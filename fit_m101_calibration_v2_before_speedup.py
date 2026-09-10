#!/usr/bin/env python3
"""First transparent M101 total-spectrum calibration draft.

The internal calibration is intentionally a visible sequence of scientific
models.  This draft runs Models 0--3 on an explicitly selected development
set.  Scattered-light, persistent-fiber, color, and external-source stages
are represented by disabled hooks and are not silently fitted.
"""

from __future__ import annotations

from argparse import ArgumentParser
import ast
import csv
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
import json
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import MatrixRankWarning, lsmr, spsolve
import warnings

import diagnose_m101_hierarchical as validated_m101

import m101_blank_fibers as blank_fibers
import m101_compact_mask as compact_mask
import m101_external_measurements as external_measurements
from m101_calibration_utils import (
    ALL_BANDS, BANDS, collapse, contrast_basis, file_identity, json_ready,
    residual_summary, robust_location, robust_location_scatter, robust_scatter,
    small_file_hash,
)
from m101_native_data import ExposureData, WAVE, load as load_native_data


SCHEMA_VERSION = "m101_calibration_v2_draft_1"
NULL_BANDS = ("HIGH1", "LOW1", "HIGH2", "LOW2", "HIGH3")
FIBER_DIAMETER_ARCSEC = 1.5
MINIMUM_BLANK_FIBERS_FOR_POPULATION = 20
MAX_DENSE_NORMAL_PARAMETERS = 1024


def _progress(started, message):
    print("[fit +%.1fs] %s" % (time.perf_counter() - started, message), flush=True)


def focal_plot_filename(exposure_key):
    """Return a stable filename for an H5 basename and exposure number."""
    return "%s_focal_plane_residual.png" % focal_plot_label(exposure_key)


def focal_plot_label(exposure_key):
    """Return the human-readable H5/exposure label used in plot titles."""
    h5_name, exposure = exposure_key
    return "%s_exp%02d" % (Path(str(h5_name)).stem, int(exposure))


def _fiber_marker_area_points2(axis, figure, x_reference, y_reference):
    """Convert a 1.5 arcsec physical fiber diameter into scatter ``s``."""
    diameter_arcmin = FIBER_DIAMETER_ARCSEC / 60.0
    transform = axis.transData
    x0, y0 = transform.transform((x_reference, y_reference))
    x1, _ = transform.transform((x_reference + diameter_arcmin, y_reference))
    diameter_points = abs(x1 - x0) * 72.0 / figure.dpi
    return float(np.pi * (diameter_points / 2.0) ** 2)


@dataclass
class ParameterLayout:
    mode: str
    ifus: list[tuple]
    ifu_index: dict[tuple, int]
    ifu_basis: np.ndarray
    ifu_columns: dict[tuple, np.ndarray]
    amp_basis: np.ndarray
    amp_columns: dict[tuple, np.ndarray]
    plane_columns: dict[tuple, tuple[int, int]]
    q_columns: dict[tuple, int]
    names: list[str]

    @property
    def size(self):
        return len(self.names)


@dataclass
class FittedModel:
    mode: str
    layout: ParameterLayout
    p_ifu: dict[tuple, float]
    p_amp: dict[tuple, float]
    ax: dict[tuple, float]
    ay: dict[tuple, float]
    alpha_q: dict[tuple, float]
    clipped_points: int
    iterations: int

    def z_for(self, data: ExposureData) -> np.ndarray:
        values = np.zeros(data.row_index.size, dtype=float)
        for index, ifu_value in enumerate(map(tuple, data.ifu)):
            amp_key = ifu_value + (str(data.amp[index]),)
            values[index] = (
                self.p_ifu.get(ifu_value, 0.0)
                + self.p_amp.get(amp_key, 0.0)
                + self.ax.get(data.key, 0.0) * data.x_arcmin[index]
                + self.ay.get(data.key, 0.0) * data.y_arcmin[index]
            )
        return values

    def additive_bands(self, data: ExposureData, fq: np.ndarray) -> np.ndarray:
        if self.mode != "model3":
            return np.zeros_like(data.band_total, dtype=float)
        alpha = np.asarray([
            self.alpha_q.get((data.key, tuple(map(int, ifu_value)), str(amp_value)), 0.0)
            for ifu_value, amp_value in zip(data.ifu, data.amp)
        ], dtype=float)
        return alpha[:, None] * fq[data.q, None] * data.K[None, :]

    def additive_full(self, data: ExposureData, fq: np.ndarray) -> np.ndarray:
        if self.mode != "model3":
            return np.zeros_like(data.total, dtype=float)
        alpha = np.asarray([
            self.alpha_q.get((data.key, tuple(map(int, ifu_value)), str(amp_value)), 0.0)
            for ifu_value, amp_value in zip(data.ifu, data.amp)
        ], dtype=float)
        return alpha[:, None] * fq[data.q, None] * data.K_wave[None, :]


def parse_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--h5", nargs="+", required=True)
    parser.add_argument("--blank-file", required=True)
    parser.add_argument("--on-filter", required=True)
    parser.add_argument("--off-filter", required=True)
    parser.add_argument("--fq-template", required=True)
    parser.add_argument("--external-cache")
    parser.add_argument("--compact-mask")
    parser.add_argument("--output-dir", default="m101_calibration_v2_development")
    parser.add_argument("--minimum-finite-fraction", type=float, default=0.8)
    parser.add_argument("--max-sky-passes", type=int, default=5)
    parser.add_argument("--disable-model3", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def estimate_initial_skies(data: list[ExposureData], minimum_finite_fraction: float,
                           progress=None):
    skies = {}
    records = {}
    for item_index, item in enumerate(data, 1):
        item_started = time.perf_counter()
        selected = item.blank_valid
        if not np.any(selected):
            raise ValueError("%s exposure %d has no valid external blank fibers" % item.key)
        sky, scatter = robust_location_scatter(item.total[selected], axis=0)
        skies[item.key] = np.asarray(sky, dtype=float)
        records[str(item.key)] = {
            "N_blank": int(np.sum(selected)),
            "minimum_finite_fraction": float(minimum_finite_fraction),
            "sky": sky, "robust_scatter": scatter,
        }
        if progress is not None:
            progress("initial sky: exposure %d/%d (%s e%d) done in %.3fs; blank-valid=%d"
                     % (item_index, len(data), item.h5_name, item.exposure,
                        time.perf_counter() - item_started, int(np.sum(selected))))
    return skies, records


def sky_band_values(skies: dict, responses: np.ndarray) -> dict:
    result = {}
    for key, sky in skies.items():
        result[key] = np.asarray([collapse(sky[None, :], response)[0][0]
                                  for response in responses], dtype=float)
    return result


def _persistent_ifus(data):
    return sorted({tuple(map(int, ifu)) for item in data for ifu in item.ifu})


def _supported_ifus(data):
    return sorted({tuple(map(int, ifu)) for item in data
                   for ifu in item.ifu[item.blank_valid]})


def _supported_q_groups(data):
    result = set()
    for item in data:
        for ifu, amp, valid in zip(item.ifu, item.amp, item.blank_valid):
            if valid:
                result.add((item.key, tuple(map(int, ifu)), str(amp)))
    return sorted(result, key=str)


def make_layout(data: list[ExposureData], mode: str) -> ParameterLayout:
    ifus = _supported_ifus(data)
    ifu_basis = contrast_basis(len(ifus))
    ifu_columns = {}
    names = []
    cursor = 0
    for index in range(ifu_basis.shape[1]):
        names.append("p_IFU_contrast_%03d" % index)
    for index, key in enumerate(ifus):
        ifu_columns[key] = np.arange(cursor, cursor + ifu_basis.shape[1], dtype=int)
    cursor += ifu_basis.shape[1]

    amp_basis = contrast_basis(4) if mode in ("model2", "model3") else np.zeros((4, 0))
    amp_columns = {}
    if mode in ("model2", "model3"):
        for ifu in ifus:
            columns = np.arange(cursor, cursor + amp_basis.shape[1], dtype=int)
            amp_columns[ifu] = columns
            names.extend("p_AMP_%s_%03d" % ("_".join(map(str, ifu)), index)
                         for index in range(amp_basis.shape[1]))
            cursor += amp_basis.shape[1]

    plane_columns = {}
    for key in sorted({item.key for item in data}, key=str):
        plane_columns[key] = (cursor, cursor + 1)
        names.extend(["ax_%s_e%d" % (key[0], key[1]),
                      "ay_%s_e%d" % (key[0], key[1])])
        cursor += 2

    q_columns = {}
    if mode == "model3":
        for key in _supported_q_groups(data):
            q_columns[key] = cursor
            names.append("alpha_q_%s_e%d_%s_%s" %
                         (key[0], key[1][0], key[2], key[1]))
            cursor += 1
    ifu_index = {ifu: index for index, ifu in enumerate(ifus)}
    return ParameterLayout(mode, ifus, ifu_index, ifu_basis, ifu_columns, amp_basis,
                           amp_columns, plane_columns, q_columns, names)


def _append_row_columns(item, row_index, band_index, layout, fq, sky):
    ifu = tuple(map(int, item.ifu[row_index]))
    amp = str(item.amp[row_index])
    columns, values = [], []
    if ifu in layout.ifu_columns:
        basis_row = layout.ifu_basis[layout.ifu_index[ifu]]
        columns.extend(layout.ifu_columns[ifu][basis_row != 0].tolist())
        values.extend((sky[band_index] * basis_row[basis_row != 0]).tolist())
    ax_column, ay_column = layout.plane_columns[item.key]
    columns.extend((ax_column, ay_column))
    values.extend((sky[band_index] * item.x_arcmin[row_index],
                   sky[band_index] * item.y_arcmin[row_index]))
    if layout.mode in ("model2", "model3") and ifu in layout.amp_columns:
        amp_index = ("LL", "LU", "RL", "RU").index(amp)
        amp_row = layout.amp_basis[amp_index]
        use = amp_row != 0
        columns.extend(layout.amp_columns[ifu][use].tolist())
        values.extend((sky[band_index] * amp_row[use]).tolist())
    if layout.mode == "model3":
        q_key = (item.key, ifu, amp)
        q_column = layout.q_columns.get(q_key)
        if q_column is not None:
            columns.append(q_column)
            values.append(float(fq[item.q[row_index]] * item.K[band_index]))
    return columns, values


def _prepare_design_rows(item, layout, fq):
    """Cache row structure shared by all seven band equations."""
    amp_indices = {amp: index for index, amp in enumerate(("LL", "LU", "RL", "RU"))}
    result = []
    for row_index, (ifu_value, amp_value, x_arcmin, y_arcmin, q_value) in enumerate(
            zip(item.ifu, item.amp, item.x_arcmin, item.y_arcmin, item.q)):
        ifu = tuple(map(int, ifu_value))
        columns = []
        factors = []
        ifu_index = layout.ifu_index.get(ifu)
        if ifu_index is not None:
            basis_row = layout.ifu_basis[ifu_index]
            use = basis_row != 0
            columns.extend(layout.ifu_columns[ifu][use].tolist())
            factors.extend(basis_row[use].tolist())
        ax_column, ay_column = layout.plane_columns[item.key]
        columns.extend((ax_column, ay_column))
        factors.extend((float(x_arcmin), float(y_arcmin)))
        if layout.mode in ("model2", "model3") and ifu in layout.amp_columns:
            amp_row = layout.amp_basis[amp_indices[str(amp_value)]]
            use = amp_row != 0
            columns.extend(layout.amp_columns[ifu][use].tolist())
            factors.extend(amp_row[use].tolist())
        q_column = None
        q_factor = 0.0
        if layout.mode == "model3":
            q_column = layout.q_columns.get((item.key, ifu, str(amp_value)))
            if q_column is not None:
                q_factor = float(fq[q_value])
        result.append((tuple(columns), np.asarray(factors, dtype=float),
                       q_column, q_factor))
    return result


def build_design(data, skies, responses, fq, mode, timings=None, progress=None):
    shared_started = time.perf_counter()
    layout = make_layout(data, mode)
    sky_bands = sky_band_values(skies, responses)
    if timings is not None:
        timings["%s_design_shared_seconds" % mode] = time.perf_counter() - shared_started
    matrix_blocks = []
    target_blocks = []
    sigma_blocks = []
    row_number = 0
    for item_index, item in enumerate(data, 1):
        item_started = time.perf_counter()
        item_row_start = row_number
        row_numbers, column_numbers, matrix_values, item_target, item_sigma = [], [], [], [], []
        sky = sky_bands[item.key]
        valid_rows = item.blank_valid
        template_started = time.perf_counter()
        row_templates = _prepare_design_rows(item, layout, fq)
        template_seconds = time.perf_counter() - template_started
        for native_index in np.flatnonzero(valid_rows):
            columns, factors, q_column, q_factor = row_templates[native_index]
            for band_index in range(len(ALL_BANDS)):
                y = item.band_total[native_index, band_index] - sky[band_index]
                error = item.band_error[native_index, band_index]
                if not np.isfinite(y) or not np.isfinite(error) or error <= 0:
                    continue
                values = (factors * sky[band_index]).tolist()
                row_columns = columns
                if q_column is not None:
                    row_columns = columns + (q_column,)
                    values.append(q_factor * item.K[band_index])
                row_numbers.extend([row_number] * len(row_columns))
                column_numbers.extend(row_columns)
                matrix_values.extend(values)
                item_target.append(float(y))
                item_sigma.append(float(error))
                row_number += 1
        csr_started = time.perf_counter()
        if item_target:
            matrix_blocks.append(
                sparse.coo_matrix(
                    (matrix_values,
                     (np.asarray(row_numbers, dtype=np.int64) - item_row_start,
                      np.asarray(column_numbers, dtype=np.int64))),
                    shape=(row_number - item_row_start, layout.size),
                ).tocsr()
            )
            target_blocks.append(np.asarray(item_target, dtype=float))
            sigma_blocks.append(np.asarray(item_sigma, dtype=float))
        csr_seconds = time.perf_counter() - csr_started
        if progress is not None:
            progress("%s design: exposure %d/%d done; %d equations in %.3fs "
                     "(templates=%.3fs, CSR=%.3fs)"
                     % (mode, item_index, len(data), row_number - item_row_start,
                        time.perf_counter() - item_started, template_seconds,
                        csr_seconds))
        if timings is not None:
            timings.setdefault("%s_design_per_exposure" % mode, []).append({
                "H5": item.h5_name, "exposure": int(item.exposure),
                "rows": int(item.row_index.size),
                "valid_rows": int(np.sum(valid_rows)),
                "equations": int(row_number - item_row_start),
                "template_seconds": template_seconds,
                "csr_seconds": csr_seconds,
                "seconds": time.perf_counter() - item_started,
            })
    if not target_blocks:
        raise ValueError("no finite blank calibration equations")
    assembly_started = time.perf_counter()
    matrix = sparse.vstack(matrix_blocks, format="csr")
    if progress is not None:
        progress("%s design: exposure blocks assembled into CSR in %.3fs"
                 % (mode, time.perf_counter() - assembly_started))
    return (layout, matrix, np.concatenate(target_blocks),
            np.concatenate(sigma_blocks), sky_bands)


def fit_active_model(data, skies, responses, fq, mode, timings=None, progress=None):
    build_started = time.perf_counter()
    layout, matrix, target, sigma, _ = build_design(
        data, skies, responses, fq, mode, timings=timings, progress=progress)
    if timings is not None:
        timings["%s_build_design_seconds" % mode] = time.perf_counter() - build_started
    if progress is not None:
        progress("%s design assembled: %d equations x %d parameters; starting 6 IRLS solves"
                 % (mode, matrix.shape[0], matrix.shape[1]))
    base_weights = 1.0 / np.maximum(sigma, np.finfo(float).eps)
    robust_weights = np.ones(target.size, dtype=float)
    params = np.zeros(layout.size, dtype=float)
    clipped = 0
    solve_started = time.perf_counter()
    iteration_timings = []
    for iteration in range(6):
        iteration_started = time.perf_counter()
        if progress is not None:
            progress("%s IRLS %d/6: preparing row weights" % (mode, iteration + 1))
        weighted_started = time.perf_counter()
        weights = base_weights * np.sqrt(robust_weights)
        normal_weights = weights ** 2
        weighted_normal_matrix = matrix.multiply(normal_weights[:, None]).tocsr()
        weighted_seconds = time.perf_counter() - weighted_started
        if progress is not None:
            progress("%s IRLS %d/6: weighted design ready in %.3fs; forming normal matrix"
                     % (mode, iteration + 1, weighted_seconds))
        normal_started = time.perf_counter()
        normal = (matrix.T @ weighted_normal_matrix).tocsc()
        normal_seconds = time.perf_counter() - normal_started
        if progress is not None:
            progress("%s IRLS %d/6: normal matrix ready in %.3fs; forming RHS"
                     % (mode, iteration + 1, normal_seconds))
        rhs_started = time.perf_counter()
        rhs = np.asarray(matrix.T @ (target * normal_weights)).ravel()
        rhs_seconds = time.perf_counter() - rhs_started
        if progress is not None:
            progress("%s IRLS %d/6: RHS ready in %.3fs; solving normal system"
                     % (mode, iteration + 1, rhs_seconds))
        solver_name = "spsolve"
        solver_rank = None
        linear_solve_started = time.perf_counter()
        if layout.size <= MAX_DENSE_NORMAL_PARAMETERS:
            solver_name = "lstsq_dense"
            params, _, solver_rank, _ = np.linalg.lstsq(
                normal.toarray(), rhs, rcond=1e-12)
            params = np.asarray(params, dtype=float)
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("error", MatrixRankWarning)
                try:
                    params = np.asarray(spsolve(normal, rhs), dtype=float)
                except (MatrixRankWarning, RuntimeError, ValueError):
                    solver_name = "lsmr_fallback"
                    weighted_matrix = matrix.multiply(weights[:, None]).tocsr()
                    params = np.asarray(lsmr(weighted_matrix, target * weights,
                                             atol=1e-12, btol=1e-12)[0], dtype=float)
        if not np.all(np.isfinite(params)):
            solver_name = "lsmr_fallback"
            weighted_matrix = matrix.multiply(weights[:, None]).tocsr()
            params = np.asarray(lsmr(weighted_matrix, target * weights,
                                     atol=1e-12, btol=1e-12)[0], dtype=float)
        if not np.all(np.isfinite(params)):
            raise RuntimeError("sparse weighted solve returned non-finite parameters")
        linear_solve_seconds = time.perf_counter() - linear_solve_started
        if progress is not None:
            rank_text = ("; rank=%d/%d" % (solver_rank, layout.size)
                         if solver_rank is not None else "")
            progress("%s IRLS %d/6: %s finished in %.3fs%s; reweighting residuals"
                     % (mode, iteration + 1, solver_name, linear_solve_seconds,
                        rank_text))
        reweight_started = time.perf_counter()
        residual = target - matrix @ params
        scale = max(float(validated_m101.robust_scale(residual)), 1e-12)
        standardized = np.abs(residual) / scale
        robust_weights = np.where(standardized > 1.345,
                                  1.345 / standardized, 1.0)
        clipped = int(np.sum(robust_weights < 1.0))
        if progress is not None:
            progress("%s IRLS %d/6: complete in %.3fs; clipped=%d"
                     % (mode, iteration + 1, time.perf_counter() - iteration_started, clipped))
        iteration_timings.append({
            "iteration": int(iteration + 1), "solver": solver_name,
            "solver_rank": (int(solver_rank) if solver_rank is not None else None),
            "weighted_design_seconds": weighted_seconds,
            "normal_matrix_seconds": normal_seconds,
            "rhs_seconds": rhs_seconds,
            "linear_solve_seconds": linear_solve_seconds,
            "residual_reweight_seconds": time.perf_counter() - reweight_started,
            "total_seconds": time.perf_counter() - iteration_started,
        })
    if timings is not None:
        timings["%s_irls_solve_seconds" % mode] = time.perf_counter() - solve_started
        timings["%s_irls_iterations" % mode] = iteration_timings
    p_ifu = {}
    for index, ifu in enumerate(layout.ifus):
        cols = layout.ifu_columns[ifu]
        p_ifu[ifu] = float(layout.ifu_basis[index] @ params[cols]) if cols.size else 0.0
    p_amp = {}
    for ifu in layout.ifus:
        cols = layout.amp_columns.get(ifu, np.zeros(0, dtype=int))
        for amp_index, amp in enumerate(("LL", "LU", "RL", "RU")):
            p_amp[ifu + (amp,)] = float(layout.amp_basis[amp_index] @ params[cols]) if cols.size else 0.0
    ax = {key: float(params[cols[0]]) for key, cols in layout.plane_columns.items()}
    ay = {key: float(params[cols[1]]) for key, cols in layout.plane_columns.items()}
    alpha_q = {key: float(params[column]) for key, column in layout.q_columns.items()}
    return FittedModel(mode, layout, p_ifu, p_amp, ax, ay, alpha_q,
                       clipped, iteration + 1)


def center_persistent_ifu(model: FittedModel, skies: dict):
    if not model.p_ifu:
        return {"robust_location_before": 0.0, "sky_rescale": 1.0}
    center = float(robust_location(np.asarray(list(model.p_ifu.values()), dtype=float)))
    for key in model.p_ifu:
        model.p_ifu[key] -= center
    scale = float(np.exp(center))
    for key in skies:
        skies[key] = skies[key] * scale
    return {"robust_location_before": center, "sky_rescale": scale,
            "robust_location_after": float(robust_location(np.asarray(list(model.p_ifu.values()))))}


def update_skies(data, skies, model, fq, responses, timings=None, timing_key=None,
                 progress=None):
    history = []
    for item_index, item in enumerate(data, 1):
        item_started = time.perf_counter()
        selected = item.blank_valid
        model_eval_started = time.perf_counter()
        z = model.z_for(item)
        additive = model.additive_full(item, fq)
        model_eval_seconds = time.perf_counter() - model_eval_started
        sky_calc_started = time.perf_counter()
        corrected = ((np.asarray(item.total[selected], dtype=float) - additive[selected])
                     / np.exp(z[selected])[:, None])
        new_sky = np.asarray(robust_location(corrected, axis=0), dtype=float)
        old_sky = np.asarray(skies[item.key], dtype=float)
        finite = np.isfinite(old_sky) & np.isfinite(new_sky)
        delta = new_sky[finite] - old_sky[finite]
        denominator = max(float(np.sqrt(np.mean(old_sky[finite] ** 2))), 1e-12)
        sky_calc_seconds = time.perf_counter() - sky_calc_started
        history.append({"exposure": list(item.key),
                        "absolute_rms_change": float(np.sqrt(np.mean(delta ** 2)))
                        if delta.size else np.nan,
                        "fractional_sky_change": float(np.sqrt(np.mean(delta ** 2)) / denominator)
                        if delta.size else np.nan,
                        "blank_robust_rms": float(validated_m101.robust_scale(
                            (corrected[:, finite] - new_sky[finite]).ravel()))
                        if finite.any() else np.nan})
        skies[item.key] = new_sky
        if progress is not None:
            progress("%s: exposure %d/%d sky update done in %.3fs"
                     % (timing_key or "sky update", item_index, len(data),
                        time.perf_counter() - item_started))
        if timings is not None and timing_key is not None:
            timings.setdefault(timing_key + "_per_exposure", []).append({
                "H5": item.h5_name, "exposure": int(item.exposure),
                "model_eval_seconds": model_eval_seconds,
                "sky_calculation_seconds": sky_calc_seconds,
                "seconds": time.perf_counter() - item_started,
            })
    return history


def _group_residuals(data, residuals):
    grouped = {"exposure": {}, "ifu": {}, "amplifier": {}, "band": {}}
    for item, residual in zip(data, residuals):
        exposure_key = str(item.key)
        grouped["exposure"].setdefault(exposure_key, []).append(residual)
        for index, (ifu, amp) in enumerate(zip(item.ifu, item.amp)):
            if not item.blank_valid[index]:
                continue
            ifu_key = str(tuple(map(int, ifu)))
            amp_key = str(tuple(map(int, ifu)) + (str(amp),))
            grouped["ifu"].setdefault(ifu_key, []).append(residual[index])
            grouped["amplifier"].setdefault(amp_key, []).append(residual[index])
        for band_index, band in enumerate(ALL_BANDS):
            values = residual[item.blank_valid, band_index]
            grouped["band"].setdefault(band, []).append(values)
    output = {}
    for level, values in grouped.items():
        output[level] = {}
        for key, arrays in values.items():
            joined = np.concatenate([np.asarray(value).ravel() for value in arrays])
            output[level][key] = residual_summary(joined)
    return output


def evaluate_blank_residuals(data, skies, model, fq, responses, name,
                             timings=None, timing_key=None, progress=None):
    shared_started = time.perf_counter()
    sky_bands = sky_band_values(skies, responses)
    shared_seconds = time.perf_counter() - shared_started
    residuals = []
    focal = {}
    for item_index, item in enumerate(data, 1):
        item_started = time.perf_counter()
        model_eval_started = time.perf_counter()
        z = model.z_for(item)
        additive = model.additive_bands(item, fq)
        model_eval_seconds = time.perf_counter() - model_eval_started
        residual_started = time.perf_counter()
        corrected = (item.band_total - additive) / np.exp(z)[:, None]
        residual = corrected - sky_bands[item.key][None, :]
        residual[~item.blank_valid] = np.nan
        residual_seconds = time.perf_counter() - residual_started
        residuals.append(residual)
        focal[str(item.key)] = {
            "x_arcmin": item.x_arcmin, "y_arcmin": item.y_arcmin,
            "residual": np.nanmedian(residual, axis=1),
            "N_blank": int(np.sum(item.blank_valid)),
        }
        if progress is not None:
            progress("%s: exposure %d/%d residual evaluation done in %.3fs"
                     % (name, item_index, len(data), time.perf_counter() - item_started))
        if timings is not None and timing_key is not None:
            timings.setdefault(timing_key + "_per_exposure", []).append({
                "H5": item.h5_name, "exposure": int(item.exposure),
                "model_eval_seconds": model_eval_seconds,
                "residual_focal_seconds": residual_seconds,
                "seconds": time.perf_counter() - item_started,
            })
    grouping_started = time.perf_counter()
    joined = np.concatenate([value[np.isfinite(value)] for value in residuals
                             if np.any(np.isfinite(value))])
    if progress is not None:
        progress("%s: per-exposure residuals ready; grouping residual summaries" % name)
    qa = {"model": name, "overall": residual_summary(joined),
          "grouped": _group_residuals(data, residuals), "focal": focal,
          "residual_arrays": residuals}
    if timings is not None and timing_key is not None:
        timings[timing_key + "_shared_seconds"] = shared_seconds
        timings[timing_key + "_grouping_seconds"] = time.perf_counter() - grouping_started
    return qa


def support_report(data, progress=None):
    # Build identities and support sets in one pass. The previous version
    # scanned every native fiber three times to create identity sets, then
    # scanned them again to populate support, which is costly for large runs.
    ifu_support = {}
    amp_support = {}
    fiber_support = {}
    per_exposure = {}
    for item_index, item in enumerate(data, 1):
        item_started = time.perf_counter()
        per_exposure[item.key] = {"IFU": set(), "amplifier": set(), "fiber": set()}
        for ifu, amp, j, valid in zip(item.ifu, item.amp, item.j, item.blank_valid):
            ifu_key = tuple(map(int, ifu)); amp_key = ifu_key + (str(amp),)
            fiber_key = amp_key + (int(j),)
            ifu_support.setdefault(ifu_key, set())
            amp_support.setdefault(amp_key, set())
            fiber_support.setdefault(fiber_key, set())
            if not valid:
                continue
            per_exposure[item.key]["IFU"].add(ifu_key)
            per_exposure[item.key]["amplifier"].add(amp_key)
            per_exposure[item.key]["fiber"].add(fiber_key)
            ifu_support[ifu_key].add(item.key)
            amp_support[amp_key].add(item.key)
            fiber_support[fiber_key].add(item.key)
        if progress is not None:
            progress("support report: exposure %d/%d (%s e%d) done in %.3fs"
                     % (item_index, len(data), item.h5_name, item.exposure,
                        time.perf_counter() - item_started))
    def rows(values):
        return [{"identity": list(key), "support_in_exposures": len(exposures),
                 "supported": bool(exposures),
                 "exposures": [list(x) for x in sorted(exposures, key=str)]}
                for key, exposures in sorted(values.items(), key=lambda pair: str(pair[0]))]
    all_ifus = sorted(ifu_support, key=str)
    all_amps = sorted(amp_support, key=str)
    all_fibers = sorted(fiber_support, key=str)
    return {"ifu": rows(ifu_support), "amplifier": rows(amp_support),
            "fiber": rows(fiber_support),
            "per_exposure": {
                str(key): {name: int(len(values)) for name, values in levels.items()}
                for key, levels in sorted(per_exposure.items(), key=lambda pair: str(pair[0]))},
            "counts": {"IFU_total": len(all_ifus),
                       "IFU_supported": sum(bool(x) for x in ifu_support.values()),
                       "IFU_unsupported": sum(not x for x in ifu_support.values()),
                       "amplifier_total": len(all_amps),
                       "amplifier_supported": sum(bool(x) for x in amp_support.values()),
                       "amplifier_unsupported": sum(not x for x in amp_support.values()),
                       "fiber_total": len(all_fibers),
                       "fiber_supported": sum(bool(x) for x in fiber_support.values()),
                       "fiber_unsupported": sum(not x for x in fiber_support.values())}}


def diagnose_scattered_light_residual(*args, **kwargs):
    return {"enabled": False, "reason": "no validated scattered-light basis was found"}


def fit_scattered_light(*args, **kwargs):
    raise RuntimeError("scattered-light fitting is disabled until a validated basis is identified")


def diagnose_wavelength_response(*args, **kwargs):
    return {"enabled": False, "reason": "optional color stage is deferred until scalar calibration is stable"}


def fit_external_source_scale(*args, **kwargs):
    raise RuntimeError("external source anchor is deferred in the two-H5 draft")


def _print_rung(qa):
    values = qa["overall"]
    print("%s: N=%d median=%+.6g robust_rms=%.6g arithmetic_rms=%.6g p95|r|=%.6g p99|r|=%.6g max|r|=%.6g"
          % (qa["model"], values["N"], values["median"], values["robust_rms"],
             values["arithmetic_rms"], values["p95_abs"], values["p99_abs"], values["max_abs"]))


def _print_timing_summary(timings):
    """Print coarse global and per-exposure timings without a profiler dump."""
    print("\nTiming summary (seconds)")
    scalar_keys = [key for key, value in timings.items()
                   if isinstance(value, (int, float))]
    for key in sorted(scalar_keys):
        print("  %-42s %.3f" % (key, timings[key]))

    for row in timings.get("native_h5_shared_seconds", []):
        print("  native shared H5 %-24s total=%.3f physical=%.3f masks=%.3f"
              % (row["H5"], row["seconds"], row["physical_arrays_seconds"],
                 row["hardware_masks_seconds"]))
    for row in sorted(timings.get("native_per_exposure", []),
                      key=lambda value: value["total_seconds"], reverse=True):
        print("  native exposure %-24s e%d read=%.3f support=%.3f collapse=%.3f coords=%.3f total=%.3f"
              % (row["H5"], row["exposure"], row["read_reconstruct_seconds"],
                 row["support_mask_seconds"], row["band_collapse_seconds"],
                 row["coordinates_support_seconds"], row["total_seconds"]))

    for mode in ("model1", "model2", "model3"):
        for stage in ("design", "sky_update", "qa"):
            rows = timings.get("%s_%s_per_exposure" % (mode, stage), [])
            for row in sorted(rows, key=lambda value: value["seconds"], reverse=True)[:10]:
                print("  %s %s exposure %-24s e%d total=%.3f"
                      % (mode, stage, row["H5"], row["exposure"], row["seconds"]))
        iterations = timings.get("%s_irls_iterations" % mode, [])
        if iterations:
            print("  %s IRLS: %s" % (mode, ", ".join(
                "i%d=%0.3fs(%s)" % (row["iteration"], row["total_seconds"], row["solver"])
                for row in iterations)))


def _aggregate_amplifier_qa(records, out3, out5):
    """Aggregate exposure measurements first by H5, then across H5s."""
    h5_groups = {}
    for record, flag3, flag5 in zip(records, out3, out5):
        key = (record["h5"], record["SPECID"], record["IFUSLOT"],
               record["IFUID"], record["AMP"])
        row = h5_groups.setdefault(key, {
            "H5": record["h5"], "SPECID": record["SPECID"],
            "IFUSLOT": record["IFUSLOT"], "IFUID": record["IFUID"],
            "AMP": record["AMP"], "supported_exposures": [],
            "outlier3_exposures": [], "outlier5_exposures": []})
        if record["support_status"] == "supported":
            label = "e%d" % record["exposure"]
            row["supported_exposures"].append(label)
            if flag3:
                row["outlier3_exposures"].append(label)
            if flag5:
                row["outlier5_exposures"].append(label)
    h5_rows = []
    for row in h5_groups.values():
        supported = len(row["supported_exposures"])
        outlier3 = len(row["outlier3_exposures"])
        outlier5 = len(row["outlier5_exposures"])
        h5_rows.append({
            **row,
            "_supported_labels": list(row["supported_exposures"]),
            "_outlier3_labels": list(row["outlier3_exposures"]),
            "_outlier5_labels": list(row["outlier5_exposures"]),
            "supported_exposures": supported,
            "outlier3_exposures": outlier3,
            "outlier5_exposures": outlier5,
            "h5_outlier_3mad": bool(supported >= 2 and outlier3 >= 2),
            "h5_outlier_5mad": bool(supported >= 2 and outlier5 >= 2),
        })
    return sorted(h5_rows, key=lambda row: (
        row["H5"], row["SPECID"], row["IFUSLOT"], row["IFUID"], row["AMP"]))


def _aggregate_persistent_qa(h5_rows):
    physical = {}
    for row in h5_rows:
        key = (row["SPECID"], row["IFUSLOT"], row["IFUID"], row["AMP"])
        result = physical.setdefault(key, {
            "SPECID": key[0], "IFUSLOT": key[1], "IFUID": key[2],
            "AMP": key[3], "supported_H5": [], "outlier_H5_3mad": [],
            "outlier_H5_5mad": [], "supported_exposures": [],
            "outlier_exposures_3mad": [], "outlier_exposures_5mad": []})
        if row["supported_exposures"]:
            result["supported_H5"].append(row["H5"])
            result["supported_exposures"].extend(
                "%s/%s" % (row["H5"], label)
                for label in row.get("_supported_labels", []))
        # Preserve the exposure provenance while the H5 counts remain the
        # scientific unit of independence.
        if row["outlier3_exposures"]:
            result["outlier_exposures_3mad"].extend(
                "%s/%s" % (row["H5"], label)
                for label in row.get("_outlier3_labels", []))
        if row["outlier5_exposures"]:
            result["outlier_exposures_5mad"].extend(
                "%s/%s" % (row["H5"], label)
                for label in row.get("_outlier5_labels", []))
        if row["h5_outlier_3mad"]:
            result["outlier_H5_3mad"].append(row["H5"])
        if row["h5_outlier_5mad"]:
            result["outlier_H5_5mad"].append(row["H5"])
    output = []
    for row in physical.values():
        n_supported = len(row["supported_H5"])
        n_out3 = len(row["outlier_H5_3mad"])
        n_out5 = len(row["outlier_H5_5mad"])
        row.update({
            "N_supported_H5": n_supported,
            "N_outlier_H5_3mad": n_out3,
            "N_outlier_H5_5mad": n_out5,
            "fraction_H5_3mad": float(n_out3 / n_supported) if n_supported else 0.0,
            "fraction_H5_5mad": float(n_out5 / n_supported) if n_supported else 0.0,
            "candidate_label": (
                "persistent_candidate" if n_out5 >= 2 else
                "h5_specific_candidate" if n_out5 == 1 and n_supported >= 2 else
                "weak_candidate" if n_out3 >= 2 else "nominal"),
        })
        output.append(row)
    return sorted(output, key=lambda row: (
        -row["N_outlier_H5_5mad"], -row["fraction_H5_5mad"],
        -row["N_outlier_H5_3mad"], row["SPECID"], row["IFUSLOT"],
        row["IFUID"], row["AMP"]))


def _write_amplifier_qa_tables(output_dir, h5_rows, persistent_rows):
    output_dir = Path(output_dir)
    by_h5_fields = ["H5", "SPECID", "IFUSLOT", "IFUID", "AMP",
                    "supported_exposures", "outlier3_exposures",
                    "outlier5_exposures", "h5_outlier_3mad",
                    "h5_outlier_5mad"]
    persistent_fields = ["SPECID", "IFUSLOT", "IFUID", "AMP",
                         "N_supported_H5", "N_outlier_H5_3mad",
                         "N_outlier_H5_5mad", "fraction_H5_3mad",
                         "fraction_H5_5mad", "supported_H5",
                         "outlier_H5_3mad", "outlier_H5_5mad",
                         "candidate_label"]
    for filename, fields, rows in (
            ("amplifier_qa_by_h5.csv", by_h5_fields, h5_rows),
            ("amplifier_qa_persistent.csv", persistent_fields, persistent_rows)):
        with (output_dir / filename).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows({field: row[field] for field in fields}
                             for row in rows)


def diagnose_amplifier_residual_population(
        data, qa3, output_dir,
        minimum_blank_fibers=MINIMUM_BLANK_FIBERS_FOR_POPULATION,
        progress=None):
    """Report residual population behavior without changing fit validity."""
    records, band_values = [], [[] for _ in ALL_BANDS]
    for item_index, (item, residual) in enumerate(zip(data, qa3["residual_arrays"]), 1):
        item_started = time.perf_counter()
        groups = {}
        for index, (ifu_value, amp_value) in enumerate(zip(item.ifu, item.amp)):
            key = (tuple(map(int, ifu_value)), str(amp_value))
            groups.setdefault(key, []).append(index)
        for (ifu, amp), group_rows in sorted(groups.items(), key=lambda pair: str(pair[0])):
            rows = np.asarray(group_rows, dtype=int)
            rows = rows[item.blank_valid[rows]]
            r, s = np.asarray(residual[rows]), np.asarray(item.band_error[rows])
            good = np.isfinite(r) & np.isfinite(s) & (s > 0)
            if not np.any(good):
                continue
            ratios = r / s
            rec = {"h5": item.h5_name, "exposure": item.exposure,
                   "SPECID": int(ifu[0]), "IFUSLOT": int(ifu[1]), "IFUID": int(ifu[2]),
                   "AMP": amp, "N": int(np.sum(good)),
                   "N_blank_fibers": int(rows.size),
                   "N_valid_band_measurements": int(np.sum(good)),
                   "support_status": ("supported" if rows.size >= minimum_blank_fibers
                                       else "insufficient_support"),
                   "_ratios": ratios}
            for band_index in range(len(ALL_BANDS)):
                gb = good[:, band_index]
                if np.any(gb):
                    band_ratios = ratios[gb, band_index]
                    band_values[band_index].append(
                        float(validated_m101.robust_scale(band_ratios)))
            records.append(rec)
        if progress is not None:
            progress("amplifier population: exposure %d/%d (%s e%d) done in %.3fs; "
                     "%d groups"
                     % (item_index, len(data), item.h5_name, item.exposure,
                        time.perf_counter() - item_started, len(groups)))
    if not records:
        raise ValueError("no valid blank Model-3 amplifier residual measurements")
    supported = np.asarray([r["support_status"] == "supported" for r in records], dtype=bool)
    if not np.any(supported):
        raise ValueError("no amplifier observations meet minimum blank-fiber support")
    empirical_band_scales = []
    for band_index in range(len(ALL_BANDS)):
        values = [rec["_ratios"][np.isfinite(rec["_ratios"][:, band_index]), band_index]
                  for rec in records if rec["support_status"] == "supported"
                  and np.any(np.isfinite(rec["_ratios"][:, band_index]))]
        values = np.concatenate(values) if values else np.asarray([], dtype=float)
        scale = float(validated_m101.robust_scale(values)) if values.size else np.nan
        empirical_band_scales.append(
            scale if np.isfinite(scale) and scale > 0 else 1.0)
    empirical_band_scales = np.asarray(empirical_band_scales, dtype=float)
    for rec in records:
        ratios = rec.pop("_ratios")
        normalized = ratios / empirical_band_scales[None, :]
        normalized = normalized[np.isfinite(normalized)]
        rec["mean_significance"] = float(np.mean(normalized))
        normalized_scale = float(validated_m101.robust_scale(normalized))
        rec["mad_over_error"] = (normalized_scale if np.isfinite(normalized_scale)
                                  else 0.0)
        rec["median_normalized_residual"] = float(np.median(normalized))
    all_x = np.asarray([r["mean_significance"] for r in records])
    all_y = np.asarray([r["mad_over_error"] for r in records])
    x, y = all_x[supported], all_y[supported]
    xmed, ymed = np.median(x), np.median(y)
    xmad = max(1.4826 * np.median(np.abs(x - xmed)), np.finfo(float).eps)
    ymad = max(1.4826 * np.median(np.abs(y - ymed)), np.finfo(float).eps)
    zx, zy = (all_x - xmed) / xmad, (all_y - ymed) / ymad
    scores = np.maximum(np.abs(zx), np.abs(zy))
    out3 = supported & (scores > 3)
    out5 = supported & (scores > 5)
    for rec, score, flag3, flag5 in zip(records, scores, out3, out5):
        rec["population_score"] = float(score)
        rec["outlier_3mad"] = bool(flag3)
        rec["outlier_5mad"] = bool(flag5)
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(6, 5), constrained_layout=True)
    axis.scatter(all_x[supported & ~out5], all_y[supported & ~out5], s=10,
                 color="tab:blue", alpha=.65)
    axis.scatter(all_x[out5], all_y[out5], s=18, color="crimson", alpha=.9)
    if np.any(~supported):
        axis.scatter(all_x[~supported], all_y[~supported], s=16, color="0.55",
                     marker="x", label="insufficient support")
    axis.axvline(0, color="0.25", lw=.8)
    for value in (xmed - 5 * xmad, xmed + 5 * xmad): axis.axvline(value, color="0.45", ls="--", lw=.8)
    for value in (ymed - 5 * ymad, ymed + 5 * ymad): axis.axhline(value, color="0.45", ls="--", lw=.8)
    for rec in sorted(records, key=lambda r: r["population_score"], reverse=True)[:10]:
        axis.annotate("%s/e%d/%s-%d-%d-%d/%s" % (Path(rec["h5"]).stem, rec["exposure"], rec["SPECID"], rec["IFUSLOT"], rec["IFUID"], rec["N"], rec["AMP"]),
                      (rec["mean_significance"], rec["mad_over_error"]), fontsize=6)
    axis.set(xlabel="mean normalized residual", ylabel="robust scale of normalized residual",
             title="Amplifier residual population: %d supported, %d (%.1f%%) 5-MAD outliers" %
             (int(supported.sum()), int(out5.sum()),
              100 * out5.sum() / supported.sum()))
    fig.savefig(output_dir / "amplifier_residual_population.png", dpi=140); plt.close(fig)
    fig, axis = plt.subplots(figsize=(8, 4), constrained_layout=True)
    axis.boxplot(band_values, tick_labels=ALL_BANDS, showfliers=False)
    axis.set(ylabel="robust scale of r / sigma", title="Amplifier residual population by band")
    fig.savefig(output_dir / "amplifier_residual_population_by_band.png", dpi=140); plt.close(fig)
    print("\nAmplifier residual population diagnostic")
    print("observations=%d; supported=%d; minimum_blank_fibers=%d" %
          (len(records), int(supported.sum()), minimum_blank_fibers))
    print("empirical band scales: %s" % json.dumps(
        {band: float(scale) for band, scale in zip(ALL_BANDS, empirical_band_scales)}))
    print("mean normalized residual median=%.4g robust_MAD=%.4g; normalized robust scale median=%.4g robust_MAD=%.4g" %
          (xmed, xmad, ymed, ymad))
    print("population outliers beyond 3 MAD: %d (%.1f%%); beyond 5 MAD: %d (%.1f%%)" %
          (out3.sum(), 100 * out3.sum() / supported.sum(), out5.sum(),
           100 * out5.sum() / supported.sum()))
    print("H5 exposure SPECID IFUSLOT IFUID AMP N N_blank_fibers N_valid_band_measurements status mean_normalized_residual normalized_robust_scale")
    for rec in sorted(records, key=lambda r: r["population_score"], reverse=True)[:20]:
        print("%s %d %d %d %d %s %d %d %d %s %+.4g %.4g" %
              (Path(rec["h5"]).stem, rec["exposure"], rec["SPECID"], rec["IFUSLOT"], rec["IFUID"],
               rec["AMP"], rec["N"], rec["N_blank_fibers"],
               rec["N_valid_band_measurements"], rec["support_status"],
               rec["mean_significance"], rec["mad_over_error"]))
    h5_rows = _aggregate_amplifier_qa(records, out3, out5)
    persistent = _aggregate_persistent_qa(h5_rows)
    _write_amplifier_qa_tables(output_dir, h5_rows, persistent)
    supported_h5_rows = sum(row["supported_exposures"] > 0 for row in h5_rows)
    print("H5-level amplifier population: total=%d; supported=%d; 3-MAD=%d (%.1f%%); 5-MAD=%d (%.1f%%)" %
          (len(h5_rows), supported_h5_rows,
           sum(row["h5_outlier_3mad"] for row in h5_rows),
           100 * sum(row["h5_outlier_3mad"] for row in h5_rows) / max(1, supported_h5_rows),
           sum(row["h5_outlier_5mad"] for row in h5_rows),
           100 * sum(row["h5_outlier_5mad"] for row in h5_rows) / max(1, supported_h5_rows)))
    print("persistent physical amplifier summary")
    for row in persistent:
        print("%d/%d/%d/%s supported_H5=%s outlier_H5_3mad=%s outlier_H5_5mad=%s "
              "N=%d/%d/%d fractions=%.3f/%.3f candidate=%s" %
              (row["SPECID"], row["IFUSLOT"], row["IFUID"], row["AMP"],
               row["supported_H5"], row["outlier_H5_3mad"], row["outlier_H5_5mad"],
               row["N_supported_H5"], row["N_outlier_H5_3mad"],
               row["N_outlier_H5_5mad"], row["fraction_H5_3mad"],
               row["fraction_H5_5mad"], row["candidate_label"]))
    same_h5_ifu = []
    for h5 in sorted({row["H5"] for row in h5_rows}):
        by_ifu = {}
        for row in h5_rows:
            if row["H5"] == h5 and (row["h5_outlier_3mad"] or row["h5_outlier_5mad"]):
                key = (row["H5"], row["SPECID"], row["IFUSLOT"], row["IFUID"])
                by_ifu.setdefault(key, []).append(row["AMP"])
        same_h5_ifu.extend({"H5": key[0], "SPECID": key[1], "IFUSLOT": key[2],
                            "IFUID": key[3], "AMP": sorted(amps)}
                           for key, amps in by_ifu.items() if len(amps) > 1)
    if same_h5_ifu:
        print("multiple failing amplifiers in the same H5/IFU: %s" %
              json.dumps(same_h5_ifu, sort_keys=True))
    else:
        print("multiple failing amplifiers in the same H5/IFU: none")
    for name, values in (("H5/exposure", Counter("%s/e%d" % (r["h5"], r["exposure"]) for r, flag in zip(records, out5) if flag)),
                         ("AMP orientation", Counter(r["AMP"] for r, flag in zip(records, out5) if flag)),
                         ("physical amplifier", Counter("%d/%d/%d/%s" % (r["SPECID"], r["IFUSLOT"], r["IFUID"], r["AMP"]) for r, flag in zip(records, out5) if flag))):
        print("5-MAD outliers by %s: %s" % (name, dict(sorted(values.items()))))
    return {"N": len(records), "supported_amplifier_observations": int(supported.sum()),
            "minimum_blank_fibers": int(minimum_blank_fibers),
            "median_mean_significance": float(xmed), "robust_mad_mean_significance": float(xmad),
            "median_mad_over_error": float(ymed), "robust_mad_mad_over_error": float(ymad),
            "outliers_3_mad": int(out3.sum()), "outliers_5_mad": int(out5.sum()),
            "outlier_fraction_3_mad": float(out3.sum() / supported.sum()),
            "outlier_fraction_5_mad": float(out5.sum() / supported.sum()),
            "band_scales": {band: float(scale) for band, scale in zip(ALL_BANDS, empirical_band_scales)},
            "h5_amplifier_records": h5_rows,
            "persistent_outlier_summary": persistent,
            "multiple_failing_amplifiers_same_h5_ifu": same_h5_ifu,
            "records": records}


def write_focal_maps(output_dir, qa):
    output_dir = Path(output_dir)
    for key, values in qa["focal"].items():
        path = output_dir / focal_plot_filename(
            ast.literal_eval(key) if isinstance(key, str) else key)
        x, y, residual = (np.asarray(values[name], dtype=float)
                          for name in ("x_arcmin", "y_arcmin", "residual"))
        finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(residual)
        fig, axis = plt.subplots(figsize=(5, 4), constrained_layout=True)
        if np.any(finite):
            limit = max(float(np.nanpercentile(np.abs(residual[finite]), 99)), 1e-12)
            x_span = max(float(np.ptp(x[finite])), FIBER_DIAMETER_ARCSEC / 60.0)
            y_span = max(float(np.ptp(y[finite])), FIBER_DIAMETER_ARCSEC / 60.0)
            x_margin = max(0.05 * x_span, FIBER_DIAMETER_ARCSEC / 120.0)
            y_margin = max(0.05 * y_span, FIBER_DIAMETER_ARCSEC / 120.0)
            axis.set_xlim(float(np.min(x[finite])) - x_margin,
                          float(np.max(x[finite])) + x_margin)
            axis.set_ylim(float(np.min(y[finite])) - y_margin,
                          float(np.max(y[finite])) + y_margin)
            axis.set_aspect("equal", adjustable="box")
            fig.canvas.draw()
            marker_area = _fiber_marker_area_points2(
                axis, fig, float(np.nanmedian(x[finite])), float(np.nanmedian(y[finite])))
            image = axis.scatter(x[finite], y[finite], c=residual[finite], s=marker_area,
                                 cmap="coolwarm", vmin=-limit, vmax=limit)
            fig.colorbar(image, ax=axis, label="median blank residual")
        axis.set(xlabel="Delta RA cos(Dec) [arcmin]", ylabel="Delta Dec [arcmin]",
                 title="%s focal-plane residual" % focal_plot_label(
                     ast.literal_eval(key) if isinstance(key, str) else key))
        axis.set_aspect("equal", adjustable="box")
        fig.savefig(path, dpi=140)
        plt.close(fig)


def write_calibration_product(output_dir, args, data, skies, sky_initial,
                              models, qa_by_model, sky_history, support, provenance,
                              progress=None):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    arrays_path = output_dir / "m101_calibration_v2_arrays.npz"
    if progress is not None:
        progress("compressing final sky arrays to %s" % arrays_path.name)
    np.savez_compressed(
        arrays_path,
        wavelength=WAVE,
        **{"sky_%s" % str(key).replace("'", "").replace(" ", "_").replace("(", "").replace(")", "").replace(",", "_"): value
           for key, value in skies.items()})
    parameter_rows = {}
    for name, model in models.items():
        parameter_rows[name] = {
            "mode": model.mode, "clipped_points": model.clipped_points,
            "iterations": model.iterations,
            "p_IFU_log": {str(key): value for key, value in model.p_ifu.items()},
            "p_IFU_percent": {str(key): float(100 * np.expm1(value)) for key, value in model.p_ifu.items()},
            "p_AMP_log": {str(key): value for key, value in model.p_amp.items()},
            "p_AMP_percent": {str(key): float(100 * np.expm1(value)) for key, value in model.p_amp.items()},
            "ax": {str(key): value for key, value in model.ax.items()},
            "ay": {str(key): value for key, value in model.ay.items()},
            "alpha_q": {str(key): value for key, value in model.alpha_q.items()},
        }
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "scientific_sequence": ["model0_initial_sky", "model1_ifu_illumination",
                                 "model2_amplifier", "model3_fixed_q_additive"],
        "native_equation": "total = Fibers.spectrum / Survey.offset + Fibers.skyspectrum",
        "native_error_equation": "total_error = Fibers.error / abs(Survey.offset)",
        "skyspectrum_role": "reconstruction only; never a sky model, likelihood input, or residual comparison",
        "multiplicative_model": "log(m_rel)=p_IFU+p_AMP+ax*x+ay*y",
        "additive_model": "A_q=alpha_observation * K(lambda) * fixed_f(q)",
        "band_order": list(ALL_BANDS),
        "parameter_rows": parameter_rows,
        "initial_sky": sky_initial,
        "sky_update_history": sky_history,
        "qa": {name: {key: value for key, value in qa.items() if key != "residual_arrays"}
               for name, qa in qa_by_model.items()},
        "support": support,
        "provenance": provenance,
        "arrays": str(arrays_path),
        "configuration": vars(args),
    }
    json_path = output_dir / "m101_calibration_v2_product.json"
    if progress is not None:
        progress("writing calibration metadata to %s" % json_path.name)
    json_path.write_text(json.dumps(json_ready(metadata), indent=2, sort_keys=True))
    if progress is not None:
        progress("calibration metadata and arrays written; writing focal-plane QA maps")
    for qa_index, qa in enumerate(qa_by_model.values(), 1):
        write_focal_maps(output_dir, qa)
        if progress is not None:
            progress("focal-plane QA map set %d/%d written" %
                     (qa_index, len(qa_by_model)))
    return json_path, arrays_path


def main():
    args = parse_args()
    if args.max_sky_passes < 1 or not 0 < args.minimum_finite_fraction <= 1:
        raise SystemExit("invalid sky update controls")
    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise SystemExit("output directory is nonempty; use --overwrite: %s" % output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    timings = {}
    progress = lambda message: _progress(started, message)

    # ================================================================
    # Load validated inputs
    # ================================================================
    input_started = time.perf_counter()
    h5_paths = [Path(path).expanduser().resolve() for path in args.h5]
    progress("loading calibration inputs and H5 data")
    blank_masks, blank_provenance = blank_fibers.load(args.blank_file, h5_paths)
    fq = validated_m101.load_fq(args.fq_template)
    external_provenance = None
    compact_provenance = None
    if args.external_cache:
        _, external_provenance = external_measurements.load(args.external_cache, h5_paths)
    if args.compact_mask:
        mask_data, mask_wcs = compact_mask.load(args.compact_mask)
        compact_provenance = {"path": str(Path(args.compact_mask).resolve()),
                              "shape": list(mask_data.shape),
                              "semantics": "source-reference validity only; never a VIRUS spectrum mask"}
        del mask_data, mask_wcs

    # ================================================================
    # Reconstruct total spectra
    # ================================================================
    data, input_provenance = load_native_data(
        h5_paths, blank_masks, args.on_filter, args.off_filter,
        minimum_finite_fraction=args.minimum_finite_fraction, timings=timings,
        progress=progress)
    timings["input_loading_reconstruction_seconds"] = time.perf_counter() - input_started
    responses = input_provenance["responses"]
    print("loaded %d H5 files, %d exposure instances, %d native fibers" %
          (len(h5_paths), len(data), sum(item.row_index.size for item in data)))
    print("total reconstruction: spectrum/offset + skyspectrum; skyspectrum is not retained as a model input")
    progress("native reconstruction complete in %.3fs; estimating initial sky and support" %
             timings["input_loading_reconstruction_seconds"])

    # ================================================================
    # Initial sky from external blank fibers
    # ================================================================
    initial_started = time.perf_counter()
    progress("estimating robust initial sky spectra")
    skies, sky_initial = estimate_initial_skies(
        data, args.minimum_finite_fraction, progress=progress)
    progress("initial sky spectra ready; building support report over all native fibers")
    support = support_report(data, progress=progress)
    timings["initial_sky_support_seconds"] = time.perf_counter() - initial_started
    print("initial sky blank support: %s" % json.dumps({key: value["N_blank"]
                                                        for key, value in sky_initial.items()}))
    progress("initial sky/support complete in %.3fs" % timings["initial_sky_support_seconds"])

    # ================================================================
    # Model 0: initial sky only
    # ================================================================
    model0 = FittedModel("model0", make_layout(data, "model1"), {}, {}, {}, {}, {}, 0, 0)
    model0_qa_started = time.perf_counter()
    qa0 = evaluate_blank_residuals(
        data, skies, model0, fq, responses, "MODEL0_INITIAL_SKY",
        timings=timings, timing_key="model0_qa", progress=progress)
    timings["model0_qa_seconds"] = time.perf_counter() - model0_qa_started
    _print_rung(qa0)
    qa_by_model = {"MODEL0_INITIAL_SKY": qa0}
    models = {}
    sky_history = {"MODEL0_INITIAL_SKY": []}

    # ================================================================
    # Model 1: persistent IFU + exposure illumination
    # ================================================================
    progress("starting MODEL1 persistent IFU fit")
    model1 = fit_active_model(data, skies, responses, fq, "model1", timings, progress)
    centering1 = center_persistent_ifu(model1, skies)
    sky_update_started = time.perf_counter()
    sky_history["MODEL1_IFU_ILLUMINATION"] = update_skies(
        data, skies, model1, fq, responses, timings=timings,
        timing_key="model1_sky_update", progress=progress)
    timings["model1_sky_update_seconds"] = time.perf_counter() - sky_update_started
    qa_started = time.perf_counter()
    qa1 = evaluate_blank_residuals(
        data, skies, model1, fq, responses, "MODEL1_IFU_ILLUMINATION",
        timings=timings, timing_key="model1_qa", progress=progress)
    timings["model1_qa_seconds"] = time.perf_counter() - qa_started
    qa1["parameter_centering"] = centering1
    _print_rung(qa1)
    models["MODEL1_IFU_ILLUMINATION"] = model1
    qa_by_model["MODEL1_IFU_ILLUMINATION"] = qa1

    # ================================================================
    # Model 2: add persistent amplifier response
    # ================================================================
    progress("starting MODEL2 persistent amplifier fit")
    model2 = fit_active_model(data, skies, responses, fq, "model2", timings, progress)
    centering2 = center_persistent_ifu(model2, skies)
    sky_update_started = time.perf_counter()
    sky_history["MODEL2_IFU_AMP_ILLUMINATION"] = update_skies(
        data, skies, model2, fq, responses, timings=timings,
        timing_key="model2_sky_update", progress=progress)
    timings["model2_sky_update_seconds"] = time.perf_counter() - sky_update_started
    qa_started = time.perf_counter()
    qa2 = evaluate_blank_residuals(
        data, skies, model2, fq, responses, "MODEL2_IFU_AMP_ILLUMINATION",
        timings=timings, timing_key="model2_qa", progress=progress)
    timings["model2_qa_seconds"] = time.perf_counter() - qa_started
    qa2["parameter_centering"] = centering2
    _print_rung(qa2)
    models["MODEL2_IFU_AMP_ILLUMINATION"] = model2
    qa_by_model["MODEL2_IFU_AMP_ILLUMINATION"] = qa2

    # ================================================================
    # Model 3: add the fixed validated q additive structure
    # ================================================================
    amplifier_population = {"enabled": False, "reason": "Model 3 disabled"}
    if args.disable_model3:
        print("MODEL3_Q_ADDITIVE: disabled by command line")
    else:
        progress("starting MODEL3 fixed-q additive fit")
        model3 = fit_active_model(data, skies, responses, fq, "model3", timings, progress)
        centering3 = center_persistent_ifu(model3, skies)
        sky_update_started = time.perf_counter()
        sky_history["MODEL3_Q_ADDITIVE"] = update_skies(
            data, skies, model3, fq, responses, timings=timings,
            timing_key="model3_sky_update", progress=progress)
        timings["model3_sky_update_seconds"] = time.perf_counter() - sky_update_started
        qa_started = time.perf_counter()
        qa3 = evaluate_blank_residuals(
            data, skies, model3, fq, responses, "MODEL3_Q_ADDITIVE",
            timings=timings, timing_key="model3_qa", progress=progress)
        timings["model3_qa_seconds"] = time.perf_counter() - qa_started
        qa3["parameter_centering"] = centering3
        _print_rung(qa3)
        models["MODEL3_Q_ADDITIVE"] = model3
        qa_by_model["MODEL3_Q_ADDITIVE"] = qa3

    # ================================================================
    # Diagnostic: amplifier residual population after Model 3
    # ================================================================
    if "MODEL3_Q_ADDITIVE" in qa_by_model:
        progress("starting post-MODEL3 amplifier population diagnostic")
        diagnostic_started = time.perf_counter()
        amplifier_population = diagnose_amplifier_residual_population(
            data, qa_by_model["MODEL3_Q_ADDITIVE"], output_dir,
            progress=progress)
        timings["amplifier_diagnostic_seconds"] = time.perf_counter() - diagnostic_started

    # ================================================================
    # Model 4: diagnose scattered-light residual (disabled)
    # ================================================================
    scattered_qa = diagnose_scattered_light_residual(data, qa_by_model)

    # ================================================================
    # Model 5: persistent fiber response (inactive in the first draft)
    # ================================================================
    fiber_stage = {"enabled": False, "reason": "Phase D stops after Model 3"}

    # ================================================================
    # Model 6: wavelength-dependent response (inactive in the first draft)
    # ================================================================
    color_qa = diagnose_wavelength_response(data, skies, models.get("MODEL3_Q_ADDITIVE"))

    # ================================================================
    # External ON/OFF source gray anchor (deferred)
    # ================================================================
    source_anchor = {"enabled": False, "reason": "Phase D internal calibration draft only"}

    # ================================================================
    # Global validation and one complete calibration product
    # ================================================================
    provenance = {
        "created_unix": time.time(), "runtime_seconds": time.perf_counter() - started,
        "input_h5": [file_identity(path) for path in h5_paths],
        "input_h5_count": len(h5_paths), "input_exposure_count": len(data),
        "blank_file": file_identity(args.blank_file),
        "blank_file_sha256": small_file_hash(args.blank_file),
        "blank_provenance": blank_provenance,
        "measurement_input": input_provenance,
        "hardware_exclusions": input_provenance.get("hardware_exclusions"),
        "on_filter": file_identity(args.on_filter), "off_filter": file_identity(args.off_filter),
        "fq_template": file_identity(args.fq_template),
        "fq_template_sha256": small_file_hash(args.fq_template),
        "external_measurements": external_provenance,
        "compact_mask": compact_provenance,
        "scattered_light": scattered_qa, "fiber_stage": fiber_stage,
        "color_stage": color_qa, "source_anchor": source_anchor,
        "amplifier_residual_population": amplifier_population,
        "missing_reproduction_guide": "M101_VIRUS_calibration_cube_reproduction_guide.md was not found in the workspace",
    }
    provenance["timings"] = timings
    product_started = time.perf_counter()
    progress("writing calibration product and focal maps")
    product_json, arrays = write_calibration_product(
        output_dir, args, data, skies, sky_initial, models, qa_by_model,
        sky_history, support, provenance, progress=progress)
    timings["product_and_focal_writing_seconds"] = time.perf_counter() - product_started
    timings["total_seconds"] = time.perf_counter() - started
    print("product: %s" % product_json)
    print("arrays: %s" % arrays)
    print("runtime_seconds: %.3f" % timings["total_seconds"])
    _print_timing_summary(timings)


if __name__ == "__main__":
    main()
