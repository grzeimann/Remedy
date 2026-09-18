#!/usr/bin/env python3
"""Block-coordinate joint M101 calibration.

The frozen Model-3 fitter remains the reference implementation.  This file
adds the H5/physical-IFU topology terms in a separate alternating fitter:

    D = exp(p_IFU + p_AMP + ax*x_f + ay*y_f
            + B_a.G[h5,ifu] + x_lambda*B_a.Q[h5,ifu]) S
        + alpha_q*f(q)*K

The base update is an incremental Model-3 design around the exact current
response.  Its q columns are still one-column-per-native-row groups, and its
normal system is partitioned with the frozen diagonal-q Schur structure.  The
topology update is a collection of independent six-parameter local weighted
least-squares problems.  External source measurements are amplifier-level
robust O/X statistics with a profiled common mode; their membership is frozen
at the persisted Model-3 warm start.

This script intentionally does not edit or import the Model-3 command-line
entry point.  It imports its data classes and Schur primitive as a library.
"""

from __future__ import annotations

import argparse
import csv
from itertools import combinations
import json
import math
import pickle
from pathlib import Path
import time

import numpy as np
from scipy import sparse

import diagnose_m101_hierarchical as validated_m101
import m101_blank_fibers
import m101_compact_mask
import m101_external_measurements
from diagnose_m101_post_model3 import _group_indices, _radius_arcmin, load_model_and_skies
from fit_m101_calibration_v2 import (
    FittedModel,
    Model3BlockStructureError,
    _prepare_design_rows,
    _solve_model3_block_normal,
    make_layout,
)
from m101_calibration_utils import (
    ALL_BANDS,
    collapse,
    collapse_many,
    file_identity,
    robust_location,
    robust_scatter,
    small_file_hash,
)
from m101_native_data import discover_h5, load as load_native_data


NULL_BANDS = tuple(ALL_BANDS[:5])
SOURCE_BANDS = ("ON", "OFF")
AMP_ORDER = ("LL", "LU", "RL", "RU")
AMP_INDEX = {amp: index for index, amp in enumerate(AMP_ORDER)}
BASIS = np.asarray((
    (-1., -1., 1.),
    (-1., 1., -1.),
    (1., -1., -1.),
    (1., 1., 1.),
), dtype=float)
WAVE = np.asarray(validated_m101.DEF_WAVE, dtype=float)
X_LAMBDA = (WAVE - 4500.0) / 1000.0
N_WAVE = WAVE.size
QMAX_10 = math.log(1.10) / 2.0
QMAX_15 = math.log(1.15) / 2.0
TOPOLOGY_RELATIVE_RANK_TOL = 1e-3
SCHEMA_VERSION = "m101_calibration_joint_block_coordinate_v1"


class ArchitectureError(RuntimeError):
    """The requested preserved Model-3 block structure was not available."""


def _json_ready(value):
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_json_ready(item) for item in value.tolist()]
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _finite(values):
    values = np.asarray(values, dtype=float)
    return values[np.isfinite(values)]


def _write_rows(path, rows, fields):
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            clean = {}
            for field in fields:
                value = row.get(field, "")
                if isinstance(value, np.generic):
                    value = value.item()
                clean[field] = value
            writer.writerow(clean)


def _copy_model(model, layout):
    return FittedModel(
        "model3", layout,
        dict(model.p_ifu), dict(model.p_amp), dict(model.ax), dict(model.ay),
        dict(model.alpha_q), int(model.clipped_points), int(model.iterations))


def _load_external_subset(path, h5_paths):
    """Load the validated cache while permitting one/two-H5 staged runs."""
    path = Path(path).expanduser().resolve()
    with path.open("rb") as stream:
        cache = pickle.load(stream)
    by_name = {Path(item["h5"]).name: item for item in cache.get("calibrations", [])}
    result = {}
    for h5_path in h5_paths:
        if h5_path.name not in by_name:
            raise ValueError("external cache lacks H5 %s" % h5_path.name)
        item = by_name[h5_path.name]
        result[h5_path.name] = {
            "external_object": {
                tuple(key): np.asarray(value, dtype=float)
                for key, value in item["external_object"].items()},
            "external_valid": {
                tuple(key): np.asarray(value, dtype=bool)
                for key, value in item["external_valid"].items()},
            "global_g": {key: float(value) for key, value in cache["global_g"].items()},
        }
    provenance = {
        "schema_version": m101_external_measurements.SCHEMA_VERSION,
        "cache": file_identity(path),
        "cache_version": cache.get("version"),
        "calibration_model": cache.get("calibration_model"),
        "selected_h5": [file_identity(path) for path in h5_paths],
        "images": cache.get("images"),
        "filters_in_cache": list(cache.get("filters", {})),
        "X_definition": "X = g_global * I; I is matched external image exact-aperture measurement",
    }
    return result, provenance


def _state_key(item, ifu):
    return (item.h5_name, tuple(map(int, ifu)))


def _topology_z_matrix(item, state):
    """Return the H5/IFU topology contribution for every native row/wavelength."""
    values = np.zeros((item.row_index.size, N_WAVE), dtype=float)
    for ifu, indices in _group_indices(item)[0]:
        key = (item.h5_name, tuple(map(int, ifu)))
        g = np.asarray(state.g.get(key, np.zeros(3)), dtype=float)
        q = np.asarray(state.q.get(key, np.zeros(3)), dtype=float)
        rows = np.asarray(indices, dtype=int)
        amp_values = np.asarray([BASIS[AMP_INDEX[str(value)]] for value in item.amp[rows]])
        gray = amp_values @ g
        slope = amp_values @ q
        values[rows] = gray[:, None] + slope[:, None] * X_LAMBDA[None, :]
    return values


def _full_response(item, model, state):
    return np.exp(model.z_for(item)[:, None] + _topology_z_matrix(item, state))


def _source_effective_x(responses):
    values = {}
    flat_fnu_weight = (WAVE / 4500.0) ** -2
    response_indices = (("ON", 0), ("OFF", 1)) if len(responses) == 2 else (("ON", 5), ("OFF", 6))
    for band, index in response_indices:
        response = np.asarray(responses[index], dtype=float)
        finite = np.isfinite(response) & (response != 0)
        denominator = np.sum(response[finite])
        fnu_denominator = np.sum(response[finite] * flat_fnu_weight[finite])
        values[band] = {
            "flat_F_lambda": (float(np.sum(response[finite] * X_LAMBDA[finite]) / denominator)
                               if denominator else np.nan),
            "flat_F_nu": (float(np.sum(response[finite] * flat_fnu_weight[finite] * X_LAMBDA[finite]) /
                                 fnu_denominator) if fnu_denominator else np.nan),
        }
    return values


class JointState:
    def __init__(self, model, data):
        keys = sorted({(item.h5_name, tuple(map(int, ifu)))
                       for item in data for ifu in item.ifu}, key=str)
        self.g = {key: np.zeros(3, dtype=float) for key in keys}
        self.q = {key: np.zeros(3, dtype=float) for key in keys}
        self.active = set()
        self.layout = None
        self.source_population = None


def _candidate_population(data, external, mask_data, mask_wcs, model, skies, fq,
                          responses, minimum_source_fibers):
    """Freeze significant native source membership and its warm-start quality."""
    source_masks = {}
    group_records = {}
    band_indices = {band: ALL_BANDS.index(band) for band in SOURCE_BANDS}
    for item in data:
        cache = external[item.h5_name]
        positions = {}
        object_values = {}
        radius = _radius_arcmin(item.ra, item.dec)
        for band in SOURCE_BANDS:
            masked, inside = m101_compact_mask.mask_radec(mask_data, mask_wcs, item.ra, item.dec)
            positions[band] = (masked, inside)
            object_values[band] = (float(cache["global_g"][band]) *
                                   np.asarray(cache["external_object"][(item.exposure, band)], dtype=float))
        response = np.exp(model.z_for(item)[:, None])
        additive = model.additive_full(item, fq)
        corrected, _ = collapse_many((np.asarray(item.total, dtype=float) - additive) / response,
                                     np.asarray(responses)[[5, 6]])
        sky_band, _ = collapse_many(np.asarray(skies[item.key], dtype=float)[None, :],
                                    np.asarray(responses)[[5, 6]])
        virus_by_band = {band: corrected[:, index] - sky_band[0, index]
                         for index, band in enumerate(SOURCE_BANDS)}
        masks = {}
        for band in SOURCE_BANDS:
            external_valid = np.asarray(cache["external_valid"][(item.exposure, band)], dtype=bool)
            masked, inside = positions[band]
            comparison = m101_external_measurements.source_comparison_validity(
                external_valid, masked, inside, ~item.hardware_bad,
                np.isfinite(object_values[band]), np.ones(object_values[band].shape))
            reference = comparison & (radius > 6.)
            if int(reference.sum()) < 20:
                valid_indices = np.flatnonzero(comparison)
                keep = valid_indices[np.argsort(object_values[band][valid_indices])[
                    :max(20, valid_indices.size // 4)]]
                reference = np.zeros(comparison.shape, dtype=bool)
                reference[keep] = True
            baseline = robust_location(object_values[band][reference]) if np.any(reference) else np.nan
            scale = robust_scatter(object_values[band][reference]) if np.any(reference) else np.nan
            if not np.isfinite(scale) or scale <= 0:
                scale = float(np.nanstd(object_values[band][reference])) if np.any(reference) else np.nan
            scale = max(float(scale), 1e-12) if np.isfinite(scale) else np.nan
            threshold = baseline + 5.0 * scale if np.isfinite(baseline) and np.isfinite(scale) else np.nan
            masks[band] = (comparison & (object_values[band] > 0) &
                           (object_values[band] > threshold))
        source_masks[item.key] = {band: masks[band].copy() for band in SOURCE_BANDS}
        _, amplifier_groups = _group_indices(item)
        for (ifu, amp), indices in amplifier_groups:
            ifu = tuple(map(int, ifu))
            for band in SOURCE_BANDS:
                selected = masks[band][indices]
                x_values = object_values[band][indices][selected]
                y_values = virus_by_band[band][indices][selected]
                valid = (np.isfinite(x_values) & np.isfinite(y_values) &
                         (x_values > 0))
                ratios = y_values[valid] / x_values[valid]
                scatter = float(robust_scatter(ratios)) if ratios.size else np.nan
                raw = float(robust_location(ratios)) if ratios.size else np.nan
                key = (item.h5_name, int(item.exposure), *ifu, str(amp), band)
                group_records[key] = {
                    "H5": item.h5_name, "exposure": int(item.exposure),
                    "SPECID": ifu[0], "IFUSLOT": ifu[1], "IFUID": ifu[2],
                    "AMP": str(amp), "band": band,
                    "N_candidate_fibers": int(np.sum(selected)),
                    "N_valid_source_fibers": int(valid.sum()),
                    "raw_ratio_warm_start": raw,
                    "robust_scatter_warm_start": scatter,
                    "source_candidate": bool(np.any(selected)),
                    "eligible_source": bool(
                        np.any(selected) and valid.sum() >= minimum_source_fibers and
                        np.isfinite(scatter) and scatter <= .15),
                    "minimum_source_fibers": int(minimum_source_fibers),
                }
    return source_masks, group_records


def _recompute_source_measurements(data, external, source_masks, accepted_groups,
                                   model, state, skies, fq, responses,
                                   source_sigma_floor):
    """Recompute the validated robust amplifier O/X statistic at current state."""
    rows = []
    response_pair = np.asarray(responses)[[5, 6]]
    x_effective = _source_effective_x(responses)
    for item in data:
        cache = external[item.h5_name]
        response = _full_response(item, model, state)
        additive = model.additive_full(item, fq)
        corrected, _ = collapse_many((np.asarray(item.total, dtype=float) - additive) / response,
                                     response_pair)
        sky_band, _ = collapse_many(np.asarray(skies[item.key], dtype=float)[None, :], response_pair)
        masks = source_masks[item.key]
        _, amplifier_groups = _group_indices(item)
        for (ifu, amp), indices in amplifier_groups:
            ifu = tuple(map(int, ifu)); amp = str(amp)
            for band_index, band in enumerate(SOURCE_BANDS):
                key = (item.h5_name, int(item.exposure), *ifu, amp, band)
                if key not in accepted_groups:
                    continue
                selected = masks[band][indices]
                x_values = np.asarray(cache["global_g"][band] *
                                      cache["external_object"][(item.exposure, band)])[indices][selected]
                virus = corrected[indices, band_index] - sky_band[0, band_index]
                y_values = virus[selected]
                valid = np.isfinite(x_values) & np.isfinite(y_values) & (x_values > 0)
                ratios = y_values[valid] / x_values[valid]
                if not ratios.size:
                    continue
                raw = float(robust_location(ratios))
                scatter = float(robust_scatter(ratios))
                if not np.isfinite(raw) or raw <= 0:
                    continue
                rows.append({
                    "key": (item.h5_name, ifu), "H5": item.h5_name,
                    "exposure": int(item.exposure), "SPECID": ifu[0],
                    "IFUSLOT": ifu[1], "IFUID": ifu[2], "AMP": amp,
                    "band": band, "N_source_measurements": int(ratios.size),
                    "raw_ratio": raw, "log_ratio": float(np.log(raw)),
                    "robust_scatter": scatter,
                    "sigma": max(float(scatter) if np.isfinite(scatter) else source_sigma_floor,
                                 source_sigma_floor),
                    "x_src_flat_F_lambda": x_effective[band]["flat_F_lambda"],
                    "x_src_flat_F_nu": x_effective[band]["flat_F_nu"],
                })
    by_group = {}
    for row in rows:
        group_key = (row["H5"], row["exposure"], row["SPECID"], row["IFUSLOT"],
                     row["IFUID"], row["band"])
        by_group.setdefault(group_key, []).append(row)
    usable = []
    for group_key, group_rows in sorted(by_group.items(), key=lambda pair: str(pair[0])):
        # One amplifier has no differential information after profiling its
        # common mode; two amplifiers are sufficient for a partial group.
        if len(group_rows) < 2:
            continue
        common = float(robust_location([row["log_ratio"] for row in group_rows]))
        for row in group_rows:
            row = dict(row)
            row["common_mode_log"] = common
            row["target_log_differential"] = row["log_ratio"] - common
            row["source_group_amplifier_support"] = len(group_rows)
            usable.append(row)
    return usable, x_effective


def _membership_artifact(data, source_masks, group_records):
    rows = []
    counts = {"candidate_native_fibers": 0, "candidate_native_fibers_ON": 0,
              "candidate_native_fibers_OFF": 0, "candidate_native_fibers_union": 0,
              "blank_original": 0, "blank_final": 0}
    for item in data:
        masks = source_masks[item.key]
        union = masks["ON"] | masks["OFF"]
        counts["candidate_native_fibers_ON"] += int(masks["ON"].sum())
        counts["candidate_native_fibers_OFF"] += int(masks["OFF"].sum())
        counts["candidate_native_fibers_union"] += int(union.sum())
        counts["candidate_native_fibers"] += int(union.sum())
        counts["blank_original"] += int(item.blank_valid.sum())
        counts["blank_final"] += int(np.sum(item.blank_valid & ~union))
        _, amplifier_groups = _group_indices(item)
        for (ifu, amp), indices in amplifier_groups:
            ifu = tuple(map(int, ifu)); amp = str(amp)
            for band in SOURCE_BANDS:
                key = (item.h5_name, int(item.exposure), *ifu, amp, band)
                record = dict(group_records[key])
                candidate = bool(record["source_candidate"])
                if candidate:
                    if record["eligible_source"]:
                        transition = "candidate_to_accepted_source"
                    else:
                        transition = "candidate_to_rejected_source_quality"
                elif int(np.sum(item.blank_valid[indices] & ~union[indices])):
                    transition = "noncandidate_to_pure_blank"
                else:
                    transition = "noncandidate_to_unused"
                record.update({
                    "N_original_blank_fibers": int(np.sum(item.blank_valid[indices])),
                    "N_final_blank_fibers": int(np.sum(item.blank_valid[indices] & ~union[indices])),
                    "N_union_source_candidate_fibers": int(np.sum(union[indices])),
                    "blank_removed_by_union": int(np.sum(item.blank_valid[indices] & union[indices])),
                    "transition": transition,
                    "source_membership_frozen": True,
                })
                rows.append(record)
    counts.update({
        "candidate_groups": int(sum(row["source_candidate"] for row in rows)),
        "accepted_source_groups": int(sum(row["eligible_source"] for row in rows)),
        "rejected_source_quality_groups": int(sum(
            row["source_candidate"] and not row["eligible_source"] for row in rows)),
        "pure_blank_groups": int(sum(row["transition"] == "noncandidate_to_pure_blank" for row in rows)),
        "unused_groups": int(sum(row["transition"] == "noncandidate_to_unused" for row in rows)),
    })
    return rows, counts


def _make_blank_masks(data, source_masks):
    return {item.key: np.asarray(item.blank_valid &
                                  ~(source_masks[item.key]["ON"] |
                                    source_masks[item.key]["OFF"]), dtype=bool)
            for item in data}


def _warm_model3_null_space(data, skies, fq, responses, max_sample_rows=200000):
    """Inspect the frozen 393-column non-q Schur space on a distributed sample."""
    layout = make_layout(data, "model3")
    sky_bands = {
        item.key: np.asarray([collapse(skies[item.key][None, :], response)[0][0]
                              for response in responses], dtype=float)
        for item in data
    }
    total_valid = sum(int(item.blank_valid.sum()) * len(ALL_BANDS) for item in data)
    stride = max(1, int(math.ceil(total_valid / max_sample_rows)))
    rr, cc, vv = [], [], []
    row_number = 0
    fq_values = fq
    for item in data:
        templates = _prepare_design_rows(item, layout, fq_values)
        for native_index in np.flatnonzero(item.blank_valid)[::stride]:
            columns, factors, q_column, q_factor = templates[native_index]
            for band_index in range(len(ALL_BANDS)):
                sky = sky_bands[item.key][band_index]
                if not np.isfinite(item.band_total[native_index, band_index]):
                    continue
                row_columns = list(columns)
                values = (factors * sky).tolist()
                if q_column is not None:
                    row_columns.append(q_column)
                    values.append(q_factor * item.K[band_index])
                rr.extend([row_number] * len(row_columns))
                cc.extend(row_columns); vv.extend(values)
                row_number += 1
    matrix = sparse.coo_matrix((vv, (rr, cc)), shape=(row_number, layout.size)).tocsc()
    normal = (matrix.T @ matrix).tocsc()
    try:
        _, solver_details = _solve_model3_block_normal(
            normal, np.zeros(layout.size, dtype=float), layout)
    except Model3BlockStructureError as error:
        raise ArchitectureError("frozen Model-3 q block failed structural inspection: %s" % error)
    q_start = solver_details["model3_q_start"]
    diagonal = np.asarray(normal.diagonal()[q_start:], dtype=float)
    constrained = diagonal > max(1e-12 * max(1., float(np.max(diagonal))), 1e-14)
    A = normal[:q_start, :q_start].toarray()
    C = normal[:q_start, q_start:].toarray()[:, constrained]
    if np.any(constrained):
        inv = 1. / diagonal[constrained]
        schur = A - (C * inv) @ C.T
    else:
        schur = A
    _, singular_values, vt = np.linalg.svd(schur, full_matrices=True)
    threshold = max(float(singular_values[0]) * 1e-12, 1e-14) if singular_values.size else 1e-14
    rank = int(np.sum(singular_values > threshold))
    null_vectors = np.asarray(vt[rank:], dtype=float).T
    row_vectors = np.asarray(vt[:rank], dtype=float).T
    null_rows = []
    def parameter_group(name):
        for prefix in ("p_IFU", "p_AMP", "ax", "ay", "alpha_q"):
            if name.startswith(prefix):
                return prefix
        return name.split("_", 1)[0]

    for index in range(null_vectors.shape[1]):
        vector = null_vectors[:, index]
        order = np.argsort(np.abs(vector))[::-1][:8]
        null_rows.append({
            "null_index": index,
            "largest_columns": [layout.names[int(column)] for column in order],
            "largest_absolute_coefficients": [float(vector[column]) for column in order],
            "dominant_parameter_groups": sorted({
                parameter_group(layout.names[int(column)]) for column in order}),
        })
    return {
        "layout": layout,
        "row_space_basis": row_vectors,
        "null_space_basis": null_vectors,
        "rank": rank,
        "columns": int(q_start),
        "sample_rows": int(row_number),
        "sample_stride": int(stride),
        "singular_values": singular_values,
        "threshold": float(threshold),
        "expected_rank": 373,
        "expected_nullity": 20,
        "null_rows": null_rows,
    }


def _incremental_design(data, model, state, skies, fq, responses, blank_masks, layout):
    blocks, targets, sigmas = [], [], []
    for item in data:
        response = _full_response(item, model, state)
        sky, _ = collapse_many(response * skies[item.key][None, :], responses)
        additive = model.additive_bands(item, fq)
        templates = _prepare_design_rows(item, layout, fq)
        rows, cols, values, target, sigma = [], [], [], [], []
        row_number = 0
        selected_native = np.flatnonzero(blank_masks[item.key])
        for native_index in selected_native:
            columns, factors, q_column, q_factor = templates[native_index]
            for band_index in range(len(ALL_BANDS)):
                y = item.band_total[native_index, band_index] - additive[native_index, band_index] - sky[native_index, band_index]
                error = item.band_error[native_index, band_index]
                if not np.isfinite(y) or not np.isfinite(error) or error <= 0:
                    continue
                row_columns = list(columns)
                row_values = (factors * sky[native_index, band_index]).tolist()
                if q_column is not None:
                    row_columns.append(q_column)
                    row_values.append(q_factor * item.K[band_index])
                rows.extend([row_number] * len(row_columns)); cols.extend(row_columns)
                values.extend(row_values); target.append(float(y)); sigma.append(float(error))
                row_number += 1
        if target:
            blocks.append(sparse.coo_matrix(
                (values, (rows, cols)), shape=(len(target), layout.size)).tocsr())
            targets.append(np.asarray(target, dtype=float)); sigmas.append(np.asarray(sigma, dtype=float))
    if not targets:
        raise ValueError("no finite blank equations remain after source precedence")
    return sparse.vstack(blocks, format="csr"), np.concatenate(targets), np.concatenate(sigmas)


def _locked_schur_solution(normal, rhs, layout, row_space_basis):
    q_columns = sorted(layout.q_columns.values())
    q_start = q_columns[0]
    diagonal = np.asarray(normal.diagonal()[q_start:], dtype=float)
    tolerance = max(1e-12 * max(1., float(np.max(np.abs(diagonal)))), 1e-14)
    constrained = diagonal > tolerance
    if np.any(diagonal < -tolerance):
        raise ArchitectureError("incremental Model-3 q block has a negative diagonal")
    A = normal[:q_start, :q_start].toarray()
    C = normal[:q_start, q_start:].toarray()
    b_beta = np.asarray(rhs[:q_start], dtype=float)
    b_alpha = np.asarray(rhs[q_start:], dtype=float)
    if np.any(constrained):
        inv = np.zeros_like(diagonal); inv[constrained] = 1. / diagonal[constrained]
        cc = C[:, constrained]
        schur = A - (cc * inv[constrained]) @ cc.T
        reduced_rhs = b_beta - (cc * inv[constrained]) @ b_alpha[constrained]
    else:
        inv = np.zeros_like(diagonal); cc = np.zeros((q_start, 0))
        schur = A; reduced_rhs = b_beta
    basis = np.asarray(row_space_basis, dtype=float)
    if basis.shape[0] != q_start:
        raise ArchitectureError("fixed Model-3 null projector has the wrong dimension")
    reduced_normal = basis.T @ schur @ basis
    reduced_vector = basis.T @ reduced_rhs
    theta, _, rank, _ = np.linalg.lstsq(reduced_normal, reduced_vector, rcond=1e-12)
    beta = basis @ theta
    alpha = np.zeros(diagonal.size, dtype=float)
    if np.any(constrained):
        alpha[constrained] = inv[constrained] * (b_alpha[constrained] - cc.T @ beta)
    return np.concatenate((beta, alpha)), int(rank)


def _apply_base_increment(model, layout, increment):
    for index, ifu in enumerate(layout.ifus):
        columns = layout.ifu_columns[ifu]
        if columns.size:
            model.p_ifu[ifu] = model.p_ifu.get(ifu, 0.) + float(
                layout.ifu_basis[index] @ increment[columns])
    for ifu in layout.ifus:
        columns = layout.amp_columns.get(ifu, np.zeros(0, dtype=int))
        for amp_index, amp in enumerate(AMP_ORDER):
            delta = float(layout.amp_basis[amp_index] @ increment[columns]) if columns.size else 0.
            model.p_amp[ifu + (amp,)] = model.p_amp.get(ifu + (amp,), 0.) + delta
    for key, columns in layout.plane_columns.items():
        model.ax[key] = model.ax.get(key, 0.) + float(increment[columns[0]])
        model.ay[key] = model.ay.get(key, 0.) + float(increment[columns[1]])
    for key, column in layout.q_columns.items():
        model.alpha_q[key] = model.alpha_q.get(key, 0.) + float(increment[column])


def _base_update(data, model, state, skies, fq, responses, blank_masks,
                 null_info, timings, progress=None):
    started = time.perf_counter()
    layout = null_info["layout"]
    matrix, target, sigma = _incremental_design(
        data, model, state, skies, fq, responses, blank_masks, layout)
    build_seconds = time.perf_counter() - started
    robust_weights = np.ones(target.size, dtype=float)
    base_weights = 1. / np.maximum(sigma, np.finfo(float).eps)
    increment = np.zeros(layout.size, dtype=float)
    iteration_rows = []
    for iteration in range(6):
        iter_started = time.perf_counter()
        weights = base_weights * np.sqrt(robust_weights)
        normal = (matrix.T @ matrix.multiply(weights[:, None] ** 2)).tocsc()
        rhs = np.asarray(matrix.T @ (target * weights ** 2)).ravel()
        try:
            unconstrained, schur_details = _solve_model3_block_normal(normal, rhs, layout)
        except Model3BlockStructureError as error:
            raise ArchitectureError("incremental base update lost Model-3 q sparsity: %s" % error)
        locked, locked_rank = _locked_schur_solution(
            normal, rhs, layout, null_info["row_space_basis"])
        increment = locked
        residual = target - matrix @ increment
        scale = max(float(validated_m101.robust_scale(residual)), 1e-12)
        standardized = np.abs(residual) / scale
        robust_weights = np.where(standardized > 1.345, 1.345 / standardized, 1.)
        iteration_rows.append({
            "iteration": iteration + 1,
            "solver": "model3_block_schur_locked_null",
            "unconstrained_solver": "model3_block_schur_lstsq",
            "locked_rank": locked_rank,
            "unconstrained_rank": schur_details["model3_beta_rank"],
            "clipped_rows": int(np.sum(robust_weights < 1.)),
            "seconds": time.perf_counter() - iter_started,
        })
        if progress is not None:
            progress("base Model-3 increment IRLS %d/6: %d rows, %d parameters, clipped=%d"
                     % (iteration + 1, target.size, layout.size,
                        int(np.sum(robust_weights < 1.))))
    _apply_base_increment(model, layout, increment)
    timings["base_model3_design_seconds"] = build_seconds
    timings["base_model3_update_seconds"] = time.perf_counter() - started
    timings["base_model3_update_irls"] = iteration_rows
    return {
        "equations": int(matrix.shape[0]), "parameters": int(matrix.shape[1]),
        "nnz": int(matrix.nnz), "increment_norm": float(np.linalg.norm(increment)),
        "iterations": iteration_rows,
    }


def _blank_eval(data, model, state, skies, fq, responses, blank_masks,
                need_normals=False, scales=None):
    normals = {}
    residuals = []
    scale_values = [[] for _ in ALL_BANDS]
    for item in data:
        response = _full_response(item, model, state)
        sky = np.asarray(skies[item.key], dtype=float)
        s_values, _ = collapse_many(response * sky[None, :], responses)
        u_values, _ = collapse_many(response * sky[None, :] * X_LAMBDA[None, :], responses)
        additive = model.additive_bands(item, fq)
        residual = np.asarray(item.band_total, dtype=float) - additive - s_values
        selected = np.asarray(blank_masks[item.key], dtype=bool)
        residuals.append(residual[selected])
        for band_index in range(len(ALL_BANDS)):
            values = residual[selected, band_index]
            if scales is None:
                scale_values[band_index].extend(values[np.isfinite(values)][::10].tolist())
        if not need_normals:
            continue
        scale = np.asarray(scales, dtype=float)
        _, amplifier_groups = _group_indices(item)
        for (ifu, amp), indices in amplifier_groups:
            ifu = tuple(map(int, ifu)); amp = str(amp)
            key = (item.h5_name, ifu)
            group_rows = np.asarray(indices, dtype=int)
            group_rows = group_rows[selected[group_rows]]
            if not group_rows.size:
                continue
            amp_index = AMP_INDEX[amp]
            b = BASIS[amp_index]
            H = normals.setdefault(key, {
                "H": np.zeros((6, 6), dtype=float), "rhs": np.zeros(6, dtype=float),
                "blank_fibers": 0, "blank_equations": 0, "blank_amplifiers": set(),
                "source_rows": 0, "source_amplifiers": set()})
            H["blank_fibers"] += int(group_rows.size)
            H["blank_amplifiers"].add(amp)
            for band_index in range(len(ALL_BANDS)):
                rows = group_rows
                r = residual[rows, band_index]
                sigma = item.band_error[rows, band_index]
                good = np.isfinite(r) & np.isfinite(sigma) & (sigma > 0)
                if not np.any(good):
                    continue
                s = s_values[rows, band_index][good]
                u = u_values[rows, band_index][good]
                r = r[good]; sigma = sigma[good]
                robust = np.ones(r.size, dtype=float)
                if np.isfinite(scale[band_index]) and scale[band_index] > 0:
                    standardized = np.abs(r) / scale[band_index]
                    robust = np.where(standardized > 1.345, 1.345 / standardized, 1.)
                weight = robust / sigma ** 2
                design = np.concatenate((s[:, None] * b[None, :],
                                         u[:, None] * b[None, :]), axis=1)
                H["H"] += design.T @ (weight[:, None] * design)
                H["rhs"] += design.T @ (weight * r)
                H["blank_equations"] += int(r.size)
    if scales is None:
        scales = [max(float(validated_m101.robust_scale(values)), 1e-12)
                  if values else 1.0 for values in scale_values]
        return scales, residuals
    return normals, residuals


def _source_normal(rows, model, state):
    normals = {}
    residual_values = []
    by_group = {}
    for row in rows:
        group_key = (row["H5"], row["exposure"], row["SPECID"],
                     row["IFUSLOT"], row["IFUID"], row["band"])
        by_group.setdefault(group_key, []).append(row)
    common_design = {}
    for group_key, group_rows in by_group.items():
        amplifier_vectors = np.asarray(
            [BASIS[AMP_INDEX[row["AMP"]]] for row in group_rows], dtype=float)
        # Center the topology design over the same available amplifiers as
        # the profiled source common mode.  This keeps partial groups purely
        # differential as well as the complete four-amplifier groups.
        common_design[group_key] = np.mean(amplifier_vectors, axis=0)
    for row in rows:
        key = row["key"]
        amp = AMP_INDEX[row["AMP"]]
        group_key = (row["H5"], row["exposure"], row["SPECID"],
                     row["IFUSLOT"], row["IFUID"], row["band"])
        b = BASIS[amp] - common_design[group_key]
        # target_log_differential is recomputed from the current calibrated
        # source statistic.  Increasing the response by dz decreases the
        # corrected O/X ratio by dz, so the increment design is +[B, x_src B]
        # when solving design * delta = current differential.  The cumulative
        # state must not be subtracted here a second time.
        residual = float(row["target_log_differential"])
        residual_values.append(residual)
        entry = normals.setdefault(key, {
            "H": np.zeros((6, 6), dtype=float), "rhs": np.zeros(6, dtype=float),
            "blank_fibers": 0, "blank_equations": 0, "blank_amplifiers": set(),
            "source_rows": 0, "source_amplifiers": set()})
        entry["source_rows"] += 1
        entry["source_amplifiers"].add(row["AMP"])
    source_scale = max(float(robust_scatter(residual_values)), 1e-3) if residual_values else 1.
    for row in rows:
        key = row["key"]; amp = AMP_INDEX[row["AMP"]]
        group_key = (row["H5"], row["exposure"], row["SPECID"],
                     row["IFUSLOT"], row["IFUID"], row["band"])
        b = BASIS[amp] - common_design[group_key]
        xsrc = row["x_src_flat_F_lambda"]
        residual = float(row["target_log_differential"])
        standardized = abs(residual) / source_scale
        robust = 1.345 / standardized if standardized > 1.345 else 1.
        design = np.concatenate((b, xsrc * b))
        weight = robust / max(row["sigma"], 1e-3) ** 2
        normals[key]["H"] += weight * np.outer(design, design)
        normals[key]["rhs"] += weight * design * residual
    return normals, source_scale


def _rank_topology(H):
    singular = np.linalg.svd(H, compute_uv=False)
    threshold = (max(float(singular[0]) * TOPOLOGY_RELATIVE_RANK_TOL, 1e-14)
                 if singular.size else 1e-14)
    full = int(np.sum(singular > threshold))
    g_singular = np.linalg.svd(H[:3, :3], compute_uv=False)
    g_threshold = (max(float(g_singular[0]) * TOPOLOGY_RELATIVE_RANK_TOL, 1e-14)
                   if g_singular.size else 1e-14)
    g_rank = int(np.sum(g_singular > g_threshold))
    gg = H[:3, :3]
    cross = H[:3, 3:]
    qq = H[3:, 3:]
    conditional_q = qq - cross.T @ np.linalg.pinv(gg, rcond=1e-12) @ cross
    q_singular = np.linalg.svd(conditional_q, compute_uv=False)
    q_threshold = (max(float(q_singular[0]) * TOPOLOGY_RELATIVE_RANK_TOL, 1e-14)
                   if q_singular.size else 1e-14)
    q_rank = int(np.sum(q_singular > q_threshold))
    return full, g_rank, q_rank, singular


def _constrained_local_q(H, rhs, current_q, qmax):
    """Solve one six-D quadratic with four linear Q inequalities.

    Only Q is constrained.  Eliminating the unconstrained G coordinates first
    reduces the constrained problem to three dimensions and remains stable
    for the rank-deficient one/two-H5 validation problems.
    """
    unconstrained = np.asarray(np.linalg.lstsq(H, rhs, rcond=1e-12)[0], dtype=float)
    limits = np.concatenate((np.full(4, qmax) - BASIS @ current_q,
                             np.full(4, qmax) + BASIS @ current_q))
    unconstrained_objective = float(.5 * unconstrained @ H @ unconstrained - rhs @ unconstrained)
    if np.max(np.abs(BASIS @ (current_q + unconstrained[3:]))) <= qmax + 1e-10:
        return unconstrained, unconstrained_objective, unconstrained_objective, False
    hgg = np.asarray(H[:3, :3], dtype=float)
    hgq = np.asarray(H[:3, 3:], dtype=float)
    bg = np.asarray(rhs[:3], dtype=float)
    bq = np.asarray(rhs[3:], dtype=float)
    hgg_inverse = np.linalg.pinv(hgg, rcond=1e-12)
    reduced_hessian = H[3:, 3:] - hgq.T @ hgg_inverse @ hgq
    reduced_rhs = bq - hgq.T @ hgg_inverse @ bg
    reduced_hessian = .5 * (reduced_hessian + reduced_hessian.T)
    eigenvalues, eigenvectors = np.linalg.eigh(reduced_hessian)
    # Roundoff can make a Schur complement very slightly indefinite.  Preserve
    # the measured quadratic while removing only that numerical negative part.
    reduced_hessian = eigenvectors @ np.diag(np.maximum(eigenvalues, 0.)) @ eigenvectors.T
    constraints = np.vstack((BASIS, -BASIS))
    # This is a convex problem in three variables.  Enumerating the at most
    # three independent active inequalities gives a deterministic small active
    # set solve and avoids optimizer failures on the rank-deficient local
    # topology designs seen in the one/two-H5 stages.
    candidates = []
    feasibility_tolerance = 2e-10 * max(1., qmax)

    def add_candidate(q_delta):
        q_delta = np.asarray(q_delta, dtype=float)
        if not np.all(np.isfinite(q_delta)):
            return
        violation = float(np.max(constraints @ q_delta - limits))
        if violation > feasibility_tolerance:
            return
        # Move a numerically borderline point toward the already feasible
        # current state.  This keeps the final persisted Q inside the guardrail
        # without coordinate-wise clipping.
        if violation > 0.:
            factor = max(0., 1. - violation / max(qmax, 1e-12))
            q_delta *= factor
        g_delta = hgg_inverse @ (bg - hgq @ q_delta)
        candidate = np.concatenate((g_delta, q_delta))
        objective = float(.5 * candidate @ H @ candidate - rhs @ candidate)
        if np.isfinite(objective):
            candidates.append((objective, candidate))

    add_candidate(np.zeros(3, dtype=float))
    add_candidate(np.linalg.lstsq(reduced_hessian, reduced_rhs, rcond=1e-12)[0])
    for active_count in range(1, 4):
        for active_indices in combinations(range(constraints.shape[0]), active_count):
            active_matrix = constraints[list(active_indices)]
            active_limits = limits[list(active_indices)]
            kkt = np.block([
                [reduced_hessian, active_matrix.T],
                [active_matrix, np.zeros((active_count, active_count), dtype=float)],
            ])
            solution = np.linalg.lstsq(
                kkt,
                np.concatenate((reduced_rhs, active_limits)),
                rcond=1e-12,
            )[0]
            add_candidate(solution[:3])
    if not candidates:
        raise ArchitectureError(
            "three-dimensional q-constrained active-set solve found no feasible point"
        )
    objective, candidate = min(candidates, key=lambda value: value[0])
    return candidate, unconstrained_objective, objective, True


def _solve_topology(normals, state, qmax):
    rows = []
    violating = 0; constrained = 0; constrained_objective_change = 0.
    current_active = set()
    # Emit a support record for every H5/physical-IFU state, including states
    # with no usable blank or source measurements.  This makes unsupported
    # topology dimensions auditable instead of silently dropping them from
    # the support artifact.
    all_keys = set(state.g) | set(normals)
    empty_entry = {
        "H": np.zeros((6, 6), dtype=float), "rhs": np.zeros(6, dtype=float),
        "blank_fibers": 0, "blank_equations": 0, "blank_amplifiers": set(),
        "source_rows": 0, "source_amplifiers": set(),
    }
    for key in sorted(all_keys, key=str):
        entry = normals.get(key, empty_entry)
        H = entry["H"]; rhs = entry["rhs"]
        rank, rank_g, rank_q, singular = _rank_topology(H)
        active = rank >= 6 and rank_g >= 3 and rank_q >= 3
        if active:
            current_active.add(key)
            delta = np.linalg.lstsq(H, rhs, rcond=1e-12)[0]
            current_q = state.q.get(key, np.zeros(3))
            unconstrained_final_q = current_q + delta[3:]
            if np.max(np.abs(BASIS @ unconstrained_final_q)) > qmax + 1e-10:
                violating += 1
                candidate, objective_u, objective_c, did_constrain = _constrained_local_q(
                    H, rhs, current_q, qmax)
                if did_constrain:
                    delta = candidate
                    constrained += 1
                    constrained_objective_change += objective_c - objective_u
            state.g[key] = state.g.get(key, np.zeros(3)) + delta[:3]
            state.q[key] = state.q.get(key, np.zeros(3)) + delta[3:]
        rows.append({
            "H5": key[0], "SPECID": key[1][0], "IFUSLOT": key[1][1], "IFUID": key[1][2],
            "blank_fibers": entry["blank_fibers"], "blank_equations": entry["blank_equations"],
            "blank_amplifier_support": ",".join(sorted(entry["blank_amplifiers"])),
            "source_rows": entry["source_rows"],
            "source_amplifier_support": ",".join(sorted(entry["source_amplifiers"])),
            "topology_design_rank": rank, "G_rank": rank_g, "Q_rank_conditional": rank_q,
            "active": active, "G_constrained": bool(rank_g >= 3),
            "Q_constrained": bool(rank_q >= 3), "singular_values": singular.tolist(),
        })
    state.active = current_active
    near = sum(np.any(np.abs(BASIS @ state.q[key]) >= qmax - 1e-7)
               for key in state.active)
    return rows, {
        "active_topology_blocks": len(state.active),
        "unconstrained_violating_blocks": violating,
        "constrained_re_solved_blocks": constrained,
        "ending_near_constraint_blocks": int(near),
        "constrained_objective_change": float(constrained_objective_change),
    }


def _gauge_weights(data, model, state, skies, responses, blank_masks):
    weights = {}
    for item in data:
        response = _full_response(item, model, state)
        s_values, _ = collapse_many(response * skies[item.key][None, :], responses)
        for ifu, indices in _group_indices(item)[0]:
            key = (item.h5_name, tuple(map(int, ifu)))
            selected = blank_masks[item.key][indices]
            if np.any(selected):
                value = float(np.sum(s_values[indices][selected] ** 2))
                weights[key] = weights.get(key, 0.) + value
    return weights


def _gauge_snapshot(data, model, state, count=8):
    snapshot = {}
    for item in data:
        rows = np.arange(min(count, item.row_index.size))
        snapshot[item.key] = (model.z_for(item)[rows, None] +
                              _topology_z_matrix(item, state)[rows])
    return snapshot


def _gauge_center(data, model, state, weights):
    before = _gauge_snapshot(data, model, state)
    by_ifu = {}
    for h5, ifu in weights:
        by_ifu.setdefault(ifu, []).append((h5, weights[(h5, ifu)]))
    transfers = []
    for ifu, members in sorted(by_ifu.items(), key=lambda pair: str(pair[0])):
        denominator = sum(weight for _, weight in members if weight > 0)
        if denominator <= 0:
            continue
        gbar = sum(weight * state.g.get((h5, ifu), np.zeros(3))
                   for h5, weight in members) / denominator
        # The weighted members determine the scientific gauge value, but the
        # gauge transformation itself must cover every H5 state for this
        # persistent IFU.  A zero-blank-support state still contributes to
        # the model prediction and would otherwise see the p_AMP transfer
        # without the compensating subtraction from its local G.
        for h5, state_ifu in state.g:
            if state_ifu == ifu:
                state.g[(h5, state_ifu)] = state.g[(h5, state_ifu)] - gbar
        for amp_index, amp in enumerate(AMP_ORDER):
            amp_key = ifu + (amp,)
            model.p_amp[amp_key] = model.p_amp.get(amp_key, 0.) + float(BASIS[amp_index] @ gbar)
        transfers.append({"IFU": list(ifu), "H5_count": len(members), "Gbar": gbar.tolist(),
                          "weight": float(denominator)})
    after = _gauge_snapshot(data, model, state)
    differences = []
    for key in before:
        differences.append(np.max(np.abs(before[key] - after[key])))
    return {"transfers": transfers,
            "max_prediction_change": float(max(differences) if differences else 0.),
            "prediction_invariant": bool(max(differences, default=0.) <= 2e-12)}


def _sky_update(data, model, state, skies, fq, blank_masks):
    changes = []
    for item in data:
        response = _full_response(item, model, state)
        additive = model.additive_full(item, fq)
        selected = blank_masks[item.key]
        corrected = (np.asarray(item.total, dtype=float)[selected] - additive[selected]) / response[selected]
        new_sky = np.asarray(robust_location(corrected, axis=0), dtype=float)
        old = np.asarray(skies[item.key], dtype=float)
        finite = np.isfinite(new_sky) & np.isfinite(old)
        delta = new_sky[finite] - old[finite]
        denominator = max(float(np.sqrt(np.mean(old[finite] ** 2))), 1e-12) if np.any(finite) else 1.
        skies[item.key] = new_sky
        changes.append({"H5": item.h5_name, "exposure": int(item.exposure),
                        "absolute_rms_change": float(np.sqrt(np.mean(delta ** 2))) if delta.size else np.nan,
                        "fractional_rms_change": float(np.sqrt(np.mean(delta ** 2)) / denominator) if delta.size else np.nan,
                        "blank_fibers": int(selected.sum())})
    return changes


def _leave_one_exposure_out(source_rows, qmax):
    """Measure source topology transfer using two exposures to predict the third."""
    grouped = {}
    for row in source_rows:
        # Fit the predictor with both source bands.  Grouping by band would
        # leave only one effective x_src value and could not test Q.
        group_key = (row["H5"], row["SPECID"], row["IFUSLOT"], row["IFUID"])
        grouped.setdefault(group_key, {}).setdefault(row["exposure"], []).append(row)
    prediction_rows = []
    for group_key, by_exposure in sorted(grouped.items(), key=lambda pair: str(pair[0])):
        exposures = sorted(by_exposure)
        if len(exposures) < 3:
            continue
        for heldout in exposures:
            predictor_rows = [row for exposure in exposures if exposure != heldout
                              for row in by_exposure[exposure]]
            predictor_normals, _ = _source_normal(predictor_rows, None, None)
            predictor_state = type("SourceTopologyState", (), {
                "g": {}, "q": {}, "active": set()})()
            _, _ = _solve_topology(predictor_normals, predictor_state, qmax)
            key = (group_key[0], (group_key[1], group_key[2], group_key[3]))
            if key not in predictor_state.active:
                continue
            heldout_center = np.mean(
                [BASIS[AMP_INDEX[item["AMP"]]] for item in by_exposure[heldout]], axis=0)
            for row in by_exposure[heldout]:
                amp = AMP_INDEX[row["AMP"]]
                b = BASIS[amp] - heldout_center
                predicted = float(
                    b @ predictor_state.g[key] +
                    row["x_src_flat_F_lambda"] * (b @ predictor_state.q[key]))
                residual = float(row["target_log_differential"] - predicted)
                prediction_rows.append({
                    "H5": group_key[0], "heldout_exposure": int(heldout),
                    "SPECID": group_key[1], "IFUSLOT": group_key[2],
                    "IFUID": group_key[3], "AMP": row["AMP"], "band": row["band"],
                    "predicted_differential_log": predicted,
                    "observed_differential_log": row["target_log_differential"],
                    "residual_log": residual,
                    "residual_percent": float(100. * np.expm1(residual)),
                    "predictor_exposures": "+".join(map(str, [e for e in exposures if e != heldout])),
                })
    summary = []
    for band in SOURCE_BANDS:
        for heldout in (1, 2, 3):
            values = np.asarray([
                row["residual_percent"] for row in prediction_rows
                if row["band"] == band and row["heldout_exposure"] == heldout
            ], dtype=float)
            finite = values[np.isfinite(values)]
            summary.append({
                "band": band, "heldout_exposure": heldout, "N": int(finite.size),
                "robust_scatter_percent": (float(robust_scatter(finite))
                                             if finite.size else np.nan),
                "p95_abs_residual_percent": (float(np.percentile(np.abs(finite), 95))
                                              if finite.size else np.nan),
                "fraction_within_5pct": (float(np.mean(np.abs(finite) <= 5.))
                                          if finite.size else np.nan),
            })
    return {"rows": prediction_rows, "summary": summary,
            "definition": "two source exposures fit local G/Q and predict the held-out exposure",
            "population": "final accepted amplifier-level source measurements"}


def _objective_and_diagnostics(data, model, state, skies, fq, responses,
                               blank_masks, source_rows):
    blank_chi2 = 0.; blank_count = 0
    blank_residual = []
    for item in data:
        response = _full_response(item, model, state)
        sky, _ = collapse_many(response * skies[item.key][None, :], responses)
        residual = item.band_total - model.additive_bands(item, fq) - sky
        selected = blank_masks[item.key]
        good = selected[:, None] & np.isfinite(residual) & np.isfinite(item.band_error) & (item.band_error > 0)
        values = residual[good]; errors = item.band_error[good]
        blank_chi2 += float(np.sum((values / errors) ** 2)); blank_count += int(values.size)
        blank_residual.extend(values[::10].tolist())
    source_chi2 = 0.; source_residual = []; active_source_residual = []
    for row in source_rows:
        residual = row["target_log_differential"]
        source_chi2 += (residual / max(row["sigma"], 1e-3)) ** 2
        source_residual.append(residual)
        if row["key"] in state.active:
            active_source_residual.append(residual)
    source_residual = np.asarray(source_residual, dtype=float)
    stitching = {
        "N_source_measurements": int(source_residual.size),
        "median_residual_percent": float(100. * np.median(np.expm1(source_residual))) if source_residual.size else np.nan,
        "robust_scatter_percent": float(100. * robust_scatter(source_residual)) if source_residual.size else np.nan,
        "p95_abs_residual_percent": float(100. * np.percentile(np.abs(np.expm1(source_residual)), 95)) if source_residual.size else np.nan,
    }
    active_source_residual = np.asarray(active_source_residual, dtype=float)
    active_stitching = {
        "N_source_measurements": int(active_source_residual.size),
        "median_residual_percent": (float(100. * np.median(np.expm1(active_source_residual)))
                                     if active_source_residual.size else np.nan),
        "p95_abs_residual_percent": (float(100. * np.percentile(
            np.abs(np.expm1(active_source_residual)), 95))
                                      if active_source_residual.size else np.nan),
        "robust_scatter_percent": (float(100. * robust_scatter(active_source_residual))
                                    if active_source_residual.size else np.nan),
    }
    return {
        "total_objective_half_chi2": float(.5 * (blank_chi2 + source_chi2)),
        "blank_objective_half_chi2": float(.5 * blank_chi2),
        "source_objective_half_chi2": float(.5 * source_chi2),
        "N_blank_equations": blank_count,
        "N_source_measurements": int(source_residual.size),
        "blank_residual_sample_robust_scatter": float(robust_scatter(blank_residual)) if blank_residual else np.nan,
        "source_stitching": stitching,
        "source_stitching_active_topology": active_stitching,
        "inactive_source_measurements": int(source_residual.size - active_source_residual.size),
    }


def _frozen_reproduction(data, model, skies, fq, responses):
    """Evaluate the persisted Model-3 state using its exact current evaluator."""
    state = JointState(model, data)
    masks = {item.key: item.blank_valid.copy() for item in data}
    residual_values = []
    for item in data:
        response = _full_response(item, model, state)
        sky, _ = collapse_many(response * skies[item.key][None, :], responses)
        residual = item.band_total - model.additive_bands(item, fq) - sky
        residual_values.append(residual[masks[item.key]].ravel())
    residual_values = _finite(np.concatenate(residual_values))
    return {
        "equations": int(residual_values.size),
        "blank_residual_robust_scatter": float(validated_m101.robust_scale(residual_values)),
        "reference_model3_equation_count": 6875211,
        "equation_count_matches_reference": bool(residual_values.size == 6875211),
        "model3_state_loaded_without_refit": True,
    }


def _verify_jacobians(data, model, state, skies, responses):
    checks = []
    response = np.asarray(responses, dtype=float)
    for item in data[:min(2, len(data))]:
        ifu_groups, _ = _group_indices(item)
        for ifu, indices in ifu_groups[:2]:
            key = (item.h5_name, tuple(map(int, ifu)))
            if not np.any(item.blank_valid[indices]):
                continue
            native = int(indices[np.flatnonzero(item.blank_valid[indices])[0]])
            amp = AMP_INDEX[str(item.amp[native])]
            baseline = _full_response(item, model, state)[native] * skies[item.key]
            b = BASIS[amp]; epsilon = 1e-6
            for coordinate, label in enumerate(("G_LR", "G_UD", "G_I")):
                plus = baseline * np.exp(epsilon * b[coordinate])
                minus = baseline * np.exp(-epsilon * b[coordinate])
                fd = np.asarray([(collapse(plus[None, :], r)[0][0] - collapse(minus[None, :], r)[0][0]) /
                                  (2. * epsilon) for r in response])
                exact = np.asarray([collapse(baseline[None, :], r)[0][0] for r in response]) * b[coordinate]
                checks.append({"H5": item.h5_name, "IFU": list(map(int, ifu)), "AMP": AMP_ORDER[amp],
                               "parameter": label, "max_absolute_error": float(np.nanmax(np.abs(fd - exact))),
                               "pass": bool(np.allclose(fd, exact, rtol=2e-8, atol=2e-10))})
            for coordinate, label in enumerate(("Q_LR", "Q_UD", "Q_I")):
                plus = baseline * np.exp(epsilon * X_LAMBDA * b[coordinate])
                minus = baseline * np.exp(-epsilon * X_LAMBDA * b[coordinate])
                fd = np.asarray([(collapse(plus[None, :], r)[0][0] - collapse(minus[None, :], r)[0][0]) /
                                  (2. * epsilon) for r in response])
                exact = np.asarray([collapse((baseline * X_LAMBDA)[None, :], r)[0][0] for r in response]) * b[coordinate]
                checks.append({"H5": item.h5_name, "IFU": list(map(int, ifu)), "AMP": AMP_ORDER[amp],
                               "parameter": label, "max_absolute_error": float(np.nanmax(np.abs(fd - exact))),
                               "pass": bool(np.allclose(fd, exact, rtol=2e-8, atol=2e-10))})
            break
        if checks:
            break
    return {"checks": checks, "pass": bool(checks) and all(row["pass"] for row in checks),
            "definition": "s_b=collapse(M0*S,response_b); u_b=collapse(x_lambda*M0*S,response_b)"}


def _write_corrections(output_dir, data, model, state):
    keys = sorted(state.g, key=str)
    corrections = np.zeros((len(keys), 4, N_WAVE), dtype=float)
    rows = []
    for state_index, key in enumerate(keys):
        ifu = key[1]
        for amp_index, amp in enumerate(AMP_ORDER):
            p_amp = model.p_amp.get(ifu + (amp,), 0.)
            corrections[state_index, amp_index] = np.exp(
                p_amp + BASIS[amp_index] @ state.g.get(key, np.zeros(3)) +
                X_LAMBDA * (BASIS[amp_index] @ state.q.get(key, np.zeros(3))))
            rows.append({"state_index": state_index, "H5": key[0], "SPECID": ifu[0],
                         "IFUSLOT": ifu[1], "IFUID": ifu[2], "AMP": amp,
                         "G_LR": state.g.get(key, np.zeros(3))[0],
                         "G_UD": state.g.get(key, np.zeros(3))[1],
                         "G_I": state.g.get(key, np.zeros(3))[2],
                         "Q_LR": state.q.get(key, np.zeros(3))[0],
                         "Q_UD": state.q.get(key, np.zeros(3))[1],
                         "Q_I": state.q.get(key, np.zeros(3))[2],
                         "correction_3500": corrections[state_index, amp_index, np.argmin(abs(WAVE - 3500))],
                         "correction_4500": corrections[state_index, amp_index, np.argmin(abs(WAVE - 4500))],
                         "correction_5500": corrections[state_index, amp_index, np.argmin(abs(WAVE - 5500))],
                         "max_end_to_end_topology_change": float(np.exp(
                             np.max(X_LAMBDA) * (BASIS[amp_index] @ state.q.get(key, np.zeros(3))) -
                             np.min(X_LAMBDA) * (BASIS[amp_index] @ state.q.get(key, np.zeros(3)))) - 1.)})
    np.savez_compressed(output_dir / "m101_joint_amplifier_corrections.npz",
                        wavelength_A=WAVE, h5=np.asarray([key[0] for key in keys], dtype="U32"),
                        ifu=np.asarray([key[1] for key in keys], dtype=int),
                        amplifier=np.asarray(AMP_ORDER, dtype="U2"), correction=corrections)
    _write_rows(output_dir / "m101_joint_amplifier_corrections_index.csv", rows, list(rows[0]) if rows else [])
    return {"array_file": str(output_dir / "m101_joint_amplifier_corrections.npz"),
            "index_file": str(output_dir / "m101_joint_amplifier_corrections_index.csv"),
            "states": len(keys), "wavelengths": N_WAVE}


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5", nargs="+", required=True)
    parser.add_argument("--product", required=True, help="frozen Model-3 product JSON")
    parser.add_argument("--blank-file", required=True)
    parser.add_argument("--external-cache")
    parser.add_argument("--compact-mask")
    parser.add_argument("--on-filter", required=True)
    parser.add_argument("--off-filter", required=True)
    parser.add_argument("--fq-template", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--stage", choices=("frozen", "jacobian", "one-h5", "two-h5", "full"), default="full")
    parser.add_argument("--minimum-finite-fraction", type=float, default=.8)
    parser.add_argument("--minimum-source-fibers", type=int, default=10)
    parser.add_argument("--source-sigma-floor", type=float, default=.01)
    parser.add_argument("--max-outer-passes", type=int, default=5)
    parser.add_argument("--outer-tolerance", type=float, default=1e-5)
    parser.add_argument("--q-limit-percent", type=float, choices=(10., 15.), default=10.,
                        help="production is 10%%; 15%% is an explicit sensitivity run")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = _parse_args()
    if args.max_outer_passes < 1 or args.minimum_source_fibers < 1 or args.source_sigma_floor <= 0:
        raise SystemExit("invalid iteration or source controls")
    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise SystemExit("output directory is nonempty; use --overwrite: %s" % output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    timings = {}
    progress = lambda message: print("[joint +%.1fs] %s" % (time.perf_counter() - started, message), flush=True)

    product, warm_model, all_skies = load_model_and_skies(args.product, progress=progress)
    supplied_h5 = discover_h5(args.h5, development=True)
    expected_names = {Path(item["filename"]).name for item in product["provenance"]["input_h5"]}
    if not {path.name for path in supplied_h5}.issubset(expected_names):
        raise ValueError("supplied H5 set is not a subset of the frozen Model-3 product")
    if args.stage == "one-h5":
        selected_paths = supplied_h5[:1]
    elif args.stage == "two-h5":
        selected_paths = supplied_h5[:2]
    else:
        selected_paths = supplied_h5
    blank_masks_all, blank_provenance = m101_blank_fibers.load(args.blank_file, selected_paths)
    fq = validated_m101.load_fq(args.fq_template)
    # Native loading owns null-band response validation and returns all seven.
    data, input_provenance = load_native_data(
        selected_paths, blank_masks_all, args.on_filter, args.off_filter,
        minimum_finite_fraction=args.minimum_finite_fraction, timings=timings,
        progress=progress)
    responses = np.asarray(input_provenance["responses"], dtype=float)
    skies = {key: np.asarray(all_skies[key], dtype=float).copy() for key in (item.key for item in data)}
    layout = make_layout(data, "model3")
    model = _copy_model(warm_model, layout)
    state = JointState(model, data)
    state.layout = layout
    reproduction = _frozen_reproduction(data, model, skies, fq, responses)
    (output_dir / "frozen_model3_reproduction.json").write_text(json.dumps(_json_ready(reproduction), indent=2, sort_keys=True))
    progress("frozen Model-3 reproduction: %s" % json.dumps(_json_ready(reproduction), sort_keys=True))
    if args.stage == "frozen":
        return
    jacobian = _verify_jacobians(data, model, state, skies, responses)
    (output_dir / "topology_jacobian_finite_difference.json").write_text(json.dumps(_json_ready(jacobian), indent=2, sort_keys=True))
    if not jacobian["pass"]:
        raise ArchitectureError("exact full-wavelength topology Jacobian finite-difference gate failed")
    if args.stage == "jacobian":
        return
    if not args.external_cache or not args.compact_mask:
        raise SystemExit("topology stages require --external-cache and --compact-mask")
    external, external_provenance = _load_external_subset(args.external_cache, selected_paths)
    mask_data, mask_wcs = m101_compact_mask.load(args.compact_mask)
    source_masks, group_records = _candidate_population(
        data, external, mask_data, mask_wcs, model, skies, fq, responses,
        args.minimum_source_fibers)
    membership_rows, membership_counts = _membership_artifact(data, source_masks, group_records)
    _write_rows(output_dir / "m101_joint_source_membership_groups.csv", membership_rows,
                list(membership_rows[0]) if membership_rows else [])
    membership_payload = {
        "freeze_stage": "persisted frozen Model-3 warm start",
        "candidate_definition": "existing external-valid + compact-mask + hardware + finite/error + 5-sigma external X",
        "accepted_source_definition": "significant candidate and warm-start amplifier robust_scatter <= 0.15",
        "rejected_source_definition": "significant candidate and warm-start amplifier robust_scatter > 0.15; excluded from blanks and source likelihood",
        "pure_blank_definition": "not a source candidate and existing blank_valid",
        "counts": membership_counts,
        "groups": membership_rows,
    }
    (output_dir / "m101_joint_source_membership.json").write_text(json.dumps(_json_ready(membership_payload), indent=2, sort_keys=True))
    blank_masks = _make_blank_masks(data, source_masks)
    qmax = QMAX_15 if args.q_limit_percent == 15. else QMAX_10
    null_info = _warm_model3_null_space(data, skies, fq, responses)
    null_payload = {key: value for key, value in null_info.items()
                    if key not in ("layout", "row_space_basis", "null_space_basis")}
    null_payload["rank_preserved_by_fixed_projector"] = True
    (output_dir / "model3_nonq_null_space.json").write_text(json.dumps(_json_ready(null_payload), indent=2, sort_keys=True))
    progress("Model-3 non-q Schur rank=%d/%d; nullity=%d" %
             (null_info["rank"], null_info["columns"], null_info["null_space_basis"].shape[1]))
    if args.stage in ("one-h5", "two-h5"):
        state.active = set()
    gauge_weights = _gauge_weights(data, model, state, skies, responses, blank_masks)
    (output_dir / "source_effective_wavelength.json").write_text(json.dumps(_json_ready({
        "definition": "response-weighted effective x_lambda assuming flat F_lambda",
        "diagnostic": "flat F_nu reweighting is lambda^-2",
        "values": _source_effective_x(responses)},), indent=2, sort_keys=True))
    iteration_history = []
    previous_objective = np.inf
    for outer in range(1, args.max_outer_passes + 1):
        progress("outer pass %d/%d: updating base Model-3 block" % (outer, args.max_outer_passes))
        pass_started = time.perf_counter()
        base_result = _base_update(data, model, state, skies, fq, responses,
                                   blank_masks, null_info, timings, progress)
        source_started = time.perf_counter()
        source_rows, effective_x = _recompute_source_measurements(
            data, external, source_masks,
            {key for key, value in group_records.items() if value["eligible_source"]},
            model, state, skies, fq, responses, args.source_sigma_floor)
        timings.setdefault("source_measurement_reconstruction_seconds", 0.)
        timings["source_measurement_reconstruction_seconds"] += time.perf_counter() - source_started
        source_normals, source_scale = _source_normal(source_rows, model, state)
        blank_scales, _ = _blank_eval(data, model, state, skies, fq, responses, blank_masks)
        blank_normals, _ = _blank_eval(data, model, state, skies, fq, responses,
                                       blank_masks, need_normals=True, scales=blank_scales)
        for key, entry in source_normals.items():
            if key not in blank_normals:
                blank_normals[key] = entry
            else:
                blank_normals[key]["H"] += entry["H"]
                blank_normals[key]["rhs"] += entry["rhs"]
                blank_normals[key]["source_rows"] += entry["source_rows"]
                blank_normals[key]["source_amplifiers"].update(entry["source_amplifiers"])
        topology_started = time.perf_counter()
        support_rows, topology_summary = _solve_topology(blank_normals, state, qmax)
        timings.setdefault("topology_solve_seconds", 0.)
        timings["topology_solve_seconds"] += time.perf_counter() - topology_started
        _write_rows(output_dir / "m101_joint_topology_support.csv", support_rows,
                    list(support_rows[0]) if support_rows else [])
        gauge_started = time.perf_counter()
        gauge = _gauge_center(data, model, state, gauge_weights)
        timings.setdefault("gauge_recenter_seconds", 0.)
        timings["gauge_recenter_seconds"] += time.perf_counter() - gauge_started
        sky_started = time.perf_counter()
        sky_changes = _sky_update(data, model, state, skies, fq, blank_masks)
        timings.setdefault("sky_update_seconds", 0.)
        timings["sky_update_seconds"] += time.perf_counter() - sky_started
        objective = _objective_and_diagnostics(data, model, state, skies, fq, responses,
                                               blank_masks, source_rows)
        max_base_change = base_result["increment_norm"]
        max_sky_change = max((row["fractional_rms_change"] for row in sky_changes), default=0.)
        record = {
            "outer_pass": outer, "base": base_result,
            "source_rows": len(source_rows), "source_scale": source_scale,
            "topology": topology_summary, "gauge": gauge, "sky": sky_changes,
            "objective": objective, "max_base_increment_norm": max_base_change,
            "max_sky_fractional_change": max_sky_change,
            "pass_seconds": time.perf_counter() - pass_started,
        }
        iteration_history.append(record)
        progress("outer pass %d objective=%.6g blank=%.6g source=%.6g active=%d q-bound=%d"
                 % (outer, objective["total_objective_half_chi2"],
                    objective["blank_objective_half_chi2"], objective["source_objective_half_chi2"],
                    topology_summary["active_topology_blocks"],
                    topology_summary["ending_near_constraint_blocks"]))
        current_objective = objective["total_objective_half_chi2"]
        relative_objective_change = abs(previous_objective - current_objective) / max(abs(previous_objective), 1.)
        record["relative_objective_change"] = relative_objective_change
        if (outer > 1 and relative_objective_change <= args.outer_tolerance and
                max_base_change <= args.outer_tolerance and max_sky_change <= args.outer_tolerance):
            record["converged"] = True
            break
        previous_objective = current_objective
    else:
        if iteration_history:
            iteration_history[-1]["converged"] = False

    # ``support_rows`` was produced by the last topology pass.  Do not solve
    # again here: doing so would apply a second topology increment after the
    # final sky/gauge update.
    _write_rows(output_dir / "m101_joint_topology_support.csv", support_rows,
                list(support_rows[0]) if support_rows else [])
    correction_product = _write_corrections(output_dir, data, model, state)
    source_rows_final, effective_x = _recompute_source_measurements(
        data, external, source_masks,
        {key for key, value in group_records.items() if value["eligible_source"]},
        model, state, skies, fq, responses, args.source_sigma_floor)
    final_objective = _objective_and_diagnostics(
        data, model, state, skies, fq, responses, blank_masks, source_rows_final)
    source_measurement_rows = []
    for row in source_rows_final:
        record = dict(row)
        record["topology_active"] = row["key"] in state.active
        source_measurement_rows.append(record)
    _write_rows(output_dir / "m101_joint_source_measurements.csv",
                source_measurement_rows,
                list(source_measurement_rows[0]) if source_measurement_rows else [])
    leave_one_exposure_out = _leave_one_exposure_out(source_rows_final, qmax)
    _write_rows(output_dir / "m101_joint_leave_one_exposure_out.csv",
                leave_one_exposure_out["rows"],
                list(leave_one_exposure_out["rows"][0])
                if leave_one_exposure_out["rows"] else [])
    final_q = [float(np.max(np.abs(BASIS @ value))) for value in state.q.values()]
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "architecture": "block-coordinate Model-3/base + local six-parameter H5/physical-IFU topology",
        "reference_fitter_unchanged": True,
        "global_sparse_lsmr_used": False,
        "base_model3_block": {
            "non_q_columns": int(null_info["columns"]),
            "q_columns": int(layout.size - null_info["columns"]),
            "total_columns": int(layout.size),
            "solver": "frozen diagonal-q Schur with fixed warm-start null-space projector",
        },
        "topology_basis": BASIS.tolist(),
        "x_lambda": "(lambda - 4500 A) / 1000 A",
        "q_bound": {"percent": args.q_limit_percent, "qmax": qmax,
                    "definition": "abs(B_a dot Q) <= log(1.10)/2 in production"},
        "topology_rank_relative_tolerance": TOPOLOGY_RELATIVE_RANK_TOL,
        "outer_iteration_history": iteration_history,
        "final_objective": final_objective,
        "final_q_max": float(max(final_q, default=0.)),
        "final_q_respects_bound": bool(max(final_q, default=0.) <= qmax + 1e-10),
        "source_effective_wavelength": effective_x,
        "leave_one_exposure_out": {
            "definition": leave_one_exposure_out["definition"],
            "population": leave_one_exposure_out["population"],
            "summary": leave_one_exposure_out["summary"],
        },
        "source_membership_counts": membership_counts,
        "active_topology_blocks": len(state.active),
        "topology_support_file": str(output_dir / "m101_joint_topology_support.csv"),
        "correction_product": correction_product,
        "model_parameters": {
            "p_IFU_log": {str(key): value for key, value in model.p_ifu.items()},
            "p_AMP_log": {str(key): value for key, value in model.p_amp.items()},
            "ax": {str(key): value for key, value in model.ax.items()},
            "ay": {str(key): value for key, value in model.ay.items()},
            "alpha_q": {str(key): value for key, value in model.alpha_q.items()},
            "G": {str(key): value.tolist() for key, value in state.g.items()},
            "Q": {str(key): value.tolist() for key, value in state.q.items()},
        },
        "skies": {str(key): value.tolist() for key, value in skies.items()},
        "timings": timings,
        "provenance": {
            "product": file_identity(args.product), "product_h5": product["provenance"]["input_h5"],
            "blank_file": file_identity(args.blank_file), "blank_file_sha256": small_file_hash(args.blank_file),
            "blank_loader": blank_provenance, "native_loader": input_provenance,
            "external_cache": external_provenance, "compact_mask": file_identity(args.compact_mask),
            "on_filter": file_identity(args.on_filter), "off_filter": file_identity(args.off_filter),
            "fq_template": file_identity(args.fq_template),
        },
    }
    arrays_path = output_dir / "m101_calibration_joint_arrays.npz"
    np.savez_compressed(arrays_path, wavelength=WAVE,
                        **{"sky_%s__%d" % key: value for key, value in skies.items()})
    metadata["arrays"] = str(arrays_path)
    (output_dir / "m101_calibration_joint_product.json").write_text(
        json.dumps(_json_ready(metadata), indent=2, sort_keys=True))
    (output_dir / "timing_breakdown.json").write_text(json.dumps(_json_ready({
        "base_Model3_update": timings.get("base_model3_update_seconds", 0.),
        "source_measurement_reconstruction": timings.get("source_measurement_reconstruction_seconds", 0.),
        "topology_solves": timings.get("topology_solve_seconds", 0.),
        "constrained_Q_resolves": sum(row["topology"]["constrained_re_solved_blocks"] for row in iteration_history),
        "sky_update": timings.get("sky_update_seconds", 0.),
        "diagnostics": time.perf_counter() - started - timings.get("base_model3_update_seconds", 0.) -
                       timings.get("source_measurement_reconstruction_seconds", 0.) -
                       timings.get("topology_solve_seconds", 0.) - timings.get("sky_update_seconds", 0.),
        "all_timings": timings,
    }), indent=2, sort_keys=True))
    progress("wrote joint product to %s" % output_dir)


if __name__ == "__main__":
    main()
