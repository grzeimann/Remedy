#!/usr/bin/env python3
"""Physics, identifiability, and initialization audit for the M101 joint fit.

This is deliberately a consumer of the persisted products.  It loads the
frozen Model-3 state, the final joint state, and the existing native/external
measurements, but never calls a production fitting loop or writes a
calibration product.  All files are written below ``--output-dir``.

The audit uses two complementary residual representations:

* corrected blank residuals, ``(D-A)/M-S`` (spectral and collapsed-band);
* an exact collapsed-band amplifier residual,
  ``d=(D_band-A_band-collapse(MS))/collapse(MS)``.

The latter is the dimensionless amplifier datum used by the diagnostic
initialization prototype.  It is a first-order response residual, not a new
scientific model or a production weight change.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import time
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import diagnose_m101_hierarchical as validated_m101
from diagnose_m101_post_model3 import (
    _group_indices, _parse_map, load_model_and_skies,
)
from fit_m101_calibration_v2 import FittedModel
import fit_m101_calibration_joint as production_joint
import m101_blank_fibers
import m101_compact_mask
import m101_external_measurements
from m101_calibration_utils import (
    ALL_BANDS, collapse_many, robust_location, robust_scatter,
)
from m101_native_data import discover_h5, load as load_native_data


AMP_ORDER = tuple(production_joint.AMP_ORDER)
AMP_INDEX = dict(production_joint.AMP_INDEX)
BASIS = np.asarray(production_joint.BASIS, dtype=float)
WAVE = np.asarray(production_joint.WAVE, dtype=float)
X_LAMBDA = np.asarray(production_joint.X_LAMBDA, dtype=float)
QMAX_10 = float(production_joint.QMAX_10)
RANK_TOL = float(production_joint.TOPOLOGY_RELATIVE_RANK_TOL)
SOURCE_BANDS = ("ON", "OFF")
SCHEMA = "m101_joint_physics_identifiability_audit_v1"


def _json_ready(value):
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _write_json(path, value):
    Path(path).write_text(json.dumps(_json_ready(value), indent=2, sort_keys=True))


def _write_rows(path, rows):
    rows = list(rows)
    fields = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
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


def _finite(values):
    values = np.asarray(values, dtype=float)
    return values[np.isfinite(values)]


def _summary(values):
    values = _finite(values)
    if not values.size:
        return {"N": 0, "location": np.nan, "scatter": np.nan,
                "se_location": np.nan, "p95_abs": np.nan,
                "p99_abs": np.nan, "median": np.nan, "rms": np.nan}
    location = float(robust_location(values))
    scatter = float(robust_scatter(values))
    return {
        "N": int(values.size), "location": location, "scatter": scatter,
        "se_location": (scatter / math.sqrt(values.size)
                         if np.isfinite(scatter) and values.size else np.nan),
        "p95_abs": float(np.percentile(np.abs(values), 95)),
        "p99_abs": float(np.percentile(np.abs(values), 99)),
        "median": float(np.median(values)),
        "rms": float(np.sqrt(np.mean(values ** 2))),
    }


def _corr(left, right):
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    use = np.isfinite(left) & np.isfinite(right)
    if np.sum(use) < 3 or np.std(left[use]) <= 0 or np.std(right[use]) <= 0:
        return np.nan
    return float(np.corrcoef(left[use], right[use])[0, 1])


def _key(h5, ifu):
    return (str(h5), tuple(map(int, ifu)))


def _ifu_text(ifu):
    return "%d/%d/%d" % tuple(map(int, ifu))


def _parse_model_parameters(payload, model_parameters):
    p = model_parameters
    model = FittedModel(
        "model3", None,
        _parse_map(p["p_IFU_log"]), _parse_map(p["p_AMP_log"]),
        _parse_map(p["ax"]), _parse_map(p["ay"]), _parse_map(p["alpha_q"]),
        0, 6)
    g = {ast.literal_eval(k): np.asarray(v, dtype=float)
         for k, v in p["G"].items()}
    q = {ast.literal_eval(k): np.asarray(v, dtype=float)
         for k, v in p["Q"].items()}
    arrays_path = Path(payload["arrays"])
    if not arrays_path.is_absolute():
        arrays_path = Path(payload["_path"]).parent / arrays_path
    arrays = np.load(arrays_path)
    skies = {}
    for key in g:
        h5, ifu = key
        # Joint arrays are keyed by exposure, not IFU.  The key list is
        # recovered from the persisted base product below when needed.
        del ifu
    for array_key in arrays.files:
        if not array_key.startswith("sky_"):
            continue
        stem = array_key[4:]
        h5, exposure = stem.rsplit("__", 1)
        skies[(h5, int(exposure))] = np.asarray(arrays[array_key], dtype=float)
    return model, g, q, skies


def _topology(item, g, q):
    values = np.zeros((item.row_index.size, WAVE.size), dtype=float)
    for ifu, indices in _group_indices(item)[0]:
        state_key = _key(item.h5_name, ifu)
        gv = np.asarray(g.get(state_key, np.zeros(3)), dtype=float)
        qv = np.asarray(q.get(state_key, np.zeros(3)), dtype=float)
        rows = np.asarray(indices, dtype=int)
        amp_values = np.asarray([BASIS[AMP_INDEX[str(a)]] for a in item.amp[rows]])
        values[rows] = (amp_values @ gv)[:, None] + (amp_values @ qv)[:, None] * X_LAMBDA[None, :]
    return values


def _model_full_terms(item, model, g, q, sky, fq, responses):
    """Return response and collapsed additive/sky terms for one exposure."""
    z = model.z_for(item) + _topology(item, g, q).sum(axis=1) * 0.0
    # The expression above keeps the topology construction visibly separate;
    # the actual topology is added below without allocating a second model.
    topo = _topology(item, g, q)
    response = np.exp(model.z_for(item)[:, None] + topo)
    sky_band, _ = collapse_many(np.asarray(sky, dtype=float)[None, :], responses)
    return response, sky_band[0]


class ValueAccumulator:
    """Small grouped store for robust summaries, kept separate by band."""

    def __init__(self):
        self.values = defaultdict(list)
        self.fractions = defaultdict(list)

    def add(self, state, selection, grouping, label, band, values, fractions):
        key = (state, selection, grouping, str(label), str(band))
        values = _finite(values)
        fractions = _finite(fractions)
        if values.size:
            self.values[key].extend(values.tolist())
        if fractions.size:
            self.fractions[key].extend(fractions.tolist())

    def rows(self):
        result = []
        for key in sorted(self.values, key=str):
            state, selection, grouping, label, band = key
            stats = _summary(self.values[key])
            frac = _summary(self.fractions[key])
            result.append({
                "state": state, "selection": selection, "grouping": grouping,
                "group": label, "band": band, **stats,
                "fraction_location": frac["location"],
                "fraction_scatter": frac["scatter"],
                "fraction_se_location": frac["se_location"],
                "fraction_p95_abs": frac["p95_abs"],
            })
        return result


def _add_grouped(acc, state, selection, group_arrays, band, values, fractions):
    for grouping, labels in group_arrays:
        labels = np.asarray(labels)
        if values.size != labels.size:
            if labels.size and values.size % labels.size == 0:
                # ``ALL_BANDS`` is flattened row-major, so each fiber label
                # is repeated for its seven band values.
                labels = np.repeat(labels, values.size // labels.size)
            else:
                raise ValueError("group labels and values have incompatible sizes")
        for label in np.unique(labels):
            use = labels == label
            acc.add(state, selection, grouping, str(label), band,
                    values[use], fractions[use])


def _state_group_arrays(item, q_bins, x_bins, y_bins):
    h5 = item.h5_name
    ifus = np.asarray([_ifu_text(v) for v in item.ifu], dtype=object)
    return [
        ("overall", np.full(item.row_index.size, "ALL", dtype=object)),
        ("H5", np.full(item.row_index.size, h5, dtype=object)),
        ("exposure", np.full(item.row_index.size, str(item.exposure), dtype=object)),
        ("H5_exposure", np.full(item.row_index.size, "%s/e%d" % (h5, item.exposure), dtype=object)),
        ("physical_IFU", np.asarray([h5 + "/" + value for value in ifus], dtype=object)),
        ("amplifier", np.asarray(item.amp, dtype=object)),
        ("q", np.asarray(item.q, dtype=int)),
        ("x_bin", np.asarray(x_bins, dtype=int)),
        ("y_bin", np.asarray(y_bins, dtype=int)),
    ]


def _load_joint_payload(path):
    payload = json.loads(Path(path).read_text())
    payload["_path"] = str(Path(path).resolve())
    return payload


def _load_support(path):
    rows = {}
    with Path(path).open() as stream:
        for row in csv.DictReader(stream):
            key = _key(row["H5"], (row["SPECID"], row["IFUSLOT"], row["IFUID"]))
            try:
                singular = np.asarray(json.loads(row["singular_values"]), dtype=float)
            except (TypeError, ValueError, json.JSONDecodeError):
                singular = np.zeros(6, dtype=float)
            rows[key] = {
                **row,
                "singular_values": singular,
                "active": row["active"].lower() == "true",
            }
    return rows


def _state_model_value(model, item, g, q):
    topo = _topology(item, g, q)
    return np.exp(model.z_for(item)[:, None] + topo)


def _source_rows_for_state(data, model, g, q, skies, fq, responses,
                           source_masks, accepted_groups, source_sigma_floor):
    rows = []
    source_responses = np.asarray(responses)[[5, 6]]
    effective_x = production_joint._source_effective_x(responses)
    for item in data:
        response = _state_model_value(model, item, g, q)
        additive = model.additive_full(item, fq)
        corrected, _ = collapse_many((np.asarray(item.total, dtype=float) - additive) / response,
                                     source_responses)
        sky_band, _ = collapse_many(np.asarray(skies[item.key], dtype=float)[None, :], source_responses)
        _, amplifier_groups = _group_indices(item)
        for (ifu, amp), indices in amplifier_groups:
            ifu = tuple(map(int, ifu)); amp = str(amp)
            for band_index, band in enumerate(SOURCE_BANDS):
                key = (item.h5_name, int(item.exposure), *ifu, amp, band)
                if key not in accepted_groups:
                    continue
                selected = np.asarray(source_masks[item.key][band][indices], dtype=bool)
                cache = accepted_groups["__external__"][item.h5_name]
                x_values = (cache["global_g"][band] *
                             cache["external_object"][(item.exposure, band)])[indices][selected]
                y_values = (corrected[indices, band_index] - sky_band[0, band_index])[selected]
                valid = np.isfinite(x_values) & np.isfinite(y_values) & (x_values > 0)
                ratios = np.asarray(y_values[valid] / x_values[valid], dtype=float)
                if not ratios.size:
                    continue
                log_values = np.log(ratios[ratios > 0])
                if not log_values.size:
                    continue
                linear_scatter = float(robust_scatter(ratios))
                log_scatter = float(robust_scatter(log_values))
                row = {
                    "state": "", "key": _key(item.h5_name, ifu),
                    "H5": item.h5_name, "exposure": int(item.exposure),
                    "SPECID": ifu[0], "IFUSLOT": ifu[1], "IFUID": ifu[2],
                    "AMP": amp, "band": band,
                    "N_source_measurements": int(ratios.size),
                    "raw_ratio": float(robust_location(ratios)),
                    "log_ratio": float(robust_location(log_values)),
                    "robust_scatter_linear": linear_scatter,
                    "robust_scatter_log": log_scatter,
                    "ratio_uncertainty": (linear_scatter / math.sqrt(ratios.size)
                                           if np.isfinite(linear_scatter) else np.nan),
                    "linear_location_se": (linear_scatter / math.sqrt(ratios.size)
                                            if np.isfinite(linear_scatter) else np.nan),
                    "log_location_se": (log_scatter / math.sqrt(log_values.size)
                                         if np.isfinite(log_scatter) else np.nan),
                    "sigma_existing": max(linear_scatter if np.isfinite(linear_scatter) else source_sigma_floor,
                                           source_sigma_floor),
                    "x_src_flat_F_lambda": effective_x[band]["flat_F_lambda"],
                    "x_src_flat_F_nu": effective_x[band]["flat_F_nu"],
                    "ratios": ratios,
                }
                rows.append(row)
    by_group = defaultdict(list)
    for row in rows:
        by_group[(row["H5"], row["exposure"], row["SPECID"], row["IFUSLOT"],
                  row["IFUID"], row["band"])].append(row)
    for group_rows in by_group.values():
        if len(group_rows) < 2:
            continue
        common = float(robust_location([row["log_ratio"] for row in group_rows]))
        amps = np.asarray([BASIS[AMP_INDEX[row["AMP"]]] for row in group_rows])
        center = np.mean(amps, axis=0)
        for row in group_rows:
            row["common_mode_log"] = common
            row["target_log_differential"] = row["log_ratio"] - common
            row["group_amp_support"] = len(group_rows)
            row["group_basis_center_LR"] = center[0]
            row["group_basis_center_UD"] = center[1]
            row["group_basis_center_I"] = center[2]
    return [row for row in rows if "target_log_differential" in row]


def _blank_amp_summary(accumulator, state):
    rows = []
    for (row_state, h5, exposure, specid, slot, ifuid, amp, band), values in sorted(
            accumulator.items(), key=str):
        if row_state != state:
            continue
        d = _finite(values["d"])
        x = _finite(values["x"])
        stats = _summary(d)
        rows.append({
            "state": state, "H5": h5, "exposure": int(exposure),
            "SPECID": int(specid), "IFUSLOT": int(slot), "IFUID": int(ifuid),
            "AMP": amp, "band": band, "N_fibers": int(d.size), **stats,
            "d_location": stats["location"], "d_scatter": stats["scatter"],
            "location_se": stats["se_location"],
            "effective_x_location": (float(robust_location(x)) if x.size else np.nan),
            "effective_x_scatter": (float(robust_scatter(x)) if x.size else np.nan),
        })
    return rows


def _normal_entry(normal, key):
    if key not in normal:
        normal[key] = {"H": np.zeros((6, 6), dtype=float),
                       "rhs": np.zeros(6, dtype=float),
                       "x_num": 0., "x_den": 0.,
                       "blank_fibers": 0, "blank_equations": 0,
                       "source_rows": 0, "source_amplifiers": set(),
                       "blank_amplifiers": set()}
    return normal[key]


def _add_normal_design(entry, design, y, sigma, x=None):
    design = np.asarray(design, dtype=float)
    y = np.asarray(y, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    if design.ndim == 1:
        design = design[None, :]
    if y.ndim == 0:
        y = y[None]
    if sigma.ndim == 0:
        sigma = sigma[None]
    use = np.all(np.isfinite(design), axis=1) & np.isfinite(y) & np.isfinite(sigma) & (sigma > 0)
    if not np.any(use):
        return
    d = design[use]; target = y[use]; weight = 1. / sigma[use] ** 2
    entry["H"] += (d.T * weight) @ d
    entry["rhs"] += d.T @ (weight * target)
    if x is not None:
        x = np.asarray(x, dtype=float)
        if x.ndim == 0:
            x = np.full(y.shape, float(x), dtype=float)
        x = x[use]
        norm = np.sum(d[:, :3] ** 2, axis=1)
        entry["x_num"] += float(np.sum(weight * norm * x))
        entry["x_den"] += float(np.sum(weight * norm))


def _ranks(H):
    H = np.asarray(H, dtype=float)
    singular = np.linalg.svd(H, compute_uv=False)
    threshold = max(float(singular[0]) * RANK_TOL, 1e-14) if singular.size else 1e-14
    full = int(np.sum(singular > threshold))
    gg = H[:3, :3]
    gsv = np.linalg.svd(gg, compute_uv=False)
    gt = max(float(gsv[0]) * RANK_TOL, 1e-14) if gsv.size else 1e-14
    grank = int(np.sum(gsv > gt))
    cross = H[:3, 3:]
    qq = H[3:, 3:]
    qcond = .5 * (qq - cross.T @ np.linalg.pinv(gg, rcond=1e-12) @ cross)
    qcond += .5 * (qq - cross.T @ np.linalg.pinv(gg, rcond=1e-12) @ cross).T
    qsv = np.linalg.svd(qcond, compute_uv=False)
    qt = max(float(qsv[0]) * RANK_TOL, 1e-14) if qsv.size else 1e-14
    qrank = int(np.sum(qsv > qt))
    return {"singular": singular, "rank": full, "g_singular": gsv,
            "g_rank": grank, "q_conditional": qcond, "q_singular": qsv,
            "q_rank": qrank}


def _condition(values, rank=None):
    values = _finite(values)
    if not values.size:
        return np.nan
    if rank is None:
        rank = values.size
    positive = values[:max(1, min(rank, values.size))]
    positive = positive[positive > 0]
    return float(positive[0] / positive[-1]) if positive.size else np.nan


def _canonical(a, b, cross):
    """Canonical cosines between the two design subspaces."""
    def invsqrt(matrix):
        values, vectors = np.linalg.eigh(.5 * (matrix + matrix.T))
        keep = values > max(float(np.max(values, initial=0.)) * 1e-12, 1e-14)
        if not np.any(keep):
            return np.zeros_like(matrix)
        return (vectors[:, keep] / np.sqrt(values[keep])) @ vectors[:, keep].T
    left = invsqrt(a) @ cross @ invsqrt(b)
    return np.linalg.svd(left, compute_uv=False)


def _coupling_row(key, entry):
    H = .5 * (entry["H"] + entry["H"].T)
    rank = _ranks(H)
    covariance = np.linalg.pinv(H, rcond=1e-12)
    diag = np.diag(covariance)
    # An inverse of a rank-deficient/near-zero normal is not a covariance in
    # the usual statistical sense.  Keep those correlations unavailable
    # rather than reporting numerically meaningless values larger than one.
    covariance_valid = rank["rank"] == 6 and np.all(np.isfinite(diag)) and np.all(diag > 0)
    if covariance_valid:
        denom = np.sqrt(np.maximum(diag[:, None] * diag[None, :], 0.))
        corr = np.divide(covariance, denom, out=np.full_like(covariance, np.nan), where=denom > 0)
    else:
        corr = np.full_like(covariance, np.nan)
    xbar = entry["x_num"] / entry["x_den"] if entry["x_den"] > 0 else np.nan
    if np.isfinite(xbar):
        hgq = H[:3, 3:] - xbar * H[:3, :3]
        hqq = H[3:, 3:] - xbar * (H[:3, 3:] + H[3:, :3]) + xbar ** 2 * H[:3, :3]
        centered = np.block([[H[:3, :3], hgq], [hgq.T, hqq]])
    else:
        centered = H.copy()
    centered = .5 * (centered + centered.T)
    centered_rank = _ranks(centered)
    centered_cov = np.linalg.pinv(centered, rcond=1e-12)
    cd = np.diag(centered_cov)
    centered_covariance_valid = centered_rank["rank"] == 6 and np.all(np.isfinite(cd)) and np.all(cd > 0)
    if centered_covariance_valid:
        cden = np.sqrt(np.maximum(cd[:, None] * cd[None, :], 0.))
        ccorr = np.divide(centered_cov, cden, out=np.full_like(centered_cov, np.nan), where=cden > 0)
    else:
        ccorr = np.full_like(centered_cov, np.nan)
    design_canonical = _canonical(H[:3, :3], H[3:, 3:], H[:3, 3:])
    cov_canonical = (_canonical(covariance[:3, :3], covariance[3:, 3:], covariance[:3, 3:])
                     if covariance_valid else np.full(3, np.nan))
    source_pair = [corr[i, 3 + i] for i in range(3)]
    centered_pair = [ccorr[i, 3 + i] for i in range(3)]
    return {
        "H5": key[0], "SPECID": key[1][0], "IFUSLOT": key[1][1], "IFUID": key[1][2],
        "blank_fibers": entry["blank_fibers"], "blank_equations": entry["blank_equations"],
        "source_rows": entry["source_rows"],
        "rank": rank["rank"], "G_rank": rank["g_rank"], "Q_rank_conditional": rank["q_rank"],
        "singular_values": json.dumps(rank["singular"].tolist()),
        "G_singular_values": json.dumps(rank["g_singular"].tolist()),
        "Q_conditional_singular_values": json.dumps(rank["q_singular"].tolist()),
        "condition_number": _condition(rank["singular"], rank["rank"]),
        "G_condition_number": _condition(rank["g_singular"], rank["g_rank"]),
        "Q_condition_number": _condition(rank["q_singular"], rank["q_rank"]),
        "xbar": xbar,
        "max_GQ_covariance_correlation": float(np.nanmax(np.abs(corr[:3, 3:]))) if covariance_valid else np.nan,
        "paired_GQ_corr_LR": source_pair[0], "paired_GQ_corr_UD": source_pair[1],
        "paired_GQ_corr_I": source_pair[2],
        "canonical_design_LR_UD_I": json.dumps(design_canonical.tolist()),
        "canonical_covariance_LR_UD_I": json.dumps(cov_canonical.tolist()),
        "centered_rank": centered_rank["rank"],
        "centered_condition_number": _condition(centered_rank["singular"], centered_rank["rank"]),
        "centered_max_GQ_covariance_correlation": float(np.nanmax(np.abs(ccorr[:3, 3:]))) if centered_covariance_valid else np.nan,
        "centered_paired_GQ_corr_LR": centered_pair[0],
        "centered_paired_GQ_corr_UD": centered_pair[1],
        "centered_paired_GQ_corr_I": centered_pair[2],
        "centered_canonical_design_LR_UD_I": json.dumps(
            (_canonical(centered[:3, :3], centered[3:, 3:], centered[:3, 3:])
             if centered_covariance_valid else np.full(3, np.nan)).tolist()),
        "G_info_LR": float(H[0, 0]), "G_info_UD": float(H[1, 1]), "G_info_I": float(H[2, 2]),
        "Q_info_LR": float(rank["q_conditional"][0, 0]),
        "Q_info_UD": float(rank["q_conditional"][1, 1]),
        "Q_info_I": float(rank["q_conditional"][2, 2]),
    }


def _mode_label(info, prefix, rank):
    values = np.asarray(info, dtype=float)
    scale = max(float(np.nanmax(values)) if values.size else 0., 1e-14)
    names = ("LR", "UD", "I")
    supported = [names[i] for i, value in enumerate(values) if value > scale * RANK_TOL]
    return "%s%d:%s" % (prefix, rank, "+".join(supported) if supported else "none")


def _source_prediction(row, g, q, group_rows):
    center = np.mean([BASIS[AMP_INDEX[item["AMP"]]] for item in group_rows], axis=0)
    b = BASIS[AMP_INDEX[row["AMP"]]] - center
    state_key = row["key"]
    return float(b @ g.get(state_key, np.zeros(3)) +
                 row["x_src_flat_F_lambda"] * (b @ q.get(state_key, np.zeros(3))))


def _make_source_tail_rows(source_rows, final_g, final_q, support, warm_records):
    grouped = defaultdict(list)
    for row in source_rows:
        grouped[(row["H5"], row["exposure"], row["SPECID"], row["IFUSLOT"],
                 row["IFUID"], row["band"])].append(row)
    output = []
    for group_rows in grouped.values():
        for row in group_rows:
            predicted = _source_prediction(row, final_g, final_q, group_rows)
            residual = float(row["target_log_differential"] - predicted)
            key = row["key"]
            s = support.get(key, {})
            g = final_g.get(key, np.zeros(3))
            q = final_q.get(key, np.zeros(3))
            g_amp = BASIS @ g; q_amp = BASIS @ q
            warm_key = (row["H5"], row["exposure"], row["SPECID"], row["IFUSLOT"],
                        row["IFUID"], row["AMP"], row["band"])
            warm = warm_records.get(warm_key, {})
            singular = np.asarray(s.get("singular_values", np.zeros(6)), dtype=float)
            rank = int(s.get("topology_design_rank", 0) or 0)
            g_rank = int(s.get("G_rank", 0) or 0)
            q_rank = int(s.get("Q_rank_conditional", 0) or 0)
            cond = _condition(singular, rank)
            q_bound = bool(np.max(np.abs(q_amp)) >= QMAX_10 - 1e-7)
            poor = bool(rank < 6 or (np.isfinite(cond) and cond > 1e6))
            output.append({
                "H5": row["H5"], "exposure": row["exposure"],
                "SPECID": row["SPECID"], "IFUSLOT": row["IFUSLOT"], "IFUID": row["IFUID"],
                "AMP": row["AMP"], "band": row["band"],
                "active": bool(s.get("active", False)),
                "source_group_amplifier_support": row["group_amp_support"],
                "N_source_measurements": row["N_source_measurements"],
                "warm_start_robust_scatter": warm.get("robust_scatter_warm_start", np.nan),
                "final_robust_scatter": row["robust_scatter_linear"],
                "final_log_scatter": row["robust_scatter_log"],
                "G_rank": g_rank, "Q_rank_conditional": q_rank,
                "full_topology_rank": rank, "condition_number": cond,
                "smallest_singular_value": float(singular[-1]) if singular.size else np.nan,
                "largest_singular_value": float(singular[0]) if singular.size else np.nan,
                "Q_bound": q_bound, "G_amplitude_max": float(np.max(np.abs(g_amp))),
                "Q_amplitude_max": float(np.max(np.abs(q_amp))),
                "source_residual_log": residual,
                "source_residual_percent": float(100. * np.expm1(residual)),
                "source_abs_residual_percent": float(100. * abs(np.expm1(residual))),
                "partial_support": bool(rank < 6), "poor_condition": poor,
                "failure_class": ("Q_bound" if q_bound else
                                   "partial_or_poor_condition" if poor else
                                   "well_supported_unbounded"),
            })
    return output


def _write_plot_blank(output_dir, blank_rows):
    rows = [r for r in blank_rows if r["grouping"] == "overall" and r["band"] != "ALL_BANDS"]
    if not rows:
        return
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for state, color in (("frozen", "tab:blue"), ("final", "tab:orange")):
        subset = [r for r in rows if r["state"] == state and r["selection"] == "pure_blank"]
        x = [ALL_BANDS.index(r["band"]) for r in subset]
        y = [100. * r["fraction_location"] for r in subset]
        e = [100. * r["fraction_scatter"] for r in subset]
        axes[0, 0].errorbar(x, y, yerr=e, fmt="o-", label=state, color=color)
        axes[0, 1].plot(x, [100. * r["fraction_p95_abs"] for r in subset], "o-", label=state, color=color)
    axes[0, 0].axhline(0, color="k", lw=.8); axes[0, 0].set_title("Pure blank center ± scatter")
    axes[0, 0].set_ylabel("fraction [%]"); axes[0, 0].set_xticks(range(7), ALL_BANDS, rotation=35)
    axes[0, 1].set_title("Pure blank p95 absolute residual")
    axes[0, 1].set_ylabel("fraction [%]"); axes[0, 1].set_xticks(range(7), ALL_BANDS, rotation=35)
    for state, color in (("frozen", "tab:blue"), ("final", "tab:orange")):
        subset = [r for r in rows if r["state"] == state and r["selection"] == "original_blank"]
        axes[1, 0].plot([ALL_BANDS.index(r["band"]) for r in subset],
                        [100. * r["fraction_location"] for r in subset], "o-", label=state, color=color)
        axes[1, 1].plot([ALL_BANDS.index(r["band"]) for r in subset],
                        [100. * r["fraction_p95_abs"] for r in subset], "o-", label=state, color=color)
    axes[1, 0].axhline(0, color="k", lw=.8); axes[1, 0].set_title("Original blank center")
    axes[1, 1].set_title("Original blank p95 absolute residual")
    for axis in axes[1]:
        axis.set_ylabel("fraction [%]"); axis.set_xticks(range(7), ALL_BANDS, rotation=35)
    axes[0, 0].legend(); fig.savefig(output_dir / "blank_physics_overview.png", dpi=140); plt.close(fig)


def _write_plot_source(output_dir, source_rows, tail_rows):
    if not source_rows:
        return
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for state, color in (("frozen", "tab:blue"), ("final", "tab:orange")):
        subset = [r for r in source_rows if r["state"] == state]
        common = np.asarray([r["common_mode_log"] for r in subset])
        diff = np.asarray([r["target_log_differential"] for r in subset])
        axes[0, 0].hist(common[np.isfinite(common)], bins=80, histtype="step", density=True,
                         label=state, color=color)
        axes[0, 1].hist(100. * np.expm1(diff[np.isfinite(diff)]), bins=100, histtype="step",
                         density=True, label=state, color=color)
        for band in SOURCE_BANDS:
            values = [100. * np.expm1(r["target_log_differential"])
                      for r in subset if r["band"] == band]
            axes[1, 0].hist(values, bins=100, histtype="step", density=True,
                            label=state + " " + band, color=color, alpha=.7)
    axes[0, 0].set_title("Source common-mode log(O/X)"); axes[0, 0].legend()
    axes[0, 1].set_title("Source differential log(O/X), percent scale")
    axes[1, 0].set_title("Differential by ON/OFF")
    abs_res = [r["source_abs_residual_percent"] for r in tail_rows]
    if abs_res:
        axes[1, 1].hist(abs_res, bins=np.geomspace(max(min(abs_res), 1e-4), max(abs_res), 80),
                        histtype="step", density=True)
        axes[1, 1].set_xscale("log")
    axes[1, 1].set_title("Final source absolute residual tail")
    fig.savefig(output_dir / "source_common_differential_tail.png", dpi=140); plt.close(fig)


def _write_plot_support(output_dir, support_rows, coupling_rows, init_rows):
    if not support_rows:
        return
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    for rank, label, color in (("G_rank", "G rank", "tab:blue"),
                               ("Q_rank_conditional", "Q rank", "tab:orange"),
                               ("topology_design_rank", "full rank", "tab:green")):
        counts = defaultdict(int)
        for row in support_rows:
            counts[int(row[rank])] += 1
        axis = axes[0 if rank == "G_rank" else 1 if rank == "Q_rank_conditional" else 2]
        axis.bar(list(counts), list(counts.values()), color=color)
        axis.set_title(label); axis.set_xlabel("rank"); axis.set_ylabel("states")
    fig.savefig(output_dir / "topology_support_ranks.png", dpi=140); plt.close(fig)
    if coupling_rows:
        fig, axis = plt.subplots(figsize=(7, 5), constrained_layout=True)
        before = np.asarray([r["condition_number"] for r in coupling_rows], dtype=float)
        after = np.asarray([r["centered_condition_number"] for r in coupling_rows], dtype=float)
        use = np.isfinite(before) & np.isfinite(after) & (before > 0) & (after > 0)
        axis.scatter(before[use], after[use], s=5, alpha=.3)
        positive = np.concatenate((before[use], after[use])) if np.any(use) else np.asarray([1.])
        lo, hi = np.nanpercentile(positive, [1, 99])
        axis.plot([lo, hi], [lo, hi], "k--", lw=.8)
        axis.set_xscale("log"); axis.set_yscale("log")
        axis.set_xlabel("uncentered condition number"); axis.set_ylabel("centered condition number")
        axis.set_title("G/Q wavelength centering")
        fig.savefig(output_dir / "centered_wavelength_conditioning.png", dpi=140); plt.close(fig)
    if init_rows:
        fig, axis = plt.subplots(figsize=(7, 5), constrained_layout=True)
        x = np.arange(len(init_rows))
        axis.plot(x, [r["current_G_amplitude_max"] for r in init_rows], ".", label="current", alpha=.5)
        axis.plot(x, [r["staged_G_amplitude_max"] for r in init_rows], ".", label="staged", alpha=.5)
        axis.set_ylabel("max amplifier gray offset [log]"); axis.set_xlabel("topology state")
        axis.legend(); axis.set_title("Staged G versus current")
        fig.savefig(output_dir / "staged_initialization_comparison.png", dpi=140); plt.close(fig)


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--joint-product", default="m101_joint_calibration/m101_calibration_joint_product.json")
    parser.add_argument("--model3-product", default="m101-calibration-v2-all-h5/m101_calibration_v2_product.json")
    parser.add_argument("--h5", nargs="*")
    parser.add_argument("--blank-file")
    parser.add_argument("--external-cache")
    parser.add_argument("--compact-mask")
    parser.add_argument("--on-filter")
    parser.add_argument("--off-filter")
    parser.add_argument("--fq-template")
    parser.add_argument("--output-dir", default="m101_joint_physics_audit")
    parser.add_argument("--minimum-finite-fraction", type=float, default=.8)
    parser.add_argument("--minimum-source-fibers", type=int, default=10)
    parser.add_argument("--source-sigma-floor", type=float, default=.01)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _defaults(args, joint_payload, model3_path):
    provenance = joint_payload.get("provenance", {})
    model3 = json.loads(Path(model3_path).read_text()) if False else None
    def value(option, *members):
        if option:
            return option
        current = provenance
        for member in members:
            current = current[member]
        return current.get("full_path", current.get("path", current)) if isinstance(current, dict) else current
    # input_h5 belongs to the frozen product; it is read by load_model_and_skies.
    return {
        "blank_file": value(args.blank_file, "blank_file",),
        "external_cache": value(args.external_cache, "external_cache", "cache"),
        "compact_mask": value(args.compact_mask, "compact_mask",),
        "on_filter": value(args.on_filter, "on_filter",),
        "off_filter": value(args.off_filter, "off_filter",),
        "fq_template": value(args.fq_template, "fq_template",),
    }


def _input_path(value):
    return str(Path(value).expanduser().resolve())


def main():
    args = _parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise SystemExit("output directory is nonempty; use --overwrite: %s" % output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    progress = lambda msg: print("[joint-audit +%.1fs] %s" % (time.perf_counter() - started, msg), flush=True)

    joint_payload = _load_joint_payload(args.joint_product)
    defaults = _defaults(args, joint_payload, args.model3_product)
    progress("loading frozen Model-3 product and final joint parameters")
    _, frozen_model, frozen_skies = load_model_and_skies(args.model3_product, progress=progress)
    final_model, final_g, final_q, final_skies = _parse_model_parameters(joint_payload, joint_payload["model_parameters"])
    input_h5 = joint_payload["provenance"].get("product_h5") or []
    if not input_h5:
        base_payload = json.loads(Path(args.model3_product).read_text()) if False else None
        # load_model_and_skies exposes the frozen provenance without reading its
        # 730 MB JSON into memory.
        frozen_product, _, _ = load_model_and_skies(args.model3_product)
        input_h5 = frozen_product["provenance"]["input_h5"]
    h5_paths = [Path(item["full_path"] if isinstance(item, dict) and item.get("full_path")
                     else item["filename"] if isinstance(item, dict) else item).expanduser().resolve()
                for item in input_h5]
    if args.h5:
        supplied = {Path(path).name: Path(path).expanduser().resolve() for path in args.h5}
        h5_paths = [supplied[path.name] for path in h5_paths if path.name in supplied]
    h5_paths = discover_h5(h5_paths, development=True)
    blank_file = _input_path(defaults["blank_file"])
    external_cache = _input_path(defaults["external_cache"])
    compact_mask = _input_path(defaults["compact_mask"])
    on_filter = _input_path(defaults["on_filter"])
    off_filter = _input_path(defaults["off_filter"])
    fq_template = _input_path(defaults["fq_template"])
    progress("loading existing blank classification and native data (%d H5 files)" % len(h5_paths))
    blank_masks_all, blank_provenance = m101_blank_fibers.load(blank_file, h5_paths)
    fq = validated_m101.load_fq(fq_template)
    data, input_provenance = load_native_data(
        h5_paths, blank_masks_all, on_filter, off_filter,
        minimum_finite_fraction=args.minimum_finite_fraction,
        progress=progress)
    responses = np.asarray(input_provenance["responses"], dtype=float)
    progress("reconstructing the persisted source membership; no fit is run")
    external, external_provenance = production_joint._load_external_subset(external_cache, h5_paths)
    mask_data, mask_wcs = m101_compact_mask.load(compact_mask)
    source_masks, group_records = production_joint._candidate_population(
        data, external, mask_data, mask_wcs, frozen_model, frozen_skies, fq,
        responses, args.minimum_source_fibers)
    group_records["__external__"] = external
    accepted = {key: value for key, value in group_records.items()
                if key != "__external__" and value.get("eligible_source")}
    accepted_keys = set(accepted)
    support_path = Path(joint_payload["topology_support_file"])
    if not support_path.is_absolute():
        support_path = Path(args.joint_product).parent / support_path
    support = _load_support(support_path)
    progress("accepted source groups=%d; evaluating frozen and final physics" % len(accepted_keys))

    # Fixed binning is based only on native coordinates, not fitted residuals.
    all_x = np.concatenate([np.asarray(item.x_arcmin, dtype=float) for item in data])
    all_y = np.concatenate([np.asarray(item.y_arcmin, dtype=float) for item in data])
    x_edges = np.linspace(*np.nanpercentile(all_x[np.isfinite(all_x)], [1, 99]), 11)
    y_edges = np.linspace(*np.nanpercentile(all_y[np.isfinite(all_y)], [1, 99]), 11)
    q_bins = None

    blank_acc = ValueAccumulator()
    amp_acc = {}
    normal = {"frozen": {}, "final": {}}
    spectral_samples = defaultdict(list)
    for item_index, item in enumerate(data, 1):
        if item_index % 10 == 0:
            progress("blank physics: exposure %d/%d" % (item_index, len(data)))
        union = np.asarray(source_masks[item.key]["ON"] | source_masks[item.key]["OFF"], dtype=bool)
        selections = {
            "original_blank": np.asarray(item.blank_valid, dtype=bool),
            "pure_blank": np.asarray(item.blank_valid & ~union, dtype=bool),
        }
        xbin = np.clip(np.digitize(item.x_arcmin, x_edges[1:-1]), 0, 9)
        ybin = np.clip(np.digitize(item.y_arcmin, y_edges[1:-1]), 0, 9)
        groups = _state_group_arrays(item, q_bins, xbin, ybin)
        for state, model, g, q, skies in (
                ("frozen", frozen_model, {}, {}, frozen_skies),
                ("final", final_model, final_g, final_q, final_skies)):
            response = _state_model_value(model, item, g, q)
            sky = np.asarray(skies[item.key], dtype=float)
            sky_band, _ = collapse_many(sky[None, :], responses)
            sky_band = sky_band[0]
            alpha = np.asarray([
                model.alpha_q.get((item.key, tuple(map(int, ifu)), str(amp)), 0.)
                for ifu, amp in zip(item.ifu, item.amp)], dtype=float)
            add_band = alpha[:, None] * fq[item.q, None] * item.K[None, :]
            pure = selections["pure_blank"]
            entry_by_state = normal[state]
            for start in range(0, item.row_index.size, 4096):
                stop = min(start + 4096, item.row_index.size)
                sl = slice(start, stop)
                signal = response[sl] * sky[None, :]
                p_values, _ = collapse_many(signal, responses)
                u_values, _ = collapse_many(signal * X_LAMBDA[None, :], responses)
                d_values = np.divide(
                    item.band_total[sl] - add_band[sl] - p_values,
                    p_values, out=np.full_like(p_values, np.nan), where=np.abs(p_values) > 0)
                for selection, mask in selections.items():
                    selected = np.asarray(mask[sl], dtype=bool)
                    if not np.any(selected):
                        continue
                    ids = np.flatnonzero(selected) + start
                    corrected = ((np.asarray(item.total[ids], dtype=float) -
                                  alpha[ids, None] * fq[item.q[ids], None] * item.K_wave[None, :]) /
                                 response[ids] - sky[None, :])
                    residual, _ = collapse_many(corrected, responses)
                    fraction = np.divide(residual, sky_band[None, :],
                                         out=np.full_like(residual, np.nan),
                                         where=np.abs(sky_band[None, :]) > 0)
                    selected_groups = _state_group_arrays(
                        item, q_bins, xbin, ybin)
                    selected_groups = [(name, labels[ids]) for name, labels in selected_groups]
                    for band_index, band in enumerate(ALL_BANDS):
                        _add_grouped(blank_acc, state, selection, selected_groups, band,
                                     residual[:, band_index], fraction[:, band_index])
                    _add_grouped(blank_acc, state, selection, selected_groups, "ALL_BANDS",
                                 residual.reshape(-1), fraction.reshape(-1))
                    if len(spectral_samples[(state, selection)]) < 1800:
                        take = ids[:max(0, 1800 - len(spectral_samples[(state, selection)]))]
                        if take.size:
                            spectral_samples[(state, selection)].extend(
                                np.asarray(((np.asarray(item.total[take], dtype=float) -
                                             alpha[take, None] * fq[item.q[take], None] * item.K_wave[None, :]) /
                                            response[take] - sky[None, :]), dtype=np.float32))
                selected = pure[sl]
                if np.any(selected):
                    ids_local = np.flatnonzero(selected)
                    native_error = np.asarray(item.band_error[sl][ids_local], dtype=float)
                    for local in ids_local:
                        global_index = start + int(local)
                        ifu = tuple(map(int, item.ifu[global_index])); amp = str(item.amp[global_index])
                        state_key = _key(item.h5_name, ifu)
                        entry = _normal_entry(entry_by_state, state_key)
                        entry["blank_fibers"] += 1
                        entry["blank_amplifiers"].add(amp)
                        for band_index in range(len(ALL_BANDS)):
                            sig = float(p_values[local, band_index])
                            lev = float(u_values[local, band_index] / sig) if np.isfinite(sig) and sig != 0 else np.nan
                            design = np.concatenate((sig * BASIS[AMP_INDEX[amp]],
                                                     u_values[local, band_index] * BASIS[AMP_INDEX[amp]]))
                            y = float(d_values[local, band_index] * sig)
                            err = float(item.band_error[global_index, band_index])
                            _add_normal_design(entry, design, y, err, lev)
                            if np.isfinite(err) and err > 0 and np.isfinite(sig):
                                entry["blank_equations"] += 1
                        amp_key = (state, item.h5_name, int(item.exposure), *ifu, amp)
                        # The amplifier summaries are dimensionless and band-specific.
                        for band_index, band in enumerate(ALL_BANDS):
                            d = d_values[local, band_index]
                            lev = (u_values[local, band_index] / p_values[local, band_index]
                                   if np.isfinite(p_values[local, band_index]) and p_values[local, band_index] != 0 else np.nan)
                            akey = amp_key + (band,)
                            amp_acc.setdefault(akey, {"d": [], "x": []})
                            if np.isfinite(d):
                                amp_acc[akey]["d"].append(float(d))
                            if np.isfinite(lev):
                                amp_acc[akey]["x"].append(float(lev))
    blank_rows = blank_acc.rows()
    _write_rows(output_dir / "blank_physical_residual_summary.csv", blank_rows)
    amp_rows = _blank_amp_summary(amp_acc, "frozen") + _blank_amp_summary(amp_acc, "final")
    _write_rows(output_dir / "blank_amplifier_summaries.csv", amp_rows)
    _write_json(output_dir / "blank_spectral_sample_metadata.json", {
        "definition": "first deterministic <=1800 pure/original blank rows per state",
        "wavelength_A": WAVE.tolist(),
        "sample_counts": {str(k): len(v) for k, v in spectral_samples.items()},
    })
    spectral_profiles = {}
    for key, values in spectral_samples.items():
        array = np.asarray(values, dtype=float)
        spectral_profiles["%s__%s" % key] = (np.nanmedian(array, axis=0)
                                               if array.size else np.full(WAVE.shape, np.nan))
    np.savez_compressed(output_dir / "blank_spectral_profiles.npz", wavelength_A=WAVE,
                        **spectral_profiles)

    progress("reconstructing source ON/OFF measurements at both persisted states")
    source_rows = []
    for state, model, g, q, skies in (
            ("frozen", frozen_model, {}, {}, frozen_skies),
            ("final", final_model, final_g, final_q, final_skies)):
        rows = _source_rows_for_state(
            data, model, g, q, skies, fq, responses, source_masks,
            {**accepted, "__external__": external}, args.source_sigma_floor)
        for row in rows:
            row["state"] = state
            row.pop("ratios", None)
        source_rows.extend(rows)
    _write_rows(output_dir / "source_common_differential.csv", source_rows)

    # Blank/source same-state amplifier comparison after each group's common mode.
    blank_by_key = defaultdict(list)
    for row in amp_rows:
        blank_by_key[(row["state"], row["H5"], row["exposure"], row["SPECID"],
                      row["IFUSLOT"], row["IFUID"], row["band"])].append(row)
    source_by_key = defaultdict(list)
    for row in source_rows:
        source_by_key[(row["state"], row["H5"], row["exposure"], row["SPECID"],
                       row["IFUSLOT"], row["IFUID"], row["band"])].append(row)
    comparison_rows = []
    for group_key, rows in source_by_key.items():
        b_rows = {row["AMP"]: row for row in blank_by_key.get(group_key, [])}
        if len(b_rows) < 2:
            continue
        b_common = float(robust_location([math.log1p(row["d_location"])
                                          for row in b_rows.values()
                                          if row["d_location"] > -1]))
        for row in rows:
            if row["AMP"] not in b_rows or b_rows[row["AMP"]]["d_location"] <= -1:
                continue
            b = b_rows[row["AMP"]]
            comparison_rows.append({
                "state": group_key[0], "H5": group_key[1], "exposure": group_key[2],
                "SPECID": group_key[3], "IFUSLOT": group_key[4], "IFUID": group_key[5],
                "band": group_key[6], "AMP": row["AMP"],
                "blank_common_log": b_common,
                "blank_log_d": math.log1p(b["d_location"]),
                "blank_differential_log": math.log1p(b["d_location"]) - b_common,
                "source_common_log": row["common_mode_log"],
                "source_log_ratio": row["log_ratio"],
                "source_differential_log": row["target_log_differential"],
            })
    _write_rows(output_dir / "blank_source_amplifier_comparison.csv", comparison_rows)
    compare_summary = []
    for state in ("frozen", "final"):
        for band in ALL_BANDS:
            subset = [row for row in comparison_rows if row["state"] == state and row["band"] == band]
            compare_summary.append({
                "state": state, "band": band, "N": len(subset),
                "correlation_blank_source_differential": _corr(
                    [r["blank_differential_log"] for r in subset],
                    [r["source_differential_log"] for r in subset]),
                "blank_differential_scatter": (float(robust_scatter([r["blank_differential_log"] for r in subset]))
                                                if subset else np.nan),
                "source_differential_scatter": (float(robust_scatter([r["source_differential_log"] for r in subset]))
                                                 if subset else np.nan),
            })
    _write_rows(output_dir / "blank_source_amplifier_comparison_summary.csv", compare_summary)

    progress("building native blank/source topology normals and support modes")
    coupling_rows = []
    mode_rows = []
    for state in ("frozen", "final"):
        for key, entry in sorted(normal[state].items(), key=str):
            row = _coupling_row(key, entry)
            row["state"] = state
            row["G_mode_support"] = _mode_label(
                [row["G_info_LR"], row["G_info_UD"], row["G_info_I"]], "G", row["G_rank"])
            row["Q_mode_support"] = _mode_label(
                [row["Q_info_LR"], row["Q_info_UD"], row["Q_info_I"]], "Q", row["Q_rank_conditional"])
            row["physical_IFU"] = _ifu_text(key[1])
            coupling_rows.append(row)
            mode_rows.append(row.copy())
    _write_rows(output_dir / "gq_covariance_conditioning.csv", coupling_rows)

    # Source rows are the source-uncertainty and tail products; add official
    # support fields and current-model residuals.
    final_source = [row for row in source_rows if row["state"] == "final"]
    tail_rows = _make_source_tail_rows(final_source, final_g, final_q, support, accepted)
    _write_rows(output_dir / "source_failure_classification.csv", tail_rows)
    tail_summary = []
    for label, predicate in (
            ("all", lambda r: True),
            ("active", lambda r: r["active"]),
            ("inactive", lambda r: not r["active"]),
            ("Q_bound", lambda r: r["Q_bound"]),
            ("Q_not_bound", lambda r: not r["Q_bound"]),
            ("full_rank", lambda r: r["full_topology_rank"] >= 6),
            ("partial_rank", lambda r: r["full_topology_rank"] < 6),
            ("poor_condition", lambda r: r["poor_condition"]),
            ("good_condition", lambda r: not r["poor_condition"]),
            ("well_supported_unbounded", lambda r: r["failure_class"] == "well_supported_unbounded"),
    ):
        subset = [r for r in tail_rows if predicate(r)]
        values = np.asarray([r["source_abs_residual_percent"] for r in subset], dtype=float)
        tail_summary.append({"population": label, **_summary(values)})
    _write_rows(output_dir / "source_failure_tail_summary.csv", tail_summary)

    # Candidate support policies use the measured ranks, while the official
    # production active flag remains visible for comparison.
    support_policy_rows = []
    state_keys = sorted(set(normal["final"]) | set(support), key=str)
    state_row_by_key = {(_key(r["H5"], (r["SPECID"], r["IFUSLOT"], r["IFUID"]))): r
                        for r in tail_rows}
    source_key_counts = defaultdict(int)
    source_fiber_counts = defaultdict(int)
    for row in final_source:
        source_key_counts[row["key"]] += 1
        source_fiber_counts[row["key"]] += int(row["N_source_measurements"])
    blank_fiber_counts = {key: entry["blank_fibers"] for key, entry in normal["final"].items()}
    for policy, select in (
            ("full_6D_only", lambda r: r["rank"] >= 6 and r["G_rank"] >= 3 and r["Q_rank_conditional"] >= 3),
            ("full_G_no_Q", lambda r: r["G_rank"] >= 3),
            ("full_G_any_supported_Q", lambda r: r["G_rank"] >= 3 and r["Q_rank_conditional"] >= 1),
            ("any_supported_G_any_supported_Q", lambda r: r["G_rank"] >= 1 and r["Q_rank_conditional"] >= 1),
            ("any_supported_G_only", lambda r: r["G_rank"] >= 1),
    ):
        selected_keys = set()
        for key in state_keys:
            audit = next((r for r in coupling_rows if r["state"] == "final" and
                          _key(r["H5"], (r["SPECID"], r["IFUSLOT"], r["IFUID"])) == key), None)
            if audit is None:
                official = support.get(key, {})
                rank = int(official.get("topology_design_rank", 0) or 0)
                grank = int(official.get("G_rank", 0) or 0)
                qrank = int(official.get("Q_rank_conditional", 0) or 0)
                audit = {"rank": rank, "G_rank": grank, "Q_rank_conditional": qrank}
            if select(audit):
                selected_keys.add(key)
        source_rows_n = sum(value for key, value in source_key_counts.items() if key in selected_keys)
        source_fibers_n = sum(value for key, value in source_fiber_counts.items() if key in selected_keys)
        blank_n = sum(value for key, value in blank_fiber_counts.items() if key in selected_keys)
        support_policy_rows.append({
            "policy": policy, "state_count": len(selected_keys), "state_fraction": len(selected_keys) / max(len(state_keys), 1),
            "source_measurement_rows": source_rows_n,
            "source_measurement_fraction": source_rows_n / max(len(final_source), 1),
            "source_native_fiber_count": source_fibers_n,
            "blank_fiber_count": blank_n,
        })
    _write_rows(output_dir / "topology_support_mode_coverage.csv", support_policy_rows)

    # Source uncertainty audit against persisted leave-one-exposure-out rows.
    loo_path = Path(args.joint_product).parent / "m101_joint_leave_one_exposure_out.csv"
    loo = {}
    if loo_path.exists():
        with loo_path.open() as stream:
            for row in csv.DictReader(stream):
                key = (row["H5"], row["heldout_exposure"], row["SPECID"], row["IFUSLOT"],
                       row["IFUID"], row["AMP"], row["band"])
                loo[key] = float(row["residual_log"])
    uncertainty_rows = []
    for row in final_source:
        key = (row["H5"], str(row["exposure"]), str(row["SPECID"]), str(row["IFUSLOT"]),
               str(row["IFUID"]), row["AMP"], row["band"])
        if key not in loo:
            continue
        uncertainty_rows.append({
            "H5": row["H5"], "exposure": row["exposure"], "SPECID": row["SPECID"],
            "IFUSLOT": row["IFUSLOT"], "IFUID": row["IFUID"], "AMP": row["AMP"], "band": row["band"],
            "N_source_measurements": row["N_source_measurements"],
            "raw_linear_scatter": row["robust_scatter_linear"],
            "log_scatter": row["robust_scatter_log"],
            "ratio_uncertainty_existing": row["ratio_uncertainty"],
            "linear_location_se": row["linear_location_se"],
            "log_location_se": row["log_location_se"],
            "heldout_residual_abs_log": abs(loo[key]),
            "heldout_residual_abs_percent": abs(100. * np.expm1(loo[key])),
        })
    _write_rows(output_dir / "source_uncertainty_vs_heldout.csv", uncertainty_rows)
    uncertainty_summary = []
    heldout = np.asarray([r["heldout_residual_abs_log"] for r in uncertainty_rows], dtype=float)
    for name in ("raw_linear_scatter", "log_scatter", "ratio_uncertainty_existing",
                 "linear_location_se", "log_location_se"):
        values = np.asarray([r[name] for r in uncertainty_rows], dtype=float)
        use = np.isfinite(values) & np.isfinite(heldout) & (values > 0) & (heldout > 0)
        uncertainty_summary.append({
            "uncertainty_definition": name, "N": int(np.sum(use)),
            "correlation_abs_heldout_log_error": _corr(values[use], heldout[use]),
            "median_abs_heldout_over_sigma": (float(np.median(heldout[use] / values[use]))
                                               if np.any(use) else np.nan),
            "fraction_abs_heldout_le_sigma": (float(np.mean(heldout[use] <= values[use]))
                                               if np.any(use) else np.nan),
        })
    _write_rows(output_dir / "source_uncertainty_summary.csv", uncertainty_summary)

    # Hierarchical initialization prototype from the persisted frozen evidence.
    progress("solving diagnostic source-anchored G -> Q -> local refinement")
    frozen_amp_rows = [r for r in amp_rows if r["state"] == "frozen"]
    frozen_source_rows = [r for r in source_rows if r["state"] == "frozen"]
    evidence = defaultdict(list)
    for row in frozen_source_rows:
        group = (row["H5"], row["exposure"], row["SPECID"], row["IFUSLOT"], row["IFUID"], row["band"])
        group_rows = [r for r in frozen_source_rows if
                      (r["H5"], r["exposure"], r["SPECID"], r["IFUSLOT"], r["IFUID"], r["band"]) == group]
        center = np.mean([BASIS[AMP_INDEX[r["AMP"]]] for r in group_rows], axis=0)
        b = BASIS[AMP_INDEX[row["AMP"]]] - center
        design = np.concatenate((b, row["x_src_flat_F_lambda"] * b))
        evidence[row["key"]].append({"type": "source", "design": design,
                                      "y": row["target_log_differential"],
                                      "sigma": max(row["sigma_existing"], np.finfo(float).eps)})
    by_blank_group = defaultdict(list)
    for row in frozen_amp_rows:
        by_blank_group[(row["H5"], row["exposure"], row["SPECID"], row["IFUSLOT"], row["IFUID"], row["band"])].append(row)
    for group, group_rows in by_blank_group.items():
        valid = [r for r in group_rows if np.isfinite(r["d_location"]) and r["d_location"] > -1]
        if len(valid) < 2:
            continue
        common = float(robust_location([math.log1p(r["d_location"]) for r in valid]))
        center = np.mean([BASIS[AMP_INDEX[r["AMP"]]] for r in valid], axis=0)
        for row in valid:
            key = _key(row["H5"], (row["SPECID"], row["IFUSLOT"], row["IFUID"]))
            b = BASIS[AMP_INDEX[row["AMP"]]] - center
            design = np.concatenate((b, row["effective_x_location"] * b))
            evidence[key].append({"type": "blank", "design": design,
                                  "y": math.log1p(row["d_location"]) - common,
                                  "sigma": max(row["location_se"], np.finfo(float).eps)})

    def evidence_normal(key, types):
        H = np.zeros((6, 6)); rhs = np.zeros(6)
        for item in evidence.get(key, []):
            if item["type"] not in types:
                continue
            d = item["design"]; w = 1. / item["sigma"] ** 2
            H += w * np.outer(d, d); rhs += w * d * item["y"]
        return .5 * (H + H.T), rhs

    def objective(key, params, types):
        values = []
        for item in evidence.get(key, []):
            if item["type"] in types:
                values.append(((item["y"] - item["design"] @ params) / item["sigma"]) ** 2)
        return float(.5 * np.sum(values)) if values else np.nan

    def bounded_q(Hq, rhsq):
        result = np.linalg.lstsq(Hq, rhsq, rcond=1e-12)[0]
        if np.max(np.abs(BASIS @ result)) <= QMAX_10 + 1e-10:
            return result, False
        # Three-dimensional convex active-set enumeration; this is only a
        # diagnostic re-solve and does not alter the production constraint.
        constraints = np.vstack((BASIS, -BASIS))
        limits = np.full(8, QMAX_10)
        candidates = []
        for mask in [()] + [(i,) for i in range(8)] + [(i, j) for i in range(8) for j in range(i + 1, 8)] + [(i, j, k) for i in range(8) for j in range(i + 1, 8) for k in range(j + 1, 8)]:
            if mask:
                a = constraints[list(mask)]
                kkt = np.block([[Hq, a.T], [a, np.zeros((len(mask), len(mask)))]])
                sol = np.linalg.lstsq(kkt, np.concatenate((rhsq, limits[list(mask)])), rcond=1e-12)[0][:3]
            else:
                sol = np.zeros(3)
            if np.max(constraints @ sol - limits) <= 1e-8:
                candidates.append((float(.5 * sol @ Hq @ sol - rhsq @ sol), sol))
        return min(candidates, key=lambda x: x[0])[1], True

    init_rows = []; q_rows = []; g_rows = []
    init_params = {}
    for key in sorted(evidence, key=str):
        Hs, rs = evidence_normal(key, {"source"})
        Hc, rc = evidence_normal(key, {"source", "blank"})
        g_source = np.linalg.lstsq(Hs[:3, :3], rs[:3], rcond=1e-12)[0] if np.any(Hs[:3, :3]) else np.zeros(3)
        g_combined = np.linalg.lstsq(Hc[:3, :3], rc[:3], rcond=1e-12)[0] if np.any(Hc[:3, :3]) else np.zeros(3)
        # Use the source anchor as the staged point; retain combined G as an
        # empirical comparison, never as a hidden source-vs-blank multiplier.
        g_stage = g_source
        H_all, r_all = evidence_normal(key, {"source", "blank"})
        Hq = H_all[3:, 3:]
        rq = r_all[3:] - H_all[3:, :3] @ g_stage
        q_unbounded, q_was_bounded = bounded_q(Hq, rq)
        staged = np.concatenate((g_stage, q_unbounded))
        local = np.linalg.lstsq(H_all, r_all, rcond=1e-12)[0] if np.any(H_all) else np.zeros(6)
        try:
            local_constrained, _, _, local_bounded = production_joint._constrained_local_q(
                H_all, r_all, np.zeros(3), QMAX_10)
        except Exception:
            local_constrained, local_bounded = local, False
        current = np.concatenate((final_g.get(key, np.zeros(3)), final_q.get(key, np.zeros(3))))
        rank_info = _ranks(H_all)
        init_params[key] = {"g_source": g_source, "g_combined": g_combined,
                            "staged": staged, "local": local_constrained, "current": current,
                            "rank": rank_info}
        g_rows.append({
            "H5": key[0], "SPECID": key[1][0], "IFUSLOT": key[1][1], "IFUID": key[1][2],
            "G_LR_source": g_source[0], "G_UD_source": g_source[1], "G_I_source": g_source[2],
            "G_LR_source_plus_blank": g_combined[0], "G_UD_source_plus_blank": g_combined[1], "G_I_source_plus_blank": g_combined[2],
            "source_blank_G_difference_norm": float(np.linalg.norm(g_source - g_combined)),
            "G_source_amplitude_max": float(np.max(np.abs(BASIS @ g_source))),
            "G_combined_amplitude_max": float(np.max(np.abs(BASIS @ g_combined))),
            "G_rank": rank_info["g_rank"], "source_evidence_count": sum(i["type"] == "source" for i in evidence[key]),
            "blank_evidence_count": sum(i["type"] == "blank" for i in evidence[key]),
        })
        q_rows.append({
            "H5": key[0], "SPECID": key[1][0], "IFUSLOT": key[1][1], "IFUID": key[1][2],
            "Q_LR": q_unbounded[0], "Q_UD": q_unbounded[1], "Q_I": q_unbounded[2],
            "Q_amplitude_max": float(np.max(np.abs(BASIS @ q_unbounded))),
            "Q_bounded": q_was_bounded, "Q_rank_conditional": rank_info["q_rank"],
            "objective_Q0": objective(key, np.concatenate((g_stage, np.zeros(3))), {"source", "blank"}),
            "objective_Q_staged": objective(key, staged, {"source", "blank"}),
            "objective_Q_full_unbounded": objective(key, np.concatenate((g_stage, np.linalg.lstsq(Hq, rq, rcond=1e-12)[0])), {"source", "blank"}),
        })
        init_rows.append({
            "H5": key[0], "SPECID": key[1][0], "IFUSLOT": key[1][1], "IFUID": key[1][2],
            "G_rank": rank_info["g_rank"], "Q_rank_conditional": rank_info["q_rank"], "full_rank": rank_info["rank"],
            "staged_objective": objective(key, staged, {"source", "blank"}),
            "local_objective": objective(key, local_constrained, {"source", "blank"}),
            "current_objective": objective(key, current, {"source", "blank"}),
            "staged_G_amplitude_max": float(np.max(np.abs(BASIS @ staged[:3]))),
            "local_G_amplitude_max": float(np.max(np.abs(BASIS @ local_constrained[:3]))),
            "current_G_amplitude_max": float(np.max(np.abs(BASIS @ current[:3]))),
            "staged_Q_amplitude_max": float(np.max(np.abs(BASIS @ staged[3:]))),
            "local_Q_amplitude_max": float(np.max(np.abs(BASIS @ local_constrained[3:]))),
            "current_Q_amplitude_max": float(np.max(np.abs(BASIS @ current[3:]))),
            "local_Q_bounded": bool(local_bounded),
            "local_max_delta_G_offset_from_staged": float(np.max(np.abs(BASIS @ (local_constrained[:3] - staged[:3])))),
            "source_objective_staged": objective(key, staged, {"source"}),
            "source_objective_local": objective(key, local_constrained, {"source"}),
            "blank_objective_staged": objective(key, staged, {"blank"}),
            "blank_objective_local": objective(key, local_constrained, {"blank"}),
        })
    _write_rows(output_dir / "staged_G_initialization.csv", g_rows)
    _write_rows(output_dir / "staged_Q_initialization.csv", q_rows)
    _write_rows(output_dir / "initial_vs_current_topology.csv", init_rows)

    # Trust-region and final amplitude audit.
    trust_rows = []
    all_g_offsets = []; all_q_offsets = []
    for key in state_keys:
        gv = final_g.get(key, np.zeros(3)); qv = final_q.get(key, np.zeros(3))
        go = BASIS @ gv; qo = BASIS @ qv
        all_g_offsets.extend(go.tolist()); all_q_offsets.extend(qo.tolist())
        staged = init_params.get(key, {}).get("staged", np.zeros(6))
        local = init_params.get(key, {}).get("local", np.zeros(6))
        delta = BASIS @ (local[:3] - staged[:3])
        trust_rows.append({
            "H5": key[0], "SPECID": key[1][0], "IFUSLOT": key[1][1], "IFUID": key[1][2],
            "G_LR": gv[0], "G_UD": gv[1], "G_I": gv[2],
            "delta_LL": go[0], "delta_LU": go[1], "delta_RL": go[2], "delta_RU": go[3],
            "factor_LL": np.exp(go[0]), "factor_LU": np.exp(go[1]), "factor_RL": np.exp(go[2]), "factor_RU": np.exp(go[3]),
            "Q_offset_max": float(np.max(np.abs(qo))),
            "local_step_max_G_offset": float(np.max(np.abs(delta))),
            "trust_005_active": bool(np.max(np.abs(delta)) > .05),
            "trust_scaled_step_max": float(min(np.max(np.abs(delta)), .05)),
            "Q_near_10pct_bound": bool(np.max(np.abs(qo)) >= QMAX_10 - 1e-7),
        })
    _write_rows(output_dir / "G_amplitude_trust_region.csv", trust_rows)
    _write_json(output_dir / "G_amplitude_summary.json", {
        "G_offset_log_percentiles": np.percentile(np.asarray(all_g_offsets), [50, 68, 90, 95, 99, 100]).tolist(),
        "G_factor_percentiles": np.percentile(np.exp(np.asarray(all_g_offsets)), [50, 68, 90, 95, 99, 100]).tolist(),
        "Q_offset_log_percentiles": np.percentile(np.abs(np.asarray(all_q_offsets)), [50, 68, 90, 95, 99, 100]).tolist(),
        "Q_bound_qmax": QMAX_10,
        "Q_near_bound_state_count": int(sum(r["Q_near_10pct_bound"] for r in trust_rows)),
        "trust_region_005_active_state_count": int(sum(r["trust_005_active"] for r in trust_rows)),
        "trust_region_005_fraction": float(np.mean([r["trust_005_active"] for r in trust_rows])) if trust_rows else np.nan,
        "per_pass_changes_recoverable": False,
        "per_pass_note": "The persisted joint product records pass summaries but not per-state G snapshots; no per-pass G changes were reconstructed.",
    })

    # Exact algebraic wavelength-centering check on a deterministic sample.
    invariance = []
    for key, entry in list(normal["final"].items())[:100]:
        xbar = entry["x_num"] / entry["x_den"] if entry["x_den"] > 0 else 0.
        gv = final_g.get(key, np.zeros(3)); qv = final_q.get(key, np.zeros(3))
        old = BASIS @ gv + X_LAMBDA[:, None] * 0.0
        # Compare scalar topology for all four amplifiers and all wavelengths.
        old = np.asarray([(BASIS[a] @ gv) + X_LAMBDA * (BASIS[a] @ qv) for a in range(4)])
        new = np.asarray([(BASIS[a] @ (gv + xbar * qv) + (X_LAMBDA - xbar) * (BASIS[a] @ qv))
                          for a in range(4)])
        invariance.append(float(np.max(np.abs(old - new))))
    _write_json(output_dir / "centered_wavelength_invariance.json", {
        "definition": "G_star=G+xbar Q; topology=G_star+(x_lambda-xbar)Q",
        "sample_states": len(invariance), "max_abs_prediction_difference": max(invariance, default=0.),
    })

    _write_plot_blank(output_dir, blank_rows)
    _write_plot_source(output_dir, source_rows, tail_rows)
    _write_plot_support(output_dir, support, coupling_rows, init_rows)

    # Checklist and a compact technical report.
    def overall_blank(state, selection, field):
        rows = [r for r in blank_rows if r["state"] == state and r["selection"] == selection and
                r["grouping"] == "overall" and r["band"] == "ALL_BANDS"]
        return rows[0].get(field, np.nan) if rows else np.nan
    final_support = [r for r in coupling_rows if r["state"] == "final"]
    final_corr = [r for r in compare_summary if r["state"] == "final" and np.isfinite(r["correlation_blank_source_differential"])]
    q_meas = sum(r["Q_rank_conditional"] > 0 for r in final_support)
    checklist = {
        "schema": SCHEMA,
        "blank_zero": {"frozen_fraction_location": overall_blank("frozen", "pure_blank", "fraction_location"),
                        "final_fraction_location": overall_blank("final", "pure_blank", "fraction_location"),
                        "final_fraction_scatter": overall_blank("final", "pure_blank", "fraction_scatter"),
                        "assessment": "diagnostic; inspect band and selection summaries"},
        "additive_q": {"assessment": "reported in q-binned blank_physical_residual_summary.csv"},
        "focal_plane": {"assessment": "reported in x_bin/y_bin blank_physical_residual_summary.csv"},
        "amplifier_topology": {"matched_blank_source_correlation_median": float(np.nanmedian([r["correlation_blank_source_differential"] for r in final_corr])) if final_corr else np.nan,
                               "assessment": "see matched amplifier comparison"},
        "exposure_coherence": {"leave_one_exposure_out_product": str(Path(args.joint_product).parent / "m101_joint_leave_one_exposure_out.csv"),
                               "assessment": "existing persisted LOO rows retained; no new fit"},
        "on_off": {"assessment": "common and differential log ratios are separated in source_common_differential.csv"},
        "wavelength": {"states_with_Q_rank_gt_zero": q_meas,
                       "states_total": len(final_support),
                       "fraction_Q_measurable": q_meas / max(len(final_support), 1),
                       "assessment": "mode support measured from native blank + source normals"},
        "model_coverage": {"policies_file": "topology_support_mode_coverage.csv"},
        "large_tail": {"tail_file": "source_failure_tail_summary.csv"},
        "parameter_coupling": {"coupling_file": "gq_covariance_conditioning.csv",
                                "centering_check_file": "centered_wavelength_invariance.json"},
        "initialization": {"initialization_file": "initial_vs_current_topology.csv",
                            "assessment": "diagnostic local summary solve only"},
    }
    _write_json(output_dir / "physics_acceptance_checklist.json", checklist)
    _write_rows(output_dir / "physics_acceptance_checklist.csv",
                [{"check": key, "value": json.dumps(value, default=str)} for key, value in checklist.items()])

    support_counts = {"official_active": int(joint_payload.get("active_topology_blocks", 0)),
                      "audit_states_with_full_rank": int(sum(r["rank"] >= 6 for r in final_support)),
                      "audit_states_with_full_G": int(sum(r["G_rank"] >= 3 for r in final_support)),
                      "audit_states_with_any_Q": int(q_meas)}
    report = f"""# M101 joint physics / identifiability audit

This report is diagnostic only.  It consumed the frozen Model-3 product and the persisted final joint product; it did not modify either product and did not run a production fit.

## 1. PHYSICS CHECK

The blank residuals are in `blank_physical_residual_summary.csv` for both the frozen and final states, including original and pure blank selections, all seven bands, H5, exposure, physical IFU, amplifier, q, and focal-plane bins.  The exact collapsed-band dimensionless amplifier residuals are in `blank_amplifier_summaries.csv`.  Source absolute/common and differential quantities are in `source_common_differential.csv`.

## 2. FAILURE-TAIL ORIGIN

The row-level classification is `source_failure_classification.csv`; population summaries are in `source_failure_tail_summary.csv`.  It explicitly separates active/inactive, partial/full rank, Q-bound/non-bound, condition, warm-start scatter, and source support.

## 3. TOPOLOGY SUPPORT / COVERAGE

The persisted production active count is {support_counts['official_active']}.  The audit finds {support_counts['audit_states_with_full_rank']} states with native six-dimensional full rank, {support_counts['audit_states_with_full_G']} with all three gray modes, and {support_counts['audit_states_with_any_Q']} with at least one conditional chromatic mode.  Candidate science coverage policies are in `topology_support_mode_coverage.csv`.

## 4. PARAMETER COUPLING

`gq_covariance_conditioning.csv` contains inverse-Hessian correlations, canonical correlations, singular values, condition numbers, mode information, and the centered-wavelength comparison.  `centered_wavelength_invariance.json` verifies the reparameterization prediction identity.

## 5. BLANK VS SOURCE AMPLIFIER INFORMATION

`blank_source_amplifier_comparison_summary.csv` compares matched differential topology after profiling the common mode.  No source-vs-blank multiplier was introduced; each diagnostic summary retains its native empirical scatter/SE.

## 6. SOURCE UNCERTAINTY

`source_uncertainty_summary.csv` compares linear robust scatter, log scatter, existing `ratio_uncertainty`, and location-SE definitions against persisted held-out exposure errors.  The fixed 0.15 source QC threshold was not changed.

## 7. INITIALIZATION EXPERIMENT

`staged_G_initialization.csv`, `staged_Q_initialization.csv`, and `initial_vs_current_topology.csv` contain source-anchored G, Q=0/supported-Q comparisons, and one local six-dimensional summary refinement.  These are prototype solves on amplifier summaries only.

## 8. TWO-SCALE FITTING ASSESSMENT

The measured normal matrices and their support/coupling diagnostics are the evidence for a future block-coordinate design.  This audit makes no production recommendation beyond the evidence in those tables.

## 9. WHAT IS ESTABLISHED

The raw blank corrected residual, source common-mode/differential separation, mode support, covariance, and initialization products are now reproducible under one standalone command.

## 10. WHAT STILL BLOCKS PRODUCTION

The audit must be read quantitatively before adopting partial-mode updates, trust regions, or a new likelihood.  The persisted product did not contain per-state G snapshots for every outer pass, so only final amplitudes and a diagnostic local-step trust test are reported.

## 11. RECOMMENDED NEXT IMPLEMENTATION

Use the tables to choose support policy and sigma definition first.  If the centered coupling and staged local residuals support it, prototype a small held-out block-coordinate validation; keep all scientific thresholds and the production fitter unchanged until that validation is separately accepted.
"""
    (output_dir / "m101_joint_physics_audit_report.md").write_text(report)
    _write_json(output_dir / "audit_metadata.json", {
        "schema": SCHEMA, "elapsed_seconds": time.perf_counter() - started,
        "inputs": {"joint_product": str(Path(args.joint_product).resolve()),
                   "model3_product": str(Path(args.model3_product).resolve()),
                   "h5_count": len(h5_paths), "blank_file": blank_file,
                   "external_cache": external_cache, "compact_mask": compact_mask,
                   "on_filter": on_filter, "off_filter": off_filter, "fq_template": fq_template},
        "source_membership_reconstructed": True,
        "accepted_source_groups": len(accepted_keys),
        "native_exposure_count": len(data),
        "production_products_untouched": True,
        "external_provenance": external_provenance,
        "blank_provenance": blank_provenance,
    })
    progress("audit complete: %s" % output_dir)


if __name__ == "__main__":
    main()
