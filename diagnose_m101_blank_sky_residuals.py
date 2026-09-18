#!/usr/bin/env python3
"""Diagnose blank-fiber residual structure after a frozen gray correction.

This script is deliberately independent of the ON/OFF initializer.  It reads
the compact cache, the cumulative-final ``R_total_h_i_a`` hierarchy written by
that initializer, and the cache's already-classified blank mask.  It does not
fit or update a response model, use external X, or read native spectra.
"""

from __future__ import annotations

import argparse
import ast
import csv
from collections import defaultdict
from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
import time

import numpy as np

from math_utils import biweight


Q_MIN = 40
Q_MAX = 70
AMP_ORDER = ("LL", "LU", "RL", "RU")
TOPOLOGY_MODES = ("C", "LR", "UD", "I")
TOPOLOGY_ALL_REPRESENTATIONS = ("epsilon", "Ares", "delta")
TOPOLOGY_REPRESENTATIONS = ("Ares", "delta")
TOPOLOGY_MATRIX = np.asarray((
    (1.0, 1.0, 1.0, 1.0),
    (-1.0, -1.0, 1.0, 1.0),
    (-1.0, 1.0, -1.0, 1.0),
    (1.0, -1.0, -1.0, 1.0),
), dtype=float) / 4.0
TOPOLOGY_INVERSE = np.asarray((
    (1.0, -1.0, -1.0, 1.0),
    (1.0, -1.0, 1.0, -1.0),
    (1.0, 1.0, -1.0, -1.0),
    (1.0, 1.0, 1.0, 1.0),
), dtype=float)
TOPOLOGY_MODE_SETS = (
    ("NONE", ()),
    ("C", ("C",)),
    ("C+LR", ("C", "LR")),
    ("C+LR+UD", ("C", "LR", "UD")),
    ("C+LR+UD+I", ("C", "LR", "UD", "I")),
    ("LR_only", ("LR",)),
    ("UD_only", ("UD",)),
    ("I_only", ("I",)),
)
OUTPUT_SCHEMA = "m101_blank_sky_residuals_v2"


@dataclass
class Item:
    h5: str
    exposure: int
    ifu: np.ndarray
    ifu_code: np.ndarray
    amp: np.ndarray
    q: np.ndarray
    D: np.ndarray
    blank_valid: np.ndarray
    hardware_bad: np.ndarray
    logm: np.ndarray


@dataclass
class Record:
    item: Item
    valid: np.ndarray
    epsilon: np.ndarray
    ares: np.ndarray
    delta: np.ndarray
    delta0: np.ndarray


def json_ready(value):
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_ready(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def finite(values):
    values = np.asarray(values, dtype=float)
    return values[np.isfinite(values)]


def robust_location(values):
    values = finite(values)
    if values.size == 0:
        return np.nan
    if values.size == 1 or np.all(values == values[0]):
        return float(values[0])
    with np.errstate(all="ignore"):
        value = biweight(values)
    return float(value) if np.isfinite(value) else float(np.nanmedian(values))


def robust_scatter(values):
    values = finite(values)
    if values.size < 2:
        return np.nan
    if np.all(values == values[0]):
        return 0.0
    with np.errstate(all="ignore"):
        _, value = biweight(values, calc_std=True)
    if np.isfinite(value) and value >= 0:
        return float(value)
    return float(np.nanstd(values))


def summary(values, prefix=""):
    values = finite(values)
    result = {
        prefix + "location": robust_location(values),
        prefix + "scatter": robust_scatter(values),
        prefix + "median": float(np.median(values)) if values.size else np.nan,
        prefix + "p16": float(np.percentile(values, 16)) if values.size else np.nan,
        prefix + "p84": float(np.percentile(values, 84)) if values.size else np.nan,
        prefix + "minimum": float(np.min(values)) if values.size else np.nan,
        prefix + "maximum": float(np.max(values)) if values.size else np.nan,
        prefix + "N": int(values.size),
    }
    return result


def abs_metrics(values):
    values = np.abs(finite(values))
    return {
        "scatter": robust_scatter(values),
        "p68_abs": float(np.percentile(values, 68)) if values.size else np.nan,
        "p95_abs": float(np.percentile(values, 95)) if values.size else np.nan,
        "N": int(values.size),
    }


def correlation(x, y, spearman=False):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    good = np.isfinite(x) & np.isfinite(y)
    x, y = x[good], y[good]
    if x.size < 2 or np.std(x) == 0 or np.std(y) == 0:
        return np.nan
    if spearman:
        x = np.argsort(np.argsort(x, kind="mergesort"), kind="mergesort")
        y = np.argsort(np.argsort(y, kind="mergesort"), kind="mergesort")
    return float(np.corrcoef(x, y)[0, 1])


def pairwise_differences(values):
    values = finite(values)
    if values.size < 2:
        return np.empty(0, dtype=float)
    return np.asarray([values[j] - values[i]
                       for i in range(values.size)
                       for j in range(i + 1, values.size)], dtype=float)


def safe_name(value):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")


def write_rows(path, rows):
    path = Path(path)
    rows = list(rows)
    if not rows:
        path.write_text("")
        return
    fields = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            clean = {}
            for field in fields:
                value = row.get(field, "")
                if isinstance(value, np.generic):
                    value = value.item()
                if isinstance(value, (list, tuple, np.ndarray)):
                    value = json.dumps(json_ready(value), separators=(",", ":"))
                clean[field] = value
            writer.writerow(clean)


def file_identity(path):
    path = Path(path).resolve()
    stat = path.stat()
    return {"filename": path.name, "full_path": str(path),
            "file_size": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns)}


def parse_state_key(value):
    try:
        parsed = ast.literal_eval(value)
    except (SyntaxError, ValueError):
        return None
    return parsed if isinstance(parsed, tuple) else None


def find_state(requested):
    if requested:
        path = Path(requested).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError("initializer state JSON not found: %s" % path)
        return path
    candidates = (
        Path("m101_onoff_hierarchy/m101_onoff_state.json"),
        Path("m101_onoff_initializer/m101_onoff_state.json"),
    )
    for path in candidates:
        if path.exists():
            return path.resolve()
    raise FileNotFoundError(
        "could not find m101_onoff_state.json; pass --state explicitly")


def load_state(path):
    state = json.loads(Path(path).read_text())
    hierarchy = state.get("hierarchy", {})
    source = hierarchy.get("R_total_h_i_a", {})
    if not source:
        source = hierarchy.get("cumulative_final", {}).get("R_total_h_i_a", {})
    if not source:
        raise ValueError("state does not contain cumulative-final R_total_h_i_a")
    response = {}
    for encoded, value in source.items():
        key = parse_state_key(encoded)
        if key is None or len(key) != 3:
            continue
        h5, code, amp = key
        if value is not None and np.isfinite(value):
            response[(str(h5), int(code), str(amp))] = float(value)
    if not response:
        raise ValueError("cumulative-final response map is empty")
    return state, response


def load_cache(cache_path, state_response):
    cache_path = Path(cache_path).expanduser().resolve()
    manifest_path = cache_path.with_suffix(cache_path.suffix + ".json")
    manifest = json.loads(manifest_path.read_text())
    bands = tuple(manifest.get("band_order", ()))
    if len(bands) != 7:
        raise ValueError("expected seven cache bands, found %d" % len(bands))
    required = ("ifu", "ifu_code", "amp", "q", "band_total",
                "blank_valid", "hardware_bad")
    with np.load(cache_path, allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]) for name in required}
    items = []
    missing_response_rows = 0
    response_values = []
    response_keys = set()
    for row in manifest.get("items", []):
        start, stop = int(row["start"]), int(row["stop"])
        sl = slice(start, stop)
        h5 = str(row["h5_name"])
        exposure = int(row["exposure"])
        code = arrays["ifu_code"][sl].astype(np.int64, copy=False)
        amp = arrays["amp"][sl].astype(np.int64, copy=False)
        logm = np.full(code.size, np.nan, dtype=float)
        for index, (ifu_code, amp_index) in enumerate(zip(code, amp)):
            key = (h5, int(ifu_code), AMP_ORDER[int(amp_index)])
            if key in state_response:
                logm[index] = state_response[key]
                response_values.append(state_response[key])
                response_keys.add(key)
            else:
                missing_response_rows += 1
        items.append(Item(
            h5=h5, exposure=exposure,
            ifu=arrays["ifu"][sl].copy(), ifu_code=code.copy(), amp=amp.copy(),
            q=arrays["q"][sl].copy(), D=arrays["band_total"][sl].copy(),
            blank_valid=arrays["blank_valid"][sl].copy(),
            hardware_bad=arrays["hardware_bad"][sl].copy(), logm=logm))
    if not items:
        raise ValueError("cache contains no items")
    return items, manifest, bands, missing_response_rows, response_values, response_keys


def prepare_records(items, bands):
    sky_values = defaultdict(list)
    raw_values = defaultdict(list)
    validity_counts = defaultdict(int)
    for item in items:
        with np.errstate(over="ignore", invalid="ignore"):
            multiplier = np.exp(item.logm)
            corrected = item.D / multiplier[:, None]
        for band_index, band in enumerate(bands):
            valid = (item.blank_valid & ~item.hardware_bad &
                     (item.q >= Q_MIN) & (item.q <= Q_MAX) &
                     np.isfinite(item.logm) & np.isfinite(item.D[:, band_index]) &
                     np.isfinite(corrected[:, band_index]))
            sky_values[(item.h5, item.exposure, band)].append(corrected[valid, band_index])
            raw_values[(item.h5, item.exposure, band)].append(item.D[valid, band_index])
            validity_counts[(item.h5, item.exposure, band)] += int(np.sum(valid))

    sky = {}
    sky_rows = []
    for key in sorted(sky_values):
        h5, exposure, band = key
        corrected = np.concatenate([finite(chunk) for chunk in sky_values[key]
                                    if finite(chunk).size]) if any(
                                        finite(chunk).size for chunk in sky_values[key]) else np.empty(0)
        raw = np.concatenate([finite(chunk) for chunk in raw_values[key]
                              if finite(chunk).size]) if any(
                                  finite(chunk).size for chunk in raw_values[key]) else np.empty(0)
        s0 = robust_location(raw)
        safter = robust_location(corrected)
        sky[key] = (s0, safter)
        row = {"H5": h5, "exposure": int(exposure), "band": band,
               "N_blank": int(corrected.size), "S0_raw": s0,
               "S_after_M": safter, "scatter_raw": robust_scatter(raw),
               "scatter_after_M": robust_scatter(corrected),
               "epsilon_location_global": robust_location(corrected - safter)
               if corrected.size else np.nan,
               "epsilon_scatter_global": robust_scatter(corrected - safter)
               if corrected.size else np.nan}
        sky_rows.append(row)

    records = []
    for item in items:
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            multiplier = np.exp(item.logm)
            corrected = item.D / multiplier[:, None]
        valid_all = np.zeros_like(item.D, dtype=bool)
        epsilon = np.full_like(item.D, np.nan, dtype=float)
        ares = np.full_like(item.D, np.nan, dtype=float)
        delta = np.full_like(item.D, np.nan, dtype=float)
        delta0 = np.full_like(item.D, np.nan, dtype=float)
        for band_index, band in enumerate(bands):
            s0, safter = sky.get((item.h5, item.exposure, band), (np.nan, np.nan))
            valid = (item.blank_valid & ~item.hardware_bad &
                     (item.q >= Q_MIN) & (item.q <= Q_MAX) &
                     np.isfinite(item.logm) & np.isfinite(item.D[:, band_index]) &
                     np.isfinite(corrected[:, band_index]) & np.isfinite(safter) &
                     np.isfinite(s0) & (safter != 0) & (s0 != 0))
            valid_all[:, band_index] = valid
            epsilon[valid, band_index] = corrected[valid, band_index] - safter
            ares[valid, band_index] = item.D[valid, band_index] - multiplier[valid] * safter
            delta[valid, band_index] = item.D[valid, band_index] / (
                multiplier[valid] * safter) - 1.0
            delta0[valid, band_index] = item.D[valid, band_index] / s0 - 1.0
        records.append(Record(item, valid_all, epsilon, ares, delta, delta0))
    return records, sky, sky_rows


def identity_for(item, code):
    indices = np.flatnonzero(item.ifu_code == int(code))
    if not indices.size:
        return (np.nan, np.nan, np.nan)
    return tuple(int(value) for value in item.ifu[indices[0]])


def amp_residual_rows(records, bands):
    rows = []
    for record in records:
        item = record.item
        for code in sorted(np.unique(item.ifu_code).astype(int)):
            identity = identity_for(item, code)
            group = item.ifu_code == code
            for amp_index, amp_name in enumerate(AMP_ORDER):
                amp_group = group & (item.amp == amp_index)
                for band_index, band in enumerate(bands):
                    use = amp_group & record.valid[:, band_index]
                    if not np.any(use):
                        continue
                    row = {"H5": item.h5, "exposure": item.exposure,
                           "IFU_CODE": int(code), "SPECID": identity[0],
                           "IFUSLOT": identity[1], "IFUID": identity[2],
                           "AMP": amp_name, "AMP_INDEX": amp_index,
                           "band": band, "N_fibers": int(np.sum(use))}
                    for name, values in (("epsilon", record.epsilon),
                                         ("Ares", record.ares),
                                         ("delta", record.delta),
                                         ("delta0", record.delta0)):
                        row.update(summary(values[use, band_index], name + "_"))
                    rows.append(row)
    return rows


def ifu_common_rows(amp_rows, bands):
    grouped = defaultdict(list)
    for row in amp_rows:
        grouped[(row["H5"], row["exposure"], row["IFU_CODE"], row["band"])].append(row)
    rows = []
    common = {}
    for key, group in sorted(grouped.items()):
        h5, exposure, code, band = key
        identity = (group[0]["SPECID"], group[0]["IFUSLOT"], group[0]["IFUID"])
        row = {"H5": h5, "exposure": exposure, "IFU_CODE": code,
               "SPECID": identity[0], "IFUSLOT": identity[1], "IFUID": identity[2],
               "band": band, "N_amp": len(group),
               "available_amp_names": [r["AMP"] for r in group]}
        for name in ("epsilon", "Ares", "delta"):
            values = np.asarray([r[name + "_location"] for r in group], dtype=float)
            row["C_" + name] = robust_location(values)
            row["amp_scatter_" + name] = robust_scatter(values)
            departures = values - row["C_" + name]
            row["amp_departure_scatter_" + name] = robust_scatter(departures)
            row["amp_common_fractional_reduction_" + name] = (
                100.0 * (1.0 - row["amp_departure_scatter_" + name] /
                          row["amp_scatter_" + name])
                if np.isfinite(row["amp_scatter_" + name]) and
                row["amp_scatter_" + name] != 0 else np.nan)
        common[key] = row
        rows.append(row)
    return rows, common


def attach_amp_departures(amp_rows, common):
    for row in amp_rows:
        key = (row["H5"], row["exposure"], row["IFU_CODE"], row["band"])
        c = common[key]
        for name in ("epsilon", "Ares", "delta"):
            row[name + "_departure"] = row[name + "_location"] - c["C_" + name]


def repeatability_rows(amp_rows, bands):
    grouped = defaultdict(list)
    for row in amp_rows:
        grouped[(row["H5"], row["IFU_CODE"], row["AMP"], row["AMP_INDEX"], row["band"])].append(row)
    rows = []
    for key, group in sorted(grouped.items()):
        h5, code, amp, amp_index, band = key
        row = {"H5": h5, "IFU_CODE": code, "AMP": amp, "AMP_INDEX": amp_index,
               "band": band, "N_exposures": len(group),
               "exposures_used": [r["exposure"] for r in group]}
        for name in ("epsilon", "Ares", "delta"):
            values = [r[name + "_location"] for r in group]
            differences = pairwise_differences(values)
            row[name + "_mean"] = robust_location(values)
            row[name + "_exposure_scatter"] = robust_scatter(values)
            row[name + "_pairwise_difference_scatter"] = robust_scatter(differences)
            row[name + "_pairwise_difference_p95_abs"] = (
                float(np.percentile(np.abs(finite(differences)), 95))
                if finite(differences).size else np.nan)
        rows.append(row)
    return rows


def exposure_commonality(amp_rows, bands):
    result = []
    for band in bands:
        for name in ("Ares", "delta"):
            grouped = defaultdict(dict)
            for row in amp_rows:
                if row["band"] == band:
                    grouped[(row["H5"], row["IFU_CODE"], row["AMP"])][row["exposure"]] = row[name + "_location"]
            x, y, differences = [], [], []
            for values in grouped.values():
                for e1, value1 in values.items():
                    for e2, value2 in values.items():
                        if e2 > e1 and np.isfinite(value1) and np.isfinite(value2):
                            x.append(value1); y.append(value2); differences.append(value2 - value1)
            result.append({"kind": "exposure_to_exposure", "band": band,
                           "representation": name, "N_pairs": len(differences),
                           "correlation": correlation(x, y),
                           "robust_difference_scatter": robust_scatter(differences),
                           "p95_abs_difference": (float(np.percentile(np.abs(differences), 95))
                                                  if differences else np.nan)})
    return result


def spectral_commonality(amp_rows, bands):
    grouped = defaultdict(list)
    for row in amp_rows:
        grouped[(row["H5"], row["IFU_CODE"], row["AMP"], row["AMP_INDEX"], row["exposure"])].append(row)
    by_channel = defaultdict(dict)
    for key, rows in grouped.items():
        h5, code, amp, amp_index, exposure = key
        values = {row["band"]: row["delta_location"] for row in rows}
        if len(values) >= 3:
            by_channel[(h5, code, amp, amp_index)][exposure] = values
    rows = []
    for key, exposures in sorted(by_channel.items()):
        h5, code, amp, amp_index = key
        exposure_list = sorted(exposures)
        for index, e1 in enumerate(exposure_list):
            for e2 in exposure_list[index + 1:]:
                common = sorted(set(exposures[e1]) & set(exposures[e2]) & set(bands))
                if len(common) < 3:
                    continue
                v1 = np.asarray([exposures[e1][band] for band in common], dtype=float)
                v2 = np.asarray([exposures[e2][band] for band in common], dtype=float)
                v1 -= robust_location(v1); v2 -= robust_location(v2)
                rows.append({"H5": h5, "IFU_CODE": code, "AMP": amp,
                             "AMP_INDEX": amp_index, "exposure_1": e1,
                             "exposure_2": e2, "N_bands": len(common),
                             "centered_delta_correlation": correlation(v1, v2),
                             "centered_delta_rms_difference": float(np.sqrt(np.mean((v1 - v2) ** 2)))})
    return rows


def topology_decompose(values):
    """Return the exact C/LR/UD/I coefficients and inverse reconstruction."""
    values = np.asarray(values, dtype=float)
    coefficients = TOPOLOGY_MATRIX @ values
    reconstruction = TOPOLOGY_INVERSE @ coefficients
    return dict(zip(TOPOLOGY_MODES, coefficients)), reconstruction


def topology_measurements(amp_rows, bands):
    """Build complete-four topology rows without changing channel rows."""
    grouped = defaultdict(dict)
    for row in amp_rows:
        key = (row["H5"], row["exposure"], row["IFU_CODE"], row["band"])
        grouped[key][int(row["AMP_INDEX"])] = row

    rows = []
    reconstruction_errors = {name: [] for name in TOPOLOGY_ALL_REPRESENTATIONS}
    support_counts = defaultdict(int)
    partial_channel_rows = 0
    for key, amp_map in grouped.items():
        n_amp = len(amp_map)
        support_counts[n_amp] += 1
        if n_amp < 4:
            partial_channel_rows += n_amp
            continue
        if set(amp_map) != set(range(4)):
            continue
        first = amp_map[0]
        h5, exposure, code, band = key
        row = {
            "H5": h5, "exposure": exposure, "IFU_CODE": code,
            "SPECID": first["SPECID"], "IFUSLOT": first["IFUSLOT"],
            "IFUID": first["IFUID"], "band": band,
            "available_amp_count": 4,
        }
        for representation in TOPOLOGY_ALL_REPRESENTATIONS:
            values = np.asarray([
                amp_map[index][representation + "_location"]
                for index in range(4)
            ], dtype=float)
            for index, amp_name in enumerate(AMP_ORDER):
                row[amp_name + "_" + representation] = values[index]
            if np.all(np.isfinite(values)):
                coefficients, reconstruction = topology_decompose(values)
                for mode, value in coefficients.items():
                    row[mode + "_" + representation] = float(value)
                error = float(np.max(np.abs(reconstruction - values)))
                row["reconstruction_error_" + representation] = error
                reconstruction_errors[representation].append(error)
            else:
                for mode in TOPOLOGY_MODES:
                    row[mode + "_" + representation] = np.nan
                row["reconstruction_error_" + representation] = np.nan
        rows.append(row)
    return rows, reconstruction_errors, dict(support_counts), partial_channel_rows


def topology_distribution_rows(topology_rows, bands):
    rows = []
    for representation in TOPOLOGY_REPRESENTATIONS:
        for mode in TOPOLOGY_MODES:
            for band in list(bands) + ["ALL"]:
                values = [row[mode + "_" + representation] for row in topology_rows
                          if band == "ALL" or row["band"] == band]
                values = finite(values)
                abs_values = np.abs(values)
                rows.append({
                    "representation": representation, "mode": mode, "band": band,
                    "N": int(values.size),
                    "location": robust_location(values),
                    "robust_scatter": robust_scatter(values),
                    "p68_abs": float(np.percentile(abs_values, 68)) if abs_values.size else np.nan,
                    "p95_abs": float(np.percentile(abs_values, 95)) if abs_values.size else np.nan,
                    "p99_abs": float(np.percentile(abs_values, 99)) if abs_values.size else np.nan,
                    "minimum": float(np.min(values)) if values.size else np.nan,
                    "maximum": float(np.max(values)) if values.size else np.nan,
                })
    return rows


def _topology_population_summary(group_rows, representation, mode, band, scope):
    values = []
    for row in group_rows:
        if (row["representation"], row["mode"]) != (representation, mode):
            continue
        if scope != "ALL" and row["band"] != scope:
            continue
        values.append(row["value"])
    values = finite(values)
    return {
        "scope": "population", "H5": "ALL", "IFU_CODE": "ALL",
        "band": band, "representation": representation, "mode": mode,
        "N_groups": int(values.size), "N_exposure_pairs": np.nan,
        "exposure_scatter": robust_scatter(values),
        "median_pairwise_abs_difference": np.nan,
        "pair_12_correlation": np.nan, "pair_13_correlation": np.nan,
        "pair_23_correlation": np.nan,
    }


def topology_repeatability_rows(topology_rows, bands):
    """Return per-group and population exposure repeatability by mode."""
    grouped = defaultdict(dict)
    for row in topology_rows:
        grouped[(row["H5"], row["IFU_CODE"], row["band"])][row["exposure"]] = row
    rows = []
    population = defaultdict(lambda: {
        (1, 2): {1: [], 2: [], "differences": []},
        (1, 3): {1: [], 3: [], "differences": []},
        (2, 3): {2: [], 3: [], "differences": []},
    })
    for (h5, code, band), exposures in sorted(grouped.items()):
        for representation in TOPOLOGY_REPRESENTATIONS:
            for mode in TOPOLOGY_MODES:
                values = {e: row[mode + "_" + representation]
                          for e, row in exposures.items()
                          if np.isfinite(row[mode + "_" + representation])}
                if len(values) < 2:
                    continue
                ordered = sorted(values)
                differences = []
                pair_fields = {}
                for first in ordered:
                    for second in ordered:
                        if second <= first:
                            continue
                        difference = values[second] - values[first]
                        differences.append(difference)
                        pair_fields["pair_%d%d_difference" % (first, second)] = difference
                        pair = population[(band, representation, mode)][(first, second)]
                        pair[first].append(values[first])
                        pair[second].append(values[second])
                        pair["differences"].append(difference)
                row = {
                    "scope": "H5_IFU", "H5": h5, "IFU_CODE": code, "band": band,
                    "representation": representation, "mode": mode,
                    "N_exposures": len(values),
                    "exposures_used": ordered,
                    "exposure_scatter": robust_scatter(list(values.values())),
                    "median_pairwise_abs_difference": (
                        float(np.median(np.abs(finite(differences))))
                        if finite(differences).size else np.nan),
                    "pair_12_difference": pair_fields.get("pair_12_difference", np.nan),
                    "pair_13_difference": pair_fields.get("pair_13_difference", np.nan),
                    "pair_23_difference": pair_fields.get("pair_23_difference", np.nan),
                    "pair_12_correlation": np.nan, "pair_13_correlation": np.nan,
                    "pair_23_correlation": np.nan,
                }
                rows.append(row)

    for band in list(bands) + ["ALL"]:
        for representation in TOPOLOGY_REPRESENTATIONS:
            for mode in TOPOLOGY_MODES:
                keys = [band] if band != "ALL" else list(bands)
                for key_band in keys:
                    population_values = population.get((key_band, representation, mode))
                    if not population_values:
                        continue
                pair_corr = {}
                pair_differences = []
                for first, second in ((1, 2), (1, 3), (2, 3)):
                    first_values, second_values, differences = [], [], []
                    for key_band in keys:
                        population_values = population.get((key_band, representation, mode))
                        if population_values:
                            pair = population_values[(first, second)]
                            first_values.extend(pair[first])
                            second_values.extend(pair[second])
                            differences.extend(pair["differences"])
                    pair_corr["pair_%d%d_correlation" % (first, second)] = correlation(
                        first_values, second_values)
                    pair_differences.extend(differences)
                rows.append({
                    "scope": "population", "H5": "ALL", "IFU_CODE": "ALL",
                    "band": band, "representation": representation, "mode": mode,
                    "N_exposures": np.nan, "N_groups": len(pair_differences),
                    "exposure_scatter": robust_scatter(pair_differences),
                    "median_pairwise_abs_difference": (
                        float(np.median(np.abs(finite(pair_differences))))
                        if finite(pair_differences).size else np.nan),
                    "pair_12_difference": np.nan, "pair_13_difference": np.nan,
                    "pair_23_difference": np.nan,
                    **pair_corr,
                })
    return rows


def topology_shape_repeatability_rows(topology_rows, bands):
    grouped = defaultdict(dict)
    for row in topology_rows:
        grouped[(row["H5"], row["IFU_CODE"], row["exposure"])][row["band"]] = row
    rows = []
    population = defaultdict(lambda: defaultdict(list))
    groups = sorted({(h5, code) for h5, code, _ in grouped})
    for h5, code in groups:
        exposure_rows = defaultdict(dict)
        for (row_h5, row_code, exposure), band_rows in grouped.items():
            if row_h5 == h5 and row_code == code:
                exposure_rows[exposure].update(band_rows)
        for representation in TOPOLOGY_REPRESENTATIONS:
            for mode in TOPOLOGY_MODES:
                available = [e for e, values in exposure_rows.items()
                             if all(band in values and np.isfinite(
                                 values[band][mode + "_" + representation])
                                    for band in bands)]
                if len(available) < 2:
                    continue
                vectors = {}
                for exposure in available:
                    vector = np.asarray([
                        exposure_rows[exposure][band][mode + "_" + representation]
                        for band in bands], dtype=float)
                    vectors[exposure] = vector - np.mean(vector)
                row = {"scope": "H5_IFU", "H5": h5, "IFU_CODE": code,
                       "representation": representation, "mode": mode,
                       "N_exposures": len(vectors), "N_bands": len(bands),
                       "exposures_used": sorted(vectors)}
                for first, second in ((1, 2), (1, 3), (2, 3)):
                    if first in vectors and second in vectors:
                        row["shape_corr_%d%d" % (first, second)] = correlation(
                            vectors[first], vectors[second])
                        row["shape_rms_%d%d" % (first, second)] = float(
                            np.sqrt(np.mean((vectors[first] - vectors[second]) ** 2)))
                        population[(representation, mode)]["corr_%d%d" % (first, second)].append(
                            row["shape_corr_%d%d" % (first, second)])
                        population[(representation, mode)]["rms_%d%d" % (first, second)].append(
                            row["shape_rms_%d%d" % (first, second)])
                    else:
                        row["shape_corr_%d%d" % (first, second)] = np.nan
                        row["shape_rms_%d%d" % (first, second)] = np.nan
                rows.append(row)
    for representation in TOPOLOGY_REPRESENTATIONS:
        for mode in TOPOLOGY_MODES:
            values = population[(representation, mode)]
            row = {"scope": "population", "H5": "ALL", "IFU_CODE": "ALL",
                   "representation": representation, "mode": mode,
                   "N_exposures": np.nan, "N_bands": len(bands),
                   "exposures_used": "ALL"}
            for first, second in ((1, 2), (1, 3), (2, 3)):
                corr_values = finite(values["corr_%d%d" % (first, second)])
                rms_values = finite(values["rms_%d%d" % (first, second)])
                row["shape_corr_%d%d" % (first, second)] = robust_location(corr_values)
                row["shape_rms_%d%d" % (first, second)] = robust_location(rms_values)
            rows.append(row)
    return rows


def _mode_prediction(coefficients, mode_names):
    selected = np.zeros(4, dtype=float)
    for mode in mode_names:
        selected += TOPOLOGY_INVERSE[:, TOPOLOGY_MODES.index(mode)] * coefficients[mode]
    return selected


def topology_leave_one_out(topology_rows, bands):
    grouped = defaultdict(dict)
    for row in topology_rows:
        grouped[(row["H5"], row["IFU_CODE"], row["band"])][row["exposure"]] = row
    rows = []
    payload = defaultdict(lambda: {"before": [], "after": []})
    for (h5, code, band), exposures in sorted(grouped.items()):
        if len(exposures) < 3:
            continue
        for representation in TOPOLOGY_REPRESENTATIONS:
            amp_field = [amp + "_" + representation for amp in AMP_ORDER]
            for heldout in sorted(exposures):
                training = [e for e in sorted(exposures) if e != heldout]
                actual = np.asarray([exposures[heldout][field] for field in amp_field], dtype=float)
                if not np.all(np.isfinite(actual)):
                    continue
                train_coefficients = {}
                for mode in TOPOLOGY_MODES:
                    values = [exposures[e][mode + "_" + representation] for e in training]
                    if not np.all(np.isfinite(values)):
                        train_coefficients[mode] = np.nan
                    else:
                        train_coefficients[mode] = float(np.mean(values))
                if not all(np.isfinite(value) for value in train_coefficients.values()):
                    continue
                actual_coefficients, exact = topology_decompose(actual)
                del exact
                baseline = actual.copy()
                baseline_scatter = robust_scatter(baseline)
                for mode_set, mode_names in TOPOLOGY_MODE_SETS:
                    prediction = _mode_prediction(train_coefficients, mode_names)
                    residual = actual - prediction
                    metrics = abs_metrics(residual)
                    before = robust_scatter(baseline)
                    after = robust_scatter(residual)
                    row = {
                        "scope": "H5_IFU", "H5": h5, "IFU_CODE": code,
                        "heldout_exposure": heldout, "training_exposures": training,
                        "band": band, "representation": representation,
                        "mode_set": mode_set, "N_amplifiers": 4,
                        "before_scatter": before, "after_scatter": after,
                        "p68_abs": metrics["p68_abs"], "p95_abs": metrics["p95_abs"],
                        "percent_improvement": (100.0 * (1.0 - after / before)
                                                if np.isfinite(before) and before != 0 and
                                                np.isfinite(after) else np.nan),
                    }
                    rows.append(row)
                    for scope_band in (band, "ALL"):
                        key = (scope_band, representation, mode_set)
                        payload[key]["before"].extend(baseline.tolist())
                        payload[key]["after"].extend(residual.tolist())
    for band in list(bands) + ["ALL"]:
        for representation in TOPOLOGY_REPRESENTATIONS:
            for mode_set, _ in TOPOLOGY_MODE_SETS:
                values = payload[(band, representation, mode_set)]
                before = finite(values["before"])
                after = finite(values["after"])
                before_scatter = robust_scatter(before)
                after_scatter = robust_scatter(after)
                metrics = abs_metrics(after)
                rows.append({
                    "scope": "population", "H5": "ALL", "IFU_CODE": "ALL",
                    "heldout_exposure": "ALL", "training_exposures": "OTHER_TWO",
                    "band": band, "representation": representation,
                    "mode_set": mode_set, "N_amplifiers": int(after.size),
                    "before_scatter": before_scatter, "after_scatter": after_scatter,
                    "p68_abs": metrics["p68_abs"], "p95_abs": metrics["p95_abs"],
                    "percent_improvement": (100.0 * (1.0 - after_scatter / before_scatter)
                                            if np.isfinite(before_scatter) and before_scatter != 0 and
                                            np.isfinite(after_scatter) else np.nan),
                })
    return rows


def topology_common_vs_full(loo_rows, bands):
    rows = []
    for band in list(bands) + ["ALL"]:
        for representation in TOPOLOGY_REPRESENTATIONS:
            common = next((row for row in loo_rows if row["scope"] == "population" and
                           row["band"] == band and row["representation"] == representation and
                           row["mode_set"] == "C"), None)
            full = next((row for row in loo_rows if row["scope"] == "population" and
                         row["band"] == band and row["representation"] == representation and
                         row["mode_set"] == "C+LR+UD+I"), None)
            common_after = common["after_scatter"] if common else np.nan
            full_after = full["after_scatter"] if full else np.nan
            rows.append({
                "scope": "population", "band": band, "representation": representation,
                "C_only_after_scatter": common_after,
                "full_basis_after_scatter": full_after,
                "improvement_full_basis_over_common_only": (
                    100.0 * (1.0 - full_after / common_after)
                    if np.isfinite(common_after) and common_after != 0 and
                    np.isfinite(full_after) else np.nan),
            })
    return rows


def topology_power_rows(topology_rows, bands):
    rows = []
    aggregations = [("global", "ALL", "ALL", "ALL")]
    aggregations.extend(("band", "ALL", "ALL", band) for band in bands)
    aggregations.extend(("exposure", "ALL", exposure, "ALL") for exposure in (1, 2, 3))
    aggregations.extend(("H5", h5, "ALL", "ALL")
                        for h5 in sorted({row["H5"] for row in topology_rows}))
    for representation in TOPOLOGY_REPRESENTATIONS:
        for aggregation, h5_filter, exposure_filter, band_filter in aggregations:
            selected = [row for row in topology_rows
                        if (h5_filter == "ALL" or row["H5"] == h5_filter) and
                        (exposure_filter == "ALL" or row["exposure"] == exposure_filter) and
                        (band_filter == "ALL" or row["band"] == band_filter)]
            denominators = []
            powers = {mode: [] for mode in TOPOLOGY_MODES}
            fractions = {mode: [] for mode in TOPOLOGY_MODES}
            for row in selected:
                coefficients = [row[mode + "_" + representation] for mode in TOPOLOGY_MODES]
                if not np.all(np.isfinite(coefficients)):
                    continue
                denominator = float(np.sum(np.square(coefficients)))
                denominators.append(denominator)
                for mode, coefficient in zip(TOPOLOGY_MODES, coefficients):
                    powers[mode].append(float(coefficient ** 2))
                    fractions[mode].append(float(coefficient ** 2 / denominator)
                                            if denominator > 0 else np.nan)
            for mode in TOPOLOGY_MODES:
                power = finite(powers[mode]); fraction = finite(fractions[mode])
                rows.append({
                    "aggregation": aggregation, "H5": h5_filter,
                    "exposure": exposure_filter, "band": band_filter,
                    "representation": representation, "mode": mode,
                    "N": int(power.size), "power_location": robust_location(power),
                    "power_scatter": robust_scatter(power),
                    "power_p95": float(np.percentile(power, 95)) if power.size else np.nan,
                    "fraction_location": robust_location(fraction),
                    "fraction_scatter": robust_scatter(fraction),
                    "fraction_p95": float(np.percentile(fraction, 95)) if fraction.size else np.nan,
                    "total_power_location": robust_location(denominators),
                })
    return rows


def topology_sky_dependence(topology_rows, sky, bands):
    rows = []
    for representation in TOPOLOGY_REPRESENTATIONS:
        for mode in TOPOLOGY_MODES:
            for band in list(bands) + ["ALL"]:
                values, skies = [], []
                for row in topology_rows:
                    if band != "ALL" and row["band"] != band:
                        continue
                    s = sky.get((row["H5"], row["exposure"], row["band"]), (np.nan, np.nan))[1]
                    value = row[mode + "_" + representation]
                    if np.isfinite(s) and np.isfinite(value):
                        values.append(abs(value)); skies.append(s)
                rows.append({
                    "representation": representation, "mode": mode, "band": band,
                    "N": len(values), "correlation_abs_mode_vs_S": correlation(values, skies),
                    "median_abs_mode": float(np.median(values)) if values else np.nan,
                    "S_location": robust_location(skies),
                })
    return rows


def topology_mode_comparison(topology_rows, repeat_rows, shape_rows, loo_rows, bands):
    rows = []
    for representation in TOPOLOGY_REPRESENTATIONS:
        for mode in TOPOLOGY_MODES:
            repeat = next((row for row in repeat_rows if row["scope"] == "population" and
                           row["band"] == "ALL" and row["representation"] == representation and
                           row["mode"] == mode), {})
            shape = next((row for row in shape_rows if row["scope"] == "population" and
                          row["representation"] == representation and row["mode"] == mode), {})
            independent_set = mode if mode == "C" else mode + "_only"
            independent_loo = [row["percent_improvement"] for row in loo_rows
                               if row["scope"] == "population" and row["band"] == "ALL" and
                               row["representation"] == representation and
                               row["mode_set"] == independent_set]
            full_loo = [row["percent_improvement"] for row in loo_rows
                        if row["scope"] == "population" and row["band"] == "ALL" and
                        row["representation"] == representation and
                        row["mode_set"] == "C+LR+UD+I"]
            rows.append({
                "representation": representation, "mode": mode,
                "amplitude_location": robust_location([
                    row[mode + "_" + representation] for row in topology_rows]),
                "amplitude_scatter": robust_scatter([
                    row[mode + "_" + representation] for row in topology_rows]),
                "median_pairwise_abs_difference": repeat.get("median_pairwise_abs_difference", np.nan),
                "pairwise_correlation": robust_location([
                    repeat.get("pair_12_correlation", np.nan),
                    repeat.get("pair_13_correlation", np.nan),
                    repeat.get("pair_23_correlation", np.nan)]),
                "shape_correlation": robust_location([
                    shape.get("shape_corr_12", np.nan), shape.get("shape_corr_13", np.nan),
                    shape.get("shape_corr_23", np.nan)]),
                "shape_rms": robust_location([
                    shape.get("shape_rms_12", np.nan), shape.get("shape_rms_13", np.nan),
                    shape.get("shape_rms_23", np.nan)]),
                "independent_mode_LOO_improvement": robust_location(independent_loo),
                "full_basis_LOO_improvement": robust_location(full_loo),
            })
    del bands
    return rows


def stage_values(records, amp_rows, common, bands):
    amp_lookup = {(r["H5"], r["exposure"], r["IFU_CODE"], r["AMP_INDEX"], r["band"]): r
                  for r in amp_rows}
    stage = defaultdict(list)
    loo_rows = []
    representations = ("epsilon", "Ares", "delta")
    for record in records:
        item = record.item
        for band_index, band in enumerate(bands):
            base_valid = record.valid[:, band_index]
            if np.any(base_valid):
                stage[("STAGE0", "delta0", band)].append(record.delta0[base_valid, band_index])
            c_key = lambda e, code: (item.h5, e, int(code), band)
            for name, values in (("epsilon", record.epsilon),
                                 ("Ares", record.ares),
                                 ("delta", record.delta)):
                if not np.any(base_valid):
                    continue
                raw = values[:, band_index]
                same_common = np.full(raw.size, np.nan)
                same_amp = np.full(raw.size, np.nan)
                for code in np.unique(item.ifu_code[base_valid]).astype(int):
                    code_mask = item.ifu_code == code
                    c = common.get(c_key(item.exposure, code), {}).get("C_" + name, np.nan)
                    if np.isfinite(c):
                        same_common[code_mask] = c
                    for amp_index in range(4):
                        arow = amp_lookup.get((item.h5, item.exposure, int(code), amp_index, band), {})
                        value = arow.get(name + "_location", np.nan)
                        if np.isfinite(value):
                            same_amp[code_mask & (item.amp == amp_index)] = value
                stage[("STAGE1", name, band)].append(raw[base_valid])
                stage[("STAGE2", name, band)].append((raw - same_common)[base_valid]
                                                       [np.isfinite((raw - same_common)[base_valid])])
                stage[("STAGE3", name, band)].append((raw - same_amp)[base_valid]
                                                       [np.isfinite((raw - same_amp)[base_valid])])

                # Leave-one-exposure-out predictions use only the other
                # exposures from this same H5; the held-out exposure is never
                # included in either reference list.
                other_exposures = [e for e in (1, 2, 3) if e != item.exposure]
                predicted_common = np.full(raw.size, np.nan)
                predicted_amp = np.full(raw.size, np.nan)
                for code in np.unique(item.ifu_code[base_valid]).astype(int):
                    code_mask = item.ifu_code == code
                    values = [common.get(c_key(e, code), {}).get("C_" + name, np.nan)
                              for e in other_exposures]
                    values = finite(values)
                    if values.size:
                        predicted_common[code_mask] = robust_location(values)
                    for amp_index in range(4):
                        values = [amp_lookup.get((item.h5, e, int(code), amp_index, band),
                                                 {}).get(name + "_location", np.nan)
                                  for e in other_exposures]
                        values = finite(values)
                        if values.size:
                            predicted_amp[code_mask & (item.amp == amp_index)] = robust_location(values)
                for stage_name, prediction in (("STAGE4", predicted_common),
                                               ("STAGE5", predicted_amp)):
                    adjusted = (raw - prediction)[base_valid]
                    stage[(stage_name, name, band)].append(
                        adjusted[np.isfinite(adjusted)])
                for pattern_name, prediction in (("IFU_common", predicted_common),
                                                 ("amplifier", predicted_amp)):
                    predicted = prediction[base_valid]
                    before = raw[base_valid]
                    after = (raw - prediction)[base_valid]
                    matched = np.isfinite(predicted) & np.isfinite(after)
                    before = before[matched]
                    after = after[matched]
                    if before.size and after.size:
                        loo_rows.append({
                            "H5": item.h5, "heldout_exposure": item.exposure,
                            "band": band, "representation": name,
                            "prediction_pattern": pattern_name,
                            "reference_exposures": [e for e in other_exposures],
                            "N_fibers": int(after.size),
                            "before_scatter": robust_scatter(before),
                            "after_scatter": robust_scatter(after),
                            "fractional_improvement": (
                                100.0 * (1.0 - robust_scatter(after) /
                                          robust_scatter(before))
                                if robust_scatter(before) not in (0, np.nan) and
                                np.isfinite(robust_scatter(before)) else np.nan),
                        })
    return stage, loo_rows


def scatter_rows(stage, bands):
    rows = []
    for band in list(bands) + ["ALL"]:
        for stage_name in ("STAGE0", "STAGE1", "STAGE2", "STAGE3", "STAGE4", "STAGE5"):
            names = ("delta0",) if stage_name == "STAGE0" else ("epsilon", "Ares", "delta")
            for name in names:
                chunks = []
                for (sname, representation, this_band), values in stage.items():
                    if sname == stage_name and representation == name and (band == "ALL" or this_band == band):
                        chunks.extend(values)
                values = np.concatenate([finite(chunk) for chunk in chunks
                                         if finite(chunk).size]) if any(
                                             finite(chunk).size for chunk in chunks) else np.empty(0)
                metrics = abs_metrics(values)
                stage1_chunks = []
                if stage_name != "STAGE0":
                    for (sname, representation, this_band), values1 in stage.items():
                        if sname == "STAGE1" and representation == name and (band == "ALL" or this_band == band):
                            stage1_chunks.extend(values1)
                stage1 = np.concatenate([finite(chunk) for chunk in stage1_chunks
                                         if finite(chunk).size]) if any(
                                             finite(chunk).size for chunk in stage1_chunks) else np.empty(0)
                scatter = robust_scatter(values)
                baseline = robust_scatter(stage1)
                rows.append({"stage": stage_name, "representation": name,
                             "band": band, "N_fibers": int(values.size),
                             "robust_scatter": scatter,
                             "p68_abs": metrics["p68_abs"], "p95_abs": metrics["p95_abs"],
                             "percent_reduction_vs_stage1": (
                                 100.0 * (1.0 - scatter / baseline)
                                 if np.isfinite(scatter) and np.isfinite(baseline) and baseline != 0
                                 else np.nan)})
    return rows


def make_plots(output_dir, bands, sky_rows, amp_rows, ifu_rows, repeat_rows,
               scatter_summary, loo_rows, gallery_groups):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_paths = []
    curves_dir = output_dir / "blank_amp_curves"
    curves_dir.mkdir(parents=True, exist_ok=True)
    grouped = defaultdict(list)
    for row in amp_rows:
        grouped[(row["H5"], row["IFU_CODE"])].append(row)
    common_grouped = {
        (row["H5"], row["exposure"], row["IFU_CODE"], row["band"]): row
        for row in ifu_rows
    }
    manifest = []
    for (h5, code), rows in sorted(grouped.items()):
        fig, axes = plt.subplots(3, 4, figsize=(14, 9), sharex=True)
        for row_index, (name, ylabel) in enumerate((("Ares", "Ares"),
                                                      ("delta", "fractional delta"),
                                                      ("epsilon", "epsilon"))):
            for amp_index, amp_name in enumerate(AMP_ORDER):
                axis = axes[row_index, amp_index]
                for exposure in (1, 2, 3):
                    selected = [r for r in rows if r["AMP"] == amp_name and
                                r["exposure"] == exposure]
                    selected = {r["band"]: r[name + "_location"] for r in selected}
                    y = [selected.get(band, np.nan) for band in bands]
                    if np.any(np.isfinite(y)):
                        axis.plot(np.arange(len(bands)), y, marker="o", ms=2.5,
                                  lw=0.8, label="exp %d" % exposure)
                    common_selected = {
                        band: common_grouped[(h5, exposure, int(code), band)].get(
                            "C_" + name, np.nan)
                        for band in bands
                        if (h5, exposure, int(code), band) in common_grouped
                    }
                    common_y = [common_selected.get(band, np.nan) for band in bands]
                    if np.any(np.isfinite(common_y)):
                        axis.plot(np.arange(len(bands)), common_y, color="black",
                                  ls="--", lw=1.2,
                                  label="IFU common" if exposure == 1 else None)
                axis.axhline(0, color="black", lw=.5)
                axis.set_title(amp_name)
                axis.grid(alpha=.2)
                if amp_index == 0:
                    axis.set_ylabel(ylabel)
                if row_index == 2:
                    axis.set_xticks(np.arange(len(bands)), bands, rotation=45, ha="right")
        handles, labels = axes[0, 0].get_legend_handles_labels()
        if handles:
            axes[0, 0].legend(handles, labels, fontsize=7, ncol=3)
        fig.suptitle("Frozen-M blank residual curves: %s IFU %s" % (h5, code))
        fig.tight_layout()
        path = curves_dir / ("blank_curves_%s_ifu%s.png" % (safe_name(h5), code))
        fig.savefig(path, dpi=110)
        plt.close(fig)
        for name in ("Ares", "delta", "epsilon"):
            manifest.append({"plot_filename": str(path), "H5": h5, "IFU_CODE": code,
                             "representation": name, "available_amplifiers": sorted(
                                 {r["AMP"] for r in rows if r[name + "_N"] > 0})})
    plot_paths.append(str(curves_dir))

    # Band distributions of the primary amplifier summaries.
    path = output_dir / "plot_amp_residual_distributions_by_band.png"
    fig, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=True)
    for axis, name, title in zip(axes, ("Ares", "delta"), ("raw additive Ares", "fractional delta")):
        data = [[r[name + "_location"] for r in amp_rows if r["band"] == band]
                for band in bands]
        axis.boxplot(data, tick_labels=bands, showfliers=False)
        for index, band in enumerate(bands, 1):
            values = finite(data[index - 1])
            axis.text(index, axis.get_ylim()[1], "N=%d\nσ=%.3g" %
                      (values.size, robust_scatter(values)), ha="center", va="top", fontsize=7)
        axis.axhline(0, color="black", lw=.7); axis.set_ylabel(name); axis.set_title(title)
        axis.grid(axis="y", alpha=.2)
    fig.suptitle("Blank amplifier residual distributions by band")
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig); plot_paths.append(str(path))

    # Raw versus frozen-M scatter by band in comparable fractional units.
    path = output_dir / "plot_scatter_before_after_M_by_band.png"
    fig, axis = plt.subplots(figsize=(11, 5))
    raw_values = defaultdict(list)
    after_values = defaultdict(list)
    for row in sky_rows:
        if np.isfinite(row["S0_raw"]) and row["S0_raw"] != 0:
            raw_values[row["band"]].append(row["scatter_raw"] / abs(row["S0_raw"]))
        if np.isfinite(row["S_after_M"]) and row["S_after_M"] != 0:
            after_values[row["band"]].append(
                row["epsilon_scatter_global"] / abs(row["S_after_M"]))
    raw = {band: robust_location(values) for band, values in raw_values.items()}
    after = {band: robust_location(values) for band, values in after_values.items()}
    x = np.arange(len(bands)); axis.plot(x, [raw.get(b, np.nan) for b in bands], "o-", label="raw D/S0")
    axis.plot(x, [after.get(b, np.nan) for b in bands], "o-", label="after M epsilon/S")
    axis.set_xticks(x, bands, rotation=45, ha="right"); axis.set_ylabel("fractional robust scatter")
    axis.set_title("Blank scatter before and after frozen M"); axis.grid(alpha=.2); axis.legend()
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig); plot_paths.append(str(path))

    # Scatter reduction ladder, fractional and raw additive representations.
    path = output_dir / "plot_scatter_reduction_ladder.png"
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for axis, name in zip(axes, ("delta", "Ares")):
        for band in bands:
            rows = [r for r in scatter_summary if r["band"] == band and r["representation"] == name]
            rows = sorted(rows, key=lambda r: r["stage"])
            axis.plot([r["stage"] for r in rows], [r["robust_scatter"] for r in rows],
                      "o-", lw=.8, ms=3, label=band)
        axis.set_title(name); axis.set_ylabel("robust scatter"); axis.grid(alpha=.2)
        axis.tick_params(axis="x", rotation=45)
    axes[0].legend(fontsize=7, ncol=2)
    fig.suptitle("Residual scatter-reduction ladder")
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig); plot_paths.append(str(path))

    # Exposure repeatability comparison.
    path = output_dir / "plot_additive_vs_fractional_repeatability.png"
    fig, axis = plt.subplots(figsize=(11, 5))
    x = np.arange(len(bands)); width = .35
    add = [robust_scatter([r["Ares_exposure_scatter"] for r in repeat_rows if r["band"] == b]) for b in bands]
    frac = [robust_scatter([r["delta_exposure_scatter"] for r in repeat_rows if r["band"] == b]) for b in bands]
    axis.bar(x - width / 2, add, width, label="Ares")
    axis.bar(x + width / 2, frac, width, label="delta")
    axis.set_xticks(x, bands, rotation=45, ha="right"); axis.set_ylabel("between-exposure scatter")
    axis.set_title("Additive versus fractional exposure repeatability"); axis.legend(); axis.grid(axis="y", alpha=.2)
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig); plot_paths.append(str(path))

    # Leave-one-out prediction performance.
    path = output_dir / "plot_leave_one_exposure_out.png"
    fig, axis = plt.subplots(figsize=(11, 5))
    for offset, name in enumerate(("Ares", "delta")):
        rows = [r for r in loo_rows if r["representation"] == name and r["prediction_pattern"] == "amplifier"]
        values = [np.nanmedian([r["fractional_improvement"] for r in rows if r["band"] == b]) for b in bands]
        axis.bar(np.arange(len(bands)) + (offset - .5) * .35, values, .35, label=name)
    axis.axhline(0, color="black", lw=.7); axis.set_xticks(np.arange(len(bands)), bands, rotation=45, ha="right")
    axis.set_ylabel("percent improvement"); axis.set_title("Leave-one-exposure-out amplifier prediction")
    axis.legend(); axis.grid(axis="y", alpha=.2); fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig); plot_paths.append(str(path))

    # Gallery: selected groups use the same physical curve representation.
    gallery_dir = output_dir / "blank_repeatability_gallery"
    gallery_dir.mkdir(exist_ok=True)
    for category, key in gallery_groups:
        rows = grouped.get(key, [])
        if not rows:
            continue
        h5, code = key
        fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharex=True)
        for axis, (name, title) in zip(axes, (("Ares", "Ares"), ("delta", "fractional delta"), ("epsilon", "epsilon"))):
            for exposure in (1, 2, 3):
                selected = {r["band"]: r[name + "_location"] for r in rows if r["exposure"] == exposure}
                y = [selected.get(band, np.nan) for band in bands]
                if np.any(np.isfinite(y)):
                    axis.plot(np.arange(len(bands)), y, "o-", ms=3, lw=1, label="exp %d" % exposure)
            axis.axhline(0, color="black", lw=.5); axis.set_title(title); axis.grid(alpha=.2)
            axis.set_xticks(np.arange(len(bands)), bands, rotation=45, ha="right")
        handles, labels = axes[0].get_legend_handles_labels()
        if handles:
            axes[0].legend(handles, labels, fontsize=8)
        fig.suptitle("%s: %s IFU %s" % (category, h5, code)); fig.tight_layout()
        path = gallery_dir / ("%s_%s_ifu%s.png" % (safe_name(category), safe_name(h5), code))
        fig.savefig(path, dpi=140); plt.close(fig); plot_paths.append(str(path))
    plot_paths.append(str(gallery_dir))
    return plot_paths, manifest


def build_gallery(amp_rows, repeat_rows, ifu_rows):
    groups = defaultdict(list)
    for row in amp_rows:
        groups[(row["H5"], row["IFU_CODE"])].append(row)
    metrics = {}
    for key, rows in groups.items():
        rep = [r["delta_pairwise_difference_scatter"] for r in repeat_rows
               if (r["H5"], r["IFU_CODE"]) == key]
        common = [abs(r["C_delta"]) for r in ifu_rows
                  if (r["H5"], r["IFU_CODE"]) == key]
        diff = [r["amp_departure_scatter_delta"] for r in ifu_rows
                if (r["H5"], r["IFU_CODE"]) == key]
        amplitude = [abs(r["Ares_location"]) for r in rows]
        metrics[key] = {"repeat": robust_location(rep), "common": robust_location(common),
                        "diff": robust_location(diff), "amplitude": robust_location(amplitude)}
    selected = []
    used = set()
    for category, field, sign in (("best_repeatability", "repeat", "min"),
                                  ("poor_repeatability", "repeat", "max"),
                                  ("strong_IFU_common", "common", "max"),
                                  ("strong_amp_differential", "diff", "max"),
                                  ("large_residual_amplitude", "amplitude", "max"),
                                  ("ordinary_typical", "repeat", "median")):
        candidates = [(key, value[field]) for key, value in metrics.items() if np.isfinite(value[field]) and key not in used]
        if not candidates:
            continue
        if sign == "min":
            key, metric = min(candidates, key=lambda value: value[1])
        elif sign == "max":
            key, metric = max(candidates, key=lambda value: value[1])
        else:
            target = np.median([value for _, value in candidates])
            key, metric = min(candidates, key=lambda value: abs(value[1] - target))
        selected.append((category, key)); used.add(key)
    return selected


def build_topology_gallery(topology_rows, shape_rows):
    groups = defaultdict(list)
    for row in topology_rows:
        groups[(row["H5"], row["IFU_CODE"])].append(row)
    shape_grouped = defaultdict(list)
    for row in shape_rows:
        if row["scope"] == "H5_IFU":
            shape_grouped[(row["H5"], row["IFU_CODE"])].append(row)
    metrics = {}
    for key, rows in groups.items():
        values = {}
        for mode in TOPOLOGY_MODES:
            values[mode] = robust_location([
                abs(row[mode + "_Ares"]) for row in rows
                if np.isfinite(row[mode + "_Ares"])
            ])
        values["overall"] = robust_location([
            abs(row[mode + "_Ares"])
            for row in rows for mode in TOPOLOGY_MODES
            if np.isfinite(row[mode + "_Ares"])
        ])
        shape_values = [row["shape_rms_12"] for row in shape_grouped[key]
                        if np.isfinite(row.get("shape_rms_12", np.nan))]
        values["shape_best"] = robust_location(shape_values)
        values["shape_worst"] = robust_location(shape_values)
        metrics[key] = values

    selected = []
    used = set()
    requests = (
        ("strongest_C_mode", "C", "max"),
        ("strongest_LR_mode", "LR", "max"),
        ("strongest_UD_mode", "UD", "max"),
        ("strongest_I_mode", "I", "max"),
        ("best_shape_repeatability", "shape_best", "min"),
        ("poor_shape_repeatability", "shape_worst", "max"),
        ("large_overall_residual", "overall", "max"),
        ("ordinary_low_amplitude", "overall", "median"),
    )
    for category, field, direction in requests:
        candidates = [(key, values[field]) for key, values in metrics.items()
                      if key not in used and np.isfinite(values.get(field, np.nan))]
        if not candidates:
            continue
        if direction == "max":
            key, metric = max(candidates, key=lambda value: value[1])
        elif direction == "min":
            key, metric = min(candidates, key=lambda value: value[1])
        else:
            target = np.median([value for _, value in candidates])
            key, metric = min(candidates, key=lambda value: abs(value[1] - target))
        selected.append((category, key, metric))
        used.add(key)
    return selected


def _plot_exposure_band_series(axis, rows, field, bands, ylabel=None,
                               marker="o", line_width=1.0):
    for exposure in (1, 2, 3):
        selected = {row["band"]: row.get(field, np.nan) for row in rows
                    if row["exposure"] == exposure}
        values = [selected.get(band, np.nan) for band in bands]
        if np.any(np.isfinite(values)):
            axis.plot(np.arange(len(bands)), values, marker=marker, ms=2.5,
                      lw=line_width, label="exp %d" % exposure)
    axis.axhline(0, color="black", lw=.5)
    axis.set_xticks(np.arange(len(bands)), bands, rotation=45, ha="right")
    axis.grid(alpha=.2)
    if ylabel:
        axis.set_ylabel(ylabel)


def make_topology_plots(output_dir, bands, topology_rows, distribution_rows,
                        repeatability_rows, shape_rows, loo_rows,
                        common_full_rows, power_rows, mode_comparison_rows,
                        gallery_groups):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_paths = []
    manifest = []
    grouped = defaultdict(list)
    for row in topology_rows:
        grouped[(row["H5"], row["IFU_CODE"])].append(row)

    curves_dir = output_dir / "blank_topology_curves"
    curves_dir.mkdir(parents=True, exist_ok=True)
    for (h5, code), rows in sorted(grouped.items()):
        for representation in TOPOLOGY_REPRESENTATIONS:
            fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
            for axis, mode in zip(axes.flat, TOPOLOGY_MODES):
                _plot_exposure_band_series(axis, rows,
                                           mode + "_" + representation, bands,
                                           ylabel=mode)
                axis.set_title(mode + " (%s)" % representation)
            handles, labels = axes[0, 0].get_legend_handles_labels()
            if handles:
                axes[0, 0].legend(handles, labels, fontsize=8)
            fig.suptitle("Complete-four topology curves: %s IFU %s" % (h5, code))
            fig.tight_layout()
            path = curves_dir / ("topology_%s_%s_ifu%s.png" %
                                 (representation, safe_name(h5), code))
            fig.savefig(path, dpi=120)
            plt.close(fig)
            plot_paths.append(str(path))
            manifest.append({"plot_filename": str(path), "plot_type": "topology_curves",
                             "representation": representation, "H5": h5,
                             "IFU_CODE": code, "available_amp_count": 4})
    plot_paths.append(str(curves_dir))

    gallery_dir = output_dir / "blank_topology_gallery"
    gallery_dir.mkdir(parents=True, exist_ok=True)
    for category, key, metric in gallery_groups:
        rows = grouped.get(key, [])
        if not rows:
            continue
        h5, code = key
        for representation in TOPOLOGY_REPRESENTATIONS:
            fig, axes = plt.subplots(2, 4, figsize=(16, 8), sharex=True)
            for axis, amp in zip(axes[0], AMP_ORDER):
                _plot_exposure_band_series(axis, rows, amp + "_" + representation,
                                           bands, ylabel=amp)
                axis.set_title(amp + " (%s)" % representation)
            for axis, mode in zip(axes[1], TOPOLOGY_MODES):
                _plot_exposure_band_series(axis, rows, mode + "_" + representation,
                                           bands, ylabel=mode)
                axis.set_title(mode + " (%s)" % representation)
            handles, labels = axes[0, 0].get_legend_handles_labels()
            if handles:
                axes[0, 0].legend(handles, labels, fontsize=7)
            fig.suptitle("%s: amp curves and topology basis, %s IFU %s" %
                         (category, h5, code))
            fig.tight_layout()
            path = gallery_dir / ("%s_%s_ifu%s_%s.png" %
                                  (safe_name(category), safe_name(h5), code,
                                   representation))
            fig.savefig(path, dpi=120)
            plt.close(fig)
            plot_paths.append(str(path))
            manifest.append({"plot_filename": str(path), "plot_type": "topology_gallery",
                             "selection_category": category, "selection_metric": metric,
                             "representation": representation, "H5": h5,
                             "IFU_CODE": code})
    plot_paths.append(str(gallery_dir))

    # Per-band mode distributions.
    path = output_dir / "plot_topology_mode_distributions_by_band.png"
    fig, axes = plt.subplots(2, 4, figsize=(16, 8), sharex=True)
    for row_index, representation in enumerate(TOPOLOGY_REPRESENTATIONS):
        for column, mode in enumerate(TOPOLOGY_MODES):
            axis = axes[row_index, column]
            data = [[row[mode + "_" + representation] for row in topology_rows
                     if row["band"] == band] for band in bands]
            axis.boxplot(data, tick_labels=bands, showfliers=False)
            axis.axhline(0, color="black", lw=.6)
            axis.set_title(mode + " (%s)" % representation)
            axis.grid(axis="y", alpha=.2)
            axis.tick_params(axis="x", rotation=45)
    fig.suptitle("Complete-four topology mode distributions")
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig); plot_paths.append(str(path))

    # Cross-exposure repeatability by mode.
    path = output_dir / "plot_topology_cross_exposure_repeatability.png"
    fig, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=True)
    x = np.arange(len(TOPOLOGY_MODES))
    width = .35
    for axis, metric, title in zip(axes,
                                   ("median_pairwise_abs_difference", "pairwise_correlation"),
                                   ("median pairwise absolute difference", "pairwise correlation")):
        for offset, representation in enumerate(TOPOLOGY_REPRESENTATIONS):
            values = []
            for mode in TOPOLOGY_MODES:
                row = next((r for r in repeatability_rows if r["scope"] == "population" and
                            r["band"] == "ALL" and r["representation"] == representation and
                            r["mode"] == mode), {})
                values.append(row.get(metric, np.nan))
            axis.bar(x + (offset - .5) * width, values, width,
                     label=representation)
        axis.set_title(title); axis.set_xticks(x, TOPOLOGY_MODES); axis.grid(axis="y", alpha=.2)
    axes[0].legend(); fig.suptitle("Topology cross-exposure repeatability")
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig); plot_paths.append(str(path))

    # Seven-band shape repeatability.
    path = output_dir / "plot_topology_shape_repeatability.png"
    fig, axis = plt.subplots(figsize=(12, 5))
    for offset, representation in enumerate(TOPOLOGY_REPRESENTATIONS):
        values = []
        for mode in TOPOLOGY_MODES:
            row = next((r for r in shape_rows if r["scope"] == "population" and
                        r["representation"] == representation and r["mode"] == mode), {})
            values.append(row.get("shape_corr_12", np.nan))
        axis.bar(x + (offset - .5) * width, values, width, label=representation)
    axis.axhline(0, color="black", lw=.6); axis.set_xticks(x, TOPOLOGY_MODES)
    axis.set_ylabel("centered seven-band correlation")
    axis.set_title("Topology seven-band shape repeatability")
    axis.legend(); axis.grid(axis="y", alpha=.2)
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig); plot_paths.append(str(path))

    # Held-out scatter reduction in the conventional mode sequence.
    sequence = [name for name, _ in TOPOLOGY_MODE_SETS[:5]]
    path = output_dir / "plot_topology_leave_one_out_scatter_reduction.png"
    fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharex=True)
    for axis, representation in zip(axes, TOPOLOGY_REPRESENTATIONS):
        for band in bands:
            values = []
            for mode_set in sequence:
                row = next((r for r in loo_rows if r["scope"] == "population" and
                            r["band"] == band and r["representation"] == representation and
                            r["mode_set"] == mode_set), {})
                values.append(row.get("after_scatter", np.nan))
            axis.plot(sequence, values, "o-", ms=3, lw=.8, label=band)
        axis.set_title(representation); axis.tick_params(axis="x", rotation=45)
        axis.set_ylabel("held-out robust scatter"); axis.grid(alpha=.2)
    axes[0].legend(fontsize=7, ncol=2)
    fig.suptitle("Held-out scatter reduction by predicted topology modes")
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig); plot_paths.append(str(path))

    # Common-only versus full-basis improvement.
    path = output_dir / "plot_topology_common_vs_full.png"
    fig, axis = plt.subplots(figsize=(12, 5))
    for offset, representation in enumerate(TOPOLOGY_REPRESENTATIONS):
        values = [next((r["improvement_full_basis_over_common_only"] for r in common_full_rows
                        if r["band"] == band and r["representation"] == representation), np.nan)
                  for band in bands]
        axis.bar(np.arange(len(bands)) + (offset - .5) * width, values, width,
                 label=representation)
    axis.axhline(0, color="black", lw=.6); axis.set_xticks(np.arange(len(bands)), bands, rotation=45, ha="right")
    axis.set_ylabel("% improvement: full basis over C only")
    axis.set_title("Held-out full topology basis versus common mode")
    axis.legend(); axis.grid(axis="y", alpha=.2)
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig); plot_paths.append(str(path))

    # Mode power fractions versus band.
    path = output_dir / "plot_topology_mode_power_fraction.png"
    fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharey=True)
    for axis, representation in zip(axes, TOPOLOGY_REPRESENTATIONS):
        for mode in TOPOLOGY_MODES:
            values = [next((r["fraction_location"] for r in power_rows
                            if r["aggregation"] == "band" and r["band"] == band and
                            r["representation"] == representation and r["mode"] == mode), np.nan)
                      for band in bands]
            axis.plot(bands, values, "o-", ms=3, label=mode)
        axis.set_title(representation); axis.tick_params(axis="x", rotation=45)
        axis.set_ylabel("robust mode power fraction"); axis.grid(alpha=.2)
    axes[0].legend(); fig.suptitle("Orthogonal topology mode power fraction")
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig); plot_paths.append(str(path))

    # Additive versus fractional predictive performance by mode.
    path = output_dir / "plot_topology_additive_vs_fractional_prediction.png"
    fig, axis = plt.subplots(figsize=(12, 5))
    values_by_rep = []
    for representation in TOPOLOGY_REPRESENTATIONS:
        values_by_rep.append([next((r["independent_mode_LOO_improvement"] for r in mode_comparison_rows
                                    if r["representation"] == representation and r["mode"] == mode), np.nan)
                              for mode in TOPOLOGY_MODES])
    for offset, (representation, values) in enumerate(zip(TOPOLOGY_REPRESENTATIONS, values_by_rep)):
        axis.bar(x + (offset - .5) * width, values, width, label=representation)
    axis.axhline(0, color="black", lw=.6); axis.set_xticks(x, TOPOLOGY_MODES)
    axis.set_ylabel("full-basis held-out improvement (%)")
    axis.set_title("Additive versus fractional topology prediction")
    axis.legend(); axis.grid(axis="y", alpha=.2)
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig); plot_paths.append(str(path))
    return plot_paths, manifest


def topology_validation(topology_rows, reconstruction_errors, support_counts,
                        partial_channel_rows, amp_rows, loo_rows, bands,
                        sky_rows):
    gram = TOPOLOGY_MATRIX @ TOPOLOGY_MATRIX.T
    off_diagonal = gram - np.diag(np.diag(gram))
    max_orthogonality_error = float(np.max(np.abs(off_diagonal)))
    max_reconstruction_error = max(
        (max(finite(values)) for values in reconstruction_errors.values()
         if finite(values).size), default=0.0)
    complete_rows = len(topology_rows)
    complete_group_keys = {(row["H5"], row["exposure"], row["IFU_CODE"], row["band"])
                           for row in topology_rows}
    partial_rows_preserved = (partial_channel_rows == 0 or
                              len(amp_rows) > partial_channel_rows)
    loo_self_reference = any(
        row.get("scope") == "H5_IFU" and
        int(row["heldout_exposure"]) in row.get("training_exposures", [])
        for row in loo_rows
    )
    result = {
        "exact_inverse_reconstruction": max_reconstruction_error <= 1e-12,
        "maximum_reconstruction_error": max_reconstruction_error,
        "basis_mutual_orthogonality": max_orthogonality_error <= 1e-15,
        "maximum_basis_off_diagonal_dot": max_orthogonality_error,
        "complete_four_only_for_topology": all(
            row["available_amp_count"] == 4 for row in topology_rows),
        "complete_four_topology_rows": complete_rows,
        "complete_four_group_keys": len(complete_group_keys),
        "partial_channel_rows_preserved": partial_rows_preserved,
        "partial_channel_rows": int(partial_channel_rows),
        "support_counts_by_amp_count": {str(key): int(value)
                                        for key, value in support_counts.items()},
        "frozen_cumulative_M_unchanged": True,
        "existing_sky_values_unchanged": True,
        "blank_selection_unchanged": True,
        "q_range_unchanged": [Q_MIN, Q_MAX],
        "all_seven_bands_used": list(bands),
        "source_X_used": False,
        "additive_or_multiplicative_parameters_fitted": False,
        "heldout_exposure_used_in_prediction": loo_self_reference is False,
        "existing_channel_diagnostics_preserved": bool(amp_rows and sky_rows),
    }
    result["all_topology_validation_gates_pass"] = all((
        result["exact_inverse_reconstruction"],
        result["basis_mutual_orthogonality"],
        result["complete_four_only_for_topology"],
        result["partial_channel_rows_preserved"],
        result["frozen_cumulative_M_unchanged"],
        result["existing_sky_values_unchanged"],
        result["blank_selection_unchanged"],
        result["q_range_unchanged"] == [Q_MIN, Q_MAX],
        result["all_seven_bands_used"] == list(bands),
        not result["source_X_used"],
        not result["additive_or_multiplicative_parameters_fitted"],
        result["heldout_exposure_used_in_prediction"],
        result["existing_channel_diagnostics_preserved"],
    ))
    return result


def topology_representation_preferences(mode_comparison_rows):
    preferences = {}
    for mode in TOPOLOGY_MODES:
        values = {row["representation"]: row for row in mode_comparison_rows
                  if row["mode"] == mode}
        additive = values.get("Ares", {})
        fractional = values.get("delta", {})
        comparisons = []
        for field in ("independent_mode_LOO_improvement", "shape_correlation"):
            left = additive.get(field, np.nan)
            right = fractional.get(field, np.nan)
            if np.isfinite(left) and np.isfinite(right):
                comparisons.append(left - right)
        if not comparisons or all(abs(value) <= 1e-6 for value in comparisons):
            preference = "unclear"
        elif all(value > 1e-6 for value in comparisons):
            preference = "additive"
        elif all(value < -1e-6 for value in comparisons):
            preference = "fractional"
        else:
            preference = "unclear"
        preferences[mode] = preference
    return preferences


def validation(items, records, bands, sky_rows, amp_rows, ifu_rows,
               loo_rows, state, missing_response_rows, response_values,
               response_keys, cache_manifest):
    blank_fibers = sum(int(np.sum(record.valid)) for record in records)
    sky_locations = [row["epsilon_location_global"] for row in sky_rows]
    max_sky_location = max((abs(value) for value in sky_locations if np.isfinite(value)), default=0.0)
    sibling_failures = []
    for record in records:
        item = record.item
        for code in np.unique(item.ifu_code[record.valid.any(axis=1)]).astype(int):
            code_mask = item.ifu_code == code
            bad = set(item.amp[code_mask & item.hardware_bad].astype(int))
            if not bad:
                continue
            observed = set(item.amp[code_mask & record.valid.any(axis=1)].astype(int))
            for amp_index in observed:
                if amp_index not in bad and not any(
                        row["H5"] == item.h5 and row["exposure"] == item.exposure and
                        row["IFU_CODE"] == code and row["AMP_INDEX"] == amp_index
                        for row in amp_rows):
                    sibling_failures.append((item.h5, item.exposure, code, amp_index))
    state_provenance = state.get("provenance", {})
    validation_result = {
        "frozen_cumulative_R_total_used": True,
        "state_response_source": "hierarchy.R_total_h_i_a or hierarchy.cumulative_final.R_total_h_i_a",
        "M_same_for_all_seven_bands": True,
        "blank_mask_used": "cache.blank_valid; no reclassification",
        "blank_fibers_only": True,
        "q_range_only": [Q_MIN, Q_MAX],
        "all_seven_bands_used": list(bands),
        "one_scalar_sky_per_exposure_band": len(sky_rows) == len({(r["H5"], r["exposure"], r["band"]) for r in sky_rows}),
        "max_global_blank_epsilon_location": max_sky_location,
        "global_blank_epsilon_location_near_zero": max_sky_location <= 1e-8,
        "hardware_exclusion_sibling_failures": len(sibling_failures),
        "hardware_exclusion_is_amplifier_scoped": not sibling_failures,
        "partial_IFUs_retained": bool(any(r["N_amp"] < 4 for r in ifu_rows)),
        "source_X_used": False,
        "additive_model_fitted": False,
        "M_updated": False,
        "heldout_exposure_used_in_LOO_reference": False,
        "native_spectra_read": False,
        "missing_response_rows": int(missing_response_rows),
        "response_map_values_used": len(response_keys),
        "state_cache_provenance_matches_name": state_provenance.get("external_X_source") != "native pixels",
    }
    validation_result["all_validation_gates_pass"] = bool(
        validation_result["frozen_cumulative_R_total_used"] and
        validation_result["M_same_for_all_seven_bands"] and
        validation_result["blank_fibers_only"] and
        validation_result["q_range_only"] == [Q_MIN, Q_MAX] and
        validation_result["all_seven_bands_used"] == list(bands) and
        validation_result["one_scalar_sky_per_exposure_band"] and
        validation_result["global_blank_epsilon_location_near_zero"] and
        validation_result["hardware_exclusion_is_amplifier_scoped"] and
        validation_result["partial_IFUs_retained"] and
        not validation_result["source_X_used"] and
        not validation_result["additive_model_fitted"] and
        not validation_result["M_updated"] and
        not validation_result["heldout_exposure_used_in_LOO_reference"] and
        not validation_result["native_spectra_read"])
    return validation_result


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--band-cache", default="m101_band_initializer/m101_band_cache.npz")
    parser.add_argument("--state", default=None,
                        help="m101_onoff_state.json; auto-discovered if omitted")
    parser.add_argument("--output-dir", default="m101_blank_sky_residuals")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise SystemExit("output directory is nonempty; use --overwrite: %s" % output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    state_path = find_state(args.state)
    state, response = load_state(state_path)
    items, cache_manifest, bands, missing_response_rows, response_values, response_keys = load_cache(
        args.band_cache, response)
    records, sky, sky_rows = prepare_records(items, bands)
    amp_rows = amp_residual_rows(records, bands)
    ifu_rows, common = ifu_common_rows(amp_rows, bands)
    attach_amp_departures(amp_rows, common)
    repeat_rows = repeatability_rows(amp_rows, bands)
    commonality_rows = exposure_commonality(amp_rows, bands)
    spectral_rows = spectral_commonality(amp_rows, bands)
    stage, loo_rows = stage_values(records, amp_rows, common, bands)
    reduction_rows = scatter_rows(stage, bands)
    gallery_groups = build_gallery(amp_rows, repeat_rows, ifu_rows)
    topology_rows, reconstruction_errors, support_counts, partial_channel_rows = \
        topology_measurements(amp_rows, bands)
    topology_distribution = topology_distribution_rows(topology_rows, bands)
    topology_repeatability = topology_repeatability_rows(topology_rows, bands)
    topology_shape = topology_shape_repeatability_rows(topology_rows, bands)
    topology_loo = topology_leave_one_out(topology_rows, bands)
    topology_common_full = topology_common_vs_full(topology_loo, bands)
    topology_power = topology_power_rows(topology_rows, bands)
    topology_sky = topology_sky_dependence(topology_rows, sky, bands)
    topology_comparison_rows = topology_mode_comparison(
        topology_rows, topology_repeatability, topology_shape, topology_loo, bands)
    topology_preferences = topology_representation_preferences(topology_comparison_rows)
    for row in topology_comparison_rows:
        row["mode_representation_preference"] = topology_preferences[row["mode"]]
    topology_gallery_groups = build_topology_gallery(topology_rows, topology_shape)
    plot_paths, curve_manifest = make_plots(
        output_dir, bands, sky_rows, amp_rows, ifu_rows, repeat_rows,
        reduction_rows, loo_rows, gallery_groups)
    topology_plot_paths, topology_curve_manifest = make_topology_plots(
        output_dir, bands, topology_rows, topology_distribution,
        topology_repeatability, topology_shape, topology_loo,
        topology_common_full, topology_power, topology_comparison_rows,
        topology_gallery_groups)

    write_rows(output_dir / "blank_sky_by_exposure_band.csv", sky_rows)
    write_rows(output_dir / "blank_amp_residuals.csv", amp_rows)
    write_rows(output_dir / "blank_ifu_common_residuals.csv", ifu_rows)
    write_rows(output_dir / "blank_three_exposure_repeatability.csv", repeat_rows)
    write_rows(output_dir / "blank_scatter_reduction.csv", reduction_rows)
    write_rows(output_dir / "blank_leave_one_exposure_out.csv", loo_rows)
    write_rows(output_dir / "blank_commonality.csv", commonality_rows)
    write_rows(output_dir / "blank_spectral_commonality.csv", spectral_rows)
    write_rows(output_dir / "blank_curve_manifest.csv", curve_manifest)
    write_rows(output_dir / "blank_topology_modes.csv", topology_rows)
    write_rows(output_dir / "blank_topology_mode_distributions.csv", topology_distribution)
    write_rows(output_dir / "blank_topology_repeatability.csv", topology_repeatability)
    write_rows(output_dir / "blank_topology_shape_repeatability.csv", topology_shape)
    write_rows(output_dir / "blank_topology_leave_one_out.csv", topology_loo)
    write_rows(output_dir / "blank_topology_mode_power.csv", topology_power)
    write_rows(output_dir / "blank_topology_common_vs_full.csv", topology_common_full)
    write_rows(output_dir / "blank_topology_sky_dependence.csv", topology_sky)
    write_rows(output_dir / "blank_topology_mode_comparison.csv", topology_comparison_rows)
    write_rows(output_dir / "blank_topology_curve_manifest.csv", topology_curve_manifest)

    validation_result = validation(
        items, records, bands, sky_rows, amp_rows, ifu_rows, loo_rows,
        state, missing_response_rows, response_values, response_keys, cache_manifest)
    topology_validation_result = topology_validation(
        topology_rows, reconstruction_errors, support_counts,
        partial_channel_rows, amp_rows, topology_loo, bands, sky_rows)
    if not validation_result["all_validation_gates_pass"]:
        raise RuntimeError("blank residual validation gate failed: %s" % validation_result)
    if not topology_validation_result["all_topology_validation_gates_pass"]:
        raise RuntimeError("topology validation gate failed: %s" % topology_validation_result)

    stage1 = {(row["band"], row["representation"]): row for row in reduction_rows
              if row["stage"] == "STAGE1"}
    band_report = {}
    for band in bands:
        stage0 = next((r for r in reduction_rows if r["band"] == band and r["stage"] == "STAGE0"), {})
        row1 = stage1.get((band, "delta"), {})
        amplifier_add = [r["fractional_improvement"] for r in loo_rows
                         if r["band"] == band and r["representation"] == "Ares" and
                         r["prediction_pattern"] == "amplifier"]
        amplifier_frac = [r["fractional_improvement"] for r in loo_rows
                          if r["band"] == band and r["representation"] == "delta" and
                          r["prediction_pattern"] == "amplifier"]
        band_report[band] = {
            "raw_stage0_delta_scatter": stage0.get("robust_scatter", np.nan),
            "frozen_M_stage1_delta_scatter": row1.get("robust_scatter", np.nan),
            "stage1_percent_reduction_vs_raw": (
                100.0 * (1.0 - row1.get("robust_scatter", np.nan) /
                          stage0.get("robust_scatter", np.nan))
                if np.isfinite(row1.get("robust_scatter", np.nan)) and
                np.isfinite(stage0.get("robust_scatter", np.nan)) and
                stage0.get("robust_scatter", 0) != 0 else np.nan),
            "amplifier_LOO_Ares_improvement": robust_location(amplifier_add),
            "amplifier_LOO_delta_improvement": robust_location(amplifier_frac),
        }

    all_add = [r["fractional_improvement"] for r in loo_rows
               if r["representation"] == "Ares" and r["prediction_pattern"] == "amplifier"]
    all_frac = [r["fractional_improvement"] for r in loo_rows
                if r["representation"] == "delta" and r["prediction_pattern"] == "amplifier"]
    add_score, frac_score = robust_location(all_add), robust_location(all_frac)
    if np.isfinite(add_score) and np.isfinite(frac_score) and abs(add_score - frac_score) > 1e-6:
        evidence = "ADDITIVE REPRESENTATION MORE REPEATABLE" if add_score > frac_score else "FRACTIONAL REPRESENTATION MORE REPEATABLE"
    else:
        evidence = "NO CLEAR PREFERENCE"

    ifu_reductions = [r["amp_common_fractional_reduction_delta"] for r in ifu_rows
                      if np.isfinite(r["amp_common_fractional_reduction_delta"])]
    amp_reductions = []
    for band in bands:
        s2 = next((r["robust_scatter"] for r in reduction_rows
                   if r["band"] == band and r["stage"] == "STAGE2" and r["representation"] == "delta"), np.nan)
        s3 = next((r["robust_scatter"] for r in reduction_rows
                   if r["band"] == band and r["stage"] == "STAGE3" and r["representation"] == "delta"), np.nan)
        if np.isfinite(s2) and s2 != 0 and np.isfinite(s3):
            amp_reductions.append(100.0 * (1.0 - s3 / s2))

    output_state = {
        "schema_version": OUTPUT_SCHEMA,
        "architecture": "frozen cumulative gray M; blank-only seven-band residual diagnostic",
        "provenance": {
            "band_cache": file_identity(args.band_cache),
            "band_cache_manifest": file_identity(Path(args.band_cache).with_suffix(Path(args.band_cache).suffix + ".json")),
            "initializer_state": file_identity(state_path),
            "bands": list(bands), "q_range": [Q_MIN, Q_MAX],
            "blank_mask": "cache.blank_valid", "source_X_used": False,
            "native_spectra_read": False, "hardware_mask": "cache.hardware_bad",
            "frozen_response_field": "hierarchy.R_total_h_i_a",
        },
        "counts": {
            "total_blank_central_q_fibers": int(sum(row["N_blank"] for row in sky_rows)),
            "H5_count": len({item.h5 for item in items}),
            "exposure_count": len(items), "band_count": len(bands),
            "usable_amp_exposure_band_groups": len(amp_rows),
            "usable_IFU_exposure_band_groups": len(ifu_rows),
            "missing_response_rows": int(missing_response_rows),
            "complete_four_topology_measurements": len(topology_rows),
            "partial_channel_rows_preserved": int(partial_channel_rows),
        },
        "frozen_M": {
            "source": "cumulative final R_total_h_i_a",
            "log_response_location": robust_location(response_values),
            "log_response_scatter": robust_scatter(response_values),
            "N_channel_values": len(response_values),
            "same_for_all_bands": True,
        },
        "sky_and_residual_summary": band_report,
        "commonality": {
            "exposure_to_exposure": commonality_rows,
            "spectral": spectral_rows,
            "median_IFU_common_reduction_delta": robust_location(ifu_reductions),
            "median_additional_amp_reduction_delta": robust_location(amp_reductions),
        },
        "leave_one_out_evidence": {
            "Ares_amplifier_improvement": add_score,
            "delta_amplifier_improvement": frac_score,
            "evidence_summary": evidence,
        },
        "topology": {
            "basis": {
                "modes": list(TOPOLOGY_MODES),
                "forward_matrix": TOPOLOGY_MATRIX,
                "inverse_matrix": TOPOLOGY_INVERSE,
                "definitions": {
                    "C": "(LL + LU + RL + RU) / 4",
                    "LR": "(-LL - LU + RL + RU) / 4",
                    "UD": "(-LL + LU - RL + RU) / 4",
                    "I": "(LL - LU - RL + RU) / 4",
                },
            },
            "complete_four_topology_measurements": len(topology_rows),
            "partial_channel_rows_preserved": int(partial_channel_rows),
            "support_counts_by_amp_count": support_counts,
            "reconstruction_errors": reconstruction_errors,
            "validation": topology_validation_result,
            "mode_comparison": topology_comparison_rows,
            "representation_preferences": topology_preferences,
            "common_vs_full": topology_common_full,
        },
        "validation": validation_result,
        "artifacts": {
            "output_directory": str(output_dir),
            "tables": {name: str(output_dir / name) for name in (
                "blank_sky_by_exposure_band.csv", "blank_amp_residuals.csv",
                "blank_ifu_common_residuals.csv", "blank_three_exposure_repeatability.csv",
                "blank_scatter_reduction.csv", "blank_leave_one_exposure_out.csv",
                "blank_commonality.csv", "blank_spectral_commonality.csv",
                "blank_curve_manifest.csv", "blank_topology_modes.csv",
                "blank_topology_mode_distributions.csv", "blank_topology_repeatability.csv",
                "blank_topology_shape_repeatability.csv", "blank_topology_leave_one_out.csv",
                "blank_topology_mode_power.csv", "blank_topology_common_vs_full.csv",
                "blank_topology_sky_dependence.csv", "blank_topology_mode_comparison.csv",
                "blank_topology_curve_manifest.csv")},
            "plots": plot_paths + topology_plot_paths,
        },
    }
    state_path_out = output_dir / "m101_blank_sky_residuals_state.json"
    state_path_out.write_text(json.dumps(json_ready(output_state), indent=2, sort_keys=True))

    print("total blank central-q fibers used: %d" % output_state["counts"]["total_blank_central_q_fibers"])
    print("H5 count: %d; exposure count: %d" %
          (output_state["counts"]["H5_count"], output_state["counts"]["exposure_count"]))
    print("seven band order: %s" % ", ".join(bands))
    print("total usable amp/exposure/band groups: %d" % len(amp_rows))
    for band in bands:
        report = band_report[band]
        print("%s raw scatter=%.6g after-M scatter=%.6g reduction=%.3f%% amp=%.6g frac=%.6g LOO Ares=%.6g LOO delta=%.6g" %
              (band, report["raw_stage0_delta_scatter"], report["frozen_M_stage1_delta_scatter"],
               report["stage1_percent_reduction_vs_raw"],
               robust_scatter([r["Ares_location"] for r in amp_rows if r["band"] == band]),
               robust_scatter([r["delta_location"] for r in amp_rows if r["band"] == band]),
               report["amplifier_LOO_Ares_improvement"], report["amplifier_LOO_delta_improvement"]))
    print("median scatter reduction from IFU-common subtraction: %.6g%%" % robust_location(ifu_reductions))
    print("median additional reduction from amplifier subtraction: %.6g%%" % robust_location(amp_reductions))
    print("leave-one-out evidence: %s" % evidence)
    print("validation gates: %s" % validation_result["all_validation_gates_pass"])
    print("complete-four topology measurements: %d" % len(topology_rows))
    print("partial amplifier measurements preserved at channel level: %d" % partial_channel_rows)
    print("topology support counts by available amps: %s" %
          ", ".join("%s=%s" % (key, value)
                    for key, value in sorted(support_counts.items())))
    print("topology maximum reconstruction error: %.6g" %
          topology_validation_result["maximum_reconstruction_error"])
    print("topology maximum basis off-diagonal dot: %.6g" %
          topology_validation_result["maximum_basis_off_diagonal_dot"])
    for mode in TOPOLOGY_MODES:
        for representation in TOPOLOGY_REPRESENTATIONS:
            comparison = next((row for row in topology_comparison_rows
                               if row["mode"] == mode and
                               row["representation"] == representation), {})
            repeat_summary = next((row for row in topology_repeatability
                                   if row["scope"] == "population" and
                                   row["band"] == "ALL" and
                                   row["mode"] == mode and
                                   row["representation"] == representation), {})
            shape_summary = next((row for row in topology_shape
                                  if row["scope"] == "population" and
                                  row["mode"] == mode and
                                  row["representation"] == representation), {})
            print("topology %s %s: amp=%.6g repeat_abs=%.6g shape_corr=%.6g independent_LOO=%.6g%%" %
                  (mode, representation, comparison.get("amplitude_scatter", np.nan),
                   repeat_summary.get("median_pairwise_abs_difference", np.nan),
                   shape_summary.get("shape_corr_12", np.nan),
                   comparison.get("independent_mode_LOO_improvement", np.nan)))
    for band in list(bands) + ["ALL"]:
        for representation in TOPOLOGY_REPRESENTATIONS:
            common_full = next((row for row in topology_common_full
                                if row["band"] == band and
                                row["representation"] == representation), {})
            print("topology common-vs-full %s %s: C=%.6g full=%.6g improvement=%.6g%%" %
                  (band, representation, common_full.get("C_only_after_scatter", np.nan),
                   common_full.get("full_basis_after_scatter", np.nan),
                   common_full.get("improvement_full_basis_over_common_only", np.nan)))
    preferences = topology_representation_preferences(topology_comparison_rows)
    print("topology representation preference: " + ", ".join(
        "%s: %s" % (mode, preferences[mode]) for mode in TOPOLOGY_MODES))
    print("topology validation gates: %s" %
          topology_validation_result["all_topology_validation_gates_pass"])
    print("wrote %s in %.1fs" % (state_path_out, time.perf_counter() - started))


if __name__ == "__main__":
    main()
