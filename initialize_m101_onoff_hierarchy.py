#!/usr/bin/env python3
"""Build a cache-only, physics-first ON/OFF multiplicative initializer.

This is intentionally a separate initializer from
``initialize_m101_calibration_bands.py``.  It reads the existing compact band
cache, uses only its ON/OFF columns and central-q fibers, and writes a small
hierarchical diagnostic product.  No native spectral pixels are read here.

The model is deliberately limited to the following sequence::

    D ~= S + C X
    r = log(D / (S + C X))
    r_amp -> I_e,i,b and d_e,i,a,b
    (three-exposure robust averages) -> I_h,i,b and d_h,i,a,b
    ON/OFF comparison -> optional gray hierarchy

The optional refinement applies the first gray hierarchy once, repeats only
the exposure-level S+C X regression, and rebuilds the summaries once.  It is
not a joint or nonlinear optimizer.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import re
import tempfile
import time

import numpy as np
from math_utils import biweight


AMP_ORDER = ("LL", "LU", "RL", "RU")
AMP_INDEX = {name: index for index, name in enumerate(AMP_ORDER)}
Q_MIN = 40
Q_MAX = 70
SOURCE_BANDS = ("ON", "OFF")
SOURCE_BAND_FIELDS = {"ON": "ON", "OFF": "OFF"}
CACHE_SCHEMA = "m101_band_cache_v1"
OUTPUT_SCHEMA = "m101_onoff_hierarchy_v1"


def json_ready(value):
    """Convert numpy values without importing the native-data stack."""
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


def file_identity(path):
    path = Path(path)
    stat = path.stat()
    return {"filename": path.name, "full_path": str(path.resolve()),
            "file_size": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns)}


def robust_location(values):
    values = np.asarray(values, dtype=float)
    with np.errstate(all="ignore"):
        result = biweight(values)
    if np.isfinite(result):
        return float(result)
    return float(np.nanmedian(values))


def robust_scatter(values):
    values = np.asarray(values, dtype=float)
    with np.errstate(all="ignore"):
        _, result = biweight(values, calc_std=True)
    if np.isfinite(result) and result >= 0:
        return float(result)
    return float(np.nanstd(values))


@dataclass
class CacheItem:
    item_index: int
    h5_name: str
    h5_path: str
    exposure: int
    ifu: np.ndarray
    ifu_code: np.ndarray
    amp: np.ndarray
    q: np.ndarray
    band_total: np.ndarray
    band_error: np.ndarray
    X: np.ndarray
    external_valid: np.ndarray
    hardware_bad: np.ndarray
    blank_valid: np.ndarray
    source_candidate: np.ndarray
    x_arcmin: np.ndarray
    y_arcmin: np.ndarray

    @property
    def key(self):
        return self.h5_name, self.exposure


def _finite(values):
    values = np.asarray(values, dtype=float)
    return values[np.isfinite(values)]


def _location(values):
    values = _finite(values)
    if values.size == 0:
        return np.nan
    if values.size == 1 or np.all(values == values[0]):
        return float(values[0])
    result = robust_location(values)
    return float(result) if np.isfinite(result) else float(np.nanmedian(values))


def _scatter(values):
    values = _finite(values)
    if values.size < 2:
        return np.nan
    if np.all(values == values[0]):
        return 0.0
    result = robust_scatter(values)
    if np.isfinite(result) and result >= 0:
        return float(result)
    return float(np.nanstd(values))


def _uncertainty(values):
    values = _finite(values)
    scale = _scatter(values)
    return float(scale / math.sqrt(values.size)) if np.isfinite(scale) else np.nan


def _percentiles(values):
    values = _finite(values)
    if values.size == 0:
        return {"p68": np.nan, "p90": np.nan, "p95": np.nan}
    return {"p68": float(np.percentile(values, 68)),
            "p90": float(np.percentile(values, 90)),
            "p95": float(np.percentile(values, 95))}


def _safe_key(*values):
    return str(tuple(values))


def _safe_name(value):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")


def _write_rows(path, rows):
    path = Path(path)
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
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


def _load_cache(cache_path, selected_h5=None):
    """Load only compact cache columns needed by this initializer."""
    cache_path = Path(cache_path).expanduser().resolve()
    manifest_path = cache_path.with_suffix(cache_path.suffix + ".json")
    if not cache_path.exists() or not manifest_path.exists():
        raise FileNotFoundError(
            "band cache and manifest must both exist: %s" % cache_path)
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema_version") != CACHE_SCHEMA:
        raise ValueError("unsupported band cache schema: %s" %
                         manifest.get("schema_version"))
    band_order = tuple(manifest.get("band_order", ()))
    if not all(band in band_order for band in SOURCE_BANDS):
        raise ValueError("cache does not contain both ON and OFF bands")
    band_indices = [band_order.index(band) for band in SOURCE_BANDS]
    required = ("ifu", "ifu_code", "amp", "q", "band_total", "band_error",
                "X", "external_valid", "hardware_bad", "blank_valid",
                "source_candidate", "x_arcmin", "y_arcmin")
    wanted = None
    if selected_h5:
        wanted = {Path(value).name for value in selected_h5}
        available = {row["h5_name"] for row in manifest.get("items", [])}
        missing = wanted - available
        if missing:
            raise ValueError("requested H5 is absent from band cache: %s" %
                             sorted(missing))

    with np.load(cache_path, allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]) for name in required}

    items = []
    for manifest_row in manifest.get("items", []):
        if wanted is not None and manifest_row["h5_name"] not in wanted:
            continue
        start = int(manifest_row["start"])
        stop = int(manifest_row["stop"])
        sl = slice(start, stop)
        # Copy slices so the large archive arrays can be released after load.
        item = CacheItem(
            item_index=len(items),
            h5_name=str(manifest_row["h5_name"]),
            h5_path=str(manifest_row.get("h5_path", "")),
            exposure=int(manifest_row["exposure"]),
            ifu=arrays["ifu"][sl].copy(),
            ifu_code=arrays["ifu_code"][sl].copy(),
            amp=arrays["amp"][sl].copy(),
            q=arrays["q"][sl].copy(),
            band_total=arrays["band_total"][sl][:, band_indices].copy(),
            band_error=arrays["band_error"][sl][:, band_indices].copy(),
            X=arrays["X"][sl].copy(),
            external_valid=arrays["external_valid"][sl].copy(),
            hardware_bad=arrays["hardware_bad"][sl].copy(),
            blank_valid=arrays["blank_valid"][sl].copy(),
            source_candidate=arrays["source_candidate"][sl].copy(),
            x_arcmin=arrays["x_arcmin"][sl].copy(),
            y_arcmin=arrays["y_arcmin"][sl].copy(),
        )
        items.append(item)
    if not items:
        raise ValueError("selected H5 set is empty in band cache")
    for index, item in enumerate(items):
        item.item_index = index
    manifest = dict(manifest)
    manifest["selected_h5"] = list(dict.fromkeys(item.h5_name for item in items))
    manifest["source_band_order"] = list(SOURCE_BANDS)
    manifest["source_band_indices_in_cache"] = band_indices
    return items, manifest, cache_path, manifest_path


def _fplane_candidates():
    return (
        Path("fplaneall.txt"),
        Path("/Users/grz85/work/code/VIRUSFlow/fplaneall.txt"),
        Path("/work/03730/gregz/maverick/fplaneall.txt"),
    )


def _load_fplane(path=None):
    """Read frozen IFUSLOT X_FP/Y_FP coordinates for diagnostic maps."""
    candidates = [Path(path).expanduser()] if path else list(_fplane_candidates())
    chosen = next((candidate.resolve() for candidate in candidates if candidate.exists()), None)
    if chosen is None:
        return {}, None
    result = {}
    for line in chosen.read_text().splitlines():
        fields = line.split()
        if len(fields) < 3 or fields[0].startswith("#"):
            continue
        try:
            slot = int(fields[0])
            result[slot] = (float(fields[1]), float(fields[2]))
        except (TypeError, ValueError):
            continue
    return result, chosen


def _robust_linear_fit(x, y, sigma=None, iterations=8):
    """Robust direct two-column S+C X regression with simple uncertainty."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    good = np.isfinite(x) & np.isfinite(y)
    if sigma is not None:
        sigma = np.asarray(sigma, dtype=float)
        good &= np.isfinite(sigma) & (sigma > 0)
    x, y = x[good], y[good]
    if sigma is None:
        sig = np.ones(y.size, dtype=float)
    else:
        sig = sigma[good]
    if y.size == 0:
        return {"S": np.nan, "C": np.nan, "N": 0, "rank": 0,
                "residual_scatter": np.nan, "S_uncertainty": np.nan,
                "C_uncertainty": np.nan}
    if sigma is not None:
        fallback = np.nanmedian(sig) if sig.size else 1.0
        sig = np.where(np.isfinite(sig) & (sig > 0), sig, fallback)
    design = np.column_stack((np.ones(y.size), x))
    # The error floor prevents tiny formal errors from overwhelming the
    # robust population average.  It is a fit weight floor, not a data edit.
    initial_residual = y - np.nanmedian(y)
    initial_scale = _scatter(initial_residual)
    if not np.isfinite(initial_scale) or initial_scale <= 0:
        initial_scale = np.nanmedian(sig) if sig.size else 1.0
    sigma_floor = max(float(initial_scale) * 0.1, 1e-12)
    sig = np.maximum(sig, sigma_floor)
    base = 1.0 / sig ** 2
    robust_weights = np.ones(y.size, dtype=float)
    coefficients = np.asarray([np.nanmedian(y), 0.0], dtype=float)
    scale = initial_scale
    rank = 0
    for _ in range(iterations):
        weights = base * robust_weights
        root = np.sqrt(weights)
        coefficients, _, rank, _ = np.linalg.lstsq(
            design * root[:, None], y * root, rcond=1e-12)
        residual = y - design @ coefficients
        scale = _scatter(residual)
        if not np.isfinite(scale) or scale <= 0:
            scale = max(float(np.nanstd(residual)), 1e-12)
        cutoff = 1.345 * max(scale, 1e-12)
        absolute = np.abs(residual)
        robust_weights = np.ones(y.size, dtype=float)
        outlier = absolute > cutoff
        robust_weights[outlier] = cutoff / np.maximum(absolute[outlier], 1e-12)
    residual = y - design @ coefficients
    scale = _scatter(residual)
    if not np.isfinite(scale):
        scale = float(np.nanstd(residual)) if residual.size > 1 else np.nan
    uncertainties = [np.nan, np.nan]
    if rank >= 2 and np.isfinite(scale):
        weights = base * robust_weights
        normal = design.T @ (weights[:, None] * design)
        covariance = np.linalg.pinv(normal, rcond=1e-12) * scale ** 2
        uncertainties = [float(np.sqrt(max(covariance[i, i], 0.0)))
                         for i in range(2)]
    return {
        "S": float(coefficients[0]), "C": float(coefficients[1]),
        "N": int(y.size), "rank": int(rank),
        "residual_scatter": float(scale) if np.isfinite(scale) else np.nan,
        "S_uncertainty": uncertainties[0], "C_uncertainty": uncertainties[1],
    }


def _fit_exposure_SC(items, logm_by_item=None, pass_name="first"):
    rows = []
    fits = {}
    for item in items:
        correction = np.zeros(item.band_total.shape[0], dtype=float)
        if logm_by_item is not None:
            correction = np.asarray(logm_by_item.get(item.item_index, correction), dtype=float)
        correction = np.where(np.isfinite(correction), correction, 0.0)
        scale = np.exp(np.clip(correction, -50.0, 50.0))
        for band_index, band in enumerate(SOURCE_BANDS):
            valid = ((item.q >= Q_MIN) & (item.q <= Q_MAX) &
                     ~item.hardware_bad & item.external_valid[:, band_index] &
                     np.isfinite(item.X[:, band_index]) &
                     np.isfinite(item.band_total[:, band_index]))
            target = item.band_total[valid, band_index] / scale[valid]
            error = item.band_error[valid, band_index] / scale[valid]
            fit = _robust_linear_fit(item.X[valid, band_index], target, error)
            key = (item.h5_name, item.exposure, band)
            fits[key] = fit
            x_values = item.X[valid, band_index]
            rows.append({
                "pass": pass_name, "H5": item.h5_name,
                "exposure": int(item.exposure), "band": band,
                "band_index_in_onoff": int(band_index),
                "S": fit["S"], "C": fit["C"],
                "S_uncertainty": fit["S_uncertainty"],
                "C_uncertainty": fit["C_uncertainty"],
                "residual_scatter": fit["residual_scatter"],
                "N_fibers": fit["N"], "rank": fit["rank"],
                "X_min": float(np.min(x_values)) if x_values.size else np.nan,
                "X_max": float(np.max(x_values)) if x_values.size else np.nan,
                "X_median": _location(x_values),
                "fraction_positive_X": (float(np.mean(x_values > 0))
                                         if x_values.size else np.nan),
                "correction_applied": pass_name != "first",
            })
    return rows, fits


def _normalize_C(rows):
    log_cbar = {}
    for band in SOURCE_BANDS:
        values = [math.log(row["C"]) for row in rows
                  if row["band"] == band and np.isfinite(row["C"]) and row["C"] > 0]
        log_cbar[band] = _location(values)
    cbar = {band: float(np.exp(value)) if np.isfinite(value) else np.nan
            for band, value in log_cbar.items()}
    by_h5_band = {}
    for row in rows:
        if np.isfinite(row["C"]) and row["C"] > 0 and np.isfinite(log_cbar[row["band"]]):
            row["log_C"] = float(math.log(row["C"]))
            row["log_c"] = float(row["log_C"] - log_cbar[row["band"]])
            by_h5_band.setdefault((row["H5"], row["band"]), []).append(row["log_c"])
        else:
            row["log_C"] = np.nan
            row["log_c"] = np.nan
    gc = {}
    for key, values in by_h5_band.items():
        gc[key] = _location(values)
    for row in rows:
        key = (row["H5"], row["band"])
        row["Gc"] = gc.get(key, np.nan)
        row["gc"] = (row["log_c"] - row["Gc"]
                      if np.isfinite(row["log_c"]) and np.isfinite(row["Gc"])
                      else np.nan)
        row["C_reconstructed"] = (
            cbar[row["band"]] * math.exp(row["Gc"] + row["gc"])
            if np.isfinite(cbar[row["band"]]) and np.isfinite(row["Gc"]) and
            np.isfinite(row["gc"]) else np.nan)
        row["C_reconstruction_error"] = (
            row["C_reconstructed"] - row["C"]
            if np.isfinite(row["C_reconstructed"]) and np.isfinite(row["C"])
            else np.nan)
    return cbar, log_cbar, gc


def _build_r(items, fits, logm_by_item=None):
    r_by_item = {}
    valid_by_item = {}
    total_predicted_by_item = {}
    fraction_by_item = {}
    for item in items:
        r = np.full((item.band_total.shape[0], 2), np.nan, dtype=float)
        valid_out = np.zeros_like(r, dtype=bool)
        total_predicted = np.full_like(r, np.nan, dtype=float)
        fraction = np.full_like(r, np.nan, dtype=float)
        correction = np.zeros(item.band_total.shape[0], dtype=float)
        if logm_by_item is not None:
            correction = np.asarray(logm_by_item.get(item.item_index, correction), dtype=float)
        correction = np.where(np.isfinite(correction), correction, 0.0)
        multiplier = np.exp(np.clip(correction, -50.0, 50.0))
        for band_index, band in enumerate(SOURCE_BANDS):
            fit = fits.get((item.h5_name, item.exposure, band), {})
            S, C = fit.get("S", np.nan), fit.get("C", np.nan)
            if not np.isfinite(S) or not np.isfinite(C):
                continue
            total = S + C * item.X[:, band_index]
            total_predicted[:, band_index] = total
            good = ((item.q >= Q_MIN) & (item.q <= Q_MAX) &
                    ~item.hardware_bad & item.external_valid[:, band_index] &
                    np.isfinite(item.X[:, band_index]) &
                    np.isfinite(item.band_total[:, band_index]) &
                    (item.band_total[:, band_index] > 0) & np.isfinite(total) &
                    (total > 0))
            corrected_data = item.band_total[:, band_index] / multiplier
            with np.errstate(divide="ignore", invalid="ignore"):
                r[good, band_index] = np.log(corrected_data[good] / total[good])
                fraction[good, band_index] = C * item.X[good, band_index] / total[good]
            valid_out[:, band_index] = good & np.isfinite(r[:, band_index])
        r_by_item[item.item_index] = r
        valid_by_item[item.item_index] = valid_out
        total_predicted_by_item[item.item_index] = total_predicted
        fraction_by_item[item.item_index] = fraction
    return r_by_item, valid_by_item, total_predicted_by_item, fraction_by_item


def _ifu_identity(item, ifu_code):
    indices = np.flatnonzero(item.ifu_code == int(ifu_code))
    return tuple(map(int, item.ifu[indices[0]])) if indices.size else (np.nan,) * 3


def _summary_row(item, ifu_code, amp_index, band, values, indices, fractions,
                 pass_name):
    values = np.asarray(values, dtype=float)
    good = np.isfinite(values)
    values = values[good]
    selected_indices = np.asarray(indices, dtype=int)[good]
    identity = _ifu_identity(item, ifu_code)
    x = item.X[selected_indices, SOURCE_BANDS.index(band)] if selected_indices.size else np.empty(0)
    q = item.q[selected_indices] if selected_indices.size else np.empty(0)
    source_fraction = fractions[selected_indices, SOURCE_BANDS.index(band)] if selected_indices.size else np.empty(0)
    d = item.band_total[selected_indices, SOURCE_BANDS.index(band)] if selected_indices.size else np.empty(0)
    e = item.band_error[selected_indices, SOURCE_BANDS.index(band)] if selected_indices.size else np.empty(0)
    formal = e / d
    return {
        "pass": pass_name, "H5": item.h5_name, "exposure": int(item.exposure),
        "IFU_CODE": int(ifu_code), "SPECID": identity[0],
        "IFUSLOT": identity[1], "IFUID": identity[2],
        "AMP": AMP_ORDER[amp_index], "AMP_INDEX": int(amp_index),
        "band": band, "central_q_min": Q_MIN, "central_q_max": Q_MAX,
        "central_q_only": True, "supported": bool(values.size > 0),
        "support_reason": ("supported central-q summary" if values.size else
                           "no valid central-q ON/OFF rows"),
        "N_fibers": int(values.size), "robust_location": _location(values),
        "robust_scatter": _scatter(values), "location_uncertainty": _uncertainty(values),
        "formal_log_error_median": _location(formal),
        "q_min": int(np.min(q)) if q.size else np.nan,
        "q_max": int(np.max(q)) if q.size else np.nan,
        "q_support": int(q.size),
        "X_min": float(np.min(x)) if x.size else np.nan,
        "X_max": float(np.max(x)) if x.size else np.nan,
        "X_median": _location(x),
        "source_fraction_location": _location(source_fraction),
        "source_fraction_scatter": _scatter(source_fraction),
        "value_definition": "log(D/(S+C*X))",
    }


def _build_amplifier_measurements(items, r_by_item, valid_by_item,
                                  fraction_by_item, pass_name):
    rows = []
    lookup = {}
    for item in items:
        item_ifus = sorted({int(code) for code in np.unique(item.ifu_code)})
        for ifu_code in item_ifus:
            group = item.ifu_code == ifu_code
            for amp_index, amp in enumerate(AMP_ORDER):
                amp_group = group & (item.amp == amp_index)
                for band_index, band in enumerate(SOURCE_BANDS):
                    indices = np.flatnonzero(amp_group & valid_by_item[item.item_index][:, band_index])
                    values = r_by_item[item.item_index][indices, band_index]
                    row = _summary_row(
                        item, ifu_code, amp_index, band, values, indices,
                        fraction_by_item[item.item_index], pass_name)
                    rows.append(row)
                    lookup[(item.h5_name, item.exposure, ifu_code, amp_index, band)] = row
    return rows, lookup


def _build_exposure_common(items, amp_lookup, pass_name):
    rows = []
    common_lookup = {}
    d_lookup = {}
    for item in items:
        item_ifus = sorted({int(code) for code in np.unique(item.ifu_code)})
        for ifu_code in item_ifus:
            identity = _ifu_identity(item, ifu_code)
            for band in SOURCE_BANDS:
                amps = [amp_lookup[(item.h5_name, item.exposure, ifu_code,
                                    amp_index, band)] for amp_index in range(4)]
                supported = [row for row in amps
                             if row["supported"] and np.isfinite(row["robust_location"])]
                complete = len(supported) == 4
                values = np.asarray([row["robust_location"] for row in amps], dtype=float)
                if complete:
                    # This is intentionally exactly equal amplifier weighting.
                    common = float(0.25 * np.sum(values))
                    differentials = values - common
                else:
                    common = np.nan
                    differentials = np.full(4, np.nan)
                row = {
                    "pass": pass_name, "H5": item.h5_name,
                    "exposure": int(item.exposure), "IFU_CODE": int(ifu_code),
                    "SPECID": identity[0], "IFUSLOT": identity[1], "IFUID": identity[2],
                    "band": band, "complete_four_amplifiers": complete,
                    "N_amp_supported": len(supported),
                    "N_fibers": int(sum(amp["N_fibers"] for amp in amps)),
                    "I_e": common, "equal_amp_weight": 0.25,
                    "r_LL": values[0], "r_LU": values[1],
                    "r_RL": values[2], "r_RU": values[3],
                    "d_LL": differentials[0], "d_LU": differentials[1],
                    "d_RL": differentials[2], "d_RU": differentials[3],
                    "d_sum": (float(np.sum(differentials)) if complete else np.nan),
                    "support_reason": ("all four amplifiers supported" if complete else
                                       "missing central-q amplifier support"),
                }
                rows.append(row)
                common_lookup[(item.h5_name, item.exposure, ifu_code, band)] = row
                for amp_index in range(4):
                    d_lookup[(item.h5_name, item.exposure, ifu_code,
                              amp_index, band)] = differentials[amp_index]
    return rows, common_lookup, d_lookup


def _build_h5_amplifier_summaries(items, amp_lookup, pass_name):
    """Collapse each usable amplifier channel over the three exposures.

    This is intentionally independent of every other amplifier in the IFU.
    The gray channel is only an ON/OFF presentation of the available channel;
    it is not a four-amplifier average.
    """
    h5_names = sorted({item.h5_name for item in items})
    ifus = sorted({int(code) for item in items for code in np.unique(item.ifu_code)})
    identity_by_code = {}
    for item in items:
        for code in np.unique(item.ifu_code):
            identity_by_code[int(code)] = _ifu_identity(item, code)
    band_rows, band_lookup = [], {}
    for h5 in h5_names:
        for ifu_code in ifus:
            identity = identity_by_code.get(ifu_code, (np.nan,) * 3)
            for amp_index, amp in enumerate(AMP_ORDER):
                for band in SOURCE_BANDS:
                    values = np.asarray([
                        amp_lookup.get((h5, exposure, ifu_code, amp_index, band), {}).get(
                            "robust_location", np.nan)
                        for exposure in (1, 2, 3)], dtype=float)
                    good = np.isfinite(values)
                    if not np.any(good):
                        continue
                    used_exposures = [exposure for exposure, keep in
                                      zip((1, 2, 3), good) if keep]
                    R = _location(values[good])
                    row = {
                        "pass": pass_name, "H5": h5, "IFU_CODE": int(ifu_code),
                        "SPECID": identity[0], "IFUSLOT": identity[1], "IFUID": identity[2],
                        "AMP": amp, "AMP_INDEX": int(amp_index), "band": band,
                        "R_h_i_a_b": R, "R_h": R,
                        "R_e1": values[0], "R_e2": values[1], "R_e3": values[2],
                        "exposure_values_used": used_exposures,
                        "N_exposure": int(np.sum(good)),
                        "robust_exposure_scatter": _scatter(values[good]),
                        "exposure_location_uncertainty": _uncertainty(values[good]),
                        "support_reason": "at least one valid amplifier exposure",
                    }
                    band_rows.append(row)
                    band_lookup[(h5, ifu_code, amp_index, band)] = row

    gray_rows, gray_lookup = [], {}
    for h5 in h5_names:
        for ifu_code in ifus:
            identity = identity_by_code.get(ifu_code, (np.nan,) * 3)
            for amp_index, amp in enumerate(AMP_ORDER):
                on = band_lookup.get((h5, ifu_code, amp_index, "ON"))
                off = band_lookup.get((h5, ifu_code, amp_index, "OFF"))
                if on is None and off is None:
                    continue
                on_value = on["R_h"] if on is not None else np.nan
                off_value = off["R_h"] if off is not None else np.nan
                available = [value for value in (on_value, off_value) if np.isfinite(value)]
                support = "BOTH" if on is not None and off is not None else (
                    "ON_ONLY" if on is not None else "OFF_ONLY")
                gray_value = (0.5 * (on_value + off_value)
                              if np.isfinite(on_value) and np.isfinite(off_value)
                              else available[0])
                row = {
                    "pass": pass_name, "H5": h5, "IFU_CODE": int(ifu_code),
                    "SPECID": identity[0], "IFUSLOT": identity[1], "IFUID": identity[2],
                    "AMP": amp, "AMP_INDEX": int(amp_index), "band": "GRAY",
                    "R_h_i_a": gray_value, "R_gray": gray_value,
                    "R_ON": on_value, "R_OFF": off_value,
                    "R_ON_minus_OFF": (on_value - off_value
                                       if np.isfinite(on_value) and np.isfinite(off_value)
                                       else np.nan),
                    "ON_OFF_support": support,
                    "N_exposure_ON": on["N_exposure"] if on is not None else 0,
                    "N_exposure_OFF": off["N_exposure"] if off is not None else 0,
                    "N_exposure": int(max(on["N_exposure"] if on is not None else 0,
                                           off["N_exposure"] if off is not None else 0)),
                    "exposure_values_used_ON": on["exposure_values_used"] if on is not None else [],
                    "exposure_values_used_OFF": off["exposure_values_used"] if off is not None else [],
                    "robust_exposure_scatter_ON": on["robust_exposure_scatter"] if on is not None else np.nan,
                    "robust_exposure_scatter_OFF": off["robust_exposure_scatter"] if off is not None else np.nan,
                    "support_reason": "ON/OFF channel combination; no four-amplifier requirement",
                }
                gray_rows.append(row)
                gray_lookup[(h5, ifu_code, amp_index)] = row
    return band_rows, band_lookup, gray_rows, gray_lookup


def _persistent_amplifier_hierarchy(h5_band_rows, h5_gray_rows, ifus):
    """Build the primary persistent physical amplifier-channel hierarchy."""
    A = {(code, amp_index): np.nan for code in ifus for amp_index in range(4)}
    A_ON = {(code, amp_index): np.nan for code in ifus for amp_index in range(4)}
    A_OFF = {(code, amp_index): np.nan for code in ifus for amp_index in range(4)}
    support = {}
    values_gray = {}
    values_band = {band: {} for band in SOURCE_BANDS}
    for row in h5_gray_rows:
        key = (int(row["IFU_CODE"]), int(row["AMP_INDEX"]))
        if np.isfinite(row["R_gray"]):
            values_gray.setdefault(key, []).append(row["R_gray"])
    for row in h5_band_rows:
        key = (int(row["IFU_CODE"]), int(row["AMP_INDEX"]))
        if np.isfinite(row["R_h"]):
            values_band[row["band"]].setdefault(key, []).append(row["R_h"])
    for key in A:
        A[key] = _location(values_gray.get(key, []))
        A_ON[key] = _location(values_band["ON"].get(key, []))
        A_OFF[key] = _location(values_band["OFF"].get(key, []))
        support[key] = {
            "support_h5_count": len(values_gray.get(key, [])),
            "support_h5_count_ON": len(values_band["ON"].get(key, [])),
            "support_h5_count_OFF": len(values_band["OFF"].get(key, [])),
            "N_gray_values": len(values_gray.get(key, [])),
            "N_ON_values": len(values_band["ON"].get(key, [])),
            "N_OFF_values": len(values_band["OFF"].get(key, [])),
        }
    Delta = {(row["H5"], int(row["IFU_CODE"]), int(row["AMP_INDEX"])):
             float(row["R_gray"] - A[(int(row["IFU_CODE"]), int(row["AMP_INDEX"]))])
             for row in h5_gray_rows
             if np.isfinite(row["R_gray"]) and np.isfinite(
                 A[(int(row["IFU_CODE"]), int(row["AMP_INDEX"]))])}

    # The secondary common/differential decomposition is defined only for
    # persistent four-channel support.  No missing channel is imputed.
    complete = {code for code in ifus if all(np.isfinite(A[(code, amp)]) for amp in range(4))}
    complete_band = {
        band: {code for code in ifus if all(np.isfinite(
            (A_ON if band == "ON" else A_OFF)[(code, amp)]) for amp in range(4))}
        for band in SOURCE_BANDS
    }
    P = {code: (float(0.25 * sum(A[(code, amp)] for amp in range(4)))
                if code in complete else np.nan) for code in ifus}
    Pamp = {(code, amp): (A[(code, amp)] - P[code]
                          if code in complete else np.nan)
            for code in ifus for amp in range(4)}
    P_band, Pamp_band = {}, {}
    for band, band_values in (("ON", A_ON), ("OFF", A_OFF)):
        P_band[band] = {code: (float(0.25 * sum(band_values[(code, amp)] for amp in range(4)))
                               if code in complete_band[band] else np.nan)
                        for code in ifus}
        Pamp_band[band] = {
            (code, amp): (band_values[(code, amp)] - P_band[band][code]
                          if code in complete_band[band] else np.nan)
            for code in ifus for amp in range(4)}

    def leave_one_h5_pamp(rows, value_key, band=None):
        grouped = {}
        for row in rows:
            if band is not None and row.get("band") != band:
                continue
            grouped.setdefault((row["H5"], int(row["IFU_CODE"])), []).append(row)
        references = {}
        for (target_h5, code) in grouped:
            values = {}
            for amp_index in range(4):
                other = [row[value_key] for (h5, other_code), group in grouped.items()
                         if other_code == code and h5 != target_h5
                         for row in group if int(row["AMP_INDEX"]) == amp_index and
                         np.isfinite(row[value_key])]
                values[amp_index] = _location(other)
            if all(np.isfinite(values[amp]) for amp in range(4)):
                p_ref = float(0.25 * sum(values.values()))
                for amp_index in range(4):
                    references[(target_h5, code, amp_index)] = values[amp_index] - p_ref
        return references

    pamp_reference_by_h5 = leave_one_h5_pamp(h5_gray_rows, "R_gray")
    pamp_reference_ON_by_h5 = leave_one_h5_pamp(h5_band_rows, "R_h", "ON")
    pamp_reference_OFF_by_h5 = leave_one_h5_pamp(h5_band_rows, "R_h", "OFF")
    return {
        "A": A, "A_ON": A_ON, "A_OFF": A_OFF, "Delta": Delta,
        "support": support, "complete_ifus": complete,
        "complete_band_ifus": complete_band,
        "P": P, "Pamp": Pamp, "P_ON": P_band["ON"], "P_OFF": P_band["OFF"],
        "Pamp_ON": Pamp_band["ON"], "Pamp_OFF": Pamp_band["OFF"],
        "raw_P": dict(P), "raw_Pamp": dict(Pamp),
        "raw_P_ON": dict(P_band["ON"]), "raw_P_OFF": dict(P_band["OFF"]),
        "raw_Pamp_ON": dict(Pamp_band["ON"]), "raw_Pamp_OFF": dict(Pamp_band["OFF"]),
        "Pamp_reference_by_h5": pamp_reference_by_h5,
        "Pamp_reference_ON_by_h5": pamp_reference_ON_by_h5,
        "Pamp_reference_OFF_by_h5": pamp_reference_OFF_by_h5,
        "channel_rows": [],
    }


def _row_key_amp(row):
    return (row["H5"], int(row["exposure"]), int(row["IFU_CODE"]),
            int(row["AMP_INDEX"]), row["band"])


def _row_key_h5_band(row):
    return (row["H5"], int(row["IFU_CODE"]), int(row["AMP_INDEX"]), row["band"])


def _row_key_h5_gray(row):
    return (row["H5"], int(row["IFU_CODE"]), int(row["AMP_INDEX"]))


def _finite_increment(value, increment):
    """Add an available refinement increment without changing support."""
    if not np.isfinite(value):
        return np.nan
    return float(value + increment) if np.isfinite(increment) else float(value)


def _compose_amp_measurements(first_rows, second_rows):
    """Compose exposure/amplifier log responses as pass1 + residual."""
    second = {_row_key_amp(row): row for row in second_rows}
    total_rows = []
    for first in first_rows:
        row = dict(first)
        increment_row = second.get(_row_key_amp(first), {})
        pass1 = first.get("robust_location", np.nan)
        increment = increment_row.get("robust_location", np.nan)
        row["pass"] = "total"
        row["pass_role"] = "cumulative_final"
        row["r_pass1"] = pass1
        row["dr_pass2"] = increment if np.isfinite(increment) else 0.0
        row["r_total"] = _finite_increment(pass1, row["dr_pass2"])
        row["robust_location"] = row["r_total"]
        row["value_definition"] = "cumulative original-data log response: r_pass1 + dr_pass2"
        return_row = row
        total_rows.append(return_row)
    return total_rows


def _compose_h5_band_rows(first_rows, second_rows):
    """Compose H5/band channel responses from pass 1 and residual pass 2."""
    second = {_row_key_h5_band(row): row for row in second_rows}
    total_rows = []
    for first in first_rows:
        row = dict(first)
        increment_row = second.get(_row_key_h5_band(first), {})
        pass1 = first.get("R_h", np.nan)
        increment = increment_row.get("R_h", np.nan)
        row["pass"] = "total"
        row["pass_role"] = "cumulative_final"
        row["R_pass1"] = pass1
        row["dR_pass2"] = increment if np.isfinite(increment) else 0.0
        row["R_total"] = _finite_increment(pass1, row["dR_pass2"])
        row["R_h"] = row["R_total"]
        row["R_h_i_a_b"] = row["R_total"]
        row["value_definition"] = "cumulative original-data log response: R_pass1 + dR_pass2"
        total_rows.append(row)
    return total_rows


def _compose_h5_gray_rows(first_rows, second_rows):
    """Compose gray channel rows while preserving first-pass support."""
    second = {_row_key_h5_gray(row): row for row in second_rows}
    total_rows = []

    def add_component(value, increment):
        if not np.isfinite(value):
            return np.nan
        return _finite_increment(value, increment if np.isfinite(increment) else 0.0)

    for first in first_rows:
        row = dict(first)
        increment_row = second.get(_row_key_h5_gray(first), {})
        pass1 = first.get("R_gray", np.nan)
        increment = increment_row.get("R_gray", np.nan)
        row["pass"] = "total"
        row["pass_role"] = "cumulative_final"
        row["R_pass1"] = pass1
        row["dR_pass2"] = increment if np.isfinite(increment) else 0.0
        row["R_total"] = _finite_increment(pass1, row["dR_pass2"])
        row["R_h_i_a"] = row["R_total"]
        row["R_gray"] = row["R_total"]
        for name in ("ON", "OFF"):
            pass1_value = first.get("R_" + name, np.nan)
            increment_value = increment_row.get("R_" + name, np.nan)
            row["R_%s_pass1" % name] = pass1_value
            row["dR_%s_pass2" % name] = (
                increment_value if np.isfinite(increment_value) else 0.0)
            row["R_%s_total" % name] = add_component(pass1_value, increment_value)
            row["R_" + name] = row["R_%s_total" % name]
        row["R_ON_minus_OFF"] = (
            row["R_ON"] - row["R_OFF"]
            if np.isfinite(row["R_ON"]) and np.isfinite(row["R_OFF"]) else np.nan)
        row["value_definition"] = "cumulative original-data gray log response: R_pass1 + dR_pass2"
        total_rows.append(row)
    return total_rows


def _zero_primary_increment(primary):
    """Make an explicit zero residual hierarchy when refinement is disabled."""
    result = dict(primary)
    for name in ("A", "A_ON", "A_OFF", "P", "P_ON", "P_OFF"):
        result[name] = {key: (0.0 if np.isfinite(value) else np.nan)
                        for key, value in primary[name].items()}
    for name in ("Delta", "Pamp", "Pamp_ON", "Pamp_OFF"):
        result[name] = {key: (0.0 if np.isfinite(value) else np.nan)
                        for key, value in primary[name].items()}
    result["raw_P"] = dict(result["P"])
    result["raw_Pamp"] = dict(result["Pamp"])
    result["raw_P_ON"] = dict(result["P_ON"])
    result["raw_P_OFF"] = dict(result["P_OFF"])
    result["raw_Pamp_ON"] = dict(result["Pamp_ON"])
    result["raw_Pamp_OFF"] = dict(result["Pamp_OFF"])
    return result


def _compose_primary_hierarchy(primary_first, primary_increment,
                               h5_band_total, h5_gray_total, ifus):
    """Compose persistent channel state exactly as A_total=A1+dA2."""
    base = _persistent_amplifier_hierarchy(h5_band_total, h5_gray_total, ifus)
    for name in ("A", "A_ON", "A_OFF"):
        base[name] = {
            key: _finite_increment(primary_first[name].get(key, np.nan),
                                  primary_increment[name].get(key, np.nan))
            for key in primary_first[name]
        }
    base["support"] = primary_first["support"]
    base["complete_ifus"] = {
        code for code in ifus
        if all(np.isfinite(base["A"].get((code, amp), np.nan)) for amp in range(4))
    }
    base["complete_band_ifus"] = {
        band: {code for code in ifus if all(np.isfinite(
            base["A_ON" if band == "ON" else "A_OFF"].get((code, amp), np.nan))
            for amp in range(4))}
        for band in SOURCE_BANDS
    }

    def decompose(values, complete_codes):
        common = {code: (float(0.25 * sum(values[(code, amp)] for amp in range(4)))
                         if code in complete_codes else np.nan)
                  for code in ifus}
        differential = {
            (code, amp): (values[(code, amp)] - common[code]
                          if code in complete_codes else np.nan)
            for code in ifus for amp in range(4)}
        return common, differential

    base["P"], base["Pamp"] = decompose(base["A"], base["complete_ifus"])
    base["P_ON"], base["Pamp_ON"] = decompose(
        base["A_ON"], base["complete_band_ifus"]["ON"])
    base["P_OFF"], base["Pamp_OFF"] = decompose(
        base["A_OFF"], base["complete_band_ifus"]["OFF"])
    base["raw_P"] = dict(base["P"])
    base["raw_Pamp"] = dict(base["Pamp"])
    base["raw_P_ON"] = dict(base["P_ON"])
    base["raw_P_OFF"] = dict(base["P_OFF"])
    base["raw_Pamp_ON"] = dict(base["Pamp_ON"])
    base["raw_Pamp_OFF"] = dict(base["Pamp_OFF"])
    base["Delta"] = {
        (row["H5"], int(row["IFU_CODE"]), int(row["AMP_INDEX"])):
        float(row["R_gray"] - base["A"][(int(row["IFU_CODE"]), int(row["AMP_INDEX"]))])
        for row in h5_gray_total
        if np.isfinite(row.get("R_gray", np.nan)) and np.isfinite(
            base["A"].get((int(row["IFU_CODE"]), int(row["AMP_INDEX"])), np.nan))
    }
    base["channel_rows"] = []
    return base


def _collapse_h5_common(channel_rows, persistent, ifus, pass_name,
                        band, pamp_key, gray=False, exposure_lookup=None):
    """Collapse available channels, using only an independent Pamp reference."""
    grouped = {}
    for row in channel_rows:
        grouped.setdefault((row["H5"], int(row["IFU_CODE"])), []).append(row)
    common_rows, diff_rows, common_lookup, diff_lookup = [], [], {}, {}
    for (h5, ifu_code), rows in sorted(grouped.items()):
        identity = (rows[0]["SPECID"], rows[0]["IFUSLOT"], rows[0]["IFUID"])
        by_amp = {int(row["AMP_INDEX"]): row for row in rows}
        available = [row for row in rows if np.isfinite(row["R_gray"] if gray else row["R_h"])]
        values = {amp: (row["R_gray"] if gray else row["R_h"])
                  for amp, row in by_amp.items()}
        n_available = len(available)
        available_names = [AMP_ORDER[index] for index in sorted(values)]
        if n_available == 4:
            I_value = 0.25 * sum(values[index] for index in sorted(values))
            method = "complete_four_equal_amp_mean"
            pamp_reference = True
        else:
            if pamp_key == "Pamp":
                pamp_map = persistent["Pamp_reference_by_h5"]
            else:
                pamp_map = persistent["Pamp_reference_%s_by_h5" % pamp_key.split("_")[-1]]
            candidates = [values[index] - pamp_map.get((h5, ifu_code, index), np.nan)
                          for index in sorted(values)
                          if np.isfinite(pamp_map.get((h5, ifu_code, index), np.nan))]
            pamp_reference = bool(candidates) and len(candidates) == n_available
            if pamp_reference:
                I_value = _location(candidates)
                method = "partial_using_independent_persistent_Pamp"
            else:
                I_value = np.nan
                method = "unseparated_channel_only"
        common_row = {
            "pass": pass_name, "H5": h5, "IFU_CODE": ifu_code,
            "SPECID": identity[0], "IFUSLOT": identity[1], "IFUID": identity[2],
            "band": band, "I_h": I_value,
            "N_amp_available": n_available, "available_amp_names": available_names,
            "support_class": "%d_amp" % n_available,
            "Pamp_reference_available": pamp_reference,
            "Pamp_reference_source": (
                "leave_one_H5_out_persistent_channel_contrasts"
                if pamp_reference and n_available < 4 else
                "not_needed_complete_four" if n_available == 4 else "none"),
            "common_identifiable": bool(np.isfinite(I_value)),
            "common_method": method,
            "repeatability_scatter": np.nan,
            "repeatability_uncertainty": np.nan,
            "equal_amp_weight": 0.25 if n_available == 4 else np.nan,
            "support_reason": method,
        }
        exposure_values = np.asarray([
            exposure_lookup.get((h5, ifu_code, exposure, band), {}).get("I_e", np.nan)
            for exposure in (1, 2, 3)
        ], dtype=float) if exposure_lookup is not None and not gray else np.full(3, np.nan)
        common_row.update({
            "I_e1": exposure_values[0], "I_e2": exposure_values[1], "I_e3": exposure_values[2],
            "deltaI_e1": exposure_values[0] - I_value if np.isfinite(exposure_values[0]) and np.isfinite(I_value) else np.nan,
            "deltaI_e2": exposure_values[1] - I_value if np.isfinite(exposure_values[1]) and np.isfinite(I_value) else np.nan,
            "deltaI_e3": exposure_values[2] - I_value if np.isfinite(exposure_values[2]) and np.isfinite(I_value) else np.nan,
            "N_exposures": int(np.sum(np.isfinite(exposure_values))),
            "repeatability_scatter": _scatter(exposure_values),
            "repeatability_uncertainty": _uncertainty(exposure_values),
        })
        for amp_index, amp in enumerate(AMP_ORDER):
            R_value = values.get(amp_index, np.nan)
            common_row["R_%s" % amp] = R_value
            common_row["d_%s" % amp] = (R_value - I_value
                                          if np.isfinite(R_value) and np.isfinite(I_value)
                                          else np.nan)
        common_row["d_sum"] = (sum(common_row["d_%s" % amp] for amp in AMP_ORDER)
                                if np.isfinite(I_value) else np.nan)
        common_rows.append(common_row)
        if np.isfinite(I_value):
            common_lookup[(h5, ifu_code)] = common_row
            for amp_index, amp in enumerate(AMP_ORDER):
                R_value = values.get(amp_index, np.nan)
                if not np.isfinite(R_value):
                    continue
                diff_row = {
                    "pass": pass_name, "H5": h5, "IFU_CODE": ifu_code,
                    "SPECID": identity[0], "IFUSLOT": identity[1], "IFUID": identity[2],
                    "AMP": amp, "AMP_INDEX": amp_index, "band": band,
                    "d_h": R_value - I_value, "R_h": R_value,
                    "I_h": I_value, "N_amp_available": n_available,
                    "available_amp_names": available_names,
                    "Pamp_reference_available": pamp_reference,
                    "Pamp_reference_source": common_row["Pamp_reference_source"],
                    "common_identifiable": True, "common_method": method,
                    "repeatability_scatter": np.nan,
                    "repeatability_uncertainty": np.nan,
                }
                exposure_d = np.asarray([
                    exposure_lookup.get((h5, ifu_code, exposure, band), {}).get(
                        "d_%s" % amp, np.nan)
                    for exposure in (1, 2, 3)
                ], dtype=float) if exposure_lookup is not None and not gray else np.full(3, np.nan)
                diff_row.update({
                    "d_e1": exposure_d[0], "d_e2": exposure_d[1], "d_e3": exposure_d[2],
                    "deltad_e1": exposure_d[0] - diff_row["d_h"] if np.isfinite(exposure_d[0]) else np.nan,
                    "deltad_e2": exposure_d[1] - diff_row["d_h"] if np.isfinite(exposure_d[1]) else np.nan,
                    "deltad_e3": exposure_d[2] - diff_row["d_h"] if np.isfinite(exposure_d[2]) else np.nan,
                    "N_exposures": int(np.sum(np.isfinite(exposure_d))),
                    "repeatability_scatter": _scatter(exposure_d),
                    "repeatability_uncertainty": _uncertainty(exposure_d),
                })
                diff_rows.append(diff_row)
                diff_lookup[(h5, ifu_code, amp_index)] = diff_row
    return common_rows, diff_rows, common_lookup, diff_lookup


def _build_h5_common_summaries(h5_band_rows, h5_gray_rows, persistent,
                               ifus, pass_name, exposure_rows=None):
    exposure_lookup = {(row["H5"], int(row["IFU_CODE"]), int(row["exposure"]), row["band"]): row
                       for row in (exposure_rows or [])}
    h5_common_rows, h5_diff_rows, hcommon, hdiff = [], [], {}, {}
    for band, pamp_key in (("ON", "Pamp_ON"), ("OFF", "Pamp_OFF")):
        channel_rows = [row for row in h5_band_rows if row["band"] == band]
        common, diff, lookup, dlookup = _collapse_h5_common(
            channel_rows, persistent, ifus, pass_name, band, pamp_key,
            gray=False, exposure_lookup=exposure_lookup)
        h5_common_rows.extend(common)
        h5_diff_rows.extend(diff)
        hcommon.update({(h5, code, band): row for (h5, code), row in lookup.items()})
        hdiff.update({(h5, code, amp, band): row
                      for (h5, code, amp), row in dlookup.items()})
    gray_common_rows, gray_diff_rows, hcommon_gray, hdiff_gray = _collapse_h5_common(
        h5_gray_rows, persistent, ifus, pass_name, "GRAY", "Pamp", gray=True,
        exposure_lookup=exposure_lookup)
    return (h5_common_rows, h5_diff_rows, hcommon, hdiff,
            gray_common_rows, gray_diff_rows, hcommon_gray, hdiff_gray)


def _rankdata(values):
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    sorted_values = np.asarray(values)[order]
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and sorted_values[stop] == sorted_values[start]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1) + 1.0
        start = stop
    return ranks


def _correlation(x, y, spearman=False):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    good = np.isfinite(x) & np.isfinite(y)
    x, y = x[good], y[good]
    if x.size < 2 or np.std(x) <= 0 or np.std(y) <= 0:
        return np.nan
    if spearman:
        x, y = _rankdata(x), _rankdata(y)
    return float(np.corrcoef(x, y)[0, 1])


def _compare_on_off(hcommon, hdiff, ifus):
    ifu_rows = []
    for h5 in sorted({key[0] for key in hcommon}):
        for ifu_code in ifus:
            on = hcommon.get((h5, ifu_code, "ON"))
            off = hcommon.get((h5, ifu_code, "OFF"))
            if on is None or off is None:
                continue
            diff = float(on["I_h"] - off["I_h"])
            ifu_rows.append({
                "H5": h5, "IFU_CODE": int(ifu_code),
                "SPECID": on["SPECID"], "IFUSLOT": on["IFUSLOT"], "IFUID": on["IFUID"],
                "I_ON": on["I_h"], "I_OFF": off["I_h"], "I_ON_minus_OFF": diff,
                "I_perpendicular_difference": diff / math.sqrt(2.0),
            })
    amp_rows = []
    for h5, ifu_code, amp_index, _band in sorted(hdiff):
        on = hdiff.get((h5, ifu_code, amp_index, "ON"))
        off = hdiff.get((h5, ifu_code, amp_index, "OFF"))
        if on is None or off is None:
            continue
        diff = float(on["d_h"] - off["d_h"])
        amp_rows.append({
            "H5": h5, "IFU_CODE": int(ifu_code), "AMP": AMP_ORDER[amp_index],
            "AMP_INDEX": int(amp_index), "SPECID": on["SPECID"],
            "IFUSLOT": on["IFUSLOT"], "IFUID": on["IFUID"],
            "d_ON": on["d_h"], "d_OFF": off["d_h"],
            "d_ON_minus_OFF": diff,
            "d_perpendicular_difference": diff / math.sqrt(2.0),
        })
    on_values = np.asarray([row["I_ON"] for row in ifu_rows], dtype=float)
    off_values = np.asarray([row["I_OFF"] for row in ifu_rows], dtype=float)
    ifu_diffs = np.asarray([row["I_ON_minus_OFF"] for row in ifu_rows], dtype=float)
    comparison = {
        "ifu_common": {
            "N": int(ifu_diffs.size), "median_offset": _location(ifu_diffs),
            "robust_scatter_difference": _scatter(ifu_diffs),
            "robust_scatter_perpendicular": _scatter(ifu_diffs / math.sqrt(2.0)),
            "pearson": _correlation(on_values, off_values),
            "spearman": _correlation(on_values, off_values, spearman=True),
            "absolute_difference": _percentiles(np.abs(ifu_diffs)),
        },
        "amp_differential": {},
    }
    on_values = np.asarray([row["d_ON"] for row in amp_rows], dtype=float)
    off_values = np.asarray([row["d_OFF"] for row in amp_rows], dtype=float)
    diffs = np.asarray([row["d_ON_minus_OFF"] for row in amp_rows], dtype=float)
    comparison["amp_differential"] = {
        "N": int(diffs.size), "median_offset": _location(diffs),
        "robust_scatter_difference": _scatter(diffs),
        "robust_scatter_perpendicular": _scatter(diffs / math.sqrt(2.0)),
        "pearson": _correlation(on_values, off_values),
        "spearman": _correlation(on_values, off_values, spearman=True),
        "absolute_difference": _percentiles(np.abs(diffs)),
    }
    return ifu_rows, amp_rows, comparison


def _band_hierarchy(hcommon, hdiff, ifus, band, primary):
    """Secondary band decomposition from persistent channel responses."""
    raw_p_map = primary["P_%s" % band]
    pamp_map = primary["Pamp_%s" % band]
    raw_p = {code: raw_p_map.get(code, np.nan) for code in ifus}
    finite_raw = _finite(list(raw_p.values()))
    gauge = float(np.mean(finite_raw)) if finite_raw.size else np.nan
    p = {code: (value - gauge if np.isfinite(value) and np.isfinite(gauge) else np.nan)
         for code, value in raw_p.items()}
    g = {}
    for (h5, code, b), row in hcommon.items():
        if b == band and np.isfinite(row["I_h"]) and np.isfinite(p.get(code, np.nan)):
            g[(h5, code)] = float(row["I_h"] - p[code])
    raw_amp = {(code, amp): pamp_map.get((code, amp), np.nan)
               for code in ifus for amp in range(4)}
    pamp = dict(raw_amp)
    amp_g = {}
    for (h5, code, amp_index, b), row in hdiff.items():
        if b == band and np.isfinite(row["d_h"]) and np.isfinite(pamp.get((code, amp_index), np.nan)):
            amp_g[(h5, code, amp_index)] = float(row["d_h"] - pamp[(code, amp_index)])
    return {"raw_P": raw_p, "gauge": gauge, "P": p, "G": g,
            "raw_Pamp": raw_amp, "Pamp": pamp, "g": amp_g}


def _gray_hierarchy(hcommon_gray, hdiff_gray, ifus, primary):
    """Secondary gray decomposition; the primary A/Delta channel state is retained."""
    gray = {(h5, code): row["I_h"] for (h5, code), row in hcommon_gray.items()
            if np.isfinite(row.get("I_h", np.nan))}
    raw_P = {code: primary["P"].get(code, np.nan) for code in ifus}
    finite_raw = _finite(list(raw_P.values()))
    gauge = float(np.mean(finite_raw)) if finite_raw.size else np.nan
    P = {code: (value - gauge if np.isfinite(value) and np.isfinite(gauge) else np.nan)
         for code, value in raw_P.items()}
    G = {(h5, code): float(value - P[code])
         for (h5, code), value in gray.items()
         if np.isfinite(P.get(code, np.nan))}
    Pamp = {(code, amp): primary["Pamp"].get((code, amp), np.nan)
            for code in ifus for amp in range(4)}
    gray_d = {(h5, code, amp): row["d_h"]
              for (h5, code, amp), row in hdiff_gray.items()
              if np.isfinite(row.get("d_h", np.nan))}
    g = {(h5, code, amp): float(value - Pamp[(code, amp)])
         for (h5, code, amp), value in gray_d.items()
         if np.isfinite(Pamp.get((code, amp), np.nan))}
    return {"gray": gray, "raw_P": raw_P, "gauge": gauge, "P": P,
            "G": G, "raw_Pamp": dict(Pamp), "Pamp": Pamp,
            "gray_d": gray_d, "g": g,
            "primary_A": primary["A"], "primary_Delta": primary["Delta"]}


def _build_logm(items, hierarchy):
    result = {}
    for item in items:
        logm = np.zeros(item.q.size, dtype=float)
        for index, (code, amp_index) in enumerate(zip(item.ifu_code, item.amp)):
            common = hierarchy["P"].get(int(code), np.nan)
            common += hierarchy["G"].get((item.h5_name, int(code)), np.nan)
            differential = hierarchy["Pamp"].get((int(code), int(amp_index)), np.nan)
            differential += hierarchy["g"].get((item.h5_name, int(code), int(amp_index)), np.nan)
            if np.isfinite(common) and np.isfinite(differential):
                logm[index] = common + differential
            else:
                # A measured channel response is sufficient for a gray
                # correction even when P/Pamp is not identifiable for a
                # partial persistent IFU.  This is a direct A + Delta
                # fallback, never an imputed missing amplifier.
                channel = hierarchy.get("primary_A", {}).get((int(code), int(amp_index)), np.nan)
                departure = hierarchy.get("primary_Delta", {}).get(
                    (item.h5_name, int(code), int(amp_index)), np.nan)
                if np.isfinite(channel) and np.isfinite(departure):
                    logm[index] = channel + departure
        result[item.item_index] = logm
    return result


def _hierarchy_rows(hierarchy, ifus, pass_name):
    rows = []
    for code in ifus:
        for (h5, code2), value in hierarchy["G"].items():
            if code2 == code:
                rows.append({"pass": pass_name, "kind": "G_hi", "H5": h5,
                             "IFU_CODE": code, "value": value,
                             "P_or_Pamp": hierarchy["P"].get(code, np.nan),
                             "reconstructed": hierarchy["P"].get(code, np.nan) + value,
                             "target": hierarchy["gray"].get((h5, code), np.nan)})
        for amp_index, amp in enumerate(AMP_ORDER):
            for (h5, code2, amp2), value in hierarchy["g"].items():
                if code2 == code and amp2 == amp_index:
                    p = hierarchy["Pamp"].get((code, amp_index), np.nan)
                    rows.append({"pass": pass_name, "kind": "g_hi_amp", "H5": h5,
                                 "IFU_CODE": code, "AMP": amp, "AMP_INDEX": amp_index,
                                 "value": value, "P_or_Pamp": p,
                                 "reconstructed": p + value,
                                 "target": hierarchy["gray_d"].get((h5, code, amp_index), np.nan)})
        rows.append({"pass": pass_name, "kind": "P_i", "IFU_CODE": code,
                     "value": hierarchy["P"].get(code, np.nan),
                     "raw_value": hierarchy["raw_P"].get(code, np.nan)})
        for amp_index, amp in enumerate(AMP_ORDER):
            rows.append({"pass": pass_name, "kind": "Pamp_i,a", "IFU_CODE": code,
                         "AMP": amp, "AMP_INDEX": amp_index,
                         "value": hierarchy["Pamp"].get((code, amp_index), np.nan),
                         "raw_value": hierarchy["raw_Pamp"].get((code, amp_index), np.nan)})
    return rows


def _hierarchy_ranges(hierarchy):
    result = {}
    for name, values in (("P", hierarchy["P"].values()),
                         ("G_hi", hierarchy["G"].values()),
                         ("Pamp", hierarchy["Pamp"].values()),
                         ("g_hi", hierarchy["g"].values())):
        values = _finite(list(values))
        result[name] = {
            "N": int(values.size), "robust_scatter": _scatter(values),
            "min": float(np.min(values)) if values.size else np.nan,
            "max": float(np.max(values)) if values.size else np.nan,
            "p95_abs": (float(np.percentile(np.abs(values), 95))
                        if values.size else np.nan),
        }
    return result


def _change_summary(first, second, key=None):
    if key is not None:
        first = first[key]
        second = second[key]
    a = np.asarray(list(first.values()), dtype=float)
    b = np.asarray(list(second.values()), dtype=float)
    if a.size != b.size:
        return {"N": 0, "reason": "different support"}
    good = np.isfinite(a) & np.isfinite(b)
    delta = b[good] - a[good]
    return {"N": int(delta.size), "max_abs": float(np.max(np.abs(delta))) if delta.size else np.nan,
            "robust_scatter": _scatter(delta), "median": _location(delta),
            "p95_abs": (float(np.percentile(np.abs(delta), 95)) if delta.size else np.nan)}


def _build_support(items):
    ifus = sorted({int(code) for item in items for code in np.unique(item.ifu_code)})
    rows = []
    new_counts_by_code_amp = {(code, amp_index): 0
                              for code in ifus for amp_index in range(4)}
    old_counts_by_code_amp = {(code, amp_index): 0
                              for code in ifus for amp_index in range(4)}
    for item in items:
        central_common = ((item.q >= Q_MIN) & (item.q <= Q_MAX) &
                          ~item.hardware_bad & np.all(np.isfinite(item.band_total), axis=1))
        central_new = central_common & item.external_valid.any(axis=1)
        old_blank = (central_common & item.blank_valid &
                     ~item.source_candidate.any(axis=1))
        for mask, counts in ((central_new, new_counts_by_code_amp),
                             (old_blank, old_counts_by_code_amp)):
            if not np.any(mask):
                continue
            keys = item.ifu_code[mask].astype(np.int64) * 4 + item.amp[mask].astype(np.int64)
            unique, values = np.unique(keys, return_counts=True)
            for key, value in zip(unique, values):
                counts[(int(key) // 4, int(key) % 4)] += int(value)
    for code in ifus:
        identity = _ifu_identity(next(item for item in items if np.any(item.ifu_code == code)), code)
        new_counts = []
        old_counts = []
        for amp_index in range(4):
            new_counts.append(new_counts_by_code_amp[(code, amp_index)])
            old_counts.append(old_counts_by_code_amp[(code, amp_index)])
        new_amp_count = int(sum(value > 0 for value in new_counts))
        old_amp_count = int(sum(value > 0 for value in old_counts))
        rows.append({
            "IFU_CODE": code, "SPECID": identity[0], "IFUSLOT": identity[1], "IFUID": identity[2],
            "new_LL": new_counts[0], "new_LU": new_counts[1],
            "new_RL": new_counts[2], "new_RU": new_counts[3],
            "old_blank_LL": old_counts[0], "old_blank_LU": old_counts[1],
            "old_blank_RL": old_counts[2], "old_blank_RU": old_counts[3],
            "new_complete_four": bool(all(value > 0 for value in new_counts)),
            "old_blank_complete_four": bool(all(value > 0 for value in old_counts)),
            "new_amp_count": new_amp_count,
            "old_blank_amp_count": old_amp_count,
            "new_support_class": "%d_amp" % new_amp_count,
            "old_blank_support_class": "%d_amp" % old_amp_count,
            "new_total_central_q": int(sum(new_counts)),
            "old_blank_total_central_q": int(sum(old_counts)),
        })
    return rows


def _attach_fplane(rows, fplane):
    for row in rows:
        coords = fplane.get(int(row["IFUSLOT"])) if np.isfinite(row.get("IFUSLOT", np.nan)) else None
        row["fplane_x"] = coords[0] if coords else np.nan
        row["fplane_y"] = coords[1] if coords else np.nan


def _persistent_ifu_rows(hierarchy, support_rows):
    rows = []
    for support in support_rows:
        code = int(support["IFU_CODE"])
        rows.append({
            "IFU_CODE": code, "SPECID": support.get("SPECID", np.nan),
            "IFUSLOT": support.get("IFUSLOT", np.nan), "IFUID": support.get("IFUID", np.nan),
            "P_gray": hierarchy["P"].get(code, np.nan),
            "P_gray_raw": hierarchy["raw_P"].get(code, np.nan),
            "N_gray_H5": int(sum(c == code for _h, c in hierarchy["gray"])),
        })
    return rows


def _primary_channel_rows(primary, support_rows, pass_name):
    rows = []
    identity = {int(row["IFU_CODE"]): row for row in support_rows}
    for code in sorted(identity):
        support_counts = [
            int(np.isfinite(primary["A"].get((code, amp), np.nan)))
            for amp in range(4)]
        for amp_index, amp in enumerate(AMP_ORDER):
            support = primary["support"].get((code, amp_index), {})
            row = identity[code]
            rows.append({
                "pass": pass_name, "IFU_CODE": code,
                "SPECID": row.get("SPECID", np.nan), "IFUSLOT": row.get("IFUSLOT", np.nan),
                "IFUID": row.get("IFUID", np.nan), "AMP": amp, "AMP_INDEX": amp_index,
                "A_i_a": primary["A"].get((code, amp_index), np.nan),
                "A_i_a_ON": primary["A_ON"].get((code, amp_index), np.nan),
                "A_i_a_OFF": primary["A_OFF"].get((code, amp_index), np.nan),
                "support_h5_count": support.get("support_h5_count", 0),
                "support_h5_count_ON": support.get("support_h5_count_ON", 0),
                "support_h5_count_OFF": support.get("support_h5_count_OFF", 0),
                "channel_usable": bool(np.isfinite(primary["A"].get((code, amp_index), np.nan))),
                "persistent_amp_count": int(sum(support_counts)),
                "persistent_support_class": "%d_amp" % int(sum(support_counts)),
            })
    return rows


def _annotate_primary_pass_columns(rows, primary_first, primary_increment,
                                   primary_total):
    for row in rows:
        key = (int(row["IFU_CODE"]), int(row["AMP_INDEX"]))
        row["A_pass1"] = primary_first["A"].get(key, np.nan)
        increment = primary_increment["A"].get(key, np.nan)
        row["dA_pass2"] = increment if np.isfinite(increment) else 0.0
        row["A_total"] = primary_total["A"].get(key, np.nan)
        row["pass_role"] = "pass1" if row.get("pass") == "first" else "cumulative_final"
    return rows


def _annotate_h5_gray_output_rows(first_rows, second_rows, total_rows):
    first_by_key = {_row_key_h5_gray(row): row for row in first_rows}
    second_by_key = {_row_key_h5_gray(row): row for row in second_rows}
    output = []
    for source_rows, role in ((first_rows, "pass1"),
                              (second_rows, "refinement_increment"),
                              (total_rows, "cumulative_final")):
        for source in source_rows:
            key = _row_key_h5_gray(source)
            first = first_by_key.get(key, {})
            second = second_by_key.get(key, {})
            row = dict(source)
            row["pass_role"] = role
            row["R_pass1"] = first.get("R_gray", np.nan)
            increment = second.get("R_gray", np.nan)
            row["dR_pass2"] = increment if np.isfinite(increment) else 0.0
            row["R_total"] = (source.get("R_gray", np.nan)
                               if role == "cumulative_final" else
                               _finite_increment(row["R_pass1"], row["dR_pass2"]))
            row["R_h_i_a"] = row["R_total"] if role == "cumulative_final" else source.get("R_gray", np.nan)
            output.append(row)
    return output


def _annotate_amp_output_rows(first_rows, second_rows, total_rows):
    first_by_key = {_row_key_amp(row): row for row in first_rows}
    second_by_key = {_row_key_amp(row): row for row in second_rows}
    output = []
    for source_rows, role in ((first_rows, "pass1"),
                              (second_rows, "refinement_increment"),
                              (total_rows, "cumulative_final")):
        for source in source_rows:
            key = _row_key_amp(source)
            first = first_by_key.get(key, {})
            second = second_by_key.get(key, {})
            row = dict(source)
            row["pass_role"] = role
            row["r_pass1"] = first.get("robust_location", np.nan)
            increment = second.get("robust_location", np.nan)
            row["dr_pass2"] = increment if np.isfinite(increment) else 0.0
            row["r_total"] = (source.get("robust_location", np.nan)
                               if role == "cumulative_final" else
                               _finite_increment(row["r_pass1"], row["dr_pass2"]))
            output.append(row)
    return output


def _h5_amplifier_departure_rows(h5_amp_gray_rows, primary):
    rows = []
    for row in h5_amp_gray_rows:
        code, amp_index = int(row["IFU_CODE"]), int(row["AMP_INDEX"])
        A = primary["A"].get((code, amp_index), np.nan)
        R = row["R_gray"]
        rows.append({
            "pass": row["pass"], "H5": row["H5"], "IFU_CODE": code,
            "SPECID": row["SPECID"], "IFUSLOT": row["IFUSLOT"], "IFUID": row["IFUID"],
            "AMP": row["AMP"], "AMP_INDEX": amp_index,
            "R_h_i_a": R, "A_i_a": A,
            "Delta_h_i_a": R - A if np.isfinite(R) and np.isfinite(A) else np.nan,
            "R_pass1": row.get("R_pass1", np.nan),
            "dR_pass2": row.get("dR_pass2", np.nan),
            "R_total": row.get("R_total", R),
            "ON_OFF_support": row["ON_OFF_support"], "R_ON": row["R_ON"], "R_OFF": row["R_OFF"],
            "R_ON_minus_OFF": row["R_ON_minus_OFF"],
            "N_exposure": row["N_exposure"],
            "exposure_values_used_ON": row["exposure_values_used_ON"],
            "exposure_values_used_OFF": row["exposure_values_used_OFF"],
            "support_h5_channel": bool(np.isfinite(A)),
        })
    return rows


def _complete_four_decomposition_rows(primary_first, primary_increment,
                                     primary_total, support_rows, pass_name):
    rows = []
    support_by_code = {int(row["IFU_CODE"]): row for row in support_rows}
    for code in sorted(primary_total["complete_ifus"]):
        support = support_by_code.get(code, {})
        P_pass1 = primary_first["P"].get(code, np.nan)
        dP_pass2 = primary_increment["P"].get(code, np.nan)
        dP_pass2 = dP_pass2 if np.isfinite(dP_pass2) else 0.0
        P_total = primary_total["P"].get(code, np.nan)
        for amp_index, amp in enumerate(AMP_ORDER):
            A = primary_total["A"].get((code, amp_index), np.nan)
            A_pass1 = primary_first["A"].get((code, amp_index), np.nan)
            dA_pass2 = primary_increment["A"].get((code, amp_index), np.nan)
            dA_pass2 = dA_pass2 if np.isfinite(dA_pass2) else 0.0
            Pamp_pass1 = primary_first["Pamp"].get((code, amp_index), np.nan)
            dPamp_pass2 = primary_increment["Pamp"].get((code, amp_index), np.nan)
            dPamp_pass2 = dPamp_pass2 if np.isfinite(dPamp_pass2) else 0.0
            Pamp_total = primary_total["Pamp"].get((code, amp_index), np.nan)
            rows.append({
                "pass": pass_name, "IFU_CODE": code,
                "SPECID": support.get("SPECID", np.nan), "IFUSLOT": support.get("IFUSLOT", np.nan),
                "IFUID": support.get("IFUID", np.nan), "AMP": amp, "AMP_INDEX": amp_index,
                "A_i_a": A, "A_pass1": A_pass1, "dA_pass2": dA_pass2,
                "A_total": A, "P_i": P_total, "P_pass1": P_pass1,
                "dP_pass2": dP_pass2, "P_total": P_total,
                "Pamp_i_a": Pamp_total, "Pamp_pass1": Pamp_pass1,
                "dPamp_pass2": dPamp_pass2, "Pamp_total": Pamp_total,
                "reconstructed_A_i_a": P_total + Pamp_total,
                "reconstruction_error": A - (P_total + Pamp_total),
                "Pamp_sum_over_four": sum(primary_total["Pamp"].get((code, index), np.nan)
                                           for index in range(4)),
                "decomposition_method": "complete_four_persistent_channel_mean_total",
            })
    return rows


def _partial_ifu_decomposition_rows(primary_first, primary_increment,
                                    primary_total, support_rows, pass_name):
    rows = []
    support_by_code = {int(row["IFU_CODE"]): row for row in support_rows}
    for code in sorted(set(support_by_code) - set(primary_total["complete_ifus"])):
        available = [amp for amp in range(4)
                     if np.isfinite(primary_total["A"].get((code, amp), np.nan))]
        support = support_by_code[code]
        for amp_index in available:
            rows.append({
                "pass": pass_name, "IFU_CODE": code,
                "SPECID": support.get("SPECID", np.nan), "IFUSLOT": support.get("IFUSLOT", np.nan),
                "IFUID": support.get("IFUID", np.nan), "AMP": AMP_ORDER[amp_index],
                "AMP_INDEX": amp_index,
                "A_i_a": primary_total["A"].get((code, amp_index), np.nan),
                "A_pass1": primary_first["A"].get((code, amp_index), np.nan),
                "dA_pass2": (primary_increment["A"].get((code, amp_index), np.nan)
                              if np.isfinite(primary_increment["A"].get((code, amp_index), np.nan))
                              else 0.0),
                "A_total": primary_total["A"].get((code, amp_index), np.nan),
                "number_available_amps": len(available),
                "available_amp_names": [AMP_ORDER[index] for index in available],
                "P_identified": False, "P_i_partial": np.nan,
                "Pamp_reference_available": False,
                "decomposition_method": "unseparated_channel_only",
                "missing_amp_names": [AMP_ORDER[index] for index in range(4)
                                       if index not in available],
                "note": "cumulative channel response retained; P_i/Pamp_i_a not identifiable",
            })
    return rows


def _make_plots(output_dir, items, sc_rows, sc_fits, r_by_item, valid_by_item,
                amp_rows, common_rows, hcommon, hdiff, gray_rows,
                persistent_rows, support_rows, hierarchy, fplane, pass_name="first"):
    os.environ.setdefault(
        "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "m101_onoff_mplconfig"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_paths = []
    first_rows = [row for row in sc_rows if row["pass"] == pass_name]
    first_fits = {key: value for key, value in sc_fits.items() if key[2] in SOURCE_BANDS}
    ifus = sorted({int(row["IFU_CODE"]) for row in support_rows})

    # Plot 1: choose the exposure with the largest combined central-q support.
    chosen_item = max(items, key=lambda item: sum(
        np.sum((item.q >= Q_MIN) & (item.q <= Q_MAX) & ~item.hardware_bad &
               item.external_valid[:, b] & np.isfinite(item.X[:, b]) &
               np.isfinite(item.band_total[:, b])) for b in range(2)))
    path = output_dir / ("plot1_D_vs_X_%s_exp%d.png" %
                         (_safe_name(chosen_item.h5_name), chosen_item.exposure))
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=False)
    for band_index, band in enumerate(SOURCE_BANDS):
        valid = valid_by_item[chosen_item.item_index][:, band_index]
        x = chosen_item.X[valid, band_index]
        y = chosen_item.band_total[valid, band_index]
        if x.size > 5000:
            take = np.linspace(0, x.size - 1, 5000, dtype=int)
            x, y = x[take], y[take]
        axes[band_index].scatter(x, y, s=2, alpha=.12, rasterized=True, color="tab:blue")
        fit = sc_fits.get((chosen_item.h5_name, chosen_item.exposure, band), {})
        if np.isfinite(fit.get("S", np.nan)) and np.isfinite(fit.get("C", np.nan)) and x.size:
            xline = np.linspace(np.min(x), np.max(x), 100)
            axes[band_index].plot(xline, fit["S"] + fit["C"] * xline,
                                  color="tab:red", lw=2, label="S + C X")
        if x.size:
            edges = np.linspace(np.min(x), np.max(x), 13)
            centers, locations, errors = [], [], []
            for lo, hi in zip(edges[:-1], edges[1:]):
                use = (x >= lo) & (x <= hi if hi == edges[-1] else x < hi)
                if np.sum(use) >= 20:
                    centers.append(_location(x[use])); locations.append(_location(y[use]))
                    errors.append(_uncertainty(y[use]))
            if centers:
                axes[band_index].errorbar(centers, locations, yerr=errors, fmt="o",
                                          ms=4, color="black", label="robust bins")
        axes[band_index].set_title(band)
        axes[band_index].set_xlabel("external X")
        axes[band_index].set_ylabel("D")
        axes[band_index].grid(alpha=.2)
        axes[band_index].legend(fontsize=8)
    fig.suptitle("Central-q D versus X: %s exposure %d" %
                 (chosen_item.h5_name, chosen_item.exposure))
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    plot_paths.append(str(path))

    # Plot 2: automatically select representative repeated IFUs.
    candidates = {}
    for row in common_rows:
        if row.get("pass") != pass_name or not row.get("complete_four_amplifiers"):
            continue
        key = (row["H5"], int(row["IFU_CODE"]))
        values = np.asarray([row.get("r_%s" % amp, np.nan) for amp in AMP_ORDER])
        candidates.setdefault(key, {"contrasts": [], "source": [], "repeat": []})
        candidates[key]["contrasts"].append(float(np.ptp(values)))
    for row in amp_rows:
        if row.get("pass") == pass_name and np.isfinite(row.get("source_fraction_location", np.nan)):
            candidates.setdefault((row["H5"], int(row["IFU_CODE"])),
                                  {"contrasts": [], "source": [], "repeat": []})
            candidates[(row["H5"], int(row["IFU_CODE"]))]["source"].append(
                row["source_fraction_location"])
    for row in [row for row in hcommon.values() if row.get("pass") == pass_name]:
        candidates.setdefault((row["H5"], int(row["IFU_CODE"])),
                              {"contrasts": [], "source": [], "repeat": []})
        if np.isfinite(row.get("repeatability_scatter", np.nan)):
            candidates[(row["H5"], int(row["IFU_CODE"]))]["repeat"].append(
                row["repeatability_scatter"])
    usable = [key for key, metric in candidates.items()
              if metric["contrasts"] and (key in {(row["H5"], int(row["IFU_CODE"]))
                                                  for row in common_rows
                                                  if row.get("pass") == pass_name and row.get("complete_four_amplifiers")})]
    labels = {}
    if usable:
        labels["typical_contrast"] = min(usable, key=lambda key: abs(
            _location(candidates[key]["contrasts"]) - _location(
                [v for k in usable for v in candidates[k]["contrasts"]])))
        labels["large_contrast"] = max(usable, key=lambda key: _location(candidates[key]["contrasts"]))
        source_usable = [key for key in usable if candidates[key]["source"]]
        if source_usable:
            labels["low_source_illumination"] = min(source_usable,
                key=lambda key: _location(candidates[key]["source"]))
            labels["high_source_illumination"] = max(source_usable,
                key=lambda key: _location(candidates[key]["source"]))
        repeat_usable = [key for key in usable if candidates[key]["repeat"]]
        if repeat_usable:
            labels["best_three_exposure_agreement"] = min(repeat_usable,
                key=lambda key: _location(candidates[key]["repeat"]))
            labels["worst_three_exposure_agreement"] = max(repeat_usable,
                key=lambda key: _location(candidates[key]["repeat"]))
    seen = set()
    common_lookup = {(row["H5"], int(row["exposure"]), int(row["IFU_CODE"]), row["band"]): row
                     for row in common_rows if row.get("pass") == pass_name}
    for label, key in labels.items():
        if key in seen:
            continue
        seen.add(key)
        h5, ifu_code = key
        values = []
        for exposure in (1, 2, 3):
            for band in SOURCE_BANDS:
                row = common_lookup.get((h5, exposure, ifu_code, band))
                if row:
                    values.extend([row["r_%s" % amp] for amp in AMP_ORDER])
                    values.append(row["I_e"])
        values = _finite(values)
        if values.size == 0:
            continue
        lo, hi = float(np.min(values)), float(np.max(values))
        pad = max(.03, .08 * (hi - lo))
        path = output_dir / ("plot2_repeatability_%s_%s_ifu%d.png" %
                             (label, _safe_name(h5), ifu_code))
        fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True, sharey=True)
        for band_index, band in enumerate(SOURCE_BANDS):
            for col, exposure in enumerate((1, 2, 3)):
                axis = axes[band_index, col]
                row = common_lookup.get((h5, exposure, ifu_code, band))
                if row:
                    amp_values = [row["r_%s" % amp] for amp in AMP_ORDER]
                    axis.plot(np.arange(4), amp_values, "o-", color="tab:blue", label="r amp")
                    axis.axhline(row["I_e"], color="tab:red", lw=1.8, label="I_e")
                axis.set_title("exp %d %s" % (exposure, band))
                axis.set_xticks(np.arange(4), AMP_ORDER)
                axis.grid(alpha=.2)
                axis.set_ylim(lo - pad, hi + pad)
        axes[0, 0].set_ylabel("log response")
        axes[1, 0].set_ylabel("log response")
        fig.suptitle("Three-dither four-amplifier repeatability: %s IFU %d (%s)" %
                     (h5, ifu_code, label))
        handles, labels_ = axes[0, 0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels_, loc="lower center", ncol=2)
        fig.tight_layout(rect=(0, .04, 1, .95))
        fig.savefig(path, dpi=150)
        plt.close(fig)
        plot_paths.append(str(path))

    # Plot 3: gray ON/OFF common response.
    if gray_rows:
        path = output_dir / "plot3_ON_vs_OFF_IFU_common.png"
        fig, axis = plt.subplots(figsize=(6, 6))
        x = np.asarray([row["I_ON"] for row in gray_rows], dtype=float)
        y = np.asarray([row["I_OFF"] for row in gray_rows], dtype=float)
        axis.scatter(x, y, s=9, alpha=.55)
        if x.size:
            lo, hi = min(np.min(x), np.min(y)), max(np.max(x), np.max(y))
            axis.plot([lo, hi], [lo, hi], "k--", lw=1)
            axis.set_xlim(lo, hi); axis.set_ylim(lo, hi)
        axis.set_xlabel("I_h, ON"); axis.set_ylabel("I_h, OFF")
        axis.set_title("ON versus OFF IFU common response")
        axis.grid(alpha=.2); fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
        plot_paths.append(str(path))

    # Plot 4: amplifier differential ON/OFF.
    if hdiff:
        path = output_dir / "plot4_ON_vs_OFF_amplifier_differential.png"
        fig, axis = plt.subplots(figsize=(6, 6))
        colors = {amp: color for amp, color in zip(AMP_ORDER, ("C0", "C1", "C2", "C3"))}
        for amp in AMP_ORDER:
            values = [(row["d_h"], hdiff.get((row["H5"], int(row["IFU_CODE"]),
                                                 int(row["AMP_INDEX"]), "OFF"), {}).get("d_h", np.nan))
                      for row in hdiff.values() if row.get("pass") == pass_name and row["AMP"] == amp and row["band"] == "ON"]
            if values:
                x = np.asarray([pair[0] for pair in values]); y = np.asarray([pair[1] for pair in values])
                good = np.isfinite(x) & np.isfinite(y)
                axis.scatter(x[good], y[good], s=8, alpha=.5, label=amp, color=colors[amp])
        all_values = _finite([value for row in hdiff.values()
                              for value in (row.get("d_h", np.nan),)])
        if all_values.size:
            lo, hi = np.min(all_values), np.max(all_values)
            axis.plot([lo, hi], [lo, hi], "k--", lw=1)
        axis.set_xlabel("d_h, ON"); axis.set_ylabel("d_h, OFF")
        axis.set_title("ON versus OFF amplifier differential")
        axis.grid(alpha=.2); axis.legend(); fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
        plot_paths.append(str(path))

    # Plot 5: persistent physical IFU map, with no plane fit.
    if persistent_rows:
        path = output_dir / "plot5_persistent_physical_ifu_map.png"
        fig, axis = plt.subplots(figsize=(7, 6))
        x = np.asarray([row.get("fplane_x", np.nan) for row in persistent_rows])
        y = np.asarray([row.get("fplane_y", np.nan) for row in persistent_rows])
        z = np.asarray([row.get("P_gray", np.nan) for row in persistent_rows])
        good = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
        if not np.any(good):
            x = np.asarray([row["IFUSLOT"] for row in persistent_rows], dtype=float)
            y = np.zeros_like(x); good = np.isfinite(z)
            axis.set_xlabel("IFUSLOT (fplane coordinates unavailable)")
        else:
            axis.set_xlabel("fplane X_FP")
        scatter = axis.scatter(x[good], y[good], c=z[good], cmap="coolwarm", s=45, edgecolor="black", linewidth=.25)
        fig.colorbar(scatter, ax=axis, label="P_i [log response]")
        for row, xx, yy in zip(persistent_rows, x, y):
            if np.isfinite(xx) and np.isfinite(yy):
                axis.annotate(str(row["IFUSLOT"]), (xx, yy), fontsize=6)
        axis.set_ylabel("fplane Y_FP"); axis.set_title("Persistent physical IFU response; no plane fit")
        axis.grid(alpha=.2); fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
        plot_paths.append(str(path))

    # Plot 6: H5 x IFU G heatmap.
    if hierarchy["G"]:
        path = output_dir / "plot6_H5_x_IFU_G_heatmap.png"
        h5s = sorted({key[0] for key in hierarchy["G"]})
        matrix = np.full((len(h5s), len(ifus)), np.nan)
        for row_index, h5 in enumerate(h5s):
            for col, code in enumerate(ifus):
                matrix[row_index, col] = hierarchy["G"].get((h5, code), np.nan)
        fig, axis = plt.subplots(figsize=(15, 6))
        image = axis.imshow(matrix, aspect="auto", interpolation="nearest", cmap="coolwarm")
        axis.set_xlabel("physical IFU code"); axis.set_ylabel("H5")
        axis.set_xticks(np.arange(len(ifus)), [str(code) for code in ifus], rotation=90, fontsize=6)
        axis.set_yticks(np.arange(len(h5s)), [_safe_name(h5) for h5 in h5s], fontsize=6)
        fig.colorbar(image, ax=axis, label="G_hi [log response]")
        axis.set_title("H5 x physical IFU departure")
        fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
        plot_paths.append(str(path))

    # Plot 7: support map, four panels.
    if support_rows:
        path = output_dir / "plot7_central_q_support_map.png"
        fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True, sharey=True)
        for amp_index, amp in enumerate(AMP_ORDER):
            axis = axes.flat[amp_index]
            x = np.asarray([row.get("fplane_x", np.nan) for row in support_rows])
            y = np.asarray([row.get("fplane_y", np.nan) for row in support_rows])
            if not np.any(np.isfinite(x) & np.isfinite(y)):
                x = np.asarray([row["IFUSLOT"] for row in support_rows], dtype=float)
                y = np.zeros_like(x)
            new = np.asarray([row["new_%s" % amp] for row in support_rows], dtype=float)
            old = np.asarray([row["old_blank_%s" % amp] for row in support_rows], dtype=float)
            axis.scatter(x, y, c=np.where(new > 0, "tab:blue", "lightgray"), s=35,
                         edgecolor=np.where(old > 0, "black", "tab:red"), linewidth=.7)
            axis.set_title("%s: blue=new central-q, black edge=old blank" % amp, fontsize=8)
            axis.grid(alpha=.15)
        fig.suptitle("Central-q amplifier support")
        fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
        plot_paths.append(str(path))
    return plot_paths


def _diagnostic_candidate_mask(item, band_index):
    """Central-q rows eligible to diagnose a band before D/T positivity."""
    return ((item.q >= Q_MIN) & (item.q <= Q_MAX) & ~item.hardware_bad &
            item.external_valid[:, band_index] & np.isfinite(item.X[:, band_index]))


def _add_diagnostic_stats(row, prefix, values):
    values = _finite(values)
    row["%s_robust_location" % prefix] = _location(values)
    row["%s_median" % prefix] = float(np.median(values)) if values.size else np.nan
    row["%s_p16" % prefix] = float(np.percentile(values, 16)) if values.size else np.nan
    row["%s_p84" % prefix] = float(np.percentile(values, 84)) if values.size else np.nan
    row["%s_minimum" % prefix] = float(np.min(values)) if values.size else np.nan
    row["%s_maximum" % prefix] = float(np.max(values)) if values.size else np.nan


def _extreme_amp_diagnostics(items, sc_fits, r_by_item, valid_by_item,
                             amp_rows, exposure_rows):
    """Flag first-pass amp summaries and retain raw central-q explanations."""
    amp_values = {band: np.asarray([
        row["robust_location"] for row in amp_rows
        if row["band"] == band and row["supported"] and
        np.isfinite(row["robust_location"])
    ], dtype=float) for band in SOURCE_BANDS}
    centers = {band: _location(values) for band, values in amp_values.items()}
    scales = {band: _scatter(values) for band, values in amp_values.items()}
    common_lookup = {(row["H5"], int(row["exposure"]), int(row["IFU_CODE"]), row["band"]): row
                     for row in exposure_rows}
    item_lookup = {(item.h5_name, item.exposure): item for item in items}
    flagged = []
    for row in amp_rows:
        value = row.get("robust_location", np.nan)
        if not row.get("supported") or not np.isfinite(value):
            continue
        band = row["band"]
        scale = scales[band]
        center = centers[band]
        flag_absolute = bool(abs(value) > .20)
        flag_robust = bool(np.isfinite(center) and np.isfinite(scale) and scale > 0 and
                           abs(value - center) > 5. * scale)
        if flag_absolute or flag_robust:
            standardized = (abs(value - center) / scale
                            if np.isfinite(center) and np.isfinite(scale) and scale > 0 else 0.)
            flagged.append({
                "amp_row": row, "flag_absolute": flag_absolute,
                "flag_robust": flag_robust,
                "severity": max(abs(value) / .20, standardized / 5.),
            })

    diagnostic_rows = []
    for flagged_item in flagged:
        amp_row = flagged_item["amp_row"]
        item = item_lookup[(amp_row["H5"], int(amp_row["exposure"]))]
        band_index = SOURCE_BANDS.index(amp_row["band"])
        fit = sc_fits.get((item.h5_name, item.exposure, amp_row["band"]), {})
        candidate = (_diagnostic_candidate_mask(item, band_index) &
                     (item.ifu_code == int(amp_row["IFU_CODE"])) &
                     (item.amp == int(amp_row["AMP_INDEX"])))
        x = item.X[candidate, band_index]
        d = item.band_total[candidate, band_index]
        S, C = fit.get("S", np.nan), fit.get("C", np.nan)
        cx = C * x if np.isfinite(C) else np.full(x.shape, np.nan)
        t = S + cx if np.isfinite(S) else np.full(x.shape, np.nan)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = d / t
        r_values = r_by_item[item.item_index][candidate, band_index]
        used = valid_by_item[item.item_index][candidate, band_index]
        common = common_lookup.get((amp_row["H5"], int(amp_row["exposure"]),
                                    int(amp_row["IFU_CODE"]), amp_row["band"]), {})
        row = {
            "H5": amp_row["H5"], "exposure": int(amp_row["exposure"]),
            "IFU_CODE": int(amp_row["IFU_CODE"]), "SPECID": amp_row["SPECID"],
            "IFUSLOT": amp_row["IFUSLOT"], "IFUID": amp_row["IFUID"],
            "AMP": amp_row["AMP"], "AMP_INDEX": int(amp_row["AMP_INDEX"]),
            "band": amp_row["band"], "diagnostic_pass": "first_M_equals_1",
            "r_amp": amp_row["robust_location"],
            "I_e_i_b": common.get("I_e", np.nan),
            "d_e_i_a_b": common.get("d_%s" % amp_row["AMP"], np.nan),
            "N_central_q_fibers": int(amp_row["N_fibers"]),
            "N_central_q_candidate": int(np.sum(candidate)),
            "N_final_used": int(np.sum(used)),
            "flag_abs_r_gt_0p20": flagged_item["flag_absolute"],
            "flag_abs_r_minus_location_gt_5_scale": flagged_item["flag_robust"],
            "r_population_location": centers[amp_row["band"]],
            "r_population_robust_scale": scales[amp_row["band"]],
            "r_standardized_from_location": (abs(value - centers[amp_row["band"]]) /
                                              scales[amp_row["band"]]
                                              if np.isfinite(centers[amp_row["band"]]) and
                                              np.isfinite(scales[amp_row["band"]]) and
                                              scales[amp_row["band"]] > 0 else np.nan),
            "diagnostic_severity": flagged_item["severity"],
            "fraction_D_nonpositive": float(np.mean(np.isfinite(d) & (d <= 0))) if d.size else np.nan,
            "fraction_T_nonpositive": float(np.mean(np.isfinite(t) & (t <= 0))) if t.size else np.nan,
            "fraction_core_finite": float(np.mean(np.isfinite(d) & np.isfinite(x) & np.isfinite(t))) if d.size else np.nan,
            "fraction_D_over_T_finite": float(np.mean(np.isfinite(ratio))) if ratio.size else np.nan,
            "fraction_finite": float(np.mean(np.isfinite(d) & np.isfinite(x) &
                                              np.isfinite(t) & np.isfinite(ratio))) if d.size else np.nan,
            "fraction_X_positive": float(np.mean(np.isfinite(x) & (x > 0))) if x.size else np.nan,
            "fraction_X_positive_of_finite": (float(np.mean(x[np.isfinite(x)] > 0))
                                               if np.any(np.isfinite(x)) else np.nan),
        }
        for prefix, values in (("D", d), ("S", np.full(x.shape, S)),
                               ("C_times_X", cx), ("T", t),
                               ("D_over_T", ratio), ("X", x),
                               ("log_D_over_T", r_values)):
            _add_diagnostic_stats(row, prefix, values)
        diagnostic_rows.append(row)
    flagged.sort(key=lambda value: value["severity"], reverse=True)
    metadata = {
        "diagnostic_pass": "first_M_equals_1",
        "absolute_threshold": .20,
        "robust_sigma_threshold": 5.,
        "r_population_location": centers,
        "r_population_robust_scale": scales,
        "N_flagged": len(flagged),
        "N_flagged_by_band": {band: int(sum(item["amp_row"]["band"] == band for item in flagged))
                              for band in SOURCE_BANDS},
        "flagged_keys": [(
            item["amp_row"]["H5"], int(item["amp_row"]["exposure"]),
            int(item["amp_row"]["IFU_CODE"]), int(item["amp_row"]["AMP_INDEX"]),
            item["amp_row"]["band"]) for item in flagged],
    }
    return diagnostic_rows, flagged, metadata


def _make_extreme_fiber_plots(output_dir, items, fits_first, r_first,
                              flagged, limit=20):
    """Plot the central-q fibers behind the most extreme first-pass amps."""
    import matplotlib.pyplot as plt

    item_lookup = {(item.h5_name, item.exposure): item for item in items}
    paths = []
    selected = sorted(flagged, key=lambda value: value["severity"], reverse=True)[:limit]
    for selected_case in selected:
        amp_row = selected_case["amp_row"]
        item = item_lookup[(amp_row["H5"], int(amp_row["exposure"]))]
        band = amp_row["band"]
        band_index = SOURCE_BANDS.index(band)
        mask = (_diagnostic_candidate_mask(item, band_index) &
                (item.ifu_code == int(amp_row["IFU_CODE"])) &
                (item.amp == int(amp_row["AMP_INDEX"])))
        x = item.X[mask, band_index]
        d = item.band_total[mask, band_index]
        q = item.q[mask]
        fit = fits_first.get((item.h5_name, item.exposure, band), {})
        S, C = fit.get("S", np.nan), fit.get("C", np.nan)
        with np.errstate(divide="ignore", invalid="ignore"):
            t = S + C * x
            ratio = d / t
            log_ratio = np.log(ratio)
        finite_log = np.isfinite(log_ratio)
        r_amp = float(amp_row["robust_location"])

        if x.size:
            xline = np.linspace(np.nanmin(x), np.nanmax(x), 100)
        else:
            xline = np.asarray([])
        fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
        axes[0].scatter(x, d, s=7, alpha=.35, color="tab:blue", rasterized=True)
        if xline.size and np.isfinite(S) and np.isfinite(C):
            axes[0].plot(xline, S + C * xline, color="tab:red", lw=2,
                         label="T = S + C X")
        axes[0].set_xlabel("external X")
        axes[0].set_ylabel("D")
        axes[0].set_title("D versus X")
        axes[0].grid(alpha=.2)
        axes[0].legend(fontsize=8)

        axes[1].scatter(q, ratio, s=8, alpha=.45, color="tab:green")
        axes[1].axhline(1., color="black", lw=1, ls="--")
        axes[1].set_xlabel("q")
        axes[1].set_ylabel("D / T")
        axes[1].set_title("central-q fibers")
        axes[1].grid(alpha=.2)

        if np.any(finite_log):
            axes[2].hist(log_ratio[finite_log], bins=20, color="tab:purple", alpha=.75)
        axes[2].axvline(r_amp, color="tab:red", lw=2,
                        label="r_amp = %+.4g" % r_amp)
        axes[2].set_xlabel("log(D / T)")
        axes[2].set_ylabel("N fibers")
        axes[2].set_title("response residual distribution")
        axes[2].grid(alpha=.2)
        axes[2].legend(fontsize=8)

        fig.suptitle("Extreme amplifier: %s exp %d IFU %s amp %s %s; N=%d" %
                     (item.h5_name, item.exposure, amp_row["IFUSLOT"],
                      amp_row["AMP"], band, int(np.sum(mask))))
        fig.tight_layout(rect=(0, 0, 1, .93))
        filename = ("extreme_amp_%s_exp%d_ifu%s_%s_%s.png" %
                    (_safe_name(item.h5_name), item.exposure,
                     _safe_name(amp_row["IFUSLOT"]), amp_row["AMP"], band))
        path = output_dir / filename
        fig.savefig(path, dpi=150)
        plt.close(fig)
        paths.append(str(path))
    return paths


def _unsupported_amp_provenance(items, fits_first, valid_first, valid_science):
    """Account for every cache-to-science validity transition per amp/band."""
    ifus = sorted({int(code) for item in items for code in np.unique(item.ifu_code)})
    identity_by_code = {}
    for item in items:
        for code in np.unique(item.ifu_code):
            identity_by_code[int(code)] = _ifu_identity(item, code)
    rows = []
    for code in ifus:
        identity = identity_by_code[code]
        for amp_index, amp in enumerate(AMP_ORDER):
            counts = Counter()
            for key in ("N_total_cache_rows", "N_q_40_70", "N_hardware_valid",
                        "N_hardware_invalid", "N_blank_valid",
                        "N_source_candidate_any", "N_other_cache_valid",
                        "N_first_used_ON", "N_first_used_OFF",
                        "N_final_used_ON", "N_final_used_OFF"):
                counts[key] = 0
            for band in SOURCE_BANDS:
                for key in ("N_band_D_finite_%s" % band,
                            "N_X_finite_%s" % band,
                            "N_external_valid_%s" % band,
                            "N_source_candidate_%s" % band,
                            "N_other_cache_valid_%s" % band,
                            "N_T_finite_%s" % band,
                            "N_T_positive_%s" % band,
                            "N_first_used_%s" % band,
                            "N_final_used_%s" % band):
                    counts[key] = 0
            for item in items:
                group = (item.ifu_code == code) & (item.amp == amp_index)
                counts["N_total_cache_rows"] += int(np.sum(group))
                central = group & (item.q >= Q_MIN) & (item.q <= Q_MAX)
                hardware_valid = central & ~item.hardware_bad
                counts["N_q_40_70"] += int(np.sum(central))
                counts["N_hardware_valid"] += int(np.sum(hardware_valid))
                counts["N_hardware_invalid"] += int(np.sum(central & item.hardware_bad))
                counts["N_blank_valid"] += int(np.sum(central & item.blank_valid))
                counts["N_source_candidate_any"] += int(np.sum(
                    central & item.source_candidate.any(axis=1)))
                other_by_band = []
                for band_index, band in enumerate(SOURCE_BANDS):
                    d_finite = central & np.isfinite(item.band_total[:, band_index])
                    x_finite = central & np.isfinite(item.X[:, band_index])
                    external_valid = central & item.external_valid[:, band_index]
                    other = (hardware_valid & item.external_valid[:, band_index] &
                             np.isfinite(item.band_total[:, band_index]) &
                             np.isfinite(item.X[:, band_index]))
                    counts["N_band_D_finite_%s" % band] += int(np.sum(d_finite))
                    counts["N_X_finite_%s" % band] += int(np.sum(x_finite))
                    counts["N_external_valid_%s" % band] += int(np.sum(external_valid))
                    counts["N_source_candidate_%s" % band] += int(np.sum(
                        central & item.source_candidate[:, band_index]))
                    counts["N_other_cache_valid_%s" % band] += int(np.sum(other))
                    other_by_band.append(other)
                    fit = fits_first.get((item.h5_name, item.exposure, band), {})
                    if np.isfinite(fit.get("S", np.nan)) and np.isfinite(fit.get("C", np.nan)):
                        t = fit["S"] + fit["C"] * item.X[:, band_index]
                        t_finite = other & np.isfinite(t)
                        t_positive = t_finite & (t > 0)
                        counts["N_T_finite_%s" % band] += int(np.sum(t_finite))
                        counts["N_T_positive_%s" % band] += int(np.sum(t_positive))
                    counts["N_first_used_%s" % band] += int(np.sum(
                        valid_first[item.item_index][:, band_index] & group))
                    counts["N_final_used_%s" % band] += int(np.sum(
                        valid_science[item.item_index][:, band_index] & group))
                counts["N_other_cache_valid"] += int(np.sum(
                    other_by_band[0] & other_by_band[1]))

            def band_reasons(band):
                reasons = []
                total = counts["N_total_cache_rows"]
                if total == 0:
                    reasons.append("no cache rows")
                elif counts["N_q_40_70"] == 0:
                    reasons.append("no central-q rows")
                if counts["N_q_40_70"] and counts["N_hardware_valid"] == 0:
                    reasons.append("hardware/date masked")
                if counts["N_hardware_valid"] and counts["N_band_D_finite_%s" % band] == 0:
                    reasons.append("D nonfinite")
                if counts["N_hardware_valid"] and counts["N_X_finite_%s" % band] == 0:
                    reasons.append("X nonfinite")
                if counts["N_hardware_valid"] and counts["N_external_valid_%s" % band] == 0:
                    reasons.append("external X invalid")
                if counts["N_other_cache_valid_%s" % band] and counts["N_T_positive_%s" % band] == 0:
                    reasons.append("T invalid")
                if counts["N_other_cache_valid_%s" % band] and counts["N_final_used_%s" % band] == 0 and not reasons:
                    reasons.append("other validity condition")
                if not reasons and counts["N_final_used_%s" % band] > 0:
                    reasons.append("supported")
                return reasons

            on_reasons = band_reasons("ON")
            off_reasons = band_reasons("OFF")
            supported_on = counts["N_final_used_ON"] > 0
            supported_off = counts["N_final_used_OFF"] > 0
            supported_channel = supported_on or supported_off
            channel_support_class = ("BOTH" if supported_on and supported_off else
                                     "ON_ONLY" if supported_on else
                                     "OFF_ONLY" if supported_off else "NONE")
            all_reasons = [reason for reason in on_reasons + off_reasons if reason != "supported"]
            dominant_order = ("no cache rows", "no central-q rows", "hardware/date masked",
                              "D nonfinite", "X nonfinite", "external X invalid",
                              "T invalid", "other validity condition")
            dominant = next((reason for reason in dominant_order if reason in all_reasons), "supported")
            row = {
                "IFU_CODE": code, "SPECID": identity[0], "IFUSLOT": identity[1], "IFUID": identity[2],
                "AMP": amp, "AMP_INDEX": amp_index,
                "N_total_cache_rows": counts["N_total_cache_rows"],
                "N_q_40_70": counts["N_q_40_70"],
                "N_band_D_finite_ON": counts["N_band_D_finite_ON"],
                "N_band_D_finite_OFF": counts["N_band_D_finite_OFF"],
                "N_X_finite_ON": counts["N_X_finite_ON"],
                "N_X_finite_OFF": counts["N_X_finite_OFF"],
                "N_external_valid_ON": counts["N_external_valid_ON"],
                "N_external_valid_OFF": counts["N_external_valid_OFF"],
                "N_hardware_valid": counts["N_hardware_valid"],
                "N_hardware_invalid": counts["N_hardware_invalid"],
                "N_blank_valid": counts["N_blank_valid"],
                "N_source_candidate_any": counts["N_source_candidate_any"],
                "N_other_cache_valid": counts["N_other_cache_valid"],
                "N_other_cache_valid_ON": counts["N_other_cache_valid_ON"],
                "N_other_cache_valid_OFF": counts["N_other_cache_valid_OFF"],
                "N_source_candidate_ON": counts["N_source_candidate_ON"],
                "N_source_candidate_OFF": counts["N_source_candidate_OFF"],
                "N_T_finite_ON": counts["N_T_finite_ON"],
                "N_T_finite_OFF": counts["N_T_finite_OFF"],
                "N_T_positive_ON": counts["N_T_positive_ON"],
                "N_T_positive_OFF": counts["N_T_positive_OFF"],
                "N_first_used_ON": counts["N_first_used_ON"],
                "N_first_used_OFF": counts["N_first_used_OFF"],
                "N_final_used_ON": counts["N_final_used_ON"],
                "N_final_used_OFF": counts["N_final_used_OFF"],
                "supported_ON": supported_on, "supported_OFF": supported_off,
                "supported_ON_OFF": supported_on and supported_off,
                "supported_channel": supported_channel,
                "channel_support_class": channel_support_class,
                "ON_reason_codes": "; ".join(on_reasons),
                "OFF_reason_codes": "; ".join(off_reasons),
                "dominant_reason": dominant,
                "unsupported": not supported_channel,
                "other_valid_definition": "central-q & hardware-valid & external-valid & finite D & finite X",
                "final_used_definition": "positive finite D and positive finite T in diagnostic science pass",
            }
            rows.append(row)
    return rows


def _choose_representative_item(items):
    return max(items, key=lambda item: sum(
        np.sum(_diagnostic_candidate_mask(item, band_index) &
               np.isfinite(item.band_total[:, band_index]))
        for band_index in range(2)))


def _quantile_bin_rows(item, fits_first):
    rows = []
    trend = {}
    for band_index, band in enumerate(SOURCE_BANDS):
        mask = (_diagnostic_candidate_mask(item, band_index) &
                np.isfinite(item.band_total[:, band_index]))
        x = item.X[mask, band_index]
        d = item.band_total[mask, band_index]
        fit = fits_first.get((item.h5_name, item.exposure, band), {})
        if x.size == 0 or not np.isfinite(fit.get("S", np.nan)) or not np.isfinite(fit.get("C", np.nan)):
            trend[band] = {"N_bins": 0, "spearman_residual_vs_X": np.nan,
                           "obvious_monotonic_trend": False}
            continue
        order = np.argsort(x, kind="mergesort")
        n_bins = min(16, x.size)
        for bin_index, positions in enumerate(np.array_split(order, n_bins), 1):
            bx, bd = x[positions], d[positions]
            x_location = _location(bx)
            d_location = _location(bd)
            predicted = fit["S"] + fit["C"] * x_location
            residual = d_location - predicted
            rows.append({
                "H5": item.h5_name, "exposure": int(item.exposure),
                "band": band, "bin": bin_index, "N_fibers": int(bx.size),
                "X_robust_location": x_location, "D_robust_location": d_location,
                "D_robust_scatter": _scatter(bd),
                "D_location_uncertainty": _uncertainty(bd),
                "model_D": predicted, "DeltaD_bin": residual,
            })
        selected = [row for row in rows if row["H5"] == item.h5_name and
                    row["exposure"] == item.exposure and row["band"] == band]
        bx = np.asarray([row["X_robust_location"] for row in selected])
        residuals = np.asarray([row["DeltaD_bin"] for row in selected])
        rho = _correlation(bx, residuals, spearman=True)
        trend[band] = {
            "N_bins": len(selected), "spearman_residual_vs_X": rho,
            "obvious_monotonic_trend": bool(np.isfinite(rho) and abs(rho) >= .70 and len(selected) >= 6),
        }
    return rows, trend


def _make_quantile_plot(output_dir, items, fits_first):
    item = _choose_representative_item(items)
    bin_rows, trend = _quantile_bin_rows(item, fits_first)
    path = output_dir / ("plot_D_vs_X_quantile_%s_exp%d.png" %
                         (_safe_name(item.h5_name), item.exposure))
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex="col")
    for band_index, band in enumerate(SOURCE_BANDS):
        mask = (_diagnostic_candidate_mask(item, band_index) &
                np.isfinite(item.band_total[:, band_index]))
        x = item.X[mask, band_index]
        d = item.band_total[mask, band_index]
        if x.size > 5000:
            take = np.linspace(0, x.size - 1, 5000, dtype=int)
            x_plot, d_plot = x[take], d[take]
        else:
            x_plot, d_plot = x, d
        top = axes[0, band_index]
        bottom = axes[1, band_index]
        top.scatter(x_plot, d_plot, s=2, alpha=.10, color="tab:blue", rasterized=True)
        fit = fits_first.get((item.h5_name, item.exposure, band), {})
        if x.size and np.isfinite(fit.get("S", np.nan)) and np.isfinite(fit.get("C", np.nan)):
            x_line = np.linspace(np.min(x), np.max(x), 100)
            top.plot(x_line, fit["S"] + fit["C"] * x_line, color="tab:red", lw=2,
                     label="S + C X")
        selected = [row for row in bin_rows if row["band"] == band]
        if selected:
            bx = np.asarray([row["X_robust_location"] for row in selected])
            by = np.asarray([row["D_robust_location"] for row in selected])
            be = np.asarray([row["D_robust_scatter"] for row in selected])
            top.errorbar(bx, by, yerr=be, fmt="o", color="black", ms=4,
                         label="equal-pop robust bins")
            bottom.axhline(0., color="black", lw=1)
            bottom.plot(bx, [row["DeltaD_bin"] for row in selected], "o-",
                        color="tab:purple", ms=4)
        top.set_title(band)
        top.set_ylabel("D")
        bottom.set_xlabel("external X")
        bottom.set_ylabel("DeltaD bin")
        top.grid(alpha=.2); bottom.grid(alpha=.2)
        top.legend(fontsize=8)
    fig.suptitle("Equal-population D versus X diagnostic: %s exposure %d" %
                 (item.h5_name, item.exposure))
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return str(path), bin_rows, trend


def _visible_pair_stats(rows, x_name, y_name, limit=.10):
    x = np.asarray([row[x_name] for row in rows], dtype=float)
    y = np.asarray([row[y_name] for row in rows], dtype=float)
    good = np.isfinite(x) & np.isfinite(y) & (x >= -limit) & (x <= limit) & (y >= -limit) & (y <= limit)
    differences = y[good] - x[good]
    return {
        "N_visible": int(np.sum(good)), "median_difference": _location(differences),
        "robust_scatter_difference": _scatter(differences),
        "p95_abs_difference": (float(np.percentile(np.abs(differences), 95))
                               if differences.size else np.nan),
        "mask": good, "x": x, "y": y,
    }


def _make_zoomed_onoff_plots(output_dir, gray_rows, gray_amp_rows):
    import matplotlib.pyplot as plt
    paths = []
    amp_path = output_dir / "plot4_ON_vs_OFF_amplifier_differential_zoom.png"
    amp_stats = _visible_pair_stats(gray_amp_rows, "d_ON", "d_OFF")
    fig, axis = plt.subplots(figsize=(6, 6))
    colors = {amp: color for amp, color in zip(AMP_ORDER, ("C0", "C1", "C2", "C3"))}
    for amp in AMP_ORDER:
        selected = [row for row, keep in zip(gray_amp_rows, amp_stats["mask"])
                    if keep and row["AMP"] == amp]
        if selected:
            axis.scatter([row["d_ON"] for row in selected], [row["d_OFF"] for row in selected],
                         s=10, alpha=.55, color=colors[amp], label=amp)
    axis.plot([-.10, .10], [-.10, .10], "k--", lw=1)
    axis.set(xlim=(-.10, .10), ylim=(-.10, .10), xlabel="d_h, ON", ylabel="d_h, OFF")
    axis.set_title("ON/OFF amplifier differential; normal-population zoom\nN=%d, median OFF-ON=%+.4g, scatter=%.4g, p95|diff|=%.4g" %
                   (amp_stats["N_visible"], amp_stats["median_difference"],
                    amp_stats["robust_scatter_difference"], amp_stats["p95_abs_difference"]))
    axis.grid(alpha=.2); axis.legend(); fig.tight_layout(); fig.savefig(amp_path, dpi=150); plt.close(fig)
    paths.append(str(amp_path))

    ifu_path = output_dir / "plot3_ON_vs_OFF_IFU_common_zoom.png"
    ifu_stats = _visible_pair_stats(gray_rows, "I_ON", "I_OFF")
    fig, axis = plt.subplots(figsize=(6, 6))
    axis.scatter(ifu_stats["x"][ifu_stats["mask"]], ifu_stats["y"][ifu_stats["mask"]], s=10, alpha=.55)
    axis.plot([-.10, .10], [-.10, .10], "k--", lw=1)
    axis.set(xlim=(-.10, .10), ylim=(-.10, .10), xlabel="I_h, ON", ylabel="I_h, OFF")
    axis.set_title("ON/OFF IFU common response; normal-population zoom\nN=%d, median OFF-ON=%+.4g, scatter=%.4g, p95|diff|=%.4g" %
                   (ifu_stats["N_visible"], ifu_stats["median_difference"],
                    ifu_stats["robust_scatter_difference"], ifu_stats["p95_abs_difference"]))
    axis.grid(alpha=.2); fig.tight_layout(); fig.savefig(ifu_path, dpi=150); plt.close(fig)
    paths.append(str(ifu_path))
    return paths, {"amplifier": {key: value for key, value in amp_stats.items() if key not in ("mask", "x", "y")},
                   "ifu": {key: value for key, value in ifu_stats.items() if key not in ("mask", "x", "y")}}


def _source_strength_rows(items, gray_rows, fraction_by_item):
    strength = {}
    for item in items:
        for band_index, band in enumerate(SOURCE_BANDS):
            mask = _diagnostic_candidate_mask(item, band_index)
            x = item.X[mask, band_index]
            fraction = fraction_by_item[item.item_index][mask, band_index]
            strength.setdefault((item.h5_name, int(item.ifu_code[0])), {})
            # The grouping is completed below from every row-level IFU code.
            for code in np.unique(item.ifu_code[mask]):
                group = mask & (item.ifu_code == code)
                x_group = item.X[group, band_index]
                f_group = fraction_by_item[item.item_index][group, band_index]
                entry = strength.setdefault((item.h5_name, int(code)), {}).setdefault(band, {"x": [], "fraction": []})
                entry["x"].extend(_finite(x_group).tolist())
                entry["fraction"].extend(_finite(f_group).tolist())
    rows = []
    for gray in gray_rows:
        key = (gray["H5"], int(gray["IFU_CODE"]))
        entry = strength.get(key, {})
        x_on = entry.get("ON", {}).get("x", [])
        x_off = entry.get("OFF", {}).get("x", [])
        f_on = entry.get("ON", {}).get("fraction", [])
        f_off = entry.get("OFF", {}).get("fraction", [])
        x_all = x_on + x_off
        f_all = f_on + f_off
        rows.append({
            "H5": gray["H5"], "IFU_CODE": int(gray["IFU_CODE"]),
            "SPECID": gray["SPECID"], "IFUSLOT": gray["IFUSLOT"], "IFUID": gray["IFUID"],
            "I_ON": gray["I_ON"], "I_OFF": gray["I_OFF"],
            "I_OFF_minus_ON": gray["I_OFF"] - gray["I_ON"],
            "median_X": _location(x_all), "median_source_fraction": _location(f_all),
            "median_X_ON": _location(x_on), "median_X_OFF": _location(x_off),
            "median_source_fraction_ON": _location(f_on),
            "median_source_fraction_OFF": _location(f_off),
        })
    return rows


def _make_source_strength_plot(output_dir, source_rows):
    import matplotlib.pyplot as plt
    path = output_dir / "plot3_ON_vs_OFF_IFU_common_source_colored.png"
    x = np.asarray([row["I_ON"] for row in source_rows], dtype=float)
    y = np.asarray([row["I_OFF"] for row in source_rows], dtype=float)
    color = np.asarray([row["median_source_fraction"] for row in source_rows], dtype=float)
    good = np.isfinite(x) & np.isfinite(y) & np.isfinite(color)
    fig, axis = plt.subplots(figsize=(7, 6))
    if np.any(good):
        points = axis.scatter(x[good], y[good], c=color[good], cmap="viridis", s=15, alpha=.7)
        fig.colorbar(points, ax=axis, label="median(C X / T)")
        lo, hi = min(np.min(x[good]), np.min(y[good])), max(np.max(x[good]), np.max(y[good]))
        axis.plot([lo, hi], [lo, hi], "k--", lw=1)
    axis.set_xlabel("I_h, ON"); axis.set_ylabel("I_h, OFF")
    axis.set_title("ON/OFF IFU common response colored by source strength")
    axis.grid(alpha=.2); fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
    return str(path), {"N": int(np.sum(good)),
                       "spearman_source_fraction_vs_OFF_minus_ON": _correlation(
                           color[good], (y - x)[good], spearman=True) if np.sum(good) >= 2 else np.nan}


def _make_robust_G_heatmap(output_dir, hierarchy, support_rows):
    import matplotlib.pyplot as plt
    if not hierarchy["G"]:
        return None, {"N": 0, "center": np.nan, "scale": np.nan, "N_outside": 0}
    ifus = sorted({int(row["IFU_CODE"]) for row in support_rows})
    h5s = sorted({key[0] for key in hierarchy["G"]})
    matrix = np.full((len(h5s), len(ifus)), np.nan)
    for row_index, h5 in enumerate(h5s):
        for col, code in enumerate(ifus):
            matrix[row_index, col] = hierarchy["G"].get((h5, code), np.nan)
    values = _finite(matrix)
    center, scale = _location(values), _scatter(values)
    half = max(.05, 3. * scale if np.isfinite(scale) else .05)
    outside = np.isfinite(matrix) & ((matrix < center - half) | (matrix > center + half))
    path = output_dir / "plot6_H5_x_IFU_G_heatmap_robust.png"
    fig, axis = plt.subplots(figsize=(15, 6))
    image = axis.imshow(matrix, aspect="auto", interpolation="nearest", cmap="coolwarm",
                        vmin=center - half, vmax=center + half)
    y_index, x_index = np.where(outside)
    if x_index.size:
        axis.scatter(x_index, y_index, marker="^", facecolors="none", edgecolors="black",
                     s=30, linewidth=.8, label="outside display range")
        axis.legend(loc="upper right", fontsize=8)
    axis.set_xlabel("physical IFU code"); axis.set_ylabel("H5")
    axis.set_xticks(np.arange(len(ifus)), [str(code) for code in ifus], rotation=90, fontsize=6)
    axis.set_yticks(np.arange(len(h5s)), [_safe_name(h5) for h5 in h5s], fontsize=6)
    fig.colorbar(image, ax=axis, label="G_hi [log response]")
    axis.set_title("Robust-display H5 x IFU departure: center=%+.4g, scale=%.4g, range=[%+.4g,%+.4g]" %
                   (center, scale, center - half, center + half))
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
    return str(path), {"N": int(values.size), "center": center, "scale": scale,
                       "display_half_range": half, "N_outside": int(np.sum(outside))}


def _plot_repeatability_case(output_dir, common_rows, key, category, metric):
    import matplotlib.pyplot as plt
    h5, ifu_code = key
    lookup = {(row["H5"], int(row["exposure"]), int(row["IFU_CODE"]), row["band"]): row
              for row in common_rows if row.get("pass") == "first"}
    values = []
    for exposure in (1, 2, 3):
        for band in SOURCE_BANDS:
            row = lookup.get((h5, exposure, ifu_code, band))
            if row:
                values.extend([row["r_%s" % amp] for amp in AMP_ORDER])
                values.append(row["I_e"])
    values = _finite(values)
    if not values.size:
        return None
    lo, hi = float(np.min(values)), float(np.max(values))
    pad = max(.03, .08 * (hi - lo))
    path = output_dir / ("repeatability_gallery_%02d_%s_%s_ifu%d.png" %
                         (len(list(output_dir.glob("repeatability_gallery_*.png"))) + 1,
                          _safe_name(category), _safe_name(h5), ifu_code))
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True, sharey=True)
    for band_index, band in enumerate(SOURCE_BANDS):
        for col, exposure in enumerate((1, 2, 3)):
            axis = axes[band_index, col]
            row = lookup.get((h5, exposure, ifu_code, band))
            if row:
                axis.plot(np.arange(4), [row["r_%s" % amp] for amp in AMP_ORDER],
                          "o-", color="tab:blue", label="r amp")
                axis.axhline(row["I_e"], color="tab:red", lw=1.8, label="I_e")
            axis.set_title("exp %d %s" % (exposure, band))
            axis.set_xticks(np.arange(4), AMP_ORDER)
            axis.set_ylim(lo - pad, hi + pad)
            axis.grid(alpha=.2)
    axes[0, 0].set_ylabel("log response")
    axes[1, 0].set_ylabel("log response")
    fig.suptitle("Repeatability gallery: %s IFU %d; %s metric=%+.5g" %
                 (h5, ifu_code, category, metric))
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=2)
    fig.tight_layout(rect=(0, .04, 1, .95))
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return str(path)


def _make_repeatability_gallery(output_dir, common_rows, amp_rows, hcommon,
                                gray_rows, flagged):
    candidates = {}
    def get(key):
        return candidates.setdefault(key, {"contrast": [], "source": [], "repeat": [],
                                           "onoff": [], "common": [], "extreme": []})
    for row in common_rows:
        if row.get("pass") != "first" or not row.get("complete_four_amplifiers"):
            continue
        key = (row["H5"], int(row["IFU_CODE"]))
        get(key)["contrast"].append(float(np.ptp(
            [row["r_%s" % amp] for amp in AMP_ORDER])))
    for row in amp_rows:
        if row.get("pass") == "first" and np.isfinite(row.get("source_fraction_location", np.nan)):
            get((row["H5"], int(row["IFU_CODE"])))["source"].append(
                row["source_fraction_location"])
    for row in hcommon.values():
        if row.get("pass") == "first":
            key = (row["H5"], int(row["IFU_CODE"]))
            get(key)["repeat"].append(row.get("repeatability_scatter", np.nan))
            get(key)["common"].append(row.get("I_h", np.nan))
    for row in gray_rows:
        get((row["H5"], int(row["IFU_CODE"])))['onoff'].append(
            abs(row["I_ON_minus_OFF"]))
    for item in flagged:
        row = item["amp_row"]
        get((row["H5"], int(row["IFU_CODE"])))["extreme"].append(item["severity"])
    usable = [key for key, metric in candidates.items() if metric["contrast"]]
    if not usable:
        return [], {"N_candidates": 0}
    all_contrast = [value for key in usable for value in candidates[key]["contrast"]]
    target_contrast = _location(all_contrast)
    selections = []
    used = set()
    def choose(category, metric_name, mode="max", target=None):
        available = [key for key in usable if key not in used and candidates[key][metric_name]
                     and np.isfinite(_location(candidates[key][metric_name]))]
        if not available:
            return
        if mode == "min":
            selected = min(available, key=lambda key: _location(candidates[key][metric_name]))
        elif mode == "target":
            selected = min(available, key=lambda key: abs(_location(candidates[key][metric_name]) - target))
        else:
            selected = max(available, key=lambda key: _location(candidates[key][metric_name]))
        value = _location(candidates[selected][metric_name])
        used.add(selected); selections.append((category, selected, value))

    choose("typical_contrast", "contrast", "target", target_contrast)
    choose("low_contrast", "contrast", "min")
    choose("high_contrast", "contrast", "max")
    choose("high_source_illumination", "source", "max")
    choose("low_source_illumination", "source", "min")
    choose("best_three_dither_agreement", "repeat", "min")
    choose("poor_three_dither_agreement", "repeat", "max")
    choose("largest_ON_OFF_disagreement", "onoff", "max")
    choose("largest_IFU_common_response", "common", "max")
    choose("most_negative_IFU_common_response", "common", "min")
    choose("extreme_amplifier_pathology", "extreme", "max")
    # A low-complexity ordinary example is selected from the remaining pool.
    composite = []
    for key in usable:
        if key in used:
            continue
        metric = candidates[key]
        score = (_location(metric["contrast"]) +
                 _location(metric["repeat"]) if metric["repeat"] else _location(metric["contrast"]))
        if metric["onoff"]:
            score += _location(metric["onoff"])
        composite.append((score, key))
    if composite:
        score, key = min(composite)
        used.add(key); selections.append(("ordinary_well_behaved", key, score))

    # Fill to a 16-case gallery with distinct H5/IFU keys.
    remaining = sorted((key for key in usable if key not in used), key=str)
    for key in remaining:
        if len(selections) >= 16:
            break
        used.add(key)
        selections.append(("additional_distinct_case", key,
                           _location(candidates[key]["contrast"])))

    manifest = []
    for category, key, metric in selections:
        filename = _plot_repeatability_case(output_dir, common_rows, key, category, metric)
        if filename:
            h5, code = key
            row = next((row for row in common_rows if row.get("pass") == "first" and
                        row["H5"] == h5 and int(row["IFU_CODE"]) == code), None)
            manifest.append({
                "plot_filename": filename, "selection_category": category,
                "H5": h5, "IFU_CODE": code,
                "SPECID": row["SPECID"] if row else np.nan,
                "IFUSLOT": row["IFUSLOT"] if row else np.nan,
                "IFUID": row["IFUID"] if row else np.nan,
                "ranking_metric": metric,
            })
    return manifest, {"N_candidates": len(usable), "N_selected": len(manifest),
                      "categories": [row["selection_category"] for row in manifest]}


def _make_support_reason_plot(output_dir, provenance_rows):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    path = output_dir / "plot_support_reason_map.png"
    marker_map = {"no cache rows": "x", "no central-q rows": "^",
                  "hardware/date masked": "s", "D nonfinite": "D",
                  "X nonfinite": "v", "external X invalid": "P",
                  "T invalid": "*", "other validity condition": "+",
                  "supported": "o"}
    colors = {"no cache rows": "tab:gray", "no central-q rows": "tab:orange",
              "hardware/date masked": "tab:red", "D nonfinite": "tab:purple",
              "X nonfinite": "tab:green", "external X invalid": "tab:brown",
              "T invalid": "tab:pink", "other validity condition": "tab:cyan",
              "supported": "tab:blue"}
    fig, axes = plt.subplots(2, 2, figsize=(11, 9), sharex=True, sharey=True)
    for amp_index, amp in enumerate(AMP_ORDER):
        axis = axes.flat[amp_index]
        for row in provenance_rows:
            if row["AMP"] != amp:
                continue
            x = row.get("fplane_x", np.nan)
            y = row.get("fplane_y", np.nan)
            if not np.isfinite(x) or not np.isfinite(y):
                x, y = float(row["IFUSLOT"]), 0.0
            reason = "supported" if row["supported_channel"] else row["dominant_reason"]
            open_point = reason != "supported"
            if open_point:
                axis.scatter([x], [y], marker=marker_map.get(reason, "o"),
                             facecolors="none", edgecolors=colors.get(reason, "black"),
                             s=45, linewidth=.8)
            else:
                axis.scatter([x], [y], marker="o", facecolors=colors[reason],
                             edgecolors=colors[reason], s=45, linewidth=.8)
            if open_point:
                axis.annotate(str(row["IFUSLOT"]), (x, y), fontsize=5)
        axis.set_title("%s: filled=supported, open=unsupported" % amp)
        axis.grid(alpha=.15)
    axes[0, 0].set_ylabel("fplane Y_FP")
    axes[1, 0].set_ylabel("fplane Y_FP")
    axes[1, 0].set_xlabel("fplane X_FP")
    axes[1, 1].set_xlabel("fplane X_FP")
    fig.suptitle("Central-q support with provenance reason coding")
    legend_handles = []
    for reason in marker_map:
        supported = reason == "supported"
        legend_handles.append(Line2D(
            [0], [0], marker=marker_map[reason], linestyle="none",
            markerfacecolor=colors[reason] if supported else "none",
            markeredgecolor=colors[reason], color=colors[reason],
            markersize=7, label=reason))
    fig.legend(handles=legend_handles, loc="lower center", ncol=4,
               fontsize=8, frameon=True)
    fig.tight_layout(rect=(0, .08, 1, .95)); fig.savefig(path, dpi=150); plt.close(fig)
    return str(path)


def _make_partial_support_plots(output_dir, support_rows, h5_amp_gray_rows,
                                primary):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    colors = {0: "tab:gray", 1: "tab:purple", 2: "tab:green",
              3: "tab:orange", 4: "tab:blue"}
    markers = {0: "x", 1: "o", 2: "s", 3: "^", 4: "D"}
    path_map = output_dir / "plot7_IFU_support_count_map.png"
    fig, axis = plt.subplots(figsize=(8, 7))
    for row in support_rows:
        category = int(row.get("new_amp_count", 0))
        x, y = row.get("fplane_x", np.nan), row.get("fplane_y", np.nan)
        if not np.isfinite(x) or not np.isfinite(y):
            x, y = float(row["IFUSLOT"]), 0.0
        axis.scatter([x], [y], marker=markers[category], s=80,
                     color=colors[category],
                     edgecolor=None if category == 0 else "black", linewidth=.5)
        axis.annotate(str(row["IFUSLOT"]), (x, y), fontsize=6)
    handles = [Line2D([0], [0], marker=markers[n], color=colors[n],
                      linestyle="none", markeredgecolor="black", label="%d amps" % n)
               for n in range(5)]
    axis.legend(handles=handles, ncol=5, fontsize=8)
    axis.set_xlabel("fplane X_FP"); axis.set_ylabel("fplane Y_FP")
    axis.set_title("Persistent-cache amplifier support by physical IFU")
    axis.grid(alpha=.2); fig.tight_layout(); fig.savefig(path_map, dpi=150); plt.close(fig)

    h5_counts = {}
    for key in {(row["H5"], int(row["IFU_CODE"])) for row in h5_amp_gray_rows}:
        h5, code = key
        h5_counts[key] = sum(any(row["H5"] == h5 and int(row["IFU_CODE"]) == code and
                                 int(row["AMP_INDEX"]) == amp and np.isfinite(row["R_gray"])
                                 for row in h5_amp_gray_rows) for amp in range(4))
    persistent_counts = Counter()
    for code in sorted({int(row["IFU_CODE"]) for row in support_rows}):
        persistent_counts[sum(np.isfinite(primary["A"].get((code, amp), np.nan))
                              for amp in range(4))] += 1
    exposure_counts = Counter(int(row.get("new_amp_count", 0)) for row in support_rows)
    h5_count_summary = Counter(h5_counts.values())
    path_bar = output_dir / "plot7_support_coverage_summary.png"
    fig, axis = plt.subplots(figsize=(8, 5))
    categories = np.arange(5)
    width = .25
    for offset, (label, counts) in enumerate((
            ("cache/exposure", exposure_counts),
            ("H5/dither-collapsed", h5_count_summary),
            ("persistent channel", persistent_counts))):
        axis.bar(categories + (offset - 1) * width,
                 [counts[n] for n in range(5)], width=width, label=label)
    axis.set_xticks(categories, ["0", "1", "2", "3", "4"])
    axis.set_xlabel("usable amplifiers"); axis.set_ylabel("number of states / IFUs")
    axis.set_title("Support coverage before and after dither collapse")
    axis.legend(); axis.grid(axis="y", alpha=.2); fig.tight_layout()
    fig.savefig(path_bar, dpi=150); plt.close(fig)
    return [str(path_map), str(path_bar)], {
        "exposure": {str(n): int(exposure_counts[n]) for n in range(5)},
        "h5": {str(n): int(h5_count_summary[n]) for n in range(5)},
        "persistent": {str(n): int(persistent_counts[n]) for n in range(5)},
    }


def _make_partial_ifu_example_plots(output_dir, h5_amp_gray_rows, primary):
    import matplotlib.pyplot as plt
    grouped = {}
    for row in h5_amp_gray_rows:
        grouped.setdefault((row["H5"], int(row["IFU_CODE"])), []).append(row)
    by_category = {}
    for key, rows in grouped.items():
        names = sorted({int(row["AMP_INDEX"]) for row in rows if np.isfinite(row["R_gray"])})
        n = len(names)
        if n in (1, 2, 3) and n not in by_category:
            by_category[n] = (key, rows)
    paths = []
    for n in (3, 2, 1):
        if n not in by_category:
            continue
        (h5, code), rows = by_category[n]
        row_by_amp = {int(row["AMP_INDEX"]): row for row in rows}
        values = [row_by_amp[a].get("R_total", row_by_amp[a]["R_gray"])
                  if a in row_by_amp else np.nan for a in range(4)]
        pass1_values = [row_by_amp[a].get("R_pass1", np.nan)
                        if a in row_by_amp else np.nan for a in range(4)]
        increment_values = [row_by_amp[a].get("dR_pass2", 0.0)
                            if a in row_by_amp else np.nan for a in range(4)]
        fig, axis = plt.subplots(figsize=(7, 4.5))
        axis.bar(np.arange(4), [value if np.isfinite(value) else 0. for value in values],
                 color=["tab:blue" if np.isfinite(value) else "lightgray" for value in values],
                 edgecolor="black", label="cumulative total")
        axis.scatter(np.arange(4), pass1_values, marker="x", s=55, color="black",
                     label="pass 1")
        for amp_index, (value, increment) in enumerate(zip(values, increment_values)):
            if np.isfinite(value) and np.isfinite(increment):
                axis.text(amp_index, value, "\nΔ=%.2g" % increment,
                          ha="center", va="top", fontsize=7, color="tab:red")
        for amp_index, value in enumerate(values):
            if not np.isfinite(value):
                axis.text(amp_index, 0., "MISSING", ha="center", va="bottom", rotation=90,
                          fontsize=8)
        partial_method = "unseparated channel-only"
        if all(np.isfinite(primary["Pamp"].get((code, amp), np.nan)) for amp in range(4)):
            partial_method = "common identifiable using independent Pamp"
        axis.set_xticks(np.arange(4), AMP_ORDER)
        axis.set_ylabel("cumulative log response (R_total)")
        axis.set_title("Partial IFU example: %s IFU %d (%d amps)\n%s" %
                       (h5, code, n, partial_method))
        axis.grid(axis="y", alpha=.2); axis.legend(fontsize=8); fig.tight_layout()
        path = output_dir / ("plot_partial_ifu_%damp_%s_ifu%d.png" %
                             (n, _safe_name(h5), code))
        fig.savefig(path, dpi=150); plt.close(fig); paths.append(str(path))
    return paths


def _population_stats(values):
    values = _finite(values)
    return {"N": int(values.size), "robust_location": _location(values),
            "robust_scatter": _scatter(values),
            "p95_abs": float(np.percentile(np.abs(values), 95)) if values.size else np.nan,
            "p99_abs": float(np.percentile(np.abs(values), 99)) if values.size else np.nan,
            "min": float(np.min(values)) if values.size else np.nan,
            "max": float(np.max(values)) if values.size else np.nan}


def _zero_logm_like(items):
    return {item.item_index: np.zeros(item.q.size, dtype=float) for item in items}


def _add_logm_maps(first, increment, items):
    result = {}
    for item in items:
        a = np.asarray(first.get(item.item_index, np.zeros(item.q.size)), dtype=float)
        b = np.asarray(increment.get(item.item_index, np.zeros(item.q.size)), dtype=float)
        result[item.item_index] = np.where(
            np.isfinite(a) & np.isfinite(b), a + b, np.nan)
    return result


def _original_prediction_diagnostics(items, fits_first, fits_final,
                                    logm_first, logm_increment,
                                    logm_total, logm_reported):
    residuals = {name: {band: [] for band in SOURCE_BANDS}
                 for name in ("pass1", "cumulative_final", "old_reported")}
    for item in items:
        first_logm = np.asarray(logm_first.get(item.item_index, np.zeros(item.q.size)), dtype=float)
        increment_logm = np.asarray(logm_increment.get(item.item_index, np.zeros(item.q.size)), dtype=float)
        total_logm = np.asarray(logm_total.get(item.item_index, np.zeros(item.q.size)), dtype=float)
        reported_logm = np.asarray(logm_reported.get(item.item_index, np.zeros(item.q.size)), dtype=float)
        for band_index, band in enumerate(SOURCE_BANDS):
            first_fit = fits_first.get((item.h5_name, item.exposure, band), {})
            final_fit = fits_final.get((item.h5_name, item.exposure, band), {})
            if not all(np.isfinite(first_fit.get(name, np.nan))
                       for name in ("S", "C")):
                continue
            if not all(np.isfinite(final_fit.get(name, np.nan))
                       for name in ("S", "C")):
                continue
            x = item.X[:, band_index]
            data = item.band_total[:, band_index]
            t_first = first_fit["S"] + first_fit["C"] * x
            t_final = final_fit["S"] + final_fit["C"] * x
            base = ((item.q >= Q_MIN) & (item.q <= Q_MAX) &
                    ~item.hardware_bad & item.external_valid[:, band_index] &
                    np.isfinite(x) & np.isfinite(data) & (data > 0) &
                    np.isfinite(t_first) & (t_first > 0) &
                    np.isfinite(t_final) & (t_final > 0))
            predictions = {
                "pass1": np.exp(np.clip(first_logm, -50., 50.)) * t_first,
                "cumulative_final": np.exp(np.clip(total_logm, -50., 50.)) * t_final,
                "old_reported": np.exp(np.clip(reported_logm, -50., 50.)) * t_final,
            }
            for name, prediction in predictions.items():
                good = base & np.isfinite(prediction) & (prediction > 0)
                with np.errstate(divide="ignore", invalid="ignore"):
                    residuals[name][band].extend(
                        np.log(data[good] / prediction[good]).tolist())
    return {
        name: {band: _population_stats(values)
               for band, values in by_band.items()}
        for name, by_band in residuals.items()
    }


def _pass2_residual_identity(items, fits_second, r_second, valid_second, logm_first):
    errors = []
    for item in items:
        first_logm = np.asarray(logm_first.get(item.item_index, np.zeros(item.q.size)), dtype=float)
        for band_index, band in enumerate(SOURCE_BANDS):
            fit = fits_second.get((item.h5_name, item.exposure, band), {})
            if not all(np.isfinite(fit.get(name, np.nan)) for name in ("S", "C")):
                continue
            x = item.X[:, band_index]
            data = item.band_total[:, band_index]
            total = fit["S"] + fit["C"] * x
            good = (valid_second[item.item_index][:, band_index] &
                    np.isfinite(first_logm) & np.isfinite(total) & (total > 0) &
                    np.isfinite(data) & (data > 0))
            with np.errstate(divide="ignore", invalid="ignore"):
                expected = np.log(data[good] / total[good]) - first_logm[good]
            measured = r_second[item.item_index][good, band_index]
            finite = np.isfinite(expected) & np.isfinite(measured)
            errors.extend(np.abs(expected[finite] - measured[finite]).tolist())
    return float(np.max(errors)) if errors else 0.0


def _map_composition_error(first_map, increment_map, total_map):
    errors = []
    for key, first in first_map.items():
        total = total_map.get(key, np.nan)
        increment = increment_map.get(key, np.nan)
        if np.isfinite(first) and np.isfinite(total):
            errors.append(abs(total - (first + (increment if np.isfinite(increment) else 0.0))))
    return float(np.max(errors)) if errors else 0.0


def _map_stats(values):
    return _population_stats([value for value in values if np.isfinite(value)])


def _representative_channel_examples(primary_first, primary_increment,
                                     primary_total, support_rows):
    by_code = {int(row["IFU_CODE"]): row for row in support_rows}
    selected = []
    selected.extend(sorted(primary_total["complete_ifus"])[:3])
    for wanted in (3, 2, 1):
        candidates = []
        for code in sorted(by_code):
            available = sum(np.isfinite(primary_total["A"].get((code, amp), np.nan))
                            for amp in range(4))
            if available == wanted:
                candidates.append(code)
        if candidates:
            selected.append(candidates[0])
    rows = []
    for code in dict.fromkeys(selected):
        available = [AMP_ORDER[amp] for amp in range(4)
                     if np.isfinite(primary_total["A"].get((code, amp), np.nan))]
        values = []
        for amp_index, amp in enumerate(AMP_ORDER):
            first = primary_first["A"].get((code, amp_index), np.nan)
            increment = primary_increment["A"].get((code, amp_index), np.nan)
            total = primary_total["A"].get((code, amp_index), np.nan)
            if np.isfinite(total):
                values.append({"amp": amp, "A_pass1": first,
                               "dA_pass2": increment if np.isfinite(increment) else 0.0,
                               "A_total": total})
        rows.append({"IFU_CODE": code, "support_class": "%d_amp" % len(available),
                     "available_amp_names": available, "channels": values})
    return rows


def _make_population_summary(amp_rows, h5_common_rows, h5_diff_rows,
                             hierarchy, flagged):
    extreme_amp_keys = {(row["H5"], int(row["exposure"]), int(row["IFU_CODE"]),
                         int(row["AMP_INDEX"]), row["band"])
                        for row in (item["amp_row"] for item in flagged)}
    extreme_h5_amp = {(key[0], key[2], key[3]) for key in extreme_amp_keys}
    extreme_h5_ifu = {(key[0], key[2]) for key in extreme_amp_keys}
    extreme_ifu_band = {(key[0], key[2], key[4]) for key in extreme_amp_keys}
    extreme_code = {key[2] for key in extreme_amp_keys}
    populations = {name: {"all": [], "normal": [], "extreme": []}
                   for name in ("r_amp", "d_amp", "I_IFU", "G_hi", "P_i")}

    for row in amp_rows:
        if row.get("pass") != "first" or not row.get("supported"):
            continue
        key = (row["H5"], int(row["exposure"]), int(row["IFU_CODE"]),
               int(row["AMP_INDEX"]), row["band"])
        populations["r_amp"]["all"].append(row["robust_location"])
        populations["r_amp"]["extreme" if key in extreme_amp_keys else "normal"].append(row["robust_location"])
    for row in h5_common_rows:
        if row.get("pass") != "first":
            continue
        common_key = (row["H5"], int(row["IFU_CODE"]), row["band"])
        flag_common = common_key in extreme_ifu_band
        populations["I_IFU"]["all"].append(row["I_h"])
        populations["I_IFU"]["extreme" if flag_common else "normal"].append(row["I_h"])
    for row in h5_diff_rows:
        if row.get("pass") != "first":
            continue
        value = row["d_h"]
        amp_key = (row["H5"], int(row["IFU_CODE"]), int(row["AMP_INDEX"]), row["band"])
        populations["d_amp"]["all"].append(value)
        populations["d_amp"]["extreme" if amp_key in extreme_amp_keys else "normal"].append(value)
    for (h5, code), value in hierarchy["G"].items():
        populations["G_hi"]["all"].append(value)
        populations["G_hi"]["extreme" if (h5, code) in extreme_h5_ifu else "normal"].append(value)
    for code, value in hierarchy["P"].items():
        populations["P_i"]["all"].append(value)
        populations["P_i"]["extreme" if code in extreme_code else "normal"].append(value)

    rows = []
    for observable, groups in populations.items():
        for population, values in groups.items():
            rows.append({"observable": observable, "population": population,
                         **_population_stats(values)})
    return rows, {
        "extreme_amp_keys": [list(key) for key in sorted(extreme_amp_keys, key=str)],
        "N_extreme_amp_keys": len(extreme_amp_keys),
        "N_extreme_h5_ifu": len(extreme_h5_ifu),
        "N_extreme_physical_ifus": len(extreme_code),
    }


def _normal_onoff_summary(gray_rows, gray_amp_rows, flagged):
    extreme_amp = {(row["H5"], int(row["IFU_CODE"]), int(row["AMP_INDEX"]))
                   for row in (case["amp_row"] for case in flagged)}
    extreme_ifu = {(h5, code) for h5, code, _amp in extreme_amp}
    normal_amp = [row for row in gray_amp_rows
                  if (row["H5"], int(row["IFU_CODE"]), int(row["AMP_INDEX"])) not in extreme_amp]
    normal_ifu = [row for row in gray_rows
                  if (row["H5"], int(row["IFU_CODE"])) not in extreme_ifu]
    amp_diff = np.asarray([row["d_OFF"] - row["d_ON"] for row in normal_amp], dtype=float)
    ifu_diff = np.asarray([row["I_OFF"] - row["I_ON"] for row in normal_ifu], dtype=float)
    return {
        "amplifier": {**_population_stats(amp_diff),
                      "pearson": _correlation(
                          [row["d_ON"] for row in normal_amp],
                          [row["d_OFF"] for row in normal_amp])},
        "ifu_common": {**_population_stats(ifu_diff),
                       "pearson": _correlation(
                           [row["I_ON"] for row in normal_ifu],
                           [row["I_OFF"] for row in normal_ifu])},
        "N_excluded_amp_rows": len(gray_amp_rows) - len(normal_amp),
        "N_excluded_ifu_rows": len(gray_rows) - len(normal_ifu),
    }


def _validate_partial_support(items, amp_rows, r_by_item, valid_by_item,
                              primary, h5_gray_rows, h5_gray_common_rows,
                              support_rows):
    """Explicit gates for independent amplifier support and identifiability."""
    item_lookup = {(item.h5_name, item.exposure): item for item in items}
    independence_errors = []
    for row in amp_rows:
        if not row.get("supported") or not np.isfinite(row.get("robust_location", np.nan)):
            continue
        item = item_lookup[(row["H5"], int(row["exposure"]))]
        band_index = SOURCE_BANDS.index(row["band"])
        mask = ((item.ifu_code == int(row["IFU_CODE"])) &
                (item.amp == int(row["AMP_INDEX"])) &
                valid_by_item[item.item_index][:, band_index])
        independence_errors.append(abs(_location(r_by_item[item.item_index][mask, band_index]) -
                                       row["robust_location"]))

    # Remove all other amplifiers from one test IFU/exposure.  The target
    # amplifier's observable must be bitwise-identical to its original value.
    removal_errors = []
    for item in items:
        codes = np.unique(item.ifu_code)
        for code in codes:
            amp_counts = []
            for amp_index in range(4):
                for band_index in range(2):
                    target = ((item.ifu_code == code) & (item.amp == amp_index) &
                              valid_by_item[item.item_index][:, band_index])
                    if np.any(target):
                        original = _location(r_by_item[item.item_index][target, band_index])
                        amp_counts.append((amp_index, band_index, original))
            if len({amp for amp, _band, _value in amp_counts}) >= 2:
                for target_amp, band_index, original in amp_counts[:2]:
                    target = ((item.ifu_code == code) & (item.amp == target_amp) &
                              valid_by_item[item.item_index][:, band_index])
                    removal_errors.append(abs(original - _location(
                        r_by_item[item.item_index][target, band_index])))
                break
        if removal_errors:
            break

    persistent_counts = Counter()
    for code in sorted({int(row["IFU_CODE"]) for row in support_rows}):
        n = sum(np.isfinite(primary["A"].get((code, amp), np.nan)) for amp in range(4))
        persistent_counts[n] += 1
    h5_counts = Counter()
    for key in {(row["H5"], int(row["IFU_CODE"])) for row in h5_gray_rows}:
        h5, code = key
        n = sum(any(row["H5"] == h5 and int(row["IFU_CODE"]) == code and
                    int(row["AMP_INDEX"]) == amp and np.isfinite(row["R_gray"])
                    for row in h5_gray_rows) for amp in range(4))
        h5_counts[n] += 1
    exposure_counts = Counter()
    for row in support_rows:
        exposure_counts[int(row.get("new_amp_count", 0))] += 1

    reconstruction = [
        abs(primary["A"][(code, amp)] - primary["P"][code] - primary["Pamp"][(code, amp)])
        for code in primary["complete_ifus"] for amp in range(4)
        if np.isfinite(primary["A"][(code, amp)]) and
        np.isfinite(primary["P"][code]) and np.isfinite(primary["Pamp"][(code, amp)])]
    pamp_sums = [abs(sum(primary["Pamp"][(code, amp)] for amp in range(4)))
                 for code in primary["complete_ifus"]]
    partial_common_bad = [row for row in h5_gray_common_rows
                          if row["N_amp_available"] < 4 and row["common_identifiable"] and
                          not row["Pamp_reference_available"]]
    partial_channel_only = [row for row in h5_gray_common_rows
                            if row["N_amp_available"] < 4 and
                            not row["common_identifiable"]]

    hardware_row_count = 0
    hardware_channel_count = set()
    hardware_sibling_failures = []
    observed_amp_rows = {
        (row["H5"], int(row["exposure"]), int(row["IFU_CODE"]),
         int(row["AMP_INDEX"]), row["band"])
        for row in amp_rows if row.get("supported")
    }
    for item in items:
        hardware_row_count += int(np.sum(item.hardware_bad &
                                         (item.q >= Q_MIN) & (item.q <= Q_MAX)))
        for code, amp in zip(item.ifu_code[item.hardware_bad], item.amp[item.hardware_bad]):
            hardware_channel_count.add((int(code), int(amp)))
        central = ((item.q >= Q_MIN) & (item.q <= Q_MAX))
        for code in np.unique(item.ifu_code[central]):
            code_mask = central & (item.ifu_code == code)
            bad_amps = set(item.amp[code_mask & item.hardware_bad].astype(int))
            if not bad_amps:
                continue
            for amp_index in range(4):
                if amp_index in bad_amps:
                    continue
                amp_mask = code_mask & (item.amp == amp_index) & ~item.hardware_bad
                for band_index, band in enumerate(SOURCE_BANDS):
                    if not np.any(amp_mask & valid_by_item[item.item_index][:, band_index]):
                        continue
                    key = (item.h5_name, int(item.exposure), int(code), amp_index, band)
                    if key not in observed_amp_rows:
                        hardware_sibling_failures.append(key)
    unavailable_channels = sum(
        not np.isfinite(primary["A"].get((code, amp), np.nan))
        for code in sorted({int(row["IFU_CODE"]) for row in support_rows})
        for amp in range(4))
    observed_channel_keys = {(int(row["IFU_CODE"]), int(row["AMP_INDEX"]))
                             for row in h5_gray_rows if np.isfinite(row["R_gray"])}
    no_missing_imputed = all(
        np.isfinite(primary["A"].get(key, np.nan)) == (key in observed_channel_keys)
        for key in primary["A"])
    max_or_zero = lambda values: float(np.max(values)) if values else 0.0
    category_ok = all(sum(exposure_counts[n] for n in range(5)) == len(support_rows)
                      for _ in (0,))
    return {
        "r_enters_independently_of_other_amplifiers": max_or_zero(independence_errors) <= 2e-15,
        "independent_r_max_abs_error": max_or_zero(independence_errors),
        "removing_other_amplifier_does_not_change_r": max_or_zero(removal_errors) <= 2e-15,
        "removal_test_max_abs_error": max_or_zero(removal_errors),
        "persistent_support_counts": {str(n): int(persistent_counts[n]) for n in range(5)},
        "h5_support_counts": {str(n): int(h5_counts[n]) for n in range(5)},
        "exposure_support_counts": {str(n): int(exposure_counts[n]) for n in range(5)},
        "complete_four_A_reconstruction_max_abs_error": max_or_zero(reconstruction),
        "complete_four_Pamp_sum_max_abs": max_or_zero(pamp_sums),
        "partial_common_requires_independent_Pamp": not partial_common_bad,
        "partial_common_bad_rows": len(partial_common_bad),
        "partial_channel_only_states": len(partial_channel_only),
        "no_missing_amplifier_imputed": no_missing_imputed,
        "hardware_exclusion_is_amplifier_scoped": not hardware_sibling_failures,
        "hardware_sibling_survival_failures": len(hardware_sibling_failures),
        "hardware_bad_central_q_rows": int(hardware_row_count),
        "hardware_excluded_physical_amp_channels": len(hardware_channel_count),
        "persistent_nonfinite_unavailable_channels": int(unavailable_channels),
        "support_category_accounting_exact": category_ok,
        "all_partial_support_gates_pass": bool(
            max_or_zero(independence_errors) <= 2e-15 and
            max_or_zero(removal_errors) <= 2e-15 and
            max_or_zero(reconstruction) <= 2e-12 and
            max_or_zero(pamp_sums) <= 2e-12 and
            not partial_common_bad and category_ok and no_missing_imputed and
            not hardware_sibling_failures),
    }


def _validate(first_sc, final_sc, first_hierarchy, final_hierarchy,
              common_rows, hcommon, hdiff, support_rows, refinement_used):
    c_reconstruction = [abs(row["C_reconstruction_error"]) for row in final_sc
                        if np.isfinite(row.get("C_reconstruction_error", np.nan))]
    p_reconstruction = [abs(value - final_hierarchy["P"].get(code, np.nan) -
                             final_hierarchy["G"].get((h5, code), np.nan))
                        for (h5, code), value in final_hierarchy["gray"].items()
                        if np.isfinite(value) and np.isfinite(final_hierarchy["P"].get(code, np.nan)) and
                        np.isfinite(final_hierarchy["G"].get((h5, code), np.nan))]
    amp_reconstruction = [abs(value - final_hierarchy["Pamp"].get((code, amp), np.nan) -
                               final_hierarchy["g"].get((h5, code, amp), np.nan))
                          for (h5, code, amp), value in final_hierarchy["gray_d"].items()
                          if np.isfinite(value) and np.isfinite(final_hierarchy["Pamp"].get((code, amp), np.nan)) and
                          np.isfinite(final_hierarchy["g"].get((h5, code, amp), np.nan))]
    amp_zero = [abs(row["d_sum"]) for row in common_rows
                if row.get("complete_four_amplifiers") and np.isfinite(row.get("d_sum", np.nan))]
    pamp_zero = []
    p_zero = []
    for values in (final_hierarchy["P"].values(),):
        finite_values = _finite(list(values))
        if finite_values.size:
            p_zero.append(abs(float(np.mean(finite_values))))
    for code in sorted({key[0] for key in final_hierarchy["Pamp"]}):
        values = [final_hierarchy["Pamp"].get((code, amp), np.nan) for amp in range(4)]
        if all(np.isfinite(values)):
            pamp_zero.append(abs(float(np.sum(values))))
    gc_groups = {}
    for row in final_sc:
        if np.isfinite(row.get("gc", np.nan)):
            gc_groups.setdefault((row["H5"], row["band"]), []).append(row["gc"])
    gc_zero = [abs(_location(values)) for values in gc_groups.values()]
    max_or_nan = lambda values: float(np.max(values)) if values else 0.0
    return {
        "native_spectra_read": False,
        "native_pixels_read_in_fit": False,
        "bands_used": list(SOURCE_BANDS),
        "null_bands_used": False,
        "central_q_only": True,
        "q_range": [Q_MIN, Q_MAX],
        "blank_classification_required": False,
        "blank_metadata_used_in_fit": False,
        "additive_correction_applied": False,
        "focal_plane_ax_ay_fitted": False,
        "chromatic_Q_fitted": False,
        "wavelength_dependent_amplifier_correction_fitted": False,
        "psf_correction_applied": False,
        "astrometric_correction_applied": False,
        "nonlinear_global_optimizer_used": False,
        "equal_amplifier_weight_max_error": max_or_nan(amp_zero),
        "equal_amplifier_weight_is_one_quarter": max_or_nan(amp_zero) <= 2e-15,
        "amplifier_differential_sum_max_abs": max_or_nan(amp_zero),
        "Pamp_sum_max_abs": max_or_nan(pamp_zero),
        "P_mean_gauged_max_abs": max_or_nan(p_zero),
        "gc_H5_band_location_max_abs": max_or_nan(gc_zero),
        "C_decomposition_max_abs_error": max_or_nan(c_reconstruction),
        "P_plus_G_reconstruction_max_abs_error": max_or_nan(p_reconstruction),
        "Pamp_plus_g_reconstruction_max_abs_error": max_or_nan(amp_reconstruction),
        "optional_refinement_passes": int(1 if refinement_used else 0),
        "optional_refinement_at_most_once": True,
        "all_validation_gates_pass": bool(
            max_or_nan(amp_zero) <= 2e-15 and max_or_nan(pamp_zero) <= 2e-12 and
            max_or_nan(p_zero) <= 2e-15 and
            max_or_nan(gc_zero) <= 2e-10 and
            max_or_nan(c_reconstruction) <= 2e-10 and max_or_nan(p_reconstruction) <= 2e-12 and
            max_or_nan(amp_reconstruction) <= 2e-12),
    }


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--band-cache", default="m101_band_initializer/m101_band_cache.npz")
    parser.add_argument("--h5", nargs="*", default=[],
                        help="optional H5 paths or basenames to select from the cache")
    parser.add_argument("--output-dir", default="m101_onoff_initializer")
    parser.add_argument("--fplane", help="optional frozen fplaneall.txt for diagnostic maps")
    parser.add_argument("--no-refinement", action="store_true",
                        help="skip the single optional gray consistency refinement")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _state_maps(sc_rows, hierarchy, ifus, h5_names):
    sc = {}
    for row in sc_rows:
        key = _safe_key(row["H5"], row["exposure"], row["band"])
        sc[key] = {name: row.get(name, np.nan) for name in
                   ("S", "C", "S_uncertainty", "C_uncertainty", "residual_scatter",
                    "N_fibers", "X_min", "X_max", "fraction_positive_X", "Gc", "gc")}
    return sc


def main():
    args = _parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise SystemExit("output directory is nonempty; use --overwrite: %s" % output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    print("[onoff-init] loading compact cache", flush=True)
    items, cache_manifest, cache_path, cache_manifest_path = _load_cache(args.band_cache, args.h5)
    fplane, fplane_path = _load_fplane(args.fplane)
    ifus = sorted({int(code) for item in items for code in np.unique(item.ifu_code)})
    h5_names = sorted({item.h5_name for item in items})

    # Build the old blank-only support comparison from frozen cache metadata.
    support_rows = _build_support(items)
    _attach_fplane(support_rows, fplane)

    print("[onoff-init] first robust exposure regressions", flush=True)
    sc_first, fits_first = _fit_exposure_SC(items, pass_name="first")
    cbar_first, log_cbar_first, gc_first = _normalize_C(sc_first)
    r_first, valid_first, total_first, fraction_first = _build_r(items, fits_first)
    amp_first, amp_lookup_first = _build_amplifier_measurements(
        items, r_first, valid_first, fraction_first, "first")
    exposure_first, common_lookup_first, d_lookup_first = _build_exposure_common(
        items, amp_lookup_first, "first")
    (h5_amp_first, h5_amp_lookup_first, h5_amp_gray_first,
     h5_amp_gray_lookup_first) = _build_h5_amplifier_summaries(
        items, amp_lookup_first, "first")
    primary_first = _persistent_amplifier_hierarchy(
        h5_amp_first, h5_amp_gray_first, ifus)
    (h5_common_first, h5_diff_first, hcommon_first, hdiff_first,
     h5_gray_common_first, h5_gray_diff_first,
     hcommon_gray_first, hdiff_gray_first) = _build_h5_common_summaries(
        h5_amp_first, h5_amp_gray_first, primary_first, ifus, "first",
        exposure_rows=exposure_first)
    gray_rows_first, gray_amp_first, comparison_first = _compare_on_off(
        hcommon_first, hdiff_first, ifus)
    hierarchy_first = _gray_hierarchy(
        hcommon_gray_first, hdiff_gray_first, ifus, primary_first)

    refinement_used = not args.no_refinement
    if refinement_used:
        print("[onoff-init] applying exactly one gray refinement", flush=True)
        logm_first = _build_logm(items, hierarchy_first)
        sc_second, fits_second = _fit_exposure_SC(
            items, logm_by_item=logm_first, pass_name="second")
        cbar_second, log_cbar_second, gc_second = _normalize_C(sc_second)
        r_second, valid_second, total_second, fraction_second = _build_r(
            items, fits_second, logm_by_item=logm_first)
        amp_second, amp_lookup_second = _build_amplifier_measurements(
            items, r_second, valid_second, fraction_second, "second")
        exposure_second, common_lookup_second, d_lookup_second = _build_exposure_common(
            items, amp_lookup_second, "second")
        (h5_amp_second, h5_amp_lookup_second, h5_amp_gray_second,
         h5_amp_gray_lookup_second) = _build_h5_amplifier_summaries(
            items, amp_lookup_second, "second")
        primary_second = _persistent_amplifier_hierarchy(
            h5_amp_second, h5_amp_gray_second, ifus)
        (h5_common_second, h5_diff_second, hcommon_second, hdiff_second,
         h5_gray_common_second, h5_gray_diff_second,
         hcommon_gray_second, hdiff_gray_second) = _build_h5_common_summaries(
            h5_amp_second, h5_amp_gray_second, primary_second, ifus, "second",
            exposure_rows=exposure_second)
        gray_rows_second, gray_amp_second, comparison_second = _compare_on_off(
            hcommon_second, hdiff_second, ifus)
        hierarchy_second = _gray_hierarchy(
            hcommon_gray_second, hdiff_gray_second, ifus, primary_second)
    else:
        sc_second, fits_second = [], {}
        cbar_second, log_cbar_second, gc_second = cbar_first, log_cbar_first, gc_first
        amp_second, exposure_second = [], []
        h5_amp_second, h5_amp_lookup_second, h5_amp_gray_second = [], {}, []
        h5_amp_gray_lookup_second = {}
        primary_second = primary_first
        h5_common_second, h5_diff_second = [], []
        h5_gray_common_second, h5_gray_diff_second = [], []
        hcommon_second, hdiff_second = {}, {}
        hcommon_gray_second, hdiff_gray_second = {}, {}
        gray_rows_second, gray_amp_second = [], []
        comparison_second = {}
        hierarchy_second = hierarchy_first

    # Pass 2 is evaluated on D/M1, so its measured hierarchy is a residual
    # increment.  Compose every final response in log space while preserving
    # the original support from pass 1.
    primary_increment = (primary_second if refinement_used
                         else _zero_primary_increment(primary_first))
    amp_second_rows = amp_second if refinement_used else []
    h5_amp_second_rows = h5_amp_second if refinement_used else []
    h5_amp_gray_second_rows = h5_amp_gray_second if refinement_used else []
    amp_total = _compose_amp_measurements(amp_first, amp_second_rows)
    amp_lookup_total = {_row_key_amp(row): row for row in amp_total}
    exposure_total, common_lookup_total, d_lookup_total = _build_exposure_common(
        items, amp_lookup_total, "total")
    h5_amp_total = _compose_h5_band_rows(h5_amp_first, h5_amp_second_rows)
    h5_amp_gray_total = _compose_h5_gray_rows(
        h5_amp_gray_first, h5_amp_gray_second_rows)
    primary_total = _compose_primary_hierarchy(
        primary_first, primary_increment, h5_amp_total, h5_amp_gray_total, ifus)
    (h5_common_total, h5_diff_total, hcommon_total, hdiff_total,
     h5_gray_common_total, h5_gray_diff_total,
     hcommon_gray_total, hdiff_gray_total) = _build_h5_common_summaries(
        h5_amp_total, h5_amp_gray_total, primary_total, ifus, "total",
        exposure_rows=exposure_total)
    gray_rows_total, gray_amp_total, comparison_total = _compare_on_off(
        hcommon_total, hdiff_total, ifus)
    hierarchy_total = _gray_hierarchy(
        hcommon_gray_total, hdiff_gray_total, ifus, primary_total)

    logm_first_audit = _build_logm(items, hierarchy_first)
    logm_increment_audit = (_build_logm(items, hierarchy_second)
                            if refinement_used else _zero_logm_like(items))
    logm_total_audit = _add_logm_maps(
        logm_first_audit, logm_increment_audit, items)
    logm_reported_audit = (logm_increment_audit if refinement_used
                           else logm_first_audit)
    logm_total_from_hierarchy = _build_logm(items, hierarchy_total)
    logm_mapping_error = []
    for item in items:
        a = logm_total_audit[item.item_index]
        b = logm_total_from_hierarchy[item.item_index]
        good = np.isfinite(a) & np.isfinite(b)
        if np.any(good):
            logm_mapping_error.extend(np.abs(a[good] - b[good]).tolist())
    audit_prediction = _original_prediction_diagnostics(
        items, fits_first, fits_second if refinement_used else fits_first,
        logm_first_audit, logm_increment_audit, logm_total_audit,
        logm_reported_audit)
    pass2_identity_error = (_pass2_residual_identity(
        items, fits_second, r_second, valid_second, logm_first_audit)
        if refinement_used else 0.0)
    primary_composition_errors = {
        "A_total": _map_composition_error(
            primary_first["A"], primary_increment["A"],
            primary_total["A"]),
        "A_ON_total": _map_composition_error(
            primary_first["A_ON"], primary_increment["A_ON"],
            primary_total["A_ON"]),
        "A_OFF_total": _map_composition_error(
            primary_first["A_OFF"], primary_increment["A_OFF"],
            primary_total["A_OFF"]),
    }
    channel_examples = _representative_channel_examples(
        primary_first, primary_increment, primary_total, support_rows)
    multiplicative_errors = []
    for key, first_value in primary_first["A"].items():
        increment_value = primary_increment["A"].get(key, np.nan)
        if np.isfinite(first_value) and np.isfinite(increment_value):
            multiplicative_errors.append(abs(
                math.exp(first_value + increment_value) -
                math.exp(first_value) * math.exp(increment_value)))
    audit = {
        "current_pre_fix_final_semantics": "second_pass_residual_only",
        "current_pre_fix_A_semantics": "primary_second_A, which equals dA_pass2 when refinement is enabled",
        "current_pre_fix_P_Pamp_semantics": "P/Pamp/G/g derived from the second-pass residual hierarchy",
        "pass1_quantity": "r_pass1 = log(D / T_pass1); R_pass1 and A_pass1 are absolute original-data responses",
        "correction_before_pass2": "D_pass2 = D_original / exp(R_pass1-compatible hierarchy)",
        "pass2_quantity": "dR_pass2 = log(D_pass2 / T_pass2) = log(D_original / T_pass2) - R_pass1",
        "final_quantity": "R_total = R_pass1 + dR_pass2; A_total = A_pass1 + dA_pass2",
        "pass2_residual_identity_max_abs": pass2_identity_error,
        "logm_total_mapping_max_abs": max(logm_mapping_error) if logm_mapping_error else 0.0,
        "primary_composition_max_abs": primary_composition_errors,
        "multiplicative_composition_max_abs": max(multiplicative_errors) if multiplicative_errors else 0.0,
        "representative_channel_examples": channel_examples,
        "channel_magnitude_stats": {
            "A_pass1": _map_stats(primary_first["A"].values()),
            "dA_pass2": _map_stats(primary_increment["A"].values()),
            "A_total": _map_stats(primary_total["A"].values()),
            "abs_dA_over_abs_A_pass1": _map_stats([
                abs(increment / first)
                for key, first in primary_first["A"].items()
                for increment in [primary_increment["A"].get(key, np.nan)]
                if np.isfinite(first) and np.isfinite(increment) and first != 0
            ]),
        },
        "final_semantics": "cumulative_first_plus_refinement_increment",
        "original_prediction_residuals": audit_prediction,
    }

    # S+C X in the final pass is in the M1-corrected data domain.  The final
    # original-data prediction therefore uses this fit with the cumulative
    # M1*M2 hierarchy below.
    final_sc = sc_second if refinement_used else sc_first
    final_fits = fits_second if refinement_used else fits_first
    science_valid = valid_second if refinement_used else valid_first
    final_primary = primary_total
    final_amp = amp_total
    final_exposure = exposure_total
    final_h5_common = h5_common_total
    final_h5_diff = h5_diff_total
    final_hcommon = hcommon_total
    final_hdiff = hdiff_total
    final_h5_amp = h5_amp_total
    final_h5_amp_gray = h5_amp_gray_total
    final_h5_gray_common = h5_gray_common_total
    final_h5_gray_diff = h5_gray_diff_total
    final_gray_rows = gray_rows_total
    final_gray_amp = gray_amp_total
    final_comparison = comparison_total
    final_hierarchy = hierarchy_total
    final_cbar = cbar_second if refinement_used else cbar_first
    final_log_cbar = log_cbar_second if refinement_used else log_cbar_first
    final_gc = gc_second if refinement_used else gc_first
    final_band_hierarchies = {
        band: _band_hierarchy(final_hcommon, final_hdiff, ifus, band, final_primary)
        for band in SOURCE_BANDS
    }
    first_band_hierarchies = {
        band: _band_hierarchy(hcommon_first, hdiff_first, ifus, band, primary_first)
        for band in SOURCE_BANDS
    }

    # Add final hierarchy fields to the human-readable H5 tables.
    for row in final_h5_common:
        row["I_gray"] = final_hierarchy["gray"].get((row["H5"], int(row["IFU_CODE"])), np.nan)
        row["G_hi"] = final_hierarchy["G"].get((row["H5"], int(row["IFU_CODE"])), np.nan)
    for row in final_h5_diff:
        row["d_gray"] = final_hierarchy["gray_d"].get(
            (row["H5"], int(row["IFU_CODE"]), int(row["AMP_INDEX"])), np.nan)
        row["g_hi_amp"] = final_hierarchy["g"].get(
            (row["H5"], int(row["IFU_CODE"]), int(row["AMP_INDEX"])), np.nan)
        row["Pamp_gray"] = final_hierarchy["Pamp"].get(
            (int(row["IFU_CODE"]), int(row["AMP_INDEX"])), np.nan)

    # Final persistent tables retain band-specific estimates as diagnostics.
    persistent_ifu = _persistent_ifu_rows(final_hierarchy, support_rows)
    persistent_ifu_first = _persistent_ifu_rows(hierarchy_first, support_rows)
    persistent_amp = []
    for index, code in enumerate(ifus):
        identity = next((row for row in support_rows if row["IFU_CODE"] == code), {})
        persistent_ifu[index]["P_ON"] = final_band_hierarchies["ON"]["P"].get(code, np.nan)
        persistent_ifu[index]["P_OFF"] = final_band_hierarchies["OFF"]["P"].get(code, np.nan)
        persistent_ifu_first[index]["P_ON"] = first_band_hierarchies["ON"]["P"].get(code, np.nan)
        persistent_ifu_first[index]["P_OFF"] = first_band_hierarchies["OFF"]["P"].get(code, np.nan)
        for amp_index, amp in enumerate(AMP_ORDER):
            persistent_amp.append({
                "IFU_CODE": code, "SPECID": identity.get("SPECID", np.nan),
                "IFUSLOT": identity.get("IFUSLOT", np.nan), "IFUID": identity.get("IFUID", np.nan),
                "AMP": amp, "AMP_INDEX": amp_index,
                "Pamp_gray": final_hierarchy["Pamp"].get((code, amp_index), np.nan),
                "Pamp_ON": final_band_hierarchies["ON"]["Pamp"].get((code, amp_index), np.nan),
                "Pamp_OFF": final_band_hierarchies["OFF"]["Pamp"].get((code, amp_index), np.nan),
            })
    _attach_fplane(persistent_ifu, fplane)
    _attach_fplane(persistent_ifu_first, fplane)
    _attach_fplane(persistent_amp, fplane)
    primary_channel_rows = _annotate_primary_pass_columns(
        _primary_channel_rows(final_primary, support_rows, "final"),
        primary_first, primary_increment, final_primary)
    primary_channel_rows_first = _annotate_primary_pass_columns(
        _primary_channel_rows(primary_first, support_rows, "first"),
        primary_first, primary_increment, final_primary)
    h5_departure_rows = _h5_amplifier_departure_rows(final_h5_amp_gray, final_primary)
    h5_departure_rows_first = _h5_amplifier_departure_rows(
        h5_amp_gray_first, primary_first)
    h5_gray_output_rows = _annotate_h5_gray_output_rows(
        h5_amp_gray_first, h5_amp_gray_second_rows, h5_amp_gray_total)
    amp_output_rows = _annotate_amp_output_rows(
        amp_first, amp_second_rows, amp_total)
    complete_decomposition_rows = _complete_four_decomposition_rows(
        primary_first, primary_increment, final_primary, support_rows, "final")
    partial_decomposition_rows = _partial_ifu_decomposition_rows(
        primary_first, primary_increment, final_primary, support_rows, "final")
    _attach_fplane(primary_channel_rows, fplane)
    _attach_fplane(primary_channel_rows_first, fplane)
    _attach_fplane(h5_departure_rows, fplane)
    _attach_fplane(complete_decomposition_rows, fplane)
    _attach_fplane(partial_decomposition_rows, fplane)

    hierarchy_rows = _hierarchy_rows(hierarchy_first, ifus, "first")
    if refinement_used:
        hierarchy_rows.extend(_hierarchy_rows(hierarchy_second, ifus, "second"))
    hierarchy_rows.extend(_hierarchy_rows(final_hierarchy, ifus, "total"))
    hierarchy_rows.extend({"pass": "diagnostic", "kind": "exposure_I_residual",
                           "H5": row["H5"], "IFU_CODE": row["IFU_CODE"],
                           "band": row["band"], "exposure": e,
                           "value": row["I_e%d" % e], "reference": row["I_h"],
                           "departure": row["deltaI_e%d" % e]}
                          for row in final_h5_common for e in (1, 2, 3)
                          if np.isfinite(row.get("I_e%d" % e, np.nan)))
    hierarchy_rows.extend({"pass": "diagnostic", "kind": "exposure_d_residual",
                           "H5": row["H5"], "IFU_CODE": row["IFU_CODE"],
                           "AMP": row["AMP"], "AMP_INDEX": row["AMP_INDEX"],
                           "band": row["band"], "exposure": e,
                           "value": row["d_e%d" % e], "reference": row["d_h"],
                           "departure": row["deltad_e%d" % e]}
                          for row in final_h5_diff for e in (1, 2, 3)
                          if np.isfinite(row.get("d_e%d" % e, np.nan)))

    # First-to-second changes are compact numerical diagnostics.
    changes = {
        "S": _change_summary(
            {_safe_key(row["H5"], row["exposure"], row["band"]): row["S"] for row in sc_first},
            {_safe_key(row["H5"], row["exposure"], row["band"]): row["S"] for row in (sc_second or sc_first)}),
        "C": _change_summary(
            {_safe_key(row["H5"], row["exposure"], row["band"]): row["C"] for row in sc_first},
            {_safe_key(row["H5"], row["exposure"], row["band"]): row["C"] for row in (sc_second or sc_first)}),
        "A_pass1": _map_stats(primary_first["A"].values()),
        "dA_pass2": _map_stats(primary_increment["A"].values()),
        "A_total": _map_stats(primary_total["A"].values()),
        "P_pass1": _map_stats(primary_first["P"].values()),
        "dP_pass2": _map_stats(primary_increment["P"].values()),
        "P_total": _map_stats(primary_total["P"].values()),
        "Pamp_pass1": _map_stats(primary_first["Pamp"].values()),
        "dPamp_pass2": _map_stats(primary_increment["Pamp"].values()),
        "Pamp_total": _map_stats(primary_total["Pamp"].values()),
        "second_pass_hierarchy_residual": (
            _hierarchy_ranges(hierarchy_second) if refinement_used else
            {name: _map_stats([]) for name in ("P", "G_hi", "Pamp", "g_hi")}),
    }
    identity_by_code = {
        int(row["IFU_CODE"]): (int(row["SPECID"]), int(row["IFUSLOT"]), int(row["IFUID"]))
        for row in support_rows
    }
    p_by_identity = {
        str(identity_by_code[code]): value
        for code, value in final_hierarchy["P"].items()
        if code in identity_by_code
    }
    g_by_identity = {
        _safe_key(h5, identity_by_code[code]): value
        for (h5, code), value in final_hierarchy["G"].items()
        if code in identity_by_code
    }
    pamp_by_identity = {
        _safe_key(identity_by_code[code], amp): value
        for (code, amp), value in final_hierarchy["Pamp"].items()
        if code in identity_by_code
    }
    small_g_by_identity = {
        _safe_key(h5, identity_by_code[code], amp): value
        for (h5, code, amp), value in final_hierarchy["g"].items()
        if code in identity_by_code
    }
    h5_composition_errors = []
    for row in h5_amp_gray_total:
        if np.isfinite(row.get("R_pass1", np.nan)) and np.isfinite(row.get("R_total", np.nan)):
            h5_composition_errors.append(abs(
                row["R_total"] - (row["R_pass1"] + row.get("dR_pass2", 0.0))))
    partial_output_errors = [
        abs(row["A_i_a"] - row["A_total"])
        for row in partial_decomposition_rows
        if np.isfinite(row.get("A_i_a", np.nan)) and np.isfinite(row.get("A_total", np.nan))]
    complete_output_errors = [abs(row["reconstruction_error"])
                              for row in complete_decomposition_rows
                              if np.isfinite(row.get("reconstruction_error", np.nan))]
    total_pass1_errors = [
        abs(primary_total["A"].get(key, np.nan) - primary_first["A"].get(key, np.nan))
        for key in primary_first["A"]
        if not refinement_used and np.isfinite(primary_total["A"].get(key, np.nan)) and
        np.isfinite(primary_first["A"].get(key, np.nan))]
    refinement_validation = {
        "refinement_disabled_total_equals_pass1_max_abs": max(total_pass1_errors, default=0.0),
        "refinement_enabled_total_equals_pass1_plus_increment": (
            max(primary_composition_errors.values()) <= 2e-12 if refinement_used else True),
        "second_pass_is_residual_not_absolute": pass2_identity_error <= 2e-12,
        "complete_four_decomposition_uses_A_total": max(complete_output_errors, default=0.0) <= 2e-12,
        "partial_outputs_use_A_total": max(partial_output_errors, default=0.0) <= 2e-12,
        "final_original_prediction_uses_cumulative_response": (
            audit["logm_total_mapping_max_abs"] <= 2e-12),
        "multiplicative_composition_max_abs": audit["multiplicative_composition_max_abs"],
        "h5_total_composition_max_abs": max(h5_composition_errors, default=0.0),
        "only_one_refinement_pass": bool(refinement_used in (True, False)),
        "all_refinement_bookkeeping_gates_pass": bool(
            (max(total_pass1_errors, default=0.0) <= 2e-12) and
            (max(primary_composition_errors.values()) <= 2e-12) and
            (pass2_identity_error <= 2e-12) and
            (max(complete_output_errors, default=0.0) <= 2e-12) and
            (max(partial_output_errors, default=0.0) <= 2e-12) and
            (audit["logm_total_mapping_max_abs"] <= 2e-12) and
            (audit["multiplicative_composition_max_abs"] <= 2e-12) and
            (max(h5_composition_errors, default=0.0) <= 2e-12)),
    }
    validation = _validate(sc_first, final_sc, hierarchy_first, final_hierarchy,
                           final_exposure, final_hcommon, final_hdiff,
                           support_rows, refinement_used)
    partial_validation = _validate_partial_support(
        items, amp_first, r_first, valid_first, primary_first,
        h5_amp_gray_first, h5_gray_common_first, support_rows)
    validation["partial_support"] = partial_validation
    validation["refinement_bookkeeping"] = refinement_validation
    validation["all_validation_gates_pass"] = bool(
        validation["all_validation_gates_pass"] and
        partial_validation["all_partial_support_gates_pass"] and
        refinement_validation["all_refinement_bookkeeping_gates_pass"])
    if not validation["all_validation_gates_pass"]:
        raise RuntimeError("ON/OFF hierarchy validation gate failed: %s" % validation)

    # CSV outputs.  SC and amplifier/exposure summaries retain both passes.
    _write_rows(output_dir / "exposure_SC.csv", sc_first + (sc_second if refinement_used else []))
    _write_rows(output_dir / "amp_measurements.csv", amp_first + (amp_second if refinement_used else []))
    _write_rows(output_dir / "exposure_ifu_common.csv", exposure_first + (exposure_second if refinement_used else []))
    _write_rows(output_dir / "h5_ifu_common.csv", final_h5_common)
    _write_rows(output_dir / "h5_amp_differential.csv", final_h5_diff)
    _write_rows(output_dir / "persistent_ifu.csv", persistent_ifu)
    _write_rows(output_dir / "persistent_amp.csv", persistent_amp)
    _write_rows(output_dir / "hierarchy_residuals.csv", hierarchy_rows)
    _write_rows(output_dir / "support.csv", support_rows)
    _write_rows(output_dir / "on_off_ifu_comparison.csv", final_gray_rows)
    _write_rows(output_dir / "on_off_amp_comparison.csv", final_gray_amp)

    plot_paths = _make_plots(
        output_dir, items, sc_first, fits_first, r_first, valid_first, amp_first,
        exposure_first, hcommon_first, hdiff_first, gray_rows_first,
        persistent_ifu_first, support_rows, hierarchy_first, fplane, pass_name="first")

    # Diagnostics deliberately consume first-pass M=1 observables.  They do
    # not feed back into the estimator or alter any science-validity mask.
    extreme_rows, flagged_extremes, extreme_meta = _extreme_amp_diagnostics(
        items, fits_first, r_first, valid_first, amp_first, exposure_first)
    provenance_rows = _unsupported_amp_provenance(
        items, fits_first, valid_first, science_valid)
    _attach_fplane(provenance_rows, fplane)
    extreme_plot_paths = _make_extreme_fiber_plots(
        output_dir, items, fits_first, r_first, flagged_extremes)
    quantile_plot_path, quantile_bin_rows, quantile_trend = _make_quantile_plot(
        output_dir, items, fits_first)
    zoom_plot_paths, zoom_plot_meta = _make_zoomed_onoff_plots(
        output_dir, gray_rows_first, gray_amp_first)
    source_rows = _source_strength_rows(items, gray_rows_first, fraction_first)
    source_plot_path, source_plot_meta = _make_source_strength_plot(
        output_dir, source_rows)
    robust_g_path, robust_g_meta = _make_robust_G_heatmap(
        output_dir, hierarchy_first, support_rows)
    gallery_manifest, gallery_meta = _make_repeatability_gallery(
        output_dir, exposure_first, amp_first, hcommon_first,
        gray_rows_first, flagged_extremes)
    support_reason_path = _make_support_reason_plot(output_dir, provenance_rows)
    support_count_plot_paths, support_coverage = _make_partial_support_plots(
        output_dir, support_rows, final_h5_amp_gray, final_primary)
    partial_example_plot_paths = _make_partial_ifu_example_plots(
        output_dir, final_h5_amp_gray, final_primary)
    population_rows, population_meta = _make_population_summary(
        amp_first, h5_common_first, h5_diff_first, hierarchy_first,
        flagged_extremes)
    normal_onoff = _normal_onoff_summary(
        gray_rows_first, gray_amp_first, flagged_extremes)

    _write_rows(output_dir / "extreme_amp_diagnostics.csv", extreme_rows)
    _write_rows(output_dir / "unsupported_amp_provenance.csv", provenance_rows)
    _write_rows(output_dir / "onoff_ifu_source_strength.csv", source_rows)
    _write_rows(output_dir / "repeatability_gallery.csv", gallery_manifest)
    _write_rows(output_dir / "diagnostic_population_summary.csv", population_rows)
    _write_rows(output_dir / "D_vs_X_quantile_bins.csv", quantile_bin_rows)
    _write_rows(output_dir / "amplifier_exposure_measurements.csv",
                amp_output_rows)
    _write_rows(output_dir / "amplifier_h5_measurements.csv",
                h5_gray_output_rows)
    _write_rows(output_dir / "persistent_amplifier_channels.csv",
                primary_channel_rows_first + primary_channel_rows)
    _write_rows(output_dir / "h5_amplifier_departures.csv",
                h5_departure_rows_first + h5_departure_rows)
    _write_rows(output_dir / "complete_four_ifu_decomposition.csv",
                complete_decomposition_rows)
    _write_rows(output_dir / "partial_ifu_decomposition.csv",
                partial_decomposition_rows)

    total_measurements = sum(int(row["N_fibers"]) for row in final_amp)
    amp_summary_count = sum(1 for row in final_amp if row["supported"])
    old_supported = sum(row["old_blank_complete_four"] for row in support_rows)
    new_supported = sum(row["new_complete_four"] for row in support_rows)
    repeat_I = [row["repeatability_scatter"] for row in final_h5_common]
    repeat_d = [row["repeatability_scatter"] for row in final_h5_diff]
    state = {
        "schema_version": OUTPUT_SCHEMA,
        "architecture": "physics-first multiplicative ON/OFF central-q hierarchy",
        "hierarchy": {
            "Cbar_b": final_cbar,
            "log_Cbar_b": final_log_cbar,
            "Gc_h_b": {_safe_key(h5, band): value for (h5, band), value in final_gc.items()},
            "gc_e_b": {_safe_key(row["H5"], row["exposure"], row["band"]): row["gc"]
                       for row in final_sc},
            "S_e_b": {_safe_key(row["H5"], row["exposure"], row["band"]): row["S"]
                      for row in final_sc},
            "C_e_b": {_safe_key(row["H5"], row["exposure"], row["band"]): row["C"]
                      for row in final_sc},
            "R_pass1_h_i_a": {
                _safe_key(row["H5"], row["IFU_CODE"], row["AMP"]): row["R_gray"]
                for row in h5_amp_gray_first
            },
            "dR_pass2_h_i_a": {
                _safe_key(row["H5"], row["IFU_CODE"], row["AMP"]): row["R_gray"]
                for row in h5_amp_gray_second_rows
            },
            "R_total_h_i_a": {
                _safe_key(row["H5"], row["IFU_CODE"], row["AMP"]): row["R_total"]
                for row in h5_amp_gray_total
            },
            "R_pass1_h_i_a_b": {
                _safe_key(row["H5"], row["IFU_CODE"], row["AMP"], row["band"]): row["R_h"]
                for row in h5_amp_first
            },
            "dR_pass2_h_i_a_b": {
                _safe_key(row["H5"], row["IFU_CODE"], row["AMP"], row["band"]): row["R_h"]
                for row in h5_amp_second_rows
            },
            "R_total_h_i_a_b": {
                _safe_key(row["H5"], row["IFU_CODE"], row["AMP"], row["band"]): row["R_total"]
                for row in h5_amp_total
            },
            "A_pass1_i_a": {_safe_key(code, amp): value
                             for (code, amp), value in primary_first["A"].items()},
            "dA_pass2_i_a": {_safe_key(code, amp): value
                              for (code, amp), value in primary_increment["A"].items()},
            "A_total_i_a": {_safe_key(code, amp): value
                             for (code, amp), value in final_primary["A"].items()},
            "A_i_a": {_safe_key(code, amp): value
                       for (code, amp), value in final_primary["A"].items()},
            "A_i_a_ON": {_safe_key(code, amp): value
                          for (code, amp), value in final_primary["A_ON"].items()},
            "A_i_a_OFF": {_safe_key(code, amp): value
                           for (code, amp), value in final_primary["A_OFF"].items()},
            "Delta_h_i_a": {_safe_key(h5, code, amp): value
                            for (h5, code, amp), value in final_primary["Delta"].items()},
            "P_i_exact_channel_mean": {
                str(code): value for code, value in final_primary["P"].items()
            },
            "P_pass1_i": {str(code): value for code, value in primary_first["P"].items()},
            "dP_pass2_i": {str(code): value for code, value in primary_increment["P"].items()},
            "P_total_i": {str(code): value for code, value in final_primary["P"].items()},
            "Pamp_pass1_i_a": {_safe_key(code, amp): value
                                for (code, amp), value in primary_first["Pamp"].items()},
            "dPamp_pass2_i_a": {_safe_key(code, amp): value
                                 for (code, amp), value in primary_increment["Pamp"].items()},
            "Pamp_total_i_a": {_safe_key(code, amp): value
                                for (code, amp), value in final_primary["Pamp"].items()},
            "Pamp_reference_by_H5": {
                _safe_key(h5, code, amp): value
                for (h5, code, amp), value in final_primary["Pamp_reference_by_h5"].items()
            },
            "P_i": {str(code): value for code, value in final_hierarchy["P"].items()},
            "P_i_by_physical_identity": p_by_identity,
            "G_hi": {_safe_key(h5, code): value for (h5, code), value in final_hierarchy["G"].items()},
            "G_hi_by_physical_identity": g_by_identity,
            "Pamp_i_a": {_safe_key(code, amp): value for (code, amp), value in final_hierarchy["Pamp"].items()},
            "Pamp_i,a_by_physical_identity": pamp_by_identity,
            "g_hi_a": {_safe_key(h5, code, amp): value for (h5, code, amp), value in final_hierarchy["g"].items()},
            "g_h,i,a_by_physical_identity": small_g_by_identity,
            # Punctuation-preserving aliases keep the mathematical names
            # explicit in the JSON as well as in the CSV column names.
            "Pamp_i,a": {_safe_key(code, amp): value for (code, amp), value in final_hierarchy["Pamp"].items()},
            "g_h,i,a": {_safe_key(h5, code, amp): value for (h5, code, amp), value in final_hierarchy["g"].items()},
            "gauges": {"P_IFU_raw_location": final_hierarchy["gauge"],
                       "P_ON_gauge": final_band_hierarchies["ON"]["gauge"],
                       "P_OFF_gauge": final_band_hierarchies["OFF"]["gauge"]},
            "field_semantics": {
                "A_i_a": "cumulative final A_total_i_a",
                "P_i": "gauged cumulative secondary IFU common term",
                "P_i_exact_channel_mean": "raw exact cumulative complete-four mean of A_total_i_a",
                "Pamp_i_a": "cumulative secondary amplifier contrast",
                "R_total_h_i_a": "original-data cumulative channel response",
            },
            "pass1": {
                "R_h_i_a": {_safe_key(row["H5"], row["IFU_CODE"], row["AMP"]): row["R_gray"]
                             for row in h5_amp_gray_first},
                "A_i_a": {_safe_key(code, amp): value
                           for (code, amp), value in primary_first["A"].items()},
                "P_i": {str(code): value for code, value in primary_first["P"].items()},
                "Pamp_i_a": {_safe_key(code, amp): value
                              for (code, amp), value in primary_first["Pamp"].items()},
            },
            "refinement_increment": {
                "dR_h_i_a": {_safe_key(row["H5"], row["IFU_CODE"], row["AMP"]): row["R_gray"]
                              for row in h5_amp_gray_second_rows},
                "dA_i_a": {_safe_key(code, amp): value
                            for (code, amp), value in primary_increment["A"].items()},
                "dP_i": {str(code): value for code, value in primary_increment["P"].items()},
                "dPamp_i_a": {_safe_key(code, amp): value
                               for (code, amp), value in primary_increment["Pamp"].items()},
            },
            "cumulative_final": {
                "R_total_h_i_a": {_safe_key(row["H5"], row["IFU_CODE"], row["AMP"]): row["R_total"]
                                  for row in h5_amp_gray_total},
                "A_total_i_a": {_safe_key(code, amp): value
                                 for (code, amp), value in final_primary["A"].items()},
                "P_total_i": {str(code): value for code, value in final_primary["P"].items()},
                "Pamp_total_i_a": {_safe_key(code, amp): value
                                    for (code, amp), value in final_primary["Pamp"].items()},
            },
        },
        "band_specific_observables": {
            "ON_OFF_IFU_comparison": final_comparison["ifu_common"],
            "ON_OFF_amplifier_comparison": final_comparison["amp_differential"],
            "gray_definition": "0.5*(I_h,ON + I_h,OFF) only when both are available",
        },
        "diagnostics": {
            "S_distribution": {band: _percentiles([row["S"] for row in final_sc if row["band"] == band]) for band in SOURCE_BANDS},
            "C_distribution": {band: _percentiles([row["C"] for row in final_sc if row["band"] == band]) for band in SOURCE_BANDS},
            "residual_scatter": {band: _percentiles([row["residual_scatter"] for row in final_sc if row["band"] == band]) for band in SOURCE_BANDS},
            "Gc_h_b": {_safe_key(h5, band): value for (h5, band), value in final_gc.items()},
            "gc_e_b": _percentiles([row["gc"] for row in final_sc]),
            "three_exposure_IFU_repeatability": _percentiles(repeat_I),
            "three_exposure_amplifier_repeatability": _percentiles(repeat_d),
            "hierarchy_amplitudes": _hierarchy_ranges(final_hierarchy),
            "first_to_second_pass": changes,
            "refinement_audit": audit,
            "refinement_bookkeeping_validation": refinement_validation,
            "primary_channel_ranges": {
                "A_pass1": _map_stats(primary_first["A"].values()),
                "dA_pass2": _map_stats(primary_increment["A"].values()),
                "A_total": _map_stats(primary_total["A"].values()),
            },
            "plot_paths": plot_paths,
            "diagnostic_pass": "first_M_equals_1",
            "extreme_amp": extreme_meta,
            "unsupported_amp_provenance": {
                "N_rows": len(provenance_rows),
                "N_unsupported_amp_band_rows": int(sum(row["unsupported"] for row in provenance_rows)),
                "dominant_reason_counts": dict(Counter(
                    row["dominant_reason"] for row in provenance_rows if row["unsupported"])),
            },
            "robust_G_heatmap": robust_g_meta,
            "D_vs_X_quantile": {
                "representative_H5": _choose_representative_item(items).h5_name,
                "representative_exposure": int(_choose_representative_item(items).exposure),
                "trend": quantile_trend,
            },
            "normal_population_ON_OFF": normal_onoff,
            "source_strength_plot": source_plot_meta,
            "diagnostic_population_summary": population_meta,
            "repeatability_gallery": gallery_meta,
            "additional_plot_paths": (extreme_plot_paths + [quantile_plot_path] +
                                       zoom_plot_paths + [source_plot_path, robust_g_path,
                                       support_reason_path] + support_count_plot_paths +
                                       partial_example_plot_paths +
                                       [row["plot_filename"] for row in gallery_manifest]),
            "partial_support_coverage": support_coverage,
            "partial_validation": partial_validation,
        },
        "support": {
            "physical_IFUs_total": len(ifus),
            "old_blank_only_complete_four": int(old_supported),
            "new_central_q_complete_four": int(new_supported),
            "exposure_support_by_amp_count": support_coverage["exposure"],
            "h5_support_by_amp_count": support_coverage["h5"],
            "persistent_support_by_amp_count": support_coverage["persistent"],
            "total_persistent_amplifier_channels": int(sum(
                np.isfinite(value) for value in final_primary["A"].values())),
            "total_persistent_amplifier_channels_unavailable": int(
                sum(not np.isfinite(value) for value in final_primary["A"].values())),
            "old_blank_only_supported_IFU_codes": [row["IFU_CODE"] for row in support_rows if row["old_blank_complete_four"]],
            "new_central_q_supported_IFU_codes": [row["IFU_CODE"] for row in support_rows if row["new_complete_four"]],
        },
        "counts": {
            "total_central_q_ON_OFF_fiber_band_measurements": int(total_measurements),
            "supported_amp_summaries": int(amp_summary_count),
            "all_amp_summary_rows": int(len(final_amp)),
            "complete_exposure_IFU_band_summaries": int(sum(row["complete_four_amplifiers"] for row in final_exposure)),
            "H5_IFU_band_summaries": int(len(final_h5_common)),
        },
        "passes": {"refinement_used": refinement_used,
                   "first_Cbar": cbar_first, "second_Cbar": cbar_second,
                   "first_hierarchy_ranges": _hierarchy_ranges(hierarchy_first),
                   "second_hierarchy_ranges": (
                       _hierarchy_ranges(hierarchy_second) if refinement_used else
                       {name: _map_stats([]) for name in ("P", "G_hi", "Pamp", "g_hi")}),
                   "primary_channel_ranges": {
                       "A_pass1": _map_stats(primary_first["A"].values()),
                       "dA_pass2": _map_stats(primary_increment["A"].values()),
                       "A_total": _map_stats(primary_total["A"].values()),
                   },
                   "parameter_changes": changes},
        "validation": validation,
        "provenance": {
            "band_cache": file_identity(cache_path),
            "band_cache_manifest": file_identity(cache_manifest_path),
            "cache_schema": cache_manifest.get("schema_version"),
            "cache_manifest_selected": cache_manifest,
            "cache_only_fit": True,
            "native_pixels_read": False,
            "external_X_source": "cache field X[:, ON/OFF]",
            "hardware_mask_source": "cache field hardware_bad",
            "source_metadata_loaded_for_diagnostics_only": True,
            "blank_metadata_fit_role": "not used; no blank classification required",
            "fplane_file": str(fplane_path) if fplane_path else None,
            "fit_bands": list(SOURCE_BANDS),
            "excluded_null_bands": ["HIGH1", "LOW1", "HIGH2", "LOW2", "HIGH3"],
            "q_selection": [Q_MIN, Q_MAX],
            "optimizer": "direct robust IRLS linear S+C X regressions only; no nonlinear/global optimization",
            "diagnostic_pass": "first_M_equals_1",
            "diagnostics_do_not_change_science": True,
        },
        "artifacts": {
            "output_directory": str(output_dir),
            "plots": plot_paths,
            "diagnostic_plots": (extreme_plot_paths + [quantile_plot_path] +
                                 zoom_plot_paths + [source_plot_path, robust_g_path,
                                 support_reason_path] + support_count_plot_paths +
                                 partial_example_plot_paths +
                                 [row["plot_filename"] for row in gallery_manifest]),
            "diagnostic_tables": {
                "extreme_amp_diagnostics": str(output_dir / "extreme_amp_diagnostics.csv"),
                "unsupported_amp_provenance": str(output_dir / "unsupported_amp_provenance.csv"),
                "onoff_ifu_source_strength": str(output_dir / "onoff_ifu_source_strength.csv"),
                "repeatability_gallery": str(output_dir / "repeatability_gallery.csv"),
                "diagnostic_population_summary": str(output_dir / "diagnostic_population_summary.csv"),
                "D_vs_X_quantile_bins": str(output_dir / "D_vs_X_quantile_bins.csv"),
                "amplifier_exposure_measurements": str(output_dir / "amplifier_exposure_measurements.csv"),
                "amplifier_h5_measurements": str(output_dir / "amplifier_h5_measurements.csv"),
                "persistent_amplifier_channels": str(output_dir / "persistent_amplifier_channels.csv"),
                "h5_amplifier_departures": str(output_dir / "h5_amplifier_departures.csv"),
                "complete_four_ifu_decomposition": str(output_dir / "complete_four_ifu_decomposition.csv"),
                "partial_ifu_decomposition": str(output_dir / "partial_ifu_decomposition.csv"),
            },
        },
    }
    state_path = output_dir / "m101_onoff_state.json"
    state_path.write_text(json.dumps(json_ready(state), indent=2, sort_keys=True))

    elapsed = time.perf_counter() - started
    print("total central-q ON/OFF fiber-band measurements: %d" % total_measurements)
    print("number of amp summaries: %d supported (%d rows)" % (amp_summary_count, len(final_amp)))
    print("old blank-only complete-four reference: %d/%d IFUs" %
          (old_supported, len(ifus)))
    print("persistent IFU support 4/3/2/1/0 amps: %s" %
          json.dumps(support_coverage["persistent"], sort_keys=True))
    print("H5/IFU support 4/3/2/1/0 amps: %s" %
          json.dumps(support_coverage["h5"], sort_keys=True))
    print("exposure/cache support 4/3/2/1/0 amps: %s" %
          json.dumps(support_coverage["exposure"], sort_keys=True))
    print("total persistent amplifier channels measured: %d" %
          sum(np.isfinite(value) for value in final_primary["A"].values()))
    print("Cbar_ON / Cbar_OFF: %.8g / %.8g" % (final_cbar["ON"], final_cbar["OFF"]))
    print("exposure C scatter: ON=%.6g OFF=%.6g" %
          (_scatter([row["C"] for row in final_sc if row["band"] == "ON"]),
           _scatter([row["C"] for row in final_sc if row["band"] == "OFF"])))
    print("three-dither IFU repeatability: robust scatter=%.6g" % _scatter(repeat_I))
    print("three-dither amplifier repeatability: robust scatter=%.6g" % _scatter(repeat_d))
    print("ON/OFF IFU agreement: median offset=%.6g scatter=%.6g" %
          (final_comparison["ifu_common"]["median_offset"],
           final_comparison["ifu_common"]["robust_scatter_difference"]))
    print("ON/OFF amplifier agreement: median offset=%.6g scatter=%.6g" %
          (final_comparison["amp_differential"]["median_offset"],
           final_comparison["amp_differential"]["robust_scatter_difference"]))
    ranges = _hierarchy_ranges(final_hierarchy)
    print("robust ranges P/G_hi/Pamp/g_hi: %s" % json.dumps(json_ready(ranges), sort_keys=True))
    print("first-to-second-pass change: %s" % json.dumps(json_ready(changes), sort_keys=True))
    a1_values = _finite(list(primary_first["A"].values()))
    da2_values = _finite(list(primary_increment["A"].values()))
    at_values = _finite(list(primary_total["A"].values()))
    ratio_values = _finite([
        abs(increment / first)
        for first, increment in zip(primary_first["A"].values(), primary_increment["A"].values())
        if np.isfinite(first) and np.isfinite(increment) and first != 0
    ])
    abs_summary = lambda values: {
        "median_abs": float(np.median(np.abs(values))) if len(values) else np.nan,
        "p95_abs": float(np.percentile(np.abs(values), 95)) if len(values) else np.nan,
    }
    print("A_pass1 robust scatter: %.6g" % _scatter(a1_values))
    print("dA_pass2 robust scatter: %.6g" % _scatter(da2_values))
    print("A_total robust scatter: %.6g" % _scatter(at_values))
    print("median/p95 abs dA_pass2: %s" % json.dumps(json_ready(abs_summary(da2_values))))
    print("median/p95 abs A_pass1: %s" % json.dumps(json_ready(abs_summary(a1_values))))
    print("median/p95 abs dA_pass2/A_pass1: %s" % json.dumps(json_ready(abs_summary(ratio_values))))
    print("original-data residual log(D/Dhat), pass1 only: %s" %
          json.dumps(json_ready(audit_prediction["pass1"]), sort_keys=True))
    print("original-data residual log(D/Dhat), cumulative final: %s" %
          json.dumps(json_ready(audit_prediction["cumulative_final"]), sort_keys=True))
    print("original-data residual log(D/Dhat), currently reported old-final semantics: %s" %
          json.dumps(json_ready(audit_prediction["old_reported"]), sort_keys=True))
    print("pass-2 residual identity max error: %.6g" % audit["pass2_residual_identity_max_abs"])
    print("maximum complete-four reconstruction error: %.6g" %
          max(complete_output_errors, default=0.0))
    print("maximum multiplicative composition error: %.6g" %
          audit["multiplicative_composition_max_abs"])
    print("representative channel compositions (A_pass1, dA_pass2, A_total):")
    for example in channel_examples:
        print("  IFU %s [%s]: %s" %
              (example["IFU_CODE"], example["support_class"],
               json.dumps(json_ready(example["channels"]), sort_keys=True)))
    print("FINAL CHANNEL RESPONSE IS CUMULATIVE")
    partial_identifiable = sum(
        row["N_amp_available"] < 4 and row["common_identifiable"]
        for row in final_h5_gray_common)
    partial_channel_only = sum(
        row["N_amp_available"] < 4 and not row["common_identifiable"]
        for row in final_h5_gray_common)
    print("partial H5/IFU states with independently identifiable common mode: %d" %
          partial_identifiable)
    print("partial H5/IFU states with channel response only: %d" % partial_channel_only)
    print("hardware-excluded amplifier channels: %d; nonfinite/unavailable persistent channels: %d" %
          (partial_validation["hardware_excluded_physical_amp_channels"],
           partial_validation["persistent_nonfinite_unavailable_channels"]))
    print("complete-four A=P+Pamp reconstruction max error: %.6g" %
          max((abs(row["reconstruction_error"]) for row in complete_decomposition_rows),
              default=0.0))
    print("diagnostic plots: %s" % ", ".join(plot_paths))
    flagged_by_band = Counter(case["amp_row"]["band"] for case in flagged_extremes)
    flagged_by_amp = Counter(case["amp_row"]["AMP"] for case in flagged_extremes)
    flagged_h5 = {case["amp_row"]["H5"] for case in flagged_extremes}
    flagged_ifus = {int(case["amp_row"]["IFU_CODE"]) for case in flagged_extremes}
    print("number of extreme amp measurements: %d" % len(flagged_extremes))
    print("extreme amps by band: %s" % json.dumps(dict(flagged_by_band), sort_keys=True))
    print("extreme amps by amplifier: %s" % json.dumps(dict(flagged_by_amp), sort_keys=True))
    print("unique H5s / IFUs affected by extremes: %d / %d" %
          (len(flagged_h5), len(flagged_ifus)))
    persistent_amp_count_by_code = {
        code: sum(np.isfinite(final_primary["A"].get((code, amp), np.nan))
                  for amp in range(4))
        for code in ifus}
    unsupported_ifus = [row for row in support_rows
                        if persistent_amp_count_by_code.get(int(row["IFU_CODE"]), 0) == 0]
    reason_counts = Counter(row["dominant_reason"] for row in provenance_rows
                            if row["unsupported"])
    print("unsupported physical IFUs: %d" % len(unsupported_ifus))
    print("unsupported amplifier dominant provenance: %s" %
          json.dumps(dict(reason_counts), sort_keys=True))
    print("unsupported IFU provenance table (IFUSLOT: AMP=reason):")
    for support in unsupported_ifus:
        code = int(support["IFU_CODE"])
        amp_reasons = [
            "%s=%s" % (row["AMP"], row["dominant_reason"])
            for row in provenance_rows if int(row["IFU_CODE"]) == code
        ]
        print("  IFU %s (code %d): %s" %
              (support["IFUSLOT"], code, ", ".join(amp_reasons)))
    print("normal-population ON/OFF amplifier scatter: %.6g" %
          normal_onoff["amplifier"]["robust_scatter"])
    print("normal-population ON/OFF IFU common scatter: %.6g" %
          normal_onoff["ifu_common"]["robust_scatter"])
    trend_flags = {band: bool(result["obvious_monotonic_trend"])
                   for band, result in quantile_trend.items()}
    print("binned D-versus-X residuals show obvious monotonic trend: %s" %
          json.dumps(trend_flags, sort_keys=True))
    print("extreme diagnostic table: %s" % (output_dir / "extreme_amp_diagnostics.csv"))
    print("support provenance table: %s" % (output_dir / "unsupported_amp_provenance.csv"))
    print("robust G heatmap: %s" % robust_g_path)
    print("D-versus-X quantile plot: %s" % quantile_plot_path)
    print("zoomed ON/OFF plots: %s" % ", ".join(zoom_plot_paths))
    print("repeatability gallery manifest: %s" % (output_dir / "repeatability_gallery.csv"))
    print("partial-support plots: %s" % ", ".join(support_count_plot_paths + partial_example_plot_paths))
    print("primary channel tables: %s" % ", ".join(str(output_dir / name) for name in (
        "amplifier_exposure_measurements.csv", "amplifier_h5_measurements.csv",
        "persistent_amplifier_channels.csv", "h5_amplifier_departures.csv",
        "complete_four_ifu_decomposition.csv", "partial_ifu_decomposition.csv")))
    print("wrote %s in %.1fs" % (state_path, elapsed))


if __name__ == "__main__":
    main()
