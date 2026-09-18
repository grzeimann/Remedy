#!/usr/bin/env python3
"""Visualize full-resolution blank-fiber residuals after frozen gray M.

This is a read-only diagnostic.  It uses the validated native reconstruction
from :mod:`m101_native_data`, the cumulative-final response written by the
ON/OFF initializer, and the existing exact seven-band response arrays.  It
does not fit a response, estimate an additive correction, or update any
calibration state.

The primary residual is

    E(lambda) = D(lambda) - M * S(lambda)

where ``S`` is one robust full-resolution sky spectrum per H5/exposure,
estimated from classified blank fibers with 40 <= q <= 75.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import shlex
import sys
import time
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import m101_blank_fibers
import m101_native_data as native
from m101_calibration_utils import (
    collapse_many,
    robust_location,
    robust_scatter,
)
from m101_hardware_exclusions import HARDWARE_EXCLUSIONS


WAVE = np.asarray(native.WAVE, dtype=float)
WAVE_MIN = 3470.0
WAVE_MAX = 5540.0
SCIENCE_MASK = (WAVE >= WAVE_MIN) & (WAVE <= WAVE_MAX)
Q_MIN = 40
Q_MAX = 75
LEGACY_Q_MAX = 70
EXPOSURES = (1, 2, 3)
AMP_ORDER = ("LL", "LU", "RL", "RU")
BANDS = ("HIGH1", "LOW1", "HIGH2", "LOW2", "HIGH3", "ON", "OFF")
MIN_SKY_FIBERS = 1
MATCH_RTOL = 2.0e-4
MATCH_ATOL = 5.0e-5
DETAIL_MAX_FIBERS = 140
GALLERY_DEFAULT = 16


def _finite(values):
    values = np.asarray(values, dtype=float)
    return values[np.isfinite(values)]


def _json_ready(value):
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_json_ready(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _safe_name(value):
    text = str(value)
    return "".join(char if char.isalnum() or char in "._-" else "_"
                   for char in text).strip("_")


def _identity_text(identity):
    specid, ifuslot, ifuid, amp = identity
    return "%d_%d_%d_%s" % (int(specid), int(ifuslot), int(ifuid), amp)


def _identity_label(identity):
    specid, ifuslot, ifuid, amp = identity
    return "SPECID=%d IFUSLOT=%d IFUID=%d AMP=%s" % (
        int(specid), int(ifuslot), int(ifuid), amp)


def _physical_key(values):
    return tuple(int(value) for value in values)


def _file_identity(path):
    path = Path(path).expanduser().resolve()
    if not path.exists():
        return {"filename": path.name, "full_path": str(path), "exists": False}
    stat = path.stat()
    return {"filename": path.name, "full_path": str(path), "exists": True,
            "file_size": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns)}


def _write_rows(path, rows):
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
                    value = json.dumps(_json_ready(value), separators=(",", ":"))
                clean[field] = value
            writer.writerow(clean)


def _parse_state_key(value):
    try:
        parsed = ast.literal_eval(value)
    except (SyntaxError, ValueError):
        return None
    return parsed if isinstance(parsed, tuple) else None


def _resolve_value(value):
    if isinstance(value, dict):
        value = value.get("full_path", value.get("path", value.get("filename")))
    if value is None:
        return None
    return str(Path(value).expanduser().resolve())


def _provenance_candidates():
    return (
        Path("m101_stage1/m101_stage1_state.json"),
        Path("m101_band_initializer/m101_band_initial_state.json"),
        Path("m101_joint_calibration/m101_calibration_joint_product.json"),
    )


def _default_input(name):
    for candidate in _provenance_candidates():
        if not candidate.exists():
            continue
        try:
            data = json.loads(candidate.read_text())
        except (OSError, ValueError):
            continue
        value = _resolve_value(data.get("provenance", {}).get(name))
        if value and Path(value).exists():
            return value
    return None


def _load_state(state_path):
    state_path = Path(state_path).expanduser().resolve()
    state = json.loads(state_path.read_text())
    hierarchy = state.get("hierarchy", {})
    cumulative = hierarchy.get("cumulative_final", {})
    source = cumulative.get("R_total_h_i_a", {})
    if not source:
        raise ValueError("state has no hierarchy.cumulative_final.R_total_h_i_a")
    response = {}
    for encoded, value in source.items():
        key = _parse_state_key(encoded)
        if key is None or len(key) != 3:
            continue
        h5, code, amp = key
        if value is None or not np.isfinite(value):
            continue
        response[(str(h5), int(code), str(amp))] = float(value)
    if not response:
        raise ValueError("cumulative-final R_total_h_i_a is empty")
    return state, response


def _verify_response_composition(state):
    hierarchy = state.get("hierarchy", {})
    cumulative = hierarchy.get("cumulative_final", {})
    total = cumulative.get("R_total_h_i_a", {})
    pass1 = hierarchy.get("R_pass1_h_i_a", {})
    increment = hierarchy.get("dR_pass2_h_i_a", {})
    differences = []
    for key, value in total.items():
        if key not in pass1 or key not in increment:
            continue
        if value is None or pass1[key] is None or increment[key] is None:
            continue
        differences.append(float(pass1[key]) + float(increment[key]) - float(value))
    differences = np.asarray(differences, dtype=float)
    max_abs = float(np.max(np.abs(differences))) if differences.size else np.nan
    composition_pass = bool(differences.size and np.allclose(
        differences, 0.0, rtol=0.0, atol=2.0e-12))
    top = hierarchy.get("R_total_h_i_a", {})
    top_differences = []
    for key, value in total.items():
        if key in top and value is not None and top[key] is not None:
            top_differences.append(float(top[key]) - float(value))
    top_differences = np.asarray(top_differences, dtype=float)
    top_max_abs = (float(np.max(np.abs(top_differences)))
                   if top_differences.size else np.nan)
    return {
        "source_field": "hierarchy.cumulative_final.R_total_h_i_a",
        "composition": "R_pass1_h_i_a + dR_pass2_h_i_a",
        "n_composed": int(differences.size),
        "max_abs_composition_difference": max_abs,
        "composition_matches_stored_cumulative": composition_pass,
        "max_abs_top_level_difference": top_max_abs,
        "top_level_matches_stored_cumulative": bool(
            top_differences.size and np.allclose(
                top_differences, 0.0, rtol=0.0, atol=2.0e-12)),
    }


def _load_cache_identity_map(cache_path, cache_manifest, selected_h5):
    """Map native physical IFU identities to the cache's IFU_CODE.

    The native loader intentionally exposes physical identities but not the
    initializer's compact IFU code.  The cache is used only for this frozen
    identity mapping; spectra and all diagnostic values come from the native
    loader.
    """
    cache_path = Path(cache_path).expanduser().resolve()
    wanted = {Path(path).name for path in selected_h5}
    with np.load(cache_path, allow_pickle=False) as archive:
        ifu = np.asarray(archive["ifu"])
        ifu_code = np.asarray(archive["ifu_code"])
        mapping = {}
        for item in cache_manifest.get("items", []):
            h5 = str(item["h5_name"])
            if h5 not in wanted or int(item["exposure"]) != 1:
                continue
            start, stop = int(item["start"]), int(item["stop"])
            pairs = np.unique(np.column_stack((ifu[start:stop], ifu_code[start:stop])), axis=0)
            for row in pairs:
                physical = _physical_key(row[:3])
                code = int(row[3])
                old = mapping.setdefault((h5, physical), code)
                if old != code:
                    raise ValueError("cache maps one physical IFU to multiple IFU_CODE values: %s %s" %
                                     (h5, physical))
    missing = []
    for h5 in wanted:
        if not any(key[0] == h5 for key in mapping):
            missing.append(h5)
    if missing:
        raise ValueError("cache has no physical IFU mapping for %s" % sorted(missing))
    return mapping


def _load_band_manifest(cache_path):
    manifest_path = Path(cache_path).expanduser().resolve()
    manifest_path = manifest_path.with_suffix(manifest_path.suffix + ".json")
    manifest = json.loads(manifest_path.read_text())
    if tuple(manifest.get("band_order", ())) != BANDS:
        raise ValueError("band cache does not contain the expected seven bands")
    return manifest, manifest_path


def _resolve_h5_paths(args, manifest):
    by_name = {}
    for row in manifest.get("input_h5", []):
        name = Path(row["filename"]).name
        by_name[name] = _resolve_value(row.get("full_path", row.get("filename")))
    if args.h5:
        raw = args.h5
        paths = []
        for value in raw:
            name = Path(value).name
            path = str(Path(value).expanduser().resolve()) if Path(value).exists() else by_name.get(name)
            if path is None:
                raise FileNotFoundError("H5 is not in the band-cache provenance: %s" % value)
            if Path(path).name != name:
                raise ValueError("H5 basename mismatch for %s" % value)
            paths.append(path)
    else:
        paths = [path for path in by_name.values() if path and Path(path).exists()]
        missing = sorted(name for name, path in by_name.items()
                         if path is None or not Path(path).exists())
        if missing:
            raise FileNotFoundError("band-cache H5 files are unavailable: %s; pass --h5 for an alternate path" %
                                    ", ".join(missing))
    return native.discover_h5(paths, development=True)


def _nanpercentile(values, percentile):
    values = np.asarray(values, dtype=float)
    output = np.full(values.shape[1], np.nan, dtype=float)
    good = np.any(np.isfinite(values), axis=0)
    if np.any(good):
        with np.errstate(all="ignore"):
            output[good] = np.nanpercentile(values[:, good], percentile, axis=0)
    return output


def _safe_location(values, axis=None):
    values = np.asarray(values, dtype=float)
    if axis == 0 and values.ndim == 2:
        supported = np.any(np.isfinite(values), axis=0)
        result = np.full(values.shape[1], np.nan, dtype=float)
        if np.any(supported):
            with np.errstate(all="ignore"):
                result[supported] = robust_location(values[:, supported], axis=0)
        return result
    with np.errstate(all="ignore"):
        result = robust_location(values, axis=axis)
    result = np.asarray(result, dtype=float)
    if axis is None:
        return float(result) if np.isfinite(result) else np.nan
    if np.any(~np.isfinite(result)):
        fallback = np.nanmedian(values, axis=axis)
        result = np.where(np.isfinite(result), result, fallback)
    return result


def _safe_scatter(values):
    values = _finite(values)
    if values.size < 2:
        return np.nan
    with np.errstate(all="ignore"):
        value = robust_scatter(values)
    return float(value) if np.isfinite(value) else np.nan


def _amp_residual_summary(values):
    values = _finite(values)
    if not values.size:
        return {"rms": np.nan, "median": np.nan, "p01": np.nan, "p99": np.nan}
    return {
        "rms": _safe_scatter(values),
        "median": float(np.median(values)),
        "p01": float(np.percentile(values, 1.0)),
        "p99": float(np.percentile(values, 99.0)),
    }


def _band_effective_wavelengths(responses):
    responses = np.asarray(responses, dtype=float)
    totals = np.sum(responses, axis=1)
    return np.divide(responses @ WAVE, totals,
                     out=np.full(len(BANDS), np.nan), where=totals != 0)


def _band_definition_provenance(responses, effective):
    rows = {}
    for name, response, wavelength in zip(BANDS, responses, effective):
        use = np.isfinite(response) & (response != 0)
        rows[name] = {
            "effective_wavelength_A": float(wavelength),
            "sample_indices": np.flatnonzero(use).astype(int).tolist(),
            "sample_wavelengths_A": WAVE[use].astype(float).tolist(),
            "weights": response[use].astype(float).tolist(),
            "total_weight": float(np.sum(response[use])),
            "normalization": "weighted finite-response mean; exact m101_calibration_utils.collapse",
        }
    return rows


def _finite_row_mask(item, codes, response):
    total = np.asarray(item.total, dtype=float)
    logm = np.full(item.row_index.size, np.nan, dtype=float)
    for index, (code, amp) in enumerate(zip(codes, item.amp)):
        value = response.get((item.h5_name, int(code), str(amp)), np.nan)
        logm[index] = value
    base = (np.asarray(item.blank_classified, dtype=bool)
            & np.asarray(item.blank_valid, dtype=bool)
            & ~np.asarray(item.hardware_bad, dtype=bool)
            & np.isfinite(logm)
            & (np.asarray(item.q) >= Q_MIN)
            & (np.asarray(item.q) <= Q_MAX))
    return total, logm, base


def _sky_and_residuals(item, codes, response, q_max):
    total, logm, base = _finite_row_mask(item, codes, response)
    base &= np.asarray(item.q) <= q_max
    multiplier = np.exp(logm)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        corrected = total / multiplier[:, None]
    corrected[~np.isfinite(corrected)] = np.nan
    supported = base[:, None] & np.isfinite(corrected)
    n_sky = np.sum(supported, axis=0).astype(np.int64)
    sky = _safe_location(np.where(supported, corrected, np.nan), axis=0)
    sky[n_sky < MIN_SKY_FIBERS] = np.nan
    residual = total - multiplier[:, None] * sky[None, :]
    epsilon = corrected - sky[None, :]
    residual[~(base[:, None] & np.isfinite(total) & np.isfinite(sky)[None, :])] = np.nan
    epsilon[~(base[:, None] & np.isfinite(corrected) & np.isfinite(sky)[None, :])] = np.nan
    denominator = multiplier[:, None] * sky[None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        delta = total / denominator - 1.0
    delta[~(base[:, None] & np.isfinite(total) & np.isfinite(denominator)
            & (denominator != 0))] = np.nan
    return {
        "total": total,
        "logm": logm,
        "multiplier": multiplier,
        "base": base,
        "corrected": corrected,
        "sky": sky,
        "n_sky": n_sky,
        "residual": residual,
        "epsilon": epsilon,
        "delta": delta,
        "codes": np.asarray(codes, dtype=int),
    }


def _group_record(item, calculation, physical, amp, responses, effective, q_max,
                  keep_individual=False):
    physical_mask = np.all(np.asarray(item.ifu, dtype=int) == np.asarray(physical), axis=1)
    group = physical_mask & (np.asarray(item.amp, dtype=object) == amp)
    valid = group[:, None] & np.isfinite(calculation["residual"])
    residual_values = np.where(valid, calculation["residual"], np.nan)
    epsilon_values = np.where(group[:, None] & np.isfinite(calculation["epsilon"]),
                              calculation["epsilon"], np.nan)
    delta_values = np.where(group[:, None] & np.isfinite(calculation["delta"]),
                            calculation["delta"], np.nan)
    group_residual_values = residual_values[group]
    group_epsilon_values = epsilon_values[group]
    group_delta_values = delta_values[group]
    n_blank = int(np.sum(calculation["base"] & group))
    e_amp = _safe_location(group_residual_values, axis=0)
    # M is a positive scalar for one physical amplifier.  The primary
    # relation epsilon = E/M is therefore the same gray-scaled robust curve,
    # without a second expensive biweight pass over every wavelength.
    group_logm = _finite(calculation["logm"][group])
    group_multiplier = float(np.exp(group_logm[0])) if group_logm.size else np.nan
    epsilon_amp = (e_amp / group_multiplier if np.isfinite(group_multiplier)
                   else _safe_location(group_epsilon_values, axis=0))
    delta_amp = _safe_location(group_delta_values, axis=0)
    n = np.sum(np.isfinite(group_residual_values), axis=0).astype(np.int32)
    p16 = _nanpercentile(group_residual_values, 16.0)
    p84 = _nanpercentile(group_residual_values, 84.0)
    band_values, _ = collapse_many(group_residual_values, responses)
    band_amp = _safe_location(band_values, axis=0)
    band_n = np.sum(np.isfinite(band_values), axis=0).astype(np.int32)
    return {
        "H5": item.h5_name,
        "exposure": int(item.exposure),
        "physical": tuple(int(value) for value in physical),
        "IFU_CODE": int(np.unique(calculation["codes"][physical_mask])[0]) if np.any(physical_mask) else np.nan,
        "AMP": amp,
        "q_max": int(q_max),
        "blank_count": n_blank,
        "E": e_amp,
        "epsilon": epsilon_amp,
        "delta": delta_amp,
        "p16": p16,
        "p84": p84,
        "n": n,
        "band": band_amp,
        "band_n": band_n,
        "band_effective": effective.copy(),
        # Full fiber-by-wavelength arrays are retained only for selected
        # detailed-gallery targets. Keeping these for every amplifier is the
        # high-memory failure mode this diagnostic is designed to avoid.
        "individual_residual": residual_values if keep_individual else None,
        "individual_valid": valid if keep_individual else None,
    }


def _group_band_record(item, calculation, physical, amp, responses, effective):
    """Collapse legacy-q residual spectra only; no unused spectral curves."""
    physical_mask = np.all(np.asarray(item.ifu, dtype=int) == np.asarray(physical), axis=1)
    group = physical_mask & (np.asarray(item.amp, dtype=object) == amp)
    values = calculation["residual"][group]
    band_values, _ = collapse_many(values, responses)
    band_amp = _safe_location(band_values, axis=0)
    return {
        "H5": item.h5_name,
        "exposure": int(item.exposure),
        "physical": tuple(int(value) for value in physical),
        "IFU_CODE": int(np.unique(calculation["codes"][physical_mask])[0]) if np.any(physical_mask) else np.nan,
        "AMP": amp,
        "band": band_amp,
        "band_effective": effective.copy(),
    }


def _physical_groups(item, calculation):
    result = {}
    ifu = np.asarray(item.ifu, dtype=int)
    for physical in np.unique(ifu, axis=0):
        physical = _physical_key(physical)
        row_mask = np.all(ifu == np.asarray(physical), axis=1)
        for amp in AMP_ORDER:
            # Do not spend robust-statistics time on response-unavailable
            # channels.  They remain absent from the usable-amplifier set;
            # sibling channels in a partial IFU are still retained.
            usable = (row_mask & (np.asarray(item.amp, dtype=object) == amp)
                      & np.asarray(calculation["base"], dtype=bool))
            if np.any(usable):
                result[(physical, amp)] = True
    return result


def _make_records(item, codes, response, responses, effective):
    calculation75 = _sky_and_residuals(item, codes, response, Q_MAX)
    calculation70 = _sky_and_residuals(item, codes, response, LEGACY_Q_MAX)
    physical_groups = _physical_groups(item, calculation75)
    records75, records70 = {}, {}
    for (physical, amp) in physical_groups:
        records75[(physical, amp)] = _group_record(
            item, calculation75, physical, amp, responses, effective, Q_MAX)
        records70[(physical, amp)] = _group_band_record(
            item, calculation70, physical, amp, responses, effective)
    return calculation75, calculation70, records75, records70


def _load_previous_tables(previous_dir):
    previous_dir = Path(previous_dir).expanduser().resolve()
    amp_path = previous_dir / "blank_amp_residuals.csv"
    sky_path = previous_dir / "blank_sky_by_exposure_band.csv"
    if not amp_path.exists():
        raise FileNotFoundError("previous amplifier table not found: %s" % amp_path)
    if not sky_path.exists():
        raise FileNotFoundError("previous sky table not found: %s" % sky_path)
    amp_rows = {}
    with amp_path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            key = (row["H5"], int(row["exposure"]), int(row["IFU_CODE"]),
                   row["AMP"], row["band"])
            amp_rows[key] = row
    sky_rows = {}
    with sky_path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            key = (row["H5"], int(row["exposure"]), row["band"])
            sky_rows[key] = row
    return amp_rows, sky_rows, amp_path, sky_path


def _reproduction_row(old, new, old_field, new_field, fields):
    old_value = float(old[old_field]) if old is not None and old.get(old_field, "") not in ("", None) else np.nan
    new_value = float(new) if new is not None and np.isfinite(new) else np.nan
    difference = new_value - old_value if np.isfinite(old_value) and np.isfinite(new_value) else np.nan
    status = ("match" if np.isfinite(old_value) and np.isfinite(new_value)
              and np.isclose(new_value, old_value, rtol=MATCH_RTOL, atol=MATCH_ATOL)
              else "mismatch" if np.isfinite(old_value) and np.isfinite(new_value)
              else "missing")
    output = dict(fields)
    output.update({"old_value": old_value, "new_value": new_value,
                   "difference": difference, "absolute_difference": abs(difference)
                   if np.isfinite(difference) else np.nan,
                   "match_status": status})
    return output


def _add_band_reproduction_rows(item, records70, previous_amp, output_rows):
    physical_to_identity = {}
    for (physical, amp), record in records70.items():
        identity = tuple(physical) + (amp,)
        for band_index, band in enumerate(BANDS):
            key = (item.h5_name, int(item.exposure), int(record["IFU_CODE"]), amp, band)
            old = previous_amp.get(key)
            if old is None:
                continue
            output_rows.append(_reproduction_row(
                old, record["band"][band_index], "Ares_location", record["band"][band_index],
                {"H5": item.h5_name, "exposure": int(item.exposure),
                 "SPECID": physical[0], "IFUSLOT": physical[1], "IFUID": physical[2],
                 "IFU_CODE": int(record["IFU_CODE"]), "AMP": amp, "band": band,
                 "previous_Ares_location": float(old["Ares_location"]),
                 "new_spectral_collapse_Ares": float(record["band"][band_index])}))
        physical_to_identity[physical] = identity


def _band_overlay_rows(item, records75, output_rows):
    for (physical, amp), record in sorted(records75.items()):
        for band_index, band in enumerate(BANDS):
            value = record["band"][band_index]
            output_rows.append({
                "H5": item.h5_name,
                "exposure": int(item.exposure),
                "SPECID": int(physical[0]),
                "IFUSLOT": int(physical[1]),
                "IFUID": int(physical[2]),
                "IFU_CODE": int(record["IFU_CODE"]),
                "IFU_CODE_label": int(record["IFU_CODE"]),
                "AMP": amp,
                "band": band,
                "effective_wavelength_A": float(record["band_effective"][band_index]),
                "N_fibers": int(record["band_n"][band_index]),
                "band_residual_from_full_spectrum_q40_75": float(value)
                if np.isfinite(value) else np.nan,
                "units": "native calibrated/reconstructed flux",
            })


def _sky_reproduction_rows(item, calculation70, responses, effective, previous_sky, output_rows):
    sky_band, _ = collapse_many(calculation70["sky"][None, :], responses)
    n70 = int(np.sum(calculation70["base"]))
    for index, band in enumerate(BANDS):
        key = (item.h5_name, int(item.exposure), band)
        old = previous_sky.get(key)
        if old is None:
            continue
        row = _reproduction_row(
            old, sky_band[0, index], "S_after_M", sky_band[0, index],
            {"H5": item.h5_name, "exposure": int(item.exposure), "band": band,
             "effective_wavelength_A": float(effective[index]),
             "N_fibers_q40_70": n70,
             "previous_S_after_M": float(old["S_after_M"]),
             "new_spectral_collapse_S": float(sky_band[0, index])})
        output_rows.append(row)


def _summary_row(record):
    values = record["E"][SCIENCE_MASK]
    summary = _amp_residual_summary(values)
    return {
        "H5": record["H5"],
        "exposure": int(record["exposure"]),
        "SPECID": int(record["physical"][0]),
        "IFUSLOT": int(record["physical"][1]),
        "IFUID": int(record["physical"][2]),
        "IFU_CODE": int(record["IFU_CODE"]),
        "IFU_CODE_label": int(record["IFU_CODE"]),
        "AMP": record["AMP"],
        "N_blank_q40_75": int(record["blank_count"]),
        "residual_robust_rms_3560_5480": summary["rms"],
        "residual_median": summary["median"],
        "p01": summary["p01"],
        "p99": summary["p99"],
        "epsilon_robust_rms_3560_5480": _safe_scatter(record["epsilon"][SCIENCE_MASK]),
        "delta_robust_rms_3560_5480": _safe_scatter(record["delta"][SCIENCE_MASK]),
        "n_spectral_samples": int(np.sum(np.isfinite(record["E"][SCIENCE_MASK]))),
    }


def _record_by_amp(records):
    output = defaultdict(dict)
    for record in records:
        output[(record["physical"], record["AMP"])][int(record["exposure"])] = record
    return output


def _plot_limits(records, field="E"):
    values = []
    for record in records:
        values.append(_finite(record[field][SCIENCE_MASK]))
        values.append(_finite(record["band"]))
    values = _finite(np.concatenate([value for value in values if value.size])) if any(
        value.size for value in values) else np.empty(0)
    if not values.size:
        return (-1.0, 1.0)
    low, high = np.percentile(values, [1.0, 99.0])
    width = max(high - low, 1.0e-12)
    return (float(low - 0.12 * width), float(high + 0.12 * width))


def _plot_band_support(axis, responses):
    colors = ("#d9d9d9", "#c7e9c0", "#c6dbef", "#fdd0a2", "#dadaeb", "#fdae6b", "#9ecae1")
    for name, response, color in zip(BANDS, responses, colors):
        use = np.isfinite(response) & (response != 0) & SCIENCE_MASK
        if not np.any(use):
            continue
        indices = np.flatnonzero(use)
        # Preserve exact native support, including disjoint support if present.
        starts = [indices[0]]
        stops = []
        for left, right in zip(indices[:-1], indices[1:]):
            if right != left + 1:
                stops.append(left)
                starts.append(right)
        stops.append(indices[-1])
        for start, stop in zip(starts, stops):
            axis.axvspan(WAVE[start], WAVE[stop], color=color, alpha=0.08, lw=0)


def _plot_amp_spectrum(path, h5, records, responses, title_extra=""):
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(12, 5.8))
    colors = {1: "#1f77b4", 2: "#d62728", 3: "#2ca02c"}
    _plot_band_support(axis, responses)
    plotted = False
    for exposure in EXPOSURES:
        record = next((row for row in records if int(row["exposure"]) == exposure), None)
        if record is None:
            continue
        color = colors[exposure]
        axis.fill_between(WAVE[SCIENCE_MASK], record["p16"][SCIENCE_MASK],
                          record["p84"][SCIENCE_MASK], color=color, alpha=0.10,
                          linewidth=0)
        axis.plot(WAVE[SCIENCE_MASK], record["E"][SCIENCE_MASK], color=color,
                  lw=1.25, label="exposure %d" % exposure)
        band_values = record["band"]
        in_range = ((record["band_effective"] >= WAVE_MIN)
                    & (record["band_effective"] <= WAVE_MAX)
                    & np.isfinite(band_values))
        axis.plot(record["band_effective"][in_range], band_values[in_range],
                  "o", color=color, ms=4.2, mec="black", mew=0.35,
                  label="exp %d exact bands" % exposure)
        plotted = True
    if records:
        for record in records[:1]:
            high1 = np.isfinite(record["band"][0])
            if high1:
                axis.plot([WAVE_MIN], [record["band"][0]], marker="<", color="black",
                          ms=5, clip_on=False)
                axis.annotate("HIGH1 %.0f A (outside plotted range)" % record["band_effective"][0],
                              xy=(WAVE_MIN, record["band"][0]), xytext=(5, 5),
                              textcoords="offset points", fontsize=7, color="black")
    axis.axhline(0.0, color="black", lw=0.75, zorder=0)
    axis.set_xlim(WAVE_MIN, WAVE_MAX)
    axis.set_ylim(*_plot_limits(records))
    if records:
        first = records[0]
        identity = first["physical"] + (first["AMP"],)
        blank = ", ".join("e%d:%d" % (row["exposure"], row["blank_count"])
                          for row in sorted(records, key=lambda value: value["exposure"]))
        axis.set_title("%s; %s; blank q40--75 (%s)%s" %
                       (h5, _identity_label(identity), blank, title_extra))
    axis.set_xlabel("wavelength [Angstrom]")
    axis.set_ylabel("E = D - M*S [native calibrated/reconstructed flux units]")
    axis.legend(loc="upper right", fontsize=7, ncol=2)
    axis.grid(alpha=0.16)
    figure.tight_layout()
    figure.savefig(path, dpi=115, bbox_inches="tight")
    plt.close(figure)


def _plot_ifu_panels(path, h5, physical, by_amp, responses):
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
    axes = axes.ravel()
    available = [by_amp.get((physical, amp), []) for amp in AMP_ORDER]
    limits = _plot_limits([record for records in available for record in records])
    colors = {1: "#1f77b4", 2: "#d62728", 3: "#2ca02c"}
    for axis, amp, records in zip(axes, AMP_ORDER, available):
        if not records:
            axis.text(0.5, 0.5, "unavailable", ha="center", va="center",
                      transform=axis.transAxes)
            axis.set_title(amp)
            axis.axhline(0, color="black", lw=.6)
            continue
        _plot_band_support(axis, responses)
        for exposure in EXPOSURES:
            record = next((row for row in records if row["exposure"] == exposure), None)
            if record is None:
                continue
            color = colors[exposure]
            axis.plot(WAVE[SCIENCE_MASK], record["E"][SCIENCE_MASK], color=color,
                      lw=0.9, label="e%d" % exposure)
            use = ((record["band_effective"] >= WAVE_MIN)
                   & (record["band_effective"] <= WAVE_MAX)
                   & np.isfinite(record["band"]))
            axis.plot(record["band_effective"][use], record["band"][use], "o",
                      ms=2.8, color=color, mec="black", mew=.25)
        axis.axhline(0, color="black", lw=.6)
        axis.set_ylim(*limits)
        axis.set_title("%s (blank e1/e2/e3: %s)" %
                       (amp, "/".join(str(row["blank_count"]) for row in records)))
        axis.grid(alpha=.13)
    for axis in axes[2:]:
        axis.set_xlabel("wavelength [Angstrom]")
    for axis in axes[::2]:
        axis.set_ylabel("E [native flux]")
    if any(available):
        axes[0].legend(fontsize=7, loc="upper right")
    figure.suptitle("%s; IFU SPECID=%d IFUSLOT=%d IFUID=%d; q=40--75" %
                    (h5, physical[0], physical[1], physical[2]))
    figure.tight_layout()
    figure.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(figure)


def _select_gallery(summary_rows, requested, max_examples=GALLERY_DEFAULT):
    grouped = defaultdict(list)
    for row in summary_rows:
        key = (row["H5"], int(row["SPECID"]), int(row["IFUSLOT"]),
               int(row["IFUID"]), row["AMP"])
        grouped[key].append(row)
    metrics = []
    for key, rows in grouped.items():
        rms = _finite([row["residual_robust_rms_3560_5480"] for row in rows])
        if rms.size:
            metrics.append((key, float(np.median(rms))))
    metrics.sort(key=lambda value: (value[1], value[0]))
    chosen = []
    chosen_set = set()
    # Four H5 groups times four amplifier names gives a deterministic 16-row
    # core that spans observation epochs and readout channels.  The caller can
    # reduce this for a quicker gallery; explicit --amp requests are appended
    # below and are intentionally not limited by max_examples.
    h5_names = sorted({key[0] for key, _ in metrics})
    h5_groups = np.array_split(np.arange(len(h5_names)), min(4, len(h5_names))) if h5_names else []
    for group_index, group in enumerate(h5_groups):
        if len(chosen) >= max_examples:
            break
        if not len(group):
            continue
        h5_pool = set(h5_names[int(index)] for index in group)
        for amp_index, amp in enumerate(AMP_ORDER):
            if len(chosen) >= max_examples:
                break
            candidates = [(key, value) for key, value in metrics
                          if key[0] in h5_pool and key[4] == amp]
            if not candidates:
                continue
            target_fraction = (amp_index + 0.5) / len(AMP_ORDER)
            candidates.sort(key=lambda value: (value[1], value[0]))
            choice = candidates[min(len(candidates) - 1,
                                    int(target_fraction * len(candidates)))]
            if choice[0] not in chosen_set:
                chosen.append((choice[0], choice[1], "automatic_span"))
                chosen_set.add(choice[0])
    if metrics:
        for fraction in np.linspace(0.0, 1.0, 8):
            index = min(len(metrics) - 1, int(round(fraction * (len(metrics) - 1))))
            key, value = metrics[index]
            if key not in chosen_set and len(chosen) < max_examples:
                chosen.append((key, value, "automatic_rms_quantile"))
                chosen_set.add(key)
    for identity in requested:
        specid, ifuslot, ifuid, amp = identity
        found = [(key, value) for key, value in metrics
                 if key[1:] == (specid, ifuslot, ifuid, amp)]
        for key, value in found:
            if key not in chosen_set:
                chosen.append((key, value, "user_requested"))
                chosen_set.add(key)
    return chosen


def _plot_detail(path, h5, record_by_exp, responses):
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    colors = {1: "#1f77b4", 2: "#d62728", 3: "#2ca02c"}
    limits = _plot_limits(list(record_by_exp.values()))
    for axis, exposure in zip(axes, EXPOSURES):
        record = record_by_exp.get(exposure)
        if record is None:
            axis.text(0.5, .5, "unavailable", transform=axis.transAxes,
                      ha="center", va="center")
            continue
        _plot_band_support(axis, responses)
        residual = record["individual_residual"][:, SCIENCE_MASK]
        valid = record["individual_valid"][:, SCIENCE_MASK]
        finite_fibers = np.flatnonzero(np.any(valid, axis=1))
        if finite_fibers.size > DETAIL_MAX_FIBERS:
            picks = np.linspace(0, finite_fibers.size - 1,
                                DETAIL_MAX_FIBERS).round().astype(int)
            finite_fibers = finite_fibers[picks]
        for fiber_index in finite_fibers:
            values = residual[fiber_index].copy()
            values[~valid[fiber_index]] = np.nan
            axis.plot(WAVE[SCIENCE_MASK], values, color=colors[exposure],
                      alpha=.075, lw=.45)
        axis.fill_between(WAVE[SCIENCE_MASK], record["p16"][SCIENCE_MASK],
                          record["p84"][SCIENCE_MASK], color=colors[exposure],
                          alpha=.18, lw=0)
        axis.plot(WAVE[SCIENCE_MASK], record["E"][SCIENCE_MASK],
                  color=colors[exposure], lw=1.8, label="robust amplifier E")
        use = ((record["band_effective"] >= WAVE_MIN)
               & (record["band_effective"] <= WAVE_MAX)
               & np.isfinite(record["band"]))
        axis.plot(record["band_effective"][use], record["band"][use], "o",
                  color="black", ms=3.5, label="exact band collapse")
        axis.axhline(0, color="black", lw=.6)
        axis.set_ylim(*limits)
        axis.set_ylabel("e%d E" % exposure)
        axis.set_title("exposure %d; %d blank fibers shown of %d" %
                       (exposure, len(finite_fibers), record["blank_count"]), fontsize=9)
        axis.grid(alpha=.13)
    axes[-1].set_xlabel("wavelength [Angstrom]")
    if record_by_exp:
        record = next(iter(record_by_exp.values()))
        identity = record["physical"] + (record["AMP"],)
        figure.suptitle("Individual blank-fiber context; %s; %s; q=40--75" %
                        (h5, _identity_label(identity)))
        axes[0].legend(fontsize=7, loc="upper right")
    figure.tight_layout()
    figure.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(figure)


def _plot_heatmap(path, h5, record_by_exp, responses):
    path.parent.mkdir(parents=True, exist_ok=True)
    chunks = [_finite(record["individual_residual"][:, SCIENCE_MASK]).ravel()
              for record in record_by_exp.values()]
    chunks = [chunk for chunk in chunks if chunk.size]
    if not chunks:
        return
    all_values = np.concatenate(chunks)
    if not all_values.size:
        return
    limit = float(max(abs(np.percentile(all_values, 1.0)),
                     abs(np.percentile(all_values, 99.0))))
    limit = max(limit, 1.0e-12)
    figure, axes = plt.subplots(3, 1, figsize=(12, 8.5), sharex=True)
    cmap = plt.get_cmap("coolwarm").copy()
    cmap.set_bad("white")
    for axis, exposure in zip(axes, EXPOSURES):
        record = record_by_exp.get(exposure)
        if record is None:
            axis.text(.5, .5, "unavailable", transform=axis.transAxes,
                      ha="center", va="center")
            continue
        values = np.ma.masked_invalid(record["individual_residual"][:, SCIENCE_MASK])
        image = axis.imshow(values, origin="lower", aspect="auto", cmap=cmap,
                            vmin=-limit, vmax=limit,
                            extent=(WAVE_MIN, WAVE_MAX, 0.5, values.shape[0] + .5),
                            interpolation="nearest")
        axis.set_ylabel("e%d fiber row" % exposure)
        axis.set_title("exposure %d; blank q=40--75=%d" %
                       (exposure, record["blank_count"]), fontsize=9)
    axes[-1].set_xlabel("wavelength [Angstrom]")
    figure.colorbar(image, ax=axes, label="E [native calibrated/reconstructed flux units]")
    record = next(iter(record_by_exp.values()))
    identity = record["physical"] + (record["AMP"],)
    figure.suptitle("Blank-fiber E heat map; %s; %s" %
                    (h5, _identity_label(identity)))
    figure.tight_layout()
    figure.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(figure)


def _render_primary_h5(h5_name, records_by_amp, primary_dir, ifu_dir, responses):
    """Render one H5's compact primary records and release them afterward."""
    for (physical, amp), records in sorted(records_by_amp.items()):
        name = "%s_%s" % (_safe_name(Path(h5_name).stem),
                           _identity_text(physical + (amp,)))
        _plot_amp_spectrum(primary_dir / (name + ".png"), h5_name,
                           sorted(records, key=lambda value: value["exposure"]),
                           responses)
    by_ifu = defaultdict(dict)
    for (physical, amp), records in records_by_amp.items():
        by_ifu[physical][amp] = records
    for physical, amp_records in sorted(by_ifu.items()):
        _plot_ifu_panels(
            ifu_dir / ("%s_ifu_%d_%d_%d.png" %
                       (_safe_name(Path(h5_name).stem), physical[0], physical[1], physical[2])),
            h5_name, physical,
            {(physical, amp): records for amp, records in amp_records.items()},
            responses)


def _stats_from_rows(rows, difference_field="difference"):
    differences = _finite([row.get(difference_field, np.nan) for row in rows])
    absolute = np.abs(differences)
    return {
        "N_matching": int(differences.size),
        "median_difference": float(np.median(differences)) if differences.size else np.nan,
        "robust_difference_scatter": _safe_scatter(differences),
        "p95_absolute_difference": float(np.percentile(absolute, 95.0)) if absolute.size else np.nan,
        "maximum_absolute_difference": float(np.max(absolute)) if absolute.size else np.nan,
        "all_rows_within_tolerance": bool(all(
            np.isclose(row.get("new_value", np.nan), row.get("old_value", np.nan),
                       rtol=MATCH_RTOL, atol=MATCH_ATOL)
            for row in rows if row.get("match_status") != "missing")),
    }


def _parse_amp(value):
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 4 or parts[3].upper() not in AMP_ORDER:
        raise argparse.ArgumentTypeError(
            "--amp must be SPECID,IFUSLOT,IFUID,AMP with AMP in LL,LU,RL,RU")
    try:
        numbers = tuple(int(part) for part in parts[:3])
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--amp identity numbers must be integers") from exc
    return numbers + (parts[3].upper(),)


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", default="m101_onoff_hierarchy/m101_onoff_state.json")
    parser.add_argument("--band-cache", default="m101_band_initializer/m101_band_cache.npz")
    parser.add_argument("--h5", nargs="*", help="restrict processing/plots to H5 paths or basenames")
    parser.add_argument("--blank-file", default=None)
    parser.add_argument("--on-filter", default=None)
    parser.add_argument("--off-filter", default=None)
    parser.add_argument("--previous-output-dir", default="m101_blank_sky_residuals")
    parser.add_argument("--output-dir", default="m101_frozenM_spectral_residuals")
    parser.add_argument("--amp", action="append", type=_parse_amp,
                        help="additional physical amplifier SPECID,IFUSLOT,IFUID,AMP; repeatable")
    parser.add_argument("--heatmaps", action="store_true",
                        help="write secondary fiber x wavelength heat maps for the detailed gallery")
    parser.add_argument("--no-plots", action="store_true",
                        help="skip PNG generation for table/gate smoke tests")
    parser.add_argument("--force-plots", action="store_true",
                        help="render plots even when a consistency gate fails; plots remain flagged in state")
    parser.add_argument("--gallery-size", type=int, default=GALLERY_DEFAULT,
                        help="number of automatic amplifier/H5 examples to plot (default: 16); --amp requests are additional")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = _parse_args()
    if args.gallery_size <= 0:
        raise SystemExit("--gallery-size must be positive")
    started = time.perf_counter()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise SystemExit("output directory is nonempty; use --overwrite: %s" % output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    state_path = Path(args.state).expanduser().resolve()
    state, response = _load_state(state_path)
    composition = _verify_response_composition(state)
    if not composition["composition_matches_stored_cumulative"]:
        raise RuntimeError("pass-1 plus refinement response does not reproduce cumulative R_total")

    cache_path = Path(args.band_cache).expanduser().resolve()
    manifest, manifest_path = _load_band_manifest(cache_path)
    h5_paths = _resolve_h5_paths(args, manifest)
    h5_names = {path.name for path in h5_paths}
    response_identity_map = _load_cache_identity_map(cache_path, manifest, h5_paths)
    blank_file = args.blank_file or _default_input("blank_file")
    on_filter = args.on_filter or _default_input("on_filter")
    off_filter = args.off_filter or _default_input("off_filter")
    if not blank_file or not Path(blank_file).exists():
        raise FileNotFoundError("classified blank file is unavailable; pass --blank-file")
    if not on_filter or not Path(on_filter).exists():
        raise FileNotFoundError("ON filter is unavailable; pass --on-filter")
    if not off_filter or not Path(off_filter).exists():
        raise FileNotFoundError("OFF filter is unavailable; pass --off-filter")

    previous_dir = Path(args.previous_output_dir).expanduser().resolve()
    previous_amp, previous_sky, previous_amp_path, previous_sky_path = _load_previous_tables(previous_dir)
    blank_masks, blank_provenance = m101_blank_fibers.load(blank_file, h5_paths)
    on_filter_values = __import__("diagnose_m101_hierarchical").read_filter(on_filter)
    off_filter_values = __import__("diagnose_m101_hierarchical").read_filter(off_filter)
    responses, null_provenance = native._band_responses(on_filter_values, off_filter_values)
    effective = _band_effective_wavelengths(responses)
    if tuple(BANDS) != tuple(manifest.get("band_order", ())):
        raise RuntimeError("exact cache band order changed")
    expected_effective = 4500.0 + 1000.0 * np.asarray(manifest["band_effective_x_lambda"], dtype=float)
    effective_difference = float(np.max(np.abs(effective - expected_effective)))
    if effective_difference > 1.0e-8:
        raise RuntimeError("filter-derived band effective wavelengths disagree with the frozen band cache")

    print("[spectral] H5 files=%d; loading validated native spectra one H5 at a time" % len(h5_paths), flush=True)
    summary_rows = []
    overlay_rows = []
    reproduction_rows = []
    sky_reproduction_rows = []
    sky_epsilon_locations = []
    primary_dir = output_dir / "amplifier_spectra"
    ifu_dir = output_dir / "ifu_four_panel"
    for h5_index, h5_path in enumerate(h5_paths, 1):
        print("[spectral] %d/%d %s" % (h5_index, len(h5_paths), h5_path.name), flush=True)
        native_items, loader_provenance = native.load(
            [h5_path], blank_masks, on_filter, off_filter,
            include_native_errors=False, include_band_errors=False,
            collapse_band_indices=(),
            progress=lambda message: print("  " + message, flush=True))
        for item in sorted(native_items, key=lambda value: value.exposure):
            physical_codes = []
            for physical in np.asarray(item.ifu, dtype=int):
                key = (item.h5_name, _physical_key(physical))
                if key not in response_identity_map:
                    physical_codes.append(-1)
                else:
                    physical_codes.append(response_identity_map[key])
            calculation75, calculation70, records75, records70 = _make_records(
                item, physical_codes, response, responses, effective)
            sky_epsilon_locations.append(_safe_location(
                np.where(calculation75["base"][:, None] & np.isfinite(calculation75["epsilon"]),
                         calculation75["epsilon"], np.nan), axis=0))
            for record in records75.values():
                summary_rows.append(_summary_row(record))
            _band_overlay_rows(item, records75, overlay_rows)
            _add_band_reproduction_rows(item, records70, previous_amp, reproduction_rows)
            _sky_reproduction_rows(item, calculation70, responses, effective,
                                   previous_sky, sky_reproduction_rows)
        del native_items
        del item, calculation75, calculation70, records75, records70

    _write_rows(output_dir / "spectral_amp_residual_summary.csv", summary_rows)
    _write_rows(output_dir / "spectral_band_overlay.csv", overlay_rows)
    _write_rows(output_dir / "spectral_band_reproduction_q40_70.csv", reproduction_rows)
    _write_rows(output_dir / "spectral_sky_band_reproduction_q40_70.csv", sky_reproduction_rows)

    reproduction_stats = _stats_from_rows(reproduction_rows)
    sky_stats = _stats_from_rows(sky_reproduction_rows)
    epsilon_chunks = [_finite(values) for values in sky_epsilon_locations
                      if _finite(values).size]
    epsilon_locations = np.concatenate(epsilon_chunks) if epsilon_chunks else np.empty(0)
    max_epsilon_location = float(np.max(np.abs(epsilon_locations))) if epsilon_locations.size else np.nan
    preplot_gate_pass = bool(
        reproduction_stats["N_matching"] > 0
        and reproduction_stats["all_rows_within_tolerance"]
        and sky_stats["N_matching"] > 0
        and sky_stats["all_rows_within_tolerance"])
    plot_allowed = bool(not args.no_plots and (preplot_gate_pass or args.force_plots))
    if not args.no_plots and not preplot_gate_pass and not args.force_plots:
        print("[spectral] consistency gate failed; stopping before residual plots; "
              "see the two reproduction tables and listed discrepancy causes", flush=True)

    # Select a small deterministic amplifier gallery after all numerical
    # products are complete.  Numerical summaries still cover every usable
    # amplifier; only the rendered examples are culled.  Explicit --amp
    # requests are added to the automatic gallery.
    requested = args.amp or []
    gallery = _select_gallery(summary_rows, requested,
                               max_examples=args.gallery_size) if plot_allowed else []
    gallery_targets_by_h5 = defaultdict(set)
    for key, _, _ in gallery:
        h5, specid, ifuslot, ifuid, amp = key
        gallery_targets_by_h5[h5].add(((int(specid), int(ifuslot), int(ifuid)), amp))

    # Render primary plots in a streaming pass restricted to the selected
    # gallery.  This applies both with and without --force-plots, so a full
    # run never creates one primary plot per amplifier.
    if plot_allowed and gallery:
        for h5_path in h5_paths:
            h5_targets = gallery_targets_by_h5.get(h5_path.name, set())
            if not h5_targets:
                continue
            native_items, _ = native.load(
                [h5_path], blank_masks, on_filter, off_filter,
                include_native_errors=False, include_band_errors=False,
                collapse_band_indices=())
            records_by_amp = defaultdict(list)
            for item in sorted(native_items, key=lambda value: value.exposure):
                codes = [response_identity_map[(item.h5_name, _physical_key(physical))]
                         for physical in np.asarray(item.ifu, dtype=int)]
                calculation75 = _sky_and_residuals(item, codes, response, Q_MAX)
                for physical, amp in _physical_groups(item, calculation75):
                    if (physical, amp) not in h5_targets:
                        continue
                    records_by_amp[(physical, amp)].append(
                        _group_record(item, calculation75, physical, amp,
                                      responses, effective, Q_MAX))
            _render_primary_h5(h5_path.name, records_by_amp,
                               primary_dir, ifu_dir, responses)
            del native_items, records_by_amp, item, calculation75

    gallery_rows = []
    for key, rms, selection in gallery:
        h5, specid, ifuslot, ifuid, amp = key
        gallery_rows.append({
            "H5": h5, "SPECID": specid, "IFUSLOT": ifuslot, "IFUID": ifuid,
            "AMP": amp, "selection": selection,
            "median_residual_robust_rms": rms,
            "primary_plot": str((primary_dir / ("%s_%s.png" %
                (_safe_name(Path(h5).stem), _identity_text((specid, ifuslot, ifuid, amp))))).resolve()),
        })
    detail_targets = {(key[0], (int(key[1]), int(key[2]), int(key[3])), key[4])
                      for key, _, _ in gallery}
    detail_dir = output_dir / "detailed_gallery"
    heatmap_dir = output_dir / "spectral_heatmaps"
    # The second native pass is deliberately restricted to the selected
    # amplifier/H5 examples. It supplies the individual-fiber context without
    # retaining every fiber spectrum from the full population.
    for h5_path in h5_paths:
        if not plot_allowed:
            break
        h5_targets = {(physical, amp) for h5, physical, amp in detail_targets
                      if h5 == h5_path.name}
        if not h5_targets:
            continue
        native_items, _ = native.load(
            [h5_path], blank_masks, on_filter, off_filter,
            include_native_errors=False, include_band_errors=False,
            collapse_band_indices=())
        detail_records = defaultdict(dict)
        for item in sorted(native_items, key=lambda value: value.exposure):
            codes = [response_identity_map[(item.h5_name, _physical_key(physical))]
                     for physical in np.asarray(item.ifu, dtype=int)]
            calculation75 = _sky_and_residuals(item, codes, response, Q_MAX)
            for physical, amp in h5_targets:
                if not np.any(np.all(np.asarray(item.ifu) == np.asarray(physical), axis=1)
                              & (np.asarray(item.amp, dtype=object) == amp)):
                    continue
                record = _group_record(item, calculation75, physical, amp,
                                       responses, effective, Q_MAX,
                                       keep_individual=True)
                detail_records[(physical, amp)][int(item.exposure)] = record
        for (physical, amp), records in detail_records.items():
            name = "%s_%s" % (_safe_name(h5_path.stem), _identity_text(physical + (amp,)))
            _plot_detail(detail_dir / (name + ".png"), h5_path.name, records, responses)
            if args.heatmaps:
                _plot_heatmap(heatmap_dir / (name + ".png"), h5_path.name, records, responses)
        del native_items, detail_records, item, calculation75

    _write_rows(output_dir / "spectral_gallery_manifest.csv", gallery_rows)
    response_keys_used = {(row["H5"], int(row["IFU_CODE"]), row["AMP"])
                          for row in summary_rows}
    used_logm = np.asarray([response[key] for key in response_keys_used if key in response], dtype=float)
    unique_logm_checks = []
    for row in summary_rows:
        key = (row["H5"], int(row["IFU_CODE"]), row["AMP"])
        unique_logm_checks.append(response.get(key, np.nan))
    unique_logm_checks = _finite(unique_logm_checks)

    validation = {
        "current_cumulative_frozen_M_used": True,
        "response_field_used": "hierarchy.cumulative_final.R_total_h_i_a",
        "M_definition": "exp(R_total_h_i_a), one scalar per H5/IFU_CODE/amplifier, broadcast unchanged over wavelength",
        "M_gray_wavelength_invariant": True,
        "response_composition": composition,
        "pass2_residual_only_response_used": False,
        "alpha_f_q_K_applied": False,
        "Q_applied": False,
        "topology_correction_applied": False,
        "blank_classified_only": True,
        "main_q_range_inclusive": [Q_MIN, Q_MAX],
        "legacy_q_range_inclusive": [Q_MIN, LEGACY_Q_MAX],
        "wavelength_range_A": [WAVE_MIN, WAVE_MAX],
        "native_wavelength_grid_used": True,
        "interpolation_used": False,
        "hardware_exclusions_amplifier_scoped": True,
        "partial_IFUs_retained": True,
        "one_full_resolution_sky_per_H5_exposure": True,
        "global_blank_epsilon_location_max_abs": max_epsilon_location,
        "global_blank_epsilon_location_near_zero": bool(
            np.isfinite(max_epsilon_location) and max_epsilon_location <= 1.0e-7),
        "q40_q70_band_residual_reproduction_passed": bool(
            reproduction_stats["N_matching"] > 0 and reproduction_stats["all_rows_within_tolerance"]),
        "q40_q70_sky_reproduction_passed": bool(
            sky_stats["N_matching"] > 0 and sky_stats["all_rows_within_tolerance"]),
        "no_new_calibration_component_fitted": True,
    }
    all_passed = all((validation["current_cumulative_frozen_M_used"],
                      validation["M_gray_wavelength_invariant"],
                      validation["response_composition"]["composition_matches_stored_cumulative"],
                      validation["blank_classified_only"],
                      validation["global_blank_epsilon_location_near_zero"],
                      validation["q40_q70_band_residual_reproduction_passed"],
                      validation["q40_q70_sky_reproduction_passed"],
                      validation["no_new_calibration_component_fitted"]))

    gallery_manifest_path = output_dir / "spectral_gallery_manifest.csv"
    state_out = {
        "schema_version": "m101_frozenM_spectral_residuals_v1",
        "script": str(Path(__file__).resolve()),
        "fit_performed": False,
        "purpose": "full-resolution blank-fiber residual visualization and seven-band consistency diagnostic",
        "frozen_onoff_state": str(state_path),
        "response": {
            "exact_field": "hierarchy.cumulative_final.R_total_h_i_a",
            "M": "exp(R_total_h_i_a)",
            "n_state_values": len(response),
            "n_values_used": int(used_logm.size),
            "logM_min": float(np.min(used_logm)) if used_logm.size else np.nan,
            "logM_max": float(np.max(used_logm)) if used_logm.size else np.nan,
            "composition_verification": composition,
        },
        "selection": {
            "classified_blank_fibers_only": True,
            "q_range_main_inclusive": [Q_MIN, Q_MAX],
            "q_range_legacy_inclusive": [Q_MIN, LEGACY_Q_MAX],
            "source_fibers_used": False,
            "X_used": False,
            "hardware_valid_rows_only": True,
            "automatic_gallery_size_requested": int(args.gallery_size),
            "explicit_amp_requests_are_additional": True,
        },
        "wavelength": {
            "native_grid_path": "m101_native_data.WAVE / diagnose_m101_hierarchical.DEF_WAVE",
            "native_samples_total": int(WAVE.size),
            "scientific_range_A": [WAVE_MIN, WAVE_MAX],
            "n_scientific_samples": int(np.sum(SCIENCE_MASK)),
            "interpolation": False,
        },
        "band_definitions": {
            "source": "m101_native_data._band_responses and m101_calibration_utils.collapse_many",
            "cache": _file_identity(cache_path),
            "manifest": _file_identity(manifest_path),
            "order": list(BANDS),
            "effective_wavelengths_A": effective.tolist(),
            "cache_effective_wavelength_max_difference_A": effective_difference,
            "definitions": _band_definition_provenance(responses, effective),
            "normalization": "finite-response weighted mean, the same normalization used by the band cache",
        },
        "provenance": {
            "state": _file_identity(state_path),
            "h5": [_file_identity(path) for path in h5_paths],
            "blank_file": _file_identity(blank_file),
            "blank_loader": blank_provenance,
            "on_filter": _file_identity(on_filter),
            "off_filter": _file_identity(off_filter),
            "previous_amp_table": _file_identity(previous_amp_path),
            "previous_sky_table": _file_identity(previous_sky_path),
            "hardware_exclusions_module": str(Path(__import__("m101_hardware_exclusions").__file__).resolve()),
            "hardware_exclusion_record_count": len(HARDWARE_EXCLUSIONS),
            "native_loader": "m101_native_data.load; D=Fibers.spectrum/Survey.offset+Fibers.skyspectrum",
            "native_loader_provenance_last_H5": loader_provenance,
        },
        "counts": {
            "H5_count": len(h5_paths),
            "exposure_count": len(h5_paths) * len(EXPOSURES),
            "selected_blank_fiber_exposure_rows_q40_75": int(sum(row["N_blank_q40_75"] for row in summary_rows)),
            "amplifier_exposure_spectra": len(summary_rows),
            "amplifier_H5_spectra": len({
                (row["H5"], row["SPECID"], row["IFUSLOT"], row["IFUID"], row["AMP"])
                for row in summary_rows}),
            "median_blank_fibers_per_amplifier_exposure": float(np.median(
                [row["N_blank_q40_75"] for row in summary_rows])) if summary_rows else np.nan,
            "gallery_examples": len(gallery),
        },
        "band_reproduction": reproduction_stats,
        "sky_reproduction": sky_stats,
        "validation": validation,
        "diagnostic_if_sky_gate_fails": [
            "different q selection",
            "different blank selection",
            "integrated versus averaged band convention",
            "wavelength weighting",
            "missing-data treatment",
            "response mapping",
            "sky robust-location order of operations",
        ],
        "consistency_diagnosis": {
            "sky_difference_exceeds_roundoff": bool(
                np.isfinite(sky_stats["p95_absolute_difference"])
                and sky_stats["p95_absolute_difference"] > 1.0e-6),
            "most_direct_structural_difference_to_check":
                "robust_location_lambda(D/M) followed by band collapse versus "
                "robust_location_fiber(collapse_band(D/M)); these operations need not commute",
            "interpretation": "diagnostic discrepancy only; no calibration component was inferred or fitted",
        },
        "plot_policy": {
            "consistency_gate_checked_before_default_plots": True,
            "gate_passed_before_plots": preplot_gate_pass,
            "force_plots_requested": bool(args.force_plots),
            "plots_rendered": plot_allowed,
            "automatic_gallery_size_requested": int(args.gallery_size),
            "gallery_examples_rendered": len(gallery),
            "all_amplifiers_plotted": False,
        },
        "artifacts": {
            "summary": str((output_dir / "spectral_amp_residual_summary.csv").resolve()),
            "band_overlay": str((output_dir / "spectral_band_overlay.csv").resolve()),
            "band_reproduction": str((output_dir / "spectral_band_reproduction_q40_70.csv").resolve()),
            "sky_reproduction": str((output_dir / "spectral_sky_band_reproduction_q40_70.csv").resolve()),
            "gallery_manifest": str(gallery_manifest_path.resolve()),
            "primary_spectra_directory": str(primary_dir.resolve()),
            "four_panel_directory": str(ifu_dir.resolve()),
            "detail_directory": str(detail_dir.resolve()),
            "heatmap_directory": str(heatmap_dir.resolve()) if args.heatmaps else None,
        },
        "command": " ".join(shlex.quote(value) for value in sys.argv),
        "elapsed_seconds": time.perf_counter() - started,
    }
    (output_dir / "m101_frozenM_spectral_residuals_state.json").write_text(
        json.dumps(_json_ready(state_out), indent=2, sort_keys=True))

    print("H5 count: %d" % len(h5_paths))
    print("exposure count: %d" % (len(h5_paths) * len(EXPOSURES)))
    print("wavelength range: %.0f--%.0f A" % (WAVE_MIN, WAVE_MAX))
    print("main q range: %d--%d inclusive" % (Q_MIN, Q_MAX))
    print("total selected blank fibers: %d blank-fiber/exposure amplifier rows" %
          sum(row["N_blank_q40_75"] for row in summary_rows))
    print("total usable amplifier/exposure spectra: %d" % len(summary_rows))
    print("median blank fibers per amplifier: %.1f" % (
        float(np.median([row["N_blank_q40_75"] for row in summary_rows]))
        if summary_rows else np.nan))
    print("legacy q=40..70 reproduction: N matching band measurements=%d" %
          reproduction_stats["N_matching"])
    print("legacy q=40..70 residual differences: median=%s robust scatter=%s p95 abs=%s max abs=%s" %
          tuple("%.9g" % reproduction_stats[key] for key in (
              "median_difference", "robust_difference_scatter",
              "p95_absolute_difference", "maximum_absolute_difference")))
    print("q=40..70 sky reproduction: N=%d median=%s robust scatter=%s p95 abs=%s max abs=%s" %
          (sky_stats["N_matching"], *["%.9g" % sky_stats[key] for key in (
              "median_difference", "robust_difference_scatter",
              "p95_absolute_difference", "maximum_absolute_difference")]))
    if all_passed:
        print("FULL-SPECTRUM / BAND CONSISTENCY PASSED")
    else:
        print("FULL-SPECTRUM / BAND CONSISTENCY FAILED; inspect reproduction tables and state diagnostics")
    print("wrote diagnostic products to %s" % output_dir)


if __name__ == "__main__":
    main()
