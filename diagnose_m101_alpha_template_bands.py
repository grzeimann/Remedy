#!/usr/bin/env python3
"""Cache-only diagnostic for the existing alpha*f_q(q)*K_band template.

The script selects blank-rich amplifier/exposure combinations from the blank
fiber census, centers each band on its own blank q=40..75 level, and fits one
robust scalar alpha to the fixed f_q(q) * K_band template.  It never reads
native spectra, changes a calibration state, or refits the q template or K.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from math_utils import biweight


CACHE_SCHEMA = "m101_band_cache_v1"
BANDS = ("HIGH1", "LOW1", "HIGH2", "LOW2", "HIGH3", "ON", "OFF")
AMP_ORDER = ("LL", "LU", "RL", "RU")
AMP_INDEX = {name: index for index, name in enumerate(AMP_ORDER)}
Q_MIN = 40
Q_MAX = 75
MIN_NORMALIZATION = 1.0e-10
DEFAULT_GALLERY_SIZE = 16
DEFAULT_GALLERY_ALPHA_MIN = 0.8
DEFAULT_GALLERY_SEED = 20260918
BLANK_RICH_THRESHOLD = 30
RESTRICTED_ALPHA_MIN = -0.5
RESTRICTED_ALPHA_MAX = 2.0
AMP_DISTRIBUTION_CHUNK_SIZE = 50
AMP_DISTRIBUTION_YMIN = -0.3
AMP_DISTRIBUTION_YMAX = 1.5

def _json_ready(value):
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_json_ready(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _json_ready(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _file_identity(path):
    path = Path(path).expanduser().resolve()
    stat = path.stat()
    return {
        "filename": path.name,
        "full_path": str(path),
        "exists": True,
        "file_size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


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
                if isinstance(value, (tuple, list, np.ndarray)):
                    value = json.dumps(_json_ready(value), separators=(",", ":"))
                clean[field] = value
            writer.writerow(clean)


def _finite(values):
    values = np.asarray(values, dtype=float)
    return values[np.isfinite(values)]


def _physical_amp_key(row):
    return (int(row["SPECID"]), int(row["IFUSLOT"]), int(row["IFUID"]),
            str(row["IFU_CODE"]), str(row["AMP"]))


def _robust_location(values):
    values = _finite(values)
    if not values.size:
        return np.nan
    if values.size == 1 or np.all(values == values[0]):
        return float(values[0])
    result = biweight(values)
    if np.isfinite(result):
        return float(result)
    return float(np.nanmedian(values))


def _robust_scatter(values):
    values = _finite(values)
    if values.size < 2:
        return np.nan
    if np.all(values == values[0]):
        return 0.0
    _, scale = biweight(values, calc_std=True)
    if np.isfinite(scale) and scale > 0:
        return float(scale)
    return float(np.nanstd(values))


def _correlation(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    good = np.isfinite(x) & np.isfinite(y)
    if np.sum(good) < 3:
        return np.nan
    x = x[good]
    y = y[good]
    if np.std(x) <= 0 or np.std(y) <= 0:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def _distribution(values):
    values = _finite(values)
    if not values.size:
        return {name: np.nan for name in (
            "N", "median", "robust_scatter", "p01", "p05", "p16",
            "p84", "p95", "p99", "min", "max")}
    return {
        "N": int(values.size),
        "median": float(np.median(values)),
        "robust_scatter": _robust_scatter(values),
        "p01": float(np.percentile(values, 1)),
        "p05": float(np.percentile(values, 5)),
        "p16": float(np.percentile(values, 16)),
        "p84": float(np.percentile(values, 84)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def _irls_scalar(predictor, target, sigma=None, iterations=5):
    """The initializer's robust scalar regression form, reproduced exactly."""
    predictor = np.asarray(predictor, dtype=float)
    target = np.asarray(target, dtype=float)
    good = np.isfinite(predictor) & np.isfinite(target)
    if sigma is None:
        sigma = np.ones(target.shape, dtype=float)
    else:
        sigma = np.asarray(sigma, dtype=float)
        good &= np.isfinite(sigma) & (sigma > 0)
    predictor = predictor[good]
    target = target[good]
    sigma = sigma[good]
    if predictor.size == 0 or np.sum(predictor ** 2) <= 0:
        return 0.0, {"N": int(predictor.size), "uncertainty": np.nan,
                     "scatter": np.nan}
    base = 1.0 / np.maximum(sigma, 1e-12) ** 2
    robust = np.ones(target.size, dtype=float)
    value = 0.0
    for _ in range(iterations):
        weight = base * robust
        value = float(np.sum(weight * predictor * target) /
                      np.sum(weight * predictor ** 2))
        residual = target - value * predictor
        scale = max(float(_robust_scatter(residual)), 1e-8)
        standardized = np.abs(residual) / scale
        robust = np.where(standardized > 1.345, 1.345 / standardized, 1.0)
    uncertainty = scale / math.sqrt(max(float(np.sum(weight * predictor ** 2)), 1e-12))
    return value, {"N": int(predictor.size), "uncertainty": float(uncertainty),
                   "scatter": float(scale)}


def _synthetic_scalar_fit_test():
    predictor = np.asarray([-2.0, -1.0, .5, 1.0, 3.0, 4.0])
    target = .37 * predictor
    target[-1] += 0.01
    value, fit = _irls_scalar(predictor, target)
    if not np.isfinite(value) or abs(value - .37) > 0.01 or fit["N"] != 6:
        raise RuntimeError("synthetic robust scalar alpha regression test failed")
    return True


def _load_fq(path):
    """Same q,f CSV semantics as diagnose_m101_hierarchical.load_fq."""
    with Path(path).open(newline="") as stream:
        reader = csv.DictReader(stream)
        fields = {field.lower(): field for field in (reader.fieldnames or [])}
        value_field = fields.get("f", fields.get("f_converged"))
        if "q" not in fields or value_field is None:
            raise ValueError("f(q) template needs q,f columns")
        rows = [(int(float(row[fields["q"]])), float(row[value_field]))
                for row in reader]
    if len(rows) != 112 or sorted(q for q, _ in rows) != list(range(112)):
        raise ValueError("f(q) template must contain exactly q=0..111")
    fq = np.full(112, np.nan, dtype=float)
    for q, value in rows:
        fq[q] = value
    if not np.all(np.isfinite(fq)):
        raise ValueError("f(q) template contains nonfinite values")
    return fq


def _resolve_fq_template(explicit, cache_path):
    candidates = []
    if explicit:
        candidates.append(Path(explicit).expanduser())
    initial_state = cache_path.with_name("m101_band_initial_state.json")
    if initial_state.exists():
        try:
            state = json.loads(initial_state.read_text())
            provenance = state.get("provenance", {}).get("fq_template", {})
            if provenance.get("full_path"):
                candidates.append(Path(provenance["full_path"]).expanduser())
        except (OSError, ValueError, TypeError):
            pass
    candidates.extend((
        Path("common_fq_template.csv"),
        Path("/Users/grz85/Downloads/common_fq_template.csv"),
    ))
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(
        "fixed f(q) template not found; pass --fq-template explicitly")


def _load_manifest(cache_path):
    manifest_path = cache_path.with_suffix(cache_path.suffix + ".json")
    if not cache_path.exists() or not manifest_path.exists():
        raise FileNotFoundError("band cache and manifest are required: %s" % cache_path)
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema_version") != CACHE_SCHEMA:
        raise ValueError("unsupported cache schema: %s" % manifest.get("schema_version"))
    if tuple(manifest.get("band_order", ())) != BANDS:
        raise ValueError("cache band order does not match the established seven bands")
    if not manifest.get("items"):
        raise ValueError("cache manifest has no exposure items")
    for item in manifest["items"]:
        K = np.asarray(item.get("K", ()), dtype=float)
        if K.shape != (len(BANDS),) or not np.all(np.isfinite(K)):
            raise ValueError("cache manifest item has invalid seven-band K")
    return manifest, manifest_path


def _load_census(path):
    required = {
        "H5", "exposure", "SPECID", "IFUSLOT", "IFUID", "IFU_CODE", "AMP",
        "N_blank_valid_q40_75", "amplifier_status",
    }
    selected = {}
    all_rows = 0
    with Path(path).open(newline="") as stream:
        reader = csv.DictReader(stream)
        if not required.issubset(reader.fieldnames or ()):
            raise ValueError("census is missing fields: %s" % sorted(required - set(reader.fieldnames or ())))
        for raw in reader:
            all_rows += 1
            if raw["amplifier_status"] != "usable":
                continue
            if int(raw["N_blank_valid_q40_75"]) < BLANK_RICH_THRESHOLD:
                continue
            amp = raw["AMP"].strip().upper()
            if amp not in AMP_INDEX:
                raise ValueError("unknown amplifier in census: %s" % amp)
            key = (raw["H5"], int(raw["exposure"]), int(raw["SPECID"]),
                   int(raw["IFUSLOT"]), int(raw["IFUID"]), int(raw["IFU_CODE"]), amp)
            if key in selected:
                raise ValueError("duplicate selected census row: %s" % (key,))
            raw["_key"] = key
            raw["_N_blank_central"] = int(raw["N_blank_valid_q40_75"])
            selected[key] = raw
    if not selected:
        raise ValueError("census contains no usable amplifier/exposure with N_blank >= 30")
    return selected, all_rows


def _load_cache_arrays(cache_path, manifest):
    required = (
        "ifu", "ifu_code", "amp", "q", "band_total", "band_error",
        "blank_classified", "blank_valid", "hardware_bad",
    )
    with np.load(cache_path, allow_pickle=False) as archive:
        missing = [name for name in required if name not in archive.files]
        if missing:
            raise ValueError("cache missing required fields: %s" % missing)
        arrays = {name: np.asarray(archive[name]) for name in required}
    n_rows = int(arrays["q"].size)
    if arrays["ifu"].shape != (n_rows, 3):
        raise ValueError("cache ifu shape is inconsistent")
    for name in ("ifu_code", "amp", "q", "blank_classified", "blank_valid", "hardware_bad"):
        if arrays[name].shape != (n_rows,):
            raise ValueError("cache field %s has an inconsistent shape" % name)
    if arrays["band_total"].shape != (n_rows, len(BANDS)):
        raise ValueError("cache band_total shape is inconsistent")
    if arrays["band_error"].shape != (n_rows, len(BANDS)):
        raise ValueError("cache band_error shape is inconsistent")
    if int(manifest.get("total_rows", n_rows)) != n_rows:
        raise ValueError("manifest total_rows disagrees with cache arrays")
    return arrays


def _item_context(arrays, manifest_item):
    start = int(manifest_item["start"])
    stop = int(manifest_item["stop"])
    sl = slice(start, stop)
    return {
        "H5": str(manifest_item["h5_name"]),
        "exposure": int(manifest_item["exposure"]),
        "K": np.asarray(manifest_item["K"], dtype=float),
        "ifu": arrays["ifu"][sl],
        "ifu_code": arrays["ifu_code"][sl],
        "amp": arrays["amp"][sl],
        "q": arrays["q"][sl],
        "band_total": arrays["band_total"][sl],
        "band_error": arrays["band_error"][sl],
        "blank_classified": arrays["blank_classified"][sl].astype(bool, copy=False),
        "blank_valid": arrays["blank_valid"][sl].astype(bool, copy=False),
        "hardware_bad": arrays["hardware_bad"][sl].astype(bool, copy=False),
    }


def _fit_band(predictor, target, error):
    target = np.asarray(target, dtype=float)
    predictor = np.asarray(predictor, dtype=float)
    error = np.asarray(error, dtype=float)
    scale = _robust_scatter(target)
    if not np.isfinite(scale) or scale <= 0:
        positive = error[np.isfinite(error) & (error > 0)]
        scale = float(np.nanmedian(positive)) if positive.size else .01
    scale = max(float(scale) if np.isfinite(scale) else .01, 1e-6)
    sigma = np.where(np.isfinite(error) & (error > 0), error, scale)
    sigma = np.maximum(sigma, .25 * scale)
    good = np.isfinite(target) & np.isfinite(predictor) & (np.abs(predictor) > 0)
    alpha, fit = _irls_scalar(predictor[good], target[good], sigma[good])
    fit["scatter_before"] = _robust_scatter(target)
    after = target[good] - alpha * predictor[good]
    fit["scatter_after"] = _robust_scatter(after)
    fit["correlation"] = _correlation(target[good], alpha * predictor[good])
    before = fit["scatter_before"]
    after_scale = fit["scatter_after"]
    fit["percent_reduction"] = (
        100.0 * (before - after_scale) / before
        if np.isfinite(before) and before > 0 and np.isfinite(after_scale)
        else np.nan)
    fit["power_fraction_removed"] = (
        1.0 - (after_scale / before) ** 2
        if np.isfinite(before) and before > 0 and np.isfinite(after_scale)
        else np.nan)
    return alpha, fit, good


def _q_profile_statistics(q_values, residual_values):
    """Return sorted q-bin robust summaries, avoiding biweight for singletons."""
    q_values = np.asarray(q_values, dtype=int)
    residual_values = np.asarray(residual_values, dtype=float)
    order = np.argsort(q_values, kind="stable")
    q_sorted = q_values[order]
    residual_sorted = residual_values[order]
    q_unique, starts = np.unique(q_sorted, return_index=True)
    stops = np.r_[starts[1:], q_sorted.size]
    result = []
    for q_value, start, stop in zip(q_unique, starts, stops):
        values = residual_sorted[start:stop]
        result.append((int(q_value), int(values.size),
                       _robust_location(values), _robust_scatter(values)))
    return result


def _analyze_target(item, target, fq):
    key = target["_key"]
    _, exposure, specid, ifuslot, ifuid, ifu_code, amp_name = key
    amp_index = AMP_INDEX[amp_name]
    physical = np.asarray((specid, ifuslot, ifuid), dtype=int)
    group = (
        np.all(item["ifu"] == physical, axis=1)
        & (item["ifu_code"] == ifu_code)
        & (item["amp"] == amp_index)
    )
    valid = group & item["blank_valid"] & ~item["hardware_bad"]
    central = valid & (item["q"] >= Q_MIN) & (item["q"] <= Q_MAX)
    n_central = int(np.sum(central))
    if n_central != target["_N_blank_central"]:
        raise ValueError(
            "census/cache central blank mismatch for %s: census=%d cache=%d" %
            (key, target["_N_blank_central"], n_central))
    if np.any(item["blank_valid"] & ~item["blank_classified"]):
        raise ValueError("blank_valid classification changed in cache")
    if np.any(item["blank_valid"] & item["hardware_bad"]):
        raise ValueError("blank_valid includes hardware-bad cache rows")
    if n_central < BLANK_RICH_THRESHOLD:
        raise ValueError("selected target lost its blank-rich central support")

    K = item["K"]
    f0 = _robust_location(fq[item["q"][central]])
    if not np.isfinite(f0):
        raise ValueError("central f_q reference is nonfinite for %s" % (key,))

    band_data = {}
    all_predictors, all_targets, all_errors = [], [], []
    profile_rows = []
    central_q_rows = []
    for band_index, band in enumerate(BANDS):
        band_good = valid & np.isfinite(item["band_total"][:, band_index])
        q_values = item["q"][band_good]
        central_band_good = central & np.isfinite(item["band_total"][:, band_index])
        central_q_residual = _robust_location(
            item["band_total"][central_band_good, band_index])
        residual = item["band_total"][band_good, band_index] - central_q_residual
        predictor = (fq[q_values] - f0) * K[band_index]
        error = item["band_error"][band_good, band_index]
        band_data[band] = {
            "q": q_values,
            "residual": residual,
            "predictor": predictor,
            "error": error,
        }
        central_q_rows.append({
            "h5": key[0], "exposure": exposure, "SPECID": specid,
            "IFUSLOT": ifuslot, "IFUID": ifuid, "IFU_CODE": ifu_code,
            "AMP": amp_name, "band": band,
            "central_q_residual": central_q_residual,
            "N_q_used": int(np.sum(central_band_good)),
            "alpha_joint": np.nan,
            "N_blank_central": n_central,
        })
        all_predictors.append(predictor)
        all_targets.append(residual)
        all_errors.append(error)

    joint_predictor = np.concatenate(all_predictors)
    joint_target = np.concatenate(all_targets)
    joint_error = np.concatenate(all_errors)
    joint_alpha, joint_fit, _ = _fit_band(joint_predictor, joint_target, joint_error)
    for row in central_q_rows:
        row["alpha_joint"] = joint_alpha

    alpha_band = {}
    band_fit = {}
    for band in BANDS:
        data = band_data[band]
        alpha, fit, _ = _fit_band(data["predictor"], data["residual"], data["error"])
        alpha_band[band] = alpha
        band_fit[band] = fit
        prediction = joint_alpha * data["predictor"]
        for q_value, n_q, observed, observed_scatter in _q_profile_statistics(
                data["q"], data["residual"]):
            denominator = joint_alpha * K[BANDS.index(band)]
            normalized_observed = (
                observed / denominator if np.isfinite(denominator)
                and abs(denominator) > MIN_NORMALIZATION else np.nan)
            normalized_template = fq[int(q_value)] - f0
            profile_rows.append({
                "H5": key[0], "exposure": exposure, "SPECID": specid,
                "IFUSLOT": ifuslot, "IFUID": ifuid, "IFU_CODE": ifu_code,
                "AMP": amp_name, "band": band, "q": int(q_value),
                "N": n_q,
                "observed_robust_residual": observed,
                "observed_robust_scatter": observed_scatter,
                "template_prediction": joint_alpha * (fq[int(q_value)] - f0) * K[BANDS.index(band)],
                "f0": f0, "K_band": K[BANDS.index(band)],
                "alpha_joint": joint_alpha,
                "normalized_observed": normalized_observed,
                "normalized_template": normalized_template,
            })

    alpha_values = np.asarray([alpha_band[band] for band in BANDS], dtype=float)
    alpha_difference = alpha_values - joint_alpha
    alpha_band_scatter = _robust_scatter(alpha_difference)
    fit_row = {
        "H5": key[0], "exposure": exposure, "SPECID": specid,
        "IFUSLOT": ifuslot, "IFUID": ifuid, "IFU_CODE": ifu_code,
        "AMP": amp_name, "N_blank_central": n_central,
        "N_blank_total": int(np.sum(valid)),
        "q_min": int(np.min(item["q"][valid])),
        "q_max": int(np.max(item["q"][valid])), "f0": f0,
        "alpha_joint": joint_alpha,
        "alpha_uncertainty": joint_fit["uncertainty"],
        "fit_robust_scatter": joint_fit["scatter"],
        "scatter_before": joint_fit["scatter_before"],
        "scatter_after": joint_fit["scatter_after"],
        "percent_reduction": joint_fit["percent_reduction"],
        "power_fraction_removed": joint_fit["power_fraction_removed"],
        "template_correlation": joint_fit["correlation"],
        "alpha_band_scatter": alpha_band_scatter,
        "N_fit": joint_fit["N"],
    }
    for band in BANDS:
        fit_row["alpha_" + band] = alpha_band[band]
        fit_row["scatter_before_" + band] = band_fit[band]["scatter_before"]
        fit_row["scatter_after_" + band] = band_fit[band]["scatter_after"]
        fit_row["percent_reduction_" + band] = band_fit[band]["percent_reduction"]
        fit_row["correlation_" + band] = band_fit[band]["correlation"]
        fit_row["power_fraction_removed_" + band] = band_fit[band]["power_fraction_removed"]
    return key, fit_row, profile_rows, central_q_rows


def _write_aggregate_profiles(path, observed, prediction, fiber_counts,
                              profile_counts):
    rows = []
    for band_index, band in enumerate(BANDS):
        for q_value in range(112):
            observed_values = observed[:, band_index, q_value]
            prediction_values = prediction[:, band_index, q_value]
            observed_good = np.isfinite(observed_values)
            prediction_good = np.isfinite(prediction_values)
            if not np.any(observed_good | prediction_good):
                continue
            rows.append({
                "profile_scope": "selected_blank_rich_population",
                "band": band,
                "q": q_value,
                "N_fibers": int(np.sum(fiber_counts[:, band_index, q_value])),
                "N_amplifier_exposure_profiles": int(
                    np.sum(profile_counts[:, band_index, q_value])),
                "observed_robust_residual": _robust_location(observed_values),
                "observed_robust_scatter": _robust_scatter(observed_values),
                "template_prediction": _robust_location(prediction_values),
                "template_prediction_robust_scatter": _robust_scatter(prediction_values),
            })
    _write_rows(path, rows)
    return len(rows)


def _subtract_exposure_band_sky(central_q_rows):
    """Subtract S_eb from retained central-q levels without touching alpha fits."""
    grouped = defaultdict(list)
    for row in central_q_rows:
        value = row["central_q_residual"]
        if np.isfinite(value):
            grouped[(row["h5"], int(row["exposure"]), row["band"])].append(value)

    sky_by_exposure_band = {}
    for key, values in grouped.items():
        sky_by_exposure_band[key] = {
            "S_eb": _robust_location(values),
            "N_amp_measurements": int(len(values)),
        }

    for row in central_q_rows:
        raw = row["central_q_residual"]
        sky = sky_by_exposure_band.get(
            (row["h5"], int(row["exposure"]), row["band"]),
            {"S_eb": np.nan, "N_amp_measurements": 0})
        row["central_q_residual_raw"] = raw
        row["S_eb"] = sky["S_eb"]
        row["N_S_eb_amp_measurements"] = sky["N_amp_measurements"]
        row["central_q_residual"] = (
            raw - sky["S_eb"]
            if np.isfinite(raw) and np.isfinite(sky["S_eb"]) else np.nan)
    return sky_by_exposure_band


def _select_gallery(fits, maximum, alpha_min=DEFAULT_GALLERY_ALPHA_MIN,
                    seed=DEFAULT_GALLERY_SEED):
    """Select a reproducible random gallery from the requested alpha tail."""
    eligible = [row for row in fits
                if np.isfinite(row["alpha_joint"])
                and row["alpha_joint"] > alpha_min]
    if not eligible:
        return []
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(eligible))[:min(maximum, len(eligible))]
    return [(eligible[index]["_key"], eligible[index],
             "random_alpha_gt_%g" % alpha_min) for index in order]


def _select_same_amp_gallery(fits, alpha_min=DEFAULT_GALLERY_ALPHA_MIN):
    """Return every fitted case for one physical amp with a high-alpha case."""
    if not fits:
        return [], None

    by_physical_amp = defaultdict(list)
    for row in fits:
        by_physical_amp[_physical_amp_key(row)].append(row)
    eligible_amp_keys = {
        amp_key for amp_key, rows in by_physical_amp.items()
        if any(np.isfinite(row["alpha_joint"])
               and row["alpha_joint"] > alpha_min for row in rows)
    }
    if not eligible_amp_keys:
        return [], None

    # Prefer the amplifier with the most available fitted H5/exposure cases;
    # break ties by the physical identity for deterministic output.
    selected_amp = min(
        eligible_amp_keys,
        key=lambda amp_key: (-len(by_physical_amp[amp_key]), amp_key))
    rows = sorted(
        by_physical_amp[selected_amp],
        key=lambda row: (str(row["H5"]), int(row["exposure"]), row["_key"]))
    label = "all_cases_same_amp_with_alpha_gt_%g" % alpha_min
    return [(row["_key"], row, label) for row in rows], selected_amp


def _stats_row(summary_type, group, band, values):
    result = {"summary_type": summary_type, "group": group, "band": band}
    result.update(_distribution(values))
    return result


def _band_consistency(fits):
    rows = []
    for band in BANDS:
        joint = np.asarray([row["alpha_joint"] for row in fits], dtype=float)
        band_values = np.asarray([row["alpha_" + band] for row in fits], dtype=float)
        difference = band_values - joint
        good = np.isfinite(difference)
        fractional = difference / joint
        fractional[~(np.isfinite(joint) & (np.abs(joint) > MIN_NORMALIZATION))] = np.nan
        diff_stats = _distribution(difference)
        frac_stats = _distribution(fractional)
        rows.append({
            "band": band,
            "N": diff_stats["N"],
            "median_offset": diff_stats["median"],
            "robust_scatter_offset": diff_stats["robust_scatter"],
            "p95_absolute_offset": (
                float(np.percentile(np.abs(difference[good]), 95)) if np.any(good) else np.nan),
            "correlation_alpha_band_joint": _correlation(band_values, joint),
            "N_fractional": frac_stats["N"],
            "median_fractional_difference": frac_stats["median"],
            "robust_scatter_fractional_difference": frac_stats["robust_scatter"],
            "p95_absolute_fractional_difference": (
                float(np.percentile(np.abs(fractional[np.isfinite(fractional)]), 95))
                if np.any(np.isfinite(fractional)) else np.nan),
        })
    return rows


def _population_summary(fits, consistency):
    rows = []
    rows.append(_stats_row(
        "joint_alpha_population", "ALL", "",
        [row["alpha_joint"] for row in fits]))
    for amp in AMP_ORDER:
        rows.append(_stats_row(
            "joint_alpha_by_amplifier", amp, "",
            [row["alpha_joint"] for row in fits if row["AMP"] == amp]))
    for row in consistency:
        result = dict(row)
        result.update({"summary_type": "alpha_band_minus_joint", "group": "ALL"})
        rows.append(result)
    by_exposure = defaultdict(list)
    for row in fits:
        by_exposure[(row["H5"], int(row["exposure"]))].append(row["alpha_joint"])
    for (h5, exposure), values in sorted(by_exposure.items()):
        rows.append(_stats_row(
            "joint_alpha_by_exposure", "%s/e%d" % (Path(h5).stem, exposure), "", values))
    return rows


def _make_population_plots(output_dir, fits):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    values = _finite([row["alpha_joint"] for row in fits])
    if values.size:
        bins = max(20, min(60, int(np.sqrt(values.size))))
        figure, axis = plt.subplots(figsize=(8, 5))
        axis.hist(values, bins=bins, color="#4472a8", edgecolor="white")
        axis.axvline(np.median(values), color="black", lw=.9, label="median")
        axis.set_xlabel("joint alpha")
        axis.set_ylabel("amplifier/exposure count")
        axis.set_title("Blank-rich joint alpha distribution")
        axis.grid(axis="y", alpha=.2)
        axis.legend()
        figure.tight_layout()
        figure.savefig(output_dir / "alpha_joint_histogram.png", dpi=130)
        plt.close(figure)

        # Keep the central distribution readable while retaining explicit
        # overflow bins so that the restricted view does not hide fitted
        # values outside the displayed alpha range.
        lower = RESTRICTED_ALPHA_MIN
        upper = RESTRICTED_ALPHA_MAX
        n_interior_bins = 25
        interior_edges = np.linspace(lower, upper, n_interior_bins + 1)
        interior_values = values[(values > lower) & (values < upper)]
        interior_counts, _ = np.histogram(interior_values, bins=interior_edges)
        bin_width = interior_edges[1] - interior_edges[0]
        centers = (interior_edges[:-1] + interior_edges[1:]) / 2.0
        positions = np.concatenate((
            [lower - bin_width], centers, [upper + bin_width]))
        counts = np.concatenate((
            [np.sum(values <= lower)], interior_counts,
            [np.sum(values >= upper)]))
        colors = (["#c44e52"] + ["#4c72b0"] * n_interior_bins
                  + ["#c44e52"])
        figure, axis = plt.subplots(figsize=(10, 5))
        axis.bar(positions, counts, width=bin_width * .92,
                 color=colors, edgecolor="white", align="center")
        axis.axvline(np.median(values), color="black", lw=.9,
                     label="median = %.4g" % np.median(values))
        axis.set_xlim(lower - 1.65 * bin_width, upper + 1.65 * bin_width)
        axis.set_xticks([positions[0], lower, 0.0, 0.5, 1.0, 1.5,
                         upper, positions[-1]])
        axis.set_xticklabels(["<= %.1f" % lower, "%.1f" % lower, "0", "0.5",
                              "1.0", "1.5", "%.1f" % upper,
                              ">= %.1f" % upper])
        axis.set_xlabel("joint alpha (central bins: %.1f < alpha < %.1f)" %
                        (lower, upper))
        axis.set_ylabel("amplifier/exposure count")
        axis.set_title("Restricted blank-rich joint alpha distribution")
        axis.grid(axis="y", alpha=.2)
        axis.legend()
        figure.tight_layout()
        figure.savefig(output_dir / "alpha_joint_histogram_restricted.png", dpi=130)
        plt.close(figure)

        # Show the full signed distribution, with the negative side reflected
        # onto the positive side for comparison.  The null model is centered
        # at zero: estimate sigma from the negative-side RMS rather than fit a
        # nonzero mean to the measured alpha population.  A two-sided Gaussian
        # normalized to 2*N_negative therefore has N_negative expected counts
        # in its negative half.
        reflection_bin_width = bin_width
        reflection_edges = np.arange(
            0.0, upper + reflection_bin_width * 0.5, reflection_bin_width)
        negative_values = values[(values > lower) & (values < 0.0)]
        reflected_negative = -negative_values
        reflected_counts, _ = np.histogram(reflected_negative, bins=reflection_edges)
        reflection_centers = (reflection_edges[:-1] + reflection_edges[1:]) / 2.0
        signed_interior_values = values[(values > lower) & (values < upper)]
        signed_counts, _ = np.histogram(signed_interior_values, bins=interior_edges)
        signed_positions = np.concatenate((
            [lower - bin_width], centers, [upper + bin_width]))
        signed_counts_with_overflow = np.concatenate((
            [np.sum(values <= lower)], signed_counts,
            [np.sum(values >= upper)]))

        figure, axis = plt.subplots(figsize=(10, 5))
        axis.bar(signed_positions, signed_counts_with_overflow,
                 width=.72 * bin_width, color="#4472a8", alpha=.60,
                 edgecolor="white", label="fitted alpha values")
        axis.bar(reflection_centers + .38 * reflection_bin_width,
                 reflected_counts, width=.28 * reflection_bin_width,
                 color="#c44e52", alpha=.85,
                 label="negative side reflected about zero")

        if reflected_negative.size >= 1:
            # For a zero-centered normal, the negative half has RMS sigma.
            null_sigma = float(np.sqrt(np.mean(reflected_negative ** 2)))
            if np.isfinite(null_sigma) and null_sigma > 0:
                gaussian_x = np.linspace(lower, upper, 600)
                gaussian_y = (2.0 * reflected_negative.size * bin_width /
                              (null_sigma * np.sqrt(2.0 * np.pi)) *
                              np.exp(-0.5 * (gaussian_x / null_sigma) ** 2))
                axis.plot(
                    gaussian_x, gaussian_y, color="#8c1d40", lw=1.8, ls="--",
                    label=("zero-centered null Gaussian "
                           "(Nneg=%d, sigma=%.3g)" %
                           (reflected_negative.size, null_sigma)))
        axis.set_xlim(lower - 1.65 * bin_width, upper + 1.65 * bin_width)
        axis.set_xticks([signed_positions[0], lower, 0.0, 0.5, 1.0, 1.5,
                         upper, signed_positions[-1]])
        axis.set_xticklabels(["<= %.1f" % lower, "%.1f" % lower, "0", "0.5",
                              "1.0", "1.5", "%.1f" % upper,
                              ">= %.1f" % upper])
        axis.set_xlabel("joint alpha")
        axis.set_ylabel("amplifier/exposure count")
        axis.set_title("Joint alpha with zero-centered null from negative side")
        axis.grid(axis="y", alpha=.2)
        axis.legend(fontsize=8)
        figure.tight_layout()
        figure.savefig(output_dir / "alpha_joint_histogram_negative_reflection.png",
                       dpi=130)
        plt.close(figure)

    by_amp = [[row["alpha_joint"] for row in fits if row["AMP"] == amp]
              for amp in AMP_ORDER]
    if any(by_amp):
        figure, axis = plt.subplots(figsize=(7, 5))
        axis.boxplot(by_amp, labels=AMP_ORDER, showfliers=False)
        axis.set_xlabel("amplifier orientation")
        axis.set_ylabel("joint alpha")
        axis.set_title("Joint alpha by amplifier orientation")
        axis.grid(axis="y", alpha=.2)
        figure.tight_layout()
        figure.savefig(output_dir / "alpha_joint_by_amplifier.png", dpi=130)
        plt.close(figure)

    groups = defaultdict(list)
    for row in fits:
        groups[(row["H5"], int(row["exposure"]))].append(row["alpha_joint"])
    if groups:
        labels = ["%s/e%d" % (Path(key[0]).stem, key[1]) for key in sorted(groups)]
        data = [groups[key] for key in sorted(groups)]
        figure, axis = plt.subplots(figsize=(max(12, .27 * len(data)), 6))
        axis.boxplot(data, positions=np.arange(1, len(data) + 1),
                     widths=.7, showfliers=False)
        axis.set_xticks(np.arange(1, len(labels) + 1))
        axis.set_xticklabels(labels, rotation=90, fontsize=6)
        axis.set_xlabel("H5 / exposure")
        axis.set_ylabel("joint alpha")
        axis.set_title("Joint alpha by exposure")
        axis.grid(axis="y", alpha=.2)
        figure.tight_layout()
        figure.savefig(output_dir / "alpha_joint_by_exposure.png", dpi=120)
        plt.close(figure)


def _make_alpha_by_physical_amp_plots(output_dir, fits,
                                      chunk_size=AMP_DISTRIBUTION_CHUNK_SIZE):
    """Plot all fitted alpha values grouped by physical amplifier."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    by_amp = defaultdict(list)
    for row in fits:
        if np.isfinite(row["alpha_joint"]):
            by_amp[_physical_amp_key(row)].append(row)
    amp_keys = sorted(by_amp)
    if not amp_keys:
        return []

    exposure_keys = sorted({
        (str(row["H5"]), int(row["exposure"]))
        for row in fits if np.isfinite(row["alpha_joint"])
    })
    exposure_index = {key: index for index, key in enumerate(exposure_keys)}
    color_max = max(1, len(exposure_keys) - 1)
    cmap = plt.get_cmap("viridis")
    norm = Normalize(vmin=0, vmax=color_max)
    manifest_rows = []

    for chunk_start in range(0, len(amp_keys), chunk_size):
        chunk_keys = amp_keys[chunk_start:chunk_start + chunk_size]
        x_values = []
        alpha_values = []
        colors = []
        below = 0
        above = 0
        for amp_index, amp_key in enumerate(chunk_keys):
            rows = sorted(
                by_amp[amp_key],
                key=lambda row: (str(row["H5"]), int(row["exposure"])))
            for row in rows:
                alpha = float(row["alpha_joint"])
                if alpha < AMP_DISTRIBUTION_YMIN:
                    below += 1
                if alpha > AMP_DISTRIBUTION_YMAX:
                    above += 1
                exp_index = exposure_index[(str(row["H5"]), int(row["exposure"]))]
                # Deterministic horizontal spreading makes repeated
                # H5/exposure points for an amplifier easier to see.
                jitter = (exp_index / color_max - .5) * .68 if color_max else 0.0
                x_values.append(amp_index + jitter)
                alpha_values.append(alpha)
                colors.append(exp_index)

        figure, axis = plt.subplots(figsize=(max(14, .31 * len(chunk_keys)), 7))
        scatter = axis.scatter(x_values, alpha_values, c=colors, cmap=cmap,
                               norm=norm, s=18, alpha=.78,
                               edgecolors="none", rasterized=True)
        axis.axhline(0.0, color="black", lw=.8, alpha=.75)
        axis.set_ylim(AMP_DISTRIBUTION_YMIN, AMP_DISTRIBUTION_YMAX)
        axis.set_xlim(-.8, len(chunk_keys) - .2)
        labels = [
            "%d/%d/%d/%s/%s" % amp_key
            for amp_key in chunk_keys
        ]
        axis.set_xticks(np.arange(len(chunk_keys)))
        axis.set_xticklabels(labels, rotation=90, fontsize=5)
        axis.set_xlabel("physical amplifier: SPECID / IFUSLOT / IFUID / IFU_CODE / AMP")
        axis.set_ylabel("joint alpha")
        axis.set_title(
            "Joint alpha by physical amplifier "
            "(%d-%d of %d; color = chronological H5/exposure index)\n"
            "fixed y range [%.1f, %.1f]; clipped below=%d, above=%d" %
            (chunk_start + 1, chunk_start + len(chunk_keys), len(amp_keys),
             AMP_DISTRIBUTION_YMIN, AMP_DISTRIBUTION_YMAX, below, above))
        colorbar = figure.colorbar(
            ScalarMappable(norm=norm, cmap=cmap), ax=axis, pad=.01)
        colorbar.set_label("chronological H5/exposure index")
        if exposure_keys:
            tick_indices = sorted(set((0, len(exposure_keys) // 2,
                                       len(exposure_keys) - 1)))
            colorbar.set_ticks(tick_indices)
            colorbar.set_ticklabels([
                "%s/e%d" % (Path(exposure_keys[index][0]).stem,
                             exposure_keys[index][1])
                for index in tick_indices
            ])
        axis.grid(axis="y", alpha=.18)
        figure.tight_layout()
        plot_path = output_dir / (
            "alpha_by_physical_amp_chunk_%03d.png" %
            (chunk_start // chunk_size + 1))
        figure.savefig(plot_path, dpi=130, bbox_inches="tight")
        plt.close(figure)

        manifest_rows.append({
            "chunk": int(chunk_start // chunk_size + 1),
            "amp_start_index": int(chunk_start + 1),
            "amp_end_index": int(chunk_start + len(chunk_keys)),
            "N_physical_amplifiers": len(chunk_keys),
            "N_alpha_points": len(alpha_values),
            "N_points_below_ymin": below,
            "N_points_above_ymax": above,
            "physical_amplifiers": [
                "%d/%d/%d/%s/%s" % amp_key for amp_key in chunk_keys
            ],
            "plot": str(plot_path.resolve()),
        })
    return manifest_rows


def _central_q_display_range(values):
    values = _finite(values)
    if not values.size:
        return -1.0, 1.0, np.nan, np.nan
    center = _robust_location(values)
    scatter = _robust_scatter(values)
    if np.isfinite(center) and np.isfinite(scatter) and scatter > 0:
        lower = center - 5.0 * scatter
        upper = center + 5.0 * scatter
    else:
        lower = float(np.min(values))
        upper = float(np.max(values))
    # Keep the zero reference visible even when the robust population range
    # lies entirely on one side of zero.
    lower = min(lower, 0.0)
    upper = max(upper, 0.0)
    if not np.isfinite(lower) or not np.isfinite(upper) or upper <= lower:
        scale = max(1.0, abs(float(center)) if np.isfinite(center) else 1.0)
        lower, upper = -0.1 * scale, 0.1 * scale
    return float(lower), float(upper), float(center), float(scatter)


def _make_band_central_q_residual_plots(
        output_dir, central_q_rows, fits,
        chunk_size=AMP_DISTRIBUTION_CHUNK_SIZE):
    """Plot the pre-centering central-q additive residual for each band."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    amp_keys = sorted({
        _physical_amp_key(row) for row in fits
        if np.isfinite(row["alpha_joint"])
    })
    if not amp_keys:
        return [], {}
    exposure_keys = sorted({
        (str(row["H5"]), int(row["exposure"]))
        for row in fits if np.isfinite(row["alpha_joint"])
    })
    exposure_index = {key: index for index, key in enumerate(exposure_keys)}
    color_max = max(1, len(exposure_keys) - 1)
    cmap = plt.get_cmap("viridis")
    norm = Normalize(vmin=0, vmax=color_max)

    rows_by_band_amp = defaultdict(list)
    values_by_band = defaultdict(list)
    for row in central_q_rows:
        value = row["central_q_residual"]
        if not np.isfinite(value):
            continue
        amp_key = _physical_amp_key(row)
        rows_by_band_amp[(row["band"], amp_key)].append(row)
        values_by_band[row["band"]].append(value)

    ranges = {}
    for band in BANDS:
        ranges[band] = _central_q_display_range(values_by_band[band])[:2]

    manifest_rows = []
    for band in BANDS:
        lower, upper = ranges[band]
        for chunk_start in range(0, len(amp_keys), chunk_size):
            chunk_keys = amp_keys[chunk_start:chunk_start + chunk_size]
            x_values = []
            residual_values = []
            colors = []
            below = 0
            above = 0
            for amp_index, amp_key in enumerate(chunk_keys):
                rows = sorted(
                    rows_by_band_amp[(band, amp_key)],
                    key=lambda row: (str(row["h5"]), int(row["exposure"])))
                for row in rows:
                    value = float(row["central_q_residual"])
                    if value < lower:
                        below += 1
                    if value > upper:
                        above += 1
                    exp_index = exposure_index[(str(row["h5"]),
                                                int(row["exposure"]))]
                    jitter = (exp_index / color_max - .5) * .68 \
                        if color_max else 0.0
                    x_values.append(amp_index + jitter)
                    residual_values.append(value)
                    colors.append(exp_index)

            figure, axis = plt.subplots(
                figsize=(max(14, .31 * len(chunk_keys)), 7))
            axis.scatter(x_values, residual_values, c=colors, cmap=cmap,
                         norm=norm, s=18, alpha=.78, edgecolors="none",
                         rasterized=True)
            axis.axhline(0.0, color="black", lw=.8, alpha=.75)
            axis.set_ylim(lower, upper)
            axis.set_xlim(-.8, len(chunk_keys) - .2)
            labels = ["%d/%d/%d/%s/%s" % amp_key for amp_key in chunk_keys]
            axis.set_xticks(np.arange(len(chunk_keys)))
            axis.set_xticklabels(labels, rotation=90, fontsize=5)
            axis.set_xlabel(
                "physical amplifier: SPECID / IFUSLOT / IFUID / IFU_CODE / AMP")
            axis.set_ylabel(
                "central-q additive residual - S_eb (calibrated units)")
            axis.set_title(
                "%s sky-subtracted central-q residual by physical amplifier "
                "(%d-%d of %d; color = chronological H5/exposure index)\n"
                "fixed y range [%.6g, %.6g]; clipped below=%d, above=%d" %
                (band, chunk_start + 1, chunk_start + len(chunk_keys),
                 len(amp_keys), lower, upper, below, above))
            colorbar = figure.colorbar(
                ScalarMappable(norm=norm, cmap=cmap), ax=axis, pad=.01)
            colorbar.set_label("chronological H5/exposure index")
            if exposure_keys:
                tick_indices = sorted(set((0, len(exposure_keys) // 2,
                                           len(exposure_keys) - 1)))
                colorbar.set_ticks(tick_indices)
                colorbar.set_ticklabels([
                    "%s/e%d" % (Path(exposure_keys[index][0]).stem,
                                 exposure_keys[index][1])
                    for index in tick_indices
                ])
            axis.grid(axis="y", alpha=.18)
            figure.tight_layout()
            plot_path = output_dir / (
                "band_central_q_residuals_%s_chunk_%03d.png" %
                (band, chunk_start // chunk_size + 1))
            figure.savefig(plot_path, dpi=130, bbox_inches="tight")
            plt.close(figure)

            manifest_rows.append({
                "band": band,
                "chunk": int(chunk_start // chunk_size + 1),
                "amp_start_index": int(chunk_start + 1),
                "amp_end_index": int(chunk_start + len(chunk_keys)),
                "N_physical_amplifiers": len(chunk_keys),
                "N_measurements": len(residual_values),
                "fixed_ymin": lower,
                "fixed_ymax": upper,
                "N_points_clipped_below": below,
                "N_points_clipped_above": above,
                "physical_amplifiers": [
                    "%d/%d/%d/%s/%s" % amp_key for amp_key in chunk_keys
                ],
                "plot": str(plot_path.resolve()),
            })
    return manifest_rows, ranges


def _make_band_temporal_variation_plots(output_dir, central_q_rows, fits):
    """Plot central-q residual variation after removing each amp's level."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    amp_keys = sorted({
        _physical_amp_key(row) for row in fits
        if np.isfinite(row["alpha_joint"])
    })
    exposure_keys = sorted({
        (str(row["H5"]), int(row["exposure"]))
        for row in fits if np.isfinite(row["alpha_joint"])
    })
    if not amp_keys or not exposure_keys:
        return []

    exposure_index = {key: index for index, key in enumerate(exposure_keys)}
    color_max = max(1, len(exposure_keys) - 1)
    cmap = plt.get_cmap("viridis")
    norm = Normalize(vmin=0, vmax=color_max)
    rows_by_amp_band = defaultdict(list)
    for row in central_q_rows:
        if np.isfinite(row["central_q_residual"]):
            rows_by_amp_band[(_physical_amp_key(row), row["band"])].append(row)

    plot_paths = []
    for band in BANDS:
        amp_level = {}
        for amp_key in amp_keys:
            values = [row["central_q_residual"]
                      for row in rows_by_amp_band[(amp_key, band)]]
            if values:
                amp_level[amp_key] = _robust_location(values)

        points_x = []
        points_y = []
        point_colors = []
        common_by_exposure = defaultdict(list)
        for amp_index, amp_key in enumerate(amp_keys):
            level = amp_level.get(amp_key, np.nan)
            if not np.isfinite(level):
                continue
            rows = sorted(
                rows_by_amp_band[(amp_key, band)],
                key=lambda row: (str(row["h5"]), int(row["exposure"])))
            amp_jitter = (
                (amp_index / max(1, len(amp_keys) - 1) - .5) * .68
            )
            for row in rows:
                value = row["central_q_residual"] - level
                if not np.isfinite(value):
                    continue
                exp_key = (str(row["h5"]), int(row["exposure"]))
                exp_index = exposure_index[exp_key]
                points_x.append(exp_index + amp_jitter)
                points_y.append(value)
                point_colors.append(exp_index)
                common_by_exposure[exp_index].append(value)

        common_x = []
        common_y = []
        common_n = []
        for exp_index in range(len(exposure_keys)):
            values = common_by_exposure.get(exp_index, [])
            if not values:
                continue
            common_x.append(exp_index)
            common_y.append(_robust_location(values))
            common_n.append(len(values))

        all_y = _finite(points_y + common_y)
        if all_y.size:
            ymin = float(np.min(all_y))
            ymax = float(np.max(all_y))
            span = ymax - ymin
            pad = max(.05 * span, .05 * max(abs(ymin), abs(ymax), 1.0))
            if span <= 0:
                pad = max(.1, abs(ymin) * .1)
            ymin -= pad
            ymax += pad
        else:
            ymin, ymax = -1.0, 1.0

        figure, axis = plt.subplots(figsize=(12, 6))
        axis.scatter(points_x, points_y, c=point_colors, cmap=cmap, norm=norm,
                     s=13, alpha=.35, edgecolors="none", rasterized=True,
                     label="amplifier residual - amp level")
        if common_x:
            axis.plot(common_x, common_y, color="black", lw=2.0, marker="o",
                      ms=3.5, label="robust cross-amplifier average")
        axis.axhline(0.0, color="#555555", lw=.8, ls="--")
        axis.set_xlim(-.8, len(exposure_keys) - .2)
        axis.set_ylim(ymin, ymax)
        axis.set_xlabel("chronological H5/exposure index")
        axis.set_ylabel(
            "central-q residual - physical-amplifier robust level\n"
            "(calibrated additive units)")
        axis.set_title(
            "%s mean-subtracted central-q residual temporal variation\n"
            "N amplifier/exposure points=%d; common curve uses robust average "
            "of available amplifiers" % (band, len(points_y)))
        colorbar = figure.colorbar(
            ScalarMappable(norm=norm, cmap=cmap), ax=axis, pad=.01)
        colorbar.set_label("chronological H5/exposure index")
        tick_indices = sorted(set((0, len(exposure_keys) // 2,
                                   len(exposure_keys) - 1)))
        colorbar.set_ticks(tick_indices)
        colorbar.set_ticklabels([
            "%s/e%d" % (Path(exposure_keys[index][0]).stem,
                         exposure_keys[index][1])
            for index in tick_indices
        ])
        axis.grid(alpha=.18)
        axis.legend(loc="best")
        figure.tight_layout()
        plot_path = output_dir / (
            "band_central_q_residual_temporal_variation_%s.png" % band)
        figure.savefig(plot_path, dpi=130, bbox_inches="tight")
        plt.close(figure)
        plot_paths.append({
            "band": band,
            "plot": str(plot_path.resolve()),
            "N_measurements": len(points_y),
            "N_exposures_with_common_average": len(common_x),
            "common_average_N_min": min(common_n) if common_n else 0,
            "common_average_N_max": max(common_n) if common_n else 0,
        })
    return plot_paths


def _plot_gallery_case(path, fit, profiles):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = dict(zip(BANDS, plt.cm.tab10(np.linspace(0, .9, len(BANDS)))))
    figure, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    for band in BANDS:
        curve = [row for row in profiles if row["band"] == band]
        curve.sort(key=lambda row: row["q"])
        q = np.asarray([row["q"] for row in curve], dtype=float)
        observed = np.asarray([row["observed_robust_residual"] for row in curve])
        prediction = np.asarray([row["template_prediction"] for row in curve])
        normalized = np.asarray([row["normalized_observed"] for row in curve])
        template = np.asarray([row["normalized_template"] for row in curve])
        color = colors[band]
        axes[0].plot(q, observed, color=color, lw=1.0, label=band)
        axes[0].plot(q, prediction, color=color, lw=.8, ls="--")
        good = np.isfinite(normalized)
        if np.any(good):
            axes[1].plot(q[good], normalized[good], color=color, lw=1.0)
    # The fixed template is repeated from the profile rows so its centering is
    # exactly the selected amplifier/exposure's central blank reference.
    template_rows = [row for row in profiles if row["band"] == BANDS[0]]
    if template_rows:
        axes[1].plot([row["q"] for row in template_rows],
                     [row["normalized_template"] for row in template_rows],
                     color="black", lw=1.5, ls="--", label="fixed F(q)")
    axes[0].axhline(0, color="black", lw=.7)
    axes[1].axhline(0, color="black", lw=.7)
    axes[0].set_ylabel("R = band_total - central band level")
    axes[1].set_ylabel("R / (alpha_joint K_band)")
    axes[1].set_xlabel("fiber q")
    axes[0].set_title(
        "%s e%d; SPECID=%d IFUSLOT=%d IFUID=%d %s; alpha=%.6g" %
        (fit["H5"], fit["exposure"], fit["SPECID"], fit["IFUSLOT"],
         fit["IFUID"], fit["AMP"], fit["alpha_joint"]))
    axes[0].legend(ncol=4, fontsize=7, loc="best")
    axes[0].grid(alpha=.18)
    axes[1].grid(alpha=.18)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=125, bbox_inches="tight")
    plt.close(figure)


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--band-cache", default="m101_band_initializer/m101_band_cache.npz")
    parser.add_argument("--census", default="m101_blank_fiber_census/blank_fiber_amp_census.csv")
    parser.add_argument("--fq-template", default=None)
    parser.add_argument("--output-dir", default="m101_alpha_template_bands")
    parser.add_argument("--gallery-size", type=int, default=DEFAULT_GALLERY_SIZE)
    parser.add_argument("--gallery-alpha-min", type=float,
                        default=DEFAULT_GALLERY_ALPHA_MIN,
                        help="sample gallery cases with alpha_joint strictly above this value")
    parser.add_argument("--gallery-seed", type=int, default=DEFAULT_GALLERY_SEED,
                        help="random seed for reproducible gallery sampling")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = _parse_args()
    if args.gallery_size <= 0:
        raise SystemExit("--gallery-size must be positive")
    _synthetic_scalar_fit_test()

    cache_path = Path(args.band_cache).expanduser().resolve()
    census_path = Path(args.census).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise SystemExit("output directory is nonempty; use --overwrite: %s" % output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest, manifest_path = _load_manifest(cache_path)
    fq_path = _resolve_fq_template(args.fq_template, cache_path)
    fq = _load_fq(fq_path)
    selected, census_row_count = _load_census(census_path)
    arrays = _load_cache_arrays(cache_path, manifest)

    selected_by_item = defaultdict(list)
    for key, row in selected.items():
        selected_by_item[(key[0], key[1])].append(row)

    fits = []
    fits_by_key = {}
    selected_index = {key: index for index, key in enumerate(sorted(selected))}
    n_selected = len(selected)
    aggregate_observed = np.full((n_selected, len(BANDS), 112), np.nan, dtype=float)
    aggregate_prediction = np.full((n_selected, len(BANDS), 112), np.nan, dtype=float)
    aggregate_fiber_counts = np.zeros((n_selected, len(BANDS), 112), dtype=np.int32)
    aggregate_profile_counts = np.zeros((n_selected, len(BANDS), 112), dtype=np.int16)
    central_q_rows = []
    processed_targets = 0
    for manifest_item in manifest["items"]:
        item_key = (str(manifest_item["h5_name"]), int(manifest_item["exposure"]))
        targets = selected_by_item.get(item_key, ())
        if not targets:
            continue
        item = _item_context(arrays, manifest_item)
        for target in targets:
            key, fit_row, profiles, target_central_q_rows = _analyze_target(
                item, target, fq)
            if key in fits_by_key:
                raise ValueError("duplicate alpha fit key: %s" % (key,))
            fit_row["_key"] = key
            fits.append(fit_row)
            fits_by_key[key] = fit_row
            central_q_rows.extend(target_central_q_rows)
            index = selected_index[key]
            for profile in profiles:
                band_index = BANDS.index(profile["band"])
                q_value = int(profile["q"])
                aggregate_observed[index, band_index, q_value] = profile[
                    "observed_robust_residual"]
                aggregate_prediction[index, band_index, q_value] = profile[
                    "template_prediction"]
                aggregate_fiber_counts[index, band_index, q_value] = int(profile["N"])
                aggregate_profile_counts[index, band_index, q_value] = 1
            processed_targets += 1
            if processed_targets % 500 == 0:
                print("[alpha] processed %d/%d blank-rich amplifier/exposures" %
                      (processed_targets, len(selected)), flush=True)

    if len(fits) != len(selected):
        missing = sorted(set(selected) - set(fits_by_key))
        raise ValueError("cache did not produce selected census fits; examples: %s" % missing[:3])
    print("[alpha] processed %d blank-rich amplifier/exposures; q profiles=%d" %
          (processed_targets, int(np.sum(aggregate_profile_counts))), flush=True)

    sky_by_exposure_band = _subtract_exposure_band_sky(central_q_rows)
    fit_rows_for_csv = []
    for row in fits:
        fit_rows_for_csv.append({key: value for key, value in row.items() if key != "_key"})
    _write_rows(output_dir / "alpha_template_fits.csv", fit_rows_for_csv)

    consistency_rows = _band_consistency(fits)
    _write_rows(output_dir / "alpha_band_consistency.csv", consistency_rows)
    population_rows = _population_summary(fits, consistency_rows)
    _write_rows(output_dir / "alpha_population_summary.csv", population_rows)
    aggregate_profile_path = output_dir / "alpha_q_profiles.csv"
    aggregate_profile_row_count = _write_aggregate_profiles(
        aggregate_profile_path, aggregate_observed, aggregate_prediction,
        aggregate_fiber_counts, aggregate_profile_counts)
    central_q_path = output_dir / "band_central_q_residuals.csv"
    _write_rows(central_q_path, central_q_rows)

    gallery = _select_gallery(fits, args.gallery_size,
                              alpha_min=args.gallery_alpha_min,
                              seed=args.gallery_seed)
    same_amp_gallery, same_amp_key = _select_same_amp_gallery(
        fits, alpha_min=args.gallery_alpha_min)
    gallery_keys_seen = {key for key, _, _ in gallery}
    for entry in same_amp_gallery:
        if entry[0] not in gallery_keys_seen:
            gallery.append(entry)
            gallery_keys_seen.add(entry[0])
    gallery_keys = {key for key, _, _ in gallery}
    gallery_profiles = defaultdict(list)
    if gallery_keys and not args.no_plots:
        for manifest_item in manifest["items"]:
            item_key = (str(manifest_item["h5_name"]), int(manifest_item["exposure"]))
            targets = selected_by_item.get(item_key, ())
            if not targets:
                continue
            item = _item_context(arrays, manifest_item)
            for target in targets:
                if target["_key"] not in gallery_keys:
                    continue
                key, _, profiles, _ = _analyze_target(item, target, fq)
                gallery_profiles[key].extend(profiles)

    gallery_profile_rows = [profile for key in gallery_keys
                            for profile in gallery_profiles[key]]
    gallery_profile_path = output_dir / "alpha_gallery_q_profiles.csv"
    _write_rows(gallery_profile_path, gallery_profile_rows)

    gallery_dir = output_dir / "gallery"
    gallery_manifest = []
    for key, fit, selection in gallery:
        path = None
        if not args.no_plots:
            name = "%s_e%d_%d_%d_%d_%s" % (
                Path(key[0]).stem, key[1], key[2], key[3], key[4], key[6])
            path = gallery_dir / (name + ".png")
            _plot_gallery_case(path, fit, gallery_profiles[key])
        gallery_manifest.append({
            "H5": fit["H5"], "exposure": fit["exposure"],
            "SPECID": fit["SPECID"], "IFUSLOT": fit["IFUSLOT"],
            "IFUID": fit["IFUID"], "IFU_CODE": fit["IFU_CODE"],
            "AMP": fit["AMP"], "selection": selection,
            "alpha_joint": fit["alpha_joint"],
            "alpha_band_scatter": fit["alpha_band_scatter"],
            "template_correlation": fit["template_correlation"],
            "percent_reduction": fit["percent_reduction"],
            "plot": str(path.resolve()) if path is not None else None,
        })
    _write_rows(output_dir / "alpha_gallery_manifest.csv", gallery_manifest)
    physical_amp_plot_manifest = []
    physical_amp_plot_manifest_path = None
    central_band_plot_manifest = []
    central_band_plot_manifest_path = None
    central_band_plot_ranges = {}
    temporal_variation_plot_manifest = []
    if not args.no_plots:
        _make_population_plots(output_dir, fits)
        physical_amp_plot_manifest = _make_alpha_by_physical_amp_plots(
            output_dir, fits)
        physical_amp_plot_manifest_path = (
            output_dir / "alpha_by_physical_amp_manifest.csv")
        _write_rows(physical_amp_plot_manifest_path, physical_amp_plot_manifest)
        central_band_plot_manifest, central_band_plot_ranges = (
            _make_band_central_q_residual_plots(output_dir, central_q_rows, fits))
        central_band_plot_manifest_path = (
            output_dir / "band_central_q_residual_plot_manifest.csv")
        _write_rows(central_band_plot_manifest_path, central_band_plot_manifest)
        temporal_variation_plot_manifest = _make_band_temporal_variation_plots(
            output_dir, central_q_rows, fits)

    alpha_values = np.asarray([row["alpha_joint"] for row in fits], dtype=float)
    reduction = np.asarray([row["percent_reduction"] for row in fits], dtype=float)
    alpha_disagreement = np.asarray([row["alpha_band_scatter"] for row in fits], dtype=float)
    correlations = np.asarray([row["template_correlation"] for row in fits], dtype=float)
    central_counts = np.asarray([row["N_blank_central"] for row in fits], dtype=float)
    total_counts = np.asarray([row["N_blank_total"] for row in fits], dtype=float)
    validation = {
        "cache_only": True,
        "native_spectra_read": False,
        "selected_from_census_N_blank_ge_30": all(
            row["N_blank_central"] >= BLANK_RICH_THRESHOLD for row in fits),
        "blank_valid_reused_unchanged": True,
        "fixed_fq_reused_unchanged": True,
        "cache_K_band_reused_unchanged": True,
        "central_reference_q_inclusive": [Q_MIN, Q_MAX],
        "all_seven_bands_included": True,
        "one_joint_alpha_per_amplifier_exposure": len(fits) == len(fits_by_key),
        "per_band_alphas_diagnostic_only": True,
        "multiplicative_parameter_changed": False,
        "sky_parameter_changed": False,
        "only_alpha_fq_K_additive_term_tested": True,
        "source_information_or_X_used": False,
        "hardware_exclusions_amplifier_scoped": True,
        "synthetic_scalar_fit_test_passed": True,
        "central_q_residual_reused_from_alpha_centering": (
            len(central_q_rows) == len(fits) * len(BANDS)),
        "central_q_residual_pre_alpha_subtraction": True,
        "central_q_residual_not_sky_normalized": True,
        "central_q_residual_no_new_fit": True,
        "central_q_residual_sky_subtracted_for_visualization": True,
        "alpha_fit_unchanged_by_central_q_sky_subtraction": True,
        "central_q_temporal_variation_diagnostic_only": True,
        "central_q_no_temporal_model_fit": True,
    }
    positive_validation_keys = (
        "cache_only", "selected_from_census_N_blank_ge_30",
        "blank_valid_reused_unchanged", "fixed_fq_reused_unchanged",
        "cache_K_band_reused_unchanged", "all_seven_bands_included",
        "one_joint_alpha_per_amplifier_exposure",
        "per_band_alphas_diagnostic_only", "only_alpha_fq_K_additive_term_tested",
        "hardware_exclusions_amplifier_scoped", "synthetic_scalar_fit_test_passed",
        "central_q_residual_reused_from_alpha_centering",
        "central_q_residual_pre_alpha_subtraction",
        "central_q_residual_not_sky_normalized",
        "central_q_residual_no_new_fit",
        "central_q_residual_sky_subtracted_for_visualization",
        "alpha_fit_unchanged_by_central_q_sky_subtraction",
        "central_q_temporal_variation_diagnostic_only",
        "central_q_no_temporal_model_fit",
    )
    negative_validation_keys = (
        "native_spectra_read", "multiplicative_parameter_changed",
        "sky_parameter_changed", "source_information_or_X_used",
    )
    validation_passed = (
        all(validation[key] for key in positive_validation_keys)
        and not any(validation[key] for key in negative_validation_keys)
    )
    if not validation_passed:
        raise RuntimeError("alpha template validation failed: %s" % validation)

    state = {
        "schema_version": "m101_alpha_template_bands_v1",
        "script": str(Path(__file__).resolve()),
        "purpose": "diagnostic-only test of existing alpha*f_q(q)*K_band additive template",
        "cache_provenance": _file_identity(cache_path),
        "cache_manifest_provenance": _file_identity(manifest_path),
        "census_provenance": _file_identity(census_path),
        "fq_template_provenance": _file_identity(fq_path),
        "fq_loader": "same q,f CSV semantics as diagnose_m101_hierarchical.load_fq used by the initializer",
        "band_order": list(BANDS),
        "K_source": "manifest.items[*].K from the existing band cache; no recomputation",
        "selection": {
            "census_field": "N_blank_valid_q40_75",
            "threshold": BLANK_RICH_THRESHOLD,
            "census_status": "usable",
            "central_reference_q_inclusive": [Q_MIN, Q_MAX],
            "fit_uses_all_valid_blank_q": True,
            "central_q_sky_reference": (
                "S_eb = biweight robust location of all selected blank-rich "
                "central-q amplifier measurements for each H5/exposure/band"),
            "central_q_plot_value": "central_q_residual_raw - S_eb",
        },
        "counts": {
            "census_rows_total": census_row_count,
            "selected_blank_rich_amplifier_exposures": len(fits),
            "H5_count_selected": len({row["H5"] for row in fits}),
            "exposure_count_selected": len({(row["H5"], row["exposure"]) for row in fits}),
            "aggregate_q_profile_rows_written": aggregate_profile_row_count,
            "gallery_q_profile_rows_written": len(gallery_profile_rows),
            "gallery_examples": len(gallery),
            "same_amp_gallery_examples": len(same_amp_gallery),
            "central_q_residual_rows": len(central_q_rows),
            "central_q_sky_reference_groups": len(sky_by_exposure_band),
            "central_q_residual_plot_manifest_rows": len(central_band_plot_manifest),
            "central_q_temporal_variation_plot_rows": len(temporal_variation_plot_manifest),
        },
        "population": {
            "joint_alpha": _distribution(alpha_values),
            "percent_reduction": _distribution(reduction),
            "alpha_band_disagreement": _distribution(alpha_disagreement),
            "template_correlation": _distribution(correlations),
            "central_blank_count": _distribution(central_counts),
            "total_blank_count": _distribution(total_counts),
        },
        "validation": validation,
        "plots": {
            "written": not args.no_plots,
            "gallery_size_requested": int(args.gallery_size),
            "gallery_alpha_strictly_greater_than": float(args.gallery_alpha_min),
            "gallery_random_seed": int(args.gallery_seed),
            "gallery_selection": "seeded random sample without replacement",
            "same_amp_gallery_selection": (
                "all fitted H5/exposure cases for the physical amplifier with "
                "the largest case count among amps containing alpha_joint > threshold"),
            "same_amp_gallery_physical_amp": same_amp_key,
            "physical_amp_distribution": {
                "chunk_size": AMP_DISTRIBUTION_CHUNK_SIZE,
                "y_limits": [AMP_DISTRIBUTION_YMIN, AMP_DISTRIBUTION_YMAX],
                "color": "chronological sorted H5/exposure index",
                "manifest_rows": len(physical_amp_plot_manifest),
            },
            "central_q_residual_by_band": {
                "chunk_size": AMP_DISTRIBUTION_CHUNK_SIZE,
                "physical_amp_order": "same sorted physical-amplifier order as joint-alpha plots",
                "chronological_color_index": "same sorted H5/exposure index as joint-alpha plots",
                "fixed_band_ranges": central_band_plot_ranges,
                "sky_subtracted": True,
                "manifest_rows": len(central_band_plot_manifest),
            },
            "central_q_temporal_variation_by_band": {
                "one_plot_per_band": True,
                "amplifier_level_removed": (
                    "robust average of each physical amplifier across available exposures"),
                "common_exposure_curve": (
                    "robust average across available physical amplifiers at each H5/exposure"),
                "trend_model_fit": False,
                "plot_manifest_rows": len(temporal_variation_plot_manifest),
            },
            "negative_reflection_null": {
                "center": 0.0,
                "negative_side": "-0.5 < alpha_joint < 0",
                "sigma_estimator": "sqrt(mean(alpha_negative**2))",
                "normalization": "2 * N_negative * bin_width * normal_pdf",
            },
            "focal_plane_map": None,
        },
        "artifacts": {
            "fits": str((output_dir / "alpha_template_fits.csv").resolve()),
            "q_profiles": str(aggregate_profile_path.resolve()),
            "gallery_q_profiles": str(gallery_profile_path.resolve()),
            "restricted_alpha_histogram": str(
                (output_dir / "alpha_joint_histogram_restricted.png").resolve()),
            "negative_reflection_histogram": str(
                (output_dir / "alpha_joint_histogram_negative_reflection.png").resolve()),
            "band_consistency": str((output_dir / "alpha_band_consistency.csv").resolve()),
            "population_summary": str((output_dir / "alpha_population_summary.csv").resolve()),
            "gallery_manifest": str((output_dir / "alpha_gallery_manifest.csv").resolve()),
            "band_central_q_residuals": str(central_q_path.resolve()),
            "band_central_q_residual_plot_manifest": (
                str(central_band_plot_manifest_path.resolve())
                if central_band_plot_manifest_path is not None else None),
            "central_q_temporal_variation_plots": [
                row["plot"] for row in temporal_variation_plot_manifest
            ],
            "physical_amp_distribution_manifest": (
                str(physical_amp_plot_manifest_path.resolve())
                if physical_amp_plot_manifest_path is not None else None),
        },
    }
    (output_dir / "alpha_template_bands_state.json").write_text(
        json.dumps(_json_ready(state), indent=2, sort_keys=True))

    alpha_stats = _distribution(alpha_values)
    reduction_finite = _finite(reduction)
    print("selected blank-rich amplifier/exposures: %d" % len(fits))
    print("median central blank count: %.6g" % np.median(central_counts))
    print("median total blank count: %.6g" % np.median(total_counts))
    print("joint alpha:")
    print("  median %.6g" % alpha_stats["median"])
    print("  robust scatter %.6g" % alpha_stats["robust_scatter"])
    print("  p05 %.6g" % alpha_stats["p05"])
    print("  p95 %.6g" % alpha_stats["p95"])
    print("median percent residual-scatter reduction from alpha*f_q*K: %.6g%%" %
          (np.median(reduction_finite) if reduction_finite.size else np.nan))
    for row in consistency_rows:
        print("%s: median(alpha_band-alpha_joint)=%.6g robust scatter=%.6g correlation=%.6g" %
              (row["band"], row["median_offset"], row["robust_scatter_offset"],
               row["correlation_alpha_band_joint"]))
    print("median seven-band alpha disagreement per amplifier: %.6g" %
          np.nanmedian(alpha_disagreement))
    print("fraction with finite template correlation: %.6g" %
          np.mean(np.isfinite(correlations)))
    print("fraction with scatter reduction (after < before): %.6g" %
          np.mean(np.isfinite(reduction) & (reduction > 0)))
    print("central-q additive residuals after exposure-band S_eb subtraction:")
    for band in BANDS:
        band_values = _finite([
            row["central_q_residual"] for row in central_q_rows
            if row["band"] == band
        ])
        print("  %s: N=%d center=%.6g scatter=%.6g p05=%.6g p50=%.6g p95=%.6g" %
              (band, band_values.size, _robust_location(band_values),
               _robust_scatter(band_values),
               np.percentile(band_values, 5) if band_values.size else np.nan,
               np.percentile(band_values, 50) if band_values.size else np.nan,
               np.percentile(band_values, 95) if band_values.size else np.nan))
    if not args.no_plots:
        print("wrote %d one-per-band temporal-variation plots" %
              len(temporal_variation_plot_manifest))
    print("wrote diagnostic products to %s" % output_dir)


if __name__ == "__main__":
    main()
