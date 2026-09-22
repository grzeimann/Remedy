#!/usr/bin/env python3
"""Report-only M101 exposure-scale and post-scale amplifier-QA diagnostic.

This diagnostic uses the current-model calibration from
``make_m101_cube_current_model`` through its final exposure sky subtraction,
measures the established physical-amplifier-collapsed surface-brightness
scale, applies exactly two median-one relative SB iterations, and reruns the
authoritative final amplifier QA on the iteration-2 corrected spectra.

The iteration-2 IMAGE_QA residuals are also reported in direct gray/common
and ON-OFF differential c/q coordinates.  Those coordinates are diagnostic
only and are never used to alter the spectra or reject amplifiers.

The earlier compact-source experiment was diagnostic-only and is superseded
by the corrected amplifier-collapsed surface-brightness estimator. It is not
part of this active product path.

No science cube, LSF product, or amplifier rejection is produced here.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, field
from datetime import datetime, timezone
import glob
import json
from pathlib import Path
import tempfile
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from astrometry import Astrometry
from extract import Extract
import make_m101_cube_current_model as cube_model
import m101_native_data as native


WAVE = np.asarray(cube_model.DEF_WAVE, dtype=float)
EXPOSURES = (1, 2, 3)
N_EXPECTED_H5 = 19
N_EXPECTED_SHOTS = 57
SB_BANDS = ("ON", "OFF")
FIBER_AREA = float(cube_model.FIBER_AREA)
EXPECTED_FIBER_AREA = float(np.pi * 0.75 ** 2)
BOOTSTRAP_SEED = 110101
DEFAULT_BOOTSTRAP_DRAWS = 300


@dataclass
class Shot:
    h5: str
    exposure: int
    survey: dict
    ra: np.ndarray
    dec: np.ndarray
    positions: dict = field(default_factory=dict)
    values: dict = field(default_factory=dict)
    sb_predictions: dict = field(default_factory=dict)
    amp_keys: np.ndarray | None = None
    calibration_summary: dict = field(default_factory=dict)


def _json_ready(value):
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_json_ready(v) for v in value.tolist()]
    if isinstance(value, np.generic):
        return _json_ready(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _write_csv(path, rows, fields=None):
    rows = list(rows)
    path = Path(path)
    if fields is None:
        fields = []
        for row in rows:
            for key in row:
                if key not in fields:
                    fields.append(key)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            clean = {}
            for key in fields:
                value = row.get(key, "")
                if isinstance(value, np.generic):
                    value = value.item()
                clean[key] = value
            writer.writerow(clean)


def _finite(values):
    values = np.asarray(values, dtype=float)
    return values[np.isfinite(values)]


def _file_identity(path):
    path = Path(path).expanduser().resolve()
    stat = path.stat()
    return {"path": str(path), "size": int(stat.st_size),
            "mtime_ns": int(stat.st_mtime_ns)}


def _resolve_h5files(patterns):
    paths = []
    for pattern in patterns:
        matches = glob.glob(str(Path(pattern).expanduser()))
        if not matches and Path(pattern).expanduser().is_file():
            matches = [str(Path(pattern).expanduser())]
        paths.extend(Path(value).resolve() for value in matches)
    unique = {path.name: path for path in paths}
    result = [unique[name] for name in sorted(unique)]
    if not result:
        raise ValueError("no H5 files matched: %s" % " ".join(patterns))
    if any(path.name == "20200523_0000023.h5" for path in result):
        raise ValueError("20200523_0000023.h5 is explicitly excluded from M101")
    return result


def _effective_wavelengths(responses):
    responses = np.asarray(responses, dtype=float)
    totals = np.sum(responses, axis=1)
    return np.divide(responses @ WAVE, totals,
                     out=np.full(responses.shape[0], np.nan), where=totals != 0)


def _robust_scale(values):
    values = _finite(values)
    if values.size < 2:
        return np.nan
    center = float(np.median(values))
    mad = 1.4826 * float(np.median(np.abs(values - center)))
    if np.isfinite(mad) and mad > 0.0:
        return mad
    scale = float(np.std(values, ddof=1))
    return scale if np.isfinite(scale) and scale > 0.0 else 0.0


def _origin_huber(x, y, max_iter=30, tuning=1.345, frequency=None):
    """Deterministic Huber IRLS slope through the origin."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if frequency is None:
        frequency = np.ones(x.shape, dtype=float)
    else:
        frequency = np.asarray(frequency, dtype=float)
        if frequency.shape != x.shape:
            raise ValueError("frequency must match x and y")
    valid = (np.isfinite(x) & np.isfinite(y) & np.isfinite(frequency) &
             (frequency > 0.0))
    x, y, frequency = x[valid], y[valid], frequency[valid]
    denominator = float(np.sum(x * x)) if x.size else 0.0
    if denominator <= 0.0 or not np.isfinite(denominator):
        return {"slope": np.nan, "scatter": np.nan, "n": int(x.size)}
    slope = float(np.sum(frequency * x * y) /
                  np.sum(frequency * x * x))
    for _ in range(max_iter):
        residual = y - slope * x
        scale = _robust_scale(residual)
        if not np.isfinite(scale) or scale <= 0.0:
            weights = np.ones_like(residual)
        else:
            threshold = tuning * scale
            absolute = np.abs(residual)
            weights = np.ones_like(residual)
            large = absolute > threshold
            weights[large] = threshold / absolute[large]
        weights *= frequency
        denominator = float(np.sum(weights * x * x))
        if denominator <= 0.0 or not np.isfinite(denominator):
            break
        updated = float(np.sum(weights * x * y) / denominator)
        if abs(updated - slope) <= 1.0e-12 * max(1.0, abs(slope)):
            slope = updated
            break
        slope = updated
    residual = y - slope * x
    return {"slope": slope, "scatter": _robust_scale(residual),
            "n": int(x.size), "residual": residual, "x": x, "y": y}


def _bootstrap_quantiles(fit_values, groups, draws, seed):
    """Bootstrap a through-origin fit by resampling whole named groups."""
    x, y = (np.asarray(fit_values[0], dtype=float),
            np.asarray(fit_values[1], dtype=float))
    groups = np.asarray(groups, dtype=object)
    valid = np.isfinite(x) & np.isfinite(y)
    x, y, groups = x[valid], y[valid], groups[valid]
    unique, group_id = np.unique(groups, return_inverse=True)
    if unique.size < 2 or x.size < 2:
        return {"p16": np.nan, "p50": np.nan, "p84": np.nan,
                "sigma": np.nan,
                "reason": "fewer than two supported bootstrap groups"}
    rng = np.random.default_rng(int(seed))
    values = []
    for _ in range(int(draws)):
        selected = rng.integers(0, unique.size, size=unique.size)
        frequency_by_group = np.bincount(selected, minlength=unique.size)
        fit = _origin_huber(
            x, y, frequency=frequency_by_group[group_id], max_iter=12)
        if np.isfinite(fit["slope"]):
            values.append(fit["slope"])
    if len(values) < max(20, int(draws * 0.5)):
        return {"p16": np.nan, "p50": np.nan, "p84": np.nan,
                "sigma": np.nan,
                "reason": "too few finite bootstrap refits (%d/%d)" %
                          (len(values), int(draws))}
    p16, p50, p84 = np.percentile(np.asarray(values), (16, 50, 84))
    return {"p16": float(p16), "p50": float(p50), "p84": float(p84),
            "sigma": float(0.5 * (p84 - p16)),
            "reason": "available", "n_draws": int(len(values))}


def _survey_value(survey, key):
    try:
        return float(survey[key])
    except (KeyError, TypeError, ValueError, OverflowError):
        return np.nan


def _adr_positions(ra, dec, survey, response):
    """Use the current IMAGE_QA ADR convention at one band wavelength."""
    survey_ra = _survey_value(survey, "ra")
    survey_dec = _survey_value(survey, "dec")
    pa = _survey_value(survey, "pa")
    if not np.all(np.isfinite([survey_ra, survey_dec, pa])):
        raise ValueError("Survey RA/Dec/PA is invalid")
    astrometry = Astrometry(survey_ra, survey_dec, pa, 0.0, 0.0)
    extractor = Extract(wave=WAVE)
    extractor.get_ADR_RAdec(astrometry)
    effective = float(_effective_wavelengths(np.asarray(response)[None, :])[0])
    adr_ra = float(np.interp(effective, WAVE, extractor.ADRra))
    adr_dec = float(np.interp(effective, WAVE, extractor.ADRdec))
    corrected_ra = (np.asarray(ra, dtype=float) - adr_ra / 3600.0 /
                    np.cos(np.deg2rad(survey_dec)))
    corrected_dec = np.asarray(dec, dtype=float) - adr_dec / 3600.0
    return np.column_stack((corrected_ra, corrected_dec))


def _derive_geometry(shots, pixel_scale, image_center_size):
    if image_center_size:
        return cube_model._parse_image_geometry(image_center_size, pixel_scale)
    ra = np.concatenate([shot.ra for shot in shots])
    dec = np.concatenate([shot.dec for shot in shots])
    valid = np.isfinite(ra) & np.isfinite(dec)
    if not np.any(valid):
        raise ValueError("no finite fiber coordinates available for image geometry")
    ra0 = float(np.nanmedian(ra[valid]))
    dec0 = float(np.nanmedian(dec[valid]))
    x = (ra[valid] - ra0) * np.cos(np.deg2rad(dec0)) * 3600.0
    y = (dec[valid] - dec0) * 3600.0
    extent = max(float(np.nanmax(np.abs(x))), float(np.nanmax(np.abs(y))))
    size_arcmin = max(1.0, (2.0 * extent + 40.0) / 60.0)
    value = "%.10f,%.10f,%.10f" % (ra0, dec0, size_arcmin)
    return cube_model._parse_image_geometry(value, pixel_scale)


def _pixel_positions(positions, tp):
    x, y = tp.wcs_world2pix(positions[:, 0], positions[:, 1], 1)
    return np.column_stack((np.asarray(x, dtype=float),
                            np.asarray(y, dtype=float)))


def _build_image_cache(shots, band, tp, xg, yg, pixel_scale, path):
    """Build only collapsed per-shot 2-D images on a disk-backed cache."""
    shape = (len(shots), len(yg), len(xg))
    stack = np.memmap(path, dtype=np.float32, mode="w+", shape=shape)
    for index, shot in enumerate(shots):
        pixel = _pixel_positions(shot.positions[band], tp)
        values = np.asarray(shot.values[band], dtype=float)
        local = np.arange(pixel.shape[0], dtype=np.int64)
        stack[index] = cube_model._qa_shot_images(
            pixel, values, [local], xg, yg, pixel_scale)[0]
        if index % 10 == 0:
            print("[exposure-scale] %s image cache %d/%d" %
                  (band, index + 1, len(shots)), flush=True)
    stack.flush()
    return stack


def _loo_samples(stack, target_index, sky_positions, tp, minimum_support=2):
    """Return exact pixel-wise LOO median and independent-shot support."""
    pixel = _pixel_positions(np.asarray(sky_positions, dtype=float), tp)
    x, y = pixel[:, 0], pixel[:, 1]
    xi = np.rint(x).astype(int) - 1
    yi = np.rint(y).astype(int) - 1
    result = np.full(x.shape, np.nan, dtype=float)
    support_result = np.zeros(x.shape, dtype=np.int16)
    good = (np.isfinite(x) & np.isfinite(y) &
            (xi >= 0) & (xi < stack.shape[2]) &
            (yi >= 0) & (yi < stack.shape[1]))
    if not np.any(good):
        return result, support_result
    values = np.asarray(stack[:, yi[good], xi[good]], dtype=float)
    values = np.delete(values, int(target_index), axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        predicted = np.nanmedian(values, axis=0)
    support = np.sum(np.isfinite(values), axis=0)
    predicted[support < int(minimum_support)] = np.nan
    result[good] = predicted
    support_result[good] = support.astype(np.int16)
    return result, support_result


def _amp_keys_for_rows(groups, n_rows):
    result = np.empty(n_rows, dtype=object)
    result[:] = "unknown"
    for group in groups:
        key = "%d/%d/%d/%s" % (int(group["specid"]),
                                int(group["ifuslot"]),
                                int(group["ifuid"]), str(group["amp"]))
        result[np.asarray(group["indices"], dtype=int)] = key
    return result


def _robust_collapse(values):
    values = _finite(values)
    return float(np.median(values)) if values.size else np.nan


def _predictor_diagnostics(predictors):
    predictors = _finite(predictors)
    if predictors.size == 0:
        return {"max_predictor_leverage_fraction": np.nan,
                "N_effective_leverage": np.nan}
    squared = predictors * predictors
    total = float(np.sum(squared))
    fourth = float(np.sum(squared * squared))
    if total <= 0.0 or not np.isfinite(total):
        return {"max_predictor_leverage_fraction": np.nan,
                "N_effective_leverage": np.nan}
    return {
        "max_predictor_leverage_fraction": float(np.max(squared) / total),
        "N_effective_leverage": float(total * total / fourth)
        if fourth > 0.0 and np.isfinite(fourth) else np.nan,
    }


def _fit_sb(shot, bootstrap_draws, seed):
    """Fit the established robust physical-amplifier SB estimator."""
    result = {
        "g_SB_ON": np.nan, "g_SB_OFF": np.nan, "g_SB_joint": np.nan,
        "N_SB_ON": 0, "N_SB_OFF": 0, "N_SB_BOTH": 0,
        "max_abs_predicted_SB_ON": np.nan,
        "max_abs_predicted_SB_OFF": np.nan,
        "sqrt_sum_predicted_squared_SB_ON": np.nan,
        "sqrt_sum_predicted_squared_SB_OFF": np.nan,
        "sqrt_sum_predicted_squared_SB_joint": np.nan,
        "residual_scatter_SB_ON": np.nan,
        "residual_scatter_SB_OFF": np.nan,
        "residual_scatter_SB_joint": np.nan,
        "abs_g_SB_ON_minus_g_SB_OFF": np.nan,
        "g_SB_p16": np.nan, "g_SB_p50": np.nan, "g_SB_p84": np.nan,
        "sigma_g_SB": np.nan, "SB_bootstrap_reason": "unavailable",
        "SB_max_predictor_leverage_fraction": np.nan,
        "SB_N_effective_leverage": np.nan,
    }
    fits = {}
    for band in SB_BANDS:
        rows = []
        for amp in np.unique(shot.amp_keys):
            members = np.asarray(shot.amp_keys == amp)
            valid = (members & np.isfinite(shot.values[band]) &
                     np.isfinite(shot.sb_predictions[band]))
            if not np.any(valid):
                continue
            observed = _robust_collapse(shot.values[band][valid])
            predicted = _robust_collapse(
                shot.sb_predictions[band][valid] * FIBER_AREA)
            if np.isfinite(observed) and np.isfinite(predicted):
                rows.append((amp, observed, predicted))
        groups = np.asarray([row[0] for row in rows], dtype=object)
        observed = np.asarray([row[1] for row in rows], dtype=float)
        predicted = np.asarray([row[2] for row in rows], dtype=float)
        fit = _origin_huber(predicted, observed)
        fits[band] = (fit, groups, observed, predicted)
        result["g_SB_%s" % band] = fit["slope"]
        result["N_SB_%s" % band] = int(predicted.size)
        if predicted.size:
            result["max_abs_predicted_SB_%s" % band] = float(
                np.max(np.abs(predicted)))
            result["sqrt_sum_predicted_squared_SB_%s" % band] = float(
                np.sqrt(np.sum(predicted ** 2)))
        result["residual_scatter_SB_%s" % band] = fit["scatter"]

    on_groups = fits["ON"][1]
    off_groups = fits["OFF"][1]
    result["N_SB_BOTH"] = int(len(set(on_groups.tolist()).intersection(
        set(off_groups.tolist()))))
    joint_pred = np.concatenate((fits["ON"][3], fits["OFF"][3]))
    joint_obs = np.concatenate((fits["ON"][2], fits["OFF"][2]))
    joint_groups = np.concatenate((on_groups, off_groups))
    joint_fit = _origin_huber(joint_pred, joint_obs)
    result["g_SB_joint"] = joint_fit["slope"]
    result["residual_scatter_SB_joint"] = joint_fit["scatter"]
    if joint_pred.size:
        result["sqrt_sum_predicted_squared_SB_joint"] = float(
            np.sqrt(np.sum(joint_pred ** 2)))
    if np.isfinite(result["g_SB_ON"]) and np.isfinite(result["g_SB_OFF"]):
        result["abs_g_SB_ON_minus_g_SB_OFF"] = abs(
            result["g_SB_ON"] - result["g_SB_OFF"])
    bootstrap = _bootstrap_quantiles((joint_pred, joint_obs), joint_groups,
                                     bootstrap_draws, seed)
    result["g_SB_p16"] = bootstrap["p16"]
    result["g_SB_p50"] = bootstrap["p50"]
    result["g_SB_p84"] = bootstrap["p84"]
    result["sigma_g_SB"] = bootstrap["sigma"]
    result["SB_bootstrap_reason"] = bootstrap["reason"]
    result["SB_source_leverage"] = result[
        "sqrt_sum_predicted_squared_SB_joint"]
    leverage = _predictor_diagnostics(joint_pred)
    result["SB_max_predictor_leverage_fraction"] = leverage[
        "max_predictor_leverage_fraction"]
    result["SB_N_effective_leverage"] = leverage["N_effective_leverage"]
    return result


def _process_sb_predictions(shots, stack_by_band, tp):
    for band in SB_BANDS:
        for shot_index, shot in enumerate(shots):
            predictions, _support = _loo_samples(
                stack_by_band[band], shot_index, shot.positions[band], tp)
            shot.sb_predictions[band] = predictions


def _sb_exposure_row(shot, fit, chronological_index):
    row = {"H5": shot.h5, "exposure": int(shot.exposure),
           "chronological_exposure_index": int(chronological_index)}
    row.update(fit)
    row["g_SB_joint_raw"] = row["g_SB_joint"]
    return row


def _attach_exposure_scale(row, scale_rows_by_key):
    scale = scale_rows_by_key[(row["H5"], int(row["exposure"]))]
    row["g_SB_joint_raw"] = scale["g_SB_joint_raw"]
    row["g_SB_relative"] = scale["g_SB_relative"]
    row["sigma_g_SB"] = scale["sigma_g_SB"]
    row["g_SB_center"] = scale["g_SB_center"]
    row["exposure_correction"] = scale["exposure_correction"]
    row["fractional_sigma_g_SB"] = scale["fractional_sigma_g_SB"]


def _qa_grouped_stats(rows):
    stats = {}
    for h5 in sorted(set(row["H5"] for row in rows)):
        for exposure in EXPOSURES:
            selected = [row for row in rows
                        if row["H5"] == h5 and int(row["exposure"]) == exposure]
            stats[(h5, exposure)] = {}
            for band in SB_BANDS:
                finite = _finite([row["image_fractional_%s" % band]
                                  for row in selected])
                stats[(h5, exposure)]["%s_median" % band] = (
                    float(np.median(finite)) if finite.size else np.nan)
                stats[(h5, exposure)]["%s_scatter" % band] = _robust_scale(finite)
    return stats


def _robust_plot_limits(values, q_lo=0.01, q_hi=0.99, padding=0.08,
                        include_zero=False, include_values=(), symmetric=False):
    """Return display limits without changing the values being plotted."""
    values = np.asarray(values, dtype=float).ravel()
    values = values[np.isfinite(values)]
    if values.size == 0:
        values = np.asarray([0.0])
    lo, hi = np.percentile(values, (q_lo * 100.0, q_hi * 100.0))
    included = np.asarray(list(include_values), dtype=float)
    included = included[np.isfinite(included)]
    if symmetric:
        bound = max(abs(float(lo)), abs(float(hi)),
                    float(np.max(np.abs(included))) if included.size else 0.0,
                    0.5 * float(hi - lo))
        bound = max(bound * (1.0 + padding), 1.0e-6)
        return -bound, bound
    span = max(float(hi - lo), 1.0e-6)
    lo -= padding * span
    hi += padding * span
    if include_zero:
        lo = min(lo, 0.0)
        hi = max(hi, 0.0)
    if included.size:
        lo = min(lo, float(np.min(included)))
        hi = max(hi, float(np.max(included)))
    if hi <= lo:
        center = 0.5 * (lo + hi)
        half = max(abs(center) * 0.1, 1.0e-6)
        lo, hi = center - half, center + half
    return float(lo), float(hi)


def _plot_offscale_markers(ax, x, y, x_limits, y_limits, color,
                           marker="^", label=None):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    finite = np.isfinite(x) & np.isfinite(y)
    outside = (finite & ((x < x_limits[0]) | (x > x_limits[1]) |
                         (y < y_limits[0]) | (y > y_limits[1])))
    if np.any(outside):
        ax.scatter(np.clip(x[outside], *x_limits),
                   np.clip(y[outside], *y_limits),
                   s=34, marker=marker, facecolors="none",
                   edgecolors=color, linewidths=0.9, label=label,
                   zorder=5)
    return int(np.count_nonzero(outside))


def _plot_closure(exposure_rows, pre_stats, post_stats, output_path):
    x = np.asarray([row["chronological_exposure_index"]
                    for row in exposure_rows], dtype=float)
    raw = np.asarray([row["g_SB_joint_raw"] for row in exposure_rows], dtype=float)
    residual = np.asarray([row["g_SB_residual_joint"]
                           for row in exposure_rows], dtype=float)
    raw_sigma = np.asarray([row["sigma_g_SB"] for row in exposure_rows], dtype=float)
    residual_sigma = np.asarray([row["sigma_g_SB_residual"]
                                 for row in exposure_rows], dtype=float)
    pre_on = np.asarray([pre_stats[(row["H5"], row["exposure"])]
                         ["ON_scatter"] for row in exposure_rows])
    post_on = np.asarray([post_stats[(row["H5"], row["exposure"])]
                          ["ON_scatter"] for row in exposure_rows])
    pre_off = np.asarray([pre_stats[(row["H5"], row["exposure"])]
                          ["OFF_scatter"] for row in exposure_rows])
    post_off = np.asarray([post_stats[(row["H5"], row["exposure"])]
                           ["OFF_scatter"] for row in exposure_rows])
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True,
                             gridspec_kw={"height_ratios": (2, 1)})
    ax = axes[0]
    for values, errors, label, color, marker in (
            (raw, raw_sigma, "raw g_SB_joint", "tab:blue", "o"),
            (residual, residual_sigma, "post-SB g_SB_residual_joint",
             "tab:green", "s")):
        finite = np.isfinite(values)
        with_error = finite & np.isfinite(errors)
        no_error = finite & ~np.isfinite(errors)
        ax.errorbar(x[with_error], values[with_error], yerr=errors[with_error],
                    fmt=marker, ms=4, lw=0.8, capsize=2,
                    color=color, label=label)
        ax.plot(x[no_error], values[no_error], marker, ms=4, color=color)
    ax.axhline(1.0, color="0.35", lw=0.9)
    top_limits = _robust_plot_limits(
        np.concatenate((raw[np.isfinite(raw)], residual[np.isfinite(residual)])),
        include_values=(1.0,))
    ax.set_ylim(*top_limits)
    offscale_top = 0
    for values, color, marker in ((raw, "tab:blue", "v"),
                                  (residual, "tab:green", "^") ):
        offscale_top += _plot_offscale_markers(
            ax, x, values, (x.min() - 0.5, x.max() + 0.5), top_limits,
            color, marker=marker)
    if offscale_top:
        ax.text(0.99, 0.03, "%d off robust y-range" % offscale_top,
                transform=ax.transAxes, ha="right", va="bottom", fontsize=8)
    ax.set_ylabel("gray multiplicative scale")
    ax.set_title("M101 SB exposure-scale closure")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8, loc="best")
    ax = axes[1]
    for values, label, color, marker, linestyle in (
            (pre_on, "pre ON scatter", "tab:blue", "o", "--"),
            (post_on, "post ON scatter", "tab:blue", "o", "-"),
            (pre_off, "pre OFF scatter", "tab:orange", "s", "--"),
            (post_off, "post OFF scatter", "tab:orange", "s", "-")):
        finite = np.isfinite(values)
        ax.plot(x[finite], values[finite], marker + linestyle, ms=3,
                lw=0.8, color=color, label=label)
    bottom_values = np.concatenate([
        values[np.isfinite(values)]
        for values in (pre_on, post_on, pre_off, post_off)])
    bottom_limits = _robust_plot_limits(bottom_values, include_zero=True)
    ax.set_ylim(*bottom_limits)
    offscale_bottom = 0
    for values, color, marker in ((pre_on, "tab:blue", "v"),
                                  (post_on, "tab:blue", "^"),
                                  (pre_off, "tab:orange", "v"),
                                  (post_off, "tab:orange", "^") ):
        offscale_bottom += _plot_offscale_markers(
            ax, x, values, (x.min() - 0.5, x.max() + 0.5), bottom_limits,
            color, marker=marker)
    if offscale_bottom:
        ax.text(0.99, 0.03, "%d off robust y-range" % offscale_bottom,
                transform=ax.transAxes, ha="right", va="bottom", fontsize=8)
    ax.set_xlabel("chronological exposure index")
    ax.set_ylabel("amplifier IMAGE_QA robust scatter")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=7, ncol=2, loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _plot_post_fractional(rows, output_path):
    on = np.asarray([row["image_fractional_ON"] for row in rows], dtype=float)
    off = np.asarray([row["image_fractional_OFF"] for row in rows], dtype=float)
    finite = np.isfinite(on) & np.isfinite(off)
    fig, ax = plt.subplots(figsize=(7, 7))
    if np.any(finite):
        ax.scatter(on[finite], off[finite], s=13, alpha=0.55,
                   color="tab:purple")
        limits = _robust_plot_limits(
            np.concatenate((on[finite], off[finite])), symmetric=True)
        ax.set_xlim(*limits)
        ax.set_ylim(*limits)
        ax.set_aspect("equal", adjustable="box")
        offscale = _plot_offscale_markers(
            ax, on[finite], off[finite], limits, limits,
            "tab:red", marker="^")
        rank = np.argsort(np.maximum(np.abs(on[finite]), np.abs(off[finite])))[::-1]
        indices = np.flatnonzero(finite)
        annotated = 0
        for index in rank:
            row = rows[indices[index]]
            is_offscale = (
                on[indices[index]] < limits[0] or
                on[indices[index]] > limits[1] or
                off[indices[index]] < limits[0] or
                off[indices[index]] > limits[1])
            if is_offscale:
                continue
            label = "%s/e%s %s/%s/%s/%s" % (
                Path(row["H5"]).stem, row["exposure"], row["SPECID"],
                row["IFUSLOT"], row["IFUID"], row["AMP"])
            ax.annotate(label, (on[indices[index]], off[indices[index]]),
                        fontsize=5)
            annotated += 1
            if annotated >= 10:
                break
        if offscale:
            ax.text(0.99, 0.02, "%d off robust 1-99%% range" % offscale,
                    transform=ax.transAxes, ha="right", va="bottom",
                    fontsize=8)
    ax.axhline(0.0, color="0.7", lw=0.8)
    ax.axvline(0.0, color="0.7", lw=0.8)
    ax.set_xlabel("post image_fractional_ON")
    ax.set_ylabel("post image_fractional_OFF")
    ax.set_title("Post-SB amplifier IMAGE_QA strongest residuals (robust range)")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _apply_exposure_relative_scales(records, scale_rows_by_key, field):
    """Apply one named relative-scale iteration to the in-memory O products."""
    for record in records:
        h5_name = record["h5file"].name
        for exposure in EXPOSURES:
            scale_row = scale_rows_by_key[(h5_name, exposure)]
            relative = float(scale_row[field])
            if not np.isfinite(relative) or relative == 0.0:
                raise RuntimeError(
                    "invalid relative SB scale for %s exposure %d (%s)" %
                    (h5_name, exposure, field))
            selected = record["labels"] == exposure
            record["spectra"][selected] = (
                record["spectra"][selected] / relative)
            record["errors"][selected] = (
                record["errors"][selected] / abs(relative))
            record["corrected_total"][selected] = (
                record["corrected_total"][selected] / relative)
            record["sky_by_exposure"][exposure] = (
                record["sky_by_exposure"][exposure] / relative)
        for exposure in EXPOSURES:
            selected = record["labels"] == exposure
            record["spectra"][selected] = (
                record["corrected_total"][selected] -
                record["sky_by_exposure"][exposure][None, :])


def _update_shot_values_from_records(shots, records, band_responses):
    """Refresh collapsed ON/OFF shot values after an O-space correction."""
    for record in records:
        collapsed, _ = cube_model.collapse_many(
            record["spectra"], band_responses)
        for shot in shots:
            if shot.h5 != record["h5file"].name:
                continue
            selected = record["labels"] == shot.exposure
            shot.values["ON"] = np.asarray(
                collapsed[selected, 5], dtype=np.float32)
            shot.values["OFF"] = np.asarray(
                collapsed[selected, 6], dtype=np.float32)
            shot.sb_predictions = {}


def _validate_cumulative_scales(records, scale_rows_by_key):
    """Check both the cumulative scale identity and O=corrected_total-final_sky."""
    max_direct_error = 0.0
    max_identity_error = 0.0
    n_direct = 0
    n_identity = 0
    for record in records:
        original = np.asarray(record["original_spectra"], dtype=float)
        actual = np.asarray(record["spectra"], dtype=float)
        for exposure in EXPOSURES:
            row = scale_rows_by_key[(record["h5file"].name, exposure)]
            cumulative = float(row["g1_rel"] * row["g2_rel"])
            if not np.isfinite(cumulative) or cumulative == 0.0:
                raise RuntimeError("invalid cumulative scale in validation")
            if not np.isclose(row["cumulative_relative_scale"], cumulative,
                              rtol=0.0, atol=1.0e-14):
                raise RuntimeError("cumulative scale identity failed for %s/e%d" %
                                   (record["h5file"].name, exposure))
            selected = record["labels"] == exposure
            expected = original[selected] / cumulative
            observed = actual[selected]
            finite = np.isfinite(expected) & np.isfinite(observed)
            if np.any(finite):
                difference = np.abs(expected[finite] - observed[finite])
                max_direct_error = max(max_direct_error,
                                       float(np.max(difference)))
                n_direct += int(np.count_nonzero(finite))
                if not np.allclose(expected[finite], observed[finite],
                                   rtol=5.0e-6, atol=5.0e-6):
                    raise RuntimeError(
                        "final spectra/cumulative correction failed for %s/e%d" %
                        (record["h5file"].name, exposure))
            # Reproduce the actual stored operation in float32.  Upcasting
            # both operands before subtraction would test a different
            # floating-point operation than the construction of spectra.
            final_sky = np.asarray(record["sky_by_exposure"][exposure],
                                   dtype=np.float32)
            corrected_total = np.asarray(record["corrected_total"][selected],
                                         dtype=np.float32)
            identity = (corrected_total - final_sky[None, :]).astype(float)
            finite_identity = np.isfinite(identity) & np.isfinite(observed)
            if np.any(finite_identity):
                difference = np.abs(identity[finite_identity] -
                                     observed[finite_identity])
                max_identity_error = max(max_identity_error,
                                         float(np.max(difference)))
                n_identity += int(np.count_nonzero(finite_identity))
                if not np.allclose(identity[finite_identity],
                                   observed[finite_identity],
                                   rtol=0.0, atol=5.0e-6):
                    raise RuntimeError("O=corrected_total-final_sky failed for %s/e%d" %
                                       (record["h5file"].name, exposure))
    return {
        "finite_direct_comparisons": int(n_direct),
        "finite_identity_comparisons": int(n_identity),
        "max_abs_direct_error": float(max_direct_error),
        "max_abs_identity_error": float(max_identity_error),
        "direct_formula": "final spectra = original post-sky spectra / (g1_rel*g2_rel)",
        "identity_formula": "O2 = corrected_total2 - final_sky2",
    }


def _add_cq_coordinates(rows):
    """Add direct IMAGE_QA and, where available, BLANK_QA c/q coordinates."""
    for row in rows:
        row["image_c_gray"] = np.nan
        row["image_q_color"] = np.nan
        row["image_abs_c_gray"] = np.nan
        row["image_abs_q_color"] = np.nan
        row["image_residual_radius"] = np.nan
        row["blank_c_gray"] = np.nan
        row["blank_q_color"] = np.nan
        image_on = row.get("image_fractional_ON", np.nan)
        image_off = row.get("image_fractional_OFF", np.nan)
        if np.isfinite(image_on) and np.isfinite(image_off):
            row["image_c_gray"] = 0.5 * (image_on + image_off)
            row["image_q_color"] = 0.5 * (image_on - image_off)
            row["image_abs_c_gray"] = abs(row["image_c_gray"])
            row["image_abs_q_color"] = abs(row["image_q_color"])
            row["image_residual_radius"] = float(np.hypot(
                row["image_c_gray"], row["image_q_color"]))
        blank_on = row.get("blank_fractional_ON", np.nan)
        blank_off = row.get("blank_fractional_OFF", np.nan)
        if np.isfinite(blank_on) and np.isfinite(blank_off):
            row["blank_c_gray"] = 0.5 * (blank_on + blank_off)
            row["blank_q_color"] = 0.5 * (blank_on - blank_off)
        # These are aliases for the frozen, already-authoritative c/q
        # decomposition.  Keep image_c_gray/image_q_color above unchanged;
        # the focal-plane diagnostic consumes the exact same floating values.
        row["c_ea"] = row["image_c_gray"]
        row["q_ea"] = row["image_q_color"]


def _physical_ifu_key(row):
    return (str(row["SPECID"]), str(row["IFUSLOT"]),
            str(row["IFUID"]), str(row["IFU_CODE"]))


def _cache_amp_key(h5_name, exposure, specid, ifuslot, ifuid, ifu_code, amp):
    return (Path(h5_name).name, int(exposure), int(specid), int(ifuslot),
            int(ifuid), int(ifu_code), str(amp).upper())


def _load_band_cache_amp_coordinates(cache_path, cache_manifest, h5files):
    """Load historical M101 x/y and collapse cached fibers per amplifier.

    The manifest slices are authoritative for the H5/exposure association.
    Integer amplifier values are decoded with the exact AMP_ORDER exported by
    the frozen band-cache producer; this diagnostic does not infer that map.
    """
    cache_path = Path(cache_path).expanduser().resolve()
    wanted = {Path(path).name for path in h5files}
    items = [item for item in cache_manifest.get("items", [])
             if str(item["h5_name"]) in wanted]
    found = {str(item["h5_name"]) for item in items}
    missing_h5 = sorted(wanted - found)
    if missing_h5:
        raise RuntimeError("band cache has no manifest slices for H5 files: %s" %
                           missing_h5)

    amp_order = tuple(cube_model.frozen_parent.AMP_ORDER)
    required = ("row_index", "ifu", "ifu_code", "amp", "x_arcmin", "y_arcmin")
    with np.load(cache_path, allow_pickle=False) as archive:
        missing = [name for name in required if name not in archive.files]
        if missing:
            raise ValueError("band cache lacks required focal-coordinate arrays: %s" %
                             missing)
        row_index = np.asarray(archive["row_index"])
        ifu = np.asarray(archive["ifu"])
        ifu_code = np.asarray(archive["ifu_code"])
        amp = np.asarray(archive["amp"])
        x_arcmin = np.asarray(archive["x_arcmin"], dtype=float)
        y_arcmin = np.asarray(archive["y_arcmin"], dtype=float)

    lengths = {array.shape[0] for array in
               (row_index, ifu, ifu_code, amp, x_arcmin, y_arcmin)}
    if len(lengths) != 1 or ifu.ndim != 2 or ifu.shape[1] != 3:
        raise ValueError("band-cache identity/coordinate arrays have incompatible shapes")

    coordinates = {}
    item_metadata = []
    for item in items:
        h5_name = str(item["h5_name"])
        exposure = int(item["exposure"])
        start, stop = int(item["start"]), int(item["stop"])
        if start < 0 or stop < start or stop > row_index.size:
            raise ValueError("invalid band-cache manifest slice for %s exposure %d" %
                             (h5_name, exposure))
        sl = slice(start, stop)
        item_ifu = ifu[sl]
        item_code = ifu_code[sl]
        item_amp = amp[sl]
        item_x = x_arcmin[sl]
        item_y = y_arcmin[sl]
        identity_rows = np.unique(
            np.column_stack((item_ifu, item_code, item_amp)), axis=0)
        item_amp_count = 0
        for identity in identity_rows:
            specid, ifuslot, ifuid = (int(value) for value in identity[:3])
            code = int(identity[3])
            amp_index = int(identity[4])
            if amp_index < 0 or amp_index >= len(amp_order):
                raise ValueError(
                    "band cache amplifier index %d is outside AMP_ORDER=%s" %
                    (amp_index, amp_order))
            amp_name = amp_order[amp_index]
            selected = ((item_ifu[:, 0] == specid) &
                        (item_ifu[:, 1] == ifuslot) &
                        (item_ifu[:, 2] == ifuid) &
                        (item_code == code) & (item_amp == amp_index))
            finite_x = item_x[selected][np.isfinite(item_x[selected])]
            finite_y = item_y[selected][np.isfinite(item_y[selected])]
            key = _cache_amp_key(
                h5_name, exposure, specid, ifuslot, ifuid, code, amp_name)
            if key in coordinates:
                raise ValueError("duplicate band-cache physical-amplifier key: %s" %
                                 (key,))
            coordinates[key] = {
                "X_amp": _robust_collapse(finite_x),
                "Y_amp": _robust_collapse(finite_y),
                "N_cached_fibers": int(np.count_nonzero(selected)),
                "N_finite_x": int(finite_x.size),
                "N_finite_y": int(finite_y.size),
                "amp_index": amp_index,
            }
            item_amp_count += 1
        item_metadata.append({
            "H5": h5_name,
            "exposure": exposure,
            "start": start,
            "stop": stop,
            "cached_rows": int(stop - start),
            "physical_amplifiers": int(item_amp_count),
        })

    return coordinates, {
        "coordinate_source": "m101_band_cache.npz x_arcmin/y_arcmin",
        "coordinate_units": "arcmin",
        "coordinate_semantics": (
            "exposure-centered tangent-plane coordinates used by historical "
            "Model-3 illumination fit"),
        "mechanical_fplane_coordinates_claimed": False,
        "amp_order_source": "cube_model.frozen_parent.AMP_ORDER",
        "amp_order": list(amp_order),
        "manifest_items_used": item_metadata,
        "physical_amplifier_coordinates": int(len(coordinates)),
    }


def _add_focal_coordinates(rows, coordinates):
    """Join cached historical fiber coordinates onto final QA rows."""
    missing = []
    for row in rows:
        key = _cache_amp_key(
            row["H5"], row["exposure"], row["SPECID"], row["IFUSLOT"],
            row["IFUID"], row["IFU_CODE"], row["AMP"])
        coords = coordinates.get(key)
        if (coords is None or not np.isfinite(coords["X_amp"]) or
                not np.isfinite(coords["Y_amp"])):
            missing.append((row.get("SPECID"), row.get("IFUSLOT"),
                            row.get("IFUID"), row.get("IFU_CODE"),
                            row.get("AMP"), row.get("H5"), row.get("exposure")))
            row["X_amp"] = np.nan
            row["Y_amp"] = np.nan
            row["focal_coordinate_status"] = "MISSING"
        else:
            row["X_amp"] = float(coords["X_amp"])
            row["Y_amp"] = float(coords["Y_amp"])
            row["N_cached_fibers"] = int(coords["N_cached_fibers"])
            row["N_finite_cached_x"] = int(coords["N_finite_x"])
            row["N_finite_cached_y"] = int(coords["N_finite_y"])
            row["focal_coordinate_status"] = "OK"
    if missing:
        raise RuntimeError("physical amplifiers missing cached x/y coordinates: %s" %
                           missing[:10])
    unique_amp = {_physical_amp_identity(row) for row in rows}
    bad = [row for row in rows if row["focal_coordinate_status"] != "OK"]
    finite_c = sum(np.isfinite(row.get("c_ea", np.nan)) for row in rows)
    finite_q = sum(np.isfinite(row.get("q_ea", np.nan)) for row in rows)
    finite_cq = sum(np.isfinite(row.get("c_ea", np.nan)) and
                    np.isfinite(row.get("q_ea", np.nan)) for row in rows)
    matched = [row for row in rows
               if np.isfinite(row.get("X_amp", np.nan)) and
               np.isfinite(row.get("Y_amp", np.nan))]
    if finite_cq != sum(np.isfinite(row.get("X_amp", np.nan)) and
                        np.isfinite(row.get("Y_amp", np.nan)) and
                        np.isfinite(row.get("c_ea", np.nan)) and
                        np.isfinite(row.get("q_ea", np.nan)) for row in rows):
        raise RuntimeError("cached coordinate join is incomplete for finite c_ea/q_ea rows")
    x_values = np.asarray([row["X_amp"] for row in matched], dtype=float)
    y_values = np.asarray([row["Y_amp"] for row in matched], dtype=float)
    return {
        "post_iter2_qa_rows": int(len(rows)),
        "finite_c_ea_rows": int(finite_c),
        "finite_q_ea_rows": int(finite_q),
        "finite_cq_rows": int(finite_cq),
        "successfully_matched_cached_xy_rows": int(len(matched)),
        "unique_physical_amplifiers": int(len(unique_amp)),
        "unique_physical_ifus": int(len({_physical_ifu_key(row) for row in rows})),
        "finite_coordinate_rows": int(len(rows) - len(bad)),
        "finite_coordinate_amplifiers": int(len(unique_amp) if not bad else
                                              len({_physical_amp_identity(row)
                                                   for row in rows
                                                   if row["focal_coordinate_status"] == "OK"})),
        "X_amp_min": float(np.min(x_values)) if x_values.size else np.nan,
        "X_amp_max": float(np.max(x_values)) if x_values.size else np.nan,
        "Y_amp_min": float(np.min(y_values)) if y_values.size else np.nan,
        "Y_amp_max": float(np.max(y_values)) if y_values.size else np.nan,
    }


def _physical_amp_identity(row):
    return (str(row["SPECID"]), str(row["IFUSLOT"]),
            str(row["IFUID"]), str(row["AMP"]))


def _fit_focal_plane(x, y, values, x_center=None, y_center=None,
                     max_iter=30, tuning=1.345):
    """Robust no-intercept plane in centered historical M101 coordinates."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    values = np.asarray(values, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(values)
    prediction = np.full(values.shape, np.nan, dtype=float)
    residual = np.full(values.shape, np.nan, dtype=float)
    fit_weight = np.full(values.shape, np.nan, dtype=float)
    if np.count_nonzero(valid) < 3:
        return {"bx": np.nan, "by": np.nan, "x_center": np.nan,
                "y_center": np.nan, "n": int(np.count_nonzero(valid)),
                "rank": 0, "prediction": prediction, "residual": residual,
                "weight": fit_weight, "inlier": np.zeros(values.shape, dtype=bool)}
    if x_center is None:
        x_center = float(np.median(x[valid]))
    if y_center is None:
        y_center = float(np.median(y[valid]))
    design = np.column_stack((x[valid] - x_center, y[valid] - y_center))
    target = values[valid]
    if np.linalg.matrix_rank(design) < 2:
        return {"bx": np.nan, "by": np.nan, "x_center": float(x_center),
                "y_center": float(y_center), "n": int(target.size), "rank": 1,
                "prediction": prediction, "residual": residual,
                "weight": fit_weight, "inlier": np.zeros(values.shape, dtype=bool)}
    weights = np.ones(target.size, dtype=float)
    coefficients = np.linalg.lstsq(design, target, rcond=None)[0]
    for _ in range(max_iter):
        root = np.sqrt(np.maximum(weights, 0.0))
        updated = np.linalg.lstsq(design * root[:, None], target * root,
                                   rcond=None)[0]
        trial_residual = target - design @ updated
        scale = _robust_scale(trial_residual)
        if not np.isfinite(scale) or scale <= 0.0:
            updated_weights = np.ones_like(weights)
        else:
            cutoff = tuning * scale
            absolute = np.abs(trial_residual)
            updated_weights = np.ones_like(weights)
            large = absolute > cutoff
            updated_weights[large] = cutoff / np.maximum(absolute[large], 1e-15)
        if np.allclose(updated, coefficients, rtol=1e-11, atol=1e-13):
            coefficients = updated
            weights = updated_weights
            break
        coefficients = updated
        weights = updated_weights
    final_residual = target - design @ coefficients
    scale = _robust_scale(final_residual)
    if np.isfinite(scale) and scale > 0.0:
        cutoff = tuning * scale
        fit_weight_valid = np.minimum(1.0, cutoff / np.maximum(
            np.abs(final_residual), 1e-15))
    else:
        fit_weight_valid = np.ones_like(final_residual)
    prediction[valid] = design @ coefficients
    residual[valid] = final_residual
    fit_weight[valid] = fit_weight_valid
    inlier = np.zeros(values.shape, dtype=bool)
    inlier[valid] = fit_weight_valid >= 0.1
    return {"bx": float(coefficients[0]), "by": float(coefficients[1]),
            "x_center": float(x_center), "y_center": float(y_center),
            "n": int(target.size), "rank": 2, "prediction": prediction,
            "residual": residual, "weight": fit_weight, "inlier": inlier}


def _scatter_reduction(before, after):
    before = float(before) if np.isfinite(before) else np.nan
    after = float(after) if np.isfinite(after) else np.nan
    if np.isfinite(before) and before > 0.0 and np.isfinite(after):
        return float((before - after) / before)
    return np.nan


def _plane_metrics(values, fit):
    before = _robust_scale(values)
    after = _robust_scale(fit["residual"])
    return {"sigma_before": before, "sigma_after": after,
            "scatter_reduction_fraction": _scatter_reduction(before, after)}


def _spatial_ifu_folds(rows, n_folds=5):
    """Assign whole physical IFUs to deterministic checkerboard folds."""
    by_ifu = {}
    for row in rows:
        key = _physical_ifu_key(row)
        x, y = float(row.get("X_amp", np.nan)), float(row.get("Y_amp", np.nan))
        if np.isfinite(x) and np.isfinite(y):
            by_ifu.setdefault(key, []).append((x, y))
    if not by_ifu:
        return {}, 0
    # First collapse the four amplifier coordinates to one center per
    # physical IFU.  Rank construction must use these centers, rather than
    # looking up a median after ranking all amplifier coordinates.
    centers = {
        key: (_robust_collapse([x for x, _ in values]),
              _robust_collapse([y for _, y in values]))
        for key, values in by_ifu.items()
    }
    centers = {key: value for key, value in centers.items()
               if np.isfinite(value[0]) and np.isfinite(value[1])}
    if not centers:
        return {}, 0
    x_values = sorted({round(float(value[0]), 8) for value in centers.values()})
    y_values = sorted({round(float(value[1]), 8) for value in centers.values()})
    x_index = {value: index for index, value in enumerate(x_values)}
    y_index = {value: index for index, value in enumerate(y_values)}
    n_folds = max(2, min(int(n_folds), len(centers)))
    result = {}
    for key, (x, y) in centers.items():
        ix = x_index[round(float(x), 8)]
        iy = y_index[round(float(y), 8)]
        result[key] = int((ix + 2 * iy) % n_folds)
    return result, n_folds


def _cross_validated_plane(rows, field, n_folds=5):
    values = np.asarray([row.get(field, np.nan) for row in rows], dtype=float)
    x = np.asarray([row.get("X_amp", np.nan) for row in rows], dtype=float)
    y = np.asarray([row.get("Y_amp", np.nan) for row in rows], dtype=float)
    valid = np.isfinite(values) & np.isfinite(x) & np.isfinite(y)
    folds, n_folds = _spatial_ifu_folds(rows, n_folds=n_folds)
    if not folds or not np.any(valid):
        return {"sigma_before": np.nan, "sigma_after": np.nan,
                "scatter_reduction_fraction": np.nan, "n": 0,
                "n_folds": 0, "n_ifu": 0}
    observed, residual = [], []
    used_folds = 0
    for fold in range(n_folds):
        fold_id = np.asarray([folds.get(_physical_ifu_key(row), -1)
                              for row in rows], dtype=int)
        train = valid & (fold_id != fold)
        heldout = valid & (fold_id == fold)
        if np.count_nonzero(heldout) == 0 or np.count_nonzero(train) < 3:
            continue
        fit = _fit_focal_plane(x[train], y[train], values[train])
        if not np.isfinite(fit["bx"]) or not np.isfinite(fit["by"]):
            continue
        prediction = (fit["bx"] * (x[heldout] - fit["x_center"]) +
                      fit["by"] * (y[heldout] - fit["y_center"]))
        observed.extend(values[heldout].tolist())
        residual.extend((values[heldout] - prediction).tolist())
        used_folds += 1
    observed = np.asarray(observed, dtype=float)
    residual = np.asarray(residual, dtype=float)
    before = _robust_scale(observed)
    after = _robust_scale(residual)
    return {"sigma_before": before, "sigma_after": after,
            "scatter_reduction_fraction": _scatter_reduction(before, after),
            "n": int(observed.size), "n_folds": int(used_folds),
            "n_ifu": int(len(folds))}


def _collapse_focal_ifus(rows):
    grouped = {}
    for row in rows:
        grouped.setdefault(_physical_ifu_key(row), []).append(row)
    output = []
    for key in sorted(grouped):
        members = grouped[key]
        first = members[0]
        result = {name: first[name] for name in
                  ("H5", "exposure", "SPECID", "IFUSLOT", "IFUID", "IFU_CODE")}
        result["AMP"] = "IFU"
        result["X_IFU"] = _robust_collapse([row["X_amp"] for row in members])
        result["Y_IFU"] = _robust_collapse([row["Y_amp"] for row in members])
        result["X_amp"] = result["X_IFU"]
        result["Y_amp"] = result["Y_IFU"]
        for field in ("c_ea", "q_ea", "c_plane_residual", "q_plane_residual",
                      "blank_c_gray", "blank_q_color"):
            result[field] = _robust_collapse([row.get(field, np.nan)
                                               for row in members])
        result["N_amp_used"] = int(sum(np.isfinite(row.get("c_ea", np.nan))
                                       for row in members))
        result["N_q_amp_used"] = int(sum(np.isfinite(row.get("q_ea", np.nan))
                                         for row in members))
        result["N_blank_amp_used"] = int(sum(
            np.isfinite(row.get("blank_c_gray", np.nan)) for row in members))
        output.append(result)
    return output


FOCAL_MEDIAN_C_MATERIAL_THRESHOLD = 0.01
FOCAL_MIN_BLANK_IFUS = 5

# The focal-plane plane remains a historical, report-only diagnostic.  The
# active pattern experiment below is deliberately IFU-level and rank-one.
LOWRANK_MIN_C_AMP_PER_IFU = 2
LOWRANK_MIN_COMMON_IFUS = 5
LOWRANK_MIN_TEMPLATE_SUPPORT = 2
LOWRANK_MIN_BLANK_IFUS = 5
LOWRANK_N_FOLDS = 5


def _gradient_metrics(prefix, fit):
    bx, by = float(fit["bx"]), float(fit["by"])
    if not (np.isfinite(bx) and np.isfinite(by)):
        return {
            "%s_gradient_amplitude" % prefix: np.nan,
            "%s_gradient_angle" % prefix: np.nan,
        }
    return {
        "%s_gradient_amplitude" % prefix: float(np.hypot(bx, by)),
        "%s_gradient_angle" % prefix: float(np.degrees(np.arctan2(by, bx))),
    }


def _short_angle_difference_degrees(first, second):
    if not (np.isfinite(first) and np.isfinite(second)):
        return np.nan
    difference = np.radians(float(first) - float(second))
    return float(np.degrees(np.arctan2(np.sin(difference), np.cos(difference))))


def _gradient_agreement(image_fit, blank_fit, image_prefix="c",
                        blank_prefix="blank_c"):
    image = _gradient_metrics(image_prefix, image_fit)
    blank = _gradient_metrics(blank_prefix, blank_fit)
    image_angle = image["%s_gradient_angle" % image_prefix]
    blank_angle = blank["%s_gradient_angle" % blank_prefix]
    image_amplitude = image["%s_gradient_amplitude" % image_prefix]
    blank_amplitude = blank["%s_gradient_amplitude" % blank_prefix]
    if np.isfinite(image_angle) and np.isfinite(blank_angle):
        cosine = float(np.cos(np.radians(image_angle - blank_angle)))
    else:
        cosine = np.nan
    return {
        "image_blank_gradient_angle_difference":
            _short_angle_difference_degrees(image_angle, blank_angle),
        "image_blank_gradient_cosine_similarity": cosine,
        "image_blank_gradient_amplitude_ratio": (
            float(image_amplitude / blank_amplitude)
            if np.isfinite(image_amplitude) and np.isfinite(blank_amplitude) and
            blank_amplitude > 0.0 else np.nan),
    }


def _focal_component_fit(rows, field, prefix):
    """Fit one no-intercept component and attach predictions/residuals."""
    x = np.asarray([row.get("X_amp", np.nan) for row in rows], dtype=float)
    y = np.asarray([row.get("Y_amp", np.nan) for row in rows], dtype=float)
    values = np.asarray([row.get(field, np.nan) for row in rows], dtype=float)
    fit = _fit_focal_plane(x, y, values)
    for index, row in enumerate(rows):
        row["%s_plane_prediction" % prefix] = fit["prediction"][index]
        row["%s_plane_residual" % prefix] = fit["residual"][index]
        row["%s_plane_weight" % prefix] = fit["weight"][index]
        row["%s_plane_inlier" % prefix] = bool(fit["inlier"][index])
    return fit


def _focal_fit_summary(prefix, fit, values, cv):
    metrics = _plane_metrics(values, fit)
    result = {
        "%s_bx" % prefix: fit["bx"],
        "%s_by" % prefix: fit["by"],
        "%s_X0" % prefix: fit["x_center"],
        "%s_Y0" % prefix: fit["y_center"],
        "%s_rank" % prefix: int(fit["rank"]),
        "N_%s_used" % prefix: int(fit["n"]),
        "%s_sigma_before" % prefix: metrics["sigma_before"],
        "%s_sigma_after" % prefix: metrics["sigma_after"],
        "%s_scatter_reduction_fraction" % prefix:
            metrics["scatter_reduction_fraction"],
        "%s_cv_sigma_before" % prefix: cv["sigma_before"],
        "%s_cv_sigma_after" % prefix: cv["sigma_after"],
        "%s_cv_scatter_reduction_fraction" % prefix:
            cv["scatter_reduction_fraction"],
        "%s_cv_N" % prefix: int(cv["n"]),
        "%s_cv_N_folds" % prefix: int(cv["n_folds"]),
        "%s_cv_N_IFU" % prefix: int(cv["n_ifu"]),
    }
    result.update(_gradient_metrics(prefix, fit))
    return result


def _run_focal_plane_diagnostic(rows):
    """Run per-exposure c/q planes, BLANK_QA planes, and IFU CV."""
    by_exposure = {}
    for row in rows:
        by_exposure.setdefault(
            (Path(row["H5"]).name, int(row["exposure"])), []).append(row)

    summary = []
    for h5_name, exposure in sorted(by_exposure):
        exposure_rows = by_exposure[(h5_name, exposure)]
        c_values = np.asarray([row.get("c_ea", np.nan)
                               for row in exposure_rows], dtype=float)
        q_values = np.asarray([row.get("q_ea", np.nan)
                               for row in exposure_rows], dtype=float)
        median_c = _robust_collapse(c_values)
        for row in exposure_rows:
            row["median_c_ea"] = median_c
            row["median_c_ea_materially_nonzero"] = bool(
                np.isfinite(median_c) and
                abs(median_c) >= FOCAL_MEDIAN_C_MATERIAL_THRESHOLD)

        c_fit = _focal_component_fit(exposure_rows, "c_ea", "c")
        q_fit = _focal_component_fit(exposure_rows, "q_ea", "q")
        blank_c_fit = _focal_component_fit(
            exposure_rows, "blank_c_gray", "blank_c")
        blank_q_fit = _focal_component_fit(
            exposure_rows, "blank_q_color", "blank_q")
        c_cv = _cross_validated_plane(exposure_rows, "c_ea")
        q_cv = _cross_validated_plane(exposure_rows, "q_ea")
        blank_c_cv = _cross_validated_plane(exposure_rows, "blank_c_gray")
        blank_q_cv = _cross_validated_plane(exposure_rows, "blank_q_color")

        ifu_rows = _collapse_focal_ifus(exposure_rows)
        c_ifu_values = np.asarray([row.get("c_ea", np.nan)
                                   for row in ifu_rows], dtype=float)
        c_ifu_fit = _focal_component_fit(ifu_rows, "c_ea", "c_IFU")
        c_ifu_metrics = _plane_metrics(c_ifu_values, c_ifu_fit)

        blank_c_ifu_keys = {
            _physical_ifu_key(row) for row in exposure_rows
            if np.isfinite(row.get("blank_c_gray", np.nan))}
        blank_q_ifu_keys = {
            _physical_ifu_key(row) for row in exposure_rows
            if np.isfinite(row.get("blank_q_color", np.nan))}
        blank_c_ifu_count = int(len(blank_c_ifu_keys))
        blank_q_ifu_count = int(len(blank_q_ifu_keys))
        blank_support_count = blank_c_ifu_count
        blank_support_gate = bool(
            blank_support_count >= FOCAL_MIN_BLANK_IFUS and
            np.isfinite(blank_c_fit["bx"]) and np.isfinite(blank_c_fit["by"]))

        amp_ifu_agreement = {
            "c_amp_ifu_gradient_angle_difference": np.nan,
            "c_amp_ifu_gradient_cosine_similarity": np.nan,
            "c_amp_ifu_gradient_amplitude_ratio": np.nan,
        }
        c_amp_gradient = _gradient_metrics("c", c_fit)
        c_ifu_gradient = _gradient_metrics("c_IFU", c_ifu_fit)
        if (np.isfinite(c_amp_gradient["c_gradient_angle"]) and
                np.isfinite(c_ifu_gradient["c_IFU_gradient_angle"])):
            amp_ifu_agreement.update({
                "c_amp_ifu_gradient_angle_difference":
                    _short_angle_difference_degrees(
                        c_amp_gradient["c_gradient_angle"],
                        c_ifu_gradient["c_IFU_gradient_angle"]),
                "c_amp_ifu_gradient_cosine_similarity": float(np.cos(np.radians(
                    c_amp_gradient["c_gradient_angle"] -
                    c_ifu_gradient["c_IFU_gradient_angle"]))),
            })
        if (np.isfinite(c_amp_gradient["c_gradient_amplitude"]) and
                np.isfinite(c_ifu_gradient["c_IFU_gradient_amplitude"]) and
                c_ifu_gradient["c_IFU_gradient_amplitude"] > 0.0):
            amp_ifu_agreement["c_amp_ifu_gradient_amplitude_ratio"] = float(
                c_amp_gradient["c_gradient_amplitude"] /
                c_ifu_gradient["c_IFU_gradient_amplitude"])

        coordinates = np.asarray([
            (row.get("X_amp", np.nan), row.get("Y_amp", np.nan))
            for row in exposure_rows], dtype=float)
        summary_row = {
            "H5": h5_name,
            "exposure": exposure,
            "N_post_iter2_qa_rows": int(len(exposure_rows)),
            "N_finite_c_ea": int(np.count_nonzero(np.isfinite(c_values))),
            "N_finite_q_ea": int(np.count_nonzero(np.isfinite(q_values))),
            "N_finite_cq": int(np.count_nonzero(
                np.isfinite(c_values) & np.isfinite(q_values))),
            "N_coordinate_rows": int(np.count_nonzero(
                np.all(np.isfinite(coordinates), axis=1))),
            "N_physical_amplifiers": int(
                len({_physical_amp_identity(row) for row in exposure_rows})),
            "N_physical_IFUs": int(len({_physical_ifu_key(row)
                                         for row in exposure_rows})),
            "X_amp_min": float(np.nanmin(coordinates[:, 0])),
            "X_amp_max": float(np.nanmax(coordinates[:, 0])),
            "Y_amp_min": float(np.nanmin(coordinates[:, 1])),
            "Y_amp_max": float(np.nanmax(coordinates[:, 1])),
            "median_c_ea": median_c,
            "median_c_ea_materially_nonzero": bool(
                np.isfinite(median_c) and
                abs(median_c) >= FOCAL_MEDIAN_C_MATERIAL_THRESHOLD),
            "N_blank_c_ea": int(np.count_nonzero(
                np.isfinite([row.get("blank_c_gray", np.nan)
                             for row in exposure_rows]))),
            "N_blank_q_ea": int(np.count_nonzero(
                np.isfinite([row.get("blank_q_color", np.nan)
                             for row in exposure_rows]))),
            "N_blank_c_IFU": blank_c_ifu_count,
            "N_blank_q_IFU": blank_q_ifu_count,
            "blank_support_IFU_count": blank_support_count,
            "blank_support_gate_min_IFU": FOCAL_MIN_BLANK_IFUS,
            "blank_support_gate_met": blank_support_gate,
        }
        summary_row.update(_focal_fit_summary("c", c_fit, c_values, c_cv))
        summary_row.update(_focal_fit_summary("q", q_fit, q_values, q_cv))
        summary_row.update(_focal_fit_summary(
            "blank_c", blank_c_fit,
            np.asarray([row.get("blank_c_gray", np.nan)
                        for row in exposure_rows], dtype=float), blank_c_cv))
        summary_row.update(_focal_fit_summary(
            "blank_q", blank_q_fit,
            np.asarray([row.get("blank_q_color", np.nan)
                        for row in exposure_rows], dtype=float), blank_q_cv))
        summary_row.update({
            "c_IFU_bx": c_ifu_fit["bx"],
            "c_IFU_by": c_ifu_fit["by"],
            "c_IFU_X0": c_ifu_fit["x_center"],
            "c_IFU_Y0": c_ifu_fit["y_center"],
            "c_IFU_rank": int(c_ifu_fit["rank"]),
            "N_c_IFU_used": int(c_ifu_fit["n"]),
            "c_IFU_sigma_before": c_ifu_metrics["sigma_before"],
            "c_IFU_sigma_after": c_ifu_metrics["sigma_after"],
            "c_IFU_scatter_reduction_fraction":
                c_ifu_metrics["scatter_reduction_fraction"],
        })
        summary_row.update(_gradient_metrics("c_IFU", c_ifu_fit))
        summary_row.update(amp_ifu_agreement)
        summary_row.update(_gradient_agreement(c_fit, blank_c_fit)
                           if blank_support_gate else {
                               "image_blank_gradient_angle_difference": np.nan,
                               "image_blank_gradient_cosine_similarity": np.nan,
                               "image_blank_gradient_amplitude_ratio": np.nan,
                           })
        summary.append(summary_row)

    join = {
        "post_iter2_qa_rows": int(len(rows)),
        "finite_c_ea_rows": int(sum(np.isfinite(row.get("c_ea", np.nan))
                                     for row in rows)),
        "finite_q_ea_rows": int(sum(np.isfinite(row.get("q_ea", np.nan))
                                     for row in rows)),
        "finite_cq_rows": int(sum(np.isfinite(row.get("c_ea", np.nan)) and
                                   np.isfinite(row.get("q_ea", np.nan))
                                   for row in rows)),
        "unique_physical_amplifiers": int(
            len({_physical_amp_identity(row) for row in rows})),
        "unique_physical_ifus": int(
            len({_physical_ifu_key(row) for row in rows})),
        "successfully_matched_cached_xy_rows": int(sum(
            np.isfinite(row.get("X_amp", np.nan)) and
            np.isfinite(row.get("Y_amp", np.nan)) for row in rows)),
        "X_amp_min": float(np.min([row["X_amp"] for row in rows])),
        "X_amp_max": float(np.max([row["X_amp"] for row in rows])),
        "Y_amp_min": float(np.min([row["Y_amp"] for row in rows])),
        "Y_amp_max": float(np.max([row["Y_amp"] for row in rows])),
    }
    return summary, join


def _value_stats(values, thresholds=False):
    values = _finite(values)
    if values.size == 0:
        result = {"N": 0, "median": np.nan, "robust_scatter": np.nan,
                  "p16": np.nan, "p84": np.nan, "min": np.nan,
                  "max": np.nan, "p01": np.nan, "p99": np.nan}
    else:
        result = {
            "N": int(values.size),
            "median": float(np.median(values)),
            "robust_scatter": _robust_scale(values),
            "p16": float(np.percentile(values, 16)),
            "p84": float(np.percentile(values, 84)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "p01": float(np.percentile(values, 1)),
            "p99": float(np.percentile(values, 99)),
        }
    if thresholds:
        result.update({
            "abs_gt_0.01": int(np.count_nonzero(np.abs(values) > 0.01)),
            "abs_gt_0.02": int(np.count_nonzero(np.abs(values) > 0.02)),
            "abs_gt_0.05": int(np.count_nonzero(np.abs(values) > 0.05)),
        })
    return result


def _closure_group_stats(exposure_rows):
    groups = {
        "all": lambda row: True,
        "sigma_g3_lt_0.02": lambda row: (
            np.isfinite(row["sigma_g3"]) and row["sigma_g3"] < 0.02),
        "sigma_g3_0.02_to_0.05": lambda row: (
            np.isfinite(row["sigma_g3"]) and
            0.02 <= row["sigma_g3"] < 0.05),
        "sigma_g3_ge_0.05": lambda row: (
            np.isfinite(row["sigma_g3"]) and row["sigma_g3"] >= 0.05),
    }
    result = {}
    for name, predicate in groups.items():
        selected = [row for row in exposure_rows if predicate(row)]
        values = np.asarray([row["g3_raw"] for row in selected], dtype=float)
        residual_stats = _value_stats(values - 1.0, thresholds=True)
        result[name] = residual_stats
        raw_stats = _value_stats(values, thresholds=False)
        result[name].update({"g3_%s" % key: value
                             for key, value in raw_stats.items()})
        for key in ("abs_gt_0.01", "abs_gt_0.02", "abs_gt_0.05"):
            result[name]["g3_%s" % key] = residual_stats[key]
    return result


def _qa_field_stats(rows, field):
    return _value_stats([row[field] for row in rows])


def _repeatability_rows(rows):
    grouped = {}
    for row in rows:
        key = (row["H5"], str(row["SPECID"]), str(row["IFUSLOT"]),
               str(row["IFUID"]), str(row["IFU_CODE"]), str(row["AMP"]))
        grouped.setdefault(key, {})[int(row["exposure"])] = row
    output = []
    for key in sorted(grouped):
        h5, specid, ifuslot, ifuid, ifu_code, amp = key
        by_exposure = grouped[key]
        c_values = {}
        q_values = {}
        for exposure in EXPOSURES:
            row = by_exposure.get(exposure)
            if row is not None and np.isfinite(row.get("image_c_gray", np.nan)) \
                    and np.isfinite(row.get("image_q_color", np.nan)):
                c_values[exposure] = float(row["image_c_gray"])
                q_values[exposure] = float(row["image_q_color"])
        valid_exposures = sorted(c_values)
        result = {
            "H5": h5, "SPECID": specid, "IFUSLOT": ifuslot,
            "IFUID": ifuid, "IFU_CODE": ifu_code, "AMP": amp,
            "N_exposures_with_cq": len(valid_exposures),
        }
        for exposure in EXPOSURES:
            result["c_exp%d" % exposure] = c_values.get(exposure, np.nan)
            result["q_exp%d" % exposure] = q_values.get(exposure, np.nan)
        if len(valid_exposures) >= 2:
            c_array = np.asarray([c_values[e] for e in valid_exposures])
            q_array = np.asarray([q_values[e] for e in valid_exposures])
            c_median = float(np.median(c_array))
            q_median = float(np.median(q_array))
            result.update({
                "c_median": c_median, "q_median": q_median,
                "c_range": float(np.max(c_array) - np.min(c_array)),
                "q_range": float(np.max(q_array) - np.min(q_array)),
                "c_rms_about_median": float(np.sqrt(np.mean(
                    (c_array - c_median) ** 2))),
                "q_rms_about_median": float(np.sqrt(np.mean(
                    (q_array - q_median) ** 2))),
                "c_same_sign": bool(np.all(np.sign(c_array) == np.sign(c_array[0]))),
                "q_same_sign": bool(np.all(np.sign(q_array) == np.sign(q_array[0]))),
            })
        else:
            result.update({
                "c_median": np.nan, "q_median": np.nan,
                "c_range": np.nan, "q_range": np.nan,
                "c_rms_about_median": np.nan,
                "q_rms_about_median": np.nan,
                "c_same_sign": np.nan, "q_same_sign": np.nan,
            })
        output.append(result)
    return output


def _repeatability_summary(repeat_rows):
    output = []
    for first, second in ((1, 2), (1, 3), (2, 3)):
        for component in ("c", "q"):
            x = np.asarray([row["%s_exp%d" % (component, first)]
                            for row in repeat_rows], dtype=float)
            y = np.asarray([row["%s_exp%d" % (component, second)]
                            for row in repeat_rows], dtype=float)
            valid = np.isfinite(x) & np.isfinite(y)
            x, y = x[valid], y[valid]
            difference = y - x
            if x.size >= 2 and np.std(x) > 0.0 and np.std(y) > 0.0:
                pearson = float(np.corrcoef(x, y)[0, 1])
            else:
                pearson = np.nan
            same_sign = (np.sign(x) == np.sign(y)) if x.size else np.asarray([])
            output.append({
                "exposure_pair": "%d_vs_%d" % (first, second),
                "component": component,
                "N_matched_amplifiers": int(x.size),
                "Pearson_correlation": pearson,
                "median_difference_second_minus_first": (
                    float(np.median(difference)) if difference.size else np.nan),
                "robust_scatter_difference": _robust_scale(difference),
                "fraction_same_sign": (
                    float(np.mean(same_sign)) if same_sign.size else np.nan),
            })
    return output


def _plot_iteration2_closure(exposure_rows, output_path):
    x = np.asarray([row["chronological_exposure_index"]
                    for row in exposure_rows], dtype=float)
    series = (
        ("original raw g1", "g1_raw", "sigma_g1", "tab:blue", "o"),
        ("iteration-1 closure g2", "g2_raw", "sigma_g2", "tab:orange", "s"),
        ("iteration-2 closure g3", "g3_raw", "sigma_g3", "tab:green", "^"),
    )
    all_values = np.concatenate([
        np.asarray([row[field] for row in exposure_rows], dtype=float)
        for _, field, _, _, _ in series])
    limits = _robust_plot_limits(all_values, include_values=(1.0,))
    fig, ax = plt.subplots(figsize=(14, 5.5))
    offscale = 0
    for label, field, sigma_field, color, marker in series:
        values = np.asarray([row[field] for row in exposure_rows], dtype=float)
        errors = np.asarray([row[sigma_field] for row in exposure_rows], dtype=float)
        finite = np.isfinite(values)
        with_error = finite & np.isfinite(errors)
        without_error = finite & ~np.isfinite(errors)
        ax.errorbar(x[with_error], values[with_error], yerr=errors[with_error],
                    fmt=marker, ms=4, lw=0.8, capsize=2, color=color,
                    label=label)
        ax.plot(x[without_error], values[without_error], marker, ms=4,
                color=color, label="_nolegend_")
        offscale += _plot_offscale_markers(
            ax, x, values, (x.min() - 0.5, x.max() + 0.5), limits,
            color, marker="v")
    ax.axhline(1.0, color="0.35", lw=0.9)
    ax.set_xlim(x.min() - 0.5, x.max() + 0.5)
    ax.set_ylim(*limits)
    ax.set_xlabel("chronological exposure index")
    ax.set_ylabel("SB gray multiplicative scale")
    ax.set_title("M101 SB exposure-scale closure through iteration 2")
    if offscale:
        ax.text(0.99, 0.03, "%d off robust y-range" % offscale,
                transform=ax.transAxes, ha="right", va="bottom", fontsize=8)
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8, ncol=3, loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _full_symmetric_limits(values, padding=0.08):
    values = _finite(values)
    bound = float(np.max(np.abs(values))) if values.size else 0.0
    bound = max(bound * (1.0 + padding), 1.0e-6)
    return -bound, bound


def _plot_cq(rows, output_path, full_range=False):
    c = np.asarray([row["image_c_gray"] for row in rows], dtype=float)
    q = np.asarray([row["image_q_color"] for row in rows], dtype=float)
    finite = np.isfinite(c) & np.isfinite(q)
    if full_range:
        limits = _full_symmetric_limits(np.concatenate((c[finite], q[finite]))) \
            if np.any(finite) else (-1.0e-6, 1.0e-6)
    else:
        limits = _robust_plot_limits(
            np.concatenate((c[finite], q[finite])) if np.any(finite)
            else np.asarray([0.0]), symmetric=True)
    fig, ax = plt.subplots(figsize=(7, 7))
    offscale = 0
    if np.any(finite):
        ax.scatter(c[finite], q[finite], s=15, alpha=0.55, color="tab:purple")
        offscale = _plot_offscale_markers(
            ax, c[finite], q[finite], limits, limits, "tab:red", marker="^")
        rank = np.argsort(np.hypot(c[finite], q[finite]))[::-1]
        indices = np.flatnonzero(finite)
        annotated = 0
        for index in rank:
            original_index = indices[index]
            if (c[original_index] < limits[0] or c[original_index] > limits[1] or
                    q[original_index] < limits[0] or q[original_index] > limits[1]):
                continue
            row = rows[original_index]
            label = "%s/e%d %s/%s/%s/%s" % (
                Path(row["H5"]).stem, row["exposure"], row["SPECID"],
                row["IFUSLOT"], row["IFUID"], row["AMP"])
            ax.annotate(label, (c[original_index], q[original_index]), fontsize=5)
            annotated += 1
            if annotated >= 8:
                break
    ax.axhline(0.0, color="0.7", lw=0.8)
    ax.axvline(0.0, color="0.7", lw=0.8)
    ax.set_xlim(*limits)
    ax.set_ylim(*limits)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("c_ea = 0.5 (f_ON + f_OFF), gray/common")
    ax.set_ylabel("q_ea = 0.5 (f_ON - f_OFF), ON-OFF differential")
    ax.set_title("Post-iteration-2 IMAGE_QA c/q" +
                 (" (full range)" if full_range else " (robust range)"))
    if offscale:
        ax.text(0.99, 0.02, "%d off displayed range" % offscale,
                transform=ax.transAxes, ha="right", va="bottom", fontsize=8)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _plot_cq_repeatability(repeat_rows, output_path):
    c_values = np.concatenate([
        np.asarray([row["c_exp%d" % exposure] for row in repeat_rows], dtype=float)
        for exposure in EXPOSURES]) if repeat_rows else np.asarray([])
    q_values = np.concatenate([
        np.asarray([row["q_exp%d" % exposure] for row in repeat_rows], dtype=float)
        for exposure in EXPOSURES]) if repeat_rows else np.asarray([])
    c_limits = _robust_plot_limits(c_values, symmetric=True)
    q_limits = _robust_plot_limits(q_values, symmetric=True)
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=True)
    for row_index, component, limits, color in (
            (0, "c", c_limits, "tab:blue"),
            (1, "q", q_limits, "tab:orange")):
        for col_index, (first, second) in enumerate(((1, 2), (1, 3), (2, 3))):
            ax = axes[row_index, col_index]
            x = np.asarray([row["%s_exp%d" % (component, first)]
                            for row in repeat_rows], dtype=float)
            y = np.asarray([row["%s_exp%d" % (component, second)]
                            for row in repeat_rows], dtype=float)
            finite = np.isfinite(x) & np.isfinite(y)
            if np.any(finite):
                ax.scatter(x[finite], y[finite], s=12, alpha=0.55, color=color)
                offscale = _plot_offscale_markers(
                    ax, x[finite], y[finite], limits, limits, "tab:red", marker="^")
                if offscale:
                    ax.text(0.97, 0.03, "%d off" % offscale,
                            transform=ax.transAxes, ha="right", va="bottom",
                            fontsize=7)
            ax.plot(limits, limits, color="0.35", lw=0.9)
            ax.axhline(0.0, color="0.8", lw=0.6)
            ax.axvline(0.0, color="0.8", lw=0.6)
            ax.set_xlim(*limits)
            ax.set_ylim(*limits)
            ax.set_aspect("equal", adjustable="box")
            ax.set_title("%s: exposure %d vs %d" % (component, first, second))
            ax.set_xlabel("exposure %d" % first)
            ax.set_ylabel("exposure %d" % second)
            ax.grid(alpha=0.15)
    fig.suptitle("Post-iteration-2 same-amplifier c/q repeatability")
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _focal_coordinate_limits(rows):
    x = _finite([row.get("X_amp", np.nan) for row in rows])
    y = _finite([row.get("Y_amp", np.nan) for row in rows])
    def limits(values):
        if not values.size:
            return -1.0, 1.0
        lo, hi = float(np.min(values)), float(np.max(values))
        span = max(hi - lo, 1.0e-6)
        pad = 0.06 * span
        return lo - pad, hi + pad
    return limits(x), limits(y)


def _focal_symmetric_robust_limits(values, q_lo=0.01, q_hi=0.99,
                                   padding=0.08):
    values = _finite(values)
    if not values.size:
        return (-1.0e-6, 1.0e-6)
    low, high = np.percentile(values, (100.0 * q_lo, 100.0 * q_hi))
    bound = max(abs(float(low)), abs(float(high)), 1.0e-8)
    bound *= 1.0 + padding
    return -bound, bound


def _focal_global_color_limits(rows):
    return {
        "c": _focal_symmetric_robust_limits(
            [row.get("c_ea", np.nan) for row in rows]),
        "q": _focal_symmetric_robust_limits(
            [row.get("q_ea", np.nan) for row in rows]),
        "method": "full post-iteration-2 population central 1--99 percentiles",
        "symmetric": True,
        "padding": 0.08,
    }


def _focal_safe_stem(h5_name):
    return "".join(char if char.isalnum() or char in "._-" else "_"
                   for char in Path(h5_name).stem)


def _plot_focal_field(ax, fig, selected, field, title, limits, x_limits,
                      y_limits, cmap="coolwarm"):
    x = np.asarray([row.get("X_amp", np.nan) for row in selected], dtype=float)
    y = np.asarray([row.get("Y_amp", np.nan) for row in selected], dtype=float)
    values = np.asarray([row.get(field, np.nan) for row in selected], dtype=float)
    finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(values)
    offscale = np.zeros(values.shape, dtype=bool)
    if np.any(finite):
        offscale = finite & ((values < limits[0]) | (values > limits[1]))
        scatter = ax.scatter(
            x[finite], y[finite], c=values[finite], s=18, cmap=cmap,
            vmin=limits[0], vmax=limits[1], alpha=0.78, edgecolors="none")
        fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04, label=field)
        if np.any(offscale):
            ax.scatter(x[offscale], y[offscale], s=45, marker="^",
                       facecolors="none", edgecolors="black", linewidths=0.8,
                       label="off global color range", zorder=5)
            ax.text(0.98, 0.03, "%d off global range" %
                    int(np.count_nonzero(offscale)), transform=ax.transAxes,
                    ha="right", va="bottom", fontsize=7)
    else:
        ax.text(0.5, 0.5, "no finite %s" % field,
                transform=ax.transAxes, ha="center", va="center")
    ax.set_xlim(*x_limits)
    ax.set_ylim(*y_limits)
    ax.set_aspect("equal", adjustable="box")
    ax.axhline(0.0, color="0.75", lw=0.6)
    ax.axvline(0.0, color="0.75", lw=0.6)
    ax.set_xlabel("historical_M101_x_arcmin")
    ax.set_ylabel("historical_M101_y_arcmin")
    ax.set_title(title)
    ax.grid(alpha=0.15)


def _plot_focal_plane_maps(rows, summary, output_dir, color_limits=None):
    """Write one fixed-scale 2x2 historical-coordinate map per exposure."""
    if color_limits is None:
        color_limits = _focal_global_color_limits(rows)
    x_limits, y_limits = _focal_coordinate_limits(rows)
    paths = []
    for summary_row in summary:
        h5_name = summary_row["H5"]
        exposure = int(summary_row["exposure"])
        selected = [row for row in rows
                    if Path(row["H5"]).name == h5_name and
                    int(row["exposure"]) == exposure]
        ifu_rows = _collapse_focal_ifus(selected)
        fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
        _plot_focal_field(
            axes[0, 0], fig, selected, "c_ea", "amplifier c_ea",
            color_limits["c"], x_limits, y_limits)
        _plot_focal_field(
            axes[0, 1], fig, ifu_rows, "c_ea", "IFU-collapsed c_ea",
            color_limits["c"], x_limits, y_limits)
        _plot_focal_field(
            axes[1, 0], fig, selected, "c_plane_residual",
            "amplifier c residual after diagnostic plane", color_limits["c"],
            x_limits, y_limits)
        _plot_focal_field(
            axes[1, 1], fig, selected, "q_ea", "amplifier q_ea",
            color_limits["q"], x_limits, y_limits)
        fig.suptitle("%s exposure %d: post-iteration-2 focal-plane diagnostic" %
                     (h5_name, exposure))
        path = output_dir / ("focal_plane_map_%s_e%02d.png" %
                             (_focal_safe_stem(h5_name), exposure))
        fig.savefig(path, dpi=150)
        plt.close(fig)
        paths.append(str(path.resolve()))
    return paths


def _plot_focal_blank_comparisons(rows, summary, output_dir, color_limits):
    """Write compact IMAGE_QA/BLANK_QA c comparisons for gated exposures."""
    x_limits, y_limits = _focal_coordinate_limits(rows)
    paths = []
    for summary_row in summary:
        if not summary_row.get("blank_support_gate_met", False):
            continue
        h5_name = summary_row["H5"]
        exposure = int(summary_row["exposure"])
        selected = [row for row in rows
                    if Path(row["H5"]).name == h5_name and
                    int(row["exposure"]) == exposure]
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.8), constrained_layout=True)
        _plot_focal_field(
            axes[0], fig, selected, "c_ea", "IMAGE_QA c_ea",
            color_limits["c"], x_limits, y_limits)
        _plot_focal_field(
            axes[1], fig, selected, "blank_c_gray", "BLANK_QA blank_c_gray",
            color_limits["c"], x_limits, y_limits)
        fig.suptitle("%s exposure %d: gated IMAGE_QA/BLANK_QA c comparison "
                     "(%d IFUs)" %
                     (h5_name, exposure, summary_row["blank_support_IFU_count"]))
        path = output_dir / ("focal_plane_blank_comparison_%s_e%02d.png" %
                             (_focal_safe_stem(h5_name), exposure))
        fig.savefig(path, dpi=150)
        plt.close(fig)
        paths.append(str(path.resolve()))
    return paths


def _plot_focal_gradient_summary(summary, output_path):
    index = np.arange(len(summary), dtype=float)
    labels = ["%s/e%d" % (Path(row["H5"]).stem, int(row["exposure"]))
              for row in summary]
    fig, axes = plt.subplots(2, 2, figsize=(15, 8), sharex=True,
                             constrained_layout=True)
    for ax, component, direction in (
            (axes[0, 0], "c", "x"), (axes[0, 1], "c", "y"),
            (axes[1, 0], "q", "x"), (axes[1, 1], "q", "y")):
        values = np.asarray([row.get("%s_b%s" % (component, direction), np.nan)
                             for row in summary], dtype=float)
        ax.plot(index, values, "o", ms=3.2,
                color="tab:blue" if component == "c" else "tab:orange")
        ax.axhline(0.0, color="0.5", lw=0.7)
        ax.set_ylabel("b_%s (fraction / arcmin)" % direction)
        ax.set_title("%s-plane %s gradient" % (component, direction))
        ax.grid(alpha=0.2)
    axes[1, 0].set_xlabel("H5/exposure in deterministic order")
    axes[1, 1].set_xlabel("H5/exposure in deterministic order")
    if labels:
        for ax in axes[1]:
            ax.set_xticks(index[::max(1, len(index) // 12)])
            ax.set_xticklabels(labels[::max(1, len(index) // 12)],
                               rotation=65, ha="right", fontsize=7)
    fig.suptitle("Post-iteration-2 historical M101 focal-plane gradients")
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _plot_focal_plane_performance(summary, output_path):
    index = np.arange(len(summary), dtype=float)
    labels = ["%s/e%d" % (Path(row["H5"]).stem, int(row["exposure"]))
              for row in summary]
    fig, axes = plt.subplots(2, 2, figsize=(15, 8), sharex=True,
                             constrained_layout=True)
    for ax, component, title in (
            (axes[0, 0], "c", "c_ea direct plane"),
            (axes[0, 1], "q", "q_ea direct plane")):
        before = np.asarray([row.get("%s_sigma_before" % component, np.nan)
                             for row in summary], dtype=float)
        after = np.asarray([row.get("%s_sigma_after" % component, np.nan)
                            for row in summary], dtype=float)
        ax.plot(index, before, "o", ms=3, label="before")
        ax.plot(index, after, "s", ms=3, label="after")
        ax.set_title(title)
        ax.set_ylabel("robust scatter")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    for ax, component, title in (
            (axes[1, 0], "c", "c_ea held-out-IFU CV"),
            (axes[1, 1], "q", "q_ea held-out-IFU CV")):
        before = np.asarray([row.get("%s_cv_sigma_before" % component, np.nan)
                             for row in summary], dtype=float)
        after = np.asarray([row.get("%s_cv_sigma_after" % component, np.nan)
                            for row in summary], dtype=float)
        ax.plot(index, before, "o", ms=3, label="before")
        ax.plot(index, after, "s", ms=3, label="held-out after")
        ax.set_title(title)
        ax.set_ylabel("robust scatter")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    if labels:
        stride = max(1, len(index) // 12)
        for ax in axes[1]:
            ax.set_xticks(index[::stride])
            ax.set_xticklabels(labels[::stride], rotation=65, ha="right",
                               fontsize=7)
            ax.set_xlabel("H5/exposure in deterministic order")
    fig.suptitle("Post-iteration-2 focal-plane plane performance")
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _lowrank_ifu_coordinate_limits(data):
    x = _finite([row.get("X_IFU", np.nan)
                 for row in data["ifu_center_rows"]])
    y = _finite([row.get("Y_IFU", np.nan)
                 for row in data["ifu_center_rows"]])
    def limits(values):
        if not values.size:
            return -1.0, 1.0
        lo, hi = float(np.min(values)), float(np.max(values))
        span = max(hi - lo, 1.0e-6)
        pad = 0.06 * span
        return lo - pad, hi + pad
    return limits(x), limits(y)


def _plot_lowrank_ifu_field(ax, fig, records, field, title, limits,
                            x_limits, y_limits):
    x = np.asarray([row.get("X_IFU", np.nan) for row in records], dtype=float)
    y = np.asarray([row.get("Y_IFU", np.nan) for row in records], dtype=float)
    values = np.asarray([row.get(field, np.nan) for row in records], dtype=float)
    finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(values)
    if np.any(finite):
        offscale = finite & ((values < limits[0]) | (values > limits[1]))
        scatter = ax.scatter(x[finite], y[finite], c=values[finite], s=30,
                             cmap="coolwarm", vmin=limits[0], vmax=limits[1],
                             edgecolors="none")
        fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04, label=field)
        if np.any(offscale):
            ax.scatter(x[offscale], y[offscale], s=55, marker="^",
                       facecolors="none", edgecolors="black", linewidths=0.8,
                       zorder=5)
            ax.text(0.98, 0.03, "%d off global range" %
                    int(np.count_nonzero(offscale)), transform=ax.transAxes,
                    ha="right", va="bottom", fontsize=7)
    else:
        ax.text(0.5, 0.5, "no finite %s" % field,
                transform=ax.transAxes, ha="center", va="center")
    ax.set_xlim(*x_limits)
    ax.set_ylim(*y_limits)
    ax.set_aspect("equal", adjustable="box")
    ax.axhline(0.0, color="0.75", lw=0.6)
    ax.axvline(0.0, color="0.75", lw=0.6)
    ax.set_xlabel("historical_M101_x_arcmin")
    ax.set_ylabel("historical_M101_y_arcmin")
    ax.set_title(title)
    ax.grid(alpha=0.15)


def _lowrank_template_records(data, image_fit, blank_fit):
    image_template = image_fit["template"]
    blank_template = blank_fit["template"]
    records = []
    for center in data["ifu_center_rows"]:
        index = data["ifu_index"][(center["SPECID"], center["IFUSLOT"],
                                    center["IFUID"], center["IFU_CODE"])]
        record = dict(center)
        record["T_image"] = image_template[index]
        record["T_blank"] = blank_template[index]
        records.append(record)
    return records


def _plot_lowrank_outputs(lowrank, output_dir):
    data = lowrank["data"]
    image_fit = lowrank["rank1_fit"]
    blank_fit = lowrank["blank_fit"]
    x_limits, y_limits = _lowrank_ifu_coordinate_limits(data)
    pattern_limits = _focal_symmetric_robust_limits(data["matrix"])
    template_values = np.concatenate((
        _finite(image_fit["template"]), _finite(blank_fit["template"])))
    template_limits = _focal_symmetric_robust_limits(template_values)
    template_records = _lowrank_template_records(data, image_fit, blank_fit)
    paths = []

    template_path = output_dir / "post_iter2_ifu_rank1_template.png"
    fig, ax = plt.subplots(1, 1, figsize=(6.8, 5.8), constrained_layout=True)
    _plot_lowrank_ifu_field(
        ax, fig, template_records, "T_image",
        "learned rank-1 focal template T_i (IMAGE_QA c)",
        template_limits, x_limits, y_limits)
    fig.suptitle("Physical-IFU rank-1 focal template; median(T)=0, robust RMS(T)=1")
    fig.savefig(template_path, dpi=160)
    plt.close(fig)
    paths.append(str(template_path.resolve()))

    finite_amplitudes = np.flatnonzero(np.isfinite(image_fit["amplitude"]))
    representative = []
    if finite_amplitudes.size:
        representative.append(int(finite_amplitudes[0]))
        median_order = finite_amplitudes[np.argsort(
            image_fit["amplitude"][finite_amplitudes])]
        representative.append(int(median_order[len(median_order) // 2]))
        largest = int(finite_amplitudes[np.argmax(
            np.abs(image_fit["amplitude"][finite_amplitudes]))])
        if largest not in representative:
            representative.append(largest)
    representative = representative[:3]
    cell_rows = data["cell_rows"]
    representative_path = output_dir / "post_iter2_ifu_rank1_representative_maps.png"
    fig, axes = plt.subplots(max(1, len(representative)), 3,
                             figsize=(15, 4.5 * max(1, len(representative))),
                             squeeze=False, constrained_layout=True)
    for plot_index, exposure_index in enumerate(representative):
        exposure_key = data["exposure_keys"][exposure_index]
        selected = [row for row in cell_rows
                    if row["H5"] == exposure_key[0] and
                    row["exposure"] == exposure_key[1]]
        amplitude = image_fit["amplitude"][exposure_index]
        label = "%s/e%d A_e=%s" % (
            exposure_key[0], exposure_key[1], _format_number(amplitude))
        _plot_lowrank_ifu_field(
            axes[plot_index, 0], fig, selected, "C_ei",
            "%s observed C_ei" % label, pattern_limits, x_limits, y_limits)
        _plot_lowrank_ifu_field(
            axes[plot_index, 1], fig, selected, "rank1_prediction_pattern",
            "%s A_e*T_i (centered pattern prediction)" % label,
            pattern_limits, x_limits, y_limits)
        _plot_lowrank_ifu_field(
            axes[plot_index, 2], fig, selected, "rank1_residual_C_minus_AT",
            "%s C_ei-A_e*T_i residual (gauge retained)" % label,
            pattern_limits, x_limits, y_limits)
    for plot_index in range(len(representative), axes.shape[0]):
        for axis in axes[plot_index]:
            axis.axis("off")
    fig.suptitle("Observed IFU-collapsed C_ei, rank-1 prediction, and residual; "
                 "fixed global color scale")
    fig.savefig(representative_path, dpi=150)
    plt.close(fig)
    paths.append(str(representative_path.resolve()))

    blank_path = output_dir / "post_iter2_ifu_rank1_blank_template.png"
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.5),
                             constrained_layout=True)
    _plot_lowrank_ifu_field(
        axes[0], fig, template_records, "T_image", "IMAGE-derived T_i",
        template_limits, x_limits, y_limits)
    _plot_lowrank_ifu_field(
        axes[1], fig, template_records, "T_blank", "BLANK_QA-derived T_blank,i",
        template_limits, x_limits, y_limits)
    fig.suptitle("Independent IMAGE_QA / BLANK_QA rank-1 template comparison")
    fig.savefig(blank_path, dpi=160)
    plt.close(fig)
    paths.append(str(blank_path.resolve()))
    return {
        "template_plot": str(template_path.resolve()),
        "representative_plot": str(representative_path.resolve()),
        "blank_template_plot": str(blank_path.resolve()),
        "paths": paths,
        "color_limits": {
            "C_ei": pattern_limits,
            "T_image_T_blank": template_limits,
            "method": "full post-iteration-2 IFU population central 1--99 percentiles",
            "symmetric": True,
        },
    }


def _ranked_cq_rows(rows, component):
    field = "image_%s" % component
    return sorted(
        [row for row in rows if np.isfinite(row.get(field, np.nan))],
        key=lambda row: abs(float(row[field])), reverse=True)


def _write_cq_ranked_csv(path, rows):
    fields = ["H5", "exposure", "SPECID", "IFUSLOT", "IFUID", "IFU_CODE",
              "AMP", "image_fractional_ON", "image_fractional_OFF",
              "image_c_gray", "image_q_color", "image_abs_c_gray",
              "image_abs_q_color", "image_z_ON", "image_z_OFF",
              "BLANK_QA_STATUS", "blank_c_gray", "blank_q_color"]
    _write_csv(path, rows, fields)


def _focal_summary_aggregate(summary):
    def finite_field(field):
        return _finite([row.get(field, np.nan) for row in summary])

    def median_field(field):
        values = finite_field(field)
        return float(np.median(values)) if values.size else np.nan

    def median_rows(rows, field):
        values = _finite([row.get(field, np.nan) for row in rows])
        return float(np.median(values)) if values.size else np.nan

    adequate = [row for row in summary
                if row.get("blank_support_gate_met", False)]
    amp_ifu_comparable = [row for row in summary
                          if np.isfinite(row.get(
                              "c_amp_ifu_gradient_angle_difference", np.nan))]
    return {
        "N_exposures": int(len(summary)),
        "median_c_scatter_before_plane": median_field("c_sigma_before"),
        "median_c_scatter_after_in_sample_plane": median_field("c_sigma_after"),
        "median_c_scatter_after_heldout_IFU_prediction": median_field(
            "c_cv_sigma_after"),
        "median_q_scatter_before_plane": median_field("q_sigma_before"),
        "median_q_scatter_after_in_sample_plane": median_field("q_sigma_after"),
        "median_q_scatter_after_heldout_IFU_prediction": median_field(
            "q_cv_sigma_after"),
        "N_exposures_heldout_c_scatter_improves": int(sum(
            np.isfinite(row.get("c_cv_sigma_before", np.nan)) and
            np.isfinite(row.get("c_cv_sigma_after", np.nan)) and
            row["c_cv_sigma_after"] < row["c_cv_sigma_before"]
            for row in summary)),
        "N_exposures_heldout_q_scatter_improves": int(sum(
            np.isfinite(row.get("q_cv_sigma_before", np.nan)) and
            np.isfinite(row.get("q_cv_sigma_after", np.nan)) and
            row["q_cv_sigma_after"] < row["q_cv_sigma_before"]
            for row in summary)),
        "N_exposures_adequate_BLANK_QA_focal_support": int(len(adequate)),
        "blank_support_min_distinct_IFUs": FOCAL_MIN_BLANK_IFUS,
        "adequate_blank_image_gradient_cosine_median": median_rows(
            adequate, "image_blank_gradient_cosine_similarity"),
        "adequate_blank_image_gradient_abs_angle_difference_median": (
            float(np.median(_finite([
                abs(row.get("image_blank_gradient_angle_difference", np.nan))
                for row in adequate]))) if _finite([
                    row.get("image_blank_gradient_angle_difference", np.nan)
                    for row in adequate]).size else np.nan),
        "adequate_blank_image_gradient_amplitude_ratio_median": median_rows(
            adequate, "image_blank_gradient_amplitude_ratio"),
        "N_exposures_amp_IFU_gradient_comparable": int(len(amp_ifu_comparable)),
        "amp_IFU_gradient_cosine_median": median_rows(
            amp_ifu_comparable, "c_amp_ifu_gradient_cosine_similarity"),
        "amp_IFU_gradient_abs_angle_difference_median": (
            float(np.median(_finite([
                abs(row.get("c_amp_ifu_gradient_angle_difference", np.nan))
                for row in amp_ifu_comparable]))) if amp_ifu_comparable else np.nan),
        "amp_IFU_gradient_amplitude_ratio_median": median_rows(
            amp_ifu_comparable, "c_amp_ifu_gradient_amplitude_ratio"),
    }


def _lowrank_robust_rms(values):
    """Robust RMS-like scale used only to normalize a pattern template."""
    values = _finite(values)
    if values.size < 2:
        return np.nan
    center = float(np.median(values))
    mad_scale = 1.4826 * float(np.median(np.abs(values - center)))
    if np.isfinite(mad_scale) and mad_scale > 0.0:
        return mad_scale
    rms = float(np.sqrt(np.mean((values - center) ** 2)))
    return rms if np.isfinite(rms) and rms > 0.0 else np.nan


def _lowrank_rankdata(values):
    """Average ranks without adding a scipy dependency for Spearman rho."""
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(values.size, dtype=float)
    start = 0
    while start < values.size:
        stop = start + 1
        while stop < values.size and sorted_values[stop] == sorted_values[start]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1) + 1.0
        start = stop
    return ranks


def _lowrank_correlations(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    result = {"pearson": np.nan, "spearman": np.nan,
              "n": int(x.size)}
    if x.size < 2:
        return result
    if np.std(x) > 0.0 and np.std(y) > 0.0:
        result["pearson"] = float(np.corrcoef(x, y)[0, 1])
        rx, ry = _lowrank_rankdata(x), _lowrank_rankdata(y)
        if np.std(rx) > 0.0 and np.std(ry) > 0.0:
            result["spearman"] = float(np.corrcoef(rx, ry)[0, 1])
    return result


def _lowrank_scatter_metrics(observed, predicted):
    observed = np.asarray(observed, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    valid = np.isfinite(observed) & np.isfinite(predicted)
    observed, predicted = observed[valid], predicted[valid]
    residual = observed - predicted
    before = _robust_scale(observed)
    after = _robust_scale(residual)
    result = {
        "sigma_before": before,
        "sigma_after": after,
        "fractional_improvement": _scatter_reduction(before, after),
        "robust_variance_explained": (
            float(1.0 - (after / before) ** 2)
            if np.isfinite(before) and before > 0.0 and np.isfinite(after)
            else np.nan),
        "n": int(observed.size),
    }
    return result


def _lowrank_normalize_template(template, amplitudes=None):
    template = np.asarray(template, dtype=float).copy()
    finite = np.isfinite(template)
    if not np.any(finite):
        return template, amplitudes
    template[finite] -= float(np.median(template[finite]))
    scale = _lowrank_robust_rms(template[finite])
    if np.isfinite(scale) and scale > 0.0:
        template[finite] /= scale
    if amplitudes is not None:
        amplitudes = np.asarray(amplitudes, dtype=float).copy()
    # Resolve the otherwise arbitrary rank-one sign deterministically.  This
    # is a gauge choice only; the fitted product A_e*T_i is unchanged.
    anchor = np.flatnonzero(np.isfinite(template) &
                            (np.abs(template) > 1.0e-12))
    if anchor.size and template[anchor[0]] < 0.0:
        template[finite] *= -1.0
        if amplitudes is not None:
            amplitudes[np.isfinite(amplitudes)] *= -1.0
    return template, amplitudes


def _fit_robust_rank1(matrix, max_iter=40):
    """Fit a transparent robust alternating rank-one decomposition."""
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("rank-one input must be a two-dimensional matrix")
    n_exposure, n_ifu = matrix.shape
    template = np.full(n_ifu, np.nan, dtype=float)
    for ifu_index in range(n_ifu):
        values = matrix[:, ifu_index]
        finite = np.isfinite(values)
        if np.count_nonzero(finite) >= LOWRANK_MIN_TEMPLATE_SUPPORT:
            template[ifu_index] = float(np.median(values[finite]))
    template, _ = _lowrank_normalize_template(template)
    amplitudes = np.full(n_exposure, np.nan, dtype=float)
    if not np.any(np.isfinite(template)):
        return {"template": template, "amplitude": amplitudes,
                "prediction": np.full(matrix.shape, np.nan),
                "residual": np.full(matrix.shape, np.nan),
                "iterations": 0}

    iterations = 0
    for iteration in range(max_iter):
        old_template = template.copy()
        for exposure_index in range(n_exposure):
            valid = np.isfinite(matrix[exposure_index]) & np.isfinite(template)
            if np.count_nonzero(valid) >= LOWRANK_MIN_TEMPLATE_SUPPORT:
                fit = _origin_huber(template[valid], matrix[exposure_index, valid])
                amplitudes[exposure_index] = fit["slope"]
            else:
                amplitudes[exposure_index] = np.nan

        updated_template = np.full(n_ifu, np.nan, dtype=float)
        for ifu_index in range(n_ifu):
            valid = np.isfinite(matrix[:, ifu_index]) & np.isfinite(amplitudes)
            if np.count_nonzero(valid) >= LOWRANK_MIN_TEMPLATE_SUPPORT:
                fit = _origin_huber(amplitudes[valid], matrix[valid, ifu_index])
                updated_template[ifu_index] = fit["slope"]
        template, amplitudes = _lowrank_normalize_template(
            updated_template, amplitudes)
        iterations = iteration + 1
        finite = np.isfinite(template) & np.isfinite(old_template)
        if np.count_nonzero(finite) and np.max(
                np.abs(template[finite] - old_template[finite])) < 1.0e-8:
            break

    for exposure_index in range(n_exposure):
        valid = np.isfinite(matrix[exposure_index]) & np.isfinite(template)
        if np.count_nonzero(valid) >= LOWRANK_MIN_TEMPLATE_SUPPORT:
            fit = _origin_huber(template[valid], matrix[exposure_index, valid])
            amplitudes[exposure_index] = fit["slope"]
        else:
            amplitudes[exposure_index] = np.nan
    prediction = amplitudes[:, None] * template[None, :]
    prediction[~np.isfinite(matrix)] = np.nan
    residual = matrix - prediction
    return {"template": template, "amplitude": amplitudes,
            "prediction": prediction, "residual": residual,
            "iterations": int(iterations)}


def _lowrank_iff_key(key):
    return "%s/%s/%s/%s" % tuple(key)


def _build_ifu_lowrank_matrix(rows):
    """Collapse raw post-iteration-2 c values to a physical-IFU matrix."""
    by_exposure = {}
    by_ifu = {}
    for row in rows:
        exposure_key = (Path(row["H5"]).name, int(row["exposure"]))
        ifu_key = _physical_ifu_key(row)
        by_exposure.setdefault(exposure_key, {}).setdefault(ifu_key, []).append(row)
        by_ifu.setdefault(ifu_key, []).append(row)
    exposure_keys = sorted(by_exposure)
    ifu_keys = sorted(by_ifu)
    ifu_index = {key: index for index, key in enumerate(ifu_keys)}
    matrix = np.full((len(exposure_keys), len(ifu_keys)), np.nan, dtype=float)
    blank_matrix = np.full_like(matrix, np.nan)
    center_lookup = {}
    for ifu_key, members in by_ifu.items():
        center_lookup[ifu_key] = (
            _robust_collapse([row.get("X_amp", np.nan) for row in members]),
            _robust_collapse([row.get("Y_amp", np.nan) for row in members]))
    cell_rows = []
    lookup = {}
    for exposure_index, exposure_key in enumerate(exposure_keys):
        members_by_ifu = by_exposure[exposure_key]
        for ifu_key in ifu_keys:
            members = members_by_ifu.get(ifu_key, [])
            c_values = np.asarray([row.get("c_ea", np.nan) for row in members],
                                  dtype=float)
            blank_values = np.asarray([
                row.get("blank_c_gray", np.nan) for row in members], dtype=float)
            finite_c = np.isfinite(c_values)
            finite_blank = np.isfinite(blank_values)
            n_amp_used = int(np.count_nonzero(finite_c))
            n_blank_amp_used = int(np.count_nonzero(finite_blank))
            c_value = (_robust_collapse(c_values)
                        if n_amp_used >= LOWRANK_MIN_C_AMP_PER_IFU else np.nan)
            blank_value = (_robust_collapse(blank_values)
                           if n_blank_amp_used >= LOWRANK_MIN_C_AMP_PER_IFU
                           else np.nan)
            x_value, y_value = center_lookup[ifu_key]
            if np.isfinite(c_value):
                matrix[exposure_index, ifu_index[ifu_key]] = c_value
            if np.isfinite(blank_value):
                blank_matrix[exposure_index, ifu_index[ifu_key]] = blank_value
            cell = {
                "H5": exposure_key[0], "exposure": exposure_key[1],
                "SPECID": ifu_key[0], "IFUSLOT": ifu_key[1],
                "IFUID": ifu_key[2], "IFU_CODE": ifu_key[3],
                "IFU_KEY": _lowrank_iff_key(ifu_key),
                "X_IFU": x_value, "Y_IFU": y_value,
                "N_amp_used": n_amp_used,
                "C_ei": c_value,
                "N_blank_amp_used": n_blank_amp_used,
                "B_ei": blank_value,
            }
            cell_rows.append(cell)
            lookup[(exposure_key, ifu_key)] = cell

    exposure_medians = np.asarray([
        _robust_collapse(matrix[index]) for index in range(matrix.shape[0])],
        dtype=float)
    blank_exposure_medians = np.asarray([
        _robust_collapse(blank_matrix[index])
        for index in range(blank_matrix.shape[0])], dtype=float)
    pattern = matrix - exposure_medians[:, None]
    blank_pattern = blank_matrix - blank_exposure_medians[:, None]
    for exposure_index, exposure_key in enumerate(exposure_keys):
        for ifu_key in ifu_keys:
            cell = lookup.get((exposure_key, ifu_key))
            if cell is None:
                continue
            cell["C_exposure_median"] = exposure_medians[exposure_index]
            cell["C_pattern"] = pattern[exposure_index, ifu_index[ifu_key]]
            cell["B_exposure_median"] = blank_exposure_medians[exposure_index]
            cell["B_pattern"] = blank_pattern[exposure_index, ifu_index[ifu_key]]

    center_rows = []
    for ifu_key in ifu_keys:
        center_rows.append({
            "SPECID": ifu_key[0], "IFUSLOT": ifu_key[1],
            "IFUID": ifu_key[2], "IFU_CODE": ifu_key[3],
            "IFU_KEY": _lowrank_iff_key(ifu_key),
            "X_IFU": center_lookup[ifu_key][0],
            "Y_IFU": center_lookup[ifu_key][1],
        })
    return {
        "exposure_keys": exposure_keys,
        "ifu_keys": ifu_keys,
        "ifu_index": ifu_index,
        "matrix": matrix,
        "blank_matrix": blank_matrix,
        "pattern": pattern,
        "blank_pattern": blank_pattern,
        "exposure_medians": exposure_medians,
        "blank_exposure_medians": blank_exposure_medians,
        "cell_rows": cell_rows,
        "cell_lookup": lookup,
        "ifu_center_rows": center_rows,
    }


def _attach_ifu_lowrank_rows(rows, data):
    for row in rows:
        exposure_key = (Path(row["H5"]).name, int(row["exposure"]))
        ifu_key = _physical_ifu_key(row)
        cell = data["cell_lookup"].get((exposure_key, ifu_key))
        if cell is None:
            row["C_ei"] = np.nan
            row["d_eia"] = np.nan
            row["C_pattern"] = np.nan
            row["N_amp_used"] = 0
            continue
        row["C_ei"] = cell["C_ei"]
        row["N_amp_used"] = cell["N_amp_used"]
        row["C_exposure_median"] = cell["C_exposure_median"]
        row["C_pattern"] = cell["C_pattern"]
        row["d_eia"] = (
            float(row["c_ea"] - cell["C_ei"])
            if np.isfinite(row.get("c_ea", np.nan)) and
            np.isfinite(cell["C_ei"]) else np.nan)


def _lowrank_pairwise_rows(data):
    matrix = data["pattern"]
    rows = []
    for first in range(matrix.shape[0]):
        for second in range(first + 1, matrix.shape[0]):
            first_key = data["exposure_keys"][first]
            second_key = data["exposure_keys"][second]
            common = np.isfinite(matrix[first]) & np.isfinite(matrix[second])
            x, y = matrix[first, common], matrix[second, common]
            correlations = _lowrank_correlations(x, y)
            fit = {"slope": np.nan, "scatter": np.nan, "n": int(x.size)}
            metrics = {"sigma_before": np.nan, "sigma_after": np.nan,
                       "fractional_improvement": np.nan,
                       "robust_variance_explained": np.nan, "n": int(x.size)}
            if x.size >= LOWRANK_MIN_COMMON_IFUS:
                fit = _origin_huber(x, y)
                metrics = _lowrank_scatter_metrics(y, fit["slope"] * x)
            rows.append({
                "pair_type": ("within_H5" if first_key[0] == second_key[0]
                              else "across_H5"),
                "H5_1": first_key[0], "exposure_1": first_key[1],
                "H5_2": second_key[0], "exposure_2": second_key[1],
                "N_common_IFU": int(x.size),
                "robust_slope": fit.get("slope", np.nan),
                "robust_residual_scatter": metrics["sigma_after"],
                "scatter_before_prediction": metrics["sigma_before"],
                "scatter_after_prediction": metrics["sigma_after"],
                "fractional_improvement": metrics["fractional_improvement"],
                "robust_variance_explained": metrics[
                    "robust_variance_explained"],
                "Pearson": correlations["pearson"],
                "Spearman": correlations["spearman"],
                "fit_status": ("adequate" if x.size >= LOWRANK_MIN_COMMON_IFUS
                               else "insufficient_common_IFU_support"),
            })
    return rows


def _lowrank_leave_one_exposure_out(data):
    matrix = data["pattern"]
    output = []
    for heldout in range(matrix.shape[0]):
        training = np.delete(matrix, heldout, axis=0)
        fit = _fit_robust_rank1(training)
        template = fit["template"]
        valid = np.isfinite(matrix[heldout]) & np.isfinite(template)
        amplitude = np.nan
        prediction = np.full(matrix.shape[1], np.nan, dtype=float)
        if np.count_nonzero(valid) >= LOWRANK_MIN_COMMON_IFUS:
            amp_fit = _origin_huber(template[valid], matrix[heldout, valid])
            amplitude = amp_fit["slope"]
            prediction[valid] = amplitude * template[valid]
        metrics = _lowrank_scatter_metrics(matrix[heldout], prediction)
        output.append({
            "H5": data["exposure_keys"][heldout][0],
            "exposure": data["exposure_keys"][heldout][1],
            "N_IFU_used": metrics["n"],
            "rank1_A_heldout": amplitude,
            "sigma_before": metrics["sigma_before"],
            "sigma_after_rank1": metrics["sigma_after"],
            "fractional_improvement": metrics["fractional_improvement"],
            "robust_variance_explained": metrics[
                "robust_variance_explained"],
            "fit_status": ("adequate" if metrics["n"] >= LOWRANK_MIN_COMMON_IFUS
                           else "insufficient_common_IFU_support"),
        })
    return output


def _lowrank_heldout_ifu_cv(data):
    """Predict each held-out IFU cell with exposure-blocked IFU coefficients."""
    matrix = data["pattern"]
    n_exposure, n_ifu = matrix.shape
    folds = {index: index % LOWRANK_N_FOLDS for index in range(n_ifu)}
    observed_by_exposure = [[] for _ in range(n_exposure)]
    residual_by_exposure = [[] for _ in range(n_exposure)]
    cell_rows = []
    used_folds = set()
    for fold in range(LOWRANK_N_FOLDS):
        train_ifu = np.asarray([index for index in range(n_ifu)
                                if folds[index] != fold], dtype=int)
        heldout_ifu = [index for index in range(n_ifu) if folds[index] == fold]
        if train_ifu.size < LOWRANK_MIN_COMMON_IFUS:
            continue
        fit = _fit_robust_rank1(matrix[:, train_ifu])
        amplitudes = fit["amplitude"]
        used_folds.add(fold)
        for ifu_index in heldout_ifu:
            for exposure_index in range(n_exposure):
                observed = matrix[exposure_index, ifu_index]
                if not np.isfinite(observed) or not np.isfinite(amplitudes[exposure_index]):
                    continue
                training_exposures = np.arange(n_exposure) != exposure_index
                training_exposures &= np.isfinite(amplitudes)
                training_exposures &= np.isfinite(matrix[:, ifu_index])
                if np.count_nonzero(training_exposures) < LOWRANK_MIN_TEMPLATE_SUPPORT:
                    continue
                t_fit = _origin_huber(
                    amplitudes[training_exposures],
                    matrix[training_exposures, ifu_index])
                template_value = t_fit["slope"]
                prediction = amplitudes[exposure_index] * template_value
                residual = observed - prediction
                observed_by_exposure[exposure_index].append(float(observed))
                residual_by_exposure[exposure_index].append(float(residual))
                exposure_key = data["exposure_keys"][exposure_index]
                ifu_key = data["ifu_keys"][ifu_index]
                cell_rows.append({
                    "H5": exposure_key[0], "exposure": exposure_key[1],
                    "SPECID": ifu_key[0], "IFUSLOT": ifu_key[1],
                    "IFUID": ifu_key[2], "IFU_CODE": ifu_key[3],
                    "IFU_KEY": _lowrank_iff_key(ifu_key),
                    "heldout_IFU_fold": int(fold),
                    "N_train_IFUs": int(train_ifu.size),
                    "N_train_exposures_for_IFU": int(
                        np.count_nonzero(training_exposures)),
                    "C_pattern_observed": float(observed),
                    "A_e_from_training_IFUs": float(amplitudes[exposure_index]),
                    "T_i_from_other_exposures": float(template_value),
                    "C_pattern_prediction": float(prediction),
                    "C_pattern_residual": float(residual),
                })
    summary = []
    for exposure_index, exposure_key in enumerate(data["exposure_keys"]):
        metrics = _lowrank_scatter_metrics(
            observed_by_exposure[exposure_index],
            np.asarray(observed_by_exposure[exposure_index]) -
            np.asarray(residual_by_exposure[exposure_index]))
        summary.append({
            "H5": exposure_key[0], "exposure": exposure_key[1],
            "N_IFU_cells": metrics["n"],
            "N_IFUs": len({row["IFU_KEY"] for row in cell_rows
                            if row["H5"] == exposure_key[0] and
                            row["exposure"] == exposure_key[1]}),
            "N_folds": int(len(used_folds)),
            "sigma_before": metrics["sigma_before"],
            "sigma_after_rank1": metrics["sigma_after"],
            "fractional_improvement": metrics["fractional_improvement"],
            "robust_variance_explained": metrics[
                "robust_variance_explained"],
        })
    return summary, cell_rows


def _lowrank_blank_transfer(data, image_fit):
    template = image_fit["template"]
    output = []
    for exposure_index, exposure_key in enumerate(data["exposure_keys"]):
        blank = data["blank_matrix"][exposure_index]
        valid = np.isfinite(blank) & np.isfinite(template)
        support = int(np.count_nonzero(np.isfinite(blank)))
        amplitude = np.nan
        prediction = np.full(blank.shape, np.nan, dtype=float)
        if (support >= LOWRANK_MIN_BLANK_IFUS and
                np.count_nonzero(valid) >= LOWRANK_MIN_COMMON_IFUS):
            fit = _origin_huber(template[valid], blank[valid])
            amplitude = fit["slope"]
            prediction[valid] = amplitude * template[valid]
        metrics = _lowrank_scatter_metrics(blank, prediction)
        correlations = _lowrank_correlations(blank[valid], template[valid])
        output.append({
            "H5": exposure_key[0], "exposure": exposure_key[1],
            "N_blank_IFU": support,
            "blank_support_gate_min_IFU": LOWRANK_MIN_BLANK_IFUS,
            "blank_support_gate_met": bool(
                support >= LOWRANK_MIN_BLANK_IFUS and
                np.count_nonzero(valid) >= LOWRANK_MIN_COMMON_IFUS),
            "N_common_template_IFU": int(np.count_nonzero(valid)),
            "blank_exposure_median": data["blank_exposure_medians"][exposure_index],
            "A_blank_e": amplitude,
            "blank_sigma_before": metrics["sigma_before"],
            "blank_sigma_after_image_template": metrics["sigma_after"],
            "blank_fractional_improvement": metrics["fractional_improvement"],
            "blank_robust_variance_explained": metrics[
                "robust_variance_explained"],
            "blank_vs_image_template_Pearson": correlations["pearson"],
            "blank_vs_image_template_Spearman": correlations["spearman"],
        })
    return output


def _lowrank_amp_residual_repeatability(rows):
    grouped = {}
    for row in rows:
        if not np.isfinite(row.get("d_eia", np.nan)):
            continue
        key = (Path(row["H5"]).name, str(row["SPECID"]),
               str(row["IFUSLOT"]), str(row["IFUID"]),
               str(row["IFU_CODE"]), str(row["AMP"]))
        grouped.setdefault(key, {})[int(row["exposure"])] = float(row["d_eia"])
    output = []
    for h5_name in sorted({key[0] for key in grouped}):
        selected = {key: values for key, values in grouped.items()
                    if key[0] == h5_name}
        for first, second in ((1, 2), (1, 3), (2, 3)):
            x, y = [], []
            for values in selected.values():
                if first in values and second in values:
                    x.append(values[first])
                    y.append(values[second])
            x, y = np.asarray(x), np.asarray(y)
            correlations = _lowrank_correlations(x, y)
            output.append({
                "H5": h5_name, "exposure_1": first, "exposure_2": second,
                "N_matched_physical_amplifiers": int(x.size),
                "d_eia_Pearson": correlations["pearson"],
                "d_eia_Spearman": correlations["spearman"],
                "d_eia_same_sign_fraction": (
                    float(np.mean(np.sign(x) == np.sign(y)))
                    if x.size else np.nan),
                "d_eia_difference_scatter": _robust_scale(y - x),
                "d_eia_scatter_first": _robust_scale(x),
                "d_eia_scatter_second": _robust_scale(y),
            })
    return output


def _lowrank_template_comparison(data, image_fit, blank_fit):
    image_template = image_fit["template"]
    blank_template = blank_fit["template"]
    valid = np.isfinite(image_template) & np.isfinite(blank_template)
    x, y = image_template[valid], blank_template[valid]
    correlations = _lowrank_correlations(x, y)
    fit = _origin_huber(x, y) if x.size >= LOWRANK_MIN_COMMON_IFUS else {}
    sign_agreement = (float(np.mean(np.sign(x) == np.sign(y)))
                      if x.size else np.nan)
    return {
        "N_common_IFU": int(x.size),
        "Pearson": correlations["pearson"],
        "Spearman": correlations["spearman"],
        "robust_regression_slope_T_blank_on_T_image": fit.get("slope", np.nan),
        "robust_residual_scatter": (
            _robust_scale(y - fit["slope"] * x)
            if fit.get("slope") is not None and np.isfinite(fit.get("slope", np.nan))
            else np.nan),
        "sign_agreement_fraction": sign_agreement,
        "fit_status": ("adequate" if x.size >= LOWRANK_MIN_COMMON_IFUS
                       else "insufficient_common_IFU_support"),
    }


def _lowrank_aggregate(summary, pair_rows, blank_template_comparison,
                       d_repeatability, matrix, rank1_fit,
                       loo_rows, heldout_rows, blank_transfer):
    def median_field(rows, field):
        values = _finite([row.get(field, np.nan) for row in rows])
        return float(np.median(values)) if values.size else np.nan

    within = [row for row in pair_rows
              if row["pair_type"] == "within_H5" and
              row["fit_status"] == "adequate"]
    across = [row for row in pair_rows
              if row["pair_type"] == "across_H5" and
              row["fit_status"] == "adequate"]
    matrix_values = _finite(matrix)
    prediction = rank1_fit["prediction"]
    rank1_metrics = _lowrank_scatter_metrics(matrix, prediction)
    loo_adequate = [row for row in loo_rows if row["fit_status"] == "adequate"]
    heldout_adequate = [row for row in heldout_rows if row["N_IFU_cells"] > 0]
    blank_adequate = [row for row in blank_transfer
                      if row["blank_support_gate_met"]]
    d_adequate = [row for row in d_repeatability
                  if row["N_matched_physical_amplifiers"] > 0]
    result = {
        "N_exposures": int(len(summary)),
        "N_physical_IFUs": int(matrix.shape[1]),
        "N_matrix_finite_C_cells": int(matrix_values.size),
        "rank1_iterations": int(rank1_fit["iterations"]),
        "rank1_sigma_before": rank1_metrics["sigma_before"],
        "rank1_sigma_after": rank1_metrics["sigma_after"],
        "rank1_fractional_improvement": rank1_metrics["fractional_improvement"],
        "rank1_robust_variance_explained": rank1_metrics[
            "robust_variance_explained"],
        "within_H5_pair_count": int(len(within)),
        "within_H5_median_Pearson": median_field(within, "Pearson"),
        "within_H5_median_Spearman": median_field(within, "Spearman"),
        "within_H5_median_fractional_improvement": median_field(
            within, "fractional_improvement"),
        "across_H5_pair_count": int(len(across)),
        "across_H5_median_Pearson": median_field(across, "Pearson"),
        "across_H5_median_Spearman": median_field(across, "Spearman"),
        "across_H5_median_fractional_improvement": median_field(
            across, "fractional_improvement"),
        "LOEO_N_exposures_adequate": int(len(loo_adequate)),
        "LOEO_median_sigma_before": median_field(loo_adequate, "sigma_before"),
        "LOEO_median_sigma_after_rank1": median_field(
            loo_adequate, "sigma_after_rank1"),
        "LOEO_median_fractional_improvement": median_field(
            loo_adequate, "fractional_improvement"),
        "LOEO_N_exposures_improved": int(sum(
            np.isfinite(row.get("sigma_before", np.nan)) and
            np.isfinite(row.get("sigma_after_rank1", np.nan)) and
            row["sigma_after_rank1"] < row["sigma_before"]
            for row in loo_adequate)),
        "heldout_IFU_N_exposures_with_cells": int(len(heldout_adequate)),
        "heldout_IFU_median_sigma_before": median_field(
            heldout_adequate, "sigma_before"),
        "heldout_IFU_median_sigma_after_rank1": median_field(
            heldout_adequate, "sigma_after_rank1"),
        "heldout_IFU_median_fractional_improvement": median_field(
            heldout_adequate, "fractional_improvement"),
        "heldout_IFU_N_exposures_improved": int(sum(
            np.isfinite(row.get("sigma_before", np.nan)) and
            np.isfinite(row.get("sigma_after_rank1", np.nan)) and
            row["sigma_after_rank1"] < row["sigma_before"]
            for row in heldout_adequate)),
        "blank_transfer_N_exposures_adequate": int(len(blank_adequate)),
        "blank_transfer_median_sigma_before": median_field(
            blank_adequate, "blank_sigma_before"),
        "blank_transfer_median_sigma_after": median_field(
            blank_adequate, "blank_sigma_after_image_template"),
        "blank_transfer_median_fractional_improvement": median_field(
            blank_adequate, "blank_fractional_improvement"),
        "blank_transfer_median_Pearson": median_field(
            blank_adequate, "blank_vs_image_template_Pearson"),
        "blank_transfer_median_Spearman": median_field(
            blank_adequate, "blank_vs_image_template_Spearman"),
        "blank_template_comparison": blank_template_comparison,
        "d_eia_median_scatter": median_field(summary, "d_eia_robust_scatter"),
        "d_eia_repeatability_N_rows": int(len(d_adequate)),
        "d_eia_repeatability_median_Pearson": median_field(
            d_adequate, "d_eia_Pearson"),
        "d_eia_repeatability_median_Spearman": median_field(
            d_adequate, "d_eia_Spearman"),
        "d_eia_repeatability_median_same_sign": median_field(
            d_adequate, "d_eia_same_sign_fraction"),
        "c_plane_model_retired": True,
        "c_plane_applied": False,
        "lowrank_uses_c_plane_fields": False,
    }
    return result


def _run_ifu_lowrank_diagnostic(rows):
    """Discover and validate one repeatable IFU-level rank-one c pattern."""
    data = _build_ifu_lowrank_matrix(rows)
    _attach_ifu_lowrank_rows(rows, data)
    rank1_fit = _fit_robust_rank1(data["pattern"])
    pair_rows = _lowrank_pairwise_rows(data)
    loo_rows = _lowrank_leave_one_exposure_out(data)
    heldout_rows, heldout_cell_rows = _lowrank_heldout_ifu_cv(data)
    blank_transfer = _lowrank_blank_transfer(data, rank1_fit)
    blank_pattern_for_fit = data["blank_pattern"].copy()
    for exposure_index in range(blank_pattern_for_fit.shape[0]):
        support = np.count_nonzero(np.isfinite(data["blank_matrix"][exposure_index]))
        if support < LOWRANK_MIN_BLANK_IFUS:
            blank_pattern_for_fit[exposure_index, :] = np.nan
    blank_fit = _fit_robust_rank1(blank_pattern_for_fit)
    blank_template_comparison = _lowrank_template_comparison(
        data, rank1_fit, blank_fit)
    d_repeatability = _lowrank_amp_residual_repeatability(rows)

    prediction = rank1_fit["prediction"]
    residual_pattern = rank1_fit["residual"]
    for cell in data["cell_rows"]:
        exposure_index = data["exposure_keys"].index((cell["H5"], cell["exposure"]))
        ifu_index = data["ifu_index"][(cell["SPECID"], cell["IFUSLOT"],
                                        cell["IFUID"], cell["IFU_CODE"])]
        cell["T_image"] = rank1_fit["template"][ifu_index]
        cell["A_e_image"] = rank1_fit["amplitude"][exposure_index]
        cell["rank1_prediction_pattern"] = prediction[exposure_index, ifu_index]
        cell["rank1_prediction_C"] = (
            cell["C_exposure_median"] + cell["rank1_prediction_pattern"]
            if np.isfinite(cell["C_exposure_median"]) and
            np.isfinite(cell["rank1_prediction_pattern"]) else np.nan)
        cell["rank1_residual_pattern"] = residual_pattern[exposure_index, ifu_index]
        cell["rank1_residual_C_minus_AT"] = (
            cell["C_ei"] - cell["rank1_prediction_pattern"]
            if np.isfinite(cell["C_ei"]) and
            np.isfinite(cell["rank1_prediction_pattern"]) else np.nan)
        cell["T_blank"] = blank_fit["template"][ifu_index]

    summary = []
    loo_by_key = {(row["H5"], row["exposure"]): row for row in loo_rows}
    heldout_by_key = {(row["H5"], row["exposure"]): row
                      for row in heldout_rows}
    blank_by_key = {(row["H5"], row["exposure"]): row
                    for row in blank_transfer}
    for exposure_index, exposure_key in enumerate(data["exposure_keys"]):
        c_values = data["matrix"][exposure_index]
        c_pattern = data["pattern"][exposure_index]
        prediction_pattern = prediction[exposure_index]
        c_cells = _finite(c_values)
        c_pattern_metrics = _lowrank_scatter_metrics(
            c_pattern, prediction_pattern)
        d_values = []
        c_amp_values = []
        for row in rows:
            if (Path(row["H5"]).name, int(row["exposure"])) != exposure_key:
                continue
            if np.isfinite(row.get("d_eia", np.nan)):
                d_values.append(row["d_eia"])
            if np.isfinite(row.get("c_ea", np.nan)):
                c_amp_values.append(row["c_ea"])
        ifu_c_scatter = _robust_scale(c_values)
        key = exposure_key
        summary_row = {
            "H5": exposure_key[0], "exposure": exposure_key[1],
            "c_plane_model_retired": True,
            "c_plane_applied": False,
            "N_physical_IFUs_total": int(len(data["ifu_keys"])),
            "N_IFU_with_C": int(c_cells.size),
            "N_amp_rows_with_c": int(len(c_amp_values)),
            "C_exposure_median": data["exposure_medians"][exposure_index],
            "C_ei_robust_scatter": ifu_c_scatter,
            "C_pattern_robust_scatter": c_pattern_metrics["sigma_before"],
            "c_ea_robust_scatter": _robust_scale(c_amp_values),
            "d_eia_robust_scatter": _robust_scale(d_values),
            "rank1_A_e": rank1_fit["amplitude"][exposure_index],
            "rank1_sigma_before": c_pattern_metrics["sigma_before"],
            "rank1_sigma_after_in_sample": c_pattern_metrics["sigma_after"],
            "rank1_fractional_improvement_in_sample":
                c_pattern_metrics["fractional_improvement"],
            "rank1_robust_variance_explained_in_sample":
                c_pattern_metrics["robust_variance_explained"],
        }
        summary_row.update({
            "LOEO_sigma_before": loo_by_key[key]["sigma_before"],
            "LOEO_sigma_after_rank1": loo_by_key[key]["sigma_after_rank1"],
            "LOEO_fractional_improvement": loo_by_key[key][
                "fractional_improvement"],
            "LOEO_robust_variance_explained": loo_by_key[key][
                "robust_variance_explained"],
            "heldout_IFU_sigma_before": heldout_by_key[key]["sigma_before"],
            "heldout_IFU_sigma_after_rank1": heldout_by_key[key][
                "sigma_after_rank1"],
            "heldout_IFU_fractional_improvement": heldout_by_key[key][
                "fractional_improvement"],
            "heldout_IFU_robust_variance_explained": heldout_by_key[key][
                "robust_variance_explained"],
            "N_heldout_IFU_cells": heldout_by_key[key]["N_IFU_cells"],
            "N_heldout_IFUs": heldout_by_key[key]["N_IFUs"],
            "N_heldout_IFU_folds": heldout_by_key[key]["N_folds"],
        })
        summary_row.update({
            "N_blank_IFU": blank_by_key[key]["N_blank_IFU"],
            "blank_support_gate_met": blank_by_key[key][
                "blank_support_gate_met"],
            "A_blank_e": blank_by_key[key]["A_blank_e"],
            "blank_sigma_before": blank_by_key[key]["blank_sigma_before"],
            "blank_sigma_after_image_template": blank_by_key[key][
                "blank_sigma_after_image_template"],
            "blank_fractional_improvement": blank_by_key[key][
                "blank_fractional_improvement"],
            "blank_vs_image_template_Pearson": blank_by_key[key][
                "blank_vs_image_template_Pearson"],
            "blank_vs_image_template_Spearman": blank_by_key[key][
                "blank_vs_image_template_Spearman"],
        })
        summary.append(summary_row)

    aggregate = _lowrank_aggregate(
        summary, pair_rows, blank_template_comparison, d_repeatability,
        data["pattern"], rank1_fit, loo_rows, heldout_rows, blank_transfer)
    return {
        "data": data,
        "rank1_fit": rank1_fit,
        "blank_fit": blank_fit,
        "pair_rows": pair_rows,
        "loo_rows": loo_rows,
        "heldout_rows": heldout_rows,
        "heldout_cell_rows": heldout_cell_rows,
        "blank_transfer": blank_transfer,
        "blank_template_comparison": blank_template_comparison,
        "d_repeatability": d_repeatability,
        "summary": summary,
        "aggregate": aggregate,
    }


def _lowrank_matrix_wide_rows(data):
    """Write the exposure x physical-IFU C matrix in auditable wide form."""
    fields = ["H5", "exposure", "C_exposure_median"]
    labels = [_lowrank_iff_key(key).replace("/", "_")
              for key in data["ifu_keys"]]
    fields.extend("C_%s" % label for label in labels)
    rows = []
    for exposure_index, exposure_key in enumerate(data["exposure_keys"]):
        row = {
            "H5": exposure_key[0], "exposure": exposure_key[1],
            "C_exposure_median": data["exposure_medians"][exposure_index],
        }
        for ifu_index, label in enumerate(labels):
            row["C_%s" % label] = data["matrix"][exposure_index, ifu_index]
        rows.append(row)
    return rows, fields


def _write_iteration2_summary(path, exposure_rows, iteration1_qa_rows,
                              iteration2_qa_rows, repeat_rows,
                              repeat_summary, preflight, validation,
                              focal_summary=None, focal_join=None,
                              focal_aggregate=None, lowrank=None):
    g2 = _finite([row["g2_raw"] for row in exposure_rows])
    g3 = _finite([row["g3_raw"] for row in exposure_rows])
    g2_minus_one = g2 - 1.0
    g3_minus_one = g3 - 1.0
    closure_groups = _closure_group_stats(exposure_rows)
    c_stats = _qa_field_stats(iteration2_qa_rows, "image_c_gray")
    q_stats = _qa_field_stats(iteration2_qa_rows, "image_q_color")
    final_on = _qa_field_stats(iteration2_qa_rows, "image_fractional_ON")
    final_off = _qa_field_stats(iteration2_qa_rows, "image_fractional_OFF")
    iter1_on = _qa_field_stats(iteration1_qa_rows, "image_fractional_ON")
    iter1_off = _qa_field_stats(iteration1_qa_rows, "image_fractional_OFF")
    warn1 = sum(row["QA_STATUS"] == "WARN" for row in iteration1_qa_rows)
    warn2 = sum(row["QA_STATUS"] == "WARN" for row in iteration2_qa_rows)
    final_rows = [row for row in iteration2_qa_rows
                  if np.isfinite(row.get("image_c_gray", np.nan))]
    gray_ranked = _ranked_cq_rows(iteration2_qa_rows, "c_gray")
    color_ranked = _ranked_cq_rows(iteration2_qa_rows, "q_color")
    c_repeat = [row for row in repeat_summary if row["component"] == "c"]
    q_repeat = [row for row in repeat_summary if row["component"] == "q"]
    c_mean_corr = np.nanmean([row["Pearson_correlation"] for row in c_repeat]) \
        if c_repeat else np.nan
    q_mean_corr = np.nanmean([row["Pearson_correlation"] for row in q_repeat]) \
        if q_repeat else np.nan
    c_mean_sign = np.nanmean([row["fraction_same_sign"] for row in c_repeat]) \
        if c_repeat else np.nan
    q_mean_sign = np.nanmean([row["fraction_same_sign"] for row in q_repeat]) \
        if q_repeat else np.nan

    def stats_line(label, stats):
        return ("%s N=%d median=%s robust_scatter=%s p16=%s p84=%s "
                "min=%s max=%s" % (
                    label, stats["N"], _format_number(stats["median"]),
                    _format_number(stats["robust_scatter"]),
                    _format_number(stats["p16"]), _format_number(stats["p84"]),
                    _format_number(stats["min"]), _format_number(stats["max"])))

    lines = [
        "M101 SB exposure-scale iteration-2 and c/q QA summary",
        "created_utc=%s" % datetime.now(timezone.utc).isoformat(),
        "SB_iterations_applied=2",
        "third_iteration_applied=False",
        "c_q_used_for_calibration=False",
        "amplifier_rejection_applied=False",
        "science_cube_built=False",
        "LSF_run=False",
        "preflight=%s" % json.dumps(_json_ready(preflight), sort_keys=True),
        "validation=%s" % json.dumps(_json_ready(validation), sort_keys=True),
        "",
        "1. Did the second SB iteration materially improve closure?",
        "iteration1_closure_g2_raw_minus_one=" +
        json.dumps(_json_ready(_value_stats(g2_minus_one, thresholds=True)),
                   sort_keys=True),
        "iteration2_closure_g3_raw_minus_one=" +
        json.dumps(_json_ready(_value_stats(g3_minus_one, thresholds=True)),
                   sort_keys=True),
        "closure_robust_scatter_g2_minus_one=%s" %
        _format_number(_robust_scale(g2_minus_one)),
        "closure_robust_scatter_g3_minus_one=%s" %
        _format_number(_robust_scale(g3_minus_one)),
        "closure_scatter_change_g3_minus_g2=%s" % _format_number(
            _robust_scale(g3_minus_one) - _robust_scale(g2_minus_one)),
        "closure_materially_improved=%s" % (
            "yes" if (np.isfinite(_robust_scale(g2_minus_one)) and
                      np.isfinite(_robust_scale(g3_minus_one)) and
                      _robust_scale(g3_minus_one) < _robust_scale(g2_minus_one))
            else "no"),
        "",
        "g3 closure groups by uncertainty (g3_raw statistics and residual counts):",
    ]
    for name in ("all", "sigma_g3_lt_0.02", "sigma_g3_0.02_to_0.05",
                 "sigma_g3_ge_0.05"):
        group = closure_groups[name]
        raw = {key[3:]: value for key, value in group.items()
               if key.startswith("g3_")}
        lines.append(name + " " + stats_line("g3_raw", raw) +
                     " abs(g3-1)>0.01=%d abs(g3-1)>0.02=%d "
                     "abs(g3-1)>0.05=%d" % (
                         group["g3_abs_gt_0.01"], group["g3_abs_gt_0.02"],
                         group["g3_abs_gt_0.05"]))

    if closure_groups["sigma_g3_lt_0.02"]["g3_N"]:
        well = closure_groups["sigma_g3_lt_0.02"]
        well_answer = ("well_constrained_g3 median=%s robust_scatter=%s "
                       "p16=%s p84=%s" % (
                           _format_number(well["g3_median"]),
                           _format_number(well["g3_robust_scatter"]),
                           _format_number(well["g3_p16"]),
                           _format_number(well["g3_p84"])))
    else:
        well_answer = "well_constrained_g3 unavailable"
    lines.extend([
        "",
        "2. For the well-constrained exposure population, how close is final g3 to unity?",
        well_answer,
        "",
        "3. Is the ordinary amplifier residual population dominated by c_ea or q_ea?",
        stats_line("image_c_gray", c_stats),
        stats_line("image_q_color", q_stats),
        "ordinary_population_dominant=%s" % (
            "c_ea" if (np.isfinite(c_stats["robust_scatter"]) and
                       np.isfinite(q_stats["robust_scatter"]) and
                       c_stats["robust_scatter"] > q_stats["robust_scatter"])
            else "q_ea" if (np.isfinite(c_stats["robust_scatter"]) and
                            np.isfinite(q_stats["robust_scatter"]) and
                            q_stats["robust_scatter"] > c_stats["robust_scatter"])
            else "indeterminate"),
        "",
        "4. Does c_ea repeat for the same physical amplifier across exposures 1/2/3?",
        "c_repeatability_assessment=positive_pair_correlations_but_same_sign_fraction=%s" %
        _format_number(c_mean_sign),
    ])
    for row in repeat_summary:
        if row["component"] == "c":
            lines.append("c %s N=%d Pearson=%s median_difference=%s "
                         "robust_scatter_difference=%s same_sign=%s" % (
                row["exposure_pair"], row["N_matched_amplifiers"],
                _format_number(row["Pearson_correlation"]),
                _format_number(row["median_difference_second_minus_first"]),
                _format_number(row["robust_scatter_difference"]),
                _format_number(row["fraction_same_sign"])))
    lines.extend([
        "",
        "5. Does q_ea show stronger physical-amplifier persistence than c_ea?",
        "q_vs_c_mean_pair_Pearson=%s_vs_%s q_same_sign=%s_vs_c_same_sign=%s" % (
            _format_number(q_mean_corr), _format_number(c_mean_corr),
            _format_number(q_mean_sign), _format_number(c_mean_sign)),
        "q_stronger_persistence_by_mean_Pearson=%s" % (
            "yes" if np.isfinite(q_mean_corr) and np.isfinite(c_mean_corr) and
            q_mean_corr > c_mean_corr else "no_or_indeterminate"),
    ])
    for row in repeat_summary:
        if row["component"] == "q":
            lines.append("q %s N=%d Pearson=%s median_difference=%s "
                         "robust_scatter_difference=%s same_sign=%s" % (
                row["exposure_pair"], row["N_matched_amplifiers"],
                _format_number(row["Pearson_correlation"]),
                _format_number(row["median_difference_second_minus_first"]),
                _format_number(row["robust_scatter_difference"]),
                _format_number(row["fraction_same_sign"])))
    lines.extend([
        "",
        "6. Which catastrophic/repeated amplifiers remain obvious after iteration 2?",
        "strongest_abs_c_ea:",
    ])
    for row in gray_ranked[:15]:
        lines.append(_cq_summary_line(row, "c"))
    lines.append("strongest_abs_q_ea:")
    for row in color_ranked[:15]:
        lines.append(_cq_summary_line(row, "q"))

    blank_matches = []
    for row in final_rows:
        if (np.isfinite(row.get("blank_c_gray", np.nan)) and
                np.isfinite(row.get("blank_q_color", np.nan))):
            blank_matches.append(row)
    gray_blank_sign = [np.sign(row["image_c_gray"]) ==
                       np.sign(row["blank_c_gray"]) for row in blank_matches]
    q_blank_sign = [np.sign(row["image_q_color"]) ==
                    np.sign(row["blank_q_color"]) for row in blank_matches]
    lines.extend([
        "",
        "7. Where BLANK_QA exists, do its gray/color decompositions agree qualitatively?",
        "blank_cq_matches=%d image_c_same_sign=%s image_q_same_sign=%s" % (
            len(blank_matches),
            _format_number(np.mean(gray_blank_sign) if gray_blank_sign else np.nan),
            _format_number(np.mean(q_blank_sign) if q_blank_sign else np.nan)),
        "blank_qa_qualitative_agreement=%s" % (
            "yes" if gray_blank_sign and q_blank_sign and
            np.mean(gray_blank_sign) >= 0.5 and np.mean(q_blank_sign) >= 0.5
            else "no_or_unavailable"),
        "",
        "QA comparison: iteration-1 post-SB versus iteration-2",
        stats_line("iteration1 IMAGE_QA ON", iter1_on),
        stats_line("iteration2 IMAGE_QA ON", final_on),
        stats_line("iteration1 IMAGE_QA OFF", iter1_off),
        stats_line("iteration2 IMAGE_QA OFF", final_off),
        "iteration1_WARN_count=%d" % warn1,
        "iteration2_WARN_count=%d" % warn2,
        "iteration1_ON_p01_p99=%s %s" % (_format_number(iter1_on["p01"]),
                                           _format_number(iter1_on["p99"])),
        "iteration1_OFF_p01_p99=%s %s" % (_format_number(iter1_off["p01"]),
                                            _format_number(iter1_off["p99"])),
        "iteration2_ON_p01_p99=%s %s" % (_format_number(final_on["p01"]),
                                           _format_number(final_on["p99"])),
        "iteration2_OFF_p01_p99=%s %s" % (_format_number(final_off["p01"]),
                                            _format_number(final_off["p99"])),
        "warn_count_is_not_an_iteration_acceptance_rule=True",
        "",
        "No g3 correction was applied. c_ea and q_ea are QA coordinates only.",
        "No amplifier rejection, science cube, or LSF product was produced.",
    ])
    if (focal_summary is not None and focal_join is not None and
            focal_aggregate is not None):
        materially_nonzero = [row for row in focal_summary
                              if row.get("median_c_ea_materially_nonzero", False)]
        lines.extend([
            "",
            "Historical focal-plane plane diagnostic (retired; report-only):",
            "coordinate_source=m101_band_cache.npz x_arcmin/y_arcmin",
            "coordinate_units=arcmin",
            "coordinate_semantics=exposure-centered tangent-plane coordinates used by historical Model-3 illumination fit",
            "mechanical_fplane_coordinates_claimed=False",
            "plane_equation=c_ea or q_ea = bx*(X-X0) + by*(Y-Y0); no correction applied; coefficients not used by low-rank analysis",
            "post_iter2_focal_plane_exposures=%d" % len(focal_summary),
            "post_iter2_focal_plane_join=%s" % json.dumps(
                _json_ready(focal_join), sort_keys=True),
            "median_c_ea_material_threshold=%.6g" %
            FOCAL_MEDIAN_C_MATERIAL_THRESHOLD,
            "median_c_ea_materially_nonzero_exposures=%d" %
            len(materially_nonzero),
            "median_c_scatter_before_plane=%s" % _format_number(
                focal_aggregate["median_c_scatter_before_plane"]),
            "median_c_scatter_after_in_sample_plane=%s" % _format_number(
                focal_aggregate["median_c_scatter_after_in_sample_plane"]),
            "median_c_scatter_after_heldout_IFU_prediction=%s" % _format_number(
                focal_aggregate["median_c_scatter_after_heldout_IFU_prediction"]),
            "median_q_scatter_before_plane=%s" % _format_number(
                focal_aggregate["median_q_scatter_before_plane"]),
            "median_q_scatter_after_in_sample_plane=%s" % _format_number(
                focal_aggregate["median_q_scatter_after_in_sample_plane"]),
            "median_q_scatter_after_heldout_IFU_prediction=%s" % _format_number(
                focal_aggregate["median_q_scatter_after_heldout_IFU_prediction"]),
            "heldout_c_scatter_improves=%d" % focal_aggregate[
                "N_exposures_heldout_c_scatter_improves"],
            "heldout_q_scatter_improves=%d" % focal_aggregate[
                "N_exposures_heldout_q_scatter_improves"],
            "adequate_BLANK_QA_focal_support=%d (minimum distinct IFUs=%d)" % (
                focal_aggregate["N_exposures_adequate_BLANK_QA_focal_support"],
                focal_aggregate["blank_support_min_distinct_IFUs"]),
            "adequate_blank_image_gradient_cosine_median=%s" % _format_number(
                focal_aggregate["adequate_blank_image_gradient_cosine_median"]),
            "adequate_blank_image_gradient_abs_angle_difference_median=%s deg" %
            _format_number(
                focal_aggregate[
                    "adequate_blank_image_gradient_abs_angle_difference_median"]),
            "amp_IFU_gradient_cosine_median=%s" % _format_number(
                focal_aggregate["amp_IFU_gradient_cosine_median"]),
            "amp_IFU_gradient_abs_angle_difference_median=%s deg" %
            _format_number(focal_aggregate[
                "amp_IFU_gradient_abs_angle_difference_median"]),
        ])
        for row in materially_nonzero:
            lines.append("median_c_ea_materially_nonzero %s/e%d median_c_ea=%s" %
                         (row["H5"], row["exposure"],
                          _format_number(row["median_c_ea"])))
    if lowrank is not None:
        aggregate = lowrank["aggregate"]
        blank_compare = aggregate["blank_template_comparison"]
        lines.extend([
            "",
            "8. Repeatable low-rank physical-IFU focal-pattern diagnostic:",
            "c_plane_model_retired=True",
            "c_plane_applied=False",
            "lowrank_coordinate_use=historical M101 x/y only for plotting",
            "lowrank_matrix=%d exposures x %d physical IFUs" % (
                aggregate["N_exposures"], aggregate["N_physical_IFUs"]),
            "rank1_sigma_before=%s" % _format_number(
                aggregate["rank1_sigma_before"]),
            "rank1_sigma_after=%s" % _format_number(
                aggregate["rank1_sigma_after"]),
            "rank1_fractional_improvement=%s" % _format_number(
                aggregate["rank1_fractional_improvement"]),
            "rank1_robust_variance_explained=%s" % _format_number(
                aggregate["rank1_robust_variance_explained"]),
            "within_H5_pair_median_Pearson=%s Spearman=%s" % (
                _format_number(aggregate["within_H5_median_Pearson"]),
                _format_number(aggregate["within_H5_median_Spearman"])),
            "across_H5_pair_median_Pearson=%s Spearman=%s" % (
                _format_number(aggregate["across_H5_median_Pearson"]),
                _format_number(aggregate["across_H5_median_Spearman"])),
            "LOEO_median_sigma_before=%s" % _format_number(
                aggregate["LOEO_median_sigma_before"]),
            "LOEO_median_sigma_after_rank1=%s" % _format_number(
                aggregate["LOEO_median_sigma_after_rank1"]),
            "LOEO_median_fractional_improvement=%s" % _format_number(
                aggregate["LOEO_median_fractional_improvement"]),
            "LOEO_exposures_improved=%d" % aggregate[
                "LOEO_N_exposures_improved"],
            "heldout_IFU_median_sigma_before=%s" % _format_number(
                aggregate["heldout_IFU_median_sigma_before"]),
            "heldout_IFU_median_sigma_after_rank1=%s" % _format_number(
                aggregate["heldout_IFU_median_sigma_after_rank1"]),
            "heldout_IFU_median_fractional_improvement=%s" % _format_number(
                aggregate["heldout_IFU_median_fractional_improvement"]),
            "heldout_IFU_exposures_improved=%d" % aggregate[
                "heldout_IFU_N_exposures_improved"],
            "IMAGE_to_BLANK_transfer_N_exposures=%d" % aggregate[
                "blank_transfer_N_exposures_adequate"],
            "IMAGE_to_BLANK_transfer_median_sigma_before=%s" % _format_number(
                aggregate["blank_transfer_median_sigma_before"]),
            "IMAGE_to_BLANK_transfer_median_sigma_after=%s" % _format_number(
                aggregate["blank_transfer_median_sigma_after"]),
            "IMAGE_to_BLANK_transfer_median_fractional_improvement=%s" %
            _format_number(aggregate["blank_transfer_median_fractional_improvement"]),
            "IMAGE_to_BLANK_transfer_median_Pearson=%s Spearman=%s" % (
                _format_number(aggregate["blank_transfer_median_Pearson"]),
                _format_number(aggregate["blank_transfer_median_Spearman"])),
            "BLANK_only_template_common_IFUs=%d Pearson=%s Spearman=%s slope=%s sign_agreement=%s" % (
                blank_compare["N_common_IFU"],
                _format_number(blank_compare["Pearson"]),
                _format_number(blank_compare["Spearman"]),
                _format_number(blank_compare[
                    "robust_regression_slope_T_blank_on_T_image"]),
                _format_number(blank_compare["sign_agreement_fraction"])),
            "d_eia_median_scatter=%s repeatability_Pearson=%s Spearman=%s same_sign=%s" % (
                _format_number(aggregate["d_eia_median_scatter"]),
                _format_number(aggregate["d_eia_repeatability_median_Pearson"]),
                _format_number(aggregate["d_eia_repeatability_median_Spearman"]),
                _format_number(aggregate["d_eia_repeatability_median_same_sign"])),
            "rank_gt_1_fitted=False",
            "No rank-1 template was applied to spectra.",
        ])
    Path(path).write_text("\n".join(lines) + "\n")


def _cq_summary_line(row, component):
    return ("%s/e%d %s/%s/%s/%s f_ON=%s f_OFF=%s c_ea=%s q_ea=%s "
            "z_ON=%s z_OFF=%s BLANK_QA=%s blank_c=%s blank_q=%s" % (
                row["H5"], row["exposure"], row["SPECID"], row["IFUSLOT"],
                row["IFUID"], row["AMP"],
                _format_number(row["image_fractional_ON"]),
                _format_number(row["image_fractional_OFF"]),
                _format_number(row["image_c_gray"]),
                _format_number(row["image_q_color"]),
                _format_number(row["image_z_ON"]),
                _format_number(row["image_z_OFF"]),
                row.get("BLANK_QA_STATUS", ""),
                _format_number(row.get("blank_c_gray", np.nan)),
                _format_number(row.get("blank_q_color", np.nan))))


def _format_number(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return str(value)
    return "nan" if not np.isfinite(value) else "%.7g" % value


def _write_summary(path, exposure_rows, pre_qa_rows, post_qa_rows,
                   pre_summary, post_summary, preflight, provenance):
    closure = _finite([row["g_SB_residual_joint"] for row in exposure_rows])
    residual_minus_one = closure - 1.0
    pre_stats = _qa_grouped_stats(pre_qa_rows)
    post_stats = _qa_grouped_stats(post_qa_rows)
    warn_pre = sum(row["QA_STATUS"] == "WARN" for row in pre_qa_rows)
    warn_post = sum(row["QA_STATUS"] == "WARN" for row in post_qa_rows)
    weak = sorted(exposure_rows,
                  key=lambda row: (float(row["sigma_g_SB"])
                                   if np.isfinite(row["sigma_g_SB"])
                                   else -np.inf), reverse=True)[:5]
    queue = sorted(post_qa_rows,
                   key=lambda row: max([
                       abs(row["image_z_%s" % band]) for band in SB_BANDS
                       if np.isfinite(row["image_z_%s" % band])] or [0.0]),
                   reverse=True)
    lines = [
        "M101 SB exposure-scale diagnostic summary",
        "created_utc=%s" % datetime.now(timezone.utc).isoformat(),
        "calibration_source=make_m101_cube_current_model._calibrate_h5_current",
        "exposure_scale_method=amplifier-collapsed surface brightness",
        "exposure_scale_gauge=median raw g_SB_joint -> 1",
        "exposure_scale_applied=True; diagnostic_only=True",
        "science_cube_built=False; LSF_run=False; amplifier_rejection_applied=False",
        "FIBER_AREA=%.12g (pi*0.75**2)" % FIBER_AREA,
        "preflight=%s" % json.dumps(preflight, sort_keys=True),
        "",
        "index H5 exposure raw_g center relative correction residual sigma_residual",
    ]
    for row in exposure_rows:
        lines.append("%d %s %d %s %s %s %s %s %s" % (
            row["chronological_exposure_index"], row["H5"], row["exposure"],
            _format_number(row["g_SB_joint_raw"]),
            _format_number(row["g_SB_center"]),
            _format_number(row["g_SB_relative"]),
            _format_number(row["exposure_correction"]),
            _format_number(row["g_SB_residual_joint"]),
            _format_number(row["sigma_g_SB_residual"])))
    lines.extend([
        "", "closure_N=%d" % closure.size,
        "closure_median=%s" % _format_number(np.median(closure) if closure.size else np.nan),
        "closure_robust_scatter=%s" % _format_number(_robust_scale(closure)),
        "closure_p16_p84=%s %s" % tuple(_format_number(value) for value in (
            np.percentile(closure, 16) if closure.size else np.nan,
            np.percentile(closure, 84) if closure.size else np.nan)),
        "closure_min_max=%s %s" % tuple(_format_number(value) for value in (
            np.min(closure) if closure.size else np.nan,
            np.max(closure) if closure.size else np.nan)),
        "closure_abs_residual_gt_1pct=%d" % np.count_nonzero(
            np.abs(residual_minus_one) > 0.01),
        "closure_abs_residual_gt_2pct=%d" % np.count_nonzero(
            np.abs(residual_minus_one) > 0.02),
        "closure_abs_residual_gt_5pct=%d" % np.count_nonzero(
            np.abs(residual_minus_one) > 0.05), "",
        "pre_image_fractional_ON=%s" % json.dumps(
            _json_ready(pre_summary["image_fractional_ON"]), sort_keys=True),
        "post_image_fractional_ON=%s" % json.dumps(
            _json_ready(post_summary["image_fractional_ON"]), sort_keys=True),
        "pre_image_fractional_OFF=%s" % json.dumps(
            _json_ready(pre_summary["image_fractional_OFF"]), sort_keys=True),
        "post_image_fractional_OFF=%s" % json.dumps(
            _json_ready(post_summary["image_fractional_OFF"]), sort_keys=True),
        "pre_WARN_rows=%d" % warn_pre, "post_WARN_rows=%d" % warn_post,
        "pre_QA_rows=%d post_QA_rows=%d" %
        (len(pre_qa_rows), len(post_qa_rows)), "",
        "weak_SB_exposures_largest_sigma:",
    ])
    for row in weak:
        key = (row["H5"], row["exposure"])
        lines.append("%s/e%d sigma=%s pre_ON_scatter=%s post_ON_scatter=%s "
                     "pre_OFF_scatter=%s post_OFF_scatter=%s" % (
            row["H5"], row["exposure"], _format_number(row["sigma_g_SB"]),
            _format_number(pre_stats[key]["ON_scatter"]),
            _format_number(post_stats[key]["ON_scatter"]),
            _format_number(pre_stats[key]["OFF_scatter"]),
            _format_number(post_stats[key]["OFF_scatter"])))
    lines.extend(["", "strongest_post_IMAGE_QA_outliers:"])
    for row in queue[:15]:
        z = max([abs(row["image_z_%s" % band]) for band in SB_BANDS
                 if np.isfinite(row["image_z_%s" % band])] or [np.nan])
        lines.append("%s/e%d %s/%s/%s/%s max_abs_image_z=%s "
                     "fractional_ON=%s fractional_OFF=%s" % (
            row["H5"], row["exposure"], row["SPECID"], row["IFUSLOT"],
            row["IFUID"], row["AMP"], _format_number(z),
            _format_number(row["image_fractional_ON"]),
            _format_number(row["image_fractional_OFF"])))
    known = [row for row in post_qa_rows
             if str(row["SPECID"]) == "401" and str(row["IFUSLOT"]) == "79"
             and str(row["IFUID"]) == "68" and str(row["AMP"]).upper() == "RL"]
    lines.append("401/79/68/RL_post_rows=%d" % len(known))
    for row in known:
        lines.append("401/79/68/RL e%d fractional_ON=%s fractional_OFF=%s "
                     "image_z_ON=%s image_z_OFF=%s QA_STATUS=%s" % (
            row["exposure"], _format_number(row["image_fractional_ON"]),
            _format_number(row["image_fractional_OFF"]),
            _format_number(row["image_z_ON"]), _format_number(row["image_z_OFF"]),
            row["QA_STATUS"]))
    lines.extend(["", "ready_for_cube_assessment_inputs=closure, exposure-relative QA scatter, WARN count, and explicit residual outlier list above",
                  "No amplifier rejection was applied; no cube or LSF was built."])
    Path(path).write_text("\n".join(lines) + "\n")


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("h5files", nargs="+", help="H5 files or glob patterns")
    parser.add_argument("--output-dir", default="m101_exposure_scale_diagnostic")
    parser.add_argument("--state", default="m101_onoff_hierarchy/m101_onoff_state.json")
    parser.add_argument("--band-cache", default="m101_band_initializer/m101_band_cache.npz")
    parser.add_argument("--alpha-fits", default="m101_alpha_template_bands/alpha_template_fits.csv")
    parser.add_argument("--alpha-summary", default="m101_alpha_same_amp_loo/alpha_physical_amp_summary.csv")
    parser.add_argument("--amp-residual-models", default="m101-amp-residual-pca-final/amp_residual_pca_models.npz")
    parser.add_argument("--fq-template", required=True, help="fixed q,f template CSV")
    parser.add_argument("--external-blank-fibers", required=True)
    parser.add_argument("--on-filter", default=None)
    parser.add_argument("--off-filter", default=None)
    parser.add_argument("--min-sky-fibers", type=int, default=cube_model.M101_SKY_MIN_FIBERS)
    parser.add_argument("--pixel-scale", type=float, default=1.0)
    parser.add_argument("--image-center-size", default=None)
    parser.add_argument("--qa-min-blank", type=int, default=5)
    parser.add_argument("--bootstrap-draws", type=int, default=DEFAULT_BOOTSTRAP_DRAWS)
    parser.add_argument("--allow-noncanonical-counts", action="store_true",
                        help="permit subsets for development")
    return parser.parse_args()


def main():
    args = _parse_args()
    if not np.isclose(FIBER_AREA, EXPECTED_FIBER_AREA,
                      rtol=0.0, atol=1.0e-14):
        raise RuntimeError("current IMAGE_QA FIBER_AREA changed: %.15g" % FIBER_AREA)
    if args.pixel_scale <= 0 or not np.isfinite(args.pixel_scale):
        raise SystemExit("--pixel-scale must be finite and positive")
    if args.qa_min_blank < 1 or args.min_sky_fibers < 1:
        raise SystemExit("QA and sky minimums must be positive")
    if args.bootstrap_draws < 20:
        raise SystemExit("--bootstrap-draws must be at least 20")

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    h5files = _resolve_h5files(args.h5files)
    if len(h5files) != N_EXPECTED_H5 and not args.allow_noncanonical_counts:
        raise ValueError("expected 19 accepted H5 files, found %d" % len(h5files))

    state_path = Path(args.state).expanduser().resolve()
    state, response, composition, _ = cube_model._load_frozen_response(state_path)
    identity_map, cache_manifest, cache_manifest_path, cache_path = \
        cube_model._load_identity_map(args.band_cache, h5files)
    alpha_direct, alpha_fits_path = cube_model._load_alpha_fits(args.alpha_fits)
    alpha_history, alpha_summary_path = cube_model._load_alpha_history(args.alpha_summary)
    amp_means, mean_science_mask, mean_scope, amp_means_path = \
        cube_model._load_amp_means(args.amp_residual_models)
    fq_template, historical_f0, fq_path = cube_model._load_fq(args.fq_template)
    blank_path = Path(args.external_blank_fibers).expanduser().resolve()
    blank_masks, blank_provenance = cube_model.frozen_parent.m101_blank_fibers.load(
        blank_path, h5files)
    on_filter = args.on_filter or cube_model.frozen_parent._default_input("on_filter")
    off_filter = args.off_filter or cube_model.frozen_parent._default_input("off_filter")
    if not on_filter or not Path(on_filter).exists():
        raise FileNotFoundError("ON filter is unavailable; pass --on-filter")
    if not off_filter or not Path(off_filter).exists():
        raise FileNotFoundError("OFF filter is unavailable; pass --off-filter")
    on_filter_values = cube_model.validated_m101.read_filter(on_filter)
    off_filter_values = cube_model.validated_m101.read_filter(off_filter)
    band_responses, null_provenance = native._band_responses(
        on_filter_values, off_filter_values)
    band_responses = np.asarray(band_responses, dtype=float)
    if band_responses.shape != (7, WAVE.size):
        raise RuntimeError("established seven-band response shape changed")
    if tuple(cache_manifest.get("band_order", ())) != \
            tuple(cube_model.frozen_parent.BANDS):
        raise RuntimeError("band cache does not contain the established seven-band order")
    effective = np.asarray(
        cube_model.frozen_parent._band_effective_wavelengths(band_responses),
        dtype=float)
    if not np.allclose(effective, _effective_wavelengths(band_responses),
                       rtol=0.0, atol=1.0e-8):
        raise RuntimeError("current and diagnostic band effective wavelengths disagree")

    coverage_rows, preflight = cube_model._preflight_model(
        h5files, identity_map, response, alpha_direct, alpha_history, amp_means)
    expected_preflight = {
        "groups_total": 15768,
        "groups_with_frozen_M": 13002,
        "groups_without_frozen_M_masked": 2766,
        "groups_missing_alpha_given_M": 0,
        "groups_missing_mean_given_M": 0,
    }
    if not args.allow_noncanonical_counts:
        for key, expected in expected_preflight.items():
            if int(preflight.get(key, -1)) != expected:
                raise RuntimeError("current-model preflight %s=%s, expected %s" %
                                   (key, preflight.get(key), expected))
    _write_csv(output_dir / "model_coverage.csv", coverage_rows)
    print("[exposure-scale] preflight: %s" % preflight, flush=True)
    print("[exposure-scale] exposure_scale_method = amplifier-collapsed surface brightness", flush=True)
    print("[exposure-scale] exposure_scale_gauge = median raw g_SB_joint -> 1", flush=True)
    print("[exposure-scale] exposure_scale_applied = True, diagnostic only", flush=True)
    print("[exposure-scale] post_scale_QA = True", flush=True)
    print("[exposure-scale] full_cube_built = False", flush=True)
    print("[exposure-scale] LSF_run = False", flush=True)
    print("[exposure-scale] amplifier_rejection_applied = False", flush=True)
    print("[exposure-scale] FIBER_AREA=%.12g (pi*0.75**2)" % FIBER_AREA, flush=True)

    shots = []
    records = []
    calibration_summaries = []
    for h5_index, h5file in enumerate(h5files, 1):
        print("[exposure-scale] calibrating H5 %d/%d: %s" %
              (h5_index, len(h5files), h5file.name), flush=True)
        calibrated = cube_model._calibrate_h5_current(
            h5file, identity_map, response, alpha_direct, alpha_history,
            amp_means, fq_template, historical_f0, blank_masks[h5file.name],
            args.min_sky_fibers)
        (spectra, errors, ra, dec, labels, surveys, calibration_summary,
         sky_records, sky_by_exposure, groups, q_rows, hardware_bad,
         corrected_total) = calibrated
        record = {
            "h5file": h5file,
            "spectra": np.asarray(spectra, dtype=np.float32),
            "original_spectra": np.asarray(spectra, dtype=np.float64).copy(),
            "errors": np.asarray(errors, dtype=np.float32),
            "ra": np.asarray(ra, dtype=float),
            "dec": np.asarray(dec, dtype=float),
            "labels": np.asarray(labels, dtype=int),
            "surveys": surveys,
            "sky_records": sky_records,
            "sky_by_exposure": {int(e): np.asarray(s, dtype=np.float32)
                                 for e, s in sky_by_exposure.items()},
            "groups": groups,
            "q_rows": np.asarray(q_rows, dtype=int),
            "hardware_bad": np.asarray(hardware_bad, dtype=bool),
            "corrected_total": np.asarray(corrected_total, dtype=np.float32),
            "blank_mask": np.asarray(blank_masks[h5file.name], dtype=bool),
        }
        records.append(record)
        calibration_summaries.append(calibration_summary)
        collapsed, _ = cube_model.collapse_many(record["spectra"], band_responses)
        amp_keys_full = _amp_keys_for_rows(groups, len(labels))
        for exposure in EXPOSURES:
            selected = np.flatnonzero(record["labels"] == exposure)
            shots.append(Shot(
                h5=h5file.name, exposure=exposure,
                survey=surveys[exposure], ra=record["ra"][selected],
                dec=record["dec"][selected],
                values={"ON": np.asarray(collapsed[selected, 5], dtype=np.float32),
                        "OFF": np.asarray(collapsed[selected, 6], dtype=np.float32)},
                amp_keys=np.asarray(amp_keys_full[selected], dtype=object),
                calibration_summary=calibration_summary))
        del calibrated, collapsed

    if len(shots) != N_EXPECTED_SHOTS and not args.allow_noncanonical_counts:
        raise RuntimeError("expected exactly 57 exposures, found %d" % len(shots))
    for shot in shots:
        shot.positions["ON"] = _adr_positions(
            shot.ra, shot.dec, shot.survey, band_responses[5])
        shot.positions["OFF"] = _adr_positions(
            shot.ra, shot.dec, shot.survey, band_responses[6])

    geometry = _derive_geometry(shots, args.pixel_scale, args.image_center_size)
    _ra0, _dec0, size_arcsec, n_image, xg, yg, _xgrid, _ygrid, tp = geometry
    print("[exposure-scale] collapsed shots=%d; image=%dx%d at %.3g arcsec/pixel" %
          (len(shots), n_image, n_image, args.pixel_scale), flush=True)

    with tempfile.TemporaryDirectory(prefix="m101-exposure-scale-") as cache_dir:
        raw_stacks = {
            band: _build_image_cache(
                shots, band, tp, xg, yg, args.pixel_scale,
                Path(cache_dir) / (band.lower() + "_raw_shot_images.float32"))
            for band in SB_BANDS}
        _process_sb_predictions(shots, raw_stacks, tp)
        exposure_rows = []
        for index, shot in enumerate(shots, 1):
            exposure_rows.append(_sb_exposure_row(
                shot, _fit_sb(shot, args.bootstrap_draws,
                              BOOTSTRAP_SEED + index), index))
        g_values = _finite([row["g_SB_joint_raw"] for row in exposure_rows])
        if not g_values.size:
            raise RuntimeError("no finite raw g_SB_joint values")
        g_center = float(np.median(g_values))
        for row in exposure_rows:
            raw = row["g_SB_joint_raw"]
            row["g_SB_center"] = g_center
            row["g_SB_relative"] = (raw / g_center
                                     if np.isfinite(raw) and g_center != 0 else np.nan)
            row["exposure_correction"] = (
                1.0 / row["g_SB_relative"]
                if np.isfinite(row["g_SB_relative"]) and row["g_SB_relative"] != 0
                else np.nan)
            row["fractional_sigma_g_SB"] = (
                row["sigma_g_SB"] / abs(raw)
                if np.isfinite(row["sigma_g_SB"]) and np.isfinite(raw) and raw != 0
                else np.nan)
            # Explicit iteration-1 provenance; the legacy names above remain
            # unchanged for the first-pass products.
            row["g1_raw"] = raw
            row["sigma_g1"] = row["sigma_g_SB"]
            row["center1"] = row["g_SB_center"]
            row["g1_rel"] = row["g_SB_relative"]
            row["original_SB_source_leverage"] = row.get(
                "SB_source_leverage", np.nan)

        pre_qa_rows, _pre_queue, pre_qa_summary = cube_model._evaluate_amplifier_qa(
            records, band_responses[-2:], effective, tp, xg, yg,
            args.pixel_scale, args.qa_min_blank,
            str(Path(cache_dir) / "pre_sb_amplifier_qa"))

        scale_by_key = {(row["H5"], int(row["exposure"])): row
                        for row in exposure_rows}
        _apply_exposure_relative_scales(
            records, scale_by_key, "g_SB_relative")
        _update_shot_values_from_records(shots, records, band_responses)

        corrected_stacks = {
            band: _build_image_cache(
                shots, band, tp, xg, yg, args.pixel_scale,
                Path(cache_dir) / (band.lower() + "_corrected_shot_images.float32"))
            for band in SB_BANDS}
        _process_sb_predictions(shots, corrected_stacks, tp)
        for index, shot in enumerate(shots):
            residual_fit = _fit_sb(
                shot, args.bootstrap_draws, BOOTSTRAP_SEED + 5000 + index)
            row = exposure_rows[index]
            row["g_SB_residual_ON"] = residual_fit["g_SB_ON"]
            row["g_SB_residual_OFF"] = residual_fit["g_SB_OFF"]
            row["g_SB_residual_joint"] = residual_fit["g_SB_joint"]
            row["sigma_g_SB_residual"] = residual_fit["sigma_g_SB"]
            row["residual_minus_one"] = row["g_SB_residual_joint"] - 1.0
            # The current post-first-pass closure is iteration 2's
            # measurement.  It is not itself a third correction.
            row["g2_ON"] = residual_fit["g_SB_ON"]
            row["g2_OFF"] = residual_fit["g_SB_OFF"]
            row["g2_raw"] = residual_fit["g_SB_joint"]
            row["sigma_g2"] = residual_fit["sigma_g_SB"]
            row["iteration1_SB_source_leverage"] = residual_fit.get(
                "SB_source_leverage", np.nan)

        # Preserve the authoritative first-pass post-SB QA before applying
        # iteration 2.  Its rows and figures are retained as provenance.
        iteration1_qa_rows, iteration1_queue_rows, iteration1_qa_summary = \
            cube_model._evaluate_amplifier_qa(
                records, band_responses[-2:], effective, tp, xg, yg,
                args.pixel_scale, args.qa_min_blank,
                str(output_dir / "post_sb"))

        g2_values = _finite([row["g2_raw"] for row in exposure_rows])
        if not g2_values.size:
            raise RuntimeError("no finite iteration-2 raw g_SB values")
        center2 = float(np.median(g2_values))
        if not np.isfinite(center2) or center2 == 0.0:
            raise RuntimeError("invalid iteration-2 SB gauge center")
        for row in exposure_rows:
            row["center2"] = center2
            row["g2_rel"] = (row["g2_raw"] / center2
                             if np.isfinite(row["g2_raw"]) else np.nan)
            row["cumulative_relative_scale"] = (
                row["g1_rel"] * row["g2_rel"]
                if np.isfinite(row["g1_rel"]) and np.isfinite(row["g2_rel"])
                else np.nan)
            row["cumulative_flux_correction"] = (
                1.0 / row["cumulative_relative_scale"]
                if np.isfinite(row["cumulative_relative_scale"]) and
                row["cumulative_relative_scale"] != 0.0 else np.nan)

        # Exactly one second relative correction.  No g3 correction is
        # applied, even if the rebuilt closure contains residual structure.
        _apply_exposure_relative_scales(records, scale_by_key, "g2_rel")
        _update_shot_values_from_records(shots, records, band_responses)
        iteration2_stacks = {
            band: _build_image_cache(
                shots, band, tp, xg, yg, args.pixel_scale,
                Path(cache_dir) / (band.lower() + "_iteration2_shot_images.float32"))
            for band in SB_BANDS}
        _process_sb_predictions(shots, iteration2_stacks, tp)
        for index, shot in enumerate(shots):
            closure_fit = _fit_sb(
                shot, args.bootstrap_draws, BOOTSTRAP_SEED + 10000 + index)
            row = exposure_rows[index]
            row["g3_ON"] = closure_fit["g_SB_ON"]
            row["g3_OFF"] = closure_fit["g_SB_OFF"]
            row["g3_raw"] = closure_fit["g_SB_joint"]
            row["sigma_g3"] = closure_fit["sigma_g_SB"]
            row["g3_minus_one"] = row["g3_raw"] - 1.0
            row["g3_z"] = (
                row["g3_minus_one"] / row["sigma_g3"]
                if np.isfinite(row["g3_minus_one"]) and
                np.isfinite(row["sigma_g3"]) and row["sigma_g3"] > 0.0
                else np.nan)
            row["iteration2_SB_source_leverage"] = closure_fit.get(
                "SB_source_leverage", np.nan)

        for row in exposure_rows:
            if not np.isfinite(row.get("g3_minus_one", np.nan)):
                row["g3_minus_one"] = row.get("g3_raw", np.nan) - 1.0

        post_qa_rows, post_queue_rows, post_qa_summary = \
            cube_model._evaluate_amplifier_qa(
                records, band_responses[-2:], effective, tp, xg, yg,
                args.pixel_scale, args.qa_min_blank,
                str(output_dir / "post_iter2"))

    scale_by_key = {(row["H5"], int(row["exposure"])): row
                    for row in exposure_rows}
    for row in iteration1_qa_rows:
        _attach_exposure_scale(row, scale_by_key)
    for row in iteration1_queue_rows:
        _attach_exposure_scale(row, scale_by_key)
    for row in post_qa_rows:
        _attach_exposure_scale(row, scale_by_key)
    for row in post_queue_rows:
        _attach_exposure_scale(row, scale_by_key)

    _add_cq_coordinates(post_qa_rows)
    cache_amp_coordinates, cache_coordinate_provenance = \
        _load_band_cache_amp_coordinates(cache_path, cache_manifest, h5files)
    focal_join = _add_focal_coordinates(post_qa_rows, cache_amp_coordinates)
    print("[exposure-scale] focal-coordinate join: post_iter2 QA rows=%d; "
          "finite c=%d; finite q=%d; finite c/q=%d; matched x/y=%d; "
          "unique physical amps=%d; unique physical IFUs=%d; "
          "X=[%.7g, %.7g] arcmin; Y=[%.7g, %.7g] arcmin" % (
              focal_join["post_iter2_qa_rows"],
              focal_join["finite_c_ea_rows"], focal_join["finite_q_ea_rows"],
              focal_join["finite_cq_rows"],
              focal_join["successfully_matched_cached_xy_rows"],
              focal_join["unique_physical_amplifiers"],
              focal_join["unique_physical_ifus"],
              focal_join["X_amp_min"], focal_join["X_amp_max"],
              focal_join["Y_amp_min"], focal_join["Y_amp_max"]), flush=True)
    focal_summary, focal_fit_join = _run_focal_plane_diagnostic(post_qa_rows)
    if (not args.allow_noncanonical_counts and
            len(focal_summary) != N_EXPECTED_SHOTS):
        raise RuntimeError("focal-plane summary must contain 57 H5 exposures, found %d" %
                           len(focal_summary))
    for key in ("post_iter2_qa_rows", "finite_c_ea_rows", "finite_q_ea_rows",
                "finite_cq_rows", "unique_physical_amplifiers",
                "unique_physical_ifus", "successfully_matched_cached_xy_rows",
                "X_amp_min", "X_amp_max", "Y_amp_min", "Y_amp_max"):
        if focal_fit_join[key] != focal_join[key]:
            raise RuntimeError(
                "focal-plane join changed between validation and plane fitting: %s" %
                key)
    print("[exposure-scale] focal-plane diagnostic: %d exposure summaries; "
          "median_c_ea material threshold=%.6g; materially nonzero=%d" % (
              len(focal_summary), FOCAL_MEDIAN_C_MATERIAL_THRESHOLD,
              sum(row["median_c_ea_materially_nonzero"]
                  for row in focal_summary)), flush=True)
    focal_aggregate = _focal_summary_aggregate(focal_summary)
    lowrank = _run_ifu_lowrank_diagnostic(post_qa_rows)
    print("[exposure-scale] IFU rank-1 diagnostic: %d exposures x %d IFUs; "
          "in-sample robust scatter improvement=%s; LOEO median improvement=%s; "
          "held-out-IFU median improvement=%s" % (
              lowrank["aggregate"]["N_exposures"],
              lowrank["aggregate"]["N_physical_IFUs"],
              _format_number(lowrank["aggregate"][
                  "rank1_fractional_improvement"]),
              _format_number(lowrank["aggregate"][
                  "LOEO_median_fractional_improvement"]),
              _format_number(lowrank["aggregate"][
                  "heldout_IFU_median_fractional_improvement"])), flush=True)
    # post_queue_rows contains the same row dictionaries in ranked order.
    repeatability_rows = _repeatability_rows(post_qa_rows)
    repeatability_summary = _repeatability_summary(repeatability_rows)
    validation = _validate_cumulative_scales(records, scale_by_key)
    validation["iteration1_gauge_median"] = float(np.median(
        _finite([row["g1_rel"] for row in exposure_rows])))
    validation["iteration2_gauge_median"] = float(np.median(
        _finite([row["g2_rel"] for row in exposure_rows])))
    if not np.isclose(validation["iteration1_gauge_median"], 1.0,
                      rtol=0.0, atol=1.0e-12):
        raise RuntimeError("iteration-1 relative-scale gauge is not median one")
    if not np.isclose(validation["iteration2_gauge_median"], 1.0,
                      rtol=0.0, atol=1.0e-12):
        raise RuntimeError("iteration-2 relative-scale gauge is not median one")

    exposure_fields = [
        "H5", "exposure", "chronological_exposure_index",
        "g_SB_ON", "g_SB_OFF", "g_SB_joint", "g_SB_joint_raw",
        "g_SB_center", "g_SB_relative", "exposure_correction",
        "sigma_g_SB", "fractional_sigma_g_SB", "g_SB_p16", "g_SB_p50",
        "g_SB_p84", "N_SB_ON", "N_SB_OFF", "N_SB_BOTH",
        "max_abs_predicted_SB_ON", "max_abs_predicted_SB_OFF",
        "sqrt_sum_predicted_squared_SB_ON",
        "sqrt_sum_predicted_squared_SB_OFF",
        "sqrt_sum_predicted_squared_SB_joint", "residual_scatter_SB_ON",
        "residual_scatter_SB_OFF", "residual_scatter_SB_joint",
        "abs_g_SB_ON_minus_g_SB_OFF", "SB_source_leverage",
        "SB_max_predictor_leverage_fraction", "SB_N_effective_leverage",
        "SB_bootstrap_reason", "g_SB_residual_ON", "g_SB_residual_OFF",
        "g_SB_residual_joint", "sigma_g_SB_residual", "residual_minus_one",
        "g1_raw", "sigma_g1", "center1", "g1_rel",
        "g2_ON", "g2_OFF", "g2_raw", "sigma_g2", "center2", "g2_rel",
        "cumulative_relative_scale", "cumulative_flux_correction",
        "g3_ON", "g3_OFF", "g3_raw", "sigma_g3", "g3_minus_one", "g3_z",
        "original_SB_source_leverage", "iteration1_SB_source_leverage",
        "iteration2_SB_source_leverage",
    ]
    _write_csv(output_dir / "exposure_scale_diagnostic.csv",
               exposure_rows, exposure_fields)
    closure_fields = [
        "H5", "exposure", "g_SB_joint_raw", "sigma_g_SB", "g_SB_center",
        "g_SB_relative", "exposure_correction", "g_SB_residual_ON",
        "g_SB_residual_OFF", "g_SB_residual_joint", "sigma_g_SB_residual",
        "residual_minus_one",
    ]
    _write_csv(output_dir / "exposure_scale_closure.csv",
               exposure_rows, closure_fields)
    _write_csv(output_dir / "post_sb_amplifier_qa.csv", iteration1_qa_rows)
    _write_csv(output_dir / "post_sb_amplifier_qa_queue.csv", iteration1_queue_rows)
    _write_csv(output_dir / "post_iter2_amplifier_qa.csv", post_qa_rows)
    _write_csv(output_dir / "post_iter2_amplifier_qa_queue.csv", post_queue_rows)
    iteration2_closure_fields = [
        "H5", "exposure", "chronological_exposure_index",
        "g1_raw", "sigma_g1", "center1", "g1_rel",
        "g2_raw", "sigma_g2", "center2", "g2_rel",
        "cumulative_relative_scale", "cumulative_flux_correction",
        "g3_ON", "g3_OFF", "g3_raw", "sigma_g3", "g3_minus_one", "g3_z",
        "original_SB_source_leverage", "iteration1_SB_source_leverage",
        "iteration2_SB_source_leverage",
    ]
    if len(exposure_rows) != N_EXPECTED_SHOTS and not args.allow_noncanonical_counts:
        raise RuntimeError("iteration-2 closure product must contain 57 rows")
    _write_csv(output_dir / "exposure_scale_iteration2_closure.csv",
               exposure_rows, iteration2_closure_fields)
    _write_csv(output_dir / "post_iter2_cq_repeatability.csv",
               repeatability_rows)
    _write_csv(output_dir / "post_iter2_cq_repeatability_summary.csv",
               repeatability_summary)
    _write_csv(output_dir / "post_iter2_focal_plane_rows.csv", post_qa_rows)
    _write_csv(output_dir / "post_iter2_focal_plane_summary.csv", focal_summary)
    lowrank_data = lowrank["data"]
    lowrank_matrix_rows, lowrank_matrix_fields = _lowrank_matrix_wide_rows(
        lowrank_data)
    _write_csv(output_dir / "post_iter2_ifu_pattern_rows.csv",
               lowrank_data["cell_rows"])
    _write_csv(output_dir / "post_iter2_ifu_template.csv",
               _lowrank_template_records(
                   lowrank_data, lowrank["rank1_fit"], lowrank["blank_fit"]))
    _write_csv(output_dir / "post_iter2_ifu_C_matrix.csv",
               lowrank_matrix_rows, lowrank_matrix_fields)
    _write_csv(output_dir / "post_iter2_ifu_lowrank_summary.csv",
               lowrank["summary"])
    _write_csv(output_dir / "post_iter2_ifu_pairwise.csv",
               lowrank["pair_rows"])
    _write_csv(output_dir / "post_iter2_ifu_rank1_loo.csv",
               lowrank["loo_rows"])
    _write_csv(output_dir / "post_iter2_ifu_rank1_heldout_cells.csv",
               lowrank["heldout_cell_rows"])
    _write_csv(output_dir / "post_iter2_ifu_blank_transfer.csv",
               lowrank["blank_transfer"])
    _write_csv(output_dir / "post_iter2_ifu_blank_template_comparison.csv",
               [lowrank["blank_template_comparison"]])
    _write_csv(output_dir / "post_iter2_ifu_amp_residual_rows.csv",
               post_qa_rows)
    _write_csv(output_dir / "post_iter2_ifu_amp_residual_repeatability.csv",
               lowrank["d_repeatability"])
    _write_cq_ranked_csv(
        output_dir / "post_iter2_strongest_gray.csv",
        _ranked_cq_rows(post_qa_rows, "c_gray"))
    _write_cq_ranked_csv(
        output_dir / "post_iter2_strongest_differential.csv",
        _ranked_cq_rows(post_qa_rows, "q_color"))

    pre_stats = _qa_grouped_stats(pre_qa_rows)
    post_stats = _qa_grouped_stats(iteration1_qa_rows)
    focal_color_limits = _focal_global_color_limits(post_qa_rows)
    focal_map_paths = _plot_focal_plane_maps(
        post_qa_rows, focal_summary, output_dir, focal_color_limits)
    focal_blank_comparison_paths = _plot_focal_blank_comparisons(
        post_qa_rows, focal_summary, output_dir, focal_color_limits)
    if (not args.allow_noncanonical_counts and
            len(focal_map_paths) != N_EXPECTED_SHOTS):
        raise RuntimeError("expected 57 focal-plane maps, found %d" %
                           len(focal_map_paths))
    focal_gradient_path = output_dir / "focal_plane_gradient_summary.png"
    _plot_focal_gradient_summary(focal_summary, focal_gradient_path)
    focal_performance_path = output_dir / "focal_plane_plane_performance.png"
    _plot_focal_plane_performance(focal_summary, focal_performance_path)
    lowrank_plot_outputs = _plot_lowrank_outputs(lowrank, output_dir)
    focal_outputs = {
        "rows_table": str((output_dir / "post_iter2_focal_plane_rows.csv").resolve()),
        "summary_table": str((output_dir / "post_iter2_focal_plane_summary.csv").resolve()),
        "map_count": int(len(focal_map_paths)),
        "map_paths": focal_map_paths,
        "blank_comparison_count": int(len(focal_blank_comparison_paths)),
        "blank_comparison_paths": focal_blank_comparison_paths,
        "gradient_summary_plot": str(focal_gradient_path.resolve()),
        "plane_performance_plot": str(focal_performance_path.resolve()),
    }
    provenance = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "script": str(Path(__file__).resolve()),
        "input_h5_files": [str(path) for path in h5files],
        "current_model": {
            "module": str(Path(cube_model.__file__).resolve()),
            "calibration_helper": "_calibrate_h5_current",
            "equation": "D=spectrum/Survey.offset+skyspectrum; M=exp(cumulative_final.R_total_h_i_a); T=(D-A_q-mu_a)/M; O=T-S_e",
            "pca_coefficients_applied": False,
            "preflight": preflight,
            "expected_preflight": expected_preflight,
        },
        "exposure_scale": {
            "method": "physical-amplifier-collapsed surface brightness",
            "fit_row": "physical amplifier; robust median observed and FIBER_AREA-corrected predicted fibers",
            "bands": ["ON", "OFF"],
            "fiber_area": FIBER_AREA,
            "gauge": "median finite raw g_SB_joint",
            "g_SB_center": g_center,
            "SB_iterations_applied": 2,
            "third_iteration_applied": False,
            "center1": float(np.median(_finite([row["center1"]
                                                  for row in exposure_rows]))),
            "center2": float(np.median(_finite([row["center2"]
                                                  for row in exposure_rows]))),
            "correction": "O1=O0/g1_rel; O2=O1/g2_rel; errors, corrected_total, and final_sky scaled consistently",
            "sigma_is_correlated_systematic": True,
            "sigma_added_to_fiber_errors": False,
        },
        "qa": {
            "helper": "make_m101_cube_current_model._evaluate_amplifier_qa",
            "qa_min_blank": args.qa_min_blank,
            "report_only": True,
            "rejection_applied": False,
            "pre_summary": pre_qa_summary,
            "iteration1_post_sb_summary": iteration1_qa_summary,
            "iteration2_post_sb_summary": post_qa_summary,
            "repeatability_summary": repeatability_summary,
        },
        "band_provenance": {
            "band_order": list(cube_model.frozen_parent.BANDS),
            "null_provenance": _json_ready(null_provenance),
            "effective_wavelength_A": _json_ready(effective),
            "qa_bands": list(cube_model.QA_BANDS),
            "adr_geometry": "current final IMAGE_QA band-effective ADR geometry",
        },
        "model_products": {
            "state": _file_identity(state_path),
            "band_cache": _file_identity(cache_path),
            "band_cache_manifest": _file_identity(cache_manifest_path),
            "alpha_fits": _file_identity(alpha_fits_path),
            "alpha_history": _file_identity(alpha_summary_path),
            "amp_means": _file_identity(amp_means_path),
            "fq_template": _file_identity(fq_path),
            "external_blank_fibers": _file_identity(blank_path),
            "on_filter": _file_identity(on_filter),
            "off_filter": _file_identity(off_filter),
        },
        "control": {
            "SB_iterations_applied": 2,
            "third_iteration_applied": False,
            "c_q_used_for_calibration": False,
            "science_cube_built": False,
            "LSF_run": False,
            "amplifier_rejection_applied": False,
            "exposure_scale_applied": True,
            "diagnostic_only": True,
        },
        "SB_iterations_applied": 2,
        "third_iteration_applied": False,
        "c_q_used_for_calibration": False,
        "amplifier_rejection_applied": False,
        "science_cube_built": False,
        "LSF_run": False,
        "validation": validation,
        "calibration_summaries": calibration_summaries,
        "blank_loader": _json_ready(blank_provenance),
        "alpha_mean_scope": mean_scope,
        "finite_mean_wavelengths": int(np.sum(mean_science_mask)),
        "response_composition": composition,
        "focal_plane": {
            **cache_coordinate_provenance,
            **focal_join,
            "status": "historical_diagnostic_only",
            "c_plane_model_retired": True,
            "c_plane_applied": False,
            "lowrank_analysis_excludes_c_plane_fields": True,
            "plane_coordinate_fields": ["X_amp", "Y_amp"],
            "plane_coordinate_label": (
                "historical_M101_x_arcmin/historical_M101_y_arcmin"),
            "plane_equation": "value=bx*(X-X0)+by*(Y-Y0)",
            "intercept": False,
            "gradient_angle_units": "degrees",
            "gradient_orientation_comparison_caveat": (
                "historical M101 coordinates are exposure-centered; absolute "
                "orientation comparisons between exposures require known common "
                "frame rotation"),
            "color_limits": focal_color_limits,
            "summary_aggregate": focal_aggregate,
            "median_c_ea_material_threshold":
                FOCAL_MEDIAN_C_MATERIAL_THRESHOLD,
            "median_c_ea_materially_nonzero_exposures": int(sum(
                row["median_c_ea_materially_nonzero"]
                for row in focal_summary)),
            "no_plane_correction_applied": True,
            "no_spectra_modified": True,
            "outputs": focal_outputs,
        },
        "ifu_lowrank_focal_pattern": {
            "status": "pattern_identification_only",
            "matrix_shape": [
                lowrank["aggregate"]["N_exposures"],
                lowrank["aggregate"]["N_physical_IFUs"]],
            "physical_ifu_identity": (
                "SPECID/IFUSLOT/IFUID/IFU_CODE"),
            "C_definition": (
                "robust median c_ea across finite amplifiers; minimum two "
                "finite amplifiers"),
            "C_exposure_median_preserved": True,
            "C_pattern_definition": "C_ei - robust_median_i(C_ei)",
            "coordinates_for_fit": False,
            "coordinates_for_plotting": cache_coordinate_provenance[
                "coordinate_source"],
            "rank_maximum_fitted": 1,
            "rank1_gauge": "median_i(T_i)=0; robust_rms(T_i)=1",
            "pairwise_test_before_rank1": True,
            "heldout_IFU_protocol": (
                "A_e fixed from training IFUs; held-out IFU coefficient fit "
                "from other exposures, excluding each test exposure"),
            "blank_transfer_no_spatial_refit": True,
            "blank_support_min_distinct_IFUs": LOWRANK_MIN_BLANK_IFUS,
            "c_plane_model_retired": True,
            "c_plane_applied": False,
            "uses_c_bx": False,
            "uses_c_by": False,
            "uses_c_plane_prediction": False,
            "uses_c_plane_residual": False,
            "no_spectra_modified": True,
            "aggregate": lowrank["aggregate"],
            "outputs": {
            "pattern_rows": str((output_dir /
                                      "post_iter2_ifu_pattern_rows.csv").resolve()),
                "template": str((output_dir /
                                  "post_iter2_ifu_template.csv").resolve()),
                "C_matrix": str((output_dir /
                                  "post_iter2_ifu_C_matrix.csv").resolve()),
                "summary": str((output_dir /
                                 "post_iter2_ifu_lowrank_summary.csv").resolve()),
                "pairwise": str((output_dir /
                                  "post_iter2_ifu_pairwise.csv").resolve()),
                "leave_one_exposure_out": str((output_dir /
                    "post_iter2_ifu_rank1_loo.csv").resolve()),
                "heldout_IFU_cells": str((output_dir /
                    "post_iter2_ifu_rank1_heldout_cells.csv").resolve()),
                "blank_transfer": str((output_dir /
                    "post_iter2_ifu_blank_transfer.csv").resolve()),
                "blank_template_comparison": str((output_dir /
                    "post_iter2_ifu_blank_template_comparison.csv").resolve()),
                "amplifier_residual_rows": str((output_dir /
                    "post_iter2_ifu_amp_residual_rows.csv").resolve()),
                "amplifier_residual_repeatability": str((output_dir /
                    "post_iter2_ifu_amp_residual_repeatability.csv").resolve()),
                "plots": lowrank_plot_outputs,
            },
        },
    }
    (output_dir / "diagnostic_provenance.json").write_text(
        json.dumps(_json_ready(provenance), indent=2, sort_keys=True))
    _plot_closure(exposure_rows, pre_stats, post_stats,
                  output_dir / "exposure_scale_closure.png")
    _plot_post_fractional(iteration1_qa_rows,
                          output_dir / "post_sb_amplifier_qa_outliers.png")
    _write_summary(output_dir / "exposure_scale_summary.txt", exposure_rows,
                   pre_qa_rows, iteration1_qa_rows, pre_qa_summary,
                   iteration1_qa_summary,
                   preflight, provenance)
    _plot_iteration2_closure(
        exposure_rows, output_dir / "exposure_scale_iteration2_closure.png")
    _plot_cq(post_qa_rows, output_dir / "post_iter2_cq.png")
    _plot_cq(post_qa_rows, output_dir / "post_iter2_cq_full_range.png",
             full_range=True)
    _plot_cq_repeatability(
        repeatability_rows, output_dir / "post_iter2_cq_repeatability.png")
    _write_iteration2_summary(
        output_dir / "iteration2_summary.txt", exposure_rows,
        iteration1_qa_rows, post_qa_rows, repeatability_rows,
        repeatability_summary, preflight, validation,
        focal_summary, focal_join, focal_aggregate, lowrank)
    print("[exposure-scale] concise final summary:", flush=True)
    print("  closure robust scatter: g2-1=%.7g g3-1=%.7g" % (
        _robust_scale(np.asarray([row["g2_raw"] for row in exposure_rows]) - 1.0),
        _robust_scale(np.asarray([row["g3_raw"] for row in exposure_rows]) - 1.0)),
          flush=True)
    final_on = _qa_field_stats(post_qa_rows, "image_fractional_ON")
    final_off = _qa_field_stats(post_qa_rows, "image_fractional_OFF")
    final_c = _qa_field_stats(post_qa_rows, "image_c_gray")
    final_q = _qa_field_stats(post_qa_rows, "image_q_color")
    print("  final IMAGE_QA robust scatter: ON=%.7g OFF=%.7g" % (
        final_on["robust_scatter"], final_off["robust_scatter"]), flush=True)
    print("  c_ea median/scatter: %.7g %.7g; q_ea median/scatter: %.7g %.7g" % (
        final_c["median"], final_c["robust_scatter"],
        final_q["median"], final_q["robust_scatter"]), flush=True)
    for repeat in repeatability_summary:
        print("  repeatability %s %s: N=%d Pearson=%s diff_scatter=%s same_sign=%s" % (
            repeat["component"], repeat["exposure_pair"],
            repeat["N_matched_amplifiers"],
            _format_number(repeat["Pearson_correlation"]),
            _format_number(repeat["robust_scatter_difference"]),
            _format_number(repeat["fraction_same_sign"])), flush=True)
    for component in ("c_gray", "q_color"):
        strongest = _ranked_cq_rows(post_qa_rows, component)[:3]
        print("  strongest |%s|:" % component, flush=True)
        for row in strongest:
            print("    %s/e%d %s/%s/%s/%s=%s" % (
                row["H5"], row["exposure"], row["SPECID"], row["IFUSLOT"],
                row["IFUID"], row["AMP"],
                _format_number(row["image_%s" % component])), flush=True)
    print("[exposure-scale] wrote diagnostic products to %s" % output_dir,
          flush=True)


if __name__ == "__main__":
    main()
