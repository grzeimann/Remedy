#!/usr/bin/env python3
"""Compare the current QR-sky path with a fresh external-blank sky reset.

This is a diagnostic-only script.  It reads VIRUS spectra and the persisted
external blank-fiber classification, applies the existing Bayesian
calibration algebra, and makes lightweight fiber-level diagnostics.  It does
not fit, reconstruct, subtract from, or modify any production cube or H5.
"""

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tables
from astropy.io import fits
from scipy.ndimage import gaussian_filter1d
from scipy.stats import spearmanr

import diagnose_m101_hierarchical as validated_m101
import make_mosaic_cube_from_fit as cube_builder
from math_utils import biweight


M101_RA_DEG = 210.800
M101_DEC_DEG = 54.333
DEFAULT_WAVELENGTHS = (3636.0, 3700.0, 4000.0, 4500.0, 5000.0)
DEFAULT_MINIMUM_FINITE_FRACTION = 0.8
DEFAULT_MINIMUM_SKY_FIBERS = 20


def expand_h5_glob(value):
    paths = sorted(Path(path).resolve() for path in __import__("glob").glob(value))
    if not paths:
        raise ValueError("no H5 files matched: %s" % value)
    if len({path.name for path in paths}) != len(paths):
        raise ValueError("input H5 basenames are ambiguous")
    return paths


def parse_wavelengths(value):
    try:
        requested = [float(part.strip()) for part in value.split(",")]
    except (AttributeError, ValueError) as exc:
        raise ValueError("diagnostic wavelengths must be comma-separated numbers") from exc
    if not requested or not np.all(np.isfinite(requested)):
        raise ValueError("diagnostic wavelengths must be finite")
    result = []
    for wavelength in requested:
        index = int(np.argmin(np.abs(validated_m101.DEF_WAVE - wavelength)))
        actual = float(validated_m101.DEF_WAVE[index])
        if abs(actual - wavelength) > 1e-8:
            raise ValueError("diagnostic wavelength %.8g is not on DEF_WAVE; nearest is %.8g" %
                             (wavelength, actual))
        result.append(index)
    return requested, result


def robust_location(values):
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.nan
    if finite.size < 3:
        return float(np.median(finite))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        result = biweight(finite)
    return float(result) if np.isfinite(result) else float(np.median(finite))


def robust_scatter(values):
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.nan
    median = np.median(finite)
    mad = np.median(np.abs(finite - median))
    if mad > 0.0:
        return float(1.4826 * mad)
    return float(np.std(finite))


def robust_spectrum(values):
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or values.shape[0] == 0:
        return np.full(values.shape[-1] if values.ndim else 0, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        result = np.asarray(biweight(values, axis=0), dtype=float)
        fallback = np.nanmedian(values, axis=0)
    result[~np.isfinite(result)] = fallback[~np.isfinite(result)]
    return result


def column_stat(values, percentile=None):
    values = np.asarray(values, dtype=float)
    result = np.full(values.shape[1], np.nan, dtype=float)
    for index in range(values.shape[1]):
        finite = values[:, index][np.isfinite(values[:, index])]
        if finite.size:
            result[index] = (np.median(finite) if percentile is None else
                             np.percentile(finite, percentile))
    return result


def finite_spectrum_count(values):
    return np.isfinite(values).sum(axis=0).astype(int)


def exposure_radius(ra, dec):
    dra = (ra - M101_RA_DEG) * np.cos(np.deg2rad(M101_DEC_DEG)) * 60.0
    ddec = (dec - M101_DEC_DEG) * 60.0
    return np.hypot(dra, ddec)


def calibrate_h5_stages(h5file, matches, fq_template):
    """Apply the cube-builder calibration algebra and retain both sky stages."""
    read_started = time.perf_counter()
    with tables.open_file(h5file, mode="r") as h5:
        info, fibers = h5.root.Info, h5.root.Fibers
        groups, labels = validated_m101.build_groups(info)
        if "spectrum" not in fibers.colnames or "skyspectrum" not in fibers.colnames:
            raise ValueError("%s Fibers lacks spectrum/skyspectrum" % h5file)
        source = np.asarray(fibers.cols.spectrum[:], dtype=float)
        qr_sky = np.asarray(fibers.cols.skyspectrum[:], dtype=float)
        ra = np.asarray(info.cols.ra[:], dtype=float)
        dec = np.asarray(info.cols.dec[:], dtype=float)
        if source.shape != qr_sky.shape or source.shape[0] != info.nrows:
            raise ValueError("%s spectrum/skyspectrum/Info shapes differ" % h5file)
        surveys = cube_builder._survey_by_exposure(h5)
        ifuslot = np.asarray(info.cols.ifuslot[:])
        amp = np.asarray([cube_builder._text(value) for value in info.cols.amp[:]])
        hardware_bad = validated_m101.masked_rows(h5file, ifuslot, amp)
    read_seconds = time.perf_counter() - read_started

    calibration_started = time.perf_counter()
    n_fibers, n_wave = source.shape
    raw_working = np.full((n_fibers, n_wave), np.nan, dtype=float)
    stage1_m = np.full_like(raw_working, np.nan)
    stage2_m_p = np.full_like(raw_working, np.nan)
    stage3_m_p_alpha = np.full_like(raw_working, np.nan)
    stage4_full_current = np.full_like(raw_working, np.nan)
    p_component = np.full_like(raw_working, np.nan)
    alpha_component = np.full_like(raw_working, np.nan)
    sky_interaction_component = np.full_like(raw_working, np.nan)
    multiplier_by_fiber = np.full(n_fibers, np.nan, dtype=float)
    p_by_fiber = np.full(n_fibers, np.nan, dtype=float)
    alpha_by_fiber = np.full(n_fibers, np.nan, dtype=float)
    for exposure in range(1, validated_m101.N_EXPOSURES + 1):
        survey = surveys[exposure]
        offset = float(survey["offset"])
        if not np.isfinite(offset) or offset == 0.0:
            raise ValueError("%s exposure %d has invalid Survey.offset" % (h5file, exposure))
        K_work = validated_m101.raw_work_basis(survey)
        working = source / offset
        for group in groups:
            if group["exposure"] != exposure:
                continue
            key = (str(h5file), exposure, group["specid"], group["ifuslot"],
                   group["ifuid"], group["amp"])
            fit = matches[key]
            indices = group["indices"]
            j = np.arange(validated_m101.N_FIBER_AMP)
            q = j if group["amp"] in ("LL", "RU") else validated_m101.N_FIBER_AMP - 1 - j
            additive = (float(fit["alpha_mean"]) * K_work[None, :] * fq_template[q, None])
            multiplier = float(np.exp(float(fit["posterior_z_mean"])))
            p_value = float(fit["p_mean"])
            raw_working[indices] = working[indices]
            multiplier_by_fiber[indices] = multiplier
            p_by_fiber[indices] = p_value
            alpha_by_fiber[indices] = float(fit["alpha_mean"])
            stage1_m[indices] = working[indices] / multiplier
            p_component[indices] = -p_value / multiplier
            alpha_component[indices] = -additive / multiplier
            sky_interaction_component[indices] = qr_sky[indices] * (1.0 / multiplier - 1.0)
            stage2_m_p[indices] = stage1_m[indices] + p_component[indices]
            stage3_m_p_alpha[indices] = stage2_m_p[indices] + alpha_component[indices]
            stage4_full_current[indices] = stage3_m_p_alpha[indices] + sky_interaction_component[indices]
    direct_current = stage4_full_current.copy()
    for exposure in range(1, validated_m101.N_EXPOSURES + 1):
        indices = np.flatnonzero(labels == exposure)
        survey = surveys[exposure]
        offset = float(survey["offset"])
        K_work = validated_m101.raw_work_basis(survey)
        working = source / offset
        for group in groups:
            if group["exposure"] != exposure:
                continue
            fit = matches[(str(h5file), exposure, group["specid"], group["ifuslot"],
                           group["ifuid"], group["amp"])]
            group_indices = group["indices"]
            j = np.arange(validated_m101.N_FIBER_AMP)
            q = j if group["amp"] in ("LL", "RU") else validated_m101.N_FIBER_AMP - 1 - j
            additive = (float(fit["alpha_mean"]) * K_work[None, :] * fq_template[q, None])
            multiplier = float(np.exp(float(fit["posterior_z_mean"])))
            direct_current[group_indices] = ((working[group_indices] - float(fit["p_mean"])
                                              - additive + qr_sky[group_indices]) / multiplier
                                              - qr_sky[group_indices])
    finite_identity = np.isfinite(stage4_full_current) & np.isfinite(direct_current)
    identity_delta = np.abs(stage4_full_current[finite_identity] - direct_current[finite_identity])
    identity_scale = np.maximum(np.abs(direct_current[finite_identity]), 1.0)
    identity_max_abs = float(np.max(identity_delta)) if identity_delta.size else 0.0
    identity_max_relative = float(np.max(identity_delta / identity_scale)) if identity_delta.size else 0.0
    identity_ok = identity_max_relative <= 5.0e-13
    if not identity_ok:
        raise ValueError("expanded calibration identity failed: max abs=%g max relative=%g" %
                         (identity_max_abs, identity_max_relative))
    calibrated_total = stage3_m_p_alpha + qr_sky * (1.0 / multiplier_by_fiber[:, None] - 1.0) + qr_sky
    current_pre = stage4_full_current
    stages = {
        "raw_working": raw_working, "after_m": stage1_m, "after_p": stage2_m_p,
        "after_alpha": stage3_m_p_alpha, "full_current": stage4_full_current,
    }
    components = {"p_component": p_component, "alpha_component": alpha_component,
                  "sky_interaction_component": sky_interaction_component}
    for array in list(stages.values()) + list(components.values()) + [calibrated_total, current_pre]:
        array[hardware_bad] = np.nan
    calibrated_total[~np.isfinite(current_pre)] = np.nan
    current_pre[~np.isfinite(calibrated_total)] = np.nan
    calibration_seconds = time.perf_counter() - calibration_started
    return {
        "h5": h5file.name, "path": str(h5file), "groups": groups,
        "labels": np.asarray(labels, dtype=int), "surveys": surveys,
        "source": source, "qr_sky": qr_sky, "calibrated_total": calibrated_total,
        "current_pre": current_pre, "ra": ra, "dec": dec, "stages": stages,
        "components": components, "multiplier": multiplier_by_fiber,
        "p_by_fiber": p_by_fiber, "alpha_by_fiber": alpha_by_fiber,
        "identity": {"ok": identity_ok, "finite_values": int(finite_identity.sum()),
                      "max_abs": identity_max_abs, "max_relative": identity_max_relative},
        "read_seconds": read_seconds, "calibration_seconds": calibration_seconds,
    }


def residual_stages(record, external_blank, minimum_finite_fraction, minimum_sky_fibers):
    stage_arrays = record["stages"]
    current_pre = record["current_pre"]
    total = record["calibrated_total"]
    labels = record["labels"]
    n_wave = current_pre.shape[1]
    selected = np.zeros(current_pre.shape[0], dtype=bool)
    exposure_rows = []
    current_residuals = {}
    fresh_skies = {}
    stage_selected = {name: np.zeros(current_pre.shape[0], dtype=bool)
                      for name in stage_arrays}
    stage_residuals = {name: {} for name in stage_arrays}
    for exposure in range(1, validated_m101.N_EXPOSURES + 1):
        indices = np.flatnonzero(labels == exposure)
        blank_candidates = external_blank[indices]
        sufficient_by_stage = {}
        selected_by_stage = {}
        for name, values in stage_arrays.items():
            sufficient = (np.isfinite(values[indices]).sum(axis=1) >=
                          int(np.ceil(minimum_finite_fraction * n_wave)))
            sufficient_by_stage[name] = sufficient
            selected_indices = indices[blank_candidates & sufficient]
            selected_by_stage[name] = selected_indices
            stage_selected[name][selected_indices] = True
            if selected_indices.size < minimum_sky_fibers:
                raise ValueError("%s exposure %d stage %s has only %d externally blank finite fibers" %
                                 (record["h5"], exposure, name, selected_indices.size))
            stage_residuals[name][exposure] = robust_spectrum(values[selected_indices])
        sufficient = sufficient_by_stage["full_current"]
        selected_indices = selected_by_stage["full_current"]
        selected[selected_indices] = True
        current_residual = robust_spectrum(current_pre[selected_indices])
        fresh_sky = robust_spectrum(total[selected_indices])
        current_residuals[exposure] = current_residual
        fresh_skies[exposure] = fresh_sky
        radius = exposure_radius(record["ra"][selected_indices], record["dec"][selected_indices])
        exposure_rows.append({
            "H5": record["h5"], "exposure": exposure,
            "N_blank": int(selected_indices.size),
            "N_external_blank_candidates": int(blank_candidates.sum()),
            "N_sufficient_finite": int(sufficient.sum()),
            "N_blank_inside_6arcmin": int(np.sum(radius <= 6.0)),
            "N_blank_outside_6arcmin": int(np.sum(radius > 6.0)),
            "stage_selected_counts": {name: int(values.size)
                                       for name, values in selected_by_stage.items()},
        })
    centered_stages = {name: np.full_like(values, np.nan)
                       for name, values in stage_arrays.items()}
    current_after = np.full_like(current_pre, np.nan)
    sky_reset = np.full_like(total, np.nan)
    for exposure in range(1, validated_m101.N_EXPOSURES + 1):
        indices = np.flatnonzero(labels == exposure)
        for name, values in stage_arrays.items():
            centered_stages[name][indices] = values[indices] - stage_residuals[name][exposure]
        current_after[indices] = current_pre[indices] - current_residuals[exposure]
        sky_reset[indices] = total[indices] - fresh_skies[exposure]
    component_residuals = {}
    centered_components = {}
    for name, values in record["components"].items():
        component_residuals[name] = {}
        centered_components[name] = np.full_like(values, np.nan)
        for exposure in range(1, validated_m101.N_EXPOSURES + 1):
            indices = np.flatnonzero(labels == exposure)
            # robust_spectrum handles wavelength-specific NaNs; requiring an
            # entirely finite row would discard every fiber with one bad plane.
            selected_indices = indices[selected[indices]]
            component_residuals[name][exposure] = robust_spectrum(values[selected_indices])
            centered_components[name][indices] = values[indices] - component_residuals[name][exposure]
    record.update({
        "external_blank": external_blank,
        "selected_blank": selected,
        "current_residuals": current_residuals,
        "fresh_skies": fresh_skies,
        "current_after": current_after,
        "sky_reset": sky_reset,
        "exposure_rows": exposure_rows,
        "stage_selected_blank": stage_selected,
        "stage_residuals": stage_residuals,
        "centered_stages": centered_stages,
        "component_residuals": component_residuals,
        "centered_components": centered_components,
    })
    return record


def group_level_rows(record, wavelength_indices):
    rows = []
    labels = record["labels"]
    blank = record["selected_blank"]
    stages = {
        "total": record["calibrated_total"],
        "current_pre": record["current_pre"],
        "current_after": record["current_after"],
        "reset": record["sky_reset"],
    }
    for wave_index in wavelength_indices:
        wavelength = float(validated_m101.DEF_WAVE[wave_index])
        for group in record["groups"]:
            indices = group["indices"]
            group_blank = indices[blank[indices]]
            row = {
                "H5": record["h5"], "exposure": group["exposure"],
                "wavelength": wavelength, "SPECID": group["specid"],
                "IFUSLOT": group["ifuslot"], "IFUID": group["ifuid"], "AMP": group["amp"],
                "N_all": int(np.isfinite(stages["total"][indices, wave_index]).sum()),
                "N_blank": int(np.isfinite(stages["total"][group_blank, wave_index]).sum()),
            }
            for name, values in stages.items():
                all_values = values[indices, wave_index]
                blank_values = values[group_blank, wave_index]
                row["%s_all_location" % name] = robust_location(all_values)
                row["%s_all_scatter" % name] = robust_scatter(all_values)
                row["%s_blank_location" % name] = robust_location(blank_values)
                row["%s_blank_scatter" % name] = robust_scatter(blank_values)
            rows.append(row)
    return rows


def nonuniformity_rows(group_rows):
    stages = ("total", "current_pre", "current_after", "reset")
    rows = []
    for key in sorted(set((row["H5"], row["exposure"], row["wavelength"])
                          for row in group_rows)):
        h5, exposure, wavelength = key
        subset = [row for row in group_rows if (row["H5"], row["exposure"], row["wavelength"]) == key]
        for population in ("all", "blank"):
            for stage in stages:
                source_stage = "reset" if stage == "total" else stage
                values = np.asarray([row["%s_%s_location" % (source_stage, population)]
                                     for row in subset], dtype=float)
                values = values[np.isfinite(values)]
                rows.append({
                    "H5": h5, "exposure": exposure, "wavelength": wavelength,
                    "population": population,
                    "stage": "calibrated_total_minus_fresh_sky" if stage == "total" else stage,
                    "N_groups": int(values.size),
                    "median": float(np.median(values)) if values.size else np.nan,
                    "p16": float(np.percentile(values, 16)) if values.size else np.nan,
                    "p84": float(np.percentile(values, 84)) if values.size else np.nan,
                    "robust_scatter": robust_scatter(values),
                    "peak_to_peak": float(np.ptp(values)) if values.size else np.nan,
                })
    return rows


CALIBRATION_STAGE_NAMES = ("raw_working", "after_m", "after_p", "after_alpha", "full_current")
CALIBRATION_STAGE_LABELS = {
    "raw_working": "raw",
    "after_m": "m",
    "after_p": "m_p",
    "after_alpha": "m_p_alpha",
    "full_current": "full",
}


def _group_values(values, indices, wave_index):
    return np.asarray(values[indices, wave_index], dtype=float)


def calibration_stage_group_rows(record, wavelength_indices):
    """Summarize centered sequential calibration stages by IFU/AMP group."""
    rows = []
    stages = record["centered_stages"]
    # Component locations are retained in their native exposure units.  The
    # correlation calculation below centers each group population itself.
    components = record["components"]
    full_blank = record["selected_blank"]
    stage_blanks = record["stage_selected_blank"]
    effects = {
        "m_effect": stages["after_m"] - stages["raw_working"],
        "p_effect": stages["after_p"] - stages["after_m"],
        "alpha_effect": stages["after_alpha"] - stages["after_p"],
        "sky_effect": stages["full_current"] - stages["after_alpha"],
    }
    for wave_index in wavelength_indices:
        wavelength = float(validated_m101.DEF_WAVE[wave_index])
        for group in record["groups"]:
            indices = group["indices"]
            blank_indices = indices[full_blank[indices]]
            row = {
                "H5": record["h5"], "exposure": group["exposure"],
                "wavelength": wavelength, "SPECID": group["specid"],
                "IFUSLOT": group["ifuslot"], "IFUID": group["ifuid"], "AMP": group["amp"],
                "N_blank": int(np.isfinite(record["current_after"][blank_indices, wave_index]).sum()),
            }
            for name in CALIBRATION_STAGE_NAMES:
                stage_indices = indices[stage_blanks[name][indices]]
                values = _group_values(stages[name], stage_indices, wave_index)
                label = CALIBRATION_STAGE_LABELS[name]
                row["%s_N_finite" % label] = int(np.isfinite(values).sum())
                row["%s_location" % label] = robust_location(values)
                row["%s_scatter" % label] = robust_scatter(values)
            for name in ("p_component", "alpha_component", "sky_interaction_component"):
                values = _group_values(components[name], blank_indices, wave_index)
                row["%s_location" % name] = robust_location(values)
                row["%s_scatter" % name] = robust_scatter(values)
            row["m_coefficient_location"] = robust_location(
                (1.0 / record["multiplier"][blank_indices]) - 1.0)
            for name, values in effects.items():
                row["%s_location" % name] = robust_location(values[blank_indices, wave_index])
            rows.append(row)
    return rows


def _summary_values(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not values.size:
        return {"N_groups": 0, "median": np.nan, "p16": np.nan, "p84": np.nan,
                "robust_scatter": np.nan, "peak_to_peak": np.nan}
    return {"N_groups": int(values.size), "median": float(np.median(values)),
            "p16": float(np.percentile(values, 16)),
            "p84": float(np.percentile(values, 84)),
            "robust_scatter": robust_scatter(values),
            "peak_to_peak": float(np.ptp(values))}


def calibration_stage_nonuniformity_rows(group_rows):
    """Return one wide row per H5/exposure/wavelength for blank groups."""
    rows = []
    keys = sorted(set((row["H5"], row["exposure"], row["wavelength"]) for row in group_rows))
    stage_fields = {"raw_working": "raw", "after_m": "m", "after_p": "m_p",
                    "after_alpha": "m_p_alpha", "full_current": "full"}
    for h5, exposure, wavelength in keys:
        subset = [row for row in group_rows
                  if (row["H5"], row["exposure"], row["wavelength"]) ==
                  (h5, exposure, wavelength)]
        output = {"H5": h5, "exposure": exposure, "wavelength": wavelength}
        scatters = {}
        for stage, field in stage_fields.items():
            stats = _summary_values([row["%s_location" % field] for row in subset])
            scatters[stage] = stats["robust_scatter"]
            for metric in ("N_groups", "median", "p16", "p84", "robust_scatter", "peak_to_peak"):
                output["%s_%s" % (field, metric)] = stats[metric]
        output["delta_scatter_m"] = scatters["after_m"] - scatters["raw_working"]
        output["delta_scatter_p"] = scatters["after_p"] - scatters["after_m"]
        output["delta_scatter_alpha"] = scatters["after_alpha"] - scatters["after_p"]
        output["delta_scatter_sky"] = scatters["full_current"] - scatters["after_alpha"]
        output["ratio_scatter_m"] = scatters["after_m"] / scatters["raw_working"] if scatters["raw_working"] > 0 else np.nan
        output["ratio_scatter_p"] = scatters["after_p"] / scatters["after_m"] if scatters["after_m"] > 0 else np.nan
        output["ratio_scatter_alpha"] = scatters["after_alpha"] / scatters["after_p"] if scatters["after_p"] > 0 else np.nan
        output["ratio_scatter_sky"] = scatters["full_current"] / scatters["after_alpha"] if scatters["after_alpha"] > 0 else np.nan
        rows.append(output)
    return rows


def _correlation_pair(x, y):
    result = {"pearson": pearson(x, y), "spearman": np.nan}
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    finite = np.isfinite(x) & np.isfinite(y)
    if finite.sum() >= 3 and np.std(x[finite]) > 0.0 and np.std(y[finite]) > 0.0:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            value = spearmanr(x[finite], y[finite]).statistic
        result["spearman"] = float(value) if np.isfinite(value) else np.nan
    return result


def calibration_component_correlations(group_rows):
    rows = []
    keys = sorted(set((row["H5"], row["exposure"], row["wavelength"]) for row in group_rows))
    comparisons = (
        ("final_vs_m", "full_location", "m_coefficient_location"),
        ("final_vs_p_component", "full_location", "p_component_location"),
        ("final_vs_alpha_component", "full_location", "alpha_component_location"),
        ("final_vs_sky_interaction", "full_location", "sky_interaction_component_location"),
        ("m_increment_vs_m_effect", "m_effect_location", "m_effect_location"),
        ("p_increment_vs_p_component", "p_effect_location", "p_component_location"),
        ("alpha_increment_vs_alpha_component", "alpha_effect_location", "alpha_component_location"),
        ("sky_increment_vs_sky_interaction", "sky_effect_location", "sky_interaction_component_location"),
    )
    for key in keys:
        h5, exposure, wavelength = key
        subset = [row for row in group_rows
                  if (row["H5"], row["exposure"], row["wavelength"]) == key]
        output = {"H5": h5, "exposure": exposure, "wavelength": wavelength,
                  "N_groups": len(subset)}
        for name, left, right in comparisons:
            pair = _correlation_pair([row[left] for row in subset],
                                     [row[right] for row in subset])
            output["%s_pearson" % name] = pair["pearson"]
            output["%s_spearman" % name] = pair["spearman"]
        rows.append(output)
    return rows


def component_spectra(records):
    rows = []
    arrays = {name: [] for name in ("p_component", "alpha_component",
                                    "sky_interaction_component", "current_residual")}
    for record in records:
        for exposure in range(1, validated_m101.N_EXPOSURES + 1):
            selected = (record["labels"] == exposure) & record["selected_blank"]
            values = {
                "p_component": record["components"]["p_component"][selected],
                "alpha_component": record["components"]["alpha_component"][selected],
                "sky_interaction_component": record["components"]["sky_interaction_component"][selected],
                # The final current residual is the exposure-wide spectrum
                # that was subtracted, rather than the already-centered
                # blank fibers (whose robust location is identically zero).
                "current_residual": record["current_residuals"][exposure][None, :],
            }
            for name, value in values.items():
                arrays[name].append(robust_spectrum(value))
            spectra = {name: robust_spectrum(value) for name, value in values.items()}
            for index, wavelength in enumerate(validated_m101.DEF_WAVE):
                rows.append({"H5": record["h5"], "exposure": exposure,
                             "wavelength": float(wavelength),
                             **{name: spectrum[index] for name, spectrum in spectra.items()}})
    metrics = {}
    for name, values in arrays.items():
        matrix = np.asarray(values, dtype=float)
        median = column_stat(matrix)
        high_frequency = median - gaussian_filter1d(np.nan_to_num(median, nan=0.0), 10.0)
        metrics[name] = {
            "median_high_frequency_rms": float(np.sqrt(np.nanmean(high_frequency ** 2))),
            "median_at_3636": float(median[np.argmin(np.abs(validated_m101.DEF_WAVE - 3636.0))]),
            "median_at_5000": float(median[np.argmin(np.abs(validated_m101.DEF_WAVE - 5000.0))]),
        }
    return rows, arrays, metrics


def plot_component_spectra(path, arrays):
    labels = {"p_component": "p component", "alpha_component": "alpha*K*f component",
        "sky_interaction_component": "QR sky interaction", "current_residual": "final current residual"}
    colors = {"p_component": "tab:blue", "alpha_component": "tab:orange",
              "sky_interaction_component": "tab:green", "current_residual": "tab:red"}
    fig, ax = plt.subplots(figsize=(12, 5))
    for name, matrix in arrays.items():
        median = column_stat(matrix)
        p16, p84 = column_stat(matrix, 16), column_stat(matrix, 84)
        ax.plot(validated_m101.DEF_WAVE, median, color=colors[name], label=labels[name])
        ax.fill_between(validated_m101.DEF_WAVE, p16, p84, color=colors[name], alpha=0.12)
    ax.axhline(0.0, color="black", lw=0.8)
    for wavelength in (3636.0, 4000.0, 4500.0, 5000.0):
        ax.axvline(wavelength, color="0.5", ls=":" if wavelength != 3636.0 else "--", lw=0.8)
    ax.set(xlabel="Wavelength (A)", ylabel="Centered native calibrated units",
           title="Calibration components in externally blank fibers")
    ax.grid(alpha=0.2); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def group_qr_rows(record, wave_index):
    rows = []
    for group in record["groups"]:
        indices = group["indices"]
        blank_indices = indices[record["selected_blank"][indices]]
        rows.append({
            "H5": record["h5"], "exposure": group["exposure"],
            "SPECID": group["specid"], "IFUSLOT": group["ifuslot"],
            "IFUID": group["ifuid"], "AMP": group["amp"],
            "ra": float(np.nanmedian(record["ra"][indices])),
            "dec": float(np.nanmedian(record["dec"][indices])),
            "qr_location": robust_location(record["qr_sky"][indices, wave_index]),
            "qr_blank_location": robust_location(record["qr_sky"][blank_indices, wave_index]),
        })
    return rows


def qr_structure_summary(records, group_rows, wavelength_indices):
    rows = []
    metrics = []
    for record in records:
        for wave_index in wavelength_indices:
            wavelength = float(validated_m101.DEF_WAVE[wave_index])
            for exposure in range(1, validated_m101.N_EXPOSURES + 1):
                exposure_indices = np.flatnonzero(record["labels"] == exposure)
                qr_values = record["qr_sky"][exposure_indices, wave_index]
                rows.append({"H5": record["h5"], "exposure": exposure,
                             "wavelength": wavelength, "level": "exposure",
                             "group": "all", "N_fibers": int(np.isfinite(qr_values).sum()),
                             "location": robust_location(qr_values)})
                for level, key in (("IFUSLOT", "ifuslot"), ("AMP", "amp"),
                                   ("IFUSLOT_AMP", "identity")):
                    keys = sorted(set(group[key] for group in record["groups"]
                                      if group["exposure"] == exposure))
                    for group_key in keys:
                        groups = [group for group in record["groups"]
                                  if group["exposure"] == exposure and group[key] == group_key]
                        indices = np.concatenate([group["indices"] for group in groups])
                        values = record["qr_sky"][indices, wave_index]
                        rows.append({"H5": record["h5"], "exposure": exposure,
                                     "wavelength": wavelength, "level": level,
                                     "group": str(group_key),
                                     "N_fibers": int(np.isfinite(values).sum()),
                                     "location": robust_location(values)})
                group_subset = [row for row in group_rows
                                if row["H5"] == record["h5"] and row["exposure"] == exposure
                                and row["wavelength"] == wavelength]
                group_values = np.asarray([row["total_all_location"] - row["current_pre_all_location"]
                                           for row in group_subset], dtype=float)
                amp_values = [robust_location(group_values[np.asarray(
                    [row["AMP"] == amp for row in group_subset], dtype=bool)])
                              for amp in sorted(set(row["AMP"] for row in group_subset))]
                ifuslot_values = [robust_location(group_values[np.asarray(
                    [row["IFUSLOT"] == slot for row in group_subset], dtype=bool)])
                                  for slot in sorted(set(row["IFUSLOT"] for row in group_subset))]
                metrics.append({
                    "H5": record["h5"], "exposure": exposure, "wavelength": wavelength,
                    "exposure_qr_median": robust_location(qr_values),
                    "ifuslot_median_scatter": robust_scatter(ifuslot_values),
                    "amp_median_scatter": robust_scatter(amp_values),
                    "ifuslot_amp_median_scatter": robust_scatter(group_values),
                })
    return rows, metrics


def spectrum_rows(records, wave_count):
    rows = []
    difference_spectra = []
    for record in records:
        for exposure in range(1, validated_m101.N_EXPOSURES + 1):
            indices = np.flatnonzero(record["labels"] == exposure)
            selected = indices[record["selected_blank"][indices]]
            qr = robust_spectrum(record["qr_sky"][selected])
            fresh = record["fresh_skies"][exposure]
            current = record["current_residuals"][exposure]
            fresh_minus_qr = fresh - qr
            current_minus_reset = robust_spectrum(
                record["current_after"][selected] - record["sky_reset"][selected])
            difference_spectra.append(current_minus_reset)
            nfinite = finite_spectrum_count(record["current_pre"][selected])
            for wave_index in range(wave_count):
                rows.append({
                    "H5": record["h5"], "exposure": exposure,
                    "wavelength": float(validated_m101.DEF_WAVE[wave_index]),
                    "N_blank_finite": int(nfinite[wave_index]),
                    "QR_sky_blank": qr[wave_index], "fresh_sky": fresh[wave_index],
                    "current_residual": current[wave_index],
                    "fresh_minus_QR": fresh_minus_qr[wave_index],
                    "fresh_minus_current_residual": fresh[wave_index] - current[wave_index],
                    "current_minus_sky_reset": current_minus_reset[wave_index],
                })
    return rows, np.asarray(difference_spectra, dtype=float)


def write_csv(path, rows, fields):
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: "" if isinstance(row.get(field), float) and
                             not np.isfinite(row[field]) else row.get(field, "")
                             for field in fields})


def write_spectral_fits(path, rows):
    fields = ("H5", "exposure", "wavelength", "N_blank_finite", "QR_sky_blank",
              "fresh_sky", "current_residual", "fresh_minus_QR",
              "fresh_minus_current_residual", "current_minus_sky_reset")
    columns = []
    for field in fields:
        values = [row[field] for row in rows]
        if field == "H5":
            columns.append(fits.Column(name=field, format="40A", array=np.asarray(values, dtype="S40")))
        elif field in ("exposure", "N_blank_finite"):
            columns.append(fits.Column(name=field, format="J", array=np.asarray(values, dtype=np.int32)))
        else:
            columns.append(fits.Column(name=field, format="D", array=np.asarray(values, dtype=float)))
    fits.HDUList([fits.PrimaryHDU(), fits.BinTableHDU.from_columns(columns, name="SKY_RESET")]).writeto(
        path, overwrite=True)


def summarize_spectra_plot(path, rows, difference_spectra):
    wavelengths = validated_m101.DEF_WAVE
    keys = sorted(set((row["H5"], row["exposure"]) for row in rows))
    arrays = {name: np.asarray([[row[name] for row in rows
                                 if (row["H5"], row["exposure"]) == key]
                                for key in keys])
              for name in ("QR_sky_blank", "fresh_sky", "current_residual")}
    difference = np.asarray(difference_spectra, dtype=float)
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    colors = {"QR_sky_blank": "tab:orange", "fresh_sky": "tab:green",
              "current_residual": "tab:red"}
    labels = {"QR_sky_blank": "QR sky, blank fibers", "fresh_sky": "fresh sky from calibrated total",
              "current_residual": "current residual from current pre-residual"}
    for name, values in arrays.items():
        median = column_stat(values)
        p16, p84 = column_stat(values, 16), column_stat(values, 84)
        axes[0].plot(wavelengths, median, color=colors[name], label=labels[name])
        axes[0].fill_between(wavelengths, p16, p84, color=colors[name], alpha=0.15)
    median = column_stat(arrays["fresh_sky"] - arrays["QR_sky_blank"])
    axes[1].plot(wavelengths, median, color="tab:blue", label="fresh sky - QR sky")
    if difference.size:
        p16, p84 = column_stat(difference, 16), column_stat(difference, 84)
        axes[1].fill_between(wavelengths, p16, p84, color="tab:purple", alpha=0.18,
                             label="current after - sky reset")
    axes[1].axhline(0.0, color="black", lw=0.8)
    for ax in axes:
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("Native calibrated units")
    axes[0].set_title("QR sky, fresh sky, and current residual")
    axes[1].set_ylabel("Difference")
    axes[1].set_xlabel("Wavelength (A)")
    axes[1].set_title("Sky-model disagreement")
    for wavelength in (3636.0, 4000.0, 4500.0, 5000.0):
        axes[1].axvline(wavelength, color="0.4", ls=":", lw=0.7)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_difference_spectrum(path, difference_spectra):
    wave = validated_m101.DEF_WAVE
    fig, ax = plt.subplots(figsize=(11, 5))
    if difference_spectra.size:
        median = column_stat(difference_spectra)
        p16, p84 = column_stat(difference_spectra, 16), column_stat(difference_spectra, 84)
        ax.plot(wave, median, color="tab:blue", label="median: current after - sky reset")
        ax.fill_between(wave, p16, p84, color="tab:blue", alpha=0.18, label="p16-p84")
    ax.axhline(0.0, color="black", lw=0.8)
    for wavelength in (3636.0, 4000.0, 4500.0, 5000.0):
        ax.axvline(wavelength, color="tab:red" if wavelength == 3636 else "0.5",
                   ls="--" if wavelength == 3636 else ":", lw=0.9)
    ax.set(xlabel="Wavelength (A)", ylabel="Native calibrated units",
           title="Current residual path minus fresh sky-reset path")
    ax.grid(alpha=0.2); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_stage_map(path, records, wave_index, title_prefix):
    arrays = {"sky reset": [], "current pre-residual": [],
              "current after residual": [], "current after - sky reset": []}
    ras, decs = [], []
    for record in records:
        ras.append(record["ra"]); decs.append(record["dec"])
        arrays["sky reset"].append(record["sky_reset"][:, wave_index])
        arrays["current pre-residual"].append(record["current_pre"][:, wave_index])
        arrays["current after residual"].append(record["current_after"][:, wave_index])
        arrays["current after - sky reset"].append(
            record["current_after"][:, wave_index] - record["sky_reset"][:, wave_index])
    ra = np.concatenate(ras); dec = np.concatenate(decs)
    x = (ra - M101_RA_DEG) * np.cos(np.deg2rad(M101_DEC_DEG)) * 60.0
    y = (dec - M101_DEC_DEG) * 60.0
    values = {key: np.concatenate(value) for key, value in arrays.items()}
    usable = np.concatenate([values[key][np.isfinite(values[key])]
                             for key in list(values)[:3]])
    limit = float(np.percentile(np.abs(usable), 99.0)) if usable.size else 1.0
    limit = max(limit, 1e-12)
    display = np.arange(x.size)
    if display.size > 100000:
        display = display[np.linspace(0, display.size - 1, 100000, dtype=int)]
    fig, axes = plt.subplots(2, 2, figsize=(13, 10), sharex=True, sharey=True)
    for ax, (name, values) in zip(axes.flat, values.items()):
        finite = np.isfinite(values[display])
        if name == "current after - sky reset":
            difference_limit = float(np.percentile(np.abs(values[np.isfinite(values)]), 99.0)) if np.any(np.isfinite(values)) else limit
            difference_limit = max(difference_limit, 1e-12)
            norm = plt.Normalize(-difference_limit, difference_limit)
        else:
            norm = plt.Normalize(-limit, limit)
        scatter = ax.scatter(x[display][finite], y[display][finite], c=values[display][finite],
                             s=2, alpha=0.35, cmap="coolwarm", norm=norm, linewidths=0)
        ax.set_title(name)
        ax.grid(alpha=0.15)
        fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
    axes[0, 0].set_ylabel("Dec offset (arcmin)"); axes[1, 0].set_ylabel("Dec offset (arcmin)")
    axes[1, 0].set_xlabel("RA offset (arcmin)"); axes[1, 1].set_xlabel("RA offset (arcmin)")
    fig.suptitle("%s at %.0f A; common robust limits for first three panels" %
                 (title_prefix, validated_m101.DEF_WAVE[wave_index]))
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def _map_coordinates(records):
    ra = np.concatenate([record["ra"] for record in records])
    dec = np.concatenate([record["dec"] for record in records])
    x = (ra - M101_RA_DEG) * np.cos(np.deg2rad(M101_DEC_DEG)) * 60.0
    y = (dec - M101_DEC_DEG) * 60.0
    display = np.arange(x.size)
    if display.size > 100000:
        display = display[np.linspace(0, display.size - 1, 100000, dtype=int)]
    return x, y, display


def plot_calibration_stage_map(path, records, wave_index):
    names = ("raw_working", "after_m", "after_p", "after_alpha", "full_current")
    titles = ("raw working", "after m", "after p", "after alpha", "full current")
    arrays = {name: np.concatenate([record["centered_stages"][name][:, wave_index]
                                    for record in records]) for name in names}
    x, y, display = _map_coordinates(records)
    finite = np.concatenate([values[display][np.isfinite(values[display])] for values in arrays.values()])
    limit = float(np.percentile(np.abs(finite), 99.0)) if finite.size else 1.0
    limit = max(limit, 1e-12)
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), sharex=True, sharey=True)
    for ax, name, title in zip(axes.flat, names, titles):
        values = arrays[name][display]
        good = np.isfinite(values)
        scatter = ax.scatter(x[display][good], y[display][good], c=values[good], s=2,
                             alpha=0.35, cmap="coolwarm", norm=plt.Normalize(-limit, limit), linewidths=0)
        ax.set_title(title); ax.grid(alpha=0.15)
        fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
    axes.flat[-1].axis("off")
    axes[0, 0].set_ylabel("Dec offset (arcmin)"); axes[1, 0].set_ylabel("Dec offset (arcmin)")
    axes[1, 0].set_xlabel("RA offset (arcmin)"); axes[1, 1].set_xlabel("RA offset (arcmin)")
    fig.suptitle("Centered calibration stages at %.0f A" % validated_m101.DEF_WAVE[wave_index])
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_calibration_components_map(path, records, wave_index):
    names = ("p_component", "alpha_component", "sky_interaction_component", "current_after")
    titles = ("p component", "alpha*K*f component", "QR sky interaction", "final current residual")
    arrays = {}
    for name in names:
        if name == "current_after":
            arrays[name] = np.concatenate([record["current_after"][:, wave_index] for record in records])
        else:
            arrays[name] = np.concatenate([record["centered_components"][name][:, wave_index]
                                           for record in records])
    x, y, display = _map_coordinates(records)
    finite = np.concatenate([values[display][np.isfinite(values[display])] for values in arrays.values()])
    limit = float(np.percentile(np.abs(finite), 99.0)) if finite.size else 1.0
    limit = max(limit, 1e-12)
    fig, axes = plt.subplots(2, 2, figsize=(13, 10), sharex=True, sharey=True)
    for ax, name, title in zip(axes.flat, names, titles):
        values = arrays[name][display]
        good = np.isfinite(values)
        scatter = ax.scatter(x[display][good], y[display][good], c=values[good], s=2,
                             alpha=0.35, cmap="coolwarm", norm=plt.Normalize(-limit, limit), linewidths=0)
        ax.set_title(title); ax.grid(alpha=0.15)
        fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
    axes[0, 0].set_ylabel("Dec offset (arcmin)"); axes[1, 0].set_ylabel("Dec offset (arcmin)")
    axes[1, 0].set_xlabel("RA offset (arcmin)"); axes[1, 1].set_xlabel("RA offset (arcmin)")
    fig.suptitle("Centered calibration components at %.0f A" % validated_m101.DEF_WAVE[wave_index])
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_qr_structure(path, qr_rows_by_wave):
    fig, axes = plt.subplots(1, len(qr_rows_by_wave), figsize=(13, 5), squeeze=False)
    for ax, (wavelength, rows) in zip(axes[0], qr_rows_by_wave.items()):
        values = np.asarray([row["qr_location"] for row in rows], dtype=float)
        finite = np.isfinite(values)
        limit = max(float(np.percentile(np.abs(values[finite] - np.median(values[finite])), 99.0)), 1e-12) if finite.any() else 1.0
        centered = values - np.nanmedian(values)
        scatter = ax.scatter([row["ra"] for row in rows], [row["dec"] for row in rows],
                             c=centered, s=9, cmap="coolwarm", norm=plt.Normalize(-limit, limit),
                             linewidths=0)
        ax.set_title("QR sky %.0f A" % wavelength)
        ax.set_xlabel("RA (deg)"); ax.grid(alpha=0.15)
        fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04, label="group QR sky - median")
    axes[0, 0].set_ylabel("Dec (deg)")
    fig.suptitle("Stored Quick Reduction sky spatial structure")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def pearson(x, y):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    finite = np.isfinite(x) & np.isfinite(y)
    if finite.sum() < 3 or np.std(x[finite]) == 0.0 or np.std(y[finite]) == 0.0:
        return np.nan
    return float(np.corrcoef(x[finite], y[finite])[0, 1])


def write_exposure_summary(path, records, wavelength_indices, group_rows):
    rows = []
    for record in records:
        for exposure in range(1, validated_m101.N_EXPOSURES + 1):
            selected = (record["labels"] == exposure) & record["selected_blank"]
            current_after = record["current_after"][selected]
            reset = record["sky_reset"][selected]
            row = {"H5": record["h5"], "exposure": exposure,
                   "N_blank": int(selected.sum())}
            for wave_index in wavelength_indices:
                wavelength = float(validated_m101.DEF_WAVE[wave_index])
                suffix = "3636" if wavelength == 3636.0 else "5000" if wavelength == 5000.0 else str(int(wavelength))
                group_subset = [item for item in group_rows
                                if item["H5"] == record["h5"] and item["exposure"] == exposure
                                and item["wavelength"] == wavelength]
                current_levels = np.asarray([item["current_after_blank_location"] for item in group_subset], dtype=float)
                reset_levels = np.asarray([item["reset_blank_location"] for item in group_subset], dtype=float)
                row.update({
                    "current_residual_%s" % suffix: record["current_residuals"][exposure][wave_index],
                    "fresh_sky_%s" % suffix: record["fresh_skies"][exposure][wave_index],
                    "QR_sky_blank_%s" % suffix: robust_location(record["qr_sky"][selected, wave_index]),
                    "current_blank_after_median_%s" % suffix: robust_location(current_after[:, wave_index]),
                    "reset_blank_after_median_%s" % suffix: robust_location(reset[:, wave_index]),
                    "current_blank_spatial_scatter_%s" % suffix: robust_scatter(current_levels),
                    "reset_blank_spatial_scatter_%s" % suffix: robust_scatter(reset_levels),
                    "current_blank_amp_range_%s" % suffix: float(np.ptp(current_levels[np.isfinite(current_levels)])) if np.any(np.isfinite(current_levels)) else np.nan,
                    "reset_blank_amp_range_%s" % suffix: float(np.ptp(reset_levels[np.isfinite(reset_levels)])) if np.any(np.isfinite(reset_levels)) else np.nan,
                })
            rows.append(row)
    fields = list(rows[0]) if rows else ["H5", "exposure", "N_blank"]
    write_csv(path, rows, fields)
    return rows


def full_spectral_summary(spectral_rows, difference_spectra):
    rows = []
    for wave_index, wavelength in enumerate(validated_m101.DEF_WAVE):
        values = np.asarray([row["fresh_minus_QR"] for row in spectral_rows
                             if row["wavelength"] == float(wavelength)], dtype=float)
        values = values[np.isfinite(values)]
        reset_values = (difference_spectra[:, wave_index]
                        if difference_spectra.size else np.asarray([], dtype=float))
        reset_values = reset_values[np.isfinite(reset_values)]
        rows.append({
            "wavelength": float(wavelength),
            "fresh_minus_QR_median": robust_location(values),
            "fresh_minus_QR_p16": np.percentile(values, 16) if values.size else np.nan,
            "fresh_minus_QR_p84": np.percentile(values, 84) if values.size else np.nan,
            "current_minus_reset_median": robust_location(reset_values),
            "current_minus_reset_p16": np.percentile(reset_values, 16) if reset_values.size else np.nan,
            "current_minus_reset_p84": np.percentile(reset_values, 84) if reset_values.size else np.nan,
        })
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("h5files", help="input VIRUS H5 glob")
    parser.add_argument("--fit-h5", required=True)
    parser.add_argument("--fq-template", required=True)
    parser.add_argument("--external-blank-fibers", required=True)
    parser.add_argument("--output-dir", default="m101_sky_reset_diagnostic")
    parser.add_argument("--diagnostic-wavelengths", default=",".join(str(v) for v in DEFAULT_WAVELENGTHS))
    parser.add_argument("--minimum-finite-fraction", type=float, default=DEFAULT_MINIMUM_FINITE_FRACTION)
    parser.add_argument("--minimum-sky-fibers", type=int, default=DEFAULT_MINIMUM_SKY_FIBERS)
    args = parser.parse_args()
    if not 0.0 < args.minimum_finite_fraction <= 1.0:
        parser.error("--minimum-finite-fraction must be in (0, 1]")
    if args.minimum_sky_fibers < 1:
        parser.error("--minimum-sky-fibers must be positive")
    h5files = expand_h5_glob(args.h5files)
    requested_wavelengths, wavelength_indices = parse_wavelengths(args.diagnostic_wavelengths)
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    timings = {}

    fit_started = time.perf_counter()
    calibration, _, fit_provenance = cube_builder._read_fit_calibration(args.fit_h5)
    matches, input_amplifiers = cube_builder._preflight_matches(h5files, calibration)
    timings["bayesian_h5_read_and_matching"] = time.perf_counter() - fit_started
    fq_template = validated_m101.load_fq(args.fq_template)

    external_started = time.perf_counter()
    external_by_h5, external_provenance = cube_builder._load_external_blank_fibers(
        args.external_blank_fibers, h5files)
    timings["external_blank_csv_read"] = time.perf_counter() - external_started
    print("Sky reset diagnostic")
    print("  H5 files=%d; external blank table=%s" % (len(h5files), external_provenance["path"]))
    print("  DEF_WAVE=%d planes from %.1f to %.1f A" %
          (validated_m101.DEF_WAVE.size, validated_m101.DEF_WAVE[0], validated_m101.DEF_WAVE[-1]))
    print("  requested diagnostic wavelengths=%s" % ", ".join("%.0f" % v for v in requested_wavelengths))
    print("  minimum finite fraction=%.3f; minimum sky fibers=%d" %
          (args.minimum_finite_fraction, args.minimum_sky_fibers))

    records = []
    h5_read_seconds = 0.0
    calibration_seconds = 0.0
    residual_started = time.perf_counter()
    for h5file in h5files:
        record = calibrate_h5_stages(h5file, matches, fq_template)
        h5_read_seconds += record.pop("read_seconds")
        calibration_seconds += record.pop("calibration_seconds")
        residual_stages(record, external_by_h5[h5file.name],
                        args.minimum_finite_fraction, args.minimum_sky_fibers)
        records.append(record)
        for row in record["exposure_rows"]:
            print("  %s exposure %d: N_blank=%d; finite candidates=%d; inside/outside 6 arcmin=%d/%d" %
                  (row["H5"], row["exposure"], row["N_blank"], row["N_sufficient_finite"],
                   row["N_blank_inside_6arcmin"], row["N_blank_outside_6arcmin"]))
    timings["h5_spectral_reads"] = h5_read_seconds
    timings["calibration_construction"] = calibration_seconds
    timings["residual_calculations"] = time.perf_counter() - residual_started

    group_started = time.perf_counter()
    group_rows = [row for record in records for row in group_level_rows(record, wavelength_indices)]
    nonuniformity = nonuniformity_rows(group_rows)
    calibration_stage_rows = [row for record in records
                              for row in calibration_stage_group_rows(record, wavelength_indices)]
    calibration_stage_nonuniformity = calibration_stage_nonuniformity_rows(calibration_stage_rows)
    component_correlations = calibration_component_correlations(calibration_stage_rows)
    qr_structure_rows, qr_structure_metrics = qr_structure_summary(
        records, group_rows, wavelength_indices)
    qr_rows_by_wave = {}
    for wave_index in wavelength_indices:
        wavelength = float(validated_m101.DEF_WAVE[wave_index])
        qr_rows_by_wave[wavelength] = [row for record in records
                                       for row in group_qr_rows(record, wave_index)]
    timings["group_aggregation"] = time.perf_counter() - group_started

    spectral_rows, difference_spectra = spectrum_rows(records, validated_m101.DEF_WAVE.size)
    spectral_summary = full_spectral_summary(spectral_rows, difference_spectra)
    component_spectral_rows, component_arrays, component_spectral_metrics = component_spectra(records)
    total_positions = int(sum(record["ra"].size for record in records))
    used_blank = int(sum(np.sum(record["selected_blank"]) for record in records))
    total_blank = int(external_provenance["blank_rows"])

    output_started = time.perf_counter()
    exposure_summary = write_exposure_summary(
        output_dir / "exposure_sky_reset_summary.csv", records, wavelength_indices, group_rows)
    write_spectral_fits(output_dir / "sky_reset_spectral_comparison.fits", spectral_rows)
    spectral_fields = ("H5", "exposure", "wavelength", "N_blank_finite", "QR_sky_blank",
                       "fresh_sky", "current_residual", "fresh_minus_QR",
                       "fresh_minus_current_residual", "current_minus_sky_reset")
    write_csv(output_dir / "sky_reset_spectral_comparison.csv", spectral_rows, spectral_fields)
    group_fields = ("H5", "exposure", "wavelength", "SPECID", "IFUSLOT", "IFUID", "AMP",
                    "N_all", "N_blank")
    for stage in ("total", "current_pre", "current_after", "reset"):
        for population in ("all", "blank"):
            group_fields += ("%s_%s_location" % (stage, population),
                             "%s_%s_scatter" % (stage, population))
    write_csv(output_dir / "spatial_group_levels.csv", group_rows, group_fields)
    write_csv(output_dir / "spatial_nonuniformity_summary.csv", nonuniformity,
              ("H5", "exposure", "wavelength", "population", "stage", "N_groups",
               "median", "p16", "p84", "robust_scatter", "peak_to_peak"))
    calibration_group_fields = ["H5", "exposure", "wavelength", "SPECID", "IFUSLOT",
                                "IFUID", "AMP", "N_blank"]
    for field in ("raw", "m", "m_p", "m_p_alpha", "full"):
        calibration_group_fields += ["%s_N_finite" % field, "%s_location" % field,
                                     "%s_scatter" % field]
    calibration_group_fields += ["%s_location" % field for field in
                                 ("p_component", "alpha_component", "sky_interaction_component",
                                  "m_coefficient", "m_effect", "p_effect", "alpha_effect", "sky_effect")]
    write_csv(output_dir / "calibration_stage_group_levels.csv", calibration_stage_rows,
              calibration_group_fields)
    calibration_nonuniformity_fields = ["H5", "exposure", "wavelength"]
    for field in ("raw", "m", "m_p", "m_p_alpha", "full"):
        calibration_nonuniformity_fields += ["%s_%s" % (field, metric) for metric in
                                             ("N_groups", "median", "p16", "p84",
                                              "robust_scatter", "peak_to_peak")]
    calibration_nonuniformity_fields += ["delta_scatter_m", "delta_scatter_p",
                                         "delta_scatter_alpha", "delta_scatter_sky",
                                         "ratio_scatter_m", "ratio_scatter_p",
                                         "ratio_scatter_alpha", "ratio_scatter_sky"]
    write_csv(output_dir / "calibration_stage_nonuniformity.csv",
              calibration_stage_nonuniformity, calibration_nonuniformity_fields)
    correlation_fields = list(component_correlations[0]) if component_correlations else [
        "H5", "exposure", "wavelength", "N_groups"]
    write_csv(output_dir / "calibration_component_correlations.csv",
              component_correlations, correlation_fields)
    write_csv(output_dir / "qr_sky_spatial_summary.csv", qr_structure_rows,
              ("H5", "exposure", "wavelength", "level", "group", "N_fibers", "location"))
    write_csv(output_dir / "calibration_component_spectra.csv", component_spectral_rows,
               ("H5", "exposure", "wavelength", "p_component", "alpha_component",
               "sky_interaction_component", "current_residual"))
    summarize_spectra_plot(output_dir / "sky_model_comparison.png", spectral_rows, difference_spectra)
    plot_difference_spectrum(output_dir / "current_minus_sky_reset_spectrum.png", difference_spectra)
    for wavelength in (3636.0, 5000.0):
        if wavelength in [float(validated_m101.DEF_WAVE[i]) for i in wavelength_indices]:
            index = wavelength_indices[[float(validated_m101.DEF_WAVE[i]) for i in wavelength_indices].index(wavelength)]
            plot_stage_map(output_dir / ("sky_reset_%d_spatial_comparison.png" % wavelength),
                           records, index, "Sky-reset spatial comparison")
            plot_calibration_stage_map(output_dir / ("calibration_stage_%d_spatial.png" % wavelength),
                                       records, index)
            plot_calibration_components_map(output_dir / ("calibration_components_%d_spatial.png" % wavelength),
                                            records, index)
    plot_qr_structure(output_dir / "qr_sky_3636_spatial_structure.png",
                      {w: qr_rows_by_wave[w] for w in qr_rows_by_wave if w in (3636.0, 5000.0)})
    plot_component_spectra(output_dir / "calibration_component_spectra.png", component_arrays)
    timings["plotting_output"] = time.perf_counter() - output_started
    timings["total_runtime"] = time.perf_counter() - started

    def stage_metric(wavelength, field):
        values = np.asarray([row[field] for row in calibration_stage_nonuniformity
                             if row["wavelength"] == wavelength], dtype=float)
        values = values[np.isfinite(values)]
        return float(np.median(values)) if values.size else np.nan

    def correlation_metric(wavelength, field):
        values = np.asarray([row[field] for row in component_correlations
                             if row["wavelength"] == wavelength], dtype=float)
        values = values[np.isfinite(values)]
        return float(np.median(values)) if values.size else np.nan

    decision_metrics = {}
    for wavelength in (3636.0, 5000.0):
        if wavelength not in [float(v) for v in validated_m101.DEF_WAVE[wavelength_indices]]:
            continue
        stage_scatter = {name: stage_metric(wavelength, "%s_robust_scatter" % field)
                         for name, field in (("raw", "raw"), ("m", "m"),
                                             ("m_p", "m_p"), ("m_p_alpha", "m_p_alpha"),
                                             ("full", "full"))}
        stage_range = {name: stage_metric(wavelength, "%s_peak_to_peak" % field)
                       for name, field in (("raw", "raw"), ("m", "m"),
                                           ("m_p", "m_p"), ("m_p_alpha", "m_p_alpha"),
                                           ("full", "full"))}
        decision_metrics[str(int(wavelength))] = {
            "stage_scatter": stage_scatter,
            "stage_peak_to_peak": stage_range,
            "incremental_scatter": {
                "m": stage_scatter["m"] - stage_scatter["raw"],
                "p": stage_scatter["m_p"] - stage_scatter["m"],
                "alpha": stage_scatter["m_p_alpha"] - stage_scatter["m_p"],
                "sky_interaction": stage_scatter["full"] - stage_scatter["m_p_alpha"],
            },
            "correlations": {
                "final_vs_m": correlation_metric(wavelength, "final_vs_m_pearson"),
                "final_vs_p_component": correlation_metric(wavelength, "final_vs_p_component_pearson"),
                "final_vs_alpha_component": correlation_metric(wavelength, "final_vs_alpha_component_pearson"),
                "final_vs_sky_interaction": correlation_metric(wavelength, "final_vs_sky_interaction_pearson"),
            },
        }

    h5_comparison = {}
    for wavelength in (3636.0, 5000.0):
        h5_comparison[str(int(wavelength))] = {}
        for h5 in sorted(set(row["H5"] for row in calibration_stage_nonuniformity)):
            h5_rows = [row for row in calibration_stage_nonuniformity
                       if row["H5"] == h5 and row["wavelength"] == wavelength]
            h5_comparison[str(int(wavelength))][h5] = {}
            for field in ("raw_robust_scatter", "m_robust_scatter", "m_p_robust_scatter",
                          "m_p_alpha_robust_scatter", "full_robust_scatter"):
                values = np.asarray([row[field] for row in h5_rows], dtype=float)
                values = values[np.isfinite(values)]
                h5_comparison[str(int(wavelength))][h5][field] = {
                    "median": float(np.median(values)) if values.size else np.nan,
                    "min": float(np.min(values)) if values.size else np.nan,
                    "max": float(np.max(values)) if values.size else np.nan,
                }

    identity = [record["identity"] for record in records]
    all_identity_deltas = [item["max_abs"] for item in identity]
    all_identity_relative = [item["max_relative"] for item in identity]
    no_sky_counterfactual = {}
    for wavelength in (3636.0, 5000.0):
        if wavelength not in [float(v) for v in validated_m101.DEF_WAVE[wavelength_indices]]:
            continue
        current = stage_metric(wavelength, "full_robust_scatter")
        no_sky = stage_metric(wavelength, "m_p_alpha_robust_scatter")
        no_sky_counterfactual[str(int(wavelength))] = {
            "current_scatter": current,
            "no_sky_interaction_scatter": no_sky,
            "improvement_factor": current / no_sky if np.isfinite(current) and no_sky > 0 else np.nan,
        }

    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "script": str(Path(__file__).resolve()),
        "input_h5_files": [str(path) for path in h5files],
        "fit_h5": fit_provenance,
        "external_blank_table": external_provenance,
        "H5_count": len(h5files),
        "exposure_count": len(h5files) * validated_m101.N_EXPOSURES,
        "total_fibers": total_positions,
        "external_blank_fibers_in_table": total_blank,
        "external_blank_fibers_used": used_blank,
        "wavelength_grid": {"start_A": float(validated_m101.DEF_WAVE[0]),
                            "stop_A": float(validated_m101.DEF_WAVE[-1]),
                            "planes": int(validated_m101.DEF_WAVE.size)},
        "requested_diagnostic_wavelengths": requested_wavelengths,
        "minimum_finite_fraction": args.minimum_finite_fraction,
        "minimum_sky_fibers": args.minimum_sky_fibers,
        "calibration_formula": "(Fibers.spectrum/Survey.offset-p_mean-alpha_mean*K_work*f(q)+Fibers.skyspectrum)/exp(posterior_z_mean)",
        "expanded_calibration_components": {
            "science_component": "raw_working/m",
            "p_component": "-p_mean/m",
            "alpha_component": "-alpha_mean*K_work*f(q)/m",
            "sky_interaction_component": "Fibers.skyspectrum*(1/m-1)",
            "stage_sequence": ["raw_working", "raw_working/m", "raw_working/m-p/m",
                                "raw_working/m-p/m-alpha*K_work*f(q)/m", "full current pre-residual"],
            "identity_max_abs": float(np.max(all_identity_deltas)) if all_identity_deltas else 0.0,
            "identity_max_relative": float(np.max(all_identity_relative)) if all_identity_relative else 0.0,
            "identity_verified": all(item["ok"] for item in identity),
        },
        "current_pre_residual_formula": "calibrated_total-Fibers.skyspectrum",
        "fresh_sky_formula": "biweight(calibrated_total[external_blank & sufficient_finite], axis=0)",
        "sky_reset_formula": "calibrated_total-fresh_sky; no post-total Fibers.skyspectrum subtraction",
        "spectral_rows": spectral_rows,
        "exposure_sky_reset_summary": exposure_summary,
        "spatial_nonuniformity_summary": nonuniformity,
        "qr_sky_spatial_metrics": qr_structure_metrics,
        "full_spectral_summaries": spectral_summary,
        "calibration_stage_nonuniformity": calibration_stage_nonuniformity,
        "calibration_component_correlations": component_correlations,
        "component_spectral_metrics": component_spectral_metrics,
        "h5_stage_comparison": h5_comparison,
        "no_sky_interaction_counterfactual": no_sky_counterfactual,
        "decision_metrics": decision_metrics,
        "timing_seconds": timings,
        "input_amplifier_observations": input_amplifiers,
        "no_cube_reconstruction": True,
        "no_h5_modification": True,
    }
    with (output_dir / "sky_reset_diagnostic_summary.json").open("w") as stream:
        json.dump(summary, stream, indent=2, default=lambda value: None if isinstance(value, float) and not np.isfinite(value) else value)

    print("\nCalibration-stage decision metrics")
    for wavelength in (3636.0, 5000.0):
        metric = decision_metrics.get(str(int(wavelength)))
        if not metric:
            continue
        print("  %.0f A median blank-group scatter: raw=%.6g; after m=%.6g; after p=%.6g; "
              "after alpha=%.6g; full=%.6g" %
              (wavelength, metric["stage_scatter"]["raw"], metric["stage_scatter"]["m"],
               metric["stage_scatter"]["m_p"], metric["stage_scatter"]["m_p_alpha"],
               metric["stage_scatter"]["full"]))
        increment = metric["incremental_scatter"]
        print("    incremental m=%.6g; p=%.6g; alpha=%.6g; sky interaction=%.6g" %
              (increment["m"], increment["p"], increment["alpha"], increment["sky_interaction"]))
        correlation = metric["correlations"]
        print("    final correlation m=%.6g; p=%.6g; alpha=%.6g; sky interaction=%.6g" %
              (correlation["final_vs_m"], correlation["final_vs_p_component"],
               correlation["final_vs_alpha_component"], correlation["final_vs_sky_interaction"]))
        print("    peak-to-peak raw=%.6g; after m=%.6g; after p=%.6g; after alpha=%.6g; full=%.6g" %
              tuple(metric["stage_peak_to_peak"][name]
                    for name in ("raw", "m", "m_p", "m_p_alpha", "full")))
    print("  expanded-equation identity: max abs=%.6g; max relative=%.6g; verified=%s" %
          (float(np.max(all_identity_deltas)) if all_identity_deltas else 0.0,
           float(np.max(all_identity_relative)) if all_identity_relative else 0.0,
           all(item["ok"] for item in identity)))
    print("  QR sky group structure: 3636 A scatter=%.6g; 5000 A scatter=%.6g" %
          (robust_scatter([row["qr_location"] for row in qr_rows_by_wave.get(3636.0, [])]),
           robust_scatter([row["qr_location"] for row in qr_rows_by_wave.get(5000.0, [])])))
    print("\nH5 stage comparison")
    for wavelength in (3636.0, 5000.0):
        for h5, values in h5_comparison.get(str(int(wavelength)), {}).items():
            full = values["full_robust_scatter"]
            print("  %.0f A %-24s full scatter median/min/max=%.6g/%.6g/%.6g" %
                  (wavelength, h5, full["median"], full["min"], full["max"]))
    print("\nCalibration-stage timing")
    for name, value in timings.items():
        print("  %-34s %.3f s" % (name + ":", value))
    print("  fiber positions:                    %d" % total_positions)

    blue = decision_metrics.get("3636", {})
    red = decision_metrics.get("5000", {})
    def dominant(metric):
        if not metric:
            return "mixed / not measured"
        increments = metric["incremental_scatter"]
        finite = {name: value for name, value in increments.items() if np.isfinite(value)}
        if not finite:
            return "mixed / not uniquely attributable"
        name = max(finite, key=lambda item: abs(finite[item]))
        return name
    print("\nComponent attribution")
    print("  dominant spatial contributor at 3636 A: %s" % dominant(blue))
    print("  dominant spatial contributor at 5000 A: %s" % dominant(red))
    print("  dominant spectral-structure contributor: inspect component_spectral_metrics; high-frequency RMS=%s" %
          ", ".join("%s=%.6g" % (name, values["median_high_frequency_rms"])
                    for name, values in component_spectral_metrics.items()))
    print("  interpretation: component attribution is diagnostic and does not change production code")


if __name__ == "__main__":
    main()
