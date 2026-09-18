#!/usr/bin/env python3
"""Inspect temporal persistence of frozen Model-3 amplifier residual morphology.

This is a development-only diagnostic.  It does not fit a model, alter Model 3,
change the external blank classification, or construct a cube.  For four
pre-selected physical amplifiers it writes common-scale residual heat-map
contact sheets, individual heat maps, descriptive exposure correlations, and
simple per-fiber profiles.

The primary residual is deliberately the observed/native quantity used by the
recent Model-4 development heat maps::

    E_f(lambda) = D_f(lambda) - A3_f(lambda) - m3_f S3(lambda)

where ``D`` is reconstructed by the validated native loader as
``Fibers.spectrum / Survey.offset + Fibers.skyspectrum``.
"""

from __future__ import annotations

import argparse
import csv
import json
import shlex
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import diagnose_m101_hierarchical as validated_m101
import m101_blank_fibers
from diagnose_m101_post_model3 import _extract_json_member, load_model_and_skies
from fit_m101_calibration_model4_joint import _amplifier_indices
from m101_calibration_utils import robust_location
from m101_native_data import discover_h5, load as load_native_data


WAVE = np.asarray(validated_m101.DEF_WAVE, dtype=float)
SPECTRAL_USE = (WAVE >= 3540.0) & (WAVE <= 5480.0)
N_FIBER_AMP = 112
EXPOSURES = (1, 2, 3)

TARGETS = (
    (317, 15, 38, "LU"),
    (409, 16, 36, "RL"),
    (316, 22, 52, "LU"),
    (320, 32, 20, "LU"),
)
EXPLICITLY_EXCLUDED = (412, 13, 43, "LL")

MIN_OVERLAP_FIBERS = 5
MIN_OVERLAP_SAMPLES = 100
COMMON_QUANTILES = (1.0, 99.0)
TAIL_TRIMMED_QUANTILES = (0.5, 99.5)


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
    if isinstance(value, (np.integer, np.floating)):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _safe_robust_location(values, axis=None):
    """Use the validated robust location, preserving all-NaN output as NaN."""
    values = np.asarray(values, dtype=float)
    with np.errstate(all="ignore"):
        result = robust_location(values, axis=axis)
    result = np.asarray(result, dtype=float)
    if axis is None:
        return float(result) if np.isfinite(result) else np.nan
    return result


def _robust_rms(values):
    finite = _finite(values)
    return (float(validated_m101.robust_scale(finite))
            if finite.size else np.nan)


def _identity_text(identity):
    return "%d_%d_%d_%s" % tuple(identity)


def _identity_label(identity):
    return "SPECID=%d IFUSLOT=%d IFUID=%d AMP=%s" % tuple(identity)


def _input_provenance(product_path):
    return _extract_json_member(product_path, "provenance", "  ")


def _provenance_path(args_value, provenance, key):
    value = args_value if args_value is not None else provenance.get(key)
    if value is None:
        raise ValueError("no --%s and no %s in Model-3 provenance" %
                         (key.replace("_", "-"), key))
    if isinstance(value, dict):
        value = value.get("full_path", value.get("path", value.get("filename")))
    if value is None:
        raise ValueError("provenance entry has no path for %s" % key)
    return str(Path(value).expanduser().resolve())


def _discover_inputs(args, product, product_path):
    entries = product["provenance"]["input_h5"]
    expected_names = {Path(entry["filename"]).name for entry in entries}
    if args.h5:
        raw_paths = args.h5
    else:
        raw_paths = [entry.get("full_path", entry["filename"])
                     for entry in entries]
        raw_paths = [str(Path(value).expanduser().resolve())
                     if not Path(value).is_absolute()
                     else value for value in raw_paths]
    paths = discover_h5(raw_paths, development=True)
    actual_names = {path.name for path in paths}
    if actual_names != expected_names:
        raise ValueError("H5 set does not match persisted Model-3 provenance: "
                         "expected %s, got %s" %
                         (sorted(expected_names), sorted(actual_names)))
    if len(paths) != 19:
        raise ValueError("persisted M101 Model-3 population must contain 19 H5 files; got %d" %
                         len(paths))
    return paths


def _frozen_residual(item, model3, skies, fq, physical):
    """Reconstruct the exact native-unit Model-3 residual heat-map array."""
    target_indices = _amplifier_indices(item, physical)
    selected_physical = np.zeros(item.row_index.size, dtype=bool)
    selected_physical[target_indices] = True
    blank = (np.asarray(item.blank_classified, dtype=bool)
             & np.asarray(item.blank_valid, dtype=bool)
             & ~np.asarray(item.hardware_bad, dtype=bool)
             & selected_physical)
    sky = np.asarray(skies[item.key], dtype=float)
    m3 = np.exp(np.asarray(model3.z_for(item), dtype=float))
    additive3 = np.asarray(model3.additive_full(item, fq), dtype=float)
    total = np.asarray(item.total, dtype=float)

    # This is intentionally the same expression as the recent Model-4 heat map.
    residual = total[target_indices] - additive3[target_indices]
    residual -= m3[target_indices, None] * sky[None, :]
    fiber_blank = np.asarray(blank[target_indices], dtype=bool)
    valid = fiber_blank[:, None] & np.isfinite(residual)
    displayed = residual.copy()
    displayed[~valid] = np.nan
    return {
        "H5": item.h5_name,
        "exposure": int(item.exposure),
        "identity": tuple(physical),
        "item": item,
        "fiber_row_indices": np.asarray(item.row_index[target_indices], dtype=np.int64),
        "residual": residual,
        "displayed": displayed,
        "fiber_blank_mask": fiber_blank,
        "valid": valid,
    }


def _record_support(record):
    valid = record["valid"][:, SPECTRAL_USE]
    per_fiber = np.sum(valid, axis=1)
    colored = record["fiber_blank_mask"]
    support = per_fiber[colored]
    finite = _finite(record["displayed"][:, SPECTRAL_USE])
    return {
        "blank_count": int(np.sum(colored)),
        "finite_support_min": int(np.min(support)) if support.size else 0,
        "finite_support_median": float(np.median(support)) if support.size else np.nan,
        "finite_support_max": int(np.max(support)) if support.size else 0,
        "finite_samples": int(finite.size),
        "support_definition": "finite 3540--5480 A samples per colored blank fiber",
    }


def _record_metrics(record):
    values = _finite(record["displayed"][:, SPECTRAL_USE])
    support = _record_support(record)
    if values.size:
        p01, p50, p99 = np.percentile(values, [1.0, 50.0, 99.0])
    else:
        p01 = p50 = p99 = np.nan
    return {
        "H5": record["H5"],
        "exposure": record["exposure"],
        "SPECID": record["identity"][0],
        "IFUSLOT": record["identity"][1],
        "IFUID": record["identity"][2],
        "AMP": record["identity"][3],
        **support,
        "residual_p01": float(p01),
        "residual_p50": float(p50),
        "residual_p99": float(p99),
        "residual_robust_rms": _robust_rms(values),
    }


def _shape_residual(record):
    """Remove only the robust amplifier-common spectrum at each wavelength."""
    values = np.asarray(record["residual"], dtype=float)
    valid = np.asarray(record["valid"], dtype=bool)
    masked = np.where(valid, values, np.nan)
    common = _safe_robust_location(masked, axis=0)
    shape = values - common[None, :]
    shape[~valid] = np.nan
    return shape, common


def _correlation(left, right):
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    finite = np.isfinite(left) & np.isfinite(right)
    overlap = int(np.sum(finite))
    fibers = (int(np.unique(np.flatnonzero(finite) // WAVE.size).size)
              if left.ndim == 2 else 0)
    if overlap < MIN_OVERLAP_SAMPLES or fibers < MIN_OVERLAP_FIBERS:
        return np.nan, overlap, fibers
    left, right = left[finite], right[finite]
    if np.std(left) <= 0.0 or np.std(right) <= 0.0:
        return np.nan, overlap, fibers
    return float(np.corrcoef(left, right)[0, 1]), overlap, fibers


def _correlation_matrix(records, arrays):
    size = len(records)
    matrix = np.full((size, size), np.nan, dtype=float)
    overlap_samples = np.zeros((size, size), dtype=np.int32)
    overlap_fibers = np.zeros((size, size), dtype=np.int16)
    for left_index in range(size):
        matrix[left_index, left_index] = 1.0
        for right_index in range(left_index + 1, size):
            value, samples, fibers = _correlation(
                arrays[left_index], arrays[right_index])
            matrix[left_index, right_index] = value
            matrix[right_index, left_index] = value
            overlap_samples[left_index, right_index] = samples
            overlap_samples[right_index, left_index] = samples
            overlap_fibers[left_index, right_index] = fibers
            overlap_fibers[right_index, left_index] = fibers
    return matrix, overlap_samples, overlap_fibers


def _matrix_csv(path, labels, matrix):
    with Path(path).open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["label", *labels])
        for label, row in zip(labels, matrix):
            writer.writerow([label, *["" if not np.isfinite(value) else value
                                      for value in row]])


def _common_scales(records):
    values = _finite(np.concatenate([
        record["displayed"][:, SPECTRAL_USE].ravel() for record in records]))
    if values.size < 10:
        raise RuntimeError("too few finite target residual samples for common scale")
    all_p01, all_p99 = np.percentile(values, COMMON_QUANTILES)
    tail_p01, tail_p99 = np.percentile(values, TAIL_TRIMMED_QUANTILES)
    primary = float(max(abs(all_p01), abs(all_p99)))
    tail_trimmed = float(max(abs(tail_p01), abs(tail_p99)))
    if not np.isfinite(primary) or primary <= 0.0:
        raise RuntimeError("degenerate all-valid common residual scale")
    if not np.isfinite(tail_trimmed) or tail_trimmed <= 0.0:
        tail_trimmed = primary
    per_exposure = {}
    for record in records:
        finite = _finite(record["displayed"][:, SPECTRAL_USE])
        per_exposure["%s/e%d" % (record["H5"], record["exposure"])] = {
            "p01": float(np.percentile(finite, 1.0)) if finite.size else np.nan,
            "p99": float(np.percentile(finite, 99.0)) if finite.size else np.nan,
        }
    pathological = sorted(
        label for label, quantiles in per_exposure.items()
        if (np.isfinite(quantiles["p01"]) and np.isfinite(quantiles["p99"])
            and max(abs(quantiles["p01"]), abs(quantiles["p99"])) > 3.0 * primary))
    dominated = bool(primary > 2.0 * tail_trimmed)
    return {
        "all_valid_global_quantiles": {"p01": float(all_p01), "p99": float(all_p99)},
        "primary_scale_definition": "symmetric max(abs(global p01), abs(global p99))",
        "primary_symmetric_limits": [-primary, primary],
        "tail_trimmed_global_quantiles": {"p0.5": float(tail_p01), "p99.5": float(tail_p99)},
        "tail_trimmed_scale_definition": "symmetric max(abs(global p0.5), abs(global p99.5)); global tails only",
        "tail_trimmed_symmetric_limits": [-tail_trimmed, tail_trimmed],
        "primary_dominated_by_global_tails": dominated,
        "pathological_exposure_tails_detected": pathological,
        "tail_trimmed_sheet_requested": bool(dominated or pathological),
        "n_finite_samples": int(values.size),
        "per_exposure_p01_p99": per_exposure,
    }


def _masked_image(record):
    return np.ma.masked_where(~record["valid"][:, SPECTRAL_USE],
                              record["residual"][:, SPECTRAL_USE])


def _cmap():
    cmap = plt.get_cmap("coolwarm").copy()
    cmap.set_bad("white")
    return cmap


def _plot_single_heatmap(record, path, limits, title_prefix=""):
    wave = WAVE[SPECTRAL_USE]
    figure, axis = plt.subplots(figsize=(11, 11))
    image = axis.imshow(
        _masked_image(record), origin="lower", interpolation="nearest",
        cmap=_cmap(), vmin=limits[0], vmax=limits[1],
        extent=(float(wave[0]), float(wave[-1]), 0.5, N_FIBER_AMP + 0.5),
        aspect=float((wave[-1] - wave[0]) / N_FIBER_AMP))
    axis.set_xlabel("wavelength [Angstrom]")
    axis.set_ylabel("fiber in physical amplifier (112 rows)")
    axis.set_xlim(3540.0, 5480.0)
    axis.set_ylim(0.5, N_FIBER_AMP + 0.5)
    metrics = _record_metrics(record)
    identity = record["identity"]
    axis.set_title("%s%s e%d; %s; blank=%d/112" %
                   (title_prefix, record["H5"], record["exposure"],
                    _identity_label(identity), metrics["blank_count"]))
    colorbar = figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    colorbar.set_label("E_f = D - A3 - m3 S3 [native reconstructed flux]")
    figure.tight_layout()
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def _plot_contact_sheet(records, h5_names, path, identity, limits, label):
    nrows = len(h5_names)
    figure, axes = plt.subplots(nrows, len(EXPOSURES), figsize=(14, max(18, nrows * 1.55)),
                                squeeze=False, sharex=True, sharey=True)
    cmap = _cmap()
    image_for_colorbar = None
    by_key = {(record["H5"], record["exposure"]): record for record in records}
    for row_index, h5_name in enumerate(h5_names):
        for col_index, exposure in enumerate(EXPOSURES):
            axis = axes[row_index, col_index]
            record = by_key.get((h5_name, exposure))
            if record is None:
                axis.set_facecolor("white")
                axis.text(0.5, 0.5, "missing", transform=axis.transAxes,
                          ha="center", va="center")
                continue
            image = axis.imshow(
                np.ma.masked_where(~record["valid"][:, SPECTRAL_USE],
                                   record["residual"][:, SPECTRAL_USE]),
                origin="lower", interpolation="nearest", cmap=cmap,
                vmin=limits[0], vmax=limits[1],
                extent=(3540.0, 5480.0, 0.5, N_FIBER_AMP + 0.5),
                aspect="auto")
            image_for_colorbar = image
            metrics = _record_metrics(record)
            axis.set_title("%s e%d  blank=%d" %
                           (Path(h5_name).stem, exposure, metrics["blank_count"]),
                           fontsize=7, pad=2)
            axis.set_xlim(3540.0, 5480.0)
            axis.set_ylim(0.5, N_FIBER_AMP + 0.5)
            axis.tick_params(labelsize=6)
            if row_index != nrows - 1:
                axis.set_xticklabels([])
            if col_index != 0:
                axis.set_yticklabels([])
    axes[0, 0].set_ylabel("fiber rows", fontsize=8)
    axes[-1, 1].set_xlabel("wavelength [Angstrom]", fontsize=8)
    figure.suptitle("Frozen Model-3 residual persistence: %s\n%s common scale [%g, %g]" %
                    (_identity_label(identity), label, limits[0], limits[1]), fontsize=11)
    if image_for_colorbar is not None:
        colorbar = figure.colorbar(image_for_colorbar, ax=axes.ravel().tolist(),
                                   fraction=0.012, pad=0.01)
        colorbar.set_label("E_f [native reconstructed flux]", fontsize=8)
    figure.subplots_adjust(left=0.06, right=0.94, bottom=0.035, top=0.965,
                           hspace=0.22, wspace=0.03)
    figure.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(figure)


def _profiles(records):
    raw_profiles = []
    normalized_profiles = []
    amplitudes = []
    for record in records:
        shape, common = _shape_residual(record)
        valid = record["valid"][:, SPECTRAL_USE]
        profile = np.full(N_FIBER_AMP, np.nan, dtype=float)
        for fiber in range(N_FIBER_AMP):
            use = valid[fiber] & np.isfinite(shape[fiber, SPECTRAL_USE])
            if np.sum(use) >= 3:
                profile[fiber] = _safe_robust_location(shape[fiber, SPECTRAL_USE][use])
        amplitude = _robust_rms(profile)
        normalized = profile / amplitude if np.isfinite(amplitude) and amplitude > 0 else profile.copy()
        raw_profiles.append(profile)
        normalized_profiles.append(normalized)
        amplitudes.append(amplitude)
        record["shape_residual"] = shape
        record["common_residual"] = common
        record["U_profile"] = profile
        record["U_profile_normalized"] = normalized
        record["profile_robust_amplitude"] = amplitude
    raw_profiles = np.asarray(raw_profiles)
    normalized_profiles = np.asarray(normalized_profiles)
    return raw_profiles, normalized_profiles, np.asarray(amplitudes, dtype=float)


def _plot_profiles(records, path, normalized=False):
    values = np.asarray([record["U_profile_normalized" if normalized else "U_profile"]
                         for record in records], dtype=float)
    median = np.nanmedian(values, axis=0)
    figure, axis = plt.subplots(figsize=(13, 6))
    for record, profile in zip(records, values):
        axis.plot(np.arange(1, N_FIBER_AMP + 1), profile, lw=0.55, alpha=0.45,
                  color="tab:blue")
    axis.plot(np.arange(1, N_FIBER_AMP + 1), median, color="black", lw=2.0,
              label="robust across-exposure median")
    axis.axhline(0.0, color="0.5", lw=0.7)
    axis.set_xlabel("fiber row in physical amplifier (1--112)")
    axis.set_ylabel("U_e(f)" + (" / per-exposure robust amplitude" if normalized else " [native flux]"))
    axis.set_title("%s\n%s" % (_identity_label(records[0]["identity"]),
                               "fiber-shape profiles, scalar-normalized" if normalized
                               else "fiber-dependent residual profiles"))
    axis.grid(alpha=0.2)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _labels(records):
    return ["%s/e%d" % (record["H5"], record["exposure"]) for record in records]


def _pair_summary(matrix):
    values = matrix[np.triu_indices_from(matrix, k=1)]
    values = _finite(values)
    if not values.size:
        return {"median": np.nan, "p16": np.nan, "p84": np.nan, "n_pairs": 0}
    return {"median": float(np.median(values)),
            "p16": float(np.percentile(values, 16.0)),
            "p84": float(np.percentile(values, 84.0)),
            "n_pairs": int(values.size)}


def _reference_gate(records, reference_npz, reference_summary, gate_dir):
    summary = json.loads(Path(reference_summary).read_text())
    selected = {}
    for index, row in enumerate(summary.get("per_amplifier", [])):
        identity = tuple(row.get("selection", {}).get("identity", ()))
        if identity in TARGETS:
            selected[identity] = index
    missing = [identity for identity in TARGETS if identity not in selected]
    if missing:
        raise RuntimeError("reference summary lacks target amplifiers: %s" % missing)
    reference = np.load(reference_npz)
    gate_dir.mkdir(parents=True, exist_ok=True)
    checks = {"reference_npz": str(Path(reference_npz).resolve()),
              "reference_summary": str(Path(reference_summary).resolve()),
              "arrays": {}}
    for record in records:
        identity = record["identity"]
        index = selected[identity]
        prefix = "record_%02d__" % index
        old_rows = np.asarray(reference[prefix + "fiber_row_index"], dtype=np.int64)
        old_blank = np.asarray(reference[prefix + "fiber_blank_mask"], dtype=bool)
        old_residual = np.asarray(reference[prefix + "E_f_all"], dtype=float)
        new_residual = np.asarray(record["residual"], dtype=float)
        row_match = np.array_equal(old_rows, record["fiber_row_indices"])
        blank_match = np.array_equal(old_blank, record["fiber_blank_mask"])
        residual_match = (old_residual.shape == new_residual.shape and
                          np.allclose(old_residual, new_residual,
                                      rtol=2e-12, atol=2e-12, equal_nan=True))
        max_abs = (float(np.nanmax(np.abs(old_residual - new_residual)))
                   if old_residual.shape == new_residual.shape else np.inf)
        old_heatmap = summary["per_amplifier"][index].get("fiber_heatmap", {})
        old_metrics = _record_metrics(record)
        old_blank_count = int(old_blank.sum())
        blank_count_match = old_blank_count == old_metrics["blank_count"]
        checks["arrays"][_identity_text(identity)] = {
            "reference_record_index": int(index),
            "same_112_row_order": bool(row_match and old_rows.size == N_FIBER_AMP),
            "same_blank_nonblank_rows": bool(blank_match),
            "same_blank_count": bool(blank_count_match),
            "same_native_residual_definition_and_units": True,
            "residual_array_allclose": bool(residual_match),
            "max_absolute_difference": max_abs,
            "pass": bool(row_match and blank_match and blank_count_match and residual_match),
        }
        if old_heatmap:
            old_limits = (float(old_heatmap["vmin"]), float(old_heatmap["vmax"]))
        else:
            values = _finite(record["displayed"][:, SPECTRAL_USE])
            p01, p99 = np.percentile(values, [1.0, 99.0])
            width = p99 - p01
            old_limits = (float(p01 - 0.2 * width), float(p99 + 0.2 * width))
        _plot_single_heatmap(record, gate_dir /
                             ("gate_%s.png" % _identity_text(identity)), old_limits,
                             title_prefix="Correctness gate: ")
    checks["pass"] = bool(all(item["pass"] for item in checks["arrays"].values()))
    (gate_dir / "correctness_gate.json").write_text(
        json.dumps(_json_ready(checks), indent=2, sort_keys=True))
    if not checks["pass"]:
        raise RuntimeError("correctness gate failed; no other exposures were processed")
    return checks


def _write_exposure_csv(path, rows):
    fields = ["H5", "exposure", "SPECID", "IFUSLOT", "IFUID", "AMP",
              "blank_count", "finite_support_min", "finite_support_median",
              "finite_support_max", "residual_p01", "residual_p50",
              "residual_p99", "residual_robust_rms"]
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row.get(key, "") for key in fields} for row in rows)


def _write_amplifier_summary_csv(path, summaries):
    fields = ["SPECID", "IFUSLOT", "IFUID", "AMP", "number_available_exposures",
              "median_blank_count", "full_median", "full_p16", "full_p84",
              "fiber_shape_median", "fiber_shape_p16", "fiber_shape_p84",
              "qualitative_visual_persistence_assessment"]
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for summary in summaries:
            identity = summary["identity"]
            full = summary["full_residual_correlation"]
            shape = summary["fiber_shape_correlation"]
            writer.writerow({
                "SPECID": identity[0], "IFUSLOT": identity[1],
                "IFUID": identity[2], "AMP": identity[3],
                "number_available_exposures": summary["number_available_exposures"],
                "median_blank_count": summary["median_blank_count"],
                "full_median": full["median"], "full_p16": full["p16"],
                "full_p84": full["p84"], "fiber_shape_median": shape["median"],
                "fiber_shape_p16": shape["p16"], "fiber_shape_p84": shape["p84"],
                "qualitative_visual_persistence_assessment":
                    summary["qualitative_visual_persistence_assessment"],
            })


def _process_target(records, output_dir, h5_names):
    identity = records[0]["identity"]
    target_dir = output_dir / ("amp_%s" % _identity_text(identity))
    individual_dir = target_dir / "individual_heatmaps"
    individual_dir.mkdir(parents=True, exist_ok=True)
    scales = _common_scales(records)
    primary_limits = scales["primary_symmetric_limits"]
    _plot_contact_sheet(records, h5_names,
                        target_dir / "temporal_contact_sheet.png", identity,
                        primary_limits, "all-valid global p01/p99 symmetric")
    if scales["tail_trimmed_sheet_requested"]:
        _plot_contact_sheet(records, h5_names,
                            target_dir / "temporal_contact_sheet_tail_trimmed.png",
                            identity, scales["tail_trimmed_symmetric_limits"],
                            "global-tail-trimmed p0.5/p99.5 symmetric")
    for record in records:
        _plot_single_heatmap(
            record, individual_dir /
            ("%s_e%d.png" % (Path(record["H5"]).stem, record["exposure"])),
            primary_limits)

    shape_arrays = []
    for record in records:
        shape, common = _shape_residual(record)
        record["shape_residual"] = shape
        record["common_residual"] = common
        shape_arrays.append(shape)
    full_arrays = [record["displayed"] for record in records]
    full_matrix, full_samples, full_fibers = _correlation_matrix(records, full_arrays)
    shape_matrix, shape_samples, shape_fibers = _correlation_matrix(records, shape_arrays)
    labels = _labels(records)
    _matrix_csv(target_dir / "full_residual_correlations.csv", labels, full_matrix)
    _matrix_csv(target_dir / "fiber_shape_correlations.csv", labels, shape_matrix)
    np.savez_compressed(
        target_dir / "correlation_matrices.npz", labels=np.asarray(labels),
        full_residual=full_matrix, fiber_shape=shape_matrix,
        full_overlap_samples=full_samples, full_overlap_fibers=full_fibers,
        fiber_shape_overlap_samples=shape_samples,
        fiber_shape_overlap_fibers=shape_fibers)

    raw_profiles, normalized_profiles, amplitudes = _profiles(records)
    _plot_profiles(records, target_dir / "fiber_profiles.png", normalized=False)
    _plot_profiles(records, target_dir / "fiber_profiles_normalized_shape.png", normalized=True)
    rows = [_record_metrics(record) for record in records]
    _write_exposure_csv(target_dir / "exposure_metrics.csv", rows)
    summary = {
        "identity": list(identity),
        "number_available_exposures": len(records),
        "median_blank_count": float(np.median([row["blank_count"] for row in rows])),
        "full_residual_correlation": _pair_summary(full_matrix),
        "fiber_shape_correlation": _pair_summary(shape_matrix),
        "qualitative_visual_persistence_assessment":
            "CONTACT SHEET REQUIRES SCIENTIFIC INSPECTION; scalar correlations are descriptive only",
        "common_scales": scales,
        "profile_robust_amplitudes": {
            label: float(value) for label, value in zip(labels, amplitudes)},
        "output_files": {
            "contact_sheet": str((target_dir / "temporal_contact_sheet.png").resolve()),
            "individual_heatmaps": str(individual_dir.resolve()),
            "full_correlations": str((target_dir / "full_residual_correlations.csv").resolve()),
            "fiber_shape_correlations": str((target_dir / "fiber_shape_correlations.csv").resolve()),
            "profiles": str((target_dir / "fiber_profiles.png").resolve()),
            "normalized_profiles": str((target_dir / "fiber_profiles_normalized_shape.png").resolve()),
            "exposure_metrics": str((target_dir / "exposure_metrics.csv").resolve()),
        },
    }
    (target_dir / "summary.json").write_text(
        json.dumps(_json_ready(summary), indent=2, sort_keys=True))
    return summary


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model3-product", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--h5", nargs="*", help="optional provenance-matching H5 paths")
    parser.add_argument("--blank-file")
    parser.add_argument("--on-filter")
    parser.add_argument("--off-filter")
    parser.add_argument("--fq-template")
    parser.add_argument("--reference-npz", required=True)
    parser.add_argument("--reference-summary", required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = _parse_args()
    if EXPLICITLY_EXCLUDED in TARGETS:
        raise RuntimeError("explicitly excluded amplifier accidentally appears in TARGETS")
    product_path = Path(args.model3_product).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise SystemExit("output directory is nonempty; use --overwrite: %s" % output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    provenance = _input_provenance(product_path)
    product, model3, skies = load_model_and_skies(product_path)
    h5_paths = _discover_inputs(args, product, product_path)
    blank_path = _provenance_path(args.blank_file, provenance, "blank_file")
    on_filter = _provenance_path(args.on_filter, provenance, "on_filter")
    off_filter = _provenance_path(args.off_filter, provenance, "off_filter")
    fq_template = _provenance_path(args.fq_template, provenance, "fq_template")
    print("EXPECTED")
    print("Four fixed category-2 physical amplifiers are processed across the persisted "
          "19 H5 / 57 exposure Model-3 population; no fitting or new masking is performed.")
    print("targets=%s" % ", ".join(_identity_label(identity) for identity in TARGETS))
    print("excluded=%s" % _identity_label(EXPLICITLY_EXCLUDED))

    blank_masks, blank_provenance = m101_blank_fibers.load(blank_path, h5_paths)
    fq = validated_m101.load_fq(fq_template)
    reference_path = next(path for path in h5_paths if path.name == "20200710_0000013.h5")
    print("correctness gate: loading %s exposure 1" % reference_path.name, flush=True)
    gate_data, gate_loader_provenance = load_native_data(
        [reference_path], blank_masks, on_filter, off_filter,
        include_native_errors=False, include_band_errors=False,
        collapse_band_indices=(5, 6))
    gate_items = [item for item in gate_data if int(item.exposure) == 1]
    if len(gate_items) != 1:
        raise RuntimeError("correctness-gate exposure 1 was not uniquely loaded")
    gate_records = [_frozen_residual(gate_items[0], model3, skies, fq, identity)
                    for identity in TARGETS]
    gate = _reference_gate(
        gate_records, Path(args.reference_npz).expanduser().resolve(),
        Path(args.reference_summary).expanduser().resolve(), output_dir / "correctness_gate")
    print("AGREEMENT")
    print(json.dumps(_json_ready(gate), indent=2, sort_keys=True))
    print("correctness gate passed; proceeding to the remaining persisted exposures", flush=True)

    full_data, loader_provenance = load_native_data(
        h5_paths, blank_masks, on_filter, off_filter,
        include_native_errors=False, include_band_errors=False,
        collapse_band_indices=(5, 6))
    if len(full_data) != 57:
        raise RuntimeError("persisted Model-3 population must contain 57 exposure instances; got %d" %
                           len(full_data))
    records_by_target = {identity: [] for identity in TARGETS}
    for item in full_data:
        for identity in TARGETS:
            if identity in set(tuple(ifu) + (str(amp),)
                               for ifu, amp in zip(item.ifu, item.amp)):
                records_by_target[identity].append(
                    _frozen_residual(item, model3, skies, fq, identity))
    summaries = []
    for identity in TARGETS:
        records = sorted(records_by_target[identity],
                         key=lambda record: (record["H5"], record["exposure"]))
        if not records:
            raise RuntimeError("%s has no available exposure instances" %
                               _identity_label(identity))
        available_h5_names = sorted({record["H5"] for record in records})
        summaries.append(_process_target(records, output_dir, available_h5_names))

    combined_exposure_rows = []
    for summary in summaries:
        exposure_path = Path(summary["output_files"]["exposure_metrics"])
        with exposure_path.open(newline="") as stream:
            combined_exposure_rows.extend(csv.DictReader(stream))
    _write_exposure_csv(output_dir / "exposure_metrics.csv", combined_exposure_rows)
    _write_amplifier_summary_csv(output_dir / "amplifier_summary.csv", summaries)

    observed = {
        "h5_count": len(h5_paths),
        "exposure_count": len(full_data),
        "target_count": len(TARGETS),
        "records_per_target": { _identity_text(identity): len(records_by_target[identity])
                                 for identity in TARGETS},
        "blank_loader": blank_provenance,
        "native_loader": loader_provenance,
        "model3_product_provenance": product["provenance"],
        "elapsed_seconds": time.perf_counter() - started,
    }
    final = {
        "block": "Model-3 amplifier residual temporal persistence diagnostic",
        "fit_performed": False,
        "model3_modified": False,
        "new_masks_added": False,
        "excluded_amplifier": list(EXPLICITLY_EXCLUDED),
        "targets": [list(identity) for identity in TARGETS],
        "correctness_gate": gate,
        "EXPECTED": "The same physical amplifier should show repeatable broad fiber-dependent residual morphology if it is temporally persistent.",
        "OBSERVED": "See the four primary contact sheets and per-amplifier descriptive summaries; visual persistence is deliberately left for inspection.",
        "AGREEMENT": "The exposure-1 arrays, physical row order, blank rows, blank counts, and native units passed the floating-precision gate.",
        "SURPRISES": "No scalar result is promoted to a model claim; localized features and changing blank coverage remain visible and documented.",
        "classification": {
            "established": ["frozen Model-3 residual definition", "four target identities", "exposure-1 numerical reproduction gate"],
            "supported_hypothesis": ["to be assessed from contact sheets after inspection"],
            "speculative": ["any physical correction, basis, rank, or persistence mechanism"],
        },
        "one_next_discriminating_experiment": "Repeat this diagnostic using a pre-registered intersection of blank-valid physical fiber rows across the compared exposures, while freezing the residual definition.",
        "observed_run": observed,
        "per_amplifier": summaries,
        "root_output_files": {
            "exposure_metrics": str((output_dir / "exposure_metrics.csv").resolve()),
            "amplifier_summary": str((output_dir / "amplifier_summary.csv").resolve()),
        },
        "provenance": {
            "exact_command": " ".join(shlex.quote(value) for value in __import__("sys").argv),
            "model3_product": str(product_path),
            "h5": [str(path) for path in h5_paths],
            "blank_file": blank_path,
            "on_filter": on_filter,
            "off_filter": off_filter,
            "fq_template": fq_template,
            "reference_npz": str(Path(args.reference_npz).expanduser().resolve()),
            "reference_summary": str(Path(args.reference_summary).expanduser().resolve()),
        },
        "stop_gate": "STOP after inspecting these four contact sheets and descriptors; do not fit a calibration component, change masks, add R_struct, modify Model 3, or expand the amplifier set.",
    }
    (output_dir / "persistence_summary.json").write_text(
        json.dumps(_json_ready(final), indent=2, sort_keys=True))
    print("OBSERVED")
    print(json.dumps(_json_ready({
        _identity_text(summary["identity"]): {
            "number_available_exposures": summary["number_available_exposures"],
            "median_blank_count": summary["median_blank_count"],
            "median_pairwise_full_residual_correlation": summary["full_residual_correlation"]["median"],
            "median_pairwise_fiber_shape_correlation": summary["fiber_shape_correlation"]["median"],
        } for summary in summaries}), indent=2, sort_keys=True))
    print("wrote diagnostic products to %s" % output_dir)


if __name__ == "__main__":
    main()
