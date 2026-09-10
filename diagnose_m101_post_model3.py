#!/usr/bin/env python3
"""Post-Model-3 M101 blank-residual and source-stitching diagnostics.

This script loads a persisted Model-3 product and reconstructs native spectra
only.  It never calls the Model-0--3 fitting functions, fits an external
image, changes a calibration parameter, or builds a cube.
"""

from __future__ import annotations

import argparse
import ast
import csv
import itertools
import json
from pathlib import Path
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

import diagnose_m101_hierarchical as validated_m101
import build_m101_measurements as validated_measurements
import m101_compact_mask
import m101_external_measurements
from fit_m101_calibration_v2 import FittedModel
from m101_calibration_utils import (
    ALL_BANDS, collapse, json_ready, robust_location, robust_scatter,
)
from m101_native_data import discover_h5, load as load_native_data


NATIVE_BANDS = tuple(ALL_BANDS)
SOURCE_BANDS = ("ON", "OFF")
AMP_ORDER = ("LL", "LU", "RL", "RU")
M101_RA = float(validated_m101.M101_CENTER_RA)
M101_DEC = float(validated_m101.M101_CENTER_DEC)


def _progress(started, message):
    print("[diagnose +%.1fs] %s" % (time.perf_counter() - started, message), flush=True)


def _group_indices(item):
    """Build reusable row groups once per exposure."""
    ifu_groups = {}
    amplifier_groups = {}
    for index, (ifu_value, amp_value) in enumerate(zip(item.ifu, item.amp)):
        ifu = tuple(map(int, ifu_value))
        amp = str(amp_value)
        ifu_groups.setdefault(ifu, []).append(index)
        amplifier_groups.setdefault((ifu, amp), []).append(index)
    return (
        [(key, np.asarray(indices, dtype=int))
         for key, indices in sorted(ifu_groups.items(), key=lambda pair: str(pair[0]))],
        [(key, np.asarray(indices, dtype=int))
         for key, indices in sorted(amplifier_groups.items(), key=lambda pair: str(pair[0]))],
    )


def _parse_map(values):
    return {ast.literal_eval(key): float(value) for key, value in values.items()}


def _extract_json_member(path, member, indentation):
    """Stream one JSON member without loading the large QA product."""
    marker = "%s\"%s\":" % (indentation, member)
    collecting = False
    chunks = []
    depth = 0
    in_string = False
    escaped = False
    with Path(path).open() as stream:
        for line in stream:
            if not collecting:
                if not line.startswith(marker):
                    continue
                line = line[len(marker):].lstrip()
                collecting = True
            chunks.append(line)
            for character in line:
                if in_string:
                    if escaped:
                        escaped = False
                    elif character == "\\":
                        escaped = True
                    elif character == '"':
                        in_string = False
                elif character == '"':
                    in_string = True
                elif character in "[{":
                    depth += 1
                elif character in "]}":
                    depth -= 1
            if collecting and depth == 0:
                return json.JSONDecoder().raw_decode("".join(chunks))[0]
    raise ValueError("JSON member not found: %s" % member)


def load_model_and_skies(product_path, progress=None):
    product_path = Path(product_path).expanduser().resolve()
    if progress is not None:
        progress("reading persisted Model-3 parameter rows from %s" % product_path.name)
    parameter_rows = _extract_json_member(product_path, "parameter_rows", "  ")
    if "MODEL3_Q_ADDITIVE" not in parameter_rows:
        raise ValueError("product has no persisted MODEL3_Q_ADDITIVE solution")
    row = parameter_rows["MODEL3_Q_ADDITIVE"]
    model = FittedModel(
        "model3", None, _parse_map(row["p_IFU_log"]), _parse_map(row["p_AMP_log"]),
        _parse_map(row["ax"]), _parse_map(row["ay"]), _parse_map(row["alpha_q"]),
        int(row.get("clipped_points", 0)), int(row.get("iterations", 6)))
    arrays_path = Path(_extract_json_member(product_path, "arrays", "  "))
    if not arrays_path.is_absolute():
        arrays_path = product_path.parent / arrays_path
    if progress is not None:
        progress("loading persisted sky arrays from %s" % arrays_path.name)
    arrays = np.load(arrays_path)
    skies = {}
    input_h5 = _extract_json_member(product_path, "input_h5", "    ")
    for h5_name in input_h5:
        h5_name = Path(h5_name["filename"]).name
        for exposure in (1, 2, 3):
            key = "sky_%s__%d" % (h5_name, exposure)
            if key not in arrays:
                raise ValueError("final sky array is missing: %s" % key)
            skies[(h5_name, exposure)] = np.asarray(arrays[key], dtype=float).copy()
    if progress is not None:
        progress("persisted Model-3 model and %d sky arrays loaded" % len(skies))
    return {"provenance": {"input_h5": input_h5}}, model, skies


def _finite_pair(left, right):
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    mask = np.isfinite(left) & np.isfinite(right)
    return left[mask], right[mask]


def _correlation(left, right):
    left, right = _finite_pair(left, right)
    if left.size < 3 or np.std(left) <= 0 or np.std(right) <= 0:
        return np.nan
    return float(np.corrcoef(left, right)[0, 1])


def _scalar_projection(target, reference):
    target, reference = _finite_pair(target, reference)
    denominator = float(np.dot(reference, reference)) if reference.size else 0.0
    if denominator <= 0:
        return np.nan, np.nan
    amplitude = float(np.dot(target, reference) / denominator)
    remaining = target - amplitude * reference
    return amplitude, float(validated_m101.robust_scale(remaining))


def _sky_projection(residual, sky):
    residual, sky = _finite_pair(residual, sky)
    denominator = float(np.dot(sky, sky)) if sky.size else 0.0
    if denominator <= 0:
        return np.nan
    return float(np.dot(residual, sky) / denominator)


def _projection_metrics(residual, basis):
    residual, basis = _finite_pair(residual, basis)
    if residual.size < 3:
        return {"coefficient": np.nan, "fraction_variance_explained": np.nan,
                "remaining_robust_rms": np.nan}
    residual_centered = residual - np.median(residual)
    basis_centered = basis - np.median(basis)
    denominator = float(np.dot(basis_centered, basis_centered))
    total = float(np.dot(residual_centered, residual_centered))
    if denominator <= 0 or total <= 0:
        return {"coefficient": np.nan, "fraction_variance_explained": np.nan,
                "remaining_robust_rms": np.nan}
    coefficient = float(np.dot(residual_centered, basis_centered) / denominator)
    remaining = residual_centered - coefficient * basis_centered
    return {"coefficient": coefficient,
            "fraction_variance_explained": float(1. - np.dot(remaining, remaining) / total),
            "remaining_robust_rms": float(validated_m101.robust_scale(remaining))}


def _fractional_spectrum(spectrum, sky):
    finite_sky = np.isfinite(sky)
    if not np.any(finite_sky):
        return np.full(np.asarray(spectrum).shape, np.nan, dtype=float)
    scale = max(1e-12, .01 * float(np.nanmedian(np.abs(sky[finite_sky]))))
    valid = finite_sky & (np.abs(sky) > scale)
    result = np.full(np.asarray(spectrum).shape, np.nan, dtype=float)
    result[valid] = np.asarray(spectrum)[valid] / sky[valid]
    return result


def _robust_spectrum(values):
    """Fast robust column combine for diagnostic hardware spectra."""
    with np.errstate(all="ignore"):
        result = np.nanmedian(np.asarray(values, dtype=float), axis=0)
    return np.asarray(result, dtype=float)


def compute_ifu_residuals(data, model, skies, fq, minimum_blank_fibers,
                          progress=None, source_masks_by_item=None, return_aux=False):
    exposure_rows, amplifier_exposure_rows = [], []
    auxiliary = {"strict_exposure": [], "incident_exposure": []}
    for item_index, item in enumerate(data, 1):
        item_started = time.perf_counter()
        sky = skies[item.key]
        z = model.z_for(item)
        additive = model.additive_full(item, fq)
        residual = (np.asarray(item.total, dtype=float) - additive) / np.exp(z)[:, None]
        residual -= sky[None, :]
        dsky = np.gradient(sky, validated_m101.DEF_WAVE)
        ifu_groups, amplifier_groups = _group_indices(item)

        def append_record(target, identity, indices):
            blank_selected = item.blank_valid[indices]
            n_blank = int(np.sum(blank_selected))
            if n_blank < minimum_blank_fibers:
                return
            spectrum = _robust_spectrum(residual[indices][blank_selected])
            projection = {}
            for name, basis in (("sky", sky), ("K", item.K_wave), ("dsky_dlambda", dsky)):
                projection[name] = _projection_metrics(spectrum, basis)
            target.append({
                "H5": item.h5_name, "exposure": int(item.exposure),
                "SPECID": identity[0], "IFUSLOT": identity[1], "IFUID": identity[2],
                **({"AMP": identity[3]} if len(identity) == 4 else {}),
                "n_blank_fibers": n_blank, "spectrum": spectrum,
                "fractional": _fractional_spectrum(spectrum, sky),
                "sky": sky.copy(), "K": np.asarray(item.K_wave, dtype=float).copy(),
                "dsky_dlambda": dsky.copy(), "projections": projection,
            })
        for ifu, indices in ifu_groups:
            append_record(exposure_rows, ifu, indices)
        for (ifu, amp), indices in amplifier_groups:
            append_record(amplifier_exposure_rows, ifu + (amp,), indices)
        if source_masks_by_item is not None:
            masks = source_masks_by_item.get(item.key, {})
            significant_any = (np.asarray(masks.get("ON", np.zeros(item.row_index.size)), dtype=bool) |
                               np.asarray(masks.get("OFF", np.zeros(item.row_index.size)), dtype=bool))
            strict_blank = item.blank_valid & ~significant_any
            source_templates = {}
            source_counts = {}
            for (ifu, amp), indices in amplifier_groups:
                selected = significant_any[indices]
                source_counts[(ifu, amp)] = int(np.sum(selected))
                source_templates[(ifu, amp)] = (
                    _robust_spectrum(residual[indices][selected]) if np.any(selected)
                    else np.full(residual.shape[1], np.nan))
                strict_selected = strict_blank[indices]
                if int(np.sum(strict_selected)) >= minimum_blank_fibers:
                    auxiliary["strict_exposure"].append({
                        "H5": item.h5_name, "exposure": int(item.exposure),
                        "SPECID": ifu[0], "IFUSLOT": ifu[1], "IFUID": ifu[2], "AMP": amp,
                        "n_strict_blank_fibers": int(np.sum(strict_selected)),
                        "spectrum": _robust_spectrum(residual[indices][strict_selected]),
                    })
            for (ifu, amp), indices in amplifier_groups:
                other_templates = [value for other_key, value in source_templates.items()
                                   if other_key[0] == ifu and other_key[1] != amp and
                                   source_counts.get(other_key, 0) > 0]
                auxiliary["incident_exposure"].append({
                    "H5": item.h5_name, "exposure": int(item.exposure),
                    "SPECID": ifu[0], "IFUSLOT": ifu[1], "IFUID": ifu[2], "AMP": amp,
                    "n_source_fibers_other_amp": int(sum(source_counts.get(other_key, 0)
                                                          for other_key in source_templates
                                                          if other_key[0] == ifu and other_key[1] != amp)),
                    "spectrum": (_robust_spectrum(other_templates) if other_templates else
                                 np.full(residual.shape[1], np.nan)),
                })
        if progress is not None:
            progress("IFU residuals: exposure %d/%d (%s e%d) done in %.3fs; "
                     "%d IFU and %d amplifier groups"
                     % (item_index, len(data), item.h5_name, item.exposure,
                        time.perf_counter() - item_started, len(ifu_groups),
                        len(amplifier_groups)))

    def h5_aggregate(rows):
        h5_rows = []
        grouped = {}
        for row in rows:
            identity = (row["H5"], row["SPECID"], row["IFUSLOT"], row["IFUID"])
            if "AMP" in row:
                identity += (row["AMP"],)
            grouped.setdefault(identity, []).append(row)
        for key, records in sorted(grouped.items(), key=lambda pair: str(pair[0])):
            spectrum = _robust_spectrum([row["spectrum"] for row in records])
            sky_value = _robust_spectrum([row["sky"] for row in records])
            k_value = _robust_spectrum([row["K"] for row in records])
            dsky_value = _robust_spectrum([row["dsky_dlambda"] for row in records])
            projections = {name: _projection_metrics(spectrum, basis)
                           for name, basis in (("sky", sky_value), ("K", k_value),
                                               ("dsky_dlambda", dsky_value))}
            h5_row = {"H5": key[0], "SPECID": key[1], "IFUSLOT": key[2], "IFUID": key[3],
                      "n_supported_exposures": len(records),
                      "supported_exposures": ";".join("e%d" % row["exposure"] for row in records),
                      "spectrum": spectrum, "fractional": _fractional_spectrum(spectrum, sky_value),
                      "sky": sky_value, "K": k_value, "dsky_dlambda": dsky_value,
                      "projections": projections}
            if len(key) == 5:
                h5_row["AMP"] = key[4]
            h5_rows.append(h5_row)
        return h5_rows

    if progress is not None:
        progress("aggregating exposure-level IFU residuals by H5")
    h5_exposure_rows = h5_aggregate(exposure_rows)
    if progress is not None:
        progress("aggregating exposure-level amplifier residuals by H5")
    h5_amplifier_rows = h5_aggregate(amplifier_exposure_rows)
    if return_aux:
        return exposure_rows, h5_exposure_rows, amplifier_exposure_rows, h5_amplifier_rows, auxiliary
    return exposure_rows, h5_exposure_rows, amplifier_exposure_rows, h5_amplifier_rows


def pairwise_ifu_metrics(h5_rows):
    by_ifu = {}
    for row in h5_rows:
        by_ifu.setdefault((row["SPECID"], row["IFUSLOT"], row["IFUID"]), []).append(row)
    output = []
    for ifu, rows in sorted(by_ifu.items(), key=lambda pair: str(pair[0])):
        for left, right in itertools.combinations(sorted(rows, key=lambda row: row["H5"]), 2):
            amplitude, remaining = _scalar_projection(right["spectrum"], left["spectrum"])
            output.append({
                "SPECID": ifu[0], "IFUSLOT": ifu[1], "IFUID": ifu[2],
                "reference_H5": left["H5"], "comparison_H5": right["H5"],
                "spectral_shape_correlation": _correlation(left["spectrum"], right["spectrum"]),
                "best_scalar_amplitude": amplitude,
                "robust_rms_after_scalar": remaining,
                "wavelength_overlap_fraction": float(np.sum(np.isfinite(left["spectrum"]) &
                                                              np.isfinite(right["spectrum"])) /
                                                       left["spectrum"].size),
                "reference_rms": float(validated_m101.robust_scale(left["spectrum"])),
                "comparison_rms": float(validated_m101.robust_scale(right["spectrum"])),
            })
    annotate_spectral_rows(h5_rows)
    return output


def pairwise_amplifier_metrics(amplifier_h5_rows):
    by_amplifier = {}
    for row in amplifier_h5_rows:
        key = (row["SPECID"], row["IFUSLOT"], row["IFUID"], row["AMP"])
        by_amplifier.setdefault(key, []).append(row)
    output = []
    for amplifier, rows in sorted(by_amplifier.items(), key=lambda pair: str(pair[0])):
        for left, right in itertools.combinations(sorted(rows, key=lambda row: row["H5"]), 2):
            amplitude, remaining = _scalar_projection(right["spectrum"], left["spectrum"])
            finite = np.isfinite(left["spectrum"]) & np.isfinite(right["spectrum"])
            output.append({
                "SPECID": amplifier[0], "IFUSLOT": amplifier[1],
                "IFUID": amplifier[2], "AMP": amplifier[3],
                "reference_H5": left["H5"], "comparison_H5": right["H5"],
                "spectral_shape_correlation": _correlation(left["spectrum"], right["spectrum"]),
                "best_scalar_amplitude": amplitude,
                "robust_rms_after_scalar": remaining,
                "wavelength_overlap_fraction": float(np.sum(finite) / left["spectrum"].size),
                "reference_rms": float(validated_m101.robust_scale(left["spectrum"])),
                "comparison_rms": float(validated_m101.robust_scale(right["spectrum"])),
            })
    annotate_spectral_rows(amplifier_h5_rows)
    return output


def within_h5_amplifier_metrics(amplifier_exposure_rows):
    by_amplifier = {}
    for row in amplifier_exposure_rows:
        key = (row["H5"], row["SPECID"], row["IFUSLOT"], row["IFUID"], row["AMP"])
        by_amplifier.setdefault(key, []).append(row)
    output = []
    for amplifier, rows in sorted(by_amplifier.items(), key=lambda pair: str(pair[0])):
        for left, right in itertools.combinations(sorted(rows, key=lambda row: row["exposure"]), 2):
            amplitude, remaining = _scalar_projection(right["spectrum"], left["spectrum"])
            output.append({
                "H5": amplifier[0], "SPECID": amplifier[1], "IFUSLOT": amplifier[2],
                "IFUID": amplifier[3], "AMP": amplifier[4],
                "reference_exposure": left["exposure"], "comparison_exposure": right["exposure"],
                "spectral_shape_correlation": _correlation(left["spectrum"], right["spectrum"]),
                "best_scalar_amplitude": amplitude, "robust_rms_after_scalar": remaining,
            })
    return output


def select_representative_amplifiers(amplifier_h5_rows, pair_rows, count=16):
    by_amplifier = {}
    for row in amplifier_h5_rows:
        key = (row["SPECID"], row["IFUSLOT"], row["IFUID"], row["AMP"])
        by_amplifier.setdefault(key, []).append(row)
    pair_by_amplifier = {}
    for row in pair_rows:
        key = (row["SPECID"], row["IFUSLOT"], row["IFUID"], row["AMP"])
        pair_by_amplifier.setdefault(key, []).append(row)
    features = []
    for key, rows in by_amplifier.items():
        if len(rows) < 1:
            continue
        rms = [row["spectral_rms"] for row in rows]
        blue = [np.nanmedian(np.abs(row["spectrum"][:200])) for row in rows]
        correlations = [row["spectral_shape_correlation"]
                        for row in pair_by_amplifier.get(key, [])]
        features.append({
            "amplifier": key, "rows": rows,
            "rms": float(np.nanmedian(rms)),
            "blue": float(np.nanmedian(blue)),
            "high_correlation": float(np.nanmax(correlations)) if correlations else np.nan,
            "low_correlation": float(np.nanmin(correlations)) if correlations else np.nan,
        })
    if not features:
        return [], {}
    median_rms = float(np.nanmedian([row["rms"] for row in features]))
    selected, reasons = [], {}

    def add(label, ordered):
        for candidate in ordered:
            if candidate["amplifier"] not in reasons and len(selected) < count:
                selected.append(candidate)
                reasons[candidate["amplifier"]] = label
                return

    add("high spectral RMS", sorted(features, key=lambda row: (-row["rms"], row["amplifier"])))
    add("moderate spectral RMS", sorted(features, key=lambda row: (abs(row["rms"] - median_rms), row["amplifier"])))
    add("low spectral RMS", sorted(features, key=lambda row: (row["rms"], row["amplifier"])))
    add("strong blue structure", sorted(features, key=lambda row: (-row["blue"], row["amplifier"])))
    add("high cross-H5 correlation", sorted(features, key=lambda row: (-np.nan_to_num(row["high_correlation"], nan=-np.inf), row["amplifier"])))
    add("low cross-H5 correlation", sorted(features, key=lambda row: (np.nan_to_num(row["low_correlation"], nan=np.inf), row["amplifier"])))
    for amp in AMP_ORDER:
        add("%s example" % amp, sorted(
            (row for row in features if row["amplifier"][3] == amp),
            key=lambda row: (-row["rms"], row["amplifier"])))
    for candidate in sorted(features, key=lambda row: (-row["rms"], row["amplifier"])):
        if len(selected) >= count:
            break
        if candidate["amplifier"] not in reasons:
            selected.append(candidate)
            reasons[candidate["amplifier"]] = "representative fill"
    return selected, reasons


def annotate_spectral_rows(rows):
    for row in rows:
        row["spectral_rms"] = float(validated_m101.robust_scale(row["spectrum"]))
        row["sky_correlation"] = _correlation(row["spectrum"], row["sky"])
        row["sky_projection"] = _sky_projection(row["spectrum"], row["sky"])
        for name, basis in (("sky", row["sky"]), ("K", row["K"]),
                            ("dsky_dlambda", row["dsky_dlambda"])):
            for metric, value in row["projections"][name].items():
                row["%s_%s" % (name, metric)] = value


def select_representative_ifus(h5_rows, pair_rows, count=16):
    by_ifu = {}
    for row in h5_rows:
        by_ifu.setdefault((row["SPECID"], row["IFUSLOT"], row["IFUID"]), []).append(row)
    pair_by_ifu = {}
    for row in pair_rows:
        key = (row["SPECID"], row["IFUSLOT"], row["IFUID"])
        pair_by_ifu.setdefault(key, []).append(row)
    features = []
    for key, rows in by_ifu.items():
        if len(rows) < 2:
            continue
        rms = np.asarray([row["spectral_rms"] for row in rows], dtype=float)
        blue = np.asarray([np.nanmedian(np.abs(row["spectrum"][:200])) for row in rows])
        changes = [row["robust_rms_after_scalar"] for row in pair_by_ifu.get(key, [])]
        correlations = [row["spectral_shape_correlation"] for row in pair_by_ifu.get(key, [])]
        features.append({
            "ifu": key, "rows": rows, "rms": float(np.nanmedian(rms)),
            "peak": float(np.nanmax([np.nanmax(np.abs(row["spectrum"])) for row in rows])),
            "blue": float(np.nanmedian(blue)),
            "date_change": float(np.nanmedian(changes)) if changes else np.nan,
            "repeatability": float(np.nanmax(correlations)) if correlations else np.nan,
        })
    if not features:
        return [], []
    median_rms = float(np.nanmedian([row["rms"] for row in features]))
    selected, reasons = [], {}

    def add(label, ordered):
        for candidate in ordered:
            if candidate["ifu"] not in reasons and len(selected) < count:
                selected.append(candidate)
                reasons[candidate["ifu"]] = label
                return

    add("large residual structure", sorted(features, key=lambda row: (-row["peak"], row["ifu"])))
    add("strong blue structure", sorted(features, key=lambda row: (-row["blue"], row["ifu"])))
    add("largest date change", sorted(features, key=lambda row: (-np.nan_to_num(row["date_change"], nan=-np.inf), row["ifu"])))
    add("repeatable spectral shape", sorted(features, key=lambda row: (-np.nan_to_num(row["repeatability"], nan=-np.inf), row["ifu"])))
    add("quiet comparison", sorted(features, key=lambda row: (row["rms"], row["ifu"])))
    add("moderate comparison", sorted(features, key=lambda row: (abs(row["rms"] - median_rms), row["ifu"])))
    for candidate in sorted(features, key=lambda row: (-row["rms"], row["ifu"])):
        if len(selected) >= count:
            break
        if candidate["ifu"] not in reasons:
            selected.append(candidate)
            reasons[candidate["ifu"]] = "representative fill"
    return selected, reasons


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


def write_ifu_products(output_dir, wave, exposure_rows, h5_rows,
                       amplifier_exposure_rows, amplifier_h5_rows, pair_rows,
                       selected, reasons):
    output_dir = Path(output_dir)
    projection_fields = [
        "%s_%s" % (name, metric)
        for name in ("sky", "K", "dsky_dlambda")
        for metric in ("coefficient", "fraction_variance_explained", "remaining_robust_rms")]
    for row in exposure_rows + amplifier_exposure_rows:
        for name in ("sky", "K", "dsky_dlambda"):
            for metric, value in row["projections"][name].items():
                row["%s_%s" % (name, metric)] = value
    metadata_fields = ["H5", "exposure", "SPECID", "IFUSLOT", "IFUID", "n_blank_fibers"] + projection_fields
    _write_rows(output_dir / "m101_ifu_residual_exposures.csv", exposure_rows, metadata_fields)
    amplifier_fields = metadata_fields + ["AMP"]
    _write_rows(output_dir / "m101_amplifier_residual_exposures.csv",
                amplifier_exposure_rows, amplifier_fields)
    h5_fields = ["H5", "SPECID", "IFUSLOT", "IFUID", "n_supported_exposures",
                 "supported_exposures", "spectral_rms", "sky_correlation", "sky_projection",
                 "sky_coefficient", "sky_fraction_variance_explained", "sky_remaining_robust_rms",
                 "K_coefficient", "K_fraction_variance_explained", "K_remaining_robust_rms",
                 "dsky_dlambda_coefficient", "dsky_dlambda_fraction_variance_explained",
                 "dsky_dlambda_remaining_robust_rms"]
    annotate_spectral_rows(h5_rows)
    _write_rows(output_dir / "m101_ifu_residual_h5.csv", h5_rows, h5_fields)
    amplifier_h5_fields = [field for field in h5_fields if field != "supported_exposures"] + ["AMP", "supported_exposures"]
    _write_rows(output_dir / "m101_amplifier_residual_h5.csv",
                amplifier_h5_rows, amplifier_h5_fields)
    pair_fields = ["SPECID", "IFUSLOT", "IFUID", "reference_H5", "comparison_H5",
                   "spectral_shape_correlation", "best_scalar_amplitude",
                   "robust_rms_after_scalar", "wavelength_overlap_fraction",
                   "reference_rms", "comparison_rms"]
    _write_rows(output_dir / "m101_ifu_residual_pairs.csv", pair_rows, pair_fields)
    np.savez_compressed(
        output_dir / "m101_ifu_residual_spectra.npz", wavelength=wave,
        exposure_spectra=np.asarray([row["spectrum"] for row in exposure_rows]),
        exposure_fractional=np.asarray([row["fractional"] for row in exposure_rows]),
        amplifier_exposure_spectra=np.asarray([row["spectrum"] for row in amplifier_exposure_rows]),
        amplifier_exposure_fractional=np.asarray([row["fractional"] for row in amplifier_exposure_rows]),
        h5_spectra=np.asarray([row["spectrum"] for row in h5_rows]),
        h5_fractional=np.asarray([row["fractional"] for row in h5_rows]),
        amplifier_h5_spectra=np.asarray([row["spectrum"] for row in amplifier_h5_rows]),
        amplifier_h5_fractional=np.asarray([row["fractional"] for row in amplifier_h5_rows]),
        exposure_h5=np.asarray([row["H5"] for row in exposure_rows], dtype="U32"),
        exposure_number=np.asarray([row["exposure"] for row in exposure_rows], dtype=int),
        exposure_ifu=np.asarray([[row["SPECID"], row["IFUSLOT"], row["IFUID"]]
                                 for row in exposure_rows], dtype=int),
        amplifier_exposure_h5=np.asarray([row["H5"] for row in amplifier_exposure_rows], dtype="U32"),
        amplifier_exposure_number=np.asarray([row["exposure"] for row in amplifier_exposure_rows], dtype=int),
        amplifier_exposure_hardware=np.asarray([[row["SPECID"], row["IFUSLOT"], row["IFUID"]]
                                                for row in amplifier_exposure_rows], dtype=int),
        amplifier_exposure_amp=np.asarray([row["AMP"] for row in amplifier_exposure_rows], dtype="U2"),
        h5_names=np.asarray([row["H5"] for row in h5_rows], dtype="U32"),
        h5_ifu=np.asarray([[row["SPECID"], row["IFUSLOT"], row["IFUID"]]
                           for row in h5_rows], dtype=int),
        amplifier_h5_names=np.asarray([row["H5"] for row in amplifier_h5_rows], dtype="U32"),
        amplifier_h5_hardware=np.asarray([[row["SPECID"], row["IFUSLOT"], row["IFUID"]]
                                          for row in amplifier_h5_rows], dtype=int),
        amplifier_h5_amp=np.asarray([row["AMP"] for row in amplifier_h5_rows], dtype="U2"))
    selection = [{"SPECID": key[0], "IFUSLOT": key[1], "IFUID": key[2],
                  "reason": reasons[key]} for key in reasons]
    (output_dir / "m101_ifu_residual_selection.json").write_text(
        json.dumps(json_ready({"selected": selection}), indent=2, sort_keys=True))


def write_amplifier_products(output_dir, wave, amplifier_exposure_rows,
                             amplifier_h5_rows, pair_rows):
    output_dir = Path(output_dir)
    projection_fields = [
        "%s_%s" % (name, metric)
        for name in ("sky", "K", "dsky_dlambda")
        for metric in ("coefficient", "fraction_variance_explained", "remaining_robust_rms")]
    for row in amplifier_exposure_rows:
        for name in ("sky", "K", "dsky_dlambda"):
            for metric, value in row["projections"][name].items():
                row["%s_%s" % (name, metric)] = value
    exposure_fields = ["H5", "exposure", "SPECID", "IFUSLOT", "IFUID", "n_blank_fibers"] + projection_fields + ["AMP"]
    _write_rows(output_dir / "m101_amplifier_residual_exposures.csv",
                amplifier_exposure_rows, exposure_fields)
    annotate_spectral_rows(amplifier_h5_rows)
    h5_fields = ["H5", "SPECID", "IFUSLOT", "IFUID", "AMP", "n_supported_exposures",
                 "supported_exposures", "spectral_rms", "sky_correlation", "sky_projection",
                 "sky_coefficient", "sky_fraction_variance_explained", "sky_remaining_robust_rms",
                 "K_coefficient", "K_fraction_variance_explained", "K_remaining_robust_rms",
                 "dsky_dlambda_coefficient", "dsky_dlambda_fraction_variance_explained",
                 "dsky_dlambda_remaining_robust_rms"]
    _write_rows(output_dir / "m101_amplifier_residual_h5.csv", amplifier_h5_rows, h5_fields)
    pair_fields = ["SPECID", "IFUSLOT", "IFUID", "AMP", "reference_H5", "comparison_H5",
                   "spectral_shape_correlation", "best_scalar_amplitude",
                   "robust_rms_after_scalar", "wavelength_overlap_fraction",
                   "reference_rms", "comparison_rms"]
    _write_rows(output_dir / "m101_amplifier_residual_pairs.csv", pair_rows, pair_fields)
    np.savez_compressed(
        output_dir / "m101_amplifier_residual_spectra.npz", wavelength=wave,
        exposure_spectra=np.asarray([row["spectrum"] for row in amplifier_exposure_rows]),
        exposure_fractional=np.asarray([row["fractional"] for row in amplifier_exposure_rows]),
        h5_spectra=np.asarray([row["spectrum"] for row in amplifier_h5_rows]),
        h5_fractional=np.asarray([row["fractional"] for row in amplifier_h5_rows]),
        exposure_h5=np.asarray([row["H5"] for row in amplifier_exposure_rows], dtype="U32"),
        exposure_number=np.asarray([row["exposure"] for row in amplifier_exposure_rows], dtype=int),
        exposure_hardware=np.asarray([[row["SPECID"], row["IFUSLOT"], row["IFUID"]]
                                      for row in amplifier_exposure_rows], dtype=int),
        exposure_amp=np.asarray([row["AMP"] for row in amplifier_exposure_rows], dtype="U2"),
        h5_names=np.asarray([row["H5"] for row in amplifier_h5_rows], dtype="U32"),
        h5_hardware=np.asarray([[row["SPECID"], row["IFUSLOT"], row["IFUID"]]
                                for row in amplifier_h5_rows], dtype=int),
        h5_amp=np.asarray([row["AMP"] for row in amplifier_h5_rows], dtype="U2"))


def plot_ifu_panels(output_dir, wave, exposure_rows, h5_rows, selected, reasons):
    h5_names = sorted({row["H5"] for row in h5_rows})
    colors = {name: plt.cm.tab20(i % 20) for i, name in enumerate(h5_names)}
    handles = [Line2D([], [], color=colors[name], lw=1.5, label=Path(name).stem)
               for name in h5_names]
    for fractional, suffix, ylabel in ((False, "", "native residual"),
                                       (True, "_fractional", "residual / sky") ):
        fig, axes = plt.subplots(4, 4, figsize=(15, 13), squeeze=False)
        for axis, candidate in zip(axes.flat, selected):
            key = candidate["ifu"]
            for row in exposure_rows:
                if (row["SPECID"], row["IFUSLOT"], row["IFUID"]) != key:
                    continue
                values = row["fractional"] if fractional else row["spectrum"]
                axis.plot(wave, values, color=colors[row["H5"]], alpha=.18, lw=.45)
            for row in h5_rows:
                if (row["SPECID"], row["IFUSLOT"], row["IFUID"]) != key:
                    continue
                values = row["fractional"] if fractional else row["spectrum"]
                axis.plot(wave, values, color=colors[row["H5"]], lw=1.1)
            axis.axhline(0, color="0.25", lw=.6)
            axis.set_title("%d / %d / %d\n%s; %d H5; %s" %
                           (key[0], key[1], key[2], reasons[key],
                            len(candidate["rows"]), candidate["ifu"]))
            axis.set_xlabel("wavelength")
            axis.set_ylabel(ylabel)
            axis.grid(alpha=.15)
        for axis in axes.flat[len(selected):]:
            axis.set_visible(False)
        fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=6,
                   bbox_to_anchor=(.5, .005))
        fig.tight_layout(rect=(0, .04, 1, 1))
        fig.savefig(output_dir / ("m101_ifu_residual_panels%s.png" % suffix), dpi=140)
        plt.close(fig)


def plot_amplifier_panels(output_dir, wave, exposure_rows, h5_rows):
    """Show multiple H5 residual spectra for representative physical amplifiers."""
    if not h5_rows:
        return
    pair_rows = pairwise_amplifier_metrics(h5_rows)
    selected, reasons = select_representative_amplifiers(h5_rows, pair_rows)
    h5_names = sorted({row["H5"] for row in h5_rows})
    colors = {name: plt.cm.tab20(i % 20) for i, name in enumerate(h5_names)}
    for fractional, suffix, ylabel in ((False, "", "native residual"),
                                       (True, "_fractional", "residual / sky")):
        fig, axes = plt.subplots(4, 4, figsize=(15, 13), squeeze=False)
        for axis, candidate in zip(axes.flat, selected):
            key = candidate["amplifier"]
            for row in exposure_rows:
                if (row["SPECID"], row["IFUSLOT"], row["IFUID"], row["AMP"]) != key:
                    continue
                values = row["fractional"] if fractional else row["spectrum"]
                axis.plot(wave, values, color=colors[row["H5"]], alpha=.16, lw=.4)
            for row in h5_rows:
                if (row["SPECID"], row["IFUSLOT"], row["IFUID"], row["AMP"]) != key:
                    continue
                values = row["fractional"] if fractional else row["spectrum"]
                axis.plot(wave, values, color=colors[row["H5"]], lw=1.0,
                          label=Path(row["H5"]).stem)
            axis.axhline(0, color="0.25", lw=.6)
            axis.set_title("%d / %d / %d / %s\n%s; %s" %
                           (key[0], key[1], key[2], key[3], reasons[key],
                            ", ".join("%s" % row["H5"] for row in candidate["rows"])), fontsize=7)
            axis.set_xlabel("wavelength"); axis.set_ylabel(ylabel); axis.grid(alpha=.15)
        for axis in axes.flat[len(selected):]:
            axis.set_visible(False)
        handles = [Line2D([], [], color=colors[name], lw=1.2, label=Path(name).stem)
                   for name in h5_names]
        fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=6,
                   bbox_to_anchor=(.5, .005))
        fig.tight_layout(rect=(0, .04, 1, 1))
        fig.savefig(output_dir / ("m101_amplifier_residual_panels%s.png" % suffix), dpi=140)
        plt.close(fig)


def plot_null_band_maps(output_dir, data, model, skies, fq, wave, progress=None):
    """Reproduce the five null-band focal-plane residual maps from Model 3."""
    fig, axes = plt.subplots(1, 5, figsize=(21, 4.5), squeeze=False)
    null_responses, _ = validated_measurements._validate_sky_null_responses()
    x_by_band = [[] for _ in NATIVE_BANDS[:5]]
    y_by_band = [[] for _ in NATIVE_BANDS[:5]]
    residual_by_band = [[] for _ in NATIVE_BANDS[:5]]
    for item_index, item in enumerate(data, 1):
        item_started = time.perf_counter()
        z = model.z_for(item)
        additive = model.additive_full(item, fq)
        native_residual = ((item.total - additive) /
                           np.exp(z)[:, None] - skies[item.key][None, :])
        for band_index, response in enumerate(null_responses):
            collapsed, _ = collapse(native_residual, response)
            selected = item.blank_valid & np.isfinite(collapsed)
            x_by_band[band_index].append(item.x_arcmin[selected])
            y_by_band[band_index].append(item.y_arcmin[selected])
            residual_by_band[band_index].append(collapsed[selected])
        if progress is not None:
            progress("null-band maps: prepared residuals for exposure %d/%d in %.3fs"
                     % (item_index, len(data), time.perf_counter() - item_started))
    for band_index, band in enumerate(NATIVE_BANDS[:5]):
        x_values = np.concatenate(x_by_band[band_index])
        y_values = np.concatenate(y_by_band[band_index])
        residual_values = np.concatenate(residual_by_band[band_index])
        axis = axes[0, band_index]
        limit = float(np.nanpercentile(np.abs(residual_values), 99)) if residual_values.size else 1.
        limit = max(limit, 1e-12)
        image = axis.hexbin(x_values, y_values, C=residual_values, gridsize=100,
                            reduce_C_function=np.nanmedian, cmap="coolwarm",
                            vmin=-limit, vmax=limit, mincnt=1)
        axis.set_title(band); axis.set_xlabel("x [arcmin]"); axis.set_ylabel("y [arcmin]")
        axis.set_aspect("equal", adjustable="box"); axis.grid(alpha=.12)
        fig.colorbar(image, ax=axis, label="Model-3 blank residual")
        if progress is not None:
            progress("null-band maps: rendered %s (%d/5)" % (band, band_index + 1))
    fig.tight_layout(); fig.savefig(output_dir / "m101_model3_null_band_focal_maps.png", dpi=140); plt.close(fig)


def _radius_arcmin(ra, dec):
    dra = (np.asarray(ra) - M101_RA) * np.cos(np.deg2rad(M101_DEC)) * 60.
    ddec = (np.asarray(dec) - M101_DEC) * 60.
    return np.hypot(dra, ddec)


def build_source_rows(data, external, mask_data, mask_wcs, model, skies, fq,
                      responses, source_sigma, minimum_source_fibers,
                      progress=None, return_source_masks=False):
    rows, fiber_ratios = [], {band: [] for band in SOURCE_BANDS}
    by_exposure = {}
    source_masks_by_item = {}
    band_indices = {band: NATIVE_BANDS.index(band) for band in SOURCE_BANDS}
    for item_index, item in enumerate(data, 1):
        item_started = time.perf_counter()
        cache = external[item.h5_name]
        z = model.z_for(item)
        additive = model.additive_bands(item, fq)
        inverse_offset = np.exp(z)
        sky_bands = np.asarray([
            collapse(skies[item.key][None, :], response)[0][0]
            for response in responses], dtype=float)
        object_values = {
            band: float(cache["global_g"][band]) *
            np.asarray(cache["external_object"][(item.exposure, band)], dtype=float)
            for band in SOURCE_BANDS}
        positions = {band: m101_compact_mask.mask_radec(
            mask_data, mask_wcs, item.ra, item.dec) for band in SOURCE_BANDS}
        radius = _radius_arcmin(item.ra, item.dec)
        virus_by_band = {
            band: ((item.band_total[:, band_indices[band]] -
                    additive[:, band_indices[band]]) / inverse_offset -
                   sky_bands[band_indices[band]])
            for band in SOURCE_BANDS}
        source_masks = {}
        source_values = {}
        for band_index, band in enumerate(SOURCE_BANDS):
            external_valid = np.asarray(cache["external_valid"][(item.exposure, band)], dtype=bool)
            masked, inside = positions[band]
            comparison = m101_external_measurements.source_comparison_validity(
                external_valid, masked, inside, ~item.hardware_bad,
                np.isfinite(object_values[band]), np.ones(object_values[band].shape))
            reference = comparison & (radius > 6.)
            if int(reference.sum()) < 20:
                valid_indices = np.flatnonzero(comparison)
                keep = valid_indices[np.argsort(object_values[band][valid_indices])[:max(20, valid_indices.size // 4)]]
                reference = np.zeros(comparison.shape, dtype=bool)
                reference[keep] = True
                reference_method = "lowest external-X quartile fallback"
            else:
                reference_method = "external-X valid fibers at radius > 6 arcmin"
            baseline = robust_location(object_values[band][reference]) if np.any(reference) else np.nan
            scale = robust_scatter(object_values[band][reference]) if np.any(reference) else np.nan
            if not np.isfinite(scale) or scale <= 0:
                scale = float(np.nanstd(object_values[band][reference])) if np.any(reference) else np.nan
            scale = max(scale, 1e-12) if np.isfinite(scale) else np.nan
            threshold = baseline + source_sigma * scale if np.isfinite(baseline) and np.isfinite(scale) else np.nan
            selected = comparison & (object_values[band] > 0) & (object_values[band] > threshold)
            source_masks[band] = selected
            source_values[band] = object_values[band]
            by_exposure.setdefault((item.h5_name, item.exposure, band), {
                "threshold": threshold, "baseline": baseline, "scale": scale,
                "reference_method": reference_method, "N_significant_source_fibers": 0,
                "N_supported_amplifiers": 0})
            by_exposure[(item.h5_name, item.exposure, band)]["N_significant_source_fibers"] += int(selected.sum())
        source_masks_by_item[item.key] = {
            band: np.asarray(source_masks[band], dtype=bool).copy()
            for band in SOURCE_BANDS}

        _, amplifier_groups = _group_indices(item)
        for (ifu, amp), indices in amplifier_groups:
            for band in SOURCE_BANDS:
                selected = source_masks[band][indices]
                virus = virus_by_band[band][indices]
                ratios = np.asarray(virus / source_values[band][indices], dtype=float)
                valid = selected & np.isfinite(ratios)
                fiber_ratios[band].extend(ratios[valid].tolist())
                if int(valid.sum()) < minimum_source_fibers:
                    continue
                ratio_values = ratios[valid]
                raw = robust_location(ratio_values)
                scatter = robust_scatter(ratio_values)
                x_values = source_values[band][indices][valid]
                y_values = virus[valid]
                x_norm = float(np.dot(x_values, x_values))
                slope = float(np.dot(x_values, y_values) / x_norm) if x_norm > 0 else np.nan
                slope_scatter = (float(validated_m101.robust_scale(y_values - slope * x_values))
                                 if np.isfinite(slope) else np.nan)
                key = (item.h5_name, item.exposure, ifu[0], ifu[1], ifu[2], amp, band)
                rows.append({
                    "H5": item.h5_name, "exposure": int(item.exposure),
                    "SPECID": ifu[0], "IFUSLOT": ifu[1], "IFUID": ifu[2], "AMP": amp,
                    "band": band, "N_source_measurements": int(valid.sum()),
                    "raw_robust_ratio": raw, "normalized_ratio": np.nan,
                    "zero_intercept_slope": slope,
                    "slope_residual_robust_rms": slope_scatter,
                    "slope_minus_ratio": slope - raw if np.isfinite(slope) and np.isfinite(raw) else np.nan,
                    "robust_scatter": scatter,
                    "ratio_uncertainty": scatter / np.sqrt(valid.sum()) if np.isfinite(scatter) else np.nan,
                    "x_arcmin": float(np.nanmean(item.x_arcmin[indices][valid])),
                    "y_arcmin": float(np.nanmean(item.y_arcmin[indices][valid])),
                    "source_threshold": by_exposure[(item.h5_name, item.exposure, band)]["threshold"],
                    "source_reference_method": by_exposure[(item.h5_name, item.exposure, band)]["reference_method"],
                    "_virus_values": np.asarray(y_values, dtype=float).copy(),
                    "_source_values": np.asarray(x_values, dtype=float).copy(),
                    "_key": key,
                })
                by_exposure[(item.h5_name, item.exposure, band)]["N_supported_amplifiers"] += 1
        if progress is not None:
            progress("source stitching: exposure %d/%d (%s e%d) done in %.3fs; "
                     "%d amplifier groups"
                     % (item_index, len(data), item.h5_name, item.exposure,
                        time.perf_counter() - item_started, len(amplifier_groups)))
    centers = {band: robust_location(np.asarray(fiber_ratios[band], dtype=float))
               for band in SOURCE_BANDS}
    for row in rows:
        row["raw_center"] = centers[row["band"]]
        row["normalized_ratio"] = row["raw_robust_ratio"] / centers[row["band"]]
    if return_source_masks:
        return rows, centers, by_exposure, fiber_ratios, source_masks_by_item
    return rows, centers, by_exposure, fiber_ratios


def _source_row_groups(rows):
    """Index source rows once for the exposure, H5, and hardware products."""
    groups = {
        "rows_by_h5_exposure_band": {},
        "rows_by_h5_band": {},
        "rows_by_physical_ifu_band": {},
        "rows_by_h5_physical_ifu_band": {},
        "rows_by_h5_physical_amp": {},
        "rows_by_physical_amp_band": {},
    }
    for row in rows:
        h5 = row["H5"]
        exposure = row["exposure"]
        ifu = (row["SPECID"], row["IFUSLOT"], row["IFUID"])
        amp = row["AMP"]
        band = row["band"]
        groups["rows_by_h5_exposure_band"].setdefault((h5, exposure, band), []).append(row)
        groups["rows_by_h5_band"].setdefault((h5, band), []).append(row)
        groups["rows_by_physical_ifu_band"].setdefault(ifu + (band,), []).append(row)
        groups["rows_by_h5_physical_ifu_band"].setdefault((h5,) + ifu + (band,), []).append(row)
        groups["rows_by_h5_physical_amp"].setdefault((h5,) + ifu + (amp,), []).append(row)
        groups.setdefault("rows_by_h5_physical_amp_band", {}).setdefault((h5,) + ifu + (amp, band), []).append(row)
        groups["rows_by_physical_amp_band"].setdefault(ifu + (amp, band), []).append(row)
    return groups


def _finite_values(values):
    values = np.asarray(values, dtype=float)
    return values[np.isfinite(values)]


def _tiny_median(values):
    values = _finite_values(values)
    return float(np.nanmedian(values)) if values.size else np.nan


def _safe_ratio(numerator, denominator):
    return (float(numerator / denominator)
            if np.isfinite(numerator) and np.isfinite(denominator) and denominator != 0 else np.nan)


def compute_exposure_gray_diagnostics(data, source_rows):
    """Estimate relative exposure gray factors from equal-weight amplifier rows."""
    grouped = _source_row_groups(source_rows)["rows_by_h5_exposure_band"]
    records = []
    for item in data:
        key = (item.h5_name, int(item.exposure))
        survey = item.survey
        band_values = {}
        for band in SOURCE_BANDS:
            values = np.asarray([row["raw_robust_ratio"] for row in grouped.get(key + (band,), [])], dtype=float)
            values = _finite_values(values)
            scatter = robust_scatter(values) if values.size >= 3 else np.nan
            center = robust_location(values) if values.size else np.nan
            uncertainty = scatter / np.sqrt(values.size) if np.isfinite(scatter) and values.size else np.nan
            source_count = int(sum(row["N_source_measurements"] for row in grouped.get(key + (band,), [])))
            slope_delta = np.asarray([
                row["slope_minus_ratio"] for row in grouped.get(key + (band,), [])], dtype=float)
            band_values[band] = {
                "G": center, "N_supported_amplifiers": int(values.size),
                "N_source_measurements": source_count,
                "robust_amplifier_scatter": scatter,
                "uncertainty": uncertainty,
                "median_slope_minus_ratio": _tiny_median(slope_delta),
            }
        on, off = band_values["ON"], band_values["OFF"]
        weights = np.asarray([
            1. / on["uncertainty"] ** 2 if np.isfinite(on["uncertainty"]) and on["uncertainty"] > 0 else 0.,
            1. / off["uncertainty"] ** 2 if np.isfinite(off["uncertainty"]) and off["uncertainty"] > 0 else 0.], dtype=float)
        centers = np.asarray([on["G"], off["G"]], dtype=float)
        good = np.isfinite(centers) & (weights > 0)
        if np.any(good):
            joint = float(np.sum(centers[good] * weights[good]) / np.sum(weights[good]))
            joint_uncertainty = float(np.sqrt(1. / np.sum(weights[good])))
        else:
            joint = np.nan
            joint_uncertainty = np.nan
        records.append({
            "H5": item.h5_name, "exposure": int(item.exposure),
            "millum": float(survey["millum"]), "throughput": float(survey["throughput"]),
            "qr_guider_ratio": float(survey["millum"] * survey["throughput"] / 5e5),
            "offset": float(survey["offset"]), "exptime": float(survey["exptime"]),
            "nstarsphotom": int(survey["nstarsphotom"]), "nstarsastrom": int(survey["nstarsastrom"]),
            "G_ON": on["G"], "G_OFF": off["G"], "G_joint": joint,
            "G_joint_uncertainty": joint_uncertainty,
            "G_OFF_over_G_ON": _safe_ratio(off["G"], on["G"]),
            "N_supported_amplifiers_ON": on["N_supported_amplifiers"],
            "N_supported_amplifiers_OFF": off["N_supported_amplifiers"],
            "N_source_measurements_ON": on["N_source_measurements"],
            "N_source_measurements_OFF": off["N_source_measurements"],
            "robust_amplifier_scatter_ON": on["robust_amplifier_scatter"],
            "robust_amplifier_scatter_OFF": off["robust_amplifier_scatter"],
            "uncertainty_ON": on["uncertainty"], "uncertainty_OFF": off["uncertainty"],
            "median_slope_minus_ratio_ON": on["median_slope_minus_ratio"],
            "median_slope_minus_ratio_OFF": off["median_slope_minus_ratio"],
        })
    finite_uncertainties = np.asarray([row["G_joint_uncertainty"] for row in records], dtype=float)
    finite_uncertainties = _finite_values(finite_uncertainties)
    n_amp = np.asarray([
        min(row["N_supported_amplifiers_ON"], row["N_supported_amplifiers_OFF"]) for row in records], dtype=float)
    n_source = np.asarray([
        min(row["N_source_measurements_ON"], row["N_source_measurements_OFF"]) for row in records], dtype=float)
    amp_floor = max(3, int(np.nanpercentile(n_amp, 25))) if n_amp.size else 3
    source_floor = int(np.nanpercentile(n_source, 25)) if n_source.size else 0
    uncertainty_limit = float(np.nanpercentile(finite_uncertainties, 75)) if finite_uncertainties.size else np.inf
    supported = []
    for row, amp_count, source_count in zip(records, n_amp, n_source):
        row["well_supported"] = bool(
            np.isfinite(row["G_joint"]) and amp_count >= amp_floor and
            source_count >= source_floor and row["G_joint_uncertainty"] <= uncertainty_limit)
        if row["well_supported"]:
            supported.append(row["G_joint"])
    gray_reference = robust_location(np.asarray(supported, dtype=float)) if supported else np.nan
    for row in records:
        row["gray_reference"] = gray_reference
        row["G_relative"] = _safe_ratio(row["G_joint"], gray_reference)
    return records


def _write_gray_products(output_dir, gray_rows, source_rows):
    gray_fields = list(gray_rows[0]) if gray_rows else []
    _write_rows(Path(output_dir) / "m101_exposure_gray_qr_metadata.csv", gray_rows, gray_fields)
    extreme_names = {"20200523_0000024.h5", "20200525_0000021.h5", "20200525_0000022.h5"}
    extreme = [row.copy() for row in gray_rows if row["H5"] in extreme_names]
    for row in extreme:
        flags = []
        if abs(row["throughput"] - 1.) < 1e-5:
            flags.append("throughput_near_one")
        if abs(row["millum"] - 5e5) < 2e3 or row["millum"] >= 5e5:
            flags.append("millum_default_or_maximal")
        if not np.isfinite(row["qr_guider_ratio"]) or row["qr_guider_ratio"] < .5 or row["qr_guider_ratio"] > 1.5:
            flags.append("unusual_qr_guider_ratio")
        if not np.isfinite(row["offset"]) or row["offset"] < .9 or row["offset"] > 1.1:
            flags.append("unusual_survey_offset")
        if min(row["N_source_measurements_ON"], row["N_source_measurements_OFF"]) < 100:
            flags.append("small_source_support")
        if min(row["N_supported_amplifiers_ON"], row["N_supported_amplifiers_OFF"]) < 20:
            flags.append("few_supported_amplifiers")
        if np.isfinite(row["G_joint_uncertainty"]) and row["G_joint_uncertainty"] > .03:
            flags.append("large_gray_uncertainty")
        if (np.isfinite(row["median_slope_minus_ratio_ON"]) and abs(row["median_slope_minus_ratio_ON"]) > .1) or \
                (np.isfinite(row["median_slope_minus_ratio_OFF"]) and abs(row["median_slope_minus_ratio_OFF"]) > .1):
            flags.append("slope_ratio_disagreement")
        if np.isfinite(row["G_OFF_over_G_ON"]) and abs(row["G_OFF_over_G_ON"] - 1.) > .1:
            flags.append("large_on_off_disagreement")
        row["diagnostic_flags"] = ";".join(flags)
    fields = list(extreme[0]) if extreme else []
    _write_rows(Path(output_dir) / "m101_extreme_G_exposure_report.csv", extreme, fields)
    return extreme


def _gray_by_exposure(gray_rows):
    return {(row["H5"], row["exposure"]): row for row in gray_rows}


def compute_source_ifu_products(source_rows, gray_rows, minimum_source_fibers):
    groups = _source_row_groups(source_rows)["rows_by_h5_physical_ifu_band"]
    gray = _gray_by_exposure(gray_rows)
    exposure_rows = []
    for key, values in sorted(groups.items(), key=lambda pair: str(pair[0])):
        h5, specid, slot, uid, band = key
        good = [row for row in values if row["N_source_measurements"] >= minimum_source_fibers]
        if len(good) < 2:
            continue
        g = gray.get((h5, good[0]["exposure"]))
        # The key intentionally remains H5/IFU/band; the exposure is split below.
        by_exposure = {}
        for row in good:
            by_exposure.setdefault(row["exposure"], []).append(row)
        for exposure, amp_rows in sorted(by_exposure.items()):
            if len(amp_rows) < 2:
                continue
            g = gray.get((h5, exposure))
            value = _tiny_median([row["raw_robust_ratio"] / g["G_joint"] for row in amp_rows]) if g and np.isfinite(g["G_joint"]) else np.nan
            exposure_rows.append({
                "H5": h5, "exposure": int(exposure), "SPECID": specid, "IFUSLOT": slot,
                "IFUID": uid, "band": band, "H_IFU_e": value,
                "N_supported_amplifiers": len(amp_rows),
                "N_source_measurements": int(sum(row["N_source_measurements"] for row in amp_rows)),
                "x_arcmin": _tiny_median([row["x_arcmin"] for row in amp_rows]),
                "y_arcmin": _tiny_median([row["y_arcmin"] for row in amp_rows]),
                "G_joint": g["G_joint"] if g else np.nan,
                "G_joint_uncertainty": g["G_joint_uncertainty"] if g else np.nan,
            })
    by_h5_ifu = {}
    for row in exposure_rows:
        by_h5_ifu.setdefault((row["H5"], row["SPECID"], row["IFUSLOT"], row["IFUID"], row["band"]), []).append(row)
    h5_rows = []
    for key, values in sorted(by_h5_ifu.items(), key=lambda pair: str(pair[0])):
        if len(values) < 1:
            continue
        h5_rows.append({
            "H5": key[0], "SPECID": key[1], "IFUSLOT": key[2], "IFUID": key[3], "band": key[4],
            "H_IFU_h": _tiny_median([row["H_IFU_e"] for row in values]),
            "N_supported_exposures": len(values),
            "N_supported_amplifiers": int(np.nanmedian([row["N_supported_amplifiers"] for row in values])),
            "N_source_measurements": int(np.nanmedian([row["N_source_measurements"] for row in values])),
            "x_arcmin": _tiny_median([row["x_arcmin"] for row in values]),
            "y_arcmin": _tiny_median([row["y_arcmin"] for row in values]),
        })
    return exposure_rows, h5_rows


def _fit_plane(rows):
    usable = [row for row in rows if np.isfinite(row["H_IFU_h"]) and np.isfinite(row["x_arcmin"]) and np.isfinite(row["y_arcmin"])]
    if len(usable) < 3:
        return None
    design = np.column_stack(([row["x_arcmin"] for row in usable], [row["y_arcmin"] for row in usable]))
    target = np.asarray([row["H_IFU_h"] - 1. for row in usable], dtype=float)
    coefficients, _, _, _ = np.linalg.lstsq(design, target, rcond=None)
    return np.asarray(coefficients, dtype=float)


def _prediction_metrics(before, after):
    before, after = _finite_pair(before, after)
    if not before.size:
        return {"N": 0, "robust_rms": np.nan, "median_abs": np.nan}
    return {"N": int(before.size), "robust_rms": _finite_robust_rms(before),
            "median_abs": float(np.median(np.abs(after)))}


def compute_source_ifu_crossvalidation(h5_rows):
    output = []
    for band in SOURCE_BANDS + ("JOINT",):
        band_rows = [row for row in h5_rows if row["band"] == band]
        if band == "JOINT":
            by_key = {}
            for row in h5_rows:
                by_key.setdefault((row["H5"], row["SPECID"], row["IFUSLOT"], row["IFUID"]), {})[row["band"]] = row
            band_rows = []
            for key, values in by_key.items():
                if "ON" not in values or "OFF" not in values:
                    continue
                on, off = values["ON"], values["OFF"]
                if np.isfinite(on["H_IFU_h"]) and np.isfinite(off["H_IFU_h"]) and abs(on["H_IFU_h"] - off["H_IFU_h"]) <= .1:
                    joint = on.copy(); joint["band"] = "JOINT"; joint["H_IFU_h"] = _tiny_median([on["H_IFU_h"], off["H_IFU_h"]])
                    band_rows.append(joint)
        h5_names = sorted({row["H5"] for row in band_rows})
        for heldout in h5_names:
            train = [row for row in band_rows if row["H5"] != heldout]
            plane = _fit_plane(train)
            by_ifu = {}
            for row in train:
                by_ifu.setdefault((row["SPECID"], row["IFUSLOT"], row["IFUID"]), []).append(row["H_IFU_h"])
            test = [row for row in band_rows if row["H5"] == heldout]
            for row in test:
                key = (row["SPECID"], row["IFUSLOT"], row["IFUID"])
                persistent = _tiny_median(by_ifu.get(key, []))
                plane_prediction = (1. + plane[0] * row["x_arcmin"] + plane[1] * row["y_arcmin"]
                                    if plane is not None else np.nan)
                output.append({
                    "H5": heldout, "SPECID": row["SPECID"], "IFUSLOT": row["IFUSLOT"], "IFUID": row["IFUID"],
                    "band": band, "H_observed": row["H_IFU_h"],
                    "persistent_prediction": persistent, "plane_prediction": plane_prediction,
                    "baseline_residual": row["H_IFU_h"] - 1.,
                    "persistent_residual": row["H_IFU_h"] - persistent if np.isfinite(persistent) else np.nan,
                    "plane_residual": row["H_IFU_h"] - plane_prediction if np.isfinite(plane_prediction) else np.nan,
                    "persistent_training_H5": len(by_ifu.get(key, [])),
                })
    return output


def summarize_ifu_crossvalidation(rows):
    summary = []
    for band in SOURCE_BANDS + ("JOINT",):
        values = [row for row in rows if row["band"] == band]
        for method, field in (("NULL", "baseline_residual"), ("PLANE", "plane_residual"), ("PERSISTENT_IFU", "persistent_residual")):
            residual = _finite_values([row[field] for row in values])
            summary.append({"band": band, "method": method, "N": int(residual.size),
                            "robust_RMS": _finite_robust_rms(residual),
                            "median_absolute_error": float(np.median(np.abs(residual))) if residual.size else np.nan})
    return summary


def _collapse_spectrum(spectrum, response):
    value, _ = collapse(np.asarray(spectrum, dtype=float)[None, :], np.asarray(response, dtype=float))
    return float(value[0]) if value.size else np.nan


def compute_amplifier_loo(amplifier_exposure_rows, source_rows, gray_rows, responses):
    """Direct three-fold held-out prediction within each H5/physical amplifier."""
    by_key = {}
    for row in amplifier_exposure_rows:
        key = (row["H5"], row["SPECID"], row["IFUSLOT"], row["IFUID"], row["AMP"])
        by_key.setdefault(key, []).append(row)
    band_index = {band: NATIVE_BANDS.index(band) for band in NATIVE_BANDS}
    folds, null_bands = [], []
    predictors = {}
    for key, values in sorted(by_key.items(), key=lambda pair: str(pair[0])):
        if len(values) < 3:
            continue
        values = sorted(values, key=lambda row: row["exposure"])
        for heldout in values:
            training = [row["spectrum"] for row in values if row["exposure"] != heldout["exposure"]]
            predictor = _robust_spectrum(training)
            pred_key = (heldout["H5"], heldout["exposure"], heldout["SPECID"], heldout["IFUSLOT"], heldout["IFUID"], heldout["AMP"])
            predictors[pred_key] = predictor
            baseline = np.asarray(heldout["spectrum"], dtype=float)
            corrected = baseline - predictor
            baseline_rms = _finite_robust_rms(baseline)
            corrected_rms = _finite_robust_rms(corrected)
            folds.append({
                "H5": heldout["H5"], "exposure": heldout["exposure"], "SPECID": heldout["SPECID"],
                "IFUSLOT": heldout["IFUSLOT"], "IFUID": heldout["IFUID"], "AMP": heldout["AMP"],
                "n_training_exposures": len(training), "baseline_robust_RMS": baseline_rms,
                "corrected_robust_RMS": corrected_rms,
                "RMS_ratio": _safe_ratio(corrected_rms, baseline_rms),
                "fraction_improvement": 1. - _safe_ratio(corrected_rms, baseline_rms),
                "prediction_correlation": _correlation(predictor, baseline),
            })
            for band, response in zip(NATIVE_BANDS, responses):
                before = _collapse_spectrum(baseline, response)
                after = _collapse_spectrum(corrected, response)
                null_bands.append({
                    "H5": heldout["H5"], "exposure": heldout["exposure"],
                    "SPECID": heldout["SPECID"], "IFUSLOT": heldout["IFUSLOT"],
                    "IFUID": heldout["IFUID"], "AMP": heldout["AMP"], "band": band,
                    "residual_before": before, "residual_after": after,
                })

    source_groups = _source_row_groups(source_rows)["rows_by_h5_physical_amp"]
    gray = _gray_by_exposure(gray_rows)
    source_validation = []
    for key, rows in source_groups.items():
        h5, specid, slot, uid, amp = key
        for row in rows:
            predictor = predictors.get((h5, row["exposure"], specid, slot, uid, amp))
            g = gray.get((h5, row["exposure"]))
            if predictor is None or g is None or not np.isfinite(g["G_joint"]):
                continue
            for source_row in rows:
                if source_row["exposure"] != row["exposure"]:
                    continue
                band = source_row["band"]
                response = responses[band_index[band]]
                predicted_additive = _collapse_spectrum(predictor, response)
                before_values = np.asarray(source_row["_virus_values"], dtype=float) / source_row["_source_values"]
                after_values = ((np.asarray(source_row["_virus_values"], dtype=float) - predicted_additive) /
                                source_row["_source_values"])
                before = _tiny_median(before_values)
                after = _tiny_median(after_values)
                source_validation.append({
                    "H5": h5, "exposure": row["exposure"], "SPECID": specid,
                    "IFUSLOT": slot, "IFUID": uid, "AMP": amp, "band": band,
                    "N_source_measurements": source_row["N_source_measurements"],
                    "predicted_additive_correction": predicted_additive,
                    "raw_ratio_before": before, "raw_ratio_after": after,
                    "G_joint": g["G_joint"],
                    "gray_normalized_before": _safe_ratio(before, g["G_joint"]),
                    "gray_normalized_after": _safe_ratio(after, g["G_joint"]),
                })
    return folds, null_bands, source_validation


def summarize_source_safety(source_validation):
    before = [row["gray_normalized_before"] for row in source_validation]
    after = [row["gray_normalized_after"] for row in source_validation]
    by_ifu = {}
    for row in source_validation:
        key = (row["H5"], row["exposure"], row["SPECID"], row["IFUSLOT"], row["IFUID"], row["band"])
        by_ifu.setdefault(key, []).append(row["gray_normalized_after"])
    before_ifu = {}
    for row in source_validation:
        key = (row["H5"], row["exposure"], row["SPECID"], row["IFUSLOT"], row["IFUID"], row["band"])
        before_ifu.setdefault(key, []).append(row["gray_normalized_before"])
    on_off_before, on_off_after = [], []
    paired = {}
    for row in source_validation:
        paired.setdefault((row["H5"], row["exposure"], row["SPECID"], row["IFUSLOT"], row["IFUID"], row["AMP"]), {})[row["band"]] = row
    for values in paired.values():
        if "ON" in values and "OFF" in values:
            on_off_before.append(values["ON"]["gray_normalized_before"] - values["OFF"]["gray_normalized_before"])
            on_off_after.append(values["ON"]["gray_normalized_after"] - values["OFF"]["gray_normalized_after"])
    return {
        "N": len(source_validation),
        "robust_amplifier_scatter_before": _finite_robust_rms(before),
        "robust_amplifier_scatter_after": _finite_robust_rms(after),
        "robust_ifu_scatter_before": _finite_robust_rms([_tiny_median(value) for value in before_ifu.values()]),
        "robust_ifu_scatter_after": _finite_robust_rms([_tiny_median(value) for value in by_ifu.values()]),
        "ON_OFF_agreement_before": _finite_robust_rms(on_off_before),
        "ON_OFF_agreement_after": _finite_robust_rms(on_off_after),
        "median_absolute_deviation_from_local_unity_before": float(np.nanmedian(np.abs(np.asarray(before, dtype=float) - 1.))) if _finite_values(before).size else np.nan,
        "median_absolute_deviation_from_local_unity_after": float(np.nanmedian(np.abs(np.asarray(after, dtype=float) - 1.))) if _finite_values(after).size else np.nan,
    }


def plot_amplifier_loo_summary(output_dir, folds, null_bands):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    before = np.asarray([row["baseline_robust_RMS"] for row in folds], dtype=float)
    after = np.asarray([row["corrected_robust_RMS"] for row in folds], dtype=float)
    finite = np.isfinite(before) & np.isfinite(after)
    axes[0].scatter(before[finite], after[finite], s=10, alpha=.35)
    if np.any(finite):
        lo, hi = min(np.nanmin(before[finite]), np.nanmin(after[finite])), max(np.nanmax(before[finite]), np.nanmax(after[finite]))
        axes[0].plot([lo, hi], [lo, hi], "k--")
    axes[0].set(xlabel="held-out baseline robust RMS", ylabel="after predicted correction",
                title="Amplifier LOO residual prediction")
    bands = list(NATIVE_BANDS[:5])
    before_band = [_finite_robust_rms([row["residual_before"] for row in null_bands if row["band"] == band]) for band in bands]
    after_band = [_finite_robust_rms([row["residual_after"] for row in null_bands if row["band"] == band]) for band in bands]
    x = np.arange(len(bands))
    axes[1].bar(x - .18, before_band, width=.36, label="before")
    axes[1].bar(x + .18, after_band, width=.36, label="after")
    axes[1].set_xticks(x, bands); axes[1].set_ylabel("robust RMS")
    axes[1].set_title("Five null bands"); axes[1].legend(); axes[1].grid(axis="y", alpha=.2)
    for axis in axes:
        axis.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(Path(output_dir) / "m101_amplifier_loo_summary.png", dpi=150); plt.close(fig)


def plot_gray_extremes(output_dir, gray_rows):
    extremes = {"20200523_0000024.h5", "20200525_0000021.h5", "20200525_0000022.h5"}
    colors = {name: color for name, color in zip(sorted(extremes), ("tab:red", "tab:purple", "tab:green"))}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for row in gray_rows:
        if row["H5"] not in extremes or not np.isfinite(row["G_relative"]):
            continue
        label = Path(row["H5"]).stem if row["exposure"] == 1 else None
        axes[0].scatter(row["qr_guider_ratio"], row["G_relative"], color=colors[row["H5"]], s=30, label=label)
        axes[1].scatter(row["offset"], row["G_relative"], color=colors[row["H5"]], s=30, label=label)
    axes[0].set(xlabel="QR guider ratio", ylabel="G relative", title="Exposure gray versus QR metadata")
    axes[1].set(xlabel="Survey.offset", ylabel="G relative", title="Exposure gray versus photometric offset")
    for axis in axes:
        axis.grid(alpha=.2)
    axes[0].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(Path(output_dir) / "m101_extreme_G_diagnostic.png", dpi=150); plt.close(fig)


def plot_ifu_persistence(output_dir, h5_rows):
    values = [row for row in h5_rows if row["band"] in SOURCE_BANDS and np.isfinite(row["H_IFU_h"])]
    if not values:
        return
    ifus = sorted({(row["SPECID"], row["IFUSLOT"], row["IFUID"]) for row in values}, key=str)
    h5s = sorted({row["H5"] for row in values})
    index = {key: i for i, key in enumerate(ifus)}
    matrix = np.full((len(ifus), len(h5s)), np.nan)
    for row in values:
        matrix[index[(row["SPECID"], row["IFUSLOT"], row["IFUID"])], h5s.index(row["H5"])] = row["H_IFU_h"]
    fig, axis = plt.subplots(figsize=(max(10, len(h5s) * .55), max(6, len(ifus) * .12)))
    image = axis.imshow(matrix, aspect="auto", interpolation="nearest", cmap="coolwarm", vmin=.75, vmax=1.25)
    axis.set(xlabel="independent H5 date", ylabel="physical IFU (SPECID / IFUSLOT / IFUID)",
             title="Locally gray-normalized source IFU persistence")
    axis.set_xticks(np.arange(len(h5s)), [Path(value).stem for value in h5s], rotation=70, ha="right", fontsize=7)
    axis.set_yticks(np.arange(len(ifus)), ["%d/%d/%d" % value for value in ifus], fontsize=5)
    fig.colorbar(image, ax=axis, label="H_IFU")
    fig.tight_layout(); fig.savefig(Path(output_dir) / "m101_source_ifu_persistence.png", dpi=150); plt.close(fig)


def plot_ifu_crossvalidation(output_dir, cv_rows):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), sharey=False)
    methods = [("NULL", "baseline_residual", "no correction"),
               ("PLANE", "plane_residual", "x/y plane"),
               ("PERSISTENT_IFU", "persistent_residual", "persistent physical IFU")]
    for axis, (method, field, label) in zip(axes, methods):
        groups = []
        labels = []
        for band in SOURCE_BANDS + ("JOINT",):
            values = _finite_values([row[field] for row in cv_rows if row["band"] == band])
            if values.size:
                groups.append(values); labels.append(band)
        if groups:
            axis.boxplot(groups, labels=labels, showfliers=False)
        axis.axhline(0, color="k", lw=.7); axis.set_title(label); axis.grid(axis="y", alpha=.2)
        axis.set_ylabel("held-out H_IFU residual")
    fig.tight_layout(); fig.savefig(Path(output_dir) / "m101_source_ifu_crossvalidation.png", dpi=150); plt.close(fig)


def _finite_robust_rms(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return float(validated_m101.robust_scale(values)) if values.size else np.nan


def compute_blank_source_diagnostics(data, model, skies, fq, source_masks_by_item,
                                     amplifier_h5_rows, minimum_blank_fibers):
    labels = ("blank_valid_and_significant_ON", "blank_valid_and_significant_OFF",
              "blank_valid_and_significant_ANY")
    summary = {"global": {label: 0 for label in labels}, "by_h5_exposure": {}}
    strict_exposure = []
    for item in data:
        masks = source_masks_by_item.get(item.key, {
            band: np.zeros(item.blank_valid.shape, dtype=bool) for band in SOURCE_BANDS})
        significant_on = np.asarray(masks["ON"], dtype=bool)
        significant_off = np.asarray(masks["OFF"], dtype=bool)
        significant_any = significant_on | significant_off
        selections = {
            labels[0]: item.blank_valid & significant_on,
            labels[1]: item.blank_valid & significant_off,
            labels[2]: item.blank_valid & significant_any,
        }
        by_key = "%s/e%d" % (item.h5_name, item.exposure)
        summary["by_h5_exposure"][by_key] = {
            label: int(np.sum(selection)) for label, selection in selections.items()}
        for label, selection in selections.items():
            summary["global"][label] += int(np.sum(selection))

        z = model.z_for(item)
        additive = model.additive_full(item, fq)
        residual = ((np.asarray(item.total, dtype=float) - additive) /
                    np.exp(z)[:, None] - skies[item.key][None, :])
        _, amplifier_groups = _group_indices(item)
        strict_blank = item.blank_valid & ~significant_any
        for (ifu, amp), indices in amplifier_groups:
            selected = strict_blank[indices]
            if int(np.sum(selected)) < minimum_blank_fibers:
                continue
            strict_exposure.append({
                "H5": item.h5_name, "exposure": int(item.exposure),
                "SPECID": ifu[0], "IFUSLOT": ifu[1], "IFUID": ifu[2], "AMP": amp,
                "n_strict_blank_fibers": int(np.sum(selected)),
                "spectrum": _robust_spectrum(residual[indices][selected]),
            })

    strict_by_amplifier = {}
    for row in strict_exposure:
        key = (row["H5"], row["SPECID"], row["IFUSLOT"], row["IFUID"], row["AMP"])
        strict_by_amplifier.setdefault(key, []).append(row)
    strict_h5 = {
        key: _robust_spectrum([row["spectrum"] for row in rows])
        for key, rows in strict_by_amplifier.items()}
    comparisons = []
    for primary in amplifier_h5_rows:
        key = (primary["H5"], primary["SPECID"], primary["IFUSLOT"],
               primary["IFUID"], primary["AMP"])
        strict = strict_h5.get(key)
        if strict is None:
            continue
        finite = np.isfinite(primary["spectrum"]) & np.isfinite(strict)
        difference = primary["spectrum"][finite] - strict[finite]
        comparisons.append({
            "H5": primary["H5"], "SPECID": primary["SPECID"],
            "IFUSLOT": primary["IFUSLOT"], "IFUID": primary["IFUID"],
            "AMP": primary["AMP"],
            "spectral_correlation": _correlation(primary["spectrum"], strict),
            "robust_rms_difference": _finite_robust_rms(difference),
            "primary_residual_rms": float(validated_m101.robust_scale(primary["spectrum"])),
            "strict_blank_residual_rms": float(validated_m101.robust_scale(strict)),
            "n_strict_blank_exposures": len(strict_by_amplifier[key]),
        })
    summary["strict_blank_h5_comparisons"] = len(comparisons)
    return summary, comparisons


def _two_basis_projection_metrics(residual, first, second):
    target = np.asarray(residual, dtype=float)
    b = np.asarray(first, dtype=float)
    l = np.asarray(second, dtype=float)
    finite = np.isfinite(target) & np.isfinite(b) & np.isfinite(l)
    target, b, l = target[finite], b[finite], l[finite]
    if target.size < 3:
        return {"B_coefficient": np.nan, "L_coefficient": np.nan,
                "B_variance_explained": np.nan, "L_variance_explained": np.nan,
                "B_plus_L_variance_explained": np.nan,
                "incremental_L_after_B": np.nan, "remaining_robust_rms": np.nan}
    target = target - np.median(target)
    b = b - np.median(b)
    l = l - np.median(l)
    total = float(np.dot(target, target))
    if total <= 0:
        return {"B_coefficient": np.nan, "L_coefficient": np.nan,
                "B_variance_explained": np.nan, "L_variance_explained": np.nan,
                "B_plus_L_variance_explained": np.nan,
                "incremental_L_after_B": np.nan, "remaining_robust_rms": np.nan}
    b_metrics = _projection_metrics(target, b)
    l_metrics = _projection_metrics(target, l)
    design = np.column_stack((b, l))
    coefficients, _, _, _ = np.linalg.lstsq(design, target, rcond=None)
    remaining = target - design @ coefficients
    joint_explained = float(1. - np.dot(remaining, remaining) / total)
    return {
        "B_coefficient": float(coefficients[0]), "L_coefficient": float(coefficients[1]),
        "B_variance_explained": b_metrics["fraction_variance_explained"],
        "L_variance_explained": l_metrics["fraction_variance_explained"],
        "B_plus_L_variance_explained": joint_explained,
        "incremental_L_after_B": joint_explained - b_metrics["fraction_variance_explained"],
        "remaining_robust_rms": _finite_robust_rms(remaining),
    }


def _multi_basis_projection_metrics(residual, bases):
    """Projection QA for actual sky, Model-3 K, and incident-light bases."""
    target = np.asarray(residual, dtype=float)
    arrays = [np.asarray(value, dtype=float) for value in bases]
    finite = np.isfinite(target)
    for value in arrays:
        finite &= np.isfinite(value)
    target = target[finite]
    arrays = [value[finite] for value in arrays]
    result = {"B_variance_explained": np.nan, "K_variance_explained": np.nan,
              "B_plus_K_variance_explained": np.nan,
              "B_plus_K_plus_L_variance_explained": np.nan,
              "incremental_L_after_B_plus_K": np.nan,
              "remaining_robust_rms": np.nan}
    if target.size < 3:
        return result
    target = target - np.median(target)
    arrays = [value - np.median(value) for value in arrays]
    total = float(np.dot(target, target))
    if total <= 0:
        return result

    def explained(columns):
        design = np.column_stack(columns)
        coefficients, _, _, _ = np.linalg.lstsq(design, target, rcond=None)
        remaining = target - design @ coefficients
        return float(1. - np.dot(remaining, remaining) / total), remaining

    result["B_variance_explained"], _ = explained([arrays[0]])
    result["K_variance_explained"], _ = explained([arrays[1]])
    result["B_plus_K_variance_explained"], _ = explained(arrays[:2])
    result["B_plus_K_plus_L_variance_explained"], remaining = explained(arrays)
    result["incremental_L_after_B_plus_K"] = (
        result["B_plus_K_plus_L_variance_explained"] - result["B_plus_K_variance_explained"])
    result["remaining_robust_rms"] = _finite_robust_rms(remaining)
    return result


def compute_incident_light_diagnostics(amplifier_h5_rows, incident_exposure_rows):
    leave_amp_by_h5 = {}
    source_count_by_h5_amp = {}
    for row in incident_exposure_rows:
        key = (row["H5"], row["SPECID"], row["IFUSLOT"], row["IFUID"], row["AMP"])
        leave_amp_by_h5.setdefault(key, []).append(row["spectrum"])
        source_count_by_h5_amp.setdefault(key, []).append(row["n_source_fibers_other_amp"])
    leave_amp_by_h5 = {key: _robust_spectrum(values) for key, values in leave_amp_by_h5.items()}
    output = []
    for row in amplifier_h5_rows:
        key = (row["H5"], row["SPECID"], row["IFUSLOT"], row["IFUID"], row["AMP"])
        source_spectrum = leave_amp_by_h5.get(key, np.full(row["spectrum"].shape, np.nan))
        metrics = _multi_basis_projection_metrics(row["spectrum"],
                                                   (row["sky"], row["K"], source_spectrum))
        output.append({
            "H5": row["H5"], "SPECID": row["SPECID"], "IFUSLOT": row["IFUSLOT"],
            "IFUID": row["IFUID"], "AMP": row["AMP"], "B_basis": "actual_contemporaneous_sky",
            "L_basis": "other_source_bearing_amplifiers",
            "n_source_fibers_other_amp": int(np.nanmedian(source_count_by_h5_amp.get(key, [0]))),
            **metrics,
        })
    return output


def write_blank_source_diagnostics(output_dir, summary, comparisons):
    fields = ["H5", "SPECID", "IFUSLOT", "IFUID", "AMP", "spectral_correlation",
              "robust_rms_difference", "primary_residual_rms",
              "strict_blank_residual_rms", "n_strict_blank_exposures"]
    _write_rows(Path(output_dir) / "m101_amplifier_blank_residual_comparison.csv",
                comparisons, fields)
    (Path(output_dir) / "m101_amplifier_blank_source_diagnostics.json").write_text(
        json.dumps(json_ready(summary), indent=2, sort_keys=True))


def summarize_blank_source_from_aux(data, source_masks_by_item, amplifier_h5_rows, auxiliary):
    labels = ("blank_valid_and_significant_ON", "blank_valid_and_significant_OFF",
              "blank_valid_and_significant_ANY")
    summary = {"global": {label: 0 for label in labels}, "by_h5_exposure": {}}
    for item in data:
        masks = source_masks_by_item.get(item.key, {band: np.zeros(item.row_index.size, dtype=bool) for band in SOURCE_BANDS})
        selections = (item.blank_valid & masks["ON"], item.blank_valid & masks["OFF"],
                      item.blank_valid & (masks["ON"] | masks["OFF"]))
        key = "%s/e%d" % (item.h5_name, item.exposure)
        summary["by_h5_exposure"][key] = {label: int(np.sum(selection)) for label, selection in zip(labels, selections)}
        for label, selection in zip(labels, selections):
            summary["global"][label] += int(np.sum(selection))
    strict_by_amp = {}
    for row in auxiliary["strict_exposure"]:
        strict_by_amp.setdefault((row["H5"], row["SPECID"], row["IFUSLOT"], row["IFUID"], row["AMP"]), []).append(row["spectrum"])
    comparisons = []
    for primary in amplifier_h5_rows:
        key = (primary["H5"], primary["SPECID"], primary["IFUSLOT"], primary["IFUID"], primary["AMP"])
        values = strict_by_amp.get(key)
        if not values:
            continue
        strict = _robust_spectrum(values)
        comparisons.append({
            "H5": primary["H5"], "SPECID": primary["SPECID"], "IFUSLOT": primary["IFUSLOT"],
            "IFUID": primary["IFUID"], "AMP": primary["AMP"],
            "spectral_correlation": _correlation(primary["spectrum"], strict),
            "robust_rms_difference": _finite_robust_rms(primary["spectrum"] - strict),
            "primary_residual_rms": _finite_robust_rms(primary["spectrum"]),
            "strict_blank_residual_rms": _finite_robust_rms(strict),
            "n_strict_blank_exposures": len(values),
        })
    summary["strict_blank_h5_comparisons"] = len(comparisons)
    return summary, comparisons


def write_incident_light_diagnostics(output_dir, rows):
    fields = ["H5", "SPECID", "IFUSLOT", "IFUID", "AMP", "B_basis", "L_basis",
              "n_source_fibers_other_amp", "B_variance_explained", "K_variance_explained",
              "B_plus_K_variance_explained", "B_plus_K_plus_L_variance_explained",
              "incremental_L_after_B_plus_K", "remaining_robust_rms"]
    _write_rows(Path(output_dir) / "m101_amplifier_incident_light.csv", rows, fields)


def ratio_summary(values, center):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    normalized = values / center if np.isfinite(center) and center != 0 else np.full(values.shape, np.nan)
    normalized = normalized[np.isfinite(normalized)]
    if not normalized.size:
        return {"N": 0, "p16": np.nan, "median": np.nan, "p84": np.nan,
                "robust_rms_about_center": np.nan, "normalized_rms_about_unity": np.nan,
                "p95_abs_deviation_from_center": np.nan,
                "p95_abs_deviation_from_unity": np.nan}
    return {
        "N": int(values.size), "p16": float(np.percentile(values, 16)),
        "median": float(np.median(values)), "p84": float(np.percentile(values, 84)),
        "robust_rms_about_center": float(validated_m101.robust_scale(values - center)),
        "normalized_rms_about_unity": float(np.sqrt(np.mean((normalized - 1.) ** 2))),
        "p95_abs_deviation_from_center": float(np.percentile(np.abs(values - center), 95)),
        "p95_abs_deviation_from_unity": float(np.percentile(np.abs(normalized - 1.), 95)),
    }


def write_source_products(output_dir, rows, centers, by_exposure, fiber_ratios,
                          amplifier_weighted_centers=None, timings=None):
    if amplifier_weighted_centers is None:
        amplifier_weighted_centers = centers.copy()
    fields = ["H5", "exposure", "SPECID", "IFUSLOT", "IFUID", "AMP", "band",
              "N_source_measurements", "raw_robust_ratio", "normalized_ratio",
              "zero_intercept_slope", "slope_residual_robust_rms", "robust_scatter",
              "slope_minus_ratio",
              "ratio_uncertainty", "x_arcmin", "y_arcmin",
              "source_threshold", "source_reference_method"]
    stage_started = time.perf_counter()
    _write_rows(output_dir / "m101_external_stitching_amplifiers.csv", rows, fields)
    if timings is not None:
        timings["source_write_csv_seconds"] = time.perf_counter() - stage_started
    summary = {"fiber_weighted_centers": centers,
               "amplifier_weighted_centers": amplifier_weighted_centers,
               "raw_robust_centers": centers, "by_band": {},
               "by_exposure_band": {}, "by_h5_exposure_band": {}}
    for band in SOURCE_BANDS:
        summary["by_band"][band] = ratio_summary(fiber_ratios[band], centers[band])
    rows_by_h5_exposure_band = _source_row_groups(rows)["rows_by_h5_exposure_band"]
    for key, info in sorted(by_exposure.items(), key=lambda pair: str(pair[0])):
        band = key[2]
        values = [row["raw_robust_ratio"] for row in rows_by_h5_exposure_band.get(key, [])]
        summary["by_h5_exposure_band"]["%s/e%d/%s" % key] = {
            **info, **ratio_summary(values, centers[band])}
    for exposure in (1, 2, 3):
        for band in SOURCE_BANDS:
            values = [row["raw_robust_ratio"] for row in rows
                      if row["exposure"] == exposure and row["band"] == band]
            summary["by_exposure_band"]["e%d/%s" % (exposure, band)] = {
                **ratio_summary(values, centers[band]),
                "N_supported_amplifiers": len(values),
            }
    paired = {}
    for row in rows:
        paired.setdefault(row["_key"][:-1], {})[row["band"]] = row
    joint = [value for value in paired.values()
             if "ON" in value and "OFF" in value]
    x = np.asarray([value["ON"]["normalized_ratio"] for value in joint], dtype=float)
    y = np.asarray([value["OFF"]["normalized_ratio"] for value in joint], dtype=float)
    finite = np.isfinite(x) & np.isfinite(y)
    summary["ON_OFF_joint"] = {
        "N": int(np.sum(finite)),
        "correlation": _correlation(x[finite], y[finite]) if np.any(finite) else np.nan,
        "median_normalized_ON_minus_OFF": float(np.median(x[finite] - y[finite])) if np.any(finite) else np.nan,
        "normalized_difference_rms": float(np.sqrt(np.mean((x[finite] - y[finite]) ** 2))) if np.any(finite) else np.nan,
    }
    stage_started = time.perf_counter()
    (output_dir / "m101_external_stitching_summary.json").write_text(
        json.dumps(json_ready(summary), indent=2, sort_keys=True))
    if timings is not None:
        timings["source_write_json_seconds"] = time.perf_counter() - stage_started
    return summary


def plot_source_products(output_dir, rows, centers, amplifier_weighted_centers=None, timings=None):
    if amplifier_weighted_centers is None:
        amplifier_weighted_centers = centers.copy()
    colors = {"ON": "tab:blue", "OFF": "tab:orange"}
    markers = {"LL": "o", "LU": "s", "RL": "^", "RU": "D"}
    groups = _source_row_groups(rows)
    rows_by_band = {band: [row for row in rows if row["band"] == band and np.isfinite(row["normalized_ratio"])]
                    for band in SOURCE_BANDS}
    for band in SOURCE_BANDS:
        values = rows_by_band[band]
        fig, axis = plt.subplots(figsize=(14, 5))
        for amp in AMP_ORDER:
            group = [row for row in values if row["AMP"] == amp]
            axis.scatter(np.arange(len(group)), [row["normalized_ratio"] for row in group],
                         marker=markers[amp], s=14, alpha=.65, label=amp)
        axis.axhline(1, color="k", lw=.8)
        axis.set(xlabel="H5 / exposure / physical-amplifier observation index",
                 ylabel="normalized O/X", title="%s amplifier source stitching" % band)
        axis.legend(); axis.grid(alpha=.2)
        fig.tight_layout()
        stage_started = time.perf_counter()
        fig.savefig(output_dir / ("m101_external_stitching_%s_ratios.png" % band.lower()), dpi=140); plt.close(fig)
        if timings is not None:
            timings["source_ratio_plot_seconds"] = timings.get("source_ratio_plot_seconds", 0.) + time.perf_counter() - stage_started

        fig, axis = plt.subplots(figsize=(7, 5))
        axis.hist([row["raw_robust_ratio"] for row in values], bins=40,
                  color=colors[band], alpha=.8)
        axis.axvline(centers[band], color="k", lw=.8, label="robust center")
        axis.set(xlabel="raw O/X", ylabel="amplifier observations",
                 title="%s raw amplifier O/X ratios" % band)
        axis.legend(); fig.tight_layout()
        stage_started = time.perf_counter()
        fig.savefig(output_dir / ("m101_external_stitching_%s_raw_histogram.png" % band.lower()), dpi=140)
        plt.close(fig)
        if timings is not None:
            timings["source_histogram_seconds"] = timings.get("source_histogram_seconds", 0.) + time.perf_counter() - stage_started

    paired = {}
    for row in rows:
        paired.setdefault(row["_key"][:-1], {})[row["band"]] = row
    joint = [value for value in paired.values() if "ON" in value and "OFF" in value and
             np.isfinite(value["ON"]["normalized_ratio"]) and np.isfinite(value["OFF"]["normalized_ratio"])]
    fig, axis = plt.subplots(figsize=(6, 6))
    x = [value["ON"]["normalized_ratio"] for value in joint]
    y = [value["OFF"]["normalized_ratio"] for value in joint]
    axis.scatter(x, y, s=16, alpha=.65)
    if joint:
        lo, hi = min(x + y), max(x + y)
        axis.plot([lo, hi], [lo, hi], "k--")
    axis.axhline(1, color="0.75", lw=.7); axis.axvline(1, color="0.75", lw=.7)
    axis.set(xlabel="normalized ON O/X", ylabel="normalized OFF O/X", title="ON versus OFF amplifier stitching")
    axis.grid(alpha=.2); fig.tight_layout()
    stage_started = time.perf_counter()
    fig.savefig(output_dir / "m101_external_stitching_on_vs_off.png", dpi=140); plt.close(fig)
    if timings is not None:
        timings["source_on_off_plot_seconds"] = time.perf_counter() - stage_started

    combined = groups["rows_by_h5_physical_amp_band"]
    h5_level_rows = []
    for key, grouped in combined.items():
        h5_level_rows.append({
            "H5": key[0], "SPECID": key[1], "IFUSLOT": key[2], "IFUID": key[3],
            "AMP": key[4], "band": key[5],
            "raw_robust_ratio": _tiny_median([row["raw_robust_ratio"] for row in grouped]),
            "x_arcmin": _tiny_median([row["x_arcmin"] for row in grouped]),
            "y_arcmin": _tiny_median([row["y_arcmin"] for row in grouped]),
        })
    h5_counts = {}
    for row in h5_level_rows:
        h5_counts[row["H5"]] = h5_counts.get(row["H5"], 0) + 1
    representative_h5 = [key for key, _ in sorted(h5_counts.items(), key=lambda pair: (-pair[1], pair[0]))[:3]]
    fig, axes = plt.subplots(len(representative_h5), 2, figsize=(12, 4 * len(representative_h5)), squeeze=False)
    for i, h5 in enumerate(representative_h5):
        for j, band in enumerate(SOURCE_BANDS):
            subset = [row for row in h5_level_rows
                      if row["H5"] == h5 and row["band"] == band and
                      np.isfinite(row["raw_robust_ratio"])]
            local_center = robust_location([row["raw_robust_ratio"] for row in subset]) if subset else np.nan
            axis = axes[i, j]
            values = [row["raw_robust_ratio"] / local_center for row in subset]
            image = axis.scatter([row["x_arcmin"] for row in subset], [row["y_arcmin"] for row in subset],
                                 c=values, cmap="coolwarm", vmin=.9, vmax=1.1, s=35)
            axis.axhline(0, color="0.8", lw=.5); axis.axvline(0, color="0.8", lw=.5)
            axis.set_title("%s %s (3-exposure H5 values)" % (Path(h5).stem, band)); axis.set_xlabel("x [arcmin]"); axis.set_ylabel("y [arcmin]"); axis.grid(alpha=.15)
            fig.colorbar(image, ax=axis, label="amplifier ratio / H5 robust center")
    fig.tight_layout()
    stage_started = time.perf_counter()
    fig.savefig(output_dir / "m101_external_stitching_focal_maps.png", dpi=140); plt.close(fig)
    if timings is not None:
        timings["source_focal_map_seconds"] = time.perf_counter() - stage_started
    fig, axis = plt.subplots(figsize=(8, 5))
    h5_names = sorted({row["H5"] for row in rows})
    for band, color in colors.items():
        for h5_index, h5 in enumerate(h5_names):
            points = []
            for exposure in (1, 2, 3):
                values = [row["raw_robust_ratio"] for row in groups["rows_by_h5_exposure_band"].get((h5, exposure, band), [])]
                if values:
                    points.append((h5_index * 4 + exposure,
                                   _tiny_median(values), robust_scatter(values)))
            if points:
                axis.errorbar([point[0] + (-.08 if band == "ON" else .08) for point in points],
                              [point[1] for point in points],
                              yerr=[point[2] for point in points], fmt="o-", color=color,
                              alpha=.75, label=band if h5_index == 0 else None)
    axis.set_xticks([index * 4 + 2 for index in range(len(h5_names))])
    axis.set_xticklabels([Path(name).stem for name in h5_names], rotation=45, ha="right", fontsize=7)
    axis.set_xlabel("H5; points within each group are e1, e2, e3")
    axis.set_ylabel("raw O/X amplifier-weighted center +/- robust scatter")
    axis.set_title("H5-aware external source stitching repeatability")
    axis.grid(alpha=.2); axis.legend(); fig.tight_layout()
    stage_started = time.perf_counter()
    fig.savefig(output_dir / "m101_external_stitching_per_exposure.png", dpi=140); plt.close(fig)
    if timings is not None:
        timings["source_h5_repeatability_plot_seconds"] = time.perf_counter() - stage_started
    return len(joint)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5", nargs="+", required=True)
    parser.add_argument("--product", required=True)
    parser.add_argument("--blank-file", required=True)
    parser.add_argument("--external-cache", required=True)
    parser.add_argument("--compact-mask", required=True)
    parser.add_argument("--on-filter", required=True)
    parser.add_argument("--off-filter", required=True)
    parser.add_argument("--fq-template", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--minimum-blank-fibers", type=int, default=20)
    parser.add_argument("--minimum-source-fibers", type=int, default=10)
    parser.add_argument("--source-sigma", type=float, default=5.)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise SystemExit("output directory is nonempty; use --overwrite: %s" % output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    progress = lambda message: _progress(started, message)
    progress("loading persisted Model-3 product")
    product, model, skies = load_model_and_skies(args.product, progress=progress)
    progress("persisted product load complete")
    timings = {}
    input_started = time.perf_counter()
    progress("validating H5 set and loading blank masks, filters, and native spectra")
    h5_paths = discover_h5(args.h5, development=True)
    expected = {Path(item["filename"]).name for item in product["provenance"]["input_h5"]}
    if {path.name for path in h5_paths} != expected:
        raise ValueError("diagnostic H5 set does not match persisted Model-3 product")
    blank_masks, blank_provenance = __import__("m101_blank_fibers").load(args.blank_file, h5_paths)
    fq = validated_m101.load_fq(args.fq_template)
    data, input_provenance = load_native_data(
        h5_paths, blank_masks, args.on_filter, args.off_filter,
        timings=timings, include_band_errors=False, collapse_band_indices=(5, 6),
        include_native_errors=False, progress=progress)
    timings["input_loading_seconds"] = time.perf_counter() - input_started
    progress("native data load complete in %.3fs" % timings["input_loading_seconds"])
    external_started = time.perf_counter()
    progress("loading external measurement cache and compact source mask")
    external, external_provenance = m101_external_measurements.load(args.external_cache, h5_paths)
    mask_data, mask_wcs = m101_compact_mask.load(args.compact_mask)
    progress("external cache and compact mask loaded in %.3fs" %
             (time.perf_counter() - external_started))
    print("loaded %d H5 / %d exposures / %d native fibers" %
          (len(h5_paths), len(data), sum(item.row_index.size for item in data)))
    print("current registry excluded legacy=%d persistent=%d overlap=%d" %
          (input_provenance["hardware_exclusions"]["legacy_date_slot_rows"],
           input_provenance["hardware_exclusions"]["persistent_physical_rows"],
           input_provenance["hardware_exclusions"]["overlap_rows"]))

    source_started = time.perf_counter()
    progress("starting external ON/OFF source stitching")
    source_rows, centers, by_exposure, fiber_ratios, source_masks_by_item = build_source_rows(
        data, external, mask_data, mask_wcs, model, skies, fq,
        input_provenance["responses"], args.source_sigma, args.minimum_source_fibers,
        progress=progress, return_source_masks=True)
    timings["source_computation_seconds"] = time.perf_counter() - source_started
    progress("source stitching analysis complete in %.3fs" % timings["source_computation_seconds"])
    gray_rows = compute_exposure_gray_diagnostics(data, source_rows)
    _write_gray_products(output_dir, gray_rows, source_rows)
    gray_plot_started = time.perf_counter()
    plot_gray_extremes(output_dir, gray_rows)
    timings["source_extreme_plot_seconds"] = time.perf_counter() - gray_plot_started

    diagnostic_started = time.perf_counter()
    progress("starting amplifier residual analysis and reusable source residual templates")
    exposure_rows, h5_rows, amplifier_exposure_rows, amplifier_h5_rows, auxiliary = compute_ifu_residuals(
        data, model, skies, fq, args.minimum_blank_fibers, progress=progress,
        source_masks_by_item=source_masks_by_item, return_aux=True)
    timings["amplifier_residual_seconds"] = time.perf_counter() - diagnostic_started
    progress("residual analysis complete in %.3fs" % timings["amplifier_residual_seconds"])
    pair_started = time.perf_counter()
    progress("computing cross-H5 amplifier repeatability pairs")
    pair_rows = pairwise_amplifier_metrics(amplifier_h5_rows)
    within_rows = within_h5_amplifier_metrics(amplifier_exposure_rows)
    annotate_spectral_rows(amplifier_h5_rows)
    progress("amplifier pairing complete in %.3fs" %
             (time.perf_counter() - pair_started))
    output_started = time.perf_counter()
    progress("writing amplifier products; IFU products disabled")
    write_amplifier_products(output_dir, validated_m101.DEF_WAVE,
                             amplifier_exposure_rows, amplifier_h5_rows, pair_rows)
    progress("amplifier products written; plotting multi-date amplifier panels")
    plot_amplifier_panels(output_dir, validated_m101.DEF_WAVE,
                          amplifier_exposure_rows, amplifier_h5_rows)
    progress("amplifier products and diagnostic panels written in %.3fs" %
             (time.perf_counter() - output_started))
    print("amplifier residual spectra: %d exposure records, %d H5 records, %d cross-H5 pairs" %
          (len(amplifier_exposure_rows), len(amplifier_h5_rows), len(pair_rows)))
    if within_rows:
        print("within-H5 amplifier repeatability: median correlation=%.4g; median scalar residual robust RMS=%.4g" %
              (robust_location([row["spectral_shape_correlation"] for row in within_rows]),
               robust_location([row["robust_rms_after_scalar"] for row in within_rows])))
    if pair_rows:
        print("cross-H5 amplifier repeatability: median correlation=%.4g; median scalar residual robust RMS=%.4g" %
              (robust_location([row["spectral_shape_correlation"] for row in pair_rows]),
               robust_location([row["robust_rms_after_scalar"] for row in pair_rows])))

    amplifier_weighted_centers = {
        band: robust_location([row["raw_robust_ratio"] for row in source_rows
                               if row["band"] == band])
        for band in SOURCE_BANDS}
    progress("writing source products")
    source_summary = write_source_products(output_dir, source_rows, centers, by_exposure,
                                           fiber_ratios, amplifier_weighted_centers, timings=timings)
    progress("source tables written; plotting source stitching products")
    joint_count = plot_source_products(output_dir, source_rows, centers,
                                       amplifier_weighted_centers, timings=timings)
    timings["source_product_plotting_seconds"] = sum(
        timings.get(key, 0.) for key in ("source_ratio_plot_seconds", "source_histogram_seconds",
                                         "source_on_off_plot_seconds", "source_focal_map_seconds",
                                         "source_h5_repeatability_plot_seconds"))
    blank_summary, blank_comparisons = summarize_blank_source_from_aux(
        data, source_masks_by_item, amplifier_h5_rows, auxiliary)
    incident_rows = compute_incident_light_diagnostics(amplifier_h5_rows, auxiliary["incident_exposure"])
    write_blank_source_diagnostics(output_dir, blank_summary, blank_comparisons)
    write_incident_light_diagnostics(output_dir, incident_rows)
    timings["source_product_writing_seconds"] = timings.get("source_write_csv_seconds", 0.) + timings.get("source_write_json_seconds", 0.)
    progress("source products written in %.3fs" %
             (timings["source_product_writing_seconds"] + timings["source_product_plotting_seconds"]))

    ifu_started = time.perf_counter()
    source_ifu_exposure_rows, source_ifu_h5_rows = compute_source_ifu_products(
        source_rows, gray_rows, args.minimum_source_fibers)
    source_ifu_cv_rows = compute_source_ifu_crossvalidation(source_ifu_h5_rows)
    source_ifu_summary = summarize_ifu_crossvalidation(source_ifu_cv_rows)
    _write_rows(output_dir / "m101_source_ifu_h5.csv", source_ifu_h5_rows,
                list(source_ifu_h5_rows[0]) if source_ifu_h5_rows else [])
    _write_rows(output_dir / "m101_source_ifu_persistence.csv", source_ifu_exposure_rows,
                list(source_ifu_exposure_rows[0]) if source_ifu_exposure_rows else [])
    _write_rows(output_dir / "m101_source_ifu_crossvalidation.csv", source_ifu_cv_rows,
                list(source_ifu_cv_rows[0]) if source_ifu_cv_rows else [])
    final_plot_started = time.perf_counter()
    plot_ifu_persistence(output_dir, source_ifu_h5_rows)
    plot_ifu_crossvalidation(output_dir, source_ifu_cv_rows)
    timings["ifu_persistence_experiment_seconds"] = time.perf_counter() - ifu_started

    loo_started = time.perf_counter()
    amp_loo_rows, amp_loo_null_rows, amp_loo_source_rows = compute_amplifier_loo(
        amplifier_exposure_rows, source_rows, gray_rows, input_provenance["responses"])
    _write_rows(output_dir / "m101_amplifier_loo_validation.csv", amp_loo_rows,
                list(amp_loo_rows[0]) if amp_loo_rows else [])
    _write_rows(output_dir / "m101_amplifier_loo_null_band_validation.csv", amp_loo_null_rows,
                list(amp_loo_null_rows[0]) if amp_loo_null_rows else [])
    _write_rows(output_dir / "m101_amplifier_loo_source_validation.csv", amp_loo_source_rows,
                list(amp_loo_source_rows[0]) if amp_loo_source_rows else [])
    plot_amplifier_loo_summary(output_dir, amp_loo_rows, amp_loo_null_rows)
    timings["final_experiment_plot_seconds"] = time.perf_counter() - final_plot_started
    timings["amplifier_leave_one_out_seconds"] = time.perf_counter() - loo_started
    global_g = next(iter(external.values()))["global_g"]
    print("external cache global g: ON=%.8g OFF=%.8g; X = g * cached I exactly once" %
          (global_g["ON"], global_g["OFF"]))
    for band in SOURCE_BANDS:
        print("%s O/X: fiber_weighted_center=%.8g amplifier_weighted_center=%.8g N=%d p16=%.6g median=%.6g p84=%.6g raw_robust_RMS=%.6g normalized_RMS=%.6g p95_abs_delta=%.6g supported_amps=%d" %
              (band, centers[band], amplifier_weighted_centers[band], source_summary["by_band"][band]["N"],
               source_summary["by_band"][band]["p16"], source_summary["by_band"][band]["median"],
               source_summary["by_band"][band]["p84"], source_summary["by_band"][band]["robust_rms_about_center"],
               source_summary["by_band"][band]["normalized_rms_about_unity"],
               source_summary["by_band"][band]["p95_abs_deviation_from_unity"],
               sum(row["band"] == band for row in source_rows)))
    print("ON/OFF raw center ratio=%.8g; paired supported amplifier measurements=%d" %
          (centers["ON"] / centers["OFF"], joint_count))
    print("blank/source overlap: global ON=%d OFF=%d ANY=%d; H5/exposure records=%d; strict-blank H5 comparisons=%d" %
          (blank_summary["global"]["blank_valid_and_significant_ON"],
           blank_summary["global"]["blank_valid_and_significant_OFF"],
           blank_summary["global"]["blank_valid_and_significant_ANY"],
           len(blank_summary["by_h5_exposure"]), blank_summary["strict_blank_h5_comparisons"]))
    if blank_comparisons:
        print("blank/source residual comparison: median correlation=%.4g; median RMS difference=%.4g; median primary RMS=%.4g; median strict-blank RMS=%.4g" %
              (robust_location([row["spectral_correlation"] for row in blank_comparisons]),
               robust_location([row["robust_rms_difference"] for row in blank_comparisons]),
               robust_location([row["primary_residual_rms"] for row in blank_comparisons]),
               robust_location([row["strict_blank_residual_rms"] for row in blank_comparisons])))
    finite_incident = [row for row in incident_rows
                       if np.isfinite(row["B_variance_explained"])]
    if finite_incident:
        print("incident-light projections: median variance explained B=%.4g K=%.4g B+K=%.4g B+K+L=%.4g incremental L after B+K=%.4g remaining robust RMS=%.4g" %
              (robust_location([row["B_variance_explained"] for row in finite_incident]),
               robust_location([row["K_variance_explained"] for row in finite_incident]),
               robust_location([row["B_plus_K_variance_explained"] for row in finite_incident]),
               robust_location([row["B_plus_K_plus_L_variance_explained"] for row in finite_incident]),
               robust_location([row["incremental_L_after_B_plus_K"] for row in finite_incident]),
               robust_location([row["remaining_robust_rms"] for row in finite_incident])))

    print("timings: %s" % json.dumps({key: value for key, value in timings.items()
                                      if isinstance(value, (int, float))}, sort_keys=True))
    for row in gray_rows:
        if row["H5"] in {"20200523_0000024.h5", "20200525_0000021.h5", "20200525_0000022.h5"}:
            print("extreme G %s e%d: millum=%.6g throughput=%.6g qr=%.6g offset=%.6g exptime=%.6g nphot=%d GON=%.6g GOFF=%.6g Gjoint=%.6g sigma=%.6g amps=%d/%d sources=%d/%d" %
                  (row["H5"], row["exposure"], row["millum"], row["throughput"], row["qr_guider_ratio"],
                   row["offset"], row["exptime"], row["nstarsphotom"], row["G_ON"], row["G_OFF"],
                   row["G_joint"], row["G_joint_uncertainty"], row["N_supported_amplifiers_ON"],
                   row["N_supported_amplifiers_OFF"], row["N_source_measurements_ON"],
                   row["N_source_measurements_OFF"]))
    print("IFU persistence CV summary: %s" % json.dumps(json_ready(source_ifu_summary), sort_keys=True))
    if amp_loo_rows:
        improvements = _finite_values([row["fraction_improvement"] for row in amp_loo_rows])
        print("amplifier LOO: N=%d median baseline RMS=%.6g median corrected RMS=%.6g median improvement=%.6g p16/p50/p84=%.6g/%.6g/%.6g improved_fraction=%.6g" %
              (len(amp_loo_rows), robust_location([row["baseline_robust_RMS"] for row in amp_loo_rows]),
               robust_location([row["corrected_robust_RMS"] for row in amp_loo_rows]),
               robust_location([row["fraction_improvement"] for row in amp_loo_rows]),
               np.percentile(improvements, 16), np.percentile(improvements, 50), np.percentile(improvements, 84),
               float(np.mean(improvements > 0))))
        for amp in AMP_ORDER:
            subset = [row for row in amp_loo_rows if row["AMP"] == amp]
            if subset:
                print("amplifier LOO %s: N=%d baseline=%.6g corrected=%.6g improvement=%.6g" %
                      (amp, len(subset), robust_location([row["baseline_robust_RMS"] for row in subset]),
                       robust_location([row["corrected_robust_RMS"] for row in subset]),
                       robust_location([row["fraction_improvement"] for row in subset])))
    null_summary = []
    for band in NATIVE_BANDS[:5]:
        before = [row["residual_before"] for row in amp_loo_null_rows if row["band"] == band]
        after = [row["residual_after"] for row in amp_loo_null_rows if row["band"] == band]
        null_summary.append((band, _finite_robust_rms(before), _finite_robust_rms(after)))
    print("five null-band held-out RMS before/after: %s" % json.dumps(null_summary))
    print("source safety: %s" % json.dumps(json_ready(summarize_source_safety(amp_loo_source_rows)), sort_keys=True))
    print("outputs: %s" % output_dir)
    print("runtime_seconds: %.3f" % (time.perf_counter() - started))


if __name__ == "__main__":
    main()
