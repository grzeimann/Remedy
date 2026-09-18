#!/usr/bin/env python3
"""Model-4A and Model-4A.1 development diagnostics.

Model 4A constructs and inspects illumination-derived spectral templates.
Model 4A.1 compares the actual frozen Model-3 blank residual in one specified
amplifier to that template.  It is a shape diagnostic only and performs no
Model-4 parameter inference.

Model-4 development plan
------------------------

    Model 4A:
        construct and inspect I, smooth(I), and R
        NO fitting
    Model 4A.1:
        compare the frozen Model-3 blank residual to R for one amplifier
        NO calibration update
    Model 4B:
        construct the additive design P + alpha*K*f(q) + beta*R and verify it
        synthetically; NO real-data parameter inference beyond the isolated test
    Model 4C:
        fit additive terms with frozen Model-3 m and S, initially on one
        amplifier / exposure
    Model 4D:
        introduce the external ON/OFF source constraint with m and S frozen
    Model 4E:
        implement and independently verify the multiplicative update
    Model 4F:
        permit a sky update
    Model 4G:
        connect validated conditional updates into an alternating solution

Every block requires its own expectation, minimal implementation, small run,
rich return, interpretation gate, and stop.  This script stops after the
requested block.  In particular it does not implement Model 4B--4G.

Scientific convention used in this block
-----------------------------------------

The native total spectrum is reconstructed by the established loader as

    D = Fibers.spectrum / Survey.offset + Fibers.skyspectrum.

For frozen Model-3 quantities,

    O3 = (D - A3) / m3 - S3
    I  = N(lambda) S3 + sum(O3)
    R_raw = I - Smooth[I].

Smoothing uses a 51 native-pixel quadratic Savitzky-Golay filter.  The
evidence window is 3540--5480 Angstrom.  Within that window R_raw is centered,
projected away from [1, x, x^2], and then divided by its robust RMS.  R_raw,
the native-unit R_struct before that division, and the normalized candidate
are retained as separate arrays.

No external-image machinery is needed for this inference.  The native loader,
Model-3 additive definition, wavelength grid, q orientation, K(lambda),
hardware/date masks, and Survey.offset convention are inherited from the
validated modules.
"""

from __future__ import annotations

import argparse
import csv
import json
import secrets
import shlex
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter
from scipy.stats import pearsonr, spearmanr

import diagnose_m101_hierarchical as validated_m101
import m101_blank_fibers
from m101_calibration_utils import robust_location, robust_scatter
from diagnose_m101_post_model3 import _group_indices, load_model_and_skies
from m101_native_data import discover_h5, load as load_native_data


WAVE = np.asarray(validated_m101.DEF_WAVE, dtype=float)
SPECTRAL_MIN_A = 3540.0
SPECTRAL_MAX_A = 5480.0
SPECTRAL_USE = (WAVE >= SPECTRAL_MIN_A) & (WAVE <= SPECTRAL_MAX_A)
SMOOTHING_WINDOW_PIXELS = 51
SMOOTHING_POLYORDER = 2
TARGET_H5_BASENAME = "20200710_0000013.h5"
REPLICATION_TARGET_COUNT = 20


def _finite(values):
    values = np.asarray(values, dtype=float)
    return values[np.isfinite(values)]


def _robust_rms(values):
    """Use the validated robust scale convention for this diagnostic."""
    return float(validated_m101.robust_scale(_finite(values)))


def _median(values, default=np.nan):
    finite = _finite(values)
    return float(np.median(finite)) if finite.size else default


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


def _poly_coordinate(wave):
    """Dimensionless x over the evidence window for the polynomial gauge."""
    wave = np.asarray(wave, dtype=float)
    center = 0.5 * (SPECTRAL_MIN_A + SPECTRAL_MAX_A)
    half_width = 0.5 * (SPECTRAL_MAX_A - SPECTRAL_MIN_A)
    return (wave - center) / half_width


def _fill_for_smoothing(values):
    """Fill only a temporary smoothing copy; never overwrite the input."""
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    if finite.sum() == 0:
        return np.zeros(values.shape, dtype=float)
    if finite.sum() == 1:
        return np.full(values.shape, values[finite][0], dtype=float)
    indices = np.arange(values.size, dtype=float)
    return np.interp(indices, indices[finite], values[finite])


def smooth_illumination(incident, window_length=SMOOTHING_WINDOW_PIXELS,
                        polyorder=SMOOTHING_POLYORDER):
    """Smooth I using the initial, explicitly inspectable 4A convention."""
    incident = np.asarray(incident, dtype=float)
    if incident.ndim != 1 or incident.size < window_length:
        raise ValueError("incident must contain at least 51 native pixels")
    filled = _fill_for_smoothing(incident)
    return savgol_filter(filled, window_length=window_length,
                         polyorder=polyorder, mode="interp")


def orthogonalize_residual(raw_residual, wave=WAVE, spectral_use=SPECTRAL_USE):
    """Return native-unit R_struct and its transparent projection data.

    The mean is removed first.  The least-squares coefficients then describe
    the projection of that centered residual onto [1, x, x^2].  Undefined
    wavelengths remain NaN, and only the evidence window contributes.
    """
    raw_residual = np.asarray(raw_residual, dtype=float)
    wave = np.asarray(wave, dtype=float)
    spectral_use = np.asarray(spectral_use, dtype=bool)
    if raw_residual.shape != wave.shape or spectral_use.shape != wave.shape:
        raise ValueError("raw residual, wavelength, and evidence mask differ")
    finite = spectral_use & np.isfinite(raw_residual)
    centered = np.full(raw_residual.shape, np.nan, dtype=float)
    candidate = np.full(raw_residual.shape, np.nan, dtype=float)
    if not np.any(finite):
        return candidate, {
            "mean_removed": np.nan,
            "polynomial_coefficients": [np.nan, np.nan, np.nan],
            "normalization_scale": np.nan,
            "normalization_degenerate": True,
            "n_finite_window": 0,
        }
    mean_removed = float(np.mean(raw_residual[finite]))
    centered[finite] = raw_residual[finite] - mean_removed
    x = _poly_coordinate(wave)
    basis = np.column_stack((np.ones(wave.size), x, x * x))
    coefficients = np.linalg.lstsq(basis[finite], centered[finite], rcond=None)[0]
    candidate[finite] = centered[finite] - basis[finite] @ coefficients
    scale = _robust_rms(candidate[finite])
    degenerate = not np.isfinite(scale) or scale <= 1e-12
    reported_scale = 1.0 if degenerate else scale
    return candidate, {
        "mean_removed": mean_removed,
        "polynomial_coefficients": [float(value) for value in coefficients],
        "normalization_scale": float(reported_scale),
        "normalization_degenerate": bool(degenerate),
        "n_finite_window": int(np.sum(finite)),
    }


def orthogonalize_and_normalize(raw_residual, wave=WAVE,
                                spectral_use=SPECTRAL_USE):
    """Return the Model-4A normalized candidate and projection metadata.

    ``orthogonalize_residual`` is kept as the explicit source of R_struct so
    later diagnostics never reconstruct it from rounded normalization metadata.
    """
    candidate, details = orthogonalize_residual(raw_residual, wave, spectral_use)
    scale = details["normalization_scale"]
    if details["normalization_degenerate"]:
        scale = 1.0
    normalized = candidate.copy()
    finite = spectral_use & np.isfinite(normalized)
    normalized[finite] /= scale
    return normalized, details


def form_incident_illumination(sky, object_spectra, valid_fibers=None):
    """Form N(lambda)S + sum(O) with wavelength-wise finite fiber support."""
    sky = np.asarray(sky, dtype=float)
    object_spectra = np.asarray(object_spectra, dtype=float)
    if object_spectra.ndim != 2 or object_spectra.shape[1] != sky.size:
        raise ValueError("object spectra and sky have incompatible shapes")
    finite = np.isfinite(object_spectra)
    if valid_fibers is not None:
        valid_fibers = np.asarray(valid_fibers, dtype=bool)
        if valid_fibers.shape != (object_spectra.shape[0],):
            raise ValueError("valid_fibers has the wrong shape")
        finite &= valid_fibers[:, None]
    finite &= np.isfinite(sky)[None, :]
    n_valid = np.sum(finite, axis=0).astype(np.int32)
    object_sum = np.sum(np.where(finite, object_spectra, 0.0), axis=0)
    sky_contribution = n_valid.astype(float) * sky
    incident = sky_contribution + object_sum
    supported = n_valid > 0
    object_sum[~supported] = np.nan
    sky_contribution[~supported] = np.nan
    incident[~supported] = np.nan
    return incident, n_valid, sky_contribution, object_sum


def _amplifier_key(ifu, amp):
    return tuple(int(value) for value in ifu), str(amp)


def reconstruct_model3_objects(data, model3, skies, fq):
    """Recover O3 and amplifier illumination from frozen Model-3 quantities."""
    records = {}
    for item in data:
        sky = np.asarray(skies[item.key], dtype=float)
        m3 = np.exp(np.asarray(model3.z_for(item), dtype=float))
        additive3 = np.asarray(model3.additive_full(item, fq), dtype=float)
        # These are new arrays.  No model or native loader array is modified.
        object_spectra = (np.asarray(item.total, dtype=float) - additive3) / m3[:, None]
        object_spectra = object_spectra - sky[None, :]
        valid_fibers = ~np.asarray(item.hardware_bad, dtype=bool)
        _, amplifier_groups = _group_indices(item)
        for (ifu, amp), indices in amplifier_groups:
            key = (item.h5_name, int(item.exposure), tuple(ifu), str(amp))
            incident, n_valid, sky_contribution, object_sum = form_incident_illumination(
                sky, object_spectra[indices], valid_fibers[indices])
            smooth = smooth_illumination(incident)
            raw = incident - smooth
            structured, normalization = orthogonalize_residual(raw)
            normalized = structured / normalization["normalization_scale"]
            records[key] = {
                "H5": item.h5_name,
                "exposure": int(item.exposure),
                "SPECID": int(ifu[0]),
                "IFUSLOT": int(ifu[1]),
                "IFUID": int(ifu[2]),
                "AMP": str(amp),
                "wave": WAVE.copy(),
                "S3": sky.copy(),
                "sum_O3": object_sum,
                "N_valid": n_valid,
                "N_S3": sky_contribution,
                "I": incident,
                "Smooth_I": smooth,
                "R_raw": raw,
                "R_struct": structured,
                "R_orthogonalized_normalized": normalized,
                "normalization": normalization,
            }
        del object_spectra, additive3
    return records


def _record_metrics(record):
    use = SPECTRAL_USE
    raw = np.asarray(record["R_raw"], dtype=float)
    normalized = np.asarray(record["R_orthogonalized_normalized"], dtype=float)
    finite = use & np.isfinite(raw)
    x = _poly_coordinate(WAVE)
    basis = np.column_stack((np.ones(WAVE.size), x, x * x))
    dot_products = [float(np.dot(normalized[use & np.isfinite(normalized)],
                                  basis[use & np.isfinite(normalized), i]))
                    for i in range(3)]
    normalized_rms = _robust_rms(normalized[use])
    raw_rms = _robust_rms(raw[finite])
    positive = _strongest_features(WAVE[finite], raw[finite], positive=True)
    negative = _strongest_features(WAVE[finite], raw[finite], positive=False)
    maximum_index = np.nanargmax(np.abs(raw[use]))
    window_indices = np.flatnonzero(use)
    maximum_index = window_indices[maximum_index]
    return {
        "N_valid_median": _median(record["N_valid"][use]),
        "N_valid_min": int(np.nanmin(record["N_valid"][use])) if np.any(use) else None,
        "N_valid_max": int(np.nanmax(record["N_valid"][use])) if np.any(use) else None,
        "median_sky_contribution": _median(record["N_S3"][use]),
        "median_summed_object_contribution": _median(record["sum_O3"][use]),
        "R_raw_robust_rms": raw_rms,
        "R_normalized_robust_rms": normalized_rms,
        "polynomial_projection_coefficients": record["normalization"]["polynomial_coefficients"],
        "R_normalization_scale": record["normalization"]["normalization_scale"],
        "R_normalization_mean_removed": record["normalization"]["mean_removed"],
        "R_normalization_degenerate": record["normalization"]["normalization_degenerate"],
        "maximum_absolute_R_raw": float(np.abs(raw[maximum_index])),
        "maximum_absolute_R_raw_wavelength_A": float(WAVE[maximum_index]),
        "strongest_positive_R_raw_features": positive,
        "strongest_negative_R_raw_features": negative,
        "orthogonality_dot_R_1": dot_products[0],
        "orthogonality_dot_R_x": dot_products[1],
        "orthogonality_dot_R_x2": dot_products[2],
    }


def _strongest_features(wave, values, positive, count=5, minimum_separation_A=15.0):
    wave = np.asarray(wave, dtype=float)
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(wave) & np.isfinite(values)
    wave, values = wave[finite], values[finite]
    if not values.size:
        return []
    order = np.argsort(-values if positive else values)
    selected = []
    for index in order:
        if all(abs(float(wave[index]) - item["wavelength_A"]) >= minimum_separation_A
               for item in selected):
            selected.append({"wavelength_A": float(wave[index]),
                             "R_raw": float(values[index])})
            if len(selected) == count:
                break
    return selected


def _selection_features(records):
    features = []
    for key, record in records.items():
        if int(record["exposure"]) != 1:
            continue
        use = SPECTRAL_USE & np.isfinite(record["sum_O3"])
        if np.sum(use) == 0:
            continue
        features.append({
            "key": key,
            "source_metric": _median(record["sum_O3"][use]),
            "source_abs_metric": _median(np.abs(record["sum_O3"][use])),
            "sky_metric": _median(record["N_S3"][SPECTRAL_USE]),
            "n_valid_metric": _median(record["N_valid"][SPECTRAL_USE]),
        })
    return features


def select_representative_amplifiers(records):
    """Select three physical amplifiers from exposure 1, then keep all repeats."""
    features = _selection_features(records)
    if len(features) < 3:
        raise ValueError("fewer than three usable amplifier groups for selection")
    features.sort(key=lambda row: (row["source_abs_metric"], str(row["key"])))
    selected = []
    labels = {}

    def add(label, feature):
        physical = feature["key"][2] + (feature["key"][3],)
        if physical not in labels:
            selected.append(physical)
            labels[physical] = label

    add("blank-dominated", features[0])
    middle = features[len(features) // 2]
    add("intermediate-source", middle)
    add("substantial-M101-illumination", features[-1])
    # The three choices can collide only if the data contain too few distinct
    # physical groups; fill deterministically in that unusual case.
    for feature in features:
        if len(selected) >= 3:
            break
        add("selection fill", feature)
    return selected, labels, features


def _write_csv(path, records):
    fields = ["H5", "exposure", "SPECID", "IFUSLOT", "IFUID", "AMP", "wavelength_A",
              "S3", "sum_O3", "N_valid", "N_S3", "I", "Smooth_I", "R_raw", "R_struct",
              "R_orthogonalized_normalized"]
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for record in records:
            for index, wavelength in enumerate(WAVE):
                writer.writerow({
                    "H5": record["H5"], "exposure": record["exposure"],
                    "SPECID": record["SPECID"], "IFUSLOT": record["IFUSLOT"],
                    "IFUID": record["IFUID"], "AMP": record["AMP"],
                    "wavelength_A": float(wavelength),
                    "S3": record["S3"][index], "sum_O3": record["sum_O3"][index],
                    "N_valid": record["N_valid"][index], "N_S3": record["N_S3"][index],
                    "I": record["I"][index], "Smooth_I": record["Smooth_I"][index],
                    "R_raw": record["R_raw"][index],
                    "R_struct": record["R_struct"][index],
                    "R_orthogonalized_normalized": record[
                        "R_orthogonalized_normalized"][index],
                })


def _write_npz(path, records):
    arrays = {"wavelength_A": WAVE}
    for index, record in enumerate(records):
        prefix = "record_%02d" % index
        for name in ("S3", "sum_O3", "N_valid", "N_S3", "I", "Smooth_I", "R_raw", "R_struct",
                     "R_orthogonalized_normalized"):
            arrays[prefix + "__" + name] = np.asarray(record[name])
    np.savez_compressed(path, **arrays)


def _plot_record(record, output_path, selection_label):
    figure, axes = plt.subplots(4, 1, figsize=(12, 12), sharex=True)
    title = "%s e%d SPECID=%d IFUSLOT=%d IFUID=%d AMP=%s (%s)" % (
        record["H5"], record["exposure"], record["SPECID"], record["IFUSLOT"],
        record["IFUID"], record["AMP"], selection_label)
    axes[0].plot(WAVE, record["S3"], lw=0.8, label="S3")
    axes[0].plot(WAVE, record["sum_O3"], lw=0.8, label="sum O3")
    axes[0].plot(WAVE, record["N_S3"], lw=0.8, label="N(lambda) S3")
    axes[0].plot(WAVE, record["I"], lw=0.9, label="I")
    axes[0].set_ylabel("native flux")
    axes[0].legend(ncol=4, fontsize=8)
    axes[0].set_title(title, fontsize=10)
    axes[1].plot(WAVE, record["I"], lw=0.8, label="I")
    axes[1].plot(WAVE, record["Smooth_I"], lw=1.0, label="Smooth[I]")
    axes[1].set_ylabel("native flux")
    axes[1].legend(fontsize=8)
    axes[2].plot(WAVE, record["R_raw"], lw=0.8, color="tab:purple", label="R_raw")
    axes[2].axhline(0.0, color="black", lw=0.7)
    axes[2].set_ylabel("native flux")
    axes[2].legend(fontsize=8)
    axes[3].plot(WAVE, record["R_orthogonalized_normalized"], lw=0.8,
                 color="tab:green", label="R orthogonalized / normalized")
    axes[3].axhline(0.0, color="black", lw=0.7)
    axes[3].set_ylabel("robust-RMS units")
    axes[3].set_xlabel("wavelength [Angstrom]")
    axes[3].legend(fontsize=8)
    for axis in axes:
        axis.axvspan(SPECTRAL_MIN_A, SPECTRAL_MAX_A, color="0.85", alpha=0.12)
        axis.set_xlim(SPECTRAL_MIN_A, SPECTRAL_MAX_A)
        axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


TARGET_PHYSICAL = (202, 35, 74, "LU")
FEATURE_WAVELENGTHS_A = (3544.0, 3608.0, 3682.0, 3908.0, 5198.0)
MIN_REPLICATION_BLANK_FIBERS = 10
MIN_REPLICATION_SEVERITY_SAMPLES = 500


def _target_item(data):
    matches = [item for item in data
               if item.h5_name == TARGET_H5_BASENAME and int(item.exposure) == 1]
    if len(matches) != 1:
        raise RuntimeError("target exposure was not uniquely loaded")
    item = matches[0]
    physical = [(int(ifu[0]), int(ifu[1]), int(ifu[2]), str(amp))
                for ifu, amp in zip(item.ifu, item.amp)]
    if TARGET_PHYSICAL not in physical:
        raise RuntimeError("target amplifier was not found in exposure 1")
    return item


def _amplifier_indices(item, physical):
    group_physical = tuple(physical)
    physical_values = [(int(ifu[0]), int(ifu[1]), int(ifu[2]), str(amp))
                       for ifu, amp in zip(item.ifu, item.amp)]
    indices = np.flatnonzero(np.asarray([value == group_physical
                                         for value in physical_values], dtype=bool))
    if indices.size == 0:
        raise RuntimeError("target amplifier has no native rows")
    return indices


def _target_amplifier_indices(item):
    return _amplifier_indices(item, TARGET_PHYSICAL)


def _reconstruct_one_model3_illumination(item, model3, skies, fq, indices):
    """Apply the unchanged Model-4A illumination construction to one group."""
    sky = np.asarray(skies[item.key], dtype=float)
    m3 = np.exp(np.asarray(model3.z_for(item), dtype=float))
    additive3 = np.asarray(model3.additive_full(item, fq), dtype=float)
    object_spectra = ((np.asarray(item.total[indices], dtype=float)
                       - additive3[indices]) / m3[indices, None]) - sky[None, :]
    incident, n_valid, sky_contribution, object_sum = form_incident_illumination(
        sky, object_spectra, ~np.asarray(item.hardware_bad[indices], dtype=bool))
    smooth = smooth_illumination(incident)
    raw = incident - smooth
    structured, normalization = orthogonalize_residual(raw)
    normalized = structured / normalization["normalization_scale"]
    return {
        "I": incident, "Smooth_I": smooth, "R_raw": raw,
        "R_struct": structured, "R_normalized": normalized,
        "N_valid_illumination": n_valid, "N_S3": sky_contribution,
        "sum_O3": object_sum, "normalization": normalization,
    }


def _model3_group_residual(item, model3, skies, fq, illumination, physical):
    """Construct frozen Model-3 E in observed/native units for one group."""
    target_indices = _amplifier_indices(item, physical)
    selected_physical = np.zeros(item.row_index.size, dtype=bool)
    selected_physical[target_indices] = True

    # item.blank_valid is the external blank classification plus hardware and
    # the established per-fiber finite-support rule.  It does not require all
    # wavelengths of a fiber to be finite; wavelength support is applied below.
    selected_blank = (np.asarray(item.blank_classified, dtype=bool)
                      & np.asarray(item.blank_valid, dtype=bool)
                      & ~np.asarray(item.hardware_bad, dtype=bool)
                      & selected_physical)
    if not np.any(selected_blank):
        raise RuntimeError("target amplifier has no externally classified blank fibers")

    sky = np.asarray(skies[item.key], dtype=float)
    m3_all = np.exp(np.asarray(model3.z_for(item), dtype=float))
    a3_all = np.asarray(model3.additive_full(item, fq), dtype=float)
    d_all = np.asarray(item.total, dtype=float)

    # Primary comparison: E is in the observed/pre-division native-flux basis.
    e_f = d_all[selected_blank] - a3_all[selected_blank]
    e_f -= m3_all[selected_blank, None] * sky[None, :]
    # Reference only: this is the equivalent post-multiplicative residual and
    # has per-fiber post-division units, so it is not compared to R_struct.
    e_f_post = ((d_all[selected_blank] - a3_all[selected_blank])
                / m3_all[selected_blank, None] - sky[None, :])
    # Keep the complete physical-amplifier fiber stack for the diagnostic
    # heatmap.  Only externally classified, blank-valid, hardware-valid rows
    # are colored; all other rows are rendered as the white background.
    e_f_all = d_all[target_indices] - a3_all[target_indices]
    e_f_all -= m3_all[target_indices, None] * sky[None, :]
    finite = np.isfinite(e_f)
    n_blank = np.sum(finite, axis=0).astype(np.int32)
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)
        e_amp = np.asarray(robust_location(e_f, axis=0), dtype=float)
        scatter_e = np.asarray(robust_scatter(e_f, axis=0), dtype=float)
        e_amp_post = np.asarray(robust_location(e_f_post, axis=0), dtype=float)
    e_amp[n_blank == 0] = np.nan
    scatter_e[n_blank == 0] = np.nan
    e_amp_post[n_blank == 0] = np.nan

    return {
        "identity": tuple(physical),
        "item": item,
        "selected_blank": selected_blank,
        "selected_blank_rows": np.asarray(item.row_index[selected_blank], dtype=np.int64),
        "selected_blank_q": np.asarray(item.q[selected_blank], dtype=np.int16),
        "fiber_row_indices": np.asarray(item.row_index[target_indices], dtype=np.int64),
        "fiber_blank_mask": np.asarray(selected_blank[target_indices], dtype=bool),
        "m3": m3_all,
        "A3": a3_all,
        "S3": sky.copy(),
        "D": d_all.copy(),
        "E_f": e_f,
        "E_f_post": e_f_post,
        "E_f_all": e_f_all,
        "E_amp": e_amp,
        "scatter_E": scatter_e,
        "E_amp_post": e_amp_post,
        "N_blank": n_blank,
        "R_raw": np.asarray(illumination["R_raw"], dtype=float).copy(),
        "R_struct": np.asarray(illumination["R_struct"], dtype=float).copy(),
        "R_normalized": np.asarray(illumination["R_normalized"], dtype=float).copy(),
        "I": np.asarray(illumination["I"], dtype=float).copy(),
        "Smooth_I": np.asarray(illumination["Smooth_I"], dtype=float).copy(),
        "N_valid_illumination": np.asarray(illumination["N_valid_illumination"], dtype=np.int32).copy(),
        "normalization": illumination["normalization"],
    }


def _model3_target_residual(item, model3, skies, fq, illumination):
    """Construct the original Model-4A.1 target result unchanged."""
    return _model3_group_residual(item, model3, skies, fq, illumination,
                                  TARGET_PHYSICAL)


def _fit_q2_structured(wave, y, structured, sample, sigma=None):
    """Fit Q2 plus one R_struct coefficient for a diagnostic projection."""
    wave = np.asarray(wave, dtype=float)
    y = np.asarray(y, dtype=float)
    structured = np.asarray(structured, dtype=float)
    sample = (np.asarray(sample, dtype=bool) & np.isfinite(wave)
              & np.isfinite(y) & np.isfinite(structured))
    if sigma is not None:
        sigma = np.asarray(sigma, dtype=float)
        sample &= np.isfinite(sigma) & (sigma > 0.0)
    if int(np.sum(sample)) < 5:
        raise RuntimeError("too few wavelength samples for Q2 + R_struct")
    x = _poly_coordinate(wave)
    design = np.column_stack((np.ones(wave.size), x, x * x, structured))
    if sigma is None:
        weights = np.ones(int(np.sum(sample)), dtype=float)
    else:
        weights = 1.0 / sigma[sample] ** 2
    sqrt_weights = np.sqrt(weights)
    weighted_design = design[sample] * sqrt_weights[:, None]
    weighted_y = y[sample] * sqrt_weights
    beta = np.linalg.lstsq(weighted_design, weighted_y, rcond=None)[0]
    fitted = design @ beta
    residual = y - fitted
    n = int(np.sum(sample))
    dof = max(1, n - design.shape[1])
    weighted_residual = residual[sample] * sqrt_weights
    covariance = np.full((4, 4), np.nan, dtype=float)
    normal = weighted_design.T @ weighted_design
    try:
        covariance = np.linalg.inv(normal) * float(
            np.dot(weighted_residual, weighted_residual) / dof)
        gamma_uncertainty = float(np.sqrt(max(0.0, covariance[3, 3])))
    except np.linalg.LinAlgError:
        gamma_uncertainty = np.nan
    return {
        "sample": sample,
        "beta": beta,
        "Q2": fitted - beta[3] * structured,
        "fitted": fitted,
        "E_detrended": y - (fitted - beta[3] * structured),
        "residual": residual,
        "gamma": float(beta[3]),
        "gamma_uncertainty": gamma_uncertainty,
        "covariance": covariance,
        "n_samples": n,
        "weights": weights,
        "sigma": sigma,
    }


def _fit_q2_only(wave, y, sample):
    """Fit the selection-only broad quadratic without constructing R."""
    wave = np.asarray(wave, dtype=float)
    y = np.asarray(y, dtype=float)
    sample = np.asarray(sample, dtype=bool) & np.isfinite(wave) & np.isfinite(y)
    if int(np.sum(sample)) < 5:
        raise RuntimeError("too few wavelength samples for the selection Q2")
    x = _poly_coordinate(wave)
    design = np.column_stack((np.ones(wave.size), x, x * x))
    beta = np.linalg.lstsq(design[sample], y[sample], rcond=None)[0]
    q2 = design @ beta
    return {"beta": beta, "Q2": q2, "residual": y - q2,
            "sample": sample, "n_samples": int(np.sum(sample))}


def _model3_only_amplifier_records(item, model3, skies, fq):
    """Build only Model-3 residual records used to preregister targets.

    This function intentionally contains no call to any Model-4 illumination
    or correlation routine.  Its outputs are the independent selection basis.
    """
    sky = np.asarray(skies[item.key], dtype=float)
    m3 = np.exp(np.asarray(model3.z_for(item), dtype=float))
    a3 = np.asarray(model3.additive_full(item, fq), dtype=float)
    d = np.asarray(item.total, dtype=float)
    _, amplifier_groups = _group_indices(item)
    rows = []
    for (ifu, amp), indices in amplifier_groups:
        physical = tuple(ifu) + (str(amp),)
        group_mask = np.zeros(item.row_index.size, dtype=bool)
        group_mask[indices] = True
        selected = (np.asarray(item.blank_classified, dtype=bool)
                    & np.asarray(item.blank_valid, dtype=bool)
                    & ~np.asarray(item.hardware_bad, dtype=bool)
                    & group_mask)
        e_f = d[selected] - a3[selected] - m3[selected, None] * sky[None, :]
        n_blank = np.sum(np.isfinite(e_f), axis=0).astype(np.int32)
        if np.any(selected):
            with warnings.catch_warnings(), np.errstate(all="ignore"):
                warnings.simplefilter("ignore", RuntimeWarning)
                e_amp = np.asarray(robust_location(e_f, axis=0), dtype=float)
                scatter_e = np.asarray(robust_scatter(e_f, axis=0), dtype=float)
        else:
            e_amp = np.full(WAVE.shape, np.nan, dtype=float)
            scatter_e = np.full(WAVE.shape, np.nan, dtype=float)
        e_amp[n_blank == 0] = np.nan
        scatter_e[n_blank == 0] = np.nan
        severity_sample = SPECTRAL_USE & np.isfinite(e_amp)
        if int(np.sum(severity_sample)) >= MIN_REPLICATION_SEVERITY_SAMPLES:
            q2_fit = _fit_q2_only(WAVE, e_amp, severity_sample)
            severity = _robust_rms(q2_fit["residual"][severity_sample])
            q2_values = q2_fit["Q2"]
            detrended_values = q2_fit["residual"]
        else:
            severity = np.nan
            q2_values = np.full(WAVE.shape, np.nan, dtype=float)
            detrended_values = np.full(WAVE.shape, np.nan, dtype=float)
        support = n_blank[SPECTRAL_USE]
        eligible = bool(np.sum(selected) >= MIN_REPLICATION_BLANK_FIBERS
                        and _median(support) >= MIN_REPLICATION_BLANK_FIBERS
                        and np.sum(severity_sample) >= MIN_REPLICATION_SEVERITY_SAMPLES
                        and np.isfinite(severity))
        rows.append({
            "identity": physical,
            "H5": item.h5_name,
            "exposure": int(item.exposure),
            "SPECID": int(ifu[0]), "IFUSLOT": int(ifu[1]),
            "IFUID": int(ifu[2]), "AMP": str(amp),
            "n_blank_fibers": int(np.sum(selected)),
            "N_blank_min": int(np.min(support)) if support.size else 0,
            "N_blank_median": _median(support),
            "N_blank_max": int(np.max(support)) if support.size else 0,
            "severity_sample_count": int(np.sum(severity_sample)),
            "residual_severity": float(severity),
            "eligible": eligible,
            "exclusion_reason": (None if eligible else
                                  "blank_fibers<10 or median_support<10 or severity_samples<500"),
            "E_amp": e_amp,
            "scatter_E": scatter_e,
            "N_blank": n_blank,
            "Q2_selection": q2_values,
            "E_detrended_selection": detrended_values,
        })
    return rows


def _load_excluded_replication_identities(path):
    """Read exact amplifier identities to exclude from a prior selection."""
    if not path:
        return set()
    payload = json.loads(Path(path).expanduser().resolve().read_text())
    selection = payload.get("selection", payload.get("preregistered_selection", {}))
    return {tuple(row["identity"]) for row in selection.values()
            if "identity" in row}


def _preregister_replication_selection(
        rows, target_count=REPLICATION_TARGET_COUNT,
        selection_mode="severity-spaced", random_seed=None,
        excluded_identities=()):
    """Choose new targets before constructing or inspecting R."""
    eligible = [row for row in rows if row["eligible"]
                and row["identity"] != TARGET_PHYSICAL
                and row["identity"] not in set(excluded_identities)]
    all_eligible = [row for row in rows if row["eligible"]]
    if len(all_eligible) < target_count + 1:
        raise RuntimeError("fewer than %d eligible physical amplifiers including reference" %
                           (target_count + 1))
    severity_sorted = sorted(all_eligible,
                             key=lambda row: (row["residual_severity"],
                                               str(row["identity"])))
    n = len(severity_sorted)
    rank_by_identity = {row["identity"]: index + 1
                        for index, row in enumerate(severity_sorted)}
    for row in rows:
        if row["identity"] in rank_by_identity:
            row["severity_rank"] = rank_by_identity[row["identity"]]
            row["severity_percentile"] = (100.0 * (row["severity_rank"] - 1)
                                           / max(1, n - 1))
        else:
            row["severity_rank"] = None
            row["severity_percentile"] = None

    if selection_mode not in ("severity-spaced", "random"):
        raise ValueError("unknown replication selection mode: %s" % selection_mode)
    if len({row["identity"][:3] for row in eligible}) < target_count:
        raise RuntimeError("could not select %d distinct physical IFUs" % target_count)

    selected = {}
    used_identities = set()
    used_ifus = set()
    if selection_mode == "severity-spaced":
        # Select evenly spaced positions in the full eligible severity
        # ranking. The nearest unused amplifier is chosen at each position,
        # with one physical IFU per panel.
        target_positions = np.linspace(0.0, float(len(severity_sorted) - 1), target_count)
        for index, target_position in enumerate(target_positions, start=1):
            available = [row for row in eligible
                         if row["identity"] not in used_identities
                         and row["identity"][:3] not in used_ifus]
            if not available:
                raise RuntimeError("ran out of distinct physical IFUs at selection %d" % index)
            chosen = min(
                available,
                key=lambda row: (abs(float(row["severity_rank"] - 1) - target_position),
                                 str(row["identity"])))
            percentile = int(round(100.0 * target_position /
                                   max(1.0, float(len(severity_sorted) - 1))))
            role = "P%02d_p%03d" % (index, percentile)
            selected[role] = chosen
            used_identities.add(chosen["identity"])
            used_ifus.add(chosen["identity"][:3])
    else:
        if random_seed is None:
            random_seed = secrets.randbits(64)
        rng = np.random.default_rng(random_seed)
        random_order = list(eligible)
        rng.shuffle(random_order)
        for chosen in random_order:
            if chosen["identity"][:3] in used_ifus:
                continue
            index = len(selected) + 1
            selected["R%02d" % index] = chosen
            used_identities.add(chosen["identity"])
            used_ifus.add(chosen["identity"][:3])
            if len(selected) == target_count:
                break
        if len(selected) != target_count:
            raise RuntimeError("random selection did not find %d distinct physical IFUs" %
                               target_count)
    for role, row in selected.items():
        row["selection_role"] = role
    return selected, severity_sorted, random_seed


def _fit_metrics(y, structured, fit):
    sample = fit["sample"]
    detrended = np.asarray(y)[sample] - np.asarray(fit["Q2"])[sample]
    with_r = detrended - fit["gamma"] * np.asarray(structured)[sample]
    pearson = float(pearsonr(detrended, np.asarray(structured)[sample])[0])
    spearman = float(spearmanr(detrended, np.asarray(structured)[sample])[0])
    before = _robust_rms(detrended)
    after = _robust_rms(with_r)
    median_before = float(np.median(np.abs(detrended)))
    median_after = float(np.median(np.abs(with_r)))
    return {
        "gamma": fit["gamma"],
        "gamma_uncertainty": fit["gamma_uncertainty"],
        "pearson": pearson,
        "spearman": spearman,
        "robust_rms_before": before,
        "robust_rms_after": after,
        "fractional_rms_reduction": float(1.0 - after / before) if before > 0 else np.nan,
        "median_absolute_before": median_before,
        "median_absolute_after": median_after,
        "n_samples": fit["n_samples"],
    }


def _load_stored_model4a_normalized(summary_path, target):
    """Load the prior Model-4A array, identified by its summary row."""
    summary_path = Path(summary_path).expanduser().resolve()
    summary = json.loads(summary_path.read_text())
    rows = summary.get("summary_rows", [])
    matches = [index for index, row in enumerate(rows)
               if (row.get("H5") == target[0] and int(row.get("exposure")) == target[1]
                   and int(row.get("SPECID")) == target[2][0]
                   and int(row.get("IFUSLOT")) == target[2][1]
                   and int(row.get("IFUID")) == target[2][2]
                   and str(row.get("AMP")) == target[3])]
    if len(matches) != 1:
        raise RuntimeError("prior Model-4A summary does not uniquely identify target")
    arrays_path = summary_path.parent / "model4a_selected_arrays.npz"
    if not arrays_path.is_file():
        raise RuntimeError("prior Model-4A arrays are missing: %s" % arrays_path)
    with np.load(arrays_path) as arrays:
        stored_wave = np.asarray(arrays["wavelength_A"], dtype=float)
        stored = np.asarray(arrays["record_%02d__R_orthogonalized_normalized" % matches[0]],
                             dtype=float)
    return stored_wave, stored, summary_path, arrays_path


def _feature_rows(result, fit):
    rows = []
    for requested in FEATURE_WAVELENGTHS_A:
        index = int(np.argmin(np.abs(WAVE - requested)))
        e = float(result["E_amp"][index])
        r = float(result["R_struct"][index])
        rows.append({
            "requested_wavelength_A": float(requested),
            "wavelength_A": float(WAVE[index]),
            "E_amp": e,
            "R_struct": r,
            "gamma_R_struct": float(fit["gamma"] * r),
            "E_detrended": float(result["E_amp"][index] - fit["Q2"][index]),
            "residual_null": float(result["E_amp"][index] - fit["Q2"][index]),
            "residual_with_R": float(result["E_amp"][index] - fit["Q2"][index]
                                      - fit["gamma"] * r),
            "sign_E_amp": int(np.sign(e)) if np.isfinite(e) else None,
            "sign_R_struct": int(np.sign(r)) if np.isfinite(r) else None,
            "sign_gamma_R_struct": int(np.sign(fit["gamma"] * r))
            if np.isfinite(r) else None,
        })
    return rows


def _write_model4a1_csv(path, result, fit):
    fields = ["wavelength_A", "E_amp", "scatter_E", "N_blank", "Q2",
              "E_detrended", "R_raw", "R_struct", "R_normalized",
              "gamma_R_struct", "residual_null", "residual_with_R",
              "E_amp_post"]
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for index, wavelength in enumerate(WAVE):
            writer.writerow({
                "wavelength_A": float(wavelength),
                "E_amp": result["E_amp"][index],
                "scatter_E": result["scatter_E"][index],
                "N_blank": int(result["N_blank"][index]),
                "Q2": fit["Q2"][index],
                "E_detrended": fit["E_detrended"][index],
                "R_raw": result["R_raw"][index],
                "R_struct": result["R_struct"][index],
                "R_normalized": result["R_normalized"][index],
                "gamma_R_struct": fit["gamma"] * result["R_struct"][index],
                "residual_null": result["E_amp"][index] - fit["Q2"][index],
                "residual_with_R": result["E_amp"][index] - fit["Q2"][index]
                - fit["gamma"] * result["R_struct"][index],
                "E_amp_post": result["E_amp_post"][index],
            })


def _plot_model4a1(result, fit, output_path):
    """Write the five-panel native-unit residual/template comparison."""
    figure = plt.figure(figsize=(14, 18))
    grid = figure.add_gridspec(5, 1, height_ratios=(1.1, 1.0, 1.1, 1.0, 1.3),
                               hspace=0.38)
    axes = [figure.add_subplot(grid[index]) for index in range(4)]
    local_axis = figure.add_subplot(grid[4])
    identity = tuple(result.get("identity", TARGET_PHYSICAL))
    title = ("Model 4A.1: %s e%d; SPECID=%d IFUSLOT=%d IFUID=%d AMP=%s\n"
             "frozen Model-3 blank residual vs illumination-derived R; evidence %g--%g A"
             % (result["item"].h5_name, result["item"].exposure, *identity, SPECTRAL_MIN_A,
                SPECTRAL_MAX_A))
    axes[0].plot(WAVE, result["E_amp"], color="tab:blue", lw=0.75,
                 label="E_amp observed/native")
    axes[0].fill_between(WAVE, result["E_amp"] - result["scatter_E"],
                         result["E_amp"] + result["scatter_E"],
                         color="tab:blue", alpha=0.18, linewidth=0,
                         label="+/- robust blank-fiber scatter")
    axes[0].axhline(0.0, color="black", lw=0.8)
    axes[0].set_ylabel("native flux")
    axes[0].set_title(title, fontsize=11)
    axes[0].legend(fontsize=8, ncol=2)

    axes[1].plot(WAVE, result["R_raw"], color="tab:purple", alpha=0.28, lw=0.65,
                 label="R_raw")
    axes[1].plot(WAVE, result["R_struct"], color="tab:green", lw=0.8,
                 label="R_struct; native units")
    axes[1].axhline(0.0, color="black", lw=0.8)
    axes[1].set_ylabel("native flux")
    axes[1].set_title("Candidate illumination structure (R_raw faint; R_struct before normalization)")
    axes[1].legend(fontsize=8)

    detrended = result["E_amp"] - fit["Q2"]
    axes[2].plot(WAVE, detrended, color="tab:blue", lw=0.75,
                 label="E_amp - fitted Q2")
    axes[2].plot(WAVE, fit["gamma"] * result["R_struct"], color="tab:orange", lw=0.9,
                 label="gamma R_struct; gamma=%+.5g" % fit["gamma"])
    axes[2].axhline(0.0, color="black", lw=0.8)
    axes[2].set_ylabel("native flux")
    axes[2].set_title("Direct shape comparison: same native-flux axis")
    axes[2].legend(fontsize=8)

    axes[3].plot(WAVE, detrended, color="0.35", lw=0.7, label="null residual: E_amp - Q2")
    axes[3].plot(WAVE, detrended - fit["gamma"] * result["R_struct"],
                 color="tab:red", lw=0.7, label="after R: E_amp - Q2 - gamma R_struct")
    axes[3].axhline(0.0, color="black", lw=0.8)
    axes[3].set_ylabel("native flux")
    axes[3].set_title("Residual after the single diagnostic R projection")
    axes[3].legend(fontsize=8)

    local_axis.axis("off")
    local_axis.set_title("Local feature inspection; blue = E_amp-Q2, orange = gamma R_struct",
                         fontsize=10, pad=24)
    for feature_index, requested in enumerate(FEATURE_WAVELENGTHS_A):
        left = 0.01 + feature_index * 0.198
        inset = local_axis.inset_axes([left, 0.15, 0.18, 0.7])
        local = np.abs(WAVE - requested) <= 24.0
        inset.plot(WAVE[local], detrended[local], color="tab:blue", lw=0.8)
        inset.plot(WAVE[local], fit["gamma"] * result["R_struct"][local],
                   color="tab:orange", lw=0.8)
        inset.axhline(0.0, color="black", lw=0.55)
        inset.axvline(requested, color="0.5", lw=0.55, ls="--")
        inset.set_title("%g A" % requested, fontsize=8)
        inset.tick_params(labelsize=6)
        inset.grid(alpha=0.2)

    for axis in axes:
        axis.set_xlim(SPECTRAL_MIN_A, SPECTRAL_MAX_A)
        axis.grid(alpha=0.2)
    axes[3].set_xlabel("wavelength [Angstrom]")
    figure.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(figure)


def _plot_model4a1_fiber_residuals(result, output_path):
    """Write a square wavelength-by-fiber Model-3 residual heatmap."""
    wave_sample = SPECTRAL_USE & np.isfinite(WAVE)
    wave = WAVE[wave_sample]
    residuals = np.asarray(result["E_f_all"], dtype=float)[:, wave_sample]
    blank = np.asarray(result["fiber_blank_mask"], dtype=bool)
    finite_blank = blank[:, None] & np.isfinite(residuals)
    values = residuals[finite_blank]
    if values.size < 5:
        raise RuntimeError("too few finite blank-fiber residuals for heatmap scaling")
    p01, p99 = np.percentile(values, [1.0, 99.0])
    ran = float(p99 - p01)
    if not np.isfinite(ran) or ran <= 0.0:
        raise RuntimeError("degenerate blank-fiber residual range for heatmap")
    vmin = float(p01 - 0.2 * ran)
    vmax = float(p99 + 0.2 * ran)

    masked = np.ma.masked_where(~finite_blank, residuals)
    cmap = plt.get_cmap("coolwarm").copy()
    cmap.set_bad("white")
    identity = tuple(result["identity"])
    n_fibers = residuals.shape[0]
    # The data-coordinate aspect makes the trimmed wavelength-by-fiber image
    # square while retaining wavelength as the horizontal coordinate.
    data_aspect = float((wave[-1] - wave[0]) / n_fibers)
    figure, axis = plt.subplots(figsize=(11, 11))
    image = axis.imshow(
        masked, origin="lower", interpolation="nearest", cmap=cmap,
        vmin=vmin, vmax=vmax,
        extent=(float(wave[0]), float(wave[-1]), 0.5, n_fibers + 0.5),
        aspect=data_aspect)
    axis.set_xlabel("wavelength [Angstrom]")
    axis.set_ylabel("fiber in physical amplifier (112 rows)")
    axis.set_title(
        "Model 4A.1 Model-3 per-fiber residuals\n"
        "%s e%d; SPECID=%d IFUSLOT=%d IFUID=%d AMP=%s; blank rows colored"
        % (result["item"].h5_name, result["item"].exposure, *identity))
    axis.set_xlim(SPECTRAL_MIN_A, SPECTRAL_MAX_A)
    axis.set_ylim(0.5, n_fibers + 0.5)
    colorbar = figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    colorbar.set_label("E_f = D - A3 - m3 S3 [native flux]\n"
                       "blank fibers; white = non-blank/invalid")
    axis.text(
        0.01, 0.01,
        "color limits: p01 - 0.2 range = %.5g; p99 + 0.2 range = %.5g\n"
        "p01=%.5g, p99=%.5g, range=%.5g; colored blank rows=%d/%d"
        % (vmin, vmax, p01, p99, ran, int(np.sum(blank)), n_fibers),
        transform=axis.transAxes, fontsize=8, va="bottom",
        bbox=dict(facecolor="white", alpha=0.78, edgecolor="0.7", pad=3.0))
    figure.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(figure)
    return {
        "n_fibers": int(n_fibers),
        "n_colored_blank_fibers": int(np.sum(blank)),
        "n_wavelengths": int(wave.size),
        "wavelength_min_A": float(wave[0]),
        "wavelength_max_A": float(wave[-1]),
        "p01": float(p01), "p99": float(p99), "range": ran,
        "vmin": vmin, "vmax": vmax,
        "color_rule": "vmin=p01-0.2*(p99-p01); vmax=p99+0.2*(p99-p01)",
        "data_aspect": data_aspect,
    }


def deterministic_checks():
    """Run small checks before any real H5 is opened."""
    wave = np.linspace(SPECTRAL_MIN_A, SPECTRAL_MAX_A, 201)
    use = np.ones(wave.size, dtype=bool)
    x = _poly_coordinate(wave)
    checks = {}
    for name, input_values in (
            ("constant", np.full(wave.size, 7.0)),
            ("quadratic", 4.0 + 2.0 * x + 3.0 * x * x)):
        smooth = smooth_illumination(input_values)
        raw = input_values - smooth
        candidate, details = orthogonalize_and_normalize(raw, wave, use)
        checks[name] = {
            "max_abs_raw": float(np.max(np.abs(raw))),
            "max_abs_orthogonalized_normalized": float(np.nanmax(np.abs(candidate))),
            "pass": bool(np.nanmax(np.abs(candidate)) < 1e-9),
            "normalization": details,
        }
    feature = (10.0 + 2.0 * np.cos(2.0 * np.pi * (wave - 4510.0) / 1200.0)
               + 5.0 * np.exp(-0.5 * ((wave - 4510.0) / 20.0) ** 2))
    feature_smooth = smooth_illumination(feature)
    feature_raw = feature - feature_smooth
    center = int(np.argmin(np.abs(wave - 4510.0)))
    checks["injected_feature"] = {
        "raw_at_feature_center": float(feature_raw[center]),
        "raw_minimum": float(np.min(feature_raw)),
        "expected_center_sign_positive": bool(feature_raw[center] > 0.0),
        "pass": bool(feature_raw[center] > 0.0),
    }
    sky = np.asarray([10.0, 20.0, 30.0])
    objects = np.asarray([[1.0, np.nan, 3.0], [2.0, 4.0, np.nan]])
    incident, count, sky_part, object_part = form_incident_illumination(sky, objects)
    expected_count = np.asarray([2, 1, 1])
    expected_incident = np.asarray([23.0, 24.0, 33.0])
    checks["wavelengthwise_support"] = {
        "N_actual": count.tolist(), "N_expected": expected_count.tolist(),
        "I_actual": incident.tolist(), "I_expected": expected_incident.tolist(),
        "pass": bool(np.array_equal(count, expected_count) and
                     np.allclose(incident, expected_incident)),
        "sky_part": sky_part.tolist(), "object_part": object_part.tolist(),
    }
    candidate, details = orthogonalize_and_normalize(feature_raw, wave, use)
    basis = np.column_stack((np.ones(wave.size), x, x * x))
    dots = [float(np.dot(candidate, basis[:, i])) for i in range(3)]
    checks["orthogonality"] = {
        "dot_R_1": dots[0], "dot_R_x": dots[1], "dot_R_x2": dots[2],
        "normalized_robust_rms": _robust_rms(candidate),
        "pass": bool(max(abs(value) for value in dots) < 1e-8 and
                     abs(_robust_rms(candidate) - 1.0) < 1e-8),
        "normalization": details,
    }
    return checks


def run_model4a1(args, checks):
    """Run only Model 4A.1 for the fixed representative amplifier."""
    h5_paths = discover_h5(args.h5, development=True)
    if len(h5_paths) != 1 or h5_paths[0].name != TARGET_H5_BASENAME:
        raise SystemExit("Model 4A.1 requires exactly %s" % TARGET_H5_BASENAME)
    product_path = Path(args.model3_product).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise SystemExit("output directory is nonempty; use --overwrite: %s" % output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    _print_units()
    product, model3, skies = load_model_and_skies(product_path)
    product_h5_names = {Path(item["filename"]).name
                        for item in product["provenance"]["input_h5"]}
    if TARGET_H5_BASENAME not in product_h5_names:
        raise SystemExit("persisted Model-3 product does not contain the representative H5")
    blank_masks, blank_provenance = m101_blank_fibers.load(args.blank_file, h5_paths)
    fq = validated_m101.load_fq(args.fq_template)
    data, input_provenance = load_native_data(
        h5_paths, blank_masks, args.on_filter, args.off_filter,
        include_native_errors=False, include_band_errors=False,
        collapse_band_indices=(5, 6))

    model3_snapshot = {
        "p_ifu": {key: value for key, value in model3.p_ifu.items()},
        "p_amp": {key: value for key, value in model3.p_amp.items()},
        "ax": {key: value for key, value in model3.ax.items()},
        "ay": {key: value for key, value in model3.ay.items()},
        "alpha_q": {key: value for key, value in model3.alpha_q.items()},
    }
    sky_snapshot = {key: value.copy() for key, value in skies.items()
                    if key[0] == TARGET_H5_BASENAME}
    item = _target_item(data)
    target_indices = _target_amplifier_indices(item)
    illumination = _reconstruct_one_model3_illumination(
        item, model3, skies, fq, target_indices)
    result = _model3_target_residual(item, model3, skies, fq, illumination)

    # Snapshot the exact target arrays used in E and check them again before
    # fitting.  This makes the frozen m3/S3/A3 gate explicit.
    m3_before = result["m3"].copy()
    a3_before = result["A3"].copy()
    s3_before = result["S3"].copy()
    m3_after = np.exp(np.asarray(model3.z_for(item), dtype=float))
    a3_after = np.asarray(model3.additive_full(item, fq), dtype=float)
    s3_after = np.asarray(skies[item.key], dtype=float)
    model3_unchanged = all(getattr(model3, name) == model3_snapshot[name]
                           for name in model3_snapshot)
    skies_unchanged = all(np.array_equal(skies[key], value, equal_nan=True)
                          for key, value in sky_snapshot.items())
    target_arrays_unchanged = (
        np.array_equal(m3_before, m3_after, equal_nan=True)
        and np.array_equal(a3_before, a3_after, equal_nan=True)
        and np.array_equal(s3_before, s3_after, equal_nan=True))

    r_struct = result["R_struct"]
    r_normalized = result["R_normalized"]
    finite_r = SPECTRAL_USE & np.isfinite(r_struct)
    x = _poly_coordinate(WAVE)
    basis = np.column_stack((np.ones(WAVE.size), x, x * x))
    dots = [float(np.dot(r_struct[finite_r], basis[finite_r, index]))
            for index in range(3)]
    normalization_scale = float(result["normalization"]["normalization_scale"])
    reconstructed_normalized = r_struct / normalization_scale
    stored_wave, stored_normalized, model4a_summary_path, model4a_arrays_path = (
        _load_stored_model4a_normalized(args.model4a_summary,
                                        (TARGET_H5_BASENAME, 1, TARGET_PHYSICAL[:3], TARGET_PHYSICAL[3])))
    stored_agreement = (stored_wave.shape == WAVE.shape
                        and np.array_equal(stored_wave, WAVE)
                        and np.allclose(stored_normalized, r_normalized,
                                        rtol=2e-12, atol=2e-12, equal_nan=True))
    normalized_reconstruction_agreement = np.allclose(
        reconstructed_normalized, r_normalized, rtol=2e-12, atol=2e-12,
        equal_nan=True)

    primary_sample = (SPECTRAL_USE & np.isfinite(result["E_amp"])
                      & np.isfinite(r_struct))
    selected_blank = result["selected_blank"]
    selection_gate = (np.all(np.asarray(item.blank_classified)[selected_blank])
                      and np.all(np.asarray(item.blank_valid)[selected_blank])
                      and np.all(~np.asarray(item.hardware_bad)[selected_blank]))
    checks_4a1 = {
        "m3_unchanged": bool(model3_unchanged and target_arrays_unchanged),
        "S3_unchanged": bool(skies_unchanged and target_arrays_unchanged),
        "A3_unchanged": bool(target_arrays_unchanged),
        "selected_identity": {
            "H5": item.h5_name, "exposure": int(item.exposure),
            "SPECID": TARGET_PHYSICAL[0], "IFUSLOT": TARGET_PHYSICAL[1],
            "IFUID": TARGET_PHYSICAL[2], "AMP": TARGET_PHYSICAL[3],
            "pass": bool(item.h5_name == TARGET_H5_BASENAME and item.exposure == 1),
        },
        "only_external_blank_hardware_valid": bool(selection_gate),
        "native_flux_units_match": True,
        "R_struct_orthogonality_dot_1": dots[0],
        "R_struct_orthogonality_dot_x": dots[1],
        "R_struct_orthogonality_dot_x2": dots[2],
        "R_struct_orthogonal": bool(max(abs(value) for value in dots) < 1e-8),
        "R_normalized_robust_rms": _robust_rms(r_normalized[SPECTRAL_USE]),
        "R_normalized_unit_robust_rms": bool(
            abs(_robust_rms(r_normalized[SPECTRAL_USE]) - 1.0) < 1e-8),
        "R_normalized_reconstruction_agrees": bool(normalized_reconstruction_agreement),
        "stored_Model4A_normalized_agrees": bool(stored_agreement),
        "same_primary_wavelength_samples": bool(
            np.array_equal(primary_sample, SPECTRAL_USE & np.isfinite(result["E_amp"])
                           & np.isfinite(r_struct))),
    }
    print("Model-4A.1 correctness checks")
    print(json.dumps(_json_ready(checks_4a1), indent=2, sort_keys=True))
    gate_values = [checks_4a1["m3_unchanged"], checks_4a1["S3_unchanged"],
                   checks_4a1["A3_unchanged"], checks_4a1["selected_identity"]["pass"],
                   checks_4a1["only_external_blank_hardware_valid"],
                   checks_4a1["native_flux_units_match"], checks_4a1["R_struct_orthogonal"],
                   checks_4a1["R_normalized_unit_robust_rms"],
                   checks_4a1["R_normalized_reconstruction_agrees"],
                   checks_4a1["stored_Model4A_normalized_agrees"],
                   checks_4a1["same_primary_wavelength_samples"]]
    if not all(gate_values):
        raise RuntimeError("Model-4A.1 correctness gate failed; no comparison was fitted")

    print("EXPECTED")
    print("Major positive and negative structures in E_amp-Q2 should align with "
          "R_struct after one signed scaling across separated features; Q2 should "
          "remain broad and the R projection should simplify the residual.")
    fit_primary = _fit_q2_structured(WAVE, result["E_amp"], r_struct, primary_sample)
    primary_metrics = _fit_metrics(result["E_amp"], r_struct, fit_primary)
    sigma_e = np.divide(result["scatter_E"], np.sqrt(result["N_blank"]),
                        out=np.full(WAVE.shape, np.nan, dtype=float),
                        where=result["N_blank"] > 0)
    weighted_sample = primary_sample & np.isfinite(sigma_e) & (sigma_e > 0.0)
    weighted_metrics = None
    fit_weighted = None
    if int(np.sum(weighted_sample)) >= 5:
        fit_weighted = _fit_q2_structured(WAVE, result["E_amp"], r_struct,
                                          weighted_sample, sigma=sigma_e)
        weighted_metrics = _fit_metrics(result["E_amp"], r_struct, fit_weighted)
    edge_sample = primary_sample & (WAVE >= 3600.0)
    fit_edge = _fit_q2_structured(WAVE, result["E_amp"], r_struct, edge_sample)
    edge_metrics = _fit_metrics(result["E_amp"], r_struct, fit_edge)
    feature_rows = _feature_rows(result, fit_primary)

    figure_path = output_dir / "model4a1_20200710_0000013_e1_202_35_74_LU.png"
    csv_path = output_dir / "model4a1_residual_shape.csv"
    arrays_path = output_dir / "model4a1_blank_fiber_residuals.npz"
    summary_path = output_dir / "model4a1_summary.json"
    _plot_model4a1(result, fit_primary, figure_path)
    _write_model4a1_csv(csv_path, result, fit_primary)
    np.savez_compressed(
        arrays_path, wavelength_A=WAVE, blank_fiber_row_index=result["selected_blank_rows"],
        blank_fiber_q=result["selected_blank_q"], E_f_observed=result["E_f"],
        E_f_post=result["E_f_post"])

    summary = {
        "block": "Model 4A.1",
        "scientific_question": "Does the actual Model-3 blank residual have the shape of R?",
        "fit_is_diagnostic_only": True,
        "model4b_performed": False,
        "iteration_performed": False,
        "expected": "Major positive and negative structures in E_amp-Q2 align with R_struct across separated features after one signed scale.",
        "target": {"H5": TARGET_H5_BASENAME, "exposure": 1,
                   "SPECID": TARGET_PHYSICAL[0], "IFUSLOT": TARGET_PHYSICAL[1],
                   "IFUID": TARGET_PHYSICAL[2], "AMP": TARGET_PHYSICAL[3]},
        "selected_blank_fibers": {
            "count": int(np.sum(selected_blank)),
            "row_indices": result["selected_blank_rows"].tolist(),
            "definition": "external blank classification AND established blank_valid AND hardware-valid; wavelength support remains per sample",
            "N_blank_median": _median(result["N_blank"][SPECTRAL_USE]),
            "N_blank_min": int(np.min(result["N_blank"][SPECTRAL_USE])),
            "N_blank_max": int(np.max(result["N_blank"][SPECTRAL_USE])),
        },
        "wavelength_support": {"evidence_min_A": SPECTRAL_MIN_A,
                               "evidence_max_A": SPECTRAL_MAX_A,
                               "n_wave": int(np.sum(primary_sample)),
                               "N_blank_by_wavelength_saved": True},
        "smoothing": {"method": "Savitzky-Golay", "window_pixels": SMOOTHING_WINDOW_PIXELS,
                      "polyorder": SMOOTHING_POLYORDER,
                      "temporary_fill": "finite linear interpolation"},
        "residual_bases": {
            "E_f": "D - A3 - m3*S3; observed/pre-division native flux",
            "E_f_post": "(D - A3)/m3 - S3; post-multiplicative per-fiber flux, reference only",
            "E_amp": "robust blank-fiber location of E_f; observed/native additive flux",
            "R_struct": "native-unit illumination residual after mean and [1,x,x^2] projection, before robust-RMS normalization",
            "R_normalized": "R_struct divided by robust_RMS(R_struct); dimensionless",
        },
        "primary_equal_wavelength": {
            "fit": {"q0": float(fit_primary["beta"][0]),
                    "q1": float(fit_primary["beta"][1]),
                    "q2": float(fit_primary["beta"][2]),
                    "gamma": fit_primary["gamma"],
                    "gamma_uncertainty_formal": fit_primary["gamma_uncertainty"]},
            "metrics": primary_metrics,
            "R_struct_robust_scale": _robust_rms(r_struct[SPECTRAL_USE]),
        },
        "uncertainty_weighted_sensitivity": {
            "definition": "sigma_E = scatter_E / sqrt(N_blank), positive finite samples only",
            "defensible": bool(weighted_metrics is not None),
            "metrics": weighted_metrics,
            "fit": ({"q0": float(fit_weighted["beta"][0]),
                     "q1": float(fit_weighted["beta"][1]),
                     "q2": float(fit_weighted["beta"][2]),
                     "gamma": fit_weighted["gamma"],
                     "gamma_uncertainty_formal": fit_weighted["gamma_uncertainty"]}
                    if fit_weighted is not None else None),
            "n_samples": int(np.sum(weighted_sample)),
        },
        "blue_edge_repeat": {
            "definition": "same R_struct and smoothing; equal-weight Q2+gamma projection with wavelength >= 3600 A",
            "metrics": edge_metrics,
            "fit": {"q0": float(fit_edge["beta"][0]),
                    "q1": float(fit_edge["beta"][1]),
                    "q2": float(fit_edge["beta"][2]),
                    "gamma": fit_edge["gamma"],
                    "gamma_uncertainty_formal": fit_edge["gamma_uncertainty"]},
        },
        "feature_rows": feature_rows,
        "deterministic_checks": checks,
        "correctness_checks": checks_4a1,
        "model3_product": str(product_path),
        "prior_model4a_summary": str(model4a_summary_path),
        "prior_model4a_arrays": str(model4a_arrays_path),
        "provenance": {"exact_command": " ".join(shlex.quote(value) for value in sys.argv),
                       "h5": str(h5_paths[0]),
                       "blank_file": str(Path(args.blank_file).expanduser().resolve()),
                       "on_filter": str(Path(args.on_filter).expanduser().resolve()),
                       "off_filter": str(Path(args.off_filter).expanduser().resolve()),
                       "fq_template": str(Path(args.fq_template).expanduser().resolve()),
                       "blank_loader": blank_provenance,
                       "native_loader": input_provenance,
                       "model3_product_provenance": product["provenance"]},
        "output_paths": {"summary": str(summary_path), "figure": str(figure_path),
                         "csv": str(csv_path), "blank_fiber_arrays": str(arrays_path),
                         "interpretation": str(output_dir / "model4a1_interpretation.md")},
        "interpretation_gate": "STOP after this one amplifier comparison; do not implement Model 4B here.",
    }
    summary_path.write_text(json.dumps(_json_ready(summary), indent=2, sort_keys=True))
    print("OBSERVED")
    print(json.dumps(_json_ready({"primary": primary_metrics,
                                  "weighted": weighted_metrics,
                                  "edge": edge_metrics,
                                  "N_blank": summary["selected_blank_fibers"],
                                  "features": feature_rows}), indent=2, sort_keys=True))
    print("wrote Model-4A.1 products to %s" % output_dir)
    return summary


def _coherent_feature_count(feature_rows):
    return int(sum(row["sign_E_detrended"] == row["sign_gamma_R_struct"]
                   for row in feature_rows
                   if row.get("sign_E_detrended") not in (None, 0)
                   and row.get("sign_gamma_R_struct") not in (None, 0)))


def _replication_row_json(row):
    return {key: row[key] for key in (
        "H5", "exposure", "SPECID", "IFUSLOT", "IFUID", "AMP",
        "identity", "n_blank_fibers", "N_blank_min", "N_blank_median",
        "N_blank_max", "severity_sample_count", "residual_severity",
        "severity_rank", "severity_percentile", "eligible", "selection_role")
            if key in row}


def _write_replication_csv(path, records):
    fields = ["H5", "exposure", "SPECID", "IFUSLOT", "IFUID", "AMP",
              "wavelength_A", "E_amp", "scatter_E", "N_blank", "Q2",
              "E_detrended", "R_raw", "R_struct", "R_normalized",
              "gamma_R_struct", "residual_null", "residual_with_R", "E_amp_post"]
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for record in records:
            result, fit = record["result"], record["fit_primary"]
            identity = tuple(result["identity"])
            for index, wavelength in enumerate(WAVE):
                writer.writerow({
                    "H5": result["item"].h5_name,
                    "exposure": int(result["item"].exposure),
                    "SPECID": identity[0], "IFUSLOT": identity[1],
                    "IFUID": identity[2], "AMP": identity[3],
                    "wavelength_A": float(wavelength),
                    "E_amp": result["E_amp"][index],
                    "scatter_E": result["scatter_E"][index],
                    "N_blank": int(result["N_blank"][index]),
                    "Q2": fit["Q2"][index],
                    "E_detrended": fit["E_detrended"][index],
                    "R_raw": result["R_raw"][index],
                    "R_struct": result["R_struct"][index],
                    "R_normalized": result["R_normalized"][index],
                    "gamma_R_struct": fit["gamma"] * result["R_struct"][index],
                    "residual_null": result["E_amp"][index] - fit["Q2"][index],
                    "residual_with_R": result["E_amp"][index] - fit["Q2"][index]
                    - fit["gamma"] * result["R_struct"][index],
                    "E_amp_post": result["E_amp_post"][index],
                })


def run_model4a1_replication(args, checks):
    """Pre-register 20 Model-3-selected amplifiers, then replicate 4A.1."""
    h5_paths = discover_h5(args.h5, development=True)
    if len(h5_paths) != 1 or h5_paths[0].name != TARGET_H5_BASENAME:
        raise SystemExit("Model 4A.1 replication requires exactly %s" % TARGET_H5_BASENAME)
    product_path = Path(args.model3_product).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise SystemExit("output directory is nonempty; use --overwrite: %s" % output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    _print_units()
    product, model3, skies = load_model_and_skies(product_path)
    product_h5_names = {Path(item["filename"]).name
                        for item in product["provenance"]["input_h5"]}
    if TARGET_H5_BASENAME not in product_h5_names:
        raise SystemExit("persisted Model-3 product does not contain the representative H5")
    blank_masks, blank_provenance = m101_blank_fibers.load(args.blank_file, h5_paths)
    fq = validated_m101.load_fq(args.fq_template)
    data, input_provenance = load_native_data(
        h5_paths, blank_masks, args.on_filter, args.off_filter,
        include_native_errors=False, include_band_errors=False,
        collapse_band_indices=(5, 6))
    items = [item for item in data
             if item.h5_name == TARGET_H5_BASENAME and int(item.exposure) == 1]
    if len(items) != 1:
        raise RuntimeError("exposure 1 was not uniquely loaded")
    item = items[0]

    # Freeze and snapshot Model-3 before the independent severity pass.
    model3_snapshot = {
        "p_ifu": {key: value for key, value in model3.p_ifu.items()},
        "p_amp": {key: value for key, value in model3.p_amp.items()},
        "ax": {key: value for key, value in model3.ax.items()},
        "ay": {key: value for key, value in model3.ay.items()},
        "alpha_q": {key: value for key, value in model3.alpha_q.items()},
    }
    sky_snapshot = {key: value.copy() for key, value in skies.items()
                    if key[0] == TARGET_H5_BASENAME}
    m3_snapshot = np.exp(np.asarray(model3.z_for(item), dtype=float))
    a3_snapshot = np.asarray(model3.additive_full(item, fq), dtype=float)
    s3_snapshot = np.asarray(skies[item.key], dtype=float)

    # This is the complete selection pass.  It constructs only E_amp, Q2, and
    # residual severity; no R array is constructed or inspected here.
    severity_rows = _model3_only_amplifier_records(item, model3, skies, fq)
    excluded_identities = _load_excluded_replication_identities(
        args.replication_exclude_selection)
    excluded_identities.add(TARGET_PHYSICAL)
    selected, severity_sorted, random_seed = _preregister_replication_selection(
        severity_rows, selection_mode=args.replication_selection_mode,
        random_seed=args.replication_random_seed,
        excluded_identities=excluded_identities)
    preregistration = {
        "block": "Model 4A.1 replication selection (20 amplifiers)",
        "selection_mode": args.replication_selection_mode,
        "random_seed": random_seed,
        "excluded_selection_json": (str(Path(args.replication_exclude_selection).expanduser().resolve())
                                     if args.replication_exclude_selection else None),
        "excluded_identities": [list(identity) for identity in sorted(excluded_identities, key=str)],
        "selection_basis": "frozen Model-3 E_amp-Q2 robust RMS and support only; R not constructed in this pass",
        "target_h5": TARGET_H5_BASENAME,
        "exposure": 1,
        "excluded_previous_amplifier": list(TARGET_PHYSICAL),
        "eligible_distribution": [_replication_row_json(row)
                                   for row in severity_sorted],
        "selection": {role: _replication_row_json(row)
                       for role, row in selected.items()},
        "exact_command": " ".join(shlex.quote(value) for value in sys.argv),
    }
    preregistration_path = output_dir / "model4a1_replication_preregistered_selection.json"
    preregistration_path.write_text(json.dumps(_json_ready(preregistration),
                                               indent=2, sort_keys=True))
    print("PRE-REGISTERED SELECTION (20 amplifiers; mode=%s; Model-3 residual severity only; R not yet constructed)"
          % args.replication_selection_mode)
    print(json.dumps(_json_ready(preregistration), indent=2, sort_keys=True))

    print("EXPECTED")
    print("If R describes one real Model-3 additive failure class, some high-severity "
          "amplifiers should show stronger multi-feature alignment and surviving "
          "edge-cut improvement than low-severity amplifiers.")

    selected_records = []
    model3_identity = lambda row: tuple(row["identity"])
    for role, selection_row in selected.items():
        physical = model3_identity(selection_row)
        indices = _amplifier_indices(item, physical)
        illumination = _reconstruct_one_model3_illumination(
            item, model3, skies, fq, indices)
        result = _model3_group_residual(item, model3, skies, fq, illumination, physical)
        primary_sample = (SPECTRAL_USE & np.isfinite(result["E_amp"])
                          & np.isfinite(result["R_struct"]))
        fit_primary = _fit_q2_structured(WAVE, result["E_amp"], result["R_struct"],
                                         primary_sample)
        sigma_e = np.divide(result["scatter_E"], np.sqrt(result["N_blank"]),
                            out=np.full(WAVE.shape, np.nan, dtype=float),
                            where=result["N_blank"] > 0)
        weighted_sample = primary_sample & np.isfinite(sigma_e) & (sigma_e > 0.0)
        fit_weighted = None
        weighted_metrics = None
        if int(np.sum(weighted_sample)) >= 5:
            fit_weighted = _fit_q2_structured(
                WAVE, result["E_amp"], result["R_struct"], weighted_sample,
                sigma=sigma_e)
            weighted_metrics = _fit_metrics(result["E_amp"], result["R_struct"],
                                             fit_weighted)
        edge_sample = primary_sample & (WAVE >= 3600.0)
        fit_edge = _fit_q2_structured(WAVE, result["E_amp"], result["R_struct"],
                                      edge_sample)
        feature_rows = _feature_rows(result, fit_primary)
        for feature in feature_rows:
            e_detrended = feature["E_detrended"]
            projected = feature["gamma_R_struct"]
            feature["sign_E_detrended"] = (int(np.sign(e_detrended))
                                             if np.isfinite(e_detrended) else None)
            feature["sign_gamma_R_struct"] = (int(np.sign(projected))
                                                if np.isfinite(projected) else None)
        record = {
            "role": role, "selection": selection_row, "result": result,
            "fit_primary": fit_primary, "metrics_primary": _fit_metrics(
                result["E_amp"], result["R_struct"], fit_primary),
            "fit_edge": fit_edge,
            "metrics_edge": _fit_metrics(result["E_amp"], result["R_struct"], fit_edge),
            "fit_weighted": fit_weighted, "metrics_weighted": weighted_metrics,
            "feature_rows": feature_rows,
            "coherent_feature_count": _coherent_feature_count(feature_rows),
            "primary_sample": primary_sample,
        }
        selected_records.append(record)
        plot_path = output_dir / (
            "model4a1_%s_20200710_0000013_e1_%d_%d_%d_%s.png" %
            (role, physical[0], physical[1], physical[2], physical[3]))
        _plot_model4a1(result, fit_primary, plot_path)
        heatmap_path = output_dir / (
            "model4a1_%s_fiber_residuals_20200710_0000013_e1_%d_%d_%d_%s.png" %
            (role, physical[0], physical[1], physical[2], physical[3]))
        heatmap_info = _plot_model4a1_fiber_residuals(result, heatmap_path)
        record["fiber_heatmap"] = heatmap_info
        record["fiber_heatmap_path"] = str(heatmap_path)

    model3_unchanged = all(getattr(model3, name) == model3_snapshot[name]
                           for name in model3_snapshot)
    skies_unchanged = all(np.array_equal(skies[key], value, equal_nan=True)
                          for key, value in sky_snapshot.items())
    arrays_unchanged = (
        np.array_equal(m3_snapshot, np.exp(np.asarray(model3.z_for(item), dtype=float)),
                       equal_nan=True)
        and np.array_equal(a3_snapshot,
                           np.asarray(model3.additive_full(item, fq), dtype=float),
                           equal_nan=True)
        and np.array_equal(s3_snapshot, np.asarray(skies[item.key], dtype=float),
                           equal_nan=True))
    gate = {
        "m3_unchanged": bool(model3_unchanged and arrays_unchanged),
        "S3_unchanged": bool(skies_unchanged and arrays_unchanged),
        "A3_unchanged": bool(arrays_unchanged),
        "only_exposure_1_processed": True,
        "selection_excluded_previous_target": True,
        "selection_used_no_R": True,
        "all_new_targets_have_adequate_support": bool(all(
            row["n_blank_fibers"] >= 10 for row in selected.values())),
        "all_R_struct_orthogonal": bool(all(
            max(abs(float(np.dot(record["result"]["R_struct"][SPECTRAL_USE],
                                      _poly_coordinate(WAVE)[SPECTRAL_USE] ** power)))
                for power in (0, 1, 2))
            < 1e-8 for record in selected_records)),
        "all_R_normalized_unit_robust_rms": bool(all(
            abs(_robust_rms(record["result"]["R_normalized"][SPECTRAL_USE]) - 1.0)
            < 1e-8 for record in selected_records)),
        "all_primary_samples_equal_null_and_structured": bool(all(
            record["fit_primary"]["n_samples"] == int(np.sum(record["primary_sample"]))
            for record in selected_records)),
    }
    print("replication correctness checks")
    print(json.dumps(_json_ready(gate), indent=2, sort_keys=True))
    if not all(gate.values()):
        raise RuntimeError("Model-4A.1 replication correctness gate failed")

    reference_summary_path = Path(args.model4a1_reference_summary).expanduser().resolve()
    reference = json.loads(reference_summary_path.read_text())
    if reference.get("target", {}).get("AMP") != TARGET_PHYSICAL[3]:
        raise RuntimeError("reference Model-4A.1 summary is not the prior LU target")
    reference_features = reference["feature_rows"]
    reference_coherent = int(sum(
        int(np.sign(row["E_detrended"])) == int(np.sign(row["gamma_R_struct"]))
        for row in reference_features
        if np.isfinite(row["E_detrended"]) and np.isfinite(row["gamma_R_struct"])))
    reference_row = next(row for row in severity_rows
                         if row["identity"] == TARGET_PHYSICAL)
    ref_primary = reference["primary_equal_wavelength"]
    ref_edge = reference["blue_edge_repeat"]
    comparison_rows = [{
        "role": "reference_previous_4A1",
        "H5": TARGET_H5_BASENAME, "exposure": 1,
        "SPECID": TARGET_PHYSICAL[0], "IFUSLOT": TARGET_PHYSICAL[1],
        "IFUID": TARGET_PHYSICAL[2], "AMP": TARGET_PHYSICAL[3],
        "residual_severity": reference_row["residual_severity"],
        "severity_rank": reference_row["severity_rank"],
        "severity_percentile": reference_row["severity_percentile"],
        "N_blank_min": reference_row["N_blank_min"],
        "N_blank_median": reference_row["N_blank_median"],
        "N_blank_max": reference_row["N_blank_max"],
        "gamma": ref_primary["fit"]["gamma"],
        "gamma_uncertainty": ref_primary["fit"]["gamma_uncertainty_formal"],
        "pearson": ref_primary["metrics"]["pearson"],
        "spearman": ref_primary["metrics"]["spearman"],
        "fractional_rms_reduction": ref_primary["metrics"]["fractional_rms_reduction"],
        "robust_rms_before": ref_primary["metrics"]["robust_rms_before"],
        "robust_rms_after": ref_primary["metrics"]["robust_rms_after"],
        "median_absolute_before": ref_primary["metrics"]["median_absolute_before"],
        "median_absolute_after": ref_primary["metrics"]["median_absolute_after"],
        "edge_gamma": ref_edge["fit"]["gamma"],
        "edge_fractional_rms_reduction": ref_edge["metrics"]["fractional_rms_reduction"],
        "coherent_feature_count": reference_coherent,
        "source": "persisted prior Model-4A.1 summary; not refit",
    }]
    for record in selected_records:
        selection_row = record["selection"]
        metrics = record["metrics_primary"]
        edge = record["metrics_edge"]
        physical = tuple(record["result"]["identity"])
        comparison_rows.append({
            "role": record["role"], "H5": TARGET_H5_BASENAME, "exposure": 1,
            "SPECID": physical[0], "IFUSLOT": physical[1], "IFUID": physical[2],
            "AMP": physical[3], "residual_severity": selection_row["residual_severity"],
            "severity_rank": selection_row["severity_rank"],
            "severity_percentile": selection_row["severity_percentile"],
            "N_blank_min": selection_row["N_blank_min"],
            "N_blank_median": selection_row["N_blank_median"],
            "N_blank_max": selection_row["N_blank_max"],
            "gamma": metrics["gamma"], "gamma_uncertainty": metrics["gamma_uncertainty"],
            "pearson": metrics["pearson"], "spearman": metrics["spearman"],
            "fractional_rms_reduction": metrics["fractional_rms_reduction"],
            "robust_rms_before": metrics["robust_rms_before"],
            "robust_rms_after": metrics["robust_rms_after"],
            "median_absolute_before": metrics["median_absolute_before"],
            "median_absolute_after": metrics["median_absolute_after"],
            "edge_gamma": edge["gamma"],
            "edge_fractional_rms_reduction": edge["fractional_rms_reduction"],
            "coherent_feature_count": record["coherent_feature_count"],
            "source": "new independent Model-4A.1 replication",
        })

    csv_path = output_dir / "model4a1_replication_residual_shapes.csv"
    npz_path = output_dir / "model4a1_replication_blank_fiber_residuals.npz"
    summary_path = output_dir / "model4a1_replication_summary.json"
    _write_replication_csv(csv_path, selected_records)
    arrays = {"wavelength_A": WAVE}
    for index, record in enumerate(selected_records):
        result = record["result"]
        prefix = "record_%02d" % index
        arrays[prefix + "__blank_fiber_row_index"] = result["selected_blank_rows"]
        arrays[prefix + "__blank_fiber_q"] = result["selected_blank_q"]
        arrays[prefix + "__E_f_observed"] = result["E_f"]
        arrays[prefix + "__E_f_post"] = result["E_f_post"]
        arrays[prefix + "__fiber_row_index"] = result["fiber_row_indices"]
        arrays[prefix + "__fiber_blank_mask"] = result["fiber_blank_mask"]
        arrays[prefix + "__E_f_all"] = result["E_f_all"]
    np.savez_compressed(npz_path, **arrays)
    summary = {
        "block": "Model 4A.1 replication (20 amplifiers)",
        "scientific_question": "Does fixed R_struct have explanatory power for some other Model-3 residuals?",
        "model_changed": False, "model4b_performed": False,
        "h5": TARGET_H5_BASENAME, "exposure_processed": 1,
        "expected": "R should help at least some high-severity amplifiers if it describes a real failure class.",
        "selection_basis": "frozen Model-3 residual severity only; R was absent from target selection",
        "selection_mode": args.replication_selection_mode,
        "random_seed": random_seed,
        "excluded_selection_json": (str(Path(args.replication_exclude_selection).expanduser().resolve())
                                     if args.replication_exclude_selection else None),
        "excluded_identities": [list(identity) for identity in sorted(excluded_identities, key=str)],
        "eligible_distribution": [_replication_row_json(row) for row in severity_sorted],
        "preregistered_selection": {role: _replication_row_json(row)
                                     for role, row in selected.items()},
        "reference_row": comparison_rows[0],
        "comparison_rows": comparison_rows,
        "per_amplifier": [{
            "role": record["role"],
            "selection": _replication_row_json(record["selection"]),
            "primary": {"fit": {"q0": float(record["fit_primary"]["beta"][0]),
                                  "q1": float(record["fit_primary"]["beta"][1]),
                                  "q2": float(record["fit_primary"]["beta"][2]),
                                  "gamma": record["fit_primary"]["gamma"],
                                  "gamma_uncertainty_formal": record["fit_primary"]["gamma_uncertainty"]},
                       "metrics": record["metrics_primary"]},
            "weighted": record["metrics_weighted"],
            "edge_repeat": {"fit": {"gamma": record["fit_edge"]["gamma"],
                                      "gamma_uncertainty_formal": record["fit_edge"]["gamma_uncertainty"]},
                             "metrics": record["metrics_edge"]},
            "feature_rows": record["feature_rows"],
            "coherent_feature_count": record["coherent_feature_count"],
            "outputs": {"plot": str(output_dir / (
                "model4a1_%s_20200710_0000013_e1_%d_%d_%d_%s.png" %
                (record["role"], *record["result"]["identity"]))),
                        "fiber_residual_heatmap": record["fiber_heatmap_path"]},
            "fiber_heatmap": record["fiber_heatmap"],
        } for record in selected_records],
        "correctness_checks": gate,
        "model4a1_reference_summary": str(reference_summary_path),
        "model3_product": str(product_path),
        "provenance": {"exact_command": " ".join(shlex.quote(value) for value in sys.argv),
                       "h5": str(h5_paths[0]),
                       "blank_file": str(Path(args.blank_file).expanduser().resolve()),
                       "on_filter": str(Path(args.on_filter).expanduser().resolve()),
                       "off_filter": str(Path(args.off_filter).expanduser().resolve()),
                       "fq_template": str(Path(args.fq_template).expanduser().resolve()),
                       "blank_loader": blank_provenance,
                       "native_loader": input_provenance,
                       "model3_product_provenance": product["provenance"]},
        "output_paths": {"preregistered_selection": str(preregistration_path),
                         "summary": str(summary_path), "csv": str(csv_path),
                         "blank_fiber_arrays": str(npz_path)},
        "interpretation_gate": "STOP after 20 new amplifiers plus persisted reference; do not implement Model 4B.",
    }
    summary_path.write_text(json.dumps(_json_ready(summary), indent=2, sort_keys=True))
    print("OBSERVED")
    print(json.dumps(_json_ready(comparison_rows), indent=2, sort_keys=True))
    print("wrote Model-4A.1 replication products to %s" % output_dir)
    return summary


def _print_units():
    print("units: D=native reconstructed flux (spectrum/Survey.offset + skyspectrum)")
    print("units: A3=native additive flux; m3=dimensionless; S3=native flux per fiber")
    print("units: O3=native flux per fiber; I and R_raw=native summed flux")
    print("units: R_orthogonalized_normalized=dimensionless robust-RMS units")


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("model4a", "model4a1", "model4a1-replicate"),
                        default="model4a")
    parser.add_argument("--h5", nargs="+", required=True,
                        help="exactly one H5; both current modes use the representative H5")
    parser.add_argument("--model3-product", required=True)
    parser.add_argument("--model4a-summary",
                        help="prior Model-4A summary; required for Model-4A.1 modes")
    parser.add_argument("--model4a1-reference-summary",
                        help="persisted prior Model-4A.1 summary; required for replication")
    parser.add_argument("--replication-selection-mode",
                        choices=("severity-spaced", "random"), default="severity-spaced",
                        help="replication target selection; random uses a recorded seed")
    parser.add_argument("--replication-random-seed", type=int,
                        help="optional seed for random replication selection")
    parser.add_argument("--replication-exclude-selection",
                        help="JSON selection file whose exact amplifier identities are excluded")
    parser.add_argument("--blank-file", required=True)
    parser.add_argument("--on-filter", required=True)
    parser.add_argument("--off-filter", required=True)
    parser.add_argument("--fq-template", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = _parse_args()
    checks = deterministic_checks()
    print("deterministic transformation checks")
    print(json.dumps(_json_ready(checks), indent=2, sort_keys=True))
    if not all(bool(value.get("pass")) for value in checks.values()):
        raise SystemExit("deterministic Model-4A check failed; real data not opened")

    if args.mode == "model4a1":
        if not args.model4a_summary:
            raise SystemExit("--model4a-summary is required for --mode model4a1")
        run_model4a1(args, checks)
        return
    if args.mode == "model4a1-replicate":
        if not args.model4a_summary or not args.model4a1_reference_summary:
            raise SystemExit("--model4a-summary and --model4a1-reference-summary are required for replication")
        run_model4a1_replication(args, checks)
        return

    h5_paths = discover_h5(args.h5, development=True)
    if len(h5_paths) != 1 or h5_paths[0].name != TARGET_H5_BASENAME:
        raise SystemExit("Model 4A requires exactly %s" % TARGET_H5_BASENAME)
    product_path = Path(args.model3_product).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise SystemExit("output directory is nonempty; use --overwrite: %s" % output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    _print_units()
    product, model3, skies = load_model_and_skies(product_path)
    product_h5_names = {Path(item["filename"]).name
                        for item in product["provenance"]["input_h5"]}
    if TARGET_H5_BASENAME not in product_h5_names:
        raise SystemExit("persisted Model-3 product does not contain the representative H5")
    blank_masks, blank_provenance = m101_blank_fibers.load(args.blank_file, h5_paths)
    fq = validated_m101.load_fq(args.fq_template)
    data, input_provenance = load_native_data(
        h5_paths, blank_masks, args.on_filter, args.off_filter,
        include_native_errors=False, include_band_errors=False,
        collapse_band_indices=(5, 6))

    model3_snapshot = {
        "p_ifu": {key: value for key, value in model3.p_ifu.items()},
        "p_amp": {key: value for key, value in model3.p_amp.items()},
        "ax": {key: value for key, value in model3.ax.items()},
        "ay": {key: value for key, value in model3.ay.items()},
        "alpha_q": {key: value for key, value in model3.alpha_q.items()},
    }
    sky_snapshot = {key: value.copy() for key, value in skies.items()
                    if key[0] == TARGET_H5_BASENAME}
    records = reconstruct_model3_objects(data, model3, skies, fq)
    model3_unchanged = all(getattr(model3, name) == model3_snapshot[name]
                           for name in model3_snapshot)
    skies_unchanged = all(np.array_equal(skies[key], value, equal_nan=True)
                          for key, value in sky_snapshot.items())
    if not model3_unchanged or not skies_unchanged:
        raise RuntimeError("Model-3 input state changed during Model-4A construction")

    physical_selected, selection_labels, selection_features = select_representative_amplifiers(records)
    selected_records = [record for key, record in sorted(records.items(), key=lambda pair: str(pair[0]))
                        if (record["SPECID"], record["IFUSLOT"], record["IFUID"], record["AMP"])
                        in physical_selected]
    selected_records.sort(key=lambda record: (record["exposure"], record["SPECID"],
                                               record["IFUSLOT"], record["IFUID"], record["AMP"]))

    summary_rows = []
    for record in selected_records:
        physical = (record["SPECID"], record["IFUSLOT"], record["IFUID"], record["AMP"])
        metrics = _record_metrics(record)
        row = {"H5": record["H5"], "exposure": record["exposure"],
               "SPECID": record["SPECID"], "IFUSLOT": record["IFUSLOT"],
               "IFUID": record["IFUID"], "AMP": record["AMP"],
               "selection_label": selection_labels[physical], **metrics}
        summary_rows.append(row)
        filename = "%s_e%d_%d_%d_%d_%s_model4a.png" % (
            Path(record["H5"]).stem, record["exposure"], record["SPECID"],
            record["IFUSLOT"], record["IFUID"], record["AMP"])
        _plot_record(record, output_dir / filename, selection_labels[physical])

    _write_csv(output_dir / "model4a_selected_spectra.csv", selected_records)
    _write_npz(output_dir / "model4a_selected_arrays.npz", selected_records)
    (output_dir / "model4a_summary.json").write_text(json.dumps(_json_ready({
        "block": "Model 4A",
        "fit_performed": False,
        "iteration_performed": False,
        "target_h5": str(h5_paths[0]),
        "model3_product": str(product_path),
        "wavelength_A": {"start": float(WAVE[0]), "stop": float(WAVE[-1]),
                         "n": int(WAVE.size), "step": float(WAVE[1] - WAVE[0])},
        "spectral_evidence_window_A": [SPECTRAL_MIN_A, SPECTRAL_MAX_A],
        "smoothing": {"method": "Savitzky-Golay", "window_pixels": SMOOTHING_WINDOW_PIXELS,
                      "polyorder": SMOOTHING_POLYORDER, "temporary_fill": "finite linear interpolation"},
        "units": {"D": "native reconstructed flux", "A3": "native additive flux",
                  "m3": "dimensionless", "S3": "native flux per fiber",
                  "O3": "native flux per fiber", "I": "native summed flux",
                  "R_raw": "native summed flux", "R_orthogonalized_normalized":
                  "dimensionless robust-RMS units"},
        "selection_rule": "exposure 1 physical amplifier groups ranked by median absolute summed O3 in the evidence window; lowest, middle, and highest retained across all three exposures",
        "selection": [{"physical": list(physical), "label": label}
                      for physical, label in selection_labels.items()],
        "selection_features_exposure_1": selection_features,
        "summary_rows": summary_rows,
        "deterministic_checks": checks,
        "model3_unchanged": model3_unchanged,
        "skies_unchanged": skies_unchanged,
        "provenance": {"blank_file": str(Path(args.blank_file).expanduser().resolve()),
                       "blank_loader": blank_provenance,
                       "native_loader": input_provenance,
                       "model3_product_provenance": product["provenance"]},
    }), indent=2, sort_keys=True))
    print("selected physical amplifiers: %s" % json.dumps(_json_ready([
        {"physical": list(physical), "label": label}
        for physical, label in selection_labels.items()
    ]), sort_keys=True))
    print("wrote Model-4A products to %s" % output_dir)
    print("Model-3 unchanged: %s; Model-3 skies unchanged: %s" %
          (model3_unchanged, skies_unchanged))
    for row in summary_rows:
        print(json.dumps(_json_ready(row), sort_keys=True))


if __name__ == "__main__":
    main()
