#!/usr/bin/env python3
"""Model-4A: construct and inspect illumination-derived spectral templates.

This file deliberately implements only the first Model-4 development block.
It loads a persisted Model-3 state, reconstructs O3, forms the total incident
illumination I for amplifier groups, subtracts a documented smooth spectrum,
and writes inspection products.  It performs no Model-4 parameter inference.

Model-4 development plan
------------------------

    Model 4A:
        construct and inspect I, smooth(I), and R
        NO fitting
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
rich return, interpretation gate, and stop.  This script stops after Model
4A.  In particular it does not implement 4B--4G.

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
projected away from [1, x, x^2], and divided by its robust RMS.  R_raw and the
orthogonalized normalized candidate are retained as separate arrays.

No external-image machinery is needed for this inference.  The native loader,
Model-3 additive definition, wavelength grid, q orientation, K(lambda),
hardware/date masks, and Survey.offset convention are inherited from the
validated modules.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter

import diagnose_m101_hierarchical as validated_m101
import m101_blank_fibers
from diagnose_m101_post_model3 import _group_indices, load_model_and_skies
from m101_native_data import discover_h5, load as load_native_data


WAVE = np.asarray(validated_m101.DEF_WAVE, dtype=float)
SPECTRAL_MIN_A = 3540.0
SPECTRAL_MAX_A = 5480.0
SPECTRAL_USE = (WAVE >= SPECTRAL_MIN_A) & (WAVE <= SPECTRAL_MAX_A)
SMOOTHING_WINDOW_PIXELS = 51
SMOOTHING_POLYORDER = 2
TARGET_H5_BASENAME = "20200710_0000013.h5"


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


def orthogonalize_and_normalize(raw_residual, wave=WAVE,
                                spectral_use=SPECTRAL_USE):
    """Return a separately retained candidate and its transparent gauge data.

    The mean is removed first.  The least-squares coefficients then describe
    the projection of that centered residual onto [1, x, x^2].  Undefined
    wavelengths remain NaN, and only the evidence window contributes to the
    coefficients and normalization scale.
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
    if degenerate:
        scale = 1.0
    candidate[finite] /= scale
    return candidate, {
        "mean_removed": mean_removed,
        "polynomial_coefficients": [float(value) for value in coefficients],
        "normalization_scale": float(scale),
        "normalization_degenerate": bool(degenerate),
        "n_finite_window": int(np.sum(finite)),
    }


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
            normalized, normalization = orthogonalize_and_normalize(raw)
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
              "S3", "sum_O3", "N_valid", "N_S3", "I", "Smooth_I", "R_raw",
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
                    "R_orthogonalized_normalized": record[
                        "R_orthogonalized_normalized"][index],
                })


def _write_npz(path, records):
    arrays = {"wavelength_A": WAVE}
    for index, record in enumerate(records):
        prefix = "record_%02d" % index
        for name in ("S3", "sum_O3", "N_valid", "N_S3", "I", "Smooth_I", "R_raw",
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


def _print_units():
    print("units: D=native reconstructed flux (spectrum/Survey.offset + skyspectrum)")
    print("units: A3=native additive flux; m3=dimensionless; S3=native flux per fiber")
    print("units: O3=native flux per fiber; I and R_raw=native summed flux")
    print("units: R_orthogonalized_normalized=dimensionless robust-RMS units")


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5", nargs="+", required=True,
                        help="exactly one H5; Model 4A accepts only the representative H5")
    parser.add_argument("--model3-product", required=True)
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
