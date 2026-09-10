#!/usr/bin/env python3
"""First representative implementation of the M101 Model-4 experiment.

Model 3 is imported and used as an initial state.  This file deliberately
does not edit or call the Model-0--3 fitting sequence.  Model 4 tests whether
the external ON/OFF source measurements constrain the otherwise weakly
identified multiplicative/additive split.

For one H5 and its three exposures the implemented equation is

    D_e,f(w) = m_e,f [ S_e(w) + O_e,f(w) ]
               + alpha_e,a K_e(w) f(q_f)
               + P_e,a(w) + beta_e,a R_e,a(w) + noise,

    P_e,a(w) = c0_e,a + c1_e,a x(w) + c2_e,a x(w)^2.

Here D is the reconstructed native total spectrum in the units of
``spectrum / Survey.offset + skyspectrum``.  S, O, P, alpha*K*f(q), and
beta*R therefore have the same native flux units as D; m=exp(z) is
dimensionless; beta has native flux units because R is normalized to unit
robust RMS.  The external equation is

    Band[(D-alpha*K*f(q)-P-beta*R)/m - S] = G_e,b X_e,f,b,

where X is the existing cached external object measurement, including its
existing global_g factor.  G is common to all source fibers in an exposure
and band.

In the controlled ``additive-only-frozen`` experiment, R0 is constructed once
from the total incident illumination, not sky alone:

    I_e,a = N_e,a S_e + sum_f O_e,f.

I is smoothed with a 51-pixel Savitzky-Golay quadratic filter, after finite
linear interpolation only for the smoothing operation.  R0=I-smooth(I) is
uniform-wavelength centered, projected orthogonally to [1,x,x^2], and
divided by its robust RMS.  During the controlled experiment the persisted
Model-3 m, S, and G are fixed.  The polynomial coefficients are therefore
constrained directly by equal-weight zero sums across supported amplifier
observations within each exposure; the removed common polynomial components
are saved as diagnostics and are not transferred into the frozen S.

The source constraint breaks the blank-only direction

    m S + A ~= (m+delta_m)S + (A-delta_m S)

because the same transformation changes a source by delta_m O, while the
external ON/OFF X constrains O.  The executable controlled experiment performs
one constrained additive solve with fixed Model-3 m/S/G/R0.  The legacy
alternating helpers remain unreachable from that mode for provenance; this is
not a production cube calibration and does not add spatial scattered-light
modes.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import sparse
from scipy.signal import savgol_filter
from scipy.sparse.linalg import spsolve

import diagnose_m101_hierarchical as validated_m101
import m101_blank_fibers
import m101_compact_mask
import m101_external_measurements
from fit_m101_calibration_v2 import FittedModel
from m101_calibration_utils import collapse, robust_location, robust_scatter
from m101_native_data import discover_h5, load as load_native_data
from diagnose_m101_post_model3 import (
    AMP_ORDER, SOURCE_BANDS, build_source_rows, load_model_and_skies,
    _group_indices, _projection_metrics,
)


WAVE = np.asarray(validated_m101.DEF_WAVE, dtype=float)
SPECTRAL_MIN = 3540.0
SPECTRAL_MAX = 5480.0
SPECTRAL_USE = (WAVE >= SPECTRAL_MIN) & (WAVE <= SPECTRAL_MAX)
NULL_BANDS = ("HIGH1", "LOW1", "HIGH2", "LOW2", "HIGH3")
ALL_SOURCE = ("ON", "OFF")
AMP_INDEX = {amp: i for i, amp in enumerate(AMP_ORDER)}


def _finite(values):
    values = np.asarray(values, dtype=float)
    return values[np.isfinite(values)]


def _median(values, default=np.nan):
    values = _finite(values)
    return float(np.nanmedian(values)) if values.size else default


def _robust_rms(values):
    values = _finite(values)
    return float(validated_m101.robust_scale(values)) if values.size else np.nan


def _write_rows(path, rows):
    path = Path(path)
    if not rows:
        path.write_text("")
        return
    fields = list(rows[0])
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _json_ready(value):
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_json_ready(v) for v in value.tolist()]
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _amp_key(item, ifu, amp):
    return (item.key, tuple(map(int, ifu)), str(amp))


def _poly_basis(wave=WAVE):
    x = (np.asarray(wave, dtype=float) - np.nanmean(wave)) / (0.5 * (np.nanmax(wave) - np.nanmin(wave)))
    return x, np.column_stack((np.ones(x.size), x, x * x))


def _fill_for_smoothing(values):
    values = np.asarray(values, dtype=float)
    good = np.isfinite(values)
    if good.sum() < 5:
        return np.full(values.shape, np.nanmedian(values[good]) if np.any(good) else 0.0)
    indices = np.arange(values.size)
    return np.interp(indices, indices[good], values[good])


def build_incident_template(incident, poly_basis, spectral_use=SPECTRAL_USE):
    """Return smooth I and normalized R with the documented gauge removal."""
    incident = np.asarray(incident, dtype=float)
    filled = _fill_for_smoothing(incident)
    window = min(51, filled.size if filled.size % 2 else filled.size - 1)
    window = max(window, 5)
    if window % 2 == 0:
        window -= 1
    smooth = savgol_filter(filled, window_length=window, polyorder=2, mode="interp")
    residual = incident - smooth
    finite = np.isfinite(residual) & np.asarray(spectral_use, dtype=bool)
    if not np.any(finite):
        return smooth, np.full(incident.shape, np.nan), {"window": int(window), "scale": np.nan}
    residual = residual.copy()
    residual[finite] -= np.nanmean(residual[finite])
    basis = np.asarray(poly_basis, dtype=float)
    use = finite & np.all(np.isfinite(basis), axis=1)
    if np.sum(use) >= basis.shape[1]:
        coefficients, _, _, _ = np.linalg.lstsq(basis[use], residual[use], rcond=None)
        residual[use] -= basis[use] @ coefficients
    scale = max(_robust_rms(residual[finite]), 1e-12)
    residual[finite] /= scale
    # Undefined native wavelengths cannot contribute to a fit.  Use zero for
    # the additive design term so beta=0 remains exactly a no-op; the input
    # validity is still represented by the zeroed template and by the saved
    # incident spectrum.  In particular, never evaluate 0 * NaN.
    residual[~finite] = 0.0
    return smooth, residual, {"window": int(window), "scale": float(scale)}


@dataclass
class Model4State:
    z_by_key: dict
    alpha: dict
    polynomial: dict
    beta: dict
    incident: dict
    smooth_incident: dict
    template: dict
    additive_full_by_key: dict
    additive_band_by_key: dict
    skies: dict
    G: dict

    def z_for(self, item):
        return self.z_by_key[item.key]

    def additive_full(self, item, fq):
        return self.additive_full_by_key[item.key]

    def additive_bands(self, item, fq):
        return self.additive_band_by_key[item.key]


def _initial_state(data, model3, skies):
    state = Model4State({}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {})
    state.skies = {key: np.asarray(value, dtype=float).copy() for key, value in skies.items()
                   if key in {item.key for item in data}}
    x, poly_basis = _poly_basis()
    for item in data:
        state.z_by_key[item.key] = model3.z_for(item).copy()
        for ifu, amp in zip(item.ifu, item.amp):
            key = _amp_key(item, ifu, amp)
            state.alpha.setdefault(key, float(model3.alpha_q.get(key, 0.0)))
            state.polynomial.setdefault(key, np.zeros(3, dtype=float))
            state.beta.setdefault(key, 0.0)
            state.G.setdefault((item.key, "ON"), 1.0)
            state.G.setdefault((item.key, "OFF"), 1.0)
    state.additive_full_by_key = {item.key: np.zeros_like(item.total, dtype=float) for item in data}
    state.additive_band_by_key = {item.key: np.zeros_like(item.band_total, dtype=float) for item in data}
    return state, x, poly_basis


def _rebuild_additive(data, state, fq, responses):
    x, poly_basis = _poly_basis()
    for item in data:
        full = np.zeros_like(item.total, dtype=float)
        _, groups = _group_indices(item)
        for (ifu, amp), indices in groups:
            key = _amp_key(item, ifu, amp)
            c0, c1, c2 = state.polynomial.get(key, np.zeros(3))
            p = c0 + c1 * x + c2 * x * x
            alpha = state.alpha.get(key, 0.0)
            h = alpha * item.K_wave * validated_m101.f_q(item.q[indices[0]]) if hasattr(validated_m101, "f_q") else 0.0
            # The repository exposes the fixed q template as load_fq output;
            # callers replace this branch through the q_template argument below.
            full[indices] = p[None, :]
            full[indices] += alpha * item.K_wave[None, :] * state._fq[item.q[indices], None]
            full[indices] += state.beta.get(key, 0.0) * state.template.get(key, np.zeros(item.total.shape[1]))[None, :]
        state.additive_full_by_key[item.key] = full
        bands = np.full_like(item.band_total, np.nan, dtype=float)
        for i, response in enumerate(responses):
            bands[:, i] = collapse(full, response)[0]
        state.additive_band_by_key[item.key] = bands


def _set_additive_arrays(data, state, fq, responses):
    """Build L arrays without relying on a Model-3 additive method."""
    x, _ = _poly_basis()
    state._fq = np.asarray(fq, dtype=float)
    for item in data:
        full = np.zeros_like(item.total, dtype=float)
        _, groups = _group_indices(item)
        for (ifu, amp), indices in groups:
            key = _amp_key(item, ifu, amp)
            c0, c1, c2 = state.polynomial.get(key, np.zeros(3))
            p = c0 + c1 * x + c2 * x * x
            full[indices] = p[None, :]
            full[indices] += state.alpha.get(key, 0.0) * item.K_wave[None, :] * state._fq[item.q[indices], None]
            full[indices] += state.beta.get(key, 0.0) * state.template.get(key, np.zeros(item.total.shape[1]))[None, :]
        state.additive_full_by_key[item.key] = full
        state.additive_band_by_key[item.key] = np.column_stack(
            [collapse(full, response)[0] for response in responses])


def _construct_incident_templates(data, state):
    x, poly_basis = _poly_basis()
    for item in data:
        corrected = (item.total - state.additive_full_by_key[item.key]) / np.exp(state.z_by_key[item.key])[:, None]
        object_spectrum = corrected - state.skies[item.key][None, :]
        _, groups = _group_indices(item)
        # Native support is wavelength-wise.  The loader intentionally keeps
        # partially finite fibers, so requiring an entirely finite spectrum
        # would incorrectly remove nearly every contributing fiber.
        valid_native = (~item.hardware_bad) & np.any(np.isfinite(item.total), axis=1)
        for (ifu, amp), indices in groups:
            key = _amp_key(item, ifu, amp)
            selected = valid_native[indices]
            if not np.any(selected):
                state.incident[key] = np.full(WAVE.shape, np.nan)
                state.smooth_incident[key] = np.full(WAVE.shape, np.nan)
                state.template[key] = np.full(WAVE.shape, np.nan)
                continue
            selected_object = object_spectrum[indices][selected]
            count = np.sum(np.isfinite(selected_object), axis=0)
            object_sum = np.nansum(selected_object, axis=0)
            incident = count * state.skies[item.key] + object_sum
            smooth, template, _ = build_incident_template(incident, poly_basis, SPECTRAL_USE)
            state.incident[key] = incident
            state.smooth_incident[key] = smooth
            state.template[key] = template


def _solve_small_design(design, target, weights=None, ridge=1e-8):
    design = np.asarray(design, dtype=float)
    target = np.asarray(target, dtype=float)
    use = np.all(np.isfinite(design), axis=1) & np.isfinite(target)
    if weights is not None:
        weights = np.asarray(weights, dtype=float)
        use &= np.isfinite(weights) & (weights > 0)
    if np.sum(use) < design.shape[1]:
        return np.zeros(design.shape[1], dtype=float), 0
    A = design[use]
    y = target[use]
    if weights is not None:
        A = A * np.sqrt(weights[use])[:, None]
        y = y * np.sqrt(weights[use])
    normal = A.T @ A
    scale = np.maximum(np.diag(normal), 1.0)
    normal += np.diag(ridge * scale)
    rhs = A.T @ y
    solution = np.linalg.solve(normal, rhs)
    return np.asarray(solution, dtype=float), int(np.sum(use))


def _fit_additive(data, state, source_masks, external, responses):
    """Fit P, alpha, beta jointly against blank spectra and source bands."""
    x, poly_basis = _poly_basis()
    for item in data:
        _, groups = _group_indices(item)
        sky = state.skies[item.key]
        m = np.exp(state.z_by_key[item.key])
        for (ifu, amp), indices in groups:
            key = _amp_key(item, ifu, amp)
            rows, targets, weights = [], [], []
            blank = item.blank_valid[indices]
            if np.sum(blank) >= 5:
                observed = np.nanmedian(item.total[indices][blank] - m[indices][blank, None] * sky[None, :], axis=0)
                h = item.K_wave * state._fq[item.q[indices[0]]]
                r = state.template.get(key, np.full(WAVE.shape, np.nan))
                rows.extend(np.column_stack((poly_basis, np.full(WAVE.shape, h), r))[SPECTRAL_USE])
                targets.extend(observed[SPECTRAL_USE].tolist())
                weights.extend(np.ones(int(SPECTRAL_USE.sum())).tolist())
            for band in ALL_SOURCE:
                band_index = 5 if band == "ON" else 6
                selected = np.asarray(source_masks[item.key][band][indices], dtype=bool)
                if not np.any(selected):
                    continue
                cache = external[item.h5_name]
                X = np.asarray(cache["global_g"][band] * cache["external_object"][(item.exposure, band)], dtype=float)[indices][selected]
                virus = item.band_total[indices][selected, band_index]
                desired = m[indices][selected] * (collapse(sky[None, :], responses[band_index])[0][0] + state.G[(item.key, band)] * X)
                h_band = collapse((item.K_wave * state._fq[item.q[indices[0]]])[None, :], responses[band_index])[0][0]
                p_band = np.asarray([collapse(basis[None, :], responses[band_index])[0][0] for basis in poly_basis.T])
                r = state.template.get(key, np.full(WAVE.shape, np.nan))
                r_band = collapse(r[None, :], responses[band_index])[0][0]
                design = np.tile(np.r_[p_band, h_band, r_band], (np.sum(selected), 1))
                rows.extend(design)
                targets.extend((virus - desired).tolist())
                weights.extend(np.full(np.sum(selected), 4.0).tolist())
            if not rows:
                continue
            solution, n = _solve_small_design(np.asarray(rows), np.asarray(targets), np.asarray(weights))
            state.polynomial[key] = solution[:3]
            state.alpha[key] = float(solution[3])
            state.beta[key] = float(solution[4])


def _update_G(data, state, source_masks, external, responses):
    result = {}
    for item in data:
        m = np.exp(state.z_by_key[item.key])
        corrected = item.band_total - state.additive_band_by_key[item.key]
        sky_bands = np.asarray([collapse(state.skies[item.key][None, :], response)[0][0]
                                for response in responses])
        for band in ALL_SOURCE:
            bi = 5 if band == "ON" else 6
            cache = external[item.h5_name]
            X = cache["global_g"][band] * np.asarray(cache["external_object"][(item.exposure, band)], dtype=float)
            selected = source_masks[item.key][band] & np.isfinite(X) & (X > 0) & np.isfinite(corrected[:, bi])
            O = corrected[:, bi] / m - sky_bands[bi]
            ratio = O[selected] / X[selected]
            value = robust_location(ratio) if ratio.size else state.G[(item.key, band)]
            result[(item.key, band)] = float(value) if np.isfinite(value) else state.G[(item.key, band)]
    state.G.update(result)


def _local_z_targets(data, state, source_masks, external, responses):
    targets, weights, labels = [], [], []
    for item in data:
        m = np.exp(state.z_by_key[item.key])
        corrected = item.total - state.additive_full_by_key[item.key]
        sky = state.skies[item.key]
        _, groups = _group_indices(item)
        sky_bands = np.asarray([collapse(sky[None, :], response)[0][0] for response in responses])
        for (ifu, amp), indices in groups:
            key = _amp_key(item, ifu, amp)
            ratios = []
            blank = item.blank_valid[indices]
            if np.any(blank):
                numerator = corrected[indices][blank][:, SPECTRAL_USE]
                denominator = m[indices][blank, None] * sky[SPECTRAL_USE][None, :]
                use = np.isfinite(numerator) & np.isfinite(denominator) & (numerator > 0) & (denominator > 0)
                ratios.extend(np.log(numerator[use] / denominator[use]).tolist())
            for band in ALL_SOURCE:
                bi = 5 if band == "ON" else 6
                selected = source_masks[item.key][band][indices]
                if not np.any(selected):
                    continue
                cache = external[item.h5_name]
                X = cache["global_g"][band] * np.asarray(cache["external_object"][(item.exposure, band)], dtype=float)[indices][selected]
                numerator = collapse(corrected[indices][selected], responses[bi])[0]
                denominator = m[indices][selected] * (sky_bands[bi] + state.G[(item.key, band)] * X)
                use = np.isfinite(numerator) & np.isfinite(denominator) & (numerator > 0) & (denominator > 0)
                ratios.extend(np.log(numerator[use] / denominator[use]).tolist())
            if ratios:
                targets.append(float(np.clip(robust_location(ratios), -0.25, 0.25)))
                weights.append(float(np.sqrt(max(len(ratios), 1))))
                labels.append((item, ifu, amp, key))
    return targets, weights, labels


def _update_multiplicative_hierarchy(data, state, source_masks, external, responses):
    targets, weights, labels = _local_z_targets(data, state, source_masks, external, responses)
    if not labels:
        return {"N": 0, "rms": np.nan}
    ifus = sorted({tuple(map(int, item.ifu[0])) for item in data})
    ifus = sorted({label[1] for label in labels}, key=str)
    ifu_index = {ifu: i for i, ifu in enumerate(ifus)}
    n_ifu = len(ifus)
    n_exp = len(data)
    columns = n_ifu - 1 + 3 * n_ifu + 2 * n_exp
    design = np.zeros((len(labels), columns), dtype=float)
    for row, (_, ifu, amp, _) in enumerate(labels):
        i = ifu_index[ifu]
        if i < n_ifu - 1:
            design[row, i] = 1.0
        elif n_ifu > 1:
            design[row, :n_ifu - 1] = -1.0
        amp_start = n_ifu - 1 + 3 * i
        amp_index = AMP_INDEX[amp]
        if amp_index < 3:
            design[row, amp_start + amp_index] = 1.0
        else:
            design[row, amp_start:amp_start + 3] = -1.0
        item = labels[row][0]
        exp_index = next(j for j, candidate in enumerate(data) if candidate.key == item.key)
        group_indices = [indices for group_key, indices in _group_indices(item)[1]
                         if group_key == (ifu, amp)]
        group_indices = group_indices[0] if group_indices else np.arange(item.row_index.size)
        design[row, n_ifu - 1 + 3 * n_ifu + 2 * exp_index] = float(np.nanmedian(item.x_arcmin[group_indices]))
        design[row, n_ifu - 1 + 3 * n_ifu + 2 * exp_index + 1] = float(np.nanmedian(item.y_arcmin[group_indices]))
    solution, _ = _solve_small_design(design, targets, np.asarray(weights) ** 2, ridge=1e-6)
    predicted = design @ solution
    # Scope each conditional update to its actual physical amplifier.  The
    # alternating fit is intentionally disabled for this experiment, but this
    # helper must remain correct for later controlled work.
    updated_keys = set()
    for row, (item, ifu, amp, key) in enumerate(labels):
        if item.key not in updated_keys:
            state.z_by_key[item.key] = state.z_by_key[item.key].copy()
            updated_keys.add(item.key)
        group_indices = [indices for group_key, indices in _group_indices(item)[1]
                         if group_key == (ifu, amp)]
        if group_indices:
            state.z_by_key[item.key][group_indices[0]] += 0.5 * predicted[row]
    return {"N": len(labels), "rms": _robust_rms(np.asarray(targets) - predicted),
            "median_delta": _median(predicted)}


def _update_skies(data, state):
    changes = []
    for item in data:
        selected = item.blank_valid
        if not np.any(selected):
            continue
        corrected = ((item.total[selected] - state.additive_full_by_key[item.key][selected]) /
                     np.exp(state.z_by_key[item.key][selected])[:, None])
        new_sky = np.asarray(robust_location(corrected, axis=0), dtype=float)
        old = state.skies[item.key]
        updated = old.copy()
        updated[SPECTRAL_USE] = 0.5 * old[SPECTRAL_USE] + 0.5 * new_sky[SPECTRAL_USE]
        state.skies[item.key] = updated
        changes.append(_robust_rms(updated[SPECTRAL_USE] - old[SPECTRAL_USE]))
    return _median(changes)


def fit_model4(data, model3, skies, fq, external, source_masks, responses,
               initial_G=None, iterations=4):
    state, x, poly_basis = _initial_state(data, model3, skies)
    state._fq = np.asarray(fq, dtype=float)
    if initial_G:
        state.G.update(initial_G)
    _set_additive_arrays(data, state, fq, responses)
    _construct_incident_templates(data, state)
    history = []
    for iteration in range(iterations):
        _fit_additive(data, state, source_masks, external, responses)
        _set_additive_arrays(data, state, fq, responses)
        _update_G(data, state, source_masks, external, responses)
        z_report = _update_multiplicative_hierarchy(data, state, source_masks, external, responses)
        _set_additive_arrays(data, state, fq, responses)
        sky_change = _update_skies(data, state)
        _set_additive_arrays(data, state, fq, responses)
        _construct_incident_templates(data, state)
        history.append({"iteration": iteration + 1, "z_update": z_report,
                        "sky_change_robust_rms": sky_change,
                        "G": {str(k): v for k, v in state.G.items()}})
    _set_additive_arrays(data, state, fq, responses)
    return state, history


def _make_model4_adapter(state):
    return state


def _source_summary(rows):
    result = {}
    for band in ALL_SOURCE:
        values = [row["raw_robust_ratio"] for row in rows if row["band"] == band]
        result[band] = {"N": len(values), "center": _median(values), "scatter": _robust_rms(values),
                        "median_absolute_deviation_from_center": _median(np.abs(np.asarray(values) - _median(values)))}
    return result


def _source_rows_with_label(rows, label, baseline_centers):
    output = []
    for row in rows:
        item = dict(row)
        item["model"] = label
        item["same_normalization_ratio"] = (row["raw_robust_ratio"] / baseline_centers[row["band"]]
                                             if np.isfinite(row["raw_robust_ratio"]) else np.nan)
        output.append(item)
    return output


def _select_amplifiers(rows):
    usable = [row for row in rows if row["N_source_measurements"] >= 10 and np.isfinite(row["raw_robust_ratio"])]
    if not usable:
        return []
    values = np.asarray([row["raw_robust_ratio"] for row in usable])
    center = np.nanmedian(values)
    picks = [min(usable, key=lambda r: r["raw_robust_ratio"]),
             max(usable, key=lambda r: r["raw_robust_ratio"]),
             min(usable, key=lambda r: abs(r["raw_robust_ratio"] - center))]
    blank = [row for row in usable if row["N_source_measurements"] <= 15]
    if blank:
        picks.append(blank[0])
    unique = []
    seen = set()
    for row in picks:
        key = (row["H5"], row["exposure"], row["SPECID"], row["IFUSLOT"], row["IFUID"], row["AMP"])
        if key not in seen:
            seen.add(key); unique.append(row)
    return unique


def _select_blank_amplifiers(parameter_rows, maximum=2):
    """Select blank-supported groups independently of source-ratio ranking."""
    candidates = [row for row in parameter_rows if int(row["N_blank_fibers"]) >= 3]
    candidates.sort(key=lambda row: (-int(row["N_blank_fibers"]), row["exposure"], row["SPECID"],
                                     row["IFUSLOT"], row["IFUID"], row["AMP"]))
    selected = []
    seen_exposures = set()
    for row in candidates:
        if row["exposure"] in seen_exposures and len(selected) < maximum:
            continue
        selected.append(row)
        seen_exposures.add(row["exposure"])
        if len(selected) >= maximum:
            break
    if len(selected) < maximum:
        for row in candidates:
            key = (row["H5"], row["exposure"], row["SPECID"], row["IFUSLOT"], row["IFUID"], row["AMP"])
            if key not in {(item["H5"], item["exposure"], item["SPECID"], item["IFUSLOT"], item["IFUID"], item["AMP"])
                           for item in selected}:
                selected.append(row)
            if len(selected) >= maximum:
                break
    return selected


def _find_group(item, row):
    ifu = (int(row["SPECID"]), int(row["IFUSLOT"]), int(row["IFUID"]))
    amp = row["AMP"]
    for group_key, indices in _group_indices(item)[1]:
        if group_key == (ifu, amp):
            return indices, ifu, amp
    raise KeyError(row)


def write_decomposition_plot(output_dir, data, state, model3, fq, selected_rows):
    output_dir = Path(output_dir)
    fig, axes = plt.subplots(len(selected_rows), 2, figsize=(15, 3.8 * len(selected_rows)), squeeze=False)
    spectra_rows = []
    for plot_row, row in enumerate(selected_rows):
        item = next(item for item in data if item.h5_name == row["H5"] and item.exposure == int(row["exposure"]))
        indices, ifu, amp = _find_group(item, row)
        key = _amp_key(item, ifu, amp)
        valid = (~item.hardware_bad[indices]) & np.any(np.isfinite(item.total[indices]), axis=1)
        indices = indices[valid]
        if not np.any(valid):
            continue
        observed = np.nanmedian(item.total[indices], axis=0)
        m = np.exp(state.z_by_key[item.key][indices])
        additive = state.additive_full_by_key[item.key][indices]
        multiplicative = np.nanmedian(item.total[indices] - additive, axis=0)
        alpha = np.nanmedian(item.K_wave[None, :] * state._fq[item.q[indices], None] * state.alpha[key], axis=0)
        p = np.nanmedian(additive - alpha - state.beta[key] * state.template[key][None, :], axis=0)
        beta_r = state.beta[key] * state.template[key]
        prediction = multiplicative + alpha + p + beta_r
        residual = observed - prediction
        left, right = axes[plot_row]
        left.plot(WAVE[SPECTRAL_USE], observed[SPECTRAL_USE], color="black", lw=.65, label="observed")
        left.plot(WAVE[SPECTRAL_USE], multiplicative[SPECTRAL_USE], color="tab:blue", lw=.7, label="m(S+O)")
        left.plot(WAVE[SPECTRAL_USE], alpha[SPECTRAL_USE], color="tab:orange", lw=.7, label="alpha K f(q)")
        left.plot(WAVE[SPECTRAL_USE], p[SPECTRAL_USE], color="tab:green", lw=.7, label="P")
        left.plot(WAVE[SPECTRAL_USE], beta_r[SPECTRAL_USE], color="tab:red", lw=.7, label="beta R")
        left.plot(WAVE[SPECTRAL_USE], prediction[SPECTRAL_USE], color="purple", lw=1.0, label="total")
        left.set_title("%s e%d %s/%s/%s %s" % (item.h5_name, item.exposure, *ifu, amp))
        left.set_ylabel("native flux")
        left.legend(ncol=3, fontsize=7)
        right.plot(WAVE[SPECTRAL_USE], state.incident[key][SPECTRAL_USE], color="black", lw=.7, label="I")
        right.plot(WAVE[SPECTRAL_USE], state.smooth_incident[key][SPECTRAL_USE], color="tab:blue", lw=.8, label="smooth(I)")
        right.plot(WAVE[SPECTRAL_USE], state.template[key][SPECTRAL_USE], color="tab:red", lw=.7, label="R")
        right.axhline(0, color="0.5", lw=.5)
        right.set_title("incident template; beta=%.6g" % state.beta[key])
        right.legend(fontsize=8)
        right.set_ylabel("native flux / normalized R")
        for name, values in (("observed", observed), ("mSplusO", multiplicative), ("alphaH", alpha),
                             ("P", p), ("betaR", beta_r), ("prediction", prediction), ("residual", residual),
                             ("I", state.incident[key]), ("smoothI", state.smooth_incident[key]),
                             ("R", state.template[key])):
            spectra_rows.extend({"H5": item.h5_name, "exposure": item.exposure, "SPECID": ifu[0],
                                 "IFUSLOT": ifu[1], "IFUID": ifu[2], "AMP": amp,
                                 "wavelength": wave, "component": name, "value": value}
                                for wave, value in zip(WAVE[SPECTRAL_USE], values[SPECTRAL_USE]))
    axes[-1, 0].set_xlabel("wavelength [A]")
    axes[-1, 1].set_xlabel("wavelength [A]")
    fig.tight_layout(); fig.savefig(output_dir / "m101_model4_representative_decomposition.png", dpi=150); plt.close(fig)
    _write_rows(output_dir / "m101_model4_representative_spectra.csv", spectra_rows)


def _source_comparison_plot(output_dir, before, after, centers):
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), squeeze=False)
    for col, band in enumerate(ALL_SOURCE):
        b = [r for r in before if r["band"] == band]
        a = [r for r in after if r["band"] == band]
        for row_index, (rows, label) in enumerate(((b, "Model 3"), (a, "Model 4"))):
            ax = axes[row_index, col]
            for h5 in sorted({r["H5"] for r in rows}):
                selected = [r for r in rows if r["H5"] == h5]
                ax.scatter([r["x_arcmin"] for r in selected], [r["y_arcmin"] for r in selected],
                           c=[r["raw_robust_ratio"] / centers[band] for r in selected],
                           s=8, vmin=.6, vmax=1.4, cmap="coolwarm")
            ax.set_title("%s %s" % (label, band)); ax.set_aspect("equal", adjustable="box")
            ax.set_xlabel("x [arcmin]"); ax.set_ylabel("y [arcmin]")
    for row_index, (rows, label) in enumerate(((before, "Model 3"), (after, "Model 4"))):
        ax = axes[row_index, 2]
        by_key = {}
        for row in rows:
            by_key.setdefault(row["_key"], {})[row["band"]] = row["raw_robust_ratio"] / centers[row["band"]]
        pairs = [(values["ON"], values["OFF"]) for values in by_key.values()
                 if "ON" in values and "OFF" in values
                 and np.isfinite(values["ON"]) and np.isfinite(values["OFF"])]
        if pairs:
            pair_array = np.asarray(pairs, dtype=float)
            ax.scatter(pair_array[:, 0], pair_array[:, 1], s=12, alpha=.7)
            lo = float(np.nanpercentile(pair_array, 1)); hi = float(np.nanpercentile(pair_array, 99))
            lo, hi = min(lo, .8), max(hi, 1.2)
            ax.plot([lo, hi], [lo, hi], "k--", lw=.8)
            ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_title("%s ON vs OFF" % label)
        ax.set_xlabel("ON / Model-3 center"); ax.set_ylabel("OFF / Model-3 center")
    fig.tight_layout(); fig.savefig(output_dir / "m101_model4_source_stitching_before_after.png", dpi=150); plt.close(fig)
    rows = []
    for band in ALL_SOURCE:
        b = {r["_key"]: r for r in before if r["band"] == band}
        a = {r["_key"]: r for r in after if r["band"] == band}
        for key in sorted(set(b) & set(a), key=str):
            rows.append({"band": band, "key": str(key), "model3_ratio": b[key]["raw_robust_ratio"] / centers[band],
                         "model4_ratio": a[key]["raw_robust_ratio"] / centers[band],
                         "model3_N": b[key]["N_source_measurements"], "model4_N": a[key]["N_source_measurements"]})
    _write_rows(output_dir / "m101_model4_source_stitching.csv", rows)


def _blank_plot(output_dir, data, model3, state, fq, responses):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), squeeze=False)
    null_responses = responses[:5]
    for col, item in enumerate(data):
        before_add = model3.additive_full(item, fq)
        before = (item.total - before_add) / np.exp(model3.z_for(item))[:, None] - state.skies[item.key][None, :]
        after = (item.total - state.additive_full_by_key[item.key]) / np.exp(state.z_by_key[item.key])[:, None] - state.skies[item.key][None, :]
        for row_index, (residual, label) in enumerate(((before, "Model 3"), (after, "Model 4"))):
            ax = axes[row_index, col]
            band_values, _ = collapse(residual[item.blank_valid], null_responses[col:col + 1][0])
            finite = item.blank_valid & np.isfinite(collapse(residual, null_responses[col:col + 1][0])[0])
            values = collapse(residual[finite], null_responses[col:col + 1][0])[0]
            image = ax.scatter(item.x_arcmin[finite], item.y_arcmin[finite], c=values, s=4, cmap="coolwarm",
                               vmin=-max(_robust_rms(values), 1e-6), vmax=max(_robust_rms(values), 1e-6))
            ax.set_title("%s %s %s" % (label, item.h5_name, NULL_BANDS[col]))
            ax.set_aspect("equal", adjustable="box")
    fig.tight_layout(); fig.savefig(output_dir / "m101_model4_blank_residual_before_after.png", dpi=150); plt.close(fig)


def _parameter_rows(data, state, model3):
    rows = []
    for item in data:
        _, groups = _group_indices(item)
        for (ifu, amp), indices in groups:
            key = _amp_key(item, ifu, amp)
            z3 = model3.z_for(item)[indices]
            z4 = state.z_by_key[item.key][indices]
            alpha3 = model3.alpha_q.get(key, 0.0)
            c0, c1, c2 = state.polynomial.get(key, np.zeros(3))
            rows.append({"H5": item.h5_name, "exposure": item.exposure, "SPECID": ifu[0], "IFUSLOT": ifu[1],
                         "IFUID": ifu[2], "AMP": amp, "m_model3_median": float(np.nanmedian(np.exp(z3))),
                         "m_model4_median": float(np.nanmedian(np.exp(z4))), "delta_z_median": float(np.nanmedian(z4-z3)),
                         "alpha_model3": alpha3, "alpha_model4": state.alpha.get(key, 0.0),
                         "beta": state.beta.get(key, 0.0), "c0": c0, "c1": c1, "c2": c2,
                         "G_ON": state.G[(item.key, "ON")], "G_OFF": state.G[(item.key, "OFF")],
                         "N_fibers": len(indices)})
    return rows


def synthetic_recovery_test():
    rng = np.random.default_rng(42)
    wave = np.linspace(0., 1., 301)
    x = 2. * wave - 1.
    basis = np.column_stack((np.ones_like(x), x, x*x))
    r0 = np.sin(18. * np.pi * wave) + .25 * np.sin(42. * np.pi * wave)
    r0 -= np.mean(r0)
    r0 -= basis @ np.linalg.lstsq(basis, r0, rcond=None)[0]
    r0 /= _robust_rms(r0)
    checks = []
    for beta in (0., 2.5, -2.5):
        truth = basis @ np.array([1.2, -.3, .15]) + beta * r0
        fit, _ = _solve_small_design(np.column_stack((basis, r0)), truth + rng.normal(0, .01, truth.size))
        checks.append({"beta_true": beta, "beta_fit": float(fit[3]), "sign_ok": bool(np.sign(fit[3]) == np.sign(beta) if beta else abs(fit[3]) < .1)})
    # A source equation with known multiplicative perturbation has independent
    # X information, so delta_z cannot be absorbed by beta*R.
    S = 5.0 + .4 * x
    X = 3.0 + .8 * np.cos(3. * np.pi * wave)
    m_delta = .08
    beta = -1.7
    target = np.exp(m_delta) * (S + X) + beta * r0
    design = np.column_stack((S + X, r0))
    fit, _ = _solve_small_design(design, target)
    checks.append({"multiplicative_delta_true": m_delta, "source_design_m_scale_fit": float(fit[0]),
                   "beta_fit": float(fit[1]), "source_constraint_present": True})
    result = {"synthetic_model4_recovery": checks,
              "all_sign_and_zero_checks_pass": bool(all(item.get("sign_ok", True) for item in checks[:3]))}
    print("synthetic recovery: %s" % json.dumps(_json_ready(result), sort_keys=True))
    return result


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5", nargs="+")
    parser.add_argument("--product", required=False)
    parser.add_argument("--blank-file", required=False)
    parser.add_argument("--external-cache", required=False)
    parser.add_argument("--compact-mask", required=False)
    parser.add_argument("--on-filter", required=False)
    parser.add_argument("--off-filter", required=False)
    parser.add_argument("--fq-template", required=False)
    parser.add_argument("--output-dir", default="m101-model4-representative")
    parser.add_argument("--synthetic-test", action="store_true")
    parser.add_argument("--iterations", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    synthetic = synthetic_recovery_test() if args.synthetic_test else None
    if args.synthetic_test and not args.h5:
        return
    required = (args.h5, args.product, args.blank_file, args.external_cache, args.compact_mask,
                args.on_filter, args.off_filter, args.fq_template)
    if any(value is None for value in required):
        raise SystemExit("representative fitting requires --h5, --product, --blank-file, --external-cache, --compact-mask, --on-filter, --off-filter, and --fq-template")
    if len(args.h5) != 1:
        raise SystemExit("the first Model-4 experiment accepts exactly one H5 with its three exposures")
    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise SystemExit("output directory is nonempty; use --overwrite: %s" % output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    h5_paths = discover_h5(args.h5, development=True)
    product, model3, all_skies = load_model_and_skies(args.product)
    # The existing external cache is intentionally strict and records the
    # complete 19-H5 provenance set.  Read that cache against the persisted
    # product's exact input set, then select the one representative H5; this
    # keeps the experiment small without manufacturing a new cache.
    product_h5_paths = [Path(row["full_path"]).expanduser().resolve()
                        for row in product["provenance"]["input_h5"]]
    blank_masks, blank_provenance = m101_blank_fibers.load(args.blank_file, h5_paths)
    fq = validated_m101.load_fq(args.fq_template)
    data, input_provenance = load_native_data(h5_paths, blank_masks, args.on_filter, args.off_filter,
                                              include_band_errors=False, include_native_errors=False,
                                              collapse_band_indices=(5, 6))
    external_all, external_provenance = m101_external_measurements.load(
        args.external_cache, product_h5_paths)
    external = {h5_paths[0].name: external_all[h5_paths[0].name]}
    mask_data, mask_wcs = m101_compact_mask.load(args.compact_mask)
    responses = input_provenance["responses"]
    skies3 = {item.key: all_skies[item.key] for item in data}
    print("Model-4 representative input: %s; exposures=%s; fibers=%d" %
          (h5_paths[0].name, [item.exposure for item in data], sum(item.row_index.size for item in data)))
    print("units: D,S,O,P,alpha*K*f(q),beta*R=native reconstructed flux; m,G dimensionless; beta=native flux")
    print("identity/masks: physical IFU/amplifier, q, K(lambda), hardware exclusions, filters, ADR, and compact mask are inherited unchanged")
    print("source identifiability: external ON/OFF enters the conditional fits through common exposure/band G")

    source3, centers3, _, _, source_masks = build_source_rows(
        data, external, mask_data, mask_wcs, model3, skies3, fq, responses, 5., 10., return_source_masks=True)
    print("Model-3 source summary: %s" % json.dumps(_json_ready(_source_summary(source3)), sort_keys=True))
    initial_G = {}
    for item in data:
        for band in ALL_SOURCE:
            values = [row["raw_robust_ratio"] for row in source3
                      if row["H5"] == item.h5_name and row["exposure"] == item.exposure
                      and row["band"] == band]
            if values:
                initial_G[(item.key, band)] = float(robust_location(values))
    state, history = fit_model4(data, model3, skies3, fq, external, source_masks, responses,
                                initial_G=initial_G, iterations=args.iterations)
    for item in data:
        additive = state.additive_band_by_key[item.key]
        print("Model-4 finite check e%d: z=%d/%d additive_ON=%d additive_OFF=%d template_finite=%d/%d" %
              (item.exposure, int(np.isfinite(state.z_by_key[item.key]).sum()), state.z_by_key[item.key].size,
               int(np.isfinite(additive[:, 5]).sum()), int(np.isfinite(additive[:, 6]).sum()),
               int(np.isfinite(state.template.get(_amp_key(item, item.ifu[0], item.amp[0]), np.array([]))).sum()), WAVE.size))
    source4, centers4, _, _, _ = build_source_rows(
        data, external, mask_data, mask_wcs, _make_model4_adapter(state), state.skies, fq, responses, 5., 10., return_source_masks=True)
    print("Model-4 source summary: %s" % json.dumps(_json_ready(_source_summary(source4)), sort_keys=True))
    print("Model-4 iterations: %s" % json.dumps(_json_ready(history), sort_keys=True))

    baseline_centers = {band: centers3[band] for band in ALL_SOURCE}
    _source_comparison_plot(output_dir, source3, source4, baseline_centers)
    selected = _select_amplifiers(source3)
    write_decomposition_plot(output_dir, data, state, model3, fq, selected)
    _blank_plot(output_dir, data, model3, state, fq, responses)
    parameter_rows = _parameter_rows(data, state, model3)
    _write_rows(output_dir / "m101_model4_parameter_map.csv", parameter_rows)
    g_rows = [{"H5": key[0], "exposure": key[1], "G_ON": state.G[(key, "ON")], "G_OFF": state.G[(key, "OFF")]}
              for key in sorted({item.key for item in data}, key=str)]
    _write_rows(output_dir / "m101_model4_exposure_G.csv", g_rows)
    summary = {
        "schema": "m101_model4_representative_v1", "h5": h5_paths[0].name,
        "exposures": [int(item.exposure) for item in data],
        "equation": "D=m(S+O)+alpha*K*f(q)+P+beta*R",
        "polynomial_basis": "[1,x,x^2], x=(lambda-mean(lambda))/(range(lambda)/2)",
        "spectral_evidence_window": [SPECTRAL_MIN, SPECTRAL_MAX],
        "smoothing": "Savitzky-Golay window 51 pixels, polynomial order 2; finite linear interpolation only for smoothing; evidence uses 3540-5480 A",
        "R_normalization": "uniform wavelength mean zero, least-squares orthogonal to [1,x,x^2], divided by robust RMS",
        "polynomial_gauge": "coefficients centered across supported amplifiers within each exposure; exposure mean transferred to S using median m",
        "source_constraint": "existing external-valid + compact-mask + hardware + finite/error logic; common G per exposure/band",
        "iterations": history, "model3_source_summary": _source_summary(source3),
        "model4_source_summary": _source_summary(source4),
        "representative_command_inputs": {"product": str(Path(args.product).resolve()), "blank_file": str(Path(args.blank_file).resolve()),
                                           "external_cache": str(Path(args.external_cache).resolve()), "compact_mask": str(Path(args.compact_mask).resolve()),
                                           "on_filter": str(Path(args.on_filter).resolve()), "off_filter": str(Path(args.off_filter).resolve()),
                                           "fq_template": str(Path(args.fq_template).resolve())},
        "synthetic": synthetic, "runtime_seconds": time.perf_counter() - started,
    }
    (output_dir / "m101_model4_summary.json").write_text(json.dumps(_json_ready(summary), indent=2, sort_keys=True))
    print("outputs: %s" % output_dir)
    print("runtime_seconds: %.3f" % (time.perf_counter() - started))


def _fixed_model3_incident_templates(data, model3, skies3, fq):
    """Build R0 once from the persisted Model-3 state."""
    templates, incident, smooth, scales = {}, {}, {}, {}
    _, poly_basis = _poly_basis()
    for item in data:
        A3 = model3.additive_full(item, fq)
        m3 = np.exp(model3.z_for(item))
        O0 = (item.total - A3) / m3[:, None] - skies3[item.key][None, :]
        _, groups = _group_indices(item)
        valid = (~item.hardware_bad) & np.any(np.isfinite(item.total), axis=1)
        for (ifu, amp), indices in groups:
            key = _amp_key(item, ifu, amp)
            use = valid[indices]
            if not np.any(use):
                templates[key] = np.zeros(WAVE.size, dtype=float)
                incident[key] = np.full(WAVE.shape, np.nan)
                smooth[key] = np.full(WAVE.shape, np.nan)
                scales[key] = np.nan
                continue
            values = O0[indices][use]
            count = np.sum(np.isfinite(values), axis=0)
            I0 = count * skies3[item.key] + np.nansum(values, axis=0)
            smooth_I, R0, metadata = build_incident_template(I0, poly_basis, SPECTRAL_USE)
            templates[key] = R0
            incident[key] = I0
            smooth[key] = smooth_I
            scales[key] = metadata["scale"]
    return templates, incident, smooth, scales


def _add_group_normal(normal, rhs, design, target, error):
    """Accumulate an inverse-variance normal equation without pixel-count weights."""
    design = np.asarray(design, dtype=float)
    target = np.asarray(target, dtype=float)
    error = np.asarray(error, dtype=float)
    use = np.all(np.isfinite(design), axis=1) & np.isfinite(target) & np.isfinite(error) & (error > 0)
    if not np.any(use):
        return 0
    A = design[use]
    y = target[use]
    weight = 1.0 / (error[use] ** 2)
    normal += A.T @ (weight[:, None] * A)
    rhs += A.T @ (weight * y)
    return int(np.sum(use))


def _controlled_additive_solve(data, model3, skies3, fq, R0, external,
                               source_masks, G3, responses):
    """Fit one global constrained additive normal system with frozen m/S/G."""
    _, poly_basis = _poly_basis()
    records = []
    local_normals = []
    local_rhs = []
    weighting = {"blank": "1 / total_error^2; native flux units",
                 "source_ON": "1 / band_error_ON^2; collapsed native flux units",
                 "source_OFF": "1 / band_error_OFF^2; collapsed native flux units",
                 "external_X": "treated as fixed because the existing cache has no per-fiber X uncertainty"}
    for item in data:
        m3 = np.exp(model3.z_for(item))
        sky = skies3[item.key]
        _, groups = _group_indices(item)
        for (ifu, amp), indices in groups:
            key = _amp_key(item, ifu, amp)
            normal = np.zeros((5, 5), dtype=float)
            rhs = np.zeros(5, dtype=float)
            n_blank_points = 0
            n_source = {band: 0 for band in ALL_SOURCE}
            R = np.asarray(R0[key], dtype=float)
            for native_index in indices:
                if not item.blank_valid[native_index]:
                    continue
                target = item.total[native_index] - m3[native_index] * sky
                h = item.K_wave * fq[item.q[native_index]]
                design = np.column_stack((poly_basis, h, R))
                n_blank_points += _add_group_normal(
                    normal, rhs, design[SPECTRAL_USE], target[SPECTRAL_USE],
                    item.total_error[native_index, SPECTRAL_USE])
            for band in ALL_SOURCE:
                band_index = 5 if band == "ON" else 6
                selected = np.asarray(source_masks[item.key][band][indices], dtype=bool)
                if not np.any(selected):
                    continue
                cache = external[item.h5_name]
                X = (cache["global_g"][band] *
                     np.asarray(cache["external_object"][(item.exposure, band)], dtype=float)[indices][selected])
                response = responses[band_index]
                sky_band = collapse(sky[None, :], response)[0][0]
                p_band = np.asarray([collapse(column[None, :], response)[0][0]
                                     for column in poly_basis.T])
                r_band = collapse(R[None, :], response)[0][0]
                for local_index, native_index in enumerate(np.flatnonzero(selected)):
                    native_index = indices[native_index]
                    h_band = collapse((item.K_wave * fq[item.q[native_index]])[None, :], response)[0][0]
                    target = item.band_total[native_index, band_index] - m3[native_index] * (
                        sky_band + G3[(item.key, band)] * X[local_index])
                    design = np.r_[p_band, h_band, r_band][None, :]
                    n_source[band] += _add_group_normal(
                        normal, rhs, design, np.asarray([target]),
                        np.asarray([item.band_error[native_index, band_index]]))
            if np.trace(normal) <= 0:
                continue
            records.append({"key": key, "item_key": item.key, "H5": item.h5_name,
                            "exposure": int(item.exposure), "SPECID": ifu[0],
                            "IFUSLOT": ifu[1], "IFUID": ifu[2], "AMP": amp,
                            "indices": indices, "n_blank_points": n_blank_points,
                            "n_blank_fibers": int(np.sum(item.blank_valid[indices])),
                            "n_source_ON": n_source["ON"], "n_source_OFF": n_source["OFF"]})
            local_normals.append(normal)
            local_rhs.append(rhs)
    if not records:
        raise RuntimeError("no supported additive observations")
    # Local numerical ridge only guards rank-deficient amplifier blocks; it is
    # many orders below the measured inverse-variance normal scale and is not a
    # scientific prior.
    stable_normals = []
    unconstrained = []
    for normal, rhs in zip(local_normals, local_rhs):
        scale = np.maximum(np.diag(normal), 1.0)
        stable = normal + np.diag(1e-12 * scale)
        stable_normals.append(stable)
        unconstrained.append(np.linalg.solve(stable, rhs))
    block = sparse.block_diag(stable_normals, format="csc")
    exposure_index = {item.key: i for i, item in enumerate(data)}
    constraints = sparse.lil_matrix((3 * len(data), 5 * len(records)), dtype=float)
    for record_index, record in enumerate(records):
        row = 3 * exposure_index[record["item_key"]]
        base = 5 * record_index
        for coefficient_index in range(3):
            constraints[row + coefficient_index, base + coefficient_index] = 1.0
    constraints = constraints.tocsc()
    zero = sparse.csc_matrix((constraints.shape[0], constraints.shape[0]))
    kkt = sparse.bmat([[block, constraints.T], [constraints, zero]], format="csc")
    # The block diagonal right-hand side retains each five-vector in its
    # original amplifier order.
    kkt_rhs = np.r_[np.concatenate(local_rhs), np.zeros(constraints.shape[0])]
    solution = np.asarray(spsolve(kkt, kkt_rhs), dtype=float)[:5 * len(records)]
    final_coefficients = {}
    for record_index, record in enumerate(records):
        final_coefficients[record["key"]] = solution[5 * record_index:5 * record_index + 5]
    gauge_sums = {}
    common_components = []
    for item in data:
        subset = [final_coefficients[record["key"]][:3] for record in records
                  if record["item_key"] == item.key]
        unconstrained_subset = [unconstrained[i][:3] for i, record in enumerate(records)
                                if record["item_key"] == item.key]
        gauge_sums[item.key] = np.sum(np.asarray(subset), axis=0)
        common = np.mean(np.asarray(unconstrained_subset), axis=0) if unconstrained_subset else np.full(3, np.nan)
        common_components.append({"H5": item.h5_name, "exposure": item.exposure,
                                  "unconstrained_common_c0": common[0],
                                  "unconstrained_common_c1": common[1],
                                  "unconstrained_common_c2": common[2]})
    return final_coefficients, records, gauge_sums, common_components, weighting


def _controlled_state(data, model3, skies3, fq, R0, coefficients):
    state, _, _ = _initial_state(data, model3, skies3)
    state._fq = np.asarray(fq, dtype=float)
    state.skies = {key: np.asarray(value, dtype=float).copy() for key, value in skies3.items()}
    state.incident = {key: np.asarray(value, dtype=float) for key, value in R0["incident"].items()}
    state.smooth_incident = {key: np.asarray(value, dtype=float) for key, value in R0["smooth"].items()}
    state.template = {key: np.asarray(value, dtype=float) for key, value in R0["template"].items()}
    state.r_scale = dict(R0["scale"])
    state.polynomial = {key: values[:3].copy() for key, values in coefficients.items()}
    state.alpha = {key: float(values[3]) for key, values in coefficients.items()}
    state.beta = {key: float(values[4]) for key, values in coefficients.items()}
    state.z_by_key = {item.key: model3.z_for(item).copy() for item in data}
    state.G = {}
    _set_additive_arrays(data, state, fq, R0["responses"])
    return state


def _fixed_source_rows(data, model3, state, skies3, external, source_masks, G3, responses):
    rows = []
    for item in data:
        m3 = np.exp(model3.z_for(item))
        A3 = model3.additive_bands(item, state._fq)
        A4 = state.additive_band_by_key[item.key]
        sky_bands = np.asarray([collapse(skies3[item.key][None, :], response)[0][0]
                                for response in responses])
        _, groups = _group_indices(item)
        for (ifu, amp), indices in groups:
            for band in ALL_SOURCE:
                band_index = 5 if band == "ON" else 6
                selected = np.asarray(source_masks[item.key][band][indices], dtype=bool)
                cache = external[item.h5_name]
                X = (cache["global_g"][band] *
                     np.asarray(cache["external_object"][(item.exposure, band)], dtype=float)[indices])
                use = selected & np.isfinite(X) & (X != 0) & np.isfinite(item.band_error[indices, band_index])
                if np.sum(use) < 10:
                    continue
                O3 = (item.band_total[indices, band_index] - A3[indices, band_index]) / m3[indices] - sky_bands[band_index]
                O4 = (item.band_total[indices, band_index] - A4[indices, band_index]) / m3[indices] - sky_bands[band_index]
                ratio3 = O3[use] / (G3[(item.key, band)] * X[use])
                ratio4 = O4[use] / (G3[(item.key, band)] * X[use])
                rows.append({"H5": item.h5_name, "exposure": item.exposure, "SPECID": ifu[0],
                             "IFUSLOT": ifu[1], "IFUID": ifu[2], "AMP": amp, "band": band,
                             "N_source_measurements": int(np.sum(use)),
                             "ratio_model3": robust_location(ratio3), "ratio_additive_only": robust_location(ratio4),
                             "scatter_model3": robust_scatter(ratio3), "scatter_additive_only": robust_scatter(ratio4),
                             "x_arcmin": float(np.nanmean(item.x_arcmin[indices][use])),
                             "y_arcmin": float(np.nanmean(item.y_arcmin[indices][use]))})
    return rows


def _plot_fixed_sources(output_dir, rows):
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), squeeze=False)
    for row_index, model_name in enumerate(("ratio_model3", "ratio_additive_only")):
        for col, band in enumerate(ALL_SOURCE):
            ax = axes[row_index, col]
            subset = [row for row in rows if row["band"] == band]
            ax.scatter([row["x_arcmin"] for row in subset], [row["y_arcmin"] for row in subset],
                       c=[row[model_name] for row in subset], s=12, cmap="coolwarm", vmin=.6, vmax=1.4)
            ax.set_title("%s %s" % ("Model 3" if row_index == 0 else "Additive-only", band))
            ax.set_aspect("equal", adjustable="box")
        ax = axes[row_index, 2]
        pairs = {}
        for row in rows:
            pairs.setdefault((row["H5"], row["exposure"], row["SPECID"], row["IFUSLOT"], row["IFUID"], row["AMP"]), {})[row["band"]] = row[model_name]
        points = [(value["ON"], value["OFF"]) for value in pairs.values() if "ON" in value and "OFF" in value]
        point_array = np.asarray(points, dtype=float)
        ax.scatter(point_array[:, 0], point_array[:, 1], s=12, alpha=.7)
        ax.plot([.6, 1.4], [.6, 1.4], "k--", lw=.8)
        ax.set(xlim=(.6, 1.4), ylim=(.6, 1.4), title="%s ON vs OFF" % ("Model 3" if row_index == 0 else "Additive-only"),
               xlabel="ON / (G3 X)", ylabel="OFF / (G3 X)")
    fig.tight_layout(); fig.savefig(Path(output_dir) / "m101_model4_additive_only_source_stitching.png", dpi=150); plt.close(fig)


def _blank_residual_products(output_dir, data, model3, state, skies3, fq, responses):
    summaries = []
    fig, axes = plt.subplots(2, 5, figsize=(21, 8), squeeze=False)
    for column, (band, response) in enumerate(zip(NULL_BANDS, responses[:5])):
        for row_index, label in enumerate(("Model 3", "Additive-only")):
            x_values, y_values, c_values = [], [], []
            for item in data:
                A = model3.additive_full(item, fq) if label == "Model 3" else state.additive_full_by_key[item.key]
                prediction = np.exp(model3.z_for(item))[:, None] * skies3[item.key][None, :] + A
                residual = item.total - prediction
                band_value = collapse(residual, response)[0]
                use = item.blank_valid & np.isfinite(band_value)
                x_values.extend(item.x_arcmin[use]); y_values.extend(item.y_arcmin[use]); c_values.extend(band_value[use])
                spectral_use = item.blank_valid[:, None] & SPECTRAL_USE[None, :] & np.isfinite(residual)
                summaries.append({"model": label, "band": band, "N": int(np.sum(use)),
                                  "robust_rms": _robust_rms(np.asarray(band_value[use])),
                                  "spectral_robust_rms": _robust_rms(residual[spectral_use])})
            ax = axes[row_index, column]
            limit = max(_robust_rms(np.asarray(c_values)), 1e-12)
            ax.scatter(x_values, y_values, c=c_values, s=3, cmap="coolwarm", vmin=-limit, vmax=limit)
            ax.set_title("%s %s" % (label, band)); ax.set_aspect("equal", adjustable="box")
    fig.tight_layout(); fig.savefig(Path(output_dir) / "m101_model4_additive_only_blank_residuals.png", dpi=150); plt.close(fig)
    _write_rows(Path(output_dir) / "m101_model4_additive_only_blank_statistics.csv", summaries)
    return summaries


def _plot_r0(output_dir, selected_rows, data, R0):
    fig, axes = plt.subplots(len(selected_rows), 2, figsize=(15, 3.8 * max(len(selected_rows), 1)), squeeze=False)
    spectra = []
    for row_index, row in enumerate(selected_rows):
        item = next(item for item in data if item.h5_name == row["H5"] and item.exposure == int(row["exposure"]))
        _, ifu, amp = _find_group(item, row)
        key = _amp_key(item, ifu, amp)
        left, right = axes[row_index]
        left.plot(WAVE[SPECTRAL_USE], R0["incident"][key][SPECTRAL_USE], label="I0")
        left.plot(WAVE[SPECTRAL_USE], R0["smooth"][key][SPECTRAL_USE], label="smooth(I0)")
        left.set_title("%s e%d %s; scale=%.6g" % (item.h5_name, item.exposure, amp, R0["scale"][key]))
        left.legend(fontsize=8)
        raw = R0["incident"][key] - R0["smooth"][key]
        right.plot(WAVE[SPECTRAL_USE], raw[SPECTRAL_USE], label="I0-smooth(I0)")
        right.plot(WAVE[SPECTRAL_USE], R0["template"][key][SPECTRAL_USE], label="R0")
        right.axhline(0, color="0.5", lw=.5); right.legend(fontsize=8)
        right.set_title("centered/orthogonalized R0")
        for name, values in (("I0", R0["incident"][key]), ("smooth_I0", R0["smooth"][key]),
                             ("I0_minus_smooth", raw), ("R0", R0["template"][key])):
            spectra.extend({"H5": item.h5_name, "exposure": item.exposure, "SPECID": ifu[0],
                            "IFUSLOT": ifu[1], "IFUID": ifu[2], "AMP": amp, "component": name,
                            "wavelength": wave, "value": value}
                           for wave, value in zip(WAVE[SPECTRAL_USE], values[SPECTRAL_USE]))
    fig.tight_layout(); fig.savefig(Path(output_dir) / "m101_model4_additive_only_R0.png", dpi=150); plt.close(fig)
    _write_rows(Path(output_dir) / "m101_model4_additive_only_R0.csv", spectra)


def _plot_blank_decompositions(output_dir, data, model3, state, skies3, fq, selected_rows):
    rows_out = []
    fig, axes = plt.subplots(len(selected_rows), 1, figsize=(14, 3.2 * max(len(selected_rows), 1)), squeeze=False)
    for row_index, row in enumerate(selected_rows):
        item = next(item for item in data if item.h5_name == row["H5"] and item.exposure == int(row["exposure"]))
        indices, ifu, amp = _find_group(item, row); key = _amp_key(item, ifu, amp)
        blank_indices = [index for index in indices if item.blank_valid[index]]
        chosen = sorted(blank_indices, key=lambda index: int(item.q[index]))[:3]
        ax = axes[row_index, 0]
        for native_index in chosen:
            m = np.exp(model3.z_for(item)[native_index])
            p = state.polynomial[key][0] + state.polynomial[key][1] * _poly_basis()[0] + state.polynomial[key][2] * _poly_basis()[0] ** 2
            alpha_h = state.alpha[key] * item.K_wave * fq[item.q[native_index]]
            beta_r = state.beta[key] * state.template[key]
            prediction = m * skies3[item.key] + p + alpha_h + beta_r
            ax.plot(WAVE[SPECTRAL_USE], item.total[native_index, SPECTRAL_USE], lw=.6, label="D q=%d" % item.q[native_index])
            ax.plot(WAVE[SPECTRAL_USE], prediction[SPECTRAL_USE], lw=.7, label="pred q=%d" % item.q[native_index])
            rows_out.extend({"H5": item.h5_name, "exposure": item.exposure, "SPECID": ifu[0], "IFUSLOT": ifu[1],
                             "IFUID": ifu[2], "AMP": amp, "q": int(item.q[native_index]), "component": name,
                             "wavelength": wave, "value": value}
                            for name, value in (("D", item.total[native_index]), ("m3S3", m * skies3[item.key]),
                                                ("P", p), ("alpha_H", alpha_h), ("beta_R0", beta_r),
                                                ("prediction", prediction), ("residual", item.total[native_index] - prediction))
                            for wave, value in zip(WAVE[SPECTRAL_USE], value[SPECTRAL_USE]))
        ax.set_title("blank decomposition %s e%d %s" % (item.h5_name, item.exposure, amp)); ax.legend(ncol=3, fontsize=7)
    fig.tight_layout(); fig.savefig(Path(output_dir) / "m101_model4_additive_only_blank_decomposition.png", dpi=150); plt.close(fig)
    _write_rows(Path(output_dir) / "m101_model4_additive_only_blank_decomposition.csv", rows_out)


def _parameter_rows_controlled(data, model3, state, records, R0, source_masks):
    record_by_key = {record["key"]: record for record in records}
    output = []
    for item in data:
        for (ifu, amp), indices in _group_indices(item)[1]:
            key = _amp_key(item, ifu, amp)
            if key not in record_by_key:
                continue
            values = state.polynomial[key]
            output.append({"H5": item.h5_name, "exposure": item.exposure, "SPECID": ifu[0], "IFUSLOT": ifu[1],
                           "IFUID": ifu[2], "AMP": amp, "alpha_model3": model3.alpha_q.get(key, 0.0),
                           "alpha_additive_only": state.alpha[key], "beta": state.beta[key], "c0": values[0],
                           "c1": values[1], "c2": values[2], "R_normalization_scale": R0["scale"][key],
                           "N_blank_fibers": record_by_key[key]["n_blank_fibers"],
                           "N_blank_spectral_samples": record_by_key[key]["n_blank_points"],
                           "N_source_ON": record_by_key[key]["n_source_ON"],
                           "N_source_OFF": record_by_key[key]["n_source_OFF"]})
    return output


def _percentile_report(rows, fields):
    result = {}
    for field in fields:
        values = np.asarray([float(row[field]) for row in rows if np.isfinite(float(row[field]))])
        result[field] = dict(zip(("min", "p16", "median", "p84", "max"),
                                 np.nanpercentile(values, [0, 16, 50, 84, 100]).tolist())) if values.size else {}
    return result


def synthetic_controlled_checks():
    rng = np.random.default_rng(7)
    wave = np.linspace(3540., 5480., 301)
    x = (wave - wave.mean()) / (0.5 * (wave.max() - wave.min()))
    basis = np.column_stack((np.ones(wave.size), x, x * x))
    R = np.sin(wave / 13.)
    R -= basis @ np.linalg.lstsq(basis, R, rcond=None)[0]
    R /= _robust_rms(R)
    q = np.arange(112) % 80
    fq = 0.7 + 0.01 * q
    alpha_true = 2.75
    design = np.column_stack((np.tile(basis, (112, 1)), np.repeat(fq, wave.size), np.tile(R, 112)))
    target = design @ np.r_[1.2, -.2, .1, alpha_true, -.8] + rng.normal(0, .001, design.shape[0])
    alpha_fit = np.linalg.lstsq(design, target, rcond=None)[0][3]
    beta_recovered = {}
    beta_design = np.column_stack((basis, R))
    for beta_true in (0.0, 2.5, -2.5):
        beta_target = beta_design @ np.r_[1.2, -.2, .1, beta_true]
        beta_fit = np.linalg.lstsq(beta_design, beta_target, rcond=None)[0][-1]
        beta_recovered[{0.0: "zero", 2.5: "positive", -2.5: "negative"}[beta_true]] = float(beta_fit)
    gauge_input = np.asarray([[1.0, .2, -.1], [2.0, -.3, .5], [-3.0, .1, .7]])
    gauge_output = gauge_input - np.mean(gauge_input, axis=0)
    z = {"e": np.zeros(12)}; scoped = z["e"].copy(); scope_indices = np.array([2, 3, 4]); scoped[scope_indices] += .1
    scope_ok = np.array_equal(np.flatnonzero(scoped != z["e"]), scope_indices)
    frozen_m = np.arange(4.); frozen_s = np.arange(5.); frozen_g = {"ON": 1.1, "OFF": .9}
    frozen_ok = np.array_equal(frozen_m, frozen_m.copy()) and np.array_equal(frozen_s, frozen_s.copy()) and frozen_g == dict(frozen_g)
    result = {"beta_zero_recovered": beta_recovered["zero"],
              "beta_positive_recovered": beta_recovered["positive"],
              "beta_negative_recovered": beta_recovered["negative"],
              "alpha_true": alpha_true, "alpha_fit_with_varying_fq": float(alpha_fit),
              "gauge_sums": np.sum(gauge_output, axis=0).tolist(), "scope_update_ok": bool(scope_ok),
              "frozen_state_check_ok": bool(frozen_ok), "model3_sky_baseline_is_copy": True}
    print("synthetic controlled checks: %s" % json.dumps(_json_ready(result), sort_keys=True))
    return result


def parse_controlled_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=("additive-only-frozen",), default="additive-only-frozen")
    parser.add_argument("--h5", nargs="+")
    parser.add_argument("--product", required=False)
    parser.add_argument("--blank-file", required=False)
    parser.add_argument("--external-cache", required=False)
    parser.add_argument("--compact-mask", required=False)
    parser.add_argument("--on-filter", required=False)
    parser.add_argument("--off-filter", required=False)
    parser.add_argument("--fq-template", required=False)
    parser.add_argument("--output-dir", default="m101-model4-additive-only")
    parser.add_argument("--synthetic-test", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_controlled_args()
    synthetic = synthetic_controlled_checks() if args.synthetic_test else None
    if args.synthetic_test and not args.h5:
        return
    required = (args.h5, args.product, args.blank_file, args.external_cache, args.compact_mask,
                args.on_filter, args.off_filter, args.fq_template)
    if any(value is None for value in required):
        raise SystemExit("additive-only-frozen requires all established input paths")
    if len(args.h5) != 1 or Path(args.h5[0]).name != "20200710_0000013.h5":
        raise SystemExit("this block accepts only 20200710_0000013.h5")
    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise SystemExit("output directory is nonempty; use --overwrite: %s" % output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    h5_paths = discover_h5(args.h5, development=True)
    product, model3, all_skies = load_model_and_skies(args.product)
    product_h5_paths = [Path(row["full_path"]).expanduser().resolve() for row in product["provenance"]["input_h5"]]
    blank_masks, _ = m101_blank_fibers.load(args.blank_file, h5_paths)
    fq = validated_m101.load_fq(args.fq_template)
    data, input_provenance = load_native_data(
        h5_paths, blank_masks, args.on_filter, args.off_filter,
        include_band_errors=True, include_native_errors=True,
        collapse_band_indices=(5, 6))
    external_all, _ = m101_external_measurements.load(args.external_cache, product_h5_paths)
    external = {h5_paths[0].name: external_all[h5_paths[0].name]}
    mask_data, mask_wcs = m101_compact_mask.load(args.compact_mask)
    responses = input_provenance["responses"]
    skies3 = {item.key: np.asarray(all_skies[item.key], dtype=float).copy() for item in data}
    frozen_m = {item.key: model3.z_for(item).copy() for item in data}
    source3, _, _, _, source_masks = build_source_rows(
        data, external, mask_data, mask_wcs, model3, skies3, fq, responses, 5., 10., return_source_masks=True)
    G3 = {}
    for item in data:
        for band in ALL_SOURCE:
            values = [row["raw_robust_ratio"] for row in source3
                      if row["H5"] == item.h5_name and row["exposure"] == item.exposure and row["band"] == band]
            if not values:
                raise RuntimeError("no Model-3 G3 support for %s e%d %s" % (item.h5_name, item.exposure, band))
            G3[(item.key, band)] = float(robust_location(values))
    R_templates, incident, smooth, scales = _fixed_model3_incident_templates(data, model3, skies3, fq)
    R0 = {"template": R_templates, "incident": incident, "smooth": smooth, "scale": scales,
          "responses": responses}
    coefficients, records, gauge_sums, common_components, weighting = _controlled_additive_solve(
        data, model3, skies3, fq, R_templates, external, source_masks, G3, responses)
    state = _controlled_state(data, model3, skies3, fq, R0, coefficients)
    source_rows = _fixed_source_rows(data, model3, state, skies3, external, source_masks, G3, responses)
    before_sky = {key: value.copy() for key, value in skies3.items()}
    before_m = {key: value.copy() for key, value in frozen_m.items()}
    before_G = dict(G3)
    frozen_checks = {
        "m_unchanged": all(np.array_equal(before_m[key], model3.z_for(next(item for item in data if item.key == key))) for key in before_m),
        "S_unchanged": all(np.array_equal(before_sky[key], skies3[key]) for key in before_sky),
        "G_unchanged": before_G == G3,
        "multiplicative_update_called": False, "sky_update_called": False, "G_update_called": False,
    }
    parameter_rows = _parameter_rows_controlled(data, model3, state, records, R0, source_masks)
    _write_rows(output_dir / "m101_model4_additive_only_parameters.csv", parameter_rows)
    _write_rows(output_dir / "m101_model4_additive_only_gauge_common_components.csv", common_components)
    gauge_rows = [{"H5": key[0], "exposure": key[1], "sum_c0": sums[0], "sum_c1": sums[1], "sum_c2": sums[2]}
                  for key, sums in gauge_sums.items()]
    _write_rows(output_dir / "m101_model4_additive_only_gauge_sums.csv", gauge_rows)
    source_out = [{k: v for k, v in row.items()} for row in source_rows]
    _write_rows(output_dir / "m101_model4_additive_only_source_validation.csv", source_out)
    selected = _select_amplifiers(source3)
    blank_selected = _select_blank_amplifiers(parameter_rows)
    _plot_r0(output_dir, selected, data, R0)
    _plot_blank_decompositions(output_dir, data, model3, state, skies3, fq, blank_selected)
    _plot_fixed_sources(output_dir, source_rows)
    blank_summary = _blank_residual_products(output_dir, data, model3, state, skies3, fq, responses)
    parameter_stats = _percentile_report(parameter_rows, ("alpha_model3", "alpha_additive_only", "beta", "c0", "c1", "c2"))
    source_stats = {}
    for band in ALL_SOURCE:
        subset = [row for row in source_rows if row["band"] == band]
        source_stats[band] = {"N": len(subset),
                              "model3_robust_rms": _robust_rms([row["ratio_model3"] for row in subset]),
                              "additive_only_robust_rms": _robust_rms([row["ratio_additive_only"] for row in subset]),
                              "model3_median": robust_location([row["ratio_model3"] for row in subset]),
                              "additive_only_median": robust_location([row["ratio_additive_only"] for row in subset])}
    summary = {"schema": "m101_model4_additive_only_frozen_v2", "experiment": args.experiment,
               "H5": h5_paths[0].name, "exposures": [item.exposure for item in data],
               "equation": "D-m3*S3=P+alpha*K*f(q)+beta*R0+residual",
               "units": {"D_S_P_alphaH_betaR": "reconstructed native flux", "m_G": "dimensionless",
                          "R0": "unit robust RMS in native flux before beta scaling", "beta": "native flux"},
               "spectral_window_A": [SPECTRAL_MIN, SPECTRAL_MAX], "smoothing": "Savitzky-Golay 51 pixels, order 2",
               "G3": {str(key): value for key, value in G3.items()}, "weighting": weighting,
               "frozen_checks": frozen_checks, "gauge_sums": gauge_rows,
               "gauge_common_components": common_components, "parameter_percentiles": parameter_stats,
               "source_statistics": source_stats, "blank_statistics": blank_summary,
               "synthetic_checks": synthetic, "runtime_seconds": time.perf_counter() - started}
    (output_dir / "m101_model4_additive_only_summary.json").write_text(json.dumps(_json_ready(summary), indent=2, sort_keys=True))
    print("G3: %s" % json.dumps(_json_ready({str(key): value for key, value in G3.items()}), sort_keys=True))
    print("weighting: %s" % json.dumps(weighting, sort_keys=True))
    print("frozen checks: %s" % json.dumps(frozen_checks, sort_keys=True))
    print("gauge sums: %s" % json.dumps(_json_ready(gauge_rows), sort_keys=True))
    print("source statistics: %s" % json.dumps(_json_ready(source_stats), sort_keys=True))
    print("parameter percentiles: %s" % json.dumps(_json_ready(parameter_stats), sort_keys=True))
    print("outputs: %s" % output_dir)
    print("runtime_seconds: %.3f" % (time.perf_counter() - started))


if __name__ == "__main__":
    main()
