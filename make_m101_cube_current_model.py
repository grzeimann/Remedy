#!/usr/bin/env python3
"""Build an M101 VIRUS cube from the current staged calibration model.

This is a cube-ingestion script, not a calibration fitter.  It deliberately
uses the current physics-first products that were developed after Model 3:

  1. Frozen gray multiplicative response

         M(h, i, a) = exp(R_total_h_i_a)

     from m101_onoff_hierarchy/m101_onoff_state.json, specifically
     hierarchy.cumulative_final.R_total_h_i_a.

  2. q-dependent additive correction

         A_q(f, lambda) = alpha * (f_q(q_f) - f0) * K_e(lambda)

     using a directly measured alpha_joint for blank-rich amplifier/exposures
     when available, otherwise the historical robust alpha for the same
     physical amplifier.

  3. Persistent full-resolution additive amplifier spectrum

         mu_a(lambda)

     from the robust-mean arrays written by diagnose_m101_amp_residual_pca.py.
     The 3 PCA coefficients are intentionally NOT applied in this first cube
     model; they are descriptive/current-state degrees of freedom that are not
     predictable when an amplifier has no blank fibers.

The native total spectrum is

    D_f(lambda) = Fibers.spectrum / Survey.offset + Fibers.skyspectrum.

The model-corrected total is

    T_f(lambda) = [D_f(lambda) - A_q(f,lambda) - mu_a(lambda)] / M(h,i,a).

For each H5 exposure, the final sky is estimated from externally classified,
hardware-valid blank fibers with 40 <= q <= 75:

    S_e(lambda) = robust_location_blank T_f(lambda)

and the spectrum sent to the cube reconstruction is

    O_f(lambda) = T_f(lambda) - S_e(lambda).

The ADR, Gaussian-splat image reconstruction, variance, DQ, coverage, and FITS
machinery is inherited from the older make_mosaic_cube_from_fit.py workflow.
Unsupported frozen-M channels are masked, matching the current diagnostic
semantics.  A channel that has M but is missing alpha or mu_a is treated as a
product-consistency error and the script stops rather than silently changing
models.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
from datetime import datetime, timezone
import glob
import json
from pathlib import Path
import time

import numpy as np
import tables
from astropy.io import fits
from numba import njit

from astrometry import Astrometry
from extract import Extract
import diagnose_m101_hierarchical as validated_m101
import diagnose_m101_frozenM_spectral_residuals as frozen_parent
from math_utils import biweight


DEF_WAVE = np.asarray(validated_m101.DEF_WAVE, dtype=float)
N_FIBER_AMP = validated_m101.N_FIBER_AMP
N_EXPOSURES = validated_m101.N_EXPOSURES
FIBER_RADIUS_ARCSEC = validated_m101.FIBER_RADIUS_ARCSEC
Q_MIN = 40
Q_MAX = 75
M101_SKY_MIN_FINITE_FRACTION = 0.8
M101_SKY_MIN_FIBERS = 20
GAUSSIAN_FWHM_ARCSEC = 1.8

DQ_INSUFFICIENT_SUPPORT = np.uint16(1 << 0)
DQ_VAR_INCOMPLETE = np.uint16(1 << 1)
DQ_EMPIRICAL_VAR_USED = np.uint16(1 << 2)
DQ_FORMAL_VAR_USED = np.uint16(1 << 3)
DQ_VAR_EMPIRICAL_ONLY = np.uint16(1 << 4)


# -----------------------------------------------------------------------------
# Small utilities and model-product readers
# -----------------------------------------------------------------------------


def _text(value):
    return validated_m101.as_text(value)


def _file_identity(path):
    path = Path(path).expanduser().resolve()
    stat = path.stat()
    return {
        "path": str(path),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _distribution(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not values.size:
        return {
            "N": 0, "min": np.nan, "p16": np.nan, "p50": np.nan,
            "p84": np.nan, "max": np.nan,
        }
    p16, p50, p84 = np.percentile(values, (16, 50, 84))
    return {
        "N": int(values.size),
        "min": float(np.min(values)),
        "p16": float(p16),
        "p50": float(p50),
        "p84": float(p84),
        "max": float(np.max(values)),
    }


def _write_rows(path, rows):
    rows = list(rows)
    path = Path(path)
    if not rows:
        path.write_text("")
        return
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})


def _physical_key(specid, ifuslot, ifuid, amp):
    return (int(specid), int(ifuslot), int(ifuid), str(amp).upper())


def _parse_amp_label(value):
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    parts = str(value).strip().split("/")
    if len(parts) != 4:
        raise ValueError("invalid physical-amplifier label %r" % value)
    return _physical_key(parts[0], parts[1], parts[2], parts[3])


def _load_frozen_response(path):
    path = Path(path).expanduser().resolve()
    state, response = frozen_parent._load_state(path)
    composition = frozen_parent._verify_response_composition(state)
    if not composition.get("composition_matches_stored_cumulative", False):
        raise RuntimeError(
            "pass-1 plus refinement response does not reproduce cumulative R_total")
    if not response:
        raise RuntimeError("frozen cumulative response map is empty")
    return state, response, composition, path


def _load_identity_map(cache_path, h5files):
    cache_path = Path(cache_path).expanduser().resolve()
    manifest, manifest_path = frozen_parent._load_band_manifest(cache_path)
    mapping = frozen_parent._load_cache_identity_map(cache_path, manifest, h5files)
    return mapping, manifest, manifest_path, cache_path


def _load_alpha_fits(path):
    """Read directly measured blank-rich alpha_joint and its exposure-specific f0."""
    path = Path(path).expanduser().resolve()
    required = {
        "H5", "exposure", "SPECID", "IFUSLOT", "IFUID", "IFU_CODE",
        "AMP", "alpha_joint", "f0",
    }
    result = {}
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError("alpha-fit table lacks columns: %s" % sorted(missing))
        for row in reader:
            alpha = float(row["alpha_joint"])
            f0 = float(row["f0"])
            if not (np.isfinite(alpha) and np.isfinite(f0)):
                continue
            key = (
                Path(row["H5"]).name,
                int(row["exposure"]),
                int(row["SPECID"]),
                int(row["IFUSLOT"]),
                int(row["IFUID"]),
                str(row["AMP"]).upper(),
            )
            if key in result:
                raise ValueError("duplicate direct alpha key: %s" % (key,))
            result[key] = {
                "alpha": alpha,
                "f0": f0,
                "IFU_CODE": int(row["IFU_CODE"]),
                "N_blank_central": int(row.get("N_blank_central", 0) or 0),
            }
    if not result:
        raise ValueError("direct alpha-fit table contains no finite alpha_joint rows")
    return result, path


def _load_alpha_history(path):
    """Read the robust same-physical-amplifier alpha fallback."""
    path = Path(path).expanduser().resolve()
    required = {
        "SPECID", "IFUSLOT", "IFUID", "IFU_CODE", "AMP",
        "historical_alpha_robust_location",
    }
    result = {}
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError("historical alpha table lacks columns: %s" % sorted(missing))
        for row in reader:
            alpha = float(row["historical_alpha_robust_location"])
            if not np.isfinite(alpha):
                continue
            key = _physical_key(
                row["SPECID"], row["IFUSLOT"], row["IFUID"], row["AMP"])
            if key in result:
                raise ValueError("duplicate historical alpha key: %s" % (key,))
            result[key] = {
                "alpha": alpha,
                "IFU_CODE": int(row["IFU_CODE"]),
                "N_H5": int(row.get("N_H5", 0) or 0),
                "N_exposure_rows": int(row.get("N_exposure_rows", 0) or 0),
            }
    if not result:
        raise ValueError("historical alpha table contains no finite robust locations")
    return result, path


def _pick_npz_key(archive, names):
    for name in names:
        if name in archive.files:
            return name
    return None


def _load_amp_means(path):
    """Read historical robust full-resolution residual spectra by physical amp.

    The current PCA diagnostic writes robust_mean_* arrays for every physical
    amplifier with at least one retained residual spectrum.  A compatibility
    path for the earlier PCA-only product is retained so Luna can diagnose old
    products explicitly, but full-model preflight will still fail if a required
    frozen-M channel lacks a mean spectrum.
    """
    path = Path(path).expanduser().resolve()
    with np.load(path, allow_pickle=False) as archive:
        if "wavelength" not in archive.files:
            raise ValueError("residual-model NPZ has no wavelength array")
        wave = np.asarray(archive["wavelength"], dtype=float)
        if wave.shape != DEF_WAVE.shape or not np.allclose(
                wave, DEF_WAVE, rtol=0.0, atol=1.0e-6):
            raise ValueError("residual-model wavelength grid differs from DEF_WAVE")

        if "robust_mean_amp_label" in archive.files and "robust_mean" in archive.files:
            label_key = "robust_mean_amp_label"
            mean_key = "robust_mean"
            ifu_key = "robust_mean_IFU_CODE"
            n_key = "robust_mean_N_usable_exposures"
            product_scope = "all_retained_amp_means"
        else:
            label_key = _pick_npz_key(archive, ("amp_label", "labels"))
            mean_key = _pick_npz_key(
                archive, ("mean", "mean_residual", "robust_mean", "amp_mean"))
            ifu_key = _pick_npz_key(archive, ("IFU_CODE", "ifu_code"))
            n_key = _pick_npz_key(
                archive, ("N_usable_exposures", "n_exposures"))
            product_scope = "legacy_pca_support_only"
        if label_key is None or mean_key is None:
            raise ValueError(
                "residual-model NPZ lacks robust_mean_amp_label/robust_mean arrays; "
                "rerun the cube-ready PCA diagnostic")

        labels = np.asarray(archive[label_key])
        means = np.asarray(archive[mean_key], dtype=float)
        if means.ndim != 2 or means.shape != (labels.size, DEF_WAVE.size):
            raise ValueError("residual mean array has unexpected shape %s" % (means.shape,))
        ifu_codes = (
            np.asarray(archive[ifu_key], dtype=int)
            if ifu_key is not None else np.full(labels.size, -1, dtype=int))
        n_exposures = (
            np.asarray(archive[n_key], dtype=int)
            if n_key is not None else np.full(labels.size, -1, dtype=int))
        science_mask = (
            np.asarray(archive["science_mask"], dtype=bool)
            if "science_mask" in archive.files
            else np.isfinite(means).any(axis=0))

    result = {}
    for index, label in enumerate(labels):
        key = _parse_amp_label(label)
        if key in result:
            raise ValueError("duplicate residual-mean amp key: %s" % (key,))
        result[key] = {
            "mean": np.asarray(means[index], dtype=float),
            "IFU_CODE": int(ifu_codes[index]),
            "N_usable_exposures": int(n_exposures[index]),
        }
    if not result:
        raise ValueError("residual-model NPZ contains no amplifier means")
    return result, science_mask, product_scope, path


def _load_fq(path):
    path = Path(path).expanduser().resolve()
    fq = np.asarray(validated_m101.load_fq(path), dtype=float)
    if fq.shape != (N_FIBER_AMP,) or not np.all(np.isfinite(fq)):
        raise ValueError("f(q) template must contain finite q=0..%d" % (N_FIBER_AMP - 1))
    central = fq[Q_MIN:Q_MAX + 1]
    try:
        f0 = float(biweight(central))
    except Exception:
        f0 = float(np.nanmedian(central))
    if not np.isfinite(f0):
        f0 = float(np.nanmedian(central))
    if not np.isfinite(f0):
        raise ValueError("could not define fixed central-q f0")
    return fq, f0, path


# -----------------------------------------------------------------------------
# Input identity/model preflight
# -----------------------------------------------------------------------------


def _survey_by_exposure(h5):
    survey = {}
    for row in h5.root.Survey:
        exposure = int(row["exp"])
        if exposure in survey:
            raise ValueError("Survey has duplicate exposure %d" % exposure)
        survey[exposure] = {name: row[name] for name in h5.root.Survey.colnames}
    if set(survey) != set(range(1, N_EXPOSURES + 1)):
        raise ValueError("Survey must contain exposures 1..%d" % N_EXPOSURES)
    return survey


def _group_q(group):
    indices = np.asarray(group["indices"], dtype=int)
    if indices.size != N_FIBER_AMP:
        raise ValueError(
            "expected %d fibers per amplifier group, found %d for %s/%s/%s/%s" %
            (N_FIBER_AMP, indices.size, group["specid"], group["ifuslot"],
             group["ifuid"], group["amp"]))
    j = np.arange(N_FIBER_AMP, dtype=int)
    return j if group["amp"] in ("LL", "RU") else N_FIBER_AMP - 1 - j


def _preflight_model(h5files, identity_map, response, alpha_direct,
                     alpha_history, amp_means):
    rows = []
    counts = {
        "groups_total": 0,
        "groups_with_frozen_M": 0,
        "groups_without_frozen_M_masked": 0,
        "groups_with_direct_alpha": 0,
        "groups_with_historical_alpha": 0,
        "groups_missing_alpha_given_M": 0,
        "groups_missing_mean_given_M": 0,
    }
    for h5file in h5files:
        with tables.open_file(h5file, mode="r") as h5:
            if "/Info" not in h5 or "/Fibers" not in h5 or "/Survey" not in h5:
                raise ValueError("input is not a VIRUS H5 with Info/Fibers/Survey: %s" % h5file)
            groups, _ = validated_m101.build_groups(h5.root.Info)
            for group in groups:
                counts["groups_total"] += 1
                physical3 = (int(group["specid"]), int(group["ifuslot"]), int(group["ifuid"]))
                amp = str(group["amp"])
                physical_amp = physical3 + (amp,)
                map_key = (Path(h5file).name, physical3)
                if map_key not in identity_map:
                    raise ValueError("band cache has no IFU_CODE for %s %s" % (h5file.name, physical3))
                ifu_code = int(identity_map[map_key])
                m_key = (Path(h5file).name, ifu_code, amp)
                has_m = m_key in response and np.isfinite(response.get(m_key, np.nan))
                direct_alpha_key = (
                    Path(h5file).name, int(group["exposure"]), physical_amp[0],
                    physical_amp[1], physical_amp[2], physical_amp[3])
                has_direct_alpha = direct_alpha_key in alpha_direct
                has_historical_alpha = physical_amp in alpha_history
                has_alpha = has_direct_alpha or has_historical_alpha
                has_mean = physical_amp in amp_means
                status = "usable"
                if not has_m:
                    counts["groups_without_frozen_M_masked"] += 1
                    status = "masked_missing_frozen_M"
                else:
                    counts["groups_with_frozen_M"] += 1
                    if has_direct_alpha:
                        counts["groups_with_direct_alpha"] += 1
                        if alpha_direct[direct_alpha_key]["IFU_CODE"] != ifu_code:
                            raise ValueError(
                                "direct alpha IFU_CODE mismatch for %s: %d vs cache %d" %
                                (direct_alpha_key, alpha_direct[direct_alpha_key]["IFU_CODE"],
                                 ifu_code))
                    elif has_historical_alpha:
                        counts["groups_with_historical_alpha"] += 1
                    if not has_alpha:
                        counts["groups_missing_alpha_given_M"] += 1
                        status = "ERROR_missing_alpha"
                    if not has_mean:
                        counts["groups_missing_mean_given_M"] += 1
                        status = "ERROR_missing_mean"
                    if has_historical_alpha and alpha_history[physical_amp]["IFU_CODE"] not in (-1, ifu_code):
                        raise ValueError(
                            "historical alpha IFU_CODE mismatch for %s: %d vs cache %d" %
                            (physical_amp, alpha_history[physical_amp]["IFU_CODE"], ifu_code))
                    if has_mean and amp_means[physical_amp]["IFU_CODE"] not in (-1, ifu_code):
                        raise ValueError(
                            "residual-mean IFU_CODE mismatch for %s: %d vs cache %d" %
                            (physical_amp, amp_means[physical_amp]["IFU_CODE"], ifu_code))
                rows.append({
                    "H5": Path(h5file).name,
                    "exposure": int(group["exposure"]),
                    "SPECID": physical3[0],
                    "IFUSLOT": physical3[1],
                    "IFUID": physical3[2],
                    "IFU_CODE": ifu_code,
                    "AMP": amp,
                    "has_frozen_M": bool(has_m),
                    "has_direct_alpha": bool(has_direct_alpha),
                    "has_historical_alpha": bool(has_historical_alpha),
                    "alpha_source": (
                        "direct_blank_rich" if has_direct_alpha
                        else "historical_same_amp" if has_historical_alpha
                        else "missing"),
                    "has_robust_mean": bool(has_mean),
                    "status": status,
                })
    if counts["groups_missing_alpha_given_M"]:
        raise RuntimeError(
            "%d frozen-M groups lack direct or historical alpha; do not silently use alpha=0" %
            counts["groups_missing_alpha_given_M"])
    if counts["groups_missing_mean_given_M"]:
        raise RuntimeError(
            "%d frozen-M groups lack robust mean spectra; rerun the cube-ready PCA "
            "diagnostic so robust_mean arrays cover all retained physical amplifiers" %
            counts["groups_missing_mean_given_M"])
    return rows, counts


# -----------------------------------------------------------------------------
# Current-model spectral calibration
# -----------------------------------------------------------------------------


def _alpha_for_group(alpha_direct, alpha_history, h5_name, exposure,
                     physical_amp, ifu_code, global_f0):
    direct_key = (
        h5_name, int(exposure), physical_amp[0], physical_amp[1],
        physical_amp[2], physical_amp[3],
    )
    if direct_key in alpha_direct:
        row = alpha_direct[direct_key]
        if row["IFU_CODE"] != ifu_code:
            raise ValueError(
                "direct alpha IFU_CODE mismatch for %s: %d vs %d" %
                (direct_key, row["IFU_CODE"], ifu_code))
        return float(row["alpha"]), float(row["f0"]), "direct_blank_rich"
    row = alpha_history[physical_amp]
    return float(row["alpha"]), float(global_f0), "historical_same_amp"


def _subtract_exposure_sky(corrected_total, labels, q_rows, blank_mask,
                           hardware_bad, h5_name, min_sky_fibers):
    """Subtract one robust corrected-space sky per exposure using central-q blanks."""
    spectra = np.full_like(corrected_total, np.nan, dtype=float)
    records = []
    n_wave = corrected_total.shape[1]
    for exposure in range(1, N_EXPOSURES + 1):
        exposure_mask = np.asarray(labels, dtype=int) == exposure
        finite_fraction = np.mean(np.isfinite(corrected_total), axis=1)
        sufficient = finite_fraction >= M101_SKY_MIN_FINITE_FRACTION
        sky_mask = (
            exposure_mask
            & np.asarray(blank_mask, dtype=bool)
            & ~np.asarray(hardware_bad, dtype=bool)
            & (np.asarray(q_rows, dtype=int) >= Q_MIN)
            & (np.asarray(q_rows, dtype=int) <= Q_MAX)
            & sufficient
        )
        selected = int(np.sum(sky_mask))
        if selected < min_sky_fibers:
            raise RuntimeError(
                "%s exposure %d has only %d corrected central-q blank fibers; "
                "minimum is %d" % (h5_name, exposure, selected, min_sky_fibers))

        values = np.asarray(corrected_total[sky_mask], dtype=float)
        finite = np.isfinite(values)
        n_sky = np.sum(finite, axis=0).astype(int)
        sky = frozen_parent._safe_location(np.where(finite, values, np.nan), axis=0)
        sky[n_sky < min_sky_fibers] = np.nan
        finite_sky = np.isfinite(sky)
        finite_sky_fraction = float(np.mean(finite_sky))
        if finite_sky_fraction < M101_SKY_MIN_FINITE_FRACTION:
            raise RuntimeError(
                "%s exposure %d corrected sky is only %.3f finite across wavelength" %
                (h5_name, exposure, finite_sky_fraction))

        indices = np.flatnonzero(exposure_mask)
        spectra[indices] = corrected_total[indices] - sky[None, :]
        spectra[np.ix_(indices, ~finite_sky)] = np.nan

        blank_after = spectra[sky_mask]
        blank_location = frozen_parent._safe_location(blank_after, axis=0)
        blank_location_finite = blank_location[np.isfinite(blank_location)]
        records.append({
            "H5": h5_name,
            "exposure": exposure,
            "selected_blank_central_q": selected,
            "finite_sky_fraction": finite_sky_fraction,
            "min_sky_support_per_wavelength": int(np.min(n_sky[finite_sky]))
            if np.any(finite_sky) else 0,
            "median_sky_support_per_wavelength": float(np.median(n_sky[finite_sky]))
            if np.any(finite_sky) else np.nan,
            "median_sky": float(np.nanmedian(sky)),
            "max_abs_post_sky_blank_location": float(
                np.nanmax(np.abs(blank_location_finite)))
            if blank_location_finite.size else np.nan,
            "n_wavelengths": int(n_wave),
        })
    return spectra, records


def _calibrate_h5_current(h5file, identity_map, response, alpha_direct,
                          alpha_history, amp_means, fq_template, global_f0,
                          blank_mask, min_sky_fibers):
    """Apply frozen M + alpha(q) + historical amplifier mean, then current sky."""
    h5_name = Path(h5file).name
    with tables.open_file(h5file, mode="r") as h5:
        info, fibers = h5.root.Info, h5.root.Fibers
        groups, labels = validated_m101.build_groups(info)
        if "skyspectrum" not in fibers.colnames:
            raise ValueError("%s Fibers lacks skyspectrum" % h5file)
        source = np.asarray(fibers.cols.spectrum[:], dtype=float)
        error_source = np.asarray(fibers.cols.error[:], dtype=float)
        sky_native = np.asarray(fibers.cols.skyspectrum[:], dtype=float)
        ra = np.asarray(info.cols.ra[:], dtype=float)
        dec = np.asarray(info.cols.dec[:], dtype=float)
        if source.shape != error_source.shape or source.shape != sky_native.shape:
            raise ValueError("%s spectrum/error/skyspectrum shapes differ" % h5file)
        if source.shape[1] != DEF_WAVE.size:
            raise ValueError("%s wavelength dimension differs from DEF_WAVE" % h5file)

        surveys = _survey_by_exposure(h5)
        ifuslot_rows = np.asarray(info.cols.ifuslot[:])
        amp_rows = np.asarray([_text(value) for value in info.cols.amp[:]])
        hardware_bad = validated_m101.masked_rows(h5file, ifuslot_rows, amp_rows)

        corrected_total = np.full(source.shape, np.nan, dtype=float)
        errors = np.full(error_source.shape, np.nan, dtype=float)
        q_rows = np.full(source.shape[0], -1, dtype=np.int16)
        alpha_source_counts = {"direct_blank_rich": 0, "historical_same_amp": 0}
        supported_groups = 0
        masked_missing_m_groups = 0
        logm_used = []
        alpha_used = []
        mean_support_used = []

        for exposure in range(1, N_EXPOSURES + 1):
            survey = surveys[exposure]
            offset = float(survey["offset"])
            if not np.isfinite(offset) or offset == 0.0:
                raise ValueError(
                    "%s exposure %d has invalid Survey.offset" % (h5file, exposure))
            K_work = np.asarray(validated_m101.raw_work_basis(survey), dtype=float)
            if K_work.shape != DEF_WAVE.shape or not np.all(np.isfinite(K_work)):
                raise ValueError(
                    "%s exposure %d raw_work_basis is invalid" % (h5file, exposure))
            total = source / offset + sky_native
            error_work = error_source / abs(offset)

            for group in groups:
                if int(group["exposure"]) != exposure:
                    continue
                indices = np.asarray(group["indices"], dtype=int)
                q = _group_q(group)
                q_rows[indices] = q
                physical3 = (
                    int(group["specid"]), int(group["ifuslot"]), int(group["ifuid"]))
                amp = str(group["amp"])
                physical_amp = physical3 + (amp,)
                ifu_code = int(identity_map[(h5_name, physical3)])
                m_key = (h5_name, ifu_code, amp)
                if m_key not in response or not np.isfinite(response[m_key]):
                    masked_missing_m_groups += 1
                    continue

                if physical_amp not in alpha_history:
                    raise RuntimeError("frozen-M channel lacks alpha fallback: %s" % (physical_amp,))
                if physical_amp not in amp_means:
                    raise RuntimeError("frozen-M channel lacks robust mean spectrum: %s" % (physical_amp,))

                alpha, f0, alpha_source = _alpha_for_group(
                    alpha_direct, alpha_history, h5_name, exposure,
                    physical_amp, ifu_code, global_f0)
                mu = np.asarray(amp_means[physical_amp]["mean"], dtype=float)
                # mu_a(lambda) was only estimated where the PCA diagnostic had
                # scientific support.  Outside that support, apply no spectral
                # mean correction rather than propagate NaN into the cube.
                mu_apply = np.where(np.isfinite(mu), mu, 0.0)
                multiplier = float(np.exp(float(response[m_key])))
                additive_q = (
                    alpha * (fq_template[q] - f0)[:, None] * K_work[None, :])

                corrected_total[indices] = (
                    total[indices] - additive_q - mu_apply[None, :]) / multiplier
                errors[indices] = error_work[indices] / multiplier

                supported_groups += 1
                alpha_source_counts[alpha_source] += 1
                logm_used.append(float(response[m_key]))
                alpha_used.append(float(alpha))
                mean_support_used.append(int(amp_means[physical_amp]["N_usable_exposures"]))

        corrected_total[hardware_bad] = np.nan
        errors[hardware_bad] = np.nan

    spectra, sky_records = _subtract_exposure_sky(
        corrected_total, labels, q_rows, blank_mask, hardware_bad,
        h5_name, min_sky_fibers)

    calibration_summary = {
        "H5": h5_name,
        "supported_groups": int(supported_groups),
        "masked_missing_frozen_M_groups": int(masked_missing_m_groups),
        "alpha_direct_groups": int(alpha_source_counts["direct_blank_rich"]),
        "alpha_historical_groups": int(alpha_source_counts["historical_same_amp"]),
        "logM": _distribution(logm_used),
        "alpha": _distribution(alpha_used),
        "robust_mean_training_exposures": _distribution(mean_support_used),
    }
    return spectra, errors, ra, dec, labels, surveys, calibration_summary, sky_records


# -----------------------------------------------------------------------------
# Cube geometry and reconstruction (kept close to the prior tested script)
# -----------------------------------------------------------------------------


def _parse_image_geometry(value, pixel_scale):
    try:
        ra, dec, size_arcmin = [float(part.strip()) for part in value.split(",")]
    except (AttributeError, ValueError):
        raise ValueError("image center/size must be RA,Dec,size_arcmin")
    if not np.all(np.isfinite([ra, dec, size_arcmin])) or size_arcmin <= 0:
        raise ValueError("image center/size must contain finite positive size")
    size_arcsec = int(size_arcmin * 60.0 / pixel_scale / 2.0) * 2 * pixel_scale
    n = int(size_arcsec / pixel_scale / 2.0) * 2 + 1
    xg = np.arange(n, dtype=float) + 1.0
    yg = np.arange(n, dtype=float) + 1.0
    xgrid, ygrid = np.meshgrid(xg, yg)
    astrometry = Astrometry(ra, dec, 0.0, (n + 1) / 2.0, (n + 1) / 2.0)
    tp = astrometry.setup_TP(
        ra, dec, 0.0, x0=(n + 1) / 2.0, y0=(n + 1) / 2.0)
    return ra, dec, size_arcsec, n, xg, yg, xgrid, ygrid, tp


@njit(nogil=True, cache=True)
def _gaussian_splat_shot_xy(indices, xpos, ypos, fluxes, errors,
                            x_origin, y_origin, nx, ny, sigma, radius,
                            support_radius, area, flux_sum, weight_sum,
                            variance_numerator, error_weight_sum, support_map):
    sigma_sq = sigma * sigma
    support_radius_sq = (2.0 * sigma) * (2.0 * sigma)
    width = 2 * radius + 1
    gx = np.empty(width, dtype=np.float32)
    gy = np.empty(width, dtype=np.float32)
    for p in range(indices.shape[0]):
        j = indices[p]
        flux = fluxes[j]
        xi = xpos[j]
        yi = ypos[j]
        if not np.isfinite(flux) or not np.isfinite(xi) or not np.isfinite(yi):
            continue
        ix_center = int(np.floor(xi - x_origin))
        iy_center = int(np.floor(yi - y_origin))
        for ox in range(-radius, radius + 1):
            gx[ox + radius] = np.exp(
                -0.5 * ((x_origin + ix_center + ox) - xi) ** 2 / sigma_sq)
        for oy in range(-radius, radius + 1):
            gy[oy + radius] = np.exp(
                -0.5 * ((y_origin + iy_center + oy) - yi) ** 2 / sigma_sq)
        for ox in range(-radius, radius + 1):
            px = ix_center + ox
            if px < 0 or px >= nx:
                continue
            for oy in range(-radius, radius + 1):
                py = iy_center + oy
                if py < 0 or py >= ny:
                    continue
                weight = gx[ox + radius] * gy[oy + radius]
                flux_sum[py, px] += weight * flux / area
                weight_sum[py, px] += weight
                err = errors[j]
                if np.isfinite(err) and err > 0.0:
                    variance_numerator[py, px] += weight * weight * (err / area) ** 2
                    error_weight_sum[py, px] += weight
        for ox in range(-support_radius, support_radius + 1):
            px = ix_center + ox
            if px < 0 or px >= nx:
                continue
            dx = (x_origin + px) - xi
            for oy in range(-support_radius, support_radius + 1):
                py = iy_center + oy
                if py < 0 or py >= ny:
                    continue
                dy = (y_origin + py) - yi
                if dx * dx + dy * dy <= support_radius_sq:
                    support_map[py, px] = 1


def _compute_final_variance(shot_images, shot_variances, ncontrib):
    ny, nx = ncontrib.shape
    valid_sci = ncontrib >= 2
    supported = np.isfinite(shot_images)
    n = supported.sum(axis=0)
    finite_variance = np.isfinite(shot_variances)
    complete = np.all(~supported | finite_variance, axis=0)
    positive_or_zero = np.all(~supported | (shot_variances >= 0.0), axis=0)
    formal = np.full((ny, nx), np.nan, dtype=np.float64)
    two_shot = (n == 2) & complete
    if np.any(two_shot):
        formal[two_shot] = (
            np.sum(np.where(supported, shot_variances, 0.0), axis=0)[two_shot] / 4.0)
    three_or_more = (n >= 3) & complete & positive_or_zero
    if np.any(three_or_more):
        sigmas = np.sqrt(np.where(supported, shot_variances, 1.0))
        inverse_sigma = np.zeros_like(sigmas, dtype=np.float64)
        np.divide(1.0, sigmas, out=inverse_sigma, where=supported)
        inverse_sigma_sum = np.sum(inverse_sigma, axis=0)
        positive_sigma = np.all(~supported | (sigmas > 0.0), axis=0)
        formal_valid = three_or_more & positive_sigma
        formal[formal_valid] = (
            n[formal_valid] * np.pi / (2.0 * inverse_sigma_sum[formal_valid] ** 2))
    empirical = np.full((ny, nx), np.nan, dtype=np.float64)
    five_or_more = n >= 5
    if np.any(five_or_more):
        center = np.nanmedian(shot_images, axis=0)
        mad = np.nanmedian(np.abs(shot_images - center[None, :, :]), axis=0)
        empirical[five_or_more] = (
            1.2533 * 1.4826 * mad / np.sqrt(n))[five_or_more] ** 2
    formal_available = np.isfinite(formal)
    empirical_available = np.isfinite(empirical)
    both = formal_available & empirical_available
    variance = np.full((ny, nx), np.nan, dtype=np.float64)
    variance[formal_available & ~empirical_available] = formal[
        formal_available & ~empirical_available]
    variance[empirical_available & ~formal_available] = empirical[
        empirical_available & ~formal_available]
    variance[both] = np.maximum(formal[both], empirical[both])
    varianceimage = variance.astype(np.float32)
    dq = np.zeros((ny, nx), dtype=np.uint16)
    dq[~valid_sci] |= DQ_INSUFFICIENT_SUPPORT
    formal_used = valid_sci & formal_available & (
        ~empirical_available | (formal >= empirical))
    empirical_used = valid_sci & both & (empirical > formal)
    empirical_only = valid_sci & ~formal_available & empirical_available
    variance_incomplete = valid_sci & ~formal_available & ~empirical_available
    dq[formal_used] |= DQ_FORMAL_VAR_USED
    dq[empirical_used] |= DQ_EMPIRICAL_VAR_USED
    dq[empirical_only] |= DQ_VAR_EMPIRICAL_ONLY
    dq[variance_incomplete] |= DQ_VAR_INCOMPLETE
    ratio = np.full((ny, nx), np.nan, dtype=np.float64)
    ratio[both] = np.sqrt(empirical[both] / formal[both])
    return varianceimage, dq, {
        "valid_sci_voxels": int(np.sum(valid_sci)),
        "finite_variance_voxels": int(
            np.sum(valid_sci & np.isfinite(varianceimage))),
        "median_ncontrib": float(np.median(ncontrib[valid_sci]))
        if np.any(valid_sci) else np.nan,
        "variance_unavailable_voxels": int(
            np.sum(valid_sci & ~np.isfinite(varianceimage))),
        "insufficient_support_voxels": int(np.sum(~valid_sci)),
        "median_ratio": float(np.nanmedian(ratio))
        if np.any(np.isfinite(ratio)) else np.nan,
    }


def make_image_gaussian(Pos, y, ye, xg, yg, xgrid, ygrid, sigma, cnt_array):
    """Current VIRUS cube Gaussian-splat reconstruction and variance path."""
    nshots = len(cnt_array)
    ny, nx = xgrid.shape
    radius = 3
    support_radius = int(np.ceil(2.0 * sigma))
    area = np.pi * FIBER_RADIUS_ARCSEC ** 2
    xpos, ypos = Pos[:, 0], Pos[:, 1]
    shot_images = np.full((nshots, ny, nx), np.nan, dtype=np.float32)
    shot_variances = np.full((nshots, ny, nx), np.nan, dtype=np.float32)
    coverage = np.zeros((ny, nx), dtype=np.float32)
    ncontrib = np.zeros((ny, nx), dtype=np.uint8)
    for k, indices in enumerate(cnt_array):
        indices = np.asarray(indices, dtype=np.int64)
        flux_sum = np.zeros((ny, nx), dtype=np.float32)
        weight_sum = np.zeros((ny, nx), dtype=np.float32)
        variance_numerator = np.zeros((ny, nx), dtype=np.float32)
        error_weight_sum = np.zeros((ny, nx), dtype=np.float32)
        support_map = np.zeros((ny, nx), dtype=np.uint8)
        _gaussian_splat_shot_xy(
            indices, xpos, ypos, y, ye, float(xg[0]), float(yg[0]),
            nx, ny, float(sigma), radius, support_radius, float(area),
            flux_sum, weight_sum, variance_numerator,
            error_weight_sum, support_map)
        supported = (support_map != 0) & (weight_sum > 0.0)
        if np.any(supported):
            shot_images[k, supported] = flux_sum[supported] / weight_sum[supported]
            complete = supported & np.isclose(
                error_weight_sum, weight_sum, rtol=1.e-5, atol=1.e-7)
            shot_variances[k, complete] = (
                variance_numerator[complete] / (weight_sum[complete] ** 2))
            ncontrib[supported] += 1
            coverage += np.minimum(weight_sum, 1.0) * supported
    with np.errstate(all="ignore"):
        image = np.nanmedian(shot_images, axis=0).astype(np.float32)
    varianceimage, dq, variance_stats = _compute_final_variance(
        shot_images, shot_variances, ncontrib)
    coverage = np.clip(coverage / float(max(1, nshots)), 0.0, 1.0)
    image[ncontrib < 2] = 0.0
    image[~np.isfinite(image)] = 0.0
    varianceimage[ncontrib < 2] = 0.0
    return image, varianceimage, coverage, ncontrib, dq, variance_stats


# -----------------------------------------------------------------------------
# Provenance/output
# -----------------------------------------------------------------------------


def _synthetic_calibration_identity():
    wave = np.linspace(3470.0, 5540.0, 17)
    truth = np.linspace(0.2, 0.8, wave.size)
    sky = np.linspace(1.1, 1.4, wave.size)
    K = np.linspace(0.8, 1.2, wave.size)
    mu = 0.03 + 0.01 * np.sin(np.linspace(0.0, 2.0 * np.pi, wave.size))
    M = np.exp(0.12)
    alpha = 0.41
    fq_minus_f0 = 0.37
    D = M * (truth + sky) + alpha * fq_minus_f0 * K + mu
    corrected_total = (D - alpha * fq_minus_f0 * K - mu) / M
    recovered = corrected_total - sky
    if not np.allclose(recovered, truth, rtol=0.0, atol=1.e-13):
        raise AssertionError("synthetic current-model calibration identity failed")
    return {
        "status": "PASS",
        "max_abs_source_error": float(np.max(np.abs(recovered - truth))),
        "equation": "O=(D-alpha*(f-f0)*K-mu)/M-S",
    }


def _write_cube_products(surname, cube, variancecube, dqcube, weightcube,
                         ncontribcube, tp, n, pixel_scale, model_provenance,
                         summary):
    header = tp.to_header()
    header["WCSAXES"] = 3
    header["CD1_1"] = -pixel_scale / 3600.0
    header["CD1_2"] = 0.0
    header["CD2_1"] = 0.0
    header["CD2_2"] = pixel_scale / 3600.0
    header.pop("CDELT1", None)
    header.pop("CDELT2", None)
    header["CRPIX1"] = (n + 1) / 2.0
    header["CRPIX2"] = (n + 1) / 2.0
    header["CDELT3"] = float(DEF_WAVE[1] - DEF_WAVE[0])
    header["CRPIX3"] = 1.0
    header["CRVAL3"] = float(DEF_WAVE[0])
    header["CTYPE1"] = "RA---TAN"
    header["CTYPE2"] = "DEC--TAN"
    header["CTYPE3"] = "WAVE"
    header["CUNIT1"] = "deg"
    header["CUNIT2"] = "deg"
    header["CUNIT3"] = "Angstrom"
    header["SPECSYS"] = "TOPOCENT"
    header["CALMOD"] = "M101CUR1"
    header["NMSUP"] = int(summary["preflight"]["groups_with_frozen_M"])
    header["NMMISS"] = int(summary["preflight"]["groups_without_frozen_M_masked"])
    header["FQREF"] = float(summary["model"]["historical_f0"])

    for text in (
            "M101 current model: M=exp(hierarchy.cumulative_final.R_total_h_i_a).",
            "M is one wavelength-independent scalar per H5/IFU_CODE/amplifier.",
            "q additive: alpha*(f_q(q)-f0)*K(lambda).",
            "alpha uses direct blank-rich alpha_joint when available, otherwise same-amp historical robust alpha.",
            "spectral additive: historical physical-amplifier robust mean residual mu_a(lambda).",
            "3-PC exposure coefficients are NOT applied in this first cube model.",
            "D=Fibers.spectrum/Survey.offset+Fibers.skyspectrum.",
            "Corrected total T=(D-alpha*(f-f0)*K-mu)/M.",
            "Exposure sky S is a robust spectrum from classified hardware-valid blank q=40..75 fibers.",
            "Cube object spectrum is O=T-S.",
            "mu is subtracted only where its saved robust-mean spectrum is finite; outside that support mu=0.",
            "Frozen-M unsupported channels are masked; sibling channels remain usable.",
            "Additive-model uncertainty and exposure-sky uncertainty are not yet propagated into formal fiber errors.",
            "Gaussian subpixel reconstruction; NCONTRIB >= 2 required for valid SCI."):
        header.add_history(text)
    header.add_history("Frozen-M state: %s" % model_provenance["state"]["path"])
    header.add_history("Alpha fits: %s" % model_provenance["alpha_fits"]["path"])
    header.add_history("Alpha history: %s" % model_provenance["alpha_history"]["path"])
    header.add_history("Residual means: %s" % model_provenance["amp_means"]["path"])

    fits.PrimaryHDU(
        np.asarray(cube, dtype=np.float32), header=header).writeto(
            "%s_cube.fits" % surname, overwrite=True)
    variance_header = header.copy()
    variance_header["BUNIT"] = "SCI units squared"
    variance_header["EXTNAME"] = "VARIANCE"
    fits.PrimaryHDU(
        np.asarray(variancecube, dtype=np.float32), header=variance_header).writeto(
            "%s_variance_cube.fits" % surname, overwrite=True)
    error_header = header.copy()
    error_header["BUNIT"] = "SCI units"
    error_header["EXTNAME"] = "ERROR"
    fits.PrimaryHDU(
        np.sqrt(np.asarray(variancecube, dtype=np.float32)),
        header=error_header).writeto(
            "%s_errorcube.fits" % surname, overwrite=True)
    dq_header = header.copy()
    dq_header["BUNIT"] = "bit mask"
    dq_header["EXTNAME"] = "DQ"
    dq_header["DQBIT0"] = "NCONTRIB < 2"
    dq_header["DQBIT1"] = "SCI valid, VAR unavailable"
    dq_header["DQBIT2"] = "empirical VAR adopted over formal"
    dq_header["DQBIT3"] = "formal VAR adopted"
    dq_header["DQBIT4"] = "empirical-only VAR adopted"
    fits.PrimaryHDU(
        np.asarray(dqcube, dtype=np.uint16), header=dq_header).writeto(
            "%s_dq_cube.fits" % surname, overwrite=True)
    weight_header = header.copy()
    weight_header["BUNIT"] = "Gaussian support"
    weight_header["EXTNAME"] = "COVERAGE"
    fits.PrimaryHDU(
        np.asarray(weightcube, dtype=np.float32), header=weight_header).writeto(
            "%s_weight_cube.fits" % surname, overwrite=True)
    ncontrib_header = header.copy()
    ncontrib_header["BUNIT"] = "count"
    ncontrib_header["EXTNAME"] = "NCONTRIB"
    ncontrib_header["COMMENT"] = "Independent shots with a valid fiber within 2 sigma."
    fits.PrimaryHDU(
        np.asarray(ncontribcube, dtype=np.uint8), header=ncontrib_header).writeto(
            "%s_ncontrib_cube.fits" % surname, overwrite=True)


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("h5files", help="input VIRUS H5 glob, e.g. '/data/*.h5'")
    parser.add_argument("surname", help="output filename prefix")
    parser.add_argument(
        "image_center_size", help="RA,Dec center and size in arcmin")
    parser.add_argument(
        "--state", default="m101_onoff_hierarchy/m101_onoff_state.json",
        help="current ON/OFF hierarchy state containing cumulative-final R_total")
    parser.add_argument(
        "--band-cache", default="m101_band_initializer/m101_band_cache.npz",
        help="band cache used only for frozen physical-IFU to IFU_CODE identity")
    parser.add_argument(
        "--alpha-fits", default="m101_alpha_template_bands/alpha_template_fits.csv",
        help="direct blank-rich alpha_joint table")
    parser.add_argument(
        "--alpha-summary",
        default="m101_alpha_same_amp_loo/alpha_physical_amp_summary.csv",
        help="historical same-physical-amplifier alpha fallback table")
    parser.add_argument(
        "--amp-residual-models",
        default="m101-amp-residual-pca-final/amp_residual_pca_models.npz",
        help="PCA diagnostic NPZ containing robust_mean arrays for physical amps")
    parser.add_argument(
        "--fq-template", required=True, help="fixed q,f template CSV")
    parser.add_argument(
        "--external-blank-fibers", required=True,
        help="validated external blank-fiber CSV or CSV.GZ")
    parser.add_argument(
        "--pixel-scale", type=float, default=1.0,
        help="output pixel scale in arcsec")
    parser.add_argument(
        "--wave-workers", type=int, default=1,
        help="wavelength reconstruction threads")
    parser.add_argument(
        "--min-sky-fibers", type=int, default=M101_SKY_MIN_FIBERS,
        help="minimum corrected central-q blank fibers per exposure/wavelength")
    parser.add_argument(
        "--preflight-only", action="store_true",
        help="validate all model products/identity coverage and stop before reading spectra")
    return parser.parse_args()


def main():
    args = _parse_args()
    if args.pixel_scale <= 0 or not np.isfinite(args.pixel_scale):
        raise SystemExit("--pixel-scale must be finite and positive")
    if args.wave_workers < 1:
        raise SystemExit("--wave-workers must be at least 1")
    if args.min_sky_fibers < 1:
        raise SystemExit("--min-sky-fibers must be positive")

    started = time.perf_counter()
    synthetic = _synthetic_calibration_identity()
    if (Q_MIN, Q_MAX) != (frozen_parent.Q_MIN, frozen_parent.Q_MAX):
        raise RuntimeError("cube and frozen-M diagnostic q gauges disagree")
    if DEF_WAVE.shape != frozen_parent.WAVE.shape or not np.allclose(
            DEF_WAVE, frozen_parent.WAVE, rtol=0.0, atol=0.0):
        raise RuntimeError("cube and frozen-M diagnostic wavelength grids disagree")

    h5files = sorted(Path(path).resolve() for path in glob.glob(args.h5files))
    if not h5files:
        raise ValueError("no H5 files matched: %s" % args.h5files)
    if any(path.name == "20200523_0000023.h5" for path in h5files):
        raise ValueError("20200523_0000023.h5 is explicitly excluded from the M101 sample")

    state, response, composition, state_path = _load_frozen_response(args.state)
    identity_map, cache_manifest, cache_manifest_path, cache_path = _load_identity_map(
        args.band_cache, h5files)
    alpha_direct, alpha_fits_path = _load_alpha_fits(args.alpha_fits)
    alpha_history, alpha_summary_path = _load_alpha_history(args.alpha_summary)
    amp_means, mean_science_mask, mean_scope, amp_means_path = _load_amp_means(
        args.amp_residual_models)
    fq_template, historical_f0, fq_path = _load_fq(args.fq_template)

    blank_masks, blank_provenance = frozen_parent.m101_blank_fibers.load(
        Path(args.external_blank_fibers).expanduser().resolve(), h5files)

    print("[cube] current-model provenance", flush=True)
    print("  frozen M state      : %s" % state_path, flush=True)
    print("  response field      : hierarchy.cumulative_final.R_total_h_i_a", flush=True)
    print("  band identity cache : %s" % cache_path, flush=True)
    print("  direct alpha fits   : %s (%d rows)" %
          (alpha_fits_path, len(alpha_direct)), flush=True)
    print("  alpha fallback      : %s (%d physical amps)" %
          (alpha_summary_path, len(alpha_history)), flush=True)
    print("  amplifier means     : %s (%d physical amps; %s)" %
          (amp_means_path, len(amp_means), mean_scope), flush=True)
    print("  f(q)                : %s; historical fallback f0=%.8g" %
          (fq_path, historical_f0), flush=True)
    print("  external blanks     : %s" %
          Path(args.external_blank_fibers).expanduser().resolve(), flush=True)
    print("  PCA coefficients    : NOT APPLIED", flush=True)

    coverage_rows, preflight = _preflight_model(
        h5files, identity_map, response, alpha_direct, alpha_history, amp_means)
    coverage_path = Path("%s_model_coverage.csv" % args.surname)
    _write_rows(coverage_path, coverage_rows)
    print("[cube] preflight: %s" % preflight, flush=True)

    if args.preflight_only:
        preflight_summary = {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "script": str(Path(__file__).resolve()),
            "input_h5_files": [str(path) for path in h5files],
            "response_composition_validation": composition,
            "preflight": preflight,
            "coverage_table": str(coverage_path.resolve()),
            "model_products": {
                "state": _file_identity(state_path),
                "band_cache": _file_identity(cache_path),
                "band_cache_manifest": _file_identity(cache_manifest_path),
                "alpha_fits": _file_identity(alpha_fits_path),
                "alpha_history": _file_identity(alpha_summary_path),
                "amp_means": _file_identity(amp_means_path),
                "fq_template": _file_identity(fq_path),
                "blank_file": _file_identity(args.external_blank_fibers),
            },
            "alpha": {
                "direct_rows": len(alpha_direct),
                "historical_physical_amps": len(alpha_history),
                "historical_f0": float(historical_f0),
            },
            "amp_means": {
                "physical_amps": len(amp_means),
                "product_scope": mean_scope,
                "finite_wavelength_samples": int(np.sum(mean_science_mask)),
            },
            "blank_loader": blank_provenance,
            "synthetic_validation": synthetic,
        }
        path = Path("%s_preflight.json" % args.surname)
        path.write_text(json.dumps(preflight_summary, indent=2, default=str))
        print("[cube] preflight-only PASS: %s" % path.resolve(), flush=True)
        return

    ra0, dec0, size_arcsec, n, xg, yg, xgrid, ygrid, tp = _parse_image_geometry(
        args.image_center_size, args.pixel_scale)
    print("[cube] image grid: %d x %d at %.6g arcsec/pixel" %
          (n, n, args.pixel_scale), flush=True)

    records = []
    calibration_rows = []
    sky_rows = []
    total_fibers = 0
    calibration_seconds = 0.0
    for h5file in h5files:
        one_started = time.perf_counter()
        spectra, errors, ra, dec, labels, surveys, calibration_summary, one_sky = (
            _calibrate_h5_current(
                h5file, identity_map, response, alpha_direct, alpha_history,
                amp_means, fq_template, historical_f0,
                blank_masks[h5file.name], args.min_sky_fibers))
        calibration_seconds += time.perf_counter() - one_started
        calibration_rows.append(calibration_summary)
        sky_rows.extend(one_sky)
        total_fibers += len(ra)
        records.append({
            "h5file": h5file,
            "spectra": spectra.astype(np.float32),
            "errors": errors.astype(np.float32),
            "ra": ra,
            "dec": dec,
            "labels": labels,
            "surveys": surveys,
        })
        print("[cube] calibrated %s in %.3f s; supported groups=%d; masked no-M=%d" %
              (h5file.name, time.perf_counter() - one_started,
               calibration_summary["supported_groups"],
               calibration_summary["masked_missing_frozen_M_groups"]), flush=True)

    sky_path = Path("%s_sky_summary.csv" % args.surname)
    _write_rows(sky_path, sky_rows)
    calibration_path = Path("%s_calibration_summary.csv" % args.surname)
    _write_rows(calibration_path, calibration_rows)

    offsets = []
    offset = 0
    for record in records:
        offsets.append(offset)
        offset += len(record["ra"])
    specarray = np.concatenate([record["spectra"] for record in records], axis=0)
    errarray = np.concatenate([record["errors"] for record in records], axis=0)
    raarray = np.empty((total_fibers, len(DEF_WAVE)), dtype=np.float32)
    decarray = np.empty_like(raarray)
    shot_indices = []
    shot_names = []

    adr_started = time.perf_counter()
    for record, start in zip(records, offsets):
        extractor = Extract(wave=DEF_WAVE)
        for exposure in range(1, N_EXPOSURES + 1):
            selected = record["labels"] == exposure
            survey = record["surveys"][exposure]
            astrometry = Astrometry(
                float(survey["ra"]), float(survey["dec"]),
                float(survey["pa"]), 0.0, 0.0)
            extractor.get_ADR_RAdec(astrometry)
            indices = np.flatnonzero(selected)
            raarray[start + indices] = (
                record["ra"][indices, None]
                - extractor.ADRra[None, :] / 3600.0
                / np.cos(np.deg2rad(float(survey["dec"]))))
            decarray[start + indices] = (
                record["dec"][indices, None] - extractor.ADRdec[None, :] / 3600.0)
            shot_indices.append(start + indices)
            shot_names.append("%s exposure %d" % (record["h5file"].name, exposure))
    adr_seconds = time.perf_counter() - adr_started
    print("[cube] ADR construction: %.3f s" % adr_seconds, flush=True)

    cube = np.zeros((len(DEF_WAVE), n, n), dtype=np.float32)
    variancecube = np.zeros_like(cube)
    weightcube = np.zeros_like(cube)
    ncontribcube = np.zeros((len(DEF_WAVE), n, n), dtype=np.uint8)
    dqcube = np.zeros((len(DEF_WAVE), n, n), dtype=np.uint16)
    reconstruction_started = time.perf_counter()
    variance_stats_by_wave = []

    def render_wavelength(index):
        x, y = tp.wcs_world2pix(raarray[:, index], decarray[:, index], 1)
        Pos = np.column_stack((x, y))
        return index, make_image_gaussian(
            Pos, specarray[:, index], errarray[:, index],
            xg, yg, xgrid, ygrid,
            (GAUSSIAN_FWHM_ARCSEC / 2.35) / args.pixel_scale,
            shot_indices)

    with ThreadPoolExecutor(max_workers=args.wave_workers) as executor:
        for index, result in executor.map(render_wavelength, range(len(DEF_WAVE))):
            image, variance, coverage, ncontrib, dq, variance_stats = result
            cube[index] = image
            variancecube[index] = variance
            weightcube[index] = coverage
            ncontribcube[index] = ncontrib
            dqcube[index] = dq
            if index in (0, len(DEF_WAVE) // 2, len(DEF_WAVE) - 1):
                variance_stats_by_wave.append({
                    "index": int(index),
                    "wavelength_A": float(DEF_WAVE[index]),
                    **variance_stats,
                })
    reconstruction_seconds = time.perf_counter() - reconstruction_started
    print("[cube] wavelength-plane reconstruction: %.3f s" %
          reconstruction_seconds, flush=True)

    direct_groups = int(sum(row["alpha_direct_groups"] for row in calibration_rows))
    historical_groups = int(sum(row["alpha_historical_groups"] for row in calibration_rows))
    model_provenance = {
        "state": _file_identity(state_path),
        "band_cache": _file_identity(cache_path),
        "band_cache_manifest": _file_identity(cache_manifest_path),
        "alpha_fits": _file_identity(alpha_fits_path),
        "alpha_history": _file_identity(alpha_summary_path),
        "amp_means": _file_identity(amp_means_path),
        "fq_template": _file_identity(fq_path),
        "blank_file": _file_identity(args.external_blank_fibers),
    }
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "script": str(Path(__file__).resolve()),
        "input_h5_glob": args.h5files,
        "input_h5_files": [str(path) for path in h5files],
        "model": {
            "name": "current_staged_m101_v1",
            "D_definition": "Fibers.spectrum/Survey.offset + Fibers.skyspectrum",
            "M_definition": "exp(hierarchy.cumulative_final.R_total_h_i_a)",
            "M_wavelength_independent": True,
            "alpha_definition": "alpha*(f_q(q)-f0)*K_e(lambda)",
            "alpha_priority": ["direct blank-rich alpha_joint", "historical same-amp robust alpha"],
            "historical_f0": float(historical_f0),
            "spectral_additive": "historical physical-amplifier robust mean residual mu_a(lambda)",
            "spectral_mean_product_scope": mean_scope,
            "spectral_mean_finite_wavelength_samples": int(np.sum(mean_science_mask)),
            "pca_coefficients_applied": False,
            "corrected_total_equation": "T=(D-alpha*(f-f0)*K-mu)/M",
            "sky_definition": "robust corrected total from classified hardware-valid blank q=40..75 fibers",
            "object_equation": "O=T-S_e",
            "formal_error_equation": "Fibers.error/abs(Survey.offset)/M",
            "additive_model_uncertainty_propagated": False,
            "sky_uncertainty_propagated": False,
            "direct_alpha_groups": direct_groups,
            "historical_alpha_groups": historical_groups,
        },
        "provenance": model_provenance,
        "response_composition_validation": composition,
        "blank_loader": blank_provenance,
        "preflight": preflight,
        "coverage_table": str(coverage_path.resolve()),
        "calibration_summary_table": str(calibration_path.resolve()),
        "sky_summary_table": str(sky_path.resolve()),
        "image_center_size": {
            "RA": ra0, "Dec": dec0, "size_arcsec": size_arcsec,
            "pixels": n, "pixel_scale_arcsec": args.pixel_scale,
        },
        "wave_grid": {
            "start_A": float(DEF_WAVE[0]),
            "stop_A": float(DEF_WAVE[-1]),
            "step_A": float(DEF_WAVE[1] - DEF_WAVE[0]),
            "planes": len(DEF_WAVE),
        },
        "gaussian_reconstruction": {
            "fwhm_arcsec": GAUSSIAN_FWHM_ARCSEC,
            "fiber_radius_arcsec": FIBER_RADIUS_ARCSEC,
            "sigma_pixels": (GAUSSIAN_FWHM_ARCSEC / 2.35) / args.pixel_scale,
            "shots": len(shot_indices),
        },
        "variance_sample_wavelengths": variance_stats_by_wave,
        "synthetic_validation": synthetic,
        "timing_seconds": {
            "spectral_calibration_and_sky": calibration_seconds,
            "adr_construction": adr_seconds,
            "wavelength_plane_reconstruction": reconstruction_seconds,
            "total_before_write": time.perf_counter() - started,
        },
        "no_calibration_fitting_or_optimization": True,
    }

    writing_started = time.perf_counter()
    _write_cube_products(
        args.surname, cube, variancecube, dqcube, weightcube,
        ncontribcube, tp, n, args.pixel_scale, model_provenance, summary)
    summary["timing_seconds"]["fits_writing"] = time.perf_counter() - writing_started
    summary["timing_seconds"]["total_runtime"] = time.perf_counter() - started
    summary_path = Path("%s_current_model_summary.json" % args.surname)
    summary_path.write_text(json.dumps(summary, indent=2, default=str))
    print("[cube] FITS writing: %.3f s" %
          summary["timing_seconds"]["fits_writing"], flush=True)
    print("[cube] total runtime: %.3f s" %
          summary["timing_seconds"]["total_runtime"], flush=True)
    print("[cube] summary: %s" % summary_path.resolve(), flush=True)


if __name__ == "__main__":
    main()
