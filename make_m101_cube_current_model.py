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
from dataclasses import dataclass, field
from datetime import datetime, timezone
import gc
import glob
import json
import logging
import os.path as op
from pathlib import Path
import tempfile
import time
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tables
from astropy.io import fits
from astropy.table import Table
from numba import njit
from scipy.interpolate import PchipInterpolator
from scipy.interpolate import griddata

from astrometry import Astrometry
from extract import Extract
import diagnose_m101_hierarchical as validated_m101
import diagnose_m101_frozenM_spectral_residuals as frozen_parent
from m101_hardware_exclusions import HARDWARE_EXCLUSIONS, hardware_excluded
import m101_native_data as native
from m101_calibration_utils import collapse_many, robust_scatter
from math_utils import biweight


DEF_WAVE = np.asarray(validated_m101.DEF_WAVE, dtype=float)
N_FIBER_AMP = validated_m101.N_FIBER_AMP
N_EXPOSURES = validated_m101.N_EXPOSURES
FIBER_RADIUS_ARCSEC = validated_m101.FIBER_RADIUS_ARCSEC
FIBER_AREA = np.pi * 0.75 ** 2
Q_MIN = 40
Q_MAX = 75
M101_SKY_MIN_FINITE_FRACTION = 0.8
M101_SKY_MIN_FIBERS = 20
GAUSSIAN_FWHM_ARCSEC = 1.8
QA_NUMERICAL_FLOOR = np.finfo(float).eps
QA_WARN_Z = 5.0
QA_BANDS = ("ON", "OFF")
SB_BANDS = ("ON", "OFF")
EXPOSURES = (1, 2, 3)
BOOTSTRAP_SEED = 110101
DEFAULT_BOOTSTRAP_DRAWS = 300
HARDWARE_REGISTRY_PATH = Path(hardware_excluded.__code__.co_filename).resolve()

DQ_INSUFFICIENT_SUPPORT = np.uint16(1 << 0)
DQ_VAR_INCOMPLETE = np.uint16(1 << 1)
DQ_EMPIRICAL_VAR_USED = np.uint16(1 << 2)
DQ_FORMAL_VAR_USED = np.uint16(1 << 3)
DQ_VAR_EMPIRICAL_ONLY = np.uint16(1 << 4)

# Empirical LSF/FWHM calibration anchors ported from the latest established
# make_mosaic_cube_revised.py path.  The two known blends remain QA-visible
# but are never published as usable LSF anchors.
REFERENCE_ARC_WAVELENGTHS = np.array([
    3610.508, 3650.153, 4046.565, 4358.335, 4678.149,
    4799.912, 4916.068, 5085.822, 5460.750,
], dtype=float)
ARC_WINDOW_ANGSTROM = 15.0
ARC_USABLE_FOR_LSF = np.array(
    [False, False, True, True, True, True, True, True, True], dtype=bool)
ARC_BLEND_STATES = np.array([
    "KNOWN_BLEND", "KNOWN_BLEND", "UNKNOWN", "UNKNOWN", "UNKNOWN",
    "UNKNOWN", "UNKNOWN", "UNKNOWN", "UNKNOWN",
], dtype="U16")
ARC_BLEND_NOTES = {
    3610.508: ("Known Cd I blend with 3612.873 A and 3614.453 A; "
               "excluded from LSF"),
    3650.153: ("Known Hg/Cd blend; Hg I 3654.84 A produces red shoulder; "
               "excluded from LSF"),
    4046.565: "No significant known blend affecting half-maximum width",
    4358.335: "No significant known blend affecting half-maximum width",
    4678.149: "No significant known blend affecting half-maximum width",
    4799.912: "No significant known blend affecting half-maximum width",
    4916.068: "No significant known blend affecting half-maximum width",
    5085.822: "No significant known blend affecting half-maximum width",
    5460.750: "No significant known blend affecting half-maximum width",
}


# -----------------------------------------------------------------------------
# Small utilities and model-product readers
# -----------------------------------------------------------------------------


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


def _write_rows(path, rows, fields=None):
    rows = list(rows)
    path = Path(path)
    if not rows:
        path.write_text("")
        return
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
            writer.writerow({key: row.get(key, "") for key in fields})


def _physical_key(specid, ifuslot, ifuid, amp):
    return (int(specid), int(ifuslot), int(ifuid), str(amp).upper())


def _persistent_hardware_registry_match(h5_name, specid, ifuslot, ifuid, amp):
    """Classify a match using persistent records from the shared registry."""
    physical = _physical_key(specid, ifuslot, ifuid, amp)
    for record in HARDWARE_EXCLUSIONS:
        if "specid" not in record or record.get("scope") != "fit_and_cube":
            continue
        if "h5" in record and Path(str(h5_name)).name != record["h5"]:
            continue
        if physical == (int(record["specid"]), int(record["ifuslot"]),
                        int(record["ifuid"]), str(record["amp"]).upper()):
            return True
    return False


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
    sky_by_exposure = {}
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
        sky_by_exposure[exposure] = sky.copy()

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
    return spectra, records, sky_by_exposure


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
        hardware_bad = np.zeros(source.shape[0], dtype=bool)
        hardware_excluded_groups = 0
        persistent_hardware_excluded_groups = 0
        hardware_excluded_fiber_rows = 0
        persistent_hardware_excluded_fiber_rows = 0
        h5_date = h5_name[:8]
        if len(h5_date) != 8 or not h5_date.isdigit():
            raise ValueError("H5 basename does not begin with a UT date: %s" % h5_name)
        for group in groups:
            specid = int(group["specid"])
            ifuslot = int(group["ifuslot"])
            ifuid = int(group["ifuid"])
            amp = str(group["amp"]).upper()
            if not hardware_excluded(
                    date=h5_date, h5=h5_name, specid=specid,
                    ifuslot=ifuslot, ifuid=ifuid, amp=amp,
                    purpose="cube"):
                continue
            indices = np.asarray(group["indices"], dtype=int)
            hardware_bad[indices] = True
            hardware_excluded_groups += 1
            hardware_excluded_fiber_rows += int(indices.size)
            if _persistent_hardware_registry_match(
                    h5_name, specid, ifuslot, ifuid, amp):
                persistent_hardware_excluded_groups += 1
                persistent_hardware_excluded_fiber_rows += int(indices.size)

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
                group["IFU_CODE"] = ifu_code
                m_key = (h5_name, ifu_code, amp)
                if m_key not in response or not np.isfinite(response[m_key]):
                    group["has_frozen_M"] = False
                    masked_missing_m_groups += 1
                    continue
                group["has_frozen_M"] = True

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

    spectra, sky_records, sky_by_exposure = _subtract_exposure_sky(
        corrected_total, labels, q_rows, blank_mask, hardware_bad,
        h5_name, min_sky_fibers)

    calibration_summary = {
        "H5": h5_name,
        "supported_groups": int(supported_groups),
        "masked_missing_frozen_M_groups": int(masked_missing_m_groups),
        "alpha_direct_groups": int(alpha_source_counts["direct_blank_rich"]),
        "alpha_historical_groups": int(alpha_source_counts["historical_same_amp"]),
        "hardware_excluded_groups": int(hardware_excluded_groups),
        "persistent_hardware_excluded_groups": int(
            persistent_hardware_excluded_groups),
        "hardware_excluded_fiber_rows": int(hardware_excluded_fiber_rows),
        "persistent_hardware_excluded_fiber_rows": int(
            persistent_hardware_excluded_fiber_rows),
        "logM": _distribution(logm_used),
        "alpha": _distribution(alpha_used),
        "robust_mean_training_exposures": _distribution(mean_support_used),
    }
    return (spectra, errors, ra, dec, labels, surveys, calibration_summary,
            sky_records, sky_by_exposure, groups, q_rows, hardware_bad,
            corrected_total)


# -----------------------------------------------------------------------------
# Accepted two-pass exposure-scale calibration
# -----------------------------------------------------------------------------


@dataclass
class Shot:
    """One H5/exposure view used by the accepted SB exposure-scale fit."""

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


def _finite(values):
    values = np.asarray(values, dtype=float)
    return values[np.isfinite(values)]


def _effective_wavelengths(responses):
    responses = np.asarray(responses, dtype=float)
    totals = np.sum(responses, axis=1)
    return np.divide(
        responses @ DEF_WAVE, totals,
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
    """Bootstrap a through-origin fit by resampling whole physical amplifiers."""
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


def _derive_exposure_scale_geometry(shots, pixel_scale):
    """Use Diagnose's shot-derived geometry for the accepted SB images."""
    ra = np.concatenate([shot.ra for shot in shots])
    dec = np.concatenate([shot.dec for shot in shots])
    valid = np.isfinite(ra) & np.isfinite(dec)
    if not np.any(valid):
        raise ValueError("no finite fiber coordinates for exposure-scale geometry")
    ra0 = float(np.nanmedian(ra[valid]))
    dec0 = float(np.nanmedian(dec[valid]))
    x = (ra[valid] - ra0) * np.cos(np.deg2rad(dec0)) * 3600.0
    y = (dec[valid] - dec0) * 3600.0
    extent = max(float(np.nanmax(np.abs(x))), float(np.nanmax(np.abs(y))))
    size_arcmin = max(1.0, (2.0 * extent + 40.0) / 60.0)
    value = "%.10f,%.10f,%.10f" % (ra0, dec0, size_arcmin)
    return _parse_image_geometry(value, pixel_scale)


def _derive_final_qa_geometry(records, pixel_scale):
    """Use the full fiber-covering geometry for final IMAGE_QA."""
    ra = np.concatenate([np.asarray(record["ra"], dtype=float)
                         for record in records])
    dec = np.concatenate([np.asarray(record["dec"], dtype=float)
                          for record in records])
    valid = np.isfinite(ra) & np.isfinite(dec)
    if not np.any(valid):
        raise ValueError("no finite fiber coordinates for final QA geometry")
    ra0 = float(np.nanmedian(ra[valid]))
    dec0 = float(np.nanmedian(dec[valid]))
    x = (ra[valid] - ra0) * np.cos(np.deg2rad(dec0)) * 3600.0
    y = (dec[valid] - dec0) * 3600.0
    extent = max(float(np.nanmax(np.abs(x))), float(np.nanmax(np.abs(y))))
    size_arcmin = max(1.0, (2.0 * extent + 40.0) / 60.0)
    value = "%.10f,%.10f,%.10f" % (ra0, dec0, size_arcmin)
    return _parse_image_geometry(value, pixel_scale)


def _adr_positions(ra, dec, survey, response):
    """Use the established IMAGE_QA ADR convention at one band wavelength."""
    survey_ra = _survey_value(survey, "ra")
    survey_dec = _survey_value(survey, "dec")
    pa = _survey_value(survey, "pa")
    if not np.all(np.isfinite([survey_ra, survey_dec, pa])):
        raise ValueError("Survey RA/Dec/PA is invalid")
    astrometry = Astrometry(survey_ra, survey_dec, pa, 0.0, 0.0)
    extractor = Extract(wave=DEF_WAVE)
    extractor.get_ADR_RAdec(astrometry)
    effective = float(_effective_wavelengths(np.asarray(response)[None, :])[0])
    adr_ra = float(np.interp(effective, DEF_WAVE, extractor.ADRra))
    adr_dec = float(np.interp(effective, DEF_WAVE, extractor.ADRdec))
    corrected_ra = (np.asarray(ra, dtype=float) - adr_ra / 3600.0 /
                    np.cos(np.deg2rad(survey_dec)))
    corrected_dec = np.asarray(dec, dtype=float) - adr_dec / 3600.0
    return np.column_stack((corrected_ra, corrected_dec))


def _pixel_positions(positions, tp):
    x, y = tp.wcs_world2pix(positions[:, 0], positions[:, 1], 1)
    return np.column_stack((np.asarray(x, dtype=float),
                            np.asarray(y, dtype=float)))


def _build_exposure_scale_image_cache(shots, band, tp, xg, yg,
                                      pixel_scale, path):
    """Build the collapsed per-shot image stack used by the SB estimator."""
    shape = (len(shots), len(yg), len(xg))
    stack = np.memmap(path, dtype=np.float32, mode="w+", shape=shape)
    for index, shot in enumerate(shots):
        pixel = _pixel_positions(shot.positions[band], tp)
        values = np.asarray(shot.values[band], dtype=float)
        local = np.arange(pixel.shape[0], dtype=np.int64)
        stack[index] = _qa_shot_images(
            pixel, values, [local], xg, yg, pixel_scale)[0]
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
    """Fit one accepted robust physical-amplifier SB estimator."""
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
    result["SB_source_leverage"] = result["sqrt_sum_predicted_squared_SB_joint"]
    leverage = _predictor_diagnostics(joint_pred)
    result["SB_max_predictor_leverage_fraction"] = leverage[
        "max_predictor_leverage_fraction"]
    result["SB_N_effective_leverage"] = leverage["N_effective_leverage"]
    return result


def _process_sb_prediction_band(shots, stack, band, tp):
    """Fill one SB band's predictions while only one image stack is mapped."""
    for shot_index, shot in enumerate(shots):
        predictions, _support = _loo_samples(
            stack, shot_index, shot.positions[band], tp)
        shot.sb_predictions[band] = predictions


def _exposure_scale_row(shot, fit, chronological_index, prefix):
    row = {"H5": shot.h5, "exposure": int(shot.exposure),
           "chronological_exposure_index": int(chronological_index)}
    row["%s_ON" % prefix] = fit["g_SB_ON"]
    row["%s_OFF" % prefix] = fit["g_SB_OFF"]
    row["%s_raw" % prefix] = fit["g_SB_joint"]
    row["sigma_%s" % prefix] = fit["sigma_g_SB"]
    row["%s_N_SB_ON" % prefix] = fit["N_SB_ON"]
    row["%s_N_SB_OFF" % prefix] = fit["N_SB_OFF"]
    row["%s_N_SB_BOTH" % prefix] = fit["N_SB_BOTH"]
    row["%s_residual_scatter_SB_joint" % prefix] = fit[
        "residual_scatter_SB_joint"]
    row["%s_SB_source_leverage" % prefix] = fit["SB_source_leverage"]
    row["%s_SB_max_predictor_leverage_fraction" % prefix] = fit[
        "SB_max_predictor_leverage_fraction"]
    row["%s_SB_N_effective_leverage" % prefix] = fit[
        "SB_N_effective_leverage"]
    return row


def _apply_exposure_relative_scales(records, scale_rows_by_key, field_name):
    """Apply one relative-scale iteration to the already sky-subtracted data."""
    for record in records:
        h5_name = record["h5file"].name
        for exposure in EXPOSURES:
            scale_row = scale_rows_by_key[(h5_name, exposure)]
            relative = float(scale_row[field_name])
            if not np.isfinite(relative) or relative == 0.0:
                raise RuntimeError(
                    "invalid relative SB scale for %s exposure %d (%s)" %
                    (h5_name, exposure, field_name))
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
        collapsed, _ = collapse_many(record["spectra"], band_responses)
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
    """Validate O2=O0/(g1_rel*g2_rel) and O2=corrected_total2-final_sky2."""
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
            # Reproduce the stored float32 subtraction itself.
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
                    raise RuntimeError(
                        "O=corrected_total-final_sky failed for %s/e%d" %
                        (record["h5file"].name, exposure))
    return {
        "finite_direct_comparisons": int(n_direct),
        "finite_identity_comparisons": int(n_identity),
        "max_abs_direct_error": float(max_direct_error),
        "max_abs_identity_error": float(max_identity_error),
        "direct_formula": "final spectra = original post-sky spectra / (g1_rel*g2_rel)",
        "identity_formula": "O2 = corrected_total2 - final_sky2",
    }


def _apply_two_pass_exposure_scale(records, band_responses, pixel_scale,
                                   bootstrap_draws=DEFAULT_BOOTSTRAP_DRAWS):
    """Apply exactly two accepted relative SB exposure-scale corrections.

    The third fit is closure QA only.  This helper intentionally does not run
    amplifier QA or apply amplifier rejection; both remain in the production
    sequence immediately after this helper returns.
    """
    shots = []
    for record in records:
        collapsed, _ = collapse_many(record["spectra"], band_responses)
        amp_keys_full = _amp_keys_for_rows(record["groups"],
                                           len(record["labels"]))
        for exposure in EXPOSURES:
            selected = np.flatnonzero(record["labels"] == exposure)
            shots.append(Shot(
                h5=record["h5file"].name,
                exposure=exposure,
                survey=record["surveys"][exposure],
                ra=record["ra"][selected],
                dec=record["dec"][selected],
                values={
                    "ON": np.asarray(collapsed[selected, 5], dtype=np.float32),
                    "OFF": np.asarray(collapsed[selected, 6], dtype=np.float32),
                },
                amp_keys=np.asarray(amp_keys_full[selected], dtype=object),
                calibration_summary=record.get("calibration_summary", {})))

    if len(shots) != len(records) * len(EXPOSURES):
        raise RuntimeError("exposure-scale shot construction is incomplete")
    for shot in shots:
        shot.positions["ON"] = _adr_positions(
            shot.ra, shot.dec, shot.survey, band_responses[5])
        shot.positions["OFF"] = _adr_positions(
            shot.ra, shot.dec, shot.survey, band_responses[6])
    _ra0, _dec0, _size_arcsec, _n, xg, yg, _xgrid, _ygrid, tp = \
        _derive_exposure_scale_geometry(shots, pixel_scale)

    with tempfile.TemporaryDirectory(prefix="m101-exposure-scale-") as cache_dir:
        for band in SB_BANDS:
            stack = _build_exposure_scale_image_cache(
                shots, band, tp, xg, yg, pixel_scale,
                Path(cache_dir) / (band.lower() + "_raw_shot_images.float32"))
            _process_sb_prediction_band(shots, stack, band, tp)
            del stack
            gc.collect()
        exposure_rows = []
        for index, shot in enumerate(shots, 1):
            fit = _fit_sb(shot, bootstrap_draws,
                          BOOTSTRAP_SEED + index)
            exposure_rows.append(_exposure_scale_row(shot, fit, index, "g1"))

        g1_values = _finite([row["g1_raw"] for row in exposure_rows])
        if not g1_values.size:
            raise RuntimeError("no finite raw g1 values")
        center1 = float(np.median(g1_values))
        if not np.isfinite(center1) or center1 == 0.0:
            raise RuntimeError("invalid first-iteration SB gauge center")
        for row in exposure_rows:
            row["center1"] = center1
            row["g1_rel"] = (row["g1_raw"] / center1
                              if np.isfinite(row["g1_raw"]) else np.nan)

        scale_by_key = {(row["H5"], int(row["exposure"])): row
                        for row in exposure_rows}
        _apply_exposure_relative_scales(records, scale_by_key, "g1_rel")
        _update_shot_values_from_records(shots, records, band_responses)

        for band in SB_BANDS:
            stack = _build_exposure_scale_image_cache(
                shots, band, tp, xg, yg, pixel_scale,
                Path(cache_dir) / (band.lower() + "_corrected_shot_images.float32"))
            _process_sb_prediction_band(shots, stack, band, tp)
            del stack
            gc.collect()
        for index, shot in enumerate(shots):
            fit = _fit_sb(shot, bootstrap_draws,
                          BOOTSTRAP_SEED + 5000 + index)
            row = exposure_rows[index]
            row["g2_ON"] = fit["g_SB_ON"]
            row["g2_OFF"] = fit["g_SB_OFF"]
            row["g2_raw"] = fit["g_SB_joint"]
            row["sigma_g2"] = fit["sigma_g_SB"]
            row["g2_N_SB_ON"] = fit["N_SB_ON"]
            row["g2_N_SB_OFF"] = fit["N_SB_OFF"]
            row["g2_N_SB_BOTH"] = fit["N_SB_BOTH"]
            row["g2_residual_scatter_SB_joint"] = fit[
                "residual_scatter_SB_joint"]
            row["g2_SB_source_leverage"] = fit["SB_source_leverage"]
            row["g2_SB_max_predictor_leverage_fraction"] = fit[
                "SB_max_predictor_leverage_fraction"]
            row["g2_SB_N_effective_leverage"] = fit[
                "SB_N_effective_leverage"]

        g2_values = _finite([row["g2_raw"] for row in exposure_rows])
        if not g2_values.size:
            raise RuntimeError("no finite raw g2 values")
        center2 = float(np.median(g2_values))
        if not np.isfinite(center2) or center2 == 0.0:
            raise RuntimeError("invalid second-iteration SB gauge center")
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

        # Exactly one second relative correction; g3 below is never applied.
        _apply_exposure_relative_scales(records, scale_by_key, "g2_rel")
        _update_shot_values_from_records(shots, records, band_responses)
        closure_state = []
        for record in records:
            closure_state.append((
                np.array(record["spectra"], copy=True),
                np.array(record["errors"], copy=True),
                np.array(record["corrected_total"], copy=True),
                {key: np.array(value, copy=True)
                 for key, value in record["sky_by_exposure"].items()},
            ))

        for band in SB_BANDS:
            stack = _build_exposure_scale_image_cache(
                shots, band, tp, xg, yg, pixel_scale,
                Path(cache_dir) / (band.lower() + "_iteration2_shot_images.float32"))
            _process_sb_prediction_band(shots, stack, band, tp)
            del stack
            gc.collect()
        for index, shot in enumerate(shots):
            fit = _fit_sb(shot, bootstrap_draws,
                          BOOTSTRAP_SEED + 10000 + index)
            row = exposure_rows[index]
            row["g3_raw"] = fit["g_SB_joint"]
            row["sigma_g3"] = fit["sigma_g_SB"]
            row["g3_minus_one"] = row["g3_raw"] - 1.0

        for record, before in zip(records, closure_state):
            after = (
                record["spectra"], record["errors"],
                record["corrected_total"], record["sky_by_exposure"])
            for before_value, after_value in zip(before[:3], after[:3]):
                if not np.array_equal(before_value, after_value,
                                      equal_nan=True):
                    raise RuntimeError("g3 closure measurement modified calibrated arrays")
            for key, before_value in before[3].items():
                if not np.array_equal(before_value, after[3][key],
                                      equal_nan=True):
                    raise RuntimeError("g3 closure measurement modified final sky")
        del closure_state
        gc.collect()

    scale_by_key = {(row["H5"], int(row["exposure"])): row
                    for row in exposure_rows}
    validation = _validate_cumulative_scales(records, scale_by_key)
    for row in exposure_rows:
        row["validation_max_abs_direct_error"] = validation[
            "max_abs_direct_error"]
        row["validation_max_abs_identity_error"] = validation[
            "max_abs_identity_error"]

    if len(exposure_rows) == len(EXPOSURES) * 19:
        for field_name in ("g1_raw", "g1_rel", "g2_raw", "g2_rel",
                           "cumulative_relative_scale", "g3_raw"):
            if np.count_nonzero(np.isfinite(
                    [row[field_name] for row in exposure_rows])) != 57:
                raise RuntimeError("canonical exposure-scale field is not finite: %s" %
                                   field_name)
    for prefix in ("g1", "g2"):
        relatives = _finite([row["%s_rel" % prefix]
                             for row in exposure_rows])
        if relatives.size and not np.isclose(np.median(relatives), 1.0,
                                             rtol=0.0, atol=1.0e-12):
            raise RuntimeError("median %s_rel is not unity" % prefix)
    for record in records:
        record.pop("original_spectra", None)
    del shots
    gc.collect()
    return exposure_rows


# -----------------------------------------------------------------------------
# Final amplifier/exposure QA (after O, before cube concatenation)
# -----------------------------------------------------------------------------


def _qa_location(values, axis=None):
    values = np.asarray(values, dtype=float)
    if axis == 0 and values.ndim == 2:
        supported = np.any(np.isfinite(values), axis=0)
        result = np.full(values.shape[1], np.nan, dtype=float)
        if np.any(supported):
            result[supported] = frozen_parent._safe_location(
                values[:, supported], axis=0)
        return result
    finite = values[np.isfinite(values)]
    if not finite.size:
        return np.nan
    return float(frozen_parent._safe_location(finite))


def _qa_scatter(values):
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size < 2:
        return np.nan
    with np.errstate(all="ignore"):
        result = robust_scatter(finite)
    return float(result) if np.isfinite(result) else np.nan


def _qa_scale(*values):
    finite = []
    for value in values:
        value = float(value)
        if np.isfinite(value):
            finite.append(abs(value))
    return max(finite + [QA_NUMERICAL_FLOOR])


def _qa_group_key(h5_name, group):
    return (
        Path(h5_name).name,
        int(group["exposure"]),
        int(group["specid"]),
        int(group["ifuslot"]),
        int(group["ifuid"]),
        str(group["amp"]).upper(),
    )


def _qa_row_key(row):
    return (
        Path(row["H5"]).name,
        int(row["exposure"]),
        int(row["SPECID"]),
        int(row["IFUSLOT"]),
        int(row["IFUID"]),
        str(row["AMP"]).upper(),
    )


def _load_qa_reject_file(path):
    if path is None:
        return set(), None
    path = Path(path).expanduser().resolve()
    required = {"H5", "exposure", "SPECID", "IFUSLOT", "IFUID", "AMP"}
    keys = set()
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError("QA reject file lacks columns: %s" % sorted(missing))
        for row in reader:
            key = (
                Path(row["H5"]).name,
                int(row["exposure"]),
                int(row["SPECID"]),
                int(row["IFUSLOT"]),
                int(row["IFUID"]),
                str(row["AMP"]).strip().upper(),
            )
            if key in keys:
                raise ValueError("duplicate QA reject group: %s" % (key,))
            keys.add(key)
    return keys, path


def _qa_band_scalar(spectrum, responses):
    values, _ = collapse_many(
        np.asarray(spectrum, dtype=float)[None, :], responses)
    return np.asarray(values[0], dtype=float)


def _build_qa_band_arrays(records, qa_responses, effective, tp):
    """Collapse final O and build band-effective ADR positions."""
    total_rows = int(sum(len(record["ra"]) for record in records))
    values = np.full((total_rows, len(QA_BANDS)), np.nan, dtype=float)
    positions = np.full((len(QA_BANDS), total_rows, 2), np.nan, dtype=float)
    shot_indices = []
    shot_names = []
    offsets = []
    offset = 0
    for record in records:
        n_rows = len(record["ra"])
        offsets.append(offset)
        values[offset:offset + n_rows], _ = collapse_many(
            np.asarray(record["spectra"], dtype=float), qa_responses)
        extractor = Extract(wave=DEF_WAVE)
        for exposure in range(1, N_EXPOSURES + 1):
            selected = np.asarray(record["labels"], dtype=int) == exposure
            local = np.flatnonzero(selected)
            if not local.size:
                continue
            survey = record["surveys"][exposure]
            astrometry = Astrometry(
                float(survey["ra"]), float(survey["dec"]),
                float(survey["pa"]), 0.0, 0.0)
            extractor.get_ADR_RAdec(astrometry)
            for band_index, wavelength in enumerate(effective[-2:]):
                adr_ra = float(np.interp(wavelength, DEF_WAVE, extractor.ADRra))
                adr_dec = float(np.interp(wavelength, DEF_WAVE, extractor.ADRdec))
                positions[band_index, offset + local, 0] = (
                    record["ra"][local] - adr_ra / 3600.0 /
                    np.cos(np.deg2rad(float(survey["dec"]))))
                positions[band_index, offset + local, 1] = (
                    record["dec"][local] - adr_dec / 3600.0)
            shot_indices.append(offset + local)
            shot_names.append("%s exposure %d" % (record["h5file"].name, exposure))
        offset += n_rows
    return values, positions, shot_indices, shot_names, offsets


def _qa_shot_images(pos, values, shot_indices, xg, yg, pixel_scale):
    """Build per-shot Gaussian images using the production splat kernel."""
    ny, nx = len(yg), len(xg)
    shot_images = np.full((len(shot_indices), ny, nx), np.nan, dtype=np.float32)
    errors = np.ones(values.shape[0], dtype=np.float32)
    sigma = (GAUSSIAN_FWHM_ARCSEC / 2.35) / pixel_scale
    for shot_number, indices in enumerate(shot_indices):
        flux_sum = np.zeros((ny, nx), dtype=np.float32)
        weight_sum = np.zeros((ny, nx), dtype=np.float32)
        variance_numerator = np.zeros((ny, nx), dtype=np.float32)
        error_weight_sum = np.zeros((ny, nx), dtype=np.float32)
        support_map = np.zeros((ny, nx), dtype=np.uint8)
        _gaussian_splat_shot_xy(
            np.asarray(indices, dtype=np.int64), pos[:, 0], pos[:, 1], values,
            errors, float(xg[0]), float(yg[0]), nx, ny,
            float(sigma), 3, int(np.ceil(2.0 * sigma)),
            float(np.pi * FIBER_RADIUS_ARCSEC ** 2), flux_sum, weight_sum,
            variance_numerator, error_weight_sum, support_map)
        supported = (support_map != 0) & (weight_sum > 0.0)
        shot_images[shot_number, supported] = (
            flux_sum[supported] / weight_sum[supported])
    return shot_images


def _qa_loo_predictions(pos, shot_images, shot_indices, tp, xg, yg):
    """Evaluate each target shot against a median of all other shot images."""
    predictions = np.full(pos.shape[0], np.nan, dtype=float)
    if len(shot_indices) < 3:
        return predictions
    all_shots = np.arange(len(shot_indices))
    for shot_number, indices in enumerate(shot_indices):
        x, y = tp.wcs_world2pix(pos[indices, 0], pos[indices, 1], 1)
        x_index = np.rint(x).astype(int) - 1
        y_index = np.rint(y).astype(int) - 1
        in_bounds = (
            np.isfinite(x) & np.isfinite(y) &
            (x_index >= 0) & (x_index < len(xg)) &
            (y_index >= 0) & (y_index < len(yg)))
        if not np.any(in_bounds):
            continue
        reference = shot_images[all_shots != shot_number][:, y_index[in_bounds],
                                                           x_index[in_bounds]]
        finite = np.isfinite(reference)
        support = np.sum(finite, axis=0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            predicted = np.nanmedian(reference, axis=0)
        predicted[support < 2] = np.nan
        destination = np.asarray(indices)[in_bounds]
        predictions[destination] = predicted
    return predictions


def _qa_distribution(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not values.size:
        return {"N": 0, "median": np.nan, "robust_scatter": np.nan,
                "p01": np.nan, "p99": np.nan}
    median = float(np.median(values))
    return {
        "N": int(values.size),
        "median": median,
        "robust_scatter": float(1.4826 * np.median(np.abs(values - median))),
        "p01": float(np.percentile(values, 1.0)),
        "p99": float(np.percentile(values, 99.0)),
    }


def _qa_positive_source_ratio(observed, predicted):
    observed = np.asarray(observed, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    valid = (
        np.isfinite(observed) & np.isfinite(predicted) &
        (observed > 0.0) & (predicted > 0.0))
    if not np.any(valid):
        return 0, np.nan
    return int(np.count_nonzero(valid)), float(
        np.median(observed[valid] / predicted[valid]))


def _qa_robust_plot_limits(values, q_lo=0.01, q_hi=0.99, padding=0.08,
                           include_zero=False, symmetric=False):
    """Set display limits from the central population, not catastrophic rows."""
    values = np.asarray(values, dtype=float).ravel()
    values = values[np.isfinite(values)]
    if values.size == 0:
        values = np.asarray([0.0])
    lo, hi = np.percentile(values, (q_lo * 100.0, q_hi * 100.0))
    if symmetric:
        bound = max(abs(float(lo)), abs(float(hi)), 0.5 * float(hi - lo))
        bound = max(bound * (1.0 + padding), 1.0e-6)
        return -bound, bound
    span = max(float(hi - lo), 1.0e-6)
    lo -= padding * span
    hi += padding * span
    if include_zero:
        lo = min(lo, 0.0)
        hi = max(hi, 0.0)
    if hi <= lo:
        center = 0.5 * (lo + hi)
        half = max(abs(center) * 0.1, 1.0e-6)
        lo, hi = center - half, center + half
    return float(lo), float(hi)


def _qa_plot_offscale(ax, x, y, x_limits, y_limits, color, marker):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    finite = np.isfinite(x) & np.isfinite(y)
    outside = (finite & ((x < x_limits[0]) | (x > x_limits[1]) |
                         (y < y_limits[0]) | (y > y_limits[1])))
    if np.any(outside):
        ax.scatter(np.clip(x[outside], *x_limits),
                   np.clip(y[outside], *y_limits), s=30, marker=marker,
                   facecolors="none", edgecolors=color, linewidths=0.8,
                   zorder=5)
    return int(np.count_nonzero(outside))


def _write_qa_figures(surname, rows):
    paths = []
    by_h5 = {}
    for row in rows:
        by_h5.setdefault(str(row["H5"]), []).append(row)
    for h5_name, h5_rows in sorted(by_h5.items()):
        fig, axes = plt.subplots(
            N_EXPOSURES, 2, figsize=(11, 3.2 * N_EXPOSURES),
            squeeze=False, constrained_layout=True)
        for exposure in range(1, N_EXPOSURES + 1):
            selected = [row for row in h5_rows if int(row["exposure"]) == exposure]
            ax_match, ax_fraction = axes[exposure - 1]
            match_x, match_y = [], []
            match_colors = []
            for band, color in (("ON", "tab:blue"), ("OFF", "tab:orange")):
                observed = np.asarray([
                    row["observed_%s_amp" % band] for row in selected], dtype=float)
                predicted = np.asarray([
                    row["predicted_%s_amp" % band] for row in selected], dtype=float)
                good = np.isfinite(observed) & np.isfinite(predicted)
                if np.any(good):
                    match_x.append(predicted[good])
                    match_y.append(observed[good])
                    match_colors.append(color)
                    ax_match.scatter(predicted[good], observed[good], s=16,
                                     alpha=0.75, label=band, color=color)
            if match_x:
                match_limits = _qa_robust_plot_limits(
                    np.concatenate(match_x + match_y), include_zero=True)
                ax_match.set_xlim(*match_limits)
                ax_match.set_ylim(*match_limits)
                ax_match.set_aspect("equal", adjustable="box")
                ax_match.plot(match_limits, match_limits, color="0.35", lw=1)
                offscale_match = 0
                for x_values, y_values, color in zip(match_x, match_y,
                                                      match_colors):
                    offscale_match += _qa_plot_offscale(
                        ax_match, x_values, y_values,
                        match_limits, match_limits, color, "v")
                if offscale_match:
                    ax_match.text(0.98, 0.03,
                                  "%d off robust range" % offscale_match,
                                  transform=ax_match.transAxes, ha="right",
                                  va="bottom", fontsize=7)
            ax_match.set_title("exposure %d: observed vs LOO predicted" % exposure)
            ax_match.set_xlabel("predicted amplifier value")
            ax_match.set_ylabel("observed amplifier value")
            ax_match.legend(fontsize=8)
            on = np.asarray([row["image_fractional_ON"] for row in selected], dtype=float)
            off = np.asarray([row["image_fractional_OFF"] for row in selected], dtype=float)
            good = np.isfinite(on) & np.isfinite(off)
            if np.any(good):
                ax_fraction.scatter(on[good], off[good], s=16, alpha=0.75,
                                    color="tab:purple")
                fraction_limits = _qa_robust_plot_limits(
                    np.concatenate((on[good], off[good])), symmetric=True)
                ax_fraction.set_xlim(*fraction_limits)
                ax_fraction.set_ylim(*fraction_limits)
                ax_fraction.set_aspect("equal", adjustable="box")
                offscale_fraction = _qa_plot_offscale(
                    ax_fraction, on[good], off[good], fraction_limits,
                    fraction_limits, "tab:red", "^")
                if offscale_fraction:
                    ax_fraction.text(
                        0.98, 0.03, "%d off robust range" % offscale_fraction,
                        transform=ax_fraction.transAxes, ha="right",
                        va="bottom", fontsize=7)
                ranked = np.argsort(np.maximum(np.abs(on[good]), np.abs(off[good])))[::-1]
                good_indices = np.flatnonzero(good)
                for index in ranked[:3]:
                    row = selected[good_indices[index]]
                    ax_fraction.annotate(
                        "%s/%s/%s/%s" % (row["SPECID"], row["IFUSLOT"],
                                          row["IFUID"], row["AMP"]),
                        (on[good_indices[index]], off[good_indices[index]]),
                        fontsize=6)
            ax_fraction.axhline(0.0, color="0.7", lw=0.8)
            ax_fraction.axvline(0.0, color="0.7", lw=0.8)
            ax_fraction.set_title("exposure %d: fractional ON vs OFF" % exposure)
            ax_fraction.set_xlabel("image_fractional_ON")
            ax_fraction.set_ylabel("image_fractional_OFF")
        fig.suptitle("%s final amplifier/exposure QA" % h5_name)
        path = Path("%s_amplifier_qa_%s.png" % (
            surname, Path(h5_name).stem))
        fig.savefig(path, dpi=150)
        plt.close(fig)
        paths.append(str(path.resolve()))
    return paths


def _evaluate_amplifier_qa(records, qa_responses, effective, tp, xg, yg,
                           pixel_scale, qa_min_blank, surname):
    band_values, band_positions, shot_indices, shot_names, offsets = (
        _build_qa_band_arrays(records, qa_responses, effective, tp))
    raw_predictions = []
    for band_index in range(len(QA_BANDS)):
        pos = band_positions[band_index]
        x, y = tp.wcs_world2pix(pos[:, 0], pos[:, 1], 1)
        pixel_pos = np.column_stack((x, y))
        shot_images = _qa_shot_images(
            pixel_pos, band_values[:, band_index], shot_indices, xg, yg,
            pixel_scale)
        raw_predictions.append(_qa_loo_predictions(
            pos, shot_images, shot_indices, tp, xg, yg))
        del shot_images
        gc.collect()

    predictions = [prediction * FIBER_AREA for prediction in raw_predictions]
    validation_observed = {band: [] for band in QA_BANDS}
    validation_raw = {band: [] for band in QA_BANDS}
    validation_integrated = {band: [] for band in QA_BANDS}

    rows = []
    for record_index, record in enumerate(records):
        start = offsets[record_index]
        blank_mask = np.asarray(record["blank_mask"], dtype=bool)
        hardware_bad = np.asarray(record["hardware_bad"], dtype=bool)
        q_rows = np.asarray(record["q_rows"], dtype=int)
        corrected_total = np.asarray(record["corrected_total"], dtype=float)
        spectra = np.asarray(record["spectra"], dtype=float)
        sufficient = np.mean(np.isfinite(spectra), axis=1) >= M101_SKY_MIN_FINITE_FRACTION
        sufficient_total = (np.mean(np.isfinite(corrected_total), axis=1) >=
                            M101_SKY_MIN_FINITE_FRACTION)
        for group in record["groups"]:
            indices = np.asarray(group["indices"], dtype=int)
            global_indices = start + indices
            key = _qa_group_key(record["h5file"].name, group)
            row = {
                "H5": key[0], "exposure": key[1], "SPECID": key[2],
                "IFUSLOT": key[3], "IFUID": key[4],
                "IFU_CODE": int(group["IFU_CODE"]), "AMP": key[5],
                "N_blank_QA": 0,
                "blank_residual_ON": np.nan,
                "blank_residual_OFF": np.nan,
                "blank_fractional_ON": np.nan,
                "blank_fractional_OFF": np.nan,
                "blank_spectral_scatter": np.nan,
                "BLANK_QA_STATUS": "NO_DIRECT_QA",
                "observed_ON_amp": np.nan,
                "predicted_ON_amp": np.nan,
                "image_difference_ON": np.nan,
                "image_fractional_ON": np.nan,
                "image_z_ON": np.nan,
                "observed_OFF_amp": np.nan,
                "predicted_OFF_amp": np.nan,
                "image_difference_OFF": np.nan,
                "image_fractional_OFF": np.nan,
                "image_z_OFF": np.nan,
                "QA_STATUS": "NO_DIRECT_QA",
                "QA_REASON": "unsupported frozen-M channel",
                "rejected_from_cube": False,
            }
            if not bool(group.get("has_frozen_M", False)):
                rows.append(row)
                continue

            blank = (
                blank_mask[indices] & ~hardware_bad[indices] &
                (q_rows[indices] >= Q_MIN) & (q_rows[indices] <= Q_MAX) &
                sufficient[indices])
            row["N_blank_QA"] = int(np.sum(blank))
            central = (
                ~hardware_bad[indices] & (q_rows[indices] >= Q_MIN) &
                (q_rows[indices] <= Q_MAX) & sufficient_total[indices])
            if row["N_blank_QA"] >= qa_min_blank:
                blank_residual = _qa_location(spectra[indices[blank]], axis=0)
                blank_band = _qa_band_scalar(blank_residual, qa_responses)
                row["blank_residual_ON"] = float(blank_band[0])
                row["blank_residual_OFF"] = float(blank_band[1])
                row["blank_spectral_scatter"] = _qa_scatter(blank_residual)
                row["BLANK_QA_STATUS"] = "PASS"
            sky = np.asarray(record["sky_by_exposure"][key[1]], dtype=float)
            if np.any(central):
                total_amp = _qa_location(corrected_total[indices[central]], axis=0)
                total_band = _qa_band_scalar(total_amp, qa_responses)
                sky_band = _qa_band_scalar(sky, qa_responses)
                row["blank_fractional_ON"] = (
                    row["blank_residual_ON"] /
                    _qa_scale(total_band[0], sky_band[0]))
                row["blank_fractional_OFF"] = (
                    row["blank_residual_OFF"] /
                    _qa_scale(total_band[1], sky_band[1]))

            for band_index, band in enumerate(QA_BANDS):
                observed = band_values[global_indices, band_index]
                raw_predicted = raw_predictions[band_index][global_indices]
                predicted = predictions[band_index][global_indices]
                usable = np.isfinite(observed) & np.isfinite(raw_predicted)
                if not np.any(usable):
                    continue
                observed_amp = _qa_location(observed[usable])
                raw_predicted_amp = _qa_location(raw_predicted[usable])
                predicted_amp = _qa_location(predicted[usable])
                difference = _qa_location(
                    observed[usable] - predicted[usable])
                row["observed_%s_amp" % band] = observed_amp
                row["predicted_%s_amp" % band] = predicted_amp
                row["image_difference_%s" % band] = difference
                sky_value = _qa_band_scalar(sky, qa_responses)[band_index]
                row["image_fractional_%s" % band] = (
                    difference / _qa_scale(predicted_amp, sky_value))
                if (
                        np.isfinite(observed_amp) and observed_amp > 0.0 and
                        np.isfinite(raw_predicted_amp) and raw_predicted_amp > 0.0 and
                        np.isfinite(predicted_amp) and predicted_amp > 0.0):
                    validation_observed[band].append(observed_amp)
                    validation_raw[band].append(raw_predicted_amp)
                    validation_integrated[band].append(predicted_amp)

            reasons = []
            z_values = []
            if row["BLANK_QA_STATUS"] == "NO_DIRECT_QA":
                reasons.append("NO_DIRECT_QA")
            for band in QA_BANDS:
                fraction = row["image_fractional_%s" % band]
                if np.isfinite(fraction):
                    z_values.append(fraction)
            if z_values:
                row["QA_STATUS"] = "PASS"
                row["QA_REASON"] = "IMAGE_QA leave-one-shot-out source consistency"
            else:
                row["QA_STATUS"] = "NO_DIRECT_QA"
                row["QA_REASON"] = "; ".join(reasons + ["no finite LOO prediction"])
            rows.append(row)

    print("[cube] IMAGE_QA unit validation: FIBER_AREA=%.10f" % FIBER_AREA,
          flush=True)
    for band in QA_BANDS:
        raw_n, raw_ratio = _qa_positive_source_ratio(
            validation_observed[band], validation_raw[band])
        integrated_n, integrated_ratio = _qa_positive_source_ratio(
            validation_observed[band], validation_integrated[band])
        print(
            "[cube] IMAGE_QA unit validation %s: "
            "N_positive=%d raw_sampled_ratio_median=%.10g "
            "predicted_integrated_ratio_median=%.10g" % (
                band, min(raw_n, integrated_n), raw_ratio, integrated_ratio),
            flush=True)

    # Exposure-relative IMAGE_QA scores are intentionally computed separately
    # for every H5/exposure; observations from different exposures are never
    # pooled into the robust center or MAD.
    for h5_name in sorted(set(row["H5"] for row in rows)):
        for exposure in range(1, N_EXPOSURES + 1):
            selected = [row for row in rows
                        if row["H5"] == h5_name and row["exposure"] == exposure]
            for band in QA_BANDS:
                field = "image_fractional_%s" % band
                values = np.asarray([row[field] for row in selected], dtype=float)
                finite = np.isfinite(values)
                if np.count_nonzero(finite) < 3:
                    continue
                center = float(np.median(values[finite]))
                sigma = float(1.4826 * np.median(np.abs(values[finite] - center)))
                if not np.isfinite(sigma) or sigma == 0.0:
                    continue
                for row in selected:
                    if np.isfinite(row[field]):
                        row["image_z_%s" % band] = (
                            row[field] - center) / sigma
            for row in selected:
                z = [abs(row["image_z_%s" % band]) for band in QA_BANDS
                     if np.isfinite(row["image_z_%s" % band])]
                if z and max(z) >= QA_WARN_Z:
                    row["QA_STATUS"] = "WARN"
                    row["QA_REASON"] = (
                        "extreme robust IMAGE_QA z; review before rejection")

    figure_paths = _write_qa_figures(surname, rows)
    queue_rows = sorted(
        rows,
        key=lambda row: (
            max([abs(row["image_z_%s" % band]) for band in QA_BANDS
                 if np.isfinite(row["image_z_%s" % band])] or [0.0]),
            max([abs(row["image_fractional_%s" % band]) for band in QA_BANDS
                 if np.isfinite(row["image_fractional_%s" % band])] or [0.0]),
        ), reverse=True)
    qa_summary = {
        "unit": "H5 + exposure + SPECID + IFUSLOT + IFUID + IFU_CODE + AMP",
        "qa_min_blank": int(qa_min_blank),
        "image_qa_amplifier_exposures": int(sum(
            np.isfinite(row["image_fractional_ON"]) or
            np.isfinite(row["image_fractional_OFF"]) for row in rows)),
        "blank_qa_amplifier_exposures": int(sum(
            row["BLANK_QA_STATUS"] == "PASS" for row in rows)),
        "without_direct_blank_qa": int(sum(
            row["BLANK_QA_STATUS"] == "NO_DIRECT_QA" for row in rows)),
        "image_fractional_ON": _qa_distribution(
            [row["image_fractional_ON"] for row in rows]),
        "image_fractional_OFF": _qa_distribution(
            [row["image_fractional_OFF"] for row in rows]),
        "image_z_ON": _qa_distribution([row["image_z_ON"] for row in rows]),
        "image_z_OFF": _qa_distribution([row["image_z_OFF"] for row in rows]),
        "figure_paths": figure_paths,
        "shot_names": shot_names,
        "band_positions": "final cube WCS and ADR at exact ON/OFF effective wavelengths",
        "loo_reference": "all other H5/exposure shots; target shot absent",
        "automatic_rejection": False,
    }
    return rows, queue_rows, qa_summary


def _apply_qa_rejection(records, qa_rows, reject_keys):
    found = set()
    rejected_rows = 0
    for record in records:
        for group in record["groups"]:
            key = _qa_group_key(record["h5file"].name, group)
            if key not in reject_keys:
                continue
            found.add(key)
            indices = np.asarray(group["indices"], dtype=int)
            record["spectra"][indices] = np.nan
            record["errors"][indices] = np.nan
            rejected_rows += int(indices.size)
            qa_row = None
            for row in qa_rows:
                if _qa_row_key(row) == key:
                    qa_row = row
                    break
            if qa_row is None:
                raise RuntimeError("QA reject group has no final QA row: %s" %
                                   (key,))
            qa_row["rejected_from_cube"] = True
            qa_row["QA_STATUS"] = "REJECT"
            qa_row["QA_REASON"] = "explicit reviewed QA rejection file"
            if not (np.all(np.isnan(record["spectra"][indices])) and
                    np.all(np.isnan(record["errors"][indices]))):
                raise RuntimeError("QA rejection did not NaN all fiber rows: %s" %
                                   (key,))
    missing = reject_keys - found
    if missing:
        raise ValueError("QA reject file contains groups absent from selected inputs: %s" %
                         sorted(missing))
    return {
        "rejected_groups": int(len(found)),
        "rejected_fiber_rows": int(rejected_rows),
    }


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


# Latest established sparse LSF splat helper.  It deliberately shares the
# production Gaussian kernel and only retains selected amplifier-center pixels.
@njit(nogil=True, cache=True)
def _gaussian_splat_sparse_shot_xy(indices, xpos, ypos, fluxes,
                                   sample_lookup, x_origin, y_origin,
                                   nx, ny, sigma, radius, support_radius,
                                   area, flux_sum, weight_sum, support_map):
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
                sample_id = sample_lookup[py, px]
                if sample_id < 0:
                    continue
                weight = gx[ox + radius] * gy[oy + radius]
                flux_sum[sample_id] += weight * flux / area
                weight_sum[sample_id] += weight
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
                sample_id = sample_lookup[py, px]
                if sample_id >= 0 and dx * dx + dy * dy <= support_radius_sq:
                    support_map[sample_id] = 1


# -----------------------------------------------------------------------------
# Optional empirical LSF path, ported from the latest revised cube builder
# -----------------------------------------------------------------------------


def _h5_text(value):
    if isinstance(value, (bytes, np.bytes_)):
        return value.decode("utf-8", errors="replace").strip()
    return str(value).strip()


def _exp_text(value):
    if isinstance(value, (bytes, np.bytes_)):
        value = value.decode("utf-8", errors="replace")
    try:
        return "%g" % float(value)
    except (TypeError, ValueError):
        return str(value).strip()


def _verify_lsf_exposure_partition(info_exp, n_fibers, h5file, log=None,
                                   inferred_nexp=None):
    labels = [_exp_text(value) for value in np.asarray(info_exp)]
    unique_labels = sorted(set(labels))
    info_unpopulated = len(unique_labels) == 1 and unique_labels[0] in ("0", "")
    if info_unpopulated:
        if log is not None:
            log.warning(
                "LSF QA %s: Info.exp is unpopulated (%s); inferred nexp=%s "
                "from the established 112-fiber row partition.",
                op.basename(str(h5file)), unique_labels[0], inferred_nexp)
        return None
    blocks = []
    for block_start in range(0, n_fibers, N_FIBER_AMP):
        block = labels[block_start:block_start + N_FIBER_AMP]
        blocks.append(sorted(set(block)))
    observed = [values[0] if len(values) == 1 else "MIXED"
                for values in blocks]
    agree = (n_fibers % N_FIBER_AMP == 0 and
             all(len(values) == 1 for values in blocks) and
             all(value == str(k % N_EXPOSURES)
                 for k, value in enumerate(observed)))
    agree_one_based = (n_fibers % N_FIBER_AMP == 0 and
                       all(len(values) == 1 for values in blocks) and
                       all(value == str((k % N_EXPOSURES) + 1)
                           for k, value in enumerate(observed)))
    agree = agree or agree_one_based
    if log is not None:
        log.info("LSF QA %s: Info.exp partition agrees=%s",
                 op.basename(str(h5file)), bool(agree))
    return agree


def _build_lsf_amplifier_centers(h5file, h5table, log=None):
    """Build one median RA/Dec center per shot x IFUSLOT x amplifier."""
    if "Raw" not in h5table.root._v_children:
        raise ValueError("LSF requested for %s, but the H5 file has no Raw table "
                         "containing arcspectrum/wave." % h5file)
    info = h5table.root.Info.cols
    ra = np.asarray(info.ra[:], dtype=float)
    dec = np.asarray(info.dec[:], dtype=float)
    ifuslot = np.asarray(info.ifuslot[:])
    amp = np.asarray([_h5_text(value) for value in info.amp[:]])
    specid = np.asarray([_h5_text(value) for value in info.specid[:]])
    ifuid = np.asarray([_h5_text(value) for value in info.ifuid[:]])
    n_info = len(ra)
    n_fiber = int(h5table.root.Fibers.nrows)
    n_raw = int(h5table.root.Raw.nrows)
    if not (n_info == n_fiber == n_raw):
        raise ValueError("LSF row alignment failure for %s: Info=%d Fibers=%d Raw=%d"
                         % (h5file, n_info, n_fiber, n_raw))
    nslots = len(np.unique(ifuslot))
    inferred_nexp = (int(n_info / float(4 * N_FIBER_AMP * nslots))
                     if nslots > 0 and n_info % (4 * N_FIBER_AMP * nslots) == 0
                     else None)
    if "exp" in h5table.root.Info.colnames:
        _verify_lsf_exposure_partition(
            h5table.root.Info.cols.exp[:], n_info, h5file, log=log,
            inferred_nexp=inferred_nexp)
    records = []
    keys = sorted(set(zip(ifuslot.tolist(), amp.tolist())),
                  key=lambda item: (int(item[0]), item[1]))
    for slot, amplifier in keys:
        selected = (ifuslot == slot) & (amp == amplifier)
        ra_values = ra[selected]
        dec_values = dec[selected]
        if not np.any(np.isfinite(ra_values) & np.isfinite(dec_values)):
            continue
        spec_values = sorted(set(specid[selected].tolist()))
        ifu_values = sorted(set(ifuid[selected].tolist()))
        identity_ok = len(spec_values) == 1 and len(ifu_values) == 1
        if not identity_ok and log is not None:
            log.warning("LSF QA %s IFUSLOT=%s AMP=%s has inconsistent identity",
                        op.basename(str(h5file)), slot, amplifier)
        records.append({
            "shot": op.basename(str(h5file)),
            "specid": spec_values[0] if spec_values else "",
            "ifuslot": int(slot), "ifuid": ifu_values[0] if ifu_values else "",
            "amp": amplifier,
            "ra": float(np.nanmedian(ra_values)),
            "dec": float(np.nanmedian(dec_values)),
            "identity_ok": bool(identity_ok),
        })
    return records


def _map_lsf_centers_to_spaxels(records, tp, xg, yg):
    sample_lookup = np.full((len(yg), len(xg)), -1, dtype=np.int32)
    unique_positions = []
    position_ids = {}
    for record in records:
        xw, yw = tp.wcs_world2pix(record["ra"], record["dec"], 1)
        valid = np.isfinite(xw) and np.isfinite(yw)
        if valid:
            x_index = int(np.rint(xw)) - 1
            y_index = int(np.rint(yw)) - 1
            valid = (0 <= x_index < len(xg)) and (0 <= y_index < len(yg))
        if not valid:
            record["cube_x"] = record["cube_y"] = record["sample_id"] = -1
            continue
        key = (y_index, x_index)
        if key not in position_ids:
            position_ids[key] = len(unique_positions)
            unique_positions.append(key)
            sample_lookup[key] = position_ids[key]
        record["cube_x"] = x_index + 1
        record["cube_y"] = y_index + 1
        record["sample_id"] = position_ids[key]
    return sample_lookup, unique_positions


def _sparse_reconstruct_one_wave(Pos, values, xg, yg, sample_lookup,
                                 shot_indices, sigma=1.8 / 2.35):
    n_samples = int(sample_lookup.max()) + 1 if np.any(sample_lookup >= 0) else 0
    shot_values = np.full((len(shot_indices), n_samples), np.nan, dtype=np.float32)
    ncontrib = np.zeros(n_samples, dtype=np.uint8)
    area = np.pi * FIBER_RADIUS_ARCSEC ** 2
    for shot_number, indices in enumerate(shot_indices):
        flux_sum = np.zeros(n_samples, dtype=np.float32)
        weight_sum = np.zeros(n_samples, dtype=np.float32)
        support_map = np.zeros(n_samples, dtype=np.uint8)
        _gaussian_splat_sparse_shot_xy(
            np.asarray(indices, dtype=np.int64), Pos[:, 0], Pos[:, 1], values,
            sample_lookup, float(xg[0]), float(yg[0]), len(xg), len(yg),
            float(sigma), 3, int(np.ceil(2.0 * sigma)), float(area),
            flux_sum, weight_sum, support_map)
        supported = (support_map != 0) & (weight_sum > 0.0)
        if np.any(supported):
            shot_values[shot_number, supported] = (
                flux_sum[supported] / weight_sum[supported])
            ncontrib[supported] += 1
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        result = np.nanmedian(shot_values, axis=0).astype(np.float32)
    result[ncontrib < 2] = 0.0
    return result, ncontrib


def _sparse_reconstruct_arc(raarray, decarray, arc_rect, full_indices,
                            sample_lookup, xg, yg, tp, shot_indices):
    n_samples = int(sample_lookup.max()) + 1 if np.any(sample_lookup >= 0) else 0
    propagated = np.full((n_samples, len(full_indices)), np.nan, dtype=np.float32)
    sample_ncontrib = np.zeros((n_samples, len(full_indices)), dtype=np.uint8)
    for column, full_index in enumerate(full_indices):
        x, y = tp.wcs_world2pix(raarray[:, full_index],
                                decarray[:, full_index], 1)
        values, ncontrib = _sparse_reconstruct_one_wave(
            np.column_stack((x, y)), arc_rect[:, column], xg, yg,
            sample_lookup, shot_indices)
        values[ncontrib < 2] = np.nan
        propagated[:, column] = values
        sample_ncontrib[:, column] = ncontrib
    return propagated, sample_ncontrib


def _rectify_lsf_arcs(h5files, def_wave, lsf_def_wave,
                      lsf_wave_indices, specarray, log=None):
    arc_rect = np.full((specarray.shape[0], len(lsf_def_wave)), np.nan,
                       dtype=np.float32)
    offset = 0
    for h5file in h5files:
        with tables.open_file(h5file, mode="r") as table:
            n_info = int(table.root.Info.nrows)
            n_raw = int(table.root.Raw.nrows)
            if n_info != n_raw:
                raise ValueError("LSF row alignment failure for %s" % h5file)
            native_arc = table.root.Raw.cols.arcspectrum[:]
            native_wave = table.root.Raw.cols.wave[:]
            if native_arc.shape[0] != n_info or native_wave.shape[0] != n_info:
                raise ValueError("LSF Raw array row alignment failure for %s" % h5file)
            for row in range(n_info):
                compact = np.interp(lsf_def_wave, native_wave[row], native_arc[row],
                                     left=np.nan, right=np.nan)
                if row == 0:
                    full = np.interp(def_wave, native_wave[row], native_arc[row],
                                     left=np.nan, right=np.nan)
                    np.testing.assert_array_equal(compact, full[lsf_wave_indices])
                    if log is not None:
                        log.info("LSF rectification validation %s: exact compact/full agreement",
                                 op.basename(str(h5file)))
                arc_rect[offset + row] = compact.astype(np.float32)
        offset += n_info
    if offset != specarray.shape[0]:
        raise ValueError("LSF row alignment failure across files: Raw=%d SCI=%d" %
                         (offset, specarray.shape[0]))
    return arc_rect


def _normalize_lsf_arc_window(arc_rect, lsf_def_wave, reference):
    """Subtract outer-window backgrounds and normalize each fiber line flux."""
    window = np.abs(lsf_def_wave - reference) <= ARC_WINDOW_ANGSTROM
    left = window & (lsf_def_wave <= reference - 10.0)
    right = window & (lsf_def_wave >= reference + 10.0)
    local = np.where(window)[0]
    normalized = np.full_like(arc_rect, np.nan, dtype=np.float32)
    if not local.size:
        return normalized, np.zeros(arc_rect.shape[0], dtype=bool), np.full(arc_rect.shape[0], np.nan)
    left_bg = (np.nanmedian(arc_rect[:, left], axis=1)
               if np.any(left) else np.full(arc_rect.shape[0], np.nan))
    right_bg = (np.nanmedian(arc_rect[:, right], axis=1)
                if np.any(right) else np.full(arc_rect.shape[0], np.nan))
    background = np.nanmedian(np.column_stack((left_bg, right_bg)), axis=1)
    line = arc_rect[:, local].astype(float) - background[:, None]
    signal = np.nansum(line, axis=1) * 2.0
    valid = (np.isfinite(left_bg) & np.isfinite(right_bg) &
             np.isfinite(background) & np.isfinite(signal) & (signal > 0.0))
    for row in np.where(valid)[0]:
        normalized[row, local] = ((arc_rect[row, local].astype(float) - background[row]) /
                                  signal[row]).astype(np.float32)
    return normalized, valid, signal


def _native_arc_clipping_diagnostic(arc_rect, lsf_def_wave, reference):
    window = np.abs(lsf_def_wave - reference) <= ARC_WINDOW_ANGSTROM
    local = np.where(window)[0]
    if not local.size:
        return {"finite": 0, "flat_top": 0, "exact_plateau": 0, "suspicious": 0}
    left = window & (lsf_def_wave <= reference - 10.0)
    right = window & (lsf_def_wave >= reference + 10.0)
    left_bg = np.nanmedian(arc_rect[:, left], axis=1)
    right_bg = np.nanmedian(arc_rect[:, right], axis=1)
    background = np.nanmedian(np.column_stack((left_bg, right_bg)), axis=1)
    values = arc_rect[:, local].astype(float) - background[:, None]
    finite = np.isfinite(values)
    peak = np.full(values.shape[0], np.nan)
    for row in range(values.shape[0]):
        if np.any(finite[row]):
            peak[row] = np.nanmax(values[row])
    finite_rows = np.isfinite(peak) & (peak > 0.0)
    flat_top = np.zeros(values.shape[0], dtype=bool)
    exact_plateau = np.zeros(values.shape[0], dtype=bool)
    for row in np.where(finite_rows)[0]:
        top = finite[row] & (values[row] >= 0.995 * peak[row])
        flat_top[row] = np.any(top[:-1] & top[1:])
        exact_plateau[row] = np.any(
            top[:-1] & top[1:] &
            (np.abs(values[row, :-1] - values[row, 1:]) <=
             max(1.e-6, 1.e-6 * abs(peak[row]))))
    suspicious = flat_top | exact_plateau
    return {"finite": int(np.count_nonzero(finite_rows)),
            "flat_top": int(np.count_nonzero(flat_top)),
            "exact_plateau": int(np.count_nonzero(exact_plateau)),
            "suspicious": int(np.count_nonzero(suspicious))}


def _measure_direct_fwhm(wave, profile, reference):
    """Measure FWHM by the established PCHIP/direct half-height method."""
    wave = np.asarray(wave, dtype=float)
    profile = np.asarray(profile, dtype=float)
    finite = np.isfinite(wave) & np.isfinite(profile)
    core = np.isfinite(wave) & (np.abs(wave - reference) <= 6.0)
    invalid = {"valid": False, "center": np.nan, "fwhm": np.nan, "phase": np.nan}
    if np.count_nonzero(core) < 2 or not np.all(finite[core]):
        return {**invalid, "status": "UNSUPPORTED_CORE"}
    if np.count_nonzero(finite) < 5:
        return {**invalid, "status": "TOO_FEW_SAMPLES"}
    x = wave[finite]
    y = profile[finite]
    order = np.argsort(x)
    x, y = x[order], y[order]
    left_background = np.nanmedian(y[x <= reference - 10.0])
    right_background = np.nanmedian(y[x >= reference + 10.0])
    if not np.isfinite(left_background) or not np.isfinite(right_background):
        return {**invalid, "status": "NO_BACKGROUND"}
    y = y - float(np.nanmedian([left_background, right_background]))
    search = (x >= reference - 5.0) & (x <= reference + 5.0)
    if np.count_nonzero(search) < 2:
        return {**invalid, "status": "NO_PEAK_NEAR_LINE"}
    try:
        interpolator = PchipInterpolator(x, y, extrapolate=False)
    except (ValueError, TypeError):
        return {**invalid, "status": "INTERPOLATION_FAILED"}
    sampled_search = np.where(search)[0]
    imax = int(sampled_search[np.argmax(y[sampled_search])])
    if (imax <= 0 or imax >= len(x) - 1 or
            not np.all(np.isfinite([x[imax - 1], x[imax + 1],
                                    y[imax - 1], y[imax + 1]]))):
        return {**invalid, "status": "INVALID_PEAK_INTERPOLATION"}
    ym, y0, yp = y[imax - 1], y[imax], y[imax + 1]
    dx_left = x[imax] - x[imax - 1]
    dx_right = x[imax + 1] - x[imax]
    if (dx_left <= 0.0 or not np.isclose(dx_left, dx_right,
                                          rtol=1.e-7, atol=1.e-10)):
        return {**invalid, "status": "INVALID_PEAK_INTERPOLATION"}
    denom = ym - 2.0 * y0 + yp
    tolerance = 100.0 * np.finfo(float).eps * max(1.0, abs(ym), abs(y0), abs(yp))
    if not np.isfinite(denom) or denom >= 0.0 or abs(denom) <= tolerance:
        return {**invalid, "status": "INVALID_PEAK_INTERPOLATION"}
    delta_pix = 0.5 * (ym - yp) / denom
    if not np.isfinite(delta_pix) or abs(delta_pix) > 1.0:
        return {**invalid, "status": "INVALID_PEAK_INTERPOLATION"}
    dx = 0.5 * (dx_left + dx_right)
    center = float(x[imax] + delta_pix * dx)
    peak_height = float(y0 - (yp - ym) ** 2 / (8.0 * denom))
    if not np.isfinite(peak_height) or peak_height <= 0.0:
        return {**invalid, "status": "NONPOSITIVE_LINE"}
    dense = np.linspace(x[0], x[-1],
                        max(401, int(np.ceil((x[-1] - x[0]) * 100)) + 1))
    dense_y = np.asarray(interpolator(dense), dtype=float)
    search_dense = (dense >= reference - 5.0) & (dense <= reference + 5.0)
    if not np.any(search_dense & np.isfinite(dense_y)):
        return {**invalid, "status": "NO_PEAK_NEAR_LINE"}
    peak_candidates = np.where(
        search_dense & np.isfinite(dense_y) &
        (dense_y >= np.roll(dense_y, 1)) &
        (dense_y >= np.roll(dense_y, -1)))[0]
    peak_candidates = peak_candidates[(peak_candidates > 0) &
                                      (peak_candidates < len(dense) - 1)]
    strong = peak_candidates[dense_y[peak_candidates] >= 0.80 * peak_height]
    if strong.size > 1 and any(abs(dense[idx] - center) >= 1.0 for idx in strong):
        return {**invalid, "status": "AMBIGUOUS_BLEND"}
    half = 0.5 * peak_height
    peak_index = int(np.argmin(np.abs(dense - center)))

    def crossing(direction):
        indices = (range(peak_index - 1, -1, -1) if direction < 0
                   else range(peak_index, len(dense) - 1))
        for index in indices:
            j = index + 1 if direction > 0 else index
            k = index if direction > 0 else index + 1
            y_left, y_right = dense_y[j] - half, dense_y[k] - half
            if np.isfinite(y_left) and np.isfinite(y_right) and y_left * y_right <= 0.0:
                lo, hi = min(dense[j], dense[k]), max(dense[j], dense[k])
                flo = float(interpolator(lo) - half)
                for _ in range(45):
                    mid = 0.5 * (lo + hi)
                    fmid = float(interpolator(mid) - half)
                    if flo * fmid <= 0.0:
                        hi = mid
                    else:
                        lo, flo = mid, fmid
                return 0.5 * (lo + hi)
        return None

    left_cross, right_cross = crossing(-1), crossing(1)
    if left_cross is None:
        return {**invalid, "status": "NO_LEFT_CROSSING"}
    if right_cross is None:
        return {**invalid, "status": "NO_RIGHT_CROSSING"}
    if left_cross <= x[0] + 1.e-6 or right_cross >= x[-1] - 1.e-6:
        return {**invalid, "status": "CROSSING_AT_BOUNDARY"}
    center = float(dense[peak_index])
    nearest_grid = 3470.0 + 2.0 * np.rint((center - 3470.0) / 2.0)
    return {"valid": True, "status": "OK", "center": center,
            "fwhm": float(right_cross - left_cross),
            "phase": float(center - nearest_grid)}


def _lsf_spatial_header(tp, n, pixel_scale):
    spatial = tp.to_header()
    scale = pixel_scale / 3600.0
    spatial["WCSAXES"] = 2
    spatial["CD1_1"] = -scale
    spatial["CD1_2"] = 0.0
    spatial["CD2_1"] = 0.0
    spatial["CD2_2"] = scale
    for key in ("CDELT1", "CDELT2", "CDELT3", "CRPIX3", "CRVAL3",
                "CTYPE3", "CUNIT3", "SPECSYS"):
        if key in spatial:
            del spatial[key]
    spatial["CRPIX1"] = (n + 1) / 2.0
    spatial["CRPIX2"] = (n + 1) / 2.0
    spatial["CTYPE1"] = "RA---TAN"
    spatial["CTYPE2"] = "DEC--TAN"
    spatial["CUNIT1"] = "deg"
    spatial["CUNIT2"] = "deg"
    return spatial


def _phase_summary(phases, fwhm):
    finite = np.isfinite(phases) & np.isfinite(fwhm)
    if np.count_nonzero(finite) < 10:
        return np.nan, np.nan, False
    x, y = phases[finite], fwhm[finite]
    edges = np.linspace(-1.0, 1.0, 9)
    medians = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        selected = (x >= lo) & (x < hi if hi < edges[-1] else x <= hi)
        if np.count_nonzero(selected) >= 3:
            medians.append(float(np.nanmedian(y[selected])))
    phase_range = (float(np.nanmax(medians) - np.nanmin(medians))
                   if len(medians) >= 2 else np.nan)
    correlation = (float(np.corrcoef(x, y)[0, 1])
                   if np.std(x) > 0.0 and np.std(y) > 0.0 else np.nan)
    median = float(np.nanmedian(y))
    evidence = bool(np.isfinite(phase_range) and np.isfinite(median) and
                    median != 0.0 and phase_range > 0.03 * abs(median))
    return phase_range, correlation, evidence


def _run_lsf_measurement(h5files, surname, def_wave, raarray, decarray,
                         specarray, shot_indices, shot_names, tp, xg, yg,
                         xgrid, ygrid, cube, ncontribcube, pixel_scale,
                         log, records=None):
    """Run the established sparse empirical effective-resolution experiment."""
    if records is None:
        records = []
        for h5file in h5files:
            with tables.open_file(h5file, mode="r") as table:
                records.extend(_build_lsf_amplifier_centers(h5file, table, log=log))
    sample_lookup, unique_positions = _map_lsf_centers_to_spaxels(records, tp, xg, yg)
    log.info("LSF amplifier-center sampling: %d centers, %d unique in-cube spaxels",
             len(records), len(unique_positions))
    if not unique_positions:
        raise RuntimeError("LSF requested but no amplifier centers map into the cube")

    line_masks = [np.abs(def_wave - reference) <= ARC_WINDOW_ANGSTROM
                  for reference in REFERENCE_ARC_WAVELENGTHS]
    lsf_wave_indices = np.unique(np.concatenate(
        [np.where(mask)[0] for mask in line_masks])).astype(np.int64)
    lsf_def_wave = def_wave[lsf_wave_indices]
    arc_rect = _rectify_lsf_arcs(
        h5files, def_wave, lsf_def_wave, lsf_wave_indices, specarray, log=log)

    normalized_by_line = []
    for reference in REFERENCE_ARC_WAVELENGTHS:
        clipping = _native_arc_clipping_diagnostic(arc_rect, lsf_def_wave, reference)
        log.info("LSF native arc diagnostic %.3f A: finite=%d suspicious=%d",
                 reference, clipping["finite"], clipping["suspicious"])
        normalized, _, _ = _normalize_lsf_arc_window(
            arc_rect, lsf_def_wave, reference)
        normalized_by_line.append(normalized)

    fwhm_by_line, center_by_line, phase_by_line = [], [], []
    status_by_line, ncontrib_by_line, summary = [], [], []
    for line_number, reference in enumerate(REFERENCE_ARC_WAVELENGTHS):
        local = np.abs(lsf_def_wave - reference) <= ARC_WINDOW_ANGSTROM
        local_wave = lsf_def_wave[local]
        line_indices = lsf_wave_indices[local]
        normalized_arc = normalized_by_line[line_number][:, local].copy()
        # Final SCI availability is imposed after line normalization.  This
        # is where unsupported-M, hardware, and explicit QA-rejected rows
        # disappear from the propagated LSF exactly as they do from SCI.
        for column, full_index in enumerate(line_indices):
            normalized_arc[~np.isfinite(specarray[:, full_index]), column] = np.nan
        propagated, sample_ncontrib = _sparse_reconstruct_arc(
            raarray, decarray, normalized_arc, line_indices,
            sample_lookup, xg, yg, tp, shot_indices)
        fwhm = np.full(len(unique_positions), np.nan, dtype=float)
        center = np.full(len(unique_positions), np.nan, dtype=float)
        phase = np.full(len(unique_positions), np.nan, dtype=float)
        status = np.full(len(unique_positions), "NO_PROFILE", dtype="U32")
        nnear = np.full(len(unique_positions), -1, dtype=np.int16)
        for sample_id in range(len(unique_positions)):
            profile = propagated[sample_id]
            support = sample_ncontrib[sample_id]
            if np.any(np.isfinite(profile)):
                nnear[sample_id] = int(np.nanmedian(support[np.isfinite(profile)]))
            core = np.abs(local_wave - reference) <= 6.0
            if (np.count_nonzero(core) >= 2 and np.all(np.isfinite(profile[core])) and
                    np.all(support[core] >= 2)):
                result = _measure_direct_fwhm(local_wave, profile, reference)
            else:
                result = {"valid": False, "status": "INSUFFICIENT_SUPPORT",
                          "center": np.nan, "fwhm": np.nan, "phase": np.nan}
            status[sample_id] = result["status"]
            if result["valid"]:
                center[sample_id] = result["center"]
                fwhm[sample_id] = result["fwhm"]
                phase[sample_id] = result["phase"]
        fwhm_by_line.append(fwhm)
        center_by_line.append(center)
        phase_by_line.append(phase)
        status_by_line.append(status)
        ncontrib_by_line.append(nnear)
        good = np.isfinite(fwhm)
        values = fwhm[good]
        if values.size:
            p16, p50, p84 = np.nanpercentile(values, [16, 50, 84])
            scatter = 1.4826 * np.nanmedian(np.abs(values - p50))
        else:
            p16 = p50 = p84 = scatter = np.nan
        phase_range, phase_corr, phase_evidence = _phase_summary(phase, fwhm)
        summary.append((reference, p16, p50, p84, scatter,
                        phase_range, phase_corr, phase_evidence))
        log.info("LSF %.3f A: valid=%d/%d; usable_for_lsf=%s; blend=%s",
                 reference, int(values.size), len(unique_positions),
                 bool(ARC_USABLE_FOR_LSF[line_number]), ARC_BLEND_STATES[line_number])

    table = Table()
    table["shot"] = np.array([record["shot"] for record in records
                               for _ in REFERENCE_ARC_WAVELENGTHS], dtype="U80")
    table["specid"] = np.array([record["specid"] for record in records
                                 for _ in REFERENCE_ARC_WAVELENGTHS], dtype="U16")
    table["ifuslot"] = np.array([record["ifuslot"] for record in records
                                  for _ in REFERENCE_ARC_WAVELENGTHS], dtype=np.int32)
    table["ifuid"] = np.array([record["ifuid"] for record in records
                                for _ in REFERENCE_ARC_WAVELENGTHS], dtype="U16")
    table["amp"] = np.array([record["amp"] for record in records
                              for _ in REFERENCE_ARC_WAVELENGTHS], dtype="U8")
    table["identity_ok"] = np.array([record["identity_ok"] for record in records
                                      for _ in REFERENCE_ARC_WAVELENGTHS], dtype=bool)
    table["ra"] = np.array([record["ra"] for record in records
                             for _ in REFERENCE_ARC_WAVELENGTHS], dtype=float)
    table["dec"] = np.array([record["dec"] for record in records
                              for _ in REFERENCE_ARC_WAVELENGTHS], dtype=float)
    table["cube_x"] = np.array([record["cube_x"] for record in records
                                 for _ in REFERENCE_ARC_WAVELENGTHS], dtype=np.int32)
    table["cube_y"] = np.array([record["cube_y"] for record in records
                                 for _ in REFERENCE_ARC_WAVELENGTHS], dtype=np.int32)
    table["sample_id"] = np.array([record["sample_id"] for record in records
                                    for _ in REFERENCE_ARC_WAVELENGTHS], dtype=np.int32)
    table["reference_wavelength"] = np.tile(
        REFERENCE_ARC_WAVELENGTHS, len(records))
    table["measured_center"] = np.array([
        center_by_line[j][record["sample_id"]] if record["sample_id"] >= 0 else np.nan
        for record in records for j in range(len(REFERENCE_ARC_WAVELENGTHS))], dtype=float)
    table["fwhm"] = np.array([
        fwhm_by_line[j][record["sample_id"]] if record["sample_id"] >= 0 else np.nan
        for record in records for j in range(len(REFERENCE_ARC_WAVELENGTHS))], dtype=float)
    table["grid_phase"] = np.array([
        phase_by_line[j][record["sample_id"]] if record["sample_id"] >= 0 else np.nan
        for record in records for j in range(len(REFERENCE_ARC_WAVELENGTHS))], dtype=float)
    table["ncontrib"] = np.array([
        ncontrib_by_line[j][record["sample_id"]] if record["sample_id"] >= 0 else -1
        for record in records for j in range(len(REFERENCE_ARC_WAVELENGTHS))], dtype=np.int16)
    table["valid"] = np.array([
        bool(record["sample_id"] >= 0 and
             np.isfinite(fwhm_by_line[j][record["sample_id"]]))
        for record in records for j in range(len(REFERENCE_ARC_WAVELENGTHS))], dtype=bool)
    table["usable_for_lsf"] = np.array([
        bool(ARC_USABLE_FOR_LSF[j] and record["sample_id"] >= 0 and
             np.isfinite(fwhm_by_line[j][record["sample_id"]]))
        for record in records for j in range(len(REFERENCE_ARC_WAVELENGTHS))], dtype=bool)
    table["status"] = np.array([
        status_by_line[j][record["sample_id"]] if record["sample_id"] >= 0
        else "OUTSIDE_CUBE"
        for record in records for j in range(len(REFERENCE_ARC_WAVELENGTHS))], dtype="U32")
    table["lab_blend"] = np.array([
        ARC_BLEND_STATES[j] == "KNOWN_BLEND" for _ in records
        for j in range(len(REFERENCE_ARC_WAVELENGTHS))], dtype=bool)
    table["lab_blend_state"] = np.array([
        ARC_BLEND_STATES[j] for _ in records
        for j in range(len(REFERENCE_ARC_WAVELENGTHS))], dtype="U16")
    table["blend_note"] = np.array([
        ARC_BLEND_NOTES[float(REFERENCE_ARC_WAVELENGTHS[j])] for _ in records
        for j in range(len(REFERENCE_ARC_WAVELENGTHS))], dtype="U96")
    sample_path = Path("%s_lsf_samples.fits" % surname)
    # Older TACC Astropy releases accept overwrite in Table.write but drop it
    # before calling the FITS writer.  Write the HDU directly for compatibility.
    fits.table_to_hdu(table).writeto(sample_path, overwrite=True)

    spatial_header = _lsf_spatial_header(tp, len(xg), pixel_scale)
    public_line_indices = np.where(ARC_USABLE_FOR_LSF)[0]
    for line_number in public_line_indices:
        reference = REFERENCE_ARC_WAVELENGTHS[line_number]
        valid = np.isfinite(fwhm_by_line[line_number])
        image = np.full(xgrid.shape, np.nan, dtype=np.float32)
        if np.any(valid):
            points = np.array([(unique_positions[sid][1], unique_positions[sid][0])
                               for sid in np.where(valid)[0]], dtype=float)
            image[:, :] = griddata(
                points, fwhm_by_line[line_number][valid],
                (xgrid - 1.0, ygrid - 1.0), method="nearest").astype(np.float32)
        science_index = int(np.argmin(np.abs(def_wave - reference)))
        image[ncontribcube[science_index] < 2] = np.nan
        header = spatial_header.copy()
        header["REFWAVE"] = (float(reference), "laboratory air wavelength [Angstrom]")
        header["WAVECONV"] = ("AIR", "Hg/Cd reference wavelength convention")
        header["METHOD"] = ("DIRECT/NONPARAMETRIC", "PCHIP peak and half-height crossings")
        header["INTERP"] = ("NEAREST", "nearest-neighbor spatial sample interpolation")
        header["LSFMODEL"] = ("NONE", "profile shape not specified")
        header["BUNIT"] = "Angstrom"
        header.add_history("Effective cube FWHM from propagated master arcs.")
        fits.PrimaryHDU(
            image, header=header).writeto(
                "%s_lsf_fwhm_%04d.fits" % (surname, int(np.floor(reference + 0.5))),
                overwrite=True)

    fig, axes = plt.subplots(2, 2, figsize=(13, 10), constrained_layout=True)
    amp_colors = {"LL": "tab:blue", "LU": "tab:orange",
                  "RL": "tab:green", "RU": "tab:red"}
    for amplifier, color in amp_colors.items():
        amplifier_records = [record for record in records
                             if record["sample_id"] >= 0 and record["amp"] == amplifier]
        axes[0, 0].scatter([record["cube_x"] for record in amplifier_records],
                           [record["cube_y"] for record in amplifier_records],
                           s=4, alpha=.45, color=color, label=amplifier)
    axes[0, 0].set_title("Amplifier-center sample spaxels")
    axes[0, 0].set_aspect("equal", adjustable="box")
    axes[0, 0].legend(title="amplifier", fontsize=8, markerscale=1.5)
    if public_line_indices.size:
        median_sample = np.nanmedian(
            np.vstack([fwhm_by_line[i] for i in public_line_indices]), axis=0)
        good_sample = np.isfinite(median_sample)
        if np.any(good_sample):
            axes[0, 1].scatter(
                [unique_positions[sid][1] + 1 for sid in np.where(good_sample)[0]],
                [unique_positions[sid][0] + 1 for sid in np.where(good_sample)[0]],
                c=median_sample[good_sample], s=5, cmap="viridis")
    axes[0, 1].set_title("Median valid FWHM at sample spaxels")
    for line_number, reference in enumerate(REFERENCE_ARC_WAVELENGTHS):
        values = fwhm_by_line[line_number]
        blend = ARC_BLEND_STATES[line_number] == "KNOWN_BLEND"
        axes[1, 0].plot(reference,
                         np.nanmedian(values) if np.any(np.isfinite(values)) else np.nan,
                         "x" if blend else "o",
                         color="tab:red" if blend else "tab:blue",
                         label="%.0f blend QA-only" % reference if blend else None)
        axes[1, 1].scatter(phase_by_line[line_number], values, s=3, alpha=.25,
                           label=("%.0f blend QA-only" % reference)
                           if blend else "%.0f" % reference)
    axes[1, 0].set_title("Median FWHM versus reference wavelength (blend QA-only)")
    axes[1, 0].set_xlabel("reference wavelength [A]")
    axes[1, 0].set_ylabel("FWHM [A]")
    axes[1, 0].legend(fontsize=7, ncol=2)
    axes[1, 1].set_title("FWHM versus 2-Angstrom grid phase (blend QA-only)")
    axes[1, 1].set_xlabel("center minus nearest grid bin [A]")
    axes[1, 1].set_ylabel("FWHM [A]")
    axes[1, 1].legend(fontsize=7, ncol=3)
    qa_path = Path("%s_lsf_qa.png" % surname)
    fig.savefig(qa_path, dpi=180)
    plt.close(fig)

    usable_counts = {
        "%.3f" % reference: int(np.count_nonzero(
            table["usable_for_lsf"][np.isclose(
                np.asarray(table["reference_wavelength"], dtype=float), reference)]))
        for reference in REFERENCE_ARC_WAVELENGTHS
    }
    return table, summary, len(unique_positions), usable_counts


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
            "Final amplifier QA was evaluated after final exposure sky subtraction.",
            "IMAGE_QA used leave-one-shot-out ON/OFF source-image consistency.",
            "Real source flux was compared to the source image and was not treated as a residual.",
            "Two median-one LOO SB exposure-scale iterations applied.",
            "Final exposure correction = 1/(g1_rel*g2_rel).",
            "Final sky scaled with exposure correction; sky not re-estimated.",
            "g3 measured as closure QA only and not applied.",
            "Reviewed amplifier QA rejection applied before cube reconstruction: %s." % (
                "yes" if summary["qa"]["rejection"]["rejected_groups"] else "no"),
            "Explicit QA rejection file: %s." % (
                "used" if summary["qa"]["reject_file_used"] else "not used"),
            "Rejected amplifier/exposure groups=%d; rejected fiber rows=%d." % (
                summary["qa"]["rejection"]["rejected_groups"],
                summary["qa"]["rejection"]["rejected_fiber_rows"]),
            "--make-lsf %s." % ("enabled" if summary["lsf"]["enabled"] else "disabled"),
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
        "--on-filter", default=None,
        help="validated M101 ON filter; defaults to the established project provenance")
    parser.add_argument(
        "--off-filter", default=None,
        help="validated M101 OFF filter; defaults to the established project provenance")
    parser.add_argument(
        "--qa-min-blank", type=int, default=5,
        help="minimum final blank fibers for direct BLANK_QA (default: 5)")
    parser.add_argument(
        "--qa-report-only", action="store_true",
        help="write final QA and review artifacts without automatic rejection")
    parser.add_argument(
        "--qa-reject-file", default=None,
        help="reviewed CSV of explicit H5/exposure/physical-amplifier rejections")
    parser.add_argument(
        "--make-lsf", action="store_true",
        help="Also propagate master-arc features sparsely and write empirical FWHM products")
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
    if args.qa_min_blank < 1:
        raise SystemExit("--qa-min-blank must be positive")

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

    on_filter = args.on_filter or frozen_parent._default_input("on_filter")
    off_filter = args.off_filter or frozen_parent._default_input("off_filter")
    if not on_filter or not Path(on_filter).exists():
        raise FileNotFoundError("ON filter is unavailable; pass --on-filter")
    if not off_filter or not Path(off_filter).exists():
        raise FileNotFoundError("OFF filter is unavailable; pass --off-filter")
    if tuple(cache_manifest.get("band_order", ())) != tuple(frozen_parent.BANDS):
        raise RuntimeError("band cache does not contain the established seven-band order")
    on_filter_values = validated_m101.read_filter(on_filter)
    off_filter_values = validated_m101.read_filter(off_filter)
    band_responses, null_provenance = native._band_responses(
        on_filter_values, off_filter_values)
    effective_bands = frozen_parent._band_effective_wavelengths(band_responses)
    expected_effective = 4500.0 + 1000.0 * np.asarray(
        cache_manifest["band_effective_x_lambda"], dtype=float)
    if not np.allclose(effective_bands, expected_effective,
                       rtol=0.0, atol=1.0e-8):
        raise RuntimeError("filter-derived band responses disagree with the frozen band cache")
    qa_responses = np.asarray(band_responses[-2:], dtype=float)
    reject_keys, reject_path = _load_qa_reject_file(args.qa_reject_file)
    print("  ON filter           : %s" % Path(on_filter).expanduser().resolve(), flush=True)
    print("  OFF filter          : %s" % Path(off_filter).expanduser().resolve(), flush=True)
    print("  final QA            : %s; explicit reject groups=%d" %
          ("report-only" if args.qa_report_only else "enabled", len(reject_keys)),
          flush=True)
    print("  make-lsf            : %s" %
          ("enabled" if args.make_lsf else "disabled"), flush=True)

    records = []
    calibration_rows = []
    sky_rows = []
    total_fibers = 0
    calibration_seconds = 0.0
    for h5file in h5files:
        one_started = time.perf_counter()
        (spectra, errors, ra, dec, labels, surveys, calibration_summary, one_sky,
         sky_by_exposure, groups, q_rows, hardware_bad, corrected_total) = (
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
            "original_spectra": spectra.astype(np.float32).copy(),
            "errors": errors.astype(np.float32),
            "ra": ra,
            "dec": dec,
            "labels": labels,
            "surveys": surveys,
            "sky_by_exposure": {
                int(exposure): np.asarray(sky, dtype=np.float32)
                for exposure, sky in sky_by_exposure.items()
            },
            "groups": groups,
            "q_rows": q_rows,
            "hardware_bad": hardware_bad,
            "corrected_total": np.asarray(corrected_total, dtype=np.float32),
            "blank_mask": np.asarray(blank_masks[h5file.name], dtype=bool),
            "calibration_summary": calibration_summary,
        })
        print("[cube] calibrated %s in %.3f s; supported groups=%d; masked no-M=%d" %
              (h5file.name, time.perf_counter() - one_started,
               calibration_summary["supported_groups"],
               calibration_summary["masked_missing_frozen_M_groups"]), flush=True)

    exposure_scale_started = time.perf_counter()
    exposure_scale_rows = _apply_two_pass_exposure_scale(
        records, band_responses, args.pixel_scale)
    exposure_scale_path = Path("%s_exposure_scale.csv" % args.surname)
    exposure_scale_fields = [
        "H5", "exposure", "chronological_exposure_index",
        "g1_raw", "sigma_g1", "center1", "g1_rel",
        "g2_raw", "sigma_g2", "center2", "g2_rel",
        "cumulative_relative_scale", "cumulative_flux_correction",
        "g3_raw", "sigma_g3", "g3_minus_one",
        "g1_ON", "g1_OFF", "g1_N_SB_ON", "g1_N_SB_OFF", "g1_N_SB_BOTH",
        "g2_ON", "g2_OFF", "g2_N_SB_ON", "g2_N_SB_OFF", "g2_N_SB_BOTH",
        "g1_residual_scatter_SB_joint", "g2_residual_scatter_SB_joint",
        "g1_SB_source_leverage", "g2_SB_source_leverage",
        "g1_SB_max_predictor_leverage_fraction",
        "g2_SB_max_predictor_leverage_fraction",
        "g1_SB_N_effective_leverage", "g2_SB_N_effective_leverage",
        "validation_max_abs_direct_error",
        "validation_max_abs_identity_error",
    ]
    _write_rows(exposure_scale_path, exposure_scale_rows,
                fields=exposure_scale_fields)
    print("[cube] two-pass exposure scale: %d rows in %.3f s" %
          (len(exposure_scale_rows), time.perf_counter() - exposure_scale_started),
          flush=True)

    _qa_ra0, _qa_dec0, _qa_size_arcsec, _qa_n, qa_xg, qa_yg, _qa_xgrid, _qa_ygrid, qa_tp = \
        _derive_final_qa_geometry(records, args.pixel_scale)
    print("[cube] final QA image grid: %d x %d at %.6g arcsec/pixel" %
          (_qa_n, _qa_n, args.pixel_scale), flush=True)
    qa_started = time.perf_counter()
    qa_rows, qa_queue_rows, qa_summary = _evaluate_amplifier_qa(
        records, qa_responses, effective_bands, qa_tp, qa_xg, qa_yg,
        args.pixel_scale, args.qa_min_blank, args.surname)
    scale_by_key = {(row["H5"], int(row["exposure"])): row
                    for row in exposure_scale_rows}
    exposure_scale_qa_fields = (
        "g1_raw", "sigma_g1", "center1", "g1_rel",
        "g2_raw", "sigma_g2", "center2", "g2_rel",
        "cumulative_relative_scale", "cumulative_flux_correction",
        "g3_raw", "sigma_g3", "g3_minus_one")
    for row in qa_rows + qa_queue_rows:
        scale = scale_by_key[(row["H5"], int(row["exposure"]))]
        for field_name in exposure_scale_qa_fields:
            row[field_name] = scale[field_name]
    qa_rejection = _apply_qa_rejection(records, qa_rows, reject_keys)
    qa_summary["reject_file_used"] = bool(reject_path is not None)
    qa_summary["reject_file"] = (_file_identity(reject_path)
                                 if reject_path is not None else None)
    qa_summary["rejection"] = qa_rejection
    qa_summary["qa_report_only"] = bool(args.qa_report_only)
    qa_summary["row_count"] = int(len(qa_rows))
    qa_summary["warn_count_after_rejection"] = int(sum(
        row["QA_STATUS"] == "WARN" for row in qa_rows))
    qa_summary["on_filter"] = _file_identity(on_filter)
    qa_summary["off_filter"] = _file_identity(off_filter)
    qa_path = Path("%s_amplifier_qa.csv" % args.surname)
    qa_queue_path = Path("%s_amplifier_qa_queue.csv" % args.surname)
    _write_rows(qa_path, qa_rows)
    _write_rows(qa_queue_path, qa_queue_rows)
    print("[cube] final amplifier QA: %d rows; IMAGE_QA=%d; BLANK_QA=%d; "
          "NO_DIRECT_QA=%d; rejected groups=%d; rejected rows=%d in %.3f s" %
          (len(qa_rows), qa_summary["image_qa_amplifier_exposures"],
           qa_summary["blank_qa_amplifier_exposures"],
           qa_summary["without_direct_blank_qa"],
           qa_rejection["rejected_groups"], qa_rejection["rejected_fiber_rows"],
           time.perf_counter() - qa_started), flush=True)

    ra0, dec0, size_arcsec, n, xg, yg, xgrid, ygrid, tp = _parse_image_geometry(
        args.image_center_size, args.pixel_scale)
    print("[cube] image grid: %d x %d at %.6g arcsec/pixel" %
          (n, n, args.pixel_scale), flush=True)

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

    lsf_summary = {
        "enabled": bool(args.make_lsf),
        "sample_fits": None,
        "public_fwhm_fits": [],
        "qa_png": None,
        "usable_for_lsf_sample_counts": {},
        "known_blends_qa_only": [3610.508, 3650.153],
    }
    if args.make_lsf:
        logging.basicConfig(level=logging.INFO, format="[cube] %(message)s")
        lsf_log = logging.getLogger("make_m101_cube_current_model.lsf")
        _, _, lsf_unique_positions, usable_counts = _run_lsf_measurement(
            h5files, args.surname, DEF_WAVE, raarray, decarray, specarray,
            shot_indices, shot_names, tp, xg, yg, xgrid, ygrid, cube,
            ncontribcube, args.pixel_scale, lsf_log)
        lsf_summary.update({
            "sample_fits": str(Path("%s_lsf_samples.fits" % args.surname).resolve()),
            "public_fwhm_fits": [str(Path(
                "%s_lsf_fwhm_%04d.fits" %
                (args.surname, int(np.floor(reference + 0.5)))).resolve())
                for reference, usable in zip(
                    REFERENCE_ARC_WAVELENGTHS, ARC_USABLE_FOR_LSF) if usable],
            "qa_png": str(Path("%s_lsf_qa.png" % args.surname).resolve()),
            "usable_for_lsf_sample_counts": usable_counts,
            "unique_sample_spaxels": int(lsf_unique_positions),
            "reference_wavelengths": REFERENCE_ARC_WAVELENGTHS.tolist(),
            "arc_usable_for_lsf": ARC_USABLE_FOR_LSF.tolist(),
            "arc_blend_states": ARC_BLEND_STATES.tolist(),
            "arc_blend_notes": ARC_BLEND_NOTES,
        })
        print("[cube] LSF products written: %s" % lsf_summary["sample_fits"],
              flush=True)

    direct_groups = int(sum(row["alpha_direct_groups"] for row in calibration_rows))
    historical_groups = int(sum(row["alpha_historical_groups"] for row in calibration_rows))
    hardware_exclusion_summary = {
        "registry": _file_identity(HARDWARE_REGISTRY_PATH),
        "purpose": "cube",
        "registry_records": int(len(HARDWARE_EXCLUSIONS)),
        "persistent_registry_records": int(sum(
            "specid" in record for record in HARDWARE_EXCLUSIONS)),
        "hardware_excluded_groups": int(sum(
            row["hardware_excluded_groups"] for row in calibration_rows)),
        "persistent_hardware_excluded_groups": int(sum(
            row["persistent_hardware_excluded_groups"]
            for row in calibration_rows)),
        "hardware_excluded_fiber_rows": int(sum(
            row["hardware_excluded_fiber_rows"] for row in calibration_rows)),
        "persistent_hardware_excluded_fiber_rows": int(sum(
            row["persistent_hardware_excluded_fiber_rows"]
            for row in calibration_rows)),
    }
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
        "exposure_scale": {
            "method": "physical-amplifier-collapsed LOO surface brightness",
            "bands": ["ON", "OFF"],
            "SB_iterations_applied": 2,
            "third_iteration_applied": False,
            "gauge": "median finite raw g at each iteration",
            "correction": "O_final = O0 / (g1_rel * g2_rel)",
            "sky_reestimated_after_scale": False,
            "errors_scaled": True,
            "sigma_g_added_to_fiber_errors": False,
            "exposure_scale_csv": _file_identity(exposure_scale_path),
            "validation": {
                "rows": int(len(exposure_scale_rows)),
                "max_abs_direct_error": float(max(
                    row["validation_max_abs_direct_error"]
                    for row in exposure_scale_rows)),
                "max_abs_identity_error": float(max(
                    row["validation_max_abs_identity_error"]
                    for row in exposure_scale_rows)),
            },
        },
        "provenance": model_provenance,
        "filters": {
            "ON": _file_identity(on_filter),
            "OFF": _file_identity(off_filter),
            "band_order": list(frozen_parent.BANDS),
            "band_response_source": "m101_native_data._band_responses",
            "null_response_provenance": null_provenance,
            "effective_wavelengths_A": effective_bands.tolist(),
        },
        "response_composition_validation": composition,
        "blank_loader": blank_provenance,
        "hardware_exclusions": hardware_exclusion_summary,
        "preflight": preflight,
        "coverage_table": str(coverage_path.resolve()),
        "calibration_summary_table": str(calibration_path.resolve()),
        "sky_summary_table": str(sky_path.resolve()),
        "qa": {
            **qa_summary,
            "qa_table": str(qa_path.resolve()),
            "queue_table": str(qa_queue_path.resolve()),
        },
        "lsf": lsf_summary,
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
