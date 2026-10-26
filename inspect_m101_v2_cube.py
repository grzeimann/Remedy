#!/usr/bin/env python3
"""Focused, read-only diagnostic for the completed M101 VIRUS v2 cubes.

The script audits the current production source, the six delivered primary
products, and the spectral WCS.  It does not import or rerun the production
reconstruction.  The optional --write-fixed-headers mode writes new FITS
files with only a demonstrably corrected WCS; array values are verified
before and after writing.

Typical TACC invocation::

    python inspect_m101_v2_cube.py \
        --base-dir /work/03730/gregz/maverick/m101_recon

The default output directory is ``m101_v2_cube_audit`` below the base
directory.  The cube products are read through FITS memmaps and spectral
diagnostics are accumulated plane-by-plane so that the full 1201 x 1201 x
1036 release cube need not be loaded into RAM.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "m101_v2_cube_audit_mpl"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
import astropy.units as u
from astropy.wcs import WCS


PRODUCT_NAMES = {
    "SCI": "m101_virus_v2_cube.fits",
    "VAR": "m101_virus_v2_variance_cube.fits",
    "ERROR": "m101_virus_v2_errorcube.fits",
    "DQ": "m101_virus_v2_dq_cube.fits",
    "NCONTRIB": "m101_virus_v2_ncontrib_cube.fits",
    "WEIGHT": "m101_virus_v2_weight_cube.fits",
}
FIXED_NAMES = {
    key: value.replace(".fits", "_wcsfixed.fits")
    for key, value in PRODUCT_NAMES.items()
}

DQ_INSUFFICIENT_SUPPORT = 1 << 0
DQ_VAR_INCOMPLETE = 1 << 1
DQ_EMPIRICAL_VAR_USED = 1 << 2
DQ_FORMAL_VAR_USED = 1 << 3
DQ_VAR_EMPIRICAL_ONLY = 1 << 4
FIBER_RADIUS_ARCSEC = 0.75
FIBER_AREA_ARCSEC2 = np.pi * FIBER_RADIUS_ARCSEC**2
GAUSSIAN_FWHM_ARCSEC = 1.8
EXPECTED_WAVE = 3470.0 + 2.0 * np.arange(1036, dtype=float)
HBETA_A = 4861.0


class _ScaledArray:
    """Small memmap-backed view that applies FITS BSCALE/BZERO per slice."""

    def __init__(self, raw, bscale, bzero):
        self.raw = raw
        self.bscale = float(bscale)
        self.bzero = float(bzero)
        self.shape = raw.shape
        self.dtype = np.result_type(raw.dtype, np.float64)

    def __getitem__(self, item):
        values = self.raw[item]
        if self.bscale == 1.0 and self.bzero == 0.0:
            return values
        return values * self.bscale + self.bzero


def _open_memmap(path):
    """Open a FITS image without expanding unsigned integer scaling in RAM."""
    hdul = fits.open(path, mode="readonly", memmap=True,
                     do_not_scale_image_data=True)
    header = hdul[0].header
    raw = hdul[0].data
    bscale = header.get("BSCALE", 1.0)
    bzero = header.get("BZERO", 0.0)
    data = (_ScaledArray(raw, bscale, bzero)
            if float(bscale) != 1.0 or float(bzero) != 0.0 else raw)
    return hdul, data


def _json_value(value):
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def _finite_stats(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"N": 0, "median": None, "p16": None, "p84": None,
                "robust_scatter": None}
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    return {"N": int(values.size), "median": median,
            "p16": float(np.percentile(values, 16)),
            "p84": float(np.percentile(values, 84)),
            "robust_scatter": float(1.4826 * mad)}


def _safe_median(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return float(np.median(values)) if values.size else np.nan


def _safe_percentile(values, percentile):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return float(np.percentile(values, percentile)) if values.size else np.nan


def _source_line_numbers(path, patterns):
    """Return the first source line matching each requested regex."""
    lines = Path(path).read_text(errors="replace").splitlines()
    result = {}
    for name, pattern in patterns.items():
        match = next((i for i, line in enumerate(lines, 1)
                      if re.search(pattern, line)), None)
        function = None
        if match is not None:
            for candidate in range(match - 1, -1, -1):
                function_match = re.match(r"\s*(?:def|class)\s+([A-Za-z0-9_]+)", lines[candidate])
                if function_match:
                    function = function_match.group(1)
                    break
        result[name] = {"file": str(Path(path).resolve()), "line": match,
                        "function": function, "pattern": pattern}
    return result


def _extract_def_wave(source_dir):
    path = source_dir / "diagnose_m101_hierarchical.py"
    if not path.exists():
        return EXPECTED_WAVE.copy(), {"status": "source_missing",
                                      "file": str(path)}
    text = path.read_text(errors="replace")
    match = re.search(
        r"DEF_WAVE\s*=\s*np\.linspace\(\s*([0-9.eE+-]+)\s*,\s*"
        r"([0-9.eE+-]+)\s*,\s*(\d+)\s*\)", text)
    if not match:
        return EXPECTED_WAVE.copy(), {"status": "pattern_not_found",
                                      "file": str(path)}
    start, stop, count = float(match.group(1)), float(match.group(2)), int(match.group(3))
    wave = np.linspace(start, stop, count)
    return wave, {"status": "parsed", "file": str(path.resolve()),
                  "expression": match.group(0), "start_A": start,
                  "stop_A": stop, "planes": count,
                  "line": text[:match.start()].count("\n") + 1}


def _production_source_audit(source_dir):
    production = source_dir / "make_m101_cube_current_model.py"
    quick = source_dir / "quick_reduction.py"
    validation = source_dir / "validate_m101_referee_release.py"
    patterns = {
        "DEF_WAVE": r"DEF_WAVE\s*=",
        "fiber_area": r"FIBER_AREA\s*=",
        "gaussian_fwhm": r"GAUSSIAN_FWHM_ARCSEC\s*=",
        "native_total": r"total\s*=\s*source\s*/\s*offset\s*\+\s*sky_native",
        "formal_error": r"errors\[indices\]\s*=\s*error_work\[indices\]\s*/\s*multiplier",
        "gaussian_splat_flux": r"flux_sum\[py, px\]\s*\+=\s*weight\s*\*\s*flux\s*/\s*area",
        "gaussian_splat_variance": r"variance_numerator\[py, px\].*err\s*/\s*area",
        "formal_variance": r"formal\[formal_valid\]\s*=",
        "empirical_variance": r"empirical\[five_or_more\]\s*=",
        "larger_variance": r"variance\[both\]\s*=\s*np\.maximum",
        "error_output": r"np\.sqrt\(np\.asarray\(variancecube",
        "wcs_cd_spatial": r'header\["CD1_1"\]',
        "wcs_cdelt_spectral": r'header\["CDELT3"\]',
        "wcs_cd3_3_missing_or_repair_target": r'header\["CD3_3"\]',
        "plane_assignment": r"cube\[index\]\s*=\s*image",
        "wcs_crpix3": r'header\["CRPIX3"\]',
        "bunit_var": r'variance_header\["BUNIT"\]',
        "bunit_error": r'error_header\["BUNIT"\]',
    }
    return {
        "production_script": str(production.resolve()),
        "source_lines": _source_line_numbers(production, patterns)
        if production.exists() else {},
        "quick_reduction_lines": _source_line_numbers(
            quick, {
                "fiber_table_types": r"class Fibers",
                "fiber_spectrum_write": r"specrow\['spectrum'\]\s*=\s*scispectra",
                "fiber_error_write": r"specrow\['error'\]\s*=\s*errspectra",
            }) if quick.exists() else {},
        "validation_script": str(validation.resolve()),
        "validation_wcs_note": _source_line_numbers(
            validation, {"direct_linear_wcs_workaround":
                         r"Reading.*cards directly avoids a WCSLIB"})
        if validation.exists() else {},
        "interpretation": {
            "native_total": "D = Fibers.spectrum / Survey.offset + Fibers.skyspectrum",
            "corrected_total": "T = (D - alpha*(f(q)-f0)*K - mu_a) / M",
            "object_spectrum": "O = T - exposure_sky",
            "formal_error": "Fibers.error / abs(Survey.offset) / M, then exposure scale",
            "fiber_area_arcsec2": float(FIBER_AREA_ARCSEC2),
            "spatial_reconstruction": "Gaussian weighted splat; flux and error are divided by fiber area, then normalized by weight sum",
            "variance_propagation": (
                "formal: n=2 uses sum(shot variance)/4; n>=3 uses "
                "n*pi/(2*(sum(1/sigma_shot))^2). Empirical: for n>=5, "
                "(1.2533*1.4826*MAD/sqrt(n))^2. When both exist, max(formal, empirical) is selected."
            ),
            "gaussian_fwhm_arcsec": GAUSSIAN_FWHM_ARCSEC,
            "spatial_covariance": "not represented in any output product",
            "spectral_covariance": "not represented; each wavelength plane is processed independently",
            "omitted_uncertainty_terms": [
                "additive calibration-model uncertainty",
                "exposure-sky uncertainty",
                "exposure-scale uncertainty sigma_g",
                "spatial covariance induced by Gaussian reconstruction",
            ],
            "upstream_units": (
                "The production source does not attach a physical BUNIT to the H5 Fibers fields. "
                "The project CUBE_README convention identifies the calibrated working scale as "
                "1e-17 erg s^-1 cm^-2 Angstrom^-1 before the cube's fiber-area normalization."
            ),
            "final_sci_units": (
                "1e-17 erg s^-1 cm^-2 Angstrom^-1 arcsec^-2, surface-brightness-like, "
                "if the documented project upstream scale is used. This scale is provenance, "
                "not an explicit production SCI BUNIT."
            ),
            "variance_units": "SCI units squared; ERROR is sqrt(VAR) in SCI units",
            "production_bunit": "The current writer assigns BUNIT to VAR/ERROR/DQ/WEIGHT/NCONTRIB, but does not assign an explicit physical SCI BUNIT; the SCI primary header therefore remains generic/blank.",
        },
    }


def _open_products(base_dir):
    handles = {}
    data = {}
    headers = {}
    paths = {}
    for key, name in PRODUCT_NAMES.items():
        path = base_dir / name
        if not path.exists():
            raise FileNotFoundError(f"missing {key} product: {path}")
        hdul, array = _open_memmap(path)
        if not hdul or hdul[0].data is None:
            hdul.close()
            raise ValueError(f"{path} has no primary array")
        handles[key] = hdul
        data[key] = array
        headers[key] = hdul[0].header.copy()
        paths[key] = path
    shape = data["SCI"].shape
    if len(shape) != 3:
        raise ValueError(f"SCI is not a 3-D cube: {shape}")
    for key, array in data.items():
        if array.shape != shape:
            raise ValueError(f"{key} shape {array.shape} differs from SCI {shape}")
    return handles, data, headers, paths


def _header_cards(header):
    keys = ["WCSAXES", "NAXIS3", "CRPIX3", "CRVAL3", "CDELT3", "CD3_3",
            "PC3_3", "CTYPE3", "CUNIT3"]
    spatial = []
    for prefix in ("CD", "PC", "CDELT", "CRPIX", "CRVAL", "CTYPE", "CUNIT"):
        for axis in (1, 2):
            key = f"{prefix}{axis}"
            if key in header:
                spatial.append(key)
        for row in (1, 2):
            for col in (1, 2):
                key = f"{prefix}{row}_{col}"
                if key in header:
                    spatial.append(key)
    cards = {key: header.get(key) for key in keys}
    cards["spatial_cards"] = {key: header.get(key) for key in sorted(set(spatial))}
    return cards


def _wcs_wavelength(header, n_wave):
    """Evaluate the spectral coordinate through standard Astropy WCS."""
    wcs = WCS(header, relax=True)
    pixel = np.arange(1, n_wave + 1, dtype=float)
    x = np.ones(n_wave, dtype=float)
    y = np.ones(n_wave, dtype=float)
    world = wcs.wcs_pix2world(x, y, pixel, 1)
    values = np.asarray(world[2], dtype=float)
    # WCSLIB commonly returns SI-valued world coordinates for a wavelength
    # axis even when the FITS CUNIT card says Angstrom.  Report all three
    # extraction methods in Angstrom so the numerical comparison is fair.
    unit = u.Unit(str(wcs.wcs.cunit[2])) if wcs.wcs.cunit[2] else u.AA
    return np.asarray(values * unit.to(u.AA), dtype=float), wcs


def _direct_wavelength(header, n_wave):
    if "CDELT3" not in header:
        raise ValueError("CDELT3 is absent; direct legacy relation cannot be evaluated")
    pixel = np.arange(1, n_wave + 1, dtype=float)
    return (float(header["CRVAL3"])
            + (pixel - float(header["CRPIX3"])) * float(header["CDELT3"]))


def _wcs_audit(header, n_wave, project_wave):
    direct = _direct_wavelength(header, n_wave)
    result = {
        "header_cards": _header_cards(header),
        "direct_relation": {"first_A": float(direct[0]), "last_A": float(direct[-1]),
                             "step_A": float(np.median(np.diff(direct))),
                             "N": int(direct.size)},
        "project_DEF_WAVE": {"first_A": float(project_wave[0]),
                             "last_A": float(project_wave[-1]),
                             "step_A": float(np.median(np.diff(project_wave))),
                             "N": int(project_wave.size)},
        "failure_demonstration": {},
    }
    try:
        astropy_wave, _ = _wcs_wavelength(header, n_wave)
        result["astropy_wcs"] = {
            "status": "success", "first_A": float(astropy_wave[0]),
            "last_A": float(astropy_wave[-1]),
            "step_A": float(np.median(np.diff(astropy_wave))),
            "N": int(astropy_wave.size),
            "max_abs_direct_difference_A": float(np.max(np.abs(astropy_wave - direct))),
        }
        result["failure_demonstration"] = {
            "mixed_spatial_CD_without_CD3_3": bool(
                any(f"CD{i}_{j}" in header for i in (1, 2) for j in (1, 2))
                and "CD3_3" not in header and "CDELT3" in header),
            "astropy_uses_effective_increment_A": float(np.median(np.diff(astropy_wave))),
            "legacy_CDELT3_A": float(header.get("CDELT3", np.nan)),
            "observed_effective_increment_ratio": float(
                np.median(np.diff(astropy_wave)) / float(header["CDELT3"])),
        }
    except Exception as exc:
        result["astropy_wcs"] = {"status": "error", "error": repr(exc)}
    result["stored_data_grid_test"] = {
        "matches_project_DEF_WAVE": bool(np.allclose(project_wave, EXPECTED_WAVE,
                                                       rtol=0.0, atol=1e-8)),
        "conclusion": (
            "Production source constructs cube[index] while iterating index over DEF_WAVE; "
            "the stored arrays therefore use 3470 + 2*index A, and the WCS defect is metadata-only."
        ),
        "metadata_only_if_true": True,
    }
    result["comparison"] = {
        "direct_minus_project_max_abs_A": float(np.max(np.abs(direct - project_wave))),
    }
    if "astropy_wcs" in result and result["astropy_wcs"].get("status") == "success":
        try:
            astropy_wave, _ = _wcs_wavelength(header, n_wave)
            result["comparison"]["astropy_minus_project_max_abs_A"] = float(
                np.max(np.abs(astropy_wave - project_wave)))
        except Exception:
            pass
    return result


def _fixed_header(header, project_wave):
    """Use one CD convention when the existing header has spatial CD cards."""
    fixed = header.copy()
    step = float(np.median(np.diff(project_wave)))
    if any(f"CD{i}_{j}" in fixed for i in (1, 2) for j in (1, 2)):
        fixed["CD3_3"] = (step, "linear wavelength increment [Angstrom]")
        # CD and CDELT/PC are alternate matrix conventions.  The production
        # header already uses CD for the spatial axes, so remove the lone
        # spectral CDELT card rather than leave an ignored conflict.
        fixed.pop("CDELT3", None)
        fixed.pop("PC3_3", None)
    else:
        fixed["CDELT3"] = (step, "linear wavelength increment [Angstrom]")
    fixed.add_history("M101 v2 audit: spectral WCS repaired in a copy only.")
    fixed.add_history("CD3_3 set to the production DEF_WAVE increment; data unchanged.")
    return fixed


def _test_fixed_wcs(header, project_wave):
    fixed = _fixed_header(header, project_wave)
    try:
        wave, _ = _wcs_wavelength(fixed, len(project_wave))
        step = np.diff(wave)
        passed = (np.isclose(wave[0], 3470.0, rtol=0, atol=1e-8)
                  and np.isclose(wave[-1], 5540.0, rtol=0, atol=1e-8)
                  and np.allclose(step, 2.0, rtol=0, atol=1e-8)
                  and len(wave) == 1036)
        return fixed, {"status": "PASS" if passed else "FAIL",
                       "first_A": float(wave[0]), "last_A": float(wave[-1]),
                       "step_A": float(np.median(step)), "N": int(len(wave)),
                       "max_step_error_A": float(np.max(np.abs(step - 2.0)))}
    except Exception as exc:
        return fixed, {"status": "FAIL", "error": repr(exc)}


def _cube_summary(data, shape):
    sci, var, error = data["SCI"], data["VAR"], data["ERROR"]
    dq, ncontrib = data["DQ"], data["NCONTRIB"]
    supported_count = 0
    common_count = 0
    mismatch_count = 0
    max_difference = 0.0
    finite_counts = {key: 0 for key in ("SCI", "VAR", "ERROR", "WEIGHT")}
    counts = {str(bit): 0 for bit in (1, 2, 4, 8, 16)}
    n_lt_2 = 0
    for index in range(shape[0]):
        sci_plane = np.asarray(sci[index])
        var_plane = np.asarray(var[index])
        error_plane = np.asarray(error[index])
        dq_plane = np.asarray(dq[index], dtype=np.uint64)
        ncontrib_plane = np.asarray(ncontrib[index])
        supported = np.isfinite(sci_plane) & (ncontrib_plane >= 2)
        both = np.isfinite(sci_plane) & np.isfinite(var_plane) & np.isfinite(error_plane)
        close = both & np.isclose(error_plane, np.sqrt(np.maximum(var_plane, 0.0)),
                                  rtol=2e-6, atol=2e-7)
        supported_count += int(np.count_nonzero(supported))
        common_count += int(np.count_nonzero(supported & np.isfinite(var_plane)))
        mismatch_count += int(np.count_nonzero(both & ~close))
        if np.any(both & ~close):
            max_difference = max(max_difference, float(np.max(
                np.abs(error_plane[both & ~close]
                       - np.sqrt(np.maximum(var_plane[both & ~close], 0.0))))))
        for key, plane in (("SCI", sci_plane), ("VAR", var_plane),
                           ("ERROR", error_plane), ("WEIGHT", data["WEIGHT"][index])):
            finite_counts[key] += int(np.count_nonzero(np.isfinite(plane)))
        for bit in (1, 2, 4, 8, 16):
            counts[str(bit)] += int(np.count_nonzero((dq_plane & bit) != 0))
        n_lt_2 += int(np.count_nonzero(ncontrib_plane < 2))
    total = int(np.prod(shape))
    return {
        "shape_Nwave_Ny_Nx": [int(x) for x in shape],
        "total_voxels": total,
        "supported_voxels_NCONTRIB_ge_2": supported_count,
        "NCONTRIB_lt_2_fraction": float(n_lt_2 / total),
        "ERROR_equals_sqrt_VAR": bool(mismatch_count == 0),
        "ERROR_sqrt_VAR_mismatch_voxels": mismatch_count,
        "ERROR_sqrt_VAR_max_abs_difference": max_difference,
        "variance_available_on_supported_fraction": float(common_count / supported_count)
        if supported_count else np.nan,
        "DQ_bit_counts": counts,
        "DQ_bit_definitions": {
            "1": "NCONTRIB < 2",
            "2": "SCI valid, VAR unavailable",
            "4": "empirical VAR adopted over formal",
            "8": "formal VAR adopted",
            "16": "empirical-only VAR adopted",
        },
        "finite_value_fractions": {
            key: float(value / total) for key, value in finite_counts.items()
        },
    }


def _variance_categories(data):
    dq, sci, var, ncontrib = data["DQ"], data["SCI"], data["VAR"], data["NCONTRIB"]
    counts = {"formal_variance_used": 0, "empirical_over_formal": 0,
              "empirical_only": 0, "variance_unavailable": 0,
              "ncontrib_lt_2": 0}
    denominator = 0
    for index in range(sci.shape[0]):
        valid = np.isfinite(sci[index]) & (ncontrib[index] >= 2)
        denominator += int(np.count_nonzero(valid))
        dq_plane = np.asarray(dq[index], dtype=np.uint64)
        counts["formal_variance_used"] += int(np.count_nonzero(
            valid & ((dq_plane & DQ_FORMAL_VAR_USED) != 0)))
        counts["empirical_over_formal"] += int(np.count_nonzero(
            valid & ((dq_plane & DQ_EMPIRICAL_VAR_USED) != 0)))
        counts["empirical_only"] += int(np.count_nonzero(
            valid & ((dq_plane & DQ_VAR_EMPIRICAL_ONLY) != 0)))
        counts["variance_unavailable"] += int(np.count_nonzero(
            valid & ((dq_plane & DQ_VAR_INCOMPLETE) != 0)))
        counts["ncontrib_lt_2"] += int(np.count_nonzero(ncontrib[index] < 2))
    denominator = max(1, denominator)
    return {key: {"voxels": int(value),
                  "fraction_of_supported": float(value / denominator)}
            for key, value in counts.items()}


def _continuum_split(sci, ncontrib, wave):
    control = ((wave >= 4500.0) & (wave <= 4750.0)) | ((wave >= 5000.0) & (wave <= 5250.0))
    indices = np.flatnonzero(control)
    if indices.size == 0:
        indices = np.array([0, len(wave) - 1])
    # A small evenly spaced stack is sufficient for a robust population split
    # and avoids materializing all control planes for the 6-GB release cube.
    chosen = indices[np.linspace(0, indices.size - 1, min(12, indices.size)).astype(int)]
    stack = []
    for index in chosen:
        plane = np.asarray(sci[index], dtype=float).copy()
        plane[ncontrib[index] < 2] = np.nan
        stack.append(plane)
    continuum = np.nanmedian(np.stack(stack, axis=0), axis=0)
    finite = np.isfinite(continuum)
    low_cut = _safe_percentile(continuum[finite], 40)
    high_cut = _safe_percentile(continuum[finite], 80)
    return ((finite & (continuum <= low_cut)), (finite & (continuum >= high_cut)),
            {"control_plane_count": int(chosen.size),
             "control_wavelength_range_A": [4500.0, 5250.0],
             "low_background_percentile": 40, "low_background_cut": low_cut,
             "bright_source_percentile": 80, "bright_source_cut": high_cut,
             "finite_continuum_spaxels": int(np.count_nonzero(finite))})


def _spatial_series(data, wave, population):
    sci, error, var = data["SCI"], data["ERROR"], data["VAR"]
    dq, ncontrib = data["DQ"], data["NCONTRIB"]
    fields = {name: [] for name in (
        "sci", "sci_scatter", "error", "var", "ncontrib", "ncontrib_p16",
        "ncontrib_p84", "empirical_fraction", "formal_fraction",
        "empirical_only_fraction", "variance_unavailable_fraction",
        "insufficient_support_fraction")}
    for index in range(len(wave)):
        valid = population & np.isfinite(sci[index]) & (ncontrib[index] >= 2)
        values = sci[index][valid]
        fields["sci"].append(_safe_median(values))
        if values.size:
            median = np.median(values)
            fields["sci_scatter"].append(float(1.4826 * np.median(np.abs(values - median))))
        else:
            fields["sci_scatter"].append(np.nan)
        fields["error"].append(_safe_median(error[index][valid]))
        fields["var"].append(_safe_median(var[index][valid]))
        fields["ncontrib"].append(_safe_median(ncontrib[index][valid]))
        fields["ncontrib_p16"].append(_safe_percentile(ncontrib[index][valid], 16))
        fields["ncontrib_p84"].append(_safe_percentile(ncontrib[index][valid], 84))
        denominator = max(1, int(values.size))
        dq_plane = np.asarray(dq[index], dtype=np.uint64)
        fields["empirical_fraction"].append(
            float(np.count_nonzero(valid & ((dq_plane & DQ_EMPIRICAL_VAR_USED) != 0)) / denominator))
        fields["formal_fraction"].append(
            float(np.count_nonzero(valid & ((dq_plane & DQ_FORMAL_VAR_USED) != 0)) / denominator))
        fields["empirical_only_fraction"].append(
            float(np.count_nonzero(valid & ((dq_plane & DQ_VAR_EMPIRICAL_ONLY) != 0)) / denominator))
        fields["variance_unavailable_fraction"].append(
            float(np.count_nonzero(valid & ((dq_plane & DQ_VAR_INCOMPLETE) != 0)) / denominator))
        fields["insufficient_support_fraction"].append(
            float(np.count_nonzero(population & (ncontrib[index] < 2)) /
                  max(1, int(np.count_nonzero(population)))))
    return {key: np.asarray(value, dtype=float) for key, value in fields.items()}


def _integrated_spectrum(data, pixel_scale):
    sci, var, ncontrib = data["SCI"], data["VAR"], data["NCONTRIB"]
    area = float(pixel_scale) ** 2
    field = np.zeros(sci.shape[1:], dtype=bool)
    for index in range(sci.shape[0]):
        field |= np.isfinite(sci[index]) & (ncontrib[index] >= 2)
    flux, sigma = [], []
    for index in range(sci.shape[0]):
        selected = field & np.isfinite(sci[index]) & np.isfinite(var[index]) & (ncontrib[index] >= 2)
        flux.append(float(np.sum(sci[index][selected], dtype=np.float64) * area))
        sigma.append(float(np.sqrt(np.sum(np.maximum(var[index][selected], 0.0), dtype=np.float64)) * area))
    flux, sigma = np.asarray(flux), np.asarray(sigma)
    fraction = np.full(flux.shape, np.nan)
    np.divide(sigma, np.abs(flux), out=fraction, where=np.abs(flux) > 0)
    finite = np.isfinite(fraction) & (np.abs(flux) > np.nanpercentile(np.abs(flux[np.isfinite(flux)]), 10))
    return {
        "selection": "union of spatial pixels with finite SCI and NCONTRIB >= 2 at any wavelength",
        "pixel_scale_arcsec": float(pixel_scale),
        "pixel_area_arcsec2": area,
        "flux_density_scale": "product units * arcsec^2; multiply by 1e-17 for documented physical scale",
        "fractional_uncertainty": _finite_stats(fraction[finite]),
        "median_integrated_flux_product_units": float(np.nanmedian(flux)),
        "median_diagonal_sigma_product_units": float(np.nanmedian(sigma)),
        "caveat": "sqrt(sum(VAR)) assumes independent spatial pixels and excludes Gaussian-reconstruction covariance",
    }, flux, sigma


def _blank_aperture_check(data, background_mask, pixel_scale, wave):
    sci, var, ncontrib = data["SCI"], data["VAR"], data["NCONTRIB"]
    ny, nx = background_mask.shape
    ntiles_y, ntiles_x = min(4, ny), min(4, nx)
    tile_spectra, tile_sigmas = [], []
    for iy in range(ntiles_y):
        y0, y1 = iy * ny // ntiles_y, (iy + 1) * ny // ntiles_y
        for ix in range(ntiles_x):
            x0, x1 = ix * nx // ntiles_x, (ix + 1) * nx // ntiles_x
            mask = np.zeros_like(background_mask, dtype=bool)
            mask[y0:y1, x0:x1] = background_mask[y0:y1, x0:x1]
            if np.count_nonzero(mask) < 10:
                continue
            sums, errors = [], []
            for index in range(len(wave)):
                selected = mask & np.isfinite(sci[index]) & np.isfinite(var[index]) & (ncontrib[index] >= 2)
                sums.append(float(np.sum(sci[index][selected], dtype=np.float64) * pixel_scale**2))
                errors.append(float(np.sqrt(np.sum(np.maximum(var[index][selected], 0.0), dtype=np.float64)) * pixel_scale**2))
            tile_spectra.append(sums)
            tile_sigmas.append(errors)
    if len(tile_spectra) < 3:
        return {"status": "insufficient_background_tiles", "tiles": len(tile_spectra)}
    observed = 1.4826 * np.median(np.abs(np.asarray(tile_spectra) - np.median(tile_spectra, axis=0)), axis=0)
    predicted = np.median(np.asarray(tile_sigmas), axis=0)
    ratio = np.divide(observed, predicted, out=np.full_like(observed, np.nan), where=predicted > 0)
    control = ((wave < 4820) | (wave > 4900)) & np.isfinite(ratio)
    return {
        "status": "diagnostic_only",
        "tiles": len(tile_spectra),
        "description": "4x4 large spatial background tiles; observed tile scatter versus median diagonal sigma",
        "ratio_observed_to_sqrt_sum_var": _finite_stats(ratio[control]),
        "hbeta_ratio_observed_to_sqrt_sum_var": _safe_median(ratio[(wave >= 4820) & (wave <= 4900)]),
        "caveat": "tile scatter also contains real spatial gradients and is not used to inflate VAR",
    }


def _feature_assessment(wave, series, label):
    region = (wave >= 4820.0) & (wave <= 4900.0)
    control = ((wave >= 4750.0) & (wave <= 4800.0)) | ((wave >= 4920.0) & (wave <= 4970.0))
    result = {"population": label, "region_A": [4820.0, 4900.0],
              "control_A": [[4750.0, 4800.0], [4920.0, 4970.0]]}
    for key in ("sci", "sci_scatter", "error", "var", "ncontrib",
                "empirical_fraction", "formal_fraction", "empirical_only_fraction",
                "variance_unavailable_fraction"):
        values = series[key]
        result[key] = {
            "hbeta_median": _safe_median(values[region]),
            "control_median": _safe_median(values[control]),
            "ratio_hbeta_to_control": float(_safe_median(values[region]) /
                                             _safe_median(values[control]))
            if np.isfinite(_safe_median(values[control])) and _safe_median(values[control]) != 0 else None,
        }
    return result


def _feature_verdict(feature):
    """Turn the requested diagnostics into cautious, reproducible evidence."""
    def number(key, field):
        value = feature[key][field]
        return float(value) if value is not None and np.isfinite(value) else np.nan

    def ratio(key):
        value = feature[key]["ratio_hbeta_to_control"]
        return float(value) if value is not None and np.isfinite(value) else np.nan

    sci_increase = ratio("sci") > 1.05
    scatter_increase = ratio("sci_scatter") > 1.10
    error_increase = ratio("error") > 1.10
    var_increase = ratio("var") > 1.20
    empirical_increase = (ratio("empirical_fraction") > 1.20 or
                          number("empirical_fraction", "hbeta_median") >
                          number("empirical_fraction", "control_median") + 0.10)
    formal_increase = (ratio("formal_fraction") > 1.20 or
                       number("formal_fraction", "hbeta_median") >
                       number("formal_fraction", "control_median") + 0.10)
    ncontrib_drop = ratio("ncontrib") < 0.95
    unavailable_increase = (ratio("variance_unavailable_fraction") > 1.20 or
                            number("variance_unavailable_fraction", "hbeta_median") >
                            number("variance_unavailable_fraction", "control_median") + 0.10)
    causes = []
    if sci_increase:
        causes.append("nonzero mean background residual")
    if scatter_increase or var_increase or empirical_increase:
        causes.append("increased shot-to-shot scatter / empirical variance")
    if formal_increase or error_increase:
        causes.append("formal-error contribution")
    if ncontrib_drop:
        causes.append("reduced NCONTRIB")
    if unavailable_increase:
        causes.append("increased variance unavailability")
    return {
        "thresholds": {
            "SCI_ratio_gt": 1.05, "scatter_or_error_ratio_gt": 1.10,
            "VAR_ratio_gt": 1.20, "fraction_absolute_increase_gt": 0.10,
            "NCONTRIB_ratio_lt": 0.95,
        },
        "real_background_level_increase": bool(sci_increase),
        "corresponding_noise_increase": bool(scatter_increase or error_increase or var_increase),
        "evidence_for_empirical_shot_scatter": bool(empirical_increase or scatter_increase),
        "evidence_for_formal_variance": bool(formal_increase or error_increase),
        "evidence_for_NCONTRIB_effect": bool(ncontrib_drop),
        "evidence_for_variance_unavailable": bool(unavailable_increase),
        "supported_causes": causes,
        "classification": ("wavelength-localized diagnostic feature is present in this population"
                            if sci_increase or scatter_increase or var_increase else
                            "no threshold-level broad H-beta feature in this population"),
    }


def _plot_wavelength_diagnostic(output, header, project_wave):
    direct = _direct_wavelength(header, len(project_wave))
    try:
        astropy, _ = _wcs_wavelength(header, len(project_wave))
    except Exception:
        astropy = np.full_like(project_wave, np.nan)
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    axes[0].plot(np.arange(len(project_wave)), project_wave, label="production DEF_WAVE", lw=2)
    axes[0].plot(np.arange(len(project_wave)), direct, label="header CRVAL/CRPIX/CDELT", ls="--")
    axes[0].plot(np.arange(len(project_wave)), astropy, label="Astropy WCS", alpha=.8)
    axes[0].set_ylabel("Wavelength [A]")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=.25)
    axes[1].plot(project_wave, direct - project_wave, label="direct - DEF_WAVE")
    axes[1].plot(project_wave, astropy - project_wave, label="Astropy - DEF_WAVE")
    axes[1].axhline(0, color="k", lw=.7)
    axes[1].set_xlabel("Production wavelength [A]")
    axes[1].set_ylabel("Difference [A]")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=.25)
    fig.suptitle("M101 v2 spectral WCS audit")
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def _plot_hbeta(output, wave, background, source):
    select = (wave >= 4500) & (wave <= 5100)
    fig, axes = plt.subplots(5, 1, figsize=(9, 12), sharex=True)
    panels = [
        ("sci", "SCI robust location", "product units"),
        ("sci_scatter", "SCI robust spatial scatter", "product units"),
        ("error", "median ERROR", "product units"),
        ("ncontrib", "NCONTRIB", "count"),
        ("empirical_fraction", "empirical VAR adopted fraction", "fraction"),
    ]
    for axis, (key, title, ylabel) in zip(axes, panels):
        axis.plot(wave[select], background[key][select], label="background", lw=1.5)
        axis.plot(wave[select], source[key][select], label="bright/source", lw=1.0, alpha=.8)
        axis.axvline(HBETA_A, color="crimson", ls="--", lw=1, label="H-beta 4861 A")
        for control in (4750, 4950):
            axis.axvline(control, color="0.5", ls=":", lw=.8)
        axis.set_ylabel(ylabel)
        axis.set_title(title, loc="left", fontsize=10)
        axis.grid(alpha=.2)
    axes[0].legend(fontsize=8, ncol=2)
    axes[-1].set_xlabel("Wavelength [A]")
    fig.suptitle("M101 v2 H-beta background/source diagnostic")
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def _plot_variance(output, wave, background, cube_summary, categories):
    fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    axes[0].plot(wave, background["var"], label="background median VAR")
    axes[0].plot(wave, background["error"] ** 2, label="background median ERROR squared", alpha=.7)
    axes[0].axvline(HBETA_A, color="crimson", ls="--")
    axes[0].set_ylabel("variance [product units²]")
    axes[0].legend(fontsize=8)
    axes[1].plot(wave, background["empirical_fraction"], label="empirical over formal")
    axes[1].plot(wave, background["formal_fraction"], label="formal adopted")
    axes[1].plot(wave, background["empirical_only_fraction"], label="empirical only")
    axes[1].set_ylabel("DQ fraction")
    axes[1].legend(fontsize=8)
    axes[2].plot(wave, background["ncontrib"], label="background median NCONTRIB")
    axes[2].set_ylabel("NCONTRIB")
    axes[2].set_xlabel("Wavelength [A]")
    axes[2].axvline(HBETA_A, color="crimson", ls="--")
    for axis in axes:
        axis.grid(alpha=.2)
    fig.suptitle("M101 v2 variance and support diagnostic")
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def _array_digest(array):
    digest = hashlib.sha256()
    for start in range(0, array.shape[0], 4):
        digest.update(np.ascontiguousarray(array[start:start + 4]).tobytes())
    return digest.hexdigest()


def _finite_digest(array):
    digest = hashlib.sha256()
    for start in range(0, array.shape[0], 4):
        digest.update(np.ascontiguousarray(
            np.isfinite(array[start:start + 4])).tobytes())
    return digest.hexdigest()


def _write_fixed_copies(data, headers, paths, output_dir, project_wave):
    records = {}
    _, test = _test_fixed_wcs(headers["SCI"], project_wave)
    if test["status"] != "PASS":
        raise RuntimeError(f"refusing to write fixed headers: {test}")
    for key, path in paths.items():
        original_digest = _array_digest(data[key])
        original_shape = tuple(data[key].shape)
        original_finite_digest = _finite_digest(data[key])
        destination = output_dir / FIXED_NAMES[key]
        with fits.open(path, mode="readonly", memmap=True,
                       do_not_scale_image_data=True) as source:
            hdus = fits.HDUList([hdu.copy() for hdu in source])
        product_header, product_test = _test_fixed_wcs(headers[key], project_wave)
        if product_test["status"] != "PASS":
            raise RuntimeError(f"fixed WCS test failed for {key}: {product_test}")
        hdus[0].header = product_header
        hdus[0].header["EXTNAME"] = headers[key].get("EXTNAME", key)
        hdus.writeto(destination, overwrite=True, checksum=True)
        hdus.close()
        with fits.open(destination, mode="readonly", memmap=True,
                       do_not_scale_image_data=True) as check:
            new_header = check[0].header
            raw = check[0].data
            bscale = new_header.get("BSCALE", 1.0)
            bzero = new_header.get("BZERO", 0.0)
            new_data = (_ScaledArray(raw, bscale, bzero)
                        if float(bscale) != 1.0 or float(bzero) != 0.0 else raw)
            new_digest = _array_digest(new_data)
            records[key] = {
                "output": str(destination.resolve()),
                "shape_unchanged": tuple(new_data.shape) == original_shape,
                "array_bytes_unchanged": new_digest == original_digest,
                "finite_mask_unchanged": _finite_digest(new_data) == original_finite_digest,
                "original_sha256": original_digest,
                "copy_sha256": new_digest,
            }
        if not all(records[key][field] for field in
                   ("shape_unchanged", "array_bytes_unchanged", "finite_mask_unchanged")):
            raise RuntimeError(f"array verification failed for {key}: {records[key]}")
    return {"fixed_wcs_test": test, "products": records}


def _format_text(summary):
    wcs = summary["wcs_audit"]
    cube = summary["cube_validation"]
    feature = summary["hbeta_diagnostic"]["feature_assessment"]
    verdict = summary["hbeta_diagnostic"]["feature_verdict"]
    units = summary["source_audit"]["interpretation"]
    lines = [
        "M101 VIRUS v2 cube audit",
        f"Created: {summary['created_utc']}",
        "",
        "1. UNITS / UNCERTAINTY",
        f"SCI: {units['final_sci_units']}",
        f"VAR: {units['variance_units']}; ERROR=sqrt(VAR): {cube['ERROR_equals_sqrt_VAR']}",
        "A naive integrated sigma uses sqrt(sum(VAR)) times pixel area. It can look small because it is a diagonal-only sum and does not include spatial covariance from the Gaussian reconstruction.",
        "Omitted terms: additive calibration-model uncertainty, exposure-sky uncertainty, exposure-scale uncertainty, and spatial covariance.",
        f"Gaussian reconstruction FWHM={GAUSSIAN_FWHM_ARCSEC:.3g} arcsec; fiber area={FIBER_AREA_ARCSEC2:.6g} arcsec^2.",
        f"The nominal Gaussian sigma-footprint is {summary['spatial_reconstruction']['nominal_gaussian_area_in_output_pixels']:.6g} output pixels at the measured pixel scale; this is a scale estimate, not a covariance correction.",
        f"Integrated diagonal fractional uncertainty: {summary['integrated_spectrum']['fractional_uncertainty']}",
        "",
        "2. H-BETA",
        "The diagnostic compares 4820-4900 A with 4750-4800 and 4920-4970 A controls, separately for low-continuum background and bright/source spaxels.",
        f"Background feature assessment: {feature['background']}",
        f"Source feature assessment: {feature['bright_source']}",
        f"Background evidence verdict: {verdict['background']}",
        f"Source evidence verdict: {verdict['bright_source']}",
        "Interpretation is evidence-based: increased SCI location indicates mean residual background; increased SCI scatter/VAR and empirical fraction indicate shot-to-shot disagreement; reduced NCONTRIB or increased formal fraction indicates support/formal-error effects.",
        "No correction or variance inflation is applied by this audit.",
        "",
        "3. WCS",
        f"SCI WCS cards: {json.dumps(wcs['header_cards'], default=_json_value)}",
        f"Direct header relation: {wcs['direct_relation']}",
        f"Standard Astropy WCS: {wcs.get('astropy_wcs')}",
        f"Failure demonstration: {wcs.get('failure_demonstration')}",
        f"Stored production grid: {wcs['project_DEF_WAVE']}; metadata-only repair: {wcs['stored_data_grid_test']}",
        f"Fixed-header Astropy test: {summary['fixed_wcs_test']}",
        "Minimal repair for the existing spatial-CD convention: set CD3_3=2.0 and remove the conflicting CDELT3 card; write copies only.",
        "",
        "Variance DQ categories:",
        json.dumps(summary["variance_categories"], indent=2),
        "",
        "Output files:",
        *[f"- {path}" for path in summary["outputs"].values()],
    ]
    return "\n".join(lines) + "\n"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", type=Path,
                        default=Path("/work/03730/gregz/maverick/m101_recon"))
    parser.add_argument("--source-dir", type=Path,
                        default=Path(__file__).resolve().parent,
                        help="Remedy source tree containing production scripts")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--write-fixed-headers", action="store_true",
                        help="write verified WCS-fixed copies; never modifies originals")
    return parser.parse_args()


def main():
    args = parse_args()
    base_dir = args.base_dir.expanduser().resolve()
    output_dir = (args.output_dir.expanduser().resolve() if args.output_dir else
                  base_dir / "m101_v2_cube_audit")
    output_dir.mkdir(parents=True, exist_ok=True)
    project_wave, wave_provenance = _extract_def_wave(args.source_dir.expanduser().resolve())

    handles, data, headers, paths = _open_products(base_dir)
    try:
        shape = data["SCI"].shape
        if shape[0] != len(project_wave):
            raise ValueError(f"cube has {shape[0]} planes but DEF_WAVE has {len(project_wave)}")
        wcs_audit = _wcs_audit(headers["SCI"], shape[0], project_wave)
        fixed_header, fixed_test = _test_fixed_wcs(headers["SCI"], project_wave)
        cube_validation = _cube_summary(data, shape)
        variance_categories = _variance_categories(data)
        pixel_scale = abs(float(headers["SCI"].get("CD1_1",
                              headers["SCI"].get("CDELT1", 1.0))) * 3600.0)
        if not np.isfinite(pixel_scale) or pixel_scale == 0:
            pixel_scale = 1.0
        background, source, split = _continuum_split(data["SCI"], data["NCONTRIB"], project_wave)
        background_series = _spatial_series(data, project_wave, background)
        source_series = _spatial_series(data, project_wave, source)
        integrated, integrated_flux, integrated_sigma = _integrated_spectrum(data, pixel_scale)
        blank_check = _blank_aperture_check(data, background, pixel_scale, project_wave)
        feature_assessment = {
            "background": _feature_assessment(project_wave, background_series, "low-continuum/background"),
            "bright_source": _feature_assessment(project_wave, source_series, "high-continuum/source"),
        }
        feature_verdict = {key: _feature_verdict(value)
                           for key, value in feature_assessment.items()}
        outputs = {
            "summary_json": str((output_dir / "m101_v2_cube_audit_summary.json").resolve()),
            "summary_text": str((output_dir / "m101_v2_cube_audit_summary.txt").resolve()),
            "wavelength_plot": str((output_dir / "m101_v2_wavelength_diagnostic.png").resolve()),
            "hbeta_plot": str((output_dir / "m101_v2_hbeta_diagnostic.png").resolve()),
            "variance_plot": str((output_dir / "m101_v2_variance_diagnostic.png").resolve()),
        }
        _plot_wavelength_diagnostic(Path(outputs["wavelength_plot"]), headers["SCI"], project_wave)
        _plot_hbeta(Path(outputs["hbeta_plot"]), project_wave, background_series, source_series)
        _plot_variance(Path(outputs["variance_plot"]), project_wave, background_series,
                       cube_validation, variance_categories)

        fixed_write = {"requested": bool(args.write_fixed_headers),
                       "written": False, "fixed_wcs_test": fixed_test}
        if args.write_fixed_headers:
            fixed_write = {"requested": True, "written": True,
                           **_write_fixed_copies(data, headers, paths, output_dir, project_wave)}

        source_audit = _production_source_audit(args.source_dir.expanduser().resolve())
        summary = {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "script": str(Path(__file__).resolve()),
            "base_dir": str(base_dir),
            "source_dir": str(args.source_dir.expanduser().resolve()),
            "products": {key: {"path": str(path), "header_BUNIT": headers[key].get("BUNIT"),
                               "dtype": str(data[key].dtype), "shape": list(data[key].shape)}
                         for key, path in paths.items()},
            "source_audit": source_audit,
            "DEF_WAVE_provenance": wave_provenance,
            "wcs_audit": wcs_audit,
            "fixed_wcs_test": fixed_test,
            "cube_validation": cube_validation,
            "variance_categories": variance_categories,
            "spatial_reconstruction": {
                "gaussian_fwhm_arcsec": GAUSSIAN_FWHM_ARCSEC,
                "fiber_radius_arcsec": FIBER_RADIUS_ARCSEC,
                "fiber_area_arcsec2": float(FIBER_AREA_ARCSEC2),
                "output_pixel_scale_arcsec": pixel_scale,
                "fwhm_in_output_pixels": float(GAUSSIAN_FWHM_ARCSEC / pixel_scale),
                "nominal_gaussian_area_in_output_pixels": float(
                    np.pi * (GAUSSIAN_FWHM_ARCSEC / (2.35482 * pixel_scale))**2),
                "covariance_statement": "VAR is a per-voxel diagonal variance product; summing it assumes independent spatial pixels and does not include covariance introduced by Gaussian reconstruction.",
            },
            "integrated_spectrum": integrated,
            "blank_sky_empirical_check": blank_check,
            "hbeta_diagnostic": {
                "population_split": split,
                "feature_assessment": feature_assessment,
                "feature_verdict": feature_verdict,
                "background_series": {key: value.tolist() for key, value in background_series.items()},
                "source_series": {key: value.tolist() for key, value in source_series.items()},
                "wavelength_A": project_wave.tolist(),
                "diagnostic_only": True,
            },
            "fixed_header_copies": fixed_write,
            "metadata_recommendations": {
                "SCI": "If the documented 1e-17 upstream scale is formally accepted, use BUNIT='1e-17 erg s-1 cm-2 Angstrom-1 arcsec-2'; otherwise retain generic units and explain the scale in COMMENT/HISTORY.",
                "VAR": "Use the square of the adopted SCI unit and state that this is a diagonal per-voxel variance product.",
                "ERROR": "Use the adopted SCI unit and state ERROR=sqrt(VAR).",
                "DQ": "Retain DQ bits 0-4 with their existing definitions.",
                "HISTORY": [
                    "Gaussian reconstruction FWHM=1.8 arcsec; fiber radius=0.75 arcsec; fiber area=pi*(0.75 arcsec)^2.",
                    "Spatial covariance from Gaussian reconstruction is not encoded.",
                    "Exposure-sky, additive calibration-model, and exposure-scale uncertainties are not propagated.",
                ],
            },
            "outputs": outputs,
        }
        summary["summary_text"] = _format_text(summary)
        Path(outputs["summary_json"]).write_text(json.dumps(summary, indent=2, default=_json_value))
        Path(outputs["summary_text"]).write_text(summary["summary_text"])
        print(summary["summary_text"], end="")
    finally:
        for handle in handles.values():
            handle.close()


if __name__ == "__main__":
    main()
