#!/usr/bin/env python3
"""Validate and package the completed current-model M101 VIRUS cube.

This is intentionally a downstream QA/release script.  It never imports the
cube builder, refits a calibration, changes production SCI, or interpolates a
line.  It uses the independent Burrell Schmidt Hbeta images, atomic line
physics, propagated cube uncertainties, and the same shared hardware registry
as the production cube for H5 bookkeeping.

Burrell Schmidt provenance (Garner et al. 2022, ApJ 941, 182): the imaging
pixel scale is approximately 1.45 arcsec pixel^-1.  Hbeta-on has central
wavelength 4875 A, width 82 A, exposure 59 x 1200 s, and flux zeropoint
7.65e-18 erg s^-1 cm^-2 ADU^-1.  Hbeta-off has central wavelength 4757 A,
width 81 A, exposure 55 x 1200 s, and flux zeropoint
7.91e-18 erg s^-1 cm^-2 ADU^-1.  The published independent calibration
agreement is approximately +/-5 percent.  The zeropoints are used only when
the image is explicitly identified as ADU/count data; BUNIT is inspected
first and the decision is recorded in the report.
"""

from argparse import ArgumentParser
import csv
from datetime import datetime, timezone
import glob
import json
from pathlib import Path
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tables
from astropy import units as u
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
from reproject import reproject_interp
from scipy.ndimage import gaussian_filter, label, find_objects

import diagnose_m101_hierarchical as validated_m101
from m101_calibration_utils import robust_location
from m101_hardware_exclusions import HARDWARE_EXCLUSIONS, hardware_excluded


SCRIPT_VERSION = "1.1"
SPEED_OF_LIGHT_KM_S = 299792.458
M101_RA_DEG = 210.800
M101_DEC_DEG = 54.333
SYSTEMIC_VELOCITY_KM_S = 241.0
FIBER_RADIUS_ARCSEC = 0.75
SCI_SCALE = 1e-17
BS_ON_ZEROPOINT = 7.65e-18
BS_OFF_ZEROPOINT = 7.91e-18
BS_ABSOLUTE_CALIBRATION_FRACTION = 0.05
DQ_INSUFFICIENT_SUPPORT = np.uint16(1 << 0)
DQ_VAR_INCOMPLETE = np.uint16(1 << 1)
DQ_OIII5007_VALIDATION = np.uint16(1 << 5)
HARDWARE_REGISTRY_PATH = Path(hardware_excluded.__code__.co_filename).resolve()

LINES = {
    "OII_3727": (3727.0, 10.0),
    "Hdelta_4101": (4101.74, 10.0),
    "Hgamma_4340": (4340.47, 10.0),
    "Hbeta_4861": (4861.33, 10.0),
    "OIII_4959": (4958.91, 10.0),
    "OIII_5007": (5006.84, 10.0),
}


def finite_stats(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"N": 0, "min": None, "p16": None, "median": None,
                "robust_location": None, "p84": None, "max": None,
                "robust_scatter": None}
    med = float(np.median(values))
    with np.errstate(all="ignore"):
        location = float(robust_location(values))
    scatter = 1.4826 * float(np.median(np.abs(values - med)))
    return {"N": int(values.size), "min": float(np.min(values)),
            "p16": float(np.percentile(values, 16)), "median": med,
            "robust_location": location,
            "p84": float(np.percentile(values, 84)), "max": float(np.max(values)),
            "robust_scatter": float(scatter)}


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return value


def reportable(value):
    """Drop diagnostic NumPy image arrays before writing the compact JSON."""
    if isinstance(value, np.ndarray):
        return None
    if isinstance(value, dict):
        return {str(k): reportable(v) for k, v in value.items()
                if not isinstance(v, np.ndarray)}
    if isinstance(value, (list, tuple)):
        return [reportable(v) for v in value]
    return value


def _require_file(path, label):
    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise ValueError("%s does not exist: %s" % (label, path))
    return path


def _file_identity(path):
    path = Path(path).resolve()
    stat = path.stat()
    return {"path": str(path), "size": int(stat.st_size),
            "mtime_ns": int(stat.st_mtime_ns)}


def _persistent_hardware_match(h5_name, group):
    """Classify a group using persistent records from the shared registry."""
    physical = (int(group["specid"]), int(group["ifuslot"]),
                int(group["ifuid"]), str(group["amp"]).upper())
    for record in HARDWARE_EXCLUSIONS:
        if record.get("scope") != "fit_and_cube" or "specid" not in record:
            continue
        if "h5" in record and Path(str(h5_name)).name != record["h5"]:
            continue
        if physical == (int(record["specid"]), int(record["ifuslot"]),
                        int(record["ifuid"]), str(record["amp"]).upper()):
            return True
    return False


def _validate_current_model_summary(summary):
    """Require the provenance fields that identify the accepted cube model."""
    model = summary.get("model", {})
    exposure_scale = summary.get("exposure_scale", {})
    hardware = summary.get("hardware_exclusions", {})
    qa = summary.get("qa", {})
    checks = {
        "model.name": model.get("name") == "current_staged_m101_v1",
        "model.pca_coefficients_applied": model.get("pca_coefficients_applied") is False,
        "exposure_scale.SB_iterations_applied": exposure_scale.get("SB_iterations_applied") == 2,
        "exposure_scale.third_iteration_applied": exposure_scale.get("third_iteration_applied") is False,
        "exposure_scale.sky_reestimated_after_scale": exposure_scale.get(
            "sky_reestimated_after_scale") is False,
        "exposure_scale.errors_scaled": exposure_scale.get("errors_scaled") is True,
        "exposure_scale.sigma_g_added_to_fiber_errors": exposure_scale.get(
            "sigma_g_added_to_fiber_errors") is False,
        "hardware_exclusions.purpose": hardware.get("purpose") == "cube",
        "hardware_exclusions.registry": bool(hardware.get("registry")),
        "qa.automatic_rejection": qa.get("automatic_rejection") is False,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(
            "cube summary is not from the accepted current-model production path: %s"
            % ", ".join(failed))
    return {"checks": checks, "status": "accepted current-model provenance"}


def _read_image_hdu(path, label, require_celestial=True):
    path = _require_file(path, label)
    with fits.open(path, memmap=False) as hdul:
        selected = None
        for hdu in hdul:
            if hdu.data is not None and np.ndim(hdu.data) == 2:
                selected = hdu
                break
        if selected is None:
            raise ValueError("%s has no 2D image HDU: %s" % (label, path))
        data = np.asarray(selected.data, dtype=float).copy()
        header = selected.header.copy()
    try:
        wcs = WCS(header).celestial
    except Exception as exc:
        raise ValueError("could not construct celestial WCS for %s" % path) from exc
    if require_celestial and not wcs.has_celestial:
        raise ValueError("%s has no celestial WCS: %s" % (label, path))
    scale = np.asarray(proj_plane_pixel_scales(wcs), dtype=float) * 3600.0
    if scale.size != 2 or not np.all(np.isfinite(scale)) or np.any(scale <= 0):
        raise ValueError("%s has no positive celestial pixel scale: %s" % (label, path))
    return data, header, wcs, float(np.mean(scale)), path


def _image_unit_decision(header, mode, label, allow_adu=True):
    bunit = str(header.get("BUNIT", "")).strip()
    normalized = bunit.lower().replace(" ", "").replace("/", "")
    flux_markers = ("erg", "cm-2", "cm^-2", "s-1", "s^-1", "ergs")
    adu_markers = ("adu", "count", "counts", "dn", "datanumber")
    clearly_flux = "erg" in normalized and ("cm" in normalized or "s" in normalized)
    clearly_adu = any(marker in normalized for marker in adu_markers)
    if mode == "flux":
        return {"mode": "flux", "bunit": bunit, "zeropoint_applied": False,
                "interpretation": "user specified already-calibrated flux"}
    if mode == "adu":
        if not allow_adu:
            raise ValueError("%s cannot use ADU mode without a published zeropoint" % label)
        return {"mode": "adu", "bunit": bunit, "zeropoint_applied": True,
                "interpretation": "user specified ADU/count image"}
    if clearly_flux:
        return {"mode": "flux", "bunit": bunit, "zeropoint_applied": False,
                "interpretation": "BUNIT indicates calibrated erg/s/cm2 values"}
    if clearly_adu:
        if not allow_adu:
            raise ValueError("%s is ADU-like but no zeropoint is available" % label)
        return {"mode": "adu", "bunit": bunit, "zeropoint_applied": True,
                "interpretation": "BUNIT indicates ADU/count values"}
    raise ValueError("could not identify %s units from BUNIT=%r; use --hb-image-units" %
                     (label, bunit))


def _read_filter_summary(path):
    path = _require_file(path, "filter curve")
    try:
        table = Table.read(path, format="ascii")
        names = {str(name).lower(): name for name in table.colnames}
        wave_name = next((names[n] for n in ("wavelength", "wave", "lambda") if n in names), None)
        response_name = next((names[n] for n in ("r", "response", "throughput") if n in names), None)
        if wave_name is None or response_name is None:
            raise ValueError("filter needs wavelength and response columns")
        wave = np.asarray(table[wave_name], dtype=float)
        response = np.asarray(table[response_name], dtype=float)
    except Exception as exc:
        raise ValueError("could not read filter %s: %s" % (path, exc)) from exc
    good = np.isfinite(wave) & np.isfinite(response) & (response > 0)
    if not np.any(good):
        raise ValueError("filter has no positive finite response: %s" % path)
    wave, response = wave[good], response[good]
    integral = float(np.trapz(response, wave))
    center = float(np.trapz(wave * response, wave) / integral)
    return {"path": str(path), "positive_samples": int(good.sum()),
            "response_weighted_center_A": center,
            "equivalent_width_A": integral / float(np.max(response))}


def _cube_wavelength(header, nplane):
    try:
        wcs = WCS(header)
        axis = int(wcs.wcs.spec)
        if axis < 0:
            axis = 2
        card_axis = axis + 1
        crval_key = "CRVAL%d" % card_axis
        crpix_key = "CRPIX%d" % card_axis
        cunit_key = "CUNIT%d" % card_axis
        cdelt_key = "CDELT%d" % card_axis
        cd_key = "CD%d_%d" % (card_axis, card_axis)
        if not all(key in header for key in (crval_key, crpix_key, cunit_key)):
            raise ValueError("spectral WCS lacks explicit linear axis cards")
        # The production writer emits a linear FITS spectral axis.  Reading
        # these cards directly avoids a WCSLIB unit-normalization issue for
        # CTYPE=WAVE/CUNIT=Angstrom, which can otherwise halve CDELT3.
        delta = header.get(cd_key, header.get(cdelt_key))
        if delta is None:
            raise ValueError("spectral WCS lacks CDELT/CD diagonal")
        native = (float(header[crval_key])
                  + (np.arange(nplane, dtype=float) + 1.0
                     - float(header[crpix_key])) * float(delta))
        unit = u.Unit(str(header[cunit_key]))
        if unit is None or not unit.is_equivalent(u.AA):
            raise ValueError("spectral WCS unit is not convertible to Angstrom")
        wave = np.asarray(native * unit.to(u.AA), dtype=float)
    except Exception as exc:
        raise ValueError("could not evaluate spectral WCS") from exc
    if wave.size != nplane or not np.all(np.isfinite(wave)):
        raise ValueError("spectral WCS did not produce one finite wavelength per plane")
    step = np.diff(wave)
    if step.size and (not np.allclose(step, np.median(step), rtol=1e-5, atol=1e-5)):
        raise ValueError("spectral WCS is not linear")
    expected = np.asarray(validated_m101.DEF_WAVE, dtype=float)
    if wave.size != expected.size or not np.allclose(
            wave, expected, rtol=0.0, atol=1.0e-6):
        raise ValueError(
            "spectral WCS does not match the current-model DEF_WAVE grid: "
            "got %d planes %.6f--%.6f A at %.6f A/pixel; expected %d planes "
            "%.6f--%.6f A at %.6f A/pixel" % (
                wave.size, wave[0], wave[-1], np.median(step),
                expected.size, expected[0], expected[-1],
                np.median(np.diff(expected))))
    return wave


def _read_cube_products(args):
    paths = {"SCI": _require_file(args.cube, "SCI cube"),
             "VAR": _require_file(args.variance, "variance cube"),
             "ERROR": _require_file(args.error, "error cube"),
             "DQ": _require_file(args.dq, "DQ cube"),
             "COVERAGE": _require_file(args.coverage, "coverage cube"),
             "NCONTRIB": _require_file(args.ncontrib, "NCONTRIB cube")}
    arrays = {}
    headers = {}
    for key, path in paths.items():
        with fits.open(path, memmap=False) as hdul:
            hdu = hdul[0]
            if hdu.data is None or np.ndim(hdu.data) != 3:
                raise ValueError("%s is not a 3D image cube: %s" % (key, path))
            arrays[key] = np.asarray(hdu.data).copy()
            headers[key] = hdu.header.copy()
    shape = arrays["SCI"].shape
    dimension_checks = {key: tuple(value.shape) for key, value in arrays.items()}
    if any(value.shape != shape for value in arrays.values()):
        raise ValueError("SCI, VAR, ERROR, DQ, COVERAGE, and NCONTRIB dimensions differ: %s" %
                         dimension_checks)
    header = headers["SCI"]
    wave = _cube_wavelength(header, shape[0])
    celestial = WCS(header).celestial
    scales = np.asarray(proj_plane_pixel_scales(celestial), dtype=float) * 3600.0
    if not np.all(np.isfinite(scales)) or np.any(scales <= 0):
        raise ValueError("cube spatial WCS has no positive pixel scale")
    pixel_scale = float(np.mean(scales))
    return arrays, headers, wave, celestial, pixel_scale, paths


def _validate_products(arrays, wave, pixel_scale):
    sci, var, error = arrays["SCI"], arrays["VAR"], arrays["ERROR"]
    dq, coverage, ncontrib = (arrays["DQ"], arrays["COVERAGE"], arrays["NCONTRIB"])
    finite_sci = np.isfinite(sci)
    valid_sci = finite_sci & ((dq & DQ_INSUFFICIENT_SUPPORT) == 0)
    finite_var = np.isfinite(var)
    finite_error = np.isfinite(error)
    common_error = finite_var & finite_error & (var >= 0)
    error_expected = np.sqrt(np.where(var >= 0, var, np.nan))
    error_mismatch = common_error & ~np.isclose(error, error_expected, rtol=3e-4, atol=1e-7)
    coverage_finite = np.isfinite(coverage)
    coverage_in_range = coverage_finite & (coverage >= -1e-6) & (coverage <= 1.0 + 1e-6)
    dq_integer = np.issubdtype(dq.dtype, np.integer)
    ncontrib_integer = np.issubdtype(ncontrib.dtype, np.integer)
    semantics_mismatch = ((dq & DQ_INSUFFICIENT_SUPPORT) == 0) != (ncontrib >= 2)
    bit_counts = {}
    for bit in range(16):
        count = int(np.sum((dq & np.uint16(1 << bit)) != 0))
        bit_counts[str(bit)] = count
    nc_values, nc_counts = np.unique(ncontrib, return_counts=True)
    ncontrib_distribution = {str(int(v)): int(c) for v, c in zip(nc_values, nc_counts)}
    return {
        "cube_dimensions": list(sci.shape),
        "wavelength": {"start_A": float(wave[0]), "end_A": float(wave[-1]),
                       "step_A": float(np.median(np.diff(wave))), "planes": int(wave.size)},
        "spatial_pixel_scale_arcsec": pixel_scale,
        "finite_SCI_voxels": int(finite_sci.sum()),
        "valid_SCI_voxels": int(valid_sci.sum()),
        "finite_VAR_voxels": int(finite_var.sum()),
        "finite_VAR_fraction": float(finite_var.mean()),
        "VAR_nonnegative_finite": bool(np.all(var[finite_var] >= 0)),
        "ERROR_sqrt_VAR_mismatch_voxels": int(error_mismatch.sum()),
        "ERROR_matches_sqrt_VAR": bool(not np.any(error_mismatch)),
        "DQ_integer": bool(dq_integer), "NCONTRIB_integer": bool(ncontrib_integer),
        "COVERAGE_finite_voxels": int(coverage_finite.sum()),
        "COVERAGE_in_expected_0_1_range": bool(np.all(coverage_in_range[coverage_finite])),
        "COVERAGE_out_of_range_voxels": int(np.sum(coverage_finite & ~coverage_in_range)),
        "DQ_counts_by_bit": bit_counts,
        "NCONTRIB_distribution": ncontrib_distribution,
        "DQ_bit0_NCONTRIB_semantics_mismatch_voxels": int(semantics_mismatch.sum()),
        "DQ_bit0_semantics_agree": bool(not np.any(semantics_mismatch)),
        "DQ_bit5_input_voxels": int(np.sum((dq & DQ_OIII5007_VALIDATION) != 0)),
        "SCI_BUNIT": str(arrays.get("_SCI_HEADER", {}).get("BUNIT", "")),
    }


def _science_unit_scale(header):
    """Return the physical scale documented for the current cube SCI values."""
    bunit = str(header.get("BUNIT", "")).lower().replace(" ", "")
    if "1e-17" in bunit or "10^-17" in bunit or "10**-17" in bunit:
        return 1e-17, "SCI BUNIT explicitly carries a 1e-17 scale"
    if "erg" in bunit and "cm" in bunit:
        return 1.0, "SCI BUNIT already states physical cgs flux-density units"
    if not bunit:
        return 1e-17, ("SCI BUNIT is absent; using the existing project Quick Reduction "
                       "convention documented in CUBE_README.md: cube values are in 1e-17 cgs units")
    raise ValueError("unrecognized SCI BUNIT=%r; cannot determine physical scaling" % header.get("BUNIT"))


def _spectrum_accounting(h5_glob):
    paths = sorted(Path(path).expanduser().resolve() for path in glob.glob(h5_glob))
    if not paths:
        raise ValueError("no H5 files matched --h5-glob=%s" % h5_glob)
    total_input = total_masked = total_historical = total_persistent = 0
    total_masked_groups = total_historical_groups = total_persistent_groups = 0
    amplifier_observations = shots = 0
    per_file = []
    exposure_ids = set()
    for path in paths:
        with tables.open_file(path, mode="r") as h5:
            if not {"Info", "Survey"}.issubset(set(h5.root._v_children)):
                raise ValueError("H5 lacks Info/Survey: %s" % path)
            info = h5.root.Info
            groups, labels = validated_m101.build_groups(info)
            if labels.shape[0] != info.nrows:
                raise ValueError("validated exposure labels do not match Info rows: %s" % path)
            h5_name = path.name
            h5_date = h5_name[:8]
            if len(h5_date) != 8 or not h5_date.isdigit():
                raise ValueError("H5 basename does not begin with a UT date: %s" % h5_name)
            masked = np.zeros(info.nrows, dtype=bool)
            persistent_masked = np.zeros(info.nrows, dtype=bool)
            historical_masked = np.zeros(info.nrows, dtype=bool)
            masked_groups = historical_groups = persistent_groups = 0
            for group in groups:
                excluded = hardware_excluded(
                    date=h5_date, h5=h5_name, specid=group["specid"],
                    ifuslot=group["ifuslot"], ifuid=group["ifuid"],
                    amp=group["amp"], purpose="cube")
                if not excluded:
                    continue
                indices = np.asarray(group["indices"], dtype=int)
                masked[indices] = True
                masked_groups += 1
                if _persistent_hardware_match(h5_name, group):
                    persistent_masked[indices] = True
                    persistent_groups += 1
                else:
                    historical_masked[indices] = True
                    historical_groups += 1
            survey_exposures = [int(row["exp"]) for row in h5.root.Survey]
            if len(set(survey_exposures)) != len(survey_exposures):
                raise ValueError("duplicate Survey exposure in %s" % path)
            total_input += int(info.nrows)
            total_masked += int(masked.sum())
            total_historical += int(historical_masked.sum())
            total_persistent += int(persistent_masked.sum())
            total_masked_groups += masked_groups
            total_historical_groups += historical_groups
            total_persistent_groups += persistent_groups
            amplifier_observations += len(groups)
            shots += len(survey_exposures)
            exposure_ids.update((path.name, exp) for exp in survey_exposures)
            per_file.append({"file": str(path), "info_rows": int(info.nrows),
                             "hardware_excluded_fiber_rows": int(masked.sum()),
                             "historical_hardware_excluded_fiber_rows": int(historical_masked.sum()),
                             "persistent_hardware_excluded_fiber_rows": int(persistent_masked.sum()),
                             "hardware_excluded_groups": int(masked_groups),
                             "historical_hardware_excluded_groups": int(historical_groups),
                             "persistent_hardware_excluded_groups": int(persistent_groups),
                             # Retain the old field name for consumers of the
                             # pre-registry accounting output.
                             "hardware_date_masked": int(historical_masked.sum()),
                             "retained_rows": int(info.nrows - masked.sum()),
                             "amplifier_observations": len(groups),
                             "survey_exposures": survey_exposures})
    return {"N_H5": len(paths), "N_input_fiber_spectra": total_input,
            "N_hardware_excluded_fiber_rows": total_masked,
            "N_historical_hardware_excluded_fiber_rows": total_historical,
            "N_persistent_hardware_excluded_fiber_rows": total_persistent,
            "N_hardware_excluded_groups": total_masked_groups,
            "N_historical_hardware_excluded_groups": total_historical_groups,
            "N_persistent_hardware_excluded_groups": total_persistent_groups,
            # Compatibility alias; this now means historical-only exclusions.
            "N_hardware_date_masked": total_historical,
            "N_retained_fiber_spectra": total_input - total_masked,
            "N_exposures_shots": shots, "N_amplifier_observations": amplifier_observations,
            "hardware_registry": _file_identity(HARDWARE_REGISTRY_PATH),
            "hardware_registry_purpose": "cube",
            "hardware_registry_records": int(len(HARDWARE_EXCLUSIONS)),
            "persistent_hardware_registry_records": int(sum(
                "specid" in record for record in HARDWARE_EXCLUSIONS)),
            "input_h5_files": [str(p) for p in paths], "per_file": per_file}


def _line_center(rest):
    return float(rest) * (1.0 + SYSTEMIC_VELOCITY_KM_S / SPEED_OF_LIGHT_KM_S)


def _line_map(sci, var, dq, wave, rest_center, half_width, unit_scale=SCI_SCALE):
    """Local weighted linear continuum and propagated line integral."""
    center = _line_center(rest_center)
    line_sel = np.abs(wave - center) <= half_width
    blue = (wave >= center - 35.0) & (wave <= center - 15.0)
    red = (wave >= center + 15.0) & (wave <= center + 35.0)
    cont_sel = blue | red
    if line_sel.sum() < 2 or cont_sel.sum() < 3:
        raise ValueError("line window is not represented on the wavelength grid at %.2f A" % center)
    valid = np.isfinite(sci) & np.isfinite(var) & (var >= 0) & ((dq & DQ_INSUFFICIENT_SUPPORT) == 0)
    # Positive variance is needed for a meaningful weighted continuum fit.
    fit_valid = valid & (var > 0)
    x = wave - center
    xc = x[cont_sel]
    y = sci[cont_sel]
    v = var[cont_sel]
    good = fit_valid[cont_sel]
    weight = np.zeros_like(v, dtype=float)
    np.divide(1.0, np.maximum(v, np.finfo(float).tiny), out=weight, where=good)
    s00 = np.sum(weight, axis=0)
    s01 = np.sum(weight * xc[:, None, None], axis=0)
    s11 = np.sum(weight * xc[:, None, None] ** 2, axis=0)
    r0 = np.sum(weight * np.where(good, y, 0.0), axis=0)
    r1 = np.sum(weight * np.where(good, y, 0.0) * xc[:, None, None], axis=0)
    det = s00 * s11 - s01 * s01
    fit_good = (s00 > 0) & (det > 0) & np.isfinite(det)
    a = np.full(s00.shape, np.nan, dtype=float)
    b = np.full(s00.shape, np.nan, dtype=float)
    a[fit_good] = (r0[fit_good] * s11[fit_good] - r1[fit_good] * s01[fit_good]) / det[fit_good]
    b[fit_good] = (r1[fit_good] * s00[fit_good] - r0[fit_good] * s01[fit_good]) / det[fit_good]
    line_x = x[line_sel]
    line_y = sci[line_sel]
    line_v = var[line_sel]
    line_good = valid[line_sel] & fit_good[None, :, :]
    dl = float(np.median(np.diff(wave)))
    residual = line_y - (a[None, :, :] + b[None, :, :] * line_x[:, None, None])
    flux = np.sum(np.where(line_good, residual, 0.0), axis=0) * dl * unit_scale
    flux[~fit_good] = np.nan
    nline = np.sum(line_good, axis=0)
    flux[nline == 0] = np.nan
    cov00 = np.full(a.shape, np.nan)
    cov01 = np.full(a.shape, np.nan)
    cov11 = np.full(a.shape, np.nan)
    cov00[fit_good] = s11[fit_good] / det[fit_good]
    cov01[fit_good] = -s01[fit_good] / det[fit_good]
    cov11[fit_good] = s00[fit_good] / det[fit_good]
    line_sum_v = np.sum(np.where(line_good, line_v, 0.0), axis=0) * dl * dl * unit_scale ** 2
    line_sum_1 = np.sum(np.where(line_good, dl, 0.0), axis=0)
    line_sum_x = np.sum(np.where(line_good, dl * line_x[:, None, None], 0.0), axis=0)
    continuum_variance = (line_sum_1 ** 2 * cov00 + 2 * line_sum_1 * line_sum_x * cov01
                          + line_sum_x ** 2 * cov11) * unit_scale ** 2
    line_variance = line_sum_v + continuum_variance
    line_variance[~fit_good | (nline == 0)] = np.nan
    snr = np.full(flux.shape, np.nan)
    positive_var = line_variance > 0
    snr[positive_var] = flux[positive_var] / np.sqrt(line_variance[positive_var])
    continuum = a * unit_scale
    ew = np.full(flux.shape, np.nan)
    positive_continuum = np.isfinite(continuum) & (continuum > 0)
    ew[positive_continuum] = flux[positive_continuum] / continuum[positive_continuum]
    return {"flux": flux.astype(np.float32), "variance": line_variance.astype(np.float32),
            "snr": snr.astype(np.float32), "continuum": continuum.astype(np.float32),
            "ew": ew.astype(np.float32), "center_observed_A": center,
            "line_planes": np.flatnonzero(line_sel).tolist(),
            "sideband_planes": np.flatnonzero(cont_sel).tolist()}


def _map_header(celestial, bunit, title):
    header = celestial.to_header() if hasattr(celestial, "to_header") else celestial.copy()
    header["BUNIT"] = bunit
    header["HISTORY"] = title
    return header


def _write_image(path, data, header, dtype=np.float32):
    fits.PrimaryHDU(np.asarray(data, dtype=dtype), header=header).writeto(path, overwrite=True)


def _calibrate_external(data, header, mode, zero_point, label, allow_adu=True):
    decision = _image_unit_decision(header, mode, label, allow_adu=allow_adu)
    finite = np.isfinite(data)
    result = np.asarray(data, dtype=float).copy()
    if decision["zeropoint_applied"]:
        result *= float(zero_point)
    result[~finite] = np.nan
    decision["zero_point_erg_s_cm2_per_ADU"] = float(zero_point) if decision["zeropoint_applied"] else None
    return result, decision


def _outer_sky(data, wcs, radius_arcmin):
    yy, xx = np.indices(data.shape, dtype=float)
    try:
        ra, dec = wcs.pixel_to_world_values(xx, yy)
    except Exception:
        return data.copy(), {"applied": False, "reason": "WCS coordinate evaluation failed"}
    radius = np.hypot((ra - M101_RA_DEG) * np.cos(np.deg2rad(M101_DEC_DEG)), dec - M101_DEC_DEG) * 60.0
    outer = (radius >= float(radius_arcmin)) & np.isfinite(data)
    values = data[outer]
    if values.size < 100:
        return data.copy(), {"applied": False, "reason": "fewer than 100 finite outer pixels",
                             "N_outer": int(values.size), "offset": 0.0}
    med = float(np.median(values))
    scatter = 1.4826 * float(np.median(np.abs(values - med)))
    corrected = data - med
    corrected[~np.isfinite(data)] = np.nan
    return corrected, {"applied": True, "N_outer": int(values.size), "offset": med,
                       "robust_scatter": scatter, "radius_arcmin": float(radius_arcmin)}


def _surface_brightness(flux_per_pixel, pixel_scale):
    return flux_per_pixel / float(pixel_scale ** 2)


def _pixel_area_arcsec2(wcs):
    scales = np.asarray(proj_plane_pixel_scales(wcs), dtype=float) * 3600.0
    return float(abs(scales[0] * scales[1]))


def _reproject_sb(sb, source_wcs, target_wcs, target_shape):
    result, footprint = reproject_interp((sb, source_wcs), target_wcs,
                                         shape_out=target_shape, order="bilinear")
    result[footprint <= 0] = np.nan
    return result, footprint


def _smooth_nan(data, sigma):
    if not np.isfinite(sigma) or sigma <= 0:
        return np.asarray(data, dtype=float).copy()
    finite = np.isfinite(data)
    numerator = gaussian_filter(np.where(finite, data, 0.0), sigma=sigma, mode="nearest")
    denominator = gaussian_filter(finite.astype(float), sigma=sigma, mode="nearest")
    result = np.full(data.shape, np.nan, dtype=float)
    good = denominator > 1e-6
    result[good] = numerator[good] / denominator[good]
    return result


def _robust_through_zero_scale(x, y):
    good = np.isfinite(x) & np.isfinite(y) & (x > 0)
    if good.sum() < 3:
        return np.nan, np.zeros_like(good)
    keep = good.copy()
    for _ in range(5):
        slope = float(np.sum(x[keep] * y[keep]) / np.sum(x[keep] ** 2))
        residual = y - slope * x
        scatter = 1.4826 * np.median(np.abs(residual[keep] - np.median(residual[keep])))
        if not np.isfinite(scatter) or scatter <= 0:
            break
        new_keep = good & (np.abs(residual) <= 4.0 * scatter)
        if new_keep.sum() == keep.sum():
            break
        keep = new_keep
    return slope, keep


def _comparison_stats(virus, external, variance, ew, snr_min, min_ew=None):
    good = (np.isfinite(virus) & np.isfinite(external) & np.isfinite(variance)
            & (variance > 0) & (external > 0) & (virus > 0)
            & (virus / np.sqrt(variance) >= snr_min))
    if min_ew is not None:
        good &= np.isfinite(ew) & (ew >= min_ew)
    ratio = np.full(virus.shape, np.nan, dtype=float)
    ratio[good] = virus[good] / external[good]
    result = finite_stats(ratio)
    result["mask_count"] = int(good.sum())
    return result, good, ratio


def _connected_regions(mask, significance, minimum_area):
    components, n = label(mask, structure=np.ones((3, 3), dtype=bool))
    rows = []
    accepted = np.zeros(mask.shape, dtype=bool)
    for component_id, slc in enumerate(find_objects(components), start=1):
        if slc is None:
            continue
        pixels = components[slc] == component_id
        area = int(pixels.sum())
        if area < int(minimum_area):
            continue
        values = significance[slc][pixels]
        y, x = np.where(pixels)
        accepted[slc][pixels] = True
        rows.append({"region_id": len(rows) + 1, "area_pixels": area,
                     "x_min": int(x.min() + slc[1].start), "x_max": int(x.max() + slc[1].start),
                     "y_min": int(y.min() + slc[0].start), "y_max": int(y.max() + slc[0].start),
                     "median_significance": float(np.median(values)),
                     "minimum_significance": float(np.min(values))})
    return accepted, rows


def _write_csv(path, rows, fieldnames=None):
    rows = list(rows)
    if fieldnames is None:
        fieldnames = list(rows[0]) if rows else []
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_line_products(output_dir, line_products, map_header):
    for name, result in line_products.items():
        _write_image(output_dir / (name + "_flux.fits"), result["flux"],
                     _map_header(map_header, "erg s-1 cm-2 arcsec-2", name + " local-continuum line flux"))
        _write_image(output_dir / (name + "_variance.fits"), result["variance"],
                     _map_header(map_header, "(erg s-1 cm-2 arcsec-2)^2", name + " propagated line variance"))
        _write_image(output_dir / (name + "_snr.fits"), result["snr"],
                     _map_header(map_header, "dimensionless", name + " line S/N"))


def _calzetti_k(wavelength_A, rv=4.05):
    x = np.asarray(wavelength_A, dtype=float) / 1e4
    result = np.empty_like(x)
    short = x < 0.63
    result[short] = 2.659 * (-2.156 + 1.509 / x[short] - 0.198 / x[short] ** 2 + 0.011 / x[short] ** 3) + rv
    result[~short] = 2.659 * (-1.857 + 1.040 / x[~short]) + rv
    return result


def _balmer_validation(line_products, args):
    hb, hg, hd = (line_products["Hbeta_4861"], line_products["Hgamma_4340"],
                  line_products["Hdelta_4101"])
    good = (np.isfinite(hb["flux"]) & np.isfinite(hg["flux"]) & np.isfinite(hd["flux"])
            & np.isfinite(hb["variance"]) & np.isfinite(hg["variance"]) & np.isfinite(hd["variance"])
            & (hb["flux"] > 0) & (hg["flux"] > 0) & (hd["flux"] > 0)
            & (hb["snr"] >= args.hbeta_snr) & (hg["snr"] >= args.hgamma_snr)
            & (hd["snr"] >= args.hdelta_snr))
    rb = hg["flux"] / hb["flux"]
    rd = hd["flux"] / hb["flux"]
    with np.errstate(divide="ignore", invalid="ignore"):
        vb = hg["variance"] / hb["flux"] ** 2 + hg["flux"] ** 2 * hb["variance"] / hb["flux"] ** 4
        vd = hd["variance"] / hb["flux"] ** 2 + hd["flux"] ** 2 * hb["variance"] / hb["flux"] ** 4
    k = _calzetti_k(np.array([4340.47, 4101.74, 4861.33]), args.calzetti_rv)
    with np.errstate(divide="ignore", invalid="ignore"):
        e_gamma = -2.5 * np.log10(rb / args.caseb_hgamma_hbeta) / (k[0] - k[2])
        e_delta = -2.5 * np.log10(rd / args.caseb_hdelta_hbeta) / (k[1] - k[2])
    e_gamma_sigma = 2.5 / np.log(10.0) * np.sqrt(vb) / rb / abs(k[0] - k[2])
    e_delta_sigma = 2.5 / np.log(10.0) * np.sqrt(vd) / rd / abs(k[1] - k[2])
    difference = e_gamma - e_delta
    difference_sigma = np.sqrt(e_gamma_sigma ** 2 + e_delta_sigma ** 2)
    locus_good = good & np.isfinite(difference)
    within_1 = locus_good & (np.abs(difference) <= difference_sigma)
    within_2 = locus_good & (np.abs(difference) <= 2.0 * difference_sigma)
    return {"N_high_quality_samples": int(good.sum()),
            "median_Hgamma_over_Hbeta": finite_stats(rb[good]),
            "median_Hdelta_over_Hbeta": finite_stats(rd[good]),
            "caseB_intrinsic_ratios": {"Hgamma_Hbeta": args.caseb_hgamma_hbeta,
                                        "Hdelta_Hbeta": args.caseb_hdelta_hbeta},
            "attenuation_law": "Calzetti et al. 2000", "calzetti_Rv": args.calzetti_rv,
            "systemic_velocity_km_s": SYSTEMIC_VELOCITY_KM_S,
            "E_BV_gamma": finite_stats(e_gamma[good]), "E_BV_delta": finite_stats(e_delta[good]),
            "E_BV_gamma_minus_delta": finite_stats(difference[locus_good]),
            "fraction_consistent_1sigma": float(within_1[locus_good].mean()) if locus_good.any() else None,
            "fraction_consistent_2sigma": float(within_2[locus_good].mean()) if locus_good.any() else None,
            "arrays": {"good": good, "rb": rb, "rd": rd, "difference": difference,
                       "difference_sigma": difference_sigma}}


def _oiii_validation(line_products, args):
    a, b = line_products["OIII_5007"], line_products["OIII_4959"]
    good = (np.isfinite(a["flux"]) & np.isfinite(b["flux"]) & np.isfinite(a["variance"])
            & np.isfinite(b["variance"]) & (b["flux"] > 0) & (b["snr"] >= args.oiii_4959_snr)
            & (a["snr"] > 0))
    ratio = np.full(a["flux"].shape, np.nan, dtype=float)
    ratio[good] = a["flux"][good] / b["flux"][good]
    ratio_vs_snr, ratio_vs_surface_brightness = _oiii_ratio_signal_summaries(
        ratio, b["snr"], b["flux"], good)
    residual = a["flux"] - args.oiii_ratio * b["flux"]
    residual_variance = a["variance"] + args.oiii_ratio ** 2 * b["variance"]
    residual_significance = np.full(residual.shape, np.nan, dtype=float)
    positive = good & (residual_variance > 0)
    residual_significance[positive] = residual[positive] / np.sqrt(residual_variance[positive])
    candidate = positive & (residual_significance <= args.oiii_deficit_sigma)
    coherent, regions = _connected_regions(candidate, residual_significance, args.min_defect_area)
    return {"N_high_SN_4959_samples": int(good.sum()), "ratio": ratio,
            "median_5007_over_4959": finite_stats(ratio[good]),
            "adopted_branching_ratio": args.oiii_ratio,
            "residual_significance": residual_significance,
            "candidate_mask": candidate, "coherent_mask": coherent,
            "coherent_regions": regions,
            "DQ_voxels_assigned": int(coherent.sum()) * len(a["line_planes"]),
            "ratio_vs_4959_snr": ratio_vs_snr,
            "ratio_vs_4959_surface_brightness": ratio_vs_surface_brightness,
            "status": ("N coherent candidate regions detected and requiring inspection" if regions
                       else "no coherent 5007 residual-deficit regions detected")}


def _oiii_ratio_summary(values, reference_ratio=2.98):
    stats = finite_stats(values)
    result = {key: stats[key] for key in
              ("N", "median", "robust_location", "robust_scatter", "p16", "p84")}
    location = stats["robust_location"]
    result["robust_location_minus_2_98"] = (
        float(location - reference_ratio) if location is not None else None)
    return result


def _oiii_ratio_signal_summaries(ratio, snr, surface_brightness, good):
    """Summarize the existing valid OIII population by signal, without refitting."""
    snr_bins = ((5.0, 7.5), (7.5, 10.0), (10.0, 20.0),
                (20.0, 50.0), (50.0, None))
    snr_rows = []
    for lower, upper in snr_bins:
        selected = good & (snr >= lower)
        if upper is not None:
            selected &= snr < upper
        row = {"snr_lower": lower, "snr_upper": upper}
        row.update(_oiii_ratio_summary(ratio[selected]))
        snr_rows.append(row)

    valid_ratio = np.asarray(ratio[good], dtype=float)
    valid_surface = np.asarray(surface_brightness[good], dtype=float)
    surface_rows = []
    if valid_surface.size:
        order = np.argsort(valid_surface, kind="mergesort")
        for quantile_index, ranks in enumerate(np.array_split(np.arange(order.size), 5), start=1):
            if ranks.size == 0:
                row = {"quantile_bin": quantile_index,
                       "surface_brightness_lower": None,
                       "surface_brightness_upper": None}
                row.update(_oiii_ratio_summary([]))
                surface_rows.append(row)
                continue
            selected_values = valid_ratio[order[ranks]]
            surface_values = valid_surface[order[ranks]]
            row = {"quantile_bin": quantile_index,
                   "surface_brightness_lower": float(np.min(surface_values)),
                   "surface_brightness_upper": float(np.max(surface_values))}
            row.update(_oiii_ratio_summary(selected_values))
            surface_rows.append(row)
    else:
        for quantile_index in range(1, 6):
            row = {"quantile_bin": quantile_index,
                   "surface_brightness_lower": None,
                   "surface_brightness_upper": None}
            row.update(_oiii_ratio_summary([]))
            surface_rows.append(row)
    return snr_rows, surface_rows


def _hbeta_validation(virus, virus_var, virus_snr, ew, external, external_support, args,
                      pixel_scale, output_dir, map_header):
    sigma = args.comparison_smoothing_fwhm / 2.354820045 / pixel_scale
    virus_sm = _smooth_nan(virus, sigma)
    ext_sm = _smooth_nan(external, sigma)
    var_sm = _smooth_nan(virus_var, sigma) / max(1.0, (2.0 * np.pi * sigma ** 2))
    ext_unc = BS_ABSOLUTE_CALIBRATION_FRACTION * np.abs(ext_sm)
    common = np.isfinite(virus_sm) & np.isfinite(ext_sm) & (ext_sm > 0) & (var_sm > 0)
    scale, scale_keep = _robust_through_zero_scale(ext_sm.ravel(), virus_sm.ravel())
    residual = virus_sm - scale * ext_sm
    residual_sigma = np.sqrt(np.maximum(var_sm, 0) + (scale * ext_unc) ** 2)
    residual_significance = residual / residual_sigma
    high_external = common & (ext_sm / np.maximum(ext_unc, np.finfo(float).tiny) >= args.hb_external_snr)
    expected = high_external & (scale * ext_sm / np.sqrt(np.maximum(var_sm, np.finfo(float).tiny)) >= args.hbeta_snr)
    candidate = expected & (residual_significance <= args.hbeta_deficit_sigma)
    coherent, regions = _connected_regions(candidate, residual_significance, args.min_defect_area)
    _write_image(output_dir / "hbeta_deficit_candidates.fits", coherent.astype(np.uint8),
                 _map_header(map_header, "bit mask", "coherent Hbeta deficit candidates; diagnostic only"), dtype=np.uint8)
    rows = []
    for row in regions:
        row["diagnostic"] = "Hbeta external morphology residual"
        rows.append(row)
    _write_csv(output_dir / "hbeta_deficit_candidates.csv", rows,
               ["region_id", "area_pixels", "x_min", "x_max", "y_min", "y_max",
                "median_significance", "minimum_significance", "diagnostic"])
    all_stats, all_mask, all_ratio = _comparison_stats(virus_sm, ext_sm, var_sm, ew, args.hbeta_snr)
    ew_stats, ew_mask, ew_ratio = _comparison_stats(virus_sm, ext_sm, var_sm, ew,
                                                    args.hbeta_snr, args.min_hbeta_ew)
    return {"virus_smoothed": virus_sm, "external_smoothed": ext_sm,
            "residual": residual, "residual_significance": residual_significance,
            "global_robust_through_zero_morphology_scale": scale,
            "morphology_scale_samples": int(scale_keep.sum()),
            "coherent_regions": regions, "coherent_mask": coherent,
            "status": ("N coherent candidate regions detected and requiring inspection" if regions
                       else "no coherent Hbeta bright-line deficit population detected"),
            "absolute_flux_all_high_SN": all_stats, "absolute_flux_high_EW": ew_stats,
            "absolute_flux_calibration_reference_fraction": BS_ABSOLUTE_CALIBRATION_FRACTION,
            "raw_external_narrowband_excess_is_not_absorption_corrected": True,
            "arrays": {"all_mask": all_mask, "ew_mask": ew_mask, "all_ratio": all_ratio,
                       "ew_ratio": ew_ratio, "external_support": external_support,
                       "virus_snr": virus_snr}}


def _plot_hbeta(hb, map_header, output_dir):
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    all_stats = hb["absolute_flux_all_high_SN"]
    ew_stats = hb["absolute_flux_high_EW"]
    images = [(hb["external_smoothed"], "External Hbeta SB"),
              (hb["virus_smoothed"], "VIRUS Hbeta"),
              (hb["residual"], "VIRUS - morphology-scaled external"),
              (hb["arrays"]["all_ratio"],
               "VIRUS / external ratio\nrobust location=%.4g; median=%.4g" %
               (all_stats["robust_location"], all_stats["median"]))]
    for ax, (data, title) in zip(axes.flat, images):
        im = ax.imshow(data, origin="lower", cmap="magma" if "ratio" not in title else "coolwarm")
        ax.set_title(title); fig.colorbar(im, ax=ax, shrink=.8)
    fig.savefig(output_dir / "hbeta_external_vs_virus.png", dpi=160); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for ax, mask, title, stats in (
            (axes[0], hb["arrays"]["all_mask"], "All high-S/N Hbeta", all_stats),
            (axes[1], hb["arrays"]["ew_mask"], "High-EW Hbeta", ew_stats)):
        x = hb["external_smoothed"][mask]; y = hb["virus_smoothed"][mask]
        ax.scatter(x, y, s=5, alpha=.35, rasterized=True)
        if x.size:
            lo, hi = np.nanpercentile(np.r_[x, y], [1, 99]); grid = np.linspace(lo, hi, 100)
            ax.plot(grid, grid, "k--", label="y=x")
            ax.fill_between(grid, .95 * grid, 1.05 * grid, color="gray", alpha=.2, label="+/-5%")
        ax.set_xlabel("Burrell Schmidt Hbeta flux SB"); ax.set_ylabel("VIRUS Hbeta SB")
        ax.set_title("%s\nrobust location=%.4g; median=%.4g" %
                     (title, stats["robust_location"], stats["median"]))
        ax.legend(loc="best")
    fig.savefig(output_dir / "hbeta_bright_region_flux_comparison.png", dpi=160); plt.close(fig)


def _plot_balmer(balmer, output_dir, args):
    fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
    good = balmer["arrays"]["good"]
    x = balmer["arrays"]["rb"][good]; y = balmer["arrays"]["rd"][good]
    ax.scatter(x, y, s=5, alpha=.3, rasterized=True, label="high-S/N VIRUS")
    e = np.linspace(0, 2.0, 200)
    k = _calzetti_k(np.array([4340.47, 4101.74, 4861.33]), args.calzetti_rv)
    ax.plot(args.caseb_hgamma_hbeta * 10 ** (-.4 * e * (k[0] - k[2])),
            args.caseb_hdelta_hbeta * 10 ** (-.4 * e * (k[1] - k[2])),
            "k-", label="Calzetti reddening locus")
    ax.plot(args.caseb_hgamma_hbeta, args.caseb_hdelta_hbeta, "r*", ms=12, label="intrinsic Case B")
    gamma_stats = balmer["median_Hgamma_over_Hbeta"]
    delta_stats = balmer["median_Hdelta_over_Hbeta"]
    ax.set_xlabel("Hgamma / Hbeta"); ax.set_ylabel("Hdelta / Hbeta"); ax.legend(loc="best")
    ax.set_title(
        "Balmer ratios: Case B and reddening\n"
        "robust locations Hgamma/Hbeta=%.4g, Hdelta/Hbeta=%.4g" %
        (gamma_stats["robust_location"], delta_stats["robust_location"]))
    fig.savefig(output_dir / "balmer_caseb_validation.png", dpi=160); plt.close(fig)


def _plot_oiii(oiii, output_dir, args):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    good = np.isfinite(oiii["ratio"])
    x = np.asarray(oiii["ratio"])[good]
    stats = oiii["median_5007_over_4959"]
    axes[0].hist(x, bins=60, histtype="step", color="k")
    axes[0].axvline(stats["robust_location"], color="tab:blue",
                    label="robust location %.3f" % stats["robust_location"])
    axes[0].axvline(stats["median"], color="tab:gray", linestyle="--",
                    label="median %.3f" % stats["median"])
    axes[0].axvline(args.oiii_ratio, color="r", label="theory %.3f" % args.oiii_ratio)
    axes[0].set_xlabel("F5007 / F4959"); axes[0].legend()
    im = axes[1].imshow(oiii["residual_significance"], origin="lower", cmap="coolwarm", vmin=-8, vmax=8)
    axes[1].set_title("5007 residual significance"); fig.colorbar(im, ax=axes[1], shrink=.8)
    axes[2].imshow(oiii["coherent_mask"], origin="lower", cmap="gray_r")
    axes[2].set_title("flagged coherent regions")
    fig.savefig(output_dir / "oiii5007_doublet_validation.png", dpi=160); plt.close(fig)


def _plot_oiii_ratio_vs_signal(oiii, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)

    def plot_rows(ax, rows, labels, xlabel):
        x = np.arange(len(rows), dtype=float)
        location = np.array([row["robust_location"] for row in rows], dtype=float)
        median = np.array([row["median"] for row in rows], dtype=float)
        p16 = np.array([row["p16"] for row in rows], dtype=float)
        p84 = np.array([row["p84"] for row in rows], dtype=float)
        finite = np.isfinite(location) & np.isfinite(p16) & np.isfinite(p84)
        lower = np.maximum(0.0, location - p16)
        upper = np.maximum(0.0, p84 - location)
        if np.any(finite):
            ax.errorbar(x[finite], location[finite],
                        yerr=np.vstack((lower[finite], upper[finite])),
                        fmt="o", color="tab:blue", capsize=4,
                        label="robust location; p16--p84")
        median_finite = np.isfinite(median)
        if np.any(median_finite):
            ax.scatter(x[median_finite], median[median_finite], marker="x",
                       color="tab:gray", label="median")
        ax.axhline(2.98, color="tab:red", linestyle="--", label="theory 2.98")
        ax.set_xticks(x, labels)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("F5007 / F4959")
        ax.grid(alpha=.25)
        ax.legend(loc="best")

    plot_rows(axes[0], oiii["ratio_vs_4959_snr"],
              ["5--7.5", "7.5--10", "10--20", "20--50", ">=50"],
              "F4959 S/N bin")
    plot_rows(axes[1], oiii["ratio_vs_4959_surface_brightness"],
              ["Q1", "Q2", "Q3", "Q4", "Q5"],
              "F4959 surface-brightness quantile bin")
    fig.suptitle("[O III] 5007/4959 ratio versus F4959 signal")
    fig.savefig(output_dir / "oiii5007_4959_ratio_vs_signal.png", dpi=160)
    plt.close(fig)


def _lsf_summary(path, wave, output_dir):
    path = _require_file(path, "LSF samples")
    with fits.open(path, memmap=True) as hdul:
        table_hdu = next((hdu for hdu in hdul if isinstance(hdu, fits.BinTableHDU)), None)
        if table_hdu is None:
            raise ValueError("LSF samples has no binary table: %s" % path)
        table = Table(table_hdu.data)
    lower = {str(name).lower(): name for name in table.colnames}
    def field(candidates):
        return next((lower[name] for name in candidates if name in lower), None)
    wave_name = field(("reference_wavelength", "reference_wave", "refwave", "wavelength", "wave"))
    fwhm_name = field(("fwhm", "fwhm_a", "fwhm_angstrom"))
    usable_name = field(("usable_for_lsf",))
    blend_name = field(("lab_blend", "known_blend", "known_blend_state", "lab_blend_state"))
    if wave_name is None or fwhm_name is None or usable_name is None:
        raise ValueError("LSF table needs reference wavelength, FWHM, and usable_for_lsf fields")
    ref = np.asarray(table[wave_name], dtype=float)
    fwhm = np.asarray(table[fwhm_name], dtype=float)
    usable = np.asarray(table[usable_name], dtype=bool)
    not_blend = np.ones(ref.shape, dtype=bool)
    if blend_name is not None:
        values = np.asarray(table[blend_name])
        if values.dtype.kind in "SUO":
            not_blend = np.array(["blend" not in str(v).lower() for v in values])
        else:
            not_blend = ~values.astype(bool)
    explicit_blend_anchors = np.isclose(ref, 3610.0, atol=2.0) | np.isclose(ref, 3650.0, atol=2.0)
    not_blend &= ~explicit_blend_anchors
    selected = usable & not_blend & np.isfinite(ref) & np.isfinite(fwhm) & (fwhm > 0)
    rows = []
    for reference in sorted(np.unique(ref[selected])):
        values = fwhm[selected & np.isclose(ref, reference)]
        stats = finite_stats(values)
        rows.append({"reference_wavelength_A": float(reference), "N": stats["N"],
                     "min_FWHM_A": stats["min"], "p16_FWHM_A": stats["p16"],
                     "median_FWHM_A": stats["median"],
                     "robust_location_FWHM_A": stats["robust_location"],
                     "p84_FWHM_A": stats["p84"], "max_FWHM_A": stats["max"],
                     "robust_scatter_A": stats["robust_scatter"],
                     # Keep the existing release-table resolving power
                     # definition unchanged; expose the robust counterpart
                     # for validation plots and interpretation.
                     "resolving_power": float(reference / stats["median"]),
                     "robust_resolving_power": float(reference / stats["robust_location"])})
    _write_csv(output_dir / "m101_lsf_summary.csv", rows)
    anchors = np.array([row["reference_wavelength_A"] for row in rows], dtype=float)
    meds = np.array([row["median_FWHM_A"] for row in rows], dtype=float)
    locations = np.array([row["robust_location_FWHM_A"] for row in rows], dtype=float)
    interp = np.full(wave.shape, np.nan, dtype=float)
    median_interp = np.full(wave.shape, np.nan, dtype=float)
    if anchors.size >= 2:
        inside = (wave >= anchors.min()) & (wave <= anchors.max())
        interp[inside] = np.interp(wave[inside], anchors, locations)
        median_interp[inside] = np.interp(wave[inside], anchors, meds)
    elif anchors.size == 1:
        interp[np.isclose(wave, anchors[0])] = locations[0]
        median_interp[np.isclose(wave, anchors[0])] = meds[0]
    _write_csv(output_dir / "m101_lsf_interpolated.csv",
               ({"wavelength_A": float(w), "FWHM_A": float(f),
                 "median_FWHM_A": float(m),
                 "resolving_power": float(w / f) if np.isfinite(f) else None}
                for w, f, m in zip(wave, interp, median_interp)),
               ["wavelength_A", "FWHM_A", "median_FWHM_A", "resolving_power"])
    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    if rows:
        p16 = np.array([r["p16_FWHM_A"] for r in rows], dtype=float)
        p84 = np.array([r["p84_FWHM_A"] for r in rows], dtype=float)
        ax.errorbar(anchors, locations,
                    yerr=np.vstack((np.maximum(0.0, locations - p16),
                                    np.maximum(0.0, p84 - locations))),
                    fmt="o", label="robust location; p16--p84")
        ax.scatter(anchors, meds, marker="x", color="tab:gray",
                   label="median")
    ax.set_xlabel("Wavelength (A)"); ax.set_ylabel("LSF FWHM (A)")
    ax.set_title("M101 instrumental LSF (robust-location FWHM)")
    ax.grid(alpha=.25); ax.legend(loc="best")
    fig.savefig(output_dir / "m101_lsf_summary.png", dpi=160); plt.close(fig)
    overall = finite_stats(fwhm[selected]) if selected.any() else finite_stats([])
    return {"input": str(path), "usable_unblended_rows": int(selected.sum()),
            "excluded_usable_known_blend_rows": int((usable & ~not_blend).sum()),
            "by_reference_wavelength": rows,
            "overall_fwhm": overall,
            "median_FWHM_A": overall["median"],
            "robust_location_FWHM_A": overall["robust_location"],
            "range_FWHM_A": [float(np.min(fwhm[selected])), float(np.max(fwhm[selected]))]
            if selected.any() else [None, None]}


def _write_release(path, arrays, sci_header, lsf_rows, info, oiii_mask, wave):
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    header = sci_header.copy()
    header["EXTNAME"] = "SCI"
    header.add_history("Cube reconstructed from calibrated VIRUS fiber spectra.")
    header.add_history("Current M101 staged model: frozen M, alpha q correction, persistent mu_a, and final exposure sky.")
    header.add_history("Two median-one physical-amplifier-collapsed LOO SB exposure-scale iterations applied.")
    header.add_history("Final exposure correction = 1/(g1_rel*g2_rel); sky was scaled and not re-estimated.")
    header.add_history("g3 was measured as closure QA only and was not applied.")
    header.add_history("Shared hardware registry exclusions and reviewed QA decisions were inherited from production.")
    header.add_history("VARIANCE/DQ/COVERAGE/NCONTRIB ancillary extensions are included.")
    header.add_history("[O III] 5007 bright-line artifact was explicitly retested against 4959.")
    header.add_history("Hbeta was independently validated against Burrell Schmidt imaging.")
    header.add_history("Hgamma/Hbeta and Hdelta/Hbeta were checked against reddened Case B.")
    header.add_history("Instrumental LSF is supplied in the LSF extension.")
    header.add_history("Remaining OIII-defect pixels, if any, are represented by DQ bit 5.")
    hdus = [fits.PrimaryHDU(np.asarray(arrays["SCI"], dtype=np.float32), header=header)]
    ext_specs = [("VAR", arrays["VAR"], "SCI units squared", np.float32),
                 ("ERROR", arrays["ERROR"], "SCI units", np.float32),
                 ("DQ", arrays["DQ"], "bit mask", np.uint16),
                 ("COVERAGE", arrays["COVERAGE"], "Gaussian support", np.float32),
                 ("NCONTRIB", arrays["NCONTRIB"], "count", np.uint8)]
    for name, data, bunit, dtype in ext_specs:
        h = sci_header.copy(); h["EXTNAME"] = name; h["BUNIT"] = bunit
        if name == "DQ":
            h["DQBIT0"] = "NCONTRIB < 2"; h["DQBIT1"] = "SCI valid, VAR unavailable"
            h["DQBIT2"] = "empirical VAR adopted over formal"; h["DQBIT3"] = "formal VAR adopted"
            h["DQBIT4"] = "empirical-only VAR adopted"
            h["DQBIT5"] = "residual bright-line [O III] 5007 deficit from 5007/4959 validation"
        hdus.append(fits.ImageHDU(np.asarray(data, dtype=dtype), header=h, name=name))
    if lsf_rows:
        lsf_table = Table()
        for name in ("reference_wavelength_A", "N", "median_FWHM_A", "p16_FWHM_A",
                     "p84_FWHM_A", "robust_scatter_A", "resolving_power"):
            values = [row.get(name, "") for row in lsf_rows]
            if name == "N":
                lsf_table[name] = np.asarray([int(float(v)) for v in values], dtype=np.int32)
            else:
                lsf_table[name] = np.asarray([float(v) if v not in ("", "None") else np.nan
                                              for v in values], dtype=np.float64)
    else:
        lsf_table = Table()
    hdus.append(fits.BinTableHDU(lsf_table, name="LSF"))
    release_rows = [{"KEY": str(k), "VALUE": json.dumps(json_safe(v), separators=(",", ":"))}
                    for k, v in info.items()]
    release_table = Table(rows=[(r["KEY"], r["VALUE"]) for r in release_rows], names=("KEY", "VALUE"))
    hdus.append(fits.BinTableHDU(release_table, name="RELEASE_INFO"))
    fits.HDUList(hdus).writeto(path, overwrite=True)
    return path


def _plot_release_qa(arrays, lsf, output_dir):
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    dq = arrays["DQ"]
    axes[0, 0].bar(list(map(int, lsf.get("dq_bits", {}).keys())) or [0], list(lsf.get("dq_bits", {}).values()) or [0])
    axes[0, 0].set_title("DQ bit counts"); axes[0, 0].set_xlabel("bit")
    axes[0, 1].hist(arrays["NCONTRIB"].ravel(), bins=np.arange(-.5, np.max(arrays["NCONTRIB"]) + 1.5), histtype="step")
    axes[0, 1].set_title("NCONTRIB distribution")
    var_fraction = np.mean(np.isfinite(arrays["VAR"]), axis=0)
    axes[1, 0].imshow(var_fraction, origin="lower", cmap="gray_r", vmin=0, vmax=1)
    axes[1, 0].set_title("finite VAR fraction by spaxel")
    rows = lsf.get("by_reference_wavelength", [])
    if rows:
        axes[1, 1].plot([r["reference_wavelength_A"] for r in rows],
                        [r.get("robust_resolving_power", r["resolving_power"])
                         for r in rows], "o-")
    axes[1, 1].set_title("LSF resolving power (robust location)"); axes[1, 1].set_xlabel("A")
    fig.savefig(output_dir / "release_product_qa.png", dpi=160); plt.close(fig)


def _markdown(summary):
    cube = summary["cube_validation"]; hb = summary["hbeta_validation"]
    balmer = summary["balmer_validation"]; oiii = summary["oiii_validation"]
    counts = summary["spectrum_accounting"]
    lsf = summary["lsf"]
    production = summary["production_cube_provenance"]
    hb_all = hb["absolute_flux_all_high_SN"]
    hb_ew = hb["absolute_flux_high_EW"]
    hg = balmer["median_Hgamma_over_Hbeta"]
    hd = balmer["median_Hdelta_over_Hbeta"]
    ebv_diff = balmer["E_BV_gamma_minus_delta"]
    oiii_ratio = oiii["median_5007_over_4959"]
    overall_lsf = lsf["overall_fwhm"]
    lines = ["# M101 referee validation and public release", "",
             "This report is downstream validation and packaging of the accepted current-model production cube. No calibration was refit and no production SCI voxel was changed.", "",
             "## Production calibration provenance", "",
             "- Model: `%s`; exposure correction: `%s`." %
             (production["model_name"], production["exposure_scale_correction"]),
             "- Applied SB iterations: %d; third iteration applied: %s; sky re-estimated after scaling: %s." %
             (production["SB_iterations_applied"], production["third_iteration_applied"],
              production["sky_reestimated_after_scale"]),
             "- Hardware registry: `%s`; purpose=`%s`; persistent excluded groups=%d." %
             (counts["hardware_registry"]["path"], counts["hardware_registry_purpose"],
              counts["N_persistent_hardware_excluded_groups"]),
             "- Explicit post-QA rejection groups inherited from production: %d." %
             production["explicit_qa_rejected_groups"], "",
             "## Hbeta bright-line validation", "",
             "- Status: **%s**." % hb["status"],
             "- Coherent candidate regions: %d; morphology scale (diagnostic only): %s." %
             (len(hb["coherent_regions"]), hb["global_robust_through_zero_morphology_scale"]),
             "- Hbeta candidates are flagged for inspection only; this script does not mask or interpolate Hbeta.", "",
             "## Independent Hbeta imaging comparison", "",
             "- All high-S/N samples: N=%d, robust location VIRUS/external=%s, median=%s, p16/p84=%s/%s, robust scatter=%s." %
             (hb_all["mask_count"], hb_all["robust_location"], hb_all["median"],
              hb_all["p16"], hb_all["p84"], hb_all["robust_scatter"]),
             "- High-EW subset (EW >= %s A): N=%d, robust location=%s, median=%s, p16/p84=%s/%s, robust scatter=%s." %
             (summary["configuration"]["min_hbeta_ew"], hb_ew["mask_count"],
              hb_ew["robust_location"], hb_ew["median"], hb_ew["p16"],
              hb_ew["p84"], hb_ew["robust_scatter"]),
             "- Raw external narrowband excess is HB_ON - HB_OFF and is not claimed to be exactly absorption-corrected nebular Hbeta.", "",
             "## Balmer decrement validation", "",
             "- High-quality samples: N=%d; robust locations Hgamma/Hbeta=%s and Hdelta/Hbeta=%s; medians=%s and %s." %
             (balmer["N_high_quality_samples"], hg["robust_location"], hd["robust_location"],
              hg["median"], hd["median"]),
             "- E(B-V)_gamma - E(B-V)_delta: robust location=%s, median=%s, robust scatter=%s; 1-sigma fraction=%s; 2-sigma fraction=%s." %
             (ebv_diff["robust_location"], ebv_diff["median"], ebv_diff["robust_scatter"],
              balmer["fraction_consistent_1sigma"], balmer["fraction_consistent_2sigma"]),
             "- E(B-V)_gamma and E(B-V)_delta robust locations=%s and %s; medians=%s and %s." %
             (balmer["E_BV_gamma"]["robust_location"], balmer["E_BV_delta"]["robust_location"],
              balmer["E_BV_gamma"]["median"], balmer["E_BV_delta"]["median"]),
             "- Case B ratios: Hgamma/Hbeta=%.4f, Hdelta/Hbeta=%.4f; attenuation: Calzetti et al. 2000, Rv=%.3f." %
             (balmer["caseB_intrinsic_ratios"]["Hgamma_Hbeta"], balmer["caseB_intrinsic_ratios"]["Hdelta_Hbeta"], balmer["calzetti_Rv"]), "",
             "## [O III] 5007 artifact validation", "",
             "- High-S/N 4959 samples: N=%d; robust location 5007/4959=%s; median=%s; robust scatter=%s; adopted ratio=%.4f." %
             (oiii["N_high_SN_4959_samples"], oiii_ratio["robust_location"],
              oiii_ratio["median"], oiii_ratio["robust_scatter"], oiii["adopted_branching_ratio"]),
             "- Status: %s; DQ bit-5 voxels assigned: %d." % (oiii["status"], oiii["DQ_voxels_assigned"]), "",
             "## Public cube ancillary products", "",
             "- Release FITS: `%s`. Extensions: PRIMARY/SCI, VAR, ERROR, DQ, COVERAGE, NCONTRIB, LSF, RELEASE_INFO." % summary["release_cube"],
             "- Dimensions: %s; wavelength %.3f--%.3f A, step %.3f A; spatial scale %.4f arcsec pixel^-1." %
             (cube["cube_dimensions"], cube["wavelength"]["start_A"], cube["wavelength"]["end_A"],
              cube["wavelength"]["step_A"], cube["spatial_pixel_scale_arcsec"]),
             "- VAR >= 0: %s; ERROR=sqrt(VAR): %s; coverage in [0,1]: %s; DQ bit-0 semantics agree: %s." %
             (cube["VAR_nonnegative_finite"], cube["ERROR_matches_sqrt_VAR"],
              cube["COVERAGE_in_expected_0_1_range"], cube["DQ_bit0_semantics_agree"]), "",
             "## Instrumental LSF", "",
             "- Usable unblended sample rows: %d; robust-location FWHM=%s A; median=%s A; range=%s A." %
             (lsf["usable_unblended_rows"], overall_lsf["robust_location"],
              overall_lsf["median"], lsf["range_FWHM_A"]),
             "- LSF files: `m101_lsf_summary.csv`, `m101_lsf_interpolated.csv`, `m101_lsf_summary.png`.", "",
             "## Input/retained spectrum accounting", "",
             "- H5=%d; input fiber spectra=%d; hardware excluded=%d; historical=%d; persistent=%d; retained=%d; shots=%d; amplifier observations=%d." %
             (counts["N_H5"], counts["N_input_fiber_spectra"], counts["N_hardware_excluded_fiber_rows"],
              counts["N_historical_hardware_excluded_fiber_rows"],
              counts["N_persistent_hardware_excluded_fiber_rows"],
              counts["N_retained_fiber_spectra"], counts["N_exposures_shots"], counts["N_amplifier_observations"]),
             "- Hardware exclusions are separate from explicit post-iteration-2 QA rejection decisions.", "",
             "## Release DQ definitions", "",
             "- Existing bits 0--4 are retained. New bit 5 is residual bright-line [O III] 5007 deficit identified by the post-reduction 5007/4959 validation.",
             "- SCI was not removed, interpolated, or replaced by this script.", ""]
    return "\n".join(lines)


def _print_oiii_ratio_signal_tables(oiii):
    fields = ("N", "median", "robust_location", "robust_scatter",
              "p16", "p84", "robust_location_minus_2_98")
    print("OIII ratio versus F4959 S/N")
    print("S/N lower\tS/N upper\t" + "\t".join(fields))
    for row in oiii["ratio_vs_4959_snr"]:
        values = [row["snr_lower"], row["snr_upper"]] + [row[key] for key in fields]
        print("\t".join(str(value) for value in values))
    print("OIII ratio versus F4959 surface brightness")
    print("quantile\tSB lower\tSB upper\t" + "\t".join(fields))
    for row in oiii["ratio_vs_4959_surface_brightness"]:
        values = [row["quantile_bin"], row["surface_brightness_lower"],
                  row["surface_brightness_upper"]] + [row[key] for key in fields]
        print("\t".join(str(value) for value in values))


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--cube", required=True); parser.add_argument("--variance", required=True)
    parser.add_argument("--error", required=True); parser.add_argument("--dq", required=True)
    parser.add_argument("--coverage", required=True); parser.add_argument("--ncontrib", required=True)
    parser.add_argument("--cube-summary", required=True); parser.add_argument("--h5-glob", required=True)
    parser.add_argument("--hb-on-image", required=True); parser.add_argument("--hb-off-image", required=True)
    parser.add_argument("--hb-on-filter", required=True); parser.add_argument("--hb-off-filter", required=True)
    parser.add_argument("--lsf-samples", required=True); parser.add_argument("--output-dir", default="m101_referee_validation")
    parser.add_argument("--release-cube", default="m101_public_release_cube.fits")
    parser.add_argument("--hb-line-image")
    parser.add_argument("--hb-image-units", choices=("auto", "adu", "flux"), default="auto")
    parser.add_argument("--hb-line-image-units", choices=("auto", "adu", "flux"), default="auto")
    parser.add_argument("--comparison-smoothing-fwhm", type=float, default=6.0)
    parser.add_argument("--min-hbeta-ew", type=float, default=20.0)
    parser.add_argument("--hbeta-snr", type=float, default=10.0); parser.add_argument("--hgamma-snr", type=float, default=5.0)
    parser.add_argument("--hdelta-snr", type=float, default=3.0); parser.add_argument("--oiii-4959-snr", type=float, default=5.0)
    parser.add_argument("--oiii-ratio", type=float, default=2.98); parser.add_argument("--oiii-deficit-sigma", type=float, default=-5.0)
    parser.add_argument("--hbeta-deficit-sigma", type=float, default=-5.0); parser.add_argument("--hb-external-snr", type=float, default=5.0)
    parser.add_argument("--min-defect-area", type=int, default=3); parser.add_argument("--outer-sky-radius-arcmin", type=float, default=16.0)
    parser.add_argument("--caseb-hgamma-hbeta", type=float, default=0.468); parser.add_argument("--caseb-hdelta-hbeta", type=float, default=0.259)
    parser.add_argument("--calzetti-rv", type=float, default=4.05)
    args = parser.parse_args()
    if args.comparison_smoothing_fwhm < 0 or args.min_hbeta_ew < 0 or args.min_defect_area < 1:
        parser.error("smoothing, EW, and minimum defect area must be nonnegative/positive")
    output_dir = Path(args.output_dir).expanduser().resolve(); output_dir.mkdir(parents=True, exist_ok=True)
    arrays, headers, wave, celestial, pixel_scale, paths = _read_cube_products(args)
    arrays["_SCI_HEADER"] = headers["SCI"]
    cube_validation = _validate_products(arrays, wave, pixel_scale)
    sci_scale, sci_scale_provenance = _science_unit_scale(headers["SCI"])
    cube_validation["SCI_physical_scale"] = sci_scale
    cube_validation["SCI_physical_scale_provenance"] = sci_scale_provenance
    cube_summary_path = _require_file(args.cube_summary, "cube summary")
    try:
        cube_summary = json.loads(cube_summary_path.read_text())
    except Exception as exc:
        raise ValueError("could not read cube summary JSON: %s" % cube_summary_path) from exc
    production_provenance = _validate_current_model_summary(cube_summary)
    accounting = _spectrum_accounting(args.h5_glob)
    production_hardware = cube_summary["hardware_exclusions"]
    accounting_pairs = {
        "hardware_excluded_groups": accounting["N_hardware_excluded_groups"],
        "persistent_hardware_excluded_groups": accounting[
            "N_persistent_hardware_excluded_groups"],
        "hardware_excluded_fiber_rows": accounting[
            "N_hardware_excluded_fiber_rows"],
        "persistent_hardware_excluded_fiber_rows": accounting[
            "N_persistent_hardware_excluded_fiber_rows"],
    }
    production_pairs = {
        key: int(production_hardware.get(key, -1)) for key in accounting_pairs
    }
    if accounting_pairs != production_pairs:
        raise ValueError(
            "hardware accounting does not match the production summary: "
            "computed=%s production=%s" % (accounting_pairs, production_pairs))
    production_provenance["hardware_accounting_matches"] = True
    on, on_header, on_wcs, on_scale, on_path = _read_image_hdu(args.hb_on_image, "Hbeta-on image")
    off, off_header, off_wcs, off_scale, off_path = _read_image_hdu(args.hb_off_image, "Hbeta-off image")
    on_area = _pixel_area_arcsec2(on_wcs)
    off_area = _pixel_area_arcsec2(off_wcs)
    on_cal, on_decision = _calibrate_external(on, on_header, args.hb_image_units, BS_ON_ZEROPOINT, "Hbeta-on")
    off_cal, off_decision = _calibrate_external(off, off_header, args.hb_image_units, BS_OFF_ZEROPOINT, "Hbeta-off")
    on_sky, on_sky_info = _outer_sky(on_cal, on_wcs, args.outer_sky_radius_arcmin)
    off_sky, off_sky_info = _outer_sky(off_cal, off_wcs, args.outer_sky_radius_arcmin)
    on_sb = on_sky / on_area; off_sb = off_sky / off_area
    on_target, on_fp = _reproject_sb(on_sb, on_wcs, celestial, arrays["SCI"].shape[1:])
    off_target, off_fp = _reproject_sb(off_sb, off_wcs, celestial, arrays["SCI"].shape[1:])
    external = on_target - off_target
    external_support = (on_fp > 0) & (off_fp > 0) & np.isfinite(external)
    external[~external_support] = np.nan
    line_image_info = None
    if args.hb_line_image:
        line_data, line_header, line_wcs, line_scale, line_path = _read_image_hdu(args.hb_line_image, "corrected Hbeta line image")
        line_cal, line_decision = _calibrate_external(line_data, line_header, args.hb_line_image_units, 1.0,
                                                       "corrected Hbeta line image", allow_adu=False)
        line_sb = line_cal / _pixel_area_arcsec2(line_wcs)
        corrected_target, corrected_fp = _reproject_sb(line_sb, line_wcs, celestial, arrays["SCI"].shape[1:])
        line_image_info = {"path": str(line_path), "units": line_decision,
                           "support_fraction": float(np.mean(corrected_fp > 0))}
        absolute_external = corrected_target
    else:
        absolute_external = external
    line_products = {name: _line_map(arrays["SCI"], arrays["VAR"], arrays["DQ"], wave, rest, half,
                                     unit_scale=sci_scale)
                     for name, (rest, half) in LINES.items()}
    map_header = celestial.to_header()
    _write_line_products(output_dir, line_products, map_header)
    hb = _hbeta_validation(line_products["Hbeta_4861"]["flux"], line_products["Hbeta_4861"]["variance"],
                           line_products["Hbeta_4861"]["snr"], line_products["Hbeta_4861"]["ew"],
                           absolute_external, external_support, args, pixel_scale, output_dir, map_header)
    _write_image(output_dir / "hbeta_residual_map.fits", hb["residual"],
                 _map_header(map_header, "erg s-1 cm-2 arcsec-2", "Hbeta morphology residual"))
    _plot_hbeta(hb, map_header, output_dir)
    balmer = _balmer_validation(line_products, args); _plot_balmer(balmer, output_dir, args)
    oiii = _oiii_validation(line_products, args)
    _write_image(output_dir / "oiii5007_ratio_map.fits", oiii["ratio"], _map_header(map_header, "dimensionless", "OIII 5007/4959 ratio"))
    _write_image(output_dir / "oiii5007_residual_significance.fits", oiii["residual_significance"], _map_header(map_header, "sigma", "OIII 5007 - R*4959 residual significance"))
    _write_image(output_dir / "oiii5007_defect_mask.fits", oiii["coherent_mask"].astype(np.uint8), _map_header(map_header, "bit mask", "OIII 5007 validation defect mask"), dtype=np.uint8)
    _plot_oiii(oiii, output_dir, args)
    _plot_oiii_ratio_vs_signal(oiii, output_dir)
    oii = line_products["OII_3727"]
    oii_valid = np.isfinite(oii["flux"]) & np.isfinite(oii["snr"])
    oii_high = oii_valid & (oii["snr"] >= 5)
    oii_control = {"finite_map_pixels": int(np.isfinite(oii["flux"]).sum()),
                   "support_fraction": float(oii_valid.mean()), "N_SNR_ge_5": int(np.sum(oii["snr"] >= 5)),
                   "zero_flux_fraction_on_supported": float(np.mean(oii["flux"][oii_valid] == 0)) if oii_valid.any() else None,
                   "zero_flux_fraction_at_SNR_ge_5": float(np.mean(oii["flux"][oii_high] == 0)) if oii_high.any() else None,
                   "doublet_ratio_test": "not attempted; VIRUS does not resolve 3726/3729"}
    lsf = _lsf_summary(args.lsf_samples, wave, output_dir)
    release_dq = np.asarray(arrays["DQ"], dtype=np.uint16).copy()
    oiii_planes = np.abs(wave - _line_center(LINES["OIII_5007"][0])) <= LINES["OIII_5007"][1]
    for plane in np.flatnonzero(oiii_planes):
        release_dq[plane][oiii["coherent_mask"]] |= DQ_OIII5007_VALIDATION
    arrays["DQ"] = release_dq
    lsf_rows = []
    lsf_csv = output_dir / "m101_lsf_summary.csv"
    with open(lsf_csv, newline="") as handle:
        lsf_rows = list(csv.DictReader(handle))
    _plot_release_qa(arrays, {**lsf, "dq_bits": cube_validation["DQ_counts_by_bit"]}, output_dir)
    exposure_scale = cube_summary["exposure_scale"]
    qa_production = cube_summary["qa"]
    production_release_info = {
        "model_name": cube_summary["model"]["name"],
        "exposure_scale_method": exposure_scale["method"],
        "exposure_scale_correction": exposure_scale["correction"],
        "SB_iterations_applied": int(exposure_scale["SB_iterations_applied"]),
        "third_iteration_applied": bool(exposure_scale["third_iteration_applied"]),
        "sky_reestimated_after_scale": bool(exposure_scale["sky_reestimated_after_scale"]),
        "explicit_qa_rejected_groups": int(
            qa_production.get("rejection", {}).get("rejected_groups", 0)),
        "explicit_qa_rejected_fiber_rows": int(
            qa_production.get("rejection", {}).get("rejected_fiber_rows", 0)),
        "qa_reject_file_used": bool(qa_production.get("reject_file_used", False)),
        "qa_reject_file": qa_production.get("reject_file"),
        "hardware_registry": accounting["hardware_registry"],
    }
    release_info = {"N_H5": accounting["N_H5"], "N_exposures_shots": accounting["N_exposures_shots"],
                    "N_input_fiber_spectra": accounting["N_input_fiber_spectra"],
                    "N_retained_fiber_spectra": accounting["N_retained_fiber_spectra"],
                    "N_hardware_excluded_fiber_rows": accounting[
                        "N_hardware_excluded_fiber_rows"],
                    "N_historical_hardware_excluded_fiber_rows": accounting[
                        "N_historical_hardware_excluded_fiber_rows"],
                    "N_persistent_hardware_excluded_fiber_rows": accounting[
                        "N_persistent_hardware_excluded_fiber_rows"],
                    "N_hardware_excluded_groups": accounting[
                        "N_hardware_excluded_groups"],
                    "N_historical_hardware_excluded_groups": accounting[
                        "N_historical_hardware_excluded_groups"],
                    "N_persistent_hardware_excluded_groups": accounting[
                        "N_persistent_hardware_excluded_groups"],
                    "N_amplifier_observations": accounting["N_amplifier_observations"],
                    "cube_dimensions": cube_validation["cube_dimensions"],
                    "wavelength": cube_validation["wavelength"],
                    "spatial_pixel_scale_arcsec": pixel_scale,
                    "SCI_BUNIT": str(headers["SCI"].get("BUNIT", "")),
                    "calibration_provenance": "completed current-model cube; validation did not refit",
                    "production_cube_provenance": production_release_info,
                    "validation_script_version": SCRIPT_VERSION,
                    "validation_utc": datetime.now(timezone.utc).isoformat(),
                    "DQ_bit5_count": int(np.sum((release_dq & DQ_OIII5007_VALIDATION) != 0))}
    release_path = _write_release(args.release_cube, arrays, headers["SCI"], lsf_rows, release_info, oiii["coherent_mask"], wave)
    summary = {"script": str(Path(__file__).resolve()), "version": SCRIPT_VERSION,
               "created_utc": datetime.now(timezone.utc).isoformat(), "release_cube": str(release_path),
               "inputs": {key: _file_identity(path) for key, path in paths.items()},
               "cube_summary": {"path": str(cube_summary_path), "source": cube_summary},
               "production_cube_provenance": {
                   **production_release_info,
                   **production_provenance,
               },
               "cube_validation": cube_validation,
               "spectrum_accounting": accounting,
               "external_hbeta": {"on": {"path": str(on_path), "pixel_scale_arcsec": on_scale, "pixel_area_arcsec2": on_area, "units": on_decision,
                                           "sky": on_sky_info},
                                  "off": {"path": str(off_path), "pixel_scale_arcsec": off_scale, "pixel_area_arcsec2": off_area, "units": off_decision,
                                          "sky": off_sky_info},
                                  "filters": {"on": _read_filter_summary(args.hb_on_filter), "off": _read_filter_summary(args.hb_off_filter)},
                                  "raw_definition": "HB_ON - HB_OFF; underlying stellar Hbeta absorption not removed",
                                  "raw_surface_brightness_units": "erg s-1 cm-2 arcsec-2",
                                  "corrected_line_image": line_image_info,
                                  "published_absolute_uncertainty_fraction": BS_ABSOLUTE_CALIBRATION_FRACTION},
               "line_products": {name: {key: value for key, value in result.items() if key != "arrays"}
                                 for name, result in line_products.items()},
               "hbeta_validation": reportable({key: value for key, value in hb.items() if key != "arrays"}),
               "balmer_validation": reportable({key: value for key, value in balmer.items() if key != "arrays"}),
               "oiii_validation": {key: value for key, value in oiii.items() if key not in ("ratio", "residual_significance", "candidate_mask", "coherent_mask")},
               "oii_control": oii_control, "lsf": lsf, "release_dq_bit5_voxels": release_info["DQ_bit5_count"],
               "configuration": vars(args)}
    summary = json_safe(reportable(summary))
    (output_dir / "referee_validation_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False))
    markdown = _markdown(summary)
    (output_dir / "referee_validation_summary.md").write_text(markdown)
    hbeta_all = summary["hbeta_validation"]["absolute_flux_all_high_SN"]
    hbeta_ew = summary["hbeta_validation"]["absolute_flux_high_EW"]
    gamma = summary["balmer_validation"]["median_Hgamma_over_Hbeta"]
    delta = summary["balmer_validation"]["median_Hdelta_over_Hbeta"]
    ebv_difference = summary["balmer_validation"]["E_BV_gamma_minus_delta"]
    oiii_ratio = summary["oiii_validation"]["median_5007_over_4959"]
    overall_lsf = summary["lsf"]["overall_fwhm"]
    print("Hbeta: N=%d; robust location=%s; median=%s; robust scatter=%s; "
          "high-EW robust location=%s; high-EW median=%s; coherent regions=%d" %
          (hbeta_all["mask_count"], hbeta_all["robust_location"], hbeta_all["median"],
           hbeta_all["robust_scatter"], hbeta_ew["robust_location"],
           hbeta_ew["median"], len(summary["hbeta_validation"]["coherent_regions"])))
    print("Balmer: N=%d; robust locations Hgamma/Hbeta=%s Hdelta/Hbeta=%s; "
          "medians=%s/%s; robust dE(B-V)=%s; median=%s; robust scatter=%s" %
          (summary["balmer_validation"]["N_high_quality_samples"],
           gamma["robust_location"], delta["robust_location"], gamma["median"],
           delta["median"], ebv_difference["robust_location"], ebv_difference["median"],
           ebv_difference["robust_scatter"]))
    print("OIII: N=%d; robust location ratio=%s; median=%s; robust scatter=%s; "
          "coherent regions=%d; DQ bit-5 voxels=%d" %
          (summary["oiii_validation"]["N_high_SN_4959_samples"],
           oiii_ratio["robust_location"], oiii_ratio["median"],
           oiii_ratio["robust_scatter"], len(summary["oiii_validation"]["coherent_regions"]),
           summary["release_dq_bit5_voxels"]))
    _print_oiii_ratio_signal_tables(summary["oiii_validation"])
    print("Release: dimensions=%s; input=%d; retained=%d; shots=%d; valid SCI=%d; "
          "finite VAR fraction=%s; LSF robust location/median/range=%s/%s/%s A" %
          (summary["cube_validation"]["cube_dimensions"], summary["spectrum_accounting"]["N_input_fiber_spectra"],
           summary["spectrum_accounting"]["N_retained_fiber_spectra"], summary["spectrum_accounting"]["N_exposures_shots"],
           summary["cube_validation"]["valid_SCI_voxels"], summary["cube_validation"]["finite_VAR_fraction"],
           overall_lsf["robust_location"], overall_lsf["median"], summary["lsf"]["range_FWHM_A"]))
    print("Central-value summary")
    print("quantity\tN\told median\tnew robust_location\trobust_scatter")
    central_rows = (
        ("OIII 5007/4959", oiii_ratio),
        ("Hgamma/Hbeta", gamma),
        ("Hdelta/Hbeta", delta),
        ("Hbeta VIRUS/Burrell all-high-S/N", hbeta_all),
        ("Hbeta VIRUS/Burrell high-EW", hbeta_ew),
        ("overall LSF FWHM", overall_lsf),
    )
    for label, stats in central_rows:
        print("%s\t%s\t%s\t%s\t%s" %
              (label, stats["N"], stats["median"], stats["robust_location"],
               stats["robust_scatter"]))
    print("Final release FITS: %s" % release_path)
    print("Referee Markdown report: %s" % (output_dir / "referee_validation_summary.md"))


if __name__ == "__main__":
    main()
