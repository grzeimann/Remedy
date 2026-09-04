#!/usr/bin/env python3
"""Build a VIRUS cube using an already-computed Bayesian M101 calibration.

This is deliberately a cube-ingestion script, not a calibration fitter.  It
uses the tested M101 H5 grouping, ADR, residual-sky, Gaussian-splat, variance,
DQ, and FITS conventions while taking the per-amplifier calibration directly
from a completed Bayesian solution H5.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
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
from math_utils import biweight


DEF_WAVE = validated_m101.DEF_WAVE
N_FIBER_AMP = validated_m101.N_FIBER_AMP
N_EXPOSURES = validated_m101.N_EXPOSURES
FIBER_RADIUS_ARCSEC = validated_m101.FIBER_RADIUS_ARCSEC
M101_RA_DEG = 210.800
M101_DEC_DEG = 54.333
M101_SKY_MIN_RADIUS_ARCMIN = 6.0
M101_SKY_MIN_FINITE_FRACTION = 0.8
M101_SKY_MIN_FIBERS = 20
GAUSSIAN_FWHM_ARCSEC = 1.8

DQ_INSUFFICIENT_SUPPORT = np.uint16(1 << 0)
DQ_VAR_INCOMPLETE = np.uint16(1 << 1)
DQ_EMPIRICAL_VAR_USED = np.uint16(1 << 2)
DQ_FORMAL_VAR_USED = np.uint16(1 << 3)
DQ_VAR_EMPIRICAL_ONLY = np.uint16(1 << 4)


def _text(value):
    return validated_m101.as_text(value)


def _file_identity(path):
    path = Path(path).expanduser().resolve()
    stat = path.stat()
    return {"path": str(path), "size": int(stat.st_size),
            "mtime_ns": int(stat.st_mtime_ns)}


def _distribution(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not values.size:
        return {"N": 0, "min": np.nan, "p16": np.nan, "p50": np.nan,
                "p84": np.nan, "max": np.nan}
    p16, p50, p84 = np.percentile(values, (16, 50, 84))
    return {"N": int(values.size), "min": float(np.min(values)),
            "p16": float(p16), "p50": float(p50),
            "p84": float(p84), "max": float(np.max(values))}


def _read_fit_calibration(path):
    """Read and validate the Bayesian amplifier table before spectra."""
    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise ValueError("Bayesian fit H5 does not exist: %s" % path)
    required = {"H5", "exposure", "SPECID", "IFUSLOT", "IFUID", "AMP",
                "posterior_z_mean", "p_mean", "alpha_mean", "p_good"}
    started = time.perf_counter()
    rows = []
    metadata = {}
    with tables.open_file(path, mode="r") as h5:
        schema = _text(getattr(h5.root._v_attrs, "schema_version", ""))
        if not schema.startswith("m101_bayesian_calibration_v1"):
            raise ValueError("unsupported Bayesian calibration schema: %s" % schema)
        if "/amplifier_observations" not in h5:
            raise ValueError("Bayesian H5 lacks /amplifier_observations")
        table = h5.root.amplifier_observations
        missing = required - set(table.colnames)
        if missing:
            raise ValueError("Bayesian amplifier table lacks columns: %s" % sorted(missing))
        if "/provenance/metadata" in h5:
            for row in h5.root.provenance.metadata:
                try:
                    metadata[_text(row["key"])] = json.loads(_text(row["value"]))
                except (TypeError, ValueError, json.JSONDecodeError):
                    metadata[_text(row["key"])] = _text(row["value"])
        for row in table:
            record = {name: row[name] for name in table.colnames}
            record["H5"] = _text(record["H5"])
            record["AMP"] = _text(record["AMP"])
            rows.append(record)
    if not rows:
        raise ValueError("Bayesian amplifier table is empty")
    calibration = {}
    for record in rows:
        key = (Path(record["H5"]).name, int(record["exposure"]),
               int(record["SPECID"]), int(record["IFUSLOT"]),
               int(record["IFUID"]), record["AMP"])
        if key in calibration:
            raise ValueError("duplicate Bayesian calibration key: %s" % (key,))
        calibration[key] = record
    band_contrast = metadata.get("band_contrast", {})
    delta_z = band_contrast.get("delta_z_band", np.nan) if isinstance(band_contrast, dict) else np.nan
    delta_p = band_contrast.get("delta_p_band", np.nan) if isinstance(band_contrast, dict) else np.nan
    if not np.isfinite(delta_z) or not np.isfinite(delta_p):
        raise ValueError("Bayesian provenance lacks finite band_contrast delta_z_band/delta_p_band")
    print("Bayesian H5 read: %.3f s" % (time.perf_counter() - started))
    return calibration, metadata, {"path": str(path), "identity": _file_identity(path),
                                   "rows": len(rows), "delta_z_band": float(delta_z),
                                   "delta_p_band": float(delta_p)}


def _preflight_matches(h5files, calibration):
    """Match every input amplifier by explicit identity, never row order."""
    started = time.perf_counter()
    matches = {}
    input_count = 0
    for h5file in h5files:
        with tables.open_file(h5file, mode="r") as h5:
            if "/Info" not in h5 or "/Fibers" not in h5 or "/Survey" not in h5:
                raise ValueError("input is not a VIRUS H5 with Info/Fibers/Survey: %s" % h5file)
            groups, labels = validated_m101.build_groups(h5.root.Info)
            for group in groups:
                key = (Path(h5file).name, group["exposure"], group["specid"],
                       group["ifuslot"], group["ifuid"], group["amp"])
                if key not in calibration:
                    raise ValueError("missing Bayesian calibration row for %s" % (key,))
                record = calibration[key]
                for field in ("posterior_z_mean", "p_mean", "alpha_mean"):
                    if not np.isfinite(float(record[field])):
                        raise ValueError("nonfinite %s for Bayesian key %s" % (field, key))
                matches[(str(Path(h5file).resolve()), group["exposure"],
                        group["specid"], group["ifuslot"], group["ifuid"],
                        group["amp"])] = record
                input_count += 1
    print("calibration-key matching: %.3f s" % (time.perf_counter() - started))
    return matches, input_count


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
    tp = astrometry.setup_TP(ra, dec, 0.0, x0=(n + 1) / 2.0,
                             y0=(n + 1) / 2.0)
    return ra, dec, size_arcsec, n, xg, yg, xgrid, ygrid, tp


def subtract_m101_residual_sky(spectra, ra, dec, labels, xg, yg, tp, log=None, h5file=""):
    """The established M101 radial/finite-spectrum residual-sky correction."""
    n_fib, n_wave = spectra.shape
    labels = np.asarray(labels, dtype=int)
    if labels.shape != (n_fib,):
        raise ValueError("residual-sky exposure labels do not match spectra")
    dra_arcmin = ((ra - M101_RA_DEG) * np.cos(np.deg2rad(M101_DEC_DEG)) * 60.0)
    ddec_arcmin = (dec - M101_DEC_DEG) * 60.0
    sky_region = np.hypot(dra_arcmin, ddec_arcmin) > M101_SKY_MIN_RADIUS_ARCMIN
    x, y = tp.wcs_world2pix(ra, dec, 1)
    xc = np.rint(np.interp(x, xg, np.arange(len(xg)), left=0., right=len(xg))).astype(int)
    yc = np.rint(np.interp(y, yg, np.arange(len(yg)), left=0., right=len(yg))).astype(int)
    in_region = (np.isfinite(x) & np.isfinite(y) & (xc >= 0) & (xc < len(xg))
                 & (yc >= 0) & (yc < len(yg)))
    for exposure_index in range(N_EXPOSURES):
        exposure_indices = np.flatnonzero(labels == exposure_index + 1)
        if exposure_indices.size == 0:
            continue
        finite_counts = np.isfinite(spectra[exposure_indices]).sum(axis=1)
        sufficient = finite_counts >= int(np.ceil(M101_SKY_MIN_FINITE_FRACTION * n_wave))
        sky_mask = np.zeros(n_fib, dtype=bool)
        sky_mask[exposure_indices] = sky_region[exposure_indices] & in_region[exposure_indices] & sufficient
        selected = int(np.sum(sky_mask))
        if selected < M101_SKY_MIN_FIBERS:
            if log is not None:
                log.warning("M101 residual sky %s exposure %d skipped: only %d candidates (minimum %d)",
                            h5file, exposure_index + 1, selected, M101_SKY_MIN_FIBERS)
            continue
        residual = biweight(spectra[sky_mask], axis=0)
        finite = np.isfinite(residual)
        if finite.sum() < int(np.ceil(M101_SKY_MIN_FINITE_FRACTION * n_wave)):
            if log is not None:
                log.warning("M101 residual sky %s exposure %d skipped: residual mostly nonfinite",
                            h5file, exposure_index + 1)
            continue
        spectra[np.ix_(exposure_indices, finite)] -= residual[finite]
        if log is not None:
            log.info("M101 residual sky %s exposure %d: selected=%d, median=%0.5g",
                     h5file, exposure_index + 1, selected, np.nanmedian(residual))
    return spectra


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


def _calibrate_h5(h5file, matches, fq_template, tp, xg, yg):
    """Read one H5 and apply only the fit-derived spectral correction."""
    with tables.open_file(h5file, mode="r") as h5:
        info, fibers = h5.root.Info, h5.root.Fibers
        groups, labels = validated_m101.build_groups(info)
        if "skyspectrum" not in fibers.colnames:
            raise ValueError("%s Fibers lacks skyspectrum" % h5file)
        source = np.asarray(fibers.cols.spectrum[:], dtype=float)
        error_source = np.asarray(fibers.cols.error[:], dtype=float)
        sky = np.asarray(fibers.cols.skyspectrum[:], dtype=float)
        ra = np.asarray(info.cols.ra[:], dtype=float)
        dec = np.asarray(info.cols.dec[:], dtype=float)
        if source.shape != error_source.shape or source.shape != sky.shape:
            raise ValueError("%s spectrum/error/skyspectrum shapes differ" % h5file)
        surveys = _survey_by_exposure(h5)
        ifuslot = np.asarray(info.cols.ifuslot[:])
        amp = np.asarray([_text(value) for value in info.cols.amp[:]])
        hardware_bad = validated_m101.masked_rows(h5file, ifuslot, amp)
        spectra = np.full(source.shape, np.nan, dtype=float)
        errors = np.full(error_source.shape, np.nan, dtype=float)
        for exposure in range(1, N_EXPOSURES + 1):
            survey = surveys[exposure]
            offset = float(survey["offset"])
            if not np.isfinite(offset) or offset == 0.0:
                raise ValueError("%s exposure %d has invalid Survey.offset" % (h5file, exposure))
            K_work = validated_m101.raw_work_basis(survey)
            working = source / offset
            error_work = error_source / abs(offset)
            for group in groups:
                if group["exposure"] != exposure:
                    continue
                key = (str(Path(h5file).resolve()), exposure, group["specid"],
                       group["ifuslot"], group["ifuid"], group["amp"])
                fit = matches[key]
                indices = group["indices"]
                j = np.arange(N_FIBER_AMP)
                q = j if group["amp"] in ("LL", "RU") else N_FIBER_AMP - 1 - j
                additive = (float(fit["alpha_mean"]) * K_work[None, :] *
                            fq_template[q, None])
                multiplier = float(np.exp(float(fit["posterior_z_mean"])))
                spectra[indices] = ((working[indices] - float(fit["p_mean"])
                                     - additive + sky[indices]) / multiplier
                                    - sky[indices])
                errors[indices] = error_work[indices] / multiplier
        spectra[hardware_bad] = np.nan
        errors[hardware_bad] = np.nan
    return spectra, errors, ra, dec, labels, surveys


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
            gx[ox + radius] = np.exp(-0.5 * ((x_origin + ix_center + ox) - xi) ** 2 / sigma_sq)
        for oy in range(-radius, radius + 1):
            gy[oy + radius] = np.exp(-0.5 * ((y_origin + iy_center + oy) - yi) ** 2 / sigma_sq)
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
        formal[two_shot] = (np.sum(np.where(supported, shot_variances, 0.0), axis=0)[two_shot] / 4.0)
    three_or_more = (n >= 3) & complete & positive_or_zero
    if np.any(three_or_more):
        sigmas = np.sqrt(np.where(supported, shot_variances, 1.0))
        inverse_sigma = np.zeros_like(sigmas, dtype=np.float64)
        np.divide(1.0, sigmas, out=inverse_sigma, where=supported)
        inverse_sigma_sum = np.sum(inverse_sigma, axis=0)
        positive_sigma = np.all(~supported | (sigmas > 0.0), axis=0)
        formal_valid = three_or_more & positive_sigma
        formal[formal_valid] = n[formal_valid] * np.pi / (2.0 * inverse_sigma_sum[formal_valid] ** 2)
    empirical = np.full((ny, nx), np.nan, dtype=np.float64)
    five_or_more = n >= 5
    if np.any(five_or_more):
        center = np.nanmedian(shot_images, axis=0)
        mad = np.nanmedian(np.abs(shot_images - center[None, :, :]), axis=0)
        empirical[five_or_more] = (1.2533 * 1.4826 * mad / np.sqrt(n))[five_or_more] ** 2
    formal_available = np.isfinite(formal)
    empirical_available = np.isfinite(empirical)
    both = formal_available & empirical_available
    variance = np.full((ny, nx), np.nan, dtype=np.float64)
    variance[formal_available & ~empirical_available] = formal[formal_available & ~empirical_available]
    variance[empirical_available & ~formal_available] = empirical[empirical_available & ~formal_available]
    variance[both] = np.maximum(formal[both], empirical[both])
    varianceimage = variance.astype(np.float32)
    dq = np.zeros((ny, nx), dtype=np.uint16)
    dq[~valid_sci] |= DQ_INSUFFICIENT_SUPPORT
    formal_used = valid_sci & formal_available & (~empirical_available | (formal >= empirical))
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
        "finite_variance_voxels": int(np.sum(valid_sci & np.isfinite(varianceimage))),
        "median_ncontrib": float(np.median(ncontrib[valid_sci])) if np.any(valid_sci) else np.nan,
        "variance_unavailable_voxels": int(np.sum(valid_sci & ~np.isfinite(varianceimage))),
        "insufficient_support_voxels": int(np.sum(~valid_sci)),
        "median_ratio": float(np.nanmedian(ratio)) if np.any(np.isfinite(ratio)) else np.nan,
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
        _gaussian_splat_shot_xy(indices, xpos, ypos, y, ye, float(xg[0]), float(yg[0]),
                                nx, ny, float(sigma), radius, support_radius, float(area),
                                flux_sum, weight_sum, variance_numerator,
                                error_weight_sum, support_map)
        supported = (support_map != 0) & (weight_sum > 0.0)
        if np.any(supported):
            shot_images[k, supported] = flux_sum[supported] / weight_sum[supported]
            complete = supported & np.isclose(error_weight_sum, weight_sum, rtol=1.e-5, atol=1.e-7)
            shot_variances[k, complete] = variance_numerator[complete] / (weight_sum[complete] ** 2)
            ncontrib[supported] += 1
            coverage += np.minimum(weight_sum, 1.0) * supported
    with np.errstate(all="ignore"):
        image = np.nanmedian(shot_images, axis=0).astype(np.float32)
    varianceimage, dq, variance_stats = _compute_final_variance(shot_images, shot_variances, ncontrib)
    coverage = np.clip(coverage / float(max(1, nshots)), 0.0, 1.0)
    image[ncontrib < 2] = 0.0
    image[~np.isfinite(image)] = 0.0
    varianceimage[ncontrib < 2] = 0.0
    return image, varianceimage, coverage, ncontrib, dq, variance_stats


def _write_cube_products(surname, cube, variancecube, dqcube, weightcube, ncontribcube,
                         tp, n, pixel_scale, fit_provenance, matched, summary):
    header = tp.to_header()
    header["WCSAXES"] = 3
    header["CD1_1"] = -pixel_scale / 3600.0
    header["CD1_2"] = 0.0
    header["CD2_1"] = 0.0
    header["CD2_2"] = pixel_scale / 3600.0
    header.pop("CDELT1", None); header.pop("CDELT2", None)
    header["CRPIX1"] = (n + 1) / 2.0; header["CRPIX2"] = (n + 1) / 2.0
    header["CDELT3"] = 2.0; header["CRPIX3"] = 1.0; header["CRVAL3"] = 3470.0
    header["CTYPE1"] = "RA---TAN"; header["CTYPE2"] = "DEC--TAN"; header["CTYPE3"] = "WAVE"
    header["CUNIT1"] = "deg"; header["CUNIT2"] = "deg"; header["CUNIT3"] = "Angstrom"
    header["SPECSYS"] = "TOPOCENT"
    header["FIT_H5"] = str(fit_provenance["path"])[:68]
    header["N_MATCH"] = int(matched)
    header["DZBAND"] = fit_provenance["delta_z_band"]
    header["DPBAND"] = fit_provenance["delta_p_band"]
    for text in (
            "Bayesian calibration: posterior_z_mean applied as exp(z).",
            "Bayesian calibration: p_mean applied as wavelength-independent additive.",
            "Bayesian calibration: alpha_mean*K(lambda)*f(q) applied.",
            "NOT applied: delta_z_band spectral tilt or delta_p_band color term.",
            "NOT applied: external-image g or second gray normalization.",
            "p_good retained as QA only; never used to censor cube spectra.",
            "Low-p_good amplifier observations retained in cube.",
            "Source is observed VIRUS data; external_prediction is not cube SCI.",
            "Calibration equation: (Fibers.spectrum/Survey.offset-p-alpha*K*f(q)+skyspectrum)/exp(z)-skyspectrum.",
            "Residual sky applied after Bayesian amplifier calibration.",
            "Gaussian subpixel reconstruction; NCONTRIB >= 2 required for valid SCI."):
        header.add_history(text)
    header.add_history("Bayesian fit H5 identity: %s" % fit_provenance["identity"])
    header.add_history("delta_z_band=%0.12g and delta_p_band=%0.12g were provenance only." %
                       (fit_provenance["delta_z_band"], fit_provenance["delta_p_band"]))
    fits.PrimaryHDU(np.asarray(cube, dtype=np.float32), header=header).writeto(
        "%s_cube.fits" % surname, overwrite=True)
    variance_header = header.copy(); variance_header["BUNIT"] = "SCI units squared"; variance_header["EXTNAME"] = "VARIANCE"
    fits.PrimaryHDU(np.asarray(variancecube, dtype=np.float32), header=variance_header).writeto(
        "%s_variance_cube.fits" % surname, overwrite=True)
    error_header = header.copy(); error_header["BUNIT"] = "SCI units"; error_header["EXTNAME"] = "ERROR"
    fits.PrimaryHDU(np.sqrt(np.asarray(variancecube, dtype=np.float32)), header=error_header).writeto(
        "%s_errorcube.fits" % surname, overwrite=True)
    dq_header = header.copy(); dq_header["BUNIT"] = "bit mask"; dq_header["EXTNAME"] = "DQ"
    dq_header["DQBIT0"] = "NCONTRIB < 2"; dq_header["DQBIT1"] = "SCI valid, VAR unavailable"
    dq_header["DQBIT2"] = "empirical VAR adopted over formal"; dq_header["DQBIT3"] = "formal VAR adopted"
    dq_header["DQBIT4"] = "empirical-only VAR adopted"
    fits.PrimaryHDU(np.asarray(dqcube, dtype=np.uint16), header=dq_header).writeto(
        "%s_dq_cube.fits" % surname, overwrite=True)
    weight_header = header.copy(); weight_header["BUNIT"] = "Gaussian support"; weight_header["EXTNAME"] = "COVERAGE"
    fits.PrimaryHDU(np.asarray(weightcube, dtype=np.float32), header=weight_header).writeto(
        "%s_weight_cube.fits" % surname, overwrite=True)
    ncontrib_header = header.copy(); ncontrib_header["BUNIT"] = "count"; ncontrib_header["EXTNAME"] = "NCONTRIB"
    ncontrib_header["COMMENT"] = "Independent shots with a valid fiber within 2 sigma."
    fits.PrimaryHDU(np.asarray(ncontribcube, dtype=np.uint8), header=ncontrib_header).writeto(
        "%s_ncontrib_cube.fits" % surname, overwrite=True)
    Path("%s_from_fit_summary.json" % surname).write_text(json.dumps(summary, indent=2, default=str))


def _synthetic_calibration_identity():
    wave = np.linspace(3470.0, 5540.0, 17)
    truth = np.linspace(1.0, 2.0, wave.size)
    sky = np.linspace(.1, .2, wave.size)
    z, p, alpha = .13, .07, .41
    K = np.linspace(.8, 1.2, wave.size)
    f = .5
    working = np.exp(z) * (truth + sky) + p + alpha * K * f - sky
    recovered = (working - p - alpha * K * f + sky) / np.exp(z) - sky
    error_in = np.full(wave.size, .03)
    error_out = error_in / np.exp(z)
    if not np.allclose(recovered, truth, rtol=0.0, atol=1e-14):
        raise AssertionError("synthetic Bayesian calibration identity failed")
    if not np.allclose(error_out, error_in / np.exp(z), rtol=0.0, atol=0.0):
        raise AssertionError("synthetic calibrated-error identity failed")
    return {"status": "PASS", "max_abs_source_error": float(np.max(np.abs(recovered - truth))),
            "error_identity": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("h5files", help="input VIRUS H5 glob, e.g. '/data/*.h5'")
    parser.add_argument("surname", help="output filename prefix")
    parser.add_argument("image_center_size", help="RA,Dec center and size in arcmin")
    parser.add_argument("--fit-h5", required=True, help="completed Bayesian calibration H5")
    parser.add_argument("--fq-template", required=True, help="fixed q,f template CSV")
    parser.add_argument("--pixel-scale", type=float, default=1.0, help="output pixel scale in arcsec")
    parser.add_argument("--wave-workers", type=int, default=1, help="wavelength reconstruction threads")
    args = parser.parse_args()
    if args.pixel_scale <= 0 or not np.isfinite(args.pixel_scale):
        parser.error("--pixel-scale must be finite and positive")
    if args.wave_workers < 1:
        parser.error("--wave-workers must be at least 1")
    started = time.perf_counter()
    synthetic = _synthetic_calibration_identity()
    h5files = sorted(Path(path).resolve() for path in glob.glob(args.h5files))
    if not h5files:
        raise ValueError("no H5 files matched: %s" % args.h5files)
    if any(Path(path).name == "20200523_0000023.h5" for path in h5files):
        raise ValueError("20200523_0000023.h5 is explicitly excluded from the M101 sample")
    bayesian_read_started = time.perf_counter()
    calibration, _fit_metadata, fit_provenance = _read_fit_calibration(args.fit_h5)
    fit_provenance["read_seconds"] = time.perf_counter() - bayesian_read_started
    matching_started = time.perf_counter()
    matches, input_amplifiers = _preflight_matches(h5files, calibration)
    fit_provenance["matching_seconds"] = time.perf_counter() - matching_started
    print("Bayesian rows=%d; input amplifier observations=%d; matched=%d; unmatched=%d" %
          (fit_provenance["rows"], input_amplifiers, len(matches),
           input_amplifiers - len(matches)))
    fq_template = validated_m101.load_fq(args.fq_template)
    ra0, dec0, size_arcsec, n, xg, yg, xgrid, ygrid, tp = _parse_image_geometry(
        args.image_center_size, args.pixel_scale)
    print("Image grid: %d x %d at %.6g arcsec/pixel" % (n, n, args.pixel_scale))
    records = []
    total_fibers = 0
    virus_read_calibration_seconds = 0.0
    residual_seconds = 0.0
    for h5file in h5files:
        calibration_one_started = time.perf_counter()
        spectra, errors, ra, dec, labels, surveys = _calibrate_h5(
            h5file, matches, fq_template, tp, xg, yg)
        virus_read_calibration_seconds += time.perf_counter() - calibration_one_started
        total_fibers += len(ra)
        residual_sky_started = time.perf_counter()
        spectra = subtract_m101_residual_sky(
            spectra, ra, dec, labels, xg, yg, tp, h5file=h5file.name)
        residual_one_seconds = time.perf_counter() - residual_sky_started
        residual_seconds += residual_one_seconds
        print("residual sky %s: %.3f s" % (h5file.name, residual_one_seconds))
        records.append({"h5file": h5file, "spectra": spectra.astype(np.float32),
                        "errors": errors.astype(np.float32), "ra": ra, "dec": dec,
                        "labels": labels, "surveys": surveys})
    print("VIRUS H5 read/calibration: %.3f s" % virus_read_calibration_seconds)
    print("residual sky: %.3f s" % residual_seconds)

    offsets = []
    offset = 0
    for record in records:
        offsets.append(offset)
        offset += len(record["ra"])
    specarray = np.concatenate([record["spectra"] for record in records], axis=0)
    errarray = np.concatenate([record["errors"] for record in records], axis=0)
    raarray = np.empty((total_fibers, len(DEF_WAVE)), dtype=np.float32)
    decarray = np.empty_like(raarray)
    shot_indices, shot_names = [], []
    adr_started = time.perf_counter()
    for record, start in zip(records, offsets):
        extractor = Extract(wave=DEF_WAVE)
        for exposure in range(1, N_EXPOSURES + 1):
            selected = record["labels"] == exposure
            survey = record["surveys"][exposure]
            astrometry = Astrometry(float(survey["ra"]), float(survey["dec"]),
                                    float(survey["pa"]), 0.0, 0.0)
            extractor.get_ADR_RAdec(astrometry)
            indices = np.flatnonzero(selected)
            raarray[start + indices] = (record["ra"][indices, None]
                                         - extractor.ADRra[None, :] / 3600.0 /
                                         np.cos(np.deg2rad(float(survey["dec"]))))
            decarray[start + indices] = record["dec"][indices, None] - extractor.ADRdec[None, :] / 3600.0
            shot_indices.append(start + indices)
            shot_names.append("%s exposure %d" % (record["h5file"].name, exposure))
    print("ADR construction: %.3f s" % (time.perf_counter() - adr_started))

    cube = np.zeros((len(DEF_WAVE), n, n), dtype=np.float32)
    variancecube = np.zeros_like(cube)
    weightcube = np.zeros_like(cube)
    ncontribcube = np.zeros((len(DEF_WAVE), n, n), dtype=np.uint8)
    dqcube = np.zeros((len(DEF_WAVE), n, n), dtype=np.uint16)
    reconstruction_started = time.perf_counter()

    def render_wavelength(index):
        x, y = tp.wcs_world2pix(raarray[:, index], decarray[:, index], 1)
        Pos = np.column_stack((x, y))
        return index, make_image_gaussian(
            Pos, specarray[:, index], errarray[:, index], xg, yg, xgrid, ygrid,
            (GAUSSIAN_FWHM_ARCSEC / 2.35) / args.pixel_scale, shot_indices)

    with ThreadPoolExecutor(max_workers=args.wave_workers) as executor:
        for index, result in executor.map(render_wavelength, range(len(DEF_WAVE))):
            image, variance, coverage, ncontrib, dq, _ = result
            cube[index] = image; variancecube[index] = variance; weightcube[index] = coverage
            ncontribcube[index] = ncontrib; dqcube[index] = dq
    reconstruction_seconds = time.perf_counter() - reconstruction_started
    print("wavelength-plane reconstruction: %.3f s" % reconstruction_seconds)

    p_values = np.asarray([float(matches[key]["p_good"]) for key in matches
                           if np.isfinite(float(matches[key]["p_good"]))])
    z_values = np.asarray([float(record["posterior_z_mean"]) for record in calibration.values()])
    pmean_values = np.asarray([float(record["p_mean"]) for record in calibration.values()])
    alpha_values = np.asarray([float(record["alpha_mean"]) for record in calibration.values()])
    fit_summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "script": str(Path(__file__).resolve()),
        "input_h5_glob": args.h5files,
        "input_h5_files": [str(path) for path in h5files],
        "fit_h5": fit_provenance,
        "fq_template": str(Path(args.fq_template).expanduser().resolve()),
        "wave_grid": {"start_A": float(DEF_WAVE[0]), "stop_A": float(DEF_WAVE[-1]),
                      "step_A": 2.0, "planes": len(DEF_WAVE)},
        "image_center_size": {"RA": ra0, "Dec": dec0, "size_arcsec": size_arcsec,
                               "pixels": n, "pixel_scale_arcsec": args.pixel_scale},
        "bayesian_rows": fit_provenance["rows"],
        "input_amplifier_observations": input_amplifiers,
        "matched_amplifier_observations": len(matches),
        "unmatched_amplifier_observations": input_amplifiers - len(matches),
        "posterior_z_mean": _distribution(z_values),
        "posterior_m": _distribution(np.exp(z_values)),
        "p_mean": _distribution(pmean_values),
        "alpha_mean": _distribution(alpha_values),
        "p_good": _distribution(p_values),
        "p_good_lt_0.5": int(np.sum(p_values < .5)),
        "p_good_lt_0.1": int(np.sum(p_values < .1)),
        "delta_z_band": fit_provenance["delta_z_band"],
        "delta_p_band": fit_provenance["delta_p_band"],
        "delta_z_band_spectral_tilt_applied": False,
        "delta_p_band_spectral_color_applied": False,
        "low_p_good_retained": True,
        "external_image_g_applied": False,
        "second_gray_normalization_applied": False,
        "external_valid_used_as_cube_mask": False,
        "compact_external_mask_used_as_cube_mask": False,
        "hardware_date_masks": "validated_m101.masked_rows only",
        "calibration_equation": "(Fibers.spectrum/Survey.offset-p_mean-alpha_mean*K(lambda)*f(q)+Fibers.skyspectrum)/exp(posterior_z_mean)-Fibers.skyspectrum",
        "error_equation": "error_work/exp(posterior_z_mean), with error_work=Fibers.error/abs(Survey.offset)",
        "gaussian_reconstruction": {"fwhm_arcsec": GAUSSIAN_FWHM_ARCSEC,
                                    "fiber_radius_arcsec": FIBER_RADIUS_ARCSEC,
                                    "sigma_pixels": (GAUSSIAN_FWHM_ARCSEC / 2.35) / args.pixel_scale,
                                    "shots": len(shot_indices)},
        "synthetic_validation": synthetic,
        "timing_seconds": {
            "bayesian_h5_read": fit_provenance.get("read_seconds", np.nan),
            "calibration_key_matching": fit_provenance.get("matching_seconds", np.nan),
            "virus_h5_read_calibration": virus_read_calibration_seconds,
            "residual_sky": residual_seconds,
            "adr_construction": time.perf_counter() - adr_started,
            "wavelength_plane_reconstruction": reconstruction_seconds,
            "total_before_write": time.perf_counter() - started,
        },
        "no_calibration_fitting_or_optimization": True,
        "no_production_calibration_applied": True,
    }
    writing_started = time.perf_counter()
    _write_cube_products(args.surname, cube, variancecube, dqcube, weightcube,
                          ncontribcube, tp, n, args.pixel_scale, fit_provenance,
                          len(matches), fit_summary)
    fit_summary["timing_seconds"]["fits_writing"] = time.perf_counter() - writing_started
    fit_summary["timing_seconds"]["total_runtime"] = time.perf_counter() - started
    Path("%s_from_fit_summary.json" % args.surname).write_text(
        json.dumps(fit_summary, indent=2, default=str))
    print("FITS writing: %.3f s" % fit_summary["timing_seconds"]["fits_writing"])
    print("total runtime: %.3f s" % fit_summary["timing_seconds"]["total_runtime"])
    print(json.dumps(fit_summary, indent=2, default=str))


if __name__ == "__main__":
    main()
