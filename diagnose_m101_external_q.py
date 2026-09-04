#!/usr/bin/env python3
"""Externally referenced M101 amplifier/readout-coordinate diagnostic.

This is a read-only, one-H5 experiment.  It deliberately does not import the
production cube builder: no cube, H5 update, production normalization,
residual-sky subtraction, or amplifier correction is performed here.

The OFF-band provenance is retained explicitly.  The historical FITS header
uses ``Ooff OIII+30nm k1031`` / ``k1031`` while the matching SVO profile is
``KPNO/MOSAIC.OIIIoff`` (KPNO Mosaic OIII +29nm Offband, k1015).  This script
does not resolve or silently change that identifier discrepancy; it uses the
filter file supplied on the command line.
"""

from argparse import ArgumentParser
import csv
from pathlib import Path
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tables
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS
from scipy.ndimage import gaussian_filter
from scipy.optimize import least_squares
from scipy.ndimage import gaussian_filter1d
from scipy import sparse
from scipy.sparse.linalg import lsqr

from astrometry import Astrometry
from extract import Extract
from math_utils import biweight


DEF_WAVE = np.linspace(3470.0, 5540.0, 1036)
FIBERS_PER_AMPLIFIER = 112
FIBERS_PER_IFU = 448
EXPECTED_EXPOSURES = 3
AMPLIFIERS = ("LL", "LU", "RL", "RU")

# Copied from make_mosaic_cube_revised.py.  This diagnostic intentionally does
# not discover, infer, or add masks.
mask_dict = {
    "20200430-20200501": ["057RL", "057RU", "057LL", "057LU", "058RU",
                           "058RL", "021RL", "021RU", "021LL", "021LU"],
    "20200430-20200715": ["092LU", "094RU", "046RU", "104LL", "104LU",
                           "028RU", "028RL", "027LL", "027LU", "067LL",
                           "067LU", "025LU", "106RU", "106RL", "026LL",
                           "026LU", "026RL", "026RU", "103LL", "103LU"],
    "20200525-20200526": ["057RL", "057LL", "089RU", "089RL"],
    "20200517-20200522": ["030RL", "030RU", "030LL", "030LU"],
    "20200523-20200526": ["039RL", "039RU", "039LL", "039LU"],
    "20200622-20200715": ["089RL", "089RU", "096RL", "096RU"],
}


def text(value):
    if isinstance(value, (bytes, np.bytes_)):
        return value.decode("utf-8", errors="replace").strip()
    return str(value).strip()


def exposure_labels(nrows, nslots):
    """Reproduce the validated 112-row interleaved exposure grouping."""
    if nslots <= 0 or nrows % (FIBERS_PER_IFU * nslots) != 0:
        raise ValueError("cannot infer 3 interleaved exposures from H5 rows")
    nexp = nrows // (FIBERS_PER_IFU * nslots)
    if nexp != EXPECTED_EXPOSURES:
        raise ValueError("expected 3 exposures, inferred %d" % nexp)
    rows = np.arange(nrows, dtype=np.int64)
    return ((rows // FIBERS_PER_AMPLIFIER) % nexp + 1).astype(np.int16)


def build_groups(info):
    """Build physical exposure x IFUSLOT x amplifier groups in row order."""
    nrows = int(info.nrows)
    required = {"ifuslot", "amp", "specid", "ifuid"}
    if not required.issubset(info.colnames):
        raise ValueError("Info lacks validated amplifier bookkeeping columns")
    ifuslot = np.asarray(info.cols.ifuslot[:])
    amp = np.asarray([text(v) for v in info.cols.amp[:]])
    specid = np.asarray(info.cols.specid[:])
    ifuid = np.asarray(info.cols.ifuid[:])
    labels = exposure_labels(nrows, len(np.unique(ifuslot)))
    groups = []
    keys = sorted(set(zip(labels.tolist(), ifuslot.tolist(), amp.tolist())),
                  key=lambda k: (int(k[0]), int(k[1]), k[2]))
    for exposure, slot, amplifier in keys:
        indices = np.flatnonzero(
            (labels == exposure) & (ifuslot == slot) & (amp == amplifier))
        if indices.size != FIBERS_PER_AMPLIFIER:
            raise ValueError("exposure %d IFUSLOT %s AMP %s has %d rows, not 112" %
                             (exposure, slot, amplifier, indices.size))
        if np.unique(specid[indices]).size != 1 or np.unique(ifuid[indices]).size != 1:
            raise ValueError("inconsistent SPECID/IFUID in amplifier group")
        groups.append({
            "exposure": int(exposure), "ifuslot": int(slot),
            "ifuid": int(np.unique(ifuid[indices])[0]),
            "specid": int(np.unique(specid[indices])[0]),
            "amp": amplifier, "indices": indices,
            "j": np.arange(FIBERS_PER_AMPLIFIER, dtype=int),
        })
        groups[-1]["identity"] = (
            groups[-1]["exposure"], groups[-1]["specid"],
            groups[-1]["ifuslot"], groups[-1]["ifuid"],
            groups[-1]["amp"])
    if set(g["amp"] for g in groups) - set(AMPLIFIERS):
        raise ValueError("unexpected amplifier label in H5")
    return groups, labels


def masked_rows(h5_path, ifuslot, amp):
    """Return only the existing date-dependent production mask."""
    try:
        date = int(Path(h5_path).name.split("_")[0])
    except (IndexError, ValueError):
        return np.zeros(ifuslot.shape, dtype=bool)
    bad_names = []
    for key, values in mask_dict.items():
        start, stop = (int(v) for v in key.split("-"))
        if start <= date < stop:
            bad_names.extend(values)
    bad = np.zeros(ifuslot.shape, dtype=bool)
    for ifuamp in bad_names:
        bad |= ((ifuslot == int(ifuamp[:3])) & (amp == ifuamp[3:]))
    return bad


def read_filter(path):
    """Read filters exactly as the cube builder does."""
    table = Table.read(path, format="ascii")
    wavelength = np.asarray(table["Wavelength"], dtype=float)
    transmission = np.asarray(table["R"], dtype=float)
    response = np.interp(DEF_WAVE, wavelength, transmission,
                         left=0.0, right=0.0)
    if not np.any(np.isfinite(response) & (response != 0.0)):
        raise ValueError("filter %s has no usable response on DEF_WAVE" % path)
    return response


def synthetic_mean(spectra, errors, response):
    """Response-weighted mean, with a finite-spectrum denominator per fiber."""
    response = np.asarray(response, dtype=float)
    finite_response = np.isfinite(response) & (response != 0.0)
    finite = np.isfinite(spectra) & finite_response[np.newaxis, :]
    weights = np.where(finite, response[np.newaxis, :], 0.0)
    denominator = np.sum(weights, axis=1)
    numerator = np.sum(np.where(finite, spectra, 0.0) *
                       response[np.newaxis, :], axis=1)
    value = np.full(spectra.shape[0], np.nan, dtype=float)
    good = np.isfinite(denominator) & (denominator != 0.0)
    value[good] = numerator[good] / denominator[good]

    # Proper weighted-mean formal error is retained as lightweight QA only.
    band_error = np.full(spectra.shape[0], np.nan, dtype=float)
    finite_error = np.isfinite(errors) & finite
    variance = np.sum(np.where(finite_error, errors ** 2 * weights ** 2, 0.0), axis=1)
    band_error[good] = np.sqrt(variance[good]) / np.abs(denominator[good])
    return value, band_error


def adr_positions(ra, dec, survey_row, response):
    """Apply the same Extract/Astrometry ADR machinery used by the builder."""
    survey_ra = float(survey_row["ra"])
    survey_dec = float(survey_row["dec"])
    pa = float(survey_row["pa"])
    if not all(np.isfinite(v) for v in (survey_ra, survey_dec, pa)):
        raise ValueError("Survey RA/Dec/PA is invalid")
    effective = Astrometry(survey_ra, survey_dec, pa, 0.0, 0.0)
    extractor = Extract(wave=DEF_WAVE)
    extractor.get_ADR_RAdec(effective)
    dra = extractor.ADRra / 3600.0 / np.cos(np.deg2rad(survey_dec))
    ddec = extractor.ADRdec / 3600.0
    # The historical code uses the fiber Info position minus ADR displacement.
    ra_wave = ra[:, None] - dra[None, :]
    dec_wave = dec[:, None] - ddec[None, :]
    finite_response = np.isfinite(response) & (response != 0.0)
    weights = np.where(finite_response, response, 0.0)
    denom = np.sum(weights)
    if denom == 0.0:
        raise ValueError("filter has zero ADR weighting")
    return (np.sum(ra_wave * weights[None, :], axis=1) / denom,
            np.sum(dec_wave * weights[None, :], axis=1) / denom)


def load_image(path):
    with fits.open(path, memmap=True) as hdul:
        data = np.asarray(hdul[0].data, dtype=float).copy()
        header = hdul[0].header.copy()
    if data.ndim != 2:
        raise ValueError("%s must contain a 2D primary image" % path)
    return {"data": data, "wcs": WCS(header), "path": Path(path)}


def smooth_image(data, sigma):
    if sigma <= 0.0:
        return np.asarray(data, dtype=float).copy()
    finite = np.isfinite(data)
    numerator = gaussian_filter(np.where(finite, data, 0.0), sigma,
                                mode="nearest")
    denominator = gaussian_filter(finite.astype(float), sigma,
                                  mode="nearest")
    result = np.full(data.shape, np.nan, dtype=float)
    good = denominator > 0.0
    result[good] = numerator[good] / denominator[good]
    return result


def sample_image(image, ra, dec, sigma):
    """Nearest-pixel WCS sample with no radial or blank-sky selection."""
    sampled_image = smooth_image(image["data"], sigma)
    x, y = image["wcs"].world_to_pixel_values(ra, dec)
    finite_xy = np.isfinite(x) & np.isfinite(y)
    xi = np.zeros(x.shape, dtype=int)
    yi = np.zeros(y.shape, dtype=int)
    xi[finite_xy] = np.rint(x[finite_xy]).astype(int)
    yi[finite_xy] = np.rint(y[finite_xy]).astype(int)
    valid = (finite_xy & (xi >= 0) & (xi < sampled_image.shape[1]) &
             (yi >= 0) & (yi < sampled_image.shape[0]))
    value = np.full(ra.shape, np.nan, dtype=float)
    value[valid] = sampled_image[yi[valid], xi[valid]]
    valid &= np.isfinite(value)

    # Pixel-coordinate gradient is a deliberately descriptive mismatch QA.
    gradient = np.full(ra.shape, np.nan, dtype=float)
    finite_for_gradient = np.isfinite(sampled_image)
    if finite_for_gradient.any():
        gy, gx = np.gradient(np.where(finite_for_gradient, sampled_image, 0.0))
        gradient[valid] = np.hypot(gx[yi[valid], xi[valid]],
                                   gy[yi[valid], xi[valid]])
    return value, gradient, valid


def robust_scale(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan
    center = np.median(values)
    mad = np.median(np.abs(values - center))
    if np.isfinite(mad) and mad > 0.0:
        return 1.4826 * mad
    rms = np.sqrt(np.mean((values - center) ** 2))
    return float(rms) if rms > 0.0 else 1.0


def robust_rms(residual):
    """Robust RMS reported throughout: 1.4826 times residual MAD."""
    return robust_scale(residual)


def fit_global(x, y, intercept):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if x.size < 3:
        return {"g": np.nan, "z": np.nan, "rms": np.nan, "n": int(x.size),
                "residual": np.full(x.shape, np.nan)}
    g0 = np.dot(x, y) / np.dot(x, x) if np.dot(x, x) != 0.0 else 0.0
    z0 = float(np.median(y - g0 * x)) if intercept else 0.0
    p0 = np.array([g0, z0]) if intercept else np.array([g0])

    def residual(params):
        model = params[0] * x + (params[1] if intercept else 0.0)
        return y - model

    scale = max(robust_scale(y), np.finfo(float).eps)
    fit = least_squares(residual, p0, loss="soft_l1", f_scale=scale,
                        x_scale="jac", max_nfev=1000)
    fitted = residual(fit.x)
    return {"g": float(fit.x[0]),
            "z": float(fit.x[1]) if intercept else 0.0,
            "rms": float(robust_rms(fitted)), "n": int(x.size),
            "residual": fitted}


def fixed_intercept_slope(x, y, z):
    """Descriptive amplifier slope with the global z held fixed."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if x.size < 3 or np.dot(x, x) == 0.0:
        return np.nan
    p0 = np.array([np.dot(x, y - z) / np.dot(x, x)])
    scale = max(robust_scale(y), np.finfo(float).eps)
    fit = least_squares(lambda p: y - (p[0] * x + z), p0,
                        loss="soft_l1", f_scale=scale, x_scale="jac",
                        max_nfev=1000)
    return float(fit.x[0])


def raw_work_basis(survey_row):
    """Quick Reduction Raw -> working Fibers basis, explicitly no offset."""
    exptime = float(survey_row["exptime"])
    millum = float(survey_row["millum"])
    guider_throughput = float(survey_row["throughput"])
    if not np.isfinite(exptime) or exptime == 0.0:
        raise ValueError("Survey.exptime is invalid")
    gratio = millum * guider_throughput / 5e5
    if not np.isfinite(gratio) or gratio == 0.0:
        raise ValueError("Survey guider ratio is invalid")
    throughput_path = Path(__file__).resolve().parent / "CALS" / "throughput.txt"
    table = Table.read(throughput_path, format="ascii.fixed_width_two_line")
    standard = np.asarray(table["throughput"], dtype=float)
    standard_wave = np.asarray(table["wavelength"], dtype=float)
    if standard.size != DEF_WAVE.size or not np.allclose(
            standard_wave, DEF_WAVE, rtol=0.0, atol=1e-6):
        raise ValueError("CALS/throughput.txt does not match DEF_WAVE")
    if np.any(~np.isfinite(standard) | (standard == 0.0)):
        raise ValueError("CALS/throughput.txt contains invalid throughput")

    mult_fac = (6.626e-27 * (3e18 / DEF_WAVE) / 360.0 /
                5e5 / 0.92 * 5)
    mult_fac *= 1e29 * DEF_WAVE ** 2 / 2.99792e18
    final_norm = 1e-29 * 2.99792e18 / DEF_WAVE ** 2 * 1e17
    # Survey.offset is intentionally absent because Fibers.spectrum was
    # already divided by it above.
    return mult_fac * (360.0 / exptime) / standard / gratio * final_norm


def weighted_scalar(values, response):
    valid = np.isfinite(values) & np.isfinite(response) & (response != 0.0)
    denominator = np.sum(np.where(valid, response, 0.0))
    return (float(np.sum(np.where(valid, values * response, 0.0)) /
                  denominator) if denominator != 0.0 else np.nan)


def load_template(path):
    """Load one explicit A(q) column and freeze its q morphology."""
    with Path(path).open(newline="") as stream:
        reader = csv.DictReader(stream)
        fields = {f.lower(): f for f in (reader.fieldnames or [])}
        if "amplifier" not in fields or "q" not in fields:
            raise ValueError("template CSV needs amplifier and q columns")
        source_key = "a_raw_applied_average"
        if source_key not in fields:
            raise ValueError("template CSV needs A_raw_applied_average")
        source = fields[source_key]
        print("template source column: A_raw_applied_average")
        values = {
            amp: [[] for _ in range(FIBERS_PER_AMPLIFIER)]
            for amp in AMPLIFIERS
        }
        for row in reader:
            amp = row[fields["amplifier"]].strip()
            if amp not in values:
                continue
            q = int(float(row[fields["q"]]))
            if 0 <= q < FIBERS_PER_AMPLIFIER:
                value = float(row[source])
                if np.isfinite(value):
                    values[amp][q].append(value)
    templates = {}
    for amp, value in values.items():
        T = np.full(FIBERS_PER_AMPLIFIER, np.nan, dtype=float)
        for q, samples in enumerate(value):
            if samples:
                T[q] = np.median(samples)
        scale = np.nanmedian(T[:20])
        if not np.isfinite(scale) or scale == 0.0:
            raise ValueError("template %s lacks a finite nonzero q<20 median" % amp)
        template = T / scale
        if not np.all(np.isfinite(template)):
            raise ValueError("template %s has missing/nonfinite q values" % amp)
        tail_max_abs = float(np.max(np.abs(template[40:])))
        print("%s q>=40 max abs template = %g" % (amp, tail_max_abs))
        if not np.allclose(template[40:], 0.0, rtol=0.0, atol=1e-12):
            raise ValueError("template %s has nonzero q>=40 tail" % amp)
        # Remove insignificant CSV floating-point roundoff so the plotted
        # frozen production-style template is exactly flat beyond q=40.
        template[40:] = 0.0
        templates[amp] = template
    return templates


def fit_template(y, template):
    valid = np.isfinite(y) & np.isfinite(template)
    y, template = np.asarray(y)[valid], np.asarray(template)[valid]
    if y.size < 3:
        return {"C": np.nan, "alpha": np.nan, "rms_before": np.nan,
                "rms_after": np.nan, "alpha_c0": np.nan,
                "rms_c0": np.nan, "reduction": np.nan}
    scale = max(robust_scale(y), np.finfo(float).eps)
    fit = least_squares(lambda p: y - (p[0] + p[1] * template),
                        [float(np.median(y)), 1.0], loss="soft_l1",
                        f_scale=scale, x_scale="jac", max_nfev=1000)
    fit0 = least_squares(lambda p: y - p[0] * template, [1.0],
                         loss="soft_l1", f_scale=scale, x_scale="jac",
                         max_nfev=1000)
    after = y - (fit.x[0] + fit.x[1] * template)
    before = y
    c0 = y - fit0.x[0] * template
    return {"C": float(fit.x[0]), "alpha": float(fit.x[1]),
            "rms_before": float(robust_rms(before)),
            "rms_after": float(robust_rms(after)),
            "alpha_c0": float(fit0.x[0]), "rms_c0": float(robust_rms(c0)),
            "reduction": float(robust_rms(before) - robust_rms(after))}


def joint_model_residual(params, image, observed, k_band, amp_code,
                         fiber_template):
    """Residual for J0: V = g I + z + K alpha_amp f_amp(q)."""
    g, z = params[:2]
    alpha = params[2:6]
    model = g * image + z + k_band * alpha[amp_code] * fiber_template
    return observed - model


def fit_joint_model(image, observed, k_band, amp_code, fiber_template,
                    sequential):
    """Fit the six-parameter individual-fiber J0 model robustly."""
    initial = np.array([
        sequential["g"], sequential["z"], 0.0, 0.0, 0.0, 0.0
    ], dtype=float)
    scale = max(robust_scale(observed), np.finfo(float).eps)
    fit = least_squares(
        joint_model_residual, initial,
        args=(image, observed, k_band, amp_code, fiber_template),
        loss="soft_l1", f_scale=scale, x_scale="jac", max_nfev=2000)
    residual = joint_model_residual(
        fit.x, image, observed, k_band, amp_code, fiber_template)
    return {"params": fit.x, "residual": residual,
            "rms": float(robust_rms(residual))}


def relative_c_values(c_parameters, amp_counts):
    """Derive C_RU from a valid-count-weighted zero-sum constraint."""
    counts = np.asarray(amp_counts, dtype=float)
    if counts[3] <= 0.0:
        raise ValueError("cannot impose relative-C constraint without RU fibers")
    c = np.zeros(4, dtype=float)
    c[:3] = c_parameters
    c[3] = -np.dot(counts[:3], c[:3]) / counts[3]
    return c


def relative_c_model_residual(params, image, observed, k_band, amp_code,
                              fiber_template, amp_counts):
    """Residual for J1 with count-weighted sum(C_amp)=0."""
    g, z = params[:2]
    alpha = params[2:6]
    c = relative_c_values(params[6:9], amp_counts)
    model = (g * image + z + k_band *
             (alpha[amp_code] * fiber_template + c[amp_code]))
    return observed - model


def fit_relative_c_model(image, observed, k_band, amp_code, fiber_template,
                         j0):
    """Fit descriptive J1; C_RU is derived, never independently fitted."""
    initial = np.r_[j0["params"], 0.0, 0.0, 0.0]
    amp_counts = np.bincount(amp_code, minlength=4)
    scale = max(robust_scale(observed), np.finfo(float).eps)
    fit = least_squares(
        relative_c_model_residual, initial,
        args=(image, observed, k_band, amp_code, fiber_template, amp_counts),
        loss="soft_l1", f_scale=scale, x_scale="jac", max_nfev=2000)
    residual = relative_c_model_residual(
        fit.x, image, observed, k_band, amp_code, fiber_template, amp_counts)
    return {"params": fit.x, "C": relative_c_values(fit.x[6:9], amp_counts),
            "residual": residual, "rms": float(robust_rms(residual)),
            "amp_counts": amp_counts}


def state_name(sigma):
    return "native" if sigma == 0.0 else "sigma_%g_pix" % sigma


def stack_profile(values, ifus):
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return np.nan, np.nan, np.nan, 0, 0
    finite = np.isfinite(values)
    if not finite.any():
        return np.nan, np.nan, np.nan, 0, 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        median = float(np.nanmedian(values))
        location = float(biweight(values[finite]))
        p16, p84 = np.nanpercentile(values, [16, 84])
    return (median, location if np.isfinite(location) else median,
            float((p84 - p16) / 2.0), int(finite.sum()),
            int(np.unique(np.asarray(ifus)[finite]).size))


def write_csvs(output_dir, profile_rows, calibration_rows):
    profile_fields = [
        "h5", "exposure", "band", "smoothing_state", "amplifier", "q",
        "D_median", "D_scatter", "D_origin_median", "D_raw_equiv_e_per_A",
        "D_biweight", "D_origin_raw_equiv_e_per_A", "n_fibers", "n_physical_ifus",
        "external_gradient_median", "C_template_e_per_A",
        "joint_residual_raw_equiv", "joint_plus_C_residual_raw_equiv",
        "alpha_template_e_per_A", "template_rms_before_e_per_A",
        "template_rms_after_e_per_A", "template_alpha_c0_e_per_A",
        "template_rms_c0_e_per_A",
    ]
    with (output_dir / "external_q_profiles.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=profile_fields)
        writer.writeheader()
        writer.writerows(profile_rows)

    calibration_fields = [
        "h5", "exposure", "band", "smoothing_state", "valid_fibers",
        "g_origin", "rms_origin", "g_intercept", "z", "rms_intercept",
        "z_relative_importance", "K_band", "g_LL_over_global",
        "g_LU_over_global", "g_RL_over_global", "g_RU_over_global",
        "abs_D_gradient_correlation", "high_gradient_abs_D_ratio",
        "g_sequential", "z_sequential", "rms_before_joint", "g_joint",
        "z_joint", "alpha_LL_joint", "alpha_LU_joint", "alpha_RL_joint",
        "alpha_RU_joint", "rms_joint", "C_LL_sequential",
        "C_LU_sequential", "C_RL_sequential", "C_RU_sequential",
        "alpha_LL_sequential", "alpha_LU_sequential",
        "alpha_RL_sequential", "alpha_RU_sequential", "C_LL_relative",
        "C_LU_relative", "C_RL_relative", "C_RU_relative",
        "rms_joint_with_C", "fractional_improvement_relative_C",
    ]
    with (output_dir / "external_q_calibration_summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=calibration_fields)
        writer.writeheader()
        writer.writerows(calibration_rows)


def plot_profiles(output_dir, profiles, bands, sigmas, primary_sigma):
    """Write one q-profile figure per smoothing state plus the required name."""
    x = np.arange(FIBERS_PER_AMPLIFIER)
    for sigma in sigmas:
        fig, axes = plt.subplots(2, 4, figsize=(16, 7), sharex=True)
        label = state_name(sigma)
        for iband, band in enumerate(bands):
            for iamp, amp in enumerate(AMPLIFIERS):
                axis = axes[iband, iamp]
                key = (band, sigma, amp)
                records = profiles[key]
                for exposure, record in sorted(records.items()):
                    axis.plot(x, record["D"], lw=0.8, alpha=0.28,
                              label="exp %d" % exposure)
                    axis.fill_between(x, record["D"] - record["scatter"],
                                      record["D"] + record["scatter"],
                                      alpha=0.06)
                if records:
                    stack = np.asarray([r["D"] for r in records.values()])
                    median = np.nanmedian(stack, axis=0)
                    p16, p84 = np.nanpercentile(stack, [16, 84], axis=0)
                    axis.fill_between(x, p16, p84, alpha=0.14)
                    axis.plot(x, median, lw=1.5, label="exposure median")
                    stack_a = np.asarray([r["D_origin"] for r in records.values()])
                    axis.plot(x, np.nanmedian(stack_a, axis=0), "--", lw=0.9,
                              label="model A" if iamp == 0 and iband == 0 else None)
                axis.axhline(0.0, color="k", lw=0.7)
                axis.axvline(40.0, color="tab:red", ls=":", lw=0.8)
                axis.set_title(amp if iband == 0 else "")
                axis.grid(alpha=0.2)
                if iamp == 0:
                    axis.set_ylabel("%s D in working units" % band)
                if iband == 1:
                    axis.set_xlabel("folded readout coordinate q")
        axes[0, 0].legend(fontsize=7)
        fig.suptitle("External-image q profiles (%s); no central anchoring" % label)
        fig.tight_layout()
        filename = "external_q_profiles.png" if sigma == primary_sigma else \
            "external_q_profiles_%s.png" % label
        fig.savefig(output_dir / filename, dpi=170)
        plt.close(fig)


def plot_raw_equivalent(output_dir, profiles, bands, sigmas, primary_sigma):
    x = np.arange(FIBERS_PER_AMPLIFIER)
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5), sharex=True)
    for iamp, amp in enumerate(AMPLIFIERS):
        axis = axes[iamp]
        for sigma in sigmas:
            styles = ["-", "--"] if sigma == primary_sigma else [":", "-."]
            for iband, band in enumerate(bands):
                records = profiles[(band, sigma, amp)]
                if not records:
                    continue
                stack = np.asarray([r["D_raw"] for r in records.values()])
                median = np.nanmedian(stack, axis=0)
                axis.plot(x, median, styles[iband], lw=1.25,
                          label="%s %s" % (band, state_name(sigma)))
        axis.axhline(0.0, color="k", lw=0.7)
        axis.axvline(40.0, color="tab:red", ls=":", lw=0.8)
        axis.set_title(amp)
        axis.set_xlabel("q")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("D_raw_equiv [e-/A]")
    axes[0].legend(fontsize=7)
    fig.suptitle("ON/OFF external q profiles in Raw-equivalent units")
    fig.tight_layout()
    fig.savefig(output_dir / "external_q_profiles_raw_equivalent.png", dpi=170)
    plt.close(fig)


def plot_amp_qa(output_dir, calibration_rows, bands, sigmas):
    fig, axes = plt.subplots(1, len(bands), figsize=(10, 4), sharey=True)
    axes = np.atleast_1d(axes)
    colors = {"LL": "tab:blue", "LU": "tab:orange", "RL": "tab:green", "RU": "tab:red"}
    for iband, band in enumerate(bands):
        axis = axes[iband]
        for state in [state_name(s) for s in sigmas]:
            rows = [r for r in calibration_rows if r["band"] == band and
                    r["smoothing_state"] == state]
            for amp in AMPLIFIERS:
                vals = [float(r["g_%s_over_global" % amp]) for r in rows
                        if np.isfinite(float(r["g_%s_over_global" % amp]))]
                if vals:
                    axis.plot(np.arange(len(vals)), vals, "o-", ms=3,
                              color=colors[amp], alpha=0.8,
                              label=amp if state == state_name(sigmas[0]) else None)
        axis.axhline(1.0, color="k", lw=0.7)
        axis.set_title(band)
        axis.set_xlabel("exposure / smoothing row")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("descriptive g_amp / g_global")
    axes[0].legend(fontsize=8)
    fig.suptitle("Amplifier-dependent multiplicative QA only")
    fig.tight_layout()
    fig.savefig(output_dir / "external_q_amplifier_normalization_qa.png", dpi=170)
    plt.close(fig)


def plot_template(output_dir, profiles, templates, bands, sigma):
    x = np.arange(FIBERS_PER_AMPLIFIER)
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5), sharex=True)
    for iamp, amp in enumerate(AMPLIFIERS):
        axis = axes[iamp]
        for band, color in zip(bands, ("tab:blue", "tab:orange")):
            records = profiles[(band, sigma, amp)]
            for exposure, record in sorted(records.items()):
                axis.plot(x, record["D_raw"], color=color, alpha=0.22, lw=0.8)
            if records:
                stack = np.asarray([r["D_raw"] for r in records.values()])
                y = np.nanmedian(stack, axis=0)
                axis.plot(x, y, color=color, lw=1.4, label=band)
                fits = [r["template"] for r in records.values()]
                c = np.nanmedian([r["C"] for r in fits])
                alpha = np.nanmedian([r["alpha"] for r in fits])
                axis.plot(x, c + alpha * templates[amp], color=color,
                          ls="--", lw=1.0)
                annotation = "%s C=%+.2g, α=%+.2g" % (band, c, alpha)
                axis.text(0.02, 0.96 - 0.10 * bands.index(band), annotation,
                          transform=axis.transAxes, color=color, fontsize=8,
                          va="top")
        axis2 = axis.twinx()
        axis2.plot(x, templates[amp], "k:", lw=0.9, label="fixed f(q)")
        axis2.set_ylabel("f(q) normalized")
        axis.axhline(0.0, color="k", lw=0.7)
        axis.axvline(40.0, color="tab:red", ls=":", lw=0.8)
        axis.set_title(amp)
        axis.set_xlabel("q")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("D_raw_equiv [e-/A]")
    axes[0].legend(fontsize=8)
    fig.suptitle("Frozen A(q) template fits (%s); C is free" % state_name(sigma))
    fig.tight_layout()
    fig.savefig(output_dir / "external_q_template_fits.png", dpi=170)
    plt.close(fig)


def profile_median(records, field):
    if not records:
        return np.full(FIBERS_PER_AMPLIFIER, np.nan)
    stack = np.asarray([record[field] for record in records.values()])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmedian(stack, axis=0)


def parameter_median(rows, field):
    values = [float(row[field]) for row in rows
              if np.isfinite(float(row[field]))]
    return float(np.median(values)) if values else np.nan


def plot_joint_residuals(output_dir, profiles, calibration_rows, bands, sigmas,
                         primary_sigma):
    """Plot individual-fiber J0 residuals stacked only afterward in q."""
    x = np.arange(FIBERS_PER_AMPLIFIER)
    fig, axes = plt.subplots(2, 4, figsize=(16, 7), sharex=True)
    primary_label = state_name(primary_sigma)
    for iband, band in enumerate(bands):
        for iamp, amp in enumerate(AMPLIFIERS):
            axis = axes[iband, iamp]
            records = profiles[(band, primary_sigma, amp)]
            primary = profile_median(records, "joint_raw")
            axis.plot(x, primary, lw=1.5, label=primary_label)
            if 0.0 in sigmas and primary_sigma != 0.0:
                native = profile_median(profiles[(band, 0.0, amp)], "joint_raw")
                axis.plot(x, native, "--", lw=0.9, alpha=0.75, label="native")
            axis.axhline(0.0, color="k", lw=0.7)
            axis.axvline(40.0, color="tab:red", ls=":", lw=0.8)
            rows = [row for row in calibration_rows
                    if row["band"] == band and
                    row["smoothing_state"] == primary_label]
            alpha = parameter_median(rows, "alpha_%s_joint" % amp)
            low = np.nanmedian(primary[:20])
            high = np.nanmedian(primary[40:])
            axis.text(0.03, 0.96,
                      "α=%+.3g\nq<20=%+.3g\nq≥40=%+.3g" %
                      (alpha, low, high), transform=axis.transAxes,
                      va="top", fontsize=8)
            axis.set_title(amp if iband == 0 else "")
            axis.grid(alpha=0.2)
            if iamp == 0:
                axis.set_ylabel("%s joint residual [e-/A]" % band)
            if iband == 1:
                axis.set_xlabel("folded readout coordinate q")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("J0 simultaneous-fit residuals (%s)" % primary_label)
    fig.tight_layout()
    fig.savefig(output_dir / "external_q_joint_fit_residuals.png", dpi=170)
    plt.close(fig)


def plot_joint_vs_sequential(output_dir, profiles, bands, primary_sigma):
    x = np.arange(FIBERS_PER_AMPLIFIER)
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5), sharex=True)
    colors = {"ON": "tab:blue", "OFF": "tab:orange"}
    for iamp, amp in enumerate(AMPLIFIERS):
        axis = axes[iamp]
        for band in bands:
            records = profiles[(band, primary_sigma, amp)]
            axis.plot(x, profile_median(records, "D_raw"), color=colors[band],
                      lw=1.1, label="%s sequential" % band)
            axis.plot(x, profile_median(records, "joint_raw"), "--",
                      color=colors[band], lw=1.2, label="%s joint" % band)
        axis.axhline(0.0, color="k", lw=0.7)
        axis.axvline(40.0, color="tab:red", ls=":", lw=0.8)
        axis.set_title(amp)
        axis.set_xlabel("q")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("Raw-equivalent residual [e-/A]")
    axes[0].legend(fontsize=7)
    fig.suptitle("Joint versus sequential q residuals (%s)" % state_name(primary_sigma))
    fig.tight_layout()
    fig.savefig(output_dir / "external_q_joint_vs_sequential.png", dpi=170)
    plt.close(fig)


def plot_joint_parameters(output_dir, calibration_rows, bands, primary_sigma):
    """Compare independent ON/OFF J0 alpha values and global zero points."""
    label = state_name(primary_sigma)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    colors = {"LL": "tab:blue", "LU": "tab:orange", "RL": "tab:green", "RU": "tab:red"}
    on_rows = {(int(row["exposure"]), row["band"], row["smoothing_state"]): row
               for row in calibration_rows if row["band"] == "ON" and
               row["smoothing_state"] == label}
    off_rows = {(int(row["exposure"]), row["band"], row["smoothing_state"]): row
                for row in calibration_rows if row["band"] == "OFF" and
                row["smoothing_state"] == label}
    for amp in AMPLIFIERS:
        xs, ys = [], []
        for exposure in sorted(set(e for e, _, _ in on_rows) &
                               set(e for e, _, _ in off_rows)):
            on = on_rows[(exposure, "ON", label)]
            off = off_rows[(exposure, "OFF", label)]
            x, y = float(on["alpha_%s_joint" % amp]), float(off["alpha_%s_joint" % amp])
            if np.isfinite(x) and np.isfinite(y):
                xs.append(x); ys.append(y)
        axes[0].plot(xs, ys, "o", color=colors[amp], label=amp)
    finite_alpha = np.isfinite(axes[0].get_xlim()).all()
    if finite_alpha:
        lo = min(axes[0].get_xlim()); hi = max(axes[0].get_xlim())
        axes[0].plot([lo, hi], [lo, hi], "k:", lw=0.8)
    axes[0].set_xlabel("ON alpha_joint [e-/A]")
    axes[0].set_ylabel("OFF alpha_joint [e-/A]")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.2)

    xs, ys = [], []
    for exposure in sorted(set(e for e, _, _ in on_rows) &
                           set(e for e, _, _ in off_rows)):
        x, y = float(on_rows[(exposure, "ON", label)]["z_joint"]), \
            float(off_rows[(exposure, "OFF", label)]["z_joint"])
        if np.isfinite(x) and np.isfinite(y):
            xs.append(x); ys.append(y)
    axes[1].plot(xs, ys, "o", color="k")
    if xs:
        lo = min(min(xs), min(ys)); hi = max(max(xs), max(ys))
        axes[1].plot([lo, hi], [lo, hi], "k:", lw=0.8)
    axes[1].set_xlabel("ON z_joint [working units]")
    axes[1].set_ylabel("OFF z_joint [working units]")
    if xs:
        axes[1].text(0.03, 0.97,
                     "ON median=%+.3g\nOFF median=%+.3g" %
                     (np.median(xs), np.median(ys)),
                     transform=axes[1].transAxes, va="top", fontsize=8)
    axes[1].grid(alpha=0.2)
    fig.suptitle("ON/OFF joint parameters (%s)" % label)
    fig.tight_layout()
    fig.savefig(output_dir / "external_q_joint_parameters.png", dpi=170)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Common-f(q) external ON/OFF diagnostic.
#
# This is deliberately self-contained.  In particular, it does not import
# make_mosaic_cube_revised.py, and it never writes an H5 file or a production
# product.


def load_common_fq_template(path):
    """Read and strictly validate the explicit q=0..111 initialization."""
    path = Path(path)
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        fields = {f.strip().lower(): f for f in (reader.fieldnames or [])}
        if "q" not in fields or "f" not in fields:
            raise ValueError("common f(q) CSV needs q and f columns")
        rows = []
        for row in reader:
            try:
                q_float = float(row[fields["q"]])
                q = int(q_float)
                value = float(row[fields["f"]])
            except (TypeError, ValueError) as exc:
                raise ValueError("invalid q/f row in %s" % path) from exc
            if not np.isfinite(q_float) or q_float != q:
                raise ValueError("q must contain integer values in %s" % path)
            rows.append((q, value))
    if len(rows) != FIBERS_PER_AMPLIFIER:
        raise ValueError("%s must contain exactly q=0..111" % path)
    q_values = [q for q, _ in rows]
    if sorted(q_values) != list(range(FIBERS_PER_AMPLIFIER)):
        raise ValueError("%s must contain each q exactly once for q=0..111" % path)
    f_initial = np.full(FIBERS_PER_AMPLIFIER, np.nan, dtype=float)
    for q, value in rows:
        f_initial[q] = value
    if not np.all(np.isfinite(f_initial)):
        raise ValueError("%s contains nonfinite f values" % path)

    low = float(np.median(f_initial[:20]))
    tail = float(np.median(f_initial[40:]))
    peak_q = int(np.argmax(f_initial))
    peak_f = float(f_initial[peak_q])
    print("common f(q) initialization: %s" % path)
    print("  median f(q<20)=%+.8g, median f(q>=40)=%+.8g, "
          "peak q=%d, peak f=%+.8g" % (low, tail, peak_q, peak_f))
    if not np.isclose(low, 1.0, atol=0.05) or not np.isclose(tail, 0.0, atol=0.05):
        print("  WARNING: input normalization differs materially from the "
              "requested convention; input was not silently renormalized")
    if peak_q < 0 or peak_q >= FIBERS_PER_AMPLIFIER:
        raise ValueError("common f(q) peak is outside q=0..111")
    return f_initial


def _huber_weights(residual, scale):
    residual = np.asarray(residual, dtype=float)
    if not np.isfinite(scale) or scale <= 0.0:
        return np.ones(residual.shape, dtype=float)
    threshold = 1.345 * scale
    absolute = np.abs(residual)
    return np.where(absolute > threshold, threshold / absolute, 1.0)


def _finite_percentiles(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan, np.nan, np.nan
    p16, p84 = np.percentile(values, [16, 84])
    return float(np.median(values)), float(p16), float(p84)


def _residual_summary(values):
    median, p16, p84 = _finite_percentiles(values)
    return {"median": median, "p16": p16, "p84": p84,
            "robust_rms": robust_rms(values)}


def _smooth_fq(values, fallback):
    """Mildly smooth q samples without allowing NaNs to bleed into f(q)."""
    values = np.asarray(values, dtype=float)
    fallback = np.asarray(fallback, dtype=float)
    finite = np.isfinite(values)
    numerator = gaussian_filter1d(np.where(finite, values, 0.0), 2.5,
                                  mode="nearest")
    denominator = gaussian_filter1d(finite.astype(float), 2.5,
                                   mode="nearest")
    result = np.array(fallback, copy=True)
    good = denominator > 1.e-12
    result[good] = numerator[good] / denominator[good]
    return result


def _normalize_fq(f_raw, alpha):
    """Apply the requested convention and preserve the multiplicative scale.

    First the smoothed shape is centered by its far-side median.  The
    subsequent division by ``norm`` is a pure multiplicative reparameterization:

        f_new = f_centered / norm
        alpha_new = alpha_old * norm

    Therefore alpha_new*f_new equals alpha_old*f_centered exactly.  The
    centering subtraction is the explicit requested zero-baseline convention;
    its amplifier-dependent constant is allowed to be re-estimated by the
    next sparse Step-A solve rather than being hidden in a global z.
    """
    f_raw = np.asarray(f_raw, dtype=float)
    alpha = np.asarray(alpha, dtype=float)
    baseline = float(np.median(f_raw[40:]))
    centered = f_raw - baseline
    norm = float(np.median(centered[:20]))
    if not np.isfinite(norm) or abs(norm) <= np.finfo(float).eps:
        raise ValueError("common f(q) update has zero q<20 normalization")
    return centered / norm, alpha * norm, baseline, norm


def _build_sparse_design(datasets, f_current, alpha_count):
    """Build V = gI + z + K alpha*f with one alpha column per identity."""
    global_keys = sorted({(d["exposure"], d["band"]) for d in datasets})
    global_index = {key: index for index, key in enumerate(global_keys)}
    n_global = 2 * len(global_keys)
    rows, cols, values, target = [], [], [], []
    row_number = 0
    for dataset in datasets:
        valid = dataset["valid"]
        indices = np.flatnonzero(valid)
        global_number = global_index[(dataset["exposure"], dataset["band"])]
        g_column = 2 * global_number
        z_column = g_column + 1
        q = dataset["q"][indices]
        alpha_column = n_global + dataset["alpha_index"][indices]
        rows.extend(row_number + np.arange(indices.size))
        cols.extend(np.full(indices.size, g_column, dtype=int))
        values.extend(dataset["image"][indices])
        rows.extend(row_number + np.arange(indices.size))
        cols.extend(np.full(indices.size, z_column, dtype=int))
        values.extend(np.ones(indices.size, dtype=float))
        rows.extend(row_number + np.arange(indices.size))
        cols.extend(alpha_column.tolist())
        values.extend(dataset["K"] * f_current[q])
        target.extend(dataset["observed"][indices].tolist())
        row_number += indices.size
    matrix = sparse.coo_matrix(
        (np.asarray(values, dtype=float),
         (np.asarray(rows, dtype=int), np.asarray(cols, dtype=int))),
        shape=(row_number, n_global + alpha_count)).tocsr()
    return matrix, np.asarray(target, dtype=float), global_index


def _solve_sparse_step_a(datasets, f_current, alpha_count):
    """Robust sparse IRLS solve for all ON/OFF globals and physical alphas."""
    matrix, target, global_index = _build_sparse_design(
        datasets, f_current, alpha_count)
    if target.size < matrix.shape[1]:
        raise ValueError("not enough valid fibers for sparse Step-A model")
    weights = np.ones(target.size, dtype=float)
    params = np.zeros(matrix.shape[1], dtype=float)
    for _ in range(6):
        weighted_matrix = matrix.multiply(np.sqrt(weights)[:, None])
        fit = lsqr(weighted_matrix, target * np.sqrt(weights),
                   atol=1.e-10, btol=1.e-10, iter_lim=10000)
        params = fit[0]
        residual = target - matrix @ params
        weights = _huber_weights(residual, robust_scale(residual))
    residuals = {}
    row_number = 0
    for dataset in datasets:
        valid = dataset["valid"]
        indices = np.flatnonzero(valid)
        local_residual = np.full(dataset["observed"].shape, np.nan, dtype=float)
        local_residual[indices] = residual[row_number:row_number + indices.size]
        residuals[id(dataset)] = local_residual
        row_number += indices.size
    globals_out = {}
    for key, number in global_index.items():
        globals_out[key] = {"g": float(params[2 * number]),
                            "z": float(params[2 * number + 1])}
    alpha = np.asarray(params[2 * len(global_index):], dtype=float)
    return {"params": params, "alpha": alpha, "residuals": residuals,
            "globals": globals_out, "rms": robust_rms(residual)}


def _update_common_fq(datasets, step_a, f_current, alpha_count):
    """Step-B weighted common-shape update using both ON and OFF rows."""
    numerator = np.zeros(FIBERS_PER_AMPLIFIER, dtype=float)
    denominator = np.zeros(FIBERS_PER_AMPLIFIER, dtype=float)
    for dataset in datasets:
        valid = dataset["valid"]
        indices = np.flatnonzero(valid)
        if indices.size == 0:
            continue
        global_fit = step_a["globals"][(dataset["exposure"], dataset["band"])]
        raw = ((dataset["observed"] -
                (global_fit["g"] * dataset["image"] + global_fit["z"])) /
               dataset["K"])
        alpha = step_a["alpha"][dataset["alpha_index"]]
        residual = step_a["residuals"][id(dataset)]
        weights = _huber_weights(residual[indices], robust_scale(residual[indices]))
        q = dataset["q"][indices]
        a = alpha[indices]
        r = raw[indices]
        np.add.at(numerator, q, weights * a * r)
        np.add.at(denominator, q, weights * a * a)
    raw_f = np.array(f_current, copy=True)
    good = denominator > np.finfo(float).eps
    raw_f[good] = numerator[good] / denominator[good]
    smoothed = _smooth_fq(raw_f, f_current)
    f_next, alpha_rescaled, baseline, norm = _normalize_fq(
        smoothed, step_a["alpha"])
    return {"f": f_next, "alpha_rescaled": alpha_rescaled,
            "baseline": baseline, "norm": norm}


def _iteration_alpha_fractional_change(current, previous):
    if previous is None:
        return np.nan
    finite = np.isfinite(current) & np.isfinite(previous)
    if not finite.any():
        return np.nan
    denominator = np.maximum(np.maximum(np.abs(current[finite]),
                                        np.abs(previous[finite])), 1.e-12)
    return float(np.median(np.abs(current[finite] - previous[finite]) /
                           denominator))


def _reconstruct_synthetic_image(image, ra, dec, values, valid, sigma):
    """Lightweight Gaussian fiber deposition, equivalent to historical QA."""
    values = np.asarray(values, dtype=float)
    valid = np.asarray(valid, dtype=bool) & np.isfinite(values)
    x, y = image["wcs"].world_to_pixel_values(ra, dec)
    valid &= np.isfinite(x) & np.isfinite(y)
    xi = np.zeros(x.shape, dtype=int)
    yi = np.zeros(y.shape, dtype=int)
    xi[valid] = np.rint(x[valid]).astype(int)
    yi[valid] = np.rint(y[valid]).astype(int)
    valid &= ((xi >= 0) & (xi < image["data"].shape[1]) &
              (yi >= 0) & (yi < image["data"].shape[0]))
    numerator = np.zeros(image["data"].shape, dtype=float)
    denominator = np.zeros(image["data"].shape, dtype=float)
    np.add.at(numerator, (yi[valid], xi[valid]), values[valid])
    np.add.at(denominator, (yi[valid], xi[valid]), 1.0)
    if sigma > 0.0:
        numerator = gaussian_filter(numerator, sigma, mode="nearest")
        denominator = gaussian_filter(denominator, sigma, mode="nearest")
    result = np.full(image["data"].shape, np.nan, dtype=float)
    good = denominator > 1.e-12
    result[good] = numerator[good] / denominator[good]
    return result


def _plot_four_panel(path, panels, title):
    finite_main_parts = [p[np.isfinite(p)] for p in panels[:3]
                         if np.any(np.isfinite(p))]
    finite_main = (np.concatenate(finite_main_parts)
                   if finite_main_parts else np.asarray([], dtype=float))
    finite_residual = panels[3][np.isfinite(panels[3])]
    if finite_main.size:
        vmin, vmax = np.percentile(finite_main, [1, 99])
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
            center = float(np.median(finite_main)); vmin, vmax = center - 1., center + 1.
    else:
        vmin, vmax = -1., 1.
    if finite_residual.size:
        residual_limit = float(np.percentile(np.abs(finite_residual), 99))
        residual_limit = max(residual_limit, np.finfo(float).eps)
    else:
        residual_limit = 1.
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.2))
    labels = ("observed synthetic VIRUS", "fitted external gI+z",
              "corrected synthetic VIRUS", "corrected - external")
    for axis, panel, label in zip(axes, panels, labels):
        if label == labels[-1]:
            im = axis.imshow(panel, origin="lower", cmap="coolwarm",
                             vmin=-residual_limit, vmax=residual_limit)
        else:
            im = axis.imshow(panel, origin="lower", cmap="viridis",
                             vmin=vmin, vmax=vmax)
        axis.set_title(label)
        axis.set_xticks([]); axis.set_yticks([])
        fig.colorbar(im, ax=axis, fraction=0.046, pad=0.04)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def _plot_common_fq(output_dir, f_initial, f_converged):
    q = np.arange(FIBERS_PER_AMPLIFIER)
    delta = f_converged - f_initial
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True,
                             gridspec_kw={"height_ratios": (3, 1)})
    axes[0].plot(q, f_initial, label="initial template", lw=1.4)
    axes[0].plot(q, f_converged, label="converged external f(q)", lw=1.4)
    axes[0].axhline(0.0, color="k", lw=0.7)
    axes[0].axvline(40.0, color="tab:red", ls=":", lw=0.8)
    axes[0].set_ylabel("f(q)"); axes[0].grid(alpha=0.2); axes[0].legend()
    axes[1].plot(q, delta, color="tab:purple", lw=1.1)
    axes[1].axhline(0.0, color="k", lw=0.7)
    axes[1].set_xlabel("folded readout distance q")
    axes[1].set_ylabel("converged - initial"); axes[1].grid(alpha=0.2)
    initial_peak = int(np.argmax(f_initial)); final_peak = int(np.argmax(f_converged))
    rms = robust_rms(delta)
    annotation = ("initial peak q=%d, f=%+.4g\nconverged peak q=%d, f=%+.4g\n"
                  "RMS difference=%+.4g\nconverged medians: q<20=%+.4g, q≥40=%+.4g" %
                  (initial_peak, f_initial[initial_peak], final_peak,
                   f_converged[final_peak], rms, np.median(f_converged[:20]),
                   np.median(f_converged[40:])))
    axes[0].text(0.98, 0.96, annotation, transform=axes[0].transAxes,
                 ha="right", va="top", fontsize=8)
    fig.suptitle("Common detector/readout shape: initialization and external refinement")
    fig.tight_layout()
    fig.savefig(output_dir / "external_common_fq.png", dpi=170)
    plt.close(fig)


def _plot_alpha_by_ifuslot(output_dir, alpha_rows):
    exposures = sorted({int(row["exposure"]) for row in alpha_rows})
    fig, axes = plt.subplots(len(exposures), 1, figsize=(11, 3.4 * len(exposures)),
                             squeeze=False, sharex=True)
    colors = dict(zip(AMPLIFIERS, ("tab:blue", "tab:orange", "tab:green", "tab:red")))
    slots = sorted({int(row["IFUSLOT"]) for row in alpha_rows})
    offsets = dict(zip(AMPLIFIERS, (-0.27, -0.09, 0.09, 0.27)))
    for axis, exposure in zip(axes[:, 0], exposures):
        subset = [row for row in alpha_rows if int(row["exposure"]) == exposure]
        for amp in AMPLIFIERS:
            rows = [row for row in subset if row["AMP"] == amp]
            x = [int(row["IFUSLOT"]) + offsets[amp] for row in rows]
            y = [float(row["alpha_e_per_A"]) for row in rows]
            axis.scatter(x, y, color=colors[amp], label=amp, s=18)
        axis.set_ylabel("alpha [e-/A]"); axis.set_title("exposure %d" % exposure)
        axis.grid(alpha=0.2)
    axes[-1, 0].set_xlabel("IFUSLOT (small offsets identify AMP)")
    axes[-1, 0].set_xticks(slots); axes[-1, 0].legend(ncol=4, fontsize=8)
    fig.suptitle("External common-f(q) physical-amplifier amplitudes")
    fig.tight_layout()
    fig.savefig(output_dir / "external_alpha_by_ifuslot.png", dpi=170)
    plt.close(fig)


def _plot_alpha_exposure_comparison(output_dir, alpha_rows):
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.2), sharey=True)
    colors = plt.cm.tab20(np.linspace(0, 1, 20))
    for index, amp in enumerate(AMPLIFIERS):
        keys = sorted({(row["SPECID"], row["IFUSLOT"], row["IFUID"])
                       for row in alpha_rows if row["AMP"] == amp})
        for color, key in zip(colors, keys):
            rows = [row for row in alpha_rows if row["AMP"] == amp and
                    (row["SPECID"], row["IFUSLOT"], row["IFUID"]) == key]
            rows.sort(key=lambda row: int(row["exposure"]))
            axes[index].plot([int(row["exposure"]) for row in rows],
                             [float(row["alpha_e_per_A"]) for row in rows],
                             "o-", color=color, alpha=0.7, ms=3)
        axes[index].set_title(amp); axes[index].set_xlabel("exposure")
        axes[index].set_xticks([1, 2, 3]); axes[index].grid(alpha=0.2)
    axes[0].set_ylabel("alpha [e-/A]")
    fig.suptitle("Same physical amplifiers across exposures")
    fig.tight_layout()
    fig.savefig(output_dir / "external_alpha_exposure_comparison.png", dpi=170)
    plt.close(fig)


def _plot_residual_distribution(output_dir, residual_values):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), sharey=True)
    colors = {"ON": "tab:blue", "OFF": "tab:orange"}
    for axis, state in zip(axes, ("before", "after")):
        for band in ("ON", "OFF"):
            values = np.asarray(residual_values[(state, band)], dtype=float)
            values = values[np.isfinite(values)]
            if values.size:
                axis.hist(values, bins=80, histtype="step", color=colors[band],
                          density=True, label=band)
                summary = _residual_summary(values)
                axis.text(0.03, 0.96 - 0.13 * (band == "OFF"),
                          "%s med=%+.3g\np16/p84=%+.3g/%+.3g\nRMS=%+.3g e-/A" %
                          (band, summary["median"], summary["p16"],
                           summary["p84"], summary["robust_rms"]),
                          transform=axis.transAxes, va="top", fontsize=8,
                          color=colors[band])
        axis.axvline(0.0, color="k", lw=0.7); axis.set_title(state)
        axis.set_xlabel("residual [e-/A]"); axis.grid(alpha=0.2)
    axes[0].set_ylabel("density"); axes[0].legend()
    fig.suptitle("Joint external-model residual distribution")
    fig.tight_layout()
    fig.savefig(output_dir / "external_joint_residual_distribution.png", dpi=170)
    plt.close(fig)


def _write_common_csvs(output_dir, f_initial, f_converged, alpha_rows,
                       global_rows):
    with (output_dir / "external_common_fq_converged.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["q", "f_initial",
                                                    "f_converged", "delta_f"])
        writer.writeheader()
        for q in range(FIBERS_PER_AMPLIFIER):
            writer.writerow({"q": q, "f_initial": f_initial[q],
                             "f_converged": f_converged[q],
                             "delta_f": f_converged[q] - f_initial[q]})

    alpha_fields = ["h5", "exposure", "SPECID", "IFUSLOT", "IFUID", "AMP",
                    "alpha_e_per_A", "n_on_fibers", "n_off_fibers",
                    "n_total_fibers", "median_residual_on",
                    "median_residual_off", "robust_rms_on", "robust_rms_off",
                    "uncertainty_e_per_A", "fit_flag"]
    with (output_dir / "external_physical_amp_alpha.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=alpha_fields)
        writer.writeheader(); writer.writerows(alpha_rows)

    global_fields = ["h5", "exposure", "band", "g_e_b", "z_e_b", "K_e_b",
                     "robust_rms_before", "robust_rms_after",
                     "robust_rms_before_e_per_A", "robust_rms_after_e_per_A",
                     "residual_median", "residual_p16", "residual_p84",
                     "residual_median_e_per_A", "residual_p16_e_per_A",
                     "residual_p84_e_per_A"]
    with (output_dir / "external_joint_global_parameters.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=global_fields)
        writer.writeheader(); writer.writerows(global_rows)


def run_common_fq_external_fit(args, output_dir):
    """Run the requested shared-f(q), shared-ON/OFF-alpha experiment."""
    f_initial = load_common_fq_template(args.common_fq_template)
    bands = ("ON", "OFF")
    filters = {"ON": read_filter(args.on_filter),
               "OFF": read_filter(args.off_filter)}
    images = {"ON": load_image(args.on_image),
              "OFF": load_image(args.off_image)}

    datasets = []
    all_groups = []
    ra = dec = row_q = row_alpha = None
    with tables.open_file(args.h5, mode="r") as h5:
        if not {"Info", "Fibers", "Survey"}.issubset(h5.root._v_children):
            raise ValueError("H5 needs Info, Fibers, and Survey tables")
        info, fibers, survey = h5.root.Info, h5.root.Fibers, h5.root.Survey
        nrows = int(info.nrows)
        if int(fibers.nrows) != nrows:
            raise ValueError("Info/Fibers row mismatch")
        if fibers.coldtypes["spectrum"].shape != (DEF_WAVE.size,):
            raise ValueError("Fibers.spectrum does not use the 1036-bin grid")
        groups, labels = build_groups(info)
        all_groups = groups
        ra = np.asarray(info.cols.ra[:], dtype=float)
        dec = np.asarray(info.cols.dec[:], dtype=float)
        ifuslot = np.asarray(info.cols.ifuslot[:])
        amp = np.asarray([text(v) for v in info.cols.amp[:]])
        bad = masked_rows(args.h5, ifuslot, amp)
        row_q = np.full(nrows, -1, dtype=int)
        row_alpha = np.full(nrows, -1, dtype=int)
        identity_to_alpha = {}
        for alpha_index, group in enumerate(groups):
            if group["identity"] in identity_to_alpha:
                raise ValueError("duplicate physical amplifier identity %s" %
                                 (group["identity"],))
            identity_to_alpha[group["identity"]] = alpha_index
            j = group["j"]
            q = j if group["amp"] in ("LL", "RU") else 111 - j
            row_q[group["indices"]] = q
            row_alpha[group["indices"]] = alpha_index
        if np.any(row_q < 0) or np.any(row_alpha < 0):
            raise ValueError("physical amplifier/q bookkeeping did not cover all rows")
        survey_by_exp = {}
        for row in survey:
            exposure = int(row["exp"])
            if exposure in survey_by_exp:
                raise ValueError("Survey has duplicate exposure %d" % exposure)
            survey_by_exp[exposure] = row
        spectra_h5 = np.asarray(fibers.cols.spectrum[:], dtype=float)
        errors_h5 = np.asarray(fibers.cols.error[:], dtype=float)

        exposures = [args.exposure] if args.exposure else list(range(1, 4))
        for exposure in exposures:
            if exposure not in survey_by_exp:
                raise ValueError("Survey has no row for exposure %d" % exposure)
            survey_row = survey_by_exp[exposure]
            offset = float(survey_row["offset"])
            if np.isfinite(offset) and offset != 0.0:
                working_spectrum = spectra_h5 / offset
                working_error = errors_h5 / offset
            else:
                working_spectrum = spectra_h5.copy()
                working_error = errors_h5.copy()
                print("WARNING exposure %d: invalid Survey.offset=%s; using "
                      "Fibers units unchanged" % (exposure, offset))
            exposure_rows = labels == exposure
            for band in bands:
                response = filters[band]
                observed, observed_error = synthetic_mean(
                    working_spectrum, working_error, response)
                eff_ra, eff_dec = adr_positions(ra, dec, survey_row, response)
                K_work = raw_work_basis(survey_row)
                K_band = weighted_scalar(K_work, response)
                if not np.isfinite(K_band) or K_band == 0.0:
                    raise ValueError("invalid K_band for %s exposure %d" %
                                     (band, exposure))
                image_value, _gradient, image_valid = sample_image(
                    images[band], eff_ra, eff_dec, args.external_image_sigma_pix)
                valid = (exposure_rows & ~bad & np.isfinite(observed) &
                         np.isfinite(image_value))
                datasets.append({
                    "exposure": exposure, "band": band, "observed": observed,
                    "observed_error": observed_error, "image": image_value,
                    "ra": eff_ra, "dec": eff_dec, "K": float(K_band),
                    "q": row_q.copy(), "alpha_index": row_alpha.copy(),
                    "valid": valid,
                })

    if not datasets or not any(np.any(d["valid"]) for d in datasets):
        raise ValueError("common f(q) fit has no valid fibers")
    f_current = np.array(f_initial, copy=True)
    previous_alpha = None
    history = []
    for iteration in range(1, args.max_iterations + 1):
        step_a = _solve_sparse_step_a(datasets, f_current, len(all_groups))
        update = _update_common_fq(datasets, step_a, f_current, len(all_groups))
        max_delta = float(np.max(np.abs(update["f"] - f_current)))
        alpha_fractional = _iteration_alpha_fractional_change(
            update["alpha_rescaled"], previous_alpha)
        f_current = update["f"]
        previous_alpha = update["alpha_rescaled"]
        history.append((iteration, step_a["rms"], max_delta,
                        alpha_fractional, f_current.copy()))
        print("iteration %d: robust RMS=%g, max delta f=%g, median fractional "
              "alpha change=%g, median f(q<20)=%g, median f(q>=40)=%g, "
              "peak q=%d, peak f=%g" %
              (iteration, step_a["rms"], max_delta, alpha_fractional,
               np.median(f_current[:20]), np.median(f_current[40:]),
               int(np.argmax(f_current)), np.max(f_current)))
        if (iteration > 1 and max_delta < 1.e-3 and
                np.isfinite(alpha_fractional) and alpha_fractional < 1.e-3):
            break

    # Refit at the final normalized shape so all saved products use one
    # consistent parameterization and one shared ON/OFF alpha per identity.
    final_step = _solve_sparse_step_a(datasets, f_current, len(all_groups))
    for dataset in datasets:
        valid = dataset["valid"]
        global_fit = final_step["globals"][(dataset["exposure"], dataset["band"])]
        model = (global_fit["g"] * dataset["image"] + global_fit["z"] +
                 dataset["K"] * final_step["alpha"][dataset["alpha_index"]] *
                 f_current[dataset["q"]])
        residual = np.full(dataset["observed"].shape, np.nan, dtype=float)
        residual[valid] = dataset["observed"][valid] - model[valid]
        dataset["model"] = model
        dataset["residual"] = residual

    _plot_common_fq(output_dir, f_initial, f_current)

    alpha_rows = []
    for alpha_index, group in enumerate(all_groups):
        band_stats = {}
        for band in bands:
            dataset = next(d for d in datasets if d["exposure"] == group["exposure"] and
                           d["band"] == band)
            selected = dataset["valid"] & (dataset["alpha_index"] == alpha_index)
            residual_e = dataset["residual"][selected] / dataset["K"]
            band_stats[band] = (int(selected.sum()), _residual_summary(residual_e))
        n_on, stats_on = band_stats["ON"]
        n_off, stats_off = band_stats["OFF"]
        total = n_on + n_off
        uncertainty = (np.nanmedian([stats_on["robust_rms"], stats_off["robust_rms"]]) /
                        np.sqrt(total) if total > 0 else np.nan)
        alpha_rows.append({
            "h5": Path(args.h5).name, "exposure": group["exposure"],
            "SPECID": group["specid"], "IFUSLOT": group["ifuslot"],
            "IFUID": group["ifuid"], "AMP": group["amp"],
            "alpha_e_per_A": final_step["alpha"][alpha_index],
            "n_on_fibers": n_on, "n_off_fibers": n_off,
            "n_total_fibers": total,
            "median_residual_on": stats_on["median"],
            "median_residual_off": stats_off["median"],
            "robust_rms_on": stats_on["robust_rms"],
            "robust_rms_off": stats_off["robust_rms"],
            "uncertainty_e_per_A": uncertainty,
            "fit_flag": "ok" if total >= 6 else "low_n",
        })
    # Write the global CSV after calculating its per exposure/band statistics.
    global_rows = []
    residual_values = {(state, band): [] for state in ("before", "after")
                       for band in bands}
    for dataset in datasets:
        global_fit = final_step["globals"][(dataset["exposure"], dataset["band"])]
        valid = dataset["valid"]
        before = dataset["observed"] - (global_fit["g"] * dataset["image"] +
                                         global_fit["z"])
        after = dataset["residual"]
        before_e = before[valid] / dataset["K"]
        after_e = after[valid] / dataset["K"]
        residual_values[("before", dataset["band"])].extend(before_e.tolist())
        residual_values[("after", dataset["band"])].extend(after_e.tolist())
        before_summary = _residual_summary(before[valid])
        after_summary = _residual_summary(after[valid])
        before_e_summary = _residual_summary(before_e)
        after_e_summary = _residual_summary(after_e)
        global_rows.append({
            "h5": Path(args.h5).name, "exposure": dataset["exposure"],
            "band": dataset["band"], "g_e_b": global_fit["g"],
            "z_e_b": global_fit["z"], "K_e_b": dataset["K"],
            "robust_rms_before": before_summary["robust_rms"],
            "robust_rms_after": after_summary["robust_rms"],
            "robust_rms_before_e_per_A": before_e_summary["robust_rms"],
            "robust_rms_after_e_per_A": after_e_summary["robust_rms"],
            "residual_median": after_summary["median"],
            "residual_p16": after_summary["p16"],
            "residual_p84": after_summary["p84"],
            "residual_median_e_per_A": after_e_summary["median"],
            "residual_p16_e_per_A": after_e_summary["p16"],
            "residual_p84_e_per_A": after_e_summary["p84"],
        })
    _write_common_csvs(output_dir, f_initial, f_current, alpha_rows, global_rows)
    _plot_alpha_by_ifuslot(output_dir, alpha_rows)
    _plot_alpha_exposure_comparison(output_dir, alpha_rows)
    _plot_residual_distribution(output_dir, residual_values)

    rendered = {}
    for dataset in datasets:
        alpha_rows_for_fibers = final_step["alpha"][dataset["alpha_index"]]
        detector = dataset["K"] * alpha_rows_for_fibers * f_current[dataset["q"]]
        corrected = dataset["observed"] - detector
        panels = [
            _reconstruct_synthetic_image(images[dataset["band"]], dataset["ra"],
                                          dataset["dec"], dataset["observed"],
                                          dataset["valid"], args.reconstruction_sigma_pix),
            _reconstruct_synthetic_image(images[dataset["band"]], dataset["ra"],
                                          dataset["dec"], dataset["model"] - detector,
                                          dataset["valid"], args.reconstruction_sigma_pix),
            _reconstruct_synthetic_image(images[dataset["band"]], dataset["ra"],
                                          dataset["dec"], corrected,
                                          dataset["valid"], args.reconstruction_sigma_pix),
            _reconstruct_synthetic_image(images[dataset["band"]], dataset["ra"],
                                          dataset["dec"], dataset["residual"],
                                          dataset["valid"], args.reconstruction_sigma_pix),
        ]
        rendered[(dataset["exposure"], dataset["band"])] = panels
        _plot_four_panel(output_dir / ("external_synthetic_image_exp%d_%s.png" %
                                       (dataset["exposure"], dataset["band"])),
                         panels, "external synthetic image exposure %d %s" %
                         (dataset["exposure"], dataset["band"]))
    if set(exposures for exposures, _band in rendered) == {1, 2, 3}:
        for band in bands:
            combined = [np.nanmedian(np.asarray([rendered[(exp, band)][panel]
                                                  for exp in (1, 2, 3)]), axis=0)
                        for panel in range(4)]
            _plot_four_panel(output_dir / ("external_synthetic_image_combined_%s.png" % band),
                             combined, "external synthetic image combined %s" % band)

    final_finite_delta = robust_rms(f_current - f_initial)
    print("common f(q) fit complete: iterations=%d, final RMS shape difference=%g, "
          "median q<20=%g, median q>=40=%g, peak q=%d, peak f=%g" %
          (len(history), final_finite_delta, np.median(f_current[:20]),
           np.median(f_current[40:]), int(np.argmax(f_current)), np.max(f_current)))


# ---------------------------------------------------------------------------
# Read-only inspection of the already-converged physical-amplifier solution.


def _read_dict_rows(path):
    with Path(path).open(newline="") as stream:
        return list(csv.DictReader(stream))


def _load_converged_solution(solution_dir):
    """Load saved parameters without fitting or changing any value."""
    solution_dir = Path(solution_dir)
    fq_rows = _read_dict_rows(solution_dir / "external_common_fq_converged.csv")
    if len(fq_rows) != FIBERS_PER_AMPLIFIER:
        raise ValueError("converged f(q) CSV must contain 112 rows")
    f_converged = np.full(FIBERS_PER_AMPLIFIER, np.nan, dtype=float)
    for row in fq_rows:
        q = int(row["q"])
        if q < 0 or q >= FIBERS_PER_AMPLIFIER:
            raise ValueError("converged f(q) contains q outside 0..111")
        f_converged[q] = float(row["f_converged"])
    if not np.all(np.isfinite(f_converged)):
        raise ValueError("converged f(q) contains missing/nonfinite values")

    global_rows = {}
    for row in _read_dict_rows(solution_dir / "external_joint_global_parameters.csv"):
        key = (int(row["exposure"]), row["band"])
        global_rows[key] = {
            "g": float(row["g_e_b"]), "z": float(row["z_e_b"]),
            "K": float(row["K_e_b"]),
        }
    alpha_rows = []
    for row in _read_dict_rows(solution_dir / "external_physical_amp_alpha.csv"):
        parsed = dict(row)
        parsed["exposure"] = int(row["exposure"])
        for field in ("SPECID", "IFUSLOT", "IFUID"):
            parsed[field] = int(row[field])
        for field in ("alpha_e_per_A", "robust_rms_on", "robust_rms_off"):
            parsed[field] = float(row[field])
        parsed["identity"] = (parsed["exposure"], parsed["SPECID"],
                               parsed["IFUSLOT"], parsed["IFUID"], parsed["AMP"])
        alpha_rows.append(parsed)
    if not alpha_rows:
        raise ValueError("converged solution has no physical amplifier rows")
    return f_converged, global_rows, alpha_rows


def _parse_inspection_selector(value):
    fields = [field.strip() for field in value.split(",")]
    if len(fields) == 3:
        exposure, ifuslot, amp = fields
        return (int(exposure), int(ifuslot), amp)
    if len(fields) == 5:
        exposure, specid, ifuslot, ifuid, amp = fields
        return (int(exposure), int(specid), int(ifuslot), int(ifuid), amp)
    raise ValueError("--inspect-amplifier expects exp,IFUSLOT,AMP or "
                     "exp,SPECID,IFUSLOT,IFUID,AMP")


def _select_inspection_rows(alpha_rows, selectors):
    exposure_one = [row for row in alpha_rows if row["exposure"] == 1]
    finite = [row for row in exposure_one
              if np.isfinite(row["alpha_e_per_A"]) and
              np.isfinite(row["robust_rms_on"]) and
              np.isfinite(row["robust_rms_off"])]
    selected = []
    selected_ids = set()

    def add_rows(rows, count):
        added = 0
        for row in rows:
            if row["identity"] in selected_ids:
                continue
            selected.append(row)
            selected_ids.add(row["identity"])
            added += 1
            if added >= count:
                break

    # Keep category selection deterministic and avoid duplicate rows between
    # categories.  These are samples for visual inspection, not QA flags.
    add_rows(sorted([r for r in finite if 0.2 <= r["alpha_e_per_A"] <= 0.6],
                    key=lambda r: abs(r["alpha_e_per_A"] - 0.4)), 3)
    add_rows(sorted(finite, key=lambda r: abs(r["alpha_e_per_A"])), 3)
    add_rows(sorted([r for r in finite if 1.0 <= abs(r["alpha_e_per_A"]) <= 2.0],
                    key=lambda r: abs(abs(r["alpha_e_per_A"]) - 1.5)), 3)
    add_rows(sorted([r for r in finite if r["alpha_e_per_A"] > 2.0],
                    key=lambda r: r["alpha_e_per_A"], reverse=True), 3)
    add_rows(sorted([r for r in finite if r["alpha_e_per_A"] < 0.0],
                    key=lambda r: r["alpha_e_per_A"]), 3)
    add_rows(sorted(finite, key=lambda r: max(r["robust_rms_on"],
                                               r["robust_rms_off"]),
                    reverse=True), 3)

    if selectors:
        selected = []
        for selector in selectors:
            if len(selector) == 3:
                matches = [row for row in alpha_rows
                           if (row["exposure"], row["IFUSLOT"], row["AMP"]) == selector]
            else:
                matches = [row for row in alpha_rows if row["identity"] == selector]
            if not matches:
                raise ValueError("no converged alpha matches selector %s" % (selector,))
            if len(matches) > 1:
                raise ValueError("selector %s matches multiple physical amplifiers; "
                                 "use the full identity" % (selector,))
            if matches[0]["identity"] not in {row["identity"] for row in selected}:
                selected.append(matches[0])
    if not selected:
        raise ValueError("no representative exposure-1 amplifiers were available")
    return selected


def _print_inspection_selection(rows):
    print("selected physical amplifiers for component inspection:")
    for row in rows:
        rms = max(row["robust_rms_on"], row["robust_rms_off"])
        print("  exposure=%d SPECID=%d IFUSLOT=%03d IFUID=%d AMP=%s "
              "alpha=%+.6g RMS=%+.6g e-/A (ON=%+.6g OFF=%+.6g)" %
              (row["exposure"], row["SPECID"], row["IFUSLOT"], row["IFUID"],
               row["AMP"], row["alpha_e_per_A"], rms,
               row["robust_rms_on"], row["robust_rms_off"]))


def _load_inspection_measurements(args, selected_rows, global_rows):
    """Build Y, M, and D from existing data and saved globals only."""
    bands = ("ON", "OFF")
    filters = {"ON": read_filter(args.on_filter),
               "OFF": read_filter(args.off_filter)}
    images = {"ON": load_image(args.on_image),
              "OFF": load_image(args.off_image)}
    wanted = {row["identity"] for row in selected_rows}
    wanted_exposures = sorted({row["exposure"] for row in selected_rows})
    records = {}
    with tables.open_file(args.h5, mode="r") as h5:
        if not {"Info", "Fibers", "Survey"}.issubset(h5.root._v_children):
            raise ValueError("H5 needs Info, Fibers, and Survey tables")
        info, fibers, survey = h5.root.Info, h5.root.Fibers, h5.root.Survey
        nrows = int(info.nrows)
        if int(fibers.nrows) != nrows:
            raise ValueError("Info/Fibers row mismatch")
        groups, labels = build_groups(info)
        ra = np.asarray(info.cols.ra[:], dtype=float)
        dec = np.asarray(info.cols.dec[:], dtype=float)
        ifuslot = np.asarray(info.cols.ifuslot[:])
        amp = np.asarray([text(v) for v in info.cols.amp[:]])
        bad = masked_rows(args.h5, ifuslot, amp)
        survey_by_exp = {int(row["exp"]): row for row in survey}
        spectra_h5 = np.asarray(fibers.cols.spectrum[:], dtype=float)
        errors_h5 = np.asarray(fibers.cols.error[:], dtype=float)
        for exposure in wanted_exposures:
            if exposure not in survey_by_exp:
                raise ValueError("Survey has no row for exposure %d" % exposure)
            survey_row = survey_by_exp[exposure]
            offset = float(survey_row["offset"])
            if np.isfinite(offset) and offset != 0.0:
                working_spectrum = spectra_h5 / offset
                working_error = errors_h5 / offset
            else:
                working_spectrum = spectra_h5.copy()
                working_error = errors_h5.copy()
            exposure_rows = labels == exposure
            for band in bands:
                solution = global_rows.get((exposure, band))
                if solution is None:
                    raise ValueError("missing saved global solution for exposure %d %s" %
                                     (exposure, band))
                response = filters[band]
                V, _V_error = synthetic_mean(working_spectrum, working_error, response)
                eff_ra, eff_dec = adr_positions(ra, dec, survey_row, response)
                I, _gradient, image_valid = sample_image(
                    images[band], eff_ra, eff_dec, args.external_image_sigma_pix)
                K = solution["K"]
                if not np.isfinite(K) or K == 0.0:
                    raise ValueError("saved K is invalid for exposure %d %s" %
                                     (exposure, band))
                valid = (exposure_rows & ~bad & np.isfinite(V) &
                         np.isfinite(I) & image_valid)
                Y = V / K
                M = (solution["g"] * I + solution["z"]) / K
                D = Y - M
                for group in groups:
                    if group["exposure"] != exposure or group["identity"] not in wanted:
                        continue
                    alpha_row = next(row for row in selected_rows
                                     if row["identity"] == group["identity"])
                    q = group["j"] if group["amp"] in ("LL", "RU") else 111 - group["j"]
                    y_q = np.full(FIBERS_PER_AMPLIFIER, np.nan, dtype=float)
                    m_q = np.full(FIBERS_PER_AMPLIFIER, np.nan, dtype=float)
                    d_q = np.full(FIBERS_PER_AMPLIFIER, np.nan, dtype=float)
                    group_indices = group["indices"]
                    good = valid[group_indices]
                    y_q[q[good]] = Y[group_indices[good]]
                    m_q[q[good]] = M[group_indices[good]]
                    d_q[q[good]] = D[group_indices[good]]
                    key = group["identity"]
                    if key not in records:
                        records[key] = {
                            "identity": key, "alpha": alpha_row["alpha_e_per_A"],
                            "Y": {}, "M": {}, "D": {}, "K": {},
                        }
                    records[key]["Y"][band] = y_q
                    records[key]["M"][band] = m_q
                    records[key]["D"][band] = d_q
                    records[key]["K"][band] = K
    missing = [row["identity"] for row in selected_rows if row["identity"] not in records]
    if missing:
        raise ValueError("selected physical amplifiers were not found in H5: %s" % missing)
    return records, images


def _inspection_ylim(record, f_converged):
    alpha = record["alpha"]
    arrays = [record["D"][band] for band in ("ON", "OFF")]
    arrays += [alpha * f_converged]
    arrays += [record["D"]["ON"] - record["D"]["OFF"],
               record["D"]["ON"] - alpha * f_converged,
               record["D"]["OFF"] - alpha * f_converged]
    parts = [a[np.isfinite(a)] for a in arrays if np.any(np.isfinite(a))]
    values = np.concatenate(parts) if parts else np.asarray([], dtype=float)
    if values.size == 0:
        return -1.0, 1.0
    lo, hi = np.percentile(values, [1.0, 99.0])
    span = max(hi - lo, 1.e-6)
    return float(lo - 0.08 * span), float(hi + 0.08 * span)


def _mark_q(axis):
    axis.axvline(20, color="0.5", ls=":", lw=0.7)
    axis.axvline(40, color="0.5", ls=":", lw=0.7)
    axis.set_xlim(-1, 112)
    axis.grid(alpha=0.2)


def _plot_one_amplifier_component(record, f_converged, path):
    q = np.arange(FIBERS_PER_AMPLIFIER)
    alpha = record["alpha"]
    model = alpha * f_converged
    d_on, d_off = record["D"]["ON"], record["D"]["OFF"]
    after_on, after_off = d_on - model, d_off - model
    difference = d_on - d_off
    fig, axes = plt.subplots(4, 1, figsize=(12, 13), sharex=True)

    axes[0].plot(q, record["Y"]["ON"], ".-", ms=3, lw=0.55,
                 color="tab:blue", label="ON Y=V/K")
    axes[0].plot(q, record["M"]["ON"], "--", lw=1.0,
                 color="tab:blue", label="ON M=(gI+z)/K")
    axes[0].plot(q, record["Y"]["OFF"], ".-", ms=3, lw=0.55,
                 color="tab:orange", label="OFF Y=V/K")
    axes[0].plot(q, record["M"]["OFF"], "--", lw=1.0,
                 color="tab:orange", label="OFF M=(gI+z)/K")
    axes[0].set_ylabel("absolute [e-/A]"); axes[0].set_title("Raw-equivalent values")
    axes[0].legend(ncol=2, fontsize=8); _mark_q(axes[0])

    axes[1].plot(q, d_on, ".-", ms=3, lw=0.55, color="tab:blue", label="D_ON")
    axes[1].plot(q, d_off, ".-", ms=3, lw=0.55, color="tab:orange", label="D_OFF")
    axes[1].plot(q, model, "k-", lw=1.5, label="alpha*f(q)")
    axes[1].axhline(0.0, color="k", lw=0.7); axes[1].set_ylabel("D [e-/A]")
    axes[1].set_title("External-image residual and detector model (PRIMARY)")
    axes[1].legend(fontsize=8); _mark_q(axes[1])
    summary = ("alpha=%+.5g e-/A\nD_ON med=%+.5g RMS=%+.5g\n"
               "D_OFF med=%+.5g RMS=%+.5g" %
               (alpha, np.nanmedian(d_on), robust_rms(d_on),
                np.nanmedian(d_off), robust_rms(d_off)))
    axes[1].text(0.99, 0.96, summary, transform=axes[1].transAxes,
                 ha="right", va="top", fontsize=8)

    axes[2].plot(q, difference, ".-", ms=3, lw=0.55, color="tab:purple",
                 label="D_ON-D_OFF")
    axes[2].axhline(0.0, color="k", lw=0.7); axes[2].set_ylabel("difference [e-/A]")
    axes[2].set_title("ON/OFF disagreement (no new fit)"); axes[2].legend(fontsize=8)
    _mark_q(axes[2])

    axes[3].plot(q, after_on, ".-", ms=3, lw=0.55, color="tab:blue",
                 label="D_ON-alpha*f(q)")
    axes[3].plot(q, after_off, ".-", ms=3, lw=0.55, color="tab:orange",
                 label="D_OFF-alpha*f(q)")
    axes[3].axhline(0.0, color="k", lw=0.7); axes[3].set_ylabel("after [e-/A]")
    axes[3].set_xlabel("folded readout distance q")
    axes[3].set_title("Residual after saved alpha*f(q)"); axes[3].legend(fontsize=8)
    _mark_q(axes[3])

    lower, upper = _inspection_ylim(record, f_converged)
    for axis in axes[1:]:
        axis.set_ylim(lower, upper)
    identity = record["identity"]
    fig.suptitle("Physical amplifier: exp %d SPECID %d IFUSLOT %03d IFUID %d AMP %s\n"
                 "All quantities Raw-equivalent [e-/A]; saved converged solution only" %
                 identity)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=170)
    plt.close(fig)


def _plot_component_gallery(records, f_converged, path):
    rows = list(records.values())
    ncols = 2
    nrows = int(np.ceil(len(rows) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 3.4 * nrows),
                             squeeze=False, sharex=True, sharey=True)
    all_values = []
    for record in rows:
        all_values.extend([a[np.isfinite(a)] for a in
                           (record["D"]["ON"], record["D"]["OFF"],
                            record["alpha"] * f_converged)
                           if np.any(np.isfinite(a))])
    values = np.concatenate(all_values) if all_values else np.asarray([], dtype=float)
    if values.size:
        lo, hi = np.percentile(values, [1, 99]); span = max(hi - lo, 1.e-6)
        ylim = (lo - 0.08 * span, hi + 0.08 * span)
    else:
        ylim = (-1., 1.)
    q = np.arange(FIBERS_PER_AMPLIFIER)
    for index, record in enumerate(rows):
        axis = axes[index // ncols, index % ncols]
        axis.plot(q, record["D"]["ON"], ".-", ms=2, lw=0.45,
                  color="tab:blue", label="D_ON")
        axis.plot(q, record["D"]["OFF"], ".-", ms=2, lw=0.45,
                  color="tab:orange", label="D_OFF")
        axis.plot(q, record["alpha"] * f_converged, "k-", lw=1.1,
                  label="alpha*f(q)")
        identity = record["identity"]
        axis.set_title("exp%d IFU%03d %s alpha=%+.3g" %
                       (identity[0], identity[2], identity[4], record["alpha"]), fontsize=9)
        axis.axhline(0.0, color="k", lw=0.6); axis.set_ylim(*ylim); _mark_q(axis)
    for index in range(len(rows), nrows * ncols):
        axes[index // ncols, index % ncols].set_visible(False)
    axes[-1, 0].set_xlabel("q"); axes[-1, 1].set_xlabel("q")
    axes[0, 0].set_ylabel("D [e-/A]"); axes[min(nrows - 1, 1), 0].set_ylabel("D [e-/A]")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Selected physical-amplifier component gallery (saved values only)")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=170)
    plt.close(fig)


def run_component_inspection(args):
    """Produce component plots from saved parameters, with no new model fit."""
    f_converged, global_rows, alpha_rows = _load_converged_solution(
        args.inspect_solution_dir)
    selectors = [_parse_inspection_selector(value)
                 for value in (args.inspect_amplifier or [])]
    selected_rows = _select_inspection_rows(alpha_rows, selectors)
    _print_inspection_selection(selected_rows)
    records, _images = _load_inspection_measurements(
        args, selected_rows, global_rows)
    output_dir = Path(args.inspect_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for row in selected_rows:
        identity = row["identity"]
        record = records[identity]
        filename = "exp%d_ifuslot%03d_%s_components.png" % (
            identity[0], identity[2], identity[4])
        _plot_one_amplifier_component(record, f_converged, output_dir / filename)
    _plot_component_gallery(records, f_converged,
                            output_dir / "amplifier_component_gallery.png")
    print("component inspection complete: %d amplifier figures plus gallery in %s" %
          (len(selected_rows), output_dir))


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--h5", required=True)
    parser.add_argument("--on-image", required=True)
    parser.add_argument("--off-image", required=True)
    parser.add_argument("--on-filter", required=True)
    parser.add_argument("--off-filter", required=True)
    parser.add_argument("--output-dir", default="external_q_test")
    parser.add_argument("--exposure", type=int, choices=(1, 2, 3))
    parser.add_argument("--image-sigma-pix", nargs="+", type=float,
                        default=[0.0, 4.5],
                        help="one or more image Gaussian sigmas; default: 0 4.5")
    parser.add_argument("--template-csv")
    parser.add_argument("--common-fq-template",
                        help="q,f CSV used to initialize the shared external f(q) fit")
    parser.add_argument("--external-image-sigma-pix", type=float, default=4.5,
                        help="image smoothing used for the common-f(q) external fit")
    parser.add_argument("--reconstruction-sigma-pix", type=float, default=1.5,
                        help="Gaussian width for diagnostic synthetic-image reconstruction")
    parser.add_argument("--max-iterations", type=int, default=10,
                        help="maximum common-f(q) alternating iterations")
    parser.add_argument("--inspect-solution-dir",
                        help="existing common-f(q) output directory to inspect without refitting")
    parser.add_argument("--inspect-amplifier", action="append",
                        help="exp,IFUSLOT,AMP or exp,SPECID,IFUSLOT,IFUID,AMP; repeatable")
    parser.add_argument("--inspect-output-dir", default="amplifier_component_inspection",
                        help="output directory for individual amplifier inspection plots")
    args = parser.parse_args()
    if any(s < 0.0 for s in args.image_sigma_pix):
        parser.error("--image-sigma-pix values must be nonnegative")
    if args.common_fq_template and args.template_csv:
        parser.error("--common-fq-template and --template-csv are separate workflows")
    if args.external_image_sigma_pix < 0.0:
        parser.error("--external-image-sigma-pix must be nonnegative")
    if args.reconstruction_sigma_pix < 0.0:
        parser.error("--reconstruction-sigma-pix must be nonnegative")
    if args.max_iterations < 1:
        parser.error("--max-iterations must be positive")
    if args.inspect_solution_dir and (args.common_fq_template or args.template_csv):
        parser.error("inspection mode cannot be combined with a fit workflow")
    sigmas = list(dict.fromkeys(float(s) for s in args.image_sigma_pix))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.common_fq_template:
        run_common_fq_external_fit(args, output_dir)
        return
    if args.inspect_solution_dir:
        run_component_inspection(args)
        return

    bands = ("ON", "OFF")
    filters = {"ON": read_filter(args.on_filter),
               "OFF": read_filter(args.off_filter)}
    images = {"ON": load_image(args.on_image),
              "OFF": load_image(args.off_image)}
    templates = load_template(args.template_csv) if args.template_csv else None

    with tables.open_file(args.h5, mode="r") as h5:
        if "Info" not in h5.root._v_children or "Fibers" not in h5.root._v_children:
            raise ValueError("H5 needs Info and Fibers tables")
        if "Survey" not in h5.root._v_children:
            raise ValueError("H5 needs Survey table")
        info, fibers, survey = h5.root.Info, h5.root.Fibers, h5.root.Survey
        nrows = int(info.nrows)
        if int(fibers.nrows) != nrows:
            raise ValueError("Info/Fibers row mismatch")
        if fibers.coldtypes["spectrum"].shape != (DEF_WAVE.size,):
            raise ValueError("Fibers.spectrum does not use the 1036-bin grid")
        groups, labels = build_groups(info)
        ra = np.asarray(info.cols.ra[:], dtype=float)
        dec = np.asarray(info.cols.dec[:], dtype=float)
        ifuslot = np.asarray(info.cols.ifuslot[:])
        amp = np.asarray([text(v) for v in info.cols.amp[:]])
        bad = masked_rows(args.h5, ifuslot, amp)
        row_q = np.full(nrows, -1, dtype=int)
        amp_code = np.full(nrows, -1, dtype=int)
        amp_number = {name: number for number, name in enumerate(AMPLIFIERS)}
        for group in groups:
            j = group["j"]
            q = j if group["amp"] in ("LL", "RU") else 111 - j
            row_q[group["indices"]] = q
            amp_code[group["indices"]] = amp_number[group["amp"]]
        if np.any(row_q < 0) or np.any(amp_code < 0):
            raise ValueError("amplifier/q bookkeeping did not cover all H5 rows")
        survey_by_exp = {}
        for row in survey:
            exp = int(row["exp"])
            if exp in survey_by_exp:
                raise ValueError("Survey has duplicate exposure %d" % exp)
            survey_by_exp[exp] = row
        spectra_h5 = np.asarray(fibers.cols.spectrum[:], dtype=float)
        errors_h5 = np.asarray(fibers.cols.error[:], dtype=float)

        profile_arrays = {(band_name, sigma, amp_name): {}
                          for band_name in bands for sigma in sigmas
                          for amp_name in AMPLIFIERS}
        calibration_rows = []
        profile_rows = []
        exposure_values = [args.exposure] if args.exposure else range(1, 4)

        for exposure in exposure_values:
            if exposure not in survey_by_exp:
                raise ValueError("Survey has no row for exposure %d" % exposure)
            survey_row = survey_by_exp[exposure]
            offset = float(survey_row["offset"])
            if np.isfinite(offset) and offset != 0.0:
                # This is the requested spectral state immediately before
                # historical external-image normalization.
                working_spectrum = spectra_h5 / offset
                working_error = errors_h5 / offset
            else:
                working_spectrum = spectra_h5.copy()
                working_error = errors_h5.copy()
                print("WARNING exposure %d: invalid Survey.offset=%s; "
                      "using Fibers units unchanged" % (exposure, offset))

            exposure_rows = labels == exposure
            for band in bands:
                response = filters[band]
                V, _V_error = synthetic_mean(working_spectrum, working_error,
                                              response)
                eff_ra, eff_dec = adr_positions(ra, dec, survey_row, response)
                K_work = raw_work_basis(survey_row)
                K_band = weighted_scalar(K_work, response)
                if not np.isfinite(K_band) or K_band == 0.0:
                    raise ValueError("invalid K_band for %s exposure %d" % (band, exposure))

                for sigma in sigmas:
                    I, gradient, image_valid = sample_image(
                        images[band], eff_ra, eff_dec, sigma)
                    valid = exposure_rows & ~bad & np.isfinite(V) & image_valid
                    fit_a = fit_global(I[valid], V[valid], intercept=False)
                    fit_b = fit_global(I[valid], V[valid], intercept=True)
                    if fit_b["n"] < 3:
                        print("WARNING exposure %d %s %s: no valid fibers" %
                              (exposure, band, state_name(sigma)))
                        continue
                    model_b = fit_b["g"] * I + fit_b["z"]
                    model_a = fit_a["g"] * I
                    D = V - model_b
                    D_a = V - model_a
                    joint = None
                    relative_c = None
                    joint_residual = np.full(nrows, np.nan, dtype=float)
                    joint_c_residual = np.full(nrows, np.nan, dtype=float)
                    fiber_template = np.full(nrows, np.nan, dtype=float)
                    if templates:
                        for amp_name in AMPLIFIERS:
                            amp_rows = amp_code == amp_number[amp_name]
                            fiber_template[amp_rows] = templates[amp_name][row_q[amp_rows]]
                        joint_valid = valid & np.isfinite(fiber_template)
                        joint = fit_joint_model(
                            I[joint_valid], V[joint_valid], K_band,
                            amp_code[joint_valid], fiber_template[joint_valid],
                            fit_b)
                        joint_residual[joint_valid] = joint["residual"]
                        relative_c = fit_relative_c_model(
                            I[joint_valid], V[joint_valid], K_band,
                            amp_code[joint_valid], fiber_template[joint_valid],
                            joint)
                        joint_c_residual[joint_valid] = relative_c["residual"]
                    yscale = robust_scale(V[valid])
                    relative_z = abs(fit_b["z"]) / yscale if yscale else np.nan
                    gradient_valid = valid & np.isfinite(gradient) & np.isfinite(D)
                    if gradient_valid.sum() >= 3:
                        abs_d = np.abs(D[gradient_valid])
                        gradient_values = gradient[gradient_valid]
                        gradient_correlation = np.corrcoef(
                            abs_d, gradient_values)[0, 1]
                        high_gradient = gradient_values >= np.nanpercentile(
                            gradient_values, 75.0)
                        high_ratio = (np.nanmedian(abs_d[high_gradient]) /
                                      np.nanmedian(abs_d)) if np.nanmedian(abs_d) != 0 else np.nan
                    else:
                        gradient_correlation, high_ratio = np.nan, np.nan
                    amp_ratios = {}
                    for amp_name in AMPLIFIERS:
                        amp_valid = valid & (amp == amp_name)
                        g_amp = fixed_intercept_slope(I[amp_valid], V[amp_valid],
                                                      fit_b["z"])
                        amp_ratios[amp_name] = (g_amp / fit_b["g"]
                                                if np.isfinite(g_amp) and fit_b["g"] != 0
                                                else np.nan)
                    calrow = {
                        "h5": Path(args.h5).name, "exposure": exposure,
                        "band": band, "smoothing_state": state_name(sigma),
                        "valid_fibers": fit_b["n"],
                        "g_origin": fit_a["g"], "rms_origin": fit_a["rms"],
                        "g_intercept": fit_b["g"], "z": fit_b["z"],
                        "rms_intercept": fit_b["rms"],
                        "z_relative_importance": relative_z, "K_band": K_band,
                        "abs_D_gradient_correlation": gradient_correlation,
                        "high_gradient_abs_D_ratio": high_ratio,
                        "g_sequential": fit_b["g"],
                        "z_sequential": fit_b["z"],
                        "rms_before_joint": fit_b["rms"],
                        "g_joint": joint["params"][0] if joint else np.nan,
                        "z_joint": joint["params"][1] if joint else np.nan,
                        "rms_joint": joint["rms"] if joint else np.nan,
                        "rms_joint_with_C": relative_c["rms"] if relative_c else np.nan,
                        "fractional_improvement_relative_C": (
                            (joint["rms"] - relative_c["rms"]) / joint["rms"]
                            if joint and relative_c and joint["rms"] != 0.0 else np.nan),
                    }
                    for amp_name in AMPLIFIERS:
                        calrow["g_%s_over_global" % amp_name] = amp_ratios[amp_name]
                        number = amp_number[amp_name]
                        calrow["alpha_%s_joint" % amp_name] = (
                            joint["params"][2 + number] if joint else np.nan)
                        calrow["alpha_%s_sequential" % amp_name] = np.nan
                        calrow["C_%s_sequential" % amp_name] = np.nan
                        calrow["C_%s_relative" % amp_name] = (
                            relative_c["C"][number] if relative_c else np.nan)

                    template_fit_by_amp = {}
                    for amp_name in AMPLIFIERS:
                        amp_records = [g for g in groups if
                                       g["exposure"] == exposure and
                                       g["amp"] == amp_name]
                        d_profile = np.full(FIBERS_PER_AMPLIFIER, np.nan)
                        d_median_profile = np.full(FIBERS_PER_AMPLIFIER, np.nan)
                        d_a_profile = np.full(FIBERS_PER_AMPLIFIER, np.nan)
                        scatter = np.full(FIBERS_PER_AMPLIFIER, np.nan)
                        joint_profile = np.full(FIBERS_PER_AMPLIFIER, np.nan)
                        joint_c_profile = np.full(FIBERS_PER_AMPLIFIER, np.nan)
                        n_fibers = np.zeros(FIBERS_PER_AMPLIFIER, dtype=int)
                        n_ifus = np.zeros(FIBERS_PER_AMPLIFIER, dtype=int)
                        grad_profile = np.full(FIBERS_PER_AMPLIFIER, np.nan)
                        for q in range(FIBERS_PER_AMPLIFIER):
                            vals, vals_a, vals_joint, vals_joint_c = [], [], [], []
                            grads, slots = [], []
                            for group in amp_records:
                                j = q if amp_name in ("LL", "RU") else 111 - q
                                row = group["indices"][j]
                                if valid[row] and np.isfinite(D[row]):
                                    vals.append(D[row]); vals_a.append(D_a[row])
                                    grads.append(gradient[row]); slots.append(group["ifuslot"])
                                    if np.isfinite(joint_residual[row]):
                                        vals_joint.append(joint_residual[row] / K_band)
                                    if np.isfinite(joint_c_residual[row]):
                                        vals_joint_c.append(joint_c_residual[row] / K_band)
                            median, location, spread, count, slot_count = stack_profile(vals, slots)
                            d_median_profile[q] = median
                            d_profile[q], scatter[q] = location, spread
                            n_fibers[q], n_ifus[q] = count, slot_count
                            if grads:
                                grad_profile[q] = np.nanmedian(grads)
                            # Model A has the same q bookkeeping but no center
                            # subtraction or recentering of either profile.
                            if vals_a:
                                d_a_profile[q] = float(biweight(np.asarray(vals_a)))
                            if vals_joint:
                                joint_profile[q] = float(biweight(np.asarray(vals_joint)))
                            if vals_joint_c:
                                joint_c_profile[q] = float(biweight(np.asarray(vals_joint_c)))
                        template_fit = fit_template(
                            d_profile / K_band,
                            templates[amp_name] if templates else np.full(112, np.nan)) \
                            if templates else None
                        template_fit_by_amp[amp_name] = template_fit
                        if template_fit:
                            calrow["alpha_%s_sequential" % amp_name] = template_fit["alpha"]
                            calrow["C_%s_sequential" % amp_name] = template_fit["C"]
                        profile_arrays[(band, sigma, amp_name)][exposure] = {
                            "D": d_profile, "D_origin": d_a_profile,
                            "D_raw": d_profile / K_band,
                            "D_origin_raw": d_a_profile / K_band,
                            "joint_raw": joint_profile,
                            "joint_c_raw": joint_c_profile,
                            "scatter": scatter / abs(K_band),
                            "n": n_fibers, "n_ifus": n_ifus,
                            "gradient": grad_profile, "template": template_fit,
                        }
                        for q in range(FIBERS_PER_AMPLIFIER):
                            row = {
                                "h5": Path(args.h5).name, "exposure": exposure,
                                "band": band, "smoothing_state": state_name(sigma),
                                "amplifier": amp_name, "q": q,
                                "D_median": d_median_profile[q], "D_scatter": scatter[q],
                                "D_origin_median": d_a_profile[q],
                                "D_raw_equiv_e_per_A": d_profile[q] / K_band,
                                "D_biweight": d_profile[q],
                                "D_origin_raw_equiv_e_per_A": d_a_profile[q] / K_band,
                                "joint_residual_raw_equiv": joint_profile[q],
                                "joint_plus_C_residual_raw_equiv": joint_c_profile[q],
                                "n_fibers": n_fibers[q], "n_physical_ifus": n_ifus[q],
                                "external_gradient_median": grad_profile[q],
                            }
                            if template_fit:
                                row.update({
                                    "C_template_e_per_A": template_fit["C"],
                                    "alpha_template_e_per_A": template_fit["alpha"],
                                    "template_rms_before_e_per_A": template_fit["rms_before"],
                                    "template_rms_after_e_per_A": template_fit["rms_after"],
                                    "template_alpha_c0_e_per_A": template_fit["alpha_c0"],
                                    "template_rms_c0_e_per_A": template_fit["rms_c0"],
                                })
                            else:
                                row.update({field: np.nan for field in (
                                    "C_template_e_per_A", "alpha_template_e_per_A",
                                    "template_rms_before_e_per_A",
                                    "template_rms_after_e_per_A",
                                    "template_alpha_c0_e_per_A",
                                    "template_rms_c0_e_per_A")})
                            profile_rows.append(row)

                    calibration_rows.append(calrow)

                    print("exposure %d %s [%s]: valid=%d, g=%+.6g, z=%+.6g, "
                          "RMS(A)=%g, RMS(B)=%g, K=%g, |z|/scale=%g" %
                          (exposure, band, state_name(sigma), fit_b["n"],
                           fit_b["g"], fit_b["z"], fit_a["rms"], fit_b["rms"],
                           K_band, relative_z))
                    if joint:
                        print("  JOINT MODEL J0: g=%+.6g z=%+.6g "
                              "alpha_LL=%+.6g alpha_LU=%+.6g alpha_RL=%+.6g "
                              "alpha_RU=%+.6g RMS=%g (before=%g)" %
                              (joint["params"][0], joint["params"][1],
                               joint["params"][2], joint["params"][3],
                               joint["params"][4], joint["params"][5],
                               joint["rms"], fit_b["rms"]))
                        for amp_name in AMPLIFIERS:
                            number = amp_number[amp_name]
                            joint_records = profile_arrays[(band, sigma, amp_name)][exposure]
                            print("    %s median post-fit residual q<20=%+.6g "
                                  "q>=40=%+.6g" %
                                  (amp_name,
                                   np.nanmedian(joint_records["joint_raw"][:20]),
                                   np.nanmedian(joint_records["joint_raw"][40:])))
                        print("  RELATIVE-C MODEL J1: C_LL=%+.6g C_LU=%+.6g "
                              "C_RL=%+.6g C_RU=%+.6g RMS=%g "
                              "fractional improvement=%+.6g" %
                              (relative_c["C"][0], relative_c["C"][1],
                               relative_c["C"][2], relative_c["C"][3],
                               relative_c["rms"],
                               (joint["rms"] - relative_c["rms"]) / joint["rms"]
                               if joint["rms"] != 0.0 else np.nan))

    write_csvs(output_dir, profile_rows, calibration_rows)
    primary_sigma = sigmas[0]
    plot_profiles(output_dir, profile_arrays, bands, sigmas, primary_sigma)
    plot_raw_equivalent(output_dir, profile_arrays, bands, sigmas, primary_sigma)
    plot_amp_qa(output_dir, calibration_rows, bands, sigmas)
    joint_sigma = 4.5 if 4.5 in sigmas else primary_sigma
    plot_joint_residuals(output_dir, profile_arrays, calibration_rows,
                         bands, sigmas, joint_sigma)
    plot_joint_vs_sequential(output_dir, profile_arrays, bands, joint_sigma)
    plot_joint_parameters(output_dir, calibration_rows, bands, joint_sigma)
    if templates:
        plot_template(output_dir, profile_arrays, templates, bands, primary_sigma)

    for sigma in sigmas:
        print("\n%s profile comparisons" % state_name(sigma))
        for amp_name in AMPLIFIERS:
            on_records = profile_arrays[("ON", sigma, amp_name)]
            off_records = profile_arrays[("OFF", sigma, amp_name)]
            on = np.nanmedian(np.asarray([r["D_raw"] for r in on_records.values()]), axis=0) if on_records else np.full(112, np.nan)
            off = np.nanmedian(np.asarray([r["D_raw"] for r in off_records.values()]), axis=0) if off_records else np.full(112, np.nan)
            valid = np.isfinite(on) & np.isfinite(off)
            correlation = np.corrcoef(on[valid], off[valid])[0, 1] if valid.sum() >= 3 else np.nan
            difference_rms = robust_rms(on[valid] - off[valid]) if valid.any() else np.nan
            print("%s: ON/OFF correlation=%+.4f, robust RMS difference=%g e-/A" %
                  (amp_name, correlation, difference_rms))

            if on_records:
                on_q = on
                print("  ON  median q<20=%+.6g, q>=40=%+.6g, peak q=%d value=%+.6g" %
                      (np.nanmedian(on_q[:20]), np.nanmedian(on_q[40:]),
                       int(np.nanargmax(on_q)), np.nanmax(on_q)))
            if off_records:
                off_q = off
                print("  OFF median q<20=%+.6g, q>=40=%+.6g, peak q=%d value=%+.6g" %
                      (np.nanmedian(off_q[:20]), np.nanmedian(off_q[40:]),
                       int(np.nanargmax(off_q)), np.nanmax(off_q)))
            if templates:
                for band in bands:
                    fits = [r["template"] for r in profile_arrays[(band, sigma, amp_name)].values()]
                    if fits:
                        print("  %s template: C=%+.6g, alpha=%+.6g, RMS before=%g, "
                              "after=%g, C=0 after=%g" %
                              (band, np.nanmedian([f["C"] for f in fits]),
                               np.nanmedian([f["alpha"] for f in fits]),
                               np.nanmedian([f["rms_before"] for f in fits]),
                               np.nanmedian([f["rms_after"] for f in fits]),
                               np.nanmedian([f["rms_c0"] for f in fits])))
        print("\n%s ON/OFF joint-parameter comparison" % state_name(sigma))
        for amp_name in AMPLIFIERS:
            on_rows = {(int(row["exposure"]), row["smoothing_state"]): row
                       for row in calibration_rows if row["band"] == "ON" and
                       row["smoothing_state"] == state_name(sigma)}
            off_rows = {(int(row["exposure"]), row["smoothing_state"]): row
                        for row in calibration_rows if row["band"] == "OFF" and
                        row["smoothing_state"] == state_name(sigma)}
            paired = sorted(set(on_rows) & set(off_rows))
            alpha_on = np.asarray([float(on_rows[key]["alpha_%s_joint" % amp_name])
                                   for key in paired])
            alpha_off = np.asarray([float(off_rows[key]["alpha_%s_joint" % amp_name])
                                    for key in paired])
            finite = np.isfinite(alpha_on) & np.isfinite(alpha_off)
            if finite.any():
                denominator = np.maximum(
                    0.5 * (np.abs(alpha_on[finite]) + np.abs(alpha_off[finite])),
                    np.finfo(float).eps)
                fractional = np.abs(alpha_on[finite] - alpha_off[finite]) / denominator
                correlation = (np.corrcoef(alpha_on[finite], alpha_off[finite])[0, 1]
                               if finite.sum() >= 3 else np.nan)
                c_on = np.asarray([float(on_rows[key]["C_%s_relative" % amp_name])
                                   for key in paired])[finite]
                c_off = np.asarray([float(off_rows[key]["C_%s_relative" % amp_name])
                                    for key in paired])[finite]
                c_finite = np.isfinite(c_on) & np.isfinite(c_off)
                c_correlation = (np.corrcoef(c_on[c_finite], c_off[c_finite])[0, 1]
                                 if c_finite.sum() >= 3 else np.nan)
                print("  %s alpha fractional differences=%s, correlation=%+.4f; "
                      "relative C correlation=%+.4f" %
                      (amp_name, np.array2string(fractional, precision=4),
                       correlation, c_correlation))
    print("\nWrote diagnostic products in %s" % output_dir)


if __name__ == "__main__":
    main()
