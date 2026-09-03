#!/usr/bin/env python3
"""Compact hierarchical M101 calibration diagnostic.

The illumination diagnostic uses these physical equations (the broad source
fit retains its empirical ``z_source_fit`` intercept):

    E_i,b = Q_i,b + C_image,b
    I_object_i,b = exact_aperture(E_i,b - C_image,b)

    V0_i,e,b = V_i,e,b - K_e,b * alpha_e,AMP * f(q_i)
    T_i,e,b = V0_i,e,b + B_sky_i,e,b
    T_i,e,b = s_e,IFU,b * [g_e,b * Q_i,b + B_sky_i,e,b] + delta_e,b

Equivalently, in the stored sky-subtracted basis,

    V0 = s_IFU*g*Q + (s_IFU - 1)*B_sky + delta.

Here ``delta`` is one global VIRUS-side additive residual per exposure/band,
fixed by the gauge median(s_IFU)=1.  It is not the broad-fit
``z_source_fit``.

where ``f(q)`` is fixed from ``--fq-template`` and alpha has only twelve
values for this three-exposure test: one per exposure and LL/LU/RL/RU
orientation.  ON and OFF share each alpha.  The FoV illumination correction
is deliberately disabled for this diagnostic: ``F(x,y) = 1`` everywhere.
This script writes diagnostic tables and plots only; it never imports or
modifies a production cube builder and never writes an H5 file.
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
from astropy.convolution import convolve, Gaussian2DKernel
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from astropy.table import Table
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
from photutils.aperture import CircularAperture, aperture_photometry
from photutils.detection import DAOStarFinder
from scipy.ndimage import gaussian_filter, gaussian_filter1d
from scipy.optimize import brentq, least_squares
from scipy.spatial import cKDTree

from astrometry import Astrometry
from extract import Extract
from math_utils import biweight


DEF_WAVE = np.linspace(3470.0, 5540.0, 1036)
N_FIBER_AMP = 112
N_EXPOSURES = 3
AMPS = ("LL", "LU", "RL", "RU")
ALPHA_LOW, ALPHA_HIGH = 0.2, 0.6
ALPHA_INITIAL = 0.4
FWHM_TO_SIGMA = 1.0 / 2.35482
FIBER_RADIUS_ARCSEC = 0.75
LEGACY_IMAGE_SIGMA_PIX = 4.5
M101_CENTER_RA = 210.800
M101_CENTER_DEC = 54.333
EXTERNAL_BACKGROUND_RADIUS_ARCMIN = 16.0

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


def as_text(value):
    if isinstance(value, (bytes, np.bytes_)):
        return value.decode("utf-8", errors="replace").strip()
    return str(value).strip()


def exposure_labels(nrows, nslots):
    if nslots <= 0 or nrows % (N_FIBER_AMP * 4 * nslots) != 0:
        raise ValueError("cannot infer three interleaved exposures from H5 rows")
    inferred = nrows // (N_FIBER_AMP * 4 * nslots)
    if inferred != N_EXPOSURES:
        raise ValueError("expected three exposures, inferred %d" % inferred)
    rows = np.arange(nrows)
    return ((rows // N_FIBER_AMP) % N_EXPOSURES + 1).astype(int)


def build_groups(info):
    required = {"ifuslot", "amp", "specid", "ifuid"}
    if not required.issubset(info.colnames):
        raise ValueError("Info lacks physical amplifier bookkeeping columns")
    ifuslot = np.asarray(info.cols.ifuslot[:])
    amp = np.asarray([as_text(v) for v in info.cols.amp[:]])
    specid = np.asarray(info.cols.specid[:])
    ifuid = np.asarray(info.cols.ifuid[:])
    labels = exposure_labels(int(info.nrows), len(np.unique(ifuslot)))
    keys = sorted(set(zip(labels.tolist(), ifuslot.tolist(), amp.tolist())))
    groups = []
    for exposure, slot, amplifier in keys:
        indices = np.flatnonzero((labels == exposure) &
                                 (ifuslot == slot) & (amp == amplifier))
        if indices.size != N_FIBER_AMP:
            raise ValueError("%d/%s/%s has %d fibers, not 112" %
                             (exposure, slot, amplifier, indices.size))
        if np.unique(specid[indices]).size != 1 or np.unique(ifuid[indices]).size != 1:
            raise ValueError("inconsistent SPECID/IFUID in physical amplifier")
        group = {
            "exposure": int(exposure), "specid": int(specid[indices[0]]),
            "ifuslot": int(slot), "ifuid": int(ifuid[indices[0]]),
            "amp": amplifier, "indices": indices,
        }
        group["identity"] = (group["exposure"], group["specid"],
                              group["ifuslot"], group["ifuid"], group["amp"])
        groups.append(group)
    if set(g["amp"] for g in groups) - set(AMPS):
        raise ValueError("unexpected amplifier orientation in H5")
    return groups, labels


def masked_rows(h5_path, ifuslot, amp):
    try:
        date = int(Path(h5_path).name.split("_")[0])
    except (IndexError, ValueError):
        return np.zeros(ifuslot.shape, dtype=bool)
    bad = np.zeros(ifuslot.shape, dtype=bool)
    for key, names in mask_dict.items():
        start, stop = (int(v) for v in key.split("-"))
        if start <= date < stop:
            for name in names:
                bad |= ((ifuslot == int(name[:3])) & (amp == name[3:]))
    return bad


def read_filter(path):
    table = Table.read(path, format="ascii")
    wave = np.asarray(table["Wavelength"], dtype=float)
    response = np.asarray(table["R"], dtype=float)
    response = np.interp(DEF_WAVE, wave, response, left=0.0, right=0.0)
    if not np.any(np.isfinite(response) & (response != 0.0)):
        raise ValueError("filter has no usable response on the VIRUS grid")
    return response


def synthetic_mean(spectra, response):
    finite = np.isfinite(spectra) & np.isfinite(response)[None, :]
    weights = np.where(finite, response[None, :], 0.0)
    denominator = np.sum(weights, axis=1)
    result = np.full(spectra.shape[0], np.nan, dtype=float)
    good = denominator != 0.0
    result[good] = np.sum(np.where(finite, spectra, 0.0) * response[None, :],
                          axis=1)[good] / denominator[good]
    return result


def adr_positions(ra, dec, survey_row, response):
    effective = Astrometry(float(survey_row["ra"]), float(survey_row["dec"]),
                           float(survey_row["pa"]), 0.0, 0.0)
    extractor = Extract(wave=DEF_WAVE)
    extractor.get_ADR_RAdec(effective)
    dra = extractor.ADRra / 3600.0 / np.cos(np.deg2rad(float(survey_row["dec"])))
    ddec = extractor.ADRdec / 3600.0
    ra_wave = ra[:, None] - dra[None, :]
    dec_wave = dec[:, None] - ddec[None, :]
    weights = np.where(np.isfinite(response) & (response != 0.0), response, 0.0)
    denominator = np.sum(weights)
    return (np.sum(ra_wave * weights[None, :], axis=1) / denominator,
            np.sum(dec_wave * weights[None, :], axis=1) / denominator)


def raw_work_basis(survey_row):
    exptime = float(survey_row["exptime"])
    millum = float(survey_row["millum"])
    guider_throughput = float(survey_row["throughput"])
    if not np.isfinite(exptime) or exptime == 0.0:
        raise ValueError("invalid Survey.exptime")
    gratio = millum * guider_throughput / 5e5
    if not np.isfinite(gratio) or gratio == 0.0:
        raise ValueError("invalid Survey guider ratio")
    table = Table.read(Path(__file__).resolve().parent / "CALS" / "throughput.txt",
                       format="ascii.fixed_width_two_line")
    standard = np.asarray(table["throughput"], dtype=float)
    if standard.size != DEF_WAVE.size or not np.allclose(
            np.asarray(table["wavelength"], dtype=float), DEF_WAVE):
        raise ValueError("CALS/throughput.txt does not match VIRUS grid")
    mult = (6.626e-27 * (3e18 / DEF_WAVE) / 360.0 / 5e5 / 0.92 * 5)
    mult *= 1e29 * DEF_WAVE**2 / 2.99792e18
    final_norm = 1e-29 * 2.99792e18 / DEF_WAVE**2 * 1e17
    return mult * (360.0 / exptime) / standard / gratio * final_norm


def weighted_scalar(values, response):
    valid = np.isfinite(values) & np.isfinite(response) & (response != 0.0)
    denominator = np.sum(np.where(valid, response, 0.0))
    return (float(np.sum(np.where(valid, values * response, 0.0)) / denominator)
            if denominator != 0.0 else np.nan)


def load_image(path):
    with fits.open(path, memmap=True) as hdul:
        data = np.asarray(hdul[0].data, dtype=float).copy()
        header = hdul[0].header.copy()
    if data.ndim != 2:
        raise ValueError("%s must contain a 2-D primary image" % path)
    wcs = WCS(header).celestial
    scales = np.asarray(proj_plane_pixel_scales(wcs), dtype=float) * 3600.0
    scales = scales[np.isfinite(scales) & (scales > 0.0)]
    if scales.size == 0:
        raise ValueError("%s WCS has no usable angular pixel scale" % path)
    pixel_scale = float(np.median(scales))
    return {"data": data, "header": header, "wcs": wcs,
            "pixel_scale_arcsec": pixel_scale, "smooth": {},
            "matched": {}, "matched_raw": {}, "psf": None, "path": Path(path),
            "background": np.nan, "background_scatter": np.nan,
            "background_npix": 0, "background_annulus": "not measured",
            "object_data": data.copy()}


def estimate_external_background(image, center_ra=M101_CENTER_RA,
                                 center_dec=M101_CENTER_DEC,
                                 radius_arcmin=EXTERNAL_BACKGROUND_RADIUS_ARCMIN):
    """Measure ``C_image`` in ``E = Q + C_image`` from an outer annulus."""
    data = image["data"]
    center_x, center_y = image["wcs"].world_to_pixel_values(center_ra, center_dec)
    if not np.isfinite(center_x) or not np.isfinite(center_y):
        raise ValueError("external image WCS cannot project the known M101 center")
    radius_pix = radius_arcmin * 60.0 / image["pixel_scale_arcsec"]
    chunks = []
    x = np.arange(data.shape[1], dtype=float)
    for y0 in range(0, data.shape[0], 512):
        y1 = min(y0 + 512, data.shape[0])
        yy = np.arange(y0, y1, dtype=float)[:, None]
        annulus = np.hypot(x[None, :] - center_x, yy - center_y) >= radius_pix
        values = data[y0:y1][annulus]
        finite = values[np.isfinite(values)]
        if finite.size:
            chunks.append(finite)
    values = np.concatenate(chunks) if chunks else np.asarray([], dtype=float)
    if values.size < 100:
        raise ValueError("external image outer background annulus has only %d valid pixels" % values.size)
    clipped, median, clipped_std = sigma_clipped_stats(values, sigma=3.0, maxiters=5,
                                                       cenfunc="median", stdfunc="std")
    del clipped
    scatter = robust_scale(values - float(median))
    if not np.isfinite(scatter):
        scatter = float(clipped_std)
    image["background"] = float(median)
    image["background_scatter"] = float(scatter)
    image["background_npix"] = int(values.size)
    image["background_annulus"] = "r >= %.3f arcmin about RA=%.3f Dec=%.3f" % \
        (radius_arcmin, center_ra, center_dec)
    image["object_data"] = data - image["background"]
    print("%s external background: C_image=%.6g scatter=%.6g Npix=%d (%s)" %
          (image["path"], image["background"], image["background_scatter"],
           image["background_npix"], image["background_annulus"]))
    return image


def sample_image_legacy(image, ra, dec, sigma=LEGACY_IMAGE_SIGMA_PIX):
    """The old fixed-sigma nearest-pixel measurement, retained for QA only."""
    if sigma not in image["smooth"]:
        data = image["data"]
        finite = np.isfinite(data)
        numerator = gaussian_filter(np.where(finite, data, 0.0), sigma,
                                    mode="nearest")
        denominator = gaussian_filter(finite.astype(float), sigma,
                                      mode="nearest")
        smoothed = np.full(data.shape, np.nan, dtype=float)
        good = denominator > 0.0
        smoothed[good] = numerator[good] / denominator[good]
        image["smooth"][sigma] = smoothed
    data = image["smooth"][sigma]
    x, y = image["wcs"].world_to_pixel_values(ra, dec)
    valid = np.isfinite(x) & np.isfinite(y)
    xi = np.zeros(x.shape, dtype=int); yi = np.zeros(y.shape, dtype=int)
    xi[valid] = np.rint(x[valid]).astype(int); yi[valid] = np.rint(y[valid]).astype(int)
    valid &= (xi >= 0) & (xi < data.shape[1]) & (yi >= 0) & (yi < data.shape[0])
    result = np.full(ra.shape, np.nan, dtype=float)
    result[valid] = data[yi[valid], xi[valid]]
    valid &= np.isfinite(result)
    return result, valid


def _header_float(header, key):
    try:
        value = float(header[key])
    except (KeyError, TypeError, ValueError, OverflowError):
        return np.nan
    return value if np.isfinite(value) and value > 0.0 else np.nan


def _coarse_background(data, tile=128):
    """Estimate a smooth background without making a full-size filter copy."""
    finite = np.isfinite(data)
    if not finite.any():
        return np.zeros(data.shape, dtype=float)
    fill = float(np.nanmedian(data))
    ny, nx = data.shape
    gy = (ny + tile - 1) // tile
    gx = (nx + tile - 1) // tile
    padded = np.full((gy * tile, gx * tile), fill, dtype=float)
    padded[:ny, :nx] = np.where(finite, data, fill)
    coarse = np.nanmedian(padded.reshape(gy, tile, gx, tile), axis=(1, 3))
    coarse = gaussian_filter(coarse, sigma=2.0, mode="nearest")
    y = np.arange(ny, dtype=float)
    x = np.arange(nx, dtype=float)
    # Two 1-D interpolations avoid a full-resolution Gaussian background
    # operation while preserving large-scale galaxy/background structure.
    centers_y = np.arange(gy, dtype=float) * tile + tile / 2.0
    centers_x = np.arange(gx, dtype=float) * tile + tile / 2.0
    vertical = np.empty((ny, gx), dtype=float)
    for col in range(gx):
        vertical[:, col] = np.interp(y, centers_y, coarse[:, col],
                                     left=coarse[0, col], right=coarse[-1, col])
    background = np.empty((ny, nx), dtype=float)
    for row in range(ny):
        background[row] = np.interp(x, centers_x, vertical[row],
                                    left=vertical[row, 0], right=vertical[row, -1])
    return background


def _measure_star_fwhm(data, x, y, half_size=8, saturation=np.nan):
    """Return an equivalent circular Gaussian FWHM in pixels from moments."""
    x0, y0 = int(np.rint(x)), int(np.rint(y))
    y1, y2 = y0 - half_size, y0 + half_size + 1
    x1, x2 = x0 - half_size, x0 + half_size + 1
    if x1 < 0 or y1 < 0 or x2 > data.shape[1] or y2 > data.shape[0]:
        return np.nan
    cut = np.asarray(data[y1:y2, x1:x2], dtype=float)
    if not np.isfinite(cut).all():
        return np.nan
    edge = np.concatenate((cut[0], cut[-1], cut[1:-1, 0], cut[1:-1, -1]))
    background = float(np.median(edge))
    signal = cut - background
    if np.isfinite(saturation) and np.nanmax(cut) >= 0.98 * saturation:
        return np.nan
    peak = np.nanmax(signal)
    if not np.isfinite(peak) or peak <= 0.0:
        return np.nan
    # A broad flat peak is a simple saturation/extended-source rejection.
    if np.count_nonzero(signal >= 0.98 * peak) > 4:
        return np.nan
    yy, xx = np.indices(signal.shape, dtype=float)
    weights = np.clip(signal, 0.0, None)
    radius = np.hypot(xx - half_size, yy - half_size)
    weights[radius > half_size - 1] = 0.0
    total = np.sum(weights)
    if not np.isfinite(total) or total <= 0.0:
        return np.nan
    cx = np.sum(weights * xx) / total
    cy = np.sum(weights * yy) / total
    dx, dy = xx - cx, yy - cy
    cov_xx = np.sum(weights * dx * dx) / total
    cov_yy = np.sum(weights * dy * dy) / total
    cov_xy = np.sum(weights * dx * dy) / total
    eig = np.linalg.eigvalsh([[cov_xx, cov_xy], [cov_xy, cov_yy]])
    if not np.all(np.isfinite(eig)) or eig[0] <= 0.0:
        return np.nan
    if np.sqrt(eig[1] / eig[0]) > 2.0:
        return np.nan
    fwhm = 2.35482 * np.sqrt(np.mean(eig))
    return float(fwhm) if 0.8 <= fwhm <= 12.0 else np.nan


def characterize_external_image(image, band):
    """Get one spatial PSF FWHM for an external image, in arcseconds."""
    header_keys = ("SEEING", "IMAGE_FWHM", "PSF_FWHM", "FWHM", "IQ", "PSF")
    for key in header_keys:
        value = _header_float(image["header"], key)
        if np.isfinite(value):
            result = {"band": band, "source": "header:%s" % key,
                      "fwhm_arcsec": value, "candidate_count": 0,
                      "accepted_fwhm_arcsec": np.asarray([], dtype=float),
                      "accepted_count": 0, "median": value,
                      "p16": np.nan, "p84": np.nan}
            image["psf"] = result
            print("%s external FWHM: %.4g arcsec from %s (PHOTFWHM ignored)" %
                  (band, value, key))
            print("  candidate stars=0, accepted=0, median=%.4g, p16/p84=nan/nan" % value)
            return result

    data = image["data"]
    finite = np.isfinite(data)
    _, median, std = sigma_clipped_stats(data[finite], sigma=3.0, stdfunc=np.std)
    background = _coarse_background(data)
    detection = np.where(finite, data - background, 0.0)
    scale = max(float(std), robust_scale(detection), 1e-12)
    finder = DAOStarFinder(fwhm=4.0, threshold=5.0 * scale,
                           exclude_border=True)
    sources = finder(detection)
    candidate_count = 0 if sources is None else len(sources)
    if sources is None or candidate_count == 0:
        raise ValueError("%s external %s image: no candidate stars found for PSF measurement" %
                         (image["path"], band))
    positions = np.column_stack((np.asarray(sources["xcentroid"], dtype=float),
                                 np.asarray(sources["ycentroid"], dtype=float)))
    tree = cKDTree(positions)
    saturation = _header_float(image["header"], "SATURATE")
    accepted = []
    for index, (x, y) in enumerate(positions):
        neighbors = tree.query_ball_point([x, y], r=6.0)
        if len(neighbors) > 1:
            continue
        fwhm_pix = _measure_star_fwhm(data, x, y, saturation=saturation)
        if np.isfinite(fwhm_pix):
            accepted.append(fwhm_pix * image["pixel_scale_arcsec"])
    accepted = np.asarray(accepted, dtype=float)
    if accepted.size < 3:
        raise ValueError("%s external %s image: only %d usable stars out of %d candidates for PSF measurement" %
                         (image["path"], band, accepted.size, candidate_count))
    p16, median_fwhm, p84 = np.percentile(accepted, [16, 50, 84])
    result = {"band": band, "source": "measured stellar FWHM",
              "fwhm_arcsec": float(median_fwhm), "candidate_count": candidate_count,
              "accepted_fwhm_arcsec": accepted, "accepted_count": int(accepted.size),
              "median": float(median_fwhm), "p16": float(p16), "p84": float(p84)}
    image["psf"] = result
    print("%s external FWHM: candidates=%d, accepted=%d, median=%.4g, p16/p84=%.4g/%.4g arcsec" %
          (band, candidate_count, accepted.size, median_fwhm, p16, p84))
    return result


def matched_external_image(image, exposure, virus_fwhm, band):
    """Return the cached Gaussian-PSF-matched image for one exposure/band."""
    if exposure in image["matched"]:
        return image["matched"][exposure]
    image_fwhm = float(image["psf"]["fwhm_arcsec"])
    if virus_fwhm <= image_fwhm:
        print("exposure %d %s: external image PSF broader than VIRUS seeing; "
              "no additional convolution" % (exposure, band))
        matched_raw = image["data"]
        matched = image["object_data"]
    else:
        kernel_fwhm = np.sqrt(virus_fwhm ** 2 - image_fwhm ** 2)
        sigma_pix = kernel_fwhm * FWHM_TO_SIGMA / image["pixel_scale_arcsec"]
        print("exposure %d %s: FWHM_virus=%.4g, FWHM_image=%.4g, "
              "FWHM_kernel=%.4g arcsec, sigma_kernel=%.4g pix" %
              (exposure, band, virus_fwhm, image_fwhm, kernel_fwhm, sigma_pix))
        kernel = Gaussian2DKernel(sigma_pix)
        matched_raw = convolve(image["data"], kernel, boundary="extend",
                               nan_treatment="interpolate", preserve_nan=True)
        matched = convolve(image["object_data"], kernel, boundary="extend",
                           nan_treatment="interpolate", preserve_nan=True)
    image["matched_raw"][exposure] = matched_raw
    image["matched"][exposure] = matched
    return matched


def sample_image_exact(image, matched_data, ra, dec):
    """Measure floating-coordinate exact circular fiber aperture sums."""
    x, y = image["wcs"].world_to_pixel_values(ra, dec)
    finite_xy = np.isfinite(x) & np.isfinite(y)
    value = np.full(ra.shape, np.nan, dtype=float)
    if finite_xy.any():
        positions = np.column_stack((x[finite_xy], y[finite_xy]))
        radius_pix = FIBER_RADIUS_ARCSEC / image["pixel_scale_arcsec"]
        aperture = CircularAperture(positions, r=radius_pix)
        table = aperture_photometry(matched_data, aperture,
                                    mask=~np.isfinite(matched_data),
                                    method="exact")
        value[finite_xy] = np.asarray(table["aperture_sum"], dtype=float)
    valid = finite_xy & np.isfinite(value)
    return value, valid


def load_fq(path):
    with Path(path).open(newline="") as stream:
        reader = csv.DictReader(stream)
        fields = {f.lower(): f for f in (reader.fieldnames or [])}
        value_field = fields.get("f", fields.get("f_converged"))
        if "q" not in fields or value_field is None:
            raise ValueError("f(q) template needs q,f columns")
        rows = [(int(float(row[fields["q"]])), float(row[value_field]))
                for row in reader]
    if len(rows) != N_FIBER_AMP or sorted(q for q, _ in rows) != list(range(N_FIBER_AMP)):
        raise ValueError("f(q) template must contain exactly q=0..111")
    f = np.full(N_FIBER_AMP, np.nan)
    for q, value in rows:
        f[q] = value
    if not np.all(np.isfinite(f)):
        raise ValueError("f(q) template contains nonfinite values")
    print("fixed f(q): median q<20=%g, median q>=40=%g, peak q=%d, peak=%g" %
          (np.median(f[:20]), np.median(f[40:]), np.argmax(f), np.max(f)))
    return f


def robust_scale(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not values.size:
        return np.nan
    center = np.median(values)
    mad = np.median(np.abs(values - center))
    return float(1.4826 * mad) if mad > 0.0 else max(float(np.std(values)), 1e-12)


def robust_location(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not values.size:
        return np.nan
    value = biweight(values)
    return float(value) if np.isfinite(value) else float(np.median(values))


def robust_rms(values):
    return robust_scale(values)


def robust_zero_slope(x, y, max_iter=30):
    """Fit ``y = slope*x`` with transparent Huber IRLS bookkeeping."""
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = np.asarray(x, dtype=float)[valid], np.asarray(y, dtype=float)[valid]
    result = {"slope": np.nan, "uncertainty": np.nan, "rms": np.nan,
              "n": int(x.size), "leverage": float(np.sum(x * x)) if x.size else np.nan,
              "x": x, "y": y, "residual": np.asarray([], dtype=float)}
    denominator = float(np.dot(x, x)) if x.size else 0.0
    if x.size < 3 or denominator <= 0.0:
        return result
    slope = float(np.dot(x, y) / denominator)
    for _ in range(max_iter):
        residual = y - slope * x
        scale = robust_scale(residual)
        scale = max(scale if np.isfinite(scale) else 0.0, 1e-12)
        standardized = np.abs(residual) / scale
        weights = np.ones_like(residual)
        outlying = standardized > 1.345
        weights[outlying] = 1.345 / standardized[outlying]
        weighted_denominator = float(np.sum(weights * x * x))
        if weighted_denominator <= 0.0:
            break
        updated = float(np.sum(weights * x * y) / weighted_denominator)
        if np.isclose(updated, slope, rtol=1e-9, atol=1e-12):
            slope = updated
            break
        slope = updated
    residual = y - slope * x
    scale = robust_scale(residual)
    scale = max(scale if np.isfinite(scale) else 0.0, 1e-12)
    standardized = np.abs(residual) / scale
    weights = np.ones_like(residual)
    outlying = standardized > 1.345
    weights[outlying] = 1.345 / standardized[outlying]
    weighted_denominator = float(np.sum(weights * x * x))
    dof = max(float(np.sum(weights) - 1.0), 1.0)
    variance = float(np.sum(weights * residual * residual) / dof)
    result.update({"slope": slope,
                   "uncertainty": float(np.sqrt(variance / weighted_denominator))
                   if weighted_denominator > 0.0 else np.nan,
                   "rms": float(scale), "residual": residual})
    return result


def source_sky_leverage(source_predictor, sky_predictor):
    """Return source/sky predictors and leverage from their sum."""
    source_predictor = np.asarray(source_predictor, dtype=float)
    sky_predictor = np.asarray(sky_predictor, dtype=float)
    total = source_predictor + sky_predictor
    valid = np.isfinite(source_predictor) & np.isfinite(sky_predictor) & np.isfinite(total)
    source = source_predictor[valid]
    sky = sky_predictor[valid]
    total = total[valid]
    fraction = np.full(total.shape, np.nan, dtype=float)
    nonzero = total != 0.0
    fraction[nonzero] = sky[nonzero] / total[nonzero]
    return {"source": source, "sky": sky, "total": total, "sky_fraction": fraction,
            "n": int(total.size), "leverage": float(np.sum(total * total))
            if total.size else np.nan,
            "median_source": robust_location(source),
            "median_sky": robust_location(sky),
            "median_total": robust_location(total),
            "median_sky_fraction": robust_location(fraction)}


def measure_ifu_scalar(x, y):
    """Measure one independent ON or OFF physical-IFU scalar."""
    fit = robust_zero_slope(x, y)
    valid = np.isfinite(x) & np.isfinite(y)
    x = np.asarray(x, dtype=float)[valid]
    fit["leverage"] = float(np.sum(x * x)) if x.size else np.nan
    fit["median_x"] = robust_location(x)
    fit["p10_x"] = float(np.percentile(x, 10)) if x.size else np.nan
    fit["p90_x"] = float(np.percentile(x, 90)) if x.size else np.nan
    fit["robust_scatter_x"] = robust_scale(x)
    return fit


def measure_common_ifu_scalar(measurements):
    """Fit one shared scalar to ON/OFF, weighting by each band's scatter."""
    valid_measurements = [m for m in measurements if m["n"] >= 3 and
                          np.isfinite(m["slope"]) and
                          np.isfinite(m["rms"]) and m["rms"] > 0.0]
    result = {"slope": np.nan, "uncertainty": np.nan, "rms": np.nan,
              "n": sum(m["n"] for m in valid_measurements),
              "residual": {}}
    if len(valid_measurements) != 2:
        return result
    slope = np.nanmedian([m["slope"] for m in valid_measurements])
    for _ in range(30):
        numerator = denominator = 0.0
        for measurement in valid_measurements:
            x, y, sigma = measurement["x"], measurement["y"], measurement["rms"]
            residual = y - slope * x
            standardized = np.abs(residual) / max(sigma, 1e-12)
            weights = np.ones_like(residual)
            outlying = standardized > 1.345
            weights[outlying] = 1.345 / standardized[outlying]
            inverse_variance = 1.0 / max(sigma * sigma, 1e-24)
            numerator += float(np.sum(weights * x * y) * inverse_variance)
            denominator += float(np.sum(weights * x * x) * inverse_variance)
        if denominator <= 0.0:
            return result
        updated = numerator / denominator
        if np.isclose(updated, slope, rtol=1e-9, atol=1e-12):
            slope = updated
            break
        slope = updated
    residuals = []
    for measurement in valid_measurements:
        residual = measurement["y"] - slope * measurement["x"]
        result["residual"][measurement["band"]] = residual
        residuals.extend(residual.tolist())
    result["slope"] = float(slope)
    result["rms"] = robust_rms(np.asarray(residuals))
    weighted_denominator = sum(float(np.sum(m["x"] * m["x"]) / max(m["rms"] ** 2, 1e-24))
                               for m in valid_measurements)
    result["uncertainty"] = float(result["rms"] / np.sqrt(weighted_denominator)) \
        if weighted_denominator > 0.0 else np.nan
    return result


def classify_illumination_scalars(rows, band):
    """Flag low-leverage scalars relative to the current exposure population."""
    for exposure in sorted({row["exposure"] for row in rows}):
        exposure_rows = [row for row in rows if row["exposure"] == exposure]
        candidates = [row for row in exposure_rows if row["n_fibers_%s" % band] >= 10 and
                      np.isfinite(row["leverage_%s" % band]) and row["leverage_%s" % band] > 0.0]
        leverages = np.asarray([row["leverage_%s" % band] for row in candidates], dtype=float)
        leverage_floor = float(np.percentile(leverages, 10)) if leverages.size else np.inf
        for row in exposure_rows:
            n = row["n_fibers_%s" % band]
            leverage = row["leverage_%s" % band]
            fit = row["s_%s_raw" % band]
            if not np.isfinite(fit) or n < 3:
                status = "qa_problem"
            elif row["n_good_amps"] < len(AMPS):
                status = "partially_masked"
            elif n < 10 or not np.isfinite(leverage) or leverage < leverage_floor:
                status = "low_leverage"
            else:
                status = "well_constrained"
            row["well_constrained_%s" % band] = status == "well_constrained"
            row["status_%s" % band] = status


def normalize_illumination_scalars(rows, exposures):
    """Keep ON/OFF's delta-fixed gauge; only record a common-fit gauge."""
    normalizers = {}
    for exposure in exposures:
        values = np.asarray([row["s_common_raw"] for row in rows
                             if row["exposure"] == exposure and
                             row["well_constrained_common"] and
                             np.isfinite(row["s_common_raw"])], dtype=float)
        normalizers[exposure] = float(np.median(values)) if values.size else np.nan
        if np.isfinite(normalizers[exposure]):
            print("illumination normalization exposure %d: median s_common_raw=%.6g N=%d" %
                  (exposure, normalizers[exposure], values.size))
        else:
            print("illumination normalization exposure %d: unavailable" % exposure)
        for row in rows:
            if row["exposure"] != exposure:
                continue
            row["common_normalizer"] = normalizers[exposure]
            for name in ("ON", "OFF"):
                row["s_%s_normalized" % name] = row["s_%s_raw" % name]
            raw = row["s_common_raw"]
            row["s_common_normalized"] = (raw / normalizers[exposure]
                                           if np.isfinite(raw) and np.isfinite(normalizers[exposure])
                                           else np.nan)
    return normalizers


def fit_illumination_plane(rows, exposure):
    """Fit ``s=1+cx*x+cy*y`` robustly to one exposure's common scalars."""
    selected = [row for row in rows if row["exposure"] == exposure and
                row["well_constrained_common"] and
                np.isfinite(row["s_common_normalized"]) and
                np.isfinite(row["mean_RA"]) and np.isfinite(row["mean_Dec"])]
    result = {"exposure": exposure, "ra0": np.nan, "dec0": np.nan, "cx": np.nan,
              "cy": np.nan, "robust_RMS_before": np.nan, "robust_RMS_after": np.nan,
              "n_IFU_used": 0, "n_IFU_rejected": 0, "used": set()}
    if len(selected) < 3:
        return result
    ra0 = float(np.median([row["mean_RA"] for row in selected]))
    dec0 = float(np.median([row["mean_Dec"] for row in selected]))
    x = np.asarray([(row["mean_RA"] - ra0) * np.cos(np.deg2rad(dec0)) * 60.0
                    for row in selected], dtype=float)
    y = np.asarray([(row["mean_Dec"] - dec0) * 60.0 for row in selected], dtype=float)
    values = np.asarray([row["s_common_normalized"] - 1.0 for row in selected], dtype=float)
    design = np.column_stack((x, y))
    keep = np.ones(values.size, dtype=bool)
    for _ in range(8):
        work_design, work_values = design[keep], values[keep]
        if work_values.size < 3:
            break
        coefficients = np.linalg.lstsq(work_design, work_values, rcond=None)[0]
        for _ in range(20):
            residual = work_values - work_design @ coefficients
            scale = max(robust_scale(residual), 1e-12)
            standardized = np.abs(residual) / scale
            weights = np.ones_like(residual)
            outlying = standardized > 1.345
            weights[outlying] = 1.345 / standardized[outlying]
            sqrt_w = np.sqrt(weights)
            weighted_design = work_design * sqrt_w[:, None]
            weighted_values = work_values * sqrt_w
            updated = np.linalg.lstsq(weighted_design, weighted_values, rcond=None)[0]
            if np.allclose(updated, coefficients, rtol=1e-9, atol=1e-12):
                coefficients = updated
                break
            coefficients = updated
        residual_all = values - design @ coefficients
        threshold = max(4.0 * robust_scale(residual_all[keep]), 1e-12)
        updated_keep = np.abs(residual_all) <= threshold
        if updated_keep.sum() < 3:
            break
        if np.array_equal(updated_keep, keep):
            break
        keep = updated_keep
    if keep.sum() < 3:
        return result
    work_design, work_values = design[keep], values[keep]
    coefficients = np.linalg.lstsq(work_design, work_values, rcond=None)[0]
    for _ in range(30):
        residual = work_values - work_design @ coefficients
        scale = max(robust_scale(residual), 1e-12)
        standardized = np.abs(residual) / scale
        weights = np.ones_like(residual)
        outlying = standardized > 1.345
        weights[outlying] = 1.345 / standardized[outlying]
        sqrt_w = np.sqrt(weights)
        updated = np.linalg.lstsq(work_design * sqrt_w[:, None],
                                   work_values * sqrt_w, rcond=None)[0]
        if np.allclose(updated, coefficients, rtol=1e-9, atol=1e-12):
            coefficients = updated
            break
        coefficients = updated
    residual_all = values - design @ coefficients
    result.update({"ra0": ra0, "dec0": dec0, "cx": float(coefficients[0]),
                   "cy": float(coefficients[1]),
                   "robust_RMS_before": robust_rms(values),
                   "robust_RMS_after": robust_rms(residual_all[keep]),
                   "n_IFU_used": int(keep.sum()),
                   "n_IFU_rejected": int((~keep).sum()),
                   "used": {selected[i]["ifu_key"] for i in np.flatnonzero(keep)}})
    for i, row in enumerate(selected):
        row["plane_used"] = bool(keep[i])
        row["plane_model"] = (1.0 + coefficients[0] * x[i] + coefficients[1] * y[i])
        row["plane_residual"] = row["s_common_normalized"] - row["plane_model"]
    return result


def build_illumination_scalars(datasets, groups, globals_, alpha, f, good_groups,
                              exposures, delta_by_band):
    """Build physical-IFU ON/OFF measurements after the final g/z and alpha."""
    by_dataset = {(d["exposure"], d["band"]): d for d in datasets}
    ifus = {}
    for index, group in enumerate(groups):
        if group["exposure"] in exposures:
            key = (group["exposure"], group["specid"], group["ifuslot"], group["ifuid"])
            ifus.setdefault(key, []).append(index)

    leverage_qa = {}
    for exposure in exposures:
        for band in ("ON", "OFF"):
            dataset = by_dataset[(exposure, band)]
            allowed = np.zeros(dataset["V"].size, dtype=bool)
            for key, group_indices in ifus.items():
                if key[0] != exposure:
                    continue
                for index in group_indices:
                    if good_groups[index]:
                        allowed[groups[index]["indices"]] = True
            fit = globals_[(exposure, band)]
            leverage_qa[(exposure, band)] = source_sky_leverage(
                fit["g"] * dataset["I"][allowed & dataset["valid"]],
                dataset["B_sky"][allowed & dataset["valid"]])

    rows = []
    for ifu_key, group_indices in sorted(ifus.items()):
        exposure, specid, ifuslot, ifuid = ifu_key
        good_indices = [index for index in group_indices if good_groups[index]]
        row = {"exposure": exposure, "SPECID": specid, "IFUSLOT": ifuslot,
               "IFUID": ifuid, "ifu_key": ifu_key, "n_good_amps": len(good_indices),
               "mean_RA": float(np.mean([groups[i]["mean_RA"] for i in group_indices])),
               "mean_Dec": float(np.mean([groups[i]["mean_Dec"] for i in group_indices]))}
        for band in ("ON", "OFF"):
            dataset = by_dataset[(exposure, band)]
            fit = globals_[(exposure, band)]
            x_values, y_values = [], []
            for index in good_indices:
                indices = groups[index]["indices"]
                selected = indices[dataset["valid"][indices]]
                if selected.size:
                    x_values.extend((fit["g"] * dataset["I"][selected] +
                                     dataset["B_sky"][selected]).tolist())
                    y_values.extend((dataset["V_total_corrected"][selected] -
                                     delta_by_band[(exposure, band)]).tolist())
            x_values, y_values = np.asarray(x_values), np.asarray(y_values)
            measurement = measure_ifu_scalar(x_values, y_values)
            measurement["band"] = band
            # Reconstruct the two predictor components alongside the fitted
            # arrays so the leverage is explicitly source+sky, not source-only.
            source_values, sky_values = [], []
            for index in good_indices:
                indices = groups[index]["indices"]
                selected = indices[dataset["valid"][indices]]
                if selected.size:
                    source_values.extend((fit["g"] * dataset["I"][selected]).tolist())
                    sky_values.extend(dataset["B_sky"][selected].tolist())
            predictor = source_sky_leverage(np.asarray(source_values), np.asarray(sky_values))
            measurement.update({"leverage": predictor["leverage"],
                                "median_source": predictor["median_source"],
                                "median_sky": predictor["median_sky"],
                                "median_total": predictor["median_total"],
                                "sky_fraction": predictor["median_sky_fraction"],
                                "p10_x": float(np.percentile(predictor["total"], 10))
                                if predictor["total"].size else np.nan,
                                "p90_x": float(np.percentile(predictor["total"], 90))
                                if predictor["total"].size else np.nan,
                                "robust_scatter_x": robust_scale(predictor["total"])})
            row.update({"n_fibers_%s" % band: measurement["n"],
                        "s_%s_raw" % band: measurement["slope"],
                        "scalar_uncertainty_%s" % band: measurement["uncertainty"],
                        "scalar_RMS_%s" % band: measurement["rms"],
                        "leverage_%s" % band: measurement["leverage"],
                        "median_source_%s" % band: measurement["median_source"],
                        "median_sky_%s" % band: measurement["median_sky"],
                        "median_total_%s" % band: measurement["median_total"],
                        "sky_fraction_%s" % band: measurement["sky_fraction"],
                        "median_x_%s" % band: measurement["median_x"],
                        "p10_x_%s" % band: measurement["p10_x"],
                        "p90_x_%s" % band: measurement["p90_x"],
                        "robust_scatter_x_%s" % band: measurement["robust_scatter_x"]})
        rows.append(row)

    for band in ("ON", "OFF"):
        classify_illumination_scalars(rows, band)
    for row in rows:
        if row["well_constrained_ON"] and row["well_constrained_OFF"]:
            common = measure_common_ifu_scalar(
                _row_measurements(row, by_dataset, groups, good_groups, globals_, delta_by_band))
        else:
            common = {"slope": np.nan, "uncertainty": np.nan, "rms": np.nan}
        row["s_common_raw"] = common["slope"]
        row["scalar_uncertainty_common"] = common["uncertainty"]
        row["scalar_RMS_common"] = common["rms"]
        row["ON_OFF_difference"] = row["s_ON_raw"] - row["s_OFF_raw"] \
            if np.isfinite(row["s_ON_raw"]) and np.isfinite(row["s_OFF_raw"]) else np.nan
        row["ON_OFF_fractional_difference"] = (row["ON_OFF_difference"] /
                                                row["s_common_raw"]
                                                if np.isfinite(row["ON_OFF_difference"]) and
                                                np.isfinite(row["s_common_raw"]) and
                                                row["s_common_raw"] != 0.0 else np.nan)
        row["well_constrained_common"] = bool(np.isfinite(common["slope"]))
        row["status_common"] = ("well_constrained" if row["well_constrained_common"]
                                 else "unavailable")
    return rows, leverage_qa


def _row_measurements(row, by_dataset, groups, good_groups, globals_, delta_by_band):
    """Rebuild band arrays for the common fit from a scalar row identity."""
    exposure = row["exposure"]
    group_indices = [i for i, group in enumerate(groups)
                     if (group["exposure"], group["specid"], group["ifuslot"], group["ifuid"])
                     == row["ifu_key"] and good_groups[i]]
    measurements = []
    for band in ("ON", "OFF"):
        dataset = by_dataset[(exposure, band)]
        fit = globals_[(exposure, band)]
        xs, ys = [], []
        for index in group_indices:
            indices = groups[index]["indices"]
            selected = indices[dataset["valid"][indices]]
            xs.extend((fit["g"] * dataset["I"][selected] + dataset["B_sky"][selected]).tolist())
            ys.extend((dataset["V_total_corrected"][selected] -
                       delta_by_band[(exposure, band)]).tolist())
        measurement = measure_ifu_scalar(np.asarray(xs), np.asarray(ys))
        measurement["band"] = band
        measurements.append(measurement)
    return measurements


def solve_illumination_delta(datasets, groups, globals_, alpha, f, good_groups,
                             exposures, exposure, band):
    """Solve the one global ``delta_e,b`` fixed by median(s_IFU)=1.

    The root is deliberately outside the IFU fits: for each trial delta the
    existing robust zero-intercept slope is refit, and only the population
    median of well-constrained physical-IFU slopes enters the gauge.
    """
    delta_by_band = {(e, b): globals_[(e, b)]["z"]
                     for e in exposures for b in ("ON", "OFF")}

    def evaluate(delta):
        trial = dict(delta_by_band)
        trial[(exposure, band)] = float(delta)
        trial_rows, _ = build_illumination_scalars(
            datasets, groups, globals_, alpha, f, good_groups, [exposure], trial)
        values = np.asarray([row["s_%s_raw" % band] for row in trial_rows
                             if row["well_constrained_%s" % band] and
                             np.isfinite(row["s_%s_raw" % band])], dtype=float)
        if not values.size:
            raise ValueError("no well-constrained physical IFU slopes for exposure %d %s" %
                             (exposure, band))
        return float(np.median(values)), int(values.size)

    dataset = next(d for d in datasets if d["exposure"] == exposure and d["band"] == band)
    allowed = np.zeros(dataset["V"].size, dtype=bool)
    for index, group in enumerate(groups):
        if group["exposure"] == exposure and good_groups[index]:
            allowed[group["indices"]] = True
    fit = globals_[(exposure, band)]
    source = fit["g"] * dataset["I"][allowed & dataset["valid"]]
    sky = dataset["B_sky"][allowed & dataset["valid"]]
    total = source + sky
    t_minus_x = dataset["V_total_corrected"][allowed & dataset["valid"]] - total
    t_minus_x = t_minus_x[np.isfinite(t_minus_x)]
    if t_minus_x.size < 3:
        raise ValueError("exposure %d %s has insufficient finite T-X values for delta bracket" %
                         (exposure, band))
    center = robust_location(t_minus_x)
    scale = robust_scale(t_minus_x)
    anchor = fit["z"] if np.isfinite(fit["z"]) else center
    width = max(5.0 * (scale if np.isfinite(scale) else 0.0),
                abs(center - anchor), 1e-8 * max(abs(center), abs(anchor), 1.0))
    low, high = anchor - width, anchor + width
    median_before, n_before = evaluate(anchor)
    f_low, f_high = evaluate(low)[0] - 1.0, evaluate(high)[0] - 1.0
    for _ in range(24):
        if f_low == 0.0:
            root = low
            break
        if f_high == 0.0:
            root = high
            break
        if f_low * f_high < 0.0:
            root = brentq(lambda value: evaluate(value)[0] - 1.0,
                          low, high, xtol=1e-10, rtol=1e-10)
            break
        width *= 2.0
        low, high = anchor - width, anchor + width
        f_low, f_high = evaluate(low)[0] - 1.0, evaluate(high)[0] - 1.0
    else:
        raise ValueError("could not bracket delta_illumination for exposure %d %s "
                         "around z_source_fit=%.6g" % (exposure, band, fit["z"]))
    median_final, n_final = evaluate(root)
    print("exposure %d %s: z_source_fit=%+.6g delta_illumination=%+.6g "
          "delta-z=%+.6g N=%d median_s_before=%.6g median_s_final=%.6g" %
          (exposure, band, fit["z"], root, root - fit["z"], n_final,
           median_before, median_final))
    return {"exposure": exposure, "band": band, "z_source_fit": fit["z"],
            "delta_illumination": float(root), "delta_minus_z": float(root - fit["z"]),
            "n_IFU_used_for_delta": n_final, "median_s_before": median_before,
            "median_s_final": median_final}


def _plot_symmetric(axis, ra, dec, values, scale, **kwargs):
    finite = np.isfinite(ra) & np.isfinite(dec) & np.isfinite(values)
    if finite.any():
        axis.scatter(ra[finite], dec[finite], c=values[finite], vmin=-scale, vmax=scale,
                     cmap="coolwarm", s=42, edgecolors="none", **kwargs)
    return finite


def plot_illumination_source_sky_leverage(leverage_qa, output_path):
    exposures = sorted({key[0] for key in leverage_qa})
    fig, axes = plt.subplots(len(exposures) * 2, 2, figsize=(12, 4.2 * len(exposures)), squeeze=False)
    for row_index, exposure in enumerate(exposures):
        for band_index, band in enumerate(("ON", "OFF")):
            values = leverage_qa[(exposure, band)]
            scatter, histogram = axes[2 * row_index, band_index], axes[2 * row_index + 1, band_index]
            scatter.scatter(values["source"], values["sky"], s=5, alpha=.3, rasterized=True)
            scatter.axline((0, 0), slope=1, color="k", ls=":", lw=.8)
            scatter.set_title("exposure %d %s: sky versus source predictor" % (exposure, band))
            scatter.set_xlabel("g I"); scatter.set_ylabel("B_sky")
            fractions = values["sky_fraction"][np.isfinite(values["sky_fraction"])]
            if fractions.size:
                histogram.hist(fractions, bins=30, color="tab:purple", alpha=.75)
                histogram.axvline(np.median(fractions), color="k", lw=1.2)
            histogram.set_title("median source=%.4g, sky=%.4g, total=%.4g; N=%d" %
                                (values["median_source"], values["median_sky"],
                                 values["median_total"], values["n"]))
            histogram.set_xlabel("B_sky / (g I + B_sky)"); histogram.set_ylabel("fibers")
            for axis in (scatter, histogram): axis.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(output_path, dpi=170); plt.close(fig)


def plot_illumination_ifu_on_vs_off(rows, output_path):
    exposures = sorted({row["exposure"] for row in rows})
    fig, axes = plt.subplots(1, len(exposures), figsize=(5.2 * len(exposures), 4.8), squeeze=False)
    for column, exposure in enumerate(exposures):
        axis = axes[0, column]
        subset = [row for row in rows if row["exposure"] == exposure]
        for status, color, label in (("well", "tab:green", "well-constrained"),
                                     ("low", "tab:orange", "low-leverage"),
                                     ("qa", "tab:red", "partially masked / QA")):
            selected = []
            for row in subset:
                if status == "well" and row["well_constrained_ON"] and row["well_constrained_OFF"]:
                    selected.append(row)
                elif status == "low" and not (row["well_constrained_ON"] and row["well_constrained_OFF"]) and \
                        "low_leverage" in (row["status_ON"], row["status_OFF"]):
                    selected.append(row)
                elif status == "qa" and ("partially_masked" in (row["status_ON"], row["status_OFF"]) or
                                         "qa_problem" in (row["status_ON"], row["status_OFF"])):
                    selected.append(row)
            if selected:
                axis.scatter([r["s_ON_normalized"] for r in selected],
                             [r["s_OFF_normalized"] for r in selected], s=28, alpha=.8,
                             color=color, label=label)
        good = [row for row in subset if row["well_constrained_ON"] and row["well_constrained_OFF"] and
                np.isfinite(row["s_ON_normalized"]) and np.isfinite(row["s_OFF_normalized"])]
        if len(good) >= 3:
            x = np.asarray([row["s_ON_normalized"] for row in good]); y = np.asarray([row["s_OFF_normalized"] for row in good])
            fit = fit_line(x, y); grid = np.linspace(min(x.min(), y.min()), max(x.max(), y.max()), 100)
            axis.plot(grid, grid, "k:", label="identity")
            axis.plot(grid, fit["g"] * grid + fit["z"], "b-", lw=1.2, label="robust fit")
            corr = np.corrcoef(x, y)[0, 1] if x.size > 1 else np.nan
            scatter = robust_rms(y - (fit["g"] * x + fit["z"]))
            axis.text(.03, .97, "scatter=%.4g\nr=%.4g\nN=%d" % (scatter, corr, len(good)),
                      transform=axis.transAxes, va="top", fontsize=8)
            discrepant = sorted(good, key=lambda r: abs(r["s_ON_normalized"] - r["s_OFF_normalized"]), reverse=True)[:3]
            for row in discrepant:
                axis.annotate(str(row["IFUSLOT"]), (row["s_ON_normalized"], row["s_OFF_normalized"]), fontsize=7)
        axis.axline((1, 1), slope=1, color="k", ls=":", lw=.8)
        axis.set_title("exposure %d" % exposure); axis.set_xlabel("normalized s_ON"); axis.set_ylabel("normalized s_OFF")
        axis.grid(alpha=.2); axis.legend(fontsize=7)
    fig.suptitle("Physical IFU illumination: ON versus OFF"); fig.tight_layout(rect=(0, 0, 1, .95))
    fig.savefig(output_path, dpi=170); plt.close(fig)


def plot_illumination_ifu_scalar_maps(rows, output_path):
    exposures = sorted({row["exposure"] for row in rows})
    values = [row["s_%s_normalized" % band] - 1.0 for row in rows for band in ("ON", "OFF", "common")
              if np.isfinite(row["s_%s_normalized" % band])]
    scale = max(float(np.percentile(np.abs(values), 95)) if values else .01, .01)
    fig, axes = plt.subplots(len(exposures), 3, figsize=(15, 4.3 * len(exposures)), squeeze=False)
    for i, exposure in enumerate(exposures):
        subset = [row for row in rows if row["exposure"] == exposure]
        ra = np.asarray([row["mean_RA"] for row in subset]); dec = np.asarray([row["mean_Dec"] for row in subset])
        for j, band in enumerate(("ON", "OFF", "common")):
            axis = axes[i, j]; values = np.asarray([row["s_%s_normalized" % band] - 1.0 for row in subset])
            _plot_symmetric(axis, ra, dec, values, scale)
            bad = ~np.isfinite(values) | ~np.asarray([row["well_constrained_%s" % band] if band != "common" else
                                                       row["well_constrained_common"] for row in subset])
            if bad.any(): axis.scatter(ra[bad], dec[bad], marker="x", color="k", s=28, label="poorly constrained")
            axis.set_title("exposure %d %s: normalized s - 1" % (exposure, band)); axis.set_xlabel("RA"); axis.set_ylabel("Dec")
            axis.grid(alpha=.2)
    fig.suptitle("Direct physical-IFU illumination scalar maps"); fig.tight_layout(rect=(0, 0, 1, .95))
    fig.savefig(output_path, dpi=170); plt.close(fig)


def plot_illumination_plane_qa(rows, planes, output_path):
    exposures = sorted(planes)
    measured = [abs(row["s_common_normalized"] - 1.0) for row in rows if np.isfinite(row["s_common_normalized"])]
    residuals = [abs(row.get("plane_residual", np.nan)) for row in rows if np.isfinite(row.get("plane_residual", np.nan))]
    measured_scale = max(float(np.percentile(measured, 95)) if measured else .01, .01)
    residual_scale = max(float(np.percentile(residuals, 95)) if residuals else .01, .01)
    fig, axes = plt.subplots(len(exposures), 3, figsize=(15, 4.3 * len(exposures)), squeeze=False)
    for i, exposure in enumerate(exposures):
        subset = [row for row in rows if row["exposure"] == exposure and np.isfinite(row["s_common_normalized"])]
        ra = np.asarray([row["mean_RA"] for row in subset]); dec = np.asarray([row["mean_Dec"] for row in subset])
        values = np.asarray([row["s_common_normalized"] - 1.0 for row in subset])
        modeled = np.asarray([row.get("plane_model", np.nan) - 1.0 for row in subset])
        residual = np.asarray([row.get("plane_residual", np.nan) for row in subset])
        for axis, data, scale, title in zip(axes[i], (values, modeled, residual),
                                            (measured_scale, measured_scale, residual_scale),
                                            ("measured", "plane", "residual")):
            _plot_symmetric(axis, ra, dec, data, scale)
            rejected = np.asarray([not row.get("plane_used", False) for row in subset])
            if rejected.any(): axis.scatter(ra[rejected], dec[rejected], marker="x", color="k", s=28)
            axis.set_title("exposure %d %s" % (exposure, title)); axis.set_xlabel("RA"); axis.set_ylabel("Dec"); axis.grid(alpha=.2)
        plane = planes[exposure]
        axes[i, 1].text(.03, .97, "cx=%.5g\ncy=%.5g\nRMS %.4g -> %.4g\nused/rejected=%d/%d" %
                        (plane["cx"], plane["cy"], plane["robust_RMS_before"], plane["robust_RMS_after"],
                         plane["n_IFU_used"], plane["n_IFU_rejected"]), transform=axes[i, 1].transAxes, va="top", fontsize=8)
    fig.suptitle("Two-parameter illumination plane QA"); fig.tight_layout(rect=(0, 0, 1, .95))
    fig.savefig(output_path, dpi=170); plt.close(fig)


def plot_illumination_ifu_exposure_comparison(rows, output_path):
    exposures = sorted({row["exposure"] for row in rows})
    grouped = {}
    for row in rows:
        grouped.setdefault((row["SPECID"], row["IFUSLOT"], row["IFUID"]), []).append(row)
    fig, axis = plt.subplots(figsize=(10, 6))
    for key, measurements in sorted(grouped.items()):
        measurements = sorted(measurements, key=lambda row: row["exposure"])
        x = [row["exposure"] for row in measurements]
        y = [row["s_common_normalized"] for row in measurements]
        axis.plot(x, y, "-", color="0.65", alpha=.35, lw=.6)
        axis.scatter(x, y, s=16, color=["tab:green" if row["well_constrained_common"] else "tab:orange" for row in measurements])
    correlations = []
    for left, right in zip(exposures[:-1], exposures[1:]):
        pairs = {(row["SPECID"], row["IFUSLOT"], row["IFUID"]): row["s_common_normalized"]
                 for row in rows if row["exposure"] == left and row["well_constrained_common"] and np.isfinite(row["s_common_normalized"])}
        for row in rows:
            key = (row["SPECID"], row["IFUSLOT"], row["IFUID"])
            if row["exposure"] == right and key in pairs and row["well_constrained_common"]:
                correlations.append((left, right, pairs[key], row["s_common_normalized"]))
    text_lines = []
    for left, right in zip(exposures[:-1], exposures[1:]):
        pair = [(a, b) for l, r, a, b in correlations if l == left and r == right]
        corr = np.corrcoef(np.asarray(pair).T)[0, 1] if len(pair) >= 3 else np.nan
        text_lines.append("%d-%d r=%.4g N=%d" % (left, right, corr, len(pair)))
    axis.text(.03, .97, "\n".join(text_lines) if text_lines else "no exposure pairs",
              transform=axis.transAxes, va="top", fontsize=9)
    axis.set_xticks(exposures); axis.set_xlabel("exposure"); axis.set_ylabel("normalized s_common")
    axis.set_title("Same physical IFUs across exposures"); axis.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(output_path, dpi=170); plt.close(fig)


def plot_illumination_model_closure(datasets, groups, globals_, delta_by_band,
                                    illumination_rows, good_groups, output_path):
    """Check the equivalent sky-subtracted model without changing spectra.

    With ``V0 = V - K*alpha*f(q)`` and ``T = V0 + B_sky``, the model is

        T = s_IFU * [g*I_object + B_sky] + delta

    or, in the stored sky-subtracted basis,

        V0 = s_IFU*g*I_object + (s_IFU - 1)*B_sky + delta.
    """
    by_dataset = {(d["exposure"], d["band"]): d for d in datasets}
    by_ifu = {row["ifu_key"]: row for row in illumination_rows}
    exposures = sorted({row["exposure"] for row in illumination_rows})
    fig, axes = plt.subplots(len(exposures), 2, figsize=(13, 4.0 * len(exposures)), squeeze=False)
    for i, exposure in enumerate(exposures):
        for j, band in enumerate(("ON", "OFF")):
            axis = axes[i, j]
            dataset = by_dataset[(exposure, band)]
            fit = globals_[(exposure, band)]
            before, after, slots = [], [], []
            for key, row in by_ifu.items():
                if key[0] != exposure:
                    continue
                group_indices = [index for index, group in enumerate(groups)
                                 if (group["exposure"], group["specid"], group["ifuslot"], group["ifuid"]) == key
                                 and good_groups[index]]
                v0_residual_before, v0_residual_after = [], []
                s = row["s_%s_raw" % band]
                for index in group_indices:
                    indices = groups[index]["indices"]
                    selected = indices[dataset["valid"][indices]]
                    if not selected.size:
                        continue
                    v0 = dataset["V0"][selected]
                    source = fit["g"] * dataset["I"][selected]
                    sky = dataset["B_sky"][selected]
                    delta = delta_by_band[(exposure, band)]
                    no_illumination = source + delta
                    with_illumination = s * source + (s - 1.0) * sky + delta
                    v0_residual_before.extend((v0 - no_illumination).tolist())
                    if np.isfinite(s):
                        v0_residual_after.extend((v0 - with_illumination).tolist())
                if v0_residual_before:
                    slots.append(row["IFUSLOT"])
                    before.append(robust_location(v0_residual_before))
                    after.append(robust_location(v0_residual_after) if v0_residual_after else np.nan)
            order = np.argsort(slots)
            slots = np.asarray(slots)[order]
            before = np.asarray(before)[order]
            after = np.asarray(after)[order]
            axis.axhline(0.0, color="k", lw=.8)
            axis.plot(slots, before, "o-", ms=3, lw=.6, color="tab:orange", label="before: s=1")
            axis.plot(slots, after, "o-", ms=3, lw=.6, color="tab:blue", label="after: measured s_IFU")
            axis.set_title("exposure %d %s" % (exposure, band)); axis.set_xlabel("IFUSLOT")
            axis.set_ylabel("robust V0 - V0_pred"); axis.grid(alpha=.2); axis.legend(fontsize=8)
            axis.text(.03, .97, "RMS before/after=%.4g/%.4g" %
                      (robust_rms(before), robust_rms(after)), transform=axis.transAxes,
                      va="top", fontsize=8)
    fig.suptitle("Sky-subtracted illumination model closure"); fig.tight_layout(rect=(0, 0, 1, .95))
    fig.savefig(output_path, dpi=170); plt.close(fig)


def fit_line(x, y, fixed_g=None):
    """Fit ``y = g*x + z`` robustly, optionally holding ``g`` fixed."""
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = np.asarray(x)[valid], np.asarray(y)[valid]
    if x.size < 3:
        return {"g": np.nan, "z": np.nan}
    if fixed_g is not None:
        fixed_g = float(fixed_g)
        if not np.isfinite(fixed_g):
            return {"g": np.nan, "z": np.nan}
        residual = y - fixed_g * x
        z0 = robust_location(residual)
        if not np.isfinite(z0):
            return {"g": fixed_g, "z": np.nan}
        scale = max(robust_scale(residual), 1e-12)
        fit = least_squares(lambda p: residual - p[0], [z0],
                            loss="soft_l1", f_scale=scale, x_scale="jac")
        return {"g": fixed_g, "z": float(fit.x[0])}
    p0 = [np.dot(x, y) / np.dot(x, x) if np.dot(x, x) else 0.0,
          float(np.median(y))]
    scale = max(robust_scale(y), 1e-12)
    fit = least_squares(lambda p: y - (p[0] * x + p[1]), p0,
                        loss="soft_l1", f_scale=scale, x_scale="jac")
    return {"g": float(fit.x[0]), "z": float(fit.x[1])}


def summarize_amp(group, dataset, v0):
    indices = group["indices"]
    valid = dataset["valid"][indices]
    if not valid.any():
        return {"Y": np.nan, "I": np.nan, "ra": np.nan, "dec": np.nan}
    selected = indices[valid]
    return {"Y": robust_location(v0[selected]),
            "I": robust_location(dataset["I"][selected]),
            "ra": float(np.nanmean(dataset["ra"][selected])),
            "dec": float(np.nanmean(dataset["dec"][selected]))}


def clip_amplifiers(groups, qa_rows, initial_good):
    values_m = np.asarray([v for row in qa_rows for v in
                           (row["median_ON"], row["median_OFF"]) if np.isfinite(v)])
    values_r = np.asarray([v for row in qa_rows for v in
                           (row["rms_ON"], row["rms_OFF"]) if np.isfinite(v)])
    med0 = np.median(values_m) if values_m.size else 0.0
    med_s = max(robust_scale(values_m), 1e-8)
    rms0 = np.median(values_r) if values_r.size else 0.0
    rms_s = max(robust_scale(values_r), 1e-8)
    good = np.array(initial_good, dtype=bool)
    for row in qa_rows:
        med_bad = any(np.isfinite(v) and abs(v - med0) > 4.0 * med_s
                      for v in (row["median_ON"], row["median_OFF"]))
        rms_bad = any(np.isfinite(v) and v > rms0 + 4.0 * rms_s
                      for v in (row["rms_ON"], row["rms_OFF"]))
        good[row["group_index"]] &= not (med_bad or rms_bad)
    for row in qa_rows:
        row["fit_good"] = bool(good[row["group_index"]])
    return good


def broad_stage(datasets, groups, f, alpha, initial_good, previous_good,
                fixed_g_by_band=None):
    """Perform one fixed-additive, g/z, and amplifier-clipping stage.

    The illumination term is intentionally not a fitted quantity in this
    diagnostic.  ``dataset["F"]`` is set to one so all residuals remain in the
    unmodified field and the same downstream diagnostic bookkeeping can be
    used without introducing a FoV model.
    """
    for dataset in datasets:
        corrected = np.zeros(dataset["V"].shape, dtype=float)
        for group in groups:
            indices = group["indices"]
            a = alpha[(dataset["exposure"], group["amp"])]
            corrected[indices] = dataset["K"] * a * f[dataset["q"][indices]]
        dataset["V0"] = dataset["V"] - corrected

    globals_, summaries = {}, {}
    for dataset in datasets:
        exposure, band = dataset["exposure"], dataset["band"]
        rows = []
        for index, group in enumerate(groups):
            summary = summarize_amp(group, dataset, dataset["V0"])
            summary.update({"identity": group["identity"],
                            "fit_good": bool(previous_good[index]),
                            "group_index": index})
            rows.append(summary)
            summaries[(group["identity"], band)] = summary
        fixed_g = (fixed_g_by_band.get(band) if fixed_g_by_band is not None
                   else None)
        fit = fit_line(np.asarray([row["I"] for row in rows]),
                       np.asarray([row["Y"] for row in rows]), fixed_g=fixed_g)
        globals_[(exposure, band)] = fit
        F = np.ones(dataset["V"].shape, dtype=float)
        dataset["F"] = F
        before = (dataset["V0"] - (fit["g"] * dataset["I"] + fit["z"])) / dataset["K"]
        after = (F * dataset["V0"] - (fit["g"] * dataset["I"] + fit["z"])) / dataset["K"]
        dataset["before"], dataset["after"] = before, after
        for index, group in enumerate(groups):
            selected = dataset["valid"] & np.isin(np.arange(dataset["V"].size), group["indices"])
            summaries[(group["identity"], band)].update({
                "before": robust_location(before[selected]),
                "after": robust_location(after[selected]),
            })

    qa_rows = []
    for index, group in enumerate(groups):
        if group["exposure"] not in {dataset["exposure"] for dataset in datasets}:
            continue
        values = {band: next(d for d in datasets
                             if d["exposure"] == group["exposure"] and d["band"] == band)
                  for band in ("ON", "OFF")}
        stats = {}
        for band, dataset in values.items():
            selected = dataset["valid"] & np.isin(np.arange(dataset["V"].size), group["indices"])
            stats[band] = {
                "median": robust_location(dataset["after"][selected]),
                "rms": robust_rms(dataset["after"][selected]),
            }
        qa_rows.append({
            "identity": group["identity"], "exposure": group["exposure"],
            "group_index": index,
            "SPECID": group["specid"], "IFUSLOT": group["ifuslot"],
            "IFUID": group["ifuid"], "AMP": group["amp"],
            "fit_good": bool(previous_good[index]),
            "median_ON": stats["ON"]["median"], "rms_ON": stats["ON"]["rms"],
            "median_OFF": stats["OFF"]["median"], "rms_OFF": stats["OFF"]["rms"],
            "mean_RA": group.get("mean_RA", np.nan),
            "mean_Dec": group.get("mean_Dec", np.nan),
        })
    # The group coordinates are filled by the caller because they are common
    # to both bands and are not part of the H5 grouping metadata.
    good = clip_amplifiers(groups, qa_rows, initial_good)
    return globals_, qa_rows, summaries, good


def fit_bounded_alphas(datasets, groups, globals_, f, good_groups,
                       current, exposures):
    alpha = dict(current)
    for exposure in exposures:
        for amp in AMPS:
            x_all, y_all = [], []
            selected_group_ids = [i for i, group in enumerate(groups)
                                  if group["exposure"] == exposure and group["amp"] == amp and
                                  good_groups[i]]
            for dataset in datasets:
                if dataset["exposure"] != exposure:
                    continue
                fit = globals_[(exposure, dataset["band"])]
                F = dataset["F"]
                for group_id in selected_group_ids:
                    indices = groups[group_id]["indices"]
                    valid = dataset["valid"][indices]
                    selected = indices[valid]
                    if selected.size:
                        x_all.extend(f[dataset["q"][selected]].tolist())
                        y_all.extend(((dataset["V"][selected] -
                                       (fit["g"] * dataset["I"][selected] + fit["z"]) /
                                       F[selected]) / dataset["K"]).tolist())
            if len(x_all) < 6:
                continue
            x, y = np.asarray(x_all), np.asarray(y_all)
            scale = max(robust_scale(y), 1e-8)
            fit = least_squares(lambda p: y - p[0] * x, [alpha[(exposure, amp)]],
                                bounds=([ALPHA_LOW], [ALPHA_HIGH]), loss="soft_l1",
                                f_scale=scale, x_scale="jac")
            alpha[(exposure, amp)] = float(fit.x[0])
    return alpha


def write_csvs(output_dir, qa_rows, global_rows, alpha_rows):
    fields = ["exposure", "SPECID", "IFUSLOT", "IFUID", "AMP", "fit_good",
              "residual_median_ON", "residual_RMS_ON", "residual_median_OFF",
              "residual_RMS_OFF", "mean_RA", "mean_Dec"]
    with (output_dir / "hierarchical_amplifier_qa.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(qa_rows)
    fields = ["exposure", "band", "g", "z", "F_p0", "F_dRA", "F_dDec",
              "F_ra0", "F_dec0", "number_good_amps"]
    with (output_dir / "hierarchical_global_parameters.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(global_rows)
    fields = ["exposure", "AMP", "alpha_e_per_A", "at_lower_bound", "at_upper_bound"]
    with (output_dir / "hierarchical_alpha.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(alpha_rows)


def write_illumination_csvs(output_dir, rows, plane_rows, global_rows):
    fields = ["exposure", "SPECID", "IFUSLOT", "IFUID", "n_good_amps",
              "n_fibers_ON", "n_fibers_OFF", "s_ON_raw", "s_OFF_raw", "s_common_raw",
              "s_ON_normalized", "s_OFF_normalized", "s_common_normalized",
              "scalar_RMS_ON", "scalar_RMS_OFF", "scalar_RMS_common",
              "scalar_uncertainty_ON", "scalar_uncertainty_OFF", "scalar_uncertainty_common",
              "leverage_ON", "leverage_OFF", "well_constrained_ON", "well_constrained_OFF",
              "well_constrained_common", "status_ON", "status_OFF", "status_common",
              "common_normalizer",
              "ON_OFF_difference", "ON_OFF_fractional_difference",
              "median_source_ON", "median_source_OFF", "median_sky_ON", "median_sky_OFF",
              "median_total_ON", "median_total_OFF", "sky_fraction_ON", "sky_fraction_OFF",
              "median_x_ON", "p10_x_ON", "p90_x_ON", "robust_scatter_x_ON",
              "median_x_OFF", "p10_x_OFF", "p90_x_OFF", "robust_scatter_x_OFF",
              "mean_RA", "mean_Dec"]
    with (output_dir / "illumination_ifu_scalars.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
        writer.writerows({field: row.get(field, np.nan) for field in fields} for row in rows)
    fields = ["exposure", "cx", "cy", "ra0", "dec0", "robust_RMS_before",
              "robust_RMS_after", "n_IFU_used", "n_IFU_rejected"]
    with (output_dir / "illumination_plane_parameters.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
        writer.writerows({field: row.get(field, np.nan) for field in fields} for row in plane_rows)
    fields = ["exposure", "band", "external_image_background_per_pixel",
              "external_image_background_scatter", "external_image_background_npix",
              "external_image_background_annulus", "z_source_fit", "delta_illumination",
              "delta_minus_z", "n_IFU_used_for_delta", "median_s_final",
              "common_normalizer"]
    with (output_dir / "illumination_global_offsets.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
        writer.writerows({field: row.get(field, np.nan) for field in fields} for row in global_rows)


def plot_components(record, output_path):
    q = np.arange(N_FIBER_AMP); model = record["alpha"] * record["f"]
    d_on, d_off = record["D"]["ON"], record["D"]["OFF"]
    panels = [d_on, d_off, d_on - d_off, d_on - model, d_off - model]
    finite = np.concatenate([v[np.isfinite(v)] for v in panels if np.any(np.isfinite(v))])
    lo, hi = np.percentile(finite, [1, 99]) if finite.size else (-1, 1)
    span = max(hi - lo, 1e-6); ylim = (lo - .08 * span, hi + .08 * span)
    fig, axes = plt.subplots(4, 1, figsize=(12, 13), sharex=True)
    axes[0].plot(q, record["Y"]["ON"], ".-", ms=3, lw=.5, color="tab:blue", label="Y ON")
    axes[0].plot(q, record["M"]["ON"], "--", color="tab:blue", label="M ON")
    axes[0].plot(q, record["Y"]["OFF"], ".-", ms=3, lw=.5, color="tab:orange", label="Y OFF")
    axes[0].plot(q, record["M"]["OFF"], "--", color="tab:orange", label="M OFF")
    axes[0].set_title("Raw-equivalent absolute values [e-/A]"); axes[0].legend(ncol=2, fontsize=8)
    axes[1].plot(q, d_on, ".-", ms=3, lw=.5, color="tab:blue", label="D ON")
    axes[1].plot(q, d_off, ".-", ms=3, lw=.5, color="tab:orange", label="D OFF")
    axes[1].plot(q, model, "k-", lw=1.4, label="alpha*f(q)")
    axes[1].set_title("PRIMARY: external residual and fixed detector model"); axes[1].legend(fontsize=8)
    axes[2].plot(q, d_on - d_off, ".-", ms=3, lw=.5, color="tab:purple", label="D ON-D OFF")
    axes[2].set_title("ON/OFF difference"); axes[2].legend(fontsize=8)
    axes[3].plot(q, d_on - model, ".-", ms=3, lw=.5, color="tab:blue", label="D ON-alpha*f")
    axes[3].plot(q, d_off - model, ".-", ms=3, lw=.5, color="tab:orange", label="D OFF-alpha*f")
    axes[3].set_title("After fixed additive correction"); axes[3].legend(fontsize=8)
    for axis in axes:
        axis.axhline(0, color="k", lw=.7); axis.axvline(20, color="0.5", ls=":", lw=.7)
        axis.axvline(40, color="0.5", ls=":", lw=.7); axis.grid(alpha=.2)
    for axis in axes[1:]: axis.set_ylim(*ylim)
    axes[-1].set_xlabel("folded readout distance q")
    identity = record["identity"]
    axes[1].text(.99, .96, "fit_good=%s\nalpha=%+.4g\nON med/RMS=%+.4g/%+.4g\nOFF med/RMS=%+.4g/%+.4g" %
                 (record["fit_good"], record["alpha"], record["qa"]["median_ON"], record["qa"]["rms_ON"],
                  record["qa"]["median_OFF"], record["qa"]["rms_OFF"]),
                 transform=axes[1].transAxes, ha="right", va="top", fontsize=8)
    fig.suptitle("exp %d SPECID %d IFUSLOT %03d IFUID %d AMP %s; Raw-equivalent [e-/A]" % identity)
    fig.tight_layout(rect=(0, 0, 1, .96)); fig.savefig(output_path, dpi=160); plt.close(fig)


def plot_gallery(records, output_path):
    records = list(records.values()); ncols = 2; nrows = int(np.ceil(len(records) / ncols))
    all_values = [v[np.isfinite(v)] for record in records for v in
                  (record["D"]["ON"], record["D"]["OFF"], record["alpha"] * record["f"])
                  if np.any(np.isfinite(v))]
    values = np.concatenate(all_values) if all_values else np.asarray([])
    lo, hi = np.percentile(values, [1, 99]) if values.size else (-1, 1)
    span = max(hi - lo, 1e-6); ylim = (lo - .08 * span, hi + .08 * span)
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 3.2 * nrows), squeeze=False,
                             sharex=True, sharey=True)
    q = np.arange(N_FIBER_AMP)
    for i, record in enumerate(records):
        axis = axes[i // ncols, i % ncols]
        axis.plot(q, record["D"]["ON"], ".-", ms=2, lw=.45, color="tab:blue", label="D ON")
        axis.plot(q, record["D"]["OFF"], ".-", ms=2, lw=.45, color="tab:orange", label="D OFF")
        axis.plot(q, record["alpha"] * record["f"], "k-", lw=1.1, label="alpha*f")
        ident = record["identity"]
        axis.set_title("exp%d IFU%03d %s alpha=%+.3g good=%s" %
                       (ident[0], ident[2], ident[4], record["alpha"], record["fit_good"]), fontsize=9)
        axis.set_ylim(*ylim); axis.axhline(0, color="k", lw=.6)
        axis.axvline(20, color="0.5", ls=":", lw=.6); axis.axvline(40, color="0.5", ls=":", lw=.6)
        axis.grid(alpha=.2)
    for i in range(len(records), nrows * ncols): axes[i // ncols, i % ncols].set_visible(False)
    axes[0, 0].legend(fontsize=8); fig.suptitle("Hierarchical physical-amplifier component gallery [e-/A]")
    fig.tight_layout(rect=(0, 0, 1, .95)); fig.savefig(output_path, dpi=160); plt.close(fig)


def plot_external_image_psf_qa(images, output_path):
    """Plot the accepted external-image stellar FWHM measurements."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), squeeze=False)
    for axis, band in zip(axes[0], ("ON", "OFF")):
        image = images[band]
        psf = image["psf"]
        measurements = psf["accepted_fwhm_arcsec"]
        if measurements.size:
            bins = min(20, max(5, int(np.sqrt(measurements.size))))
            axis.hist(measurements, bins=bins, color="tab:blue", alpha=.75)
            axis.axvline(psf["median"], color="k", lw=1.5, label="median")
            axis.axvspan(psf["p16"], psf["p84"], color="tab:orange", alpha=.25,
                         label="p16--p84")
            axis.legend(fontsize=8)
        else:
            axis.text(.5, .5, "Header value; no stellar measurements", ha="center",
                      va="center", transform=axis.transAxes)
        axis.set_title("%s external image: %s" % (band, psf["source"]))
        axis.set_xlabel("stellar FWHM [arcsec]")
        axis.set_ylabel("accepted stars")
        axis.grid(alpha=.2)
        axis.text(.03, .97,
                  "candidates=%d\naccepted=%d\nmedian=%.4g\np16/p84=%s" %
                  (psf["candidate_count"], psf["accepted_count"], psf["fwhm_arcsec"],
                   "%.4g/%.4g" % (psf["p16"], psf["p84"])
                   if np.isfinite(psf["p16"]) else "n/a"),
                  transform=axis.transAxes, va="top", fontsize=8)
        axis.text(.03, .03, "pixel scale=%.6g arcsec/pix\nadopted FWHM=%.6g arcsec" %
                  (image["pixel_scale_arcsec"], psf["fwhm_arcsec"]),
                  transform=axis.transAxes, va="bottom", fontsize=8)
    fig.suptitle("External image PSF QA (PHOTFWHM excluded)")
    fig.tight_layout(rect=(0, 0, 1, .94)); fig.savefig(output_path, dpi=170); plt.close(fig)


def plot_external_fiber_sampling_comparison(comparison, output_path):
    """Compare legacy nearest-pixel values with exact aperture sums."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), squeeze=False)
    for col, band in enumerate(("ON", "OFF")):
        old, new = comparison[band]
        valid = np.isfinite(old) & np.isfinite(new)
        old, new = old[valid], new[valid]
        scatter_axis, residual_axis = axes[0, col], axes[1, col]
        if old.size >= 3:
            fit = fit_line(old, new)
            residual = new - (fit["g"] * old + fit["z"])
            scatter_axis.scatter(old, new, s=4, alpha=.25, rasterized=True)
            lo = min(np.nanmin(old), np.nanmin(new))
            hi = max(np.nanmax(old), np.nanmax(new))
            grid = np.linspace(lo, hi, 100)
            scatter_axis.plot(grid, grid, "k:", lw=.8, label="identity")
            scatter_axis.plot(grid, fit["g"] * grid + fit["z"], "r-", lw=1.2,
                              label="robust fit")
            residual_axis.scatter(new, residual, s=4, alpha=.25, rasterized=True)
            residual_axis.axhline(0.0, color="k", lw=.8)
            residual_axis.text(.03, .97, "slope=%.5g\nintercept=%.5g\nrobust RMS=%.5g\nN=%d" %
                               (fit["g"], fit["z"], robust_rms(residual), old.size),
                               transform=residual_axis.transAxes, va="top", fontsize=8)
        else:
            residual_axis.text(.5, .5, "fewer than 3 valid fibers", ha="center",
                               va="center", transform=residual_axis.transAxes)
        scatter_axis.set_title("%s exposure 1" % band)
        scatter_axis.set_xlabel("old: fixed smoothing + nearest pixel")
        scatter_axis.set_ylabel("new: matched PSF + exact aperture")
        scatter_axis.legend(fontsize=8)
        scatter_axis.grid(alpha=.2); residual_axis.grid(alpha=.2)
        residual_axis.set_xlabel("new external value")
        residual_axis.set_ylabel("new - robust-fit(old)")
    fig.suptitle("External fiber measurement comparison")
    fig.tight_layout(rect=(0, 0, 1, .95)); fig.savefig(output_path, dpi=170); plt.close(fig)


def print_external_sampling_sanity(sanity):
    """Print a small old/new V-vs-I check without changing the hierarchy."""
    for exposure, band in sorted(sanity):
        V, old, new = sanity[(exposure, band)]
        print("external sanity exposure %d %s:" % (exposure, band))
        for label, values in (("old", old), ("new", new)):
            valid = np.isfinite(V) & np.isfinite(values)
            fit = fit_line(values[valid], V[valid])
            residual = V[valid] - (fit["g"] * values[valid] + fit["z"])
            print("  %s I: g=%+.6g z=%+.6g robust RMS=%+.6g N=%d" %
                  (label, fit["g"], fit["z"], robust_rms(residual), valid.sum()))


def choose_gallery(qa_rows, summaries):
    rows = [row for row in qa_rows if row["exposure"] == 1]
    rows = [row for row in rows if np.isfinite(row["rms_ON"]) or np.isfinite(row["rms_OFF"])]
    by_id = {row["identity"]: row for row in rows}; selected = []

    def add(candidates, count):
        added = 0
        for row in candidates:
            if row["identity"] not in {r["identity"] for r in selected}:
                selected.append(row)
                added += 1
            if added >= count:
                break

    score = lambda row: max(row["rms_ON"], row["rms_OFF"])
    add(sorted(rows, key=score), 2)
    middle = np.nanmedian([score(row) for row in rows]) if rows else np.nan
    add(sorted(rows, key=lambda row: abs(score(row) - middle)), 2)
    add(sorted(rows, key=score, reverse=True), 2)
    add([row for row in rows if not row["fit_good"]], 2)
    blank = sorted(rows, key=lambda row: max(abs(summaries[(row["identity"], "ON")]["I"]),
                                               abs(summaries[(row["identity"], "OFF")]["I"])))
    bright = list(reversed(blank))
    add(blank, 2); add(bright, 2)
    return selected


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--h5", required=True)
    parser.add_argument("--on-image", required=True); parser.add_argument("--off-image", required=True)
    parser.add_argument("--on-filter", required=True); parser.add_argument("--off-filter", required=True)
    parser.add_argument("--fq-template", required=True)
    parser.add_argument("--output-dir", default="hierarchical_test")
    parser.add_argument("--exposure", type=int, choices=(1, 2, 3))
    parser.add_argument("--iterations", type=int, default=3)
    args = parser.parse_args()
    if args.iterations < 1: parser.error("--iterations must be positive")

    output_dir = Path(args.output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    # Do not leave a focal-plane before/after image from an earlier FoV-model
    # run in this no-illumination diagnostic directory.
    stale_focal_plot = output_dir / "focal_plane_residual_before_after.png"
    if stale_focal_plot.exists():
        stale_focal_plot.unlink()
    f = load_fq(args.fq_template)
    filters = {"ON": read_filter(args.on_filter), "OFF": read_filter(args.off_filter)}
    images = {"ON": load_image(args.on_image), "OFF": load_image(args.off_image)}
    for band in ("ON", "OFF"):
        estimate_external_background(images[band])
        characterize_external_image(images[band], band)
        print("%s image pixel scale=%.6g arcsec/pix; exact aperture radius=%.6g pix" %
              (band, images[band]["pixel_scale_arcsec"],
               FIBER_RADIUS_ARCSEC / images[band]["pixel_scale_arcsec"]))
    plot_external_image_psf_qa(images, output_dir / "external_image_psf_qa.png")
    groups = []; datasets = []
    sampling_comparison = {}
    sampling_sanity = {}
    background_check = {}
    with tables.open_file(args.h5, mode="r") as h5:
        if not {"Info", "Fibers", "Survey"}.issubset(h5.root._v_children):
            raise ValueError("H5 needs Info, Fibers, and Survey")
        info, fibers, survey = h5.root.Info, h5.root.Fibers, h5.root.Survey
        groups, labels = build_groups(info)
        ra = np.asarray(info.cols.ra[:], dtype=float); dec = np.asarray(info.cols.dec[:], dtype=float)
        for group in groups:
            group["mean_RA"] = float(np.nanmean(ra[group["indices"]]))
            group["mean_Dec"] = float(np.nanmean(dec[group["indices"]]))
        ifuslot = np.asarray(info.cols.ifuslot[:]); amp = np.asarray([as_text(v) for v in info.cols.amp[:]])
        bad = masked_rows(args.h5, ifuslot, amp)
        survey_by_exp = {}
        for source_row in survey:
            survey_by_exp[int(source_row["exp"])] = {
                name: source_row[name] for name in survey.colnames}
        spectra = np.asarray(fibers.cols.spectrum[:], dtype=float)
        if "skyspectrum" not in fibers.colnames:
            raise ValueError("Fibers needs skyspectrum for pre-sky-subtraction illumination diagnostics")
        skyspectra = np.asarray(fibers.cols.skyspectrum[:], dtype=float)
        exposures = [args.exposure] if args.exposure else [1, 2, 3]
        row_q = np.full(int(info.nrows), -1, dtype=int)
        for group in groups:
            j = np.arange(N_FIBER_AMP)
            row_q[group["indices"]] = j if group["amp"] in ("LL", "RU") else 111 - j
        for exposure in exposures:
            if exposure not in survey_by_exp:
                raise ValueError("Survey has no row for exposure %d" % exposure)
            survey_row = survey_by_exp[exposure]
            virus_fwhm = float(survey_row["fwhm"])
            if not np.isfinite(virus_fwhm) or virus_fwhm <= 0.0:
                raise ValueError("Survey.fwhm[%d] is not a positive FWHM in arcsec: %s" %
                                 (exposure, virus_fwhm))
            print("exposure %d: Survey.fwhm=%.6g; units assumed: FWHM arcsec" %
                  (exposure, virus_fwhm))
            offset = float(survey_row["offset"])
            working = spectra / offset if np.isfinite(offset) and offset != 0.0 else spectra.copy()
            exp_rows = labels == exposure
            adr_cache = {}
            for band in ("ON", "OFF"):
                V = synthetic_mean(working, filters[band])
                B_sky = synthetic_mean(skyspectra, filters[band])
                V_total = V + B_sky
                if band not in adr_cache:
                    adr_cache[band] = adr_positions(ra, dec, survey_row, filters[band])
                eff_ra, eff_dec = adr_cache[band]
                matched = matched_external_image(images[band], exposure, virus_fwhm, band)
                old_I, old_valid = sample_image_legacy(images[band], eff_ra, eff_dec)
                raw_I, raw_valid = sample_image_exact(
                    images[band], images[band]["matched_raw"][exposure], eff_ra, eff_dec)
                I, image_valid = sample_image_exact(images[band], matched, eff_ra, eff_dec)
                background_check[(exposure, band)] = (raw_I, I)
                if exposure == 1:
                    sampling_comparison[band] = (old_I[exp_rows], I[exp_rows])
                sampling_sanity[(exposure, band)] = (V[exp_rows], old_I[exp_rows], I[exp_rows])
                K = weighted_scalar(raw_work_basis(survey_row), filters[band])
                valid = exp_rows & ~bad & np.isfinite(V) & image_valid & np.isfinite(I)
                datasets.append({"exposure": exposure, "band": band, "V": V, "I": I,
                                 "B_sky": B_sky, "V_total": V_total, "ra": eff_ra,
                                 "dec": eff_dec, "K": K, "q": row_q, "valid": valid})

    if set(sampling_comparison) == {"ON", "OFF"}:
        plot_external_fiber_sampling_comparison(
            sampling_comparison, output_dir / "external_fiber_sampling_comparison.png")
    else:
        print("external fiber comparison skipped: exposure 1 was not selected")
    print_external_sampling_sanity(sampling_sanity)

    initial_good = np.asarray([not np.any(bad[group["indices"]]) for group in groups], dtype=bool)
    alpha = {(exposure, band): ALPHA_INITIAL for exposure in exposures for band in AMPS}
    good_groups = initial_good.copy()
    for iteration in range(1, args.iterations + 1):
        globals_, qa_rows, summaries, good_groups = broad_stage(
            datasets, groups, f, alpha, initial_good, good_groups)
        alpha = fit_bounded_alphas(datasets, groups, globals_, f,
                                   good_groups, alpha, exposures)
        print("iteration %d:" % iteration)
        for key in sorted(globals_):
            fit = globals_[key]
            exp_good = sum(good_groups[i] for i, group in enumerate(groups)
                           if group["exposure"] == key[0])
            exp_total = sum(group["exposure"] == key[0] for group in groups)
            print("  exp%d %s g=%+.6g z=%+.6g F=1 good=%d/%d" %
                  (key[0], key[1], fit["g"], fit["z"], exp_good, exp_total))
        print("  alpha " + " ".join("%s=%+.5g" % (amp, alpha[(exposure, amp)])
                                    for amp in AMPS for exposure in exposures))

    # Recompute the fixed-additive/g/z stage once more so the saved parameters
    # correspond to the final bounded alpha values. Alpha is not refit here.
    globals_, qa_rows, summaries, good_groups = broad_stage(
        datasets, groups, f, alpha, initial_good, good_groups)
    for dataset in datasets:
        # ``V0`` is source after the fixed additive detector term.  Adding
        # ``B_sky`` here reconstructs the pre-sky-subtraction signal while
        # leaving the existing source g/z machinery untouched.
        dataset["V_total_corrected"] = dataset["V0"] + dataset["B_sky"]

    for (exposure, band), (raw_I, object_I) in sorted(background_check.items()):
        dataset = next(d for d in datasets if d["exposure"] == exposure and d["band"] == band)
        fit_raw = fit_line(raw_I[dataset["valid"]], dataset["V0"][dataset["valid"]])
        fit_object = fit_line(object_I[dataset["valid"]], dataset["V0"][dataset["valid"]])
        print("background check exposure %d %s: broad g raw/object=%.8g/%.8g; "
              "z raw/object=%.8g/%.8g" %
              (exposure, band, fit_raw["g"], fit_object["g"],
               fit_raw["z"], fit_object["z"]))

    delta_by_band = {}
    illumination_global_rows = []
    for exposure in exposures:
        for band in ("ON", "OFF"):
            offset_row = solve_illumination_delta(
                datasets, groups, globals_, alpha, f, good_groups, exposures, exposure, band)
            delta_by_band[(exposure, band)] = offset_row["delta_illumination"]
            illumination_global_rows.append(offset_row)
    illumination_rows, leverage_qa = build_illumination_scalars(
        datasets, groups, globals_, alpha, f, good_groups, exposures, delta_by_band)
    common_normalizers = normalize_illumination_scalars(illumination_rows, exposures)
    for row in illumination_global_rows:
        image = images[row["band"]]
        row.update({"external_image_background_per_pixel": image["background"],
                    "external_image_background_scatter": image["background_scatter"],
                    "external_image_background_npix": image["background_npix"],
                    "external_image_background_annulus": image["background_annulus"],
                    "common_normalizer": common_normalizers[row["exposure"]]})
    planes = {exposure: fit_illumination_plane(illumination_rows, exposure)
              for exposure in exposures}
    for row in illumination_rows:
        row.setdefault("plane_used", False)
        row.setdefault("plane_model", np.nan)
        row.setdefault("plane_residual", np.nan)
    for exposure, band in sorted(leverage_qa):
        values = leverage_qa[(exposure, band)]
        print("illumination leverage exposure %d %s: median source/sky/total="
              "%.6g/%.6g/%.6g, median sky fraction=%.6g, N=%d, sum(x_total^2)=%.6g" %
              (exposure, band, values["median_source"], values["median_sky"],
               values["median_total"], values["median_sky_fraction"], values["n"],
               values["leverage"]))
    plot_illumination_source_sky_leverage(
        leverage_qa, output_dir / "illumination_source_sky_leverage.png")
    plot_illumination_ifu_on_vs_off(
        illumination_rows, output_dir / "illumination_ifu_on_vs_off.png")
    plot_illumination_ifu_scalar_maps(
        illumination_rows, output_dir / "illumination_ifu_scalar_maps.png")
    plot_illumination_plane_qa(
        illumination_rows, planes, output_dir / "illumination_plane_qa.png")
    plot_illumination_ifu_exposure_comparison(
        illumination_rows, output_dir / "illumination_ifu_exposure_comparison.png")
    plot_illumination_model_closure(
        datasets, groups, globals_, delta_by_band, illumination_rows, good_groups,
        output_dir / "illumination_model_closure.png")

    qa_by_id = {row["identity"]: row for row in qa_rows}
    records = {}
    for group in groups:
        if group["exposure"] not in exposures:
            continue
        record = {"identity": group["identity"], "alpha": alpha[(group["exposure"], group["amp"])],
                  "f": f, "fit_good": bool(good_groups[groups.index(group)]), "Y": {}, "M": {}, "D": {},
                  "qa": qa_by_id[group["identity"]]}
        for band in ("ON", "OFF"):
            dataset = next(d for d in datasets if d["exposure"] == group["exposure"] and d["band"] == band)
            indices = group["indices"]; q = dataset["q"][indices]
            arrays = {name: np.full(N_FIBER_AMP, np.nan) for name in ("Y", "M", "D")}
            good = dataset["valid"][indices]
            arrays["Y"][q[good]] = dataset["V"][indices[good]][:] / dataset["K"]
            arrays["M"][q[good]] = ((globals_[(group["exposure"], band)]["g"] *
                                      dataset["I"][indices[good]] + globals_[(group["exposure"], band)]["z"]) /
                                     dataset["F"][indices[good]] / dataset["K"])
            arrays["D"][q[good]] = arrays["Y"][q[good]] - arrays["M"][q[good]]
            for name in arrays: record[name][band] = arrays[name]
        records[group["identity"]] = record
    selected_rows = choose_gallery(qa_rows, summaries)
    print("selected gallery amplifiers:")
    for row in selected_rows:
        print("  exp%d SPECID=%d IFUSLOT=%03d IFUID=%d AMP=%s alpha=%+.5g RMS_ON=%+.5g RMS_OFF=%+.5g good=%s" %
              (row["exposure"], row["SPECID"], row["IFUSLOT"], row["IFUID"], row["AMP"],
               alpha[(row["exposure"], row["AMP"])], row["rms_ON"], row["rms_OFF"], row["fit_good"]))
    selected_records = {row["identity"]: records[row["identity"]] for row in selected_rows}
    for identity, record in selected_records.items():
        plot_components(record, output_dir / ("exp%d_ifuslot%03d_%s_components.png" %
                                              (identity[0], identity[2], identity[4])))
    plot_gallery(selected_records, output_dir / "amplifier_component_gallery.png")
    qa_output = []
    for row in qa_rows:
        qa_output.append({key: row[key] for key in
                          ("exposure", "SPECID", "IFUSLOT", "IFUID", "AMP", "fit_good")}
                         | {"residual_median_ON": row["median_ON"], "residual_RMS_ON": row["rms_ON"],
                            "residual_median_OFF": row["median_OFF"], "residual_RMS_OFF": row["rms_OFF"],
                            "mean_RA": row["mean_RA"], "mean_Dec": row["mean_Dec"]})
    global_output = []
    for key, fit in sorted(globals_.items()):
        exp_good = sum(good_groups[i] for i, group in enumerate(groups)
                       if group["exposure"] == key[0])
        global_output.append({"exposure": key[0], "band": key[1], "g": fit["g"], "z": fit["z"],
                              "F_p0": 1.0, "F_dRA": 0.0, "F_dDec": 0.0,
                              "F_ra0": np.nan, "F_dec0": np.nan,
                              "number_good_amps": int(exp_good)})
    alpha_output = [{"exposure": exposure, "AMP": amp, "alpha_e_per_A": alpha[(exposure, amp)],
                     "at_lower_bound": bool(np.isclose(alpha[(exposure, amp)], ALPHA_LOW)),
                    "at_upper_bound": bool(np.isclose(alpha[(exposure, amp)], ALPHA_HIGH))}
                    for exposure in exposures for amp in AMPS]
    write_csvs(output_dir, qa_output, global_output, alpha_output)
    write_illumination_csvs(output_dir, illumination_rows,
                            [planes[exposure] for exposure in exposures],
                            illumination_global_rows)
    print("hierarchical diagnostic complete: %s" % output_dir)


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        main()
