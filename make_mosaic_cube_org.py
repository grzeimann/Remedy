#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 13:10:11 2020

@author: gregz
"""
import matplotlib
matplotlib.use('agg')
import argparse as ap
from concurrent.futures import ThreadPoolExecutor
from numba import njit
from astropy.convolution import convolve, Gaussian2DKernel
from astropy.convolution import Gaussian1DKernel, interpolate_replace_nans
from astropy.io import fits
from astropy.wcs import WCS
from astropy.table import Table
import tables
import numpy as np
import os.path as op
import sys
import warnings
from input_utils import setup_logging
from astrometry import Astrometry
from math_utils import biweight
from extract import Extract
from scipy.interpolate import griddata, PchipInterpolator
import matplotlib.pyplot as plt



import glob

mask_dict = {'20200430-20200501': ['057RL', '057RU', '057LL', '057LU', '058RU', 
                                   '058RL', '021RL', '021RU', '021LL', '021LU'],
             '20200430-20200715': ['092LU', '094RU', '046RU', '104LL', '104LU',
                                   '028RU', '028RL', '027LL', '027LU', '067LL',
                                   '067LU', '025LU', '106RU', '106RL', '026LL',
                                   '026LU', '026RL', '026RU', '103LL', '103LU'],
             '20200525-20200526': ['057RL', '057LL', '089RU', '089RL'],
             '20200517-20200522': ['030RL', '030RU', '030LL', '030LU'],
             '20200523-20200526': ['039RL', '039RU', '039LL', '039LU'],
             '20200622-20200715': ['089RL', '089RU', '096RL', '096RU']}

# M101-specific residual sky correction at H5-consumption time.
M101_RA_DEG = 210.800
M101_DEC_DEG = 54.333
M101_SKY_MIN_RADIUS_ARCMIN = 6.0
M101_SKY_NEXPOSURES = 3
M101_SKY_MIN_FINITE_FRACTION = 0.8
M101_SKY_MIN_FIBERS = 20

# Data-quality bits for the final SCI/VAR estimator state.  Coverage strength
# and the number of contributing shots remain continuous/independent products.
DQ_INSUFFICIENT_SUPPORT = np.uint16(1 << 0)
DQ_VAR_INCOMPLETE = np.uint16(1 << 1)
DQ_EMPIRICAL_VAR_USED = np.uint16(1 << 2)
DQ_FORMAL_VAR_USED = np.uint16(1 << 3)
DQ_VAR_EMPIRICAL_ONLY = np.uint16(1 << 4)

# Error-provenance columns used by the compact wavelength/exposure counters.
PROV_NSCI = 0
PROV_ERROR_VALID = 1
PROV_ERROR_NONFINITE = 2
PROV_ERROR_ZERO = 3
PROV_ERROR_NEGATIVE = 4
PROV_ERROR_OTHER = 5

# Empirical LSF/FWHM calibration anchors.  These are AIR wavelengths and are
# intentionally kept separate from the science wavelength grid.  The local
# project resources (Lines_list/virus_lines.dat and wave_utils.py) contain
# these anchors.  Established Hg/Cd laboratory/astronomical line lists show
# that the blue Cd I components around 3610 A and the Hg/Cd structure around
# 3650 A are known blends; they are retained for QA but excluded from LSF use.
REFERENCE_ARC_WAVELENGTHS = np.array([
    3610.508,
    3650.153,
    4046.565,
    4358.335,
    4678.149,
    4799.912,
    4916.068,
    5085.822,
    5460.750,
], dtype=float)
ARC_WINDOW_ANGSTROM = 15.0
ARC_USABLE_FOR_LSF = np.array(
    [False, False, True, True, True, True, True, True, True], dtype=bool)
ARC_BLEND_STATES = np.array([
    'KNOWN_BLEND', 'KNOWN_BLEND',
    'UNKNOWN', 'UNKNOWN', 'UNKNOWN', 'UNKNOWN', 'UNKNOWN', 'UNKNOWN', 'UNKNOWN'
], dtype='U16')
ARC_BLEND_NOTES = {
    3610.508: ('Known Cd I blend with 3612.873 A and 3614.453 A; '
               'excluded from LSF'),
    3650.153: ('Known Hg/Cd blend; Hg I 3654.84 A produces red shoulder; '
               'excluded from LSF'),
    4046.565: 'No significant known blend affecting half-maximum width',
    4358.335: 'No significant known blend affecting half-maximum width',
    4678.149: 'No significant known blend affecting half-maximum width',
    4799.912: 'No significant known blend affecting half-maximum width',
    4916.068: 'No significant known blend affecting half-maximum width',
    5085.822: 'No significant known blend affecting half-maximum width',
    5460.750: 'No significant known blend affecting half-maximum width',
}


def subtract_m101_residual_sky(spectra, ra, dec, xg, yg, tp,
                               binimage=None, log=None, h5file=''):
    """Subtract one M101-safe residual sky spectrum per exposure.

    This operates directly on already sky-subtracted H5 SCI.  It deliberately
    does not read or add back ``Fibers.skyspectrum``: this is the single
    residual-sky correction performed by the cube builder.  All masks are
    full-length masks in the local H5 fiber coordinate system.

    If an external image is available, ``binimage < 0.01`` is an additional
    blank-sky criterion.  Invalid image pixels are excluded rather than being
    silently classified as blank.  If no external image is supplied, the
    radial and mosaic-region criteria remain active and the log records that
    the external blank-image criterion was unavailable.
    """
    n_fib, n_wave = spectra.shape
    inds_local = np.arange(n_fib)
    block_ids = inds_local // 112
    dra_arcmin = ((ra - M101_RA_DEG) *
                  np.cos(np.deg2rad(M101_DEC_DEG)) * 60.0)
    ddec_arcmin = (dec - M101_DEC_DEG) * 60.0
    radius_arcmin = np.sqrt(dra_arcmin ** 2 + ddec_arcmin ** 2)
    sky_region = radius_arcmin > M101_SKY_MIN_RADIUS_ARCMIN

    # Image sampling uses the same 1-based WCS pixel convention as the
    # existing normalization code.  This is only for the optional external
    # image criterion; the M101 radial selection above uses RA/Dec directly.
    x, y = tp.wcs_world2pix(ra, dec, 1)
    xc = np.rint(np.interp(x, xg, np.arange(len(xg)), left=0., right=len(xg)))
    yc = np.rint(np.interp(y, yg, np.arange(len(yg)), left=0., right=len(yg)))
    xc = np.asarray(xc, dtype=int)
    yc = np.asarray(yc, dtype=int)
    in_valid_image_region = (
        np.isfinite(x) & np.isfinite(y) &
        (xc >= 0) & (xc < len(xg)) &
        (yc >= 0) & (yc < len(yg)))

    external_blank = None
    if binimage is not None:
        external_blank = np.zeros(n_fib, dtype=bool)
        valid_image = in_valid_image_region.copy()
        valid_image[valid_image] &= np.isfinite(binimage[yc[valid_image], xc[valid_image]])
        external_blank[valid_image] = (
            binimage[yc[valid_image], xc[valid_image]] < 0.01)
    elif log is not None:
        log.warning(
            'M101 residual sky %s: external image unavailable; not applying '
            'an external blank-image classification.', h5file)

    def report(message, *values):
        if log is not None:
            log.info(message, *values)

    for k in range(M101_SKY_NEXPOSURES):
        exp_sel = (block_ids % M101_SKY_NEXPOSURES) == k
        exp_indices = np.where(exp_sel)[0]
        if exp_indices.size == 0:
            report('M101 residual sky %s exposure %d: no fibers; skipping correction.',
                   h5file, k + 1)
            continue

        finite_counts = np.isfinite(spectra[exp_indices]).sum(axis=1)
        sufficient_finite = (
            finite_counts >= int(np.ceil(M101_SKY_MIN_FINITE_FRACTION * n_wave)))
        full_sky_mask = np.zeros(n_fib, dtype=bool)
        full_sky_mask[exp_indices] = (
            sky_region[exp_indices] &
            in_valid_image_region[exp_indices] &
            sufficient_finite)
        if external_blank is not None:
            full_sky_mask &= external_blank

        selected_sky_count = int(full_sky_mask.sum())
        if selected_sky_count < M101_SKY_MIN_FIBERS:
            report(
                'M101 residual sky %s exposure %d: total fibers=%d, beyond 6 arcmin=%d, '
                'valid image fibers=%d, blank-image candidates=%s, finite sky candidates=%d, '
                'selected sky fibers=%d; fewer than %d, skipping correction.',
                h5file, k + 1, int(exp_indices.size),
                int(np.sum(sky_region[exp_indices])),
                int(np.sum(in_valid_image_region[exp_indices])),
                ('unavailable' if external_blank is None else
                 str(int(np.sum(external_blank[exp_indices])))),
                selected_sky_count, selected_sky_count, M101_SKY_MIN_FIBERS)
            if log is not None:
                log.warning(
                    'M101 residual sky %s exposure %d skipped: only %d sky '
                    'candidates (minimum %d).',
                    h5file, k + 1, selected_sky_count, M101_SKY_MIN_FIBERS)
            continue

        residual_sky = biweight(spectra[full_sky_mask], axis=0)
        finite_residual = np.isfinite(residual_sky)
        residual_fraction = finite_residual.sum() / float(n_wave)
        if residual_fraction < M101_SKY_MIN_FINITE_FRACTION:
            report(
                'M101 residual sky %s exposure %d: residual sky finite at %d/%d '
                'wavelengths; mostly nonfinite, skipping correction.',
                h5file, k + 1, int(finite_residual.sum()), n_wave)
            if log is not None:
                log.warning(
                    'M101 residual sky %s exposure %d skipped because the residual '
                    'sky is mostly nonfinite.', h5file, k + 1)
            continue

        # Apply only finite residual bins if a small number of bins are
        # undefined; this leaves those undefined SCI bins unchanged.
        spectra[np.ix_(exp_indices, finite_residual)] -= residual_sky[finite_residual]
        before_level = np.nanmedian(
            spectra[full_sky_mask] + residual_sky[None, :])
        after_level = np.nanmedian(spectra[full_sky_mask])
        p16, p50, p84 = np.nanpercentile(residual_sky[finite_residual], [16, 50, 84])
        report(
            'M101 residual sky %s exposure %d: total fibers=%d, beyond 6 arcmin=%d, '
            'valid image fibers=%d, blank-image candidates=%s, finite sky candidates=%d, '
            'selected sky fibers=%d, median residual sky=%0.5g, '
            'residual sky p16/p50/p84=(%0.5g, %0.5g, %0.5g), finite residual wavelengths=%d/%d, '
            'median sky-region level before=%0.5g, after=%0.5g',
            h5file, k + 1, int(exp_indices.size),
            int(np.sum(sky_region[exp_indices])),
            int(np.sum(in_valid_image_region[exp_indices])),
            ('unavailable' if external_blank is None else
             str(int(np.sum(external_blank[exp_indices])))),
            selected_sky_count, selected_sky_count, p50, p16, p50, p84,
            int(finite_residual.sum()), n_wave, before_level, after_level)

    return spectra


def get_script_path():
    return op.dirname(op.realpath(sys.argv[0]))

warnings.filterwarnings("ignore")

DIRNAME = get_script_path()

parser = ap.ArgumentParser(add_help=True)

parser.add_argument("-d", "--directory",
                    help='''base directory for reductions''',
                    type=str, default="")

parser.add_argument("-c", "--caldirectory",
                    help='''cal directory for reductions''',
                    type=str, default="/work/03946/hetdex/maverick/LRS2/CALS")

parser.add_argument("h5files",
                    help='''e.g., 20200430_0000020.h5''',
                    type=str)

parser.add_argument("surname",
                    help='''file name modification''',
                    type=str)

parser.add_argument("image_center_size",
                    help='''RA, Dec center of image and size (deg, deg, arcmin)
                            "150.0, 50.0, 22.0"''',
                    default=None, type=str)

parser.add_argument("-ps", "--pixel_scale",
                    help='''Pixel scale for output image in arcsec''',
                    default=1.0, type=float)

parser.add_argument("-if", "--image_file",
                    help='''Image filename''',
                    default=None, type=str)

parser.add_argument("-ff", "--filter_file",
                    help='''Filter filename''',
                    default=None, type=str)

parser.add_argument("--wave-workers",
                    help='''Number of threads used to build wavelength planes (default: 1)''',
                    default=1, type=int)

parser.add_argument("--make-lsf", action="store_true",
                    help='''Also propagate master-arc features sparsely and write empirical FWHM products''')

def rebin(arr, new_shape):
    """Rebin 2D array arr to shape new_shape by averaging."""
    shape = (new_shape[0], arr.shape[0] // new_shape[0],
             new_shape[1], arr.shape[1] // new_shape[1])
    return arr.reshape(shape).mean(axis=(-1,1))

def make_image_interp(Pos, y, ye, xg, yg, xgrid, ygrid, sigma, cnt_array,
                      binsize=2):
    # Loop through shots, average images before convolution
    # Build a PSF-aware fractional coverage map (0..1) using Gaussian smoothing
    G = Gaussian2DKernel(sigma)
    nshots = len(cnt_array)
    image_all = np.ma.zeros((nshots,) + xgrid.shape)
    image_all.mask = True
    coveragek = 0.0 * xgrid  # accumulate per-shot smoothed coverage, later normalized to 0..1
    # grid for accumulating inverse-variance per pixel (pre-convolution)
    ivar_grid = 0.0 * xgrid
    # area scaling applied to flux; propagate to errors as well
    area = (np.pi * 0.75**2)

    xc = np.interp(Pos[:, 0], xg, np.arange(len(xg)), left=0., right=len(xg))
    yc = np.interp(Pos[:, 1], yg, np.arange(len(yg)), left=0., right=len(yg))
    xc = np.array(np.round(xc), dtype=int)
    yc = np.array(np.round(yc), dtype=int)
    gsel = np.where((xc>1) * (xc<len(xg)-1) * (yc>1) * (yc<len(yg)-1))[0]
    for k, cnt in enumerate(cnt_array):
        # Support two formats for cnt: (start, stop) range, or explicit index array
        if isinstance(cnt, (tuple, list)) and len(cnt) == 2:
            l1 = int(cnt[0])
            l2 = int(cnt[1])
            sel_idx = gsel[(gsel >= l1) & (gsel < l2)]
        else:
            # assume numpy array of global indices
            cnt_arr = np.array(cnt, dtype=int)
            # intersect with valid selection
            sel_idx = np.intersect1d(gsel, cnt_arr, assume_unique=False)
        if sel_idx.size == 0:
            continue
        # place flux values (scaled by area)
        image_all.data[k, yc[sel_idx], xc[sel_idx]] = y[sel_idx] / area
        image_all.mask[k, yc[sel_idx], xc[sel_idx]] = False
        # accumulate inverse variance at exact sample locations (propagate scaling)
        if ye is not None:
            ye_sel = ye[sel_idx]
            valid_e = np.isfinite(ye_sel) & (ye_sel > 0)
            if np.any(valid_e):
                ivar_vals = np.zeros_like(ye_sel, dtype=float)
                ivar_vals[valid_e] = 1.0 / (ye_sel[valid_e]**2) / (area**2)
                # add to grid
                ivar_grid[yc[sel_idx], xc[sel_idx]] += ivar_vals
        # build per-shot binary sampling mask, then smooth with the same PSF kernel
        valid_flux = np.isfinite(y[sel_idx])
        if np.any(valid_flux):
            mask_shot = 0.0 * xgrid
            mask_shot[yc[sel_idx][valid_flux], xc[sel_idx][valid_flux]] = 1.0
            # Convolution of binary mask (kernel is normalized to unit sum) yields 0..1 local coverage
            mask_sm = convolve(mask_shot, G, preserve_nan=False, boundary='extend')
            # Accumulate; we'll normalize by number of shots to map to 0..1 overall
            coveragek += mask_sm
    # form the image by median stacking shots then Gaussian smoothing
    image = np.ma.median(image_all, axis=0)
    y_im = image.data * 1.
    y_im[image.mask] = np.nan
    image = convolve(y_im, G, preserve_nan=False, boundary='extend')
    # normalize coverage to 0..1; avoid division by zero
    nshots = max(1, nshots)
    coveragek = np.clip(coveragek / float(nshots), 0.0, 1.0)
    image[coveragek == 0.] = np.nan
    image[np.isnan(image)] = 0.0

    # build an error image: start from per-pixel variance from inverse-variance sum
    var0 = np.empty_like(xgrid, dtype=float)
    var0[:] = np.nan
    pos = ivar_grid > 0
    if np.any(pos):
        var0[pos] = 1.0 / ivar_grid[pos]
    # propagate variance through Gaussian smoothing: convolve with kernel^2
    kernel_sq = G.array**2
    var_sm = convolve(var0, kernel_sq, normalize_kernel=False, boundary='extend')
    var_sm[coveragek == 0.] = np.nan
    # finalize error image
    errorimage = np.sqrt(var_sm)
    errorimage[np.isnan(errorimage)] = 0.0

    return image, errorimage, coveragek


@njit(nogil=True, cache=True)
def _gaussian_splat_shot_xy(indices, xpos, ypos, fluxes, errors,
                            x_origin, y_origin, nx, ny, sigma, radius,
                            support_radius, area, flux_sum, weight_sum,
                            variance_numerator, error_weight_sum,
                            support_map):
    """Numba kernel for one shot's subpixel Gaussian deposition."""
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
            dx = (x_origin + ix_center + ox) - xi
            gx[ox + radius] = np.exp(-0.5 * dx * dx / sigma_sq)
        for oy in range(-radius, radius + 1):
            dy = (y_origin + iy_center + oy) - yi
            gy[oy + radius] = np.exp(-0.5 * dy * dy / sigma_sq)

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

        # Independent-shot support is based on distance to a fiber, not on
        # whether a Gaussian stencil has a mathematically nonzero tail.
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


@njit(nogil=True, cache=True)
def _gaussian_splat_sparse_shot_xy(indices, xpos, ypos, fluxes,
                                   sample_lookup, x_origin, y_origin,
                                   nx, ny, sigma, radius, support_radius,
                                   area, flux_sum, weight_sum, support_map):
    """Splat one exposure, retaining only pixels named by ``sample_lookup``.

    The arithmetic and loop order intentionally mirror
    ``_gaussian_splat_shot_xy``.  The lookup only removes additions for output
    pixels that are not among the selected amplifier-center samples.
    """
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
            dx = (x_origin + ix_center + ox) - xi
            gx[ox + radius] = np.exp(-0.5 * dx * dx / sigma_sq)
        for oy in range(-radius, radius + 1):
            dy = (y_origin + iy_center + oy) - yi
            gy[oy + radius] = np.exp(-0.5 * dy * dy / sigma_sq)

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

        # Independent-shot support uses the same distance criterion as the
        # full image kernel, but only for selected output pixels.
        for ox in range(-support_radius, support_radius + 1):
            px = ix_center + ox
            if px < 0 or px >= nx:
                continue
            dx = (x_origin + px) - xi
            for oy in range(-support_radius, support_radius + 1):
                py = iy_center + oy
                if py < 0 or py >= ny:
                    continue
                sample_id = sample_lookup[py, px]
                if sample_id < 0:
                    continue
                dy = (y_origin + py) - yi
                if dx * dx + dy * dy <= support_radius_sq:
                    support_map[sample_id] = 1


def _classify_error_provenance(science, errors):
    """Count error states for finite SCI fiber samples.

    The returned columns are ``N_SCI``, positive finite error, nonfinite
    error, zero error, negative error, and any remaining invalid state.  This
    is input-sample provenance only; it does not alter SCI acceptance.
    """
    science_valid = np.isfinite(science)
    error_finite = np.isfinite(errors)
    error_valid = science_valid & error_finite & (errors > 0.0)
    error_nonfinite = science_valid & ~error_finite
    error_zero = science_valid & error_finite & (errors == 0.0)
    error_negative = science_valid & error_finite & (errors < 0.0)
    error_other = science_valid & ~(
        error_valid | error_nonfinite | error_zero | error_negative)
    return np.array([
        int(np.sum(science_valid)),
        int(np.sum(error_valid)),
        int(np.sum(error_nonfinite)),
        int(np.sum(error_zero)),
        int(np.sum(error_negative)),
        int(np.sum(error_other)),
    ], dtype=np.int64)


def _compute_final_variance(shot_images, shot_variances, ncontrib):
    """Compute formal/empirical variance for the existing shot median.

    ``shot_images`` and ``shot_variances`` are indexed by shot, while
    ``ncontrib`` retains the existing meaningful-support count.  This helper
    never changes SCI or support; it only evaluates variance for pixels whose
    SCI median already has at least two supported shots.
    """
    ny, nx = ncontrib.shape
    valid_sci = ncontrib >= 2

    # Keep the support selection identical to the SCI nanmedian.  The
    # vectorized calculations below only read these arrays and cannot change
    # SCI or NCONTRIB.
    supported = np.isfinite(shot_images)
    n = supported.sum(axis=0)
    finite_variance = np.isfinite(shot_variances)
    complete = np.all(~supported | finite_variance, axis=0)
    positive_or_zero = np.all(~supported | (shot_variances >= 0.0), axis=0)

    formal = np.full((ny, nx), np.nan, dtype=np.float64)
    two_shot = (n == 2) & complete
    if np.any(two_shot):
        formal[two_shot] = (
            np.sum(np.where(supported, shot_variances, 0.0), axis=0)[two_shot] /
            4.0)

    three_or_more = (n >= 3) & complete & positive_or_zero
    if np.any(three_or_more):
        sigmas = np.sqrt(np.where(supported, shot_variances, 1.0))
        inverse_sigma = np.zeros_like(sigmas, dtype=np.float64)
        np.divide(1.0, sigmas, out=inverse_sigma, where=supported)
        inverse_sigma_sum = np.sum(inverse_sigma, axis=0)
        positive_sigma = np.all(~supported | (sigmas > 0.0), axis=0)
        formal_valid = three_or_more & positive_sigma
        formal[formal_valid] = (
            n[formal_valid] * np.pi /
            (2.0 * inverse_sigma_sum[formal_valid] ** 2))

    empirical = np.full((ny, nx), np.nan, dtype=np.float64)
    five_or_more = n >= 5
    if np.any(five_or_more):
        center = np.nanmedian(shot_images, axis=0)
        mad = np.nanmedian(np.abs(shot_images - center[None, :, :]), axis=0)
        sigma_scatter = 1.4826 * mad
        sigma_median_empirical = 1.2533 * sigma_scatter / np.sqrt(n)
        empirical[five_or_more] = sigma_median_empirical[five_or_more] ** 2

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

    # The variance provenance states are mutually exclusive by construction.
    variance_state_count = (formal_used.astype(np.uint8) +
                            empirical_used.astype(np.uint8) +
                            empirical_only.astype(np.uint8) +
                            variance_incomplete.astype(np.uint8))
    assert np.all(variance_state_count <= 1)

    stats = {
        'valid_sci_voxels': int(np.sum(valid_sci)),
        'finite_variance_voxels': int(np.sum(valid_sci & np.isfinite(varianceimage))),
        'shot_samples': int(np.sum(supported)),
        'shot_samples_missing_variance': int(np.sum(supported & ~finite_variance)),
        'both_variances': int(np.sum(both)),
        'empirical_exceeds_formal': int(np.sum(both & (empirical > formal))),
        'median_ncontrib': float(np.median(ncontrib[valid_sci])) if np.any(valid_sci) else np.nan,
        'variance_unavailable_voxels': int(
            np.sum(valid_sci & ~np.isfinite(varianceimage))),
        'insufficient_support_voxels': int(np.sum(~valid_sci)),
        'formal_var_used_voxels': int(np.sum(formal_used)),
        'empirical_var_used_voxels': int(np.sum(empirical_used)),
        'empirical_only_voxels': int(np.sum(empirical_only)),
    }
    ratio = np.full((ny, nx), np.nan, dtype=np.float64)
    ratio[both] = np.sqrt(empirical[both] / formal[both])
    stats['median_ratio'] = (float(np.median(ratio[np.isfinite(ratio)]))
                             if np.any(np.isfinite(ratio)) else np.nan)
    return varianceimage, dq, stats


def make_image_gaussian(Pos, y, ye, xg, yg, xgrid, ygrid, sigma,
                        cnt_array):
    """Construct one wavelength plane with true subpixel Gaussian splatting.

    Each finite science fiber contributes ``flux / area`` to a Gaussian
    weighted interpolation within its shot.  Shot images are median-combined
    only where that shot has a fiber within ``2 * sigma`` of the output pixel.
    The returned ``coverage`` is the mean per-shot Gaussian support strength,
    capped at one per shot; it is a diagnostic and never divides SCI.

    The returned ``varianceimage`` is the variance of the final SCI median:
    formal shot variances are propagated through the median, and a robust
    shot-to-shot scatter estimate is used when at least five shots contribute.
    SCI and NCONTRIB are computed independently of this variance path.
    """
    nshots = len(cnt_array)
    ny, nx = xgrid.shape
    radius = 3
    support_radius = int(np.ceil(2.0 * sigma))
    area = np.pi * 0.75 ** 2

    xpos = Pos[:, 0]
    ypos = Pos[:, 1]
    if ye is None:
        errors = np.full(y.shape, np.nan, dtype=np.float32)
    else:
        errors = ye
    shot_images = np.full((nshots, ny, nx), np.nan, dtype=np.float32)
    shot_variances = np.full((nshots, ny, nx), np.nan, dtype=np.float32)
    coverage = np.zeros((ny, nx), dtype=np.float32)
    ncontrib = np.zeros((ny, nx), dtype=np.uint8)

    for k, cnt in enumerate(cnt_array):
        if isinstance(cnt, (tuple, list)) and len(cnt) == 2:
            indices = np.arange(int(cnt[0]), int(cnt[1]), dtype=np.int64)
        else:
            indices = np.asarray(cnt, dtype=np.int64)

        flux_sum = np.zeros((ny, nx), dtype=np.float32)
        weight_sum = np.zeros((ny, nx), dtype=np.float32)
        variance_numerator = np.zeros((ny, nx), dtype=np.float32)
        error_weight_sum = np.zeros((ny, nx), dtype=np.float32)
        support_map = np.zeros((ny, nx), dtype=np.uint8)
        _gaussian_splat_shot_xy(
            indices, xpos, ypos, y, errors, float(xg[0]), float(yg[0]),
            nx, ny, float(sigma), radius, support_radius, float(area),
            flux_sum, weight_sum, variance_numerator, error_weight_sum,
            support_map)

        supported = (support_map != 0) & (weight_sum > 0.0)
        if np.any(supported):
            shot_images[k, supported] = flux_sum[supported] / weight_sum[supported]
            # Formal variance is valid only when every SCI-contributing fiber
            # has a finite positive error.  The science denominator remains
            # the full science weight sum, never the error-support sum.
            complete_error_support = supported & np.isclose(
                error_weight_sum, weight_sum, rtol=1.e-5, atol=1.e-7)
            shot_variances[k, complete_error_support] = (
                variance_numerator[complete_error_support] /
                (weight_sum[complete_error_support] *
                 weight_sum[complete_error_support]))
            ncontrib[supported] += 1
            coverage += np.minimum(weight_sum, 1.0) * supported

    # The median ignores unsupported shots because those entries remain NaN.
    image = np.nanmedian(shot_images, axis=0).astype(np.float32)
    varianceimage, dq, variance_stats = _compute_final_variance(
        shot_images, shot_variances, ncontrib)
    coverage /= float(max(1, nshots))
    coverage = np.clip(coverage, 0.0, 1.0)

    final_valid = ncontrib >= 2
    image[~final_valid] = 0.0
    varianceimage[~final_valid] = 0.0
    image[~np.isfinite(image)] = 0.0
    # Keep NaN for a valid SCI pixel whose variance cannot be estimated.

    return image, varianceimage, coverage, ncontrib, dq, variance_stats

def make_image(Pos, y, ye, xg, yg, xgrid, ygrid, sigma, cnt_array):
    image = xgrid * 0.
    error = xgrid * 0.
    weight = xgrid * 0.
    N = int(sigma*5)
    indx = np.searchsorted(xg, Pos[:, 0])
    indy = np.searchsorted(yg, Pos[:, 1])
    nogood = ((indx < N) + (indx >= (len(xg) - N)) +
              (indy < N) + (indy >= (len(yg) - N)))
    nogood = nogood + np.isnan(y)
    y[nogood] = 0.0
    ye[nogood] = 0.0
    indx[nogood] = N+1
    indy[nogood] = N+1
    indx_array = np.zeros((len(y), 4*N**2), dtype=int)
    indy_array = np.zeros((len(y), 4*N**2), dtype=int)
    for i in np.arange(2*N):
        for j in np.arange(2*N):
            c = i * 2 * N + j
            indx_array[:, c] = indx + i - N
            indy_array[:, c] = indy + j - N
    d = np.sqrt((xgrid[indy_array, indx_array] - Pos[:, 0][:, np.newaxis])**2 + 
                (ygrid[indy_array, indx_array] - Pos[:, 1][:, np.newaxis])**2)
    G = np.exp(-0.5 * d**2 / sigma**2)
    G[:] /= G.sum(axis=1)[:, np.newaxis]
    G[nogood] = 0.
    for j in np.arange(len(y)):
        image[indy_array[j], indx_array[j]] += y[j] * G[j]
        error[indy_array[j], indx_array[j]] += ye[j]**2 * G[j]
        weight[indy_array[j], indx_array[j]] += G[j]
    weight[:] *= np.pi * 0.75**2
    return image, np.sqrt(error), weight


def _h5_text(value):
    """Return a stable human-readable value for PyTables string columns."""
    if isinstance(value, (bytes, np.bytes_)):
        return value.decode('utf-8', errors='replace').strip()
    return str(value).strip()


def _exp_text(value):
    """Normalize numeric/string exposure labels for the QA comparison."""
    if isinstance(value, (bytes, np.bytes_)):
        value = value.decode('utf-8', errors='replace')
    try:
        return '%g' % float(value)
    except (TypeError, ValueError):
        return str(value).strip()


def _verify_lsf_exposure_partition(info_exp, n_fibers, h5file, log=None,
                                   inferred_nexp=None):
    """QA-check the science grouping and diagnose unpopulated ``Info.exp``."""
    labels = [_exp_text(v) for v in np.asarray(info_exp)]
    unique_labels = sorted(set(labels))

    # quick_reduction creates Info.exp but, in the historical M101 products,
    # does not assign it when writing each fiber row.  A constant zero is
    # therefore missing metadata, not a string representation of 3 exposure
    # labels.  The exposure count is inferred from the same 448-fiber IFU
    # blocks used by the reduction, while the existing 112-row ordering
    # remains the authoritative row-level grouping.
    info_unpopulated = len(unique_labels) == 1 and unique_labels[0] in ('0', '')
    if info_unpopulated:
        if log is not None:
            if inferred_nexp is not None:
                log.warning('LSF QA %s: Info.exp is present but unpopulated '
                            '(constant %s); inferred nexp=%d from the fiber-row '
                            'structure Nfiber/(448*NIFU). The existing 112-fiber '
                            'row ordering is the exposure partition and is unchanged.',
                            op.basename(h5file), unique_labels[0], inferred_nexp)
            else:
                log.warning('LSF QA %s: Info.exp is present but unpopulated '
                            '(constant %s); no valid row-structure exposure count '
                            'was available. The existing 112-fiber row ordering '
                            'is unchanged.', op.basename(h5file), unique_labels[0])
        return None

    blocks = []
    for block_start in range(0, n_fibers, 112):
        block = labels[block_start:block_start + 112]
        blocks.append(sorted(set(block)))

    observed = [values[0] if len(values) == 1 else 'MIXED'
                for values in blocks]
    agree = (n_fibers % 112 == 0 and
             all(len(values) == 1 for values in blocks) and
             all(value == str(k % 3) for k, value in enumerate(observed)))
    agree_one_based = (n_fibers % 112 == 0 and
                       all(len(values) == 1 for values in blocks) and
                       all(value == str((k % 3) + 1)
                           for k, value in enumerate(observed)))
    agree = agree or agree_one_based
    if log is not None:
        if agree:
            log.info('LSF QA %s: Info.exp agrees with the existing 112-fiber '
                     'three-exposure partition (%s-based labels).',
                     op.basename(h5file), 'one' if agree_one_based else 'zero')
        else:
            log.warning('LSF QA %s: Info.exp does NOT agree with the existing '
                        '112-fiber exposure partition; observed block labels=%s. '
                        'Science grouping is unchanged.',
                        op.basename(h5file), observed[:12])
    return agree


def _build_lsf_amplifier_centers(h5file, h5table, log=None):
    """Build one median RA/Dec center per shot x IFUSLOT x amplifier."""
    if 'Raw' not in h5table.root._v_children:
        raise ValueError('LSF requested for %s, but the H5 file has no Raw '
                         'table containing arcspectrum/wave.' % h5file)
    info = h5table.root.Info.cols
    ra = np.asarray(info.ra[:], dtype=float)
    dec = np.asarray(info.dec[:], dtype=float)
    ifuslot = np.asarray(info.ifuslot[:])
    amp = np.asarray([_h5_text(v) for v in info.amp[:]])
    specid = np.asarray([_h5_text(v) for v in info.specid[:]])
    ifuid = np.asarray([_h5_text(v) for v in info.ifuid[:]])

    n_info = len(ra)
    n_fiber = int(h5table.root.Fibers.nrows)
    n_raw = int(h5table.root.Raw.nrows)
    if not (n_info == n_fiber == n_raw):
        raise ValueError('LSF row alignment failure for %s: Info=%d Fibers=%d Raw=%d'
                         % (h5file, n_info, n_fiber, n_raw))
    nslots = len(np.unique(ifuslot))
    inferred_nexp = (int(n_info / float(448 * nslots))
                     if nslots > 0 and n_info % (448 * nslots) == 0 else None)
    _verify_lsf_exposure_partition(info.exp[:], n_info, h5file, log=log,
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
            log.warning('LSF QA %s IFUSLOT=%s AMP=%s has inconsistent '
                        'SPECID/IFUID values: %s/%s', op.basename(h5file),
                        slot, amplifier, spec_values, ifu_values)
        records.append({
            'shot': op.basename(h5file),
            'specid': spec_values[0] if spec_values else '',
            'ifuslot': int(slot),
            'ifuid': ifu_values[0] if ifu_values else '',
            'amp': amplifier,
            'ra': float(np.nanmedian(ra_values)),
            'dec': float(np.nanmedian(dec_values)),
            'identity_ok': bool(identity_ok),
        })
    return records


def _map_lsf_centers_to_spaxels(records, tp, xg, yg):
    """Map 1-based WCS coordinates to nearest zero-based output pixels."""
    sample_lookup = np.full((len(yg), len(xg)), -1, dtype=np.int32)
    unique_positions = []
    position_ids = {}
    for record in records:
        xw, yw = tp.wcs_world2pix(record['ra'], record['dec'], 1)
        valid = np.isfinite(xw) and np.isfinite(yw)
        if valid:
            # WCS returns one-based pixel coordinates.  Round in that
            # coordinate system, then explicitly convert to zero-based array
            # indices used by sample_lookup and NumPy images.
            x_one = int(np.rint(xw))
            y_one = int(np.rint(yw))
            x_index = x_one - 1
            y_index = y_one - 1
            valid = (0 <= x_index < len(xg)) and (0 <= y_index < len(yg))
        if not valid:
            record['cube_x'] = -1
            record['cube_y'] = -1
            record['sample_id'] = -1
            continue
        key = (y_index, x_index)
        if key not in position_ids:
            position_ids[key] = len(unique_positions)
            unique_positions.append(key)
            sample_lookup[key] = position_ids[key]
        record['cube_x'] = x_one
        record['cube_y'] = y_one
        record['sample_id'] = position_ids[key]
    return sample_lookup, unique_positions


def _sparse_reconstruct_one_wave(Pos, values, xg, yg, sample_lookup,
                                 shot_indices, sigma=1.8 / 2.35):
    """Return SCI-like sparse exposure median and NCONTRIB for one wavelength."""
    n_samples = int(sample_lookup.max()) + 1 if np.any(sample_lookup >= 0) else 0
    nshots = len(shot_indices)
    shot_values = np.full((nshots, n_samples), np.nan, dtype=np.float32)
    ncontrib = np.zeros(n_samples, dtype=np.uint8)
    area = np.pi * 0.75 ** 2
    for k, cnt in enumerate(shot_indices):
        if isinstance(cnt, (tuple, list)) and len(cnt) == 2:
            indices = np.arange(int(cnt[0]), int(cnt[1]), dtype=np.int64)
        else:
            indices = np.asarray(cnt, dtype=np.int64)
        flux_sum = np.zeros(n_samples, dtype=np.float32)
        weight_sum = np.zeros(n_samples, dtype=np.float32)
        support_map = np.zeros(n_samples, dtype=np.uint8)
        _gaussian_splat_sparse_shot_xy(
            indices, Pos[:, 0], Pos[:, 1], values, sample_lookup,
            float(xg[0]), float(yg[0]), len(xg), len(yg), float(sigma),
            3, int(np.ceil(2.0 * sigma)), float(area), flux_sum, weight_sum,
            support_map)
        supported = (support_map != 0) & (weight_sum > 0.0)
        if np.any(supported):
            shot_values[k, supported] = flux_sum[supported] / weight_sum[supported]
            ncontrib[supported] += 1
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        result = np.nanmedian(shot_values, axis=0).astype(np.float32)
    # Match make_image_gaussian's SCI representation exactly: unsupported
    # pixels are zero in the returned SCI plane.  The production arc profile
    # wrapper masks these entries to NaN before FWHM measurement.
    result[ncontrib < 2] = 0.0
    return result, ncontrib


def _sparse_reconstruct_arc(raarray, decarray, arc_rect, full_indices,
                            sample_lookup, xg, yg, tp, shot_indices):
    """Propagate compact normalized arc spectra through selected spaxels."""
    n_samples = int(sample_lookup.max()) + 1 if np.any(sample_lookup >= 0) else 0
    propagated = np.full((n_samples, len(full_indices)), np.nan, dtype=np.float32)
    sample_ncontrib = np.zeros((n_samples, len(full_indices)), dtype=np.uint8)
    for j, full_index in enumerate(full_indices):
        x, y = tp.wcs_world2pix(raarray[:, full_index],
                                decarray[:, full_index], 1)
        values, ncontrib = _sparse_reconstruct_one_wave(
            np.column_stack((x, y)), arc_rect[:, j], xg, yg, sample_lookup,
            shot_indices)
        values[ncontrib < 2] = np.nan
        propagated[:, j] = values
        sample_ncontrib[:, j] = ncontrib
    return propagated, sample_ncontrib


def _rectify_lsf_arcs(h5files, def_wave, lsf_def_wave,
                      lsf_wave_indices, specarray, log=None):
    """Rectify only requested arc bins, retaining the complete arc profile."""
    arc_rect = np.full((specarray.shape[0], len(lsf_def_wave)), np.nan,
                       dtype=np.float32)
    offset = 0
    for h5file in h5files:
        t = tables.open_file(h5file)
        n_info = int(t.root.Info.nrows)
        n_raw = int(t.root.Raw.nrows)
        if n_info != n_raw:
            t.close()
            raise ValueError('LSF row alignment failure for %s: Info=%d Raw=%d'
                             % (h5file, n_info, n_raw))
        native_arc = t.root.Raw.cols.arcspectrum[:]
        native_wave = t.root.Raw.cols.wave[:]
        if native_arc.shape[0] != n_info or native_wave.shape[0] != n_info:
            t.close()
            raise ValueError('LSF Raw array row alignment failure for %s' % h5file)
        for row in range(n_info):
            compact = np.interp(lsf_def_wave, native_wave[row], native_arc[row],
                                left=np.nan, right=np.nan)
            if row == 0:
                full = np.interp(def_wave, native_wave[row], native_arc[row],
                                 left=np.nan, right=np.nan)
                np.testing.assert_array_equal(compact, full[lsf_wave_indices])
                if log is not None:
                    log.info('LSF rectification validation %s: exact np.interp '
                             'compact/full-bin agreement for representative fiber.',
                             op.basename(h5file))
            arc_rect[offset + row] = compact.astype(np.float32)
        offset += n_info
        t.close()
    if offset != specarray.shape[0]:
        raise ValueError('LSF row alignment failure across files: Raw=%d SCI=%d'
                         % (offset, specarray.shape[0]))
    return arc_rect


def _normalize_lsf_arc_window(arc_rect, lsf_def_wave, reference):
    """Subtract outer-window backgrounds and normalize each fiber's line flux."""
    window = np.abs(lsf_def_wave - reference) <= ARC_WINDOW_ANGSTROM
    left = window & (lsf_def_wave <= reference - 10.0)
    right = window & (lsf_def_wave >= reference + 10.0)
    local = np.where(window)[0]
    normalized = np.full_like(arc_rect, np.nan, dtype=np.float32)
    if local.size == 0:
        return normalized, np.zeros(arc_rect.shape[0], dtype=bool), np.full(arc_rect.shape[0], np.nan)
    left_bg = np.nanmedian(arc_rect[:, left], axis=1) if np.any(left) else np.full(arc_rect.shape[0], np.nan)
    right_bg = np.nanmedian(arc_rect[:, right], axis=1) if np.any(right) else np.full(arc_rect.shape[0], np.nan)
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
    """Flag flat-topped native arc profiles without fitting a saturation model."""
    window = np.abs(lsf_def_wave - reference) <= ARC_WINDOW_ANGSTROM
    local = np.where(window)[0]
    if local.size == 0:
        return {'finite': 0, 'flat_top': 0, 'exact_plateau': 0,
                'suspicious': 0}
    left = window & (lsf_def_wave <= reference - 10.0)
    right = window & (lsf_def_wave >= reference + 10.0)
    left_bg = (np.nanmedian(arc_rect[:, left], axis=1)
               if np.any(left) else np.full(arc_rect.shape[0], np.nan))
    right_bg = (np.nanmedian(arc_rect[:, right], axis=1)
                if np.any(right) else np.full(arc_rect.shape[0], np.nan))
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
    return {'finite': int(np.count_nonzero(finite_rows)),
            'flat_top': int(np.count_nonzero(flat_top)),
            'exact_plateau': int(np.count_nonzero(exact_plateau)),
            'suspicious': int(np.count_nonzero(suspicious))}


def _measure_direct_fwhm(wave, profile, reference):
    """Measure FWHM by PCHIP interpolation and half-height crossings only."""
    wave = np.asarray(wave, dtype=float)
    profile = np.asarray(profile, dtype=float)
    finite = np.isfinite(wave) & np.isfinite(profile)
    core = np.isfinite(wave) & (np.abs(wave - reference) <= 6.0)
    if np.count_nonzero(core) < 2 or not np.all(finite[core]):
        return {'valid': False, 'status': 'UNSUPPORTED_CORE', 'center': np.nan,
                'fwhm': np.nan, 'phase': np.nan}
    if np.count_nonzero(finite) < 5:
        return {'valid': False, 'status': 'TOO_FEW_SAMPLES', 'center': np.nan,
                'fwhm': np.nan, 'phase': np.nan}
    x = wave[finite]
    y = profile[finite]
    order = np.argsort(x)
    x, y = x[order], y[order]
    left_background = np.nanmedian(y[x <= reference - 10.])
    right_background = np.nanmedian(y[x >= reference + 10.])
    if not np.isfinite(left_background) or not np.isfinite(right_background):
        return {'valid': False, 'status': 'NO_BACKGROUND', 'center': np.nan,
                'fwhm': np.nan, 'phase': np.nan}
    background = float(np.nanmedian([left_background, right_background]))
    y = y - background
    search = (x >= reference - 5.0) & (x <= reference + 5.0)
    if np.count_nonzero(search) < 2:
        return {'valid': False, 'status': 'NO_PEAK_NEAR_LINE', 'center': np.nan,
                'fwhm': np.nan, 'phase': np.nan}
    try:
        interpolator = PchipInterpolator(x, y, extrapolate=False)
    except (ValueError, TypeError):
        return {'valid': False, 'status': 'INTERPOLATION_FAILED', 'center': np.nan,
                'fwhm': np.nan, 'phase': np.nan}
    sampled_search = np.where(search)[0]
    imax = int(sampled_search[np.argmax(y[sampled_search])])
    if (imax <= 0 or imax >= len(x) - 1 or
            not np.isfinite(x[imax - 1]) or not np.isfinite(x[imax + 1]) or
            not np.isfinite(y[imax - 1]) or not np.isfinite(y[imax + 1])):
        return {'valid': False, 'status': 'INVALID_PEAK_INTERPOLATION',
                'center': np.nan, 'fwhm': np.nan, 'phase': np.nan}
    ym, y0, yp = y[imax - 1], y[imax], y[imax + 1]
    dx_left = x[imax] - x[imax - 1]
    dx_right = x[imax + 1] - x[imax]
    if (not np.isfinite(dx_left) or not np.isfinite(dx_right) or
            dx_left <= 0.0 or not np.isclose(dx_left, dx_right,
                                              rtol=1.e-7, atol=1.e-10)):
        return {'valid': False, 'status': 'INVALID_PEAK_INTERPOLATION',
                'center': np.nan, 'fwhm': np.nan, 'phase': np.nan}
    denom = ym - 2.0 * y0 + yp
    denom_tolerance = (100.0 * np.finfo(float).eps *
                       max(1.0, abs(ym), abs(y0), abs(yp)))
    if not np.isfinite(denom) or denom >= 0.0 or abs(denom) <= denom_tolerance:
        return {'valid': False, 'status': 'INVALID_PEAK_INTERPOLATION',
                'center': np.nan, 'fwhm': np.nan, 'phase': np.nan}
    delta_pix = 0.5 * (ym - yp) / denom
    if not np.isfinite(delta_pix) or abs(delta_pix) > 1.0:
        return {'valid': False, 'status': 'INVALID_PEAK_INTERPOLATION',
                'center': np.nan, 'fwhm': np.nan, 'phase': np.nan}
    dx = 0.5 * (dx_left + dx_right)
    center = float(x[imax] + delta_pix * dx)
    peak_height = float(y0 - (yp - ym) ** 2 / (8.0 * denom))
    if not np.isfinite(peak_height) or peak_height <= 0.0:
        return {'valid': False, 'status': 'NONPOSITIVE_LINE', 'center': np.nan,
                'fwhm': np.nan, 'phase': np.nan}

    dense = np.linspace(x[0], x[-1], max(401, int(np.ceil((x[-1] - x[0]) * 100)) + 1))
    dense_y = np.asarray(interpolator(dense), dtype=float)
    search_dense = (dense >= reference - 5.0) & (dense <= reference + 5.0)
    if not np.any(search_dense) or not np.any(np.isfinite(dense_y[search_dense])):
        return {'valid': False, 'status': 'NO_PEAK_NEAR_LINE', 'center': np.nan,
                'fwhm': np.nan, 'phase': np.nan}
    peak_candidates = np.where(
        search_dense & np.isfinite(dense_y) &
        (dense_y >= np.roll(dense_y, 1)) &
        (dense_y >= np.roll(dense_y, -1)))[0]
    peak_candidates = peak_candidates[(peak_candidates > 0) &
                                      (peak_candidates < len(dense) - 1)]
    strong = peak_candidates[dense_y[peak_candidates] >= 0.80 * peak_height]
    if strong.size > 1:
        separated = [idx for idx in strong if abs(dense[idx] - center) >= 1.0]
        if separated:
            return {'valid': False, 'status': 'AMBIGUOUS_BLEND', 'center': np.nan,
                    'fwhm': np.nan, 'phase': np.nan}
    half = 0.5 * peak_height
    peak_index = int(np.argmin(np.abs(dense - center)))

    def crossing(direction):
        if direction < 0:
            indices = range(peak_index - 1, -1, -1)
        else:
            indices = range(peak_index, len(dense) - 1)
        for idx in indices:
            j = idx + 1 if direction > 0 else idx
            k = idx if direction > 0 else idx + 1
            y0, y1 = dense_y[j] - half, dense_y[k] - half
            if np.isfinite(y0) and np.isfinite(y1) and y0 * y1 <= 0.0:
                lo, hi = min(dense[j], dense[k]), max(dense[j], dense[k])
                flo, fhi = float(interpolator(lo) - half), float(interpolator(hi) - half)
                for _ in range(45):
                    mid = 0.5 * (lo + hi)
                    fmid = float(interpolator(mid) - half)
                    if flo * fmid <= 0.0:
                        hi, fhi = mid, fmid
                    else:
                        lo, flo = mid, fmid
                return 0.5 * (lo + hi)
        return None

    left_cross = crossing(-1)
    right_cross = crossing(1)
    if left_cross is None:
        return {'valid': False, 'status': 'NO_LEFT_CROSSING', 'center': np.nan,
                'fwhm': np.nan, 'phase': np.nan}
    if right_cross is None:
        return {'valid': False, 'status': 'NO_RIGHT_CROSSING', 'center': np.nan,
                'fwhm': np.nan, 'phase': np.nan}
    if left_cross <= x[0] + 1.e-6 or right_cross >= x[-1] - 1.e-6:
        return {'valid': False, 'status': 'CROSSING_AT_BOUNDARY', 'center': np.nan,
                'fwhm': np.nan, 'phase': np.nan}
    center = float(dense[peak_index])
    nearest_grid = 3470. + 2. * np.rint((center - 3470.) / 2.)
    phase = float(center - nearest_grid)
    return {'valid': True, 'status': 'OK', 'center': center,
            'fwhm': float(right_cross - left_cross), 'phase': phase}


def _lsf_spatial_header(tp, n, pixel_scale):
    """Construct the final cube's two-dimensional spatial WCS."""
    spatial = tp.to_header()
    scale = pixel_scale / 3600.
    spatial['WCSAXES'] = 2
    spatial['CD1_1'] = -scale
    spatial['CD1_2'] = 0.0
    spatial['CD2_1'] = 0.0
    spatial['CD2_2'] = scale
    for key in ('CDELT1', 'CDELT2', 'CDELT3', 'CRPIX3', 'CRVAL3',
                'CTYPE3', 'CUNIT3', 'SPECSYS'):
        if key in spatial:
            del spatial[key]
    spatial['CRPIX1'] = (n + 1) / 2.
    spatial['CRPIX2'] = (n + 1) / 2.
    spatial['CTYPE1'] = 'RA---TAN'
    spatial['CTYPE2'] = 'DEC--TAN'
    spatial['CUNIT1'] = 'deg'
    spatial['CUNIT2'] = 'deg'
    return spatial


def _phase_summary(phases, fwhm):
    """Return a transparent binned phase diagnostic, not a correction."""
    finite = np.isfinite(phases) & np.isfinite(fwhm)
    if np.count_nonzero(finite) < 10:
        return np.nan, np.nan, False
    x = phases[finite]
    y = fwhm[finite]
    edges = np.linspace(-1., 1., 9)
    medians = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        selected = (x >= lo) & (x < hi if hi < edges[-1] else x <= hi)
        if np.count_nonzero(selected) >= 3:
            medians.append(float(np.nanmedian(y[selected])))
    phase_range = (float(np.nanmax(medians) - np.nanmin(medians))
                   if len(medians) >= 2 else np.nan)
    correlation = (float(np.corrcoef(x, y)[0, 1])
                   if np.std(x) > 0. and np.std(y) > 0. else np.nan)
    median = float(np.nanmedian(y))
    evidence = bool(np.isfinite(phase_range) and np.isfinite(median) and
                    median != 0. and phase_range > 0.03 * abs(median))
    return phase_range, correlation, evidence


def _run_lsf_measurement(h5files, surname, def_wave, raarray, decarray,
                         specarray, shot_indices, shot_names, tp, xg, yg,
                         xgrid, ygrid, cube, ncontribcube, pixel_scale, log,
                         records=None):
    """Run the optional sparse empirical effective-resolution experiment."""
    if records is None:
        records = []
        for h5file in h5files:
            t = tables.open_file(h5file)
            records.extend(_build_lsf_amplifier_centers(h5file, t, log=log))
            t.close()
    sample_lookup, unique_positions = _map_lsf_centers_to_spaxels(records, tp, xg, yg)
    log.info('LSF amplifier-center sampling: %d centers, %d unique in-cube spaxels.',
             len(records), len(unique_positions))
    if not unique_positions:
        raise RuntimeError('LSF requested but no amplifier centers map into the cube.')

    line_masks = [np.abs(def_wave - reference) <= ARC_WINDOW_ANGSTROM
                  for reference in REFERENCE_ARC_WAVELENGTHS]
    lsf_wave_indices = np.unique(np.concatenate(
        [np.where(mask)[0] for mask in line_masks])).astype(np.int64)
    lsf_def_wave = def_wave[lsf_wave_indices]
    log.info('LSF compact spectral grid: %d of %d science bins.',
             len(lsf_def_wave), len(def_wave))
    arc_rect = _rectify_lsf_arcs(
        h5files, def_wave, lsf_def_wave, lsf_wave_indices, specarray, log=log)

    normalized_by_line = []
    for line_number, reference in enumerate(REFERENCE_ARC_WAVELENGTHS):
        clipping = _native_arc_clipping_diagnostic(
            arc_rect, lsf_def_wave, reference)
        log.info('LSF native arc diagnostic %0.3f A: usable native profiles=%d, '
                 'flat-top profiles=%d, exact plateaus=%d, suspicious=%d. '
                 'These are QA flags only; no saturation model or rejection applied.',
                 reference, clipping['finite'], clipping['flat_top'],
                 clipping['exact_plateau'], clipping['suspicious'])
        normalized, profile_valid, signal = _normalize_lsf_arc_window(
            arc_rect, lsf_def_wave, reference)
        normalized_by_line.append(normalized)
        if line_number in (0, len(REFERENCE_ARC_WAVELENGTHS) // 2,
                           len(REFERENCE_ARC_WAVELENGTHS) - 1):
            finite_signal = signal[np.isfinite(signal) & (signal > 0.)]
            if finite_signal.size:
                p16, p50, p84 = np.nanpercentile(finite_signal, [16, 50, 84])
                log.info('LSF native arc amplitude %0.3f A: positive integrated '
                         'line signal N=%d p16/p50/p84=%0.5g/%0.5g/%0.5g '
                         'before unit-flux normalization.', reference,
                         finite_signal.size, p16, p50, p84)
            else:
                log.warning('LSF native arc amplitude %0.3f A: no positive '
                            'integrated line signals before normalization.', reference)
    log.info('LSF arc normalization: local outer-window background subtraction '
             'and unit integrated line flux; no assumption of pre-normalized '
             'Raw.arcspectrum was made.')

    fwhm_by_line = []
    center_by_line = []
    phase_by_line = []
    status_by_line = []
    ncontrib_by_line = []
    summary = []
    for line_number, reference in enumerate(REFERENCE_ARC_WAVELENGTHS):
        local = np.abs(lsf_def_wave - reference) <= ARC_WINDOW_ANGSTROM
        local_wave = lsf_def_wave[local]
        line_indices = lsf_wave_indices[local]
        normalized_arc = normalized_by_line[line_number][:, local].copy()
        # Availability is deliberately imposed after line normalization.  A
        # missing science bin therefore removes that bin's contribution but
        # cannot renormalize the surviving arc wings.
        for compact_column, full_index in enumerate(line_indices):
            normalized_arc[~np.isfinite(specarray[:, full_index]), compact_column] = np.nan
        propagated, sample_ncontrib = _sparse_reconstruct_arc(
            raarray, decarray, normalized_arc, line_indices,
            sample_lookup, xg, yg, tp, shot_indices)
        fwhm = np.full(len(unique_positions), np.nan, dtype=float)
        center = np.full(len(unique_positions), np.nan, dtype=float)
        phase = np.full(len(unique_positions), np.nan, dtype=float)
        status = np.full(len(unique_positions), 'NO_PROFILE', dtype='U32')
        nnear = np.full(len(unique_positions), -1, dtype=np.int16)
        profile_valid = np.zeros(len(unique_positions), dtype=bool)
        for sample_id in range(len(unique_positions)):
            # ``propagated`` is already limited to this line's local columns.
            profile = propagated[sample_id]
            support = sample_ncontrib[sample_id]
            nfinite = np.isfinite(profile)
            if np.any(nfinite):
                nnear[sample_id] = int(np.nanmedian(support[nfinite]))
            core = np.abs(local_wave - reference) <= 6.0
            if (np.count_nonzero(core) >= 2 and
                    np.all(np.isfinite(profile[core])) and
                    np.all(support[core] >= 2)):
                profile_valid[sample_id] = True
                result = _measure_direct_fwhm(local_wave, profile, reference)
            else:
                result = {'valid': False, 'status': 'INSUFFICIENT_SUPPORT',
                          'center': np.nan, 'fwhm': np.nan, 'phase': np.nan}
            status[sample_id] = result['status']
            if result['valid']:
                fwhm[sample_id] = result['fwhm']
                center[sample_id] = result['center']
                phase[sample_id] = result['phase']

        fwhm_by_line.append(fwhm)
        center_by_line.append(center)
        phase_by_line.append(phase)
        status_by_line.append(status)
        ncontrib_by_line.append(nnear)
        valid = np.isfinite(fwhm)
        phase_range, phase_corr, phase_evidence = _phase_summary(phase, fwhm)
        values = fwhm[valid]
        centers = center[valid]
        if values.size:
            p16, p50, p84 = np.nanpercentile(values, [16, 50, 84])
            scatter = 1.4826 * np.nanmedian(np.abs(values - p50))
            center_offset = centers - reference
            log.info('LSF %0.3f A: centers=%d, propagated profiles=%d/%d '
                     '(%0.4f), valid FWHM=%d/%d (%0.4f), '
                     'FWHM p16/p50/p84=%0.4f/%0.4f/%0.4f, '
                     'robust spatial scatter=%0.4f, center offset median/range='
                     '%0.4f/%0.4f A, phase-bin range=%s A, phase r=%s, '
                     'phase evidence=%s, laboratory blend=%s, usable_for_lsf=%s',
                     reference,
                     len(records), int(np.count_nonzero(profile_valid)),
                     len(unique_positions),
                     np.count_nonzero(profile_valid) / float(max(1, len(unique_positions))),
                     values.size, len(unique_positions),
                     values.size / float(max(1, len(unique_positions))),
                     p16, p50, p84, scatter, np.nanmedian(center_offset),
                     np.nanmax(center_offset) - np.nanmin(center_offset),
                     ('%0.4f' % phase_range if np.isfinite(phase_range) else 'nan'),
                     ('%0.4f' % phase_corr if np.isfinite(phase_corr) else 'nan'),
                     phase_evidence, ARC_BLEND_STATES[line_number],
                     bool(ARC_USABLE_FOR_LSF[line_number]))
            summary.append((reference, p16, p50, p84, scatter, phase_range,
                            phase_corr, phase_evidence))
        else:
            log.warning('LSF %0.3f A: no valid direct FWHM measurements; '
                        'laboratory blend=%s, usable_for_lsf=%s', reference,
                        ARC_BLEND_STATES[line_number],
                        bool(ARC_USABLE_FOR_LSF[line_number]))
            summary.append((reference, np.nan, np.nan, np.nan, np.nan,
                            np.nan, np.nan, False))

        # The validation deliberately exercises the full image estimator only
        # at three compact wavelengths; production values above came only from
        # the sparse evaluator.
        if line_number == 0:
            local_columns = np.where(local)[0]
            validation_columns = np.unique(
                np.linspace(0, len(local_columns) - 1,
                            min(3, len(local_columns)), dtype=int))
            for local_column in validation_columns:
                compact_column = local_columns[local_column]
                full_index = int(lsf_wave_indices[compact_column])
                x, y = tp.wcs_world2pix(raarray[:, full_index],
                                        decarray[:, full_index], 1)
                full = make_image_gaussian(
                    np.column_stack((x, y)), normalized_arc[:, local_column],
                    None, xg, yg, xgrid, ygrid, 1.8 / 2.35, shot_indices)
                sparse, sparse_n = _sparse_reconstruct_one_wave(
                    np.column_stack((x, y)), normalized_arc[:, local_column],
                    xg, yg, sample_lookup, shot_indices)
                for sample_id, (py, px) in enumerate(unique_positions):
                    np.testing.assert_allclose(sparse[sample_id], full[0][py, px],
                                               rtol=2.e-6, atol=2.e-6)
                    assert int(sparse_n[sample_id]) == int(full[3][py, px])
            log.info('LSF sparse/full validation: SCI and NCONTRIB agree at '
                     'representative wavelengths and all selected sample spaxels.')

    # Build the requested one-row-per-amplifier-center x reference-line table.
    nrows = len(records) * len(REFERENCE_ARC_WAVELENGTHS)
    table = Table()
    table['shot'] = np.array([r['shot'] for r in records for _ in REFERENCE_ARC_WAVELENGTHS], dtype='U80')
    table['specid'] = np.array([r['specid'] for r in records for _ in REFERENCE_ARC_WAVELENGTHS], dtype='U16')
    table['ifuslot'] = np.array([r['ifuslot'] for r in records for _ in REFERENCE_ARC_WAVELENGTHS], dtype=np.int32)
    table['ifuid'] = np.array([r['ifuid'] for r in records for _ in REFERENCE_ARC_WAVELENGTHS], dtype='U16')
    table['amp'] = np.array([r['amp'] for r in records for _ in REFERENCE_ARC_WAVELENGTHS], dtype='U8')
    table['identity_ok'] = np.array([r['identity_ok'] for r in records for _ in REFERENCE_ARC_WAVELENGTHS], dtype=bool)
    table['ra'] = np.array([r['ra'] for r in records for _ in REFERENCE_ARC_WAVELENGTHS], dtype=float)
    table['dec'] = np.array([r['dec'] for r in records for _ in REFERENCE_ARC_WAVELENGTHS], dtype=float)
    table['cube_x'] = np.array([r['cube_x'] for r in records for _ in REFERENCE_ARC_WAVELENGTHS], dtype=np.int32)
    table['cube_y'] = np.array([r['cube_y'] for r in records for _ in REFERENCE_ARC_WAVELENGTHS], dtype=np.int32)
    table['sample_id'] = np.array([r['sample_id'] for r in records for _ in REFERENCE_ARC_WAVELENGTHS], dtype=np.int32)
    table['reference_wavelength'] = np.tile(REFERENCE_ARC_WAVELENGTHS, len(records))
    table['measured_center'] = np.array([center_by_line[j][r['sample_id']] if r['sample_id'] >= 0 else np.nan
                                         for r in records for j in range(len(REFERENCE_ARC_WAVELENGTHS))], dtype=float)
    table['fwhm'] = np.array([fwhm_by_line[j][r['sample_id']] if r['sample_id'] >= 0 else np.nan
                              for r in records for j in range(len(REFERENCE_ARC_WAVELENGTHS))], dtype=float)
    table['grid_phase'] = np.array([phase_by_line[j][r['sample_id']] if r['sample_id'] >= 0 else np.nan
                                    for r in records for j in range(len(REFERENCE_ARC_WAVELENGTHS))], dtype=float)
    table['ncontrib'] = np.array([ncontrib_by_line[j][r['sample_id']] if r['sample_id'] >= 0 else -1
                                  for r in records for j in range(len(REFERENCE_ARC_WAVELENGTHS))], dtype=np.int16)
    table['valid'] = np.array([bool(r['sample_id'] >= 0 and np.isfinite(fwhm_by_line[j][r['sample_id']]))
                               for r in records for j in range(len(REFERENCE_ARC_WAVELENGTHS))], dtype=bool)
    table['usable_for_lsf'] = np.array([
        bool(ARC_USABLE_FOR_LSF[j] and r['sample_id'] >= 0 and
             np.isfinite(fwhm_by_line[j][r['sample_id']]))
        for r in records for j in range(len(REFERENCE_ARC_WAVELENGTHS))], dtype=bool)
    table['status'] = np.array([status_by_line[j][r['sample_id']] if r['sample_id'] >= 0 else 'OUTSIDE_CUBE'
                                for r in records for j in range(len(REFERENCE_ARC_WAVELENGTHS))], dtype='U32')
    table['lab_blend'] = np.array([
        ARC_BLEND_STATES[j] == 'KNOWN_BLEND' for _ in records
        for j in range(len(REFERENCE_ARC_WAVELENGTHS))], dtype=bool)
    table['lab_blend_state'] = np.array([
        ARC_BLEND_STATES[j] for _ in records
        for j in range(len(REFERENCE_ARC_WAVELENGTHS))], dtype='U16')
    table['blend_note'] = np.array([ARC_BLEND_NOTES[float(REFERENCE_ARC_WAVELENGTHS[j])]
                                    for _ in records for j in range(len(REFERENCE_ARC_WAVELENGTHS))], dtype='U96')
    sample_path = op.basename('%s_lsf_samples.fits' % surname)
    table.write(sample_path, format='fits', overwrite=True)

    spatial_header = _lsf_spatial_header(tp, len(xg), pixel_scale)
    public_line_indices = np.where(ARC_USABLE_FOR_LSF)[0]
    for line_number in public_line_indices:
        reference = REFERENCE_ARC_WAVELENGTHS[line_number]
        valid = np.isfinite(fwhm_by_line[line_number])
        image = np.full(xgrid.shape, np.nan, dtype=np.float32)
        if np.any(valid):
            points = np.array([(unique_positions[sid][1], unique_positions[sid][0])
                               for sid in np.where(valid)[0]], dtype=float)
            values = fwhm_by_line[line_number][valid]
            image[:, :] = griddata(points, values, (xgrid - 1., ygrid - 1.),
                                   method='nearest').astype(np.float32)
        science_index = int(np.argmin(np.abs(def_wave - reference)))
        image[ncontribcube[science_index] < 2] = np.nan
        header = spatial_header.copy()
        header['REFWAVE'] = (float(reference), 'laboratory air wavelength [Angstrom]')
        header['WAVECONV'] = ('AIR', 'Hg/Cd reference wavelength convention')
        header['METHOD'] = ('DIRECT/NONPARAMETRIC', 'PCHIP peak and half-height crossings')
        header['INTERP'] = ('NEAREST', 'nearest-neighbor spatial sample interpolation')
        header['LSFMODEL'] = ('NONE', 'profile shape not specified')
        header['BUNIT'] = 'Angstrom'
        header.add_history('Effective cube FWHM from propagated master arcs.')
        output = op.basename('%s_lsf_fwhm_%04d.fits' % (surname, int(np.floor(reference + .5))))
        fits.PrimaryHDU(image, header=header).writeto(output, overwrite=True)

    # Compact diagnostics: raw center locations, FWHM values, medians versus
    # wavelength, and FWHM versus phase on the two-Angstrom grid.
    fig, axes = plt.subplots(2, 2, figsize=(13, 10), constrained_layout=True)
    amp_colors = {'LL': 'tab:blue', 'LU': 'tab:orange',
                  'RL': 'tab:green', 'RU': 'tab:red'}
    for amplifier, color in amp_colors.items():
        amplifier_records = [r for r in records
                             if r['sample_id'] >= 0 and r['amp'] == amplifier]
        axes[0, 0].scatter([r['cube_x'] for r in amplifier_records],
                           [r['cube_y'] for r in amplifier_records],
                           s=4, alpha=.45, color=color, label=amplifier)
    axes[0, 0].set_title('Amplifier-center sample spaxels')
    axes[0, 0].set_aspect('equal', adjustable='box')
    axes[0, 0].set_xlabel('cube x pixel (1-based)')
    axes[0, 0].set_ylabel('cube y pixel (1-based)')
    axes[0, 0].legend(title='amplifier', fontsize=8, markerscale=1.5)
    median_sample = np.nanmedian(np.vstack([fwhm_by_line[i] for i in public_line_indices]), axis=0)
    good_sample = np.isfinite(median_sample)
    if np.any(good_sample):
        axes[0, 1].scatter([unique_positions[sid][1] + 1 for sid in np.where(good_sample)[0]],
                           [unique_positions[sid][0] + 1 for sid in np.where(good_sample)[0]],
                           c=median_sample[good_sample], s=5, cmap='viridis')
    axes[0, 1].set_title('Median valid FWHM at sample spaxels')
    axes[0, 1].set_aspect('equal', adjustable='box')
    axes[0, 1].set_xlabel('cube x pixel (1-based)')
    axes[0, 1].set_ylabel('cube y pixel (1-based)')
    # Include the two known blends in these diagnostic plots so their broad
    # QA widths remain visible, while keeping them out of the public summary.
    for line_number, reference in enumerate(REFERENCE_ARC_WAVELENGTHS):
        values = fwhm_by_line[line_number]
        blend = ARC_BLEND_STATES[line_number] == 'KNOWN_BLEND'
        axes[1, 0].plot(
            reference,
            np.nanmedian(values) if np.any(np.isfinite(values)) else np.nan,
            'x' if blend else 'o',
            color='tab:red' if blend else 'tab:blue',
            label='%0.0f blend QA-only' % reference if blend else None)
        axes[1, 1].scatter(phase_by_line[line_number], values, s=3, alpha=.25,
                           label=('%0.0f blend QA-only' % reference)
                           if blend else '%0.0f' % reference)
    axes[1, 0].set_title('Median FWHM versus reference wavelength (blend points QA-only)')
    axes[1, 0].set_xlabel('reference wavelength [A]')
    axes[1, 0].set_ylabel('FWHM [A]')
    axes[1, 0].legend(fontsize=7, ncol=2)
    axes[1, 1].set_title('FWHM versus 2-Angstrom grid phase (blend QA-only)')
    axes[1, 1].set_xlabel('center minus nearest grid bin [A]')
    axes[1, 1].set_ylabel('FWHM [A]')
    axes[1, 1].legend(fontsize=7, ncol=3)
    qa_path = op.basename('%s_lsf_qa.png' % surname)
    fig.savefig(qa_path, dpi=180)
    plt.close(fig)

    public_measured_count = sum(
        np.any(np.isfinite(fwhm_by_line[i])) for i in public_line_indices)
    measured_count = sum(
        np.any(np.isfinite(fwhm_by_line[i]))
        for i in range(len(REFERENCE_ARC_WAVELENGTHS)))
    all_values = np.concatenate([
        fwhm_by_line[i][np.isfinite(fwhm_by_line[i])]
        for i in public_line_indices if np.any(np.isfinite(fwhm_by_line[i]))
    ]) if any(np.any(np.isfinite(fwhm_by_line[i])) for i in public_line_indices) else np.array([])
    if all_values.size:
        median_values = np.array([summary[i][2] for i in public_line_indices
                                  if np.isfinite(summary[i][2])])
        median_change = (np.nanmax(median_values) - np.nanmin(median_values)
                         if median_values.size else np.nan)
        typical_scatter = np.nanmedian(
            [summary[i][4] for i in public_line_indices
             if np.isfinite(summary[i][4])])
        log.info('LSF overall: %d/%d public reference lines measured; unique evaluated '
                 'spaxels=%d; FWHM range=%0.4f-%0.4f A; spatial robust scatter '
                 'per line=%s; peak-to-peak change in median FWHM=%0.4f A; '
                 'typical spatial scatter=%0.4f A. The prior ~5.35-5.39 A '
                 'single-CCD result is treated only as a sanity check, not a '
                 'constraint. Anchor accounting: 9/9 windows propagated, '
                 '%d/9 measured, 7/9 usable isolated-line anchors, 2/9 '
                 'excluded as known blends.',
                 public_measured_count, len(public_line_indices),
                 len(unique_positions), np.nanmin(all_values), np.nanmax(all_values),
                 ', '.join('%0.4f' % summary[i][4] for i in public_line_indices
                           if np.isfinite(summary[i][4])), median_change, typical_scatter,
                 measured_count)
    else:
        log.warning('LSF overall: no valid public-anchor FWHM values; '
                    'anchor accounting: 9/9 windows propagated, %d/9 measured, '
                    '7/9 usable isolated-line anchors, 2/9 excluded as known blends.',
                    measured_count)
    return table, summary, len(unique_positions)

args = parser.parse_args(args=None)
if args.wave_workers < 1:
    parser.error('--wave-workers must be at least 1')
args.log = setup_logging('make_image_from_h5')

def_wave = np.linspace(3470., 5540., 1036)

h5files = sorted(glob.glob(args.h5files))
args.log.info(f"Detected {len(h5files)} input file(s). Assuming each h5 contains 3 interleaved dithers by default.")

bounding_box = [float(corner.replace(' ', ''))
                        for corner in args.image_center_size.split(',')]

bounding_box[2] = int(bounding_box[2]*60./args.pixel_scale/2.) * 2 * args.pixel_scale

bb = int(bounding_box[2]/args.pixel_scale/2.)*args.pixel_scale
N = int(bounding_box[2]/args.pixel_scale/2.) * 2 + 1
args.log.info('Image size in pixels: %i' % N)
xg = np.arange(N) + 1
yg = np.arange(N) + 1
xgrid, ygrid = np.meshgrid(xg, yg)

cube = np.zeros((len(def_wave),) + xgrid.shape, dtype='float32')
variancecube = np.zeros((len(def_wave),) + xgrid.shape, dtype='float32')
weightcube = np.zeros((len(def_wave),) + xgrid.shape, dtype='float32')
ncontribcube = np.zeros((len(def_wave),) + xgrid.shape, dtype='uint8')
dqcube = np.zeros((len(def_wave),) + xgrid.shape, dtype='uint16')

cnt = 0
cnt_array = np.zeros((len(h5files), 2), dtype=int)
# Build per-exposure (shot) index selections assuming 3 interleaved dithers per h5
nexp_default = 3
shot_indices = []  # list of numpy arrays of global indices for each exposure
shot_names = []
lsf_center_records = []
for i, h5file in enumerate(h5files):
    t = tables.open_file(h5file)
    ra = t.root.Info.cols.ra[:]
    n_fib = len(ra)
    # window for this file in the global arrays
    start = cnt
    end = cnt + n_fib
    cnt_array[i, 0] = start
    cnt_array[i, 1] = end
    # Build local indices grouped by 112-fiber blocks, split into 3 exposures
    inds_local = np.arange(n_fib)
    block_ids = (inds_local // 112).astype(int)
    for k in range(nexp_default):
        sel_local = np.where((block_ids % nexp_default) == k)[0]
        if sel_local.size > 0:
            shot_indices.append(sel_local + start)
            shot_names.append('%s exposure %d' % (op.basename(h5file), k + 1))
    if args.make_lsf:
        lsf_center_records.extend(_build_lsf_amplifier_centers(h5file, t, log=args.log))
    cnt = end
    t.close()
args.log.info('Number of total fibers: %i' % cnt)
args.log.info(f'Total shots (exposures) assumed: {len(shot_indices)} (3 per h5file).')
if args.make_lsf:
    args.log.info('LSF QA retained %d amplifier-center records before WCS mapping.',
                  len(lsf_center_records))

raarray = np.zeros((cnt, len(def_wave)), dtype='float32')
decarray = np.zeros((cnt, len(def_wave)), dtype='float32')
specarray = np.zeros((cnt, len(def_wave)), dtype='float32')
errarray = np.zeros((cnt, len(def_wave)), dtype='float32')

# Astrometry with CRPIX at image center (not lower-left)
_x0 = (N + 1) / 2.0
_y0 = (N + 1) / 2.0
A = Astrometry(bounding_box[0], bounding_box[1], 0., _x0, _y0)
# Ensure TP uses same centered CRPIX
tp = A.setup_TP(A.ra0, A.dec0, 0., x0=_x0, y0=_y0)

# Preflight coverage check: ensure some fibers land within requested region
try:
    xmin, xmax = xg.min(), xg.max()
    ymin, ymax = yg.min(), yg.max()
    in_region_total = 0
    for h5file in h5files:
        t = tables.open_file(h5file)
        ra_chk = t.root.Info.cols.ra[:]
        dec_chk = t.root.Info.cols.dec[:]
        t.close()
        x_chk, y_chk = tp.wcs_world2pix(ra_chk, dec_chk, 1)
        mask_in = (x_chk >= xmin) & (x_chk <= xmax) & (y_chk >= ymin) & (y_chk <= ymax)
        n_in = int(mask_in.sum())
        args.log.info(f'Fibers in region for {op.basename(h5file)}: {n_in}')
        in_region_total += n_in
    if in_region_total == 0:
        args.log.error('No fibers found within the requested cube region. '
                       'The output would be all zeros. Aborting. '\
                       f"Center=({bounding_box[0]:.5f},{bounding_box[1]:.5f}), size={bounding_box[2]:.2f} arcsec")
        sys.exit(2)
    if in_region_total < 10:
        args.log.warning(f'Only {in_region_total} fibers fall within the region; output may be very noisy or near zero.')
except Exception as e:
    args.log.warning(f'Coverage precheck failed with {e}; proceeding anyway.')

if args.filter_file is not None:
    R = Table.read(args.filter_file, format='ascii')
    response = np.interp(def_wave, R['Wavelength'], R['R'], left=0.0, right=0.0)
    

    
binimage = None
if args.image_file is not None:
    name = op.basename(args.image_file)[:-5] + '_rect.fits'
    if op.exists(name):
        f = fits.open(name)
        binimage = f[0].data
    else:
        image_file = fits.open(args.image_file)
        wc = WCS(image_file[0].header)
        ny, nx = image_file[0].data.shape
        yind, xind = np.indices((ny, nx))
        xn, yn = wc.wcs_world2pix(A.ra0, A.dec0, 1)
        tpn = A.setup_TP(A.ra0, A.dec0, 0., xn, yn, x_scale=-0.25, y_scale=0.25)
        r, d = wc.wcs_pix2world(xind.ravel()+1.0, yind.ravel()+1.0, 1)
        P = np.zeros((len(r), 2))
        x, y = tp.wcs_world2pix(r, d, 1)
        P[:, 0], P[:, 1] = (x, y)
        d = np.sqrt((P[:, 0]-xg[0])**2 + (P[:, 1]-yg[0])**2)
        I = np.argmin(d)
        yi, xi = np.unravel_index(I, yind.shape)
        N = 4 * len(xg)
        x = np.reshape(x, image_file[0].data.shape)
        y = np.reshape(y, image_file[0].data.shape)
        newimage = image_file[0].data[yi:yi+N, xi:xi+N]
        binimage = rebin(newimage, (newimage.shape[0]//4, newimage.shape[1]//4))
        ximage = rebin(x[yi:yi+N, xi:xi+N], (newimage.shape[0]//4, newimage.shape[1]//4))
        yimage = rebin(y[yi:yi+N, xi:xi+N], (newimage.shape[0]//4, newimage.shape[1]//4))
        P = np.zeros((len(ximage.ravel()), 2))
        P[:, 0], P[:, 1] = (ximage.ravel(), yimage.ravel())
        binimage = griddata(P, binimage.ravel(), (xgrid, ygrid), method='cubic')
        h = tp.to_header()
        N = len(xg)
        h['CRPIX1'] = np.interp(0., xg, np.arange(len(xg)))+1.0
        h['CRPIX2'] = np.interp(0., xg, np.arange(len(xg)))+1.0
        args.log.info('Writing %s' % name)
        fits.PrimaryHDU(np.array(binimage, dtype='float32'), header=h).writeto(name, overwrite=True)

cnt = 0
norm_array = np.ones((len(h5files),))
for jk, h5file in enumerate(h5files):
    args.log.info('Working on %s' % h5file)
    t = tables.open_file(h5file)
    date = int(op.basename(h5file).split('_')[0])
    ifuslots = t.root.Info.cols.ifuslot[:]
    amps = np.array([i.decode("utf-8") for i in t.root.Info.cols.amp[:]])
                
            
    ra = t.root.Info.cols.ra[:]
    dec = t.root.Info.cols.dec[:]
    RA = t.root.Survey.cols.ra[0]
    Dec = t.root.Survey.cols.dec[0]
    pa = t.root.Survey.cols.pa[0]
    offset = t.root.Survey.cols.offset[0]
    if (not np.isfinite(offset)) or (offset == 0):
        args.log.warning(f'Offset for {op.basename(h5file)} is invalid ({offset}); proceeding without offset correction.')
        spectra = t.root.Fibers.cols.spectrum[:]
        error = t.root.Fibers.cols.error[:]
    else:
        args.log.info(f'Offset for {op.basename(h5file)}: {offset}')
        # Preserve the existing H5 units: Quick Reduction applied the final
        # photometric offset to SCI/error, so undo it before cube normalization.
        spectra = t.root.Fibers.cols.spectrum[:] / offset
        error = t.root.Fibers.cols.error[:] / offset
    for key in mask_dict.keys():
        date1 = int(key.split('-')[0])
        date2 = int(key.split('-')[1])
        if (date >= date1) and (date < date2):
            ifulist = mask_dict[key]
            for ifuamp in ifulist:
                ifu = int(ifuamp[:3])
                amp = ifuamp[3:]
                sel = np.where((ifu == ifuslots) * (amp == amps))[0]
                spectra[sel] = np.nan

    # Subtract the single authoritative M101 residual-sky correction before
    # the existing mosaic normalization.  This uses H5 sky-subtracted SCI
    # directly; Fibers.skyspectrum is intentionally not added back.  This
    # replaces the former H5-wide backspectra subtraction below normalization.
    spectra = subtract_m101_residual_sky(
        spectra, ra, dec, xg, yg, tp, binimage=binimage, log=args.log,
        h5file=op.basename(h5file))
    cnt1 = cnt + len(ra)
    E = Extract()
    Aother = Astrometry(bounding_box[0], bounding_box[1], pa, 0., 0.)
    header = tp.to_header()
    E.get_ADR_RAdec(Aother)
    raarray[cnt:cnt1, :] = ra[:, np.newaxis] - E.ADRra[np.newaxis, :] / 3600. / np.cos(np.deg2rad(A.dec0))
    decarray[cnt:cnt1, :] = dec[:, np.newaxis] - E.ADRdec[np.newaxis, :] / 3600.
    Gk = Gaussian1DKernel(1.8)
    for k in np.arange(len(spectra)):
        if (np.isfinite(spectra[k])).sum() > 800:
            spectra[k] = interpolate_replace_nans(spectra[k], Gk,
                                                  **{'boundary': 'extend'})
    if args.filter_file is not None:
        wsel = response>0.0
        mask = np.isfinite(spectra[:, wsel]) * (spectra[:, wsel] != 0.0)
        Pos = np.zeros((cnt1-cnt, 2))
        x, y = tp.wcs_world2pix(np.nanmean(raarray[cnt:cnt1, wsel], axis=1),
                                np.nanmean(decarray[cnt:cnt1, wsel], axis=1), 1)
        Pos[:, 0], Pos[:, 1] = (x, y)
        xc = np.interp(Pos[:, 0], xg, np.arange(len(xg)), left=0., right=len(xg))
        yc = np.interp(Pos[:, 1], yg, np.arange(len(yg)), left=0., right=len(yg))
        xc = np.array(np.round(xc), dtype=int)
        yc = np.array(np.round(yc), dtype=int)
        gsel = (xc>0) * (xc<len(xg)) * (yc>0) * (yc<len(yg))
        collapse_image = (np.nansum(spectra[:, wsel] * response[np.newaxis, wsel], axis=1) /
                          np.nansum(mask * response[np.newaxis, wsel], axis=1))
        collapse_eimage = np.sqrt((np.nansum(error[:, wsel]**2 * response[np.newaxis, wsel], axis=1) /
                                   np.nansum(mask * response[np.newaxis, wsel], axis=1)))


        
        # make_image_interp interprets tuple/list pairs as [start, stop]
        # ranges; passing a 2D ndarray here would be treated as two indices.
        cn = [(0, len(collapse_image))]
        image, errorimage, weight = make_image_interp(Pos, collapse_image, collapse_eimage,
                                                      xg, yg, xgrid, ygrid, 1.8 / 2.35,
                                                      cn)
        
        image[image==0.] = np.nan
        G = Gaussian2DKernel(4.5)
        cimage = convolve(image, G, preserve_nan=True, boundary='extend')
        nimage = binimage * 1.
        nimage[np.isnan(cimage)] = np.nan
        nimage = convolve(nimage, G, preserve_nan=True, boundary='extend')
        sel = np.isfinite(cimage) * (nimage > 0.05)
        yim = cimage / nimage
        d = np.sqrt(xgrid**2 + ygrid**2)
        yim[~sel] = 0.0
        nimage[np.isnan(nimage)] = 0.0
        image[np.isnan(image)] = 0.0
        bimage = binimage * 1.
        bimage[image==0.] = 0.
        xmax = np.linspace(-0.1, 0.02, 26)
        bmax = xmax*0.
        thresh = 0.05
        for i, v in enumerate(xmax):
            y = (cimage - v) / nimage
            y[~sel] = 0.0
            if i == 0:
                norm_sample = y[sel][nimage[sel] > thresh]
                args.log.info(
                    'Normalization diagnostics for %s: finite_cimage=%d, '
                    'nimage_gt_thresh=%d, norm_samples=%d, finite_norm_samples=%d',
                    h5file, int(np.isfinite(cimage).sum()),
                    int(np.sum(np.isfinite(nimage) & (nimage > thresh))),
                    int(norm_sample.size), int(np.isfinite(norm_sample).sum()))
            norm, std = biweight(y[sel][nimage[sel] > thresh], calc_std=True)
            bmax[i] = std / norm
        back = xmax[np.argmin(bmax)]
        args.log.info('Background for %s: %0.2f' % (h5file, back))
        y = (cimage - back) / nimage
        y[~sel] = 0.0
        norm, std = biweight(y[sel][nimage[sel] > thresh], calc_std=True)
        norm_array[jk] = norm
        if norm_array[jk] < 0.:
            norm_array[jk] = np.nan
        args.log.info('Normalization/STD for %s: %0.2f, %0.2f' % (h5file, norm, std/norm))
        flagged = image[yc[gsel], xc[gsel]]/norm_array[jk] < -0.03
        spectra[np.where(gsel)[0][flagged]] = np.nan
        args.log.info('%i fibers flagged for too large of a difference' % flagged.sum())
        plt.figure(figsize=(10, 8))
        plt.scatter(nimage[sel], y[sel] / norm, s=5, alpha=0.05)
        plt.plot([0.03, 0.6], [1., 1.], 'r-', lw=2)
        plt.plot([0.03, 0.6], [1.-std/norm, 1.-std/norm], 'r--', lw=1)
        plt.plot([0.03, 0.6], [1.+std/norm, 1.+std/norm], 'r--', lw=1)
        mn = np.nanpercentile(y[sel], 5)/norm
        mx = np.nanpercentile(y[sel], 95)/norm
        ran = mx - mn
        plt.axis([0.03, 0.6, 0.6, 1.4])
        name = op.basename(h5file)[:-3] + '_norm.png'
        plt.savefig(name, dpi=300)
        name = op.basename(h5file)[:-3] + '_rect.fits'
        h = tp.to_header()
        N = len(xg)
        h['CRPIX1'] = np.interp(0., xg, np.arange(len(xg)))+1.0
        h['CRPIX2'] = np.interp(0., xg, np.arange(len(xg)))+1.0
        fits.PrimaryHDU(np.array(image/norm_array[jk], dtype='float32'), header=h).writeto(name, overwrite=True)
    
    # Use a safe per-file normalization; avoid NaN/zero scaling that can zero-out data
    norm_j = norm_array[jk]
    if (not np.isfinite(norm_j)) or (norm_j == 0.0):
        args.log.warning(f'Per-file norm for {op.basename(h5file)} is invalid ({norm_j}); using 1.0 instead to avoid zeroing spectra.')
        norm_j = 1.0
    specarray[cnt:cnt1, :] = spectra / norm_j
    errarray[cnt:cnt1, :] = error / norm_j
    # Diagnostics for this chunk
    chunk = specarray[cnt:cnt1, :]
    finite_cnt = int(np.isfinite(chunk).sum())
    nonzero_cnt = int(np.count_nonzero(np.nan_to_num(chunk)))
    args.log.info(f'Chunk stats [{cnt}:{cnt1}]: finite={finite_cnt}/{chunk.size}, nonzero={nonzero_cnt}')
    if nonzero_cnt == 0:
        args.log.warning('This chunk of specarray is all zeros after normalization; check upstream masking/flagging and norms.')
    cnt = cnt + len(ra)
    t.close()

# Apply global normalization safely
_bi = biweight(norm_array)
if (not np.isfinite(_bi)) or (_bi == 0.0):
    args.log.warning(f'Global biweight(norm_array) invalid ({_bi}); skipping global scaling to avoid zeroing data.')
else:
    specarray[:] *= _bi
    errarray[:] *= _bi
# Report overall specarray stats before imaging
_nonzero_total = int(np.count_nonzero(np.nan_to_num(specarray)))
_finite_total = int(np.isfinite(specarray).sum())
args.log.info(f'specarray global stats: finite={_finite_total}/{specarray.size}, nonzero={_nonzero_total}')
if _nonzero_total == 0:
    args.log.error('specarray is all zeros before imaging. Possible causes: filter response zero everywhere, norms zero/NaN, all spectra flagged.')

def render_wavelength(i):
    """Build one wavelength plane using the test Gaussian imaging path."""
    x, y = tp.wcs_world2pix(raarray[:, i], decarray[:, i], 1)
    # Check how many positions fall within xgrid/ygrid bounds
    xmin, xmax = xg.min(), xg.max()
    ymin, ymax = yg.min(), yg.max()
    in_bounds = (x >= xmin) & (x <= xmax) & (y >= ymin) & (y <= ymax)
    n_in = int(np.sum(in_bounds))
    n_tot = int(len(raarray))

    args.log.info('Working on wavelength %0.0f' % def_wave[i])
    # Keep this per-worker; sharing the old serial-loop Pos array would make
    # concurrent wavelength workers overwrite one another's positions.
    Pos = np.column_stack((x, y))
    data = specarray[:, i]
    edata = errarray[:, i]
    provenance = _classify_error_provenance(data, edata)
    shot_provenance = np.zeros((len(shot_indices), 6), dtype=np.int64)
    for k, cnt in enumerate(shot_indices):
        if isinstance(cnt, (tuple, list)) and len(cnt) == 2:
            indices = np.arange(int(cnt[0]), int(cnt[1]), dtype=np.int64)
        else:
            indices = np.asarray(cnt, dtype=np.int64)
        shot_provenance[k, :] = _classify_error_provenance(
            data[indices], edata[indices])
    image, varianceimage, weight, ncontrib, dq, variance_stats = make_image_gaussian(
        Pos, data, edata, xg, yg, xgrid, ygrid, 1.8 / 2.35, shot_indices)
    return i, image, varianceimage, weight, ncontrib, dq, variance_stats, provenance, shot_provenance


def _warm_up_gaussian_splat():
    """Compile the Numba kernel before wavelength workers are started."""
    indices = np.array([0], dtype=np.int64)
    position = np.array([1.0], dtype=np.float64)
    flux = np.array([1.0], dtype=np.float32)
    error = np.array([1.0], dtype=np.float32)
    flux_sum = np.zeros((3, 3), dtype=np.float32)
    weight_sum = np.zeros((3, 3), dtype=np.float32)
    variance_numerator = np.zeros((3, 3), dtype=np.float32)
    error_weight_sum = np.zeros((3, 3), dtype=np.float32)
    support = np.zeros((3, 3), dtype=np.uint8)
    _gaussian_splat_shot_xy(
        indices, position, position, flux, error, 1.0, 1.0, 3, 3,
        1.8 / 2.35, 3, 2, np.pi * 0.75 ** 2, flux_sum, weight_sum,
        variance_numerator, error_weight_sum, support)


def _warm_up_sparse_gaussian_splat():
    """Compile the optional sparse kernel before the LSF reconstruction."""
    indices = np.array([0], dtype=np.int64)
    position = np.array([1.0], dtype=np.float64)
    flux = np.array([1.0], dtype=np.float32)
    lookup = np.arange(9, dtype=np.int32).reshape(3, 3)
    flux_sum = np.zeros(9, dtype=np.float32)
    weight_sum = np.zeros(9, dtype=np.float32)
    support = np.zeros(9, dtype=np.uint8)
    _gaussian_splat_sparse_shot_xy(
        indices, position, position, flux, lookup, 1.0, 1.0, 3, 3,
        1.8 / 2.35, 3, 2, np.pi * 0.75 ** 2, flux_sum, weight_sum, support)


_warm_up_gaussian_splat()
if args.make_lsf:
    _warm_up_sparse_gaussian_splat()
args.log.info('Gaussian test reconstruction enabled: SCI uses subpixel splats and NCONTRIB >= 2.')
args.log.warning(
    'Gaussian reconstruction variance semantics: VAR is the final SCI-median '
    'variance; ERROR is its 1-sigma companion (sqrt(VAR)). Formal shot '
    'variances require complete positive fiber-error support, and empirical '
    'shot scatter is used for NCONTRIB >= 5.')

args.log.info('Using %d wavelength worker(s)' % args.wave_workers)
variance_diagnostics = {
    'valid_sci_voxels': 0,
    'finite_variance_voxels': 0,
    'shot_samples': 0,
    'shot_samples_missing_variance': 0,
    'both_variances': 0,
    'empirical_exceeds_formal': 0,
    'median_ratios': [],
    'median_ncontrib_values': [],
    'variance_unavailable_voxels': 0,
    'insufficient_support_voxels': 0,
    'formal_var_used_voxels': 0,
    'empirical_var_used_voxels': 0,
    'empirical_only_voxels': 0,
}
provenance_by_wave = np.zeros((len(def_wave), 6), dtype=np.int64)
provenance_by_shot = np.zeros((len(shot_indices), 6), dtype=np.int64)
with ThreadPoolExecutor(max_workers=args.wave_workers) as executor:
    # executor.map preserves wavelength order in the returned results.  Each
    # worker only reads the shared input arrays; cube writes remain here.
    for i, image, varianceimage, weight, ncontrib, dq, plane_stats, provenance, shot_provenance in executor.map(
            render_wavelength, range(len(def_wave))):
        cube[i, :, :] = image
        variancecube[i, :, :] = varianceimage
        weightcube[i, :, :] = weight
        ncontribcube[i, :, :] = ncontrib
        dqcube[i, :, :] = dq
        provenance_by_wave[i, :] = provenance
        provenance_by_shot += shot_provenance
        for key in ('valid_sci_voxels', 'finite_variance_voxels',
                    'shot_samples', 'shot_samples_missing_variance',
                    'both_variances', 'empirical_exceeds_formal',
                    'variance_unavailable_voxels', 'insufficient_support_voxels',
                    'formal_var_used_voxels', 'empirical_var_used_voxels',
                    'empirical_only_voxels'):
            variance_diagnostics[key] += plane_stats[key]
        if np.isfinite(plane_stats['median_ratio']):
            variance_diagnostics['median_ratios'].append(plane_stats['median_ratio'])
        if np.isfinite(plane_stats['median_ncontrib']):
            variance_diagnostics['median_ncontrib_values'].append(
                plane_stats['median_ncontrib'])

_valid_sci = variance_diagnostics['valid_sci_voxels']
_finite_var = variance_diagnostics['finite_variance_voxels']
_shots = variance_diagnostics['shot_samples']
_both = variance_diagnostics['both_variances']
args.log.info(
    'Variance diagnostics: finite final VAR fraction among valid SCI=%0.4f; '
    'shot SCI samples lacking complete error support=%0.4f; '
    'empirical variance > formal variance=%0.4f; '
    'median sqrt(Var_empirical/Var_formal)=%s; '
    'median NCONTRIB for valid SCI=%s; valid SCI voxels with VAR unavailable=%d',
    (_finite_var / float(_valid_sci) if _valid_sci else np.nan),
    (variance_diagnostics['shot_samples_missing_variance'] /
     float(_shots) if _shots else np.nan),
    (variance_diagnostics['empirical_exceeds_formal'] /
     float(_both) if _both else np.nan),
    (('%0.4f' % np.median(variance_diagnostics['median_ratios']))
     if variance_diagnostics['median_ratios'] else 'nan'),
    (('%0.2f' % np.median(variance_diagnostics['median_ncontrib_values']))
     if variance_diagnostics['median_ncontrib_values'] else 'nan'),
    variance_diagnostics['variance_unavailable_voxels'])

_prov_names = ('SCI-valid', 'finite positive error', 'nonfinite error',
               'zero error', 'negative error', 'other invalid error')
_prov_total = provenance_by_wave.sum(axis=0)
_prov_sci = int(_prov_total[PROV_NSCI])
args.log.info('Error provenance: SCI-valid fiber samples=%d', _prov_sci)
for _prov_index, _prov_name in enumerate(_prov_names[1:], start=1):
    args.log.info('Error provenance: %-22s=%d (%0.4f)', _prov_name,
                  int(_prov_total[_prov_index]),
                  (_prov_total[_prov_index] / float(_prov_sci)
                   if _prov_sci else np.nan))

# A compact 12-bin wavelength summary makes edge or interval concentration
# visible without retaining per-pixel provenance masks or logging every plane.
_wave_bins = np.linspace(0, len(def_wave), 13, dtype=int)
args.log.info('Error provenance by wavelength bin: wave_start-wave_end, '
              'NSCI, frac_valid, frac_nonfinite, frac_zero, frac_negative, frac_other')
for _bin_start, _bin_stop in zip(_wave_bins[:-1], _wave_bins[1:]):
    _bin_prov = provenance_by_wave[_bin_start:_bin_stop].sum(axis=0)
    _bin_sci = int(_bin_prov[PROV_NSCI])
    _bin_den = float(_bin_sci) if _bin_sci else np.nan
    args.log.info(
        'Error provenance wavelength %0.0f-%0.0f: NSCI=%d, '
        'valid=%0.4f, nonfinite=%0.4f, zero=%0.4f, negative=%0.4f, other=%0.4f',
        def_wave[_bin_start], def_wave[_bin_stop - 1], _bin_sci,
        _bin_prov[PROV_ERROR_VALID] / _bin_den,
        _bin_prov[PROV_ERROR_NONFINITE] / _bin_den,
        _bin_prov[PROV_ERROR_ZERO] / _bin_den,
        _bin_prov[PROV_ERROR_NEGATIVE] / _bin_den,
        _bin_prov[PROV_ERROR_OTHER] / _bin_den)

_invalid_by_wave = (provenance_by_wave[:, PROV_ERROR_NONFINITE] +
                    provenance_by_wave[:, PROV_ERROR_ZERO] +
                    provenance_by_wave[:, PROV_ERROR_NEGATIVE] +
                    provenance_by_wave[:, PROV_ERROR_OTHER])
_invalid_fraction_by_wave = np.divide(
    _invalid_by_wave, provenance_by_wave[:, PROV_NSCI],
    out=np.full(len(def_wave), np.nan, dtype=float),
    where=provenance_by_wave[:, PROV_NSCI] > 0)
_peak_wave = int(np.nanargmax(_invalid_fraction_by_wave)) if np.any(np.isfinite(_invalid_fraction_by_wave)) else None
if _peak_wave is not None:
    args.log.info('Error provenance concentration: peak invalid-error wavelength '
                  '%0.0f A, fraction=%0.4f; first/last-bin fractions=%s/%s.',
                  def_wave[_peak_wave], _invalid_fraction_by_wave[_peak_wave],
                  ('%0.4f' % _invalid_fraction_by_wave[0]),
                  ('%0.4f' % _invalid_fraction_by_wave[-1]))

_invalid_by_shot = provenance_by_shot[:, PROV_ERROR_NONFINITE] + provenance_by_shot[:, PROV_ERROR_ZERO] + provenance_by_shot[:, PROV_ERROR_NEGATIVE] + provenance_by_shot[:, PROV_ERROR_OTHER]
_top_shots = np.argsort(_invalid_by_shot)[::-1]
for _shot_index in _top_shots[:3]:
    if _invalid_by_shot[_shot_index] <= 0:
        break
    args.log.info('Error provenance exposure: %s invalid=%d of SCI-valid=%d',
                  shot_names[_shot_index], int(_invalid_by_shot[_shot_index]),
                  int(provenance_by_shot[_shot_index, PROV_NSCI]))

args.log.info(
    'DQ diagnostics: valid SCI voxels=%d; insufficient-support voxels=%d; '
    'SCI-valid voxels with finite VAR=%d; VAR unavailable=%d; '
    'formal VAR adopted=%d; empirical VAR adopted over formal=%d; '
    'empirical-only VAR=%d',
    variance_diagnostics['valid_sci_voxels'],
    variance_diagnostics['insufficient_support_voxels'],
    variance_diagnostics['finite_variance_voxels'],
    variance_diagnostics['variance_unavailable_voxels'],
    variance_diagnostics['formal_var_used_voxels'],
    variance_diagnostics['empirical_var_used_voxels'],
    variance_diagnostics['empirical_only_voxels'])

if args.make_lsf:
    _run_lsf_measurement(
        h5files, args.surname, def_wave, raarray, decarray, specarray,
        shot_indices, shot_names, tp, xg, yg, xgrid, ygrid, cube,
        ncontribcube, args.pixel_scale, args.log,
        records=lsf_center_records)

name = op.basename('%s_cube.fits' % args.surname)
scale = args.pixel_scale / 3600.
header['WCSAXES'] = 3
header['CD1_1']   = -scale              # minus sign -> RA increases to the left (north up, east left)
header['CD1_2']   = 0.0
header['CD2_1']   = 0.0
header['CD2_2']   = scale
del header['CDELT1'], header['CDELT2']
header['CRPIX1'] = (N+1) / 2
header['CRPIX2'] = (N+1) / 2
# Ensure DS9 recognizes 3D WCS
header['WCSAXES'] = 3
# Spectral axis definition (linear wavelength grid)
header['CDELT3'] = 2.
header['CRPIX3'] = 1.
header['CRVAL3'] = 3470.
header['CTYPE1'] = 'RA---TAN'
header['CTYPE2'] = 'DEC--TAN'
header['CTYPE3'] = 'WAVE'
header['CUNIT1'] = 'deg'
header['CUNIT2'] = 'deg'
header['CUNIT3'] = 'Angstrom'
header['SPECSYS'] = 'TOPOCENT'
header.add_history('Gaussian subpixel reconstruction.')
header.add_history('NCONTRIB >= 2 required for valid SCI and VAR.')
header.add_history('Final VAR = max(formal median variance, empirical median variance).')
F = fits.PrimaryHDU(np.array(cube, 'float32'), header=header)
F.writeto(name, overwrite=True)
name = op.basename('%s_variance_cube.fits' % args.surname)
header['CRPIX1'] = (N+1) / 2
header['CRPIX2'] = (N+1) / 2
# Ensure DS9 recognizes 3D WCS
header['WCSAXES'] = 3
# Spectral axis definition (linear wavelength grid)
header['CDELT3'] = 2.
header['CRPIX3'] = 1.
header['CRVAL3'] = 3470.
header['CTYPE1'] = 'RA---TAN'
header['CTYPE2'] = 'DEC--TAN'
header['CTYPE3'] = 'WAVE'
header['CUNIT1'] = 'deg'
header['CUNIT2'] = 'deg'
header['CUNIT3'] = 'Angstrom'
header['SPECSYS'] = 'TOPOCENT'
variance_header = header.copy()
variance_header['BUNIT'] = 'SCI units squared'
variance_header['EXTNAME'] = 'VARIANCE'
variance_header.add_comment(
    'VAR = variance of final SCI estimator; units are SCI units squared.')
variance_header.add_comment(
    'Formal and empirical estimates are combined by VAR = max(formal, empirical) when both exist.')
F = fits.PrimaryHDU(np.array(variancecube, 'float32'), header=variance_header)
F.writeto(name, overwrite=True)
# Preserve the historical error-cube filename as a 1-sigma companion.  Its
# meaning is now explicitly ERROR = sqrt(VAR), rather than a median of shot
# interpolation errors.
name = op.basename('%s_errorcube.fits' % args.surname)
error_header = header.copy()
error_header['BUNIT'] = 'SCI units'
error_header['EXTNAME'] = 'ERROR'
error_header.add_comment(
    'ERROR = 1-sigma standard deviation = sqrt(VAR). See the variance cube.')
F = fits.PrimaryHDU(np.sqrt(np.array(variancecube, 'float32')), header=error_header)
F.writeto(name, overwrite=True)
name = op.basename('%s_dq_cube.fits' % args.surname)
dq_header = header.copy()
dq_header['BUNIT'] = 'bit mask'
dq_header['EXTNAME'] = 'DQ'
dq_header['DQBIT0'] = 'NCONTRIB < 2'
dq_header['DQBIT1'] = 'SCI valid, VAR unavailable'
dq_header['DQBIT2'] = 'empirical VAR adopted over formal'
dq_header['DQBIT3'] = 'formal VAR adopted'
dq_header['DQBIT4'] = 'empirical-only VAR adopted'
dq_header.add_comment('DQ bits are independent flags; NCONTRIB/COVERAGE describe continuous support.')
F = fits.PrimaryHDU(np.array(dqcube, 'uint16'), header=dq_header)
F.writeto(name, overwrite=True)
# Write weight cube as requested
name = op.basename('%s_weight_cube.fits' % args.surname)
header['CRPIX1'] = (N+1) / 2
header['CRPIX2'] = (N+1) / 2
header['WCSAXES'] = 3
header['CDELT3'] = 2.
header['CRPIX3'] = 1.
header['CRVAL3'] = 3470.
header['CTYPE1'] = 'RA---TAN'
header['CTYPE2'] = 'DEC--TAN'
header['CTYPE3'] = 'WAVE'
header['CUNIT1'] = 'deg'
header['CUNIT2'] = 'deg'
header['CUNIT3'] = 'Angstrom'
header['SPECSYS'] = 'TOPOCENT'
F = fits.PrimaryHDU(np.array(weightcube, 'float32'), header=header)
F.writeto(name, overwrite=True)
name = op.basename('%s_ncontrib_cube.fits' % args.surname)
ncontrib_header = header.copy()
ncontrib_header['BUNIT'] = 'count'
ncontrib_header['EXTNAME'] = 'NCONTRIB'
ncontrib_header['COMMENT'] = 'Number of independent shots with a valid fiber within 2 sigma.'
F = fits.PrimaryHDU(np.array(ncontribcube, 'uint8'), header=ncontrib_header)
F.writeto(name, overwrite=True)
