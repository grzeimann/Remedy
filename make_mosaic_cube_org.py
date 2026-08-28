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
from scipy.interpolate import griddata
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

# M101-specific H5 sky repair.  These coordinates are used only for the
# H5-consumption-time replacement of the Quick Reduction exposure sky.
M101_SKY_RA0_DEG = 210.800
M101_SKY_DEC0_DEG = 54.333
M101_SKY_MIN_RADIUS_ARCMIN = 6.0
M101_SKY_NEXPOSURES = 3
M101_SKY_MIN_FINITE_PIXELS = 800
M101_SKY_MIN_FIBERS = 20


def repair_m101_sky(spectra_skysub, stored_sky, ra, dec, log=None,
                    h5file=''):
    """Replace the Quick Reduction sky independently for M101 exposures.

    ``spectra_skysub`` and ``stored_sky`` must be in the same units.  The
    caller handles the offset convention before calling this function.  A
    full-length Boolean mask is used for each exposure so the radial cut
    cannot misalign a shortened selection with the original fiber array.

    Returns
    -------
    np.ndarray
        Sky-corrected spectra, or the original sky-subtracted spectra for an
        exposure when a trustworthy replacement sky cannot be constructed.
    """
    spectra_corrected = spectra_skysub.copy()
    n_fib = spectra_skysub.shape[0]
    inds_local = np.arange(n_fib)
    block_ids = inds_local // 112
    dra_arcmin = ((ra - M101_SKY_RA0_DEG) *
                  np.cos(np.radians(M101_SKY_DEC0_DEG)) * 60.0)
    ddec_arcmin = (dec - M101_SKY_DEC0_DEG) * 60.0
    radius_arcmin = np.sqrt(dra_arcmin ** 2 + ddec_arcmin ** 2)
    sky_region = radius_arcmin > M101_SKY_MIN_RADIUS_ARCMIN
    spectra_restored = spectra_skysub + stored_sky

    def report(message, *values):
        if log is not None:
            log.info(message, *values)

    for k in range(M101_SKY_NEXPOSURES):
        exp_sel = (block_ids % M101_SKY_NEXPOSURES) == k
        exp_indices = np.where(exp_sel)[0]
        if exp_indices.size == 0:
            report('M101 sky repair %s exposure %d: no fibers; using original sky-subtracted spectrum.',
                   h5file, k + 1)
            continue

        finite_counts = np.isfinite(spectra_restored[exp_indices]).sum(axis=1)
        broadband = biweight(spectra_restored[exp_indices, 800:900], axis=1)
        candidate = (sky_region[exp_indices] &
                     (finite_counts >= M101_SKY_MIN_FINITE_PIXELS) &
                     np.isfinite(broadband))
        candidate_indices = exp_indices[candidate]
        candidate_broadband = broadband[candidate]

        # Start from the original sky-subtracted data for the conservative
        # fallback path; only replace an exposure after all checks succeed.
        full_sky_mask = np.zeros(n_fib, dtype=bool)
        selected_sky_count = 0
        replacement_sky = None
        stored_reference = None

        if candidate_indices.size >= M101_SKY_MIN_FIBERS:
            _, bm = biweight(candidate_broadband, calc_std=True)
            low = np.nanpercentile(candidate_broadband, 5)
            high = np.nanpercentile(candidate_broadband, 60)
            if np.isfinite(bm) and bm > 0.0 and np.isfinite(low) and np.isfinite(high):
                per_array = np.linspace(low, high, 81)
                neighbors = np.sum(
                    np.abs(candidate_broadband[:, None] - per_array[None, :]) < bm,
                    axis=0)
                nsky = per_array[np.argmax(neighbors)]
                selected = (candidate_broadband - nsky) < (2.0 * bm)
                full_sky_mask[candidate_indices[selected]] = True
                selected_sky_count = int(selected.sum())

                if selected_sky_count >= M101_SKY_MIN_FIBERS:
                    replacement_sky = biweight(
                        spectra_restored[full_sky_mask], axis=0)
                    stored_reference = biweight(stored_sky[full_sky_mask], axis=0)
                    if not np.any(np.isfinite(replacement_sky)):
                        replacement_sky = None
            else:
                report('M101 sky repair %s exposure %d: degenerate broadband scatter; using original sky-subtracted spectrum.',
                       h5file, k + 1)

        if replacement_sky is None:
            report(
                'M101 sky repair %s exposure %d: total fibers=%d, beyond 6 arcmin=%d, '
                'finite candidates=%d, selected sky fibers=%d; insufficient trustworthy sky, '
                'using original sky-subtracted spectrum.',
                h5file, k + 1, int(exp_indices.size),
                int(np.sum(sky_region[exp_indices])), int(candidate_indices.size),
                selected_sky_count)
            continue

        spectra_corrected[exp_indices] = (
            spectra_restored[exp_indices] - replacement_sky[None, :])
        median_stored = np.nanmedian(stored_sky[full_sky_mask])
        median_replacement = np.nanmedian(replacement_sky)
        median_difference = np.nanmedian(replacement_sky - stored_reference)
        report(
            'M101 sky repair %s exposure %d: total fibers=%d, beyond 6 arcmin=%d, '
            'finite candidates=%d, selected sky fibers=%d, median stored QR sky=%0.5g, '
            'median replacement sky=%0.5g, median replacement - stored sky=%0.5g',
            h5file, k + 1, int(exp_indices.size),
            int(np.sum(sky_region[exp_indices])), int(candidate_indices.size),
            selected_sky_count, median_stored, median_replacement, median_difference)

    return spectra_corrected


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
                            error_sum, error_support, support_map):
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
                    error_sum[py, px] += weight * weight * (err / area) ** 2
                    error_support[py, px] += weight * weight

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


def make_image_gaussian(Pos, y, ye, xg, yg, xgrid, ygrid, sigma,
                        cnt_array):
    """Construct one wavelength plane with true subpixel Gaussian splatting.

    Each finite science fiber contributes ``flux / area`` to a Gaussian
    weighted interpolation within its shot.  Shot images are median-combined
    only where that shot has a fiber within ``2 * sigma`` of the output pixel.
    The returned ``coverage`` is the mean per-shot Gaussian support strength,
    capped at one per shot; it is a diagnostic and never divides SCI.

    ``errorimage`` is the median of the available per-shot interpolation
    standard deviations calculated from the matched weighted variance formula.
    It is not an exact propagated variance for the final shot median and must
    be revisited before a public cube is finalized.
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
    shot_errors = np.full((nshots, ny, nx), np.nan, dtype=np.float32)
    coverage = np.zeros((ny, nx), dtype=np.float32)
    ncontrib = np.zeros((ny, nx), dtype=np.uint8)

    for k, cnt in enumerate(cnt_array):
        if isinstance(cnt, (tuple, list)) and len(cnt) == 2:
            indices = np.arange(int(cnt[0]), int(cnt[1]), dtype=np.int64)
        else:
            indices = np.asarray(cnt, dtype=np.int64)

        flux_sum = np.zeros((ny, nx), dtype=np.float32)
        weight_sum = np.zeros((ny, nx), dtype=np.float32)
        error_sum = np.zeros((ny, nx), dtype=np.float32)
        error_support = np.zeros((ny, nx), dtype=np.float32)
        support_map = np.zeros((ny, nx), dtype=np.uint8)
        _gaussian_splat_shot_xy(
            indices, xpos, ypos, y, errors, float(xg[0]), float(yg[0]),
            nx, ny, float(sigma), radius, support_radius, float(area),
            flux_sum, weight_sum, error_sum, error_support, support_map)

        supported = (support_map != 0) & (weight_sum > 0.0)
        if np.any(supported):
            shot_images[k, supported] = flux_sum[supported] / weight_sum[supported]
            # Invalid errors do not invalidate SCI.  The error product is
            # simply absent for pixels with no valid error-bearing fibers.
            error_valid = supported & (error_support > 0.0)
            shot_errors[k, error_valid] = np.sqrt(
                error_sum[error_valid] /
                (weight_sum[error_valid] * weight_sum[error_valid]))
            ncontrib[supported] += 1
            coverage += np.minimum(weight_sum, 1.0) * supported

    # The median ignores unsupported shots because those entries remain NaN.
    image = np.nanmedian(shot_images, axis=0).astype(np.float32)
    errorimage = np.nanmedian(shot_errors, axis=0).astype(np.float32)
    coverage /= float(max(1, nshots))
    coverage = np.clip(coverage, 0.0, 1.0)

    final_valid = ncontrib >= 2
    image[~final_valid] = 0.0
    errorimage[~final_valid] = 0.0
    image[~np.isfinite(image)] = 0.0
    errorimage[~np.isfinite(errorimage)] = 0.0

    return image, errorimage, coverage, ncontrib

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
ecube = np.zeros((len(def_wave),) + xgrid.shape, dtype='float32')
weightcube = np.zeros((len(def_wave),) + xgrid.shape, dtype='float32')
ncontribcube = np.zeros((len(def_wave),) + xgrid.shape, dtype='uint8')

cnt = 0
cnt_array = np.zeros((len(h5files), 2), dtype=int)
# Build per-exposure (shot) index selections assuming 3 interleaved dithers per h5
nexp_default = 3
shot_indices = []  # list of numpy arrays of global indices for each exposure
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
    cnt = end
    t.close()
args.log.info('Number of total fibers: %i' % cnt)
args.log.info(f'Total shots (exposures) assumed: {len(shot_indices)} (3 per h5file).')

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
    stored_sky = t.root.Fibers.cols.skyspectrum[:]
    if (not np.isfinite(offset)) or (offset == 0):
        args.log.warning(f'Offset for {op.basename(h5file)} is invalid ({offset}); proceeding without offset correction.')
        spectra_skysub = t.root.Fibers.cols.spectrum[:]
        error = t.root.Fibers.cols.error[:]
    else:
        args.log.info(f'Offset for {op.basename(h5file)}: {offset}')
        # quick_reduction applies this photometric factor to SCI/error before
        # writing the H5 file, but not to skyspectrum.  Consequently spectrum/
        # offset and the stored skyspectrum are already in matching units.
        spectra_skysub = t.root.Fibers.cols.spectrum[:] / offset
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
                spectra_skysub[sel] = np.nan

    # Restore the Quick Reduction exposure sky, replace it with an
    # M101-safe sky independently for each interleaved exposure, and pass the
    # corrected spectra into the existing mosaic normalization below.
    spectra = repair_m101_sky(
        spectra_skysub, stored_sky, ra, dec, log=args.log,
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
        d = np.sqrt(xgrid**2 + ygrid**2)
        skyvalues = (binimage[yc[gsel], xc[gsel]] < 0.01) * (d[yc[gsel], xc[gsel]] > 420.)
        args.log.info(
            'Sky diagnostics for %s: gsel=%d, sky=%d, finite_spectra=%d',
            h5file, int(np.sum(gsel)), int(np.sum(skyvalues)),
            int(np.isfinite(spectra).sum()))
        # The M101 repair above replaced the Quick Reduction exposure sky.
        # This is separate from the existing mosaic-level residual-background
        # correction performed here with backspectra.
        backspectra = biweight(spectra[gsel][skyvalues], axis=0)
        args.log.info('Average spectrum residual value: %0.3f' % np.nanmedian(backspectra))
        spectra[:] -= backspectra[np.newaxis, :]
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
    # Swap this one call back to make_image_interp for the historical/current
    # reconstruction comparison.  The old function itself remains unchanged.
    image, errorimage, weight, ncontrib = make_image_gaussian(
        Pos, data, edata, xg, yg, xgrid, ygrid, 1.8 / 2.35, shot_indices)
    return i, image, errorimage, weight, ncontrib


def _warm_up_gaussian_splat():
    """Compile the Numba kernel before wavelength workers are started."""
    indices = np.array([0], dtype=np.int64)
    position = np.array([1.0], dtype=np.float64)
    flux = np.array([1.0], dtype=np.float32)
    error = np.array([1.0], dtype=np.float32)
    flux_sum = np.zeros((3, 3), dtype=np.float32)
    weight_sum = np.zeros((3, 3), dtype=np.float32)
    error_sum = np.zeros((3, 3), dtype=np.float32)
    error_support = np.zeros((3, 3), dtype=np.float32)
    support = np.zeros((3, 3), dtype=np.uint8)
    _gaussian_splat_shot_xy(
        indices, position, position, flux, error, 1.0, 1.0, 3, 3,
        1.8 / 2.35, 3, 2, np.pi * 0.75 ** 2, flux_sum, weight_sum,
        error_sum, error_support, support)


_warm_up_gaussian_splat()
args.log.info('Gaussian test reconstruction enabled: SCI uses subpixel splats and NCONTRIB >= 2.')
args.log.warning('Gaussian test errorimage is the median per-shot interpolation error, not an exact median-variance model.')

args.log.info('Using %d wavelength worker(s)' % args.wave_workers)
with ThreadPoolExecutor(max_workers=args.wave_workers) as executor:
    # executor.map preserves wavelength order in the returned results.  Each
    # worker only reads the shared input arrays; cube writes remain here.
    for i, image, errorimage, weight, ncontrib in executor.map(
            render_wavelength, range(len(def_wave))):
        cube[i, :, :] = image
        ecube[i, :, :] = errorimage
        weightcube[i, :, :] = weight
        ncontribcube[i, :, :] = ncontrib

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
F = fits.PrimaryHDU(np.array(cube, 'float32'), header=header)
F.writeto(name, overwrite=True)
name = op.basename('%s_errorcube.fits' % args.surname)
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
F = fits.PrimaryHDU(np.array(ecube, 'float32'), header=header)
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
