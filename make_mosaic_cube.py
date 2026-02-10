#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 13:10:11 2020

@author: gregz
"""
import matplotlib
matplotlib.use('agg')
import argparse as ap
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


args = parser.parse_args(args=None)
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

# Astrometry with CRPIX at image center (not lower-left)
_x0 = (N + 1) / 2.0
_y0 = (N + 1) / 2.0
A = Astrometry(bounding_box[0], bounding_box[1], 0., _x0, _y0)
# Ensure TP uses same centered CRPIX
tp = A.setup_TP(A.ra0, A.dec0, 0., x0=_x0, y0=_y0)
header = tp.to_header()

# Build compact arrays using only fibers that fall within the requested region
# and construct per-shot (dither) indices for only those relevant fibers.
cnt = 0
nexp_default = 3
per_file_sel = []        # list of local index arrays (within-file) that fall in-bounds
per_file_starts = []     # starting compact row for each file
shot_indices = []        # list of numpy arrays of compact indices for each exposure
xmin, xmax = xg.min(), xg.max()
ymin, ymax = yg.min(), yg.max()
for i, h5file in enumerate(h5files):
    t = tables.open_file(h5file)
    ra = t.root.Info.cols.ra[:]
    dec = t.root.Info.cols.dec[:]
    n_fib = len(ra)
    # Determine which fibers land inside the region
    x_chk, y_chk = tp.wcs_world2pix(ra, dec, 1)
    mask_in = (x_chk >= xmin) & (x_chk <= xmax) & (y_chk >= ymin) & (y_chk <= ymax)
    sel_local = np.where(mask_in)[0]
    per_file_sel.append(sel_local)
    per_file_starts.append(cnt)
    # Build mapping from local indices to compact indices for this file
    if sel_local.size > 0:
        map_loc_to_comp = -1 * np.ones(n_fib, dtype=int)
        map_loc_to_comp[sel_local] = np.arange(sel_local.size, dtype=int)
        # Group by 112-fiber blocks and interleave to 3 exposures, restricted to in-bounds fibers
        inds_local = np.arange(n_fib)
        block_ids = (inds_local // 112) % nexp_default
        for k in range(nexp_default):
            sel_shot_local = np.where(block_ids == k)[0]
            # keep only those also in sel_local
            if sel_shot_local.size > 0:
                # intersect while preserving order of sel_shot_local
                in_bounds_mask = mask_in[sel_shot_local]
                sel_shot_local = sel_shot_local[in_bounds_mask]
            if sel_shot_local.size > 0:
                comp_offsets = map_loc_to_comp[sel_shot_local]
                # filter any -1 entries just in case
                comp_offsets = comp_offsets[comp_offsets >= 0]
                if comp_offsets.size > 0:
                    shot_indices.append(cnt + comp_offsets)
    cnt += sel_local.size
    t.close()
args.log.info('Number of fibers within region: %i' % cnt)
args.log.info(f'Total shots (exposures) with in-bounds fibers: {len(shot_indices)} (up to 3 per h5file).')

raarray = np.zeros((cnt, len(def_wave)), dtype='float32')
decarray = np.zeros((cnt, len(def_wave)), dtype='float32')
specarray = np.zeros((cnt, len(def_wave)), dtype='float32')
errarray = np.zeros((cnt, len(def_wave)), dtype='float32')

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

cnt = 0
# Streaming counters to avoid memory-heavy global scans
_total_finite = 0
_total_nonzero = 0
norm_array = np.ones((len(h5files),))
for jk, h5file in enumerate(h5files):
    args.log.info('Working on %s' % h5file)
    t = tables.open_file(h5file)
    date = int(op.basename(h5file).split('_')[0])
    ifuslots = t.root.Info.cols.ifuslot[:]
    amps = np.array([i.decode("utf-8") for i in t.root.Info.cols.amp[:]])

    # Load full per-file arrays, then restrict to in-bounds fibers only
    ra_full = t.root.Info.cols.ra[:]
    dec_full = t.root.Info.cols.dec[:]
    RA = t.root.Survey.cols.ra[0]
    Dec = t.root.Survey.cols.dec[0]
    pa = t.root.Survey.cols.pa[0]
    offset = t.root.Survey.cols.offset[0]
    # Log and ignore offset scaling for now per user request
    if (not np.isfinite(offset)) or (offset == 0):
        args.log.warning(f'Offset for {op.basename(h5file)} is invalid ({offset}); proceeding without offset scaling.')
    else:
        args.log.info(f'Offset for {op.basename(h5file)}: {offset} (ignored for now)')

    sel_local = per_file_sel[jk]
    start = per_file_starts[jk]
    n_sel = sel_local.size
    if n_sel == 0:
        args.log.info('No in-bounds fibers for this file; skipping.')
        t.close()
        continue

    spectra_full = t.root.Fibers.cols.spectrum[:]  # do not divide by offset
    error_full = t.root.Fibers.cols.error[:]       # do not divide by offset

    # Apply IFU/amp mask to the selected fibers only
    spectra_sel = spectra_full[sel_local].copy()
    error_sel = error_full[sel_local].copy()
    for key in mask_dict.keys():
        date1 = int(key.split('-')[0])
        date2 = int(key.split('-')[1])
        if (date >= date1) and (date < date2):
            ifulist = mask_dict[key]
            for ifuamp in ifulist:
                ifu = int(ifuamp[:3])
                amp = ifuamp[3:]
                sel = np.where((ifu == ifuslots) * (amp == amps))[0]
                if sel.size > 0:
                    # mark overlapping selected fibers as NaN
                    mask_overlap = np.isin(sel_local, sel)
                    if np.any(mask_overlap):
                        spectra_sel[mask_overlap] = np.nan

    # Fill RA/Dec arrays for only selected fibers, applying ADR shifts
    ra_sel = ra_full[sel_local]
    dec_sel = dec_full[sel_local]
    E = Extract()
    Aother = Astrometry(bounding_box[0], bounding_box[1], pa, 0., 0.)
    E.get_ADR_RAdec(Aother)
    raarray[start:start+n_sel, :] = ra_sel[:, np.newaxis] - E.ADRra[np.newaxis, :] / 3600. / np.cos(np.deg2rad(A.dec0))
    decarray[start:start+n_sel, :] = dec_sel[:, np.newaxis] - E.ADRdec[np.newaxis, :] / 3600.

    # Interpolate only spectra with sufficient finite points
    Gk = Gaussian1DKernel(1.8)
    for k in np.arange(n_sel):
        if (np.isfinite(spectra_sel[k])).sum() > 800:
            spectra_sel[k] = interpolate_replace_nans(spectra_sel[k], Gk, **{'boundary': 'extend'})

    # Use a safe per-file normalization; avoid NaN/zero scaling that can zero-out data
    norm_j = norm_array[jk]
    if (not np.isfinite(norm_j)) or (norm_j == 0.0):
        args.log.warning(f'Per-file norm for {op.basename(h5file)} is invalid ({norm_j}); using 1.0 instead to avoid zeroing spectra.')
        norm_j = 1.0
    specarray[start:start+n_sel, :] = spectra_sel / norm_j
    errarray[start:start+n_sel, :] = error_sel / norm_j

    # Diagnostics for this chunk (avoid nan_to_num on large arrays)
    chunk = specarray[start:start+n_sel, :]
    finite_mask = np.isfinite(chunk)
    finite_cnt = int(finite_mask.sum())
    # count non-zeros only among finite values
    if finite_cnt > 0:
        nonzero_cnt = int((chunk[finite_mask] != 0).sum())
    else:
        nonzero_cnt = 0
    _total_finite += finite_cnt
    _total_nonzero += nonzero_cnt
    args.log.info(f'Chunk stats [file={jk}, rows {start}:{start+n_sel}]: finite={finite_cnt}/{chunk.size}, nonzero={nonzero_cnt}')

    t.close()

# Apply global normalization safely
_bi = biweight(norm_array)
if (not np.isfinite(_bi)) or (_bi == 0.0):
    args.log.warning(f'Global biweight(norm_array) invalid ({_bi}); skipping global scaling to avoid zeroing data.')
else:
    specarray[:] *= _bi
    errarray[:] *= _bi
# Report overall specarray stats before imaging using streaming counters (memory-safe)
_finite_total = int(_total_finite)
_nonzero_total = int(_total_nonzero)
args.log.info(f'specarray global stats: finite={_finite_total}/{specarray.size}, nonzero={_nonzero_total}')
if _nonzero_total == 0:
    args.log.error('specarray is all zeros before imaging. Possible causes: filter response zero everywhere, norms zero/NaN, all spectra flagged.')

Pos = np.zeros((len(raarray), 2))
for i in np.arange(len(def_wave)):
    x, y = tp.wcs_world2pix(raarray[:, i], decarray[:, i], 1)
    # Check how many positions fall within xgrid/ygrid bounds
    xmin, xmax = xg.min(), xg.max()
    ymin, ymax = yg.min(), yg.max()
    in_bounds = (x >= xmin) & (x <= xmax) & (y >= ymin) & (y <= ymax)
    n_in = int(np.sum(in_bounds))
    n_tot = int(len(raarray))

    args.log.info('Working on wavelength %0.0f' % def_wave[i])
    Pos[:, 0], Pos[:, 1] = (x, y)
    data = specarray[:, i]
    edata = errarray[:, i]
    # Full combination across all exposures (shots)
    image, errorimage, weight = make_image_interp(Pos, data, edata, xg, yg, 
                                                  xgrid, ygrid, 1.8 / 2.35,
                                                  shot_indices)
    cube[i, :, :] += image
    ecube[i, :, :] += errorimage
    weightcube[i, :, :] += weight


name = op.basename('%s_cube.fits' % args.surname)
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
