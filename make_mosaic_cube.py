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
from astropy.stats import mad_std
import tables
import numpy as np
import os.path as op
import sys
import warnings
import matplotlib.pyplot as plt
from input_utils import setup_logging
from astrometry import Astrometry
from math_utils import biweight
from extract import Extract
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
    """Rebin a 2D array to a new shape by averaging blocks.

    Args:
        arr (np.ndarray): Input 2D array to be rebinned (ny, nx).
        new_shape (tuple[int, int]): Desired output shape (ny_new, nx_new). Must evenly divide arr shape.

    Returns:
        np.ndarray: Rebinned 2D array of shape new_shape.
    """
    shape = (new_shape[0], arr.shape[0] // new_shape[0],
             new_shape[1], arr.shape[1] // new_shape[1])
    return arr.reshape(shape).mean(axis=(-1, 1))


def make_image_interp(Pos, y, ye, xg, yg, xgrid, ygrid, sigma, cnt_array, binsize=2):
    """Interpolate scattered fiber samples to an image with PSF smoothing.

    This routine stacks per-shot (a.k.a. exposure/dither) images on a regular grid,
    smooths with a Gaussian kernel representing the PSF, and produces both an
    image and an approximate error map via inverse-variance propagation.

    Args:
        Pos (np.ndarray): Array of shape (N, 2) with pixel coordinates (x, y) for N fibers.
        y (np.ndarray): Flux values for the N fibers.
        ye (np.ndarray or None): Flux errors for the N fibers (same length as y), or None.
        xg (np.ndarray): 1D grid coordinates along x (pixel centers, 1-indexed here).
        yg (np.ndarray): 1D grid coordinates along y (pixel centers, 1-indexed here).
        xgrid (np.ndarray): 2D meshgrid of xg.
        ygrid (np.ndarray): 2D meshgrid of yg.
        sigma (float): Gaussian sigma (in pixels) for PSF smoothing.
        cnt_array (list): List of index arrays selecting fibers for each shot (exposure).
        binsize (int, optional): Unused legacy parameter retained for API compatibility.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]:
            - image (2D) smoothed flux image.
            - errorimage (2D) smoothed error estimate (sqrt of convolved variance).
            - coveragek (2D) fractional effective coverage map in [0, 1].
    """
    G = Gaussian2DKernel(sigma)
    nshots = len(cnt_array)
    image_all = np.ma.zeros((nshots,) + xgrid.shape)
    image_all.mask = True
    coveragek = 0.0 * xgrid
    ivar_grid = 0.0 * xgrid
    area = (np.pi * 0.75**2)

    xc = np.interp(Pos[:, 0], xg, np.arange(len(xg)), left=0., right=len(xg))
    yc = np.interp(Pos[:, 1], yg, np.arange(len(yg)), left=0., right=len(yg))
    xc = np.array(np.round(xc), dtype=int)
    yc = np.array(np.round(yc), dtype=int)
    gsel = np.where((xc > 1) * (xc < len(xg) - 1) * (yc > 1) * (yc < len(yg) - 1))[0]
    for k, cnt in enumerate(cnt_array):
        if isinstance(cnt, (tuple, list)) and len(cnt) == 2:
            l1 = int(cnt[0])
            l2 = int(cnt[1])
            sel_idx = gsel[(gsel >= l1) & (gsel < l2)]
        else:
            cnt_arr = np.array(cnt, dtype=int)
            sel_idx = np.intersect1d(gsel, cnt_arr, assume_unique=False)
        if sel_idx.size == 0:
            continue
        image_all.data[k, yc[sel_idx], xc[sel_idx]] = y[sel_idx] / area
        image_all.mask[k, yc[sel_idx], xc[sel_idx]] = False
        if ye is not None:
            ye_sel = ye[sel_idx]
            valid_e = np.isfinite(ye_sel) & (ye_sel > 0)
            if np.any(valid_e):
                ivar_vals = np.zeros_like(ye_sel, dtype=float)
                ivar_vals[valid_e] = 1.0 / (ye_sel[valid_e]**2) / (area**2)
                ivar_grid[yc[sel_idx], xc[sel_idx]] += ivar_vals
        valid_flux = np.isfinite(y[sel_idx])
        if np.any(valid_flux):
            mask_shot = 0.0 * xgrid
            mask_shot[yc[sel_idx][valid_flux], xc[sel_idx][valid_flux]] = 1.0
            mask_sm = convolve(mask_shot, G, preserve_nan=False, boundary='extend')
            coveragek += mask_sm
    image = np.ma.median(image_all, axis=0)
    y_im = image.data * 1.
    y_im[image.mask] = np.nan
    image = convolve(y_im, G, preserve_nan=False, boundary='extend')
    nshots = max(1, nshots)
    coveragek = np.clip(coveragek / float(nshots), 0.0, 1.0)
    image[coveragek == 0.] = np.nan
    image[np.isnan(image)] = 0.0

    var0 = np.empty_like(xgrid, dtype=float)
    var0[:] = np.nan
    pos = ivar_grid > 0
    if np.any(pos):
        var0[pos] = 1.0 / ivar_grid[pos]
    kernel_sq = G.array**2
    var_sm = convolve(var0, kernel_sq, normalize_kernel=False, boundary='extend')
    var_sm[coveragek == 0.] = np.nan
    errorimage = np.sqrt(var_sm)
    errorimage[np.isnan(errorimage)] = 0.0

    return image, errorimage, coveragek


def parse_args():
    """Parse command-line arguments and set up logging.

    Returns:
        argparse.Namespace: Parsed arguments with an attached logger at args.log.
    """
    args = parser.parse_args(args=None)
    args.log = setup_logging('make_image_from_h5')
    return args


def setup_wcs_and_grids(bounding_box, pixel_scale):
    """Create image grids and WCS/TP objects for cube construction.

    Args:
        bounding_box (list[float]): [ra_center, dec_center, size_arcsec]. Size may be non-integer; will be rounded to odd N.
        pixel_scale (float): Pixel scale in arcsec.

    Returns:
        tuple: (N, xg, yg, xgrid, ygrid, tp, header)
    """
    bounding_box = list(bounding_box)
    bounding_box[2] = int(bounding_box[2] * 60.0 / pixel_scale / 2.0) * 2 * pixel_scale
    N = int(bounding_box[2] / pixel_scale / 2.0) * 2 + 1
    xg = np.arange(N) + 1
    yg = np.arange(N) + 1
    xgrid, ygrid = np.meshgrid(xg, yg)
    _x0 = (N + 1) / 2.0
    _y0 = (N + 1) / 2.0
    A = Astrometry(bounding_box[0], bounding_box[1], 0., _x0, _y0)
    tp = A.setup_TP(A.ra0, A.dec0, 0., x0=_x0, y0=_y0)
    header = tp.to_header()
    return N, xg, yg, xgrid, ygrid, tp, header


def preselect_inbounds(h5files, tp, xg, yg, log):
    """Preselect fibers within the image region and build per-shot indices.

    Args:
        h5files (list[str]): Input HDF5 file paths.
        tp (astrometry.TP): TAN projection/WCS helper.
        xg (np.ndarray): X grid coordinates.
        yg (np.ndarray): Y grid coordinates.
        log (logging.Logger): Logger for status messages.

    Returns:
        tuple: (total_cnt, per_file_sel, per_file_starts, shot_indices)
    """
    cnt = 0
    nexp_default = 3
    per_file_sel = []
    per_file_starts = []
    shot_indices = []
    xmin, xmax = xg.min(), xg.max()
    ymin, ymax = yg.min(), yg.max()
    for i, h5file in enumerate(h5files):
        t = tables.open_file(h5file)
        ra = t.root.Info.cols.ra[:]
        dec = t.root.Info.cols.dec[:]
        n_fib = len(ra)
        x_chk, y_chk = tp.wcs_world2pix(ra, dec, 1)
        mask_in = (x_chk >= xmin) & (x_chk <= xmax) & (y_chk >= ymin) & (y_chk <= ymax)
        sel_local = np.where(mask_in)[0]
        per_file_sel.append(sel_local)
        per_file_starts.append(cnt)
        if sel_local.size > 0:
            map_loc_to_comp = -1 * np.ones(n_fib, dtype=int)
            map_loc_to_comp[sel_local] = np.arange(sel_local.size, dtype=int)
            inds_local = np.arange(n_fib)
            block_ids = (inds_local // 112) % nexp_default
            for k in range(nexp_default):
                sel_shot_local = np.where(block_ids == k)[0]
                if sel_shot_local.size > 0:
                    in_bounds_mask = mask_in[sel_shot_local]
                    sel_shot_local = sel_shot_local[in_bounds_mask]
                if sel_shot_local.size > 0:
                    comp_offsets = map_loc_to_comp[sel_shot_local]
                    comp_offsets = comp_offsets[comp_offsets >= 0]
                    if comp_offsets.size > 0:
                        shot_indices.append(cnt + comp_offsets)
        cnt += sel_local.size
        t.close()
    log.info('Number of fibers within region: %i' % cnt)
    log.info(f'Total shots (exposures) with in-bounds fibers: {len(shot_indices)} (up to 3 per h5file).')
    return cnt, per_file_sel, per_file_starts, shot_indices


def coverage_precheck(h5files, tp, xg, yg, bounding_box, log):
    """Log how many fibers fall in the region for each file; warn/abort if empty.

    Args:
        h5files (list[str]): Input files.
        tp: WCS helper.
        xg (np.ndarray): X grid.
        yg (np.ndarray): Y grid.
        bounding_box (list[float]): [ra, dec, size_arcsec].
        log (logging.Logger): Logger.
    """
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
            log.info(f'Fibers in region for {op.basename(h5file)}: {n_in}')
            in_region_total += n_in
        if in_region_total == 0:
            log.error('No fibers found within the requested cube region. The output would be all zeros. Aborting. '
                      f"Center=({bounding_box[0]:.5f},{bounding_box[1]:.5f}), size={bounding_box[2]:.2f} arcsec")
            sys.exit(2)
        if in_region_total < 10:
            log.warning(f'Only {in_region_total} fibers fall within the region; output may be very noisy or near zero.')
    except Exception as e:
        log.warning(f'Coverage precheck failed with {e}; proceeding anyway.')


def process_files(h5files, per_file_sel, per_file_starts, bounding_box, tp, A, log):
    """Load, mask, ADR-shift, and interpolate spectra only for in-bounds fibers.

    Args:
        h5files (list[str]): Input files.
        per_file_sel (list[np.ndarray]): Local in-bounds indices per file.
        per_file_starts (list[int]): Start row in compact arrays for each file.
        bounding_box (list[float]): [ra, dec, size_arcsec].
        tp: WCS helper (unused here but kept for context consistency).
        A (Astrometry): Astrometry object for dec0 (ADR cos(dec)).
        log (logging.Logger): Logger.

    Returns:
        tuple: (raarray, decarray, specarray, errarray, norm_array, total_finite, total_nonzero)
    """
    def_wave = np.linspace(3470., 5540., 1036)
    total_cnt = int(np.sum([len(s) for s in per_file_sel]))
    raarray = np.zeros((total_cnt, len(def_wave)), dtype='float32')
    decarray = np.zeros((total_cnt, len(def_wave)), dtype='float32')
    specarray = np.zeros((total_cnt, len(def_wave)), dtype='float32')
    errarray = np.zeros((total_cnt, len(def_wave)), dtype='float32')

    _total_finite = 0
    _total_nonzero = 0
    norm_array = np.ones((len(h5files),))

    for jk, h5file in enumerate(h5files):
        log.info('Working on %s' % h5file)
        t = tables.open_file(h5file)
        date = int(op.basename(h5file).split('_')[0])
        ifuslots = t.root.Info.cols.ifuslot[:]
        amps = np.array([i.decode("utf-8") for i in t.root.Info.cols.amp[:]])

        ra_full = t.root.Info.cols.ra[:]
        dec_full = t.root.Info.cols.dec[:]
        pa = t.root.Survey.cols.pa[0]
        offset = t.root.Survey.cols.offset[0]
        if (not np.isfinite(offset)) or (offset == 0):
            log.warning(f'Offset for {op.basename(h5file)} is invalid ({offset}); proceeding without offset scaling.')
        else:
            log.info(f'Offset for {op.basename(h5file)}: {offset} (ignored for now)')

        sel_local = per_file_sel[jk]
        start = per_file_starts[jk]
        n_sel = sel_local.size
        if n_sel == 0:
            log.info('No in-bounds fibers for this file; skipping.')
            t.close()
            continue

        spectra_full = t.root.Fibers.cols.spectrum[:]
        error_full = t.root.Fibers.cols.error[:]

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
                        mask_overlap = np.isin(sel_local, sel)
                        if np.any(mask_overlap):
                            spectra_sel[mask_overlap] = np.nan

        ra_sel = ra_full[sel_local]
        dec_sel = dec_full[sel_local]
        E = Extract()
        Aother = Astrometry(bounding_box[0], bounding_box[1], pa, 0., 0.)
        E.get_ADR_RAdec(Aother)
        raarray[start:start + n_sel, :] = ra_sel[:, np.newaxis] - E.ADRra[np.newaxis, :] / 3600. / np.cos(np.deg2rad(A.dec0))
        decarray[start:start + n_sel, :] = dec_sel[:, np.newaxis] - E.ADRdec[np.newaxis, :] / 3600.

        Gk = Gaussian1DKernel(1.8)
        for k in np.arange(n_sel):
            if (np.isfinite(spectra_sel[k])).sum() > 800:
                spectra_sel[k] = interpolate_replace_nans(spectra_sel[k], Gk, **{'boundary': 'extend'})

        norm_j = norm_array[jk]
        if (not np.isfinite(norm_j)) or (norm_j == 0.0):
            log.warning(f'Per-file norm for {op.basename(h5file)} is invalid ({norm_j}); using 1.0 instead to avoid zeroing spectra.')
            norm_j = 1.0
        specarray[start:start + n_sel, :] = spectra_sel / norm_j
        errarray[start:start + n_sel, :] = error_sel / norm_j

        chunk = specarray[start:start + n_sel, :]
        finite_mask = np.isfinite(chunk)
        finite_cnt = int(finite_mask.sum())
        nonzero_cnt = int((chunk[finite_mask] != 0).sum()) if finite_cnt > 0 else 0
        _total_finite += finite_cnt
        _total_nonzero += nonzero_cnt
        log.info(f'Chunk stats [file={jk}, rows {start}:{start + n_sel}]: finite={finite_cnt}/{chunk.size}, nonzero={nonzero_cnt}')

        t.close()

    _bi = biweight(norm_array)
    if (not np.isfinite(_bi)) or (_bi == 0.0):
        log.warning(f'Global biweight(norm_array) invalid ({_bi}); skipping global scaling to avoid zeroing data.')
    else:
        specarray[:] *= _bi
        errarray[:] *= _bi

    log.info(f'specarray global stats: finite={int(_total_finite)}/{specarray.size}, nonzero={int(_total_nonzero)}')
    if int(_total_nonzero) == 0:
        log.error('specarray is all zeros before imaging. Possible causes: filter response zero everywhere, norms zero/NaN, all spectra flagged.')

    return raarray, decarray, specarray, errarray, norm_array, int(_total_finite), int(_total_nonzero)


def assemble_cubes(def_wave, raarray, decarray, specarray, errarray, shot_indices, tp, xg, yg, xgrid, ygrid, log):
    """Build data, error, and weight cubes by imaging each wavelength slice.

    Args:
        def_wave (np.ndarray): Wavelength grid.
        raarray (np.ndarray): RA per fiber per wavelength.
        decarray (np.ndarray): Dec per fiber per wavelength.
        specarray (np.ndarray): Spectra per fiber per wavelength.
        errarray (np.ndarray): Errors per fiber per wavelength.
        shot_indices (list[np.ndarray]): Per-shot index arrays.
        tp: WCS helper.
        xg, yg, xgrid, ygrid: Grids for imaging.
        log: Logger.

    Returns:
        tuple: (cube, ecube, weightcube)
    """
    cube = np.zeros((len(def_wave),) + xgrid.shape, dtype='float32')
    ecube = np.zeros_like(cube)
    weightcube = np.zeros_like(cube)

    Pos = np.zeros((len(raarray), 2))
    for i in np.arange(len(def_wave)):
        x, y = tp.wcs_world2pix(raarray[:, i], decarray[:, i], 1)
        log.info('Working on wavelength %0.0f' % def_wave[i])
        Pos[:, 0], Pos[:, 1] = (x, y)
        data = specarray[:, i]
        edata = errarray[:, i]
        image, errorimage, weight = make_image_interp(Pos, data, edata, xg, yg, xgrid, ygrid, 1.8 / 2.35, shot_indices)
        cube[i, :, :] += image
        ecube[i, :, :] += errorimage
        weightcube[i, :, :] += weight
    return cube, ecube, weightcube


def write_cube(path, data, header, N):
    """Write a 3D cube to FITS with proper WCS headers.

    Args:
        path (str): Output file path.
        data (np.ndarray): 3D cube (nz, ny, nx).
        header (astropy.io.fits.Header): WCS header to update.
        N (int): Image size (sets CRPIX1/2 to image center).
    """
    h = header.copy()
    h['CRPIX1'] = (N + 1) / 2
    h['CRPIX2'] = (N + 1) / 2
    h['WCSAXES'] = 3
    h['CDELT3'] = 2.
    h['CRPIX3'] = 1.
    h['CRVAL3'] = 3470.
    h['CTYPE1'] = 'RA---TAN'
    h['CTYPE2'] = 'DEC--TAN'
    h['CTYPE3'] = 'WAVE'
    h['CUNIT1'] = 'deg'
    h['CUNIT2'] = 'deg'
    h['CUNIT3'] = 'Angstrom'
    h['SPECSYS'] = 'TOPOCENT'
    fits.PrimaryHDU(np.array(data, 'float32'), header=h).writeto(path, overwrite=True)


def evaluate_cube_stats(cube, ecube, weightcube, surname, log,
                        valid_frac_threshold=0.8, continuum_sigma=5.0, max_iter=2):
    """Evaluate S/N statistics and continuum-based clipping on the final cubes.

    This routine summarizes per-spaxel spectral behavior, selects spaxels with
    sufficiently well-behaved spectra, estimates a continuum image and removes
    high-continuum outliers, and finally measures the distribution of flux/error
    (SNR) values. A publication-quality histogram is saved alongside a standard
    normal reference curve.

    Args:
        cube (np.ndarray): Data cube of shape (nw, ny, nx).
        ecube (np.ndarray): Error cube of same shape as cube.
        weightcube (np.ndarray): Coverage/weight cube; wavelengths with weight>0 are considered covered.
        surname (str): Output tag used to write plot and optional text summary.
        log (logging.Logger): Logger for status output.
        valid_frac_threshold (float, optional): Minimum fraction of covered wavelengths in a spaxel that must have
            positive finite flux (>0) to consider the spaxel for SNR stats. Default 0.8.
        continuum_sigma (float, optional): Sigma multiple of MAD to clip high-continuum spaxels. Default 5.0.
        max_iter (int, optional): Maximum clipping iterations. Default 2.

    Returns:
        dict: Summary statistics including counts and robust SNR location/scale.
    """
    try:
        nw, ny, nx = cube.shape
    except Exception:
        log.warning('evaluate_cube_stats: cube has unexpected shape; skipping stats.')
        return {}

    # Coverage mask per wavelength and spaxel
    covered = weightcube > 0

    # Compute fraction of positive finite flux per spaxel over covered wavelengths
    finite_flux = np.isfinite(cube)
    good_spaxels = finite_flux & covered
    frac_pos =  good_spaxels.sum(axis=0)[np.newaxis, :, :] / good_spaxels.shape[0]
    pix_sel = frac_pos >= float(valid_frac_threshold)
    spax_sel = good_spaxels & pix_sel

    n_spax_total = int((covered > 0).sum())
    n_spax_sel0 = int(spax_sel.sum())
    log.info(f'Cube stats: total covered spaxels={n_spax_total}, meeting frac_pos>{valid_frac_threshold:.2f}: {n_spax_sel0}')

    if n_spax_sel0 == 0:
        log.warning('No spaxels meet the positive-fraction criterion; skipping stats plot.')
        return {
            'n_spax_total': n_spax_total,
            'n_spax_selected': 0,
            'biweight_snr': np.nan,
            'mad_snr': np.nan,
        }

    # Continuum image via robust collapse over covered wavelengths
    cube_masked = np.where(spax_sel, cube, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=RuntimeWarning)
        continuum_img = np.nanmedian(cube_masked, axis=0)

    # Robust stats over initially selected spaxels
    spax_idx = pix_sel & np.isfinite(continuum_img)
    cont_vals = continuum_img[spax_idx]
    if cont_vals.size == 0:
        log.warning('No finite continuum values among selected spaxels; skipping stats plot.')
        return {
            'n_spax_total': n_spax_total,
            'n_spax_selected': 0,
            'biweight_snr': np.nan,
            'mad_snr': np.nan,
        }

    med_c = np.nanmedian(cont_vals)
    mad_c = mad_std(cont_vals)
    thr = continuum_sigma * (mad_c if np.isfinite(mad_c) and mad_c > 0 else np.nanstd(cont_vals))
    if not np.isfinite(thr) or thr == 0:
        thr = continuum_sigma * (np.nanstd(cont_vals) if cont_vals.size > 1 else 0.0)

    inliers = spax_idx & (np.abs(continuum_img - med_c) <= thr)

    for _ in range(int(max_iter)):
        vals = continuum_img[inliers]
        if vals.size < 10:
            break
        med_c = np.nanmedian(vals)
        mad_c = mad_std(vals)
        thr = continuum_sigma * (mad_c if np.isfinite(mad_c) and mad_c > 0 else np.nanstd(vals))
        new_inliers = spax_idx & (np.abs(continuum_img - med_c) <= thr)
        if new_inliers.sum() == inliers.sum():
            break
        inliers = new_inliers

    n_spax_final = int(inliers.sum())
    log.info(f'Continuum clipping: kept {n_spax_final}/{n_spax_sel0} spaxels after |cont - med| <= {continuum_sigma:.1f} * MAD')

    # Gather SNR samples from final spaxels over covered wavelengths with valid errors
    # Build masks lazily to avoid large temporaries
    valid_err = np.isfinite(ecube) & (ecube > 0)
    valid_flux_any = np.isfinite(cube)
    valid = covered & valid_err & valid_flux_any

    # Mask down to inlier spaxels
    inliers_3d = np.broadcast_to(inliers, (nw, ny, nx))
    valid &= inliers_3d

    if not np.any(valid):
        log.warning('No valid flux/error samples after masking; skipping stats plot.')
        return {
            'n_spax_total': n_spax_total,
            'n_spax_selected': n_spax_final,
            'biweight_snr': np.nan,
            'mad_snr': np.nan,
        }

    snr = np.zeros(valid.sum(), dtype=np.float32)
    # Efficient extraction without building full flattened arrays
    # Iterate over wavelength slabs to limit peak memory
    idx = 0
    for i in range(nw):
        m = valid[i]
        if np.any(m):
            v = cube[i][m] / ecube[i][m]
            n = v.size
            snr[idx:idx+n] = v
            idx += n
    snr = snr[:idx]

    # Compute robust location and scale on clipped range for visualization
    # Keep all values for biweight, but we will draw histogram only within [-10, 10]
    bi = biweight(snr)
    mad_s = mad_std(snr)

    # Histogram within [-10, 10]
    hmin, hmax = -10.0, 10.0
    clip_mask = (snr >= hmin) & (snr <= hmax) & np.isfinite(snr)
    snr_plot = snr[clip_mask]
    if snr_plot.size < 10:
        log.warning(f'Only {snr_plot.size} SNR samples within [{hmin},{hmax}]; histogram may be uninformative.')

    fig, ax = plt.subplots(figsize=(7, 5))
    bins = np.linspace(hmin, hmax, 201)
    ax.hist(snr_plot, bins=bins, histtype='stepfilled', alpha=0.7, color='tab:blue',
            density=True, label='Flux/Error')
    ax.set_yscale('log')
    x = 0.5 * (bins[:-1] + bins[1:])
    gauss = (1.0 / np.sqrt(2*np.pi)) * np.exp(-0.5 * x * x)
    ax.plot(x, gauss, 'k--', lw=1.5, label='N(0,1)')
    ax.axvline(bi, color='tab:red', lw=1.5, label=f'Biweight={bi:.3f}')
    ax.set_xlabel('Flux / Error')
    ax.set_ylabel('Density (log scale)')
    ax.set_title('Distribution of Flux/Error (SNR)')
    ax.set_xlim(hmin, hmax)
    ax.legend(frameon=False)
    out_png = op.basename(f'{surname}_snr_hist.png')
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    log.info(f'Saved SNR histogram to {out_png} (samples: total={snr.size}, in-plot={snr_plot.size})')

    # Log a compact summary and also write a small text file next to the plot
    summary = {
        'n_spax_total': n_spax_total,
        'n_spax_selected_initial': n_spax_sel0,
        'n_spax_selected': n_spax_final,
        'continuum_median': float(med_c),
        'continuum_mad': float(mad_c),
        'continuum_threshold': float(thr),
        'snr_count_total': int(snr.size),
        'snr_count_inplot': int(snr_plot.size),
        'biweight_snr': float(bi) if np.isfinite(bi) else np.nan,
        'mad_snr': float(mad_s) if np.isfinite(mad_s) else np.nan,
    }
    for k, v in summary.items():
        log.info(f'CubeStat {k}: {v}')

    try:
        txt_path = op.basename(f'{surname}_cube_stats.txt')
        with open(txt_path, 'w') as f:
            for k, v in summary.items():
                f.write(f'{k}: {v}\n')
        log.info(f'Wrote cube statistics to {txt_path}')
    except Exception as e:
        log.warning(f'Failed to write stats text file: {e}')

    return summary


def main():
    """Entrypoint for building a mosaic cube from H5 fiber files.

    This preserves the scientific behavior while improving structure and readability.
    """
    args = parse_args()
    def_wave = np.linspace(3470., 5540., 1036)
    h5files = sorted(glob.glob(args.h5files))
    args.log.info(f"Detected {len(h5files)} input file(s). Assuming each h5 contains 3 interleaved dithers by default.")

    # Parse bounding box string: "RA, Dec, size_arcmin"
    bounding_box = [float(corner.replace(' ', '')) for corner in args.image_center_size.split(',')]
    # Convert size from arcmin to arcsec for internal logic
    bounding_box[2] = bounding_box[2]  # value in arcmin; setup will handle rounding/scale
    # setup requires arcsec; original code multiplied by 60 within rounding; keep same effect by passing arcsec here
    bounding_box_arcsec = [bounding_box[0], bounding_box[1], bounding_box[2]]
    # setup grids and WCS
    N, xg, yg, xgrid, ygrid, tp, header = setup_wcs_and_grids([bounding_box_arcsec[0], bounding_box_arcsec[1], bounding_box_arcsec[2]], args.pixel_scale)
    args.log.info('Image size in pixels: %i' % N)

    # Preselect and coverage check
    total_cnt, per_file_sel, per_file_starts, shot_indices = preselect_inbounds(h5files, tp, xg, yg, args.log)
    coverage_precheck(h5files, tp, xg, yg, [bounding_box_arcsec[0], bounding_box_arcsec[1], int(bounding_box_arcsec[2] * 60.0)], args.log)

    # Build arrays and process files
    A = Astrometry(bounding_box_arcsec[0], bounding_box_arcsec[1], 0., (N + 1) / 2.0, (N + 1) / 2.0)
    raarray, decarray, specarray, errarray, norm_array, total_finite, total_nonzero = process_files(
        h5files, per_file_sel, per_file_starts, [bounding_box_arcsec[0], bounding_box_arcsec[1], bounding_box_arcsec[2]], tp, A, args.log)

    # Assemble cubes
    cube, ecube, weightcube = assemble_cubes(def_wave, raarray, decarray, specarray, errarray, shot_indices, tp, xg, yg, xgrid, ygrid, args.log)

    # Post-assembly cube statistics and SNR histogram
    try:
        evaluate_cube_stats(cube, ecube, weightcube, args.surname, args.log,
                            valid_frac_threshold=0.8, continuum_sigma=5.0, max_iter=2)
    except Exception as e:
        args.log.warning(f'Cube statistics evaluation failed: {e}')

    # Write outputs
    write_cube(op.basename(f'{args.surname}_cube.fits'), cube, header, N)
    write_cube(op.basename(f'{args.surname}_errorcube.fits'), ecube, header, N)
    write_cube(op.basename(f'{args.surname}_weight_cube.fits'), weightcube, header, N)


if __name__ == '__main__':
    main()
