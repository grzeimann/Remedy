#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 12 10:04:09 2021

@author: gregz
"""

import argparse as ap
import glob
import numpy as np
import os.path as op
import sys
import warnings
from astropy.convolution import convolve, Box1DKernel
from astropy.io import fits
from astropy.stats import sigma_clip
from astropy.stats import mad_std
from astropy.time import Time
from distutils.dir_util import mkpath
from input_utils import setup_logging
from math_utils import biweight
from scipy.interpolate import interp1d
from scipy.ndimage import percentile_filter
from sklearn.decomposition import PCA

# Turn off annoying warnings (even though some deserve attention)
warnings.filterwarnings("ignore")

parser = ap.ArgumentParser(add_help=True)

parser.add_argument("folder", help='''Input folder''', type=str)

parser.add_argument('outfolder', type=str,
                    help='''name of the output file''')

parser.add_argument('-n', '--name', type=str,
                    help='''Name of the science target''', default=None)

parser.add_argument("-b", "--blue",
                    help='''blue Side?''',
                    action="count", default=0)

parser.add_argument("-bn", "--binned",
                    help='''Binned?''',
                    action="count", default=0)

parser.add_argument("-ra", "--reduce_all",
                    help='''Reduce all files in folder''',
                    action="count", default=0)

folder = '/Users/gregz/cure/Remedy/virusp/raw'
outfolder = '/Users/gregz/cure/Remedy/virusp/reductions'
argv = None
args = parser.parse_args(args=argv)
folder = args.folder
outfolder = args.outfolder

# Make output folder if it doesn't exist
mkpath(outfolder)

log = setup_logging('virusp_reductions')

gain = 0.25
rdnoise = 3.7
addrows = 20
pca_comp = 125
# =============================================================================
# LowRes Mode
# =============================================================================
if args.blue:
    lines = [3611.287, 3652.103, 4046.554, 4358.325, 4413.776, 4678.147,
             4799.908, 4916.066, 5085.817, 5154.662, 5460.737, 5769.598, 
             5790.663]
    fiberref = 130
    xref = np.array([40.374, 74.775, 425.068, 699.606, 748.347,
                     981.728, 1089.410, 1192.438, 1343.708, 1405.401,
                     1681.501, 1965.260, 1984.816])
    if args.binned:
        xref = xref / 2.
    use_kernel = True
    limit = 100
    if args.binned:
        def_wave = np.linspace(3570., 5810., 1024)
    else:
        def_wave = np.linspace(3570., 5810., 2048)
else:
    lines = [4799.912, 5085.822, 5154.600, 6532.882, 6598.951, 6717.0429, 
             6929.467]
    fiberref = 130
    xref = np.array([31.350, 159.233, 189.951, 809.319, 839.854, 894.628, 
                     994.314])
    use_kernel = False
    limit = 100
    if args.binned:
        def_wave = np.linspace(4730., 6940., 1024)
    else:
        def_wave = np.linspace(4730., 6940., 2048)

def get_script_path():
    '''
    Get script path, aka, where does Remedy live?
    '''
    return op.dirname(op.realpath(sys.argv[0]))

def identify_sky_pixels(sky, per=50, size=50):
    cont = percentile_filter(sky, per, size=size)
    try:
        mask = sigma_clip(sky - cont, masked=True, maxiters=None,
                          stdfunc=mad_std, sigma=5)
    except:
        mask = sigma_clip(sky - cont, iters=None, stdfunc=mad_std,
                          sigma=5) 
    return mask.mask, cont

def get_skymask(sky, per=50, size=50, niter=3):
    sky_orig = sky * 1.
    for i in np.arange(niter):
        cont = percentile_filter(sky, per, size=size)
        try:
            mask = sigma_clip(sky - cont, masked=True, maxiters=None,
                              stdfunc=mad_std, sigma=5, sigma_lower=500)
        except:
            mask = sigma_clip(sky - cont, iters=None, stdfunc=mad_std,
                              sigma=5, sigma_lower=500)
        sky[mask.mask] = cont[mask.mask]
    try:
        mask = sigma_clip(sky_orig - cont, masked=True, maxiters=None,
                          stdfunc=mad_std, sigma=5, sigma_lower=500)
    except:
        mask = sigma_clip(sky_orig - cont, iters=None, stdfunc=mad_std,
                              sigma=5, sigma_lower=500)
    return mask.mask, cont

def rectify(scispectra, errspectra, wave_all, def_wave):
    scirect = np.zeros((scispectra.shape[0], len(def_wave)))
    errorrect = np.zeros((scispectra.shape[0], len(def_wave)))

    indices1 = np.ones(scirect.shape, dtype=int)
    for i in np.arange(scispectra.shape[0]):
        dw = np.diff(wave_all[i])
        dw = np.hstack([dw[0], dw])
        scirect[i] = np.interp(def_wave, wave_all[i], scispectra[i] / dw,
                               left=np.nan, right=np.nan)
        errorrect[i] = np.interp(def_wave, wave_all[i], errspectra[i] / dw,
                               left=np.nan, right=np.nan)
        indices1[i] = np.searchsorted(wave_all[i], def_wave) + i * 1032
    
#    x_var = (def_wave[np.newaxis, :] * np.ones((scirect.shape[0],1))).ravel()
#    x_fix = wave_all.ravel()
#    indices1 = indices1.ravel()
#    indices2 = indices1 - 1
#    indices2[indices2 < 0] = 0
#    indices1[indices1 >= len(x_fix)] = len(x_fix) - 1
#    
#    distances1 = np.abs(x_fix[indices1] - x_var)
#    distances2 = np.abs(x_fix[indices2] - x_var)
#    total_distance = distances1 + distances2
#    weight1 = distances1 / total_distance
#    weight2 = distances2 / total_distance
#    errorrect = (weight2**2 * errspectra.ravel()[indices1]**2 +
#                 weight1**2 * errspectra.ravel()[indices2]**2)
#    errorrect = np.sqrt(errorrect)
#    errorrect = np.reshape(errorrect, scirect.shape)
#    errorrect[np.isnan(scirect)] = np.nan
    return scirect, errorrect

def get_fiber_to_fiber(spectrum, n_chunks=100):
    average = biweight(spectrum, axis=0)
    initial_ftf = spectrum / average[np.newaxis, :]
    X = np.arange(spectrum.shape[1])
    x = np.array([np.mean(chunk) 
                  for chunk in np.array_split(X, n_chunks)])
    ftf = spectrum * 0.
    for i in np.arange(len(spectrum)):
        y = np.array([biweight(chunk) 
                      for chunk in np.array_split(initial_ftf[i], n_chunks)])  
        sel = np.isfinite(y)
        I = interp1d(x[sel], y[sel], kind='quadratic', bounds_error=False,
                     fill_value='extrapolate')
        ftf[i] = I(X)
        
    return initial_ftf, ftf
    
def get_wavelength(spectrum, trace, good, use_kernel=True, limit=100):
    init_fiber = fiberref
    wavelength = 0. * spectrum
    loc = xref * 1.
    W = np.zeros((trace.shape[0], len(lines)))
    W[init_fiber] = loc
    for i in np.arange(init_fiber)[::-1]:
        mask, cont = identify_sky_pixels(spectrum[i])
        y = spectrum[i] - cont
        if good[i]:
            loc = get_arclines_fiber(y, loc, limit=limit,
                                 use_kernel=use_kernel)
            W[i] = loc
    loc = xref * 1.
    for i in np.arange(init_fiber+1, spectrum.shape[0]):
        mask, cont = identify_sky_pixels(spectrum[i])
        y = spectrum[i] - cont
        if good[i]:
            loc = get_arclines_fiber(y, loc, limit=limit,
                                 use_kernel=use_kernel)
            W[i] = loc
    X = W * 0.
    xall = np.arange(trace.shape[1])
    res = W[0, :] * 0.
    for i in np.arange(W.shape[1]):
        x = 0. * W[:, i]
        bad = np.where(~good)[0]
        gind = np.where(good)[0]
        for b in bad:
            W[b, i] = W[gind[np.argmin(np.abs(b-gind))], i]
        for j in np.arange(W.shape[0]):
            x[j] = np.interp(W[j, i], xall, trace[j])
        
        sel = W[:, i] > 0.
        X[:, i] = np.polyval(np.polyfit(x[sel], W[sel, i], 4), x)
        dummy, res[i] = biweight(X[:, i] - W[:, i], calc_std=True)
    for j in np.arange(W.shape[0]):
        wavelength[j] = np.polyval(np.polyfit(X[j], lines, 3), xall)
    return wavelength, res, X, W

def get_arclines_fiber(spectrum, init_loc=None, limit=1000, use_kernel=True):
    if use_kernel:
        B = Box1DKernel(9.5)
        y1 = convolve(spectrum, B)
    else:
        y1 = spectrum * 1.
    diff_array = y1[1:] - y1[:-1]
    loc = np.where((diff_array[:-1] > 0.) * (diff_array[1:] < 0.))[0]
    peaks = y1[loc+1]
    loc = loc[peaks > limit] + 1        
    peaks = y1[loc]
    def get_trace_chunk(flat, XN):
            YM = np.arange(flat.shape[0])
            inds = np.zeros((3, len(XN)))
            inds[0] = XN - 1.
            inds[1] = XN + 0.
            inds[2] = XN + 1.
            inds = np.array(inds, dtype=int)
            Trace = (YM[inds[1]] - (flat[inds[2]] - flat[inds[0]]) /
                     (2. * (flat[inds[2]] - 2. * flat[inds[1]] + flat[inds[0]])))
            return Trace
    loc = get_trace_chunk(y1, loc)
    if init_loc is not None:
        final_loc = []
        for i in init_loc:
            final_loc.append(loc[np.argmin(np.abs(np.array(loc)-i))])
        loc = final_loc
    return loc

def get_spectra(array_flt, array_trace, npix=5):
    '''
    Extract spectra by dividing the flat field and averaging the central
    two pixels
    
    Parameters
    ----------
    array_flt : 2d numpy array
        twilight image
    array_trace : 2d numpy array
        trace for each fiber
    wave : 2d numpy array
        wavelength for each fiber
    def_wave : 1d numpy array [GLOBAL]
        rectified wavelength
    
    Returns
    -------
    twi_spectrum : 2d numpy array
        rectified twilight spectrum for each fiber  
    '''
    spec = np.zeros((array_trace.shape[0], array_trace.shape[1]))
    N = array_flt.shape[0]
    x = np.arange(array_flt.shape[1])
    LB = int((npix+1)/2)
    HB = -LB + npix + 1
    for fiber in np.arange(array_trace.shape[0]):
        if np.round(array_trace[fiber]).min() < LB:
            continue
        if np.round(array_trace[fiber]).max() >= (N-LB):
            continue
        indv = np.round(array_trace[fiber]).astype(int)
        for j in np.arange(-LB, HB):
            if j == -LB:
                w = indv + j + 1 - (array_trace[fiber] - npix/2.)
            elif j == HB-1:
                w = (npix/2. + array_trace[fiber]) - (indv + j) 
            else:
                w = 1.
            spec[fiber] += array_flt[indv+j, x] * w
    return spec / npix

def get_spectra_error(array_flt, array_trace, npix=5):
    '''
    Extract spectra by dividing the flat field and averaging the central
    two pixels
    
    Parameters
    ----------
    array_flt : 2d numpy array
        twilight image
    array_trace : 2d numpy array
        trace for each fiber
    wave : 2d numpy array
        wavelength for each fiber
    def_wave : 1d numpy array [GLOBAL]
        rectified wavelength
    
    Returns
    -------
    twi_spectrum : 2d numpy array
        rectified twilight spectrum for each fiber  
    '''

    spec = np.zeros((array_trace.shape[0], array_trace.shape[1]))
    N = array_flt.shape[0]
    x = np.arange(array_flt.shape[1])
    LB = int((npix+1)/2)
    HB = -LB + npix + 1
    for fiber in np.arange(array_trace.shape[0]):
        if np.round(array_trace[fiber]).min() < LB:
            continue
        if np.round(array_trace[fiber]).max() >= (N-LB):
            continue
        indv = np.round(array_trace[fiber]).astype(int)
        for j in np.arange(-LB, HB):
            if j == -LB:
                w = indv + j + 1 - (array_trace[fiber] - npix/2.)
            elif j == HB-1:
                w = (npix/2. + array_trace[fiber]) - (indv + j) 
            else:
                w = 1.
            spec[fiber] += array_flt[indv+j, x]**2 * w
    return np.sqrt(spec) / npix

def get_spectra_chi2(array_flt, array_sci, array_err,
                     array_trace, npix=5):
    '''
    Extract spectra by dividing the flat field and averaging the central
    two pixels
    
    Parameters
    ----------
    array_flt : 2d numpy array
        twilight image
    array_trace : 2d numpy array
        trace for each fiber
    wave : 2d numpy array
        wavelength for each fiber
    def_wave : 1d numpy array [GLOBAL]
        rectified wavelength
    
    Returns
    -------
    twi_spectrum : 2d numpy array
        rectified twilight spectrum for each fiber  
    '''
    spec = np.zeros((array_trace.shape[0], array_trace.shape[1]))
    N = array_flt.shape[0]
    x = np.arange(array_flt.shape[1])
    LB = int((npix+1)/2)
    HB = -LB + npix + 1
    for fiber in np.arange(array_trace.shape[0]):
        chi2 = np.zeros((npix+1, 3, len(x)))
        if np.round(array_trace[fiber]).min() < LB:
            continue
        if np.round(array_trace[fiber]).max() >= (N-LB):
            continue
        indv = np.round(array_trace[fiber]).astype(int)
        for j in np.arange(-LB, HB):
            if j == -LB:
                w = indv + j + 1 - (array_trace[fiber] - npix/2.)
            elif j == HB-1:
                w = (npix/2. + array_trace[fiber]) - (indv + j) 
            else:
                w = 1.
            chi2[j+LB, 0] = array_sci[indv+j, x] * w
            chi2[j+LB, 1] = array_flt[indv+j, x] * w
            chi2[j+LB, 2] = array_err[indv+j, x] * w
        norm = chi2[:, 0].sum(axis=0) / chi2[:, 1].sum(axis=0)
        num = (chi2[:, 0] - chi2[:, 1] * norm[np.newaxis, :])**2
        denom = (chi2[:, 2] + 0.01*chi2[:, 0].sum(axis=0)[np.newaxis, :])**2
        spec[fiber] = 1. / (1. + npix) * np.sum(num / denom, axis=0)
    return spec

def get_trace(twilight):
    ref = np.loadtxt(op.join(DIRNAME, 'Fiber_Locations/20210512/virusp_fibloc.txt'))
    N1 = (ref[:, 1] == 0.).sum()
    good = np.where(ref[:, 1] == 0.)[0]
    def get_trace_chunk(flat, XN):
        YM = np.arange(flat.shape[0])
        inds = np.zeros((3, len(XN)))
        inds[0] = XN - 1.
        inds[1] = XN + 0.
        inds[2] = XN + 1.
        inds = np.array(inds, dtype=int)
        Trace = (YM[inds[1]] - (flat[inds[2]] - flat[inds[0]]) /
                 (2. * (flat[inds[2]] - 2. * flat[inds[1]] + flat[inds[0]])))
        return Trace
    image = twilight
    if args.binned:
        N = 40
    else:
        N = 80
    xchunks = np.array([np.mean(x) for x in
                        np.array_split(np.arange(image.shape[1]), N)])
    chunks = np.array_split(image, N, axis=1)
    flats = [np.mean(chunk, axis=1) for chunk in chunks]
    Trace = np.zeros((len(ref), len(chunks)))
    k = 0
    P = []
    for flat, x in zip(flats, xchunks):
        diff_array = flat[1:] - flat[:-1]
        loc = np.where((diff_array[:-1] > 0.) * (diff_array[1:] < 0.))[0]
        loc = loc[loc>2]
        peaks = flat[loc+1]
        loc = loc[peaks > 0.3 * np.median(peaks)]+1
        P.append(loc)
        trace = get_trace_chunk(flat, loc)
        T = np.zeros((len(ref)))
        if len(trace) > N1:
            trace = trace[-N1:]
        if len(trace) == N1:
            T[good] = trace
            for missing in np.where(ref[:, 1] == 1.)[0]:
                gind = np.argmin(np.abs(missing - good))
                T[missing] = (T[good[gind]] + ref[missing, 0] -
                              ref[good[gind], 0])
        if len(trace) == len(ref):
            T = trace
        Trace[:, k] = T
        k += 1
    x = np.arange(twilight.shape[1])
    trace = np.zeros((Trace.shape[0], twilight.shape[1]))
    for i in np.arange(Trace.shape[0]):
        sel = Trace[i, :] != 0.
        trace[i] = np.polyval(np.polyfit(xchunks[sel], Trace[i, sel], 7), x)
    return trace, (ref[:, 1] == 0.)

def prep_image(data):
    if args.binned:
        image = data[:, :1024] - biweight(data[:, 1026:])
    else:
        image = data[:, :2048] - biweight(data[:, 2052:])
    new_image = np.zeros((len(image)+addrows, image.shape[1]))
    new_image[addrows:, :] = image
    return new_image

def base_reduction(data, masterbias):
    image = prep_image(data)
    image[:] -= masterbias
    image[:] *= gain
    E = np.sqrt(rdnoise**2 + np.where(image > 0., image, 0.))
    return image, E

def subtract_sky(spectra, good):
    nfibs, N = spectra.shape
    n1 = int(1. / 3. * N)
    n2 = int(2. / 3. * N)
    y = biweight(spectra[:, n1:n2], axis=1)
    mask, cont = identify_sky_pixels(y[good], size=15)
    m1 = ~good
    m1[good] = mask
    skyfibers = ~m1
    init_sky = biweight(spectra[skyfibers], axis=0)
    return spectra - init_sky[np.newaxis, :]

def get_pca_sky_residuals(data, ncomponents=5):
    pca = PCA(n_components=ncomponents)
    H = pca.fit_transform(data)
    A = np.dot(H, pca.components_)
    return pca, A

def get_residual_map(data, pca):
    res = data * 0.
    for i in np.arange(data.shape[1]):
        A = np.abs(data[:, i] - np.nanmedian(data[:, i]))
        good = A < (3. * np.nanmedian(A))
        sel = np.isfinite(data[:, i]) * good
        coeff = np.dot(data[sel, i], pca.components_.T[sel])
        model = np.dot(coeff, pca.components_)
        res[:, i] = model
    return res


def get_arc_pca(arcskysub, good, mask, components=15):
    X = arcskysub
    X[:, ~mask] = 0.
    X[~good] = 0.
    X = X.swapaxes(0, 1)
    M = np.nanmean(X, axis=1)
    Var = np.nanstd(X, axis=1)
    Var[Var==0.] = 1.
    X = (X-M[:, np.newaxis]) / Var[:, np.newaxis]
    X[np.isnan(X)] = 0.
    pca, A = get_pca_sky_residuals(X, ncomponents=components)
    return pca

def get_continuum(skysub, masksky, nbins=50):
    bigcont = skysub * 0.
    for j in np.arange(skysub.shape[0]):
        y = skysub[j] * 1.
        y[masksky] = np.nan
        x = np.array([np.mean(chunk) for chunk in np.array_split(np.arange(len(y)), nbins)])
        z = np.array([biweight(chunk) for chunk in np.array_split(y, nbins)])
        sel = np.isfinite(z)
        if sel.sum()>5:
            I = interp1d(x[sel], z[sel], kind='quadratic', bounds_error=False, 
                         fill_value='extrapolate')
            bigcont[j] = I(np.arange(len(y)))
    return bigcont
    
def reduce(fn, biastime_list, masterbias_list, flttime_list,
                 trace_list, wave_time, wave_list, ftf_list, pca=None):
    f = fits.open(fn)
    t = Time(f[0].header['DATE-OBS']+'T'+f[0].header['UT'])
    mtime = t.mjd
    masterbias = masterbias_list[get_cal_index(mtime, biastime_list)]
    image, E = base_reduction(f[0].data, masterbias)
    trace, good = trace_list[get_cal_index(mtime, flttime_list)]
    spec = get_spectra(image, trace)
    specerr = get_spectra_error(E, trace)
    chi2 = get_spectra_chi2(masterflt-masterbias, image, E, trace)
    badpix = chi2 > 20.
    specerr[badpix] = np.nan
    spec[badpix] = np.nan
    wavelength = wave_list[get_cal_index(mtime, wave_time)]
    specrect, errrect = rectify(spec, specerr, wavelength, def_wave)
    ftf = ftf_list[get_cal_index(mtime, flttime_list)]
    specrect[:] /= (ftf * f[0].header['EXPTIME'])
    errrect[:] /= (ftf * f[0].header['EXPTIME'])
    skymask, cont = get_skymask(biweight(specrect, axis=0), size=25)
    skysubrect = subtract_sky(specrect, good)
    if pca is None:
        pca = get_arc_pca(skysubrect, good, skymask, components=pca_comp)
        return pca
    skymask[1:] += skymask[:-1]
    skymask[:-1] += skymask[1:]
    cont1 = get_continuum(skysubrect, skymask, nbins=50)
    Z = skysubrect - cont1
    res = get_residual_map(Z, pca)
    res[:, ~skymask] = 0.0
    write_fits(skysubrect-res, skysubrect, specrect, errrect, f[0].header)
    return biweight(specrect, axis=0), cont

def write_fits(skysubrect_adv, skysubrect, specrect, errorrect, header):
    hdulist = []
    for image, ftp in zip([skysubrect_adv, skysubrect, specrect, errorrect], 
                          [fits.PrimaryHDU, fits.ImageHDU, fits.ImageHDU,
                           fits.ImageHDU]):
        hdu = ftp(np.array(image, dtype='float32'))
        hdu.header['CRVAL2'] = 1
        hdu.header['CRVAL1'] = def_wave[0]
        hdu.header['CRPIX2'] = 1
        hdu.header['CRPIX1'] = 1
        hdu.header['CTYPE2'] = 'fiber'
        hdu.header['CTYPE1'] = 'Angstrom'
        hdu.header['CDELT2'] = 1
        hdu.header['CDELT1'] = def_wave[1] - def_wave[0]
        for key in header.keys():
            if key in hdu.header:
                continue
            if ('CCDSEC' in key) or ('DATASEC' in key):
                continue
            if ('BSCALE' in key) or ('BZERO' in key):
                continue
            try:
                hdu.header[key] = header[key]
            except:
                continue
        t = Time(header['DATE-OBS']+'T'+header['UT'])
        objname = '_'.join(header['OBJECT'].split())
        iname = '_'.join([objname, t.strftime('%Y%m%dT%H%M%S'), 'multi'])
        hdulist.append(hdu)
    fits.HDUList(hdulist).writeto(op.join(outfolder, iname + '.fits'), overwrite=True)

def make_mastercal_list(filenames, breakind):
    breakind1 = np.hstack([0, breakind])
    breakind2 = np.hstack([breakind, len(filenames)+1])
    masters = []
    times = []
    for bk1, bk2 in zip(breakind1, breakind2):
        frames = [prep_image(fits.open(f)[0].data) 
                  for cnt, f in enumerate(filenames) 
                  if ((cnt > bk1) * (cnt < bk2))]
        t = [Time(fits.open(f)[0].header['DATE-OBS']).mjd 
             for cnt, f in enumerate(filenames) 
             if ((cnt > bk1) * (cnt < bk2))]
        masters.append(np.median(frames, axis=0))
        times.append(np.mean(t))
    return masters, times

def get_cal_index(mtime, time_list):
    return np.argmin(np.abs(mtime - np.array(time_list)))

def get_filenames(gnames, typelist, names):
    matches = []
    for gn, tp in zip(gnames, typelist):
        for name in names:
            if name in str(tp).lower():
                matches.append(gn)
    return np.array(matches)

# =============================================================================
# Get Folder and Filenames
# =============================================================================
filenames = sorted(glob.glob(op.join(folder, '*.fits')))
DIRNAME = get_script_path()

# =============================================================================
# Make a list of the objects for each filename, ignore those without 'OBJECT' 
# in the header
# =============================================================================
typelist = []
gnames = []
timelist = []
for f in filenames:
    try:
        obj = fits.open(f)[0].header['OBJECT']
        datestring = fits.open(f)[0].header['DATE-OBS']
    except:
        continue
    typelist.append(obj)
    gnames.append(f)
    timelist.append(Time(datestring))

# =============================================================================
# Get the bias filenames, domeflat filenames, and arc lamp filenames
# =============================================================================
log.info('Sorting Files')
bnames = ['bias', 'zero']
anames = ['arc', 'necd', 'hgcd']
tnames = ['twilight_flat']
dfnames = ['flat']
snames = ['feige', 'bd']
bias_filenames = get_filenames(gnames, typelist, bnames)
twiflt_filenames = get_filenames(gnames, typelist, tnames)
domeflt_filenames = get_filenames(gnames, typelist, dfnames)
arc_filenames = get_filenames(gnames, typelist, anames)
std_filenames = get_filenames(gnames, typelist, snames)
if args.reduce_all:
    gna = []
    for gn in gnames:
        if gn in bias_filenames:
            continue
        if gn in arc_filenames:
            continue
        if gn in twiflt_filenames:
            continue
        if gn in domeflt_filenames:
            continue
        gna.append(gn)
    sci_filenames = np.array(gna)
else:
    if args.name is not None:
        sci_filenames = get_filenames(gnames, typelist, [args.name])
    else:
        sci_filenames = []

if len(twiflt_filenames) > 0:
    flt_filenames = twiflt_filenames
else:
    flt_filenames = domeflt_filenames

# =============================================================================
# Use the file numbers for connecting blocks of observations
# =============================================================================
biasnum = [int(op.basename(f).split('.fits')[0][-4:]) for f in bias_filenames]
fltnum = [int(op.basename(f).split('.fits')[0][-4:]) for f in flt_filenames]
arcnum = [int(op.basename(f).split('.fits')[0][-4:]) for f in arc_filenames]
bias_breakind = np.where(np.diff(biasnum) > 1)[0]
flt_breakind = np.where(np.diff(fltnum) > 1)[0]
arc_breakind = np.where(np.diff(arcnum) > 1)[0]

# =============================================================================
# Make a master bias, master dome flat, and master arc for the first set of OBS
# =============================================================================
log.info('Making master bias frames')
masterbias_list, biastime_list = make_mastercal_list(bias_filenames,
                                                     bias_breakind)

log.info('Making master domeFlat frames')

masterflt_list, flttime_list = make_mastercal_list(flt_filenames,
                                                   flt_breakind)

log.info('Making master arc frames')
masterarc_list, arctime_list = make_mastercal_list(arc_filenames,
                                                   arc_breakind)


# =============================================================================
# Get trace from the dome flat
# =============================================================================
trace_list = []
fltspec = []
log.info('Getting trace for each master domeFlat')
for masterflt, mtime in zip(masterflt_list, flttime_list):
    masterbias = masterbias_list[get_cal_index(mtime, biastime_list)]
    trace, good = get_trace(masterflt-masterbias)
    trace_list.append([trace, good])
    domeflat_spec = get_spectra(masterflt-masterbias, trace)
    domeflat_error = 0. * domeflat_spec
    fltspec.append([domeflat_spec, domeflat_error])

# =============================================================================
# Get wavelength from arc lamps
# =============================================================================
wave_list = []
wave_time = []
bk1 = np.hstack([0, arc_breakind+1])
log.info('Getting wavelength for each master arc')
for masterarc, mtime, bk in zip(masterarc_list, arctime_list, bk1):
    masterbias = masterbias_list[get_cal_index(mtime, biastime_list)]
    trace, good = trace_list[get_cal_index(mtime, flttime_list)]
    lamp_spec = get_spectra(masterarc-masterbias, trace)
    try:
        wavelength, res, X, W = get_wavelength(lamp_spec, trace, good, 
                                               limit=limit, 
                                               use_kernel=use_kernel)
    except:
        log.warning('Could not get wavelength solution for masterarc')
        log.warning('First file of failed masterarc included: %s' %
                    (arc_filenames[bk]))
        continue
    wave_list.append(wavelength)
    wave_time.append(mtime)

# =============================================================================
# Rectify domeflat spectra and get fiber to fiber
# =============================================================================
ftf_list = []
log.info('Getting fiber to fiber for each master domeFlat')
for fltsp, mtime in zip(fltspec, flttime_list):
    wavelength = wave_list[get_cal_index(mtime, wave_time)]        
    domeflat_spec, domeflat_error = fltsp
    domeflat_rect, domeflat_error_rect = rectify(domeflat_spec, domeflat_error,
                                                 wavelength, def_wave)
    ftf, ftf_smooth = get_fiber_to_fiber(domeflat_rect)
    ftf_list.append(ftf)
    

pca = reduce(arc_filenames[0], biastime_list, masterbias_list, flttime_list,
             trace_list, wave_time, wave_list, ftf_list, pca=None)
for fn in sci_filenames:
    log.info('Reducing: %s' % fn)
    sky, cont = reduce(fn, biastime_list, masterbias_list, flttime_list,
                       trace_list, wave_time, wave_list, ftf_list, pca=pca)
