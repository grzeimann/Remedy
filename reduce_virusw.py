#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 12 10:04:09 2021

@author: gregz
"""

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
from input_utils import setup_logging, setup_parser
from math_utils import biweight
from scipy.interpolate import interp1d
from scipy.ndimage import percentile_filter

warnings.filterwarnings("ignore")


parser = setup_parser()

parser.add_argument("folder", help='''Input folder''', type=str)

parser.add_argument('outfolder', type=str,
                    help='''name of the output file''')

parser.add_argument("-lr", "--lowres",
                    help='''low resolution''',
                    action="count", default=0)

folder = '/Users/gregz/cure/Remedy/virusw/VIRUS-W_Jan21'
outfolder = '/Users/gregz/cure/Remedy/virusw/reductions'
argv = [folder, outfolder, "-lr"]
argv = None
args = parser.parse_args(args=argv)
folder = args.folder
outfolder = args.folder
log = setup_logging('virusw_reductions')

gain = 0.62
rdnoise = 2.55
# =============================================================================
# LowRes Mode
# =============================================================================
if args.lowres:
    lines = [4046.563, 4077.831, 4348.063, 4358.327, 4916.068, 5037.75, 
             5330.778, 5400.562, 5719.23, 5820.16, 5852.488, 5881.895, 
             5944.834, 5975.534]
    fiberref = 130
    xref = np.array([462.4524, 514.266, 990.160, 1009.781, 2042.973, 2275.308, 
                     2843.932, 2980.563, 3607.36, 
                     3806.076, 3870.173, 3928.036, 4050.715, 4109.934])
    use_kernel = True
    limit = 50
    def_wave = np.exp(np.linspace(np.log(3900), np.log(5971.295), 3650))
else:
    lines = [4916.07, 5037.751, 5116.503, 5144.938, 5203.896, 5330.778, 
             5341.094, 5400.56]
    fiberref = 130
    xref = np.array([1037.136, 1678.06, 2094.78, 2245.65, 2558.26, 3231.163, 
                     3285.645, 3599.408])
    use_kernel = False
    limit = 15
    def_wave = np.exp(np.linspace(np.log(4830), np.log(5489.81), 3200))


def identify_sky_pixels(sky, per=50, size=50):
    cont = percentile_filter(sky, per, size=size)
    try:
        mask = sigma_clip(sky - cont, masked=True, maxiters=None,
                          stdfunc=mad_std, sigma=5)
    except:
        mask = sigma_clip(sky - cont, iters=None, stdfunc=mad_std,
                          sigma=5) 
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
    
def get_wavelength(spectrum, trace, use_kernel=True, limit=50):
    init_fiber = fiberref
    wavelength = 0. * spectrum
    loc = xref * 1.
    W = np.zeros((trace.shape[0], len(lines)))
    W[init_fiber] = loc
    for i in np.arange(init_fiber)[::-1]:
        loc = get_arclines_fiber(spectrum[i], loc, limit=limit,
                                 use_kernel=use_kernel)
        W[i] = loc
    loc = xref * 1.
    for i in np.arange(init_fiber+1, spectrum.shape[0]):
        loc = get_arclines_fiber(spectrum[i], loc, limit=limit,
                                 use_kernel=use_kernel)
        W[i] = loc
    X = W * 0.
    xall = np.arange(trace.shape[1])
    res = W[0, :] * 0.
    for i in np.arange(W.shape[1]):
        x = 0. * W[:, i]
        for j in np.arange(W.shape[0]):
            x[j] = np.interp(W[j, i], xall, trace[j])
        X[:, i] = np.polyval(np.polyfit(x, W[:, i], 4), x)
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

def get_spectra(image, array_trace, npix=5):
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
    array_flt = np.zeros((image.shape[0]+2, image.shape[1]))
    array_flt[:-2, :] = image
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

def get_spectra_error(image, array_trace, npix=5):
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
    array_flt = np.zeros((image.shape[0]+2, image.shape[1]))
    array_flt[:-2, :] = image
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

def get_spectra_chi2(flt, sci, err,
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
    array_flt = np.zeros((flt.shape[0]+2, flt.shape[1]))
    array_flt[:-2, :] = flt
    array_sci = np.zeros((sci.shape[0]+2, sci.shape[1]))
    array_sci[:-2, :] = sci
    array_err = np.zeros((err.shape[0]+2, err.shape[1]))
    array_err[:-2, :] = err
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
    ref = np.loadtxt(op.join(DIRNAME, 'Fiber_Locations/20210512/virusw_fibloc.txt'))
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
        loc = loc[peaks > 0.01 * np.median(peaks)]+1
        P.append(loc)
        trace = get_trace_chunk(flat, loc)
        T = np.zeros((len(ref)))
        if len(trace) == N1:
            T[good] = trace
            for missing in np.where(ref[:, 1] == 1)[0]:
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
    return trace

def prep_image(data):
    image = np.hstack([data[:,:1024] - biweight(data[-43:,:1024]),
                      data[:,1124:] - biweight(data[-43:,1124:])])
    image = np.rot90(image)
    image = np.fliplr(image)
    return image

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

def write_fits(skysubrect, specrect, errorrect, header):
    hdulist = []
    for image, ftp, nn in zip([skysubrect, specrect, errorrect], 
                          [fits.PrimaryHDU, fits.ImageHDU,
                           fits.ImageHDU],
                           ['skysub', 'spectra', 'error']):
        hdu = ftp(np.array(image, dtype='float32'))
        hdu.header['CRVAL2'] = 1
        hdu.header['CRVAL1'] = np.log(def_wave)[0]
        hdu.header['CRPIX2'] = 1
        hdu.header['CRPIX1'] = 1
        hdu.header['CTYPE2'] = 'fiber'
        hdu.header['CTYPE1'] = 'ln(wave)'
        hdu.header['CDELT2'] = 1
        hdu.header['CDELT1'] = np.log(def_wave)[1] - np.log(def_wave)[0]
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
        t = Time(header['DATE-OBS'])
        objname = '_'.join(str(header['OBJECT']).split())
        iname = '_'.join([objname, t.strftime('%Y%m%dT%H%M%S'), 'multi'])
        hdu.header['EXTNAME'] = nn
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

def get_script_path():
    '''
    Get script path, aka, where does Remedy live?
    '''
    return op.dirname(op.realpath(sys.argv[0]))

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
anames = ['nehg']
tnames = ['twi']
dfnames = ['domeflat']
snames = ['feige', 'bd']
bias_filenames = get_filenames(gnames, typelist, bnames)
twiflt_filenames = get_filenames(gnames, typelist, tnames)
domeflt_filenames = get_filenames(gnames, typelist, dfnames)
arc_filenames = get_filenames(gnames, typelist, anames)
std_filenames = get_filenames(gnames, typelist, snames)
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

if len(domeflt_filenames) > 0:
    flt_filenames = domeflt_filenames
else:
    flt_filenames = twiflt_filenames

# =============================================================================
# Use the file numbers for connecting blocks of observations
# =============================================================================
biasnum = [int(op.basename(f).split('vw')[1][:6]) for f in bias_filenames]
fltnum = [int(op.basename(f).split('vw')[1][:6]) for f in flt_filenames]
arcnum = [int(op.basename(f).split('vw')[1][:6]) for f in arc_filenames]
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
print(flt_filenames)
for masterflt, mtime in zip(masterflt_list, flttime_list):
    masterbias = masterbias_list[get_cal_index(mtime, biastime_list)]
    fits.HDUList([fits.PrimaryHDU(masterbias), 
                  fits.ImageHDU(masterflt)]).writeto(op.join(outfolder, 'test.fits'), 
                                                     overwrite=True)
    trace = get_trace(masterflt-masterbias)
    trace_list.append(trace)
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
    trace = trace_list[get_cal_index(mtime, flttime_list)]
    lamp_spec = get_spectra(masterarc-masterbias, trace)
    try:
        wavelength, res, X, W = get_wavelength(lamp_spec, trace, 
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

for fn in sci_filenames:
    log.info('Reducing: %s' % op.basename(fn))
    f = fits.open(fn)
    t = Time(f[0].header['DATE-OBS'])
    image, E = base_reduction(f[0].data, masterbias)
    spec = get_spectra(image, trace)
    specerr = get_spectra_error(E, trace)
    chi2 = get_spectra_chi2(masterflt-masterbias, image, E, trace)
    badpix = chi2 > 20.
    specerr[badpix] = np.nan
    spec[badpix] = np.nan
    specrect, errrect = rectify(spec, specerr, wavelength, def_wave)
    specrect[:] /= (ftf * f[0].header['EXPTIME'])
    errrect[:] /= (ftf * f[0].header['EXPTIME'])
    skysubrect = subtract_sky(specrect, np.ones((len(specrect),), dtype=bool))
    write_fits(skysubrect, specrect, errrect, f[0].header)
# Build the write a fits file out