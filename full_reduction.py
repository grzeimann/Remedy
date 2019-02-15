# -*- coding: utf-8 -*-
"""
Created on Tue Jan 22 15:11:39 2019

@author: gregz
"""

import argparse as ap
import glob
import numpy as np
import os.path as op
import sys
import warnings

from astrometry import Astrometry
from astropy.convolution import convolve, Gaussian2DKernel
from astropy.io import fits
from astropy.stats import biweight_location
from datetime import datetime, timedelta
from input_utils import setup_logging
from scipy.interpolate import griddata, interp1d
from scipy.signal import savgol_filter
from tables import open_file


# Configuration    
amps = ['LL', 'LU', 'RL', 'RU']
dither_pattern = np.array([[0., 0.], [1.27, -0.73], [1.27, 0.73]])
def_wave = np.arange(3470., 5542., 2.)
color_dict = {'blue': [3600., 3900.], 'green': [4350, 4650],
              'red': [5100., 5400.]}
for col_ in color_dict:
     color_dict[col_].append(np.searchsorted(def_wave, color_dict[col_][0]))
     color_dict[col_].append(np.searchsorted(def_wave, color_dict[col_][1]))

parser = ap.ArgumentParser(add_help=True)

parser.add_argument("date",
                    help='''Date for reduction''',
                    type=str, default=None)

parser.add_argument("observation",
                    help='''Observation ID''',
                    type=int, default=None)

parser.add_argument("hdf5file",
                    help='''HDF5 calibration file''',
                    type=str, default=None)

parser.add_argument("-r", "--rootdir",
                    help='''Directory for raw data. 
                    (just before date folders)''',
                    type=str, default='/work/03946/hetdex/maverick')

parser.add_argument("-ra", "--ra",
                    help='''RA of the FP 0,0 (IHMP)''',
                    type=float, default=None)

parser.add_argument("-dec", "--dec",
                    help='''Dec of the FP 0,0 (IHMP)''',
                    type=float, default=None)

parser.add_argument("-fp", "--fplane_file",
                    help='''fplane file''',
                    type=str, default=None)


args = parser.parse_args(args=None)

def build_path(rootdir, date, obs, ifuslot, amp, base='sci', exp='exp*',
               instrument='virus'):
    if obs != '*':
        obs = '%07d' % obs
    path = op.join(rootdir, date, instrument, '%s%s' % (instrument, obs),
                   exp, instrument, '2*%s%s*%s.fits' % (ifuslot, amp, base))
    return path
    

def orient_image(image, amp, ampname):
    '''
    Orient the images from blue to red (left to right)
    Fibers are oriented to match configuration files
    '''
    if amp == "LU":
        image[:] = image[::-1, ::-1]
    if amp == "RL":
        image[:] = image[::-1, ::-1]
    if ampname is not None:
        if ampname == 'LR' or ampname == 'UL':
            image[:] = image[:, ::-1]
    return image


def base_reduction(filename, get_header=False):
    # Load fits file
    a = fits.open(filename)
    image = np.array(a[0].data, dtype=float)
    
    # Overscan subtraction
    overscan_length = 32 * (image.shape[1] / 1064)
    O = biweight_location(image[:, -(overscan_length-2):])
    image[:] = image - O
    
    # Trim image
    image = image[:, :-overscan_length]
    
    # Gain multiplication (catch negative cases)
    gain = a[0].header['GAIN']
    gain = np.where(gain > 0., gain, 0.85)
    rdnoise = a[0].header['RDNOISE']
    rdnoise = np.where(rdnoise > 0., rdnoise, 3.)
    amp = (a[0].header['CCDPOS'].replace(' ', '') +
           a[0].header['CCDHALF'].replace(' ', ''))
    try:
        ampname = a[0].header['AMPNAME']
    except:
        ampname = None
    header = a[0].header
    
    # Orient image
    a = orient_image(image, amp, ampname) * gain
    
    # Calculate error frame
    E = np.sqrt(rdnoise**2 + np.where(a > 0., a, 0.))
    if get_header:
        return a, E, header
    return a, E


def get_mastertwi(twi_path, masterbias):
    # Get twi files
    files = glob.glob(twi_path)
    listtwi = []
    # Reduce twi frames
    for filename in files:
        a, e = base_reduction(filename)
        a[:] -= masterbias
        listtwi.append(a)
    # Median combine after normalization, then re-normalize
    if len(listtwi) == 0:
        return None
    twi_array = np.array(listtwi, dtype=float)
    norm = np.median(twi_array, axis=(1, 2))[:, np.newaxis, np.newaxis]
    return np.median(twi_array / norm, axis=0) * np.median(norm)


def get_cal_path(pathname, date):
    # Get calibration date in case no cals found on initial date
    # Looking back 30 days to find cal
    datec = date
    daten = date
    cnt = 0
    while len(glob.glob(pathname)) == 0:
        datec_ = datetime(int(daten[:4]), int(daten[4:6]), int(daten[6:]))
        daten_ = datec_ - timedelta(days=1)
        daten = '%04d%02d%02d' % (daten_.year, daten_.month, daten_.day)
        pathname = pathname.replace(datec, daten)
        cnt += 1
        log.info(pathname)
        if cnt > 30:
            log.error('Could not find cal within 30 days.')
            break
    return pathname, daten


def get_script_path():
    return op.dirname(op.realpath(sys.argv[0]))


def get_interp_errors(x_fix, y_fix, x_var):
    x_repeat = np.tile(x_var[:,None], (len(x_fix),))
    distances = np.abs(x_repeat - x_fix)
    x_indices = np.searchsorted(x_fix, x_var)
    good = (x_indices > 0) * (x_indices < (len(x_indices)-1))
    weights = np.zeros_like(distances)
    idx = np.arange(len(x_indices))[good]
    x_indices = x_indices[good]
    weights[idx, x_indices] = distances[idx, x_indices-1]
    weights[idx, x_indices-1] = distances[idx, x_indices]
    weights /= np.sum(weights, axis=1)[:, None]
    errors = np.sqrt(np.sum(weights * y_fix**2, axis=0))
    return errors


def get_spectra(array_sci, error_sci, array_flt, array_trace, wave, def_wave):
    # Get twi spectra as average of two pixels closest to trace
    # Get sci spectra as average of two pixels closest to trace normalized
    # by the twi frame for same pixels (aka, flat field extraction)
    sci_spectrum = np.zeros((array_trace.shape[0], def_wave.shape[0]))
    err_spectrum = np.zeros((array_trace.shape[0], def_wave.shape[0]))
    twi_spectrum = np.zeros((array_trace.shape[0], def_wave.shape[0]))
    N = array_flt.shape[0]
    x = np.arange(array_flt.shape[1])
    for fiber in np.arange(array_trace.shape[0]):
        dw = np.diff(wave[fiber])
        dw = np.hstack([dw[0], dw])
        if array_trace[fiber].min() < 0.:
            continue
        if np.ceil(array_trace[fiber]).max() >= N:
            continue
        indl = np.floor(array_trace[fiber]).astype(int)
        indh = np.ceil(array_trace[fiber]).astype(int)
        tw = array_flt[indl, x] / 2. + array_flt[indh, x] / 2.
        twi_spectrum[fiber] = np.interp(def_wave, wave[fiber], tw / dw,
                                        left=0.0, right=0.0)
        sw = (array_sci[indl, x] / array_flt[indl, x] +
              array_sci[indh, x] / array_flt[indh, x])
        ew = np.sqrt((error_sci[indl, x]**2 / array_flt[indl, x] +
                      error_sci[indh, x]**2 / array_flt[indh, x]))
        sci_spectrum[fiber] = np.interp(def_wave, wave[fiber], sw / dw,
                                        left=0.0, right=0.0)
        err_spectrum[fiber] = get_interp_errors(wave[fiber], ew / dw,
                                                def_wave)
    twi_spectrum[~np.isfinite(twi_spectrum)] = 0.0
    sci_spectrum[~np.isfinite(sci_spectrum)] = 0.0
    err_spectrum[~np.isfinite(err_spectrum)] = 0.0
    return twi_spectrum, sci_spectrum / 2., err_spectrum / 2.


def reduce_ifuslot(ifuloop, h5table):
    p, t, s, es = ([], [], [], [])
    for ind in ifuloop:
        # Grab calibration information from hdf5 table
        ifuslot = '%03d' % h5table[ind]['ifuslot']
        amp = h5table[ind]['amp']
        amppos = h5table[ind]['ifupos']
        wave = h5table[ind]['wavelength']
        trace = h5table[ind]['trace']
        try:
            masterbias = h5table[ind]['masterbias']
        except:
            masterbias = 0.0
        
        # Get twilight frames and make master twi
        twibase = build_path(args.rootdir, args.date, '*', ifuslot, amp,
                             base='twi')
        twibase, newdate = get_cal_path(twibase, args.date)
    
        if newdate != args.date:
            log.info('Found twi files on %s and using them for %s' %
                     (newdate, args.date))
        log.info('Making mastertwi for %s%s' % (ifuslot, amp))
        masterflt = get_mastertwi(twibase, masterbias)
        if masterflt is None:
            continue

        filenames = build_path(args.rootdir, args.date, args.observation,
                               ifuslot, amp)
        filenames = sorted(glob.glob(filenames))

        log.info('Extracting spectra for %s%s' % (ifuslot, amp))
        for j, fn in enumerate(filenames):
            sciimage, scierror = base_reduction(fn)
            sciimage[:] = sciimage - masterbias
            twi, spec, espec = get_spectra(sciimage, masterflt, trace, wave,
                                           def_wave)
            pos = amppos + dither_pattern[j]
            for x, i in zip([p, t, s, es], [pos, twi, spec, espec]):
                x.append(i * 1.)        
    p, t, s, es = [np.vstack(j) for j in [p, t, s, es]]
    return p, t, s, es, fn 
            

def output_fits(image, fn):
    PA = fits.open(fn)[0].header['PARANGLE']
    imscale = 48. / image.shape[0]
    crx = image.shape[1] / 2.
    cry = image.shape[0] / 2.
    ifuslot = '%03d' % args.ifuslot
    if (args.ra is None) or (args.dec is None):
        log.info('Using header for RA and Dec')
        RA = fits.open(fn)[0].header['TRAJCRA'] * 15.
        DEC = fits.open(fn)[0].header['TRAJCDEC']
        A = Astrometry(RA, DEC, PA, 0., 0., fplane_file=args.fplane_file)
        A.get_ifuslot_projection(ifuslot, imscale, crx, cry)
        wcs = A.tp_ifuslot
    else:
        A = Astrometry(args.ra, args.dec, PA, 0., 0.,
                       fplane_file=args.fplane_file)
        wcs = A.setup_TP(args.ra, args.dec, A.rot, crx, 
                         cry, x_scale=-imscale, y_scale=imscale)
    header = wcs.to_header()
    F = fits.PrimaryHDU(np.array(image, 'float32'), header=header)
    #for hi in header:
    #    F.header[hi] = header[hi]
    F.writeto('%s_%07d_%03d.fits' %
              (args.date, args.observation, args.ifuslot), overwrite=True)


def find_cosmics(xloc, yloc, data, radius=1.5, thresh=0.5):
    D = np.sqrt((xloc - xloc[:, np.newaxis])**2 +
                (yloc - yloc[:, np.newaxis])**2)
    W = np.zeros(D.shape, dtype=bool)
    W[D < radius] = True
    mask = np.zeros(data.shape, dtype=bool)
    for k in np.arange(data.shape[1]):
        sel = (data[:, k] / (data[:, k] * W).sum(axis=1)) >= 0.5
        mask[sel, k] = True
    return mask


def extract_source(xloc, yloc, data, Dx, Dy, ftf,
                   scale=0.25, seeing_fac=2.3):
    seeing = seeing_fac / scale
    a, b = data.shape
    x = np.arange(xloc.min(),
                  xloc.max()+1*scale, scale)
    y = np.arange(yloc.min(),
                  yloc.min()+1*scale, scale)
    xgrid, ygrid = np.meshgrid(x, y)
    zgrid = np.zeros((b,)+xgrid.shape)
    area = np.pi * 0.75**2
    G = Gaussian2DKernel(seeing / 2.35)
    S = np.zeros((data.shape[0], 2))
    
    for k in np.arange(b):
        S[:, 0] = xloc - Dx[k]
        S[:, 1] = yloc - Dy[k]
        sel = np.isfinite(data[:, k]) * (data[:, k] != 0.0)
        if np.any(sel):
            grid_z = (griddata(S[sel], data[sel, k], (xgrid, ygrid),
                               method='cubic') * scale**2 / area)
            zgrid[k, :, :] = (convolve(grid_z, G, boundary='extend') *
                              scale**2 / area)
    
    return 


def write_cube(wave, xgrid, ygrid, zgrid, outname, he):
    hdu = fits.PrimaryHDU(np.array(zgrid, dtype='float32'))
    hdu.header['CRVAL1'] = xgrid[0, 0]
    hdu.header['CRVAL2'] = ygrid[0, 0]
    hdu.header['CRVAL3'] = wave[0]
    hdu.header['CRPIX1'] = 1
    hdu.header['CRPIX2'] = 1
    hdu.header['CRPIX3'] = 1
    hdu.header['CTYPE1'] = 'pixel'
    hdu.header['CTYPE2'] = 'pixel'
    hdu.header['CTYPE3'] = 'pixel'
    hdu.header['CDELT1'] = xgrid[0, 1] - xgrid[0, 0]
    hdu.header['CDELT2'] = ygrid[1, 0] - ygrid[0, 0]
    hdu.header['CDELT3'] = wave[1] - wave[0]
    for key in he.keys():
        if key in hdu.header:
            continue
        if ('CCDSEC' in key) or ('DATASEC' in key):
            continue
        if ('BSCALE' in key) or ('BZERO' in key):
            continue
        hdu.header[key] = he[key]
    hdu.writeto(outname, overwrite=True)


def subtract_sky(scispectra, nchunk=12):
    N = len(scispectra) / 112
    nexp = N / 4
    amps = np.array(np.array_split(scispectra, N, axis=0))
    if nexp > 1:
        for i in np.arange(1, nexp):
            sel = amps[i::nexp] != 0.
            r = np.median(amps[::nexp][sel] / amps[i::nexp][sel])
            amps[i::nexp] *= r
    d = np.reshape(amps, scispectra.shape)
    def evalf(x, n, avg=1.):
        X = np.arange(n*0.95, n*1.055, n*0.005)
        chi2 = X * 0.
        for j, l in enumerate(X):
            chi2[j] = (x - l)**2 + (l - avg)**2
        i = np.argmin(chi2)
        return X[i]
    W = np.array([np.mean(chunk) 
                  for chunk in np.array_split(def_wave, nchunk)])
    S = np.array_split(d, nchunk, axis=1)
    ftf = np.zeros((len(d), len(S)))
    for k, si in zip(np.arange(len(S)), S):
        log.info('Working on chunk %i of %i for fiber to fiber' %
                 (k+1, nchunk))
        s = np.ma.array(si, mask=si==0.)
        n = np.ma.median(s)
        x = np.ma.median(s, axis=1)
        y = n * np.ones(x.shape)
        ns = np.array_split(np.arange(len(y)), N)
        for i in np.arange(3):
            for nsi in ns:
                y[nsi] = np.median(y[nsi])
            for j in np.arange(len(x)):
                y[j] = evalf(x[j], n, avg=y[j])
        for nsi in ns:
            y[nsi] = savgol_filter(y[nsi], 11, 1)
        ftf[:, k] = y / n
    for i in np.arange(len(scispectra)):
        I = interp1d(W, ftf[i, :], kind='quadratic', fill_value='extrapolate')
        scispectra[i] /= I(def_wave)
    log.info('Getting average sky')
    sky = np.nanmedian(scispectra, axis=0)
    return scispectra - sky, sky

# GET DIRECTORY NAME FOR PATH BUILDING
DIRNAME = get_script_path()
instrument = 'virus'
log = setup_logging()
if args.hdf5file is None:
    args.hdf5file = op.join(DIRNAME, 'cals', 'default_cals.h5')

# OPEN HDF5 FILE
h5file = open_file(args.hdf5file, mode='r')
h5table = h5file.root.Cals

# Collect indices for ifuslot
ifuslots = h5table.cols.ifuslot[:]
ifuloop = np.arange(len(ifuslots))

# Reducing IFUSLOT
log.info('Reducing all ifus in CAL HDF5')
pos, twispectra, scispectra, errspectra, fn = reduce_ifuslot(ifuloop, h5table)
average_twi = np.percentile(twispectra, 95, axis=0)
scispectra = scispectra * average_twi
errspectra = errspectra * average_twi
# fits.PrimaryHDU(scispectra).writeto('test2.fits', overwrite=True)

# Subtracting Sky
log.info('Subtracting sky.')
scispectra, sky = subtract_sky(scispectra)
N = np.vstack([scispectra, sky])
fits.PrimaryHDU(N).writeto('test.fits', overwrite=True)
#
## Collapse image
#log.info('Making collapsed frame')
#color = color_dict['red']
#image = np.median(scispectra[:, color[2]:color[3]], axis=1)
#
## Normalize exposures and amps together using 20%-tile
#log.info('Building collapsed frame')
#grid_x, grid_y = np.meshgrid(np.linspace(-23, 25, 401),
#                             np.linspace(-23, 25, 401))
#sel = ftf > 0.5
#grid_z0 = griddata(pos[sel], image[sel], (grid_x, grid_y), method='cubic')
#G = Gaussian2DKernel(8)
#image = convolve(grid_z0, G, boundary='extend')
#output_fits(image, fn)
#
#
#wADR = [3500., 4000., 4500., 5000., 5500.]
#ADRx = [-0.74, -0.4, -0.08, 0.08, 0.20]
#ADRx = np.polyval(np.polyfit(wADR, ADRx, 3), def_wave)
#
## Making data cube
#log.info('Making Cube')
#with warnings.catch_warnings():
#    warnings.simplefilter("ignore")
#    zgrid, xgrid, ygrid = make_frame(pos[:, 0], pos[:, 1], scispectra, 
#                                     ADRx, 0. * def_wave, ftf)
#
#he = fits.open(fn)[0].header
#write_cube(def_wave, xgrid, ygrid, zgrid,'%s_%07d_%03d_cube.fits' %
#           (args.date, args.observation, args.ifuslot) , he)


h5file.close()
    
        