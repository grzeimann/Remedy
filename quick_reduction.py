# -*- coding: utf-8 -*-
"""
Created on Tue Jan 22 15:11:39 2019

@author: gregz
"""

import argparse as ap
import fnmatch
import glob
import numpy as np
import os.path as op
import sys
import tarfile
import warnings


from astrometry import Astrometry
from astropy.convolution import convolve, Gaussian2DKernel
from astropy.convolution import interpolate_replace_nans, Gaussian1DKernel
from astropy.io import fits
from astropy.stats import biweight_location, sigma_clip
from astropy.stats import sigma_clipped_stats
from astropy.table import Table
from datetime import datetime, timedelta
from extract import Extract
from input_utils import setup_logging
from photutils import DAOStarFinder, aperture_photometry, CircularAperture
from scipy.interpolate import griddata, interp1d
from tables import open_file

#################
# Configuration #
#################
# Amps
amps = ['LL', 'LU', 'RL', 'RU']
# Ignore these ifuslots
badifuslots = np.array([67, 46])
# Default dither pattern for 3 exposures
dither_pattern = np.array([[0., 0.], [1.27, -0.73], [1.27, 0.73]])
# Rectified wavelength
def_wave = np.arange(3470., 5542., 2.)
# ADR model
wADR = [3500., 4000., 4500., 5000., 5500.]
ADRx = [-0.74, -0.4, -0.08, 0.08, 0.20]
ADRx = np.polyval(np.polyfit(wADR, ADRx, 3), def_wave)

parser = ap.ArgumentParser(add_help=True)

parser.add_argument("date",
                    help='''Date for reduction''',
                    type=str, default=None)

parser.add_argument("observation",
                    help='''Observation ID''',
                    type=int, default=None)

parser.add_argument("ifuslot",
                    help='''ifuslot to reduced''',
                    type=int, default=None)

parser.add_argument("hdf5file",
                    help='''HDF5 calibration file''',
                    type=str, default=None)

parser.add_argument("-r", "--rootdir",
                    help='''Directory for raw data. 
                    (just before date folders)''',
                    type=str, default='/work/03946/hetdex/maverick')

parser.add_argument("-ra", "--ra",
                    help='''RA of the IFUSLOT to be reduced''',
                    type=float, default=None)

parser.add_argument("-dec", "--dec",
                    help='''Dec of the IFUSLOT to be reduced''',
                    type=float, default=None)

parser.add_argument("-fp", "--fplane_file",
                    help='''fplane file''',
                    type=str, default=None)

parser.add_argument("-nd", "--neighbor_dist",
                    help='''Distance in x or y in the focal plane IFUSLOT 
                    units to include in the reduction.''',
                    type=int, default=3)

parser.add_argument("-sf", "--source_file",
                    help='''file for spectrum to add in cube''',
                    type=str, default=None)

parser.add_argument("-s", "--simulate",
                    help='''Simulate source''',
                    action="count", default=0)

parser.add_argument("-sx", "--source_x",
                    help='''x-position for spectrum to add in cube''',
                    type=float, default=0.0)

parser.add_argument("-sy", "--source_y",
                    help='''y-position for spectrum to add in cube''',
                    type=float, default=0.0)

parser.add_argument("-ss", "--source_seeing",
                    help='''seeing conditions manually entered''',
                    type=float, default=1.5)

args = parser.parse_args(args=None)

def splitall(path):
    ''' Split path into list '''
    allparts = []
    while 1:
        parts = op.split(path)
        if parts[0] == path:  # sentinel for absolute paths
            allparts.insert(0, parts[0])
            break
        elif parts[1] == path: # sentinel for relative paths
            allparts.insert(0, parts[1])
            break
        else:
            path = parts[0]
            allparts.insert(0, parts[1])
    return allparts


def read_sim(filename):
    '''
    Read an ascii simulation file with wavelength (A) and F_lam as columns
    '''
    T = Table.read(filename, format='ascii')
    spectrum = np.interp(def_wave, T['col1'], T['col2'])
    return spectrum

def convert_slot_to_coords(ifuslots):
    '''
    Convert ifuslot integer to x, y focal plane coordinates [row, column]
    '''
    x, y = ([], [])
    for i in ifuslots:
        s = str(i)
        if len(s) == 3:
            x.append(int(s[:2]))
        else:
            x.append(int(s[0]))
        y.append(int(s[-1]))
    return np.array(x), np.array(y)


def get_slot_neighbors(ifuslot, u_ifuslots, dist=1):
    '''
    Get ifuslots within square distance, dist, from target ifuslot.
    
    A dist 1 will return a square of 3 x 3 ifuslots (excluding center ifuslot),
    for all ifuslots within that 3 x 3 box that are populated.  A dist = 2
    will return an equivalent 5 x 5 box.
    '''
    sel = np.zeros(u_ifuslots.shape, dtype=bool)
    xi, yi = convert_slot_to_coords([ifuslot])
    x, y = convert_slot_to_coords(u_ifuslots)
    sel = (np.abs(xi - x) <= dist) * (np.abs(yi - y) <= dist)
    unsel = (xi == x) * (yi == y)
    return u_ifuslots[sel*(~unsel)]


def build_path(rootdir, date, obs, ifuslot, amp, base='sci', exp='exp*',
               instrument='virus'):
    '''
    Build path for a given ifuslot, amplifier, observation, date, and rootdir
    '''
    if obs != '*':
        obs = '%07d' % obs
    path = op.join(rootdir, date, instrument, '%s%s' % (instrument, obs),
                   exp, instrument, '2*%s%s*%s.fits' % (ifuslot, amp, base))
    return path
    

def get_ifuslots():
    file_glob = build_path(args.rootdir, args.date, args.observation,
                           '*', 'LL', exp='exp01')
        
    filenames = sorted(glob.glob(file_glob))
    
    tfile = None
    if len(filenames) == 0:
        # do we have a tar?
        path = splitall(file_glob)
        tfile = op.join(*path[:-3]) + ".tar"
        if op.exists(tfile):
            print("This observation is tarred. Trying workaround...")
            fnames_glob = op.join(*path[-4:])
            with tarfile.open(tfile) as tf:
                filenames = fnmatch.filter(tf.getnames(), fnames_glob)
        else:
            print("Could not find %s" % tfile)
            print("Crap! How did I get here?")
            sys.exit(1)
    ifuslots = [int(op.basename(fn).split('_')[1][:3]) for fn in filenames]
    return np.unique(ifuslots)

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


def base_reduction(filename, get_header=False, tfile=None):
    '''
    Reduce filename from tarfile or fits file.
    
    Reduction steps include:
        1) Overscan subtraction
        2) Trim image
        3) Orientation
        4) Gain Multiplication
        5) Error propagation
    '''
    # Load fits file
    if tfile is not None:
        t = tarfile.open(tfile,'r')
        a = fits.open(t.extractfile(filename))
    else:
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
    '''
    Make a master flat image from the twilight frames
    '''
    files = glob.glob(twi_path)
    listtwi = []
    for filename in files:
        a, e = base_reduction(filename)
        a[:] -= masterbias
        listtwi.append(a)
    twi_array = np.array(listtwi, dtype=float)
    norm = np.median(twi_array, axis=(1, 2))[:, np.newaxis, np.newaxis]
    return np.median(twi_array / norm, axis=0) * np.median(norm)


def get_cal_path(pathname, date):
    '''
    Get appropriate cals on the closest night back 60 days in time
    '''
    datec = date
    daten = date
    cnt = 0
    pathnamen = pathname[:]
    while len(glob.glob(pathnamen)) == 0:
        datec_ = datetime(int(daten[:4]), int(daten[4:6]), int(daten[6:]))
        daten_ = datec_ - timedelta(days=1)
        daten = '%04d%02d%02d' % (daten_.year, daten_.month, daten_.day)
        pathnamen = pathname.replace(datec, daten)
        cnt += 1
        log.info(pathnamen)
        if cnt > 60:
            log.error('Could not find cal within 60 days.')
            break
    return pathnamen, daten


def get_script_path():
    '''
    Get script path 
    '''
    return op.dirname(op.realpath(sys.argv[0]))


def get_spectra(array_sci, array_flt, array_trace, wave, def_wave):
    '''
    Extract spectra by dividing the flat field and averaging the central
    two pixels
    '''
    sci_spectrum = np.zeros((array_trace.shape[0], def_wave.shape[0]))
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
        sw = (array_sci[indl, x] +#/ array_flt[indl, x] +
              array_sci[indh, x] )#/ array_flt[indh, x])
        sci_spectrum[fiber] = np.interp(def_wave, wave[fiber], sw / dw,
                                        left=0.0, right=0.0)
    twi_spectrum[~np.isfinite(twi_spectrum)] = 0.0
    sci_spectrum[~np.isfinite(sci_spectrum)] = 0.0
    return twi_spectrum, sci_spectrum / 2.


def reduce_ifuslot(ifuloop, h5table):
    p, t, s = ([], [], [])
    mult_fac = 6.626e-27 * (3e18 / def_wave) / 360. / 5e5 / 0.25
    for ind in ifuloop:
        ifuslot = '%03d' % h5table[ind]['ifuslot']
        amp = h5table[ind]['amp'].astype('U13')
        amppos = h5table[ind]['ifupos']
        wave = h5table[ind]['wavelength']
        trace = h5table[ind]['trace']
        try:
            masterbias = h5table[ind]['masterbias']
            
        except:
            masterbias = 0.0
        try:
            amp2amp = h5table[ind]['Amp2Amp']
            throughput = h5table[ind]['Throughput']
        except:
            amp2amp = np.ones((112, 1036))
            throughput = np.ones((112, 1036))
        twibase = build_path(args.rootdir, args.date, '*', ifuslot, amp,
                             base='twi')
        twibase, newdate = get_cal_path(twibase, args.date)
    
        if newdate != args.date:
            log.info('Found twi files on %s and using them for %s' %
                     (newdate, args.date))
        log.info('Making mastertwi for %s%s' % (ifuslot, amp))
        masterflt = get_mastertwi(twibase, masterbias)
        log.info('Done making mastertwi for %s%s' % (ifuslot, amp))

        file_glob = build_path(args.rootdir, args.date, args.observation,
                               ifuslot, amp)
        
        filenames = sorted(glob.glob(file_glob))
        
        tfile = None
        if len(filenames) == 0:
            # do we have a tar?
            path = splitall(file_glob)
            tfile = op.join(*path[:-3]) + ".tar"
            if op.exists(tfile):
                print("This observation is tarred. Trying workaround...")
                fnames_glob = op.join(*path[-4:])
                with tarfile.open(tfile) as tf:
                    filenames = fnmatch.filter(tf.getnames(), fnames_glob)
            else:
                print("Couldn't find observation: expanding {} was empty, and"
                      "{} doesn't exist.".format(file_glob,tfile))
                sys.exit(-1)

        for j, fn in enumerate(filenames):
            sciimage, scierror = base_reduction(fn, tfile=tfile)
            sciimage[:] = sciimage - masterbias
            twi, spec = get_spectra(sciimage, masterflt, trace, wave, def_wave)
            twi[:] = safe_division(twi, amp2amp*throughput)
            spec[:] = safe_division(spec, amp2amp*throughput) * mult_fac
            
            pos = amppos + dither_pattern[j]
            for x, i in zip([p, t, s], [pos, twi, spec]):
                x.append(i * 1.)        
    
    p, t, s = [np.vstack(j) for j in [p, t, s]]
    
    return p, t, s, fn , tfile
            

def make_fits(image, fn, name, ifuslot, tfile=None, ra=None,
              dec=None, pa=None):
    '''
    Outputing collapsed image
    '''
    if tfile is not None:
        import tarfile
        t = tarfile.open(tfile,'r')
        a = fits.open(t.extractfile(fn))
    else:
        a = fits.open(fn)

    PA = a[0].header['PARANGLE']
    imscale = 48. / image.shape[0]
    crx = image.shape[1] / 2.
    cry = image.shape[0] / 2.
    ifuslot = '%03d' % ifuslot
    if (ra is None) or (dec is None) or (pa is None):
        log.info('Using header for RA and Dec')
        RA = a[0].header['TRAJCRA'] * 15.
        DEC = a[0].header['TRAJCDEC']
        A = Astrometry(RA, DEC, PA, 0., 0., fplane_file=args.fplane_file)
        A.get_ifuslot_projection(ifuslot, imscale, crx, cry)
        wcs = A.tp_ifuslot
    else:
        A = Astrometry(ra, dec, pa, 0., 0.,
                       fplane_file=args.fplane_file)
        wcs = A.setup_TP(args.ra, args.dec, A.rot, crx, 
                         cry, x_scale=-imscale, y_scale=imscale)
    header = wcs.to_header()
    F = fits.PrimaryHDU(np.array(image, 'float32'), header=header)
    return F


def make_frame(xloc, yloc, data, Dx, Dy, ftf,
               scale=0.75, seeing_fac=2.3, radius=1.5):
    seeing = seeing_fac / scale
    a, b = data.shape
    x = np.arange(-23.-scale,
                  25.+1*scale, scale)
    y = np.arange(-23.-scale,
                  25.+1*scale, scale)
    xgrid, ygrid = np.meshgrid(x, y)
    zgrid = np.zeros((b,)+xgrid.shape)
    area = np.pi * 0.75**2
    G = Gaussian2DKernel(seeing / 2.35)
    S = np.zeros((data.shape[0], 2))
    D = np.sqrt((xloc - xloc[:, np.newaxis])**2 +
                (yloc - yloc[:, np.newaxis])**2)
    W = np.zeros(D.shape, dtype=bool)
    W[D < radius] = True
    Gp = Gaussian1DKernel(5.)
    c = data * 0.
    for i in np.arange(data.shape[0]):
        c[i] = interpolate_replace_nans(data[i], Gp)
    data = c * 1.
    for k in np.arange(b):
        S[:, 0] = xloc - Dx[k]
        S[:, 1] = yloc - Dy[k]
        sel = (data[:, k] / np.nansum(data[:, k] * W, axis=1)) <= 0.5
        sel *= np.isfinite(data[:, k]) * (ftf > 0.5)
        if np.any(sel):
            grid_z = griddata(S[sel], data[sel, k],
                              (xgrid, ygrid), method='cubic')
            zgrid[k, :, :] = (convolve(grid_z, G, boundary='extend') *
                              scale**2 / area)
    return zgrid[:, 1:-1, 1:-1], xgrid[1:-1, 1:-1], ygrid[1:-1, 1:-1]


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

def safe_division(num, denom, eps=1e-8, fillval=0.0):
    good = np.isfinite(denom) * (np.abs(denom) > eps)
    div = num * 0.
    if num.ndim == denom.ndim:
        div[good] = num[good] / denom[good]
        div[~good] = fillval
    else:
        div[:, good] = num[:, good] / denom[good]
        div[:, ~good] = fillval
    return div

def simulate_source(simulated_spectrum, pos, spectra, xc, yc,
                    seeing=1.5):
    E = Extract()
    boxsize = 10.5
    scale = 0.25
    psf = E.moffat_psf(seeing, boxsize, scale, alpha=3.5)
    weights = E.build_weights(xc, yc, pos[:, 0], pos[:, 1], psf)
    s_spec = weights * simulated_spectrum[np.newaxis, :]
    return spectra + s_spec

def correct_amplifier_offsets(data, fibers_in_amp=112, order=1):
    y = np.nanmedian(data[:, 200:-200], axis=1)
    x = np.arange(fibers_in_amp)
    model = []
    for i in np.arange(0, len(y), fibers_in_amp):
        yi = y[i:i+fibers_in_amp]
        try:
            mask = sigma_clip(yi, masked=True, maxiters=None)
        except:
            mask = sigma_clip(yi, iters=None) 
        skysel = ~mask.mask
        model.append(np.polyval(np.polyfit(x[skysel], yi[skysel], order), x))
    model = np.hstack(model)
    avg = np.mean(model)
    return model / avg

def estimate_sky(data):
    y = np.nanmedian(data[:, 200:-200], axis=1)
    try:
        mask = sigma_clip(y, masked=True, maxiters=None)
    except:
        mask = sigma_clip(y, iters=None)
    log.info('Number of masked fibers is %i / %i' % (mask.mask.sum(), len(y)))
    skyfibers = ~mask.mask
    init_sky = np.nanmedian(data[skyfibers], axis=0)
    return init_sky

def make_photometric_image(x, y, data, filtg, good_fibers, Dx, Dy,
                           nchunks=25,
                           ran=[-23, 25, -23, 25], scale=0.75):
    '''
    Make collapsed frame (uJy) for filtg
    '''    
    N1 = int((ran[1] - ran[0]) / scale) + 1
    N2 = int((ran[1] - ran[0]) / scale) + 1
    grid_x, grid_y = np.meshgrid(np.linspace(ran[0], ran[1], N1),
                                 np.linspace(ran[0], ran[3], N2))
    chunks = []
    weights = np.array([np.mean(f) for f in np.array_split(filtg, nchunks)])
    weights /= weights.sum()
    data = data * 1e29 * def_wave**2 / 3e18
    for c in np.array_split(data, nchunks, axis=1):
        chunks.append(np.nanmedian(c, axis=1))
    cDx = [np.mean(dx) for dx in np.array_split(Dx, nchunks)]
    cDy = [np.mean(dy) for dy in np.array_split(Dx, nchunks)]
    S = np.zeros((len(x), 2))
    images = []
    for k in np.arange(nchunks):
        S[:, 0] = x - cDx[k]
        S[:, 1] = y - cDy[k]
        sel = good_fibers * np.isfinite(chunks[k])
        if sel.sum():
            image = chunks[k][sel]
            grid_z0 = griddata(S[sel], image, (grid_x, grid_y),
                               method='linear')
            grid_z0 *= np.nansum(chunks[k][sel]) / np.nansum(grid_z0)
        else:
            grid_z0 = 0. * grid_x
        images.append(grid_z0)
    images = np.array(images)
    image = np.sum(images * weights[:, np.newaxis, np.newaxis], axis=0) 
    return image

# GET DIRECTORY NAME FOR PATH BUILDING
DIRNAME = get_script_path()
instrument = 'virus'
log = setup_logging()
if args.hdf5file is None:
    log.error('Please specify an hdf5file.  A default is not yet setup.')
    sys.exit(1)

# OPEN HDF5 FILE
h5file = open_file(args.hdf5file, mode='r')
h5table = h5file.root.Cals

# Get photometric filter
T = Table.read(op.join(DIRNAME, 'filters/ps1g.dat'), format='ascii')
filtg = np.interp(def_wave, T['col1'], T['col2'], left=0.0, right=0.0)
filtg /= filtg.sum()
 
# Collect indices for ifuslots (target and neighbors)
ifuslots = h5table.cols.ifuslot[:]
u_ifuslots = np.unique(ifuslots)
sel1 = list(np.where(args.ifuslot == ifuslots)[0])
ifuslotn = get_slot_neighbors(args.ifuslot, u_ifuslots,
                              dist=args.neighbor_dist)
ifuslotn = np.setdiff1d(ifuslotn, badifuslots)
IFUs = get_ifuslots()
allifus = [args.ifuslot]
for ifuslot in ifuslotn:
    if ifuslot not in IFUs:
        continue
    sel1.append(np.where(ifuslot == ifuslots)[0])
    allifus.append(ifuslot)
ifuloop = np.array(np.hstack(sel1), dtype=int)
nslots = len(ifuloop) / 4
allifus = np.hstack(allifus)


# Reducing IFUSLOT
log.info('Reducing ifuslot: %03d' % args.ifuslot)
pos, twispectra, scispectra, fn, tfile = reduce_ifuslot(ifuloop, h5table)
t = twispectra * 1.
t[t< 1e-8] = np.nan
g = t / np.nanmean(t, axis=0)[np.newaxis, :]
a = np.array([np.nanmedian(f, axis=1) for f in np.array_split(g, 25, axis=1)]).swapaxes(0, 1)
x = np.array([np.mean(xi) for xi in np.array_split(np.arange(g.shape[1]), 25)])
ftf = np.zeros(scispectra.shape)
for i, ai in enumerate(a):
    sel = np.isfinite(ai)
    if np.sum(sel)>1:
        I = interp1d(x[sel], ai[sel], kind='quadratic',
                     fill_value='extrapolate')
        ftf[i] = I(np.arange(g.shape[1]))
    else:
        ftf[i] = 0.0
scispectra = safe_division(scispectra, ftf)

# Subtracting Sky
log.info('Subtracting sky for all ifuslots')
nexp = scispectra.shape[0] / 448 / nslots
log.info('Number of exposures: %i' % nexp)
N, D = scispectra.shape
reorg = np.zeros((nexp, N/nexp, D))
for i in np.arange(nexp):
    x = np.arange(112)
    X = []
    for j in np.arange(4*nslots):
        X.append(x + j * 336 + i * 112)
    X = np.array(np.hstack(X), dtype=int)
    reorg[i] = scispectra[X]*1.
reorg[reorg < 1e-42] = np.nan
reorg = np.array([r / correct_amplifier_offsets(r)[:, np.newaxis]
                  for r in reorg])
skies = np.array([estimate_sky(r) for r in reorg])
fits.PrimaryHDU(skies).writeto('test.fits', overwrite=True)
exp_ratio = np.ones((nexp,))
for i in np.arange(1, nexp):
    exp_ratio[i] = np.nanmedian(skies[i] / skies[0])
    log.info('Ratio for exposure %i to exposure 1: %0.2f' %
             (i+1, exp_ratio[i]))
reorg[:] -= skies[:, np.newaxis, :]
reorg[:] *= exp_ratio[:, np.newaxis, np.newaxis]
cnt = 0
for j in np.arange(4*nslots): 
    for i in np.arange(nexp):
        l = j*112
        u = (j+1)*112
        scispectra[cnt:(cnt+112)] = reorg[i, l:u]*1.
        cnt += 112

daofind = DAOStarFinder(fwhm=3.0, threshold=0.1, exclude_border=True) 
for i, ui in enumerate(allifus):
    log.info('Making collapsed frame for %03d' % ui)
    N = 448 * nexp
    data = scispectra[N*i:(i+1)*N]
    F = np.nanmedian(ftf[N*i:(i+1)*N], axis=1)
    P = pos[N*i:(i+1)*N]
    name = '%s_%07d_%03d.fits' % (args.date, args.observation, ui)
    image = make_photometric_image(P[:, 0], P[:, 1], data, filtg, F > 0.5,
                                   ADRx, 0.*ADRx, nchunks=11,
                                   ran=[-23, 25, -23, 25],  scale=0.75)
    F = make_fits(image, fn, name, ui, tfile)
    mean, median, std = sigma_clipped_stats(image, sigma=3.0)
    daofind.threshold = 10. * std
    sources = daofind(image)
    log.info('Found %i sources' % len(sources))
    positions = (sources['xcentroid'], sources['ycentroid'])
    apertures = CircularAperture(positions, r=5.)
    phot_table = aperture_photometry(image, apertures)
    gmags = -2.5 * np.log10(phot_table['aperture_sum']) + 23.9
    Sources = np.zeros((len(sources), 3))
    Sources[:, 0], Sources[:, 1] = (sources['xcentroid'], sources['ycentroid'])
    Sources[:, 2] = gmags
    print(Sources)
    F.writeto(name, overwrite=True)

sys.exit(1)

if args.simulate:
    log.info('Simulating spectrum from %s' % args.source_file)
    simulated_spectrum = read_sim(args.source_file)
    scispectra = simulate_source(simulated_spectrum, P, data, 
                                 args.source_x, args.source_y, 
                                 args.source_seeing)

if args.simulate:
    name = ('%s_%07d_%03d_sim.fits' %
                (args.date, args.observation, args.ifuslot))
    cubename = ('%s_%07d_%03d_cube_sim.fits' %
                (args.date, args.observation, args.ifuslot))
else:
    name = ('%s_%07d_%03d.fits' %
                (args.date, args.observation, args.ifuslot))
    cubename = ('%s_%07d_%03d_cube.fits' %
                (args.date, args.observation, args.ifuslot))



# Making data cube
log.info('Making Cube')
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    zgrid, xgrid, ygrid = make_frame(pos[:, 0], pos[:, 1], scispectra, 
                                     ADRx, 0. * def_wave, ftf)

if tfile is not None:
    t = tarfile.open(tfile, 'r')
    a = fits.open(t.extractfile(fn))
    t.close()
else:
    a = fits.open(fn)

he = a[0].header

write_cube(def_wave, xgrid, ygrid, zgrid, cubename, he)


h5file.close()
    
        
