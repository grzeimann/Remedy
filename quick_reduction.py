# -*- coding: utf-8 -*-
"""
Created on Tue Jan 22 15:11:39 2019
 
@author: gregz
"""

import matplotlib
matplotlib.use('agg')
import argparse as ap
import astropy.units as units
import fnmatch
import glob
import matplotlib.pyplot as plt
import multiprocessing
import numpy as np
import os.path as op
import seaborn as sns
import sys
import tarfile
import warnings


from astrometry import Astrometry
from astropy.coordinates import SkyCoord, match_coordinates_sky
from astropy.convolution import convolve, Gaussian2DKernel
from astropy.convolution import interpolate_replace_nans, Gaussian1DKernel
from astropy.io import fits
from astropy.modeling.models import Polynomial2D
from astropy.modeling.fitting import LevMarLSQFitter
from astropy.stats import biweight_location, sigma_clip
from astropy.stats import sigma_clipped_stats, mad_std
from astropy.table import Table
from catalog_search import query_panstarrs, MakeRegionFile
from datetime import datetime, timedelta
from extract import Extract
from input_utils import setup_logging
from multiprocessing import Pool
from photutils import DAOStarFinder, aperture_photometry, CircularAperture
from scipy.interpolate import griddata, interp1d
from tables import open_file, IsDescription, Float32Col


# Plot Style
sns.set_context('talk')
sns.set_style('whitegrid')

# Turn off annoying warnings (even though some deserve attention)
warnings.filterwarnings("ignore")

###############################################################################
# CONFIGURATION AND GLOBAL VARIABLES 

# Amps
amps = ['LL', 'LU', 'RL', 'RU']

# Ignore these ifuslots because they are subpar or look poor
badifuslots = np.array([67, 84, 83, 96, 104, 86, 42, 91, 88, 86, 95, 75])

# Default dither pattern for 3 exposures
dither_pattern = np.array([[0., 0.], [1.27, -0.73], [1.27, 0.73]])

# Rectified wavelength
def_wave = np.arange(3470., 5542., 2.)

# ADR model
wADR = [3500., 4000., 4500., 5000., 5500.]
ADRx = [-0.74, -0.4, -0.08, 0.08, 0.20]
ADRx = np.polyval(np.polyfit(wADR, ADRx, 3), def_wave)

###############################################################################
# PARSED INPUT PARAMETERS FROM THE COMMAND LINE

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

parser.add_argument("-p", "--parallelize",
                    help='''Parallelize some of the processing''',
                    action="count", default=0)

args = parser.parse_args(args=None)


###############################################################################
# FUNCTIONS FOR THE MAIN BODY BELOW                                                       

def splitall(path):
    ''' 
    Split path into list 
    
    Parameters
    ----------
    path : str
        file path
    
    Returns
    -------
    allparts : str
        list of constituent parts of the path
    '''
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


def get_ra_dec_from_header(tfile, fn):
    ''' 
    Grab RA, Dec, and PA from header
    The fits header may be in a tar file, "tfile"
    
    Parameters
    ----------
    tfile : str
        Tarfile name
    fn : str
        fits filename containing the header in the primary HDU
    
    Returns
    -------
    ra : float
        Right Ascension
    dec : float
        Declination
    pa : float
        Parangle
    '''
    if tfile is not None:
        t = tarfile.open(tfile,'r')
        a = fits.open(t.extractfile(fn))
    else:
        a = fits.open(fn)
    ra = a[0].header['TRAJCRA'] * 15.
    dec = a[0].header['TRAJCDEC'] * 1.
    pa = a[0].header['PARANGLE'] * 1.
    return ra, dec, pa


def read_sim(filename):
    '''
    Read an ascii simulation file with wavelength (A) and F_lam as columns
    The columns should not have headers, just an ascii file with wavelength
    and F_lam (ergs/s/cm^2/A)
    
    Parameters
    ----------
    filename : str
        filename of the simulation data
    def_wave : 1d numpy array [GLOBAL]
        global variable for the rectified wavelength array
    
    Returns
    -------
    spectrum : 1d numpy array
        simulated spectrum interpolated to the rectified wavelength
    '''
    T = Table.read(filename, format='ascii')
    spectrum = np.interp(def_wave, T['col1'], T['col2'])
    return spectrum

def convert_slot_to_coords(ifuslots):
    '''
    Convert ifuslot integer to x, y focal plane coordinates [row, column]
    
    Parameters
    ----------
    ifuslots : list
        a list of integers, e.g. 47
    
    Returns
    -------
    x : 1d numpy array
        Row in focal plane: 47 --> 4
    y : 1d numpy array
        Column in focal plane: 47 --> 7
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
    
    A dist = 1 will return a square of 3 x 3 ifuslots (excluding central ifu).
    A dist = 2 will return an equivalent 5 x 5 box.
    
    If an ifu does not occupy an ifuslot, then that slot is not returned.
    
    Parameters
    ----------
    ifuslot : integer
        Target IFU slot
    u_ifuslots : 1d numpy array [type=int]
        iFU slots in the observation
    
    Returns
    -------
    u_ifuslots : 1d numpy array [type=int]
        neighbor ifuslots as explained above
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
    
    Parameters
    ----------
    rootdir : str
        root directory where raw data is stored
    date : str
        date for path
    obs : integer
        observation for path
    ifuslot : str
        IFU slot for path
    amp : str
        Amplifier for path
    base : str
        Type of exposure
    exp : str
        Exposure number for path
    instrument : str
        Instrument for path
    
    Returns
    -------
    path : str
        File path constructed for the given date, obs, ect.
    '''
    if obs != '*':
        obs = '%07d' % obs
    path = op.join(rootdir, date, instrument, '%s%s' % (instrument, obs),
                   exp, instrument, '2*%s%s*%s.fits' % (ifuslot, amp, base))
    return path
    

def get_ifuslots():
    '''
    Get ifuslots for the rootdir, date, and observation in the global variable
    args.
    
    Parameters
    ----------
    args : object [GLOBAL]
        Object containing the command line inputs from the python call
    
    Returns
    -------
    ifuslots : 1d numpy array [type=int]
        ifuslots for the input date, observation
    '''
        
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
    
    Parameters
    ----------
    image : 2d numpy array
        fits image
    amp : str
        Amplifier for the fits image
    ampname : str
        Amplifier name is the location of the amplifier
    
    Returns
    -------
    image : 2d numpy array
        Oriented fits image correcting for what amplifier it comes from
        These flips are unique to the VIRUS/LRS2 amplifiers
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
    
    Parameters
    ----------
    filename : str
        Filename of the fits file
    get_header : boolean
        Flag to get and return the header
    tfile : str
        Tar filename if the fits file is in a tarred file
    
    Returns
    -------
    a : 2d numpy array
        Reduced fits image, see steps above
    e : 2d numpy array
        Associated error frame
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


def get_mastertwi(files, masterbias, twitarfile):
    '''
    Make a master flat image from the twilight frames
    
    Parameters
    ----------
    files : list
        list of fits file names
    masterbias : 2d numpy array
        masterbias in e-
    twitarfile : str or None
        Name of the tar file if the fits file is tarred
    
    Returns
    -------
    mastertwi : 2d numy array
        median stacked twilight frame
    '''
    listtwi = []
    for filename in files:
        a, e = base_reduction(filename, tfile=twitarfile)
        a[:] -= masterbias
        listtwi.append(a)
    twi_array = np.array(listtwi, dtype=float)
    norm = np.median(twi_array, axis=(1, 2))[:, np.newaxis, np.newaxis]
    return np.median(twi_array / norm, axis=0) * np.median(norm)

def get_twi_tarfile(pathname, date):
    '''
    Go through many dates if necessary and find the tar file that contains
    twilight exposures.
    
    Parameters
    ----------
    pathname : str
        File path to look for the tar file containing twilight exposures
    date : str
        Date for initial search for twilight exposures
    
    Returns
    -------
    twitarfile : str
        Name of the tarred file containing twilight exposures
    '''
    datec = date
    datec_ = datetime(int(date[:4]), int(date[4:6]), int(date[6:]))
    daten_ = datec_ + timedelta(days=1)
    daten = '%04d%02d%02d' % (daten_.year, daten_.month, daten_.day)
    pathnamen = pathname[:]
    twitarfile = None
    while twitarfile is None:
        datec_ = datetime(int(daten[:4]), int(daten[4:6]), int(daten[6:]))
        daten_ = datec_ - timedelta(days=1)
        daten = '%04d%02d%02d' % (daten_.year, daten_.month, daten_.day)
        pathnamen = pathname.replace(datec, daten)
        tarfolders = glob.glob(pathnamen)
        for tarfolder in tarfolders:
            T = tarfile.open(tarfolder, 'r')
            flag = True
            while flag:
                a = T.next()
                try:
                    name = a.name
                except:
                    flag = False
                if name[-5:] == '.fits':
                    if name[-8:-5] == 'twi':
                        twitarfile = tarfolder
                    flag = False
            if twitarfile is not None:
                break
    return twitarfile

def roll_through_dates(pathname, date):
    '''
    Get appropriate cals on the closest night back 60 days in time
    
    Parameters
    ----------
    pathname : str
        File path to look for the tar file containing twilight exposures
    date : str
        Date for initial search for twilight exposures
    
    Returns
    -------
    twinames : list
        list of twilight fits file names
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
    return glob.glob(pathnamen)


def get_script_path():
    '''
    Get script path, aka, where does Remedy live?
    '''
    return op.dirname(op.realpath(sys.argv[0]))


def get_spectra(array_sci, array_err, array_flt, array_trace, wave, def_wave):
    '''
    Extract spectra by dividing the flat field and averaging the central
    two pixels
    
    Parameters
    ----------
    array_sci : 2d numpy array
        science image
    array_err : 2d numpy array
        error image
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
    sci_spectrum : 2d numpy array
        rectified science spectrum for each fiber
    err_spectrum : 2d numpy array
        rectified error spectrum for each fiber   
    '''
    sci_spectrum = np.zeros((array_trace.shape[0], def_wave.shape[0]))
    err_spectrum = np.zeros((array_trace.shape[0], def_wave.shape[0]))
    twi_spectrum = np.zeros((array_trace.shape[0], def_wave.shape[0]))
    N = array_flt.shape[0]
    x = np.arange(array_flt.shape[1])
    for fiber in np.arange(array_trace.shape[0]):
        dw = np.diff(wave[fiber])
        dw = np.hstack([dw[0], dw])
        if array_trace[fiber].min() < 1.:
            continue
        if np.ceil(array_trace[fiber]).max() >= (N-1):
            continue
        indl = np.floor(array_trace[fiber]).astype(int)
        indh = np.ceil(array_trace[fiber]).astype(int)
        tw = array_flt[indl, x] / 2. + array_flt[indh, x] / 2.
        twi_spectrum[fiber] = np.interp(def_wave, wave[fiber], tw / dw,
                                        left=0.0, right=0.0)
        sw = (array_sci[indl, x] + array_sci[indh, x]) / 2.
        ew = np.sqrt((array_err[indl, x]**2 + array_err[indh, x]**2) / 2.)
        sci_spectrum[fiber] = np.interp(def_wave, wave[fiber], sw / dw,
                                        left=0.0, right=0.0)
        # Not real propagation of error, but skipping math for now
        err_spectrum[fiber] = np.interp(def_wave, wave[fiber], ew / dw,
                                        left=0.0, right=0.0)
    twi_spectrum[~np.isfinite(twi_spectrum)] = 0.0
    err_spectrum[~np.isfinite(err_spectrum)] = 0.0
    sci_spectrum[~np.isfinite(sci_spectrum)] = 0.0
    return twi_spectrum, sci_spectrum, err_spectrum

def get_fiber_to_fiber(twispectra):
    '''
    Get Fiber to Fiber from twilight spectra
    
    Parameters
    ----------
    twispectra : 2d numpy array
        twilight spectra for each fiber
        
    Returns
    -------
    ftf : 2d numpy array
        fiber to fiber normalization
    '''
    t = twispectra * 1.
    t[t< 1e-8] = np.nan
    g = t / np.nanmedian(t, axis=0)[np.newaxis, :]
    
    
    # Fiber to fiber is an interpolation of median binned normalized twi spectra
    nbins = 51
    a = np.array([np.nanmedian(f, axis=1) for f in np.array_split(g, nbins, axis=1)]).swapaxes(0, 1)
    x = np.array([np.mean(xi) for xi in np.array_split(np.arange(g.shape[1]), nbins)])
    ftf = np.zeros(scispectra.shape)
    for i, ai in enumerate(a):
        sel = np.isfinite(ai)
        if np.sum(sel)>1:
            I = interp1d(x[sel], ai[sel], kind='quadratic',
                         fill_value='extrapolate')
            ftf[i] = I(np.arange(g.shape[1]))
        else:
            ftf[i] = 0.0
    T = safe_division(t, ftf)
    g = t / np.nanpercentile(T, 50, axis=0)[np.newaxis, :]
    return g

def get_sci_twi_files():
    '''
    Get the names of the fits files for the science frames of interest
    and the twilight frames of interest.  If the files are tarred,
    return the names of the tar files as well.
    
    Parameters
    ----------
    args : object [GLOBAL]
        Object containing the command line inputs from the python call
    
    Returns
    -------
    scinames : list
        names of the science fits files
    twinames : list
        names of the twilight fits files
    scitarfile : str or None
        name of the tar file containing the fits files if there is one
    twitarfile : str or None
        name of the tar file containing the fits files if there is one
    '''
    file_glob = build_path(args.rootdir, args.date, args.observation,
                           '047', 'LL')
    path = splitall(file_glob)
    scitarfile = op.join(*path[:-3]) + ".tar"
    if op.exists(scitarfile):
        with tarfile.open(scitarfile) as tf:
            scinames = sorted(tf.getnames())
    else:
        file_glob = build_path(args.rootdir, args.date, args.observation,
                           '*', '*')
        scinames = sorted(glob.glob(file_glob))
        scitarfile = None
    if scitarfile is None:
        pathname = build_path(args.rootdir, args.date, '*', '*', '*',
                             base='twi')
        twinames = sorted(roll_through_dates(pathname, args.date))
        twitarfile = None
    else:
        file_glob = build_path(args.rootdir, args.date, '*',
                           '047', 'LL')
        path = splitall(file_glob)
        tarname = op.join(*path[:-3]) + ".tar"
        twitarfile = get_twi_tarfile(tarname, args.date)
        with tarfile.open(twitarfile) as tf:
            twinames = sorted(tf.getnames())
    return scinames, twinames, scitarfile, twitarfile

def reduce_ifuslot(ifuloop, h5table):
    '''
    Parameters
    ----------
    ifuloop : list
        list of indices for ifuslots in the hdf5 table
    h5table: hdf5 table [tables object]
        the hdf5 calibration table with wavelength, trace, ect. for all ifus
    
    Returns
    -------
    p : 2d numpy array
        x and y positions for all fibers in ifuloop [ifu frame]
    t : 2d numpy array
        twilight spectrum for all fibers in ifuloop
    s : 2d numpy array
        science spectrum for all fibers in ifuloop
    fn : str
        last fits filename to get info from later
    scitarfile : str or None
        name of the tar file for the science frames if there is one
    '''
    p, t, s, e = ([], [], [], [])
    
    # Calibration factors to convert electrons to uJy
    mult_fac = 6.626e-27 * (3e18 / def_wave) / 360. / 5e5 / 0.25
    mult_fac *= 1e29 * def_wave**2 / 3e18
    # Check if tarred

    scinames, twinames, scitarfile, twitarfile = get_sci_twi_files()

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
        if np.all(amp2amp==0.):
            amp2amp = np.ones((112, 1036))
        if np.all(throughput==0.):
            T = Table.read(op.join(DIRNAME, 'CALS/throughput.txt'),
                           format='ascii.fixed_width_two_line')
            throughput = np.array([T['throughput']]*112)
        
        fnames_glob = '*/2*%s%s*%s.fits' % (ifuslot, amp, 'twi')
        twibase = fnmatch.filter(twinames, fnames_glob)
        log.info('Making mastertwi for %s%s' % (ifuslot, amp))
        masterflt = get_mastertwi(twibase, masterbias, twitarfile)
        log.info('Done making mastertwi for %s%s' % (ifuslot, amp))

        fnames_glob = '*/2*%s%s*%s.fits' % (ifuslot, amp, 'sci')
        filenames = fnmatch.filter(scinames, fnames_glob)
        for j, fn in enumerate(filenames):
            sciimage, scierror = base_reduction(fn, tfile=scitarfile)
            sciimage[:] = sciimage - masterbias
            twi, spec, espec = get_spectra(sciimage, scierror, masterflt,
                                           trace, wave, def_wave)
            twi[:] = safe_division(twi, amp2amp*throughput)
            spec[:] = safe_division(spec, amp2amp*throughput) * mult_fac
            espec[:] = safe_division(espec, amp2amp*throughput) * mult_fac
            pos = amppos + dither_pattern[j]
            for x, i in zip([p, t, s, e], [pos, twi, spec, espec]):
                x.append(i * 1.)        
    
    p, t, s, e = [np.vstack(j) for j in [p, t, s, e]]
    
    return p, t, s, e, fn , scitarfile


def make_cube(xloc, yloc, data, Dx, Dy, ftf, scale, ran,
              seeing_fac=2.3, radius=1.5):
    '''
    Make data cube for a given ifuslot
    
    Parameters
    ----------
    xloc : 1d numpy array
        fiber positions in the x-direction [ifu frame]
    yloc : 1d numpy array
        fiber positions in the y-direction [ifu frame]
    data : 2d numpy array
        fiber spectra [N fibers x M wavelength]
    Dx : 1d numpy array
        Atmospheric refraction in x-direction as a function of wavelength
    Dy : 1d numpy array
        Atmospheric refraction in y-direction as a function of wavelength
    ftf : 1d numpy array
        Average fiber to fiber normalization (1 value for each fiber)
    scale : float
        Spatial pixel scale in arcseconds
    seeing_fac : float
    
    Returns
    -------
    Dcube : 3d numpy array
        Data cube, corrected for ADR
    xgrid : 2d numpy array
        x-coordinates for data cube
    ygrid : 2d numpy array
        y-coordinates for data cube
    '''
    seeing = seeing_fac / scale
    a, b = data.shape
    N1 = int((ran[1] - ran[0]) / scale) + 1
    N2 = int((ran[3] - ran[2]) / scale) + 1
    xgrid, ygrid = np.meshgrid(np.linspace(ran[0], ran[1], N1),
                               np.linspace(ran[2], ran[3], N2))
    Dcube = np.zeros((b,)+xgrid.shape)
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
                              (xgrid, ygrid), method='linear')
            Dcube[k, :, :] = (convolve(grid_z, G, boundary='extend') *
                              scale**2 / area)
    return Dcube[:, 1:-1, 1:-1], xgrid[1:-1, 1:-1], ygrid[1:-1, 1:-1]


def write_cube(wave, xgrid, ygrid, Dcube, outname, he):
    '''
    Write data cube to fits file
    
    Parameters
    ----------
    wave : 1d numpy array
        Wavelength for data cube
    xgrid : 2d numpy array
        x-coordinates for data cube
    ygrid : 2d numpy array
        y-coordinates for data cube
    Dcube : 3d numpy array
        Data cube, corrected for ADR
    outname : str
        Name of the outputted fits file
    he : object
        hdu header object to carry original header information
    '''
    hdu = fits.PrimaryHDU(np.array(Dcube, dtype='float32'))
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
    '''
    Do a safe division in case denominator has zeros.  This is a personal
    favorite solution to this common problem.
    
    Parameters
    ----------
    num : numpy array
        Numerator in the division
    denom : numpy array
        Denominator in the division
    eps : float
        low value to check for zeros
    fillval : float
        value to fill in if the denominator is zero
    
    Returns
    -------
    div : numpy array
        The safe division product of num / denom
    '''
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
    '''
    Simulate a point source using the Extract class
    
    Start with a simulated spectrum.  Spread the flux over fibers using a
    Moffat PSF.  Then add to the original fiber spectra.
    
    Parameters
    ----------
    simulated_spectrum : 1d numpy array
        simulated spectrum at appropriate rectified wavelengths (def_wave)
    pos : 2d numpy array
        x and y positions of the fibers to place the simulated spectrum
    spectra : 2d numpy array
        all fiber spectra correspoding to the x and y positions in pos
    xc : float
        x centroid of the source [ifu frame]
    yc : float
        y centroid of the source [ifu frame]
    seeing : float
        seeing for the moffat psf profile to spread out the simulated spectrum
        
    Returns
    -------
    spectra : 2d numpy array
        original spectra + the new fiber spectra of the simulated source
    '''
    E = Extract()
    boxsize = 10.5
    scale = 0.25
    psf = E.moffat_psf(seeing, boxsize, scale, alpha=3.5)
    weights = E.build_weights(xc, yc, pos[:, 0], pos[:, 1], psf)
    s_spec = weights * simulated_spectrum[np.newaxis, :]
    return spectra + s_spec

def correct_amplifier_offsets(data, fibers_in_amp=112, order=1):
    '''
    Correct the offsets (in multiplication) between various amplifiers.
    These offsets originate from incorrect gain values in the amplifier 
    headers and/or mismatches in the amplifier to amplifier correction
    going from twilight to science exposures
    
    Parameters
    ----------
    data : 2d numpy array
        all spectra in a given exposure
    fiber_in_amp : int
        number of fibers in an amplifier
    order : int
        polynomial order of correction across amplifiers
    '''
    y = np.nanmedian(data[:, 200:-200], axis=1)
    x = np.arange(fibers_in_amp)
    model = []
    for i in np.arange(0, len(y), fibers_in_amp):
        yi = y[i:i+fibers_in_amp]
        try:
            mask = sigma_clip(yi, masked=True, maxiters=None, stdfunc=mad_std)
        except:
            mask = sigma_clip(yi, iters=None, stdfunc=mad_std) 
        skysel = ~mask.mask
        if skysel.sum() > 10:
            model.append(np.polyval(np.polyfit(x[skysel], yi[skysel], order),
                                    x))
        else:
            model.append(np.ones(yi.shape))
    model = np.hstack(model)
    avg = np.nanmedian(model)
    return model / avg

def estimate_sky(data):
    '''
    Model the sky for all fibers using a sigma clip to exlude fibers with
    sources in them, then take a median of the remaining fibers
    
    Parameters
    ----------
    data : 2d numpy array
        all spectra in a given exposure
    
    Returns
    -------
    init_sky : 1d numpy array
        estimate of the sky spectrum for all fibers
    '''
    y = np.nanmedian(data[:, 200:-200], axis=1)
    try:
        mask = sigma_clip(y, masked=True, maxiters=None, stdfunc=mad_std)
    except:
        mask = sigma_clip(y, iters=None, stdfunc=mad_std)
    log.info('Number of masked fibers is %i / %i' % (mask.mask.sum(), len(y)))
    skyfibers = ~mask.mask
    init_sky = np.nanmedian(data[skyfibers], axis=0)
    return init_sky

def make_photometric_image(x, y, data, filtg, good_fibers, Dx, Dy,
                           nchunks=25,
                           ran=[-23, 25, -23, 25], scale=0.75):
    '''
    Make collapsed frame (uJy) for g' filter.
    
    Due to ADR, we must chunk out the spectra in wavelength, make individual
    frames and then coadd them with the g' filter weight to make g-band image.
    
    Parameters
    ----------
    x : 1d numpy array
        fiber x positions
    y : 1d numpy array
        fiber y positions
    data : 2d numpy array
        fiber spectra
    filtg : 1d numpy array
        g' filter rectified to def_wave
    good_fibers : 1d numpy array
        fibers with significant throughput
    Dx : 1d numpy array
        Atmospheric refraction in x-direction as a function of wavelength
    Dy : 1d numpy array
        Atmospheric refraction in y-direction as a function of wavelength
    nchunks : int
        Number of wavelength chunks to make individual images for coaddition
    ran : list
        min(x), max(x), min(y), max(y) for image
    scale : float
        pixel scale of image
    
    Returns
    -------
    image : 2d numpy array
        g-band image from VIRUS spectra of a given IFU slot
    '''    
    N1 = int((ran[1] - ran[0]) / scale) + 1
    N2 = int((ran[3] - ran[2]) / scale) + 1
    grid_x, grid_y = np.meshgrid(np.linspace(ran[0], ran[1], N1),
                                 np.linspace(ran[2], ran[3], N2))
    chunks = []
    weights = np.array([np.mean(f) for f in np.array_split(filtg, nchunks)])
    weights /= weights.sum()
    for c in np.array_split(data, nchunks, axis=1):
        chunks.append(np.nanmedian(c, axis=1))
    cDx = [np.mean(dx) for dx in np.array_split(Dx, nchunks)]
    cDy = [np.mean(dy) for dy in np.array_split(Dx, nchunks)]
    S = np.zeros((len(x), 2))
    images = []
    G = Gaussian2DKernel(1.0)
    for k in np.arange(nchunks):
        S[:, 0] = x - cDx[k]
        S[:, 1] = y - cDy[k]
        sel = good_fibers * np.isfinite(chunks[k])
        if sel.sum():
            image = chunks[k][sel]
            grid_z0 = griddata(S[sel], image, (grid_x, grid_y),
                               method='linear')
            grid_z0 = convolve(grid_z0, G)
            grid_z0 *= np.nansum(chunks[k][sel]) / np.nansum(grid_z0)
        else:
            grid_z0 = 0. * grid_x
        images.append(grid_z0)
    images = np.array(images)
    image = np.sum(images * weights[:, np.newaxis, np.newaxis], axis=0) 
    return image

def match_to_archive(sources, image, A, ifuslot, scale, ran, coords,
                     apradius=3.5):
    '''
    Match list of sources in a given IFU slot to a set of coordinates from
    an existing survey, coords.  The sources comes from DAOStarFinder.  We
    will also perform aperture photometry on the sources, keep a series of
    information on the sources and matches, and return the information.
    
    Parameters
    ----------
    sources : DAOStarFinder table
        Output from running DAOStarFinder on the IFU slot g-band image
    image : 2d numpy array
        IFU slot g-band image
    A : Astrometry class
        astrometry object from the Astrometry Class
    ifuslot : int
        IFU slot
    scale : float
        pixel scale of the g-band image
    ran : list
        min(x), max(x), min(y), max(y) for image
    coords : Skycoord Object
        coordinates from an archival catalog (Pan-STARRS, likely)
    apradius : float
        radius of the aperture in pixels for aperture photometry
    
    Returns
    -------
    Sources : 2d numpy array
        columns include image x coordinate, image y coordinate, aperture g-band
        magnitude of IFU slot image, distance to closest catalog match,
        catalog g-band magnitude, catalog RA, catalog Dec, focal plane x-coord,
        focal plane y-coord, delta RA, delta Dec
    '''
    if len(sources) == 0:
        return None

    positions = (sources['xcentroid'], sources['ycentroid'])
    apertures = CircularAperture(positions, r=apradius)
    phot_table = aperture_photometry(image, apertures,
                                     mask=~np.isfinite(image))
    gmags = np.where(phot_table['aperture_sum'] > 0.,
                     -2.5 * np.log10(phot_table['aperture_sum']) + 23.9,
                     99.)
    Sources = np.zeros((len(sources), 11))
    Sources[:, 0], Sources[:, 1] = (sources['xcentroid'], sources['ycentroid'])
    Sources[:, 2] = gmags
    RA, Dec = A.get_ifupos_ra_dec('%03d' % ifuslot,
                                  Sources[:, 0]*scale + ran[0],
                                  Sources[:, 1]*scale + ran[2])
    ifu = A.fplane.by_ifuslot('%03d' % ifuslot)
    ifux, ifuy = (ifu.y, ifu.x) 

    # Find closest bright source, use that for offset, then match
    # Make image class that combines, does astrometry, detections, updates coords
    C = SkyCoord(RA*units.deg, Dec*units.deg, frame='fk5')
    idx, d2d, d3d = match_coordinates_sky(C, coords)
    Sources[:, 3] = d2d.arcsecond
    Sources[:, 4] = gC[idx]
    Sources[:, 5] = coords[idx].ra.deg
    Sources[:, 6] = coords[idx].dec.deg
    Sources[:, 7], Sources[:, 8] = (sources['xcentroid']*scale + ran[0] + ifux,
                                    sources['ycentroid']*scale + ran[2] + ifuy)
    Sources[:, 9], Sources[:, 10] = (RA-Sources[:,5], Dec-Sources[:, 6])
    return Sources

def fit_astrometry(f, A1, thresh=7.):
    '''
    Fit astrometry using a linear fit to the conversion of focal plane x and y
    to RA as well as a linear fit to the conversion of focal plane x and y to
    Dec.  This gets a new RA and Dec of the 0, 0 coordinate in focal plane
    units.  Then solve for the adjust rotation angle from the parallactic 
    angle.  Iterate one last time for the RA and Dec of 0, 0.  Return the
    new astometry class assuming the solution is reasonable.
    
    Parameters
    ----------
    f : astropy Table
        columns include image x coordinate, image y coordinate, aperture g-band
        magnitude of IFU slot image, distance to closest catalog match,
        catalog g-band magnitude, catalog RA, catalog Dec, focal plane x-coord,
        focal plane y-coord, delta RA, delta Dec 
    A1 : Astrometry Class
        Astrometry object 
    thresh : float
        Matching threshold for considering a match to be real (")
    
    Returns
    -------
    A1 : Astrometry Class
        Astrometry object 
    '''
    A = A1
    P = Polynomial2D(1)
    fitter = LevMarLSQFitter()
    sel = f['dist'] < thresh
    log.info('Number of sources with 7": %i' % sel.sum())
    fitr = fitter(P, f['fx'][sel], f['fy'][sel], f['RA'][sel])
    fitd = fitter(P, f['fx'][sel], f['fy'][sel], f['Dec'][sel])
    ra0 = A.ra0 * 1.
    dec0 = A.dec0 * 1.
    RA0 = fitr(0., 0.)
    Dec0 = fitd(0., 0.)
    dR = np.cos(np.deg2rad(Dec0)) * 3600. * (ra0 - RA0)
    dD = 3600. * (dec0 - Dec0)
    log.info('%s_%07d initial offsets: %0.2f, %0.2f' %(args.date, 
                                                       args.observation, dR,
                                                       dD))
    dr = np.cos(np.deg2rad(f['Dec'][sel])) * -3600. * (f['RA'][sel] - RA0)
    dd = 3600. * (f['Dec'][sel] - Dec0)
    a1 = np.arctan2(dd, dr)
    a2 = np.arctan2(f['fy'][sel], f['fx'][sel])
    da = a1 - a2
    sel1 = np.abs(da) > np.pi
    da[sel1] -= np.sign(da[sel1]) * 2. * np.pi
    rot_i = A.rot * 1.
    rot = np.rad2deg(np.median(da))
    rot_error = mad_std(np.rad2deg(da))
    for i in [rot, 360. - rot, -rot]:
        if np.abs(A.rot - i) < 1.5:
            rot = i
    A.rot = rot * 1.
    A.tp = A.setup_TP(RA0, Dec0, A.rot, A.x0,  A.y0)
    mRA, mDec = A.tp.wcs_pix2world(f['fx'][sel], f['fy'][sel], 1)
    DR = (f['RA'][sel] - mRA)
    DD = (f['Dec'][sel] - mDec)
    
    RA0 += np.median(DR)
    Dec0 += np.median(DD)
    dR = np.cos(np.deg2rad(Dec0)) * 3600. * (ra0 - RA0)
    dD = 3600. * (dec0 - Dec0)
    log.info('%s_%07d Rotation offset and error: %0.2f, %0.2f' %(args.date,
                                                      args.observation,
                                                      A.rot-rot_i,
                                                      rot_error))
    log.info('%s_%07d offsets: %0.2f, %0.2f, %0.2f' %(args.date,
                                                      args.observation,
                                                      dR, dD,
                                                      A.rot-rot_i))
    A.get_pa()
    if (np.sqrt(dR**2 + dD**2) > 7.) or (np.abs(A.rot-rot_i) > 1.):
        return A1
    A1 = Astrometry(RA0, Dec0, A.pa, 0., 0., fplane_file=args.fplane_file)
    A1.tp = A1.setup_TP(RA0, Dec0, A1.rot, A1.x0,  A1.y0)
    return A1


###############################################################################
# MAIN SCRIPT

# Get directory name for path building
DIRNAME = get_script_path()

# Instrument for reductions
instrument = 'virus'

# Setup logging
log = setup_logging()
if args.hdf5file is None:
    log.error('Please specify an hdf5file.  A default is not yet setup.')
    sys.exit(1)

# Open HDF5 file
h5file = open_file(args.hdf5file, mode='r')
h5table = h5file.root.Cals
ifuslots = h5table.cols.ifuslot[:]

# Get unique IFU slot names
u_ifuslots = np.unique(ifuslots)

# Get photometric filter
T = Table.read(op.join(DIRNAME, 'filters/ps1g.dat'), format='ascii')
filtg = np.interp(def_wave, T['col1'], T['col2'], left=0.0, right=0.0)
filtg /= filtg.sum()
 
# Collect indices for ifuslots (target and neighbors)
sel1 = list(np.where(args.ifuslot == ifuslots)[0])
ifuslotn = get_slot_neighbors(args.ifuslot, u_ifuslots,
                              dist=args.neighbor_dist)
ifuslotn = np.setdiff1d(ifuslotn, badifuslots)
IFUs = get_ifuslots()
allifus = [args.ifuslot]
for ifuslot in ifuslotn:
    if ifuslot not in IFUs:
        continue
    silly = np.where(ifuslot == ifuslots)[0]
    if len(silly) != 4:
        continue
    sel1.append(silly)
    allifus.append(ifuslot)
ifuloop = np.array(np.hstack(sel1), dtype=int)
nslots = len(ifuloop) / 4
allifus = np.hstack(allifus)


# Reducing IFUSLOT
log.info('Reducing ifuslot: %03d' % args.ifuslot)
pos, twispectra, scispectra, errspectra, fn, tfile = reduce_ifuslot(ifuloop,
                                                                    h5table)

# Get fiber to fiber from twilight spectra
ftf = get_fiber_to_fiber(twispectra)

# Correct fiber to fiber
scispectra = safe_division(scispectra, ftf)
errspectra = safe_division(errspectra, ftf)


###################
# Subtracting Sky #
###################
log.info('Subtracting sky for all ifuslots')

# Number of exposures
nexp = scispectra.shape[0] / 448 / nslots
log.info('Number of exposures: %i' % nexp)

# Reorganize spectra by exposure
N, D = scispectra.shape
reorg = np.zeros((nexp, N/nexp, D))
for i in np.arange(nexp):
    x = np.arange(112)
    X = []
    for j in np.arange(4*nslots):
        X.append(x + j * 336 + i * 112)
    X = np.array(np.hstack(X), dtype=int)
    reorg[i] = scispectra[X]*1.
# Mask 0.0 values by setting to nan
reorg[reorg < 1e-42] = np.nan

# Correct offsets in the "gain" of each amplifier
# Not yet sure why this is needed physically
# But it is needed mathematically  
# The corrections are ~few percent.
reorg = np.array([r / correct_amplifier_offsets(r)[:, np.newaxis]
                  for r in reorg])

# Estimate sky with sigma clipping to find "sky" fibers    
skies = np.array([estimate_sky(r) for r in reorg])

# Take the ratio of the 2nd and 3rd sky to the first
# Assume the ratio is due to illumination differences
# Correct multiplicatively to the first exposure's illumination (scalar corrections)
exp_ratio = np.ones((nexp,))
for i in np.arange(1, nexp):
    exp_ratio[i] = np.nanmedian(skies[i] / skies[0])
    log.info('Ratio for exposure %i to exposure 1: %0.2f' %
             (i+1, exp_ratio[i]))

# Subtract sky and correct normalization
reorg[:] -= skies[:, np.newaxis, :]
reorg[:] *= exp_ratio[:, np.newaxis, np.newaxis]

# Reformat spectra back to 2d array [all exposures, amps, ifus]
cnt = 0
for j in np.arange(4*nslots): 
    for i in np.arange(nexp):
        l = j*112
        u = (j+1)*112
        scispectra[cnt:(cnt+112)] = reorg[i, l:u]*1.
        cnt += 112

errspectra[~np.isfinite(scispectra)] = np.nan

# Get RA, Dec from header
ra, dec, pa = get_ra_dec_from_header(tfile, fn)

# Build astrometry class
A = Astrometry(ra, dec, pa, 0., 0., fplane_file=args.fplane_file)

# Query catalog Sources in the area
log.info('Querying Pan-STARRS at: %0.5f %0.5f' % (ra, dec))
pname = 'Panstarrs_%0.6f_%0.5f_%0.4f.dat' % (ra, dec, 11. / 60.)
if op.exists(pname):
    Pan = Table.read(pname, format='ascii.fixed_width_two_line')
else:
    Pan = query_panstarrs(ra, dec, 11. / 60.)
    Pan.write(pname, format='ascii.fixed_width_two_line')

raC, decC, gC = (np.array(Pan['raMean']), np.array(Pan['decMean']),
                 np.array(Pan['gApMag']))
coords = SkyCoord(raC*units.degree, decC*units.degree, frame='fk5')


# Make g-band image and find sources
Total_sources = []
info = []
# Image scale and range
scale = 0.75
ran = [-23., 25., -23., 25.]
for i, ui in enumerate(allifus):
    # Get the info for the given ifuslot
    log.info('Making collapsed frame for %03d' % ui)
    N = 448 * nexp
    data = scispectra[N*i:(i+1)*N]
    F = np.nanmedian(ftf[N*i:(i+1)*N], axis=1)
    P = pos[N*i:(i+1)*N]
    name = '%s_%07d_%03d.fits' % (args.date, args.observation, ui)
    
    # Make g-band image
    image = make_photometric_image(P[:, 0], P[:, 1], data, filtg, F > 0.5,
                                   ADRx, 0.*ADRx, nchunks=11,
                                   ran=ran,  scale=scale)
    
    # Make full fits file with wcs info (using default header)
    mean, median, std = sigma_clipped_stats(image, sigma=3.0, stdfunc=mad_std)
    daofind = DAOStarFinder(fwhm=4.0, threshold=7. * std, exclude_border=True) 
    sources = daofind(image)
    log.info('Found %i sources' % len(sources))
    
    # Keep certain info for new loops
    info.append([image, fn, name, ui, tfile, sources])
    
    # Match sources to archive catalog
    Sources = match_to_archive(sources, image, A, ui, scale, ran, coords)
    if Sources is not None:
        Total_sources.append(Sources) 

# Write temporary info out
Total_sources = np.vstack(Total_sources)
f = Table(Total_sources, names = ['imagex', 'imagey', 'gmag', 'dist', 'Cgmag',
                              'RA', 'Dec', 'fx', 'fy', 'dra', 'ddec'])
f.write('sources.dat', format='ascii.fixed_width_two_line',
        overwrite=True)

# Fit astrometric offset
for j in np.arange(1):
    A = fit_astrometry(f, A)
    Total_sources = []
    for i, ui in enumerate(allifus):
        image, fn, name, ui, tfile, sources = info[i]
        Sources = match_to_archive(sources, image, A, ui, scale, ran, coords)
        if Sources is not None:
            Total_sources.append(Sources) 
    Total_sources = np.vstack(Total_sources)
    f = Table(Total_sources, names = ['imagex', 'imagey', 'gmag', 'dist',
                                      'Cgmag', 'RA', 'Dec', 'fx', 'fy', 'dra',
                                      'ddec'])
    f.write('sources.dat', format='ascii.fixed_width_two_line', overwrite=True)

# Extract
RAFibers, DecFibers = ([], [])
for i, _info in enumerate(info):
    image, fn, name, ui, tfile, sources = _info
    N = 448 * nexp
    data = scispectra[N*i:(i+1)*N]
    P = pos[N*i:(i+1)*N]
    ra, dec = A.get_ifupos_ra_dec('%03d' % ui, P[:, 0], P[:, 1])
    RAFibers.append(ra)
    DecFibers.append(dec)

mRA, mDec = A.tp.wcs_pix2world(f['fx'], f['fy'], 1)
E = Extract()
E.psf = E.tophat_psf(4, 10.5, 0.25)
E.coords = SkyCoord(mRA*units.deg, mDec*units.deg, frame='fk5')
E.ra, E.dec = [np.hstack(x) for x in [RAFibers, DecFibers]]
E.data = scispectra
E.error = errspectra
E.mask = np.isfinite(scispectra)
E.get_ADR_RAdec(A)
log.info('Beginning Extraction')
if args.parallelize:
    N = np.array(np.max([multiprocessing.cpu_count()-2, 1]), dtype=int)
    pool = Pool(N)
    spec_list = pool.map(E.get_spectrum_by_coord_index,
                         np.arange(len(E.coords)))
else:
    spec_list = []
    for i in np.arange(len(E.coords)):
        spec_list.append(E.get_spectrum_by_coord_index(i))
log.info('Finished Extraction')
name = ('%s_%07d.h5' %
                (args.date, args.observation))
h5spec = open_file(name, mode="w", title="%s Spectra" % name[:-3])

class Spectra(IsDescription):
     ra  = Float32Col()    # float  (single-precision)
     dec  = Float32Col()    # float  (single-precision)
     spectrum  = Float32Col((1036,))    # float  (single-precision)
     error  = Float32Col((1036,))    # float  (single-precision)
     image = Float32Col((21, 21))
     xgrid = Float32Col((21, 21))
     ygrid = Float32Col((21, 21))


table = h5spec.create_table(h5spec.root, 'Spectra', Spectra, 
                            "Extracted Spectra")
specrow = table.row
for ra, dec, specinfo in zip(mRA, mDec, spec_list):
    specrow['ra'] = ra
    specrow['dec'] = dec
    if len(specinfo[0]) > 0:
        specrow['spectrum'] = specinfo[0]
        specrow['error'] = specinfo[1]
        specrow['image'] = specinfo[2]
        specrow['xgrid'] = specinfo[3]
        specrow['ygrid'] = specinfo[4]
    specrow.append()
table.flush()
h5spec.close()
# Making g-band images
for i in info:
    image, fn, name, ui, tfile, sources = i
    crx = np.abs(ran[0]) / scale + 1.
    cry = np.abs(ran[2]) / scale + 1.
    ifuslot = '%03d' % ui
    A.get_ifuslot_projection(ifuslot, scale, crx, cry)
    header = A.tp_ifuslot.to_header()
    F = fits.PrimaryHDU(np.array(image, 'float32'), header=header)
    F.writeto(name, overwrite=True)

with open('ds9_%s_%07d.reg' % (args.date, args.observation), 'w') as k:
    MakeRegionFile.writeHeader(k)
    MakeRegionFile.writeSource(k, coords.ra.deg, coords.dec.deg)

if args.simulate:
    i = 0
    N = 448 * nexp
    F = np.nanmedian(ftf[N*i:(i+1)*N], axis=1)
    data = scispectra[N*i:(i+1)*N]
    P = pos[N*i:(i+1)*N]
    log.info('Simulating spectrum from %s' % args.source_file)
    simulated_spectrum = read_sim(args.source_file)
    simdata = simulate_source(simulated_spectrum, P, data, 
                                 args.source_x, args.source_y, 
                                 args.source_seeing)
    scispectra[N*i:(i+1)*N] = simdata * 1.

sel = f['dist'] < 2.5
print('Number of sources within 2.5": %i' % sel.sum())
mRA, mDec = A.tp.wcs_pix2world(f['fx'][sel], f['fy'][sel], 1)
plt.figure(figsize=(9, 8))
plt.gca().set_position([0.2, 0.2, 0.65, 0.65])
dr = np.cos(np.deg2rad(f['Dec'][sel])) * -3600. * (f['RA'][sel] - mRA)
dd = 3600. * (f['Dec'][sel] - mDec)
D = np.sqrt((f['fx'][sel] - f['fx'][sel][:, np.newaxis])**2 +
            (f['fy'][sel] - f['fy'][sel][:, np.newaxis])**2)
D[D==0.] = 999.
noneigh = np.min(D, axis=0) > 8.
nsel = (np.sqrt(dr**2 + dd**2) < 1.) * noneigh 
plt.scatter(dr, dd, alpha=0.75, s=45)
plt.axis([-1.5, 1.5, -1.5, 1.5])
plt.xlabel(r'$\Delta$ RA (")', fontsize=20, labelpad=20)
plt.ylabel(r'$\Delta$ Dec (")', fontsize=20, labelpad=20)
plt.savefig('astrometry_%s_%07d.png'  % (args.date, args.observation), dpi=300)
plt.figure(figsize=(9, 8))
Mg = Total_sources[sel, 4][nsel]
mg = Total_sources[sel, 2][nsel]
ss = (Mg > 15) * (Mg < 22) * (mg < 25.)
print((mg - Mg)[ss])
mean, median, std = sigma_clipped_stats((mg - Mg)[ss], stdfunc=mad_std)
print('The mean, median, and std for the mag offset is for %s_%07d: '
      '%0.2f, %0.2f, %0.2f' % (args.date, args.observation, mean, median, std))
plt.gca().set_position([0.2, 0.2, 0.65, 0.65])
plt.scatter(Mg, mg - Mg - median, alpha=0.75, s=75)
plt.plot([15, 20], [std, std], 'r--', lw=1)
plt.plot([15, 20], [-std, -std], 'r--', lw=1)
plt.xlim([15, 20])
plt.ylim([-0.5, 0.5])
plt.xlabel('Pan-STARRS g (AB mag)', fontsize=20, labelpad=20)
plt.ylabel('VIRUS g - Pan-STARRS g (AB mag)', fontsize=20, labelpad=20)
plt.savefig('mag_offset_%s_%07d.png'  % (args.date, args.observation), dpi=300)
sys.exit(1)


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

if tfile is not None:
    t = tarfile.open(tfile, 'r')
    a = fits.open(t.extractfile(fn))
    t.close()
else:
    a = fits.open(fn)

he = a[0].header

# Making data cube
log.info('Making Cube')
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    zgrid, xgrid, ygrid = make_cube(pos[:, 0], pos[:, 1], scispectra, 
                                     ADRx, 0. * def_wave, ftf)

    write_cube(def_wave, xgrid, ygrid, zgrid, cubename, he)


h5file.close()
    
        
