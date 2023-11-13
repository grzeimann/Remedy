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
import numpy as np
import os
import gc
import os.path as op
import seaborn as sns
import sys
import tarfile
import warnings
import psutil


from astrometry import Astrometry
from astroquery.mast import Catalogs
from astropy.coordinates import SkyCoord, match_coordinates_sky
from astropy.convolution import convolve, Gaussian2DKernel
from astropy.convolution import interpolate_replace_nans, Gaussian1DKernel
from astropy.io import fits
from astropy.modeling.models import Polynomial2D
from astropy.modeling.fitting import LevMarLSQFitter, FittingWithOutlierRemoval
from astropy.stats import biweight_location, sigma_clip
from astropy.stats import sigma_clipped_stats, mad_std
from astropy.table import Table
from catalog_search import query_panstarrs, MakeRegionFile
from datetime import datetime, timedelta
from extract import Extract
from fiber_utils import identify_sky_pixels, measure_fiber_profile, get_trace
from fiber_utils import build_model_image, detect_sources, get_powerlaw
from fiber_utils import get_spectra, get_spectra_error, get_spectra_chi2
from input_utils import setup_logging
from math_utils import biweight
from photutils.detection import DAOStarFinder
from photutils.aperture import aperture_photometry
from photutils.aperture import CircularAperture
from photutils.centroids import centroid_com
from scipy.interpolate import griddata, interp1d, interp2d
from scipy.signal import savgol_filter
from sklearn.decomposition import PCA
from tables import open_file, IsDescription, Float32Col, StringCol, Int32Col
from matplotlib.ticker import MultipleLocator
from matplotlib.colors import ListedColormap


# Plot Style
sns.set_context('notebook')
sns.set_style('ticks')

# Turn off annoying warnings (even though some deserve attention)
warnings.filterwarnings("ignore")

###############################################################################
# CONFIGURATION AND GLOBAL VARIABLES 

# Amps
amps = ['LL', 'LU', 'RL', 'RU']

# Ignore these ifuslots because they are subpar or look poor
badifuslots = np.array([])

# Default dither pattern for 3 exposures
dither_pattern = np.array([[0., 0.], [1.215, -0.70], [1.215, 0.70]])

# Rectified wavelength
def_wave = np.linspace(3470., 5540., 1036)

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

parser.add_argument("-nt", "--nametype",
                    help='''nametype''',
                    type=str, default='sci')

parser.add_argument("-s", "--simulate",
                    help='''Simulate source''',
                    action="count", default=0)

parser.add_argument("-nm", "--no_masking",
                    help='''No Masking Employed''',
                    action="count", default=0)

parser.add_argument("-nD", "--no_detect",
                    help='''No Detection''',
                    action="count", default=0)

parser.add_argument("-qs", "--quick_sky",
                    help='''Quick Sky''',
                    action="count", default=0)

parser.add_argument("-mc", "--make_cube",
                    help='''Make Cube''',
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
    try:
        ra = a[0].header['TRAJCRA'] * 15.
        dec = a[0].header['TRAJCDEC'] * 1.
        ra1 = a[0].header['TRAJRA'] * 15.
        dec1 = a[0].header['TRAJDEC'] * 1.
        dra = (ra - ra1)*np.cos(dec*np.pi/180.)*3600.
        ddec = (dec - dec1)*3600.
        d = np.sqrt(dra**2+ddec**2)
        if d > 25.:
            ra = ra1
            dec = dec1
    except:
        ra = a[0].header['TRAJRA'] * 15.
        dec = a[0].header['TRAJDEC'] * 1.
    pa = a[0].header['PARANGLE'] * 1.
    if tfile is not None:
        t.close()
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
                           '*', 'LL', exp='exp01', base=args.nametype)
        
    filenames = sorted(glob.glob(file_glob))
    
    tfile = None
    if len(filenames) == 0:
        # do we have a tar?
        path = splitall(file_glob)
        tfile = op.join(*path[:-3]) + ".tar"
        if op.exists(tfile):
            print("This observation is tarred. Trying workaround...")
            fnames_glob = op.join(*path[-4:])
            mzip = []
            with tarfile.open(tfile) as tf:
                filenames = fnmatch.filter(tf.getnames(), fnames_glob)
                for filename in filenames:
                    f = fits.open(tf.extractfile(filename))
                    specid = '%03d' % int(f[0].header['SPECID'])
                    ifuid = '%03d' % int(f[0].header['IFUID'])
                    ifuslot = '%03d' % int(f[0].header['IFUSLOT'])
                    mzip.append('%s_%s_%s' % (specid, ifuslot, ifuid))
        else:
            print("Could not find %s" % tfile)
            print("Crap! How did I get here?")
            sys.exit(1)
    return np.unique(mzip)

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


def base_reduction(filename, get_header=False, tfile=None, rdnoise=None):
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
    overscan_length = int(32 * (image.shape[1] / 1064))
    O = biweight_location(image[:, -(overscan_length-2):])
    image[:] = image - O
    
    # Trim image
    image = image[:, :-overscan_length]
    
    # Gain multiplication (catch negative cases)
    gain = a[0].header['GAIN']
    gain = np.where(gain > 0., gain, 0.85)
    if rdnoise is None:
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
    if tfile is not None:
        t.close()
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

def get_twi_tarfile(pathname, date, kind='twi'):
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
                    if name[-8:-5] == kind:
                        twitarfile = tarfolder
                    flag = False
            if twitarfile is not None:
                break
            T.close()
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

def get_twi_spectra(array_flt, array_trace, wave, def_wave):
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
    twi_spectrum = np.zeros((array_trace.shape[0], array_trace.shape[1]))
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
        twi_spectrum[fiber] = array_flt[indl, x] / 2. + array_flt[indh, x] / 2.
    twi_spectrum[~np.isfinite(twi_spectrum)] = 0.0
    return twi_spectrum


def get_spectra_quick(array_sci, array_err, array_flt, plaw, mdark, array_trace, wave, def_wave,
                      pixelmask, mastersci):
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
    plaw_spectrum = np.zeros((array_trace.shape[0], def_wave.shape[0]))
    mdark_spectrum = np.zeros((array_trace.shape[0], def_wave.shape[0]))
    chi2_spectrum = np.zeros((array_trace.shape[0], def_wave.shape[0]))
    msci_spectrum = np.zeros((array_trace.shape[0], def_wave.shape[0]))

    mask = np.zeros((array_trace.shape[0], def_wave.shape[0]))

    N = array_flt.shape[0]
    x = np.arange(array_flt.shape[1])
    for fiber in np.arange(array_trace.shape[0]):
        dw = np.diff(wave[fiber])
        dw = np.hstack([dw[0], dw])
        if np.round(array_trace[fiber]).min() < 2:
            continue
        if np.round(array_trace[fiber]).max() >= (N-2):
            continue
        indv = np.round(array_trace[fiber]).astype(int)
        tw, sw, ew, pm, pl, md, msw = (0., 0., 0., 0., 0., 0., 0.)
        chi2 = np.zeros((len(indv), 6, 3))
        for j in np.arange(-3, 3):
            if j == -3:
                w = indv + j + 1 - (array_trace[fiber] - 2.5)
            elif j == 2:
                w = (2.5 + array_trace[fiber]) - (indv + j) 
            else:
                w = 1.
            tw += array_flt[indv+j, x] * w
            sw += array_sci[indv+j, x] * w
            msw += mastersci[indv+j, x] * w
            pl += plaw[indv+j, x] * w
            md += mdark[indv+j, x] * w
            ew += array_err[indv+j, x]**2 * w
            pm += pixelmask[indv+j, x]
            chi2[:, j+3, 0] = array_sci[indv+j, x] * w
            chi2[:, j+3, 1] = array_flt[indv+j, x] * w
            chi2[:, j+3, 2] = array_err[indv+j, x] * w
        norm = chi2[:, :, 0].sum(axis=1) / chi2[:, :, 1].sum(axis=1)
        num = (chi2[:, :, 0] - chi2[:, :, 1] * norm[:, np.newaxis])**2
        denom = (chi2[:, :, 2] + 0.01*chi2[:, :, 0].sum(axis=1)[:, np.newaxis])**2
        chi2a = 1. / (1. + 5.) * np.sum(num / denom, axis=1)
        ew = np.sqrt(ew)
        twi_spectrum[fiber] = np.interp(def_wave, wave[fiber], tw / dw,
                                        left=0.0, right=0.0)
        plaw_spectrum[fiber] = np.interp(def_wave, wave[fiber], pl / dw,
                                        left=0.0, right=0.0)
        mdark_spectrum[fiber] = np.interp(def_wave, wave[fiber], md / dw,
                                        left=0.0, right=0.0)
        sci_spectrum[fiber] = np.interp(def_wave, wave[fiber], sw / dw,
                                        left=0.0, right=0.0)
        msci_spectrum[fiber] = np.interp(def_wave, wave[fiber], msw / dw,
                                        left=0.0, right=0.0)
        mask[fiber] = np.interp(def_wave, wave[fiber], pm / dw,
                                        left=0.0, right=0.0)
        chi2_spectrum[fiber] = np.interp(def_wave, wave[fiber], chi2a,
                                        left=0.0, right=0.0)
        # Not real propagation of error, but skipping math for now
        err_spectrum[fiber] = np.interp(def_wave, wave[fiber], ew / dw,
                                        left=0.0, right=0.0)
    total_mask = (~np.isfinite(twi_spectrum)) * (mask > 0.)
    twi_spectrum[total_mask] = 0.0
    err_spectrum[total_mask] = 0.0
    sci_spectrum[total_mask] = 0.0
    msci_spectrum[total_mask] = 0.0
    return (twi_spectrum, sci_spectrum, err_spectrum, plaw_spectrum,
            mdark_spectrum, chi2_spectrum, msci_spectrum)


def get_continuum(spectra, nbins=25):
    '''
    Get continuum from sky-subtracted spectra
    
    Parameters
    ----------
    spectra : 2d numpy array
        sky-subtracted spectra for each fiber
        
    Returns
    -------
    cont : 2d numpy array
        continuum normalization
    '''
    a = np.array([biweight(f, axis=1) 
                  for f in np.array_split(spectra, nbins, axis=1)]).swapaxes(0, 1)
    x = np.array([np.mean(xi) 
                  for xi in np.array_split(np.arange(spectra.shape[1]), nbins)])
    cont = np.zeros(spectra.shape)
    X = np.arange(spectra.shape[1])
    for i, ai in enumerate(a):
        sel = np.isfinite(ai)
        if np.sum(sel)>nbins/2.:
            I = interp1d(x[sel], ai[sel], kind='quadratic',
                         fill_value='extrapolate')
            cont[i] = I(X)
        else:
            cont[i] = 0.0
    return cont

def get_fiber_to_fiber(fltspec, scispec, wave_all, twispec):
    '''
    Get Fiber to Fiber from twilight spectra
    
    Parameters
    ----------
    t : 2d numpy array
        twilight spectra for each fiber
        
    Returns
    -------
    ftf : 2d numpy array
        fiber to fiber normalization
    '''
    
    inds = np.argsort(wave_all.ravel())
    S = scispec.ravel()
    avgsci = [biweight(chunk) for chunk in np.array_split(S[inds], 3000)]
    avgwav = [np.mean(w) for w in np.array_split(wave_all.ravel()[inds], 3000)]
    S = fltspec.ravel()
    avgflt = [biweight(chunk) for chunk in np.array_split(S[inds], 3000)]
    S = twispec.ravel()
    avgtwi = [biweight(chunk) for chunk in np.array_split(S[inds], 3000)]
    S = interp1d(avgwav, avgsci, bounds_error=False, fill_value=np.nan)
    F = interp1d(avgwav, avgflt, bounds_error=False, fill_value=np.nan)
    T = interp1d(avgwav, avgtwi, bounds_error=False, fill_value=np.nan)

    ftfflt = fltspec * 0.
    ftfsci = scispec * 0.
    ftftwi = twispec * 0.
    
    for i in np.arange(scispec.shape[0]):
        w = wave_all[i]
        si = scispec[i]
        fi = fltspec[i]
        ti = twispec[i]
        ftfflt[i] = fi / F(w) 
        ftfsci[i] = si / S(w)
        ftftwi[i] = ti / T(w)
    z = get_continuum(ftftwi / ftfflt, nbins=5)
    ftf = ftfflt * z
    sky = T(wave_all) * ftf
    ratio = (twispec - sky) / sky
    zr = get_continuum(ratio, nbins=25)
    ftf = ftfflt * z * (1 + zr)
    sky = S(wave_all) * ftf 
    error = np.sqrt((5. * 3.2**2) + (scispec * 5.)) / 5.
    cont = get_continuum(scispec-sky, nbins=50)
    return ftf, (scispec - sky - cont) / error
    
def background_pixels(trace, image):
    back = np.ones(image.shape, dtype=bool)
    x = np.arange(image.shape[1])
    for fibert in trace:
        for j in np.arange(-7, 8):
            indv = np.round(fibert) + j
            m = np.max([np.zeros(fibert.shape), indv], axis=0)
            n = np.min([m, np.ones(fibert.shape)*(image.shape[0]-1)], axis=0)
            inds = np.array(n, dtype=int)
            back[inds, x] = False
    return back

def get_fiber_to_fiber_adj(scispectra, ftf, nexp):
    Adj = ftf * 0.0
    inds = np.arange(scispectra.shape[0])
    for k in np.arange(nexp):
        sel = np.where(np.array(inds / 112, dtype=int) % nexp == k)[0]
        bins = 9
        adj = np.zeros((len(sel), bins))
        X = np.array([np.mean(xi) for xi in np.array_split(def_wave, bins)])
        cnt = 0
        for o, f in zip(np.array_split(scispectra[sel], bins, axis=1),
                        np.array_split(ftf[sel], bins, axis=1)):
            y = biweight(o / f, axis=1)
            norm = biweight(y)
            mask, cont = identify_sky_pixels(y, kernel=2.5)
            adj[:, cnt] = cont / norm
            cnt += 1
        for j in np.arange(len(sel)):
            good = np.isfinite(adj[j])
            if good.sum() > bins-3:
                Adj[sel[j]] = interp1d(X, adj[j], kind='quadratic',
                                       fill_value='extrapolate')(def_wave)
            else:
                scispectra[sel(j)] = np.nan
    return scispectra, Adj

def get_mask(scispectra, C1, ftf, res, nexp):
    mask = np.zeros(scispectra.shape, dtype=bool)

    # Chi2 cut (pre-error adjustment, which is usually 0.8, so may need to think more)
    badchi2 = (C1 > 5.)
    y, x = np.where(badchi2)
    for i in np.arange(-2, 3):
        x1 = x + i
        n = (x1 > -1) * (x1 < mask.shape[1])
        mask[y[n], x1[n]] = True
    
    # Fiber to fiber < 0.5 (dead fiber) and |res| > 0.25, meaning mastersky residuals > 25%
    badftf = (ftf < 0.3)# + (res < -0.5) + (res > 0.75)
    mask[badftf] = True
    
    # Error spectra with 0.0 are either outside the wavelength range or pixels masked by bad pixel mask
    badpixmask = np.isnan(scispectra) + (scispectra == 0.)
    mask[badpixmask] = True
    
    # Flag a fiber as bad if more than 200 columns are already flagged bad
    badfiberflag = mask.sum(axis=1) > 200
    badfibers = np.where(badfiberflag)[0]
    mask[badfibers] = True
    
    inds = np.arange(scispectra.shape[0])
    # Flag an amp as bad if more than 20% of their fibers are marked as bad
    for k in np.arange(0, int(len(inds)/112./nexp), dtype=int):
        ll = int(k*112*nexp)
        hl = int((k+1)*112*nexp)
        idx = int(k / 4)
        ampid = amps[k % 4]
        bad_columns = np.sum(mask[ll:hl], axis=0) > (0.2 * nexp * 112)
        mask[ll:hl, bad_columns] = True
        mfrac = ((mask[ll:hl].sum()*1.) / 
                 (mask[ll:hl].shape[0]*mask[ll:hl].shape[1])*1.)
        if mfrac > 0.2:
            log.info('%03d%s Amplifier marked bad.' % (allifus[idx], ampid))
            mask[ll:hl] = True
    return mask

def power_law(x, c1, c2=.5, c3=.15, c4=1., sig=2.5):
    '''
    Power law for scattered light from mirror imperfections
    
    Parameters
    ----------
    x : float or 1d numpy array
        Distance from fiber in pixels
    c1 : float
        Normalization of powerlaw
    
    Returns
    -------
    plaw : float or 1d numpy array
        see function form below
    '''
    return c1 / (c2 + c3 * np.power(abs(x / sig), c4))

def get_powerlaw_ydir(trace, spec, amp, col):
    '''
    Get powerlaw in ydir for a given column
    
    Parameters
    ----------
    trace : 2d numpy array
        y position as function of x for each fiber
    spec : 2d numpy array
        fiber spectra
    amp : str
        amplifier
    col : int
        Column
    '''
    if amp in ['LL', 'RU']:
        ntrace = np.vstack([trace, 2064 - trace])
    else: 
        ntrace = np.vstack([-trace, trace])
    nspec = np.vstack([spec, spec])
    YM, XM = np.indices(ntrace.shape)
    yz = np.linspace(0, 1031, 25)
    plaw = []
    for yi in yz:
        d = np.sqrt((yi - ntrace[:, ::43])**2 + (col - XM[:, ::43])**2)
        plaw.append(np.nansum(nspec[:, ::43] *
                              power_law(d, 1.4e-5, c3=2., c4=1.0,  sig=1.5)))
    return yz, np.array(plaw)   


def get_sci_twi_files(kind='twi'):
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
                           '047', 'LL', base=args.nametype)
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
                             base=kind)
        twinames = sorted(roll_through_dates(pathname, args.date))
        twitarfile = None
    else:
        file_glob = build_path(args.rootdir, args.date, '*',
                           '047', 'LL')
        path = splitall(file_glob)
        tarname = op.join(*path[:-3]) + ".tar"
        twitarfile = get_twi_tarfile(tarname, args.date, kind=kind)
        with tarfile.open(twitarfile) as tf:
            twinames = sorted(tf.getnames())
    return scinames, twinames, scitarfile, twitarfile

def reduce_ifuslot(ifuloop, h5table, tableh5):
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
    _i, intm = ([], [])
    
    
    # Check if tarred

    scinames, twinames, scitarfile, twitarfile = get_sci_twi_files()
    
    specrow = tableh5.row
    ifuslot = '%03d' % h5table[ifuloop[0]]['ifuslot']
    amp = 'LL'
    fnames_glob = '*/2*%s%s*%s.fits' % (ifuslot, amp, args.nametype)
    filenames = fnmatch.filter(scinames, fnames_glob)
    nexposures = len(filenames)
    N = len(ifuloop) * nexposures * 112
    ExP = np.zeros((N,))
    p = np.zeros((N, 2))
    t = np.zeros((N, 1032))
    ft = np.zeros((N, 1032))
    s = np.zeros((N, 1032))
    ms = np.zeros((N, 1032))
    e = np.zeros((N, 1032))
    c1 = np.zeros((N, 1032))
    wa = np.zeros((N, 1032))
    cnt = 0
    for ind in ifuloop:
        ifuslot = '%03d' % h5table[ind]['ifuslot']
        ifuid = h5table[ind]['ifuid'].decode("utf-8")
        specid = h5table[ind]['specid'].decode("utf-8")
        amp = h5table[ind]['amp'].decode("utf-8")
        amppos = h5table[ind]['ifupos']
        wave = h5table[ind]['wavelength']
        dw = np.diff(wave, axis=1)
        dw = np.hstack([dw[:, 0:1], dw])
        trace = h5table[ind]['trace']
        if np.min(trace) < 0.:
            trace = 0. * trace
        readnoise = h5table[ind]['readnoise']
        masterflt = h5table[ind]['masterflt']
        mastertwi = h5table[ind]['mastertwi']

        mastersci = h5table[ind]['mastersci']
        maskspec = h5table[ind]['maskspec']
        masterbias = h5table[ind]['masterbias']
        try:
            masterdark = h5table[ind]['masterdark']
            pixelmask = h5table[ind]['pixelmask']
        except:
            log.warning("Can't open masterdark, pixelmask")
            masterdark = np.zeros((1032, 1032))
            pixelmask = np.zeros((1032, 1032), dtype=int)
        
        log.info('Getting powerlaw for mastertwi for %s%s' % (ifuslot, amp))
        masterdark[:] = masterdark - masterbias
        masterflt[:] = masterflt - masterbias
        try:
            plaw = get_powerlaw(masterflt, trace)
        except:
            plaw = 0.
        masterflt[:] = masterflt - plaw
        
        mastertwi[:] = mastertwi - masterbias
        try:
            plaw = get_powerlaw(mastertwi, trace)
        except:
            plaw = 0.
        mastertwi[:] = mastertwi - plaw
        
        mastersci[:] = mastersci - masterdark - masterbias
        try:
            plaw2 = get_powerlaw(mastersci, trace)
        except:
            plaw = 0.
        mastersci[:] = mastersci - plaw2
        
        log.info('Done making mastertwi for %s%s' % (ifuslot, amp))
        fnames_glob = '*/2*%s%s*%s.fits' % (ifuslot, amp, args.nametype)
        filenames = fnmatch.filter(scinames, fnames_glob)
        for j, fn in enumerate(filenames):
            sciimage, scierror, header = base_reduction(fn, tfile=scitarfile,
                                                rdnoise=readnoise, get_header=True)
            # Check here that it is the same unit
            facexp = header['EXPTIME'] / 360.
            if facexp < 0.:
                facexp = 1.
            sciimage[:] = sciimage - masterdark*facexp - masterbias
            try:
                sci_plaw = get_powerlaw(sciimage, trace)
            except:
                sci_plaw = 0.
            sciimage[:] = sciimage - sci_plaw
            flt = get_spectra(masterflt, trace) / dw
            twi = get_spectra(mastertwi, trace) / dw

            spec = get_spectra(sciimage, trace) / dw
            espec = get_spectra_error(scierror, trace) / dw
            chi21 = get_spectra_chi2(masterflt, sciimage, scierror, trace)
            mask1 = get_spectra(pixelmask, trace)
            mspec = get_spectra(mastersci, trace) / dw
            #mask1 = mask1 + maskspec
            for arr in [spec, espec, flt, twi, mspec]:
                arr[mask1>0.] = np.nan
            if j==0:
                #intpm, shifts = measure_fiber_profile(masterflt, twi, 
                #                                          trace, wave)
                intpm = None
            
            ExP[cnt:cnt+112] = np.array([header['EXPTIME']]*112)
            if nexposures == 3:
                pos = amppos + dither_pattern[j]
            else:
                pos = amppos * 1.
            N = flt.shape[0]
            _I = np.char.array(['%s_%s_%s_%s' % (specid, ifuslot, ifuid, amp)] * N)
            _V = '%s_%s_%s_%s_exp%02d' % (specid, ifuslot, ifuid, amp, j+1)
            #specrow['image'] = sciimage
            specrow['zipcode'] = _V
            #specrow['mastersci'] = mastersci
            specrow.append()
#            for loop in np.arange(len(spec)):
#                specrow['observed'] = spec1[loop]
#                specrow['observed_error'] = espec1[loop]
#                specrow['plawspec'] = plaw1[loop]
#                specrow['mdarkspec'] = mdark1[loop]
#                specrow['trace'] = trace[loop]
#                specrow['wavelength'] = wave[loop]
#                specrow.append()
            p[cnt:cnt+112, :] = pos
            t[cnt:cnt+112, :] = twi
            ft[cnt:cnt+112, :] = flt
            s[cnt:cnt+112, :] = spec
            ms[cnt:cnt+112, :] = mspec
            e[cnt:cnt+112, :] = espec
            c1[cnt:cnt+112, :] = chi21
            wa[cnt:cnt+112, :] = wave
            cnt += 112
            for x, i in zip([_i, intm], [_I, [intpm, 0.0, _V]]):
                x.append(i * 1) 
        
        process = psutil.Process(os.getpid())
        log.info('Memory Used: %0.2f GB' % (process.memory_info()[0] / 1e9))
    _i = np.vstack(_i)
    tableh5.flush()
    tableh5 = None
    return p, ft, s, e, wa, filenames, scitarfile, _i, c1, intm, ExP, ms, t


def make_cube(xloc, yloc, data, error, Dx, Dy, scale, ran,
              seeing_fac=1.8, radius=1.5):
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
    Ecube = Dcube * 0.
    area = np.pi * 0.75**2
    G = Gaussian2DKernel(seeing / 2.35)
    S = np.zeros((data.shape[0], 2))
    Gp = Gaussian1DKernel(5.)
    c = data * 0.
    for i in np.arange(data.shape[0]):
        c[i] = interpolate_replace_nans(data[i], Gp)
    data = c * 1.
    for k in np.arange(b):
        S[:, 0] = xloc - Dx[k]
        S[:, 1] = yloc - Dy[k]
        sel = np.isfinite(data[:, k])
        if np.any(sel):
            grid_z = griddata(S[sel], data[sel, k],
                              (xgrid, ygrid), method='linear')
            Dcube[k, :, :] = (convolve(grid_z, G, boundary='extend') *
                              scale**2 / area)
            grid_z = griddata(S[sel], error[sel, k],
                              (xgrid, ygrid), method='linear')
            Ecube[k, :, :] = (convolve(grid_z, G, boundary='extend') *
                              scale**2 / area)
    return Dcube[:, 1:-1, 1:-1], Ecube[:, 1:-1, 1:-1], xgrid[1:-1, 1:-1], ygrid[1:-1, 1:-1]


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

def correct_amplifier_offsets(data, ftf, fibers_in_amp=112, order=1):
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

def get_mirror_illumination_throughput(fn=None, default=51.4e4, default_t=1.,
                                       default_iq=1.8):
    ''' Use hetillum from illum_lib to calculate mirror illumination (cm^2) '''
    try:
        F = fits.open(fn)
        names = ['RHO_STRT', 'THE_STRT', 'PHI_STRT', 'X_STRT', 'Y_STRT']
        r, t, p, x, y = [F[0].header[name] for name in names]
        mirror_illum = float(os.popen('/home1/00156/drory/illum_lib/hetillum -p'
                             ' -x "[%0.4f,%0.4f,%0.4f]" "[%0.4f,%0.4f]" 256' %
                                      (x, y, p, 0.042, 0.014)).read().split('\n')[0])
        area = mirror_illum * default
        if (F[0].header['TRANSPAR'] < 0.1) or (F[0].header['TRANSPAR'] > 1.05):
            transpar = default_t
        else:
            transpar = F[0].header['TRANSPAR'] * 1.
        if (F[0].header['IQ'] < 0.8) or (F[0].header['IQ'] > 4.0):
            iq = default_iq
        else:
            iq = F[0].header['IQ']
    except:
        log.info('Using default mirror illumination value')
        area = default
        transpar = default_t
        iq = default_iq
    return area, transpar, iq


def get_mirror_illumination_guider(fn, exptime, default=51.4e4, default_t=1.,
                                   default_iq=1.8,
                                   path='/work/03946/hetdex/maverick'):
    try:
        M = []
        path = op.join(path, args.date)
        f = op.basename(fn)
        DT = f.split('_')[0]
        y, m, d, h, mi, s = [int(x) for x in [DT[:4], DT[4:6], DT[6:8], DT[9:11],
                             DT[11:13], DT[13:15]]]
        d0 = datetime(y, m, d, h, mi, s)
        tarfolders = op.join(path, 'gc*', '*.tar')
        tarfolders = glob.glob(tarfolders)
        if len(tarfolders) == 0:
            area = 51.4e4
            log.info('No guide camera tarfolders found')
            return default, default_t, default_iq
        for tarfolder in tarfolders:
            T = tarfile.open(tarfolder, 'r')

            init_list = sorted([name for name in T.getnames()
                                if name[-5:] == '.fits'])
            final_list = []
            
            for t in init_list:
                DT = op.basename(t).split('_')[0]
                y, m, d, h, mi, s = [int(x) for x in [DT[:4], DT[4:6], DT[6:8],
                                     DT[9:11], DT[11:13], DT[13:15]]]
                d = datetime(y, m, d, h, mi, s)
                p = (d - d0).seconds

                if (p > -10.) * (p < exptime+10.):
                    final_list.append(t)
            for fn in final_list:
                fobj = T.extractfile(T.getmember(fn))
                M.append(get_mirror_illumination_throughput(fobj))
        M = np.array(M)
        sel = M[:, 2] != 1.8
        if sel.sum() > 0.:
            transpar = np.median(M[sel, 1])
            area = np.median(M[sel, 0])
            iq = np.median(M[sel, 2])
        else:
            area = 51.4e4
            transpar = 1.
            iq = 1.8
        return area, transpar, iq
    except: 
        log.info('Using default mirror illumination: %0.2f m^2' % (default/1e4))
        return default, default_t, default_iq

def make_photometric_image(x, y, data, filtg, mask, Dx, Dy,
                           nchunks=25, ran=[-23, 25, -23, 25], scale=0.75,
                           kind='linear', seeing=1.76):
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
    cDy = [np.mean(dy) for dy in np.array_split(Dy, nchunks)]
    S = np.zeros((len(x), 2))
    G = Gaussian2DKernel(seeing/2.35/scale)
    area = np.pi * 0.75**2 / scale**2
    WT = []
    simage = 0. * grid_x
    for k in np.arange(nchunks):
        S[:, 0] = x - cDx[k]
        S[:, 1] = y - cDy[k]
        sel = np.isfinite(chunks[k])
        if sel.sum()>50:
            image = chunks[k]
            grid_z0 = griddata(S, image, (grid_x, grid_y),
                               method=kind)
            grid_z0 = convolve(grid_z0, G, fill_value=np.nan,
                               preserve_nan=True)
            simage += weights[k] * grid_z0 / area
            WT.append(weights[k])
    image = simage / np.sum(WT)
    if np.all(image == 0.):
        image = np.nan * grid_x
    return image

def match_to_archive(sources, image, A, ifuslot, scale, ran, coords, gC,
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
    positions = []
    X, Y = (np.array(sources['xcentroid']),
            np.array(sources['ycentroid']))
    for x, y in zip(X, Y):
        positions.append([float(x), float(y)])

    apertures = CircularAperture(positions, r=apradius)
    phot_table = aperture_photometry(image, apertures,
                                     mask=~np.isfinite(image))
    gmags = np.where(phot_table['aperture_sum'] > 0.,
                     -2.5 * np.log10(phot_table['aperture_sum']) + 23.9,
                     99.)
    Sources = np.zeros((len(sources), 12))
    Sources[:, 0], Sources[:, 1] = (X, Y)
    Sources[:, 2] = gmags
    B = A.get_ifupos_ra_dec('%03d' % ifuslot,
                                  Sources[:, 0]*scale + ran[0],
                                  Sources[:, 1]*scale + ran[2])
    RA, Dec = A.get_ifupos_ra_dec('%03d' % ifuslot,
                                  Sources[:, 0]*scale + ran[0],
                                  Sources[:, 1]*scale + ran[2])
    ifu = A.fplane.by_ifuslot('%03d' % ifuslot)
    ifux, ifuy = (ifu.y, ifu.x) 
    fx, fy = (X*scale + ran[0] + ifux,
              Y*scale + ran[2] + ifuy)

    # Find closest bright source, use that for offset, then match
    # Make image class that combines, does astrometry, detections, updates coords
    C = SkyCoord(RA*units.deg, Dec*units.deg, frame='fk5')
    idx, d2d, d3d = match_coordinates_sky(C, coords)
    Sources[:, 3] = d2d.arcsecond
    Sources[:, 4] = gC[idx]
    Sources[:, 5] = coords[idx].ra.deg
    Sources[:, 6] = coords[idx].dec.deg
    Sources[:, 7], Sources[:, 8] = (fx, fy)
    Sources[:, 9], Sources[:, 10] = (RA-Sources[:,5], Dec-Sources[:, 6])
    Sources[:, 11] = ifuslot
    ix, iy = A.tp.wcs_world2pix(Sources[:, 5], Sources[:, 6], 1)
    ix, iy = (ix - ifux, iy - ifuy)
    xc, yc = ((ix -ran[0])/scale, (iy-ran[2])/scale)
    Sources[:, 0], Sources[:, 1] = (xc, yc)
    return Sources

def fit_astrometry(f, A1, thresh=25.):
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
    fitter = FittingWithOutlierRemoval(LevMarLSQFitter(), sigma_clip)
    sel = (f['dist'] < thresh) * (f['Cgmag']<22.) * (f['Cgmag'] > 15.)
    dr = np.array(np.cos(np.deg2rad(f['Dec']))*f['dra']*3600.)
    dd = np.array(f['ddec']*3600.)
    d = np.sqrt((dr[:, np.newaxis] - dr)**2 + (dd[:, np.newaxis] - dd)**2)
    nneigh = (d < 1.5).sum(axis=1)
    ind = np.argmax(nneigh)
    sel = sel * (d[:, ind]<1.5)
    log.info('RA, Dec offset with most sources: %0.2f, %0.2f' % (dr[ind], dd[ind]))
    log.info('Number of sources within 1.5" of max density: %i' % (sel.sum()))
    if sel.sum() < 4.:
        return A
    fitr, filtered_r = fitter(P, f['fx'][sel], f['fy'][sel], f['RA'][sel])
    fitd, filtered_d = fitter(P, f['fx'][sel], f['fy'][sel], f['Dec'][sel])
    ra0 = A.ra0 * 1.
    dec0 = A.dec0 * 1.
    rot_i = A.rot * 1.
    RA0 = fitr(0., 0.)
    Dec0 = fitd(0., 0.)
    dR = np.cos(np.deg2rad(Dec0)) * 3600. * (ra0 - RA0)
    dD = 3600. * (dec0 - Dec0)
    log.info('%s_%07d initial offsets: %0.2f, %0.2f' %(args.date, 
                                                       args.observation, dR,
                                                       dD))
    min_std = 9999
    for aoff in np.linspace(-0.3, 0.3, 61):
        rot = A.rot * 1. + aoff
        A.tp = A.setup_TP(A.ra0, A.dec0, rot, A.x0,  A.y0)
        mRA, mDec = A.tp.wcs_pix2world(f['fx'][sel], f['fy'][sel], 1)
        DR = (f['RA'][sel] - mRA) * np.cos(np.deg2rad(Dec0)) * 3600.
        DD = (f['Dec'][sel] - mDec) * 3600.
        std = np.sqrt(mad_std(DR) * mad_std(DD))
        if std < min_std:
            min_std = std
            keep_rot = rot
    log.info('%s_%07d Rotation offset and spread: %0.2f, %0.2f' %(args.date,
                                                      args.observation,
                                                      keep_rot - rot_i,
                                                      min_std))
    A.rot = keep_rot
    A.tp = A.setup_TP(A.ra0, A.dec0, keep_rot, A.x0,  A.y0)
    mRA, mDec = A.tp.wcs_pix2world(f['fx'][sel], f['fy'][sel], 1)
    DR = (f['RA'][sel] - mRA) * np.cos(np.deg2rad(A.dec0)) * 3600.
    DD = (f['Dec'][sel] - mDec) * 3600.
    raoff = np.median(DR)
    decoff = np.median(DD)

    RA0 = A.ra0 + raoff / np.cos(np.deg2rad(A.dec0)) / 3600.
    Dec0 = A.dec0 + decoff / 3600.
    dR = np.cos(np.deg2rad(Dec0)) * 3600. * (ra0 - RA0)
    dD = 3600. * (dec0 - Dec0)
    A.tp = A.setup_TP(RA0, Dec0, keep_rot, A.x0,  A.y0)
    A.rot = keep_rot

    log.info('%s_%07d offsets: %0.2f, %0.2f, %0.2f' %(args.date,
                                                      args.observation,
                                                      dR, dD,
                                                      A.rot-rot_i))
    A.get_pa()
    A1 = Astrometry(RA0, Dec0, A.pa, 0., 0., fplane_file=args.fplane_file)
    A1.tp = A1.setup_TP(RA0, Dec0, A1.rot, A1.x0,  A1.y0)
    return A1


def cofes_plots(ifunums, specnums, filename_array, outfile_name, fF, vmin=-0.25,
                vmax=2.5):
    """
    filename_array is an array-like object that contains the filenames
    of fits files to plot. The output plot will be the shape of the input array.
    
    outfile is the output file name including the extension, such as out.fits.
    
    vmin and vmax set the stretch for the images. They are in units of counts;
    Pixels with vmin and below counts are black. Pixels with vmax and above counts
    are white. 
    """
    rows=10
    cols=10
    cmap = plt.get_cmap('Greys')
    from astropy.visualization.stretch import AsinhStretch
    from astropy.visualization import ImageNormalize
    stretch = AsinhStretch()
    norm = ImageNormalize(stretch=stretch, vmin=vmin, vmax=vmax)
    fig = plt.figure(figsize=(8, 8))
    for i in np.arange(10):
        for j in np.arange(1,11):
            ifuname='%02d%01d' % (j,i)
            index = [ind for ind, v in enumerate(ifunums) if ifuname==v]
            ax = plt.subplot(rows, cols, i*cols+j)
            ax.set_xticks([])
            ax.set_yticks([])
            if ((i==0 and j==1) or (i==9 and j==1) or
                (i==0 and j==10) or (i==9 and j==10)):
                ax.axis('off')
            if len(index):
                f = filename_array[index[0]]
                ax.set_xlim([0, 65])
                ax.set_ylim([0, 65])
                try:
                    data = f * 1.
                    data[np.isnan(data)] = 0.0
                    data[(data<vmin) * (data>vmax)] = 0.0
                    sel = fF['ifuslot'] == int(ifuname)
                    ax.imshow(data, vmin=vmin, vmax=vmax,
                              interpolation='nearest', origin='lower',
                              cmap=cmap, norm=norm)
                    #ax.scatter(fF['imagex'][sel], fF['imagey'][sel], color='g',
                    #           marker='x', s=8)
                    #ax.text(32.5, 32.5, 'V' + specnums[index[0]],
                    #        horizontalalignment='center',
                    #        verticalalignment='center', color='firebrick',
                    #        fontsize=10)
            
                except:
                    ax.imshow(np.zeros((65, 65)),
                              interpolation='nearest', origin='lower',
                              cmap=cmap, norm=norm)
    for i in np.arange(10):
        for j in np.arange(1, 11):
            if i == 9:
                name = '%02d' % j
                ax = plt.subplot(rows, cols, i*cols+j)
                ax.set_xticks([])
                ax.set_yticks([])
                if ((i==0 and j==1) or (i==9 and j==1) or
                    (i==0 and j==10) or (i==9 and j==10)):
                    ax.axis('off')
                pos1 = ax.get_position()
                plt.text(pos1.x0+0.015, pos1.y0-0.01, name,
                         transform=fig.transFigure,
                         horizontalalignment='left',
                         verticalalignment='top',
                         weight='bold')
            if (((i==0 or i==1 or i==2 or i==7 or i==8 or i==9) and j==1) or
                    ((i==0 or i==1 or i==2 or i==7 or i==8 or i==9) and j==10) or
                    ((i==0 or i==9) and j==2) or 
                    ((i==0 or i==9) and j==9) or
                    ((i==4 or i==5 or i==6) and j==5) or
                    ((i==4 or i==5 or i==6) and j==6)):
                ax = plt.subplot(rows, cols, i*cols+j)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.axis('off')
            if j == 1:
                name = '%01d' % i
                ax = plt.subplot(rows, cols, i*cols+j)
                ax.set_xticks([])
                ax.set_yticks([])
                
                pos1 = ax.get_position()
                plt.text(pos1.x0-0.01, pos1.y0+0.015, name,
                         transform=fig.transFigure,
                         horizontalalignment='right',
                         verticalalignment='bottom',
                         weight='bold')
                
       

    plt.subplots_adjust(wspace=0.025, hspace=0.025)    
    fig.savefig(outfile_name, dpi=300)


def advanced_analysis(tfile, fn, scispectra, allifus, pos, A, scale, ran,
                      coords, nexp, gC):
    '''
    Parameters
    ----------
    tfile : str
        Tarfile name
    fn : str
        File name
    scispectra : 2d numpy array
        Sky subtracted spectra
    allifus: 1d numpy array
        Array of all ifus
    pos : 2d numpy array
        IFU x and y positions for all fibers
    A : Astrometry Object
        initial astrometric solution from header
    scale : float
        pixel scale of the g-band image
    ran : list
        min(x), max(x), min(y), max(y) for image
    coords : SkyCoord Object
        coordinates of archival catalog
    Returns
    -------
    '''
    # Make g-band image and find sources
    Total_sources = []
    info = []
    # Image scale and range
    scale = 0.75
    ran = [-23., 25., -23., 25.]
    if nexp == 3:
        seeing = 1.76
    else:
        seeing = 3.0
    
    for i, ui in enumerate(allifus):
        # Get the info for the given ifuslot
        N = 448 * nexp
        data = scispectra[N*i:(i+1)*N]
        M = ~np.isfinite(data)
        P = pos[N*i:(i+1)*N]
        name = '%s_%07d_%03d.fits' % (args.date, args.observation, ui)
        
        # Make g-band image
        image = make_photometric_image(P[:, 0], P[:, 1], data, filtg,
                                       M, ADRx, 0.*ADRx, nchunks=11,
                                       ran=ran,  scale=scale, seeing=seeing)
        # Make full fits file with wcs info (using default header)
        mean, median, std = sigma_clipped_stats(image, sigma=3.0, stdfunc=np.std)
        daofind = DAOStarFinder(fwhm=4.0, threshold=7. * std, exclude_border=True) 
        try:
            sources = daofind(image)
        except:
            log.warning('DAOStarFinder failed')
            sources = None
        if sources is None:
            sources = []
        log.info('Found %i sources in %03d' % (len(sources), ui))
        
        # Keep certain info for new loops
        info.append([image, fn, name, ui, tfile, sources])
        
        # Match sources to archive catalog
        Sources = match_to_archive(sources, image, A, ui, scale, ran, coords,
                                   gC)
        if Sources is not None:
            Total_sources.append(Sources) 
    
    # Write temporary info out
    if len(Total_sources) > 0:
        Total_sources = np.vstack(Total_sources)
        
    f = Table(Total_sources, names = ['imagex', 'imagey', 'gmag', 'dist',
                                      'Cgmag', 'RA', 'Dec', 'fx', 'fy', 'dra',
                                      'ddec', 'ifuslot'])
    # Fit astrometric offset
    for j in np.arange(1):
        A = fit_astrometry(f, A)
        Total_sources = []
        for i, ui in enumerate(allifus):
            image, fn, name, ui, tfile, sources = info[i]
            Sources = match_to_archive(sources, image, A, ui, scale, ran, coords,
                                       gC)
            if Sources is not None:
                Total_sources.append(Sources) 
        Total_sources = np.vstack(Total_sources)
        f = Table(Total_sources, names = ['imagex', 'imagey', 'gmag', 'dist',
                                          'Cgmag', 'RA', 'Dec', 'fx', 'fy', 'dra',
                                          'ddec', 'ifuslot'])
        # f.write('sources.dat', format='ascii.fixed_width_two_line', overwrite=True)

    return f, Total_sources, info, A

def plot_astrometry(f, A, sel, colors):
    log.info('Number of sources for astrometry": %i' % len(sel))
    mRA, mDec = A.tp.wcs_pix2world(f['fx'][sel], f['fy'][sel], 1)
    plt.figure(figsize=(6, 6))
    plt.gca().set_position([0.2, 0.2, 0.65, 0.65])
    dr = np.cos(np.deg2rad(f['Dec'][sel])) * -3600. * (f['RA'][sel] - mRA)
    dd = 3600. * (f['Dec'][sel] - mDec)
    mean, medianr, stdr = sigma_clipped_stats(dr, stdfunc=np.std)
    mean, mediand, stdd = sigma_clipped_stats(dd, stdfunc=np.std)
    t = np.linspace(0, np.pi * 2., 361)
    cx = np.cos(t) * stdr + medianr
    cy = np.sin(t) * stdd + mediand
    plt.scatter(dr, dd, color='r', alpha=0.45, s=45, zorder=3)
    plt.text(-1.2, 1.2, r'$\Delta$ RA (") = %0.2f +/ %0.2f' % (medianr, stdr))
    plt.text(-1.2, 1.05, r'$\Delta$ Dec (") = %0.2f +/ %0.2f' % (mediand, stdd))
    plt.plot(cx, cy, 'k--', lw=2)
    plt.plot([0, 0], [-1.5, 1.5], 'k-', lw=1, alpha=0.5, zorder=1)
    plt.plot([-1.5, 1.5], [0, 0], 'k-', lw=1, alpha=0.5, zorder=1)
    plt.axis([-1.5, 1.5, -1.5, 1.5])
    plt.xlabel(r'$\Delta$ RA (")', fontsize=16, labelpad=10)
    plt.ylabel(r'$\Delta$ Dec (")', fontsize=16, labelpad=10)
    plt.gca().tick_params(axis='both', which='both', direction='in')
    plt.gca().tick_params(axis='y', which='both', left=True, right=True)
    plt.gca().tick_params(axis='x', which='both', bottom=True, top=True)
    plt.gca().tick_params(axis='both', which='major', length=15, width=3)
    plt.gca().tick_params(axis='both', which='minor', length=6, width=2)
    ML = MultipleLocator(0.5)
    ml = MultipleLocator(0.1)
    MLy = MultipleLocator(0.5)
    mly = MultipleLocator(0.1)
    plt.gca().xaxis.set_major_locator(ML)
    plt.gca().yaxis.set_major_locator(MLy)
    plt.gca().xaxis.set_minor_locator(ml)
    plt.gca().yaxis.set_minor_locator(mly)
    plt.savefig('astrometry_%s_%07d.png'  % (args.date, args.observation), dpi=300)
    return medianr, mediand, stdr, stdd, len(dr)

def plot_photometry(GMag, stats, vmin=1., vmax=4., fwhm_guider=1.8,
                    fwhm_virus=None):
    plt.figure(figsize=(7, 6))
    guess_sel = ((np.abs(GMag[:, 0] - 17.) < 2.) *
                 (np.abs(stats[:, 1]-fwhm_guider)<0.2))
    if guess_sel.sum() > 3:
        isel = guess_sel
    else:
        isel = np.ones(stats[:, 0].shape, dtype=bool)
    for i in np.arange(5):
        log.info('Number of sources for photometric modelling: %i' % isel.sum())
        mean, median, std = sigma_clipped_stats(stats[isel, 1], stdfunc=np.nanstd)
        fwhm_virus = median
        std = np.max([std, 0.02])
        log.info('The mean, median, and std for the best seeing for %s_%07d: '
                 '%0.2f, %0.2f, %0.2f' % (args.date, args.observation, mean,
                                          median, std))
        sel = ((GMag[:, 0] < 20.) * (stats[:, 0] < 5.) *
               (np.abs((stats[:, 1]-median)) < 2 * std))
        mean, median, std = sigma_clipped_stats((GMag[sel, 0] - GMag[sel, 1]),
                                                stdfunc=np.nanstd)
        std = np.max([std, 0.02])
        log.info('The mean, median, and std for the mag offset is for %s_%07d: '
                 '%0.2f, %0.2f, %0.2f' % (args.date, args.observation, mean,
                                          median, std))
        isel = (np.abs(GMag[:, 0] - GMag[:, 1] - median) < 2 * std ) * sel
    plt.gca().set_position([0.2, 0.2, 0.65, 0.65])
    plt.gca().tick_params(axis='both', which='both', direction='in')
    plt.gca().tick_params(axis='y', which='both', left=True, right=True)
    plt.gca().tick_params(axis='x', which='both', bottom=True, top=True)
    plt.gca().tick_params(axis='both', which='major', length=15, width=3)
    plt.gca().tick_params(axis='both', which='minor', length=6, width=2)
    ML = MultipleLocator(2)
    ml = MultipleLocator(0.5)
    MLy = MultipleLocator(0.2)
    mly = MultipleLocator(0.05)
    plt.gca().xaxis.set_major_locator(ML)
    plt.gca().yaxis.set_major_locator(MLy)
    plt.gca().xaxis.set_minor_locator(ml)
    plt.gca().yaxis.set_minor_locator(mly)
    cmap = matplotlib.cm.get_cmap('BrBG')
    newcolors = cmap(np.linspace(0, 1, 256))
    for i in np.arange(3):
        newcolors[:, i] = 1. - newcolors[:, i]
    newcmp = ListedColormap(newcolors)
    plt.scatter(GMag[:, 0], GMag[:, 0] - GMag[:, 1] - median, c=stats[:, 1],
                alpha=0.75, s=75, zorder=3, vmin=vmin, vmax=vmax,
                cmap=newcmp)
    cbar = plt.colorbar()
    cbar.set_label('FWHM', rotation=270, labelpad=20)
    plt.plot([13, 22.5], [0, 0], 'k-', lw=1, alpha=0.5, zorder=1)
    plt.plot([13, 22.5], [std, std], 'r--', lw=1)
    plt.plot([13, 22.5], [-std, -std], 'r--', lw=1)
    plt.xlim([13, 22.5])
    plt.ylim([-0.5, 0.5])
    plt.text(13.5, 0.41, 'Magnitude Offset Cor: %0.2f' % median)
    plt.text(13.5, 0.355, 'Magnitude Std: %0.2f' % std)
    plt.text(13.5, 0.3, 'FWHM of Guiders, VIRUS: %0.2f,  %0.2f' %
             (fwhm_guider, fwhm_virus))
    plt.xlabel("Pan-STARRS g' (AB mag)", fontsize=16, labelpad=10)
    plt.ylabel("Pan-STARRS g' - VIRUS g' + Cor", fontsize=16, labelpad=10)
    plt.savefig('mag_offset_%s_%07d.png'  % (args.date, args.observation), dpi=300)
    return sel, GMag[:, 0] - GMag[:, 1] - median, 10**(-0.4*median), fwhm_virus, isel.sum()

def get_sky_fibers(norm_array, pos=None):
    if pos is not None:
        dra = np.cos(np.deg2rad(53.34)) * (pos[:, 0] - 210.8) * 60.
        ddec = (pos[:, 1] - 53.34) * 60.
        d = np.sqrt(dra**2 + ddec**2) > 6.
        norm_array = norm_array[d]
    bl, bm = biweight(norm_array, calc_std=True)
    low = np.nanpercentile(norm_array, 5)
    high = np.nanpercentile(norm_array, 60)
    per_array = np.linspace(low, high, 81)
    Neighbors = np.sum(np.abs(norm_array[:, np.newaxis] - per_array) < bm,
                       axis=0)
    ind = np.argmax(Neighbors)
    nsky = per_array[ind]
    return (norm_array-nsky) < (2 * bm)

def get_skysub(S, sky):
    goodfibers = np.isnan(S).sum(axis=1) < 200
    if goodfibers.sum() == 0:
        log.warning('No good fibers for sky subtraction')
        skysub = np.nan * S
        totsky = np.nan * S
        return skysub, totsky
    dummy = S - sky
    y = biweight(dummy[:, 400:600], axis=1)
    skyfibers = get_sky_fibers(y)
    backfibers = skyfibers * goodfibers
    log.info('Number of good fibers for sky subtraction for Round 1: %i' %
             (backfibers).sum())
    hl = np.nanpercentile(dummy, 98, axis=0)
    ll = np.nanpercentile(dummy, 2, axis=0)
    dummy[(dummy<ll[np.newaxis, :]) + (dummy>hl[np.newaxis, :])] = np.nan
    dummy[~skyfibers] = np.nan
    G = Gaussian2DKernel(7.)
    for i in np.arange(4):
        if backfibers[i*112:(i+1)*112].sum() > 15:
            dummy[i*112:(i+1)*112] = convolve(dummy[i*112:(i+1)*112], G,
                                              boundary='extend')
    G1 = Gaussian1DKernel(7.)
    intermediate = S - sky - dummy
    hl = np.nanpercentile(intermediate, 98, axis=0)
    ll = np.nanpercentile(intermediate, 2, axis=0)
    y = biweight(intermediate[:, 400:600], axis=1)
    skyfibers = get_sky_fibers(y)
    backfibers = skyfibers * goodfibers
    log.info('Number of good fibers for sky subtraction for Round 2: %i' %
             (backfibers).sum())
    intermediate[(intermediate<ll[np.newaxis,:]) + (intermediate>hl[np.newaxis,:])] = np.nan
    intermediate[~skyfibers] = np.nan
    for i in np.arange(4):
        if backfibers[i*112:(i+1)*112].sum() > 15:
            intermediate[i*112:(i+1)*112] = convolve(intermediate[i*112:(i+1)*112], G,
                                              boundary='extend')
    orig = intermediate * 1.
    skysub =  S - sky - dummy - intermediate
    totsky = sky + dummy + intermediate
    log.info('Sky Subtraction Successful')
#    for k in np.arange(S.shape[1]):
#        intermediate[:, k] = interpolate_replace_nans(intermediate[:, k], G1)
#    good_cols = np.isnan(intermediate).sum(axis=0) < 1.
#    if good_cols.sum() > 60:
#        pca = PCA(n_components=15)
#        pca.fit_transform(intermediate[:, good_cols].swapaxes(0, 1))
#        res = get_residual_map(orig, pca)
#        res = 0.
#        skysub = S[goodfibers] - sky - dummy - res
#        bl, bm = biweight(skysub, calc_std=True)
#        skysub[skysub < (-4. * bm)] = np.nan
#        totsky = sky + dummy + res
#        skysub1 = np.nan * S
#        skysub1[goodfibers] = skysub
#        totsky1 = np.nan * S
#        totsky1[goodfibers] = totsky
#        skysub = skysub1
#        totsky = totsky1
#        log.info('successful skysub')
#    else:
#        log.warning('Not enough cols for skysub')
#        skysub = np.nan * S
#        totsky = np.nan * S
    return skysub, totsky

def get_residual_map(data, pca):
    res = data * 0.
    for i in np.arange(data.shape[1]):
        sel = np.isfinite(data[:, i])
        if sel.sum() > data.shape[0]/2.:
            coeff = np.dot(data[sel, i], pca.components_.T[sel])
            model = np.dot(coeff, pca.components_)
        else: 
            model = np.zeros((data.shape[0],))
        res[:, i] = model
    return res

def get_amp_norm_ftf(sci, ftf, nexp, nchunks=9):
    K = sci * 1.
    inds = np.arange(sci.shape[0])
    for k in np.arange(nexp):
        sel = np.where(np.array(inds / 112, dtype=int) % nexp == k)[0]
        skyfibers = get_sky_fibers(biweight(sci[sel, 800:900] /
                                   ftf[sel, 800:900], axis=1))
        sky = biweight((sci[sel] / ftf[sel])[skyfibers], axis=0)
        K[sel] = K[sel] / sky[np.newaxis, :]
    namps = int(K.shape[0] / (nexp*112))
    adj = np.zeros((sci.shape[0], nchunks))
    inds = np.arange(sci.shape[1])
    X = np.array([np.mean(xi) for xi in np.array_split(inds, nchunks)])
    for i in np.arange(namps):
        i1 = int(i * nexp*112)
        i2 = int((i + 1) * nexp*112)
        cnt = 0
        for schunk, fchunk in zip(np.array_split(K[i1:i2], nchunks, axis=1),
                                  np.array_split(ftf[i1:i2], nchunks, axis=1)):
            z = biweight(schunk / fchunk, axis=1)
            b = []
            for j in np.arange(nexp):
                j1 = int(j * 112)
                j2 = int((j+1) * 112)
                b.append(z[j1:j2])
            avg = np.median(b, axis=0)
            norms = [biweight(bi / avg) for bi in b]
            avg = np.median([bi / norm for bi, norm in zip(b, norms)], axis=0)
            norms = [biweight(bi / avg) for bi in b]
            if np.isfinite(avg).sum():
                mask, cont = identify_sky_pixels(avg, 5)
            else:
                cont = np.ones(avg.shape)
            for j in np.arange(nexp, dtype=int):
                 j1 = int(j * 112)
                 j2 = int((j+1) * 112)
                 adj[(i1+j1):(i1+j2), cnt] = norms[j] * cont
            cnt += 1
    Adj = ftf * 0.
    for i in np.arange(sci.shape[0]):
        good = np.isfinite(adj[i])
        if good.sum() > (nchunks-3):
            Adj[i] = interp1d(X, adj[i], kind='quadratic',
                              fill_value='extrapolate')(inds)
        else:
            Adj[i] = 0
    
    return Adj

def catch_ifuslot_swap(ifuslot, date):
    date = int(date)
    if (date > 20191201) * (date < 20200624):
        if ifuslot == 24:
            return 15
        if ifuslot == 15:
            return 24
    if (date > 20200114) * (date < 20200128):
        if ifuslot == 78:
            return 79
    return ifuslot

# =============================================================================
# MAIN SCRIPT
# =============================================================================

# =============================================================================
# Setup logging
# =============================================================================
log = setup_logging()
if args.hdf5file is None:
    log.error('Please specify an hdf5file.  A default is not yet setup.')
    sys.exit(1)

# =============================================================================
# H5 File 
# =============================================================================
name = ('%s_%07d.h5' %
                (args.date, args.observation))
if op.exists(name):
    os.remove(name)
    log.info('%s removed before writing new one' % name)
h5spec = open_file(name, mode="w", title="%s Spectra" % name[:-3])

class Fibers(IsDescription):
     spectrum  = Float32Col((1036,))    # float  (single-precision)
     error  = Float32Col((1036,))    # float  (single-precision)
     fiber_to_fiber  = Float32Col((1036,))    # float  (single-precision)
     residual = Float32Col((1032,))
     skyspectrum = Float32Col((1036,))
     
     #fiber_to_fiber_adj  = Float32Col((1036,))    # float  (single-precision)     

class Cals(IsDescription):
     trace  = Float32Col((1032,))    # float  (single-precision)
     wavelength  = Float32Col((1032,))    # float  (single-precision)
     observed  = Float32Col((1036,))
     observed_error = Float32Col((1036,))
     plawspec = Float32Col((1036,))
     mdarkspec = Float32Col((1036,))
     
class Images(IsDescription):
     zipcode  = StringCol(21)
     #mastersci  = Float32Col((1032,1032))    # float  (single-precision)
     #image  = Float32Col((1032,1032))    # float  (single-precision)

class Info(IsDescription):
     ra  = Float32Col()    # float  (single-precision)
     dec  = Float32Col()    # float  (single-precision)
     ifux  = Float32Col()    # float  (single-precision)
     ifuy  = Float32Col()    # float  (single-precision)
     specid  = Int32Col()    # float  (single-precision)
     ifuslot  = Int32Col()    # float  (single-precision)
     ifuid  = Int32Col()   # float  (single-precision)
     amp  = StringCol(2)
     exp = Int32Col()

class Survey(IsDescription):
     ra  = Float32Col()    # float  (single-precision)
     dec  = Float32Col()    # float  (single-precision)
     pa  = Float32Col()    # float  (single-precision)
     rot  = Float32Col()    # float  (single-precision)
     millum  = Float32Col()    # float  (single-precision)
     throughput  = Float32Col()    # float  (single-precision)
     fwhm  = Float32Col()   # float  (single-precision)
     fwhmv  = Float32Col()   # float  (single-precision)
     name  = StringCol(25)
     exp = Int32Col()
     offset  = Float32Col()    # float  (single-precision)
     nstarsastrom = Int32Col()
     nstarsphotom = Int32Col()
     raoffset = Float32Col()
     decoffset = Float32Col()
     rastd = Float32Col()
     decstd = Float32Col()
     exptime = Float32Col()
     
class Detections(IsDescription):
     ra  = Float32Col()    # float  (single-precision)
     dec  = Float32Col()    # float  (single-precision)
     ifux  = Float32Col()    # float  (single-precision)
     ifuy  = Float32Col()    # float  (single-precision)
     fx  = Float32Col()    # float  (single-precision)
     fy  = Float32Col()    # float  (single-precision)
     specid  = Int32Col()    # float  (single-precision)
     ifuslot  = Int32Col()    # float  (single-precision)
     ifuid  = Int32Col()   # float  (single-precision)
     wave  = Float32Col()    # float  (single-precision)
     flux  = Float32Col()    # float  (single-precision)
     sn  = Float32Col()    # float  (single-precision)
     chi2  = Float32Col()    # float  (single-precision)
     fwhm  = Float32Col()    # float  (single-precision)
     spectrum  = Float32Col((1036,))    # float  (single-precision)
     error  = Float32Col((1036,))     # float  (single-precision)
     model  = Float32Col((1036,))     # float  (single-precision)

class CatSpectra(IsDescription):
     ra  = Float32Col()    # float  (single-precision)
     dec  = Float32Col()    # float  (single-precision)
     spectrum  = Float32Col((1036,))    # float  (single-precision)
     error  = Float32Col((1036,))    # float  (single-precision)
     weight  = Float32Col((1036,))    # float  (single-precision)
     gmag  = Float32Col()    # float  (single-precision)
     rmag  = Float32Col()    # float  (single-precision)
     imag  = Float32Col()    # float  (single-precision)
     zmag  = Float32Col()    # float  (single-precision)
     ymag  = Float32Col()    # float  (single-precision)

class MatSpectra(IsDescription):
     ra  = Float32Col()    # float  (single-precision)
     dec  = Float32Col()    # float  (single-precision)
     fx  = Float32Col()    # float  (single-precision)
     fy  = Float32Col()    # float  (single-precision)
     catfx  = Float32Col()    # float  (single-precision)
     catfy  = Float32Col()    # float  (single-precision)
     gmag  = Float32Col()    # float  (single-precision)
     syngmag  = Float32Col()    # float  (single-precision)
     spectrum  = Float32Col((1036,))    # float  (single-precision)
     error  = Float32Col((1036,))    # float  (single-precision)
     weight  = Float32Col((1036,))    # float  (single-precision)
     image = Float32Col((11, 21, 21))
     xgrid = Float32Col((21, 21))
     ygrid = Float32Col((21, 21))

# =============================================================================
# Get directory name for path building
# =============================================================================
DIRNAME = get_script_path()

# =============================================================================
# Instrument for reductions
# =============================================================================
instrument = 'virus'

# =============================================================================
# Open HDF5 file
# =============================================================================
h5file = open_file(args.hdf5file, mode='r')
h5table = h5file.root.Cals
ifuslots = h5table.cols.ifuslot[:]
specids = [x.decode("utf-8") for x in h5table.cols.specid[:]]
ifuids = [x.decode("utf-8") for x in h5table.cols.ifuid[:]]
zips = np.array(['%s_%03d_%s' % (s,i1,i2) for s, i1, i2 in zip(specids, ifuslots, ifuids)])
uzips = np.unique(zips)
# =============================================================================
# Get unique IFU slot names
# =============================================================================
u_ifuslots = np.unique(ifuslots)

# =============================================================================
# Get photometric filter
# =============================================================================
T = Table.read(op.join(DIRNAME, 'filters/ps1g.dat'), format='ascii')
filtg = np.interp(def_wave, T['col1'], T['col2'], left=0.0, right=0.0)
filtg /= filtg.sum()

# =============================================================================
# Calibration factors to convert electrons to uJy
# =============================================================================
mult_fac = 6.626e-27 * (3e18 / def_wave) / 360. / 5e5 / 0.92 * 5
mult_fac *= 1e29 * def_wave**2 / 2.99792e18

# =============================================================================
# Read standard throughput
# =============================================================================
T = Table.read(op.join(DIRNAME, 'CALS/throughput.txt'),
                       format='ascii.fixed_width_two_line')
throughput = np.array(T['throughput'])
        
# =============================================================================
# Collect indices for ifuslots (target and neighbors)
# =============================================================================

ZIPs = get_ifuslots()
allzips = []
sel1 = []
for z in uzips:
    if z not in ZIPs:
        continue
    sel1.append(np.where(z == zips)[0][:4])
    allzips.append(z)

ziploop = np.sort(np.array(np.hstack(sel1), dtype=int))
nslots = len(ziploop) / 4
allifus = ifuslots[ziploop[::4]]

# =============================================================================
# Reducing IFUSLOT
# =============================================================================
tableh5 = h5spec.create_table(h5spec.root, 'Images', Images, 
                            "Image Information")
(pos, fltspectra, scispectra, errspectra, wave_all, 
 fns, tfile, _I, C1, intm, ExP, mscispectra, twispectra) = reduce_ifuslot(ziploop, h5table,
                                                              tableh5)

for i in np.arange(len(allifus)):
    allifus[i] = catch_ifuslot_swap(allifus[i], args.date)
    
fn = fns[0]
_I = np.hstack(_I)
ifuslot_i = np.zeros((len(_I),), dtype=int)
for i in np.arange(len(_I)):
    sid, isl, iid, ap = _I[i].split('_')
    ifuslot_i[i] = int(isl)
h5file.close()
h5file = None
process = psutil.Process(os.getpid())
log.info('[MRK] Memory Used: %0.2f GB' % (process.memory_info()[0] / 1e9))

# =============================================================================
# Number of exposures
# =============================================================================
nexp = int(scispectra.shape[0] / 448 / nslots)
log.info('Number of exposures: %i' % nexp)


# =============================================================================
# Get fiber to fiber from twilight spectra
# =============================================================================
log.info('Getting Fiber to Fiber')
ftf, res = get_fiber_to_fiber(fltspectra, mscispectra, wave_all, twispectra)

del twispectra, mscispectra, fltspectra
gc.collect()
process = psutil.Process(os.getpid())
log.info('Memory Used: %0.2f GB' % (process.memory_info()[0] / 1e9))

# =============================================================================
# Masking
# =============================================================================
log.info('Masking bad pixels/fibers/amps')
inds = np.arange(scispectra.shape[0])
mask = get_mask(scispectra, C1, ftf, res, nexp)
mask[:, :-1] += mask[:, 1:]
mask[:, 1:] += mask[:, :-1]
if not args.no_masking:
    scispectra[mask] = np.nan
    errspectra[mask] = np.nan
scispectra = safe_division(scispectra, ftf)
errspectra = safe_division(errspectra, ftf)
scirect = np.zeros((scispectra.shape[0], len(def_wave)))
ftfrect = np.zeros((scispectra.shape[0], len(def_wave)))

log.info('Rectifying Spectra')
indices1 = np.ones(scirect.shape, dtype=int)
for i in np.arange(scispectra.shape[0]):
    scirect[i] = np.interp(def_wave, wave_all[i], scispectra[i], left=np.nan,
                        right=np.nan)
    ftfrect[i] = np.interp(def_wave, wave_all[i], ftf[i], left=np.nan,
                        right=np.nan)
    indices1[i] = np.searchsorted(wave_all[i], def_wave) + i * 1032

x_var = (def_wave[np.newaxis, :] * np.ones((scirect.shape[0],1))).ravel()
x_fix = wave_all.ravel()
indices1 = indices1.ravel()
indices2 = indices1 - 1
indices2[indices2 < 0] = 0
indices1[indices1 >= len(x_fix)] = len(x_fix) - 1

distances1 = np.abs(x_fix[indices1] - x_var)
distances2 = np.abs(x_fix[indices2] - x_var)
total_distance = distances1 + distances2
weight1 = distances1 / total_distance
weight2 = distances2 / total_distance
errorrect = (weight2**2 * errspectra.ravel()[indices1]**2 +
             weight1**2 * errspectra.ravel()[indices2]**2)
errorrect = np.sqrt(errorrect)
errorrect = np.reshape(errorrect, scirect.shape)
errorrect[np.isnan(scirect)] = np.nan

del total_distance, distances1, distances2, indices1, indices2, x_var, x_fix, weight1, weight2
gc.collect()

qual_check = np.abs(scirect)<1e-8
scirect[qual_check] = np.nan
errorrect[qual_check] = np.nan
del qual_check
# =============================================================================
# Sky Subtraction
# =============================================================================
log.info('Subtracting sky for all ifuslots')
skies = []


skysubrect = scirect * 0.
skyrect = scirect * 0.
skyrect_orig = scirect * 0.
for k in np.arange(nexp):
    sel = np.where(np.array(inds / 112, dtype=int) % nexp == k)[0]
    nifus = int(len(sel) / 448.)
    y = biweight(scirect[sel, 800:900], axis=1)
    skyfibers = get_sky_fibers(y, pos[sel])
    sky = biweight(scirect[sel][skyfibers], axis=0)
    skies.append(sky)
    for j in np.arange(nifus):
        I = sel[int(j*448):int((j+1)*448)]
        if args.quick_sky:
            skysub = scirect[I] - sky
            totsky = sky
        else:
            skysub, totsky = get_skysub(scirect[I], sky)
        skysubrect[I] = skysub
        skyrect[I] = totsky
        skyrect_orig[I] = sky

scispectra = skysubrect * 1.
errspectra = errorrect * 1.
del errorrect, skysubrect
gc.collect()
process = psutil.Process(os.getpid())
log.info('Memory Used: %0.2f GB' % (process.memory_info()[0] / 1e9))    

# =============================================================================
# Calibration
# =============================================================================

fac = 360. / ExP
fac[ExP==0.] = 1.
mult_fac2 = mult_fac[np.newaxis, :] * fac[:, np.newaxis]
scispectra[:] = scispectra / throughput[np.newaxis, :] * mult_fac2
errspectra[:] = errspectra / throughput[np.newaxis, :] * mult_fac2
skyrect[:] = skyrect / throughput[np.newaxis, :] * mult_fac2
# =============================================================================
# Make 2d sky-sub image 
# =============================================================================
#obsskies = []
#for k in np.arange(nexp):
#    sel = np.where(np.array(inds / 112, dtype=int) % nexp == k)[0]
#    y = s1[sel] / ftf[sel] / Adj[sel]
#    y[mask[sel]] = np.nan
#    sky = biweight(y, axis=0)
#    obsskies.append(sky)
#skysub_images = []
#for k, _V in enumerate(intm):
#    ind = k % 3
#    init = _V[0]
#    image = _V[1]
#    name = 'multi_' + _V[2] + '.fits'
#    N = k * 112
#    M = (k+1) * 112
#    sky = obsskies[ind] * ftf[N:M] * Adj[N:M]
#    sky[~np.isfinite(sky)] = 0.0
#    log.info('Writing model image for %s' % _V[2])
#    if init is not None:
#        model_image = build_model_image(init, image, T1[N:M], W1[N:M], sky,
#                                        def_wave)
#    else:
#        model_image = image * 0.
#    skysub_images.append([image, image-model_image, _V[2], (N+M)/2])

# =============================================================================
# Ifuslot dictionary
# =============================================================================
ui_dict = {}
for k, _V in enumerate(intm):
    specid, ifuslot, ifuid, amp, expn = _V[2].split('_')
    ifusl = catch_ifuslot_swap(int(ifuslot), args.date)
    ifuslot = '%03d' % ifusl
    if ifuslot not in ui_dict:
        ui_dict[ifuslot] = [specid, ifuid]

# =============================================================================
# Correct error to sigma = 1., error typically too large by 25%
# =============================================================================
namps = int(len(scispectra) / 112. / nexp)
for i in np.arange(namps):
    _V = intm[int(i*nexp)]
    specid, ifuslot, ifuid, amp, expn = _V[2].split('_')
    ifusl = catch_ifuslot_swap(int(ifuslot), args.date)
    ifuslot = '%03d' % ifusl
    ll = int(i * (112*nexp))
    hl = int((i+1) * (112*nexp))
    y = biweight(scispectra[ll:hl, 800:900], axis=1)
    skyfibers = get_sky_fibers(y)
    x1 = scispectra[ll:hl]
    x2 = errspectra[ll:hl]
    if skyfibers.sum():
        back_val, error_cor = biweight(x1[skyfibers] / x2[skyfibers], axis=0,
                                       calc_std=True)
        # errspectra[ll:hl] *= error_cor[np.newaxis, :]
        log.info('Average Error Correction for %s%s: %0.2f' %
                 (ifuslot, amp, np.nanmedian(error_cor)))
        v = np.nanmedian(error_cor) - 1.
        if v < -0.3:
            log.info('Error less than 30 percent below expectation so we mask amp %s%s' %
                     (ifuslot, amp))
            errspectra[ll:hl] = np.nan
            scispectra[ll:hl] = np.nan

# =============================================================================
# Calculate ratio of skies from multiple exposures. Normalization factor??
# =============================================================================
exp_ratio_sky = np.ones((nexp,))
for i in np.arange(1, nexp):
    exp_ratio_sky[i] = np.nanmedian(skies[i] / skies[0])
    log.info('Ratio for exposure %i to exposure 1: %0.2f' %
             (i+1, exp_ratio_sky[i]))

# =============================================================================
# Get Normalization
# =============================================================================
gratio = np.ones((nexp,))
gseeing = np.ones((nexp,))
millum = np.ones((nexp,))
transpar = np.ones((nexp,))
EXPN = np.zeros((nexp,))
for i in np.arange(1, nexp+1):
    ind = int((i-1)*112)
    mill, trans, iq = get_mirror_illumination_guider(fns[i-1], ExP[ind])
    EXPN[i-1] = ExP[ind]
    log.info('Mirror Illumination, Transparency, and seeing for exposure %i'
             ': %0.2f, %0.2f, %0.2f' % (i, mill/1e4, trans, iq))
    gratio[i-1] = (mill * trans) /  5e5
    millum[i-1] = mill
    transpar[i-1] = trans
    gseeing[i-1] = iq

for i in np.arange(0, nexp):
    sel = (np.array(inds / 112, dtype=int) % nexp) == i
    scispectra[sel] = scispectra[sel] / gratio[i]
    errspectra[sel] = errspectra[sel] / gratio[i]
    skyrect[sel] = skyrect[sel] / gratio[i]
    log.info('%s_%07d Guider ratio for exposure %i to 50m^2: %0.2f' %
             (args.date, args.observation, i+1, gratio[i]))

# =============================================================================
# Build astrometry class
# =============================================================================
ra, dec, pa = get_ra_dec_from_header(tfile, fn)
A = Astrometry(ra, dec, pa, 0., 0., fplane_file=args.fplane_file)

# =============================================================================
# Query catalog Sources in the area
# =============================================================================
log.info('Querying Pan-STARRS at: %0.5f %0.5f' % (ra, dec))
pname = 'Panstarrs_%0.6f_%0.5f_%0.4f.dat' % (ra, dec, 11. / 60.)
log.info('%s_%07d: %s' % (args.date, args.observation, pname))
if op.exists(pname):
    Pan = Table.read(pname, format='ascii.fixed_width_two_line')
    try:
        raC, decC, gC, rC, iC, zC, yC = (np.array(Pan['raMean']), np.array(Pan['decMean']),
                         np.array(Pan['gMeanApMag']), np.array(Pan['rMeanApMag']),
                         np.array(Pan['iMeanApMag']), np.array(Pan['zMeanApMag']),
                         np.array(Pan['yMeanApMag']))
    except:
        raC, decC, gC, rC, iC, zC, yC = (np.array(Pan['raMean']), np.array(Pan['decMean']),
                         np.array(Pan['gApMag']), np.array(Pan['rApMag']),
                         np.array(Pan['iApMag']), np.array(Pan['zApMag']),
                         np.array(Pan['yApMag']))
else:
    try:
        Pan = Catalogs.query_region("%s %s" % (ra, dec), radius=11./60., 
                                       catalog="Panstarrs", data_release="dr2",
                                       table="stack")
        raC, decC, gC, rC, iC, zC, yC = (np.array(Pan['raMean']), np.array(Pan['decMean']),
                         np.array(Pan['gApMag']), np.array(Pan['rApMag']),
                         np.array(Pan['iApMag']), np.array(Pan['zApMag']),
                         np.array(Pan['yApMag'])) 
        
    except:
        log.info('Panstarrs initial query failed, '
                 'trying with small coord adjustment')
        Pan = Catalogs.query_region("%s %s" % (ra, dec), radius=11./60., 
                                       catalog="Panstarrs")
        raC, decC, gC, rC, iC, zC, yC = (np.array(Pan['raMean']), np.array(Pan['decMean']),
                         np.array(Pan['gMeanApMag']), np.array(Pan['rMeanApMag']),
                         np.array(Pan['iMeanApMag']), np.array(Pan['zMeanApMag']),
                         np.array(Pan['yMeanApMag']))
    Pan.write(pname, format='ascii.fixed_width_two_line')


coords = SkyCoord(raC*units.degree, decC*units.degree, frame='fk5')


# =============================================================================
# Find sources in collapsed frames, match to archive, fit for astrometry
# =============================================================================
scale = 0.75
ran = [-23., 25., -23., 25.]
f, Total_sources, info, A = advanced_analysis(tfile, fn, scispectra, allifus,
                                              pos, A, scale, ran, coords,
                                              nexp, gC)


# =============================================================================
# Get RA and Declination for each Fiber
# =============================================================================
RAFibers, DecFibers = ([], [])
for i, _info in enumerate(info):
    image, fn, name, ui, tfile, sources = _info
    N = 448 * nexp
    data = scispectra[N*i:(i+1)*N]
    P = pos[N*i:(i+1)*N]
    ra, dec = A.get_ifupos_ra_dec('%03d' % ui, P[:, 0], P[:, 1])
    RAFibers.append(ra)
    DecFibers.append(dec)
RAFibers, DecFibers = [np.hstack(x) for x in [RAFibers, DecFibers]]


# Making g-band images
filename_array = []
ifunums = []
specnums = []
for i in info:
    image, fn, name, ui, tfile, sources = i
    crx = np.abs(ran[0]) / scale + 1.
    cry = np.abs(ran[2]) / scale + 1.
    ifuslot = '%03d' % ui
    specnums.append(ui_dict[ifuslot][0])
    ifunums.append(ifuslot)
    filename_array.append(image)
    if ui == args.ifuslot:
        A.get_ifuslot_projection(ifuslot, scale, crx, cry)
        header = A.tp_ifuslot.to_header()
        #F = fits.PrimaryHDU(np.array(image, 'float32'), header=header)
        #F.writeto(name, overwrite=True)
outfile_name = '%s_%07d_recon.png' % (args.date, args.observation)
objsel = (f['Cgmag'] < 22.) * (f['dist'] < 1.5) * (f['Cgmag'] > 14.) 
nobjsel = (f['Cgmag'] < 22.) * (f['Cgmag'] > 14.) 
cofes_plots(ifunums, specnums, filename_array, outfile_name, f[nobjsel])

log.info('%s_%07d Astrometry: %0.6f %0.5f %0.2f' % (args.date, args.observation, A.ra0, A.dec0, A.rot))
# =============================================================================
# Extraction of nearby sources
# =============================================================================

E = Extract()
tophat = E.tophat_psf(3., 10.5, 0.1)
moffat = E.moffat_psf(np.median(gseeing), 10.5, 0.1)
newpsf = tophat[0] * moffat[0] / np.max(tophat[0])
#newpsf = tophat[0] / np.sum(tophat[0]) * np.sum(newpsf)
psf = [newpsf, moffat[1], moffat[2]]
E.psf = psf
E.get_ADR_RAdec(A)
# New extraction method using all ra, dec, culled to Rad within one ifu at least
R = []
D = []
GMM = []
pancat = SkyCoord(raC*units.deg, decC*units.deg, frame='fk5')
for i in info:
    image, fn, name, ui, tfile, sources = i
    ifuslot = '%03d' % ui
    raifu, decifu = A.get_ifuslot_ra_dec(ifuslot)
    c = SkyCoord(raifu*units.deg, decifu*units.deg, frame='fk5')
    sep = c.separation(pancat)
    sel = (sep.arcsec < 35.) * (gC < 23.) * (gC > 10.)
    R.append(raC[sel])
    D.append(decC[sel])
    GMM.append([gC[sel], rC[sel], iC[sel], zC[sel], yC[sel]])
R = np.hstack(R)
D = np.hstack(D)
GMM = np.hstack(GMM)
log.info('Extracting %i Bright Sources for PSF' % objsel.sum())

mRA, mDec = (f['RA'][objsel], f['Dec'][objsel])
E.coords = SkyCoord(mRA*units.deg, mDec*units.deg, frame='fk5')
GMag = np.ones((len(E.coords), 2)) * np.nan
GMag[:, 0] = f['Cgmag'][objsel]
pFX, pFY = A.tp.wcs_world2pix(f['RA'][objsel], f['Dec'][objsel], 1)
FX = f['fx'][objsel]
FY = f['fy'][objsel]
E.ra, E.dec = (RAFibers, DecFibers)
E.data = scispectra
E.error = errspectra
E.mask = np.isfinite(scispectra)
E.ifuslot_i = ifuslot_i
spec_list = []
for i in np.arange(len(E.coords)):
    specinfo = E.get_spectrum_by_coord_index(i)
    gmask = np.isfinite(specinfo[0]) * (specinfo[2] > (0.7*np.nanmedian(specinfo[2])))
    nu = 2.99792e18 / def_wave
    dnu = np.diff(nu)
    dnu = np.hstack([dnu[0], dnu])
    if gmask.sum() > 50.:
        gmag = np.dot((nu/dnu*specinfo[0])[gmask], filtg[gmask]) / np.sum((nu/dnu*filtg)[gmask])
        GMag[i, 1] = -2.5 * np.log10(gmag) + 23.9
    else:
        GMag[i, 1] = np.nan
    
    spec_list.append([mRA[i], mDec[i], FX[i], FY[i], pFX[i], pFY[i], GMag[i, 0],
                      GMag[i, 1]] + list(specinfo))

E.coords = SkyCoord(R*units.deg, D*units.deg, frame='fk5')
allspec_list = []
log.info('Extracting %i Pan-STARRS Sources' % len(R))
for i in np.arange(len(E.coords)):
    specinfo = E.get_spectrum_by_coord_index(i)
    if len(specinfo[0]) == 0:
        continue
    gmask = np.isfinite(specinfo[0]) * (specinfo[2] > (0.1)) 
    if gmask.sum() > 50.:
        allspec_list.append([R[i], D[i], GMM[0, i], GMM[1, i], GMM[2, i],
                             GMM[3, i], GMM[4, i]] + list(specinfo))
X, Y, Z = [np.zeros((len(spec_list), 11)) for k in np.arange(3)]
nwave = np.array([np.mean(xi) for xi in np.array_split(def_wave, 11)])
seeing_array = np.linspace(np.median(gseeing)-0.8, np.median(gseeing)+0.8, 161)
R = np.linspace(0, 7, 701)
Profile = np.zeros((len(seeing_array), len(R)))
for i, seeing in enumerate(seeing_array):
    Profile[i] = E.moffat_psf_integration(R, 0.*R, seeing)

chi2norm = np.zeros((len(spec_list), len(seeing_array)))
stats = np.zeros((len(spec_list), 2))
for i, ind in enumerate(spec_list):
    im = ind[11]
    for j, ima in enumerate(im):
        x, y = centroid_com(ima)
        X[i, j] = x*0.5 - 5.
        Y[i, j] = y*0.5 - 5.
        Z[i, j] = j
    spec_package = ind[14]
    d = np.sqrt(spec_package[0]**2 + spec_package[1]**2)
    fitsel = d <= 3.5
    goodsel = spec_package[4]
    B = spec_package[2] * 1.
    Ee = spec_package[3] * 1.
    M = np.array(fitsel*goodsel, dtype=float)
    
    for j, seeing in enumerate(seeing_array): 
        model = np.interp(d, R, Profile[j], right=0.0)
        norm = (np.nansum(model * B / Ee**2 * M, axis=0) /
                np.nansum(model**2 * M / Ee**2, axis=0))
        EP = np.sqrt(Ee**2 + (0.05*model*norm[np.newaxis, :])**2)
        chi2 = np.nansum((B - model*norm[np.newaxis, :])**2 / EP**2 * M)
        chi2norm[i, j] = chi2 / (M.sum())
    best_seeing = seeing_array[np.argmin(chi2norm[i])]
    log.info('Best seeing for source %i: %0.2f, chi2=%0.2f' %
             (i+1, best_seeing, np.min(chi2norm[i])))
    stats[i, 0] = np.min(chi2norm[i])
    stats[i, 1] = best_seeing

E.ADRra = E.ADRra + np.interp(def_wave, nwave,
                              np.median(X, axis=0)-np.median(X))
E.ADRdec = E.ADRdec + np.interp(def_wave, nwave,
                                np.median(Y, axis=0)-np.median(Y))

othersel, colors, offset, fwhm_virus, Nphotom = plot_photometry(GMag, stats, vmin=seeing_array.min(),
                                   vmax=seeing_array.max(),
                                   fwhm_guider=np.median(gseeing))

raoffc, decoffc, raoffs, decoffs, Nastrom = plot_astrometry(f, A, np.where(objsel)[0], colors)

norm = 1e-29 * 2.99792e18 / def_wave**2 * 1e17
skyrect[:] *= norm
if np.isfinite(offset):
    log.info('Offset multiplication: %0.2f' % offset)
    norm *= offset
scispectra[:] *= norm
errspectra[:] *= norm

# =============================================================================
# Detections
# =============================================================================
detect_info = []

if not args.no_detect:
    Ed = Extract()
    tophat = Ed.tophat_psf(3., 10.5, 0.1)
    moffat = Ed.moffat_psf(fwhm_virus, 10.5, 0.1)
    newpsf = tophat[0] * moffat[0] / np.max(tophat[0])
    psf = [newpsf, moffat[1], moffat[2]]
    for i, ui in enumerate(allifus): 
        ifuslot = '%03d' % ui
        log.info('Detections for %s' % ifuslot)
        N = 448 * nexp
        data = scispectra[N*i:(i+1)*N]
        error = errspectra[N*i:(i+1)*N]
        M = np.isnan(data)
        P = pos[N*i:(i+1)*N]
        ifu = A.fplane.by_ifuslot('%03d' % ui)
        ifux, ifuy = (ifu.y, ifu.x) 
        sources = detect_sources(P[:, 0], P[:, 1], data, error, M, def_wave, psf,
                                 ran, scale, log, spec_res=5.6, thresh=5.)
        if ifuslot == '047':
            fits.PrimaryHDU(sources[0]/sources[1]).writeto('test_047.fits', overwrite=True)
        for l, k in zip(sources[4], sources[5]):
            fx, fy = (l[0] + ifux,
                      l[1] + ifuy)
            sra, sdec = A.tp.wcs_pix2world(fx, fy, 1)
            if np.all(k[:, 0] == 0.):
                continue
            detect_info.append([ui, ui_dict[ifuslot][0], ui_dict[ifuslot][1], l[0], l[1], fx,
                                fy, sra, sdec, l[2], l[3], l[4], l[5], l[6],
                                k[:, 0], k[:, 1], k[:, 2]])

table = h5spec.create_table(h5spec.root, 'MatSpectra', MatSpectra, 
                            "Spectral Extraction Information for matches")

specrow = table.row
for specinfo in spec_list:
    specrow['ra'] = specinfo[0]
    specrow['dec'] = specinfo[1]
    specrow['fx'] = specinfo[2]
    specrow['fy'] = specinfo[3]
    specrow['catfx'] = specinfo[4]
    specrow['catfy'] = specinfo[5]
    specrow['gmag'] = specinfo[6]
    specrow['syngmag'] = specinfo[7]
    if len(specinfo[8]) > 0:
        specrow['spectrum'] = specinfo[8] * norm
        specrow['error'] = specinfo[9] * norm
        specrow['weight'] = specinfo[10]
        specrow['image'] = specinfo[11]
        specrow['xgrid'] = specinfo[12]
        specrow['ygrid'] = specinfo[13]
    specrow.append()
table.flush()

table = h5spec.create_table(h5spec.root, 'CatSpectra', CatSpectra, 
                            "Spectral Extraction Information")

specrow = table.row
for specinfo in allspec_list:
    specrow['ra'] = specinfo[0]
    specrow['dec'] = specinfo[1]
    specrow['gmag'] = specinfo[2]
    specrow['rmag'] = specinfo[3]
    specrow['imag'] = specinfo[4]
    specrow['zmag'] = specinfo[5]
    specrow['ymag'] = specinfo[6]

    if len(specinfo[7]) > 0:
        specrow['spectrum'] = specinfo[7] * norm
        specrow['error'] = specinfo[8] * norm
        specrow['weight'] = specinfo[9]
    specrow.append()
table.flush()

log.info('Finished writing spectra')

table = h5spec.create_table(h5spec.root, 'Fibers', Fibers, 
                            "Fiber Information")
specrow = table.row
for i in np.arange(len(scispectra)):
    specrow['spectrum'] = scispectra[i]
    specrow['skyspectrum'] = skyrect[i]
    specrow['error'] = errspectra[i]
    specrow['fiber_to_fiber'] = ftfrect[i]
    specrow['residual'] = res[i]
    specrow.append()
table.flush()

log.info('Finished writing Fibers')

if tfile is not None:
    t = tarfile.open(tfile, 'r')
    a = fits.open(t.extractfile(fn))
    t.close()
else:
    a = fits.open(fn)

he = a[0].header

table = h5spec.create_table(h5spec.root, 'Survey', Survey, 
                            "Survey Information")
specrow = table.row
for i in np.arange(nexp):
    specrow['ra'] = A.ra0
    specrow['dec'] = A.dec0
    specrow['rot'] = A.rot
    specrow['pa'] = pa
    specrow['millum'] = millum[i]
    specrow['throughput'] = transpar[i]    # float  (single-precision)
    specrow['fwhm']  = gseeing[i]   # float  (single-precision)
    specrow['fwhmv']  = fwhm_virus   # float  (single-precision)
    specrow['name']  = he['OBJECT'].ljust(25)
    specrow['exp'] = i+1
    specrow['offset'] = offset
    specrow['nstarsphotom'] = Nphotom
    specrow['nstarsastrom'] = Nastrom
    specrow['raoffset'] = raoffc
    specrow['rastd'] = raoffs
    specrow['decoffset'] = decoffc
    specrow['decstd'] = decoffs
    specrow['exptime'] = EXPN[i]
    specrow.append()
table.flush()

log.info('Finished writing Fibers')


table = h5spec.create_table(h5spec.root, 'Info', Info, 
                            "Fiber Information")
specrow = table.row
for i in np.arange(len(E.ra)):
    specrow['ra'] = E.ra[i]
    specrow['dec'] = E.dec[i]
    specrow['ifux'] = pos[i, 0]
    specrow['ifuy'] = pos[i, 1]
    sid, isl, iid, ap = _I[i].split('_')
    specrow['specid'] = int(sid)
    specrow['ifuid'] = int(iid)
    specrow['ifuslot'] = int(isl)
    specrow['amp'] = ap
    specrow.append()
table.flush()

if not args.no_detect:
    table = h5spec.create_table(h5spec.root, 'Detections', Detections, 
                            "Detection Information")
    specrow = table.row
    for dinfo in detect_info:
        specrow['ra'] = dinfo[7]
        specrow['dec'] = dinfo[8]
        specrow['fx'] = dinfo[5]
        specrow['fy'] = dinfo[6]
        specrow['ifux'] = dinfo[3]
        specrow['ifuy'] = dinfo[4]
        specrow['specid'] = dinfo[1]
        specrow['ifuid'] = dinfo[2]
        specrow['ifuslot'] = dinfo[0]
        specrow['wave'] = dinfo[9]
        specrow['fwhm'] = dinfo[10]
        specrow['chi2'] = dinfo[11]
        specrow['flux'] = dinfo[12]
        specrow['sn'] = dinfo[13]
        specrow['spectrum'] = dinfo[14]
        specrow['error'] = dinfo[15]
        specrow['model'] = dinfo[16]
        specrow.append()
    table.flush()

log.info('Finished writing Detections')

#table = h5spec.create_table(h5spec.root, 'Images', Images, 
#                            "Images")
#specrow = table.row
#for i in np.arange(len(skysub_images)):
#    specrow['zipcode'] = skysub_images[i][2]
#    specrow['image'] = skysub_images[i][0]
#    specrow['skysub'] = skysub_images[i][1]
#    specrow.append()
#table.flush()
#
#log.info('Finished writing Images')

process = psutil.Process(os.getpid())
log.info('Memory Used: %0.2f GB' % (process.memory_info()[0] / 1e9))
h5spec.close()
h5spec = None


#with open('ds9_%s_%07d.reg' % (args.date, args.observation), 'w') as k:
#    MakeRegionFile.writeHeader(k)
#    MakeRegionFile.writeSource(k, coords.ra.deg, coords.dec.deg)

#x = [518 for s in skysub_images]
#y = [s[3] for s in skysub_images]
#te = [s[2] for s in skysub_images]
#with open('ds9_%s_%07d_skysub.reg' % (args.date, args.observation), 'w') as k:
#    MakeRegionFile.writeHeader(k)
#    MakeRegionFile.writeText(k, x, y, te)

#if args.simulate:
#    i = 0
#    N = 448 * nexp
#    F = np.nanmedian(ftf[N*i:(i+1)*N], axis=1)
#    data = scispectra[N*i:(i+1)*N]
#    P = pos[N*i:(i+1)*N]
#    log.info('Simulating spectrum from %s' % args.source_file)
#    simulated_spectrum = read_sim(args.source_file)
#    simdata = simulate_source(simulated_spectrum, P, data, 
#                                 args.source_x, args.source_y, 
#                                 args.source_seeing)
#    scispectra[N*i:(i+1)*N] = simdata * 1.
#
#
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
    ecubename = ('%s_%07d_%03d_error_cube.fits' %
                (args.date, args.observation, args.ifuslot))



## Making data cube
if args.make_cube:
    ifuslot_arr = E.ra * 0.
    for i in np.arange(len(E.ra)):
        sid, isl, iid, ap = _I[i].split('_')
        ifuslot_arr[i] = float(isl)
    i1 = np.where(float(args.ifuslot) == ifuslot_arr)[0][0]
    log.info('Making Cube with index %i' %i1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        zgrid, egrid, xgrid, ygrid = make_cube(pos[i1:i1+448*nexp, 0], pos[i1:i1+448*nexp, 1],
                                               scispectra[i1:i1+448*nexp]*norm, 
                                               errspectra[i1:i1+448*nexp]*norm,
                                               ADRx, 0. * def_wave, scale, ran)
        crx = np.abs(ran[0]) / scale + 1.
        cry = np.abs(ran[2]) / scale + 1.
        A.get_ifuslot_projection('%03d' % args.ifuslot, scale, crx, cry)
        
        header = A.tp_ifuslot.to_header()
        for key in header.keys():
            if ('CCDSEC' in key) or ('DATASEC' in key):
                continue
            if ('BSCALE' in key) or ('BZERO' in key):
                continue
            he[key] = header[key]
        try:
            he['VSEEING'] = np.nan_to_num(fwhm_virus)
        except:
            log.warning('Whoops!')
        he['GSEEING'] = np.median(gseeing)
        write_cube(def_wave, xgrid, ygrid, zgrid, cubename, he)
        write_cube(def_wave, xgrid, ygrid, egrid, ecubename, he)
