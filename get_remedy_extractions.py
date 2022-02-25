#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 23 10:42:43 2021

@author: gregz
"""

import argparse as ap
import astropy.units as u
import glob
import numpy as np
import os.path as op
import sys
import tables
import warnings

from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.modeling.fitting import LevMarLSQFitter
from astropy.modeling.models import Polynomial1D
from astropy.table import Table, Column
from astropy.table import vstack
from astropy.time import Time
from astrometry import Astrometry
from datetime import datetime as dt
from extract import Extract
from input_utils import setup_logging

# =============================================================================
# Global variables
# =============================================================================
fitter = LevMarLSQFitter()  # Fitter for masking
def_wave = np.linspace(3470., 5540., 1036)  # Default Wavelength Solution
max_sep = 11.0 * u.arcminute  # ~FoV for a single VIRUS shot
warnings.filterwarnings("ignore")


def get_script_path():
    '''
    Get script path, aka, where does Remedy live?
    '''
    return op.dirname(op.realpath(sys.argv[0]))

def get_gmag(spec):
    '''
    Get gmag (PanSTARRS filter) for a given spectrum
    '''
    spectrum = spec * 1.
    spectrum = spectrum * 1e-17 * 1e29 * def_wave**2 / 2.99792e18
    gmask = np.isfinite(spectrum)
    nu = 2.99792e18 / def_wave
    dnu = np.diff(nu)
    dnu = np.hstack([dnu[0], dnu])
    num = np.nansum((nu / dnu) * spectrum *
                     filtg * gmask)
    denom = np.nansum(nu / dnu * filtg * gmask)
    gmag = num / denom
    gmag = -2.5 * np.log10(gmag) + 23.9
    return gmag

def get_mask(weigh):
    '''
    Get mask for an array of spectra using a polynomial for the weight array
    and finding dips from that polynomial model.
    '''
    mask = np.zeros(weigh.shape, dtype=bool)
    P = Polynomial1D(3)
    
    # Select values that are not extreme outliers (watch for nans so use nanmedian)
    wmask = weigh > 0.3 * np.nanmedian(weigh)
    if wmask.sum() < 100:
        return np.ones(weigh.shape, dtype=bool)
    
    # Fit Polynomial
    fit = fitter(P, def_wave[wmask], weigh[wmask])
    
    # Mask values below 80% of the fit
    mask = weigh < 0.8 * fit(def_wave)
    
    # Mask neighbor pixels as well because they tend to have issues but don't meet
    # the threshold.  This is a conservative step, but it is done for robustness
    # rather than completeness
    mask[1:] += mask[:-1]
    mask[:-1] += mask[1:]
    
    # Mask additionally spectra whose average weight is less than 5%
    if np.nanmedian(weigh) < 0.05:
        mask = np.ones(weigh.shape, dtype=bool)
    return mask

DIRNAME = get_script_path()
T = Table.read(op.join(DIRNAME, 'filters/ps1g.dat'), format='ascii')
filtg = np.interp(def_wave, T['col1'], T['col2'], left=0.0, right=0.0)
filtg /= filtg.sum()

T = Table.read(op.join(DIRNAME, 'calibration', 'normalization.txt'), 
               format='ascii.fixed_width_two_line')
flux_normalization = np.array(T['normalization'])

log = setup_logging('extractions')

parser = ap.ArgumentParser(add_help=True)

parser.add_argument("folder",
                    help='''Folder with h5 files''',
                    type=str)

parser.add_argument("extraction_file",
                    help='''list of RA and Decs to extract''',
                    type=str)

parser.add_argument("outputname",
                    help='''Name of fits file output''',
                    type=str)

parser.add_argument("-fp", "--fplane_file",
                    help='''fplane file''',
                    type=str, default=None)

parser.add_argument("-ra", "--RA", type=str,
		    help='''ra name''',
		    default='ra')

parser.add_argument("-dec", "--Dec", type=str,
		    help='''dec name''',
		    default='dec')

parser.add_argument("-pr", "--pmra", type=str,
		    help='''pmra name''',
		    default='pmra')

parser.add_argument("-pd", "--pmdec", type=str,
		    help='''pmdec name''',
		    default='pmdec')

parser.add_argument("-e", "--epoch", type=float,
		    help='''epoch name''',
		    default=2015.5)

parser.add_argument("-gm", "--gmag", type=str,
		    help='''gmag name''',
		    default='gmag')

parser.add_argument("-gn", "--g_normalize",
                    help='''Normalize by g magnitude''',
                    action="count", default=0)

parser.add_argument("-fn", "--flux_normalize",
                    help='''Apply flux normalization''',
                    action="count", default=0)

args = parser.parse_args(args=None)

# =============================================================================
# Get filenames for h5 files in the input folder
# =============================================================================
filenames = sorted(glob.glob(op.join(args.folder, '*.h5')))


# =============================================================================
# Get BinTable from input fits file
# =============================================================================
fitsfile = fits.open(args.extraction_file)
bintable = fitsfile[1].data
coords = SkyCoord(bintable[args.RA]*u.deg, bintable[args.Dec]*u.deg)

# =============================================================================
# Use this formalism to add columns to input BinTable for output purposes
# =============================================================================
table = Table(bintable)
table.add_column(Column(np.zeros((len(table),), dtype=int), name='obs_id'))

# =============================================================================
# Find the shots of interest and sources in each shot
# =============================================================================
matched_sources = {}
shots_of_interest = []
for filename in filenames:
    # Open h5 file
    t = tables.open_file(filename, mode='r')
    R = t.root.Survey.cols.ra[0]
    D = t.root.Survey.cols.dec[0] 
    coord = SkyCoord(R*u.deg, D*u.deg)
    
    # Check separation from shot center and the full list of sources
    dist = coords.separation(coord)
    
    # Find sources within max_sep (11') of the shot center
    sep_constraint = dist < max_sep
    idx = np.where(sep_constraint)[0]
    
    # Keep those sources and shots of interest
    matched_sources[filename] = idx
    if len(idx) > 0:
        shots_of_interest.append(filename)
    
    # Close h5 file
    t.close()

# =============================================================================
# Begin Extraction
# =============================================================================
Sources, Spectra, Error, Weights = ([], [], [], [])
for filename in shots_of_interest:
    # Get the indices of the extraction file for this shot
    ind = matched_sources[filename]
    
    # Get date, integer name of the shot (DATEOBSID), and epoch from the date
    date = op.basename(filename).split('_')[0]
    intname = int(''.join(op.basename(filename).split('.')[0].split('_')))
    epoch = Time(dt(int(date[:4]), int(date[4:6]), int(date[6:8]))).byear
    
    # Get the proper motion corrections
    deltaRA = ((epoch - args.epoch) * bintable['pmra'][ind] / 1e3 / 3600. /
                       np.cos(bintable[args.Dec][ind] * np.pi / 180.))
    deltaDE = (epoch - args.epoch) * bintable['pmdec'][ind] / 1e3 / 3600.
    deltaRA[np.isnan(deltaRA)] = 0.0
    deltaDE[np.isnan(deltaDE)] = 0.0
    ncoords = SkyCoord((bintable[args.RA][ind]+deltaRA)*u.deg,
                       (bintable[args.Dec][ind]+deltaDE)*u.deg)
    
    # Open the file and load relevant information
    t = tables.open_file(filename, mode='r')
    R = t.root.Survey.cols.ra[0]
    D = t.root.Survey.cols.dec[0]
    pa = t.root.Survey.cols.pa[0]
    rot = t.root.Survey.cols.rot[0]
    seeing = t.root.Survey.cols.fwhm[0]
    
    # Build the astrometry for the 
    A = Astrometry(R, D, pa, 0., 0., fplane_file=args.fplane_file)
    A.tp = A.setup_TP(R, D, rot, A.x0,  A.y0)
    
    # Setup extract for extraction
    E = Extract()
    E.ra, E.dec = (t.root.Info.cols.ra[:], t.root.Info.cols.dec[:])
    E.data = t.root.Fibers.cols.spectrum[:]
    E.error = t.root.Fibers.cols.error[:]
    E.ifuslot_i = t.root.Info.cols.ifuslot[:]
    E.mask = np.isfinite(E.data)
    tophat = E.tophat_psf(3., 10.5, 0.025)
    moffat = E.moffat_psf(seeing, 10.5, 0.025)

    newpsf = tophat[0] * moffat[0] / np.max(tophat[0])
    psf = [newpsf, moffat[1], moffat[2]]
    E.psf = psf

    E.get_ADR_RAdec(A)
    E.coords = ncoords
    

    # Go through each source and extract
    log.info('Extracting %i Sources from %s' % 
             (len(E.coords), op.basename(filename)))
    for i in np.arange(len(E.coords)):
        # Extract spectrum
        specinfo = E.get_spectrum_by_coord_index(i)
        spectrum, error, weight = (specinfo[0], specinfo[1], specinfo[2])

        # If the spectrum info comes back empty
        if len(spectrum) == 0:
            log.info('Did not find fibers for source %i' % (i+1))
            continue
        
        # Find mask for fibers based on weight array
        mask = get_mask(weight)
        if (mask.sum() > (len(mask) / 2.)):
            log.info('More than half of the spectrum is masked for source %i' %
                     (i+1))
            continue
        
        # Mask the bad points with low weights
        spectrum[mask] = np.nan
        error[mask] = np.nan
        if args.flux_normalize:
            spectrum[:] = spectrum * flux_normalization
            error[:] = error * flux_normalization
        if args.g_normalize:
            gmag = get_gmag(spectrum)
            pgmag = bintable[args.gmag][ind][i]
            normalization = 10**(0.4 * (gmag - pgmag))
            spectrum[:] = spectrum * normalization
            error[:] = error * normalization
            log.info('Normalization for source %i: %0.2f' %
                     (i+1, normalization))
        # Add the shot "intname" to the table
        dtable = Table(table[ind])
        dtable['obs_id'] = intname
        Sources.append(dtable)
        Spectra.append(spectrum)
        Error.append(error)
        Weights.append(weight)
    t.close()
        
F1 = fits.BinTableHDU(vstack(Sources))
F1.header['EXTNAME'] = 'catalog'
F2 = fits.ImageHDU(np.array(Spectra))
F2.header['EXTNAME'] = 'spectra'
F2.header['CRVAL1'] = 3470.
F2.header['CDELTA1'] = 2.
F3 = fits.ImageHDU(np.array(Error))
F3.header['EXTNAME'] = 'error'
F3.header['CRVAL1'] = 3470.
F3.header['CDELTA1'] = 2.
F4 = fits.ImageHDU(np.array(Weights))
F4.header['EXTNAME'] = 'weight'
F4.header['CRVAL1'] = 3470.
F4.header['CDELTA1'] = 2.
F = fits.HDUList([fits.PrimaryHDU(), F1, F2, F3, F4])
F.writeto(args.outputname, overwrite=True)
        