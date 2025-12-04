#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
M33 VIRUS reduction and analysis pipeline (monolithic script)

Overview
- This script assembles a custom, end-to-end reduction workflow for VIRUS observations of M33.
- It combines image preparation, sky modeling with PCA, HDF5 per-shot loading, photometric
  and astrometric calibration, spectral modeling (PPXF), sky subtraction, and export of
  intermediate/diagnostic products.

Why it's monolithic
- This file historically evolved as an analysis notebook turned into a single large script.
  It contains multiple stages executed at import time. The code is kept intact here to avoid
  changing scientific results; comments were added to clarify the processing chain.

Key processing stages (in order of appearance)
1) Helper image builders
   - make_image_m33(): reprojects select image frames onto a common WCS and writes imagem33.fits.
   - make_image_lgs_m33(): prepares and reprojects an LGS image; returns WCS used later.
2) Timing/Astrometry helpers
   - get_barycor(): computes the barycentric velocity correction for an exposure.
3) Spectral modeling
   - ppxf_fit(): builds stellar+gas templates, fits spectra in log-lambda space, returns separated
     star and gas models on the native VIRUS wavelength grid.
4) Global configuration and inputs
   - Load flux normalization from Diagnose outputs; define filters (HST F475W and B) and M33 coord.
5) Image preparation
   - Load PHATTER F475W-binned image and a precomputed imagem33.fits; convolve to match PSF.
6) Sky model construction
   - Build an average sky spectrum from blank frames near the observations and normalize to
     the B-band filter. Fit a low-rank PCA basis to model time-variable sky.
7) HDF5 loading and first-pass sky solution
   - Load per-shot spectra from .h5 files; estimate initial sky and scale factors per exposure
     and amplifier; compute per-fiber photometry and quality cuts.
8) Astrometric alignment and photometric calibration
   - Match fiber maps to the reference image using custom Astrometry helper; derive corrections
     and apply per-exposure normalization.
9) PPXF-based continuum and emission separation (optional paths)
   - For selected high S/N fibers, run ppxf_fit to construct stellar/gas models used to refine
     flux calibration and background subtraction.
10) Final sky subtraction and export
   - Apply exposure- and amplifier-dependent factors, subtract average + PCA sky model, and write
     per-exposure FITS files with extensions: SKYSUB, ERROR, SKY, POS.

Notes
- Many absolute paths point to the author's workspace; adapt as needed.
- Dependencies include astropy, numpy, photutils, scikit-learn (PCA), ppxf, reproject.
- For scientific reproducibility, the original sequence and calculations are preserved.
"""

import tables
import numpy as np
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.stats import biweight_location as biweight
from astropy.io import fits
from astropy.wcs import WCS
from photutils.aperture import CircularAperture
from photutils.aperture import aperture_photometry
from astropy.convolution import convolve, Moffat2DKernel
from astropy.table import Table
import os.path as op
from astropy.visualization import AsinhStretch, ImageNormalize
from astrometry import Astrometry
from sklearn.decomposition import PCA
import seaborn as sns
import ppxf.ppxf_util as util
import ppxf.sps_util as lib
from ppxf.ppxf import ppxf
from os import path
from scipy.signal import medfilt
import sys

from astropy.modeling.fitting import FittingWithOutlierRemoval, LevMarLSQFitter
from astropy.modeling.models import Polynomial2D, Polynomial1D
from astropy.stats import sigma_clip, mad_std
from astropy.coordinates import SkyCoord, EarthLocation
from astropy.time import Time


from reproject.mosaicking import find_optimal_celestial_wcs
from reproject import reproject_interp
from reproject.mosaicking import reproject_and_coadd

def make_image_m33():
    """Build a reference M33 image by reprojecting multiple frames to a common WCS.

    - Opens a local multi-extension FITS image stack, selects specific extensions,
      PSF-matches each to approximate VIRUS seeing using a Moffat kernel, and
      reprojects/coadds them to an optimal celestial WCS.
    - Writes the combined image to 'imagem33.fits' in the current working directory.

    Notes
    - Absolute file path is hard-coded to the author's environment; update as needed.
    - Uses reproject to determine an optimal WCS and to coadd.
    """
    image_file = fits.open('/Users/grz85/Downloads/m026_g.fit')

    image_list = []
    for i in [4, 5, 6, 12, 13, 14, 15, 16, 21, 22,23, 24, 25, 31, 32, 33]:
        image = image_file[i].data
        image = image - image[0, 0]
        image = image * 10**(-0.4*(33.1-23.9)) * 1e-29 * 2.99798e18 / 4800**2 * 1e17
        wcs = WCS(image_file[i].header)
        pixscale = (np.sqrt(np.abs(wcs.pixel_scale_matrix[0, 0] * 
                                    wcs.pixel_scale_matrix[1, 1])) * 3600.)
        image = convolve(image, Moffat2DKernel(1.45/pixscale/0.936, alpha=3.5))
        image_file[i].data = image
        image_list.append(image_file[i])

    wcs_out, shape_out = find_optimal_celestial_wcs(image_list)
    image147, footprint = reproject_and_coadd(image_list,
                                            wcs_out, shape_out=shape_out,
                                            reproject_function=reproject_interp)
    fits.PrimaryHDU(image147, header=wcs_out.to_header()).writeto('imagem33.fits', overwrite=True)


def make_image_lgs_m33():
    """Prepare and reproject an LGS mosaic of M33 to a target WCS.

    - Loads an LGS FITS mosaic, converts units to f_lambda, masks sentinel values,
      PSF-matches with a Moffat kernel, and reprojects/coadds to an optimal WCS
      at 0.7"/pix resolution. Also estimates and subtracts a robust background.
    - Writes 'imagem33.fits' and returns the output WCS for later alignment.

    Returns
    - wcs_out: astropy.wcs.WCS of the output mosaic.

    Notes
    - Input path is hard-coded; adjust to your environment.
    - Background level is estimated from the outer annulus using biweight.
    """
    image_file = fits.open('/Users/grz85/work/Backup/cure/Remedy/M33/M33CBm.fits')

    image = image_file[0].data
    image[image == 50000.] = np.nan
    image = image * 10**(-0.4*(26.5-23.9)) * 1e-29 * 2.99798e18 / 4800**2
    wcs = WCS(image_file[0].header)
    pixscale = (np.sqrt(np.abs(wcs.pixel_scale_matrix[0, 0] * 
                               wcs.pixel_scale_matrix[1, 1])) * 3600.)
    image = convolve(image, Moffat2DKernel(1.45/pixscale/0.936, alpha=3.5))
    image_file[0].data = image
    wcs_out, shape_out = find_optimal_celestial_wcs([image_file[0]], 
                                                    resolution=0.7*u.arcsec)
    image, footprint = reproject_and_coadd([image_file[0]],
                                            wcs_out, shape_out=shape_out,
                                            reproject_function=reproject_interp)
    y, x = image.shape
    hy = y / 2. 
    hx = x / 2. 
    xgrid, ygrid = np.meshgrid(np.arange(x), np.arange(y))
    sel = np.sqrt((xgrid-hx)**2 + (ygrid-hy)**2) > 0.8 * np.sqrt(hx*hy)
    back = biweight(image[sel], ignore_nan=True)
    image[:] = (image - back) * 0.6695 + 2.32076357e-18
    fits.PrimaryHDU(image, header=wcs_out.to_header()).writeto('imagem33.fits', 
                                                               overwrite=True)
    return wcs_out

def get_barycor(exptime, date, ra, dec):
    """Compute barycentric velocity correction for an observation.

    Parameters
    - exptime: float
        Exposure time in seconds.
    - date: str or astropy.time.Time-compatible
        Start time of the exposure (UTC). The midpoint (date + exptime/2) is used.
    - ra, dec: str
        Target coordinates as strings parsable by SkyCoord with units
        (u.hourangle, u.deg) e.g., ra='01:33:50.9', dec='+30:39:36'.

    Returns
    - float: barycentric correction in km/s at McDonald Observatory.
    """
    tm = exptime
    tk = Time(date) + tm / 2. * u.second
    posk = '%s %s' % (ra, dec)
    sc = SkyCoord(posk, unit=(u.hourangle, u.deg), obstime=tk)
    loc = EarthLocation.of_site('McDonald Observatory')
    vcorr = sc.radial_velocity_correction(kind='barycentric',
                                          location=loc)
    return vcorr.value

# =============================================================================
# Construct the function to perform a PPXF fit
# =============================================================================

def ppxf_fit(spectrum, error, z=-0.0005):
    """Fit stellar continuum and gas emission with PPXF and return models.

    Parameters
    - spectrum: 1D array (len=1036)
        Observed VIRUS spectrum on the native linear wavelength grid 3470–5540 Å.
    - error: 1D array (len=1036)
        1-sigma uncertainty per pixel on the same grid.
    - z: float, optional
        Approximate redshift used to shift to restframe for template matching.

    Returns
    - star_model: 1D array (len=1036)
        Best-fit stellar continuum model, resampled back to the native grid.
    - gas_model: 1D array (len=1036)
        Best-fit emission-line model, resampled back to the native grid.

    Method summary
    - Trim edges, log-rebin the spectrum and noise (ppxf.util.log_rebin).
    - Build E-MILES stellar templates and PPXF gas templates across VIRUS range.
    - Fit a two-component model (stars + gas) with separate kinematics,
      using bounds on velocity and dispersion, with no additive/multiplicative polynomials.
    - Interpolate the best-fit stellar and gas models back to the native grid.
    """

    lam_lin = np.linspace(3470, 5540, 1036)[20:-15]

    lam_lin /= (1 + z)               # Compute approximate restframe wavelength
    gal_lin = spectrum[20:-15]
    error_lin = error[20:-15]
    
    galaxy, lam, velscale = util.log_rebin(lam_lin, gal_lin)
    error, lam, velscale = util.log_rebin(lam_lin, error_lin)
    
    lam = np.exp(lam)
    FWHM_gal = 4.8
    R = 1e4*np.sqrt(0.347*0.554)/5.6
    c = 299792.458                      # speed of light in km/s
    sigma_inst = c/(R*2.355)

    
    FWHM_gal /= (1 + z)     # Adjust resolution in Angstrom
    
    norm = 1.#np.nanmedian(galaxy) 
    galaxy = galaxy / norm      # Normalize spectrum to avoid numerical issues
    error = error / norm       # Assume constant noise per pixel here. I adopt a noise that gives chi2/DOF~1
    
    velscale = c*(np.log(lam[1]/lam[0])) # eq.(8) of Cappellari (2017)
    
    FWHM_temp = 2.51   # Resolution of E-MILES templates in the fitted range
    
    sps_name = 'emiles'
    ppxf_dir = path.dirname(path.realpath(lib.__file__))
    basename = "spectra_%s_9.0.npz" % sps_name
    filename = path.join(ppxf_dir, 'sps_models', basename)
    miles = lib.sps_lib(filename, velscale, FWHM_gal, norm_range=[5070, 5950],
                      age_range=[0, 2.2])
    
    reg_dim = miles.templates.shape[1:]
    stars_templates = miles.templates.reshape(miles.templates.shape[0], -1)
    
    lam_range_gal = [np.min(lam), np.max(lam)]
    gas_templates, gas_names, line_wave = util.emission_lines(miles.ln_lam_temp, 
                                                              lam_range_gal, 
                                                              FWHM_gal,
                                                              tie_balmer=1)
    
    templates = np.column_stack([stars_templates, gas_templates])

    start = [0, 200.]
    
    n_stars = stars_templates.shape[1]
    n_gas = len(gas_names)
    component = [0]*n_stars + [1]*n_gas
    gas_component = np.array(component) > 0  # gas_component=True for gas templates
    
    moments = [2, 2]
    
    start = [start, start]
    
    bounds_stars = [[-500, 500], [20, 300]]
    bounds_gas = [[-500, 500], [20, 300]]
    bounds = [bounds_stars, bounds_gas]
    
    pp = ppxf(templates, galaxy, error, velscale, start,
              moments=moments, degree=-1, mdegree=-1, lam=lam, lam_temp=miles.lam_temp,
              reg_dim=reg_dim, component=component, gas_component=gas_component,
              reddening=0, gas_reddening=0, gas_names=gas_names, bounds=bounds)

    star_model = np.interp(np.linspace(3470, 5540, 1036), lam*(1+z),  
                           pp.bestfit - pp.gas_bestfit, left=np.nan,
                           right=np.nan)
    gas_model = np.interp(np.linspace(3470, 5540, 1036), lam*(1+z),  
                          pp.gas_bestfit, left=np.nan,
                          right=np.nan)

    return star_model, gas_model

# =============================================================================
# Load the flux normalization correction from Diagnose work
# =============================================================================

T = Table.read(op.join('/Users/grz85/work/Backup/cure/Diagnose', 'calibration', 'normalization.txt'), 
               format='ascii.fixed_width_two_line')
flux_normalization = np.array(T['normalization'])

# =============================================================================
# Set up the coordinates for m33 as well as the filter curve for HST/F475W
# =============================================================================

m33 = SkyCoord(23.462040*u.deg, 30.660220*u.deg)

nexp = 3
wave = np.linspace(3470, 5540, 1036)
g_filter = np.loadtxt('/Users/grz85/work/Backup/cure/HETDEX_LOFAR/MCSED/FILTERS/wfc3_f475w.res')
g_filter = np.interp(wave, g_filter[:, 0], g_filter[:, 1], left=0., right=0.0)
g_filter = g_filter / g_filter.sum()

b_filter = np.loadtxt('/Users/grz85/work/Backup/cure/Remedy/M33/B_filter.txt')
b_filter = np.interp(wave, b_filter[:, 0], b_filter[:, 1], left=0., right=0.0)
b_filter = b_filter / b_filter.sum()
# =============================================================================
# Names of the h5 files for this program 
# =============================================================================
m33_filenames = [ '20221216_0000016.h5', '20221216_0000017.h5', 
                  '20221218_0000016.h5', '20221220_0000017.h5', 
                  '20221225_0000013.h5', '20221225_0000014.h5',
                  '20221227_0000013.h5', '20221227_0000014.h5', 
                  '20230114_0000013.h5', '20230115_0000011.h5', 
                  '20230117_0000010.h5', '20230121_0000010.h5',
                  '20231205_0000020.h5', '20231206_0000022.h5',
                  #'20231206_0000023.h5', 
                  '20231207_0000018.h5',
                  '20231208_0000020.h5', '20231211_0000017.h5',
                  #'20231211_0000018.h5', 
                  '20231212_0000020.h5',
                  '20231212_0000021.h5', '20231215_0000018.h5',
                  '20231215_0000019.h5', '20231216_0000016.h5',
                  #'20231217_0000016.h5', 
                  '20231217_0000017.h5',
                  '20231218_0000015.h5', '20240101_0000012.h5']


# =============================================================================
# Load the phatter F475W binned image
# =============================================================================
f = fits.open('/Users/grz85/work/Backup/cure/Diagnose/phatter_F475W_binned.fits')
Image = f[0].data
wcs_out = WCS(f[0].header)
pixscale = (np.sqrt(np.abs(wcs_out.pixel_scale_matrix[0, 0] * 
                           wcs_out.pixel_scale_matrix[1, 1])) * 3600.)

Image = convolve(Image, Moffat2DKernel(gamma=1.75/pixscale/0.936, alpha=3.5))

f = fits.open('/Users/grz85/work/Backup/cure/Remedy/imagem33.fits')
Image = f[0].data
wcs_out = WCS(f[0].header)
pixscale = (np.sqrt(np.abs(wcs_out.pixel_scale_matrix[0, 0] * 
                           wcs_out.pixel_scale_matrix[1, 1])) * 3600.)

# =============================================================================
# Set up all of the blank lists
# =============================================================================
S = m33
ra, dec, y, catra, catdec, z, zr, zd, s, clean, ftf, im, sysm = ([], [], [], [], [], 
                                                  [], [], [], [], [], [], [], [])

# =============================================================================
# Construct average sky from "blank" frames near observations
# =============================================================================
skies = []
skyfiles = ['20221216_0000014.h5', '20221216_0000015.h5', 
            '20221218_0000013.h5', '20221218_0000014.h5', 
            '20221225_0000012.h5']
for fname in skyfiles:
    t = tables.open_file('/Users/grz85/work/Backup/cure/Remedy/M33/%s' % fname)
    skyspectra = t.root.Fibers.cols.skyspectrum[:] * flux_normalization
    spectra = t.root.Fibers.cols.spectrum[:] / t.root.Survey.cols.offset[0] * flux_normalization
    inds = np.arange(len(skyspectra))
    for i in np.arange(nexp):
        expsel = np.array(inds / 112, dtype=int) % nexp == i
        skies.append(biweight(spectra[expsel] + skyspectra[expsel], axis=0, ignore_nan=True))
    t.close()
skies = np.array(skies).swapaxes(0, 1)
msky = np.nanmean(skies / np.nanmean(skies, axis=0)[np.newaxis, :], axis=1)
c = np.nanmean(skies / msky[:, np.newaxis], axis=0)
skies = skies / c[np.newaxis, :] - msky[:, np.newaxis]
skies[np.isnan(skies)] = 0.0
msky[msky==0.0] = np.nan
norm = np.nansum(msky*b_filter)
msky = msky / norm

# =============================================================================
# Create principal components for sky frames
# =============================================================================
pca = PCA(n_components=5)
pca.fit(skies.swapaxes(0, 1))


# =============================================================================
# Loading h5 files and construction first solution for sky model
# =============================================================================
nskyspec = np.zeros((len(skyfiles)*3, 34944))
Yi = np.zeros((len(skyfiles)*3, 312))
Mi = np.zeros((len(skyfiles)*3, 312))
Zi = np.zeros((len(skyfiles)*3, 312))
Ra = np.zeros((len(skyfiles)*3, 312))
Da = np.zeros((len(skyfiles)*3, 312))
skyvalues = np.zeros((len(skyfiles)*3,))
cnt = 0
for fname in skyfiles:
    # Load h5 sky spectra, science spectra, ra, and dec
    t = tables.open_file('/Users/grz85/work/Backup/cure/Remedy/M33/%s' % fname)
    skyspectra = t.root.Fibers.cols.skyspectrum[:] * flux_normalization
    spectra = t.root.Fibers.cols.spectrum[:] / t.root.Survey.cols.offset[0] * flux_normalization
    r = t.root.Info.cols.ra[:] * 1.
    d = t.root.Info.cols.dec[:] * 1.
    t.close()

    # Go through exposure by exposure
    inds = np.arange(len(skyspectra))
    for i in np.arange(nexp):
        # Select the fibers for this exposure number
        expsel = np.array(inds / 112, dtype=int) % nexp == i
        
        # Add sky back into the spectra
        tot = spectra[expsel] + skyspectra[expsel]
        
        # First sky estimate
        sky = biweight(tot, axis=0, ignore_nan=True)
        scale = np.nanmean(sky / msky)
        initial_sky = scale * msky
        
        # Solve for the sky pca components
        sol = np.linalg.lstsq(pca.components_.T, sky - initial_sky)[0]
        full_sky = np.sum(pca.components_ * sol[:, np.newaxis], axis=0) + initial_sky
        
        # Calculate first sky subtraction spectra
        skysub = tot - full_sky
        sky_value = np.nansum(full_sky * b_filter)
        skyvalues[cnt] = sky_value
        nskyspec[cnt] = np.nansum((tot-full_sky[np.newaxis, :])*b_filter[np.newaxis, :], axis=1)
        namps = int(nskyspec.shape[1] / 112)
        
        # Build the average amplifier sky subtraction value
        yk = np.zeros((namps,))
        zk = np.zeros((namps,))
        mra = np.zeros((namps,))
        mdec = np.zeros((namps,))
        offset = np.zeros((namps,))
        for i in np.arange(namps):
            li = i * 112
            hi = (i+1) * 112
            Yi[cnt, i] = biweight(nskyspec[cnt, li:hi], ignore_nan=True) 
            mra[i] = np.nanmean(r[expsel][li:hi])
            mdec[i] = np.nanmean(d[expsel][li:hi])
        dra = np.cos(np.pi / 180. * mdec) * (mra - np.nanmean(mra)) * 60.
        ddec = (mdec - np.nanmean(mdec)) * 60.
        Ra[cnt] = dra
        Da[cnt] = ddec
        
        # Color is a term of the residual sky subtraction value divided by sky
        color = Yi[cnt] / sky_value
        selp = np.isfinite(Yi[cnt])
        
        # initialize a linear fitter
        fit = LevMarLSQFitter()

        # initialize the outlier removal fitter
        or_fit = FittingWithOutlierRemoval(fit, sigma_clip, niter=1, sigma=3.0)

        # initialize a linear model 2D model for residual map (amps across sky)
        poly_fit = Polynomial2D(1)

        # fit the data with the fitter
        fitted_line, mask = or_fit(poly_fit, dra[selp], ddec[selp], color[selp])
        
        Mi[cnt] = Yi[cnt] - fitted_line(dra, ddec) * sky_value
        
        cnt += 1

# =============================================================================
# After accessing the initial poly model correction, solve for average amp offset
# =============================================================================
MSky = Mi / skyvalues[:, np.newaxis]
avg = 1. - biweight(Mi / skyvalues[:, np.newaxis], axis=0, ignore_nan=True)
avgsky = avg * 1.
M2 = Mi * 0.

# =============================================================================
# Now we will loop through each exposure to solve for the 2D illumination correction
# =============================================================================
cnt = 0
for i in np.arange(Yi.shape[0]):
    Ym = Yi[cnt] - skyvalues[cnt] * (1.-avg)
    color = Ym / skyvalues[cnt]
    selp = np.isfinite(Ym)
    # initialize a linear fitter
    fit = LevMarLSQFitter()
    
    # initialize the outlier removal fitter
    or_fit = FittingWithOutlierRemoval(fit, sigma_clip, niter=1, sigma=3.0)
    
    # initialize a linear model
    poly_fit = Polynomial2D(1)
    
    # fit the data with the fitter
    fitted_line, mask = or_fit(poly_fit, Ra[cnt][selp], Da[cnt][selp], color[selp])
    
    M2[cnt] = Yi[cnt] - fitted_line(Ra[cnt], Da[cnt]) * skyvalues[cnt]
    
    cnt += 1

# =============================================================================
# After accessing the second iteration poly model correction, 
# solve for average amp offset
# =============================================================================
avg = 1. - biweight(M2 / skyvalues[:, np.newaxis], axis=0, ignore_nan=True)

# Mask amps with >50% offset from a normalization factor of 1.
avg[np.abs(avg-1.) > 0.5] = np.nan

# =============================================================================
# Time to reduce the science frames
# =============================================================================
nspec = np.zeros((len(m33_filenames)*3, 34944))
mspec = np.zeros((len(m33_filenames)*3, 34944))
ra = np.zeros((len(m33_filenames)*3, 34944))
dec = np.zeros((len(m33_filenames)*3, 34944))
Yi = np.zeros((len(m33_filenames)*3, 312))
Mi = np.zeros((len(m33_filenames)*3, 312))
Zi = np.zeros((len(m33_filenames)*3, 312))
Ra = np.zeros((len(m33_filenames)*3, 312))
Da = np.zeros((len(m33_filenames)*3, 312))
cnt = 0
for filename in m33_filenames:
    # Open the h5 file
    t = tables.open_file('/Users/grz85/work/Backup/cure/Remedy/M33/%s' % filename)
    
    # Load the sky and sci spectra, apply the flux normalization (from Diagnose)
    # And apply the flux multiplication in reverse ("offset") for the sci
    skyspectra = t.root.Fibers.cols.skyspectrum[:] * flux_normalization
    spectra = t.root.Fibers.cols.spectrum[:] / t.root.Survey.cols.offset[0] * flux_normalization
    spectra[spectra == 0.] = np.nan
    tot = spectra + skyspectra
    skysub = tot * np.nan
    
    # Get the Ra and Dec
    r = t.root.Info.cols.ra[:]
    d = t.root.Info.cols.dec[:]   
    
    # Create an indice array for exposure counting
    inds = np.arange(len(tot))
    
    for k in np.arange(nexp):
        expsel = np.array(inds / 112, dtype=int) % nexp == k
        sel = np.where(expsel)[0]
        
        # Get the photometry in fiber apertures from the HST F475W data
        positions = []
        for ra1, dec1 in zip(r[sel], d[sel]):
            positions.append(wcs_out.wcs_world2pix(
                ra1+0.3/3600./np.cos(np.deg2rad(dec1)), 
                dec1-0.25/3600., 1))
        aperture = CircularAperture(positions, r=0.75/pixscale)
        phot_table = aperture_photometry(Image, aperture)
        
        # The photometry from hst is "z"
        z = phot_table['aperture_sum'] * 1e17
        
        # The synthetic photometry from VIRUS is "y"
        y = np.nansum((tot[sel])*b_filter[np.newaxis, :], axis=1)
        
        # Record amplifier information for fitting later
        namps = int(nspec.shape[1] / 112)
        yk = np.zeros((namps,))
        zk = np.zeros((namps,))
        mra = np.zeros((namps,))
        mdec = np.zeros((namps,))
        for i in np.arange(namps):
            li = i * 112
            hi = (i+1) * 112
            yk[i] = biweight(y[li:hi], ignore_nan=True) 
            zk[i] = biweight(z[li:hi], ignore_nan=True)
            mra[i] = np.nanmean(r[sel][li:hi])
            mdec[i] = np.nanmean(d[sel][li:hi])
        dra = np.cos(np.pi / 180. * mdec) * (mra - np.nanmean(mra)) * 60.
        ddec = (mdec - np.nanmean(mdec)) * 60.
        
        # Save it all for fitting
        Yi[cnt] = yk
        Zi[cnt] = zk
        Ra[cnt] = dra
        Da[cnt] = ddec
        N = len(y)
        nspec[cnt][:N] = y
        mspec[cnt][:N] = z
        ra[cnt][:N] = r[sel]
        dec[cnt][:N] = d[sel]
        # Count up
        cnt += 1
        
    t.close()

# =============================================================================
# We are going to fit for flux multiplications and sky values
# Z = Y * fluxscale * FoV_poly * amplifier value - Sky
# =============================================================================
Mi = Yi * 0.
NewYi = Yi * 0.
skyvalues = np.zeros((len(m33_filenames)*3,))
multvalues = np.zeros((len(m33_filenames)*3,))
factors = Yi * 0.
for j in np.arange(1):
    for cnt in np.arange(Yi.shape[0]):
        Yi1 = Yi[cnt] * avg
        # Selection good amplifiers with HST coverage
        selfit = (np.isfinite(Zi[cnt]) * (Zi[cnt]>0.5) * (Yi[cnt]>0.5) *
                  np.isfinite(Yi1))
        
        # Initialize fitter
        fit = LevMarLSQFitter()
        or_fit = FittingWithOutlierRemoval(fit, sigma_clip, niter=1, sigma=3.0)
        
        # initialize a linear model
        poly_fit = Polynomial1D(1)
    
        # fit the data with the fitter
        fitted_line, mask = or_fit(poly_fit, Zi[cnt][selfit], Yi1[selfit])
        skyvalues[cnt] = fitted_line.c0.value
        multvalues[cnt] = fitted_line.c1.value
        sky = skyvalues[cnt] / multvalues[cnt]
        
        # Calculate the the new Y values correcting for the sky and mult value
        Yi2 = Yi1 / multvalues[cnt] - sky
        
        # Calculate the offset between VIRUS and HST
        color = (Yi2 - Zi[cnt]) / (Zi[cnt] + sky)
        
        # initialize a polynomial fitter
        fit = LevMarLSQFitter()
    
        # initialize the outlier removal fitter
        or_fit = FittingWithOutlierRemoval(fit, sigma_clip, niter=1, sigma=3.0)
    
        # initialize a polynomial model
        poly_fit = Polynomial2D(1)
    
        # fit the data with the fitter
        fitted_line, mask = or_fit(poly_fit, Ra[cnt][selfit], Da[cnt][selfit], color[selfit])
        factor = 1. - fitted_line(Ra[cnt],  Da[cnt])
        NewYi[cnt] = Yi[cnt] * avg / multvalues[cnt] * factor - sky
        factors[cnt] = factor
        
    Y = (NewYi - Zi) / (Zi + (skyvalues / multvalues)[:, np.newaxis])
    Y[Zi < 0.5] = np.nan
    Y[NewYi < 0.5] = np.nan
    newavg = 1. - biweight(Y, axis=0, ignore_nan=True)
    avg = avg * newavg

# =============================================================================
# Now apply the amplifier models to each fiber's photometry marking bad amps
# =============================================================================

avgfactor = biweight(factors, axis=0, ignore_nan=True)
s = mad_std(Y, ignore_nan=True, axis=0)
badamps = s > 0.04
badamps = (Y > 0.15) * (Zi > 0.5)
nspec1 = nspec * 1.
nspec1[nspec == 0.] = np.nan
nspec = nspec1 * 1.
for i in np.arange(Yi.shape[0]):
    for j in np.arange(Yi.shape[1]):
        li = j * 112
        hi = (j+1) * 112
        sky = skyvalues[i] / multvalues[i]
        nspec[i, li:hi] = nspec[i, li:hi] * avg[j] / multvalues[i] * factors[i, j] - sky
        if badamps[i, j]:
            nspec[i, li:hi] = np.nan

# =============================================================================
# We'll calculate the last multiplier which is the average fiber to fiber by amp (scalar)
# =============================================================================

namps = int(nspec.shape[1]/112)
ampavgs = np.zeros((namps, 112))

for j in np.arange(namps):
    li = j * 112
    hi = (j+1) * 112
    y = nspec[:, li:hi] * 1.
    y[y < 0.2] = np.nan
    z = mspec[:, li:hi] * 1.
    d = mspec[:, li:hi] + skyvalues[:, np.newaxis] / multvalues[:, np.newaxis]
    offset = biweight(y - z, axis=1, ignore_nan=True)
    good = np.isfinite(offset)
    ampavgs[j] = 1. - biweight((y - z - offset[:, np.newaxis]) / d, axis=0, ignore_nan=True)
    if good.sum() < 18:
        ampavgs[j] = 1.

# =============================================================================
# Now apply the fiber to fiber models to each fiber
# =============================================================================
    
nspec = nspec1 * 1.
for i in np.arange(Yi.shape[0]):
    for j in np.arange(Yi.shape[1]):
        li = j * 112
        hi = (j+1) * 112
        sky = skyvalues[i] / multvalues[i]
        nspec[i, li:hi] = nspec[i, li:hi] * avg[j] / multvalues[i] * factors[i, j] * ampavgs[j] - sky
        if badamps[i, j]:
            nspec[i, li:hi] = np.nan
            
# =============================================================================
# Make the image of M33 using our new calibrations
# =============================================================================

ra = np.hstack(ra)
dec = np.hstack(dec)
Y = np.hstack(nspec)
Z = np.hstack(mspec)
norm = ImageNormalize(vmin=0.5, vmax=5, stretch=AsinhStretch())
Z[np.isnan(Z)] = 0.0
Z[Y<0.3] = np.nan
Y[Y<0.3] = np.nan
Z[np.isnan(Y)] = np.nan

sns.set_context('talk')
sns.set_style('ticks')

plt.rcParams["font.family"] = "Times New Roman"
fig, ax = plt.subplots(1, 1, figsize=(8.42, 10.65), sharey=True, sharex=True, gridspec_kw={'wspace':0.0})
ax.set_position([0.15, 0.15, 0.8, 0.8])
ax.scatter(ra, dec, c=Y, s=1, edgecolor='none', cmap=plt.get_cmap('coolwarm'), norm=norm)
plt.axis([S.ra.deg+0.32, S.ra.deg-0.33, S.dec.deg-0.727*0.48, S.dec.deg+0.727*0.52])
plt.ylabel('Dec (deg)')
plt.xlabel('RA (deg)')
plt.savefig('PHATTER-VIRUS.png', dpi=150)
       
     
# =============================================================================
# Restart the process so we can do it to the spectra
# =============================================================================


pcamodel_amp = np.zeros((len(multvalues), namps, 1036))
background_amp = np.zeros((len(multvalues), namps, 1036))
spectrum_amp = np.zeros((len(multvalues), namps, 1036))
starmodel_amp = np.zeros((len(multvalues), namps, 1036))
gasmodel_amp = np.zeros((len(multvalues), namps, 1036))
indss = np.random.random_integers(312, size=20)
for kk, filename in enumerate(m33_filenames):
    t = tables.open_file('/Users/grz85/work/Backup/cure/Remedy/M33/%s' % filename)
    skyspectra = t.root.Fibers.cols.skyspectrum[:] * flux_normalization
    spectra = t.root.Fibers.cols.spectrum[:] / t.root.Survey.cols.offset[0] * flux_normalization
    error = t.root.Fibers.cols.error[:] / t.root.Survey.cols.offset[0] * flux_normalization
    r = t.root.Info.cols.ra[:] * 1.
    d = t.root.Info.cols.dec[:] * 1.
    t.close()
    spectra[spectra == 0.] = np.nan
    tot = spectra + skyspectra
    del spectra, skyspectra 
    inds = np.arange(len(tot))
    for k in np.arange(nexp):
        cnt = k + kk * 3
        expsel = np.array(inds / 112, dtype=int) % nexp == k
        sel = np.where(expsel)[0]
        initial_sky = skyvalues[cnt] / multvalues[cnt] * msky
        skysubs = tot[sel] * 0.
        sky = tot[sel] * 0.
        for i in indss:
            li = i * 112
            hi = (i+1) * 112
            skysub = tot[sel][li:hi] * avg[i] / multvalues[cnt] * factors[cnt, i] * ampavgs[i][:, np.newaxis] - initial_sky[np.newaxis, :]
            ampspc = biweight(skysub, axis=0, ignore_nan=True)     
            amperr = 5. * biweight(error[sel][li:hi], axis=0, ignore_nan=True) / np.sqrt(112.)
            if np.isnan(ampspc).sum() > 200:
                continue
            ampspc = np.interp(wave, wave[np.isfinite(ampspc)], ampspc[np.isfinite(ampspc)])
            amperr = np.interp(wave, wave[np.isfinite(amperr)], amperr[np.isfinite(amperr)])
            star_model, gas_model = ppxf_fit(ampspc, amperr)
            background = 0 * ampspc
            for ki in np.arange(2):
                ydata = ampspc - (gas_model+star_model) - background
                seld = np.isfinite(ydata)
                pcamodel = np.linalg.lstsq(pca.components_.T[seld], ydata[seld])[0]
                pcamodel = np.dot(pca.components_.T, pcamodel)
                background = ydata - pcamodel
                seld = np.isfinite(background)
                background = np.interp(wave, wave[seld], background[seld])
                background = medfilt(background, 41)
            background = -1*(background + ydata - ampspc + gas_model + star_model)
            pcamodel_amp[cnt, i] = pcamodel
            background_amp[cnt, i] = background
            spectrum_amp[cnt, i] = ampspc
            starmodel_amp[cnt, i] = star_model
            gasmodel_amp[cnt, i] = gas_model
            avgv = biweight(mspec[cnt, li:hi], ignore_nan=True)
            avgf = np.nansum((ampspc - pcamodel - background) * b_filter)

        b = background_amp[cnt] * 1.
        b[b==0.] = np.nan
        p = pcamodel_amp[cnt] * 1.
        p[p==0.] = np.nan
        avg_background = biweight(b, axis=0, ignore_nan=True)
        avg_pcamodel= biweight(p, axis=0, ignore_nan=True)
        avg_background[np.isnan(avg_background)] = 0.0
        avg_pcamodel[np.isnan(avg_pcamodel)] = 0.0
        for i in np.arange(int(len(sel) / 112)):
            li = i * 112
            hi = (i+1) * 112
            skysub = tot[sel][li:hi] * avg[i] / multvalues[cnt] * factors[cnt, i] * ampavgs[i][:, np.newaxis] - initial_sky[np.newaxis, :]
            error[sel][li:hi] *= avg[i] / multvalues[cnt] * factors[cnt, i] * ampavgs[i][:, np.newaxis]
            skysub = skysub - avg_background - avg_pcamodel
            skysubs[li:hi] = skysub
            sky[li:hi] = (initial_sky + avg_background + avg_pcamodel)[np.newaxis, :]
        expstr = '_exp%02d_' % (k+1)
        outname = filename[:-3] + expstr + '.fits'
        S = np.zeros((len(sel), 2))
        S[:, 0] = r[sel]
        S[:, 1] = d[sel]
        L = fits.HDUList([fits.PrimaryHDU(skysubs), fits.ImageHDU(error[sel]),
                          fits.ImageHDU(sky), fits.ImageHDU(S)])
        for l, name in zip(L, ['SKYSUB', 'ERROR', 'SKY', 'POS']):
            l.header['EXTNAME'] = name
        L.writeto(op.join('/Users/grz85/work/Backup/cure/Remedy/M33', outname), overwrite=True)
    del tot, error, skysubs, sky



sys.exit(1)

from astropy.stats import mad_std

fig, ax = plt.subplots(2, 1, figsize=(8.*3/4., 11.6*3./4.), sharex=True)
ax[0].set_position([0.15, 0.4, 0.8, 0.55])
ax[1].set_position([0.15, 0.15, 0.8, 0.25])
sel = (Z>0.1) * (Y > 0.1) * np.isfinite(Y) * np.isfinite(Z) * (Y < 25) * (Z < 25)# * (np.abs(Z-Y)/Z < 0.5)
xi = np.linspace(0.8, 5, 25)
yi = xi[:-1] * 0.
xx = xi[:-1] + np.diff(xi)
sig = xx * 0.
mn = xx * 0.
N = 2000
for i in np.arange(len(xi)-1):
    nsel = (Z[sel] > xi[i]) * (Z[sel] < xi[i+1])
    inds = np.array(np.random.permutation(nsel.sum()), dtype=int)
    ax[0].scatter(Z[sel][nsel][inds[:N]], Y[sel][nsel][inds[:N]], s=1, alpha=0.25, color='k', edgecolor='none')
    ax[1].scatter(Z[sel][nsel][inds[:N]], (Z[sel][nsel][inds[:N]] - Y[sel][nsel][inds[:N]]) / Z[sel][nsel][inds[:N]], s=1, alpha=0.25, color='k', edgecolor='none')
    sig[i] = mad_std((Z[sel][nsel][inds[:N]] - Y[sel][nsel][inds[:N]]) / Z[sel][nsel][inds[:N]])
    mn[i] = biweight((Z[sel][nsel][inds[:N]] - Y[sel][nsel][inds[:N]]) / Z[sel][nsel][inds[:N]])
inds = np.array(np.random.permutation(sel.sum()), dtype=int)
#plt.scatter(Z[sel][inds], Y[sel][inds], s=1, alpha=0.1, color='k')
for a in ax:
    plt.sca(a)
    f_ax10 = plt.gca()
    plt.minorticks_on()
    f_ax10.tick_params(axis='both', which='both', direction='in', zorder=3)
    f_ax10.tick_params(axis='y', which='both', left=True, right=True)
    f_ax10.tick_params(axis='x', which='both', bottom=True, top=True)
    f_ax10.tick_params(axis='both', which='major', length=8, width=2)
    f_ax10.tick_params(axis='both', which='minor', length=5, width=1)
ax[1].plot(xi, np.hstack([sig[0], sig]), 'r--', lw=2)
ax[1].plot(xi, np.hstack([mn[0], mn]), 'r-', lw=2)
ax[1].plot(xi, np.hstack([-sig[0], -sig]), 'r--', lw=2)
plt.sca(ax[0])
plt.axis([0.7, 5, 0.7, 5.])
plt.ylabel(r'VIRUS F475W f$_{\mathrm{\lambda}}$ (10$^{-17}$ erg / s / cm$^2$ / $\mathrm{\AA}$)', labelpad=12)
plt.sca(ax[1])
plt.axis([0.7, 5, -0.29, 0.29])
plt.ylabel(r'Fraction Residuals')
plt.xlabel(r'PHATTER F475W f$_{\mathrm{\lambda}}$ (10$^{-17}$ erg / s / cm$^2$ / $\mathrm{\AA}$)')
plt.savefig('phatter_virus_flux.png', dpi=300)

