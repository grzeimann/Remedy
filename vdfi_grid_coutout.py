#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 29 08:37:38 2025

@author: grz85
"""

import numpy as np
import warnings
import os
import psutil
import os.path as op

from astropy.io import fits
from astropy.modeling.models import Moffat2D
from astropy.modeling.fitting import TRFLSQFitter
from scipy.ndimage import maximum_filter
from astropy.stats import biweight_location as biweight
from astropy.wcs import WCS
from input_utils import setup_logging
from scipy.interpolate import LinearNDInterpolator, interp1d


# Turn off annoying warnings (even though some deserve attention)
warnings.filterwarnings("ignore")



def manual_convolution(a, G, error=False):
    """
    Convolve array `a` with kernel `G.array` (e.g., from Gaussian1DKernel).

    Parameters
    ----------
    a : array-like
        Input 1D signal.
    G : object
        Kernel with 1D array in `G.array`.
    error : bool, optional
        If True, return propagated uncertainty.

    Returns
    -------
    array
        Convolved result or error estimate.
    """
    b = np.zeros((len(a), len(G.array)))
    N = int(len(G.array) / 2. - 0.5)
    M = len(G.array)
    for i in np.arange(0, N):
        b[(N - i):, i] = a[:-(N - i)]

    b[:, N] = a
    for i in np.arange(N + 1, M):
        b[:-(i - N), i] = a[(i - N):]

    if error:
        return np.sqrt(np.nansum(b**2 * G.array**2, axis=1))

    return np.nansum(b * G.array, axis=1)


def pca_fit(H, data):
    '''
    Fit and reconstruct `data` using the PCA basis `H`.

    Parameters
    ----------
    H : ndarray
        PCA basis matrix (components x features).
    data : ndarray
        Input data vector (length = features).

    Returns
    -------
    res : ndarray
        Reconstructed data from least-squares PCA fit.
    '''
    sel = np.isfinite(data)
    sol = np.linalg.lstsq(H.T[sel], data[sel])[0]
    res = np.dot(H.T, sol)
    return res

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
    a = np.array([biweight(f, axis=1, ignore_nan=True) 
               for f in np.array_split(spectra, nbins, axis=1)]).swapaxes(0, 1)
    x = np.array([np.mean(xi) 
               for xi in np.array_split(np.arange(spectra.shape[1]), nbins)])
    cont = np.zeros(spectra.shape)
    X = np.arange(spectra.shape[1])
    for i, ai in enumerate(a):
        sel = np.isfinite(ai)
        if np.sum(sel)>nbins/2.:
            I = interp1d(x[sel], ai[sel], kind='quadratic',
                         fill_value='extrapolate', bounds_error=False)
            cont[i] = I(X)
        else:
            cont[i] = 0.0
    return cont

def wavelength_corrected_seeing(seeing, wave):
    '''
    Correct the seeing as a function of wavelength
    
    Parameters
    ----------
    seeing : float
        FWHM
    wave : array
        Wavelength for the spectra
    '''
    
    seeing = seeing * (wave / 4500.)**(-1. / 5.)
    return seeing

def moffat_psf_integration(r, seeing, boxsize=14.,
                           scale=0.05, alpha=3.5):
    '''
    Moffat PSF profile image

    Parameters
    ----------
    r : array
        radial distance of the fiber from the center of the Moffat profile
    seeing: array
        FWHM of the Moffat profile
    boxsize: float
        Size of image on a side for Moffat profile
    scale: float
        Pixel scale for image
    alpha: float
        Power index in Moffat profile function

    Returns
    -------
    zarray: numpy 3d array
        An array with length 3 for the first axis: PSF image, xgrid, ygrid
    '''
    xl, xh = (0. - boxsize / 2., 0. + boxsize / 2. + scale)
    yl, yh = (0. - boxsize / 2., 0. + boxsize / 2. + scale)
    x, y = (np.arange(xl, xh, scale), np.arange(yl, yh, scale))
    xgrid, ygrid = np.meshgrid(x, y)
    V = np.zeros((len(seeing), len(r)))
    for j, fwhm in enumerate(seeing):
        M = Moffat2D()
        M.alpha.value = alpha
        M.gamma.value = 0.5 * fwhm / np.sqrt(2**(1./ M.alpha.value) - 1.)
        Z = M(xgrid, ygrid)
        Z = Z / Z.sum()
        for i, ri in enumerate(r):
            d = np.sqrt((xgrid-ri)**2 + (ygrid-0.)**2)
            sel = d <= 0.75
            adj = np.pi * 0.75**2 / (sel.sum() * scale**2)
            V[j, i] = np.sum(Z[sel]) * adj
    R, S = np.meshgrid(r, seeing)
    R, S, V = [x.ravel() for x in [R, S, V]]
    interp = LinearNDInterpolator(list(zip(R, S)), V)
    return interp, R, S, V

def find_detections(sn_cube, threshold=4.5):
    """
    Identify local maxima above a threshold in a 3D S/N cube.

    Parameters
    ----------
    sn_cube : ndarray
        3D signal-to-noise array.
    threshold : float, optional
        Minimum S/N value to consider a detection (default is 4.5).

    Returns
    -------
    ndarray
        Coordinates of local maxima above the threshold.
    """
    # Step 1: Mask voxels above the threshold
    above_threshold = sn_cube > threshold

    # Step 2: Identify local maxima (compared to 26 neighbors)
    local_max = (sn_cube == maximum_filter(sn_cube, size=3, mode='constant', 
                                           cval=0))

    # Step 3: Combine both conditions
    result_mask = above_threshold & local_max

    # Get coordinates of detections
    local_max_coords = np.argwhere(result_mask)
    
    return local_max_coords


def get_spectrum_exposure(spectra, error, dra, ddec, seeing, PSF, wave,
                          goodpix_thresh=0.4):
    '''
    Get a spectrum from a single exposure

    Parameters
    ----------
    spectra : 2D array
        Spectra of the fibers within a radius threshold
    error: 2D array
        Error of the fiber spectra in the radius threshold
    dra: 2D array
        Delta RA including the wavelength term of DAR
    ddec: 2D array
        Delta Dec including the wavelength term of DAR
    seeing: float
        FWHM of the exposure
    PSF: interpolator class
        Provides the weight of the fiber as a function radius and seeing

    Returns
    -------
    spectrum: array
        A weighted spectrum using the Moffat weights
    spectrum_error: array
        A weighted error spectrum
    '''
    r = np.sqrt(dra**2 + ddec**2)
    seeing = wavelength_corrected_seeing(seeing, wave)
    Seeing = np.ones((r.shape[0],))[:, np.newaxis] * seeing[np.newaxis, :]
    weights = PSF(r, Seeing)
    var = error**2
    spectrum = (np.nansum(spectra / var * weights, axis=0) /
                np.nansum(weights**2 / var, axis=0))
    spectrum_error = np.sqrt(np.nansum(weights, axis=0) /
                             np.nansum(weights**2 / var, axis=0))
    
    mask = np.isfinite(spectra)
    summed_weights = np.nansum(weights * mask, axis=0)
    xi = [(np.mean(wi) - 4500.) / 1000. for wi in np.array_split(wave, 21)]
    yi = [np.nanmedian(wi) for wi in np.array_split(summed_weights, 21)]
    p0 = np.polyfit(xi, yi, 3)
    pfit = np.polyval(p0, (wave-4500.) / 1000.)
    mask = summed_weights / pfit < goodpix_thresh
    spectrum[mask] = np.nan
    spectrum_error[mask] = np.nan
    if np.median(pfit) < 0.1:
        spectrum = np.nan
        spectrum_error = np.nan
    return spectrum, spectrum_error, summed_weights, pfit
    
    
# Set up logging for tracking progress and issues
log = setup_logging('vdfi')

# Define the wavelength solution (e.g., for spectral axis)
wave = np.linspace(3470, 5540, 1036)

# Load spatial coordinate data (RA, DEC) from FITS file
f = fits.open('/work/03730/gregz/maverick/VDFI/all_info.fits', memmap=True)
RA = f[1].data * 1.
DEC = f[2].data * 1.
# Flatten spatial dimensions into a 2D array: (exposure, spaxels)
RA, DEC = [x.reshape((x.shape[0], x.shape[1] * x.shape[2])) for x in [RA, DEC]]
nexp = RA.shape[0]  # Number of exposures

# Load spectral data (flux and error)
g = fits.open('/work/03730/gregz/maverick/VDFI/all_flux_final.fits', memmap=True)
e = fits.open('/work/03730/gregz/maverick/VDFI/all_error.fits', memmap=True)

mask_amps = [106, 122, 123, 131, 135, 146, 256, 264, 286, 287, 299, 301]

spectra = g[0].data
spectra[:, mask_amps, :, :] = np.nan
# Reshape to (exposure, fibers, wavelength)
spectra = spectra.reshape((spectra.shape[0], 
                           spectra.shape[1] * spectra.shape[2], 
                           spectra.shape[3]))
error = e[0].data
error[:, mask_amps, :, :] = np.nan
error = error.reshape((error.shape[0], error.shape[1] * error.shape[2], 
                       error.shape[3]))

# Load Differential Atmospheric Refraction (DAR) corrections
dar = fits.open('/work/03730/gregz/maverick/VDFI/all_initial_dar.fits')
dar_ra = dar[0].data
dar_dec = dar[1].data

# Set up PSF (Point Spread Function) modeling
radius = 3.75
r = np.linspace(0, radius, 101)  # Radial profile
seeing = np.linspace(np.min(f[5].data)-0.05, np.max(f[5].data)+0.05, 51)
PSF, R, S, V = moffat_psf_integration(r, seeing)

# Load reference images
cfht = fits.open('/work/03730/gregz/ls6/PHATTER-VIRUS/CFHT_EGS_image.fits', memmap=True)
header = cfht[0].header

# Get world coordinates (RA, Dec)
wcs = WCS(header)

# Set up a coarse grid in pixel space and convert to world coordinates
xg = np.arange(30.5, 1230.5, 60)
xgrid, ygrid = np.meshgrid(xg, xg)
grid_ra, grid_dec = wcs.all_pix2world(xgrid + 1., ygrid + 1., 1)

# We are running detections over one full column of 20 rows
xs = np.arange(20)
ys = np.arange(20)

# Set up a least-squares fitter for later use
fitter = TRFLSQFitter()

# Log the beginning of the detection step
log.info('Starting Detections')


# ===============================
# Main Loop Over Grid Positions
# ===============================
for xi in xs:
    for xj in ys:
        
        # Log memory usage for current grid point
        process = psutil.Process(os.getpid())
        log.info('Memory Used for  %i, %i: %0.2f GB' % 
                 (xi, xj, process.memory_info()[0] / 1e9))
        
        filename = 'short_%02d_%02d.fits' % (xi, xj)
        if op.exists(filename):
            continue
        # Extract RA/Dec center from input grids
        ra_center = grid_ra[xi, xj]
        dec_center = grid_dec[xi, xj]
        
        # Define cutout parameters
        size = 30.
        step = 1.
        buffer = 4.5
        cx = xgrid[xi, xj]
        cy = ygrid[xi, xj]
        
        # Initialize spectra and error arrays for extraction
        S = np.zeros((nexp, len(wave)))
        E = np.zeros_like(S)
        
        log.info('Making cutout for %i, %i' % (xi, xj))
    
        # ===============================
        # Spectral Cutout Initialization
        # ===============================
        shortspec = np.zeros((spectra.shape[0], 448, 1036), dtype=np.float32)
        shorterr = np.zeros((spectra.shape[0], 448, 1036), dtype=np.float32)
        shortra = np.zeros((spectra.shape[0], 448), dtype=np.float32)
        shortdec = np.zeros((spectra.shape[0], 448), dtype=np.float32)
        npix = 0
    
        # ===============================
        # Loop Over Exposures to Select Fibers in Region
        # ===============================
        for exposure in np.arange(nexp):
            dra = (RA[exposure]-ra_center) * np.cos(np.pi/180.*dec_center) * 3600.
            ddec = (DEC[exposure]-dec_center) * 3600.
            fiber_sel = ((np.abs(dra) < (size+buffer)) * 
                         (np.abs(ddec) < (size+buffer)))
            if fiber_sel.sum():
                mask = (np.isfinite(spectra[exposure, :, 400:500]).sum(axis=1) > 50.)
                fiber_sel = fiber_sel * mask
                if fiber_sel.sum() == 0:
                    continue
                npix += 1
                N = fiber_sel.sum()
                shortspec[exposure, :N] = spectra[exposure, fiber_sel]
                shorterr[exposure, :N] = error[exposure, fiber_sel]
                shortra[exposure, :N] = RA[exposure, fiber_sel]
                shortdec[exposure, :N] = DEC[exposure, fiber_sel]
        f0 = fits.PrimaryHDU(shortspec)
        f1 = fits.ImageHDU(shorterr)
        f2 = fits.ImageHDU(shortra)
        f3 = fits.ImageHDU(shortdec)
        fits.HDUList([f0, f1, f2, f3]).writeto('short_%02d_%02d.fits' %
                                               (xi, xj), overwrite=True)