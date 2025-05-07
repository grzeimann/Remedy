#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 17 19:58:46 2025

@author: grz85
"""

import numpy as np
import warnings
import os
import os.path as op
import psutil
import multiprocessing

from astropy.nddata import Cutout2D
from astropy.convolution import Gaussian2DKernel
from astropy.convolution import convolve, Gaussian1DKernel
from astropy.io import fits
from astropy.modeling.models import Moffat2D, Gaussian1D
from astropy.modeling.fitting import TRFLSQFitter
from astropy.table import Table
from scipy.ndimage import maximum_filter
from astropy.stats import mad_std
from astropy.stats import biweight_location as biweight
from astropy.wcs import WCS
from input_utils import setup_logging
from scipy.interpolate import LinearNDInterpolator, interp1d
from sklearn.decomposition import PCA
from photutils.aperture import aperture_photometry, CircularAperture


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


# Load Differential Atmospheric Refraction (DAR) corrections
dar = fits.open('/work/03730/gregz/maverick/VDFI/all_initial_dar.fits')
dar_ra = dar[0].data
dar_dec = dar[1].data

nexp = dar_ra.shape[0]  # Number of exposures

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


# Set up a least-squares fitter for later use
fitter = TRFLSQFitter()

# Log the beginning of the detection step
log.info('Starting Detections')


# ===============================
# Main Loop Over Grid Positions
# ===============================
def run_detection(counter):
    
    xi, xj = (int(counter / 20), counter % 20)
    filename = 'short_%02d_%02d.fits' % (xi, xj)
    outname = 'detect_output_%02d_%02d.fits' % (xi, xj)
    if op.exists(outname):
        log.info('Detection exists for %i, %i' % (xi, xj))

        return None
    # Extract RA/Dec center from input grids
    ra_center = grid_ra[xi, xj]
    dec_center = grid_dec[xi, xj]
    
    # Define cutout parameters
    size = 30.
    step = 1.
    
    # Initialize spectra and error arrays for extraction
    S = np.zeros((nexp, len(wave)))
    E = np.zeros_like(S)
    
    log.info('Getting cutout for %i, %i' % (xi, xj))

    # ===============================
    # Spectral Cutout Initialization
    # ===============================
   
    with fits.open(filename, memmap=True) as hdulist:
        shortspec = hdulist[0].data
        shorterr = hdulist[1].data
        shortra = hdulist[2].data
        shortdec =hdulist[3].data
        
        npix = np.sum(np.sum(shortspec[:, :, 405] != 0., axis=1) > 0.)
        # Skip if not enough usable spectra
        if npix < 10:
            log.info('Not enough spectra in region ... moving to next grid point')
            return None
        
        # Log memory usage for current grid point
        process = psutil.Process(os.getpid())
        log.info('Memory Used for  %i, %i: %0.2f GB' % 
                 (xi, xj, process.memory_info()[0] / 1e9))
        # ===============================
        # Begin Spectral Extraction
        # ===============================
        log.info('Running Extraction for %i, %i' % (xi, xj))
        S = np.zeros((nexp, len(wave)))
        E = np.zeros_like(S)
    
        dd = np.arange(-(size+step/2.), (size+step), step)
        dr = np.arange(-(size+step/2.), (size+step), step)[::-1]
        FL = np.zeros((len(dd), len(dr), 1036))
        EL = np.zeros_like(FL)
        iratios = np.zeros((len(dr), len(dd)))
    
        # ===============================
        # Loop Over Cutout Grid Pixels
        # ===============================
        for i in np.arange(len(dd)):
            for j in np.arange(len(dr)):
    
                def change_coordinates(dr, dd, ra_center, dec_center):
                    ra_center += dr / 3600. / np.cos(np.deg2rad(dec_center))
                    dec_center += dd / 3600.
                    return ra_center, dec_center
    
                ra_center1, dec_center1 = change_coordinates(dr[j], dd[i], ra_center, dec_center)
    
                # ===============================
                # Per Exposure Extraction at Each Cutout Spaxel
                # ===============================
                for exposure in np.arange(nexp):
                    dra = (shortra[exposure] - ra_center1) * np.cos(np.pi/180.*dec_center1) * 3600.
                    ddec = (shortdec[exposure] - dec_center1) * 3600.
                    mask = np.isfinite(shortspec[exposure, :, 405])
                    fiber_sel = (np.sqrt(dra**2 + ddec**2) <= radius) * mask
                    seeing = f[5].data[exposure]
                    newdra = dra[fiber_sel][:, np.newaxis] - dar_ra[exposure][np.newaxis, :]
                    newddec = ddec[fiber_sel][:, np.newaxis] - dar_dec[exposure][np.newaxis, :]
                    y = shortspec[exposure, fiber_sel]
                    spectrum, spectrum_e, w, pw = get_spectrum_exposure(
                        y, shorterr[exposure, fiber_sel],
                        newdra, newddec, seeing, PSF, wave
                    )
                    S[exposure] = spectrum
                    E[exposure] = spectrum_e
    
                # ===============================
                # Stack Spectra and Compute Initial Extraction
                # ===============================
                Spec = biweight(S, axis=0, ignore_nan=True)
                ratio = mad_std(((S - Spec[np.newaxis]) / E)[:, 40:-25], 
                                ignore_nan=True, axis=1)
                iratios[i, j] = biweight(ratio, ignore_nan=True)
                E *= ratio[:, np.newaxis]
                W = 1. / E**2
                Spec2 = np.nansum(S * W, axis=0) / np.nansum(W, axis=0)
                Err = np.sqrt(1. / np.nansum(W, axis=0))
                e = Err * 1.
                e[np.isnan(e)] = 0.0
                FL[i, j] = Spec2
                EL[i, j] = e
    
        # ===============================
        # Perform PCA to Model Residual Background
        # ===============================
        log.info('Fitting PCA for %i, %i' % (xi, xj))
        x, y = wcs.wcs_world2pix(ra_center, dec_center, 1)
        position = (x-1, y-1)
        size_cut = (FL.shape[0], FL.shape[1])
        cutout = Cutout2D(cfht[0].data, position, size_cut)
        cfht_image = cutout.data * 1.
        if cfht_image.shape[1] < FL.shape[1]:
            cfht_image = np.zeros((FL.shape[0], FL.shape[1]))
            cfht_image[:, :-1] = cutout.data
        if cfht_image.shape[0] < FL.shape[0]:
            cfht_image = np.zeros((FL.shape[0], FL.shape[1]))
            cfht_image[:-1, :] = cutout.data
        
        mask = convolve(cfht_image > 0.2, Gaussian2DKernel(2.5/2.35)) < 0.1
        mask *= (np.isnan(FL[:, :, 20:-20]).sum(axis=2) < 1)
    
        avg_back = np.nanmean(FL[mask], axis=(0,))
        data = (FL[mask] - avg_back[np.newaxis, :]) / EL[mask]
        data = data[:, 20:-20]
        res = np.zeros_like(FL) + avg_back[np.newaxis, np.newaxis, :]

        if data.shape[0] > 10:
            # Run PCA on masked, background-subtracted data
            pca = PCA(n_components=5)
            H = pca.fit_transform(data.T)
        
            # ===============================
            # Project PCA to Calculate Residuals
            # ===============================
            image = np.zeros((mask.shape[0], mask.shape[1], H.shape[1]))
            for i in np.arange(len(dd)):
                for j in np.arange(len(dr)):
                    data = (FL[i, j] - avg_back) / EL[i, j]
                    cont = get_continuum(data[np.newaxis, :], nbins=25)[0]
                    if np.isnan(data[20:-20]).sum() < 100:
                        sel = np.isfinite((data-cont)[20:-20])
                        sol = np.linalg.lstsq(H[sel], (data-cont)[20:-20][sel])[0]
                        res[i, j, 20:-20] = (np.dot(sol, H.T) * EL[i, j, 20:-20] + avg_back[20:-20])
                        image[i, j] = sol
    
        FL1 = FL - res

        # ===============================
        # Construct S/N Maps
        # ===============================
        log.info('Getting S/N map for %i, %i' % (xi, xj))
        SN1 = np.zeros((len(dr), len(dd), 3001))
        avgratio = mad_std(FL1[mask] / EL[mask], axis=(0,), ignore_nan=True)
        avgratio = get_continuum(avgratio[np.newaxis, :], nbins=11)[0]
    
        for i in np.arange(len(dd)):
            for j in np.arange(len(dr)):
                y = FL1[i, j] - get_continuum(FL1[i, j][np.newaxis, :], nbins=25)[0]
                y[np.isnan(y)] = 0.0
                e = EL[i, j].copy()
                e[np.isnan(e)] = 0.0
                Gc = Gaussian1DKernel(5.8 / 2.35 / 2.)
                WS = manual_convolution(y, Gc)
                WE = manual_convolution(e, Gc, error=True)
                ratio = mad_std((WS / WE)[25:-25], ignore_nan=True)
                WE *= ratio
                I = interp1d(wave[25:-25], (WS / WE)[25:-25], kind='quadratic', bounds_error=False)
                xn = np.linspace(3560, 5480, 3001)
                SN1[i, j] = I(xn)
        
        DR, DD = np.meshgrid(dr, dd)
        
        # Log memory usage for current grid point
        process = psutil.Process(os.getpid())
        log.info('Memory Used for  %i, %i: %0.2f GB' % 
                 (xi, xj, process.memory_info()[0] / 1e9))
        
        # ===============================
        # Detection Emission Lines
        # ===============================
        log.info('Getting detections for %i, %i' % (xi, xj))
        locs = find_detections(SN1, threshold=5.)
        
        # Initialize detection result containers
        r, d, w, flux, chi2norm, gmag, sn = ([], [], [], [], [], [], [])
    
        # ===============================
        # Gather Emission Line Info
        # ===============================
        for loc in locs:
            # Convert pixel coordinates to celestial coordinates
            r.append(ra_center + DR[loc[0], loc[1]] / 3600. / np.cos(np.deg2rad(dec_center)))
            d.append(dec_center + DD[loc[0], loc[1]] / 3600.)
            w.append(np.interp(loc[2], np.arange(len(xn)), xn))
            
            # Fit emission line profile and compute chi2, flux, S/N
            spec = FL1[loc[0], loc[1]]
            G = Gaussian1D(mean=w[-1], stddev=5.4/2.35)
            G.stddev.bounds = (5.0/2.35, 15.0/2.35)
            G.mean.bounds = (w[-1]-2., w[-1]+2.)
            wsel = (np.abs(wave - w[-1]) < 20.) * np.isfinite(spec)
            cont = get_continuum(spec[np.newaxis, :], nbins=25)[0]
            fit = fitter(G, wave[wsel], (spec-cont)[wsel])
            num = ((spec-cont)[wsel] - fit(wave[wsel])) / EL[loc[0], loc[1]][wsel]
            chi2norm.append(np.sum(num**2) / (len(num) - 3.))
            flux.append(np.sum(fit(wave[wsel]) * 2.))
            sn.append(SN1[loc[0], loc[1], loc[2]])
            
            # Convert location to WCS coordinates for photometry
            positions = []
            positions.append(wcs.wcs_world2pix(r[-1], d[-1], 1))
            aperture = CircularAperture(positions, r=3.)
            phot_table = aperture_photometry(cfht[0].data, aperture)
            gmag.append(-2.5 * np.log10(phot_table['aperture_sum'][0]*1/0.186**2) + 30.)
            
        
        # ===============================
        # Write Info out to a MultiFITS File
        # ===============================
        T = Table([r, d, w, sn, flux, chi2norm, gmag], names=['RA', 'Dec', 'wave',
                                                              'sn', 'Flux',
                                                              'Chi2', 'gmag'])
        sspectra = FL1[locs[:, 0], locs[:, 1]]
        serror = EL[locs[:, 0], locs[:, 1]]
        sres = res[locs[:, 0], locs[:, 1]]
        L = fits.HDUList([fits.PrimaryHDU(sspectra), fits.ImageHDU(serror),
                          fits.ImageHDU(sres), fits.BinTableHDU(T),
                          fits.ImageHDU(FL1), fits.ImageHDU(EL), fits.ImageHDU(res),
                          fits.ImageHDU(SN1)])
        L.writeto(outname, overwrite=True)
        return None

TOTAL_TASKS = 400
NUM_WORKERS = 64  # adjust depending on CPU/memory

with multiprocessing.Pool(processes=NUM_WORKERS) as pool:
    results = pool.map(run_detection, range(TOTAL_TASKS))
