#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 17 19:58:46 2025

@author: grz85
"""

import matplotlib.pyplot as plt
import numpy as np
import time
import warnings

from astropy.nddata import Cutout2D
from astropy.convolution import Gaussian2DKernel
from astropy.convolution import convolve, Gaussian1DKernel
from astropy.io import fits
from astropy.modeling.models import Moffat2D
from scipy.ndimage import maximum_filter
from astropy.stats import mad_std
from astropy.stats import biweight_location as biweight
from astropy.wcs import WCS
from input_utils import setup_logging
from scipy.interpolate import LinearNDInterpolator, interp1d
from sklearn.decomposition import PCA
import seaborn as sns

warnings.filterwarnings("ignore")


sns.set_context('talk')
sns.set_style('ticks')

plt.rcParams['font.serif'] = ['P052']
plt.rcParams['font.family'] = 'serif'


def manual_convolution(a, G, error=False):
    b = np.zeros((len(a), len(G.array)))
    N = int(len(G.array) / 2. - 0.5)
    M = len(G.array)
    for i in np.arange(0, N):
        b[(N-i):, i] = a[:-(N-i)]
    b[:, N] = a
    for i in np.arange(N+1, M):
        b[:-(i-N), i] = a[(i-N):]
    if error:
        return np.sqrt(np.nansum(b**2 * G.array**2, axis=1))
    return np.nansum(b * G.array, axis=1)

def pca_fit(H, data):
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
    # Step 1: Create a boolean mask for voxels above the threshold
    above_threshold = sn_cube > threshold

    # Step 2: Find local maxima using a maximum filter
    # This will compare each voxel to its 26 neighbors (3x3x3 cube, excluding itself)
    local_max = (sn_cube == maximum_filter(sn_cube, size=3, mode='constant', 
                                           cval=0))

    # Step 3: Combine both conditions
    result_mask = above_threshold & local_max

    # Optional: Get coordinates of these local maxima
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
    Seeing = np.ones((r.shape[0],))[:, np.newaxis] * seeing[np.newaxis, :] # create a 2D array of the seeing value
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
    
log = setup_logging('vdfi')

wave = np.linspace(3470, 5540, 1036)

f = fits.open('/work/03730/gregz/maverick/VDFI/all_info.fits')

RA = f[1].data * 1.
DEC = f[2].data * 1.
RA, DEC = [x.reshape((x.shape[0], x.shape[1] * x.shape[2])) for x in [RA, DEC]]
nexp = RA.shape[0]

g = fits.open('/work/03730/gregz/maverick/VDFI/all_flux_final.fits')
e = fits.open('/work/03730/gregz/maverick/VDFI/all_error.fits')

spectra = g[0].data
spectra = spectra.reshape((spectra.shape[0], 
                           spectra.shape[1] * spectra.shape[2], 
                           spectra.shape[3]))
error = e[0].data
error = error.reshape((error.shape[0], error.shape[1] * error.shape[2], 
                       error.shape[3]))


# Loading DAR
dar = fits.open('/work/03730/gregz/maverick/VDFI/all_initial_dar.fits')
dar_ra = dar[0].data
dar_dec = dar[1].data

radius = 3.75

r = np.linspace(0, radius, 101)
seeing = np.linspace(np.min(f[5].data)-0.05, np.max(f[5].data)+0.05, 51)
PSF, R, S, V = moffat_psf_integration(r, seeing)

cfht = fits.open('/work/03730/gregz/ls6/PHATTER-VIRUS/CFHT_EGS_image.fits')
vdfi = fits.open('/work/03730/gregz/ls6/PHATTER-VIRUS/VDFI_EGS_image.fits')

header = cfht[0].header

locs = (cfht[0].data < 0.001) * (vdfi[0].data != 0.0)
yind, xind = np.indices(cfht[0].data.shape)
wcs = WCS(header)
ra_blank, dec_blank = wcs.all_pix2world(xind[locs], yind[locs], 1)
ra_pix, dec_pix = wcs.all_pix2world(xind + 1., yind + 1., 1) # python to ds9

xg = np.arange(30.5, 1230.5, 60)
xgrid, ygrid = np.meshgrid(xg, xg)
grid_ra, grid_dec = wcs.all_pix2world(xgrid + 1., ygrid + 1., 1)
xs = [15, 16]
ys = [4, 4]
log.info('Starting Detections')

for xi, xj in zip(xs, ys):
    ra_center = grid_ra[xi, xj]
    dec_center = grid_dec[xi, xj]
    plt.figure(figsize=(8, 7.5))
    yl = int(ygrid[xi, xj]-30.5)
    yh = yl + 60
    xl = int(xgrid[xi, xj]-30.5)
    xh = xl + 60
    plt.imshow(vdfi[0].data[yl:yh, xl:xh], origin='lower', 
               vmin=-0.03, vmax=0.05, cmap=plt.get_cmap('coolwarm'))
    plt.savefig('Cutout_%i_%i.png' % (xi, xj))
    size = 30.
    step = 1.
    buffer = 4.5
    S = np.zeros((nexp, len(wave)))
    E = np.zeros_like(S)
    log.info('Making cutout for %i, %i' % (xi, xj))
    
    shortspec = np.zeros((spectra.shape[0], 448, 1036))
    shorterr = np.zeros((spectra.shape[0], 448, 1036))
    shortra = np.zeros((spectra.shape[0], 448))
    shortdec = np.zeros((spectra.shape[0], 448))
    
    npix = 0
    for exposure in np.arange(nexp):
        dra = (RA[exposure]-ra_center)* np.cos(np.pi/180.*dec_center) * 3600.
        ddec = (DEC[exposure] - dec_center) * 3600.
        fiber_sel = ((np.abs(dra) < (size+buffer))  * 
                     (np.abs(ddec) < (size+buffer)))
        if fiber_sel.sum():
            mask = (np.isfinite(spectra[exposure, :, 400:500]).sum(axis=1) > 50.)
            fiber_sel = fiber_sel * mask
            npix += 1
            N = fiber_sel.sum()
            shortspec[exposure, :N] = spectra[exposure, fiber_sel]
            shorterr[exposure, :N] = error[exposure, fiber_sel]
            shortra[exposure, :N] = RA[exposure, fiber_sel]
            shortdec[exposure, :N] = DEC[exposure, fiber_sel]
    
    if npix < 10:
        log.info('Not enough spectra in region ... moving to next grid point')
        continue
    
    log.info('Running Extraction for %i, %i' % (xi, xj))
    S = np.zeros((nexp, len(wave)))
    E = np.zeros_like(S)
    
    dd = np.arange(-(size+step/2.), (size+step), step)
    dr = np.arange(-(size+step/2.), (size+step), step)
    FL = np.zeros((len(dd), len(dr), 1036))
    EL = np.zeros((len(dd), len(dr), 1036))
    
    iratios = np.zeros((len(dr), len(dd)))
    for i in np.arange(len(dd)):
        for j in np.arange(len(dr)):
    
            def change_coordinates(dr, dd, ra_center, dec_center):
                ra_center += dr / 3600. / np.cos(np.deg2rad(dec_center))
                dec_center += dd / 3600.
                return ra_center, dec_center
    
            ra_center1, dec_center1 = change_coordinates(dr[j], dd[i], 
                                                         ra_center, dec_center)
    
            for exposure in np.arange(nexp):
                dra = (shortra[exposure]-ra_center1)* np.cos(np.pi/180.*dec_center1) * 3600.
                ddec = (shortdec[exposure] - dec_center1) * 3600.
                mask = np.isfinite(shortspec[exposure, :, 405])
                fiber_sel = np.sqrt(dra**2 + ddec**2) <= radius * mask
                seeing = f[5].data[exposure]
                newdra = dra[fiber_sel][:, np.newaxis] - dar_ra[exposure][np.newaxis, :]
                newddec = ddec[fiber_sel][:, np.newaxis] - dar_dec[exposure][np.newaxis, :]
                y = shortspec[exposure, fiber_sel]
                spectrum, spectrum_e, w, pw = get_spectrum_exposure(y, 
                                                shorterr[exposure, fiber_sel], 
                                                newdra, newddec, seeing, PSF, wave)
                S[exposure] = spectrum
                E[exposure] = spectrum_e
            Spec = biweight(S, axis=0, ignore_nan=True)
            ratio = mad_std(((S - Spec[np.newaxis]) / E)[:, 40:-25], ignore_nan=True, axis=1)
            iratios[i, j] = biweight(ratio, ignore_nan=True)
            E = E * ratio[:, np.newaxis]
            W = 1. / E**2
            Spec2 = np.nansum(S * W, axis=0) / np.nansum(W, axis=0)
            Err = np.sqrt(1. / np.nansum(W, axis=0))
            e = Err * 1.
            e[np.isnan(e)] = 0.0
            FL[i, j] = Spec2
            EL[i, j] = e
    
    log.info('Fitting PCA for %i, %i' % (xi, xj))
    
    mywcs = WCS(cfht[0].header)
    x, y = mywcs.wcs_world2pix(ra_center, dec_center, 1)
    position = (x-1, y-1)
    size_cut = (FL.shape[0], FL.shape[1])     # pixels
    cutout = Cutout2D(cfht[0].data, position, size_cut)
    mask = convolve(cutout.data[:, ::-1] > 0.2, Gaussian2DKernel(2.5/2.35)) < 0.1
    mask = mask * (np.isnan(FL[:, :, 20:-20]).sum(axis=2) < 1)
    avg_back = np.nanmean(FL[mask], axis=(0,))
    data = (FL[mask] - avg_back[np.newaxis, :]) / EL[mask]
    data = data[:, 20:-20]
    ncomponents = 5
    # Initialize PCA with the specified number of components
    pca = PCA(n_components=ncomponents)
    # Fit the PCA model and transform the data into the principal components space
    H = pca.fit_transform(data.T)
    image = np.zeros((mask.shape[0], mask.shape[1], H.shape[1]))
    rows, cols = np.where(mask+True)
    res = np.zeros_like(FL) + avg_back[np.newaxis, np.newaxis, :]
    for i in np.arange(len(dd)):
        for j in np.arange(len(dr)):
            data = (FL[i, j] - avg_back) / EL[i, j]
            cont = get_continuum(data[np.newaxis, :], nbins=25)[0]
            if np.isnan(data[20:-20]).sum() < 100:
                sel = np.isfinite((data-cont)[20:-20])
                sol = np.linalg.lstsq(H[sel], (data-cont)[20:-20][sel])[0]
                res[i, j, 20:-20] = (np.dot(sol, H.T) * EL[i, j, 20:-20] +
                                     avg_back[20:-20])
                image[i, j] = sol
                
    log.info('Getting S/N map for %i, %i' % (xi, xj))
    SN1 = np.zeros((len(dr), len(dd), 3001))
    FL1 = FL - res
    avgratio = mad_std(FL1[mask] / EL[mask], axis=(0,), ignore_nan=True)
    avgratio = get_continuum(avgratio[np.newaxis, :], nbins=11)[0]
    ratios = np.zeros((FL.shape[0], FL.shape[1]))
    for i in np.arange(len(dd)):
        for j in np.arange(len(dr)):
            y = FL1[i, j]
            cont2 = get_continuum(y[np.newaxis, :], nbins=25)[0]
            y = y - cont2
            y[np.isnan(y)] = 0.0
            e = EL[i, j] * 1.
            e[np.isnan(e)] = 0.0
            Gc = Gaussian1DKernel(5.8 / 2.35 / 2.)
            WS = manual_convolution(y, Gc)
            WE = manual_convolution(e, Gc, error=True)
            ratio = mad_std((WS / WE)[25:-25], ignore_nan=True)
            ratios[i, j] = ratio
            WE = WE * ratio
            I = interp1d(wave[25:-25], (WS / WE)[25:-25], kind='quadratic', 
                         bounds_error=False)
            xn = np.linspace(3560, 5480, 3001)
            V = I(xn)
            SN1[i, j] = V
            
    log.info('Getting detections for %i, %i' % (xi, xj))
    locs = find_detections(SN1, threshold=5.)
