#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 11:09:49 2024

@author: grz85
"""

import numpy as np
from astropy.io import fits
from astrometry import Astrometry
from input_utils import setup_logging
import os.path as op
import warnings
from astropy.convolution import convolve, Moffat2DKernel, Gaussian2DKernel
from photutils.aperture import CircularAperture
from photutils.aperture import aperture_photometry
from astropy.wcs import WCS
from astropy.nddata import Cutout2D
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.stats import biweight_location as biweight
from scipy.interpolate import interp1d

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
                         fill_value=np.nan, bounds_error=False)
            cont[i] = I(X)
        else:
            cont[i] = 0.0
    return cont

warnings.filterwarnings("ignore")

log = setup_logging('vdfi')

# Load the star normalizations and the Moffat seeing for each exposure
f = fits.open('/work/03730/gregz/maverick/VDFI/all_info.fits')

RA = f[1].data * 1.
DEC = f[2].data * 1.
nexp = f[1].data.shape[0]
namp = f[1].data.shape[1]

star_average = np.nanmedian(f[6].data, axis=0)
star_sel = np.abs(star_average - np.nanmedian(star_average)) < 0.15

x = np.arange(f[6].data.shape[0])
shot_average = np.nanmedian(f[6].data[:, star_sel], axis=1)

g = fits.open('/work/03730/gregz/maverick/VDFI/all_flux.fits')
e = fits.open('/work/03730/gregz/maverick/VDFI/all_error.fits')

spectra = g[0].data * 1.
error = e[0].data * 1.

# Correcting the normalization
for exposure in np.arange(nexp):
    spectra[exposure] = spectra[exposure] / shot_average[exposure]
    error[exposure] = error[exposure] / shot_average[exposure]
    
d = fits.open('/work/03730/gregz/maverick/VDFI/CFHTLS_D-85_g_141927+524056_T0007_MEDIAN.fits')


# Get the photometry in fiber apertures from the HST F475W data
Image = d[0].data
wcs_out = WCS(d[0].header)
pixscale = (np.sqrt(np.abs(wcs_out.pixel_scale_matrix[0, 0] * 
                           wcs_out.pixel_scale_matrix[1, 1])) * 3600.)


for exposure in np.arange(nexp):
    log.info('Working on exposure: %i' % (exposure+1))
    for amplifier in np.arange(namp):
        alpha = 3.5
        seeing = f[5].data[exposure]
        gamma = 0.5 * seeing / np.sqrt(2**(1./ alpha) - 1.)
        kernel = Moffat2DKernel(gamma=gamma, alpha=alpha)
        
        ra = f[1].data[exposure, amplifier]
        dec = f[2].data[exposure, amplifier]
        
        mra = np.mean(ra)
        mdec = np.mean(dec)
        if np.isna(mra):
            continue
        position = SkyCoord(mra * u.deg, mdec * u.deg)
        cutout = Cutout2D(Image, position, (320, 320), wcs=wcs_out)
        cutout.data = convolve(cutout.data, kernel, boundary='wrap')
        
        spec = spectra[exposure, amplifier] * 1.
        
        positions = []
        for r, d in zip(ra, dec):
            positions.append(cutout.wcs.wcs_world2pix(r, d, 1))
        aperture = CircularAperture(positions, r=0.75/pixscale)
        phot_table = aperture_photometry(cutout.data, aperture)

        z = phot_table['aperture_sum'] * 10**(0.4 * (23.9 - 30))
        z = z *  1e-29 * 3e18 / 4700**2 * 1e17
        sel = z > 0.003
        newspec = spec * 1.
        G = Gaussian2DKernel(3.)
        cont = get_continuum(newspec, nbins=25)
        cont[sel] = np.nan
        cont = convolve(cont, G, boundary='wrap')
        spectra[exposure, amplifier] = spectra[exposure, amplifier] - cont
        
fits.PrimaryHDU(spectra).writeto('/work/03730/gregz/maverick/VDFI/all_flux_final.fits', overwrite=True)



# Loading DAR
dar = fits.open('/work/03730/gregz/maverick/VDFI/all_initial_dar.fits')
dar_ra = dar[0].data
dar_dec = dar[1].data

# Need astrometry for the field to conver a grid of points and get the ra and dec

field_ra = 214.9075
field_dec = 52.88

def_wave = np.linspace(3470, 5540, 1036)


masked_amp_ids = np.array([40, 48, 49, 50, 51, 55, 56, 57, 58, 59, 106, 122, 
                           123, 131, 135, 146, 172, 173, 174, 175, 201, 227, 
                           280, 281, 282, 283, 299, 301], dtype=int)

bounding_box = [field_ra, field_dec, 20.0]

pixel_scale = 1.0

bounding_box[2] = int(bounding_box[2]*60./pixel_scale/2.) * 2 * pixel_scale

bb = int(bounding_box[2]/pixel_scale/2.)*pixel_scale
N = int(bounding_box[2]/pixel_scale/2.) * 2 + 1
log.info('Image size in pixels: %i' % N)
xg = np.linspace(-bb, bb, N)
yg = np.linspace(-bb, bb, N)
xgrid, ygrid = np.meshgrid(xg, yg)

cube = np.zeros((len(def_wave),) + xgrid.shape, dtype='float32')
ecube = np.zeros((len(def_wave),) + xgrid.shape, dtype='float32')

A = Astrometry(bounding_box[0], bounding_box[1], 0., 0., 0.)
tp = A.setup_TP(A.ra0, A.dec0, 0.)

header = tp.to_header()


surname = 'VDFI_EGS'

    
def get_cube_image(i):
    log.info('Working on wavelength %0.0f' % def_wave[i])
    image_all = np.ones((nexp,) + xgrid.shape) * np.nan
    image_all_error = np.ones((nexp,) + xgrid.shape) * np.nan
    for exposure in np.arange(nexp):
        ra = (RA[exposure] -  
              dar_ra[exposure, i] / 3600. / np.cos(np.deg2rad(A.dec0)))
        dec = (DEC[exposure] - dar_dec[exposure, i] / 3600.)
        x, y = tp.wcs_world2pix(ra, dec, 1)
        x, y = [j.ravel() for j in [x, y]]
        xc = np.interp(x, xg, np.arange(len(xg)), left=0., right=len(xg))
        yc = np.interp(y, yg, np.arange(len(yg)), left=0., right=len(yg))
        xc = np.array(np.round(xc), dtype=int)
        yc = np.array(np.round(yc), dtype=int)
        gsel = np.where((xc>1) * (xc<len(xg)-1) * (yc>1) * (yc<len(yg)-1))[0]
        image_all[exposure, yc[gsel], xc[gsel]] = spectra[exposure, :, :, i].ravel()[gsel] 
        image_all_error[exposure, yc[gsel], xc[gsel]] = error[exposure, :, :, i].ravel()[gsel] 
    image = np.nanmedian(image_all, axis=0)
    N = np.sum(np.isfinite(image_all_error), axis=0)
    errorimage = np.sqrt(np.nansum(image_all_error**2, axis=0) / N) 
    return image, errorimage

    
for i in np.arange(len(def_wave)):
    image, errorimage = get_cube_image(i)
    cube[i] = image
    ecube[i] = errorimage

name = op.basename('%s_cube.fits' % surname)
header['CRPIX1'] = (N+1) / 2.
header['CRPIX2'] = (N+1) / 2.
header['CDELT3'] = 2.
header['CRPIX3'] = 1.
header['CRVAL3'] = 3470.
header['CTYPE1'] = 'RA---TAN'
header['CTYPE2'] = 'DEC--TAN'
header['CTYPE3'] = 'WAVE'
header['CUNIT1'] = 'deg'
header['CUNIT2'] = 'deg'
header['CUNIT3'] = 'Angstrom'
header['SPECSYS'] = 'TOPOCENT'
F = fits.PrimaryHDU(np.array(cube, 'float32'), header=header)
F.writeto(name, overwrite=True)
name = op.basename('%s_errorcube.fits' % surname)
header['CRPIX1'] = (N+1) / 2.
header['CRPIX2'] = (N+1) / 2.
header['CDELT3'] = 2.
header['CRPIX3'] = 1.
header['CRVAL3'] = 3470.
header['CTYPE1'] = 'RA---TAN'
header['CTYPE2'] = 'DEC--TAN'
header['CTYPE3'] = 'WAVE'
header['CUNIT1'] = 'deg'
header['CUNIT2'] = 'deg'
header['CUNIT3'] = 'Angstrom'
header['SPECSYS'] = 'TOPOCENT'
F = fits.PrimaryHDU(np.array(ecube, 'float32'), header=header)
F.writeto(name, overwrite=True)