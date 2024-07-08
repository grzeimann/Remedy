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
from multiprocessing import Pool

warnings.filterwarnings("ignore")


# Load the star normalizations and the Moffat seeing for each exposure
f = fits.open('/work/03730/gregz/maverick/VDFI/all_info.fits')

RA = f[1].data * 1.
DEC = f[2].data * 1.
nexp = f[1].data.shape[0]

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


# Loading DAR
dar = fits.open('/work/03730/gregz/maverick/VDFI/all_initial_dar.fits')
dar_ra = dar[0].data
dar_dec = dar[1].data

# Need astrometry for the field to conver a grid of points and get the ra and dec

field_ra = 214.9075
field_dec = 52.88

def_wave = np.linspace(3470, 5540, 1036)

log = setup_logging('vdfi')

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
    errorimage = np.sqrt(np.nansum(image_all_error**2, axis=0))
    return image, errorimage

    
P = Pool(8)
res = P.map(get_cube_image, np.arange(len(def_wave)))
P.close()
for i, return_list in enumerate(res):
    cube[i] = return_list[0]
    ecube[i] = return_list[1]

name = op.basename('%s_cube.fits' % surname)
header['CRPIX1'] = (N+1) / 2
header['CRPIX2'] = (N+1) / 2
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
header['CRPIX1'] = (N+1) / 2
header['CRPIX2'] = (N+1) / 2
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