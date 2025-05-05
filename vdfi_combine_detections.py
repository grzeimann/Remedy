#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  1 13:32:26 2025

@author: grz85
"""
from astropy.io import fits
import numpy as np
from astropy.wcs import WCS
import glob
import os.path as op
from input_utils import setup_logging
import warnings
from astropy.table import Table
from astropy.table import vstack

# Turn off annoying warnings (even though some deserve attention)
warnings.filterwarnings("ignore")

# Set up logging for tracking progress and issues
log = setup_logging('vdfi')

# Load reference images
cfht = fits.open('/work/03730/gregz/ls6/PHATTER-VIRUS/CFHT_EGS_image.fits', memmap=True)
header = cfht[0].header

# Get world coordinates (RA, Dec)
wcs = WCS(header)

# Set up a coarse grid in pixel space and convert to world coordinates
xg = np.arange(30.5, 1230.5, 60)
xgrid, ygrid = np.meshgrid(xg, xg)
grid_ra, grid_dec = wcs.all_pix2world(xgrid + 1., ygrid + 1., 1)

filenames = glob.glob('/work/03730/gregz/ls6/PHATTER-VIRUS/detections/detect*.fits')
N = cfht[0].data.shape[0]
vdfi_cube = np.zeros((N, N, 1036), dtype=np.float32)
vdfi_error = np.zeros((N, N, 1036), dtype=np.float32)
vdfi_sn = np.zeros((N, N, 3001), dtype=np.float32)
tables = []
for filename in filenames:
    n = op.basename(filename)
    xi, xj = (int(n.split('_')[2]), int(n.split('_')[3][:2]))
    with fits.open(filename, memmap=True) as hdulist:
        log.info('Loaded %02d, %02d' % (xi, xj))
        # Extract RA/Dec center from input grids
        x = int(xgrid[xi, xj] - 29.5)
        y = int(ygrid[xi, xj] - 29.5)
        vdfi_cube[y:y+60,x:x+60] = hdulist[4].data[1:-1,1:-1] * 1.
        vdfi_error[y:y+60,x:x+60] = hdulist[5].data[1:-1,1:-1] * 1.
        vdfi_sn[y:y+60,x:x+60] = hdulist[6].data[1:-1,1:-1] * 1.
        tables.append(Table(hdulist[3].data))

combined_table = vstack(tables)
combined_table.write('all_detections.txt', format='ascii.fixed_width_two_line',
                     overwrite=True)

header = wcs.to_header()


surname = 'VDFI_EGS'

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
F = fits.PrimaryHDU(vdfi_cube, header=header)
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
F = fits.PrimaryHDU(vdfi_error, header=header)
F.writeto(name, overwrite=True)
name = op.basename('%s_sncube.fits' % surname)
header['CRPIX1'] = (N+1) / 2.
header['CRPIX2'] = (N+1) / 2.
header['CDELT3'] = 0.64
header['CRPIX3'] = 1.
header['CRVAL3'] = 3560.
header['CTYPE1'] = 'RA---TAN'
header['CTYPE2'] = 'DEC--TAN'
header['CTYPE3'] = 'WAVE'
header['CUNIT1'] = 'deg'
header['CUNIT2'] = 'deg'
header['CUNIT3'] = 'Angstrom'
header['SPECSYS'] = 'TOPOCENT'
F = fits.PrimaryHDU(vdfi_sn, header=header)
F.writeto(name, overwrite=True)