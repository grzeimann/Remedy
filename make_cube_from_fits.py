#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 08:46:45 2024

@author: grz85
"""
from astropy.convolution import convolve, Gaussian2DKernel
from astropy.io import fits
import numpy as np
import os.path as op
import warnings
from astrometry import Astrometry
from extract import Extract
import glob
from input_utils import setup_logging

log = setup_logging()

warnings.filterwarnings("ignore")

def make_image_interp(Pos, y, ye, xg, yg, xgrid, ygrid, sigma, cnt_array,
                      binsize=2):
    # Loop through shots, average images before convolution
    # Make a +/- 1 pixel mask for places to keep
    # Mask still needs work, need to grab bad pixels and block them out
    G = Gaussian2DKernel(sigma)
    image_all = np.ma.zeros((len(cnt_array),) + xgrid.shape)
    image_all.mask = True
    weightk = 0.0 * xgrid
    xc = np.interp(Pos[:, 0], xg, np.arange(len(xg)), left=0., right=len(xg))
    yc = np.interp(Pos[:, 1], yg, np.arange(len(yg)), left=0., right=len(yg))
    xc = np.array(np.round(xc), dtype=int)
    yc = np.array(np.round(yc), dtype=int)
    gsel = np.where((xc>1) * (xc<len(xg)-1) * (yc>1) * (yc<len(yg)-1))[0]
    for k, cnt in enumerate(cnt_array):
        weight = 0.0 * xgrid
        l1 = int(cnt[0])
        l2 = int(cnt[1])
        gi = gsel[(gsel>=l1) * (gsel<=l2)]
        image_all.data[k, yc[gi], xc[gi]] = y[gi] / (np.pi * 0.75**2)
        image_all.mask[k, yc[gi], xc[gi]] = False
        bsel = np.where(np.isfinite(y[gi]))[0]
        for i in np.arange(-1, 2):
            for j in np.arange(-1, 2):
                weight[yc[gi][bsel]+i, xc[gi][bsel]+j] = 1.
        bsel = np.where(np.isnan(y[gi]))[0]
        for i in np.arange(-1, 2):
            for j in np.arange(-1, 2):
                weight[yc[gi][bsel]+i, xc[gi][bsel]+j] = 0.0
        weightk[:] += weight
    image = np.ma.median(image_all, axis=0)
    y = image.data * 1.
    y[image.mask] = np.nan
    image = convolve(y, G, preserve_nan=False, boundary='extend')
    image[weightk == 0.] = np.nan
    image[np.isnan(image)] = 0.0

    return image, 0. * image, weight

def_wave = np.linspace(3470, 5540, 1036)

image_center_size = "23.45, 30.67, 16.4, 21.6, 1."
bounding_box = [float(corner.replace(' ', ''))
                        for corner in image_center_size.split(',')]
pixel_scale = bounding_box[4]
bounding_box[2] = int(bounding_box[2]*60./pixel_scale/2.) * 2 * pixel_scale
bounding_box[3] = int(bounding_box[3]*60./pixel_scale/2.) * 2 * pixel_scale

Nx = int(bounding_box[2]/pixel_scale*2.) + 1
Ny = int(bounding_box[3]/pixel_scale*2.) + 1
bbx = bounding_box[2] 
bby = bounding_box[3] 

print('Image size in pixels: (%i, %i)' % (Nx, Ny))
xg = np.linspace(-bbx, bbx, Nx)
yg = np.linspace(-bby, bby, Ny)
xgrid, ygrid = np.meshgrid(xg, yg)

pa = 259.3424

cube = np.zeros((len(def_wave),) + xgrid.shape, dtype='float32')
ecube = np.zeros((len(def_wave),) + xgrid.shape, dtype='float32')

A = Astrometry(bounding_box[0], bounding_box[1], 0., 0., 0.)
tp = A.setup_TP(A.ra0, A.dec0, 0.)

header = tp.to_header()

h5files = sorted(glob.glob('/Users/grz85/work/Backup/cure/Remedy/M33/2*.fits'))

cnt = 0
cnt_array = np.zeros((len(h5files), 2), dtype=int)
for i, h5file in enumerate(h5files):
    f = fits.open(h5file)
    ra = f[3].data[:, 0] * 1.
    cnt += len(ra)
    cnt_array[i, 0] = cnt - len(ra)
    cnt_array[i, 1] = cnt
    
raarray = np.zeros((cnt, len(def_wave)), dtype='float32')
decarray = np.zeros((cnt, len(def_wave)), dtype='float32')
specarray = np.zeros((cnt, len(def_wave)), dtype='float32')
errarray = np.zeros((cnt, len(def_wave)), dtype='float32')

cnt = 0
for jk, h5file in enumerate(h5files):
    f = fits.open(h5file)

    ra = f[3].data[:, 0] * 1.
    dec = f[3].data[:, 1] * 1.
    
    spectra = f[0].data
    error = f[1].data
    cnt1 = cnt + len(ra)
    E = Extract()
    Aother = Astrometry(bounding_box[0], bounding_box[1], pa, 0., 0.)
    E.get_ADR_RAdec(Aother)
    raarray[cnt:cnt1, :] = ra[:, np.newaxis] - E.ADRra[np.newaxis, :] / 3600. / np.cos(np.deg2rad(A.dec0))
    decarray[cnt:cnt1, :] = dec[:, np.newaxis] - E.ADRdec[np.newaxis, :] / 3600.
    specarray[cnt:cnt1, :] = spectra
    errarray[cnt:cnt1, :] = error
    cnt += len(ra)

Pos = np.zeros((len(raarray), 2))
for i in np.arange(len(def_wave)):
    x, y = tp.wcs_world2pix(raarray[:, i], decarray[:, i], 1)
    log.info('Working on wavelength %0.0f' % def_wave[i])
    Pos[:, 0], Pos[:, 1] = (x, y)
    data = specarray[:, i]
    edata = errarray[:, i]
    image, errorimage, weight = make_image_interp(Pos, data, edata, xg, yg, 
                                                  xgrid, ygrid, 1.8 / 2.35,
                                                  cnt_array)
    cube[i, :, :] += image
    ecube[i, :, :] += errorimage

name = op.basename('%s_cube.fits' % 'M33')
header['CRPIX1'] = (Nx+1) / 2
header['CRPIX2'] = (Ny+1) / 2
header['CDELT3'] = 2.
header['CRPIX3'] = 1.
header['CRVAL3'] = 3470.
F = fits.PrimaryHDU(np.array(cube, 'float32'), header=header)
F.writeto(name, overwrite=True)
name = op.basename('%s_errorcube.fits' % 'M33')
header['CRPIX1'] = (Nx+1) / 2
header['CRPIX2'] = (Ny+1) / 2
header['CDELT3'] = 2.
header['CRPIX3'] = 1.
header['CRVAL3'] = 3470.
F = fits.PrimaryHDU(np.array(ecube, 'float32'), header=header)
F.writeto(name, overwrite=True)