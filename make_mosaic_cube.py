#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 13:10:11 2020

@author: gregz
"""

import argparse as ap
from astropy.io import fits
import tables
import numpy as np
import os.path as op
import sys
import warnings
from input_utils import setup_logging
from astrometry import Astrometry
from math_utils import biweight
from extract import Extract
import glob

def get_script_path():
    return op.dirname(op.realpath(sys.argv[0]))

warnings.filterwarnings("ignore")

DIRNAME = get_script_path()

parser = ap.ArgumentParser(add_help=True)

parser.add_argument("-d", "--directory",
                    help='''base directory for reductions''',
                    type=str, default="")

parser.add_argument("-c", "--caldirectory",
                    help='''cal directory for reductions''',
                    type=str, default="/work/03946/hetdex/maverick/LRS2/CALS")

parser.add_argument("h5files",
                    help='''e.g., 20200430_0000020.h5''',
                    type=str)

parser.add_argument("surname",
                    help='''file name modification''',
                    type=str)

parser.add_argument("image_center_size",
                    help='''RA, Dec center of image and size (deg, deg, arcmin)
                            "150.0, 50.0, 22.0"''',
                    default=None, type=str)

parser.add_argument("-ps", "--pixel_scale",
                    help='''Pixel scale for output image in arcsec''',
                    default=1.0, type=float)


def make_image(Pos, y, xg, yg, xgrid, ygrid, sigma):
    image = xgrid * 0.
    weight = xgrid * 0.
    N = int(sigma*5)
    indx = np.searchsorted(xg, Pos[:, 0])
    indy = np.searchsorted(yg, Pos[:, 1])
    nogood = ((indx < N) + (indx >= (len(xg) - N)) +
              (indy < N) + (indy >= (len(yg) - N)))
    nogood = nogood + np.isnan(y)
    good = np.where(~nogood)[0]
    indx_array = np.zeros((len(good), 4*N**2), dtype=int)
    indy_array = np.zeros((len(good), 4*N**2), dtype=int)
    for i in np.arange(2*N):
        for j in np.arange(2*N):
            c = i * 2 * N + j
            indx_array[:, c] = indx[good] + i - N
            indy_array[:, c] = indy[good] + j - N
    d = np.sqrt((xgrid[indy_array, indx_array] - Pos[good, 0][:, np.newaxis])**2 + 
                (ygrid[indy_array, indx_array] - Pos[good, 1][:, np.newaxis])**2)
    G = np.exp(-0.5 * d**2 / sigma**2)
    G[:] /= G.sum(axis=1)[:, np.newaxis]
    for j, i in enumerate(good):
        image[indy_array[j], indx_array[j]] += y[i] * G[j]
        weight[indy_array[j], indx_array[j]] += G[j]
    weight[:] *= np.pi * 0.75**2
    image[weight < 0.4] = 0.
    return image / weight

args = parser.parse_args(args=None)
args.log = setup_logging('make_image_from_h5')

def_wave = np.linspace(3470., 5540., 1036)

h5files = glob.glob(args.h5files)

bounding_box = [float(corner.replace(' ', ''))
                        for corner in args.image_center_size.split(',')]

bounding_box[2] = int(bounding_box[2]*60./args.pixel_scale/2.) * 2 * args.pixel_scale

bb = int(bounding_box[2]/args.pixel_scale/2.)*args.pixel_scale
N = int(bounding_box[2]/args.pixel_scale/2.) * 2 + 1
args.log.info('Image size in pixels: %i' % N)
xg = np.linspace(-bb, bb, N)
yg = np.linspace(-bb, bb, N)
xgrid, ygrid = np.meshgrid(xg, yg)

ralist = []
declist = []
speclist = []
errorlist = []

for h5file in h5files:
    t = tables.open_file(args.h5file)
    ra = t.root.Info.cols.ra[:]
    dec = t.root.Info.cols.dec[:]
    RA = t.root.Survey.cols.ra[0]
    Dec = t.root.Survey.cols.dec[0]
    pa = t.root.Survey.cols.pa[0]
    offset = t.root.Survey.cols.offset[0]
    spectra = t.root.Fibers.cols.spectrum[:] / offset
    error = t.root.Fibers.cols.error[:] / offset

    A = Astrometry(bounding_box[0], bounding_box[1], 0., 0., 0.)
    tp = A.setup_TP(A.ra0, A.dec0, 0.)
    E = Extract()
    Aother = Astrometry(bounding_box[0], bounding_box[1], pa, 0., 0.)
    header = tp.to_header()
    E.get_ADR_RAdec(Aother)
    ralist.append(ra[:, np.newaxis] -
                  E.ADRra[np.newaxis, :] / 3600. / np.cos(np.deg2rad(A.dec0)))
    declist.append(dec[:, np.newaxis] - E.ADRdec[np.newaxis, :] / 3600.)
    speclist.append(spectra)
    t.close()

raarray = np.hstack(ralist)
decarray = np.hstack(declist)
specarray = np.vstack(speclist)
cube = np.zeros((len(def_wave),) + xgrid.shape, dtype='float32')
Pos = np.zeros((len(raarray), 2))
for i in np.arange(len(def_wave)):
    x, y = tp.wcs_world2pix(raarray[:, i], decarray[:, i], 1)
    Pos[:, 0], Pos[:, 1] = (x, y)
    data = specarray[:, i]
    good = np.isfinite(data)
    args.log.info('Started wavelength %0.1f' % def_wave[i])
    cube[i, :, :] = make_image(Pos[good], data[good], xg, yg, xgrid, ygrid, 1.5 / 2.35)


name = op.basename(args.h5file[:-3]) + ('_%s_cube.fits' % args.surname)
header['CRPIX1'] = (N+1) / 2
header['CRPIX2'] = (N+1) / 2
header['CDELT3'] = 2.
header['CRPIX3'] = 1.
header['CRVAL3'] = 3470.
F = fits.PrimaryHDU(np.array(cube, 'float32'), header=header)
F.writeto(name, overwrite=True)