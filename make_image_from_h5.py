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
from scipy.interpolate import griddata
from math_utils import biweight

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

parser.add_argument("h5file",
                    help='''e.g., 20200430_0000020.h5''',
                    type=str)

parser.add_argument("-we", "--wave_extract",
                    help='''Central Wavelength and Sigma"''',
                    default="5010.0, 2.6", type=str)

parser.add_argument("-bw", "--back_wave",
                    help='''Background Wavelength Range"''',
                    default="5150,5350", type=str)

parser.add_argument("-ps", "--pixel_scale",
                    help='''Pixel scale for output image in arcsec''',
                    default=1.0, type=float)

parser.add_argument("-ics", "--image_center_size",
                    help='''RA, Dec center of image and size (deg, deg, arcmin)
                            "150.0, 50.0, 22.0"''',
                    default=None, type=str)

args = parser.parse_args(args=None)
args.log = setup_logging('make_image_from_h5')

def_wave = np.linspace(3470., 5540., 1036)
wave_extract = [float(i.replace(' ', '')) for i in args.wave_extract]
back_wave = [float(i.replace(' ', '')) for i in args.back_wave]
bsel = (def_wave >= back_wave[0]) * (def_wave <= back_wave[1])
wsel = np.abs(def_wave - wave_extract[0]) < (2. * wave_extract[1])
Gmodel = np.exp(-0.5 * (def_wave - wave_extract[0])**2 / wave_extract[1]**2)
Gmodel[:] = Gmodel / Gmodel[wsel].sum()

t = tables.open_file(args.h5file)
ra = t.root.Info.cols.ra[:]
dec = t.root.Info.cols.dec[:]
RA = t.root.Survey.cols.ra[0]
Dec = t.root.Survey.cols.dec[0]
offset = t.root.Survey.cols.offset[0]
spectra = t.root.Fibers.cols.spectrum[:] / offset

if args.bounding_box is not None:
    bounding_box = [float(corner.replace(' ', ''))
                    for corner in args.bounding_box.split(',')]
else:
    bounding_box = [RA, Dec, 22.]

bounding_box[2] = int(bounding_box[2]*60./args.pixel_scale/2.) * 2 * args.pixel_scale

bb = int(bounding_box[2]*60./args.pixel_scale/2.)*args.pixel_scale
N = int(bounding_box[2]*60./args.pixel_scale/2.) * 2 + 1
A = Astrometry(np.mean(bounding_box[0]), np.mean(bounding_box[1]), 0.)
tp = A.setup_TP(A.ra0, A.dec0, 0.)
header = tp.to_header()
x, y = tp.wcs_world2pix(ra, dec, 1)
xg = np.linspace(-bb, bb, N)
yg = np.linspace(-bb, bb, N)
xgrid, ygrid = np.meshgrid(xg, yg)
Pos = np.zeros((len(x), 2))
Pos[:, 0], Pos[:, 1] = (x, y)
back = biweight(spectra[:, bsel], axis=1)
data = (spectra[:, wsel] - back) * 2.
mask = np.array(np.isfinite(data), dtype=float)
weight = Gmodel[np.newaxis, :]
z = np.nansum(mask * data * weight, axis=1) / np.nansum(mask * weight**2, axis=1)
good = np.isfinite(z) 
image = griddata(Pos[good], z[good], (xgrid, ygrid), kind='linear')
name = op.basename(args.h5file[:-3]) + '_collapse.fits'
F = fits.PrimaryHDU(np.array(image, 'float32'), header=header)
F.writeto(name, overwrite=True)