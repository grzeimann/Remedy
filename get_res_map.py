#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 09:05:20 2020

@author: gregz
"""

import glob
import tables
import numpy as np
import os.path as op
from math_utils import biweight
from astropy.io import fits

filenames = sorted(glob.glob('/work/03730/gregz/maverick/20191214*h5'))
cnt = 0
for j, filename in enumerate(filenames):
    T = tables.open_file(filename)
    ifuslots = T.root.Info.cols.ifuslot[:]
    ui = np.unique(ifuslots)
    nfibers = int(len(ui) * 448)
    nexp = int(len(ifuslots) / nfibers)
    print('%s has %i exposures' % (op.basename(filename), nexp))
    cnt += nexp
res_map = np.zeros((nfibers, cnt))
cnt = 0
for j, filename in enumerate(filenames):
    print('Working on %s' % op.basename(filename))
    T = tables.open_file(filename)
    ifuslots = T.root.Info.cols.ifuslot[:]
    ui = np.unique(ifuslots)
    nfibers = int(len(ui) * 448)
    nexp = int(len(ifuslots) / nfibers)
    sci = T.root.Fibers.cols.sci[:]
    sky = T.root.fibers.cols.sky[:]
    inds = np.arange(len(ifuslots))
    for k in np.arange(nexp):
        sel = np.where(np.array(inds / 112, dtype=int) % nexp == k)[0]
        res_map[:, cnt] = biweight(sci[sel, 700:800] / sky[sel, 700:800], axis=1)
        cnt += 1
fits.PrimaryHDU(res_map).writeto('test.fits', overwrite=True)