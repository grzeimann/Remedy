#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 14:31:35 2020

@author: gregz
"""


import tables
import sys
import numpy as np
from astropy.io import fits
import os.path as op

filename = sys.argv[1]
tname1 = sys.argv[2]
tname2 = sys.argv[3]
attr = sys.argv[4]

t = tables.open_file(filename)
specid = getattr(t.root, tname1).cols.specid[:]
spectra = getattr(getattr(t.root, tname2).cols, attr)[:]
us = np.sort(np.unique(specid))
S = spectra * 0.
cnt = 0
for s in us:
    sel = np.where(specid == s)[0]
    n = len(sel)
    S[cnt:cnt+n] = spectra[sel]
    cnt += n
nexp = int(n / 448)
inds = np.arange(S.shape[0])
for k in np.arange(nexp):
    sel = np.where(np.array(inds / 112, dtype=int) % nexp == k)[0]
    fits.PrimaryHDU(np.array(S[sel], dtype='float32')).writeto(
            '%s_%s.fits' % (op.basename(filename)[:-3], attr), overwrite=True)