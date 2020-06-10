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

filename = sys.argv[1]
attr = sys.argv[2]

t = tables.open_file(filename)
specid = t.root.Cals.cols.specid[:]
spectra = getattr(t.root.Cals.cols, attr)[:]
us = np.sort(np.unique(specid))
S = spectra * 0.
cnt = 0
for s in us:
    sel = np.where(specid == s)[0]
    n = len(sel)
    S[cnt:cnt+n] = spectra[sel]
    cnt += n
nexp = 1
inds = np.arange(S.shape[0])
for k in np.arange(nexp):
    sel = np.where(np.array(inds / 112, dtype=int) % nexp == k)[0]
    fits.PrimaryHDU(np.array(S[sel], dtype='float32')).writeto(
            '%s_%s.fits' % (filename[:-3], attr), overwrite=True)