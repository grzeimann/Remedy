#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 14 09:48:10 2020

@author: gregz
"""

import tables
import sys
import numpy as np
from astropy.io import fits

filename = sys.argv[1]

t = tables.open_file(filename)
specid = t.root.Info.cols.specid[:]
spectra = t.root.Fibers.cols.spectrum[:] * 2.
us = np.sort(np.unique(specid))
S = spectra * 0.
cnt = 0
for s in us:
    sel = np.where(specid == s)[0]
    n = len(sel)
    S[cnt:cnt+n] = spectra[sel]
    cnt += n
nexp = len(t.root.Survey)
inds = np.arange(S.shape[0])
for k in np.arange(nexp):
    sel = np.where(np.array(inds / 112, dtype=int) % nexp == k)[0]
    fits.PrimaryHDU(np.array(S[sel], dtype='float32')).writeto(
            '%s_skysub_exp%02d.fits' % (filename[:-3], k+1), overwrite=True)