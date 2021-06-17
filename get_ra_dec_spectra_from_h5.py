#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 11:14:33 2020

@author: gregz
"""


import tables
import sys
from astropy.io import fits
from astropy.table import Table
import os.path as op
import numpy as np
import subprocess

filename = sys.argv[1]

base = '/work2/00115/gebhardt/maverick/gettar'

def get_structaz(filename):
    date = filename.split('_')[0]
    obs = filename.split('_')[1]
    fn = op.join(base, '%ssci' % date[:6])
    print(fn)
    process = subprocess.Popen('cat %s | grep %s | grep %s ' %
                                       (fn, date, obs),
                                       stdout=subprocess.PIPE, shell=True)
    line = process.stdout.readline()
    b = line.rstrip().decode("utf-8")
    print(b)
    b = b.split(' ')[7]
    print(b)
    return b

structaz = get_structaz(filename)
t = tables.open_file(filename)
ra = getattr(t.root, 'Info').cols.ra[:]
dec = getattr(t.root, 'Info').cols.dec[:]
fwhm =getattr(t.root, 'Survey').cols.fwhm[0]
specid = getattr(t.root, 'Info').cols.specid[:]
ifuid = getattr(t.root, 'Info').cols.ifuid[:]
ifuslot = getattr(t.root, 'Info').cols.ifuslot[:]
amp = getattr(t.root, 'Info').cols.amp[:]
N = np.arange(len(amp))
strings = []
for s, ifid, ifsl, ap, n in zip(specid, ifuid, ifuslot, amp, N):
    st = '%03d_%03d_%03d_%s_%03d' % (s, ifsl, ifid, ap, int(n % 112))
    strings.append(st)

spectra = getattr(getattr(t.root, 'Fibers').cols, 'spectrum')[:]
error = getattr(getattr(t.root, 'Fibers').cols, 'error')[:]

spectra[np.isnan(spectra)] = 0.
error[np.isnan(error)] = 0.

T = Table([ra, dec, strings], names=['ra', 'dec', 'multiname'])
h = fits.PrimaryHDU()
h.header['fwhm'] = fwhm
h.header['structaz'] = structaz
L = fits.HDUList([h, fits.ImageHDU(spectra),
                  fits.ImageHDU(error), fits.BinTableHDU(T)])