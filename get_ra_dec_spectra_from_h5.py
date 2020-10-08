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

filename = sys.argv[1]

t = tables.open_file(filename)
ra = getattr(t.root, 'Info').cols.ra[:]
dec = getattr(t.root, 'Info').cols.dec[:]

spectra = getattr(getattr(t.root, 'Fibers').cols, 'spectrum')[:]
error = getattr(getattr(t.root, 'Fibers').cols, 'error')[:]

T = Table([ra, dec], names=['ra', 'dec'])
L = fits.HDUList([fits.PrimaryHDU(), fits.BinTableHDU(T), fits.ImageHDU(spectra),
                  fits.ImageHDU(error)])
fitsname = op.basename(filename).split('.h5')[0] + '.fits'
L.writeto(fitsname, overwrite=True)
t.close()