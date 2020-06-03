#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 12:54:50 2020

@author: gregz
"""

import tables
import numpy as np
from fiber_utils import get_powerlaw, get_spectra
import sys
from astropy.io import fits
import warnings
from scipy.interpolate import interp1d

def get_bigW(array_wave, array_trace, image):
    bigW = np.zeros(image.shape)
    Y, X = np.indices(array_wave.shape)
    YY, XX = np.indices(image.shape)
    for x, at, aw, xx, yy in zip(np.array_split(X, 2, axis=0),
                                 np.array_split(array_trace, 2, axis=0),
                                 np.array_split(array_wave, 2, axis=0),
                                 np.array_split(XX, 2, axis=0),
                                 np.array_split(YY, 2, axis=0)):
        for j in np.arange(at.shape[1]):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                p0 = np.polyfit(at[:, j], aw[:, j], 7)
            bigW[yy[:, j], j] = np.polyval(p0, yy[:, j])
    return bigW


t = tables.open_file(sys.argv[1])
amp = sys.argv[2]
ifuslot = int(sys.argv[3])

ifuslots = t.root.Cals.cols.ifuslot[:]
amps = np.array([i.decode("utf-8") for i in t.root.Cals.cols.amp[:]])
inds = np.where((ifuslots == ifuslot) * (amp == amps))[0]
if not len(inds):
    print('Could not find %03d %s' % (ifuslot, amp))
    sys.exit(1)

ind = inds[0]
trace = t.root.Cals.cols.trace[ind]
wave = t.root.Cals.cols.wavelength[ind]
masterbias = t.root.Cals.cols.masterbias[ind]
masterdark = t.root.Cals.cols.masterdark[ind]
masterflt = t.root.Cals.cols.masterflt[ind]
mastertwi = t.root.Cals.cols.mastertwi[ind]
mastersci = t.root.Cals.cols.mastersci[ind]

def_wave = np.linspace(3470., 5540., 1036*2)
names = ['masterflt', 'mastertwi', 'mastersci']
for i, master in enumerate([masterflt, mastertwi, mastersci]):
    master[:] -= masterbias
    if i == 2:
        master[:] -= masterdark
    plaw = get_powerlaw(master, trace)
    master[:] -= plaw
    spectra = get_spectra(master, trace)
    scirect = np.zeros((spectra.shape[0], len(def_wave)))
    for i in np.arange(spectra.shape[0]):
        scirect[i] = np.interp(def_wave, wave[i], spectra[i], left=np.nan,
                        right=np.nan)
    Avgspec = np.nanmedian(scirect)
    bigW = get_bigW(wave, trace, master)
    sel = np.isfinite(Avgspec)
    I = interp1d(def_wave[sel], Avgspec[sel], kind='quadratic',
                 fill_value='extrapolate')
    modelimage = I(bigW)
    flatimage = master / modelimage
    fits.PrimaryHDU(flatimage).writeto('%s_%03d%s.fits' %
                   (names[i], ifuslot, amp), overwrite=True)