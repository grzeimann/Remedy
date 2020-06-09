#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 12:54:50 2020

@author: gregz
"""

import tables
import numpy as np
from fiber_utils import get_powerlaw, get_spectra, get_spectra_error
import sys
from astropy.io import fits
import warnings
from scipy.interpolate import interp1d
from astropy.convolution import convolve, Gaussian2DKernel

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


def get_bigF(array_trace, image):
    bigF = np.zeros(image.shape)
    Y, X = np.indices(array_trace.shape)
    YY, XX = np.indices(image.shape)
    n, m = array_trace.shape
    F0 = array_trace[:, int(m/2)]
    for j in np.arange(image.shape[1]):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            p0 = np.polyfit(array_trace[:, j], F0, 7)
        bigF[:, j] = np.polyval(p0, YY[:, j])
    return bigF, F0

t = tables.open_file(sys.argv[1])
amp = sys.argv[3]
ifuslot = int(sys.argv[2])

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
readnoise = t.root.Cals.cols.readnoise[ind]

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
    for j in np.arange(spectra.shape[0]):
        scirect[j] = np.interp(def_wave, wave[j], spectra[j], left=np.nan,
                        right=np.nan)
    Avgspec = np.nanmedian(scirect, axis=0)
    d1 = np.abs(Avgspec[1:-1] - Avgspec[:-2])
    d2 = np.abs(Avgspec[2:] - Avgspec[1:-1])
    ad = (d1 + d2) / 2.
    specdiff = np.hstack([ad[0], ad, ad[-1]])
    avgfiber = np.nanmedian(scirect / Avgspec, axis=1)
    bigW = get_bigW(wave, trace, master)
    bigF, F0 = get_bigF(trace, master)
    sel = np.isfinite(Avgspec)
    I = interp1d(def_wave[sel], Avgspec[sel], kind='quadratic',
                 fill_value='extrapolate')
    modelimage = I(bigW)
    J = interp1d(F0, avgfiber, kind='quadratic',
                 fill_value='extrapolate')
    modelimageF = J(bigF)
    flatimage = master / modelimage / modelimageF
    if i == 0:
        Flat = flatimage * 1.
    if i == 2:

        skyimage = Flat * modelimage * modelimageF
        skysub = master - skyimage
        error = np.sqrt(readnoise**2 + master)
        spec = get_spectra(skysub, trace)
        syserr = spec * 0.
        for k in np.arange(spec.shape[0]):
             dummy = np.interp(wave[k], def_wave, Avgspec, left=0.0,
                               right=0.0)
             d1 = np.abs(dummy[1:-1] - dummy[:-2])
             d2 = np.abs(dummy[2:] - dummy[1:-1])
             ad = (d1 + d2) / 2.
             syserr[k] = np.hstack([ad[0], ad, ad[-1]])
        specerr = get_spectra_error(error, trace)
        toterr = np.sqrt(specerr**2 + (syserr*0.2)**2)
        G = Gaussian2DKernel(7.)
        data = spec / toterr
        imask = np.abs(data)>0.5
        data[imask] = np.nan
        c = convolve(data, G, boundary='extend')
        data[:] -= c
        G = Gaussian2DKernel(3.)
        N = data.shape[0] * 1.
        arr = np.arange(data.shape[0]-2)
        if (amp == 'LU') or (amp == 'RL'):
            data = data[::-1, :]
        for j in arr:
            inds = np.where(np.sum(np.isnan(data[j:]), axis=0) > (N-j)/3.)[0]
            for ind in inds:
                if np.isnan(data[j, ind]):
                    data[j:, ind] = np.nan
        if (amp == 'LU') or (amp == 'RL'):
            data = data[::-1, :]
        fits.HDUList([fits.PrimaryHDU(np.array(master, dtype='float32')),
                      fits.ImageHDU(np.array(skyimage, dtype='float32')),
                      fits.ImageHDU(np.array(skysub, dtype='float32')),
                      fits.ImageHDU(np.array(skysub / error, dtype='float32')),
                      fits.ImageHDU(np.array(data, dtype='float32')),
                      fits.ImageHDU(np.array(specerr, dtype='float32')),
                      fits.ImageHDU(np.array(syserr*0.2, dtype='float32'))]).writeto('masterskysub_%03d%s.fits' %
                      (ifuslot, amp), overwrite=True)
    fits.PrimaryHDU(np.array(flatimage, dtype='float32')).writeto('%s_%03d%s.fits' %
                   (names[i], ifuslot, amp), overwrite=True)