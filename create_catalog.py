#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 09:22:00 2020

@author: gregz
"""
import glob
import numpy as np
import os
import os.path as op
import psutil
import sys
import tables

from astropy.io import fits
from astropy.table import Table
from input_utils import setup_logging
from math_utils import biweight

# =============================================================================
# Set Criteria for Keeping Spectrum
# Shot:
#   Throughput * MILLUM
#   OFFSET
#   Nastrom, Nphotom
#   Astrom Errors
# Source:
#   weight > some net weight as a mask
#   less than some number of pixels 
#   normalize by g-band
# =============================================================================

def get_script_path():
    '''
    Get script path, aka, where does Remedy live?
    '''
    return op.dirname(op.realpath(sys.argv[0]))

def get_gmags(spec, weight):
    spectrum = spec * 1.
    spectrum = spectrum * 1e-17 * 1e29 * def_wave**2 / 2.99792e18
    gmask = (np.isfinite(spectrum) *
             (weight > (0.7*np.nanmedian(weight, axis=1)[:, np.newaxis])))
    nu = 2.99792e18 / def_wave
    dnu = np.diff(nu)
    dnu = np.hstack([dnu[0], dnu])
    num = np.nansum((nu / dnu)[np.newaxis, :] * spectrum *
                     filtg[np.newaxis, :] * gmask, axis=1)
    denom = np.nansum((nu / dnu * filtg)[np.newaxis, :] * gmask, axis=1)
    gmag = num / denom
    gmag = -2.5 * np.log10(gmag) + 23.9
    return gmag

outname = sys.argv[1]

def_wave = np.linspace(3470, 5540, 1036)

log = setup_logging('catalog')

folder = '/work/03730/gregz/maverick/parallel'
DIRNAME = get_script_path()
T = Table.read(op.join(DIRNAME, 'filters/ps1g.dat'), format='ascii')
filtg = np.interp(def_wave, T['col1'], T['col2'], left=0.0, right=0.0)
filtg /= filtg.sum()

h5names = sorted(glob.glob(op.join(folder, '*.h5')))
totN = 0
Nshots = 0
cnt = 0
for niter in np.arange(2):
    if niter == 1:
        RA = np.zeros((totN,))
        DEC = np.zeros((totN,))
        GMAG = np.zeros((totN,))
        SN = np.zeros((totN,))
        NAME = np.zeros((totN,), dtype='|S16')
        SPEC = np.zeros((totN, 1036))
        ERROR = np.zeros((totN, 1036))
        WEIGHT = np.zeros((totN, 1036))
        process = psutil.Process(os.getpid())
        log.info('Memory Used: %0.2f GB' % (process.memory_info()[0] / 1e9))
    for h5name in h5names:
        h5file = tables.open_file(h5name)
        name = op.basename(h5name).split('.h5')[0]
        raoffset = h5file.root.Survey.cols.raoffset[0]
        decoffset = h5file.root.Survey.cols.decoffset[0]
        rastd = h5file.root.Survey.cols.rastd[0]
        decstd = h5file.root.Survey.cols.decstd[0]
        nstars = h5file.root.Survey.cols.nstarsastrom[0]
        rule1 = np.abs(raoffset) < 0.25
        rule2 = np.abs(decoffset) < 0.25
        rule3 = np.abs(rastd) < 0.5
        rule4 = np.abs(decstd) < 0.5
        rule5 = nstars >= 5
        good = rule1 * rule2 * rule3 * rule4 * rule5
        if ~good:
            if niter == 0:
                log.info('%s did not make cut' % name)
            h5file.close()
            continue
        spectra = h5file.root.CatSpectra.cols.spectrum[:]
        error = h5file.root.CatSpectra.cols.error[:]
        weight = h5file.root.CatSpectra.cols.weight[:]
        gmag = h5file.root.CatSpectra.cols.gmag[:]
        ra = h5file.root.CatSpectra.cols.ra[:]
        dec = h5file.root.CatSpectra.cols.dec[:]
        mask = (weight > 0.15) * np.isfinite(spectra)
        num = np.nansum(filtg[np.newaxis, :] * mask * (spectra / error), axis=1) 
        denom = np.nansum(filtg[np.newaxis, :] * mask, axis=1)
        sn = num / denom
        goodspec = (mask.sum(axis=1) > 0.8) * (sn > 1.)
        N = goodspec.sum()
        if N < 1:
            continue
        if niter == 0:
            totN += N
            Nshots += 1
        virus_gmags = get_gmags(spectra[goodspec], weight[goodspec])
        normalization = 10**(-0.4 * (gmag[goodspec] - virus_gmags))
        norm_spectra = spectra[goodspec] * normalization[:, np.newaxis]
        osel = sn[goodspec] > 5.
        average_norm, std = biweight(normalization[osel], calc_std=True)
        if niter == 0:
            log.info('%s has %i/%i and average normalization correction: %0.2f +/- %0.2f' %
                     (name, osel.sum(), N, average_norm, std))
        if niter == 1:
            RA[cnt:cnt+N] = ra[goodspec]
            DEC[cnt:cnt+N] = dec[goodspec]
            GMAG[cnt:cnt+N] = gmag[goodspec]
            SN[cnt:cnt+N] = sn[goodspec]
            SPEC[cnt:cnt+N] = norm_spectra
            ERROR[cnt:cnt+N] = error[goodspec] * normalization[:, np.newaxis] 
            WEIGHT[cnt:cnt+N] = weight[goodspec]
            NAME[cnt:cnt+N] = name
            cnt += N
        h5file.close()

T = Table([RA, DEC, NAME, GMAG, SN], names=['RA', 'Dec', 'shotid', 'gmag', 'sn'])
fits.HDUList([fits.PrimaryHDU(), fits.BinHDUTable(T), fits.ImageHDU(SPEC),
              fits.ImageHDU(ERROR), fits.ImageHDU(WEIGHT)]).writeto(outname, overwrite=True)

log.info('Total Number of sources is %i for %i shots' % (totN, Nshots))