#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 12 06:12:29 2022

@author: gregz
"""

from astropy.io import fits
import numpy as np
from astropy.coordinates import SkyCoord
import astropy.units as u

filename1 = '/work/03730/gregz/maverick/parallel/HETVIPS_continuum_v1.0_unique.fits'
filename2 = '/work/03730/gregz/maverick/parallel/HETVIPS_continuum_v1.0.fits'

f = fits.open(filename1)
g = fits.open(filename2)

inds = np.where(f[1].data['sn'] > 100.)[0]
inds2 = np.where(g[1].data['sn'] > 20.)[0]
print('The two arrays are %i and %i long' % (len(inds), len(inds2)))

s = SkyCoord(f[1].data['RA'][inds]*u.deg, f[1].data['Dec'][inds]*u.deg)
S = SkyCoord(g[1].data['RA'][inds2]*u.deg, g[1].data['Dec'][inds2]*u.deg)
shotid = f[1].data['shotid'][inds]
shotids = g[1].data['shotid'][inds2]
spectra = f[2].data[inds]
spectra2 = g[2].data[inds2]

keep1 = []
keep2 = []
cnt = 0
for i in np.arange(len(inds)):
    d2d = s[i].separation(S)
    sel = np.where((d2d.arcsec < 1.)  * (shotid[i] != shotids))[0]
    if (cnt % 100) == 0:
        print('We are at %i' % (cnt+1))
    if len(sel) > 0.:
        cnt += len(sel)
        keep1.append(i)
        keep2.append(sel)
        
diff = np.zeros((cnt, 1036))
cnt = 0
for i, sel in zip(keep1, keep2):
    for s in sel:
        diff[cnt] = (spectra[i] - spectra2[s]) / spectra[i]
        cnt += 1
fits.PrimaryHDU(diff).writeto('/work/03730/gregz/maverick/parallel/repeat_difference.fits',
                              overwrite=True)