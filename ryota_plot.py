#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 09:29:15 2019

@author: gregz
"""

from astroquery.sdss import SDSS
from astropy.table import Table
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import seaborn as sns
import numpy as np
import matplotlib
from tables import open_file
from astropy.io import fits
sns.set_context('talk')
ML = MultipleLocator(500)
ml = MultipleLocator(100)
MLy = MultipleLocator(1)
mly = MultipleLocator(0.2)

ncat = Table.read('ryota.dat', format='ascii.fixed_width_two_line')
sp = SDSS.get_spectra(matches=ncat)
index = -5
out_pdf = 'ryota_hetdex.pdf'
F = fits.open('ryota_hetdex_spectra.fits')
pdf = matplotlib.backends.backend_pdf.PdfPages(out_pdf)
hdfile = open_file('survey_hdr1.h5')
t = Table(hdfile.root.Survey[:])
cnt = 0
for index in np.arange(len(ncat)):
    if (cnt % 3) == 0:
        plot_num = 311
        fig = plt.figure(figsize=(8.5, 11)) # inches
    plt.subplot(plot_num)
    plt.plot(10**(sp[index][1].data['loglam']), sp[index][1].data['flux'], lw=1,
         alpha=0.5, label='SDSS')
    flam = 10**(-0.4 * (ncat[index]['g']-23.9)) * 1e-29 * 3e18 / 5000.**2 * 1e17
    plt.scatter(5000., flam, marker='x', color='k', s=150)
    for ind  in np.where(ncat['specobjid'][index] == F[1].data['source_id'])[0]:
        st = str(F[1].data['obs_id'][ind])
        sel1 = np.where((t['date'] == int(st[:8])) * (t['obsid'] == int(st[-3:])))[0]
        x = t[sel1[0]]
        sel = F[4].data[ind] < 0.8
        data = F[2].data[ind] * 1.
        data[sel] = np.nan
        plt.plot(np.linspace(3470, 5540, 1036), data, 'r-', alpha=0.4,
                 label=st)
    plt.xlim([3450, 5550])
    plt.xlabel(r'Wavelength ($\AA$)')
    plt.ylabel(r'F$_{\lambda}$ (1e-17 ergs/s/cm^2/$\AA$)')
    if (cnt % 3) == 2:
        pdf.savefig(fig)
    if index == (len(ncat)-1):
        pdf.savefig(fig)
    plot_num +=1
    cnt += 1
pdf.close()
