#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 09:14:46 2024

@author: grz85
"""

import tables
import glob
import numpy as np
import matplotlib.pyplot as plt
import os.path as op
from astroquery.mast import Catalogs
from astropy.stats import biweight_location as biweight
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.io import fits
from astropy.convolution import convolve, Gaussian1DKernel
from astropy.table import Table
from astropy.modeling.models import Moffat2D
from input_utils import setup_logging
import warnings

warnings.filterwarnings("ignore")



def moffat_psf_integration(xloc, yloc, seeing, boxsize=14.,
                               scale=0.1, alpha=3.5):
        '''
        Moffat PSF profile image
        
        Parameters
        ----------
        seeing: float
            FWHM of the Moffat profile
        boxsize: float
            Size of image on a side for Moffat profile
        scale: float
            Pixel scale for image
        alpha: float
            Power index in Moffat profile function
        
        Returns
        -------
        zarray: numpy 3d array
            An array with length 3 for the first axis: PSF image, xgrid, ygrid
        '''
        M = Moffat2D()
        M.alpha.value = alpha
        M.gamma.value = 0.5 * seeing / np.sqrt(2**(1./ M.alpha.value) - 1.)
        xl, xh = (0. - boxsize / 2., 0. + boxsize / 2. + scale)
        yl, yh = (0. - boxsize / 2., 0. + boxsize / 2. + scale)
        x, y = (np.arange(xl, xh, scale), np.arange(yl, yh, scale))
        xgrid, ygrid = np.meshgrid(x, y)
        Z = M(xgrid, ygrid)
        Z = Z / Z.sum()
        V = xloc * 0.
        cnt = 0
        for xl, yl in zip(xloc, yloc):
            d = np.sqrt((xgrid-xl)**2 + (ygrid-yl)**2)
            sel = d <= 0.75
            adj = np.pi * 0.75**2 / (sel.sum() * scale**2)
            V[cnt] = np.sum(Z[sel]) * adj
            cnt += 1
        return V

field_ra = 214.91
field_dec = 52.88

Pan = Catalogs.query_region("%s %s" % (field_ra, field_dec), radius=11./60., 
                                       catalog="Panstarrs", data_release="dr2",
                                       table="stack")
# 299 need to be masked


log = setup_logging('deep_reduction')

# Folder with the initial Remedy reductions
folder = '/work/03730/gregz/maverick/VDFI'
filenames = sorted(glob.glob(op.join(folder, '2*.h5')))

# Grabbing the unique names
unique_amps = []
for filename in filenames:
    h5file = tables.open_file(filename, mode='r')
    h5table = h5file.root.Info
    ifuslots = ['%03d' % i for i in h5table.cols.ifuslot[:]]
    amps = [x.decode("utf-8") for x in h5table.cols.amp[:]]
    combo = ['%s_%s' % (ifuslot, amp) for ifuslot, amp in zip(ifuslots, amps)]
    nui = np.unique(combo)
    for ni in nui:
        if ni not in unique_amps:
            unique_amps.append(ni)
    h5file.close()
    
    
# Get initial catalog sources to mask in the data for sky subtraction later
g = np.array(Pan['gApMag'])
qf = np.array(Pan['qualityFlag'])
pd = np.array(Pan['primaryDetection'])
D = np.array(Pan['decStack'])
sel = (g < 23.) * (g > 14.) * (qf < 64) * (pd > 0) * (np.abs(D) < 90)
pra = np.array(Pan['raStack'][sel])
pdec = np.array(Pan['decStack'][sel])
pcoord = SkyCoord(pra*u.deg, pdec*u.deg)

# Build empty arrays
allamps = np.ones((len(filenames) * 3, len(unique_amps), 112, 1036)) * np.nan
allmask = np.zeros((len(filenames) * 3, len(unique_amps), 112), dtype=bool)
allskies = np.ones((len(filenames) * 3, 1036)) * np.nan
allra = np.ones((len(filenames) * 3, len(unique_amps), 112)) * np.nan
alldec = np.ones((len(filenames) * 3, len(unique_amps), 112)) * np.nan
guider = np.ones((len(filenames)*3)) * np.nan
offsets = np.ones((len(filenames)*3)) * np.nan

# Grab the data and put it into the arrays
for j, filename in enumerate(filenames):
    log.info('Working on %s' % filename)
    h5file = tables.open_file(filename, mode='r')
    fwhm = h5file.root.Survey.cols.fwhm[:]
    offset = h5file.root.Survey.cols.offset[:]
    ra = h5file.root.Info.cols.ra[:]
    dec = h5file.root.Info.cols.dec[:]
    mask = np.zeros((len(ra),), dtype=bool)
    coord = SkyCoord(ra*u.deg, dec*u.deg)
    idx, d2d, d3d = coord.match_to_catalog_3d(pcoord)
    mask[d2d.arcsec < 4.] = True
    spec = h5file.root.Fibers.cols.spectrum[:]
    sky = h5file.root.Fibers.cols.skyspectrum[:]
    h5table = h5file.root.Info
    ifuslots = np.array(['%03d' % i for i in h5table.cols.ifuslot[:]])
    amps = [x.decode("utf-8") for x in h5table.cols.amp[:]]
    combos = np.array(['%s_%s' % (ifuslot, amp) for ifuslot, amp in zip(ifuslots, amps)])
    for i, combo in enumerate(unique_amps):
        sel = (combo == combos) * (mask != True)
        amp_spec = spec[sel]
        amp_sky = sky[sel]
        sel = (combo == combos)
        nexp = int(sel.sum() / 112)
        for k in np.arange(nexp):
            ind = j * 3 + k
            li = k * 112
            hi = (k+1) * 112
            y = spec[sel][li:hi] * 1.
            allskies[ind] = sky[sel][li]
            allmask[ind, i, :] = mask[sel][li:hi]
            allamps[ind, i, :] = y
            allra[ind, i, :] = ra[sel][li:hi]
            alldec[ind, i, :] = dec[sel][li:hi]
            guider[ind] = fwhm[k]
            offsets[ind] = offset[k]
    h5file.close()


# Sub an average number for each exposure
# Sub the average sky offset (wavelength direction)
# Sub the average smooth wavelength offset smooth
# Build the residual
# Sub the average residual (scaled? No) 


wave = np.linspace(3470, 5540, 1036)
li = np.searchsorted(wave, 3580)
hi = np.searchsorted(wave, 5450)


if op.exists('all_flux.fits'):
    allamps = fits.open('all_flux.fits')[0].data
else:
    ResidualMaps = np.ones((allamps.shape[1], allamps.shape[2], allamps.shape[3])) * np.nan
    for I in np.arange(allamps.shape[1]):
        log.info('Working on amplifier: %s' % unique_amps[I])
    
        y = allamps[:, I, :, :] * 1.
        
        # Mask all sources
        y[allmask[:, I, :]] = np.nan
        
        # Avg residual in the whole amplifier
        avg = biweight(y, axis=(1, 2), ignore_nan=True)
        
        # Subtract the average number
        image = biweight(y - avg[:, np.newaxis, np.newaxis], axis=0, ignore_nan=True)
        
        AmpImages = np.zeros((allamps.shape[0], allamps.shape[2], allamps.shape[3]))
        for X in np.arange(allamps.shape[0]):
            # Subtract the average number
            image1 = allamps[X, I, :, :] - avg[X]
            
            # Get the average residual as a funciton of wavelength
            y = biweight(image1, axis=0, ignore_nan=True)
            
            # Get the sky model - the biweight of the sky model (for scaling later)
            z = allskies[X] - biweight(allskies[X], ignore_nan=True)
            
            # Only fit a limited wavelength range
            sel = (wave > 3600) * (wave < 5450)
            
            # Find chi2 best fit scaling
            newmult = np.nansum(z[sel] * y[sel]) / np.nansum(z[sel]**2)
            
            # Find the smooth continuum residual as well
            G = Gaussian1DKernel(15.)
            other = convolve(y - z*newmult, G, boundary='extend')
            
            # Subtract those components
            newimage = image1 - z[np.newaxis, :] * newmult - other[np.newaxis, :]
            
            # Save the new sky subtracted frame
            AmpImages[X] = newimage
        
        # Calculate the biweight residual
        K = AmpImages * 1.
        K[allmask[:, I, :]] = np.nan
        avgResidual = biweight(K, axis=0, ignore_nan=True)
        
        # Loop through each exposure to find the right scaling for the residual
        mults = np.zeros((allamps.shape[0],))
        newmask = allmask[:, I, :] * True
        for X in np.arange(allamps.shape[0]):
            # Make an absolute cut
            fiber_collapse = biweight(AmpImages[X][:, li:hi] - avgResidual[:, li:hi], axis=1, ignore_nan=True)
            fiber_flag = fiber_collapse > 0.2
            for j in np.arange(1):
                fiber_flag[:-1] += fiber_flag[1:]
                fiber_flag[1:] += fiber_flag[:-1]
            newmask[X] = fiber_flag # 
        
        # Calculate the biweight residual again
        K = AmpImages * 1.
        K[newmask] = np.nan
        avgResidual = biweight(K, axis=0, ignore_nan=True)
        ResidualMaps[I, :, :] = avgResidual
        for X in np.arange(allamps.shape[0]):
            AmpImages[X] -= avgResidual
    
        allamps[:, I, :, :] = AmpImages
    
    fits.PrimaryHDU(allamps).writeto('all_flux.fits', overwrite=True)
    fits.PrimaryHDU(ResidualMaps).writeto('residual_maps.fits', overwrite=True)



# Load the PS1 filters 
T = Table.read(op.join('/work/03730/gregz/maverick/Remedy/filters/ps1g.dat'), format='ascii')
filtg = np.interp(wave, T['col1'], T['col2'], left=0.0, right=0.0)
filtg /= filtg.sum()


# Initial Star Select
g = np.array(Pan['gApMag'])
qf = np.array(Pan['qualityFlag'])
pd = np.array(Pan['primaryDetection'])
col = np.array((Pan['rPSFMag']-Pan['rKronMag']))
D = np.array(Pan['decStack'])
sel = (g < 21.) * (g > 14.) * (qf < 64) * (pd > 0) * (np.abs(col) < 0.15) * (np.abs(D)<90.)
T = SkyCoord(Pan['raStack'][sel] * u.deg, Pan['decStack'][sel] * u.deg, frame='fk5')

# Make an array that has the maximum number of fibers within your radius
photometry = np.zeros((allra.shape[0], len(T), 25))
radius = np.zeros((allra.shape[0], len(T), 25))
norm = np.ones((photometry.shape[0], photometry.shape[1])) * np.nan
exposure_seeing = np.ones((photometry.shape[0],)) *  np.nan
for k in np.arange(allra.shape[0]):
    name = op.basename(filenames[int(k/3)])[:-3] + '_exp' + ('%02d' % ((k % 3) + 1))
    log.info('Working on photometry for %s' % name)
    nstars = 0
    
    fibsel = np.isfinite(allra[k])
    goodfibers = np.isnan(allamps[k][fibsel]).sum(axis=1) < 200.
    if fibsel.sum() > 0:
        for j in np.arange(len(T)):
            S = SkyCoord(allra[k][fibsel].ravel() * u.deg, alldec[k][fibsel].ravel() * u.deg, frame='fk5')
            ind, d2d, d3d = S.match_to_catalog_sky(T[j:j+1])
            dist_sel = (d2d.arcsec <= 4.25) * goodfibers
            if dist_sel.sum() > 5:
                nstars += 1
                fig, ax = plt.subplots(1, 1, figsize=(9.5,8))
                specs = allamps[k][fibsel][dist_sel]
                gmag = np.zeros((specs.shape[0],))
                fnu = np.zeros((specs.shape[0],))
                for i, spectrum in enumerate(specs):
                    gmask = np.isfinite(spectrum)
                    nu = 2.99792e18 / wave
                    dnu = np.diff(nu)
                    dnu = np.hstack([dnu[0], dnu])
                    fnu_spec = 1e29 * wave**2 / 2.99792e18 * spectrum * 1e-17
                    fnu[i] = np.dot((nu/dnu*fnu_spec)[gmask], filtg[gmask]) / np.sum((nu/dnu*filtg)[gmask])
                    gmag[i] = -2.5 * np.log10(fnu[i]) + 23.9
                fnutotal =  10**(-0.4 * (g[sel][j]-23.9))
                photometry[k, j, :len(fnu)] = fnu / fnutotal
                dra = np.cos(T[j].dec.deg * np.pi / 180.) * (allra[k][fibsel][dist_sel] - T[j].ra.deg) * 3600.
                ddec = (alldec[k][fibsel][dist_sel] - T[j].dec.deg) * 3600.
                radius[k, j, :len(fnu)] = np.sqrt(dra**2 + ddec**2)
        X = radius[k].ravel()
        Y = photometry[k].ravel()
        selp = Y != 0.0
        r = np.linspace(0, 4.25, 51)
        seeing = np.linspace(1.0, 4.0, 61)
        chi2 = 0. * seeing
        for i, see in enumerate(seeing):
            v = moffat_psf_integration(r, 0. * r, see, alpha=3.5)
            M = np.interp(X[selp], r, v)
            chi2[i] = np.nansum((M - Y[selp])**2)
        mchi2 = np.min(chi2)
        x0 = np.argmin(chi2)
        fitted_seeing = seeing[x0]
        p0 = np.polyfit(seeing[x0-1:x0+2], chi2[x0-1:x0+2], 2)
        fitted_seeing = -1. * p0[1] / (2. * p0[0])
        exposure_seeing[k] = fitted_seeing
        for j in np.arange(photometry[k].shape[0]):
            Y = photometry[k][j]
            X = radius[k][j]
            selp = Y != 0.0
            v = moffat_psf_integration(r, 0. * r, fitted_seeing, alpha=3.5)
            if selp.sum() > 4.:
                M = np.interp(X[sel], r, v)
                norm[k, j] = np.nansum(Y[selp]) / np.nansum(M)
 
L = fits.HDUList(fits.PrimaryHDU(), fits.ImageHDU(allra), 
                 fits.ImageHDU(alldec), fits.ImageHDU(guider), fits.ImageHDU(offsets),
                 fits.ImageHDU(exposure_seeing), fits.ImageHDU(norm))
L.writeto('all_info.fits', overwrite=True)