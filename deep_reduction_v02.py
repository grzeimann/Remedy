#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 09:14:46 2024

@author: grz85
"""

import tables
import glob
import numpy as np
import os.path as op
from astroquery.mast import Catalogs
from astropy.stats import biweight_location as biweight
from astropy.stats import mad_std
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.wcs import WCS
from astropy.io import fits
from astropy.convolution import convolve, Gaussian1DKernel, Gaussian2DKernel
from astropy.table import Table
from astropy.modeling.models import Moffat2D
from input_utils import setup_logging
from astrometry import Astrometry
from sklearn.decomposition import PCA
from extract import Extract
from scipy.interpolate import interp1d
import os
import psutil
import multiprocessing
import warnings

warnings.filterwarnings("ignore")


def get_res_map(amp_spec, amp_sky, amp_error, mask_small, li, hi):
    '''
    Computes a residual map by normalizing input spectra, removing large-scale structure, 
    performing PCA-based modeling, and reconstructing residuals.

    Parameters:
        amp_spec (ndarray): 2D array of observed spectra (rows: exposures or fibers, cols: wavelengths).
        amp_sky (ndarray): 2D array of sky model spectra.
        amp_error (ndarray): 2D array of associated errors for amp_spec.
        mask_small (ndarray): Boolean mask indicating small-scale regions of interest.
        li (int): Lower index bound for wavelength slicing.
        hi (int): Upper index bound for wavelength slicing.

    Returns:
        res (ndarray): 2D residual map with large-scale features and systematics removed.
    '''
    
    # Reference sky model at lower index bound
    sky1 = amp_sky[li]

    # Normalize data by subtracting sky and dividing by error
    Y = ((amp_spec + amp_sky)[li:hi] - sky1[np.newaxis, :]) / amp_error[li:hi]
    Y_orig = Y.copy()

    # Zero out values where mask is False or values are NaN
    Y[~mask_small[li:hi]] = 0.0
    Y[np.isnan(Y)] = 0.0
    is_zero = Y == 0.0

    # Prepare for biweight estimation by replacing zeros with NaNs
    Z = Y.copy()
    Z[Z == 0.0] = np.nan

    # Estimate and subtract average large-scale structure along rows (biweight across fibers)
    avg_w = biweight(Z, axis=0, ignore_nan=True)
    avg_w = get_continuum(avg_w[np.newaxis, :])[0]
    Y -= avg_w[np.newaxis, :]
    Y[is_zero] = 0.0

    # Estimate and subtract average feature profile across wavelength (biweight across pixels)
    Z = Y.copy()
    Z[Z == 0.0] = np.nan
    avg_f = biweight(Z, axis=1, ignore_nan=True)
    avg_f = get_continuum(avg_f[np.newaxis, :], nbins=7)[0]
    Y -= avg_f[:, np.newaxis]
    Y[is_zero] = 0.0

    # Perform PCA on the transposed residual matrix
    pca = PCA(n_components=4)
    H = pca.fit_transform(Y.T)

    # Initial residual model: add back average trends
    res1 = avg_w[np.newaxis, :] + avg_f[:, np.newaxis]

    # Reconstruct modeled features from PCA components
    res = np.zeros_like(Y)
    for i in range(Y.shape[0]):
        sol = np.linalg.lstsq(H, Y[i], rcond=None)[0]
        res[i] = np.dot(sol, H.T)

    # Smooth each wavelength column with interpolation and Gaussian convolution
    x = np.arange(res.shape[0])
    for j in range(Y.shape[1]):
        sel = res[:, j] != 0.0
        if sel.sum() > 20:
            res[:, j] = np.interp(x, x[sel], res[sel, j])
            res[:, j] = convolve(res[:, j], Gaussian1DKernel(3.))

    # Add back average structure
    res += res1

    # Mask invalid original values
    res[np.isnan(Y_orig)] = np.nan

    return res * amp_error[li:hi]


def get_continuum(spectra, nbins=25):
    '''
    Get continuum from sky-subtracted spectra

    Parameters
    ----------
    spectra : 2d numpy array
        sky-subtracted spectra for each fiber

    Returns
    -------
    cont : 2d numpy array
        continuum normalization
    '''
    a = np.array([biweight(f, axis=1, ignore_nan=True) 
               for f in np.array_split(spectra, nbins, axis=1)]).swapaxes(0, 1)
    x = np.array([np.mean(xi) 
               for xi in np.array_split(np.arange(spectra.shape[1]), nbins)])
    cont = np.zeros(spectra.shape)
    X = np.arange(spectra.shape[1])
    for i, ai in enumerate(a):
        sel = np.isfinite(ai)
        if np.sum(sel)>nbins/2.:
            I = interp1d(x[sel], ai[sel], kind='quadratic',
                         fill_value='extrapolate', bounds_error=False)
            cont[i] = I(X)
        else:
            cont[i] = 0.0
    return cont

def pca_fit(H, data):
    '''
    Fit and reconstruct `data` using the PCA basis `H`.

    Parameters
    ----------
    H : ndarray
        PCA basis matrix (components x features).
    data : ndarray
        Input data vector (length = features).

    Returns
    -------
    res : ndarray
        Reconstructed data from least-squares PCA fit.
    '''
    sel = np.isfinite(data)
    sol = np.linalg.lstsq(H.T[sel], data[sel])[0]
    res = np.dot(H.T, sol)
    return res


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
    
def get_photometry_info(allra, alldec, allamps):
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
                S = SkyCoord(allra[k][fibsel].ravel() * u.deg, 
                             alldec[k][fibsel].ravel() * u.deg, frame='fk5')
                ind, d2d, d3d = S.match_to_catalog_sky(T[j:j+1])
                dist_sel = (d2d.arcsec <= 4.25) * goodfibers
                if dist_sel.sum() > 5:
                    nstars += 1
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
                    M = np.interp(X[selp], r, v)
                    norm[k, j] = np.nansum(Y[selp]) / np.nansum(M)
    return norm, exposure_seeing

# =============================================================================
# EGS
# =============================================================================
field_ra = 214.91
field_dec = 52.88
radius = 11. / 60.
catname = '/work/03730/gregz/maverick/VDFI/CFHTLS_D-85_g_141927+524056_T0007_MEDIAN.cat'
pnstars_out = 'PanSTARRS_egs_stars.cat'
cfht = fits.open('/work/03730/gregz/maverick/VDFI/CFHT_EGS_image.fits', memmap=True)


# =============================================================================
# COSMOS
# =============================================================================
field_ra = 150.1375
field_dec = 2.2920
radius = 17. / 60.
catname = '/work/03730/gregz/maverick/VDFI/CFHTLS_D-85_g_100028+021230_T0007_MEDIAN.cat'
pnstars_out = 'PanSTARRS_cosmos_stars.cat'
cfht = fits.open('/work/03730/gregz/maverick/VDFI/CFHT_COSMOS_image.fits', memmap=True)


cfht_image = cfht[0].data
header = cfht[0].header
mask = convolve(cfht_image > 0.2, Gaussian2DKernel(2.5/2.35)) < 0.1
wcs = WCS(header)

xg = np.arange(cfht_image.shape[0])
xgrid, ygrid = np.meshgrid(xg, xg)
grid_ra, grid_dec = wcs.all_pix2world(xgrid + 1., ygrid + 1., 1)
pcoord = SkyCoord(grid_ra[mask]*u.deg, grid_dec[mask]*u.deg)


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
    
# =============================================================================
# PanSTARRS
# =============================================================================

Pan = Catalogs.query_region("%s %s" % (field_ra, field_dec), radius=radius, 
                                       catalog="Panstarrs", data_release="dr2",
                                       table="stack")


wave = np.linspace(3470, 5540, 1036)
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
    

# ===============================
# Main Loop Over Grid Positions
# ===============================
def subtract_sky(counter):
    j = counter
    filename = filenames[counter]
    # Log memory usage for current grid point
    process = psutil.Process(os.getpid())
    log.info('Memory Used for  %s: %0.2f GB' % 
             (op.basename(filename), process.memory_info()[0] / 1e9))
    h5file = tables.open_file(filename, mode='r')
    fwhm = h5file.root.Survey.cols.fwhm[:]
    offset = h5file.root.Survey.cols.offset[:]
    ra = h5file.root.Info.cols.ra[:]
    dec = h5file.root.Info.cols.dec[:]
    mask = np.zeros((len(ra),), dtype=bool)
    coord = SkyCoord(ra*u.deg, dec*u.deg)
    idx, d2d, d3d = coord.match_to_catalog_3d(pcoord)
    mask[d2d.arcsec < 1.] = True
    spec = h5file.root.Fibers.cols.spectrum[:]
    error = h5file.root.Fibers.cols.error[:]
    sky = h5file.root.Fibers.cols.skyspectrum[:]
    sigma = np.zeros((3, len(unique_amps)))
    allra = np.zeros((3, len(unique_amps), 112))
    alldec = np.zeros((3, len(unique_amps), 112))
    newspec = np.zeros((3, len(unique_amps), 112, 1036), dtype=np.float32)
    newerror = np.zeros((3, len(unique_amps), 112, 1036), dtype=np.float32)
    raoff = np.zeros((3, 1036))
    decoff = np.zeros((3, 1036))
    guider = np.zeros((3,))
    offsets = np.zeros((3,))
    h5table = h5file.root.Info
    ifuslots = np.array(['%03d' % i for i in h5table.cols.ifuslot[:]])
    amps = [x.decode("utf-8") for x in h5table.cols.amp[:]]
    combos = np.array(['%s_%s' % (ifuslot, amp) for ifuslot, amp in zip(ifuslots, amps)])
    for i, combo in enumerate(unique_amps):
        log.info('Working on combo: %s for %s' % (combo, op.basename(filename)))
        sel = (combo == combos)
        amp_spec = spec[sel]
        amp_error = error[sel]
        amp_sky = sky[sel]
        mask_small = mask[sel]
        nexp = int(sel.sum() / 112)
        for k in np.arange(nexp):
            li = k * 112
            hi = (k+1) * 112
            res = get_res_map(amp_spec, amp_sky, amp_error, mask_small, li, hi)
            newspec[k, i] = amp_spec[li:hi] - res
            newerror[k, i] = amp_error[li: hi]
            z = amp_spec[li:hi] - res
            sigma[k, i] = mad_std(z / amp_error[li:hi], ignore_nan=True)
            allra[k, i, :] = ra[sel][li:hi]
            alldec[k, i, :] = dec[sel][li:hi]
            guider[k] = fwhm[k]
            offsets[k] = offset[k]
    # Open the file and load relevant information
    R = h5file.root.Survey.cols.ra[0]
    D = h5file.root.Survey.cols.dec[0]
    pa = h5file.root.Survey.cols.pa[0]
    rot = h5file.root.Survey.cols.rot[0]
    
    # Build the astrometry for the 
    A = Astrometry(R, D, pa, 0., 0., fplane_file='/work/03730/gregz/maverick/fplaneall.txt')
    A.tp = A.setup_TP(R, D, rot, A.x0,  A.y0)
    
    # Setup extract for extraction
    E = Extract()
    E.get_ADR_RAdec(A)
    for k in np.arange(3):
        i = j * 3 + k
        raoff[k] = E.ADRra
        decoff[k] = E.ADRdec
    norms, exposure_seeing = get_photometry_info(allra, alldec, newspec)
    h5file.close()
    f1 = fits.PrimaryHDU(newspec)
    f2 = fits.ImageHDU(newerror)
    fits.HDUList([f1, f2]).writeto(op.join(folder, 'temp_%i.fits' % j), 
                                   overwrite=True)
    return (allra, alldec, raoff, decoff, 
            sigma, guider, offsets, norms, exposure_seeing)


TOTAL_TASKS = len(filenames)
NUM_WORKERS = 16  # adjust depending on CPU/memory

with multiprocessing.Pool(processes=NUM_WORKERS) as pool:
    results = pool.map(subtract_sky, range(TOTAL_TASKS))

allra = np.vstack([result[0] for result in results])
alldec = np.vstack([result[1] for result in results])
raoff = np.vstack([result[2] for result in results])
decoff = np.vstack([result[3] for result in results])
sigma = np.vstack([result[4] for result in results])
guider = np.vstack([result[5] for result in results])
offsets = np.vstack([result[6] for result in results])
norm = np.vstack([result[7] for result in results])
exposure_seeing = np.vstack([result[8] for result in results])

fits.HDUList([fits.PrimaryHDU(raoff), fits.ImageHDU(decoff)]).writeto('all_initial_dar.fits', overwrite=True)
 
L = fits.HDUList([fits.PrimaryHDU(), fits.ImageHDU(allra), 
                 fits.ImageHDU(alldec), fits.ImageHDU(guider), fits.ImageHDU(offsets),
                 fits.ImageHDU(exposure_seeing), fits.ImageHDU(norm), 
                 fits.ImageHDU(sigma)])

L.writeto('all_info.fits', overwrite=True)

Pan[sel].write(pnstars_out, format='ascii.fixed_width_two_line',
               overwrite=True)
# Build empty arrays
allspec = np.ones((len(filenames) * 3, len(unique_amps), 112, 1036), dtype=np.float32) * np.nan
allerror = np.ones((len(filenames) * 3, len(unique_amps), 112, 1036), dtype=np.float32) * np.nan
ResidualMaps = np.ones((allspec.shape[1], allspec.shape[2], allspec.shape[3])) * np.nan

# Grab the data and put it into the arrays
for j, filename in enumerate(filenames):
    log.info('Working on %s' % filename)
    f = fits.open(op.join(folder, 'temp_%i.fits' % j))
    ind = j * 3
    allspec[ind:ind+3] = f[0].data
    allerror[ind:ind+3] = f[1].data

for amp in np.arange(allspec.shape[1]):
    s = allspec[:, amp]
    avgres = biweight(s, axis=0, ignore_nan=True)
    ResidualMaps[amp] = avgres
fits.PrimaryHDU(ResidualMaps).writeto('residual_maps.fits', overwrite=True)

fits.PrimaryHDU(allerror).writeto('all_error.fits', overwrite=True)
fits.PrimaryHDU(allspec).writeto('all_flux_final.fits', overwrite=True)
