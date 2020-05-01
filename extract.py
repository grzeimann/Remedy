#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 11:48:55 2019

@author: gregz
"""

import numpy as np

from astropy.convolution import Gaussian2DKernel, convolve
from astropy.modeling.models import Moffat2D, Gaussian2D
from input_utils import setup_logging
from scipy.interpolate import griddata, LinearNDInterpolator

class Extract:
    def __init__(self, wave=None):
        '''
        Initialize Extract class
        
        Parameters
        ----------
        wave: numpy 1d array
            wavelength of calfib extension for hdf5 files, does not need to be
            set unless needed by development team
        '''
        if wave is not None:
            self.wave = wave
        else:
            self.wave = self.get_wave()
        self.get_ADR()
        self.log = setup_logging('Extract')
        self.set_dither_pattern()
    
    def set_dither_pattern(self, dither_pattern=None):
        ''' 
        Set dither pattern (default if None given)
        
        Parameters
        ----------
        dither_pattern: numpy array (length of exposures)
            only necessary if the dither pattern isn't the default
        '''
        if dither_pattern is None:
            self.dither_pattern = np.array([[0., 0.], [1.27, -0.73],
                                            [1.27, 0.73]])
        else:
            self.dither_pattern = dither_pattern

    def get_wave(self):
        ''' Return implicit wavelength solution for calfib extension '''
        return np.linspace(3470, 5540, 1036)

    def get_ADR(self, angle=0.):
        ''' 
        Use default ADR from Karl Gebhardt (adjusted by Greg Zeimann)

        Parameters
        ----------
        angle: float
            angle=0 along x-direction, appropriate if observing at PA
        dither_pattern: numpy array (length of exposures)
            only necessary if the dither pattern isn't the default
        '''
        wADR = [3500., 4000., 4500., 5000., 5500.]
        ADR = [-0.74, -0.4, -0.08, 0.08, 0.20]
        ADR = np.polyval(np.polyfit(wADR, ADR, 3), self.wave)
        self.ADRx = np.cos(np.deg2rad(angle)) * ADR
        self.ADRy = np.sin(np.deg2rad(angle)) * ADR

    def get_ADR_RAdec(self, astrometry_object):
        '''
        Use an astrometry object to convert delta_x and delta_y into
        delta_ra and delta_dec
        
        Parameters
        ----------
        astrometry_object : Astrometry Class
            Contains the astrometry to convert x and y to RA and Dec
        '''
        tRA, tDec = astrometry_object.tp.wcs_pix2world(self.ADRx, self.ADRy, 1)
        self.ADRra = ((tRA - astrometry_object.ra0) * 3600. *
                      np.cos(np.deg2rad(astrometry_object.dec0)))
        self.ADRdec = (tDec - astrometry_object.dec0) * 3600.

    def intersection_area(self, d, R, r):
        """
        Return the area of intersection of two circles.
        The circles have radii R and r, and their centers are separated by d.
    
        Parameters
        ----------
        d : float
            separation of the two circles
        R : float
            Radius of the first circle
        r : float
            Radius of the second circle
        """
    
        if d <= abs(R-r):
            # One circle is entirely enclosed in the other.
            return np.pi * min(R, r)**2
        if d >= r + R:
            # The circles don't overlap at all.
            return 0
    
        r2, R2, d2 = r**2, R**2, d**2
        alpha = np.arccos((d2 + r2 - R2) / (2*d*r))
        beta = np.arccos((d2 + R2 - r2) / (2*d*R))
        answer = (r2 * alpha + R2 * beta -
                  0.5 * (r2 * np.sin(2*alpha) + R2 * np.sin(2*beta)))
        return answer

    def tophat_psf(self, radius, boxsize, scale, fibradius=0.75):
        '''
        Tophat PSF profile image 
        (taking fiber overlap with fixed radius into account)
        
        Parameters
        ----------
        radius: float
            Radius of the tophat PSF
        boxsize: float
            Size of image on a side for Moffat profile
        scale: float
            Pixel scale for image
        
        Returns
        -------
        zarray: numpy 3d array
            An array with length 3 for the first axis: PSF image, xgrid, ygrid
        '''
        xl, xh = (0. - boxsize / 2., 0. + boxsize / 2. + scale)
        yl, yh = (0. - boxsize / 2., 0. + boxsize / 2. + scale)
        x, y = (np.arange(xl, xh, scale), np.arange(yl, yh, scale))
        xgrid, ygrid = np.meshgrid(x, y)
        t = np.linspace(0., np.pi * 2., 360)
        r = np.arange(radius - fibradius, radius + fibradius * 7./6.,
                      1. / 6. * fibradius)
        xr, yr, zr = ([], [], [])
        for i, ri in enumerate(r):
            xr.append(np.cos(t) * ri)
            yr.append(np.sin(t) * ri)
            zr.append(self.intersection_area(ri, radius, fibradius) *
                      np.ones(t.shape) / (np.pi * fibradius**2))
        xr, yr, zr = [np.hstack(var) for var in [xr, yr, zr]]
        psf = griddata(np.array([xr, yr]).swapaxes(0,1), zr, (xgrid, ygrid),
                       method='linear')
        psf[np.isnan(psf)]=0.
        zarray = np.array([psf, xgrid, ygrid])
        zarray[0] /= zarray[0].sum()
        return zarray

    def gaussian_psf(self, xstd, ystd, theta, boxsize, scale):
        '''
        Gaussian PSF profile image 
        
        Parameters
        ----------
        xstd: float
            Standard deviation of the Gaussian in x before rotating by theta
        ystd: float
            Standard deviation of the Gaussian in y before rotating by theta
        theta: float
            Rotation angle in radians.
            The rotation angle increases counterclockwise
        boxsize: float
            Size of image on a side for Moffat profile
        scale: float
            Pixel scale for image
        
        Returns
        -------
        zarray: numpy 3d array
            An array with length 3 for the first axis: PSF image, xgrid, ygrid
        '''
        M = Gaussian2D()
        M.x_stddev.value = xstd
        M.y_stddev.value = ystd
        M.theta.value = theta
        xl, xh = (0. - boxsize / 2., 0. + boxsize / 2. + scale)
        yl, yh = (0. - boxsize / 2., 0. + boxsize / 2. + scale)
        x, y = (np.arange(xl, xh, scale), np.arange(yl, yh, scale))
        xgrid, ygrid = np.meshgrid(x, y)
        zarray = np.array([M(xgrid, ygrid), xgrid, ygrid])
        zarray[0] /= zarray[0].sum()
        return zarray

    def moffat_psf(self, seeing, boxsize, scale, alpha=3.5):
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
        Z = self.moffat_psf_integration(xgrid.ravel(), ygrid.ravel(),
                                        seeing, boxsize=boxsize+1.5,
                                        alpha=alpha)
        Z = np.reshape(Z, xgrid.shape)
        zarray = np.array([Z, xgrid, ygrid])
        return zarray
    
    def moffat_psf_integration(self, xloc, yloc, seeing, boxsize=14.,
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
        

    def make_collapsed_image(self, xc, yc, xloc, yloc, data, error, mask,
                             scale=0.25, seeing_fac=1.8, boxsize=4.,
                             wrange=[3470, 5540], nchunks=11,
                             convolve_image=False,
                             interp_kind='linear'):
        ''' 
        Collapse spectra to make a signle image on a rectified grid.  This
        may be done for a wavelength range and using a number of chunks
        of wavelength to take ADR into account.
        
        Parameters
        ----------
        xc: float
            The ifu x-coordinate for the center of the collapse frame
        yc: float 
            The ifu y-coordinate for the center of the collapse frame
        xloc: numpy array
            The ifu x-coordinate for each fiber
        yloc: numpy array
            The ifu y-coordinate for each fiber
        data: numpy 2d array
            The calibrated spectrum for each fiber
        error: numpy 2d array
            The calibrated error spectrum for each fiber
        mask: numpy 2d array
            The good fiber wavelengths to be used in collapsed frame
        scale: float
            Pixel scale for output collapsed image
        seeing_fac: float
            seeing_fac = 2.35 * radius of the Gaussian kernel used 
            if convolving the images to smooth out features. Unit: arcseconds
        boxsize: float
            Length of the side in arcseconds for the convolved image
        wrange: list
            The wavelength range to use for collapsing the frame
        nchunks: int
            Number of chunks used to take ADR into account when collapsing
            the fibers.  Use a larger number for a larger wavelength.
            A small wavelength may only need one chunk
        convolve_image: bool
            If true, the collapsed frame is smoothed at the seeing_fac scale
        interp_kind: str
            Kind of interpolation to pixelated grid from fiber intensity
        
        Returns
        -------
        zarray: numpy 3d array
            An array with length 3 for the first axis: PSF image, xgrid, ygrid
        '''
        a, b = data.shape
        N = int(boxsize/scale) + 1
        xl, xh = (xc - boxsize / 2., xc + boxsize / 2.)
        yl, yh = (yc - boxsize / 2., yc + boxsize / 2.)
        x, y = (np.linspace(xl, xh, N), np.linspace(yl, yh, N))
        xgrid, ygrid = np.meshgrid(x, y)
        S = np.zeros((a, 2))
        area = np.pi * 0.75**2
        sel = (self.wave > wrange[0]) * (self.wave <= wrange[1])
        I = np.arange(b)
        ichunk = [np.mean(xi) for xi in np.array_split(I[sel], nchunks)]
        ichunk = np.array(ichunk, dtype=int)
        wchunk = [np.mean(xi) for xi in np.array_split(self.wave[sel],
                  nchunks)]
        wchunk = np.array(wchunk)

        cnt = 0
        image_list = []
        if convolve_image:
            seeing = seeing_fac / scale
            G = Gaussian2DKernel(seeing / 2.35)
        if interp_kind not in ['linear', 'cubic']:
            self.log.warning('interp_kind must be "linear" or "cubic"')
            self.log.warning('Using "linear" for interp_kind')
            interp_kind='linear'
        for chunk, echunk, mchunk in zip(
                                np.array_split(data[:, sel], nchunks, axis=1),
                                np.array_split(error[:, sel], nchunks, axis=1),
                                np.array_split(mask[:,sel], nchunks, axis=1)):
            marray = np.ma.array(chunk, mask=mchunk<1e-8)
            image = np.ma.median(marray, axis=1)
            image = image / np.ma.sum(image)
            S[:, 0] = xloc - self.ADRra[ichunk[cnt]]
            S[:, 1] = yloc - self.ADRdec[ichunk[cnt]]
            cnt += 1
            try:
                grid_z = (griddata(S[~image.mask], image.data[~image.mask],
                                   (xgrid, ygrid), method=interp_kind) *
                          scale**2 / area)
            except:
                grid_z = 0.0 * xgrid
            if convolve_image:
                grid_z = convolve(grid_z, G)
            image_list.append(grid_z)
        image = np.array(image_list)
        image[np.isnan(image)] = 0.0
        zarray = np.array([image, xgrid-xc, ygrid-yc])
        return zarray

    def get_psf_curve_of_growth(self, psf):
        '''
        Analyse the curve of growth for an input psf
        
        Parameters
        ----------
        psf: numpy 3d array
            zeroth dimension: psf image, xgrid, ygrid
        
        Returns
        -------
        r: numpy array
            radius in arcseconds
        cog: numpy array
            curve of growth for psf
        '''
        r = np.sqrt(psf[1]**2 + psf[2]**2).ravel()
        inds = np.argsort(r)
        maxr = psf[1].max()
        sel = r[inds] <= maxr
        cog = (np.cumsum(psf[0].ravel()[inds][sel]) /
               np.sum(psf[0].ravel()[inds][sel]))
        return r[inds][sel], cog
    
    def build_weights(self, xc, yc, ifux, ifuy, psf):
        '''
        Build weight matrix for spectral extraction
        
        Parameters
        ----------
        xc: float
            The ifu x-coordinate for the center of the collapse frame
        yc: float 
            The ifu y-coordinate for the center of the collapse frame
        xloc: numpy array
            The ifu x-coordinate for each fiber
        yloc: numpy array
            The ifu y-coordinate for each fiber
        psf: numpy 3d array
            zeroth dimension: psf image, xgrid, ygrid
        
        Returns
        -------
        weights: numpy 2d array (len of fibers by wavelength dimension)
            Weights for each fiber as function of wavelength for extraction
        '''
        S = np.zeros((len(ifux), 2))
        T = np.array([psf[1].ravel(), psf[2].ravel()]).swapaxes(0, 1)
        I = LinearNDInterpolator(T, psf[0].ravel(),
                                 fill_value=0.0)
        weights = np.zeros((len(ifux), len(self.wave)))
        for i in np.arange(len(self.wave)):
            S[:, 0] = ifux - self.ADRra[i] - xc
            S[:, 1] = ifuy - self.ADRdec[i] - yc
            weights[:, i] = I(S[:, 0], S[:, 1])
        self.log.info('Total weight is: %0.2f' % np.median(np.sum(weights, axis=0)))
        return weights
    
    def build_weights_no_ADR(self, xc, yc, ifux, ifuy, psf):
        '''
        Build weight matrix for spectral extraction
        
        Parameters
        ----------
        xc: float
            The ifu x-coordinate for the center of the collapse frame
        yc: float 
            The ifu y-coordinate for the center of the collapse frame
        xloc: numpy array
            The ifu x-coordinate for each fiber
        yloc: numpy array
            The ifu y-coordinate for each fiber
        psf: numpy 3d array
            zeroth dimension: psf image, xgrid, ygrid
        
        Returns
        -------
        weights: numpy 2d array (len of fibers by wavelength dimension)
            Weights for each fiber as function of wavelength for extraction
        '''
        S = np.zeros((len(ifux), 2))
        T = np.array([psf[1].ravel(), psf[2].ravel()]).swapaxes(0, 1)
        I = LinearNDInterpolator(T, psf[0].ravel(),
                                 fill_value=0.0)
        scale = np.abs(psf[1][0, 1] - psf[1][0, 0])
        area = 0.75**2 * np.pi
        S[:, 0] = ifux - xc
        S[:, 1] = ifuy - yc
        weights = I(S[:, 0], S[:, 1]) * area / scale**2

        return weights

    def get_spectrum(self, data, error, mask, weights):
        '''
        Weighted spectral extraction
        
        Parameters
        ----------
        data: numpy 2d array (number of fibers by wavelength dimension)
            Flux calibrated spectra for fibers within a given radius of the 
            source.  The radius is defined in get_fiberinfo_for_coord().
        error: numpy 2d array
            Error of the flux calibrated spectra 
        mask: numpy 2d array (bool)
            Mask of good wavelength regions for each fiber
        weights: numpy 2d array (float)
            Normalized extraction model for each fiber
        
        Returns
        -------
        spectrum: numpy 1d array
            Flux calibrated extracted spectrum
        spectrum_error: numpy 1d array
            Error for the flux calibrated extracted spectrum
        '''
        var = error**2
        spectrum = (np.nansum(data * mask * weights / var, axis=0) /
                    np.nansum(mask * weights**2 / var, axis=0))
        spectrum_error = np.sqrt(np.nansum(mask * weights, axis=0) /
                                 np.nansum(mask * weights**2 / var, axis=0))
        # Only use wavelengths with enough weight to avoid large noise spikes
        w = np.sum(mask * weights, axis=0)
        bad = w < 0.05
        for i in np.arange(1, 2):
            bad[i:] += bad[:-i]
            bad[:-i] += bad[i:]
        spectrum[bad] = np.nan
        spectrum_error[bad] = np.nan
        
        return spectrum, spectrum_error, w
    
    def get_info_within_radius(self, ra, dec, data, error, mask, rac, decc,
                               thresh=7.):
        '''
        Convert RA/Dec to delta_RA and delta_Dec for grid (in arcseconds)
        
        Parameters
        ----------
        ra : numpy 1d array
            Right Ascension of the fibers
        dec : numpy 1d array
            Declination of the fibers
        data: numpy 2d array (number of fibers by wavelength dimension)
            Flux calibrated spectra for fibers in shot
        error: numpy 2d array
            Error of the flux calibrated spectra 
        mask: numpy 2d array (bool)
            Mask of good wavelength regions for each fiber
        rac : float
            Right Ascension for centroid
        decc : float
            Declination for centroid
        thresh : float
            Distance threshold in arcseconds for finding fibers near rac, decc
        
        Returns
        -------
        delta_ra : 1d numpy array
            Offset in Right Ascension (") from rac, decc for each fiber
        delta_dec : 1d numpy array
            Offset in Declination (") from rac, decc for each fiber
        data : numpy 2d array
            spectra of fibers within thresh radius
        error : numpy 2d array
            error spectra of fibers within thresh radius
        mask : numpy 2d array
            mask for spectra of fibers within thresh radius
        '''
        delta_ra = np.cos(np.deg2rad(decc)) * (ra - rac) * 3600.
        delta_dec = (dec - decc) * 3600.
        distsel = np.where(np.sqrt(delta_ra**2 + delta_dec**2) < thresh)[0]
        if len(distsel) == 0.:
            return None
        return (delta_ra[distsel], delta_dec[distsel], data[distsel],
                error[distsel], mask[distsel])
        
    
    def get_spectrum_by_coord_index(self, ind):
        '''
        Weighted spectral extraction
        
        Parameters
        ----------
        ind: int
            index of the self.coords array (need to define self.coords first)
        
        Returns
        -------
        spectrum: numpy 1d array
            Flux calibrated extracted spectrum
        spectrum_error: numpy 1d array
            Error for the flux calibrated extracted spectrum
        '''
        if not hasattr(self, 'coords'):
            self.log.warning('No "coords" attribute found."')
            return None
        if not hasattr(self, 'psf'):
            # default PSF for extraction is a 4" aperture
            self.psf = self.tophat_psf(4, 10.5, 0.25)
        ra_cen, dec_cen = (self.coords[ind].ra.deg, self.coords[ind].dec.deg)
        info_result = self.get_info_within_radius(self.ra, self.dec,
                                                  self.data, self.error,
                                                  self.mask, ra_cen, dec_cen,
                                                  thresh=7.)
        if info_result is None:
            return [], [], [], [], [], []
        
        self.log.info('Extracting %i' % ind)
        rafibers, decfibers, data, error, mask = info_result
        d = np.sqrt(rafibers**2 + decfibers**2)
        mask_extract = mask
        weights = self.build_weights(0., 0., rafibers, decfibers, self.psf)
        norm = np.sum(weights, axis=0)
        weights = weights / norm[np.newaxis, :]
        result = self.get_spectrum(data, error, mask_extract, weights)
        spectrum, spectrum_error, mweight = [res for res in result]
        spectrum, spectrum_error = [x*norm for x in [spectrum, spectrum_error]]
        spec_package = [rafibers[:, np.newaxis]-self.ADRra, 
                        decfibers[:, np.newaxis]-self.ADRdec,
                        data, error, mask]
        image_array = self.make_collapsed_image(0., 0., rafibers, decfibers,
                                                data, error, mask, 
                                                seeing_fac=1.5,
                                                scale=0.5, boxsize=10.,
                                                convolve_image=True)
        image, xgrid, ygrid = image_array
        return (spectrum / norm, spectrum_error / norm, mweight, image, xgrid, ygrid, 
               spec_package)