#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 13:36:42 2019

@author: gregz
"""

import glob
import numpy as np
import os.path as op
import tarfile

from astropy.io import fits
from math_utils import biweight
from datetime import datetime
from scipy.interpolate import interp2d

from astropy.stats import sigma_clip, mad_std, sigma_clipped_stats
from astropy.convolution import Gaussian1DKernel, convolve
from astropy.convolution import interpolate_replace_nans

def orient_image(image, amp, ampname):
    '''
    Orient the images from blue to red (left to right)
    Fibers are oriented to match configuration files
    
    Parameters
    ----------
    image : 2d numpy array
        fits image
    amp : str
        Amplifier for the fits image
    ampname : str
        Amplifier name is the location of the amplifier
    
    Returns
    -------
    image : 2d numpy array
        Oriented fits image correcting for what amplifier it comes from
        These flips are unique to the VIRUS/LRS2 amplifiers
    '''
    if amp == "LU":
        image[:] = image[::-1, ::-1]
    if amp == "RL":
        image[:] = image[::-1, ::-1]
    if ampname is not None:
        if ampname == 'LR' or ampname == 'UL':
            image[:] = image[:, ::-1]
    return image


def base_reduction(filename, get_header=False):
    '''
    Reduce filename from tarfile or fits file.
    
    Reduction steps include:
        1) Overscan subtraction
        2) Trim image
        3) Orientation
        4) Gain Multiplication
        5) Error propagation
    
    Parameters
    ----------
    filename : str
        Filename of the fits file
    get_header : boolean
        Flag to get and return the header
    tfile : str
        Tar filename if the fits file is in a tarred file
    
    Returns
    -------
    a : 2d numpy array
        Reduced fits image, see steps above
    e : 2d numpy array
        Associated error frame
    '''
    # Load fits file
    tarbase = op.dirname(op.dirname(op.dirname(filename))) + '.tar'
    if op.exists(tarbase):
        T = tarfile.open(tarbase, 'r')
        s = '/'.join(filename.split('/')[-4:])
        a = fits.open(T.extractfile(s))
    else:
        a = fits.open(filename)

    image = np.array(a[0].data, dtype=float)
    
    # Overscan subtraction
    overscan_length = 32 * (image.shape[1] / 1064)
    O = biweight(image[:, -(overscan_length-2):])
    image[:] = image - O
    
    # Trim image
    image = image[:, :-overscan_length]
    
    # Gain multiplication (catch negative cases)
    gain = a[0].header['GAIN']
    gain = np.where(gain > 0., gain, 0.85)
    rdnoise = a[0].header['RDNOISE']
    rdnoise = np.where(rdnoise > 0., rdnoise, 3.)
    amp = (a[0].header['CCDPOS'].replace(' ', '') +
           a[0].header['CCDHALF'].replace(' ', ''))
    try:
        ampname = a[0].header['AMPNAME']
    except:
        ampname = None
    header = a[0].header
    
    # Orient image
    a = orient_image(image, amp, ampname) * gain
    
    # Calculate error frame
    E = np.sqrt(rdnoise**2 + np.where(a > 0., a, 0.))
    if get_header:
        return a, E, header
    return a, E

def power_law(x, c1, c2=.5, c3=.15, c4=1., sig=2.5):
    '''
    Power law for scattered light from mirror imperfections
    
    Parameters
    ----------
    x : float or 1d numpy array
        Distance from fiber in pixels
    c1 : float
        Normalization of powerlaw
    
    Returns
    -------
    plaw : float or 1d numpy array
        see function form below
    '''
    return c1 / (c2 + c3 * np.power(abs(x / sig), c4))


def get_powerlaw_ydir(trace, spec, amp, col):
    '''
    Get powerlaw in ydir for a given column
    
    Parameters
    ----------
    trace : 2d numpy array
        y position as function of x for each fiber
    spec : 2d numpy array
        fiber spectra
    amp : str
        amplifier
    col : int
        Column
    '''
    if amp in ['LL', 'RU']:
        ntrace = np.vstack([trace, 2064 - trace])
    else: 
        ntrace = np.vstack([-trace, trace])
    nspec = np.vstack([spec, spec])
    YM, XM = np.indices(ntrace.shape)
    yz = np.linspace(0, 1031, 25)
    plaw = []
    for yi in yz:
        d = np.sqrt((yi - ntrace[:, ::43])**2 + (col - XM[:, ::43])**2)
        plaw.append(np.nansum(nspec[:, ::43] *
                              power_law(d, 1.4e-5, c3=2., c4=1.0,  sig=1.5)))
    return yz, np.array(plaw)   


def get_powerlaw(image, trace, spec, amp):
    '''
    Solve for scatter light from powerlaw
    
    Parameters
    ----------
    image : 2d numpy array
        fits image
    trace : 2d numpy array
        y position as function of x for each fiber
    spec : 2d numpy array
        fiber spectra
    amp : str
        amplifier

    Returns
    -------
    plaw : 2d numpy array
        scatter light image from powerlaw
    '''
    fibgap = np.where(np.diff(trace[:, 400]) > 10.)[0]
    X = np.arange(image.shape[1])
    yind, xind = np.indices(image.shape)
    XV = np.array_split(X, 25)
    T = np.array_split(trace, 25, axis=1)
    XM, YM, ZM = ([], [], [])
    for xchunk, tchunk in zip(XV, T):
        avgy, avgz = ([], [])
        avgx = int(np.mean(xchunk))
        x, y = ([], [])
        dy = np.array(np.ceil(trace[0, xchunk])-7, dtype=int)
        for j, xc in enumerate(xchunk):
            d = np.arange(0, dy[j])
            if len(d):
                y.append(d)
                x.append([xc] * len(d))
        if len(y):
            y, x = [np.array(np.hstack(i), dtype=int) for i in [y, x]]
            avgy.append(np.mean(y))
            avgz.append(np.median(image[y, x]))
        for fib in fibgap:
            x, y = ([], [])
            dy = np.array(np.ceil(trace[fib, xchunk])+7, dtype=int)
            dy2 = np.array(np.ceil(trace[fib+1, xchunk])-7, dtype=int)
            for j, xc in enumerate(xchunk):
                d = np.arange(dy[j], dy2[j])
                if len(d):
                    y.append(d)
                    x.append([xc] * len(d))
            if len(y):
                y, x = [np.array(np.hstack(i), dtype=int) for i in [y, x]]
                avgy.append(np.mean(y))
                avgz.append(np.median(image[y, x]))
        x, y = ([], [])
        dy = np.array(np.ceil(trace[-1, xchunk])+7, dtype=int)
        for j, xc in enumerate(xchunk):
            d = np.arange(dy[j], image.shape[1])
            if len(d):
                y.append(d)
                x.append([xc] * len(d))
        if len(y):
            y, x = [np.array(np.hstack(i), dtype=int) for i in [y, x]]
            avgy.append(np.mean(y))
            avgz.append(np.median(image[y, x]))
        yz, plaw_col = get_powerlaw_ydir(trace, spec, amp, avgx)
        norm = np.nanmedian(np.array(avgz) / np.interp(avgy, yz, plaw_col))
        XM.append([avgx] * len(yz))
        YM.append(yz)
        ZM.append(plaw_col * norm)
    XM, YM = (np.hstack(XM), np.hstack(YM))
    xi, yi = (np.unique(XM), np.unique(YM))
    I = interp2d(xi, yi, np.hstack(ZM).reshape(len(yi), len(xi)), kind='cubic',
                 bounds_error=False)
    plaw = I(xind[0, :], yind[:, 0]).swapaxes(0, 1)
    return plaw


def get_trace_reference(specid, ifuslot, ifuid, amp, obsdate,
                       virusconfig='/work/03946/hetdex/maverick/virus_config'):
    files = glob.glob(op.join(virusconfig, 'Fiber_Locations', '*',
                              'fiber_loc_%s_%s_%s_%s.txt' %
                              (specid, ifuslot, ifuid, amp)))
    dates = [op.basename(op.dirname(fn)) for fn in files]
    obsdate = datetime(int(obsdate[:4]), int(obsdate[4:6]),
                       int(obsdate[6:]))
    timediff = np.zeros((len(dates),))
    for i, datei in enumerate(dates):
        d = datetime(int(datei[:4]), int(datei[4:6]),
                     int(datei[6:]))
        timediff[i] = np.abs((obsdate - d).days)
    ref_file = np.loadtxt(files[np.argmin(timediff)])
    return ref_file


def get_trace(twilight, specid, ifuslot, ifuid, amp, obsdate, tr_folder):
    ref = get_trace_reference(specid, ifuslot, ifuid, amp, obsdate,
                              virusconfig=tr_folder)
    N1 = (ref[:, 1] == 0.).sum()
    good = np.where(ref[:, 1] == 0.)[0]

    def get_trace_chunk(flat, XN):
        YM = np.arange(flat.shape[0])
        inds = np.zeros((3, len(XN)))
        inds[0] = XN - 1.
        inds[1] = XN + 0.
        inds[2] = XN + 1.
        inds = np.array(inds, dtype=int)
        Trace = (YM[inds[1]] - (flat[inds[2]] - flat[inds[0]]) /
                 (2. * (flat[inds[2]] - 2. * flat[inds[1]] + flat[inds[0]])))
        return Trace
    image = twilight
    N = 40
    xchunks = np.array([np.mean(x) for x in
                        np.array_split(np.arange(image.shape[1]), N)])
    chunks = np.array_split(image, N, axis=1)
    flats = [np.median(chunk, axis=1) for chunk in chunks]
    Trace = np.zeros((len(ref), len(chunks)))
    k = 0
    for flat, x in zip(flats, xchunks):
        diff_array = flat[1:] - flat[:-1]
        loc = np.where((diff_array[:-1] > 0.) * (diff_array[1:] < 0.))[0]
        
        peaks = flat[loc+1]
        loc = loc[peaks > 0.1 * np.median(peaks)]+1
        trace = get_trace_chunk(flat, loc)
        T = np.zeros((len(ref)))
        if len(trace) == N1:
            T[good] = trace
            for missing in np.where(ref[:, 1] == 1)[0]:
                gind = np.argmin(np.abs(missing - good))
                T[missing] = (T[good[gind]] + ref[missing, 0] -
                              ref[good[gind], 0])
        if len(trace) == len(ref):
            T = trace
        Trace[:, k] = T
        k += 1
    x = np.arange(twilight.shape[1])
    trace = np.zeros((Trace.shape[0], twilight.shape[1]))
    for i in np.arange(Trace.shape[0]):
        sel = Trace[i, :] > 0.
        trace[i] = np.polyval(np.polyfit(xchunks[sel], Trace[i, sel], 7), x)
    return trace, ref


def get_spectra(array_flt, array_trace, npix=5):
    '''
    Extract spectra by dividing the flat field and averaging the central
    two pixels
    
    Parameters
    ----------
    array_flt : 2d numpy array
        twilight image
    array_trace : 2d numpy array
        trace for each fiber
    wave : 2d numpy array
        wavelength for each fiber
    def_wave : 1d numpy array [GLOBAL]
        rectified wavelength
    
    Returns
    -------
    twi_spectrum : 2d numpy array
        rectified twilight spectrum for each fiber  
    '''
    spec = np.zeros((array_trace.shape[0], array_trace.shape[1]))
    N = array_flt.shape[0]
    x = np.arange(array_flt.shape[1])
    LB = int(npix/2)
    HB = -LB + npix
    for fiber in np.arange(array_trace.shape[0]):
        if np.round(array_trace[fiber]).min() < LB:
            continue
        if np.round(array_trace[fiber]).max() >= (N-LB):
            continue
        indv = np.round(array_trace[fiber]).astype(int)
        for j in np.arange(-LB, HB):
            spec[fiber] += array_flt[indv+j, x] / npix
    return spec


def get_ifucenfile(folder, ifuid, amp):
    if ifuid == '004':
        ifucen = np.loadtxt(op.join(folder, 'IFUcen_files',
                            'IFUcen_HETDEX_reverse_R.txt'),
                            usecols=[0,1,2,4], skiprows=30)
        ifucen[224:,:] = ifucen[-1:223:-1,:]
    else:
        ifucen = np.loadtxt(op.join(folder, 'IFUcen_files',
                                    'IFUcen_HETDEX.txt'),
                            usecols=[0,1,2,4], skiprows=30)
        if ifuid in ['003','005','008']:
            ifucen[224:,:] = ifucen[-1:223:-1,:]
        if ifuid == '007':
            a = ifucen[37,1:3] * 1.
            b = ifucen[38,1:3] * 1.
            ifucen[37,1:3] = b
            ifucen[38,1:3] = a
        if ifuid == '025':
            a = ifucen[208,1:3] * 1.
            b = ifucen[213,1:3] * 1.
            ifucen[208,1:3] = b
            ifucen[213,1:3] = a
        if ifuid == '030':
            a = ifucen[445,1:3] * 1.
            b = ifucen[446,1:3] * 1.
            ifucen[445,1:3] = b
            ifucen[446,1:3] = a
        if ifuid == '038':
            a = ifucen[302,1:3] * 1.
            b = ifucen[303,1:3] * 1.
            ifucen[302,1:3] = b
            ifucen[303,1:3] = a
        if ifuid == '041':
            a = ifucen[251,1:3] * 1.
            b = ifucen[252,1:3] * 1.
            ifucen[251,1:3] = b
            ifucen[252,1:3] = a

    if amp=="LL":
        return ifucen[112:224,1:3][::-1,:]
    if amp=="LU":
        return ifucen[:112,1:3][::-1,:]
    if amp=="RL":
        return ifucen[224:336,1:3][::-1,:]               
    if amp=="RU":
        return ifucen[336:,1:3][::-1,:] 
    
def safe_sigma_clip(y):
    try:
        mask = sigma_clip(y, masked=True, maxiters=None,
                          stdfunc=mad_std)
    except:
        mask = sigma_clip(y, iters=None, stdfunc=mad_std) 
    return mask

def identify_sky_pixels(sky, kernel=10.0):
    G = Gaussian1DKernel(kernel)
    cont = convolve(sky, G, boundary='extend')
    mask = safe_sigma_clip(sky - cont)
    for i in np.arange(5):
        nsky = sky * 1.
        mask.mask[1:] += mask.mask[:-1]
        mask.mask[:-1] += mask.mask[1:]
        nsky[mask.mask] = np.nan
        cont = convolve(nsky, G, boundary='extend')
        while np.isnan(cont).sum():
            cont = interpolate_replace_nans(cont, G)
        mask = safe_sigma_clip(sky - cont)
    return mask.mask, cont

def find_peaks(y, thresh=10.):
    def get_peaks(flat, XN):
        YM = np.arange(flat.shape[0])
        inds = np.zeros((3, len(XN)))
        inds[0] = XN - 1.
        inds[1] = XN + 0.
        inds[2] = XN + 1.
        inds = np.array(inds, dtype=int)
        Peaks = (YM[inds[1]] - (flat[inds[2]] - flat[inds[0]]) /
                 (2. * (flat[inds[2]] - 2. * flat[inds[1]] + flat[inds[0]])))
        return Peaks
    diff_array = y[1:] - y[:-1]
    loc = np.where((diff_array[:-1] > 0.) * (diff_array[1:] < 0.))[0]
    peaks = y[loc+1]
    loc = loc[peaks > thresh]+1
    peak_loc = get_peaks(y, loc)
    peaks = y[np.round(peak_loc).astype(int)]
    return peak_loc, peaks

def get_wave_single(y, T_array, order=3, thresh=10.):
    mask, cont = identify_sky_pixels(y)
    loc, peaks = find_peaks(y - cont, thresh=thresh)
    x = np.arange(len(y))
    dw = T_array[-1] - T_array[0]
    dx = loc[-1] - loc[0]
    wave = (x-loc[0]) * dw / dx + T_array[0]
    guess_wave = np.interp(loc, x, wave)
    diff = guess_wave[:, np.newaxis] - T_array
    ind = np.argmin(np.abs(diff), axis=0)
    P = np.polyfit(loc[ind], T_array, order)
    yv = np.polyval(P, loc[ind])
    res = np.std(T_array-yv)
    wave = np.polyval(P, x)
    return wave, res

def get_wave(spec, trace, T_array, res_lim=1., order=3):
    w = trace * 0.
    r = np.zeros((trace.shape[0],))
    xi = np.hstack([np.arange(0, trace.shape[0], 8), trace.shape[0]-1])
    failed = True
    thresh=20.
    while failed:
        for j in xi:
            try:
                wave, res = get_wave_single(spec[j], T_array, order=order,
                                            thresh=thresh)
                w[j] = wave
                r[j] = res
            except:
                pass
        sel = (np.array(r) < res_lim) * (np.array(r) > 0.)
        if sel.sum() < 7:
            thresh -= 5.
        else:
            failed = False
        if thresh < 5.:
            return None
    wave = w * 0.
    xi = np.hstack([np.arange(0, trace.shape[1], 24), trace.shape[1]-1])
    for i in xi:
        x = trace[:, i]
        y = w[:, i]
        wave[:, i] = np.polyval(np.polyfit(x[sel], y[sel], order), x)
    x = np.arange(wave.shape[1])
    for j in np.arange(wave.shape[0]):
        sel = wave[j] > 0.
        wave[j] = np.polyval(np.polyfit(x[sel], wave[j][sel], order), x)
    return wave

def get_pixelmask(dark):
    y1 = dark - np.median(dark, axis=0)[np.newaxis, :]
    y1 = y1 - np.median(y1, axis=1)[:, np.newaxis]
    m, m1, s = sigma_clipped_stats(y1)
    mask = np.abs(y1) > 5 * s
    yind, xind = np.where(y1 > 100)
    for yi, xi in zip(yind, xind):
        if yi + 2 < mask.shape[0]:
            if mask[yi+1, xi]:
                mask[:, xi] = True
    return np.array(mask, dtype=int)