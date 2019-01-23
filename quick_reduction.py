# -*- coding: utf-8 -*-
"""
Created on Tue Jan 22 15:11:39 2019

@author: gregz
"""

import fitsio
import glob
import numpy as np
import os.path as op
import sys

from astropy.io import fits
from astropy.stats import biweight_location, biweight_midvariance
from datetime import datetime, timedelta
from input_utils import setup_logging


def orient_image(image, amp, ampname):
    '''
    Orient the images from blue to red (left to right)
    Fibers are oriented to match configuration files
    '''
    if amp == "LU":
        image[:] = image[::-1, ::-1]
    if amp == "RL":
        image[:] = image[::-1, ::-1]
    if ampname is not None:
        if ampname == 'LR' or ampname == 'UL':
            image[:] = image[:, ::-1]
    return image


def make_avg_spec(wave, spec, binsize=35, per=50):
    ind = np.argsort(wave.ravel())
    T = 1
    for p in wave.shape:
        T *= p
    wchunks = np.array_split(wave.ravel()[ind],
                             T / binsize)
    schunks = np.array_split(spec.ravel()[ind],
                             T / binsize)
    nwave = np.array([np.mean(chunk) for chunk in wchunks])
    nspec = np.array([np.percentile(chunk, per) for chunk in schunks])
    nwave, nind = np.unique(nwave, return_index=True)
    return nwave, nspec[nind]


def base_reduction(filename, get_header=False):
    a = fits.open(filename)
    image = np.array(a[0].data, dtype=float)
    # overscan sub
    overscan_length = 32 * (image.shape[1] / 1064)
    O = biweight_location(image[:, -(overscan_length-2):])
    image[:] = image - O
    # trim image
    image = image[:, :-overscan_length]
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
    a = orient_image(image, amp, ampname) * gain
    E = np.sqrt(rdnoise**2 + np.where(a > 0., a, 0.))
    if get_header:
        return a, E, header
    return a, E


def get_masterbias(zro_path, amp):
    files = glob.glob(zro_path.replace('LL', amp))
    listzro = []
    for filename in files:
        a, error = base_reduction(filename)
        listzro.append(a)
    return np.median(listzro, axis=0)


def get_masterarc(arc_path, amp, arc_names, masterbias, specname):
    files = glob.glob(arc_path.replace('LL', amp))
    listarc, listarce = ([], [])
    for filename in files:
        f = fits.open(filename)
        if f[0].header['OBJECT'].lower() in arc_names:
            a, e = base_reduction(filename)
            a[:] -= masterbias
            listarc.append(a)
            listarce.append(e)
    listarc, listarce = [np.array(x) for x in [listarc, listarce]]
    return np.sum(listarc, axis=0)


def get_mastertwi(twi_path, amp, masterbias):
    files = glob.glob(twi_path.replace('LL', amp))
    listtwi = []
    for filename in files:
        a, e = base_reduction(filename)
        a[:] -= masterbias
        listtwi.append(a)
    twi_array = np.array(listtwi, dtype=float)
    norm = np.median(twi_array, axis=(1, 2))[:, np.newaxis, np.newaxis]
    return np.median(twi_array / norm, axis=0)


def get_trace_reference(specid, ifuslot, ifuid, amp, obsdate,
                        virusconfig='/work/03946/hetdex/'
                        'maverick/virus_config'):
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


def get_trace(twilight, specid, ifuslot, ifuid, amp, obsdate):
    ref = get_trace_reference(specid, ifuslot, ifuid, amp, obsdate)
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
        Trace[:, k] = T
        k += 1
    x = np.arange(twilight.shape[1])
    trace = np.zeros((Trace.shape[0], twilight.shape[1]))
    for i in np.arange(Trace.shape[0]):
        sel = Trace[i, :] > 0.
        trace[i] = np.polyval(np.polyfit(xchunks[sel], Trace[i, sel], 7), x)
    return trace, ref


def find_peaks(y, thresh=8.):
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
    std = np.sqrt(biweight_midvariance(y))
    loc = loc[peaks > (thresh * std)]+1
    peak_loc = get_peaks(y, loc)
    peaks = y[np.round(peak_loc).astype(int)]
    return peak_loc, peaks/std, peaks


def get_cal_path(pathname, date):
    datec = date
    daten = date
    cnt = 0
    while len(glob.glob(pathname)) == 0:
        datec_ = datetime(int(daten[:4]), int(daten[4:6]), int(daten[6:]))
        daten_ = datec_ - timedelta(days=1)
        daten = '%04d%02d%02d' % (daten_.year, daten_.month, daten_.day)
        pathname = pathname.replace(datec, daten)
        cnt += 1
        log.info(pathname)
        if cnt > 30:
            log.error('Could not find cal within 30 days.')
            break
    return pathname, daten


def get_ifucenfile(side, amp, virusconfig=op.join(DIRNAME, 'config'),
                   skiprows=4):

    file_dict = {"uv": "LRS2_B_UV_mapping.txt",
                 "orange": "LRS2_B_OR_mapping.txt",
                 "red": "LRS2_R_NR_mapping.txt",
                 "farred": "LRS2_R_FR_mapping.txt"}

    ifucen = np.loadtxt(op.join(virusconfig, 'IFUcen_files',
                        file_dict[side]), usecols=[0, 1, 2], skiprows=skiprows)

    if amp == "LL":
        return ifucen[140:, 1:3][::-1, :]
    if amp == "LU":
        return ifucen[:140, 1:3][::-1, :]
    if amp == "RL":
        return ifucen[:140, 1:3][::-1, :]
    if amp == "RU":
        return ifucen[140:, 1:3][::-1, :]


def get_script_path():
    return op.dirname(op.realpath(sys.argv[0]))
    
def get_spectra(array_flt, array_trace):
    spectrum = array_trace * 0.
    x = np.arange(array_flt.shape[1])
    for fiber in np.arange(array_trace.shape[0]):
        indl = np.floor(array_trace[fiber]).astype(int)
        indh = np.ceil(array_trace[fiber]).astype(int)
        spectrum[fiber] = array_flt[indl, x] / 2. + array_flt[indh, x] / 2.
    return spectrum

DIRNAME = get_script_path()
instrument = 'virus'
log = setup_logging()

for ifuslot in ifuslots:
    for amp in amps:
        amppos = get_ifucenfile(specname, amp)
        ##############
        # MASTERBIAS #
        ##############
        log.info('Getting Masterbias for ifuslot, %s, and amp, %s' %
                 (ifuslot, amp))
        zro_path = bias_path % (instrument, instrument, '00000*', instrument,
                                ifuslot)
        zro_path, newdate = get_cal_path(zro_path, args.date)
        if newdate != args.date:
            log.info('Found bias files on %s and using them for %s' % (newdate, args.date))
        masterbias = get_masterbias(zro_path, amp)

        #####################
        # MASTERTWI [TRACE] #
        #####################
        log.info('Getting MasterFlat for ifuslot, %s, and amp, %s' %
                 (ifuslot, amp))
        twibase, newdate = get_cal_path(twibase, args.date)

        if newdate != args.date:
            log.info('Found trace files on %s and using them for %s' % (newdate, args.date))
        masterflt = get_mastertwi(twibase, amp, masterbias)

        log.info('Getting Trace for ifuslot, %s, and amp, %s' %
                 (ifuslot, amp))
        trace, dead = get_trace(masterflt, specid, ifuslot, ifuid, amp,
                                twi_date)
        #fits.PrimaryHDU(trace).writeto('test_trace.fits', overwrite=True)

        ##########################
        # MASTERARC [WAVELENGTH] #
        ##########################
        log.info('Getting MasterArc for ifuslot, %s, and amp, %s' %
                 (ifuslot, amp))
        lamp_path = cmp_path % (instrument, instrument, '00000*', instrument,
                                ifuslot)
        lamp_path, newdate = get_cal_path(lamp_path, args.date)
        if newdate != args.date:
            log.info('Found lamp files on %s and using them for %s' % (newdate, args.date))
        masterarc = get_masterarc(lamp_path, amp, arc_names, masterbias,
                                  specname)
        #fits.PrimaryHDU(masterarc).writeto('wtf_%s_%s.fits' % (ifuslot, amp), overwrite=True)
        log.info('Getting Wavelength for ifuslot, %s, and amp, %s' %
                 (ifuslot, amp))
        wave = get_wavelength_from_arc(masterarc, trace, arc_lines, specname, amp)
        #fits.PrimaryHDU(wave).writeto('test_wave.fits', overwrite=True)

        #################################
        # TWILIGHT FLAT [FIBER PROFILE] #
        #################################
        log.info('Getting bigW for ifuslot, %s, and amp, %s' %
                 (ifuslot, amp))
        bigW = get_bigW(amp, wave, trace, masterbias)
        package.append([wave, trace, bigW, masterbias, amppos, dead])
        marc.append([masterarc])