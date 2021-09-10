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
import subprocess
import sys
import tables
import tarfile

from astropy.coordinates import SkyCoord, EarthLocation
from astropy.io import fits
from astropy.table import Table
from astropy.time import Time
import astropy.units as u
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

def get_rv_cor(item_list):
    try:
        sc = SkyCoord(item_list[0]*u.deg, item_list[1]*u.deg)
        s = item_list[2]
        t = Time(s, format='mjd') + item_list[3] / 2. * u.second
        vcorr = sc.radial_velocity_correction(kind='barycentric', obstime=t,
                                              location=loc)
        vcorr = vcorr.value
        mjd = t.mjd
        log.info('Barycentric RV correction: %0.2f, %0.2f' % (vcorr, mjd))
    except:
        vcorr = 0.0
        mjd = 0.0
        log.warning('Barycentric RV correction failed for this source')
    return vcorr, mjd
        
def get_tarinfo(tarfolder):
    T = tarfile.open(tarfolder, 'r')
    flag = True
    while flag:
        try:
            a = T.next()
            name = a.name
        except:
            flag = False
            continue
        if name[-5:] == '.fits':
            b = fits.open(T.extractfile(a))
            item_list = [b[0].header['QRA'], b[0].header['QDEC'],
                         b[0].header['DATE-OBS'], b[0].header['EXPTIME']]
            flag = False
    T.close()
    return item_list

def get_itemlist(filename):
    date = filename.split('_')[0]
    obs = filename.split('_')[1][:7]
    fn = op.join(basedir, '%ssci' % date[:6])
    process = subprocess.Popen('cat %s | grep %s | grep %s ' %
                                       (fn, date, obs),
                                       stdout=subprocess.PIPE, shell=True)
    line = process.stdout.readline()
    b = line.rstrip().decode("utf-8")
    try:
        exptime = float(b.split(' ')[2]) 
        trajcra = float(b.split(' ')[5]) * 15.
        trajcdec = float(b.split(' ')[6])
        mjd = float(b.split(' ')[4])
    except:
        exptime = 180.
        trajcra = 180.
        trajcdec = 50.
        mjd = -999.
    return [trajcra, trajcdec, mjd, exptime]



outname = sys.argv[1]

def_wave = np.linspace(3470, 5540, 1036)

log = setup_logging('catalog')

loc = EarthLocation.of_site('McDonald Observatory')

basedir = '/work/00115/gebhardt/maverick/gettar'
folder = '/work/03946/hetdex/virus_parallels/data'
DIRNAME = get_script_path()
T = Table.read(op.join(DIRNAME, 'filters/ps1g.dat'), format='ascii')
filtg = np.interp(def_wave, T['col1'], T['col2'], left=0.0, right=0.0)
filtg /= filtg.sum()

h5names = sorted(glob.glob(op.join(folder, '*.h5')))
totN = 0
Nshots = 0
cnt = 0
C = 0
ralist, declist, exptimelist, goodlist, nstarslist  = ([], [], [], [], [])
raofflist, decofflist, rastdlist, decstdlist  = ([], [], [], [])
for niter in np.arange(2):
    if niter == 1:
        RA = np.zeros((totN,))
        DEC = np.zeros((totN,))
        GMAG = np.zeros((totN,))
        RMAG = np.zeros((totN,))
        IMAG = np.zeros((totN,))
        ZMAG = np.zeros((totN,))
        YMAG = np.zeros((totN,))

        SN = np.zeros((totN,))
        EP = np.zeros((totN,))
        VCOR = np.zeros((totN,))
        MJD = np.zeros((totN,))
        NAME = np.zeros((totN,), dtype='|S16')
        SPEC = np.zeros((totN, 1036))
        ERROR = np.zeros((totN, 1036))
        WEIGHT = np.zeros((totN, 1036))
        process = psutil.Process(os.getpid())
        EA = np.zeros((Nshots, 1036))
        SS = np.zeros((Nshots, 1036))
        log.info('Memory Used: %0.2f GB' % (process.memory_info()[0] / 1e9))
        III = 0
    for h5name in h5names:
        try:
            h5file = tables.open_file(h5name)
            name = op.basename(h5name).split('.h5')[0]
            raoffset = h5file.root.Survey.cols.raoffset[0]
        except:
            if niter == 0:
                log.info('Could not open %s' % name)
            h5file.close()
            continue
        decoffset = h5file.root.Survey.cols.decoffset[0]
        rastd = h5file.root.Survey.cols.rastd[0]
        ral = h5file.root.Survey.cols.ra[0]
        decl = h5file.root.Survey.cols.dec[0]
        exptimel = h5file.root.Survey.cols.exptime[0]
        decstd = h5file.root.Survey.cols.decstd[0]
        nstars = h5file.root.Survey.cols.nstarsastrom[0]
        rule1 = np.sqrt(raoffset**2 + decoffset**2) < 0.5
        rule2 = exptimel > 300.
        rule5 = nstars >= 5
        good = rule1 * rule5 * rule2
        if ~good:
            if niter == 0:
                log.info('%s did not make cut' % name)
            h5file.close()
            if niter == 0:
                ralist.append(ral)
                declist.append(decl)
                exptimelist.append(exptimel)
                goodlist.append(False)
                nstarslist.append(nstars)
                raofflist.append(raoffset)
                decofflist.append(decoffset)
                rastdlist.append(rastd)
                decstdlist.append(decstd)
            continue
        if niter == 0:
            ralist.append(ral)
            declist.append(decl)
            exptimelist.append(exptimel)
            goodlist.append(True)
            nstarslist.append(nstars)
            raofflist.append(raoffset)
            decofflist.append(decoffset)
            rastdlist.append(rastd)
            decstdlist.append(decstd)
            C += len(h5file.root.Info)
        spectra = h5file.root.CatSpectra.cols.spectrum[:]
        error = h5file.root.CatSpectra.cols.error[:]
        weight = h5file.root.CatSpectra.cols.weight[:]
        gmag = h5file.root.CatSpectra.cols.gmag[:]
        rmag = h5file.root.CatSpectra.cols.rmag[:]
        imag = h5file.root.CatSpectra.cols.imag[:]
        zmag = h5file.root.CatSpectra.cols.zmag[:]
        ymag = h5file.root.CatSpectra.cols.ymag[:]
        ra = h5file.root.CatSpectra.cols.ra[:]
        dec = h5file.root.CatSpectra.cols.dec[:]
        avgdec = np.mean(dec)
        dra = (ra[:, np.newaxis] - ra[np.newaxis, :])* np.cos(np.deg2rad(avgdec))
        ddec = (dec[:, np.newaxis] - dec[np.newaxis, :])
        dist = np.sqrt(dra**2 + ddec**2) * 3600.
        neigh = (dist < 1.).sum(axis=1)
        inds = np.where(neigh>1)[0]
        clean = np.zeros((len(dist),), dtype=bool)
        clean[neigh<=1] = True
        for ind in inds:
            test = (dist[ind, :ind] < 1.).sum() < 1
            if test:
                clean[ind] = True
        mask = (weight > 0.05) * np.isfinite(spectra)
        num = np.nansum(filtg[np.newaxis, :] * mask * (spectra / error), axis=1) 
        denom = np.nansum(filtg[np.newaxis, :] * mask, axis=1)
        sn = num / denom
        goodspec = (mask.sum(axis=1) > 0.8) * (sn > 1.) * clean
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
            inds = np.random.randint(N, size=20)
            es = np.zeros((20, 1036))
            ss = np.zeros((20, 1036))
            for j, ind in enumerate(inds):
                es[j] = h5file.root.Fibers.cols.error[ind]
                ss[j] = h5file.root.Fibers.cols.skyspectrum[ind]
            EA[III] = np.nanmedian(es, axis=0)
            SS[III] = np.nanmedian(ss, axis=0)
            item_list = get_itemlist(op.basename(h5name))
            vcor, mjd = get_rv_cor(item_list)
            RA[cnt:cnt+N] = ra[goodspec]
            DEC[cnt:cnt+N] = dec[goodspec]
            GMAG[cnt:cnt+N] = gmag[goodspec]
            RMAG[cnt:cnt+N] = rmag[goodspec]
            IMAG[cnt:cnt+N] = imag[goodspec]
            ZMAG[cnt:cnt+N] = zmag[goodspec]
            YMAG[cnt:cnt+N] = ymag[goodspec]
            
            VCOR[cnt:cnt+N] = vcor * np.ones((goodspec.sum(),))
            MJD[cnt:cnt+N] = mjd * np.ones((goodspec.sum(),))
            EP[cnt:cnt+N] = exptimel * np.ones((goodspec.sum(),))
            SN[cnt:cnt+N] = sn[goodspec]
            SPEC[cnt:cnt+N] = norm_spectra
            ERROR[cnt:cnt+N] = error[goodspec] * normalization[:, np.newaxis] 
            WEIGHT[cnt:cnt+N] = weight[goodspec]
            NAME[cnt:cnt+N] = name
            cnt += N
            III += 1
        h5file.close()

inds = np.argsort(RA)
dupl = []
keep = []
r = RA[inds]
d = DEC[inds]
for ind in inds:
    if ind in dupl:
        continue
    ll = int(np.max([ind-100, 0]))
    hl = int(np.min([ind+100, len(inds)+1]))
    dra = np.cos(d[ind] * np.pi / 180.) * (r[ind] - r[ll:hl])
    ddec = (d[ind] - d[ll:hl])
    d = np.sqrt(dra**2 + ddec**2)*3600.
    sel = d < 1.
    stack_inds = inds[sel]
    for i in inds[sel]:
        dupl.append(i)
    I = inds[sel][np.argmax(SN[inds[sel]])]
    keep.append(I)
log.info('Number of unique sources is %i' % (len(keep)))
IDS = np.arange(1, len(RA)+1)
IDS = ['HETVIPS%09d' % iD for iD in IDS]
process = psutil.Process(os.getpid())
log.info('Memory Used: %0.2f GB' % (process.memory_info()[0] / 1e9))
T = Table([IDS, RA, DEC, NAME, GMAG, RMAG, IMAG, ZMAG, YMAG, SN, VCOR, MJD, EP],
          names=['objID', 'RA', 'Dec', 'shotid', 'gmag', 'rmag', 'imag', 'zmag', 'ymag',
                 'sn', 'barycor', 'mjd', 'exptime'])
T2 = Table([ralist, declist, exptimelist, goodlist, nstarslist, raofflist,
            decofflist, rastdlist, decstdlist], 
           names=['RA', 'Dec', 'exptime', 'good', 'nstars', 'raoff', 'decoff',
                  'rastd', 'decstd'])
T2.write('survey_info.dat', format='ascii.fixed_width_two_line')
fits.HDUList([fits.PrimaryHDU(), fits.BinTableHDU(T), fits.ImageHDU(SPEC),
              fits.ImageHDU(ERROR), fits.ImageHDU(WEIGHT)]).writeto(outname, overwrite=True)
fits.HDUList([fits.PrimaryHDU(EA), fits.ImageHDU(SS)]).writeto('shot_error_sky.fits', overwrite=True)
IDS = np.arange(1, len(RA[keep])+1)
IDS = ['HETVIPS%09d' % iD for iD in IDS]
T = Table([IDS, RA[keep], DEC[keep], NAME[keep], GMAG[keep], RMAG[keep], 
           IMAG[keep], ZMAG[keep], YMAG[keep], SN[keep], VCOR[keep], MJD[keep],
           EP[keep]],
          names=['objID', 'RA', 'Dec', 'shotid', 'gmag', 'rmag', 'imag', 'zmag', 'ymag',
                 'sn', 'barycor', 'mjd', 'exptime'])
fits.HDUList([fits.PrimaryHDU(), fits.BinTableHDU(T), fits.ImageHDU(SPEC[keep]),
              fits.ImageHDU(ERROR[keep]), fits.ImageHDU(WEIGHT[keep])]).writeto(outname.replace('.fits', '_unique.fits'), overwrite=True)
log.info('Total Number of sources is %i for %i shots' % (totN, Nshots))
log.info('Total Number of fibers is %i' % (C))