# -*- coding: utf-8 -*-
"""
Created on Mon Jun 11 11:32:27 2018

@author: gregz
"""

import os.path as op
import numpy as np
import subprocess
import sys
import tarfile
import warnings
import tables as tb
import glob

from astropy.convolution import convolve, Gaussian2DKernel
from astropy.io import fits
from astropy.stats import mad_std
from astropy.table import Table
from datetime import datetime, timedelta
from distutils.dir_util import mkpath
from fiber_utils import base_reduction, get_trace, get_spectra, get_spectra_error
from fiber_utils import get_powerlaw, get_ifucenfile, get_wave, get_pixelmask
from fiber_utils import get_pixelmask_flt
from fiber_utils import measure_contrast, get_bigW, get_bigF
from input_utils import setup_parser, set_daterange, setup_logging
from math_utils import biweight
from scipy.interpolate import interp1d

karl_tarlist = '/work/00115/gebhardt/maverick/gettar/%starlist'
karl_scilist = '/work/00115/gebhardt/maverick/gettar/%ssci'

warnings.filterwarnings("ignore")

def get_script_path():
    '''
    Get script path, aka, where does Remedy live?
    '''
    return op.dirname(op.realpath(sys.argv[0]))

class VIRUSImage(tb.IsDescription):
    wavelength = tb.Float32Col((112, 1032))
    trace = tb.Float32Col((112, 1032))
    lampspec = tb.Float32Col((112, 1032))
    scispec = tb.Float32Col((112, 1032))
    maskspec = tb.Float32Col((112, 1032))
    masterdark = tb.Float32Col((1032, 1032))
    ifupos = tb.Float32Col((112, 2))
    ifuslot = tb.Int32Col()
    ifuid = tb.StringCol(3)
    specid = tb.StringCol(3)
    contid = tb.StringCol(8)
    amp = tb.StringCol(2)
    readnoise = tb.Float32Col()
    pixelmask = tb.Int32Col((1032, 1032))
    masterflt = tb.Float32Col((1032, 1032))
    mastertwi = tb.Float32Col((1032, 1032))
    mastercmp = tb.Float32Col((1032, 1032))
    mastersci = tb.Float32Col((1032, 1032))
    masterbias = tb.Float32Col((1032, 1032))


def append_fibers_to_table(row, wave, trace, ifupos, ifuslot, ifuid, specid,
                           amp, readnoise, pixelmask, masterdark, masterflt,
                           mastertwi, mastercmp, mastersci, masterbias, spec,
                           lampspec, maskspec, contid):
    row['wavelength'] = wave * 1.
    row['trace'] = trace * 1.
    row['ifupos'] = ifupos * 1.
    row['ifuslot'] = ifuslot
    row['ifuid'] = ifuid
    row['specid'] = specid
    row['contid'] = contid
    row['readnoise'] = readnoise
    row['pixelmask'] = pixelmask
    row['masterdark'] = masterdark
    row['mastertwi'] = mastertwi
    row['masterflt'] = masterflt
    row['mastersci'] = mastersci
    row['masterbias'] = masterbias
    row['scispec'] = spec
    row['mastercmp'] = mastercmp
    row['lampspec'] = lampspec
    row['maskspec'] = maskspec
    row['amp'] = amp
    row.append()
    return True


def get_filenames(args, daterange, kind):
    filenames = []
    ifuslot = '%03d' % int(args.ifuslot)
    dates = []
    for date in daterange:
        date = '%04d%02d' % (date.year, date.month)
        if date not in dates:
            dates.append(date)
    
    for date in dates:
        tname = karl_tarlist % date
        for daten in daterange:
            daten = '%04d%02d%02d' % (daten.year, daten.month, daten.day)
            process = subprocess.Popen('cat %s | grep %s | grep _%sLL | '
                                       'grep %s' %
                                       (tname, kind, ifuslot, daten),
                                       stdout=subprocess.PIPE, shell=True)
            while True:
                line = process.stdout.readline()
                if not line:
                    break
                b = line.rstrip().decode("utf-8")
                c = op.join(args.rootdir, b)
                filenames.append(c[:-14])
    return filenames

def get_scifilenames(args, daterange, kind):
    filenames = []
    dates = []
    for date in daterange:
        date = '%04d%02d' % (date.year, date.month)
        if date not in dates:
            dates.append(date)
    
    for date in dates:
        tname = karl_scilist % date
        for daten in daterange:
            daten = '%04d%02d%02d' % (daten.year, daten.month, daten.day)
            process = subprocess.Popen('cat %s | grep %s | grep %s ' %
                                       (tname, daten, 'DEX'),
                                       stdout=subprocess.PIPE, shell=True)
            while True:
                line = process.stdout.readline()
                if not line:
                    break
                b = line.rstrip().decode("utf-8")
                b = b.split(' ')[0].split('maverick/')[1]
                c = op.join(args.rootdir, b)
                filenames.append(c[:-14])
    return filenames

def get_tarfiles(filenames):
    tarnames = []
    for fn in filenames:
        tarbase = op.dirname(op.dirname(op.dirname(fn))) + '.tar'
        if tarbase not in tarnames:
            if '/work2' in tarbase:
                tarnames.append(tarbase.replace('/work2', '/work'))
            else:
                tarnames.append(tarbase.replace('/work', '/work'))
    return tarnames

def get_tarinfo(tarnames, filenames):
    l = []
    tn = []
    tarnames = np.unique(tarnames)
    for tarname in tarnames:
        if not op.exists(tarname):
            continue
        args.log.info('Opening %s' % tarname)
        T = tarfile.open(tarname, 'r')
        members = T.getmembers()
        names = [t.name for t in members]
        l.append([T, members, names])
        tn.append(tarname)
    tarnames = tn
    L = []
    for filename in filenames:
        tarbase = op.dirname(op.dirname(op.dirname(filename))) + '.tar'
        tarbase = tarbase.replace('/work', '/work')
        matches = np.where(tarbase == np.array(tarnames))[0]
        if not len(matches):
            continue
        ind = matches[0]
        L.append(l[ind])
    return L

def get_objects(daterange, instrument='virus', rootdir='/work/03946/hetdex/maverick'):
    dates = []
    for date in daterange:
        date = '%04d%02d%02d' % (date.year, date.month, date.day)
        if date not in dates:
            dates.append(date)
    tarfolders = []
    ifuslot = '%03d' % int(args.ifuslot)
    ifuslot_str = '_%sLL' % ifuslot
    for date in dates:
        tarnames = sorted(glob.glob(op.join(rootdir, date, instrument, '%s0000*.tar' % instrument)))
        for t in tarnames:
            tarfolders.append(t)
    objectdict = {}
    filename_list = []
    for tarfolder in tarfolders:
        date = tarfolder.split('/')[-3]
        obsnum = int(tarfolder[-11:-4])
        NEXP = 1
        T = tarfile.open(tarfolder, 'r')
        try:
            names_list = T.getnames()
            names_list = [name for name in names_list if name[-5:] == '.fits']
            for name in names_list:
                if ifuslot_str in name:
                    c = op.join(args.rootdir, date, instrument, name)
                    filename_list.append(c)
            exposures = np.unique([name.split('/')[1] for name in names_list])
            b = fits.open(T.extractfile(T.getmember(names_list[0])))
            Target = b[0].header['OBJECT']

            NEXP = len(exposures)
            for i in np.arange(NEXP):
                objectdict['%s_%07d_%02d' % (date, obsnum, i+1)] = Target
        except:
            objectdict['%s_%07d_%02d' % (date, obsnum, NEXP)] = ''
            continue
    return objectdict, filename_list

def get_ifuslots(tarfolder):
    if not op.exists(tarfolder):
        return []
    T = tarfile.open(tarfolder, 'r')
    flag = True
    ifuslots = []
    expnames = []
    while flag:
        a = T.next()
        try:
            name = a.name
        except:
            flag = False
        if name[-5:] == '.fits':
            try:
                F = fits.open(T.extractfile(a))
            except:
                continue
            quad = ''
            for keyname in ['IFUSLOT', 'SPECID', 'IFUID']:
                quad += '%03d_' % int(F[0].header[keyname])
            quad += F[0].header['CONTID']
            namelist = name.split('/')
            expn = namelist[-3]
            if quad not in ifuslots:
                ifuslots.append(quad)
            if expn not in expnames:
                expnames.append(expn)
            if len(expnames) > 1:
                flag = False
    return sorted(ifuslots)

def get_unique_ifuslots(tarfolders):
    dates = [tarfolder.split('/')[-3] for tarfolder in tarfolders]
    utars, inds = np.unique(dates, return_index=True)
    utarfolders = [tarfolders[ind] for ind in inds]
    ifuslots = []
    for tarfolder in utarfolders:
        result = get_ifuslots(tarfolder)
        ifuslots = list(np.union1d(ifuslots, result))
    return ifuslots

def expand_date_range(daterange, days):
    l1 = [daterange[0] - timedelta(days=float(x))
          for x in np.arange(1, days+1)]
    l2 = [daterange[-1] + timedelta(days=float(x))
          for x in np.arange(1, days+1)]
    return l1 + daterange + l2


def build_master_frame(file_list, tarinfo_list, ifuslot, amp, kind, log, 
                       folder, specid, ifuid, contid):
    # Create empty lists for the left edge jump, right edge jump, and structure
    objnames = ['hg', 'cd-a']
    bia_list = []
    for itm, tinfo in zip(file_list, tarinfo_list):
        fn = itm + '%s%s_%s.fits' % (ifuslot, amp, kind)
        try:            
            I, E, header = base_reduction(fn, tinfo, get_header=True)
            if (kind == 'flt') or (kind == 'twi'):
                if (np.mean(I) < 50.) or (np.mean(I) > 50000):
                    continue
            if kind == 'sci':
                if (np.mean(I) < 1.) or (np.mean(I) > 50000):
                    continue
            hspecid, hifuid = ['%03d' % int(header[name])
                               for name in ['SPECID', 'IFUID']]
            hcontid = header['CONTID']
            if (hspecid != specid) or (hifuid != ifuid) or (hcontid != contid):
                continue
            date, time = header['DATE'].split('T')
            datelist = date.split('-')
            timelist = time.split(':')
            d = datetime(int(datelist[0]), int(datelist[1]), int(datelist[2]),
                         int(timelist[0]), int(timelist[1]),
                         int(timelist[2].split('.')[0]))
            bia_list.append([I, header['SPECID'], header['OBJECT'], d,
                             header['IFUID'], header['CONTID']])
        except:
            log.warning('Could not load %s' % fn)

    # Select only the bias frames that match the input amp, e.g., "RU"
    if not len(bia_list):
        log.warning('No %s frames found for date range given' % kind)
        return None

    if kind == 'cmp':
        masterbias = np.zeros(bia_list[0][0].shape)
        masterstd = np.zeros(bia_list[0][0].shape)

        for objname in objnames:
            big_array = np.array([v[0] for v in bia_list
                                  if v[2].lower() == objname])
            masterim = np.median(big_array, axis=0)
            masterst = np.std(big_array, axis=0)
            masterbias += masterim
            masterstd += masterst / len(objnames)
    else:
        big_array = np.array([v[0] for v in bia_list])
        masterbias = np.median(big_array, axis=0)
        masterstd = (np.median(np.abs(big_array - 
                                     masterbias[np.newaxis, :, :]), axis=0) *
                     1.4826)


    log.info('Number of frames for %s: %i' % (kind, len(big_array)))
    d1 = bia_list[0][3]
    d2 = bia_list[-1][3]
    d4 = (d1 + timedelta(seconds=(d2-d1).seconds / 2.) +
          timedelta(days=(d2-d1).days / 2.))
    avgdate = '%04d%02d%02dT%02d%02d%02d' % (d4.year, d4.month, d4.day,
                                             d4.hour, d4.minute, d4.second)
    return masterbias, masterstd, avgdate, bia_list[0][1], bia_list[0][4], bia_list[0][5]

def get_specmask(amp, wave, trace, masterbias, masterdark, readnoise,
                 masterflt, mastersci):
    def_wave = np.linspace(3470., 5540., 1036*2)

    for i, master in enumerate([masterflt, mastersci]):
        master[:] -= masterbias
        if i == 1:
            master[:] -= masterdark
        plaw = get_powerlaw(master, trace)
        master[:] -= plaw
        spectra = get_spectra(master, trace)
        scirect = np.zeros((spectra.shape[0], len(def_wave)))
        for j in np.arange(spectra.shape[0]):
            scirect[j] = np.interp(def_wave, wave[j], spectra[j], left=np.nan,
                            right=np.nan)
        Avgspec = np.nanmedian(scirect, axis=0)
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
        if i == 1:
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
            toterr = np.sqrt(specerr**2 + (syserr*1.0)**2)
            G = Gaussian2DKernel(7.)
            data = spec / toterr
            imask = np.abs(data) > 0.5
            data[imask] = np.nan
            c = convolve(data, G, boundary='extend')
            data = spec / toterr
            data[:] -= c
            imask = np.abs(data) > 0.5
            data[imask] = np.nan
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
    return np.array(np.isnan(data), dtype='float32')

parser = setup_parser()
parser.add_argument('outfilename', type=str,
                    help='''name of the output file''')
parser.add_argument("-f", "--folder", help='''Output folder''', type=str,
                    default='output')

parser.add_argument("-dd", "--dark_days", help='''Extra days +/- for darks''',
                    type=int, default=7)

parser.add_argument("-i", "--ifuslot",  help='''IFUSLOT''', type=str,
                    default='047')


args = parser.parse_args(args=None)
args.log = setup_logging(logname='build_master_bias')
args = set_daterange(args)
kinds = ['zro', 'drk', 'flt', 'cmp', 'twi', 'sci']
mkpath(args.folder)
dirname = get_script_path()

filename_dict = {}
tarname_dict = {}
tarinfo_dict = {}
daterange_darks = expand_date_range(args.daterange, args.dark_days)
daterange = list(args.daterange)
objectdict, filename_list = get_objects(daterange)
for kind in kinds:
    args.log.info('Getting file names for %s' % kind)
    filename_dict[kind] = [fn[:-14] for fn in filename_list if kind in fn]
    tarname_dict[kind] = get_tarfiles(filename_dict[kind])
    tarinfo_dict[kind] = get_tarinfo(tarname_dict[kind], filename_dict[kind])
    args.log.info('Number of tarfiles for %s: %i' %
                  (kind, len(tarname_dict[kind])))
    if kind == 'flt':
        ifuslots = get_unique_ifuslots(tarname_dict[kind])

args.log.info('Number of unique ifuslot zipcodes: %i' % len(ifuslots))

fileh = tb.open_file(op.join(args.folder, args.outfilename), 'w')
imagetable = fileh.create_table(fileh.root, 'Cals', VIRUSImage,
                                'Cal Info')

T = Table.read(op.join(dirname, 'Lines_list/virus_lines.dat'),
               format='ascii.no_header')
T_array = np.array(T['col1'])

for ifuslot_key in ifuslots:
    ifuslot, specid, ifuid, contid = ifuslot_key.split('_')
    for amp in ['LL', 'LU', 'RL', 'RU']:
        row = imagetable.row
        ifupos = get_ifucenfile(dirname, ifuid, amp)
        masterdark = np.zeros((1032, 1032))
        masterflt = np.zeros((1032, 1032))
        mastertwi = np.zeros((1032, 1032))
        mastersci = np.zeros((1032, 1032))
        masterbias = np.zeros((1032, 1032))
        wave = np.zeros((112, 1032))
        trace = np.zeros((112, 1032))
        _cmp = np.zeros((112, 1032))
        spec = np.zeros((112, 1032))
        maskspec = np.zeros((112, 1032))
        readnoise = 3.
        pixelmask = np.zeros((1032, 1032))
        for kind in kinds:
            args.log.info('Making %s master frame for %s %s' %
                          (kind, ifuslot_key, amp))
            _info = build_master_frame(filename_dict[kind], tarinfo_dict[kind],
                                       ifuslot, amp, kind,
                                       args.log, args.folder, specid, ifuid,
                                       contid)
            if _info is None:
                args.log.error("Can't complete reduction for %s %s because of"
                               " lack of %s files." % (ifuslot_key, amp, kind))
                break
            specid, ifuSlot, ifuid = ['%03d' % int(z)
                                      for z in [_info[3], ifuslot, _info[4]]]
            if kind == 'sci':
                mastersci = _info[0] * 1.
                spec = get_spectra(_info[0], trace)
            if kind == 'twi':
                mastertwi = _info[0] * 1.
            if kind == 'zro':
                masterbias = _info[0] * 1.
                readnoise = biweight(_info[1])
                args.log.info('Readnoise for %s %s: %0.2f' %
                              (ifuslot_key, amp, readnoise))
            if kind == 'drk':
                masterdark = _info[0] * 1.
                args.log.info('Getting pixel mask %03d %s' %
                              (int(ifuslot), amp))
                pixelmask = get_pixelmask(masterdark)
            if kind == 'flt':
                masterflt = _info[0] * 1.
                pixelmask2 = get_pixelmask_flt(masterflt)
                pixelmask = np.array(pixelmask2 + pixelmask > 0, dtype=int)
                args.log.info('Getting trace for %s %s' %
                              (ifuslot_key, amp))
                try:
                    trace, ref = get_trace(_info[0], specid, ifuSlot, ifuid,
                                           amp, _info[2][:8], dirname)
                    flt = get_spectra(_info[0], trace)
                except:
                    args.log.error('Trace Failed for %s %s.' %
                                       (ifuslot_key, amp))
                try:
                    cm, cl, ch = measure_contrast(_info[0], flt, trace)
                    if cm is not None:
                        args.log.info('Contrast 5th, 50th, and 95th percentiles for '
                                      '%s %s: %0.2f, %0.2f, %0.2f' %
                                      (ifuslot_key, amp, cl, cm, ch))
                except:
                    args.log.error('Contrast Failed for %s %s.' %
                                       (ifuslot_key, amp))
            if kind == 'cmp':
                args.log.info('Getting wavelength for %03d %s' %
                              (int(ifuslot), amp))
                _cmp = get_spectra(_info[0], trace)
                mastercmp = _info[0] * 1.
                try:
                    wave = get_wave(_cmp, trace, T_array)
                    if wave is None:
                        wave = np.zeros((112, 1032))
                        args.log.error('Wavelength Failed for %s %s.' %
                                       (ifuslot_key, amp))
                    args.log.info('Min/Max wavelength for %03d %s: %0.2f, %0.2f' %
                              (int(ifuslot), amp, np.min(wave), np.max(wave)))
                except:
                    args.log.error('Wavelength Failed for %s %s.' %
                                       (ifuslot_key, amp))    
        success = append_fibers_to_table(row, wave, trace, ifupos, ifuslot,
                                         ifuid, specid, amp, readnoise,
                                         pixelmask, masterdark, masterflt,
                                         mastertwi, mastercmp, mastersci,
                                         masterbias, spec, _cmp, maskspec,
                                         contid)
        if success:
            imagetable.flush()
