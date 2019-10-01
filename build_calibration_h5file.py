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

from astropy.io import fits
from astropy.stats import mad_std
from astropy.table import Table
from datetime import datetime, timedelta
from distutils.dir_util import mkpath
from fiber_utils import base_reduction, get_trace, get_spectra
from fiber_utils import get_powerlaw, get_ifucenfile, get_wave, get_pixelmask
from input_utils import setup_parser, set_daterange, setup_logging
from math_utils import biweight

karl_tarlist = '/work/00115/gebhardt/maverick/gettar/%starlist'
warnings.filterwarnings("ignore")

def get_script_path():
    '''
    Get script path, aka, where does Remedy live?
    '''
    return op.dirname(op.realpath(sys.argv[0]))

class VIRUSImage(tb.IsDescription):
    wavelength = tb.Float32Col((112, 1032))
    trace = tb.Float32Col((112, 1032))
    masterdark = tb.Float32Col((1032, 1032))
    ifupos = tb.Float32Col((112, 2))
    ifuslot = tb.Int32Col()
    ifuid = tb.StringCol(3)
    specid = tb.StringCol(3)
    contid = tb.StringCol(8)
    amp = tb.StringCol(2)
    readnoise = tb.Float32Col()
    pixelmask = tb.Int32Col((1032, 1032))
    powerlaw = tb.Float32Col((1032, 1032))



def append_fibers_to_table(row, wave, trace, ifupos, ifuslot, ifuid, specid,
                           amp, readnoise, pixelmask, masterdark, plaw, contid):
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
    row['powerlaw'] = plaw
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
                b = line.rstrip()
                c = op.join(args.rootdir, b)
                filenames.append(c[:-14])
    return filenames

def get_tarfiles(filenames):
    tarnames = []
    for fn in filenames:
        tarbase = op.dirname(op.dirname(op.dirname(fn))) + '.tar'
        if tarbase not in tarnames:
            tarnames.append(tarbase)
    return tarnames

def get_ifuslots(tarfolder):
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
    l1 = [daterange[0] - timedelta(days=x)
          for x in np.arange(1, days+1)]
    l2 = [daterange[-1] + timedelta(days=x)
          for x in np.arange(1, days+1)]
    return l1 + daterange + l2


def build_master_frame(file_list, ifuslot, amp, kind, log, folder, specid,
                       ifuid, contid):
    # Create empty lists for the left edge jump, right edge jump, and structure
    objnames = ['hg', 'cd-a']
    bia_list = []
    for itm in file_list:
        fn = itm + '%s%s_%s.fits' % (ifuslot, amp, kind)
        try:
            I, E, header = base_reduction(fn, get_header=True)
            hspecid, hifuid, hcontid = ['%03d' % int(header[name])
                                     for name in ['SPECID', 'IFUID', 'CONTID']]
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
        log.warning('No bias frames found for date range given')
        return None

    if kind != 'cmp':
        big_array = np.array([v[0] for v in bia_list])
        masterbias = np.median(big_array, axis=0)
        if kind == 'drk':
            masterstd = mad_std(big_array, axis=0)
        else:
            masterstd = np.std(big_array, axis=0)
    else:
        masterbias = np.zeros(bia_list[0][0].shape)
        masterstd = np.zeros(bia_list[0][0].shape)

        for objname in objnames:
            big_array = np.array([v[0] for v in bia_list
                                  if v[2].lower() == objname])
            masterim = np.median(big_array, axis=0)
            masterst = np.std(big_array, axis=0)
            masterbias += masterim
            masterstd += masterst / len(objnames)

    d1 = bia_list[0][3]
    d2 = bia_list[-1][3]
    d4 = (d1 + timedelta(seconds=(d2-d1).seconds / 2.) +
          timedelta(days=(d2-d1).days / 2.))
    avgdate = '%04d%02d%02dT%02d%02d%02d' % (d4.year, d4.month, d4.day,
                                             d4.hour, d4.minute, d4.second)
    return masterbias, masterstd, avgdate, bia_list[0][1], bia_list[0][4], bia_list[0][5]

parser = setup_parser()
parser.add_argument('outfilename', type=str,
                    help='''name of the output file''')
parser.add_argument("-f", "--folder", help='''Output folder''', type=str,
                    default='output')

parser.add_argument("-i", "--ifuslot",  help='''IFUSLOT''', type=str,
                    default='047')


args = parser.parse_args(args=None)
args.log = setup_logging(logname='build_master_bias')
args = set_daterange(args)
kinds = ['drk', 'twi', 'cmp']
mkpath(args.folder)
dirname = get_script_path()

filename_dict = {}
tarname_dict = {}
for kind in kinds:
    args.log.info('Getting file names for %s' % kind)
    if kind == 'drk':
        daterange = expand_date_range(args.daterange, 7)
    else:
        daterange = list(args.daterange)
    filename_dict[kind] = get_filenames(args, daterange, kind)
    tarname_dict[kind] = get_tarfiles(filename_dict[kind])
    if kind == 'twi':
        ifuslots = get_unique_ifuslots(tarname_dict[kind])

args.log.info('Number of unique ifuslots: %i' % len(ifuslots))

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
        masterdark = np.zeros((1032, 1032))
        wave = np.zeros((112, 1032))
        trace = np.zeros((112, 1032))
        plaw = np.zeros((1032, 1032))
        readnoise = 3.
        pixelmask = np.zeros((1032, 1032))
        for kind in kinds:
            args.log.info('Making %s master frame for %s %s' %
                          (kind, ifuslot_key, amp))
            _info = build_master_frame(filename_dict[kind], ifuslot, amp, kind,
                                       args.log, args.folder, specid, ifuid,
                                       contid)
            specid, ifuSlot, ifuid = ['%03d' % int(z)
                                      for z in [_info[3], ifuslot, _info[4]]]
            if kind == 'drk':
                masterdark = _info[0] * 1.
                readnoise = biweight(_info[1])
                args.log.info('Readnoise for %s %s: %0.2f' %
                              (ifuslot_key, amp, readnoise))
                args.log.info('Getting pixel mask %03d %s' %
                              (int(ifuslot), amp))
                pixelmask = get_pixelmask(masterdark)
            if kind == 'twi':
                ifupos = get_ifucenfile(dirname, ifuid, amp)
                args.log.info('Getting trace for %s %s' %
                              (ifuslot_key, amp))
                if np.mean(_info[0]) < 1000.:
                    args.log.warning('Twi for %s %s below 1000 cnts on average.' %
                                     (ifuslot_key, amp))
                    break
                try:
                    trace, ref = get_trace(_info[0], specid, ifuSlot, ifuid,
                                           amp, _info[2][:8], dirname)
                except:
                    args.log.error('Trace Failed.')
                    fits.PrimaryHDU(_info[0]).writeto('testtwi.fits', overwrite=True)
                    sys.exit(1)
                twi = get_spectra(_info[0], trace)
                args.log.info('Getting powerlaw for %03d %s' %
                              (int(ifuslot), amp))
                plaw = get_powerlaw(_info[0], trace, twi, amp)
            if kind == 'cmp':
                args.log.info('Getting wavelength for %03d %s' %
                              (int(ifuslot), amp))
                cmp = get_spectra(_info[0], trace)
                try:
                    wave = get_wave(cmp, trace, T_array)
                    if wave is None:
                        wave = np.zeros((112, 1032))
                        args.log.error('Wavelength Failed for %s %s.' %
                                       (ifuslot_key, amp))
                except:
                    args.log.error('Wavelength Failed for %s %s.' %
                                       (ifuslot_key, amp))
                
        success = append_fibers_to_table(row, wave, trace, ifupos, ifuslot,
                                         ifuid, specid, amp, readnoise,
                                         pixelmask, masterdark, plaw, _info[5])
        if success:
            imagetable.flush()
