# -*- coding: utf-8 -*-
"""
Created on Mon Jun 11 11:32:27 2018

@author: gregz
"""

import os.path as op
import numpy as np
import sys
import tarfile
import warnings
import tables as tb
import glob
import time

from astropy.convolution import convolve, Gaussian2DKernel
from astropy.io import fits
from astropy.table import Table
from datetime import datetime, timedelta
from pathlib import Path
from fiber_utils import base_reduction, get_trace, get_spectra, get_spectra_error
from fiber_utils import get_powerlaw, get_ifucenfile, get_pixelmask
from wave_utils import get_wave
from mask_utils import build_model_spectra, make_spectral_mask
from fiber_utils import get_pixelmask_flt
from fiber_utils import measure_contrast, get_bigW, get_bigF
from input_utils import setup_parser, set_daterange, setup_logging
from math_utils import biweight
from scipy.interpolate import interp1d
from qa_utils import summarize_amp_metrics, plot_ref_profile_quarters, save_amp_qa_page


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

def _get_objects_tarfile(daterange, instrument='virus', rootdir='/work/03946/hetdex/maverick'):
    start_total = time.time()
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
    args.log.info(f"[get_objects:tarfile] Start scan: {len(dates)} dates, IFUSLOT={ifuslot}, {len(tarfolders)} tar files")
    total_fits = 0
    for tarfolder in tarfolders:
        t0 = time.time()
        date = tarfolder.split('/')[-3]
        dtstr = date
        obsnum = int(tarfolder[-11:-4])
        NEXP = 1
        try:
            T = tarfile.open(tarfolder, 'r')
            names_list = T.getnames()
            names_list = [name for name in names_list if name.endswith('.fits')]
            total_fits += len(names_list)
            for name in names_list:
                if ifuslot_str in name:
                    c = op.join(args.rootdir, date, instrument, name)
                    filename_list.append(c)
            exposures = np.unique([name.split('/')[1] for name in names_list])
            # Read first header for OBJECT
            Target = ''
            if names_list:
                try:
                    b = fits.open(T.extractfile(T.getmember(names_list[0])))
                    Target = b[0].header.get('OBJECT', '')
                except Exception:
                    Target = ''
            NEXP = len(exposures)
            for i in np.arange(NEXP):
                objectdict['%s_%07d_%02d' % (date, obsnum, i+1)] = Target
            dt = time.time() - t0
            args.log.info(f"[{dtstr}] [get_objects:tarfile] {op.basename(tarfolder)}: {len(names_list)} FITS, {NEXP} exp(s), {dt:.2f}s")
        except Exception as e:
            args.log.warning(f"[{dtstr}] [get_objects:tarfile] Failed {op.basename(tarfolder)}: {e}")
            objectdict['%s_%07d_%02d' % (date, obsnum, NEXP)] = ''
            continue
    elapsed = time.time() - start_total
    args.log.info(f"[get_objects:tarfile] Done. Files: {len(filename_list)} (total FITS seen {total_fits}), t={elapsed:.2f}s")
    return objectdict, filename_list


def get_objects(daterange, instrument='virus', rootdir='/work/03946/hetdex/maverick'):
    """Wrapper to choose between tarfile and ratarmountcore implementations."""
    return _get_objects_tarfile(daterange, instrument=instrument, rootdir=rootdir)

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

# --- Calibration recipe helper functions (scoped to this amp) ---
def step_zro():
    kind = 'zro'
    args.log.info('Making %s master frame for %s %s' % (kind, ifuslot_key, amp))
    _info = build_master_frame(filename_dict[kind], tarinfo_dict[kind],
                               ifuslot, amp, kind, args.log, args.folder,
                               specid, ifuid, contid)
    if _info is None:
        args.log.error("Can't complete reduction for %s %s because of lack of %s files." % (ifuslot_key, amp, kind))
        return None
    _masterbias = _info[0] * 1.
    _readnoise = biweight(_info[1])
    args.log.info('Readnoise for %s %s: %0.2f' % (ifuslot_key, amp, _readnoise))
    return _masterbias, _readnoise, _info

def step_drk():
    kind = 'drk'
    args.log.info('Making %s master frame for %s %s' % (kind, ifuslot_key, amp))
    _info = build_master_frame(filename_dict[kind], tarinfo_dict[kind],
                               ifuslot, amp, kind, args.log, args.folder,
                               specid, ifuid, contid)
    if _info is None:
        args.log.error("Can't complete reduction for %s %s because of lack of %s files." % (ifuslot_key, amp, kind))
        return None
    _masterdark = _info[0] * 1.
    args.log.info('Getting pixel mask %03d %s' % (int(ifuslot), amp))
    _pixelmask = get_pixelmask(_masterdark)
    return _masterdark, _pixelmask, _info

def step_flt_trace(curr_pixelmask):
    kind = 'flt'
    args.log.info('Making %s master frame for %s %s' % (kind, ifuslot_key, amp))
    _info = build_master_frame(filename_dict[kind], tarinfo_dict[kind],
                               ifuslot, amp, kind, args.log, args.folder,
                               specid, ifuid, contid)
    if _info is None:
        args.log.error("Can't complete reduction for %s %s because of lack of %s files." % (ifuslot_key, amp, kind))
        return None
    _masterflt = _info[0] * 1.
    pixelmask2 = get_pixelmask_flt(_masterflt)
    _pixelmask = np.array(pixelmask2 + curr_pixelmask > 0, dtype=int)
    args.log.info('Getting trace for %s %s' % (ifuslot_key, amp))
    _trace = np.zeros((112, 1032))
    try:
        _specid, _ifuSlot, _ifuid = ['%03d' % int(z) for z in [_info[3], ifuslot, _info[4]]]
        _trace, _ref = get_trace(_info[0], _specid, _ifuSlot, _ifuid, amp, _info[2][:8], dirname)
        _flt_spec = get_spectra(_info[0], _trace)
        cm, cl, ch = measure_contrast(_info[0], _flt_spec, _trace)
        if cm is not None:
            args.log.info(
                'Contrast 5th, 50th, and 95th percentiles for %s %s: %0.2f, %0.2f, %0.2f' % (ifuslot_key, amp, cl,
                                                                                             cm, ch))
    except Exception:
        args.log.error('Trace Failed for %s %s.' % (ifuslot_key, amp))
    return _masterflt, _pixelmask, _trace, _info

def step_cmp_wave(curr_trace):
    kind = 'cmp'
    args.log.info('Making %s master frame for %s %s' % (kind, ifuslot_key, amp))
    _info = build_master_frame(filename_dict[kind], tarinfo_dict[kind],
                               ifuslot, amp, kind, args.log, args.folder,
                               specid, ifuid, contid)
    if _info is None:
        args.log.error("Can't complete reduction for %s %s because of lack of %s files." % (ifuslot_key, amp, kind))
        return None
    _mastercmp = _info[0] * 1.
    _cmp_spec = get_spectra(_info[0], curr_trace)
    _wave = np.zeros((112, 1032))
    _wave_valid = False
    ref_img = None
    try:
        qa_dict = None
        if args.make_qa:
            amp_id = f"{ifuslot}_{amp}"
            qa_dict = {
                "out_folder": Path(args.qa_folder),
                "ref_plot_name": f"ref_profile_quarters_{amp_id}.png",
            }
        _wave, ref_img = get_wave(_cmp_spec, curr_trace, T_array, qa=qa_dict)
        if _wave is None or not np.isfinite(_wave).any():
            _wave = np.zeros((112, 1032))
            args.log.error('Wavelength Failed for %s %s.' % (ifuslot_key, amp))
        wmin = float(np.nanmin(_wave)) if np.isfinite(_wave).any() else 0.0
        wmax = float(np.nanmax(_wave)) if np.isfinite(_wave).any() else 0.0
        _wave_valid = (np.isfinite(_wave).sum() > 0.5 * _wave.size) and (wmax > wmin)
        args.log.info('Min/Max wavelength for %03d %s: %0.2f, %0.2f' % (int(ifuslot), amp, wmin, wmax))
    except Exception:
        args.log.error('Wavelength Failed for %s %s.' % (ifuslot_key, amp))
        _wave_valid = False
    return _mastercmp, _cmp_spec, _wave, _wave_valid, ref_img, _info

def step_twi(curr_trace):
    kind = 'twi'
    args.log.info('Making %s master frame for %s %s' % (kind, ifuslot_key, amp))
    _info = build_master_frame(filename_dict[kind], tarinfo_dict[kind],
                               ifuslot, amp, kind, args.log, args.folder,
                               specid, ifuid, contid)
    if _info is None:
        args.log.error("Can't complete reduction for %s %s because of lack of %s files." % (ifuslot_key, amp, kind))
        return None
    _mastertwi = _info[0] * 1.
    try:
        _twi_spectra = get_spectra(_mastertwi, curr_trace)
    except Exception:
        _twi_spectra = None
    return _mastertwi, _twi_spectra, _info

def step_sci_and_mask(curr_trace, curr_wave, wave_ok):
    kind = 'sci'
    args.log.info('Making %s master frame for %s %s' % (kind, ifuslot_key, amp))
    _info = build_master_frame(filename_dict[kind], tarinfo_dict[kind],
                               ifuslot, amp, kind, args.log, args.folder,
                               specid, ifuid, contid)
    if _info is None:
        args.log.error("Can't complete reduction for %s %s because of lack of %s files." % (ifuslot_key, amp, kind))
        return None
    _mastersci = _info[0] * 1.
    _spec = get_spectra(_info[0], curr_trace)
    _maskspec = np.zeros_like(_spec, dtype='float32')
    if not wave_ok:
        args.log.info('Skipping SCI model/mask for %s %s due to invalid wavelength solution.' % (ifuslot_key, amp))
        return _mastersci, _spec, _maskspec, _info
    try:
        base_spectra = _spec
        good_solutions = np.isfinite(curr_wave).sum(axis=1) > (0.8 * curr_wave.shape[1])
        sci_interp, sci_model, sci_image, _ = build_model_spectra(base_spectra, curr_wave, good_solutions)
        msci_model_spec = sci_interp(curr_wave)
        _maskspec = make_spectral_mask(_spec, msci_model_spec).astype('float32')
    except Exception as e:
        args.log.warning('SCI model/mask generation failed for %s %s: %s' % (ifuslot_key, amp, e))
    return _mastersci, _spec, _maskspec, _info

parser = setup_parser()
parser.add_argument('outfilename', type=str,
                    help='''name of the output file''')
parser.add_argument("-f", "--folder", help='''Output folder''', type=str,
                    default='output')

parser.add_argument("-dd", "--dark_days", help='''Extra days +/- for darks''',
                    type=int, default=7)

parser.add_argument("-i", "--ifuslot",  help='''IFUSLOT''', type=str,
                    default='047')

# QA options
parser.add_argument("--make-qa", action='store_true', default=False,
                    help='Generate per-amplifier QA summary pages')
parser.add_argument("--qa-folder", type=str, default='outputs/qa',
                    help='Folder to write QA summary pages into')

args = parser.parse_args(args=None)
args.log = setup_logging(logname='build_master_bias')
args = set_daterange(args)
kinds = ['zro', 'drk', 'flt', 'cmp', 'twi', 'sci']
Path(args.folder).mkdir(parents=True, exist_ok=True)
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

        
        # --- Execute the fixed calibration recipe ---
        row = imagetable.row
        ifupos = get_ifucenfile(dirname, ifuid, amp)
        masterdark = np.zeros((1032, 1032))
        masterflt = np.zeros((1032, 1032))
        mastertwi = np.zeros((1032, 1032))
        mastersci = np.zeros((1032, 1032))
        masterbias = np.zeros((1032, 1032))
        mastercmp = np.zeros((1032, 1032))
        wave = np.zeros((112, 1032))
        trace = np.zeros((112, 1032))
        _cmp = np.zeros((112, 1032))
        spec = np.zeros((112, 1032))
        maskspec = np.zeros((112, 1032))
        readnoise = 3.
        pixelmask = np.zeros((1032, 1032))
        wave_valid = False
        twi_spectra = None
        ref_img = None
        
        # zro
        out = step_zro()
        if out is not None:
            masterbias, readnoise, info_zro = out
            specid, ifuSlot, ifuid = ['%03d' % int(z) for z in [info_zro[3], ifuslot, info_zro[4]]]
        
        # drk
        out = step_drk()
        if out is not None:
            masterdark, pixelmask, info_drk = out
        
        # flt + trace
        out = step_flt_trace(pixelmask)
        if out is not None:
            masterflt, pixelmask, trace, info_flt = out
        
        # cmp + wave
        if np.isfinite(trace).any():
            out = step_cmp_wave(trace)
            if out is not None:
                mastercmp, _cmp, wave, wave_valid, ref_img, info_cmp = out
        else:
            args.log.error('Trace not available for %s %s. Skipping wavelength.' % (ifuslot_key, amp))
        
        # twi
        if np.isfinite(trace).any():
            out = step_twi(trace)
            if out is not None:
                mastertwi, twi_spectra, info_twi = out
        
        # sci + mask
        if np.isfinite(trace).any():
            out = step_sci_and_mask(trace, wave, wave_valid)
            if out is not None:
                mastersci, spec, maskspec, info_sci = out
        
        success = append_fibers_to_table(row, wave, trace, ifupos, ifuslot,
                                         ifuid, specid, amp, readnoise,
                                         pixelmask, masterdark, masterflt,
                                         mastertwi, mastercmp, mastersci,
                                         masterbias, spec, _cmp, maskspec,
                                         contid)

        if success:
            imagetable.flush()
            # Optional QA generation per amplifier
            if args.make_qa:
                try:
                    t0qa = time.time()
                    amp_id = f"{ifuslot}_{amp}"
                    # Build metrics
                    metrics = summarize_amp_metrics(
                        amp_id=amp_id,
                        masterbias=masterbias,
                        masterdark=masterdark,
                        masterflt=masterflt,
                        mastercmp=mastercmp,
                        mastersci=mastersci,
                        readnoise=readnoise,
                        wave=wave,
                        trace=trace,
                        lampspec=_cmp,
                        maskspec=maskspec,
                        extra={
                            "specid": specid,
                            "ifuid": ifuid,
                            "contid": contid,
                        },
                    )
                    qa_out_dir = Path(args.qa_folder)
                    qa_out_dir.mkdir(parents=True, exist_ok=True)
                    args.log.info(f"[QA] Starting QA for {amp_id} → {qa_out_dir}")
                    # Save QA summary page (PNG + JSON sidecar) using ref_img from get_wave
                    try:
                        png_path = save_amp_qa_page(qa_out_dir, amp_id, metrics, ref_img)
                        args.log.info(f"[QA] Wrote QA summary page: {png_path}")
                    except Exception as e_page:
                        args.log.warning(f"[QA] Failed to write QA page for {amp_id}: {e_page}")
                    args.log.info(f"[QA] Done QA for {amp_id} in {time.time()-t0qa:.2f}s")
                except Exception as e:
                    args.log.warning(f"[QA] Failed to generate QA for {ifuslot_key} {amp}: {e}")
