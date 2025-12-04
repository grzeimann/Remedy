#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build Remedy quick_reduction call file for a date range and program ID.

Notes:
- The calibration H5 path is now computed per observation date. For each date D (YYYYMMDD),
  we use month_start = YYYYMM01 and month_end = first day of the next month, so the path is
  /work/03946/hetdex/maverick/het_code/Remedy/output/{month_start}_{month_end}.h5.
- Example: for D=20250807, month_start=20250801 and month_end=20250901.
"""

import argparse
import glob
import os
import os.path as op
import tarfile
from datetime import datetime, timedelta
import numpy as np
import io

from astropy.io import fits

rootdir = '/work/03946/hetdex/maverick'
inst = 'virus'

CALL_TEMPLATE = (
    'python3 /work/03730/gregz/maverick/Remedy/quick_reduction.py %s %i '
    '82 %s -nd 8 -fp /work/03730/gregz/maverick/fplaneall.txt -nD'
)


def parse_args():
    p = argparse.ArgumentParser(
        description='Build quick_reduction call list for a date range and QPROG program ID.'
    )
    p.add_argument('start_date', help='Start date YYYYMMDD (inclusive)')
    p.add_argument('end_date', help='End date YYYYMMDD (exclusive)')
    p.add_argument('program_id', help='Program ID to match against FITS header QPROG')
    p.add_argument('--ncalls', type=int, default=10, help='Number of batches (lines) in output file')
    return p.parse_args()


def daterange(start: str, end: str):
    """Yield YYYYMMDD strings from start (inclusive) to end (exclusive)."""
    ds = datetime.strptime(start, '%Y%m%d')
    de = datetime.strptime(end, '%Y%m%d')
    d = ds
    while d < de:
        yield d.strftime('%Y%m%d')
        d += timedelta(days=1)


def build_cal_path(start: str, end: str):
    return f'/work/03946/hetdex/maverick/het_code/Remedy/output/{start}_{end}.h5'


def month_bounds(date_str: str):
    """Given YYYYMMDD, return (month_start, next_month_start) as YYYYMMDD strings.
    Example: 20250807 -> (20250801, 20250901)
    """
    d = datetime.strptime(date_str, '%Y%m%d')
    month_start_dt = d.replace(day=1)
    if d.month == 12:
        next_month_dt = datetime(d.year + 1, 1, 1)
    else:
        next_month_dt = datetime(d.year, d.month + 1, 1)
    return month_start_dt.strftime('%Y%m%d'), next_month_dt.strftime('%Y%m%d')


def build_cal_path_for_date(date_str: str):
    start, end = month_bounds(date_str)
    return f'/work/03946/hetdex/maverick/het_code/Remedy/output/{start}_{end}.h5'


def _read_qprog_fast(tarfile_obj: tarfile.TarFile, tarinfo: tarfile.TarInfo, max_bytes: int = 28800):
    """Attempt to read QPROG from a FITS header inside a tar member quickly.

    Strategy:
    - Read up to max_bytes (default 28,800 = 10 FITS blocks) from the member.
    - Try parsing a FITS header from this partial content using Header.fromfile on a BytesIO buffer.
    - If parsing fails (e.g., END card not within buffer), fall back to a full header parse.
    """
    qprog = 'None'
    # First attempt: limited bytes
    try:
        fobj = tarfile_obj.extractfile(tarinfo)
        if fobj is not None:
            buf = fobj.read(max_bytes)
            try:
                bio = io.BytesIO(buf)
                hdr = fits.Header.fromfile(bio, endcard=True, padding=False)
                qprog = hdr.get('QPROG', 'None')
                fobj.close()
                return qprog
            except Exception:
                # Fall through to full read fallback
                pass
            finally:
                try:
                    fobj.close()
                except Exception:
                    pass
    except Exception:
        # proceed to fallback
        pass

    # Fallback: let astropy parse from the file-like object fully
    try:
        fobj2 = tarfile_obj.extractfile(tarinfo)
        if fobj2 is not None:
            try:
                hdr = fits.Header.fromfile(fobj2, endcard=True, padding=False)
                qprog = hdr.get('QPROG', 'None')
            finally:
                try:
                    fobj2.close()
                except Exception:
                    pass
    except Exception:
        pass
    return qprog


def collect_matching_obs(start: str, end: str, program_id: str):
    tarlist = []  # list of (obs, date)
    for date in daterange(start, end):
        print(f'Processing {date}')
        pattern = op.join(rootdir, date, inst, f'{inst}0000*.tar')
        tarfolders = sorted(glob.glob(pattern))
        for tarfolder in tarfolders:
            try:
                # Use explicit uncompressed tar mode to avoid autodetect overhead
                T = tarfile.open(tarfolder, 'r:')
            except Exception as e:
                # Fall back to default if explicit mode fails (e.g., compressed tar)
                try:
                    T = tarfile.open(tarfolder, 'r')
                except Exception as e2:
                    print(f'Failed to open tar {tarfolder}: {e2}')
                    continue
            try:
                # Stream through entries and stop at the first sci FITS
                while True:
                    try:
                        a = T.next()
                        if a is None:
                            break
                        name = a.name
                    except Exception:
                        break
                    if not name.endswith('.fits'):
                        continue
                    if name[-8:-5] != 'sci':
                        continue
                    # Read only the header for speed (fast path with fallback)
                    try:
                        qprog = _read_qprog_fast(T, a)
                    except Exception:
                        print(f'Failed to read FITS header from {tarfolder}')
                        # move to next tar; keep behavior similar to original
                        break
                    if str(qprog) == str(program_id):
                        try:
                            base = op.basename(tarfolder)
                            obs = int(base[-11:-4])
                        except Exception:
                            # fallback: attempt to parse any 7-digit sequence
                            obs = None
                            base = op.basename(tarfolder)
                            for i in range(len(base) - 6):
                                chunk = base[i:i+7]
                                if chunk.isdigit():
                                    obs = int(chunk)
                                    break
                            if obs is None:
                                print(f'Could not parse OBS number from {tarfolder}')
                                break
                        tarlist.append((obs, date))
                    # only consider first sci FITS per tar (as in original)
                    break
            finally:
                try:
                    T.close()
                except Exception:
                    pass
    return tarlist


def main():
    args = parse_args()

    # Validate dates
    try:
        _ = datetime.strptime(args.start_date, '%Y%m%d')
        _ = datetime.strptime(args.end_date, '%Y%m%d')
    except ValueError:
        print('Dates must be in YYYYMMDD format')
        return 1
    if args.start_date >= args.end_date:
        print('start_date must be earlier than end_date')
        return 1

    # Ensure output directory for calibration files exists (same parent for all months)
    out_dir = op.dirname(build_cal_path_for_date(args.start_date))
    os.makedirs(out_dir, exist_ok=True)

    matches = collect_matching_obs(args.start_date, args.end_date, args.program_id)
    if len(matches) == 0:
        print('No matching observations found for given range and program ID.')
        # still write an empty calls file for consistency
        out_name = f'{args.start_date}_{args.end_date}_calls'
        open(out_name, 'w').close()
        print(f'Wrote empty call file: {out_name}')
        return 0

    # Sort and batch
    matches = sorted(matches, key=lambda x: (x[1], x[0]))  # by date then obs
    ncalls = max(1, int(args.ncalls))
    chunks = np.array_split(np.array(matches, dtype=object), ncalls)

    out_name = f'{args.start_date}_{args.end_date}_calls'
    with open(out_name, 'w') as out_file:
        for chunk in chunks:
            if len(chunk) == 0:
                continue
            calls = [
                CALL_TEMPLATE % (date, int(obs), build_cal_path_for_date(date))
                for (obs, date) in chunk
            ]
            out_file.write('; '.join(calls) + '\n')

    print(f'Number of calls: {len(matches)}')
    print(f'Wrote call file: {out_name}')
    # Inform that calibration path is date-dependent now
    unique_months = sorted({month_bounds(date)[0] for (_, date) in matches})
    print(f'Calibration H5 files are month-dependent; months covered: {", ".join(unique_months)}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
