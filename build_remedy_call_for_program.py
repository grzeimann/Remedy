#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build Remedy quick_reduction call file for a date range and program ID.

"""

import argparse
import glob
import os
import os.path as op
import tarfile
from datetime import datetime, timedelta
import numpy as np

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


def collect_matching_obs(start: str, end: str, program_id: str):
    tarlist = []  # list of (obs, date)
    for date in daterange(start, end):
        pattern = op.join(rootdir, date, inst, f'{inst}0000*.tar')
        tarfolders = sorted(glob.glob(pattern))
        for tarfolder in tarfolders:
            try:
                T = tarfile.open(tarfolder, 'r')
            except Exception as e:
                print(f'Failed to open tar {tarfolder}: {e}')
                continue
            try:
                # iterate members until we find a FITS sci file
                for a in T:
                    try:
                        name = a.name
                    except Exception:
                        continue
                    if not hasattr(a, 'name'):
                        continue
                    if not name.endswith('.fits'):
                        continue
                    if name[-8:-5] != 'sci':
                        continue
                    try:
                        with fits.open(T.extractfile(a)) as b:
                            qprog = b[0].header.get('QPROG', 'None')
                    except Exception:
                        print(f'Failed to open FITS from {tarfolder}')
                        break  # move to next tar; keep behavior similar to original
                    if str(qprog) == str(program_id):
                        try:
                            obs = int(op.basename(tarfolder)[-11:-4])
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
                    break  # only consider first sci FITS per tar (as in original)
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

    cal_path = build_cal_path(args.start_date, args.end_date)
    os.makedirs(op.dirname(cal_path), exist_ok=True)

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
            calls = [CALL_TEMPLATE % (date, int(obs), cal_path) for (obs, date) in chunk]
            out_file.write('; '.join(calls) + '\n')

    print(f'Number of calls: {len(matches)}')
    print(f'Wrote call file: {out_name}')
    print(f'Calibration H5 will be: {cal_path}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
