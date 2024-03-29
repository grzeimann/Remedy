#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 14:47:05 2019

@author: gregz
"""

import glob
import os.path as op
import sys
import tarfile
import numpy as np

from astropy.io import fits

rootdir = '/work/03946/hetdex/maverick'

inst = 'virus'

date = sys.argv[1]

cal = sys.argv[2]

target = sys.argv[3]

ncalls = int(sys.argv[4])

tarfolders = sorted(glob.glob(op.join(rootdir, date, inst,
                                      '%s0000*.tar' % inst)))

call = ('python3 /work/03730/gregz/maverick/Remedy/quick_reduction.py %s %i '
        '47 %s -nd 8 -fp '
        '/work/03730/gregz/maverick/fplaneall.txt -nD')
tarlist = []
dates = [op.basename(op.dirname(op.dirname(tarf))) for tarf in tarfolders]
for date, tarfolder in zip(dates, tarfolders):
    T = tarfile.open(tarfolder, 'r')
    flag = True
    while flag:
        try:
            a = T.next()
        except:
            print('This file had an issue: %s' % tarfolder)
            flag = False
            break
        try:
            name = a.name
        except:
            flag = False
        if name[-5:] == '.fits':
            try:
                b = fits.open(T.extractfile(a))
                Target = b[0].header['OBJECT']
            except:
                print('Failed to open fits file from %s' % tarfolder)
            try:
                exptime = b[0].header['EXPTIME']
            except:
                exptime = 0.0
            try:
                prog = b[0].header['QPROG']
            except:
                prog = 'None'
            obs = int(op.basename(tarfolder)[-11:-4])
            if name[-8:-5] == 'sci':
                if target in Target:
                    tarlist.append([obs, name[-8:-5], Target, prog, exptime, date])
            flag = False
print('Number of calls: %i' % len(tarlist))
with open('%s_calls' % tarlist[0][-1], 'w') as out_file:       
    for chunk in np.array_split(tarlist, ncalls):
        calls = [call % (ch[-1], int(ch[0]), cal) for ch in chunk]
        call_str = '; '.join(calls)
        out_file.write(call_str + '\n')
