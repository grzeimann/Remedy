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

ncalls = int(sys.argv[3])

tarfolders = sorted(glob.glob(op.join(rootdir, date, inst,
                                      '%s0000*.tar' % inst)))

call = ('python3 /work/03730/gregz/maverick/Remedy/quick_reduction.py %s %i '
        '47 %s -nd 8 -fp '
        '/work/03730/gregz/maverick/fplaneall.txt')
tarlist = []
dates = [op.basename(op.dirname(op.dirname(tarf))) for tarf in tarfolders]
for date, tarfolder in zip(dates, tarfolders):
    T = tarfile.open(tarfolder, 'r')
    flag = True
    while flag:
        a = T.next()
        try:
            name = a.name
        except:
            flag = False
        if name[-5:] == '.fits':
            b = fits.open(T.extractfile(a))
            Target = b[0].header['OBJECT']
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
                if Target == 'parallel':
                    tarlist.append([obs, name[-8:-5], Target, prog, exptime, date])
            flag = False
with open('%s_calls' % tarlist[0][-1], 'w') as out_file:       
    for chunk in np.array_split(tarlist, ncalls):
        calls = [call % (ch[-1], cal, int(ch[0])) for ch in chunk]
        call_str = '; '.join(calls)
        out_file.write(call_str + '\n')