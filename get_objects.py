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

from astropy.io import fits

rootdir = '/work/03946/hetdex/maverick'

date = sys.argv[1]

inst = sys.argv[2]

if inst == 'hpf':
    tarfolders = sorted(glob.glob(op.join(rootdir, date, inst, 'raw', '0*',
                                      '00*.tar')))
else:
    tarfolders = sorted(glob.glob(op.join(rootdir, date, inst,
                                          '%s0000*.tar' % inst)))
for tarfolder in tarfolders:
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
            print('%s: %s  %s  %s %0.1f' % (tarfolder, name[-8:-5], 
                                            Target, prog, exptime))
            flag = False
    T.close()