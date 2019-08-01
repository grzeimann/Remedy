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

tarfolders = sorted(glob.glob(op.join(rootdir, date, 'virus', '*.tar')))
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
            exptime = b[0].header['EXPTIME']
            print('%s: %s  %s  %0.1f' % (tarfolder, name[-8:-5], Target, exptime))
            flag = False
    T.close()