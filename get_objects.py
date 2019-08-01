#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 14:47:05 2019

@author: gregz
"""

import tarfile
import glob
import os.path as op

from astropy.io import fits

rootdir = '/work/03946/hetdex/maverick'

date = '20190401'

tarfolders = glob.glob(op.join(rootdir, date, 'virus', '*'))
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
            T.extractfile(name)
            a = fits.open(T.extractfile(a))
            Target = a[0].header['OBJECT']
            print('%s: %s' % (tarfolder, Target))