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
if inst == 'gc1':
    tarfolders = sorted(glob.glob(op.join(rootdir, date, inst, '%s.tar' % inst)))
if inst == 'gc2':
    tarfolders = sorted(glob.glob(op.join(rootdir, date, inst, '%s.tar' % inst)))
if (inst == 'lrs2') or (inst == 'virus'):
    tarfolders = sorted(glob.glob(op.join(rootdir, date, inst,
                                          '%s0000*.tar' % inst)))
for tarfolder in tarfolders:
    T = tarfile.open(tarfolder, 'r')
    flag = True
    while flag:
        try:
            a = T.next()
            name = a.name
        except:
            flag = False
            continue
        if 'gc' in inst:
            if name[-5:] == '.fits':
                b = fits.open(T.extractfile(a))
                Target = b[0].header['OBJECT']
                state = b[0].header['GUIDLOOP']
                try:
                    illum = b[0].header['PUPILLUM']
                    transpar = b[0].header['TRANSPAR']
                    iq = b[0].header['IQ']
                except:
                    illum = -99.0
                    transpar = -99.0
                    iq = -99.0
                print('%s: %s  %s %s %0.2f %0.2f %0.2f' % (tarfolder,
                                                           name[-8:-5], 
                                                           Target, state,
                                                           illum, transpar, iq))
        if 'hpf' in inst:
            if name[-5:] == '.fits':
                b = fits.open(T.extractfile(a))
                Target = b[0].header['OBJECT']
                try:
                    prog = b[0].header['QPROG']
                except:
                    prog = 'None'
                sciobj = b[0].header['SCI-OBJ']
                calobj = b[0].header['CAL-OBJ']
                skyobj = b[0].header['SKY-OBJ']
                print('%s: %s  %s  %s %s %s %s' % (tarfolder, name[-8:-5], 
                                                Target, prog, sciobj, calobj, skyobj))
                flag = False
        
        else:   
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
                try: 
                    dt = b[0].header['DATE']
                except: 
                    dt = '20220222T00:00:00.0'
                try:
                    ra = b[0].header['QRA']
                    dec = b[0].header['QDEC']
                except:
                    ra = '00:00:00'
                    dec = '+00:00:00'
                print('%s: %s  %s  %s %0.1f %s %s %s' % (tarfolder, name[-8:-5], 
                                                Target, prog, exptime, ra, dec, dt))
                flag = False
    T.close()