#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 17 10:26:20 2023

@author: gregz
"""

import os
import glob
import os.path as op
import numpy as np

call_str = 'python3 /work/03730/gregz/maverick/Remedy/quick_reduction.py %s %i 47 /work/03730/gregz/maverick/output/%s_%s.h5 -nd 8 -fp /work/03730/gregz/maverick/fplaneall.txt -nD'

catch_date = '202210'



files = sorted(glob.glob('/work/03946/hetdex/virus_parallels/reductions/*.h5'))

calls = []
for filename in files:
    file_size = os.path.getsize(filename)
    if file_size < 100e6:
        shortname = op.basename(filename)
        a = shortname.split('.')[0]
        date = a.split('_')[0]
        obs = int(a.split('_')[1])
        
        if date[:6] == catch_date:
            date1 = '20220901'
            date2 = '20221001'
        else:
            date1 = date[:6] + '01'
            if date[4:6] == '12':
                date2 = str(int(date[:4])+1) + '0101'
            else:
                date2 = str(int(date[:6])+1) + '01'
        call = call_str % (date, obs, date1, date2)
        calls.append(call)

ncalls = int(np.ceil(len(calls) / 12.))
print('Number of calls: %i' % ncalls)
with open('%s_calls' % 'unfinished', 'w') as out_file:       
    for chunk in np.array_split(calls, ncalls):
        call_str = '; '.join(chunk)
        out_file.write(call_str + '\n')
