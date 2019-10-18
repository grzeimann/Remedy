#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 09:40:02 2019

@author: gregz
"""

import matplotlib
matplotlib.use("agg")
import sys
import os.path as op
import numpy as np
from tables import open_file
from math_utils import biweight

import matplotlib.pyplot as plt

from matplotlib.ticker import MultipleLocator
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from sklearn.gaussian_process.kernels import ConstantKernel

import seaborn as sns


filename = sys.argv[1]
t = open_file(filename, mode='r')

L = len(t.root.Cals)

B = np.zeros((L, 2))
for i in np.arange(L):
    print('Working on %i' % (i+1))
    mdark = t.root.Cals.cols.masterdark[i]
    bl = biweight(mdark)
    bm = np.median(np.abs(np.diff(mdark, axis=1)))
    B[i, 0] = bl
    B[i, 1] = bm
# Plot style
sns.set_context('talk')
sns.set_style('whitegrid')
plt.figure(figsize=(7, 6))
jp = plt.scatter(B[:, 0], B[:, 1], color="firebrick", alpha=0.8, s=8)
plt.xlabel("Average Global Structure (e-)")
plt.ylabel("Average Local Structure (e-)")
plt.axis([-0.5, 3.5, -0.5, 3.5])
ax = plt.gca()
MLx = MultipleLocator(0.5)
mLx = MultipleLocator(0.1)
MLy = MultipleLocator(0.3)
mLy = MultipleLocator(0.1)
ax.xaxis.set_major_locator(MLx)
ax.yaxis.set_major_locator(MLy)
ax.xaxis.set_minor_locator(mLx)
ax.yaxis.set_minor_locator(mLy)
ax.tick_params(axis='x', which='minor', direction='in', bottom=True)
ax.tick_params(axis='x', which='major', direction='in', bottom=True)
ax.tick_params(axis='y', which='minor', direction='in', left=True)
ax.tick_params(axis='y', which='major', direction='in', left=True)
plt.savefig('dark_hist_%s.png' % op.basename(filename[:-3]), dpi = 300)
t.close()