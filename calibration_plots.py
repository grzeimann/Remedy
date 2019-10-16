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
    print('Working on %i' % i+1)
    mdark = t.root.Cals.cols.masterdark[i]
    bl, bm = biweight(mdark, calc_std=True)
    B[i, 0] = bl
    B[i, 1] = bm
# Plot style
sns.set_context('talk')
sns.set_style('ticks')
plt.figure(figsize=(10, 6))
sns.jointplot(x=B[:, 0], y=B[:, 1], xlim=[-0.4, 1.6], ylim=[2.4, 4.6],
              kind="hex", color="k").set_axis_labels("Average Structure Value", "Readnoise")
MLx = MultipleLocator(0.5)
mLx = MultipleLocator(0.1)
MLy = MultipleLocator(1.0)
mLy = MultipleLocator(0.2)
plt.gca().xaxis.set_major_locator(MLx)
plt.gca().yaxis.set_major_locator(MLy)
plt.gca().xaxis.set_minor_locator(mLx)
plt.gca().yaxis.set_minor_locator(mLy)
plt.gca().tick_params(axis='x', which='minor', direction='in', bottom=True)
plt.gca().tick_params(axis='x', which='major', direction='in', bottom=True)
plt.gca().tick_params(axis='y', which='minor', direction='in', left=True)
plt.gca().tick_params(axis='y', which='major', direction='in', left=True)
plt.savefig('dark_hist_%s.png' % op.basename(filename[:-4]), dpi = 300)
t.close()