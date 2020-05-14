#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 13 19:09:28 2020

@author: gregz
"""

import tables
import sys

filename = sys.argv[1]

t = tables.open_file(filename)

print(t.root.Survey[:])