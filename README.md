# Remedy v0.1 (Quick VIRUS Reduction Pipeline)

## Table of Contents
[Overview](https://github.com/grzeimann/Remedy/blob/master/README.md#Overview)

[Running Remedy](https://github.com/grzeimann/Remedy/blob/master/README.md#Running-Remedy)

[Data Products](https://github.com/grzeimann/Remedy/blob/master/README.md#Data-Products)

[Examples](https://github.com/grzeimann/Remedy/blob/master/README.md#Examples)

## Overview
Remedy is a quick reduction tool for VIRUS which reduces a single ifuslot at a time.  However, there is a manual option to use a
second ifuslot as a sky frame when the target ifu does not have enough sky fibers itself.  The reduction should run quickly (~30s-1min) and produce
## Running Remedy

```
python Remedy/create_cal_hdf5.py -d 20190112 -o 27 -of test_cal_20190112.h5 -r "/work/03946/hetdex/maverick/red1/reductions"
```

```
python Remedy/quick_reduction.py 20190204 17 46 test_cal_20190112.h5 --sky_ifuslot 47 --fplane_file fplane20190129.txt
```
## Data Products

## Examples
