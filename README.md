# Remedy v0.1 (Quick VIRUS Reduction Pipeline)

## Table of Contents
[Overview](https://github.com/grzeimann/Remedy/blob/master/README.md#Overview)

[Running Remedy](https://github.com/grzeimann/Remedy/blob/master/README.md#Running-Remedy)

[Data Products](https://github.com/grzeimann/Remedy/blob/master/README.md#Data-Products)

[Examples](https://github.com/grzeimann/Remedy/blob/master/README.md#Examples)

## Overview

## Running Remedy

```
python Remedy/create_cal_hdf5.py -d 20190112 -o 27 -of test_cal_20190112.h5 -r "/work/03946/hetdex/maverick/red1/reductions"
```

```
python Remedy/quick_reduction.py 20190204 17 46 test_cal_20190112.h5 --sky_ifuslot 47 --fplane_file fplane20190129.txt
```
## Data Products

## Examples
