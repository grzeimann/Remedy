# Remedy v0.1 (Quick VIRUS Reduction Pipeline)

## Table of Contents
[Overview](https://github.com/grzeimann/Remedy/blob/master/README.md#Overview)

[Running Remedy](https://github.com/grzeimann/Remedy/blob/master/README.md#Running-Remedy)

[Data Products](https://github.com/grzeimann/Remedy/blob/master/README.md#Data-Products)

[Examples](https://github.com/grzeimann/Remedy/blob/master/README.md#Examples)

## Overview
Remedy is a quick reduction tool for VIRUS which reduces a single ifuslot at a time.  However, there is a manual option to use a second ifuslot as a sky frame when the target ifu does not have enough sky fibers itself.  The reduction should run quickly (~30s-1min) and produce a fits image of the collapse frame (for a specific wavelength range) and a datacube of the target ifu.  The datacube can be viewed with QFitsView (http://www.mpe.mpg.de/~ott/QFitsView/), and an inspection of the collapsed frame with DS9 can help orient the user (as astrometry information is included) to the pointing and potential objects of interest.  This is a beta version, but the aim is to have a multi-purpose tool for the HET to do "real-time" reductions of target VIRUS ifus.

## Running Remedy
### Download and installation
Remedy is a library code that can be acquired simply with:
```
cd WHEREVER
git clone https://github.com/grzeimann/Remedy.git
```

Remedy relies on a single calibration file (HDF5 file format: https://www.hdfgroup.org/solutions/hdf5/).  An existing calibration file can be acquired with:
```
cd WHEREVER/Remedy
scp username@wrangler.tacc.utexas.edu:/work/03730/gregz/maverick/test_cal_20190112.h5 .
```
To create your own calibration file, you must first login to TACC (https://www.tacc.utexas.edu/).  After logging in, acquire
```
python Remedy/create_cal_hdf5.py -d 20190112 -o 27 -of test_cal_20190112.h5 -r "/work/03946/hetdex/maverick/red1/reductions"
```

```
python Remedy/quick_reduction.py 20190204 17 46 test_cal_20190112.h5 --sky_ifuslot 47 --fplane_file fplane20190129.txt
```
## Data Products

## Examples
