# Remedy v0.1 (Quick VIRUS Reduction Pipeline)

## Table of Contents
[Overview](https://github.com/grzeimann/Remedy/blob/master/README.md#Overview)

[Running Remedy](https://github.com/grzeimann/Remedy/blob/master/README.md#Running-Remedy)

[Download and Install](https://github.com/grzeimann/Remedy/blob/master/README.md#Download-and-Install)

[Data Products](https://github.com/grzeimann/Remedy/blob/master/README.md#Data-Products)

[Examples](https://github.com/grzeimann/Remedy/blob/master/README.md#Examples)

## Overview
Remedy is a quick reduction tool for VIRUS which reduces a single ifuslot at a time.  However, there is a manual option to use a second ifuslot as a sky frame when the target ifu does not have enough sky fibers itself.  The reduction should run quickly (~30s-1min) and produce a fits image of the collapse frame (for a specific wavelength range) and a datacube of the target ifu.  The datacube can be viewed with QFitsView (http://www.mpe.mpg.de/~ott/QFitsView/), and an inspection of the collapsed frame with DS9 can help orient the user (as astrometry information is included) to the pointing and potential objects of interest.  This is a beta version, but the aim is to have a multi-purpose tool for the HET to do "real-time" reductions of target VIRUS ifus.

## Running Remedy
### Download and Install
Remedy is a library code that can be acquired simply with:
```
cd WHEREVER
git clone https://github.com/grzeimann/Remedy.git
```

Remedy relies on a single calibration file (HDF5 file format: https://www.hdfgroup.org/solutions/hdf5/).  A default set of calibrations will be kept up to date by the creator, Greg Zeimann, and can be acquired here:
```
cd WHEREVER/Remedy/CALS
scp username@wrangler.tacc.utexas.edu:/work/03730/gregz/maverick/test_cal_20190112.h5 .
```

If you want to create your own calibration file: 
```
1) Log onto TACC
2) Get Remedy on TACC if you haven't already 
3) Add the following to your ~/.bashrc file if you haven't already
        export PATH=/home/00115/gebhardt/anaconda2/bin:/work/03946/hetdex/maverick/bin:$PATH
4) To create a new calibration file, run:
        python Remedy/create_cal_hdf5.py -d 20190112 -o 27 -of test_cal_20190112.h5 -r "/work/03946/hetdex/maverick/red1/reductions"

        In this case I choose the night 20190112, observation 27, and called it test_cal_20190112.h5
        The night and observation were ones of quality reduction, most importantly with respect to wavelength and trace, which is all that is used
5) Copy calibration file to your desired Remedy/CALS folder
        cd WHEREVER/Remedy/CALS
        scp username@wrangler.tacc.utexas.edu:/PATHONTACC/test_cal_20190112.h5 .
```

### Running the code

```
usage: quick_reduction.py [-h] [-r ROOTDIR] [-ra RA] [-dec DEC]
                          [-fp FPLANE_FILE] [-si SKY_IFUSLOT]
                          date observation ifuslot hdf5file

positional arguments:
  date                  Date for reduction
  observation           Observation ID
  ifuslot               ifuslot to reduced
  hdf5file              HDF5 calibration file

optional arguments:
  -h, --help            show this help message and exit
  -r ROOTDIR, --rootdir ROOTDIR
                        Directory for raw data. (just before date folders)
  -ra RA, --ra RA       RA of the IFUSLOT to be reduced
  -dec DEC, --dec DEC   Dec of the IFUSLOT to be reduced
  -fp FPLANE_FILE, --fplane_file FPLANE_FILE
                        fplane file
  -si SKY_IFUSLOT, --sky_ifuslot SKY_IFUSLOT
                        If sky_ifuslot is not provided, then the ifuslot
                        itself is used
```

```
python Remedy/quick_reduction.py 20190204 17 46 test_cal_20190112.h5 --sky_ifuslot 47 --fplane_file fplane20190129.txt
```
## Data Products

## Examples
