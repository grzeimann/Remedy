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

Remedy relies on a calibration file (HDF5 file format: https://www.hdfgroup.org/solutions/hdf5/) and an fplane file.  A default calibration file will be kept up to date by the creator, Greg Zeimann, and can be acquired here:
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

If you need an up-to-date fplane file, simply run Jan Snigula's get_fplane.py, which is included with Remedy:
```
python WHEREVER/Remedy/get_fplane.py
OR
python WHEREVER/Remedy/get_fplane.py 20190412
```

A new fplane file will be created in the current directory in the format, fplane{DATE}.txt. 

### Running the code
An example call to reduce ifuslot 043 on 20190204 and obseravtion 17 using the ifuslot 047 for sky is shown below:
```
python Remedy/quick_reduction.py 20190204 17 43 Remedy/CALS/test_cal_20190112.h5 --fplane_file fplane20190129.txt --rootdir /work/03946/hetdex/maverick --neighbor_dist 7
```
To run at the mountain, replace the "--rootdir" as appropriate, the "--fplane_file" as approriate, and the calibration file if needed.  The other arguments will be specific to the choice of ifuslot and if you want to use another ifu for sky.

More details with respect to the input arguments are below.
```
usage: quick_reduction.py [-h] [-r ROOTDIR] [-ra RA] [-dec DEC]
                          [-fp FPLANE_FILE] [-nd NEIGHBOR_DIST]
                          [-sf SOURCE_FILE] [-s] [-sx SOURCE_X] [-sy SOURCE_Y]
                          [-ss SOURCE_SEEING]
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
  -nd NEIGHBOR_DIST, --neighbor_dist NEIGHBOR_DIST
                        Distance in x or y in the focal plane IFUSLOT units to
                        include in the reduction.
  -sf SOURCE_FILE, --source_file SOURCE_FILE
                        file for spectrum to add in cube
  -s, --simulate        Simulate source
  -sx SOURCE_X, --source_x SOURCE_X
                        x-position for spectrum to add in cube
  -sy SOURCE_Y, --source_y SOURCE_Y
                        y-position for spectrum to add in cube
  -ss SOURCE_SEEING, --source_seeing SOURCE_SEEING
                        seeing conditions manually entered
```
## Data Products

To be filled later, but there are two products ... {DATE}_{OBS}_{IFUSLOT}.fits and {DATE}_{OBS}_{IFUSLOT}_cube.fits

## Examples

Running on TACC with following command:
```
python Remedy/quick_reduction.py 20190213 21 43 Remedy/CALS/test_cal_20190112.h5 --sky_ifuslot 47 --fplane_file fplane20190129.txt --rootdir /work/03946/hetdex/maverick
```
You can see the reduction of an IFU with a g'~12 star in it.  At this brightness, sky subtraction is inherently difficult, but using ifuslot 047 for sky information, we can see both a good job in sky subtraction and the expected diffraction spikes from such a bright star.  Below I am showing the output 20190213_0000021_043.fits image in ds9 and the 20190213_0000021_043_cube.fits cube in QFitsView.


<p align="center">
  <img src="images/ds9_example.png" width="850"/>
</p>

<p align="center">
  <img src="images/qfitsview_ginga_example.png" width="850"/>
</p>
