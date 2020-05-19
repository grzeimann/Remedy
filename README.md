# Remedy (VIRUS Reduction Pipeline)

## Table of Contents
[Overview](https://github.com/grzeimann/Remedy/blob/master/README.md#Overview)

[Running Remedy](https://github.com/grzeimann/Remedy/blob/master/README.md#Running-Remedy)

[Download and Install](https://github.com/grzeimann/Remedy/blob/master/README.md#Download-and-Install)

[Data Products](https://github.com/grzeimann/Remedy/blob/master/README.md#Data-Products)

[Examples](https://github.com/grzeimann/Remedy/blob/master/README.md#Examples)

## Overview
Remedy is a reduction tool for VIRUS which reduces a target IFU or all IFUs in a given observation.  The pipeline should run effeciently (~30-40 min) and produce an H5 file of flux-calibrated spectra for each fiber, calibration data for each fiber, extracted bright continuum sources in the field (Pan-STARRS sources), a catalog of emission-line detected sources, and a datacube of the target ifu.  The datacube can be viewed with QFitsView (http://www.mpe.mpg.de/~ott/QFitsView/). 

## Running Remedy
### Download and Install
Remedy is a code library that can be acquired simply with:
```
cd WHEREVER
git clone https://github.com/grzeimann/Remedy.git
```

Remedy relies on a calibration file (HDF5 file format: https://www.hdfgroup.org/solutions/hdf5/) and an fplane file.  A default calibration file will be kept up to date by the creator, Greg Zeimann, and can be acquired here:
```
cd WHEREVER/Remedy/CALS
scp username@wrangler.tacc.utexas.edu:/work/03730/gregz/maverick/output/general_calibration.h5 .
```

If you want to create your own calibration file: 
```
1) Log onto Wrangler on TACC
2) Get Remedy on TACC if you haven't already 
3) Add the following to your ~/.bashrc file if you haven't already
        export PATH=/home/00115/gebhardt/anaconda2/bin:/work/03946/hetdex/maverick/bin:$PATH
4) To create a new calibration file, run:
        python Remedy/build_calibration_h5file.py 20190822_20190828.h5 -r /work/03946/hetdex/maverick -i 047 -sd 20190822 -dl 7

        In this case I choose the night 20190822, and called it 20190822_20190828.h5
        
5) Copy calibration file to your desired Remedy/CALS folder
        cd WHEREVER/Remedy/CALS
        scp username@wrangler.tacc.utexas.edu:/PATH_ON_TACC/test_cal_20190112.h5 .
```

To get an fplane file with all ifuslots, simply copy:
```
cd WHEREVER/Remedy/CALS
scp username@wrangler.tacc.utexas.edu:/work/03730/gregz/maverick/fplaneall.txt .
```

### Setting up your Python environment
To begin on TACC, point to the common python environment. In your home "~/.bashrc" file, add the following line at the bottom:
```
export PATH=/home/00115/gebhardt/anaconda2/bin:/work/03946/hetdex/maverick/bin:$PATH
```

### Running the code
An example call to reduce ifuslot 043 on 20190204 and obseravtion 17 using the ifuslot 047 for sky is shown below:
```
python Remedy/quick_reduction.py 20190822 11 47 Remedy/CALS/20190822_20190828.h5 -nd 8 -fp Remedy/CALS/fplaneall.txt
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

## About Remedy
The primary goals of the VIRUS data processing pipeline, Remedy, are to produce flux-calibrated fiber spectra, data cubes, and extracted continuum and emissin line sources. Remedy also tracks bad fiber spectra either due to regions of the CCD  with low QE, issues within the IFU spectrograph amplifiers, or ``bleeding`` effects of bright continuum sources. 

We begin by describing the basic CCD processing tasks and then discuss the more advanced steps.

### Basic Reduction Steps
VIRUS, at full capacity, is composed of 78 individual IFUs and each IFU is connected to a single spectrograph with two CCDs and four amplifiers.  Each of the amplifiers, 312 in all, has its own bias structure and dark current.  

#### Bias Subtraction
The bias level of an amplifier can be decomposed into a scalar offset and a vector containing structure over the 1032x1032 pixel frame. The scalar offset changes with exposure especially in the sequence of exposures. We use the overscan region in frame of interest to obtain the scalar offset. We measure the scalar offset with a biweight estimator, excluding the first two columns of the overscan region due to potential ''bleeding'' from the data section.  Typical scalar offsets are in the range of 1000.

There are two kinds of structure we've identified in a given VIRUS amplifier. The first is a weakly evolving large scale structure that is typically much less than the amplifier read noise, and is removed easily from a master frame built over the night or even month.  The second structure is on the pixel to pixel scale and is more akin to a  interference pattern.  This structure changes rapidly throughout the night and is not easily removed.  It is however only ~1 electron in amplitude and even less when average over a fiber in the trace direction.  As the only modelable structure is weakly evolving with time, we opt not to make a master bias frame on a night by night basis or even month by month basis and rather allow the master dark frame to capture the weakly evolving large scale structure and not make a master bias frame at all.

#### Gain Multiplication
The initial units of the amplifier images are ADUs.  We convert to electrons using the gain from the header.  The gain was calculated in the laboratory prior to delivery at the mountain.

#### Dark Subtraction
The dark current in VIRUS amplifiers have <0.5 electrons in 360s exposures. In a given night, we take between 1 and 10 dark frames.  We thus build a master dark frame, which includes the dark current, large scale bias structure, and some small scale pixel to pixel  interference structure.  The master dark frame time window is a user input, and typically a period of 21 days.

#### Pixel Mask
From the master dark frame we look for hot pixels and low level charge traps to be masked out later in processing.  We first subtract the median value in each column from the master dark to take out the large scale columnar pattern. We then subtract the median value in each row to similarly remove low level readout pattern.  Finally, we apply a sigma clipping algorithm to identify 5-sigma outliers and mask all of these pixels.  Outliers are mostly hot pixels and the base of low level charge traps.

#### Fiber Trace
The trace of the fibers or distortion map can be measured from high count observations like flat-field lamps or twilights.  For an individual fiber, we use the peak pixel and use the two neighboring pixels to define a quadratic function.  The peak of the quadratic function is then used as the  trace of the fiber for a given column.  To smooth the individual measurements across columns, we use a third order polynomial fit.

We measure the trace from a master twilight taken over an input time range, typically 7 days.  The trace is quite stable.  The night to night drift, driven primarily by the ambient temperature, is <~0.2 pixels for an individual fiber and elastic in nature as it shifts around a stable location. By averaging many nights, we essentially are measuring the trace for the  stable location and then for each observation we find a single shift to the new location.

#### Fiber Wavelength
The wavelength solution for each fiber is obtained from Hg and Cd arc lamps.  We create a master arc over an input time range, typically 7 days, and use an peak finding algorithm with a priori knowledge  of the wavelength of the bright lines to solve for the wavelength for each fiber. This creates a  stable wavelength solution, but like the fiber trace, typical shifts of <~0.2 pixels occur due to changes in the ambient temperature.  We currently do not adjust for these wavelength shifts.

#### Scattered Light - Spectrograph
Due to imperfections in the mirrors within each spectrograph, light from the fiber cables is scattered on the CCD creating a background that needs to be accounted for and subtracted.  The scattered light can be modeled as a powerlaw in which a monochromatic intensity for a given fiber has a profile that falls off proportional to 1 / pixel distance.   We find that when we add up light in the central core of fibers and compare that to the total light in the CCD,  roughly 3-4% of the incident fiber light is scattered into this smoother background.  We model this background light in the master twilight frames by first creating a power-law model for the scattered light and then normalize that model to the observed master frame.  We then model the scattered light in the science frame by scaling the background model as a function of wavelength to account for differences between the smoothed twilight incident spectrum and the smoothed average observed science spectrum.  We finally subtract this background model and consider the light lost due to the system rather than trying to account for it in post-processing.

#### Error Propagation
The error value for each pixel is calculated as the propagation of the variance from the readnoise gathered from the header and the poisson error from the number of e- in each pixel.  However, we do not make any corrections for the low count regime as the observed count is not the correct value to use for evaluating the associated counting error.  

### Advanced Reduction Steps

#### Fiber Extraction
We extract fiber spectra from both the science frames as well as the master twilight frame.  We do so by taking the central 5 pixels, +/- 2.5 pixels about the trace.  The outer pixels are given linear weights for the fraction of the pixel that is within 2.5 of the trace.  We then sum up the light at each column for both the science frame and master twilight.  We also calculate a normalized chi2 by comparing the science pixel values normalized by their sum to the master twilight pixel values normalized by their sum as well.  The normalized chi2 gives us a metric by which to flag cosmics and bad columns.  We propagate the error for the spectrum in quadrature using the appropriate weight if the pixel is an outer pixel. Lastly, we linearly interpolate all extracted spectra (and errors) to a common wavelength grid.

#### Fiber Normalization
Fiber normalization relative thoughput of each fiber compared to average.  The variations occur from CCD QE differences, fiber throughput differences, and spectrograph throughput differences.  We use the master twilight frame to evaluate the fiber normalization.  We make the assumption that averaging over many twilights the illumination across out 22' FoV is uniform and the only variation is due to the fiber normalization.  Using the extracted fiber spectra from the master twilight we build our fiber to fiber corrections.  At the HET, the illumination of the FoV depends on the track of the observation and can vary by up to 10%.  To correct the fiber to fiber normalization to our current observations we perform two steps: 1. for each amplifier we exclude real sources and use the sky and an assumption that it should be uniform across fibers to find a normalization correction in 11 bins of wavelength, 2. we interpolate those wavelength bins for each fiber to find the smooth fiber to fiber correction for our science observation.  This correction runs into issues if the field is heavily crowded.  To avoid such issues, fiber normalization adjustments beyond the physical limit of 10% are excluded and set to 1.  We flag fibers with such large adjustments to be excluded later when identify stars for flux and astrometric calibration.

#### Total Masking
A typical VIRUS observation includes three exposures for 78 IFUs each with 448 fibers, totalling 104,832 spectra.  As each IFU has two CCDs and each CCD has two amplifiers, there are a number of CCD defects to map and mask as well as amplifiers that may have temporary issues like bit readout errors.  We previously discussed the CCD pixel mask creation.  In fiber extraction, when collecting the relevant pixels for a given column and fiber, if one of the 5 pixels has a masked value then the spectrum at that column/wavelength is masked.  This is a very conservative masking approach.  We also discussed earlier the evaluation of a normalized chi2 for each wavelength of each fiber's spectrum.  If the normalized chi2 is greater than 5, we mask that spectral wavelength as well as +/- 2 columns from the high chi2 column.  The primary cause for high chi2 values are cosmics and we conservatively mask neighboring columns to avoid the contribution from the fainter wings of the comsic rays.  We also mask fibers with fiber normalizations less than 50% (i.e., dead fibers), and fibers with fiber normalization corrections larger than the physical limit of 10\%.  This latter category encompasses fibers in amplifiers with issues like bit readout errors.

Finally, we flag an entire fiber if more than 20\% of the fiber is flagged already.  We then flag an entire amplifier if more than 20% of the fibers have been flagged.

#### Sky Subtraction
With the fiber normalizations made, we build a single sky spectrum for each exposure using the biweight estimator.  We then subtract the single sky spectrum from each fiber.

#### Spectral Error Normalization
Earlier, we discussed the error propagation from readnoise and poisson counting errors to the error in the extracted spectrum.  However, in the process of rectifying the extracted spectra we correlated noise between spectral columns in the rectified phase space.  Instead of keeping a large co-variance matrix for each spectrum, we instead merely re-normalize our error spectra so that the biweight midvariance of our sky-subtracted spectra divided by the error spectra has a sigma equal to 1.  The correction factor is typically 0.8 and is a single scalar for all spectra in an observation.

#### Astrometric Calibration
We astrometrically calibrate our VIRUS observations to Pan-STARRS (Chambers et al. 2016) Data Release 2.  We go IFU by IFU, collapse the sky-subtraced spectra, interpolate onto a uniform grid, detect point sources in the collapsed image, and match the detections to the Pan-STARRS catalog.  We then fit for a shift and rotation from the initial VIRUS astrometry given by the telescope.  The typical shift is less than 5" and the rotation is <~0.1 degrees.

#### Source Extraction
The goal of our source extraction is a high signal to noise spectrum without introducing large systematic errors in the relative flux calibration.  To accomplish the former we use an optimal extraction method.  However, VIRUS spectra include two complications: an undersampled spatial point spread function and differential atmospheric refraction.  VIRUS fibers are 1.5" in diameter and the typical seeing of the HET is roughly the same.  This means that up to 60-70% of the light from a source can be in a single fiber, however the weight for each fiber can change quickly with slight changes in the assumed position of the source for a given spatial point spread function.  Thus, high quality astrometry, appropriately assumed spread function, and an accurate model of the change of position as a function of wavelength from differential atmospheric refraction are required.

Luckily, since we have so many units covering a large area on sky, we typically have >20 stars to use to model both the spatial point spread function as well as the differential atmospheric refraction.  Simultaneously, we can model the normalization between the three exposures that comprise a single observation.  The mirror illumination at the HET depends on the track location, which constantly changes over an exposure.  Thus the amount of light collected for each of the three exposures is likely different and needs to be normalized to the average illumination.  We model the normalization, the spatial point spread function, and the differential atmospheric refraction in an iterative approach with convergence after just 2 or 3 iterations.

#### Flux Calibration
For flux calibration, we use a pre-computed throughput curve from the HETDEX collaboration (Gebhardt et al. 2020), but for applicability to a given observation we must take into account the conditions of that observation.  We use the stars in the field to find the normalization (transparency) for the given shot.  After applying the standard throughput curve we convolve the initially calibrated spectra with a Pan-STARRS g' filter and find the biweight estimated offset between the VIRUS g' magnitude and the Pan-STARRS g' magnitude.  That offset is the normalization we apply to the standard througphput curve and gives us our final flux calibration.

#### Data Cubes
A full FoV data cube would be quite large and sparse due to the 1/4.5 fill factor of the VIRUS IFUs.  Instead, we make a data cube for any given IFU.  We linearly interpolate at any given wavelength to go from the 1344 fibers on sky to a uniform grid.  We take into account differential atmmospheric refraction as function of wavelength, which amounts to a change in position for the fibers of roughly 1" across 3500-5500A.

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
