#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 13:10:11 2020

@author: gregz
"""
import matplotlib
matplotlib.use('agg')
import argparse as ap
from astropy.convolution import convolve, Gaussian2DKernel
from astropy.io import fits
from astropy.wcs import WCS
from astropy.table import Table
import tables
import numpy as np
import os.path as op
import sys
import warnings
from input_utils import setup_logging
from astrometry import Astrometry
from math_utils import biweight
from extract import Extract
from scipy.interpolate import griddata
import matplotlib.pyplot as plt



import glob

mask_dict = {'20200430-20200501': ['057RL', '057RU', '057LL', '057LU', '058RU', 
                                   '058RL', '021RL', '021RU', '021LL', '021LU'],
             '20200430-20200715': ['092LU', '094RU', '046RU', '104LL', '104LU',
                                   '028RU', '028RL', '027LL', '027LU', '067LL',
                                   '067LU', '025LU', '106RU', '106RL', '026LL',
                                   '026LU', '026RL', '026RU', '103LL', '103LU'],
             '20200525-20200526': ['057RL', '057LL', '089RU', '089RL'],
             '20200517-20200522': ['030RL', '030RU', '030LL', '030LU'],
             '20200523-20200526': ['039RL', '039RU', '039LL', '039LU'],
             '20200622-20200715': ['089RL', '089RU', '096RL', '096RU']}
def get_script_path():
    return op.dirname(op.realpath(sys.argv[0]))

warnings.filterwarnings("ignore")

DIRNAME = get_script_path()

parser = ap.ArgumentParser(add_help=True)

parser.add_argument("-d", "--directory",
                    help='''base directory for reductions''',
                    type=str, default="")

parser.add_argument("-c", "--caldirectory",
                    help='''cal directory for reductions''',
                    type=str, default="/work/03946/hetdex/maverick/LRS2/CALS")

parser.add_argument("h5files",
                    help='''e.g., 20200430_0000020.h5''',
                    type=str)

parser.add_argument("surname",
                    help='''file name modification''',
                    type=str)

parser.add_argument("image_center_size",
                    help='''RA, Dec center of image and size (deg, deg, arcmin)
                            "150.0, 50.0, 22.0"''',
                    default=None, type=str)

parser.add_argument("-ps", "--pixel_scale",
                    help='''Pixel scale for output image in arcsec''',
                    default=1.0, type=float)

parser.add_argument("-if", "--image_file",
                    help='''Image filename''',
                    default=None, type=str)

parser.add_argument("-ff", "--filter_file",
                    help='''Filter filename''',
                    default=None, type=str)

def rebin(arr, new_shape):
    """Rebin 2D array arr to shape new_shape by averaging."""
    shape = (new_shape[0], arr.shape[0] // new_shape[0],
             new_shape[1], arr.shape[1] // new_shape[1])
    return arr.reshape(shape).mean(axis=(-1,1))

def make_image_interp(Pos, y, ye, xg, yg, xgrid, ygrid, sigma, cnt_array,
                      binsize=2):
    # Loop through shots, average images before convolution
    # Make a +/- 1 pixel mask for places to keep
    G = Gaussian2DKernel(sigma)
    image_all = np.zeros((len(cnt_array),) + xgrid.shape)
    error_all = np.zeros((len(cnt_array),) + xgrid.shape)
    weight_all = np.zeros((len(cnt_array),) + xgrid.shape)

    for k, cnt in enumerate(cnt_array):
        l1 = int(cnt[0])
        l2 = int(cnt[1])
        P = Pos[l1:l2]
        yi = y[l1:l2]
        yei = ye[l1:l2]
        xc = np.interp(P[:, 0], xg, np.arange(len(xg)), left=0., right=len(xg))
        yc = np.interp(P[:, 1], yg, np.arange(len(yg)), left=0., right=len(yg))
        xc = np.array(np.round(xc), dtype=int)
        yc = np.array(np.round(yc), dtype=int)
        image, error, weight = (xgrid*np.nan, xgrid*np.nan, xgrid*np.nan)
        gsel = (xc>1) * (xc<len(xg)-1) * (yc>1) * (yc<len(yg)-1)
        image[yc[gsel], xc[gsel]] = yi[gsel] / (np.pi * 0.75**2)
        error[yc[gsel], xc[gsel]] = yei[gsel] / (np.pi * 0.75**2)
        image_all[k] = image
        error_all[k] = error
        bsel = np.isfinite(y[gsel])
        for i in np.arange(-1, 2):
            for j in np.arange(-1, 2):
                weight[yc[gsel][bsel]+i, xc[gsel][bsel]+j] = 1.
        weight_all[k] = weight
    image = np.nanmedian(image_all, axis=0)
    error = np.nanmedian(error_all, axis=0)
    weight = np.nanmedian(weight_all, axis=0)
    image = convolve(image, G, preserve_nan=False, boundary='extend')
    image[np.isnan(weight)] = np.nan
    error[np.isnan(weight)] = np.nan
    image[np.isnan(image)] = 0.0
    error[np.isnan(error)] = 0.0
    return image, error, weight

def make_image(Pos, y, ye, xg, yg, xgrid, ygrid, sigma, cnt_array):
    image = xgrid * 0.
    error = xgrid * 0.
    weight = xgrid * 0.
    N = int(sigma*5)
    indx = np.searchsorted(xg, Pos[:, 0])
    indy = np.searchsorted(yg, Pos[:, 1])
    nogood = ((indx < N) + (indx >= (len(xg) - N)) +
              (indy < N) + (indy >= (len(yg) - N)))
    nogood = nogood + np.isnan(y)
    y[nogood] = 0.0
    ye[nogood] = 0.0
    indx[nogood] = N+1
    indy[nogood] = N+1
    indx_array = np.zeros((len(y), 4*N**2), dtype=int)
    indy_array = np.zeros((len(y), 4*N**2), dtype=int)
    for i in np.arange(2*N):
        for j in np.arange(2*N):
            c = i * 2 * N + j
            indx_array[:, c] = indx + i - N
            indy_array[:, c] = indy + j - N
    d = np.sqrt((xgrid[indy_array, indx_array] - Pos[:, 0][:, np.newaxis])**2 + 
                (ygrid[indy_array, indx_array] - Pos[:, 1][:, np.newaxis])**2)
    G = np.exp(-0.5 * d**2 / sigma**2)
    G[:] /= G.sum(axis=1)[:, np.newaxis]
    G[nogood] = 0.
    for j in np.arange(len(y)):
        image[indy_array[j], indx_array[j]] += y[j] * G[j]
        error[indy_array[j], indx_array[j]] += ye[j]**2 * G[j]
        weight[indy_array[j], indx_array[j]] += G[j]
#    for j in np.arange(len(cnt_array)):
#        l1 = cnt_array[j, 0]
#        l2 = cnt_array[j, 1]
#        idy = indy_array[l1:l2]
#        idx = indx_array[l1:l2]
#        g = G[l1:l2]
#        yi = y[l1:l2]
#        yie = ye[l1:l2]
#        indj = 336
#        for i in np.arange(indj):
#            image[idy[i::indj].ravel(), idx[i::indj].ravel()] += (yi[i::indj][:, np.newaxis] * g[i::indj]).ravel()
#            error[idy[i::indj].ravel(), idx[i::indj].ravel()] += (yie[i::indj][:, np.newaxis]**2 * g[i::indj]).ravel()
#            weight[idy[i::indj].ravel(), idx[i::indj].ravel()] += g[i::indj].ravel()
    weight[:] *= np.pi * 0.75**2
    return image, np.sqrt(error), weight

args = parser.parse_args(args=None)
args.log = setup_logging('make_image_from_h5')

def_wave = np.linspace(3470., 5540., 1036)

h5files = sorted(glob.glob(args.h5files))

bounding_box = [float(corner.replace(' ', ''))
                        for corner in args.image_center_size.split(',')]

bounding_box[2] = int(bounding_box[2]*60./args.pixel_scale/2.) * 2 * args.pixel_scale

bb = int(bounding_box[2]/args.pixel_scale/2.)*args.pixel_scale
N = int(bounding_box[2]/args.pixel_scale/2.) * 2 + 1
args.log.info('Image size in pixels: %i' % N)
xg = np.linspace(-bb, bb, N)
yg = np.linspace(-bb, bb, N)
xgrid, ygrid = np.meshgrid(xg, yg)

cube = np.zeros((len(def_wave),) + xgrid.shape, dtype='float32')
ecube = np.zeros((len(def_wave),) + xgrid.shape, dtype='float32')
#weightcube = np.zeros((len(def_wave),) + xgrid.shape, dtype='float32')

cnt = 0
cnt_array = np.zeros((len(h5files), 2), dtype=int)
for i, h5file in enumerate(h5files):
    t = tables.open_file(h5file)
    ra = t.root.Info.cols.ra[:]
    cnt += len(ra)
    cnt_array[i, 0] = cnt - len(ra)
    cnt_array[i, 1] = cnt
    t.close()
args.log.info('Number of total fibers: %i' % cnt)
raarray = np.zeros((cnt, len(def_wave)), dtype='float32')
decarray = np.zeros((cnt, len(def_wave)), dtype='float32')
specarray = np.zeros((cnt, len(def_wave)), dtype='float32')
errarray = np.zeros((cnt, len(def_wave)), dtype='float32')

A = Astrometry(bounding_box[0], bounding_box[1], 0., 0., 0.)
tp = A.setup_TP(A.ra0, A.dec0, 0.)

if args.filter_file is not None:
    R = Table.read(args.filter_file, format='ascii')
    response = np.interp(def_wave, R['Wavelength'], R['R'], left=0.0, right=0.0)
    

    
if args.image_file is not None:
    name = op.basename(args.image_file)[:-5] + '_rect.fits'
    if op.exists(name):
        f = fits.open(name)
        binimage = f[0].data
    else:
        image_file = fits.open(args.image_file)
        wc = WCS(image_file[0].header)
        ny, nx = image_file[0].data.shape
        yind, xind = np.indices((ny, nx))
        xn, yn = wc.wcs_world2pix(A.ra0, A.dec0, 1)
        tpn = A.setup_TP(A.ra0, A.dec0, 0., xn, yn, x_scale=-0.25, y_scale=0.25)
        r, d = wc.wcs_pix2world(xind.ravel()+1.0, yind.ravel()+1.0, 1)
        P = np.zeros((len(r), 2))
        x, y = tp.wcs_world2pix(r, d, 1)
        P[:, 0], P[:, 1] = (x, y)
        d = np.sqrt((P[:, 0]-xg[0])**2 + (P[:, 1]-yg[0])**2)
        I = np.argmin(d)
        yi, xi = np.unravel_index(I, yind.shape)
        N = 4 * len(xg)
        x = np.reshape(x, image_file[0].data.shape)
        y = np.reshape(y, image_file[0].data.shape)
        newimage = image_file[0].data[yi:yi+N, xi:xi+N]
        binimage = rebin(newimage, (newimage.shape[0]//4, newimage.shape[1]//4))
        ximage = rebin(x[yi:yi+N, xi:xi+N], (newimage.shape[0]//4, newimage.shape[1]//4))
        yimage = rebin(y[yi:yi+N, xi:xi+N], (newimage.shape[0]//4, newimage.shape[1]//4))
        P = np.zeros((len(ximage.ravel()), 2))
        P[:, 0], P[:, 1] = (ximage.ravel(), yimage.ravel())
        binimage = griddata(P, binimage.ravel(), (xgrid, ygrid), method='cubic')
        h = tp.to_header()
        N = len(xg)
        h['CRPIX1'] = np.interp(0., xg, np.arange(len(xg)))+1.0
        h['CRPIX2'] = np.interp(0., xg, np.arange(len(xg)))+1.0
        args.log.info('Writing %s' % name)
        fits.PrimaryHDU(np.array(binimage, dtype='float32'), header=h).writeto(name, overwrite=True)

cnt = 0
norm_array = np.ones((len(h5files),))
for jk, h5file in enumerate(h5files):
    args.log.info('Working on %s' % h5file)
    t = tables.open_file(h5file)
    date = int(op.basename(h5file).split('_')[0])
    ifuslots = t.root.Info.cols.ifuslot[:]
    amps = np.array([i.decode("utf-8") for i in t.root.Info.cols.amp[:]])
                
            
    ra = t.root.Info.cols.ra[:]
    dec = t.root.Info.cols.dec[:]
    RA = t.root.Survey.cols.ra[0]
    Dec = t.root.Survey.cols.dec[0]
    pa = t.root.Survey.cols.pa[0]
    offset = t.root.Survey.cols.offset[0]
    spectra = t.root.Fibers.cols.spectrum[:] / offset
    error = t.root.Fibers.cols.error[:] / offset
    for key in mask_dict.keys():
        date1 = int(key.split('-')[0])
        date2 = int(key.split('-')[1])
        if (date >= date1) and (date < date2):
            ifulist = mask_dict[key]
            for ifuamp in ifulist:
                ifu = int(ifuamp[:3])
                amp = ifuamp[3:]
                sel = np.where((ifu == ifuslots) * (amp == amps))[0]
                spectra[sel] = np.nan
    cnt1 = cnt + len(ra)
    E = Extract()
    Aother = Astrometry(bounding_box[0], bounding_box[1], pa, 0., 0.)
    header = tp.to_header()
    E.get_ADR_RAdec(Aother)
    raarray[cnt:cnt1, :] = ra[:, np.newaxis] - E.ADRra[np.newaxis, :] / 3600. / np.cos(np.deg2rad(A.dec0))
    decarray[cnt:cnt1, :] = dec[:, np.newaxis] - E.ADRdec[np.newaxis, :] / 3600.
    if args.filter_file is not None:
        wsel = response>0.0
        mask = np.isfinite(spectra[:, wsel]) * (spectra[:, wsel] != 0.0)
        Pos = np.zeros((cnt1-cnt, 2))
        x, y = tp.wcs_world2pix(np.nanmean(raarray[cnt:cnt1, wsel], axis=1),
                                np.nanmean(decarray[cnt:cnt1, wsel], axis=1), 1)
        Pos[:, 0], Pos[:, 1] = (x, y)
        xc = np.interp(Pos[:, 0], xg, np.arange(len(xg)), left=0., right=len(xg))
        yc = np.interp(Pos[:, 1], yg, np.arange(len(yg)), left=0., right=len(yg))
        xc = np.array(np.round(xc), dtype=int)
        yc = np.array(np.round(yc), dtype=int)
        gsel = (xc>0) * (xc<len(xg)) * (yc>0) * (yc<len(yg))
        d = np.sqrt(xgrid**2 + ygrid**2)
        skyvalues = (binimage[yc[gsel], xc[gsel]] < 0.01) * (d[yc[gsel], xc[gsel]] > 420.)
        backspectra = biweight(spectra[gsel][skyvalues], axis=0)
        args.log.info('Average spectrum residual value: %0.3f' % np.nanmedian(backspectra))
        spectra[:] -= backspectra[np.newaxis, :]
        collapse_image = (np.nansum(spectra[:, wsel] * response[np.newaxis, wsel], axis=1) /
                          np.nansum(mask * response[np.newaxis, wsel], axis=1))
        collapse_eimage = np.sqrt((np.nansum(error[:, wsel]**2 * response[np.newaxis, wsel], axis=1) /
                                   np.nansum(mask * response[np.newaxis, wsel], axis=1)))


        
        cn = np.ones((1, 2))
        cn[0, 0] = 0
        cn[0, 1] = len(collapse_image)
        image, errorimage, weight = make_image_interp(Pos, collapse_image, collapse_eimage,
                                                      xg, yg, xgrid, ygrid, 1.8 / 2.35,
                                                      cn)
        
        image[image==0.] = np.nan
        G = Gaussian2DKernel(4.5)
        cimage = convolve(image, G, preserve_nan=True, boundary='extend')
        nimage = binimage * 1.
        nimage[np.isnan(cimage)] = np.nan
        nimage = convolve(nimage, G, preserve_nan=True, boundary='extend')
        sel = np.isfinite(cimage) * (nimage > 0.05)
        yim = cimage / nimage
        d = np.sqrt(xgrid**2 + ygrid**2)
        yim[~sel] = 0.0
        nimage[np.isnan(nimage)] = 0.0
        image[np.isnan(image)] = 0.0
        bimage = binimage * 1.
        bimage[image==0.] = 0.
        xmax = np.linspace(-0.1, 0.02, 26)
        bmax = xmax*0.
        thresh = 0.05
        for i, v in enumerate(xmax):
            y = (cimage - v) / nimage
            y[~sel] = 0.0
            norm, std = biweight(y[sel][nimage[sel] > thresh], calc_std=True)
            bmax[i] = std / norm
        back = xmax[np.argmin(bmax)]
        args.log.info('Background for %s: %0.2f' % (h5file, back))
        y = (cimage - back) / nimage
        y[~sel] = 0.0
        norm, std = biweight(y[sel][nimage[sel] > thresh], calc_std=True)
        norm_array[jk] = norm
        if norm_array[jk] < 0.:
            norm_array[jk] = np.nan
        args.log.info('Normalization/STD for %s: %0.2f, %0.2f' % (h5file, norm, std/norm))
        diff_image = (cimage-v)/norm_array[jk] - nimage
        limage = (cimage-v)/norm_array[jk]
        bl, bottom_var = biweight(limage[(binimage<0.01) * np.isfinite(cimage)], calc_std=True)

        threshold = np.sqrt((nimage * 0.25)**2 + (3.*bottom_var)**2)
        flagged = np.abs(diff_image[yc[gsel], xc[gsel]]) > threshold[yc[gsel], xc[gsel]]
        #spectra[np.where(gsel)[0][flagged]] = np.nan
        args.log.info('%i fibers flagged for too large of a difference' % flagged.sum())
        plt.figure(figsize=(10, 8))
        plt.scatter(nimage[sel], y[sel] / norm, s=5, alpha=0.05)
        plt.plot([0.03, 0.6], [1., 1.], 'r-', lw=2)
        plt.plot([0.03, 0.6], [1.-std/norm, 1.-std/norm], 'r--', lw=1)
        plt.plot([0.03, 0.6], [1.+std/norm, 1.+std/norm], 'r--', lw=1)
        mn = np.nanpercentile(y[sel], 5)/norm
        mx = np.nanpercentile(y[sel], 95)/norm
        ran = mx - mn
        plt.axis([0.03, 0.6, 0.6, 1.4])
        name = op.basename(h5file)[:-3] + '_norm.png'
        plt.savefig(name, dpi=300)
        name = op.basename(h5file)[:-3] + '_rect.fits'
        h = tp.to_header()
        N = len(xg)
        h['CRPIX1'] = np.interp(0., xg, np.arange(len(xg)))+1.0
        h['CRPIX2'] = np.interp(0., xg, np.arange(len(xg)))+1.0
        fits.PrimaryHDU(np.array(image/norm_array[jk], dtype='float32'), header=h).writeto(name, overwrite=True)
    
    specarray[cnt:cnt1, :] = spectra / norm_array[jk]
    errarray[cnt:cnt1, :] = error / norm_array[jk]
    cnt = cnt + len(ra)
    t.close()

specarray[:] *= biweight(norm_array)
errarray[:] *= biweight(norm_array)

Pos = np.zeros((len(raarray), 2))
for i in np.arange(len(def_wave)):
    x, y = tp.wcs_world2pix(raarray[:, i], decarray[:, i], 1)
    args.log.info('Working on wavelength %0.0f' % def_wave[i])
    Pos[:, 0], Pos[:, 1] = (x, y)
    data = specarray[:, i]
    edata = errarray[:, i]
    image, errorimage, weight = make_image_interp(Pos, data, edata, xg, yg, 
                                                  xgrid, ygrid, 1.8 / 2.35,
                                                  cnt_array)
    cube[i, :, :] += image
    ecube[i, :, :] += errorimage
    #weightcube[i, :, :] += weight

#cube = np.where(weightcube > 0.4, cube / weightcube, 0.0)
#ecube = np.where(weightcube > 0.4, ecube / weightcube, 0.0)
name = op.basename('%s_cube.fits' % args.surname)
header['CRPIX1'] = (N+1) / 2
header['CRPIX2'] = (N+1) / 2
header['CDELT3'] = 2.
header['CRPIX3'] = 1.
header['CRVAL3'] = 3470.
F = fits.PrimaryHDU(np.array(cube, 'float32'), header=header)
F.writeto(name, overwrite=True)
name = op.basename('%s_errorcube.fits' % args.surname)
header['CRPIX1'] = (N+1) / 2
header['CRPIX2'] = (N+1) / 2
header['CDELT3'] = 2.
header['CRPIX3'] = 1.
header['CRVAL3'] = 3470.
F = fits.PrimaryHDU(np.array(ecube, 'float32'), header=header)
F.writeto(name, overwrite=True)