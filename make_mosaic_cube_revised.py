#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 13:10:11 2020

@author: gregz
"""
import matplotlib
matplotlib.use('agg')
import argparse as ap
import csv
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from numba import njit
from astropy.convolution import convolve, Gaussian2DKernel
from astropy.convolution import Gaussian1DKernel, interpolate_replace_nans
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
from scipy.interpolate import griddata, PchipInterpolator
import matplotlib.pyplot as plt



import glob

import diagnose_m101_hierarchical as validated_m101

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

# M101-specific residual sky correction at H5-consumption time.
M101_RA_DEG = 210.800
M101_DEC_DEG = 54.333
M101_SKY_MIN_RADIUS_ARCMIN = 6.0
M101_SKY_NEXPOSURES = 3
M101_SKY_MIN_FINITE_FRACTION = 0.8
M101_SKY_MIN_FIBERS = 20

# M101-specific, empirically validated additive amplifier-edge correction.
# The residual is measured from blank-sky Raw spectra in e-/A, smoothed only
# in folded amplifier coordinate, propagated through the known Quick
# Reduction calibration, and applied to Fibers.spectrum before the existing
# M101 residual-sky correction.  This is not an FTF correction and does not
# identify the underlying detector/scattered-light cause.
M101_AMP_REFERENCE_J = np.arange(40, 71)
M101_AMP_SAFE_WAVE = (3700.0, 5350.0)
M101_AMP_SMOOTH_SIGMA = 2.5
M101_AMP_FIBERS = 112

# Data-quality bits for the final SCI/VAR estimator state.  Coverage strength
# and the number of contributing shots remain continuous/independent products.
DQ_INSUFFICIENT_SUPPORT = np.uint16(1 << 0)
DQ_VAR_INCOMPLETE = np.uint16(1 << 1)
DQ_EMPIRICAL_VAR_USED = np.uint16(1 << 2)
DQ_FORMAL_VAR_USED = np.uint16(1 << 3)
DQ_VAR_EMPIRICAL_ONLY = np.uint16(1 << 4)

# Error-provenance columns used by the compact wavelength/exposure counters.
PROV_NSCI = 0
PROV_ERROR_VALID = 1
PROV_ERROR_NONFINITE = 2
PROV_ERROR_ZERO = 3
PROV_ERROR_NEGATIVE = 4
PROV_ERROR_OTHER = 5

# Empirical LSF/FWHM calibration anchors.  These are AIR wavelengths and are
# intentionally kept separate from the science wavelength grid.  The local
# project resources (Lines_list/virus_lines.dat and wave_utils.py) contain
# these anchors.  Established Hg/Cd laboratory/astronomical line lists show
# that the blue Cd I components around 3610 A and the Hg/Cd structure around
# 3650 A are known blends; they are retained for QA but excluded from LSF use.
REFERENCE_ARC_WAVELENGTHS = np.array([
    3610.508,
    3650.153,
    4046.565,
    4358.335,
    4678.149,
    4799.912,
    4916.068,
    5085.822,
    5460.750,
], dtype=float)
ARC_WINDOW_ANGSTROM = 15.0
ARC_USABLE_FOR_LSF = np.array(
    [False, False, True, True, True, True, True, True, True], dtype=bool)
ARC_BLEND_STATES = np.array([
    'KNOWN_BLEND', 'KNOWN_BLEND',
    'UNKNOWN', 'UNKNOWN', 'UNKNOWN', 'UNKNOWN', 'UNKNOWN', 'UNKNOWN', 'UNKNOWN'
], dtype='U16')
ARC_BLEND_NOTES = {
    3610.508: ('Known Cd I blend with 3612.873 A and 3614.453 A; '
               'excluded from LSF'),
    3650.153: ('Known Hg/Cd blend; Hg I 3654.84 A produces red shoulder; '
               'excluded from LSF'),
    4046.565: 'No significant known blend affecting half-maximum width',
    4358.335: 'No significant known blend affecting half-maximum width',
    4678.149: 'No significant known blend affecting half-maximum width',
    4799.912: 'No significant known blend affecting half-maximum width',
    4916.068: 'No significant known blend affecting half-maximum width',
    5085.822: 'No significant known blend affecting half-maximum width',
    5460.750: 'No significant known blend affecting half-maximum width',
}


def subtract_m101_residual_sky(spectra, ra, dec, xg, yg, tp,
                               binimage=None, log=None, h5file=''):
    """Subtract one M101-safe residual sky spectrum per exposure.

    This operates directly on already sky-subtracted H5 SCI.  It deliberately
    does not read or add back ``Fibers.skyspectrum``: this is the single
    residual-sky correction performed by the cube builder.  All masks are
    full-length masks in the local H5 fiber coordinate system.

    If an external image is available, ``binimage < 0.01`` is an additional
    blank-sky criterion.  Invalid image pixels are excluded rather than being
    silently classified as blank.  If no external image is supplied, the
    radial and mosaic-region criteria remain active and the log records that
    the external blank-image criterion was unavailable.
    """
    n_fib, n_wave = spectra.shape
    inds_local = np.arange(n_fib)
    block_ids = inds_local // 112
    dra_arcmin = ((ra - M101_RA_DEG) *
                  np.cos(np.deg2rad(M101_DEC_DEG)) * 60.0)
    ddec_arcmin = (dec - M101_DEC_DEG) * 60.0
    radius_arcmin = np.sqrt(dra_arcmin ** 2 + ddec_arcmin ** 2)
    sky_region = radius_arcmin > M101_SKY_MIN_RADIUS_ARCMIN

    # Image sampling uses the same 1-based WCS pixel convention as the
    # existing normalization code.  This is only for the optional external
    # image criterion; the M101 radial selection above uses RA/Dec directly.
    x, y = tp.wcs_world2pix(ra, dec, 1)
    xc = np.rint(np.interp(x, xg, np.arange(len(xg)), left=0., right=len(xg)))
    yc = np.rint(np.interp(y, yg, np.arange(len(yg)), left=0., right=len(yg)))
    xc = np.asarray(xc, dtype=int)
    yc = np.asarray(yc, dtype=int)
    in_valid_image_region = (
        np.isfinite(x) & np.isfinite(y) &
        (xc >= 0) & (xc < len(xg)) &
        (yc >= 0) & (yc < len(yg)))

    external_blank = None
    if binimage is not None:
        external_blank = np.zeros(n_fib, dtype=bool)
        valid_image = in_valid_image_region.copy()
        valid_image[valid_image] &= np.isfinite(binimage[yc[valid_image], xc[valid_image]])
        external_blank[valid_image] = (
            binimage[yc[valid_image], xc[valid_image]] < 0.01)
    elif log is not None:
        log.warning(
            'M101 residual sky %s: external image unavailable; not applying '
            'an external blank-image classification.', h5file)

    def report(message, *values):
        if log is not None:
            log.info(message, *values)

    for k in range(M101_SKY_NEXPOSURES):
        exp_sel = (block_ids % M101_SKY_NEXPOSURES) == k
        exp_indices = np.where(exp_sel)[0]
        if exp_indices.size == 0:
            report('M101 residual sky %s exposure %d: no fibers; skipping correction.',
                   h5file, k + 1)
            continue

        finite_counts = np.isfinite(spectra[exp_indices]).sum(axis=1)
        sufficient_finite = (
            finite_counts >= int(np.ceil(M101_SKY_MIN_FINITE_FRACTION * n_wave)))
        full_sky_mask = np.zeros(n_fib, dtype=bool)
        full_sky_mask[exp_indices] = (
            sky_region[exp_indices] &
            in_valid_image_region[exp_indices] &
            sufficient_finite)
        if external_blank is not None:
            full_sky_mask &= external_blank

        selected_sky_count = int(full_sky_mask.sum())
        if selected_sky_count < M101_SKY_MIN_FIBERS:
            report(
                'M101 residual sky %s exposure %d: total fibers=%d, beyond 6 arcmin=%d, '
                'valid image fibers=%d, blank-image candidates=%s, finite sky candidates=%d, '
                'selected sky fibers=%d; fewer than %d, skipping correction.',
                h5file, k + 1, int(exp_indices.size),
                int(np.sum(sky_region[exp_indices])),
                int(np.sum(in_valid_image_region[exp_indices])),
                ('unavailable' if external_blank is None else
                 str(int(np.sum(external_blank[exp_indices])))),
                selected_sky_count, selected_sky_count, M101_SKY_MIN_FIBERS)
            if log is not None:
                log.warning(
                    'M101 residual sky %s exposure %d skipped: only %d sky '
                    'candidates (minimum %d).',
                    h5file, k + 1, selected_sky_count, M101_SKY_MIN_FIBERS)
            continue

        residual_sky = biweight(spectra[full_sky_mask], axis=0)
        finite_residual = np.isfinite(residual_sky)
        residual_fraction = finite_residual.sum() / float(n_wave)
        if residual_fraction < M101_SKY_MIN_FINITE_FRACTION:
            report(
                'M101 residual sky %s exposure %d: residual sky finite at %d/%d '
                'wavelengths; mostly nonfinite, skipping correction.',
                h5file, k + 1, int(finite_residual.sum()), n_wave)
            if log is not None:
                log.warning(
                    'M101 residual sky %s exposure %d skipped because the residual '
                    'sky is mostly nonfinite.', h5file, k + 1)
            continue

        # Apply only finite residual bins if a small number of bins are
        # undefined; this leaves those undefined SCI bins unchanged.
        spectra[np.ix_(exp_indices, finite_residual)] -= residual_sky[finite_residual]
        before_level = np.nanmedian(
            spectra[full_sky_mask] + residual_sky[None, :])
        after_level = np.nanmedian(spectra[full_sky_mask])
        p16, p50, p84 = np.nanpercentile(residual_sky[finite_residual], [16, 50, 84])
        report(
            'M101 residual sky %s exposure %d: total fibers=%d, beyond 6 arcmin=%d, '
            'valid image fibers=%d, blank-image candidates=%s, finite sky candidates=%d, '
            'selected sky fibers=%d, median residual sky=%0.5g, '
            'residual sky p16/p50/p84=(%0.5g, %0.5g, %0.5g), finite residual wavelengths=%d/%d, '
            'median sky-region level before=%0.5g, after=%0.5g',
            h5file, k + 1, int(exp_indices.size),
            int(np.sum(sky_region[exp_indices])),
            int(np.sum(in_valid_image_region[exp_indices])),
            ('unavailable' if external_blank is None else
             str(int(np.sum(external_blank[exp_indices])))),
            selected_sky_count, selected_sky_count, p50, p16, p50, p84,
            int(finite_residual.sum()), n_wave, before_level, after_level)

    return spectra


def _m101_amplifier_groups(info):
    """Return the validated stored-row exposure/amp bookkeeping.

    The H5 writer repeats each physical IFU/amplifier block for the three
    exposures.  As in the existing science grouping, exposure is therefore
    ``(row // 112) % 3``.  Within each exposure/IFUSLOT/amplifier group the
    stored row order is the Remedy fiber order j=0..111.
    """
    nrows = int(info.nrows)
    ifuslot = np.asarray(info.cols.ifuslot[:])
    amps = np.asarray([i.decode("utf-8") if isinstance(i, bytes) else str(i)
                       for i in info.cols.amp[:]])
    specid = np.asarray(info.cols.specid[:])
    ifuid = np.asarray(info.cols.ifuid[:])
    nslots = len(np.unique(ifuslot))
    if nslots <= 0 or nrows % (448 * nslots) != 0:
        raise ValueError(
            "M101 amplifier correction cannot infer exposure blocks: rows=%d, "
            "nslots=%d; expected divisibility by 448*nslots" %
            (nrows, nslots))
    nexp = int(nrows / float(448 * nslots))
    if nexp != M101_SKY_NEXPOSURES:
        raise ValueError("M101 amplifier correction inferred %d exposures; "
                         "expected %d" % (nexp, M101_SKY_NEXPOSURES))
    row_exposure = ((np.arange(nrows) // M101_AMP_FIBERS) % nexp + 1).astype(int)
    row_j = np.full(nrows, -1, dtype=int)
    row_amp = np.full(nrows, "", dtype="U2")
    groups = []
    keys = sorted(set(zip(row_exposure.tolist(), ifuslot.tolist(), amps.tolist())),
                  key=lambda key: (int(key[0]), int(key[1]), key[2]))
    for exposure, slot, amp in keys:
        indices = np.flatnonzero(
            (row_exposure == exposure) & (ifuslot == slot) & (amps == amp))
        if indices.size != M101_AMP_FIBERS:
            raise ValueError(
                "M101 amplifier group exposure %d IFUSLOT %s AMP %s has %d "
                "rows; expected 112" %
                (exposure, slot, amp, indices.size))
        if np.unique(specid[indices]).size != 1 or \
                np.unique(ifuid[indices]).size != 1:
            raise ValueError(
                "M101 amplifier group exposure %d IFUSLOT %s AMP %s has "
                "inconsistent SPECID/IFUID" % (exposure, slot, amp))
        row_j[indices] = np.arange(M101_AMP_FIBERS, dtype=int)
        row_amp[indices] = amp
        groups.append({"exposure": int(exposure), "ifuslot": int(slot),
                       "ifuid": int(np.unique(ifuid[indices])[0]),
                       "specid": int(np.unique(specid[indices])[0]),
                       "amp": amp, "indices": indices})
    physical_groups = {}
    for group in groups:
        physical_groups.setdefault((group["ifuslot"], group["amp"]), 0)
        physical_groups[(group["ifuslot"], group["amp"])] += 1
    if any(count != M101_SKY_NEXPOSURES
           for count in physical_groups.values()):
        raise ValueError("M101 amplifier bookkeeping does not contain three "
                         "exposures for every physical IFUSLOT/amplifier")
    return row_exposure, row_j, row_amp, groups


def _m101_amp_blank_selection(ra, dec, row_exposure, exposure,
                              fiber_spectrum, original_image_data,
                              original_image_wcs):
    """Use the production M101-safe blank-sky selection for one exposure."""
    nrows, n_wave = fiber_spectrum.shape
    if original_image_data is None or original_image_wcs is None:
        return np.zeros(nrows, dtype=bool)
    dra = ((ra - M101_RA_DEG) * np.cos(np.deg2rad(M101_DEC_DEG)) * 60.0)
    ddec = (dec - M101_DEC_DEG) * 60.0
    radial_blank = np.hypot(dra, ddec) > M101_SKY_MIN_RADIUS_ARCMIN

    x, y = original_image_wcs.world_to_pixel_values(ra, dec)
    finite = np.isfinite(x) & np.isfinite(y)
    xi = np.zeros(x.shape, dtype=int)
    yi = np.zeros(y.shape, dtype=int)
    xi[finite] = np.rint(x[finite]).astype(int)
    yi[finite] = np.rint(y[finite]).astype(int)
    valid_image = (
        finite
        & (xi >= 0)
        & (xi < original_image_data.shape[1])
        & (yi >= 0)
        & (yi < original_image_data.shape[0])
    )
    image_blank = np.zeros(nrows, dtype=bool)
    image_blank[valid_image] = (
        np.isfinite(
            original_image_data[
                yi[valid_image],
                xi[valid_image]
            ]
        )
        &
        (
            original_image_data[
                yi[valid_image],
                xi[valid_image]
            ] < 0.01
        )
    )
    sufficient = (np.isfinite(fiber_spectrum).sum(axis=1) >= int(np.ceil(
        M101_SKY_MIN_FINITE_FRACTION * n_wave)))
    return ((row_exposure == exposure) & radial_blank & valid_image &
            image_blank & sufficient)


def _m101_amp_smooth_profile(values):
    """The validated Gaussian smoothing used by the additive diagnostic."""
    from scipy.ndimage import gaussian_filter1d

    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    if not np.any(finite):
        return np.full(values.shape, np.nan, dtype=float)
    numerator = gaussian_filter1d(np.where(finite, values, 0.0),
                                  M101_AMP_SMOOTH_SIGMA, mode="nearest")
    denominator = gaussian_filter1d(finite.astype(float),
                                    M101_AMP_SMOOTH_SIGMA, mode="nearest")
    result = np.full(values.shape, np.nan, dtype=float)
    valid = denominator > 0.0
    result[valid] = numerator[valid] / denominator[valid]
    return result


def _m101_amp_raw_to_fibers_basis(h5, exposure):
    """Return the absolute Quick Reduction Raw -> Fibers transfer."""
    if "Survey" not in h5.root._v_children:
        raise ValueError("H5 lacks Survey table")
    survey = h5.root.Survey
    required = {"exptime", "millum", "throughput", "offset", "exp"}
    if not required.issubset(survey.colnames):
        raise ValueError("Survey lacks Raw->Fibers calibration columns")
    survey_exp = np.asarray(survey.cols.exp[:], dtype=int)
    selected = np.flatnonzero(survey_exp == int(exposure))
    if selected.size != 1:
        raise ValueError("Survey must contain one row for exposure %d" % exposure)
    row = survey[selected[0]]
    exptime = float(row["exptime"])
    millum = float(row["millum"])
    guider_throughput = float(row["throughput"])
    survey_offset = float(row["offset"])
    if not np.isfinite(exptime):
        raise ValueError("Survey.exptime is invalid")
    fac = 360.0 / exptime if exptime != 0.0 else 1.0
    if not np.isfinite(millum) or not np.isfinite(guider_throughput):
        raise ValueError("Survey illumination/transparency is invalid")
    gratio = millum * guider_throughput / 5e5
    if not np.isfinite(gratio) or gratio == 0.0:
        raise ValueError("Survey guider ratio is invalid")

    throughput_path = Path(DIRNAME) / "CALS" / "throughput.txt"
    table = Table.read(throughput_path, format="ascii.fixed_width_two_line")
    standard_wave = np.asarray(table["wavelength"], dtype=float)
    standard_throughput = np.asarray(table["throughput"], dtype=float)
    if (standard_wave.size != len(def_wave) or
            not np.allclose(standard_wave, def_wave, rtol=0.0, atol=1e-6)):
        raise ValueError("CALS/throughput.txt wavelength grid mismatch")
    if np.any(~np.isfinite(standard_throughput) |
              (standard_throughput == 0.0)):
        raise ValueError("CALS/throughput.txt contains invalid response")

    # These expressions mirror quick_reduction.py.  A zero/nonfinite Survey
    # offset is treated as invalid by the current cube builder and therefore
    # uses the no-offset branch here as well.
    offset = survey_offset if np.isfinite(survey_offset) and \
        survey_offset != 0.0 else 1.0
    mult_fac = (6.626e-27 * (3e18 / def_wave) / 360.0 /
                5e5 / 0.92 * 5)
    mult_fac *= 1e29 * def_wave**2 / 2.99792e18
    final_norm = 1e-29 * 2.99792e18 / def_wave**2 * 1e17
    basis = (mult_fac * fac / standard_throughput / gratio *
             final_norm * offset)
    if np.any(~np.isfinite(basis)):
        raise ValueError("Raw->Fibers transfer is nonfinite")
    metadata = {
        "exptime": exptime, "millum": millum,
        "guider_throughput": guider_throughput, "offset": survey_offset,
        "gratio": gratio,
    }
    return basis, metadata


def correct_m101_amplifier_edge_background(h5, fiber_spectrum, ra, dec,
                                            original_image_data,
                                            original_image_wcs, h5file, log):
    """Apply the measured M101 amplifier-edge additive background in memory.

    M101 has a small additive detector-coordinate residual near the
    readout-side edge of each VIRUS amplifier.  It is approximately constant
    with wavelength in Raw e-/A units, has a repeatable folded spatial profile,
    and varies in amplitude by exposure.  This helper measures one profile per
    exposure and amplifier type from M101-safe blank-sky fibers, smooths it in
    amplifier coordinate, propagates it through the known Quick Reduction
    calibration, and subtracts it from Fibers.spectrum before the existing
    M101 residual-sky correction.  It is not an FTF correction and does not
    claim to identify the underlying detector/scattered-light cause.
    """
    nrows, n_fiber_wave = fiber_spectrum.shape
    if n_fiber_wave != len(def_wave):
        raise ValueError("Fibers.spectrum has unexpected wavelength length")
    raw = h5.root.Raw
    if not {"spectrum", "wave"}.issubset(raw.colnames):
        raise ValueError("H5 Raw table lacks spectrum/wave")
    raw_spectrum = np.asarray(raw.cols.spectrum[:], dtype=float)
    raw_wave = np.asarray(raw.cols.wave[:], dtype=float)
    if raw_spectrum.shape[0] != nrows or raw_wave.shape != raw_spectrum.shape:
        raise ValueError("Raw/Fibers row or wavelength shape mismatch")
    row_exposure, row_j, row_amp, groups = _m101_amplifier_groups(h5.root.Info)
    if len(ra) != nrows or len(dec) != nrows:
        raise ValueError("Info/Fibers row mismatch for amplifier correction")

    corrected = np.asarray(fiber_spectrum, dtype=float).copy()
    qa_rows = []
    profile_states = []
    for exposure in range(1, M101_SKY_NEXPOSURES + 1):
        selected = _m101_amp_blank_selection(
            ra, dec, row_exposure, exposure, fiber_spectrum,
            original_image_data, original_image_wcs)
        selected_indices = np.flatnonzero(selected)
        common_raw_wave = None
        raw_rectified = None
        if selected_indices.size:
            wave_subset = raw_wave[selected_indices]
            common_raw_wave = np.nanmedian(
                wave_subset[:min(256, len(wave_subset))], axis=0)
            valid_common = np.isfinite(common_raw_wave)
            common_raw_wave = common_raw_wave[valid_common]
            if common_raw_wave.size >= 2:
                raw_rectified = np.full(
                    (selected_indices.size, common_raw_wave.size), np.nan,
                    dtype=float)
                for index, row in enumerate(selected_indices):
                    valid = (np.isfinite(raw_wave[row]) &
                             np.isfinite(raw_spectrum[row]))
                    if valid.sum() >= 2:
                        raw_rectified[index] = np.interp(
                            common_raw_wave, raw_wave[row, valid],
                            raw_spectrum[row, valid], left=np.nan, right=np.nan)
        try:
            basis, calibration = _m101_amp_raw_to_fibers_basis(h5, exposure)
            basis_error = None
        except Exception as error:
            basis = None
            calibration = {"exptime": np.nan, "millum": np.nan,
                           "guider_throughput": np.nan,
                           "offset": np.nan, "gratio": np.nan}
            basis_error = error
            log.warning("M101 amplifier background %s exposure %d: %s; "
                        "correction skipped for this exposure.",
                        h5file, exposure, error)

        for amp in ("LL", "LU", "RL", "RU"):
            amp_selected = selected & (row_amp == amp)
            n_blank = int(amp_selected.sum())
            physical_ifus = set()
            for group in groups:
                if group["exposure"] != exposure or group["amp"] != amp:
                    continue
                if np.any(selected[group["indices"]]):
                    physical_ifus.add((group["ifuslot"], group["ifuid"]))
            n_ifus = len(physical_ifus)
            raw_profile = np.full(M101_AMP_FIBERS, np.nan, dtype=float)
            scatter = np.full(M101_AMP_FIBERS, np.nan, dtype=float)
            smooth = np.full(M101_AMP_FIBERS, np.nan, dtype=float)
            applied = np.zeros(M101_AMP_FIBERS, dtype=float)
            profile_error = None
            if n_blank == 0 or raw_rectified is None:
                profile_error = "no usable blank-sky Raw spectra"
            else:
                candidate_positions = np.flatnonzero(amp_selected[selected_indices])
                candidate_j = row_j[selected_indices][candidate_positions]
                central = candidate_positions[np.isin(candidate_j,
                                                       M101_AMP_REFERENCE_J)]
                if central.size == 0:
                    profile_error = "no blank-sky central reference fibers"
                else:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", RuntimeWarning)
                        center_spectrum = np.nanmedian(
                            raw_rectified[central], axis=0)
                    safe = ((common_raw_wave >= M101_AMP_SAFE_WAVE[0]) &
                            (common_raw_wave <= M101_AMP_SAFE_WAVE[1]))
                    if np.sum(np.isfinite(center_spectrum[safe])) < 2:
                        profile_error = "central Raw reference is nonfinite"
                    else:
                        q_values = (candidate_j if amp in ("LL", "RU")
                                     else 111 - candidate_j)
                        for q in range(M101_AMP_FIBERS):
                            q_positions = candidate_positions[q_values == q]
                            if q_positions.size == 0:
                                continue
                            with warnings.catch_warnings():
                                warnings.simplefilter("ignore", RuntimeWarning)
                                q_spectrum = np.nanmedian(
                                    raw_rectified[q_positions], axis=0)
                            difference = q_spectrum - center_spectrum
                            finite = safe & np.isfinite(difference)
                            if np.any(finite):
                                raw_profile[q] = float(
                                    np.nanmedian(difference[finite]))
                                residual = difference[finite] - raw_profile[q]
                                scatter[q] = float(np.sqrt(
                                    np.nanmedian(residual * residual)))
                        if np.isfinite(raw_profile).sum() < M101_AMP_FIBERS // 2:
                            profile_error = (
                                "mostly nonfinite A_raw profile (%d/%d q bins)" %
                                (np.isfinite(raw_profile).sum(),
                                 M101_AMP_FIBERS))
                        else:
                            smooth = _m101_amp_smooth_profile(raw_profile)
                            if amp in ("LL", "RU"):
                                reference_q = M101_AMP_REFERENCE_J
                            else:
                                reference_q = 111 - M101_AMP_REFERENCE_J
                            center_level = np.nanmedian(smooth[reference_q])
                            if not np.isfinite(center_level):
                                profile_error = "smoothed central reference is nonfinite"
                            else:
                                smooth -= center_level
                                applied = smooth.copy()
                                applied[np.arange(M101_AMP_FIBERS) >= 40] = 0.0
                                applied[~np.isfinite(applied)] = 0.0
            if profile_error is not None:
                log.warning(
                    "M101 amplifier background %s exposure %d amp %s: %s; "
                    "correction skipped.", h5file, exposure, amp,
                    profile_error)
            if basis_error is not None or profile_error is not None:
                applied[:] = 0.0
            else:
                rows = ((row_exposure == exposure) & (row_amp == amp))
                q_for_rows = (row_j[rows] if amp in ("LL", "RU")
                              else 111 - row_j[rows])
                corrected[rows] -= applied[q_for_rows, None] * basis[None, :]

            finite_edge = np.isfinite(applied[:20])
            median_edge = (float(np.nanmedian(applied[:20]))
                           if np.any(finite_edge) else np.nan)
            finite_peak = np.isfinite(applied[:40])
            peak_q = (int(np.nanargmax(applied[:40]))
                      if np.any(finite_peak) else np.nan)
            peak_value = (float(applied[peak_q])
                          if np.any(finite_peak) else np.nan)
            if log is not None:
                log.info(
                    "M101 amplifier background %s exposure %d amp %s: "
                    "blank fibers=%d, physical IFUs=%d, "
                    "median A_raw(q<20)=%.6g e-/A, peak=%.6g e-/A at q=%s, "
                    "q0-9/q10-19/q20-29/q30-39=%.6g/%.6g/%.6g/%.6g e-/A, "
                    "q<20/exptime=%.6g e-/s/A",
                    h5file, exposure, amp, n_blank, n_ifus, median_edge,
                    peak_value, peak_q,
                    np.nanmedian(applied[0:10]), np.nanmedian(applied[10:20]),
                    np.nanmedian(applied[20:30]), np.nanmedian(applied[30:40]),
                    median_edge / calibration["exptime"]
                    if np.isfinite(calibration["exptime"]) and
                    calibration["exptime"] != 0.0 else np.nan)

            state = {"h5": op.basename(h5file), "exposure": exposure,
                     "amplifier": amp, "applied": applied.copy()}
            profile_states.append(state)
            for q in range(M101_AMP_FIBERS):
                qa_rows.append({
                    "h5": op.basename(h5file), "exposure": exposure,
                    "amplifier": amp, "q": q,
                    "A_raw_unsmoothed_e_per_A": raw_profile[q],
                    "A_raw_smoothed_e_per_A": smooth[q],
                    "A_raw_applied_e_per_A": applied[q],
                    "wavelength_scatter_e_per_A": scatter[q],
                    "n_blank_fibers": n_blank,
                    "n_physical_ifus": n_ifus,
                    "exptime": calibration["exptime"],
                    "millum": calibration["millum"],
                    "guider_throughput": calibration["guider_throughput"],
                    "offset": calibration["offset"],
                })
    return corrected, qa_rows, profile_states


def _write_m101_amplifier_background_qa(path, rows):
    fields = [
        "h5", "exposure", "amplifier", "q",
        "A_raw_unsmoothed_e_per_A", "A_raw_smoothed_e_per_A",
        "A_raw_applied_e_per_A", "wavelength_scatter_e_per_A",
        "n_blank_fibers", "n_physical_ifus", "exptime", "millum",
        "guider_throughput", "offset",
    ]
    with open(path, "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_m101_amplifier_background_figure(path, states):
    if not states:
        return
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5), sharey=True)
    colors = dict(zip(("LL", "LU", "RL", "RU"),
                      ("tab:blue", "tab:orange", "tab:green", "tab:red")))
    q = np.arange(M101_AMP_FIBERS)
    for axis, amp in zip(axes, ("LL", "LU", "RL", "RU")):
        values = np.asarray([state["applied"] for state in states
                             if state["amplifier"] == amp], dtype=float)
        if values.size == 0:
            continue
        for value in values:
            axis.plot(q, value, color=colors[amp], alpha=.12, linewidth=.6)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            median = np.nanmedian(values, axis=0)
            p16, p84 = np.nanpercentile(values, [16, 84], axis=0)
        axis.fill_between(q, p16, p84, color=colors[amp], alpha=.2)
        axis.plot(q, median, color=colors[amp], linewidth=2, label="median")
        axis.axvspan(0, 19, color="k", alpha=.08)
        axis.axvline(40, color="k", linestyle=":", linewidth=.8)
        axis.set_title(amp)
        axis.set_xlabel("folded readout distance q")
        axis.grid(alpha=.2)
    axes[0].set_ylabel("A_raw(q) [e-/A]")
    axes[0].legend(fontsize=8)
    fig.suptitle("M101 amplifier-edge additive background profiles")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def get_script_path():
    return op.dirname(op.realpath(sys.argv[0]))

warnings.filterwarnings("ignore")

DIRNAME = get_script_path()


# -----------------------------------------------------------------------------
# Validated M101 hierarchical calibration.
#
# The implementation below deliberately delegates the mathematical pieces to
# diagnose_m101_hierarchical.py.  The production builder therefore uses the
# same exact aperture, ADR, fixed-f(q), alpha, g/z_source_fit,
# delta_illumination, source+sky IFU scalar, plane, and robust beta estimators.
# Only compact calibration products survive PASS 1; spectra are reopened in
# PASS 2.  No calibration correction is applied to the historical cube
# estimator after specarray/errarray are populated.
# -----------------------------------------------------------------------------

M101_SECONDARY_H5 = {
    '20200622_0000015.h5', '20200625_0000017.h5',
    '20200710_0000013.h5', '20200710_0000014.h5',
}


def _m101_production_survey_by_exp(h5):
    if 'Survey' not in h5.root._v_children:
        raise ValueError('M101 hierarchical calibration requires Survey')
    survey_by_exp = {}
    for row in h5.root.Survey:
        exposure = int(row['exp'])
        if exposure in survey_by_exp:
            raise ValueError('Survey has duplicate exposure %d' % exposure)
        survey_by_exp[exposure] = {name: row[name] for name in h5.root.Survey.colnames}
    if set(survey_by_exp) != {1, 2, 3}:
        raise ValueError('Survey must contain exactly exposures 1, 2, 3')
    return survey_by_exp


def _m101_temporary_matched_images(image, virus_fwhm, band):
    """Match one fixed external image for one exposure, without caching it."""
    image_fwhm = float(image['psf']['fwhm_arcsec'])
    if virus_fwhm <= image_fwhm:
        return image['data'], image['object_data']
    kernel_fwhm = np.sqrt(virus_fwhm ** 2 - image_fwhm ** 2)
    sigma_pix = (kernel_fwhm * validated_m101.FWHM_TO_SIGMA /
                 image['pixel_scale_arcsec'])
    kernel = Gaussian2DKernel(sigma_pix)
    matched_raw = convolve(image['data'], kernel, boundary='extend',
                           nan_treatment='interpolate', preserve_nan=True)
    matched_object = convolve(image['object_data'], kernel, boundary='extend',
                              nan_treatment='interpolate', preserve_nan=True)
    return matched_raw, matched_object


def _m101_load_state_template(path, log):
    required = {'SPECID', 'IFUSLOT', 'IFUID', 'C_state1', 'C_state2'}
    with Path(path).open(newline='') as stream:
        reader = csv.DictReader(stream)
        if not required.issubset(reader.fieldnames or ()):
            raise ValueError('state template lacks required columns: %s' %
                             sorted(required - set(reader.fieldnames or ())))
        primary, secondary = {}, {}
        for row in reader:
            key = (int(row['SPECID']), int(row['IFUSLOT']), int(row['IFUID']))
            c1 = float(row['C_state1']) if row['C_state1'].strip() else np.nan
            c2 = float(row['C_state2']) if row['C_state2'].strip() else np.nan
            if np.isfinite(c1): primary[key] = c1
            if np.isfinite(c2): secondary[key] = c2
    log.info('M101 state template mapping verified from explicit C_state1/C_state2 '
             'columns: primary finite=%d secondary finite=%d', len(primary), len(secondary))
    return {1: primary, 2: secondary}


def _m101_fit_production_response(rows, groups, good_groups, planes,
                                  state_template, production_state, h5file, log):
    """Fit beta only and make the compact physical-IFU response mapping."""
    template = state_template[production_state]
    selected = [row for row in rows if row.get('well_constrained_common') and
                np.isfinite(row.get('s_common_normalized', np.nan)) and
                row['ifu_key'] in template and np.isfinite(template[row['ifu_key']])]
    x = np.asarray([template[row['ifu_key']] for row in selected], dtype=float)
    y = np.asarray([row['plane_residual'] for row in selected], dtype=float)
    beta_fit = validated_m101.robust_zero_slope(x, y)
    beta = beta_fit['slope']
    if not np.isfinite(beta):
        raise ValueError('%s state template beta is invalid' % h5file)
    response = {}
    response_details = {}
    missing = set()
    for group in groups:
        if group['exposure'] not in planes:
            continue
        plane = planes[group['exposure']]
        ra0, dec0 = plane['ra0'], plane['dec0']
        if not np.isfinite(ra0) or not np.isfinite(dec0):
            raise ValueError('%s exposure %d has invalid illumination plane center' %
                             (h5file, group['exposure']))
        x_arcmin = ((group['mean_RA'] - ra0) * np.cos(np.deg2rad(dec0)) * 60.0)
        y_arcmin = (group['mean_Dec'] - dec0) * 60.0
        s_plane = 1.0 + plane['cx'] * x_arcmin + plane['cy'] * y_arcmin
        key = (group['specid'], group['ifuslot'], group['ifuid'])
        c = template.get(key, 0.0)
        if key not in template:
            missing.add(key)
        s_response = s_plane + beta * c
        if not np.isfinite(s_response) or s_response <= 0.0:
            raise ValueError('%s exposure %d IFU %s has nonpositive response %.8g' %
                             (h5file, group['exposure'], key, s_response))
        response[(group['exposure'],) + key] = float(s_response)
        response_details[(group['exposure'],) + key] = {
            's_plane': float(s_plane), 'C_state': float(c),
            'beta_times_C': float(beta * c), 's_response': float(s_response),
            'template_present': key in template}
    if missing:
        log.warning('%s state %d template missing %d physical IFUs; using C=0: %s',
                    h5file, production_state, len(missing), sorted(missing))
    record_by_exposure = {}
    for exposure in sorted(planes):
        exposure_rows = [row for row in rows if row['exposure'] == exposure]
        exposure_selected = [row for row in selected if row['exposure'] == exposure]
        before = np.asarray([row['s_common_normalized'] - 1.0 for row in exposure_rows
                             if np.isfinite(row['s_common_normalized'])])
        after_plane = np.asarray([row['plane_residual'] for row in exposure_rows
                                  if np.isfinite(row['plane_residual'])])
        after_template = np.asarray([row['plane_residual'] - beta * template[row['ifu_key']]
                                     for row in exposure_selected])
        local_values = np.asarray([response[(exposure, row['SPECID'], row['IFUSLOT'], row['IFUID'])]
                                   for row in exposure_rows
                                   if (exposure, row['SPECID'], row['IFUSLOT'], row['IFUID']) in response])
        record_by_exposure[exposure] = {
            'exposure': exposure, 'production_state': production_state,
            'beta': beta, 'n_template_IFUs': len(exposure_selected),
            'n_well_constrained_IFUs': sum(row['exposure'] == exposure and
                                          row.get('well_constrained_common', False)
                                          for row in rows),
            'n_good_physical_amps': sum(group['exposure'] == exposure and good_groups[i]
                                       for i, group in enumerate(groups)),
            'RMS_scalar_before_plane': validated_m101.robust_rms(before),
            'RMS_scalar_after_plane': validated_m101.robust_rms(after_plane),
            'RMS_scalar_after_template': validated_m101.robust_rms(after_template),
            'response_min': np.min(local_values) if local_values.size else np.nan,
            'response_p16': np.percentile(local_values, 16) if local_values.size else np.nan,
            'response_median': np.median(local_values) if local_values.size else np.nan,
            'response_p84': np.percentile(local_values, 84) if local_values.size else np.nan,
            'response_max': np.max(local_values) if local_values.size else np.nan,
        }
        log.info('%s exposure %d state=%d beta=%+.8g response min/p16/med/p84/max='
                 '%.8g/%.8g/%.8g/%.8g/%.8g', h5file, exposure, production_state,
                 beta, *[record_by_exposure[exposure][name]
                          for name in ('response_min', 'response_p16', 'response_median',
                                       'response_p84', 'response_max')])
    return response, response_details, record_by_exposure


def _m101_calibrate_one_h5(h5file, images, filters, f, iterations,
                           state_template, production_state, log):
    """PASS 1: reproduce the validated single-H5 hierarchy and discard spectra."""
    groups = []
    datasets = []
    with tables.open_file(h5file, mode='r') as h5:
        info, fibers = h5.root.Info, h5.root.Fibers
        groups, labels = validated_m101.build_groups(info)
        ra = np.asarray(info.cols.ra[:], dtype=float)
        dec = np.asarray(info.cols.dec[:], dtype=float)
        for group in groups:
            group['mean_RA'] = float(np.nanmean(ra[group['indices']]))
            group['mean_Dec'] = float(np.nanmean(dec[group['indices']]))
        if 'skyspectrum' not in fibers.colnames:
            raise ValueError('%s Fibers lacks skyspectrum' % h5file)
        spectra = np.asarray(fibers.cols.spectrum[:], dtype=float)
        skyspectra = np.asarray(fibers.cols.skyspectrum[:], dtype=float)
        survey_by_exp = _m101_production_survey_by_exp(h5)
        ifuslot = np.asarray(info.cols.ifuslot[:])
        amp = np.asarray([validated_m101.as_text(value) for value in info.cols.amp[:]])
        bad = validated_m101.masked_rows(h5file, ifuslot, amp)
        row_q = np.full(int(info.nrows), -1, dtype=int)
        for group in groups:
            j = np.arange(validated_m101.N_FIBER_AMP)
            row_q[group['indices']] = j if group['amp'] in ('LL', 'RU') else 111 - j
        for exposure in (1, 2, 3):
            survey_row = survey_by_exp[exposure]
            offset = float(survey_row['offset'])
            if not np.isfinite(offset) or offset == 0.0:
                raise ValueError('%s exposure %d Survey.offset is invalid: %s' %
                                 (h5file, exposure, offset))
            working = spectra / offset
            exposure_rows = labels == exposure
            for band in ('ON', 'OFF'):
                response = filters[band]
                V = validated_m101.synthetic_mean(working, response)
                B_sky = validated_m101.synthetic_mean(skyspectra, response)
                eff_ra, eff_dec = validated_m101.adr_positions(ra, dec, survey_row, response)
                images[band]['_production_exposure'] = exposure
                matched_raw, matched_object = _m101_temporary_matched_images(
                    images[band], float(survey_row['fwhm']), band)
                raw_I, _ = validated_m101.sample_image_exact(images[band], matched_raw,
                                                               eff_ra, eff_dec)
                I, image_valid = validated_m101.sample_image_exact(images[band], matched_object,
                                                                    eff_ra, eff_dec)
                K = validated_m101.weighted_scalar(
                    validated_m101.raw_work_basis(survey_row), response)
                valid = (exposure_rows & ~bad & np.isfinite(V) & image_valid & np.isfinite(I))
                datasets.append({'exposure': exposure, 'band': band, 'V': V, 'I': I,
                                 'B_sky': B_sky, 'V_total': V + B_sky,
                                 'ra': eff_ra, 'dec': eff_dec, 'K': K, 'q': row_q,
                                 'valid': valid})

    initial_good = np.asarray([not np.any(bad[group['indices']]) for group in groups], dtype=bool)
    alpha = {(exposure, band): validated_m101.ALPHA_INITIAL
             for exposure in (1, 2, 3) for band in validated_m101.AMPS}
    good_groups = initial_good.copy()
    for _ in range(iterations):
        globals_, _, _, good_groups = validated_m101.broad_stage(
            datasets, groups, f, alpha, initial_good, good_groups)
        alpha = validated_m101.fit_bounded_alphas(
            datasets, groups, globals_, f, good_groups, alpha, (1, 2, 3))
    globals_, _, _, good_groups = validated_m101.broad_stage(
        datasets, groups, f, alpha, initial_good, good_groups)
    for dataset in datasets:
        dataset['V_total_corrected'] = dataset['V0'] + dataset['B_sky']
    delta_by_band = {}
    global_rows = []
    for exposure in (1, 2, 3):
        for band in ('ON', 'OFF'):
            row = validated_m101.solve_illumination_delta(
                datasets, groups, globals_, alpha, f, good_groups, (1, 2, 3), exposure, band)
            delta_by_band[(exposure, band)] = row['delta_illumination']
            global_rows.append(row)
    illumination_rows, leverage_qa = validated_m101.build_illumination_scalars(
        datasets, groups, globals_, alpha, f, good_groups, (1, 2, 3), delta_by_band)
    validated_m101.normalize_illumination_scalars(illumination_rows, (1, 2, 3))
    planes = {exposure: validated_m101.fit_illumination_plane(illumination_rows, exposure)
              for exposure in (1, 2, 3)}
    for row in illumination_rows:
        plane = planes[row['exposure']]
        x_arcmin = ((row['mean_RA'] - plane['ra0']) * np.cos(np.deg2rad(plane['dec0'])) * 60.0)
        y_arcmin = (row['mean_Dec'] - plane['dec0']) * 60.0
        row['plane_model'] = 1.0 + plane['cx'] * x_arcmin + plane['cy'] * y_arcmin
        row['plane_residual'] = row['s_common_normalized'] - row['plane_model']
        row['ifu_key'] = (int(row['SPECID']), int(row['IFUSLOT']), int(row['IFUID']))
    response, response_details, exposure_records = _m101_fit_production_response(
        illumination_rows, groups, good_groups, planes, state_template,
        production_state, Path(h5file).name, log)
    return {'h5': Path(h5file).name, 'alpha': alpha, 'globals': globals_,
            'delta': delta_by_band, 'global_rows': global_rows,
            'planes': planes, 'illumination_rows': illumination_rows,
            'leverage': leverage_qa, 'groups': groups, 'response': response,
            'response_details': response_details,
            'survey_by_exp': survey_by_exp,
            'exposure_records': exposure_records,
            'production_state': production_state, 'labels': labels}


def _m101_derive_gray_factors(calibrations, log):
    """Replace historical norm_array with ensemble log-g normalization."""
    logs = {band: [] for band in ('ON', 'OFF')}
    for calibration in calibrations:
        for band in logs:
            for exposure in (1, 2, 3):
                g = calibration['globals'][(exposure, band)]['g']
                if np.isfinite(g) and g > 0.0:
                    logs[band].append(np.log(g))
    reference = {band: validated_m101.robust_location(values)
                 for band, values in logs.items()}
    if not all(np.isfinite(value) for value in reference.values()):
        raise ValueError('cannot derive ensemble gray normalization: invalid g population')
    raw = []
    for calibration in calibrations:
        per_exposure = {}
        for exposure in (1, 2, 3):
            deviations = {}
            for band in ('ON', 'OFF'):
                g = calibration['globals'][(exposure, band)]['g']
                deviations[band] = (np.log(g) - reference[band]
                                    if np.isfinite(g) and g > 0.0 else np.nan)
            valid = [value for value in deviations.values() if np.isfinite(value)]
            if not valid:
                raise ValueError('%s exposure %d has no valid positive g for gray factor' %
                                 (calibration['h5'], exposure))
            if len(valid) == 1:
                log.warning('%s exposure %d gray factor uses only one valid band',
                            calibration['h5'], exposure)
            d = float(np.mean(valid))
            per_exposure[exposure] = {
                'gray_ON_relative': np.exp(-deviations['ON']) if np.isfinite(deviations['ON']) else np.nan,
                'gray_OFF_relative': np.exp(-deviations['OFF']) if np.isfinite(deviations['OFF']) else np.nan,
                'gray_combined_raw': np.exp(-d),
                'log_gray_ON_minus_OFF': (deviations['ON'] - deviations['OFF']
                                          if all(np.isfinite(deviations[b]) for b in ('ON', 'OFF')) else np.nan),
                'n_valid_bands_for_gray': len(valid)}
            if (np.isfinite(per_exposure[exposure]['gray_ON_relative']) and
                    np.isfinite(per_exposure[exposure]['gray_OFF_relative']) and
                    per_exposure[exposure]['gray_OFF_relative'] != 0.0):
                per_exposure[exposure]['gray_ON_minus_OFF_fractional'] = (
                    per_exposure[exposure]['gray_ON_relative'] /
                    per_exposure[exposure]['gray_OFF_relative'] - 1.0)
            else:
                per_exposure[exposure]['gray_ON_minus_OFF_fractional'] = np.nan
            raw.append(per_exposure[exposure]['gray_combined_raw'])
        calibration['gray_by_exposure'] = per_exposure
    center = validated_m101.robust_location(np.log(np.asarray(raw, dtype=float)))
    if not np.isfinite(center):
        raise ValueError('cannot center ensemble gray normalization')
    for calibration in calibrations:
        for exposure in (1, 2, 3):
            calibration['gray_by_exposure'][exposure]['gray_combined'] = (
                calibration['gray_by_exposure'][exposure]['gray_combined_raw'] / np.exp(center))
    log.info('M101 gray normalization: log center=%+.8g; positive g values ON=%d OFF=%d',
             center, len(logs['ON']), len(logs['OFF']))
    return reference


def _m101_write_calibration_qa(calibrations, images, output_name, log):
    calibration_rows = []
    ifu_rows = []
    for calibration in calibrations:
        h5 = calibration['h5']
        for exposure in (1, 2, 3):
            survey = calibration['survey_by_exp'][exposure]
            record = calibration['exposure_records'][exposure]
            plane = calibration['planes'][exposure]
            gray = calibration['gray_by_exposure'][exposure]
            row = {'H5': h5, 'exposure': exposure,
                   'production_state': calibration['production_state'],
                   'Survey.offset': float(survey['offset']),
                   'Survey.fwhm': float(survey['fwhm']),
                   'plane_cx': plane['cx'], 'plane_cy': plane['cy'],
                   'beta': record['beta'],
                   'gray_ON_relative': gray['gray_ON_relative'],
                   'gray_OFF_relative': gray['gray_OFF_relative'],
                   'gray_combined': gray['gray_combined'],
                   'log_gray_ON_minus_OFF': gray['log_gray_ON_minus_OFF'],
                   'gray_ON_minus_OFF_fractional': gray['gray_ON_minus_OFF_fractional'],
                   'n_good_physical_amps': record['n_good_physical_amps'],
                   'n_well_constrained_IFUs': record['n_well_constrained_IFUs'],
                   'n_template_IFUs': record['n_template_IFUs'],
                   'RMS_scalar_before_plane': record['RMS_scalar_before_plane'],
                   'RMS_scalar_after_plane': record['RMS_scalar_after_plane'],
                   'RMS_scalar_after_template': record['RMS_scalar_after_template'],
                   'response_min': record['response_min'], 'response_p16': record['response_p16'],
                   'response_median': record['response_median'], 'response_p84': record['response_p84'],
                   'response_max': record['response_max'],
                   'external_FWHM_ON': images['ON']['psf']['fwhm_arcsec'],
                   'external_FWHM_OFF': images['OFF']['psf']['fwhm_arcsec'],
                   'external_background_ON': images['ON']['background'],
                   'external_background_OFF': images['OFF']['background']}
            for band in ('ON', 'OFF'):
                fit = calibration['globals'][(exposure, band)]
                delta = calibration['delta'][(exposure, band)]
                row['g_%s' % band] = fit['g']
                row['z_source_fit_%s' % band] = fit['z']
                row['delta_illumination_%s' % band] = delta
            for amp in validated_m101.AMPS:
                row['alpha_%s' % amp] = calibration['alpha'][(exposure, amp)]
            calibration_rows.append(row)

            illumination = {tuple((int(r['SPECID']), int(r['IFUSLOT']), int(r['IFUID']))): r
                            for r in calibration['illumination_rows']
                            if int(r['exposure']) == exposure}
            identities = sorted({(group['specid'], group['ifuslot'], group['ifuid'])
                                 for group in calibration['groups'] if group['exposure'] == exposure},
                                key=lambda key: (key[1], key[0], key[2]))
            for key in identities:
                details = calibration['response_details'][(exposure,) + key]
                measured = illumination.get(key, {})
                c = details['C_state']
                if not details['template_present']:
                    c = 0.0
                ifu_rows.append({'H5': h5, 'exposure': exposure,
                                 'production_state': calibration['production_state'],
                                 'SPECID': key[0], 'IFUSLOT': key[1], 'IFUID': key[2],
                                 's_common_normalized': measured.get('s_common_normalized', np.nan),
                                 'well_constrained_common': measured.get('well_constrained_common', False),
                                 's_plane': details['s_plane'], 'C_state': c,
                                 'beta': record['beta'], 'beta_times_C': details['beta_times_C'],
                                 's_response': details['s_response'],
                                 'template_present': details['template_present']})
    calibration_fields = ['H5', 'exposure', 'production_state', 'Survey.offset', 'Survey.fwhm']
    calibration_fields += ['alpha_%s' % amp for amp in validated_m101.AMPS]
    calibration_fields += ['g_ON', 'z_source_fit_ON', 'delta_illumination_ON',
                           'g_OFF', 'z_source_fit_OFF', 'delta_illumination_OFF',
                           'plane_cx', 'plane_cy', 'beta', 'gray_ON_relative',
                           'gray_OFF_relative', 'gray_combined', 'log_gray_ON_minus_OFF',
                           'gray_ON_minus_OFF_fractional',
                           'n_good_physical_amps', 'n_well_constrained_IFUs', 'n_template_IFUs',
                           'RMS_scalar_before_plane', 'RMS_scalar_after_plane',
                           'RMS_scalar_after_template', 'response_min', 'response_p16',
                           'response_median', 'response_p84', 'response_max',
                           'external_FWHM_ON', 'external_FWHM_OFF',
                           'external_background_ON', 'external_background_OFF']
    ifu_fields = ['H5', 'exposure', 'production_state', 'SPECID', 'IFUSLOT', 'IFUID',
                  's_common_normalized', 'well_constrained_common', 's_plane', 'C_state',
                  'beta', 'beta_times_C', 's_response', 'template_present']
    with Path(output_name).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=calibration_fields)
        writer.writeheader(); writer.writerows(calibration_rows)
    with Path(output_name).with_name('m101_production_ifu_response.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=ifu_fields)
        writer.writeheader(); writer.writerows(ifu_rows)
    _m101_plot_calibration_summary(calibration_rows, output_name, log)
    _m101_plot_response_maps(calibrations, output_name, log)
    return calibration_rows, ifu_rows


def _m101_plot_calibration_summary(rows, output_name, log):
    rows = sorted(rows, key=lambda row: (row['H5'], row['exposure']))
    x = np.arange(len(rows))
    fig, axes = plt.subplots(4, 1, figsize=(15, 12), sharex=True)
    axes[0].plot(x, [row['beta'] for row in rows], 'o-', ms=2, label='beta')
    axes[0].plot(x, [row['gray_combined'] for row in rows], 'o-', ms=2, label='gray combined')
    axes[0].set_ylabel('factor'); axes[0].legend(fontsize=8)
    axes[1].plot(x, [row['log_gray_ON_minus_OFF'] for row in rows], 'o-', ms=2)
    axes[1].axhline(0, color='k', lw=.7); axes[1].set_ylabel('log gray ON-OFF')
    axes[2].plot(x, [row['RMS_scalar_before_plane'] for row in rows], 'o-', ms=2, label='before plane')
    axes[2].plot(x, [row['RMS_scalar_after_plane'] for row in rows], 'o-', ms=2, label='after plane')
    axes[2].plot(x, [row['RMS_scalar_after_template'] for row in rows], 'o-', ms=2, label='after beta*C')
    axes[2].set_ylabel('scalar RMS'); axes[2].legend(fontsize=8)
    for name in ('response_min', 'response_median', 'response_max'):
        axes[3].plot(x, [row[name] for row in rows], 'o-', ms=2, label=name)
    axes[3].set_ylabel('s_response'); axes[3].set_xlabel('chronological H5/exposure'); axes[3].legend(fontsize=8)
    boundaries = [i for i in range(1, len(rows)) if rows[i]['H5'] != rows[i - 1]['H5']]
    for axis in axes:
        for boundary in boundaries: axis.axvline(boundary - .5, color='k', lw=.7)
        axis.grid(alpha=.2)
    fig.suptitle('M101 production hierarchical calibration')
    fig.tight_layout(rect=(0, 0, 1, .95))
    fig.savefig(Path(output_name).with_name('m101_production_calibration_summary.png'), dpi=170)
    plt.close(fig)


def _m101_plot_response_maps(calibrations, output_name, log):
    selected_names = []
    for calibration in calibrations:
        if calibration['h5'] not in M101_SECONDARY_H5 and not selected_names:
            selected_names.append(calibration['h5'])
    selected_names += [name for name in ('20200622_0000015.h5', '20200710_0000013.h5',
                                         '20200710_0000014.h5') if name not in selected_names]
    selected = [calibration for calibration in calibrations if calibration['h5'] in selected_names]
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), squeeze=False)
    for axis, calibration in zip(axes.flat, selected[:4]):
        exposure = 1
        values = []; ra = []; dec = []
        seen = set()
        for group in calibration['groups']:
            if group['exposure'] != exposure:
                continue
            key = (group['specid'], group['ifuslot'], group['ifuid'])
            if key in seen:
                continue
            seen.add(key); details = calibration['response_details'][(exposure,) + key]
            values.append(details['s_response'] - 1.0); ra.append(group['mean_RA']); dec.append(group['mean_Dec'])
        scale = max(np.percentile(np.abs(values), 95) if values else .01, .01)
        axis.scatter(ra, dec, c=values, cmap='coolwarm', vmin=-scale, vmax=scale, s=25)
        axis.set_title('%s e1 state=%d' % (calibration['h5'], calibration['production_state']), fontsize=9)
        axis.set_xlabel('RA'); axis.set_ylabel('Dec'); axis.grid(alpha=.2)
    for axis in axes.flat[len(selected[:4]):]: axis.set_visible(False)
    fig.suptitle('M101 production s_response map QA (exposure 1)')
    fig.tight_layout(rect=(0, 0, 1, .95))
    fig.savefig(Path(output_name).with_name('m101_production_response_maps.png'), dpi=170)
    plt.close(fig)


def _m101_apply_h5_calibration(h5file, calibration, f_template, binimage,
                               xg, yg, tp, log):
    """PASS 2: apply detector, local response, residual sky, and gray scale."""
    with tables.open_file(h5file, mode='r') as h5:
        info, fibers = h5.root.Info, h5.root.Fibers
        groups, labels = validated_m101.build_groups(info)
        ra = np.asarray(info.cols.ra[:], dtype=float)
        dec = np.asarray(info.cols.dec[:], dtype=float)
        if 'skyspectrum' not in fibers.colnames:
            raise ValueError('%s Fibers lacks skyspectrum' % h5file)
        source = np.asarray(fibers.cols.spectrum[:], dtype=float)
        error_source = np.asarray(fibers.cols.error[:], dtype=float)
        sky = np.asarray(fibers.cols.skyspectrum[:], dtype=float)
        survey_by_exp = _m101_production_survey_by_exp(h5)
        ifuslot = np.asarray(info.cols.ifuslot[:])
        amp = np.asarray([validated_m101.as_text(value) for value in info.cols.amp[:]])
        bad = validated_m101.masked_rows(h5file, ifuslot, amp)
        spectra = np.full(source.shape, np.nan, dtype=float)
        errors = np.full(error_source.shape, np.nan, dtype=float)
        for exposure in (1, 2, 3):
            survey = survey_by_exp[exposure]
            offset = float(survey['offset'])
            if not np.isfinite(offset) or offset == 0.0:
                raise ValueError('%s exposure %d Survey.offset is invalid: %s' %
                                 (h5file, exposure, offset))
            working = source / offset
            error_work = error_source / offset
            K_work = validated_m101.raw_work_basis(survey)
            for group in groups:
                if group['exposure'] != exposure:
                    continue
                key = (exposure, group['specid'], group['ifuslot'], group['ifuid'])
                s_response = calibration['response'].get(key)
                if s_response is None:
                    raise ValueError('%s missing s_response for %s' % (h5file, key))
                indices = group['indices']
                j = np.arange(validated_m101.N_FIBER_AMP)
                q = j if group['amp'] in ('LL', 'RU') else 111 - j
                additive = (K_work[None, :] *
                            calibration['alpha'][(exposure, group['amp'])] *
                            f_template[q, None])
                # Complete validated spectral equation:
                # S_local = (S_Fibers/offset - A + B_sky)/s_response - B_sky.
                spectra[indices] = ((working[indices] - additive + sky[indices]) /
                                    s_response - sky[indices])
                errors[indices] = error_work[indices] / s_response
        spectra[bad] = np.nan
        errors[bad] = np.nan
        # binimage is retained solely for subtract_m101_residual_sky's frozen
        # blank-region criterion; it is not used for photometric normalization.
        spectra = subtract_m101_residual_sky(
            spectra, ra, dec, xg, yg, tp, binimage=binimage, log=log,
            h5file=op.basename(h5file))
        for exposure in (1, 2, 3):
            gray = calibration['gray_by_exposure'][exposure]['gray_combined']
            selected = labels == exposure
            spectra[selected] *= gray
            errors[selected] *= gray
    return spectra, errors, ra, dec, labels, survey_by_exp

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

parser.add_argument("-if", "--image_file", "--m101-on-image",
                    dest="m101_on_image",
                    help='''M101 ON external image (historical -if alias)''',
                    default=None, type=str)

parser.add_argument("-ff", "--filter_file", "--m101-on-filter",
                    dest="m101_on_filter",
                    help='''M101 ON filter (historical -ff alias)''',
                    default=None, type=str)

parser.add_argument("--m101-off-image", required=True,
                    help='''M101 OFF external image''', type=str)
parser.add_argument("--m101-off-filter", required=True,
                    help='''M101 OFF filter''', type=str)
parser.add_argument("--m101-fq-template", required=True,
                    help='''Fixed common f(q) template''', type=str)
parser.add_argument("--m101-ifu-state-template", required=True,
                    help='''Fixed physical-IFU response state template''', type=str)
parser.add_argument("--m101-calibration-only", action="store_true",
                    help='''Run one-time image setup and PASS 1 calibration, then exit''')
parser.add_argument("--m101-iterations", type=int, default=3,
                    help='''Validated M101 alpha iterations (default: 3)''')

parser.add_argument("--wave-workers",
                    help='''Number of threads used to build wavelength planes (default: 1)''',
                    default=1, type=int)

parser.add_argument("--make-lsf", action="store_true",
                    help='''Also propagate master-arc features sparsely and write empirical FWHM products''')

parser.add_argument("--no-m101-amplifier-background", action="store_true",
                    help='''Deprecated compatibility flag; the validated fixed alpha*f(q) hierarchy is always used''')

def rebin(arr, new_shape):
    """Rebin 2D array arr to shape new_shape by averaging."""
    shape = (new_shape[0], arr.shape[0] // new_shape[0],
             new_shape[1], arr.shape[1] // new_shape[1])
    return arr.reshape(shape).mean(axis=(-1,1))

def make_image_interp(Pos, y, ye, xg, yg, xgrid, ygrid, sigma, cnt_array,
                      binsize=2):
    # Loop through shots, average images before convolution
    # Build a PSF-aware fractional coverage map (0..1) using Gaussian smoothing
    G = Gaussian2DKernel(sigma)
    nshots = len(cnt_array)
    image_all = np.ma.zeros((nshots,) + xgrid.shape)
    image_all.mask = True
    coveragek = 0.0 * xgrid  # accumulate per-shot smoothed coverage, later normalized to 0..1
    # grid for accumulating inverse-variance per pixel (pre-convolution)
    ivar_grid = 0.0 * xgrid
    # area scaling applied to flux; propagate to errors as well
    area = (np.pi * 0.75**2)

    xc = np.interp(Pos[:, 0], xg, np.arange(len(xg)), left=0., right=len(xg))
    yc = np.interp(Pos[:, 1], yg, np.arange(len(yg)), left=0., right=len(yg))
    xc = np.array(np.round(xc), dtype=int)
    yc = np.array(np.round(yc), dtype=int)
    gsel = np.where((xc>1) * (xc<len(xg)-1) * (yc>1) * (yc<len(yg)-1))[0]
    for k, cnt in enumerate(cnt_array):
        # Support two formats for cnt: (start, stop) range, or explicit index array
        if isinstance(cnt, (tuple, list)) and len(cnt) == 2:
            l1 = int(cnt[0])
            l2 = int(cnt[1])
            sel_idx = gsel[(gsel >= l1) & (gsel < l2)]
        else:
            # assume numpy array of global indices
            cnt_arr = np.array(cnt, dtype=int)
            # intersect with valid selection
            sel_idx = np.intersect1d(gsel, cnt_arr, assume_unique=False)
        if sel_idx.size == 0:
            continue
        # place flux values (scaled by area)
        image_all.data[k, yc[sel_idx], xc[sel_idx]] = y[sel_idx] / area
        image_all.mask[k, yc[sel_idx], xc[sel_idx]] = False
        # accumulate inverse variance at exact sample locations (propagate scaling)
        if ye is not None:
            ye_sel = ye[sel_idx]
            valid_e = np.isfinite(ye_sel) & (ye_sel > 0)
            if np.any(valid_e):
                ivar_vals = np.zeros_like(ye_sel, dtype=float)
                ivar_vals[valid_e] = 1.0 / (ye_sel[valid_e]**2) / (area**2)
                # add to grid
                ivar_grid[yc[sel_idx], xc[sel_idx]] += ivar_vals
        # build per-shot binary sampling mask, then smooth with the same PSF kernel
        valid_flux = np.isfinite(y[sel_idx])
        if np.any(valid_flux):
            mask_shot = 0.0 * xgrid
            mask_shot[yc[sel_idx][valid_flux], xc[sel_idx][valid_flux]] = 1.0
            # Convolution of binary mask (kernel is normalized to unit sum) yields 0..1 local coverage
            mask_sm = convolve(mask_shot, G, preserve_nan=False, boundary='extend')
            # Accumulate; we'll normalize by number of shots to map to 0..1 overall
            coveragek += mask_sm
    # form the image by median stacking shots then Gaussian smoothing
    image = np.ma.median(image_all, axis=0)
    y_im = image.data * 1.
    y_im[image.mask] = np.nan
    image = convolve(y_im, G, preserve_nan=False, boundary='extend')
    # normalize coverage to 0..1; avoid division by zero
    nshots = max(1, nshots)
    coveragek = np.clip(coveragek / float(nshots), 0.0, 1.0)
    image[coveragek == 0.] = np.nan
    image[np.isnan(image)] = 0.0

    # build an error image: start from per-pixel variance from inverse-variance sum
    var0 = np.empty_like(xgrid, dtype=float)
    var0[:] = np.nan
    pos = ivar_grid > 0
    if np.any(pos):
        var0[pos] = 1.0 / ivar_grid[pos]
    # propagate variance through Gaussian smoothing: convolve with kernel^2
    kernel_sq = G.array**2
    var_sm = convolve(var0, kernel_sq, normalize_kernel=False, boundary='extend')
    var_sm[coveragek == 0.] = np.nan
    # finalize error image
    errorimage = np.sqrt(var_sm)
    errorimage[np.isnan(errorimage)] = 0.0

    return image, errorimage, coveragek


@njit(nogil=True, cache=True)
def _gaussian_splat_shot_xy(indices, xpos, ypos, fluxes, errors,
                            x_origin, y_origin, nx, ny, sigma, radius,
                            support_radius, area, flux_sum, weight_sum,
                            variance_numerator, error_weight_sum,
                            support_map):
    """Numba kernel for one shot's subpixel Gaussian deposition."""
    sigma_sq = sigma * sigma
    support_radius_sq = (2.0 * sigma) * (2.0 * sigma)
    width = 2 * radius + 1
    gx = np.empty(width, dtype=np.float32)
    gy = np.empty(width, dtype=np.float32)

    for p in range(indices.shape[0]):
        j = indices[p]
        flux = fluxes[j]
        xi = xpos[j]
        yi = ypos[j]
        if not np.isfinite(flux) or not np.isfinite(xi) or not np.isfinite(yi):
            continue

        ix_center = int(np.floor(xi - x_origin))
        iy_center = int(np.floor(yi - y_origin))

        for ox in range(-radius, radius + 1):
            dx = (x_origin + ix_center + ox) - xi
            gx[ox + radius] = np.exp(-0.5 * dx * dx / sigma_sq)
        for oy in range(-radius, radius + 1):
            dy = (y_origin + iy_center + oy) - yi
            gy[oy + radius] = np.exp(-0.5 * dy * dy / sigma_sq)

        for ox in range(-radius, radius + 1):
            px = ix_center + ox
            if px < 0 or px >= nx:
                continue
            for oy in range(-radius, radius + 1):
                py = iy_center + oy
                if py < 0 or py >= ny:
                    continue
                weight = gx[ox + radius] * gy[oy + radius]
                flux_sum[py, px] += weight * flux / area
                weight_sum[py, px] += weight
                err = errors[j]
                if np.isfinite(err) and err > 0.0:
                    variance_numerator[py, px] += weight * weight * (err / area) ** 2
                    error_weight_sum[py, px] += weight

        # Independent-shot support is based on distance to a fiber, not on
        # whether a Gaussian stencil has a mathematically nonzero tail.
        for ox in range(-support_radius, support_radius + 1):
            px = ix_center + ox
            if px < 0 or px >= nx:
                continue
            dx = (x_origin + px) - xi
            for oy in range(-support_radius, support_radius + 1):
                py = iy_center + oy
                if py < 0 or py >= ny:
                    continue
                dy = (y_origin + py) - yi
                if dx * dx + dy * dy <= support_radius_sq:
                    support_map[py, px] = 1


@njit(nogil=True, cache=True)
def _gaussian_splat_sparse_shot_xy(indices, xpos, ypos, fluxes,
                                   sample_lookup, x_origin, y_origin,
                                   nx, ny, sigma, radius, support_radius,
                                   area, flux_sum, weight_sum, support_map):
    """Splat one exposure, retaining only pixels named by ``sample_lookup``.

    The arithmetic and loop order intentionally mirror
    ``_gaussian_splat_shot_xy``.  The lookup only removes additions for output
    pixels that are not among the selected amplifier-center samples.
    """
    sigma_sq = sigma * sigma
    support_radius_sq = (2.0 * sigma) * (2.0 * sigma)
    width = 2 * radius + 1
    gx = np.empty(width, dtype=np.float32)
    gy = np.empty(width, dtype=np.float32)

    for p in range(indices.shape[0]):
        j = indices[p]
        flux = fluxes[j]
        xi = xpos[j]
        yi = ypos[j]
        if not np.isfinite(flux) or not np.isfinite(xi) or not np.isfinite(yi):
            continue

        ix_center = int(np.floor(xi - x_origin))
        iy_center = int(np.floor(yi - y_origin))

        for ox in range(-radius, radius + 1):
            dx = (x_origin + ix_center + ox) - xi
            gx[ox + radius] = np.exp(-0.5 * dx * dx / sigma_sq)
        for oy in range(-radius, radius + 1):
            dy = (y_origin + iy_center + oy) - yi
            gy[oy + radius] = np.exp(-0.5 * dy * dy / sigma_sq)

        for ox in range(-radius, radius + 1):
            px = ix_center + ox
            if px < 0 or px >= nx:
                continue
            for oy in range(-radius, radius + 1):
                py = iy_center + oy
                if py < 0 or py >= ny:
                    continue
                sample_id = sample_lookup[py, px]
                if sample_id < 0:
                    continue
                weight = gx[ox + radius] * gy[oy + radius]
                flux_sum[sample_id] += weight * flux / area
                weight_sum[sample_id] += weight

        # Independent-shot support uses the same distance criterion as the
        # full image kernel, but only for selected output pixels.
        for ox in range(-support_radius, support_radius + 1):
            px = ix_center + ox
            if px < 0 or px >= nx:
                continue
            dx = (x_origin + px) - xi
            for oy in range(-support_radius, support_radius + 1):
                py = iy_center + oy
                if py < 0 or py >= ny:
                    continue
                sample_id = sample_lookup[py, px]
                if sample_id < 0:
                    continue
                dy = (y_origin + py) - yi
                if dx * dx + dy * dy <= support_radius_sq:
                    support_map[sample_id] = 1


def _classify_error_provenance(science, errors):
    """Count error states for finite SCI fiber samples.

    The returned columns are ``N_SCI``, positive finite error, nonfinite
    error, zero error, negative error, and any remaining invalid state.  This
    is input-sample provenance only; it does not alter SCI acceptance.
    """
    science_valid = np.isfinite(science)
    error_finite = np.isfinite(errors)
    error_valid = science_valid & error_finite & (errors > 0.0)
    error_nonfinite = science_valid & ~error_finite
    error_zero = science_valid & error_finite & (errors == 0.0)
    error_negative = science_valid & error_finite & (errors < 0.0)
    error_other = science_valid & ~(
        error_valid | error_nonfinite | error_zero | error_negative)
    return np.array([
        int(np.sum(science_valid)),
        int(np.sum(error_valid)),
        int(np.sum(error_nonfinite)),
        int(np.sum(error_zero)),
        int(np.sum(error_negative)),
        int(np.sum(error_other)),
    ], dtype=np.int64)


def _compute_final_variance(shot_images, shot_variances, ncontrib):
    """Compute formal/empirical variance for the existing shot median.

    ``shot_images`` and ``shot_variances`` are indexed by shot, while
    ``ncontrib`` retains the existing meaningful-support count.  This helper
    never changes SCI or support; it only evaluates variance for pixels whose
    SCI median already has at least two supported shots.
    """
    ny, nx = ncontrib.shape
    valid_sci = ncontrib >= 2

    # Keep the support selection identical to the SCI nanmedian.  The
    # vectorized calculations below only read these arrays and cannot change
    # SCI or NCONTRIB.
    supported = np.isfinite(shot_images)
    n = supported.sum(axis=0)
    finite_variance = np.isfinite(shot_variances)
    complete = np.all(~supported | finite_variance, axis=0)
    positive_or_zero = np.all(~supported | (shot_variances >= 0.0), axis=0)

    formal = np.full((ny, nx), np.nan, dtype=np.float64)
    two_shot = (n == 2) & complete
    if np.any(two_shot):
        formal[two_shot] = (
            np.sum(np.where(supported, shot_variances, 0.0), axis=0)[two_shot] /
            4.0)

    three_or_more = (n >= 3) & complete & positive_or_zero
    if np.any(three_or_more):
        sigmas = np.sqrt(np.where(supported, shot_variances, 1.0))
        inverse_sigma = np.zeros_like(sigmas, dtype=np.float64)
        np.divide(1.0, sigmas, out=inverse_sigma, where=supported)
        inverse_sigma_sum = np.sum(inverse_sigma, axis=0)
        positive_sigma = np.all(~supported | (sigmas > 0.0), axis=0)
        formal_valid = three_or_more & positive_sigma
        formal[formal_valid] = (
            n[formal_valid] * np.pi /
            (2.0 * inverse_sigma_sum[formal_valid] ** 2))

    empirical = np.full((ny, nx), np.nan, dtype=np.float64)
    five_or_more = n >= 5
    if np.any(five_or_more):
        center = np.nanmedian(shot_images, axis=0)
        mad = np.nanmedian(np.abs(shot_images - center[None, :, :]), axis=0)
        sigma_scatter = 1.4826 * mad
        sigma_median_empirical = 1.2533 * sigma_scatter / np.sqrt(n)
        empirical[five_or_more] = sigma_median_empirical[five_or_more] ** 2

    formal_available = np.isfinite(formal)
    empirical_available = np.isfinite(empirical)
    both = formal_available & empirical_available
    variance = np.full((ny, nx), np.nan, dtype=np.float64)
    variance[formal_available & ~empirical_available] = formal[formal_available & ~empirical_available]
    variance[empirical_available & ~formal_available] = empirical[empirical_available & ~formal_available]
    variance[both] = np.maximum(formal[both], empirical[both])
    varianceimage = variance.astype(np.float32)

    dq = np.zeros((ny, nx), dtype=np.uint16)
    dq[~valid_sci] |= DQ_INSUFFICIENT_SUPPORT
    formal_used = valid_sci & formal_available & (~empirical_available | (formal >= empirical))
    empirical_used = valid_sci & both & (empirical > formal)
    empirical_only = valid_sci & ~formal_available & empirical_available
    variance_incomplete = valid_sci & ~formal_available & ~empirical_available
    dq[formal_used] |= DQ_FORMAL_VAR_USED
    dq[empirical_used] |= DQ_EMPIRICAL_VAR_USED
    dq[empirical_only] |= DQ_VAR_EMPIRICAL_ONLY
    dq[variance_incomplete] |= DQ_VAR_INCOMPLETE

    # The variance provenance states are mutually exclusive by construction.
    variance_state_count = (formal_used.astype(np.uint8) +
                            empirical_used.astype(np.uint8) +
                            empirical_only.astype(np.uint8) +
                            variance_incomplete.astype(np.uint8))
    assert np.all(variance_state_count <= 1)

    stats = {
        'valid_sci_voxels': int(np.sum(valid_sci)),
        'finite_variance_voxels': int(np.sum(valid_sci & np.isfinite(varianceimage))),
        'shot_samples': int(np.sum(supported)),
        'shot_samples_missing_variance': int(np.sum(supported & ~finite_variance)),
        'both_variances': int(np.sum(both)),
        'empirical_exceeds_formal': int(np.sum(both & (empirical > formal))),
        'median_ncontrib': float(np.median(ncontrib[valid_sci])) if np.any(valid_sci) else np.nan,
        'variance_unavailable_voxels': int(
            np.sum(valid_sci & ~np.isfinite(varianceimage))),
        'insufficient_support_voxels': int(np.sum(~valid_sci)),
        'formal_var_used_voxels': int(np.sum(formal_used)),
        'empirical_var_used_voxels': int(np.sum(empirical_used)),
        'empirical_only_voxels': int(np.sum(empirical_only)),
    }
    ratio = np.full((ny, nx), np.nan, dtype=np.float64)
    ratio[both] = np.sqrt(empirical[both] / formal[both])
    stats['median_ratio'] = (float(np.median(ratio[np.isfinite(ratio)]))
                             if np.any(np.isfinite(ratio)) else np.nan)
    return varianceimage, dq, stats


def make_image_gaussian(Pos, y, ye, xg, yg, xgrid, ygrid, sigma,
                        cnt_array):
    """Construct one wavelength plane with true subpixel Gaussian splatting.

    Each finite science fiber contributes ``flux / area`` to a Gaussian
    weighted interpolation within its shot.  Shot images are median-combined
    only where that shot has a fiber within ``2 * sigma`` of the output pixel.
    The returned ``coverage`` is the mean per-shot Gaussian support strength,
    capped at one per shot; it is a diagnostic and never divides SCI.

    The returned ``varianceimage`` is the variance of the final SCI median:
    formal shot variances are propagated through the median, and a robust
    shot-to-shot scatter estimate is used when at least five shots contribute.
    SCI and NCONTRIB are computed independently of this variance path.
    """
    nshots = len(cnt_array)
    ny, nx = xgrid.shape
    radius = 3
    support_radius = int(np.ceil(2.0 * sigma))
    area = np.pi * 0.75 ** 2

    xpos = Pos[:, 0]
    ypos = Pos[:, 1]
    if ye is None:
        errors = np.full(y.shape, np.nan, dtype=np.float32)
    else:
        errors = ye
    shot_images = np.full((nshots, ny, nx), np.nan, dtype=np.float32)
    shot_variances = np.full((nshots, ny, nx), np.nan, dtype=np.float32)
    coverage = np.zeros((ny, nx), dtype=np.float32)
    ncontrib = np.zeros((ny, nx), dtype=np.uint8)

    for k, cnt in enumerate(cnt_array):
        if isinstance(cnt, (tuple, list)) and len(cnt) == 2:
            indices = np.arange(int(cnt[0]), int(cnt[1]), dtype=np.int64)
        else:
            indices = np.asarray(cnt, dtype=np.int64)

        flux_sum = np.zeros((ny, nx), dtype=np.float32)
        weight_sum = np.zeros((ny, nx), dtype=np.float32)
        variance_numerator = np.zeros((ny, nx), dtype=np.float32)
        error_weight_sum = np.zeros((ny, nx), dtype=np.float32)
        support_map = np.zeros((ny, nx), dtype=np.uint8)
        _gaussian_splat_shot_xy(
            indices, xpos, ypos, y, errors, float(xg[0]), float(yg[0]),
            nx, ny, float(sigma), radius, support_radius, float(area),
            flux_sum, weight_sum, variance_numerator, error_weight_sum,
            support_map)

        supported = (support_map != 0) & (weight_sum > 0.0)
        if np.any(supported):
            shot_images[k, supported] = flux_sum[supported] / weight_sum[supported]
            # Formal variance is valid only when every SCI-contributing fiber
            # has a finite positive error.  The science denominator remains
            # the full science weight sum, never the error-support sum.
            complete_error_support = supported & np.isclose(
                error_weight_sum, weight_sum, rtol=1.e-5, atol=1.e-7)
            shot_variances[k, complete_error_support] = (
                variance_numerator[complete_error_support] /
                (weight_sum[complete_error_support] *
                 weight_sum[complete_error_support]))
            ncontrib[supported] += 1
            coverage += np.minimum(weight_sum, 1.0) * supported

    # The median ignores unsupported shots because those entries remain NaN.
    image = np.nanmedian(shot_images, axis=0).astype(np.float32)
    varianceimage, dq, variance_stats = _compute_final_variance(
        shot_images, shot_variances, ncontrib)
    coverage /= float(max(1, nshots))
    coverage = np.clip(coverage, 0.0, 1.0)

    final_valid = ncontrib >= 2
    image[~final_valid] = 0.0
    varianceimage[~final_valid] = 0.0
    image[~np.isfinite(image)] = 0.0
    # Keep NaN for a valid SCI pixel whose variance cannot be estimated.

    return image, varianceimage, coverage, ncontrib, dq, variance_stats

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
    weight[:] *= np.pi * 0.75**2
    return image, np.sqrt(error), weight


def _h5_text(value):
    """Return a stable human-readable value for PyTables string columns."""
    if isinstance(value, (bytes, np.bytes_)):
        return value.decode('utf-8', errors='replace').strip()
    return str(value).strip()


def _exp_text(value):
    """Normalize numeric/string exposure labels for the QA comparison."""
    if isinstance(value, (bytes, np.bytes_)):
        value = value.decode('utf-8', errors='replace')
    try:
        return '%g' % float(value)
    except (TypeError, ValueError):
        return str(value).strip()


def _verify_lsf_exposure_partition(info_exp, n_fibers, h5file, log=None,
                                   inferred_nexp=None):
    """QA-check the science grouping and diagnose unpopulated ``Info.exp``."""
    labels = [_exp_text(v) for v in np.asarray(info_exp)]
    unique_labels = sorted(set(labels))

    # quick_reduction creates Info.exp but, in the historical M101 products,
    # does not assign it when writing each fiber row.  A constant zero is
    # therefore missing metadata, not a string representation of 3 exposure
    # labels.  The exposure count is inferred from the same 448-fiber IFU
    # blocks used by the reduction, while the existing 112-row ordering
    # remains the authoritative row-level grouping.
    info_unpopulated = len(unique_labels) == 1 and unique_labels[0] in ('0', '')
    if info_unpopulated:
        if log is not None:
            if inferred_nexp is not None:
                log.warning('LSF QA %s: Info.exp is present but unpopulated '
                            '(constant %s); inferred nexp=%d from the fiber-row '
                            'structure Nfiber/(448*NIFU). The existing 112-fiber '
                            'row ordering is the exposure partition and is unchanged.',
                            op.basename(h5file), unique_labels[0], inferred_nexp)
            else:
                log.warning('LSF QA %s: Info.exp is present but unpopulated '
                            '(constant %s); no valid row-structure exposure count '
                            'was available. The existing 112-fiber row ordering '
                            'is unchanged.', op.basename(h5file), unique_labels[0])
        return None

    blocks = []
    for block_start in range(0, n_fibers, 112):
        block = labels[block_start:block_start + 112]
        blocks.append(sorted(set(block)))

    observed = [values[0] if len(values) == 1 else 'MIXED'
                for values in blocks]
    agree = (n_fibers % 112 == 0 and
             all(len(values) == 1 for values in blocks) and
             all(value == str(k % 3) for k, value in enumerate(observed)))
    agree_one_based = (n_fibers % 112 == 0 and
                       all(len(values) == 1 for values in blocks) and
                       all(value == str((k % 3) + 1)
                           for k, value in enumerate(observed)))
    agree = agree or agree_one_based
    if log is not None:
        if agree:
            log.info('LSF QA %s: Info.exp agrees with the existing 112-fiber '
                     'three-exposure partition (%s-based labels).',
                     op.basename(h5file), 'one' if agree_one_based else 'zero')
        else:
            log.warning('LSF QA %s: Info.exp does NOT agree with the existing '
                        '112-fiber exposure partition; observed block labels=%s. '
                        'Science grouping is unchanged.',
                        op.basename(h5file), observed[:12])
    return agree


def _build_lsf_amplifier_centers(h5file, h5table, log=None):
    """Build one median RA/Dec center per shot x IFUSLOT x amplifier."""
    if 'Raw' not in h5table.root._v_children:
        raise ValueError('LSF requested for %s, but the H5 file has no Raw '
                         'table containing arcspectrum/wave.' % h5file)
    info = h5table.root.Info.cols
    ra = np.asarray(info.ra[:], dtype=float)
    dec = np.asarray(info.dec[:], dtype=float)
    ifuslot = np.asarray(info.ifuslot[:])
    amp = np.asarray([_h5_text(v) for v in info.amp[:]])
    specid = np.asarray([_h5_text(v) for v in info.specid[:]])
    ifuid = np.asarray([_h5_text(v) for v in info.ifuid[:]])

    n_info = len(ra)
    n_fiber = int(h5table.root.Fibers.nrows)
    n_raw = int(h5table.root.Raw.nrows)
    if not (n_info == n_fiber == n_raw):
        raise ValueError('LSF row alignment failure for %s: Info=%d Fibers=%d Raw=%d'
                         % (h5file, n_info, n_fiber, n_raw))
    nslots = len(np.unique(ifuslot))
    inferred_nexp = (int(n_info / float(448 * nslots))
                     if nslots > 0 and n_info % (448 * nslots) == 0 else None)
    _verify_lsf_exposure_partition(info.exp[:], n_info, h5file, log=log,
                                   inferred_nexp=inferred_nexp)

    records = []
    keys = sorted(set(zip(ifuslot.tolist(), amp.tolist())),
                  key=lambda item: (int(item[0]), item[1]))
    for slot, amplifier in keys:
        selected = (ifuslot == slot) & (amp == amplifier)
        ra_values = ra[selected]
        dec_values = dec[selected]
        if not np.any(np.isfinite(ra_values) & np.isfinite(dec_values)):
            continue
        spec_values = sorted(set(specid[selected].tolist()))
        ifu_values = sorted(set(ifuid[selected].tolist()))
        identity_ok = len(spec_values) == 1 and len(ifu_values) == 1
        if not identity_ok and log is not None:
            log.warning('LSF QA %s IFUSLOT=%s AMP=%s has inconsistent '
                        'SPECID/IFUID values: %s/%s', op.basename(h5file),
                        slot, amplifier, spec_values, ifu_values)
        records.append({
            'shot': op.basename(h5file),
            'specid': spec_values[0] if spec_values else '',
            'ifuslot': int(slot),
            'ifuid': ifu_values[0] if ifu_values else '',
            'amp': amplifier,
            'ra': float(np.nanmedian(ra_values)),
            'dec': float(np.nanmedian(dec_values)),
            'identity_ok': bool(identity_ok),
        })
    return records


def _map_lsf_centers_to_spaxels(records, tp, xg, yg):
    """Map 1-based WCS coordinates to nearest zero-based output pixels."""
    sample_lookup = np.full((len(yg), len(xg)), -1, dtype=np.int32)
    unique_positions = []
    position_ids = {}
    for record in records:
        xw, yw = tp.wcs_world2pix(record['ra'], record['dec'], 1)
        valid = np.isfinite(xw) and np.isfinite(yw)
        if valid:
            # WCS returns one-based pixel coordinates.  Round in that
            # coordinate system, then explicitly convert to zero-based array
            # indices used by sample_lookup and NumPy images.
            x_one = int(np.rint(xw))
            y_one = int(np.rint(yw))
            x_index = x_one - 1
            y_index = y_one - 1
            valid = (0 <= x_index < len(xg)) and (0 <= y_index < len(yg))
        if not valid:
            record['cube_x'] = -1
            record['cube_y'] = -1
            record['sample_id'] = -1
            continue
        key = (y_index, x_index)
        if key not in position_ids:
            position_ids[key] = len(unique_positions)
            unique_positions.append(key)
            sample_lookup[key] = position_ids[key]
        record['cube_x'] = x_one
        record['cube_y'] = y_one
        record['sample_id'] = position_ids[key]
    return sample_lookup, unique_positions


def _sparse_reconstruct_one_wave(Pos, values, xg, yg, sample_lookup,
                                 shot_indices, sigma=1.8 / 2.35):
    """Return SCI-like sparse exposure median and NCONTRIB for one wavelength."""
    n_samples = int(sample_lookup.max()) + 1 if np.any(sample_lookup >= 0) else 0
    nshots = len(shot_indices)
    shot_values = np.full((nshots, n_samples), np.nan, dtype=np.float32)
    ncontrib = np.zeros(n_samples, dtype=np.uint8)
    area = np.pi * 0.75 ** 2
    for k, cnt in enumerate(shot_indices):
        if isinstance(cnt, (tuple, list)) and len(cnt) == 2:
            indices = np.arange(int(cnt[0]), int(cnt[1]), dtype=np.int64)
        else:
            indices = np.asarray(cnt, dtype=np.int64)
        flux_sum = np.zeros(n_samples, dtype=np.float32)
        weight_sum = np.zeros(n_samples, dtype=np.float32)
        support_map = np.zeros(n_samples, dtype=np.uint8)
        _gaussian_splat_sparse_shot_xy(
            indices, Pos[:, 0], Pos[:, 1], values, sample_lookup,
            float(xg[0]), float(yg[0]), len(xg), len(yg), float(sigma),
            3, int(np.ceil(2.0 * sigma)), float(area), flux_sum, weight_sum,
            support_map)
        supported = (support_map != 0) & (weight_sum > 0.0)
        if np.any(supported):
            shot_values[k, supported] = flux_sum[supported] / weight_sum[supported]
            ncontrib[supported] += 1
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        result = np.nanmedian(shot_values, axis=0).astype(np.float32)
    # Match make_image_gaussian's SCI representation exactly: unsupported
    # pixels are zero in the returned SCI plane.  The production arc profile
    # wrapper masks these entries to NaN before FWHM measurement.
    result[ncontrib < 2] = 0.0
    return result, ncontrib


def _sparse_reconstruct_arc(raarray, decarray, arc_rect, full_indices,
                            sample_lookup, xg, yg, tp, shot_indices):
    """Propagate compact normalized arc spectra through selected spaxels."""
    n_samples = int(sample_lookup.max()) + 1 if np.any(sample_lookup >= 0) else 0
    propagated = np.full((n_samples, len(full_indices)), np.nan, dtype=np.float32)
    sample_ncontrib = np.zeros((n_samples, len(full_indices)), dtype=np.uint8)
    for j, full_index in enumerate(full_indices):
        x, y = tp.wcs_world2pix(raarray[:, full_index],
                                decarray[:, full_index], 1)
        values, ncontrib = _sparse_reconstruct_one_wave(
            np.column_stack((x, y)), arc_rect[:, j], xg, yg, sample_lookup,
            shot_indices)
        values[ncontrib < 2] = np.nan
        propagated[:, j] = values
        sample_ncontrib[:, j] = ncontrib
    return propagated, sample_ncontrib


def _rectify_lsf_arcs(h5files, def_wave, lsf_def_wave,
                      lsf_wave_indices, specarray, log=None):
    """Rectify only requested arc bins, retaining the complete arc profile."""
    arc_rect = np.full((specarray.shape[0], len(lsf_def_wave)), np.nan,
                       dtype=np.float32)
    offset = 0
    for h5file in h5files:
        t = tables.open_file(h5file)
        n_info = int(t.root.Info.nrows)
        n_raw = int(t.root.Raw.nrows)
        if n_info != n_raw:
            t.close()
            raise ValueError('LSF row alignment failure for %s: Info=%d Raw=%d'
                             % (h5file, n_info, n_raw))
        native_arc = t.root.Raw.cols.arcspectrum[:]
        native_wave = t.root.Raw.cols.wave[:]
        if native_arc.shape[0] != n_info or native_wave.shape[0] != n_info:
            t.close()
            raise ValueError('LSF Raw array row alignment failure for %s' % h5file)
        for row in range(n_info):
            compact = np.interp(lsf_def_wave, native_wave[row], native_arc[row],
                                left=np.nan, right=np.nan)
            if row == 0:
                full = np.interp(def_wave, native_wave[row], native_arc[row],
                                 left=np.nan, right=np.nan)
                np.testing.assert_array_equal(compact, full[lsf_wave_indices])
                if log is not None:
                    log.info('LSF rectification validation %s: exact np.interp '
                             'compact/full-bin agreement for representative fiber.',
                             op.basename(h5file))
            arc_rect[offset + row] = compact.astype(np.float32)
        offset += n_info
        t.close()
    if offset != specarray.shape[0]:
        raise ValueError('LSF row alignment failure across files: Raw=%d SCI=%d'
                         % (offset, specarray.shape[0]))
    return arc_rect


def _normalize_lsf_arc_window(arc_rect, lsf_def_wave, reference):
    """Subtract outer-window backgrounds and normalize each fiber's line flux."""
    window = np.abs(lsf_def_wave - reference) <= ARC_WINDOW_ANGSTROM
    left = window & (lsf_def_wave <= reference - 10.0)
    right = window & (lsf_def_wave >= reference + 10.0)
    local = np.where(window)[0]
    normalized = np.full_like(arc_rect, np.nan, dtype=np.float32)
    if local.size == 0:
        return normalized, np.zeros(arc_rect.shape[0], dtype=bool), np.full(arc_rect.shape[0], np.nan)
    left_bg = np.nanmedian(arc_rect[:, left], axis=1) if np.any(left) else np.full(arc_rect.shape[0], np.nan)
    right_bg = np.nanmedian(arc_rect[:, right], axis=1) if np.any(right) else np.full(arc_rect.shape[0], np.nan)
    background = np.nanmedian(np.column_stack((left_bg, right_bg)), axis=1)
    line = arc_rect[:, local].astype(float) - background[:, None]
    signal = np.nansum(line, axis=1) * 2.0
    valid = (np.isfinite(left_bg) & np.isfinite(right_bg) &
             np.isfinite(background) & np.isfinite(signal) & (signal > 0.0))
    for row in np.where(valid)[0]:
        normalized[row, local] = ((arc_rect[row, local].astype(float) - background[row]) /
                                  signal[row]).astype(np.float32)
    return normalized, valid, signal


def _native_arc_clipping_diagnostic(arc_rect, lsf_def_wave, reference):
    """Flag flat-topped native arc profiles without fitting a saturation model."""
    window = np.abs(lsf_def_wave - reference) <= ARC_WINDOW_ANGSTROM
    local = np.where(window)[0]
    if local.size == 0:
        return {'finite': 0, 'flat_top': 0, 'exact_plateau': 0,
                'suspicious': 0}
    left = window & (lsf_def_wave <= reference - 10.0)
    right = window & (lsf_def_wave >= reference + 10.0)
    left_bg = (np.nanmedian(arc_rect[:, left], axis=1)
               if np.any(left) else np.full(arc_rect.shape[0], np.nan))
    right_bg = (np.nanmedian(arc_rect[:, right], axis=1)
                if np.any(right) else np.full(arc_rect.shape[0], np.nan))
    background = np.nanmedian(np.column_stack((left_bg, right_bg)), axis=1)
    values = arc_rect[:, local].astype(float) - background[:, None]
    finite = np.isfinite(values)
    peak = np.full(values.shape[0], np.nan)
    for row in range(values.shape[0]):
        if np.any(finite[row]):
            peak[row] = np.nanmax(values[row])
    finite_rows = np.isfinite(peak) & (peak > 0.0)
    flat_top = np.zeros(values.shape[0], dtype=bool)
    exact_plateau = np.zeros(values.shape[0], dtype=bool)
    for row in np.where(finite_rows)[0]:
        top = finite[row] & (values[row] >= 0.995 * peak[row])
        flat_top[row] = np.any(top[:-1] & top[1:])
        exact_plateau[row] = np.any(
            top[:-1] & top[1:] &
            (np.abs(values[row, :-1] - values[row, 1:]) <=
             max(1.e-6, 1.e-6 * abs(peak[row]))))
    suspicious = flat_top | exact_plateau
    return {'finite': int(np.count_nonzero(finite_rows)),
            'flat_top': int(np.count_nonzero(flat_top)),
            'exact_plateau': int(np.count_nonzero(exact_plateau)),
            'suspicious': int(np.count_nonzero(suspicious))}


def _measure_direct_fwhm(wave, profile, reference):
    """Measure FWHM by PCHIP interpolation and half-height crossings only."""
    wave = np.asarray(wave, dtype=float)
    profile = np.asarray(profile, dtype=float)
    finite = np.isfinite(wave) & np.isfinite(profile)
    core = np.isfinite(wave) & (np.abs(wave - reference) <= 6.0)
    if np.count_nonzero(core) < 2 or not np.all(finite[core]):
        return {'valid': False, 'status': 'UNSUPPORTED_CORE', 'center': np.nan,
                'fwhm': np.nan, 'phase': np.nan}
    if np.count_nonzero(finite) < 5:
        return {'valid': False, 'status': 'TOO_FEW_SAMPLES', 'center': np.nan,
                'fwhm': np.nan, 'phase': np.nan}
    x = wave[finite]
    y = profile[finite]
    order = np.argsort(x)
    x, y = x[order], y[order]
    left_background = np.nanmedian(y[x <= reference - 10.])
    right_background = np.nanmedian(y[x >= reference + 10.])
    if not np.isfinite(left_background) or not np.isfinite(right_background):
        return {'valid': False, 'status': 'NO_BACKGROUND', 'center': np.nan,
                'fwhm': np.nan, 'phase': np.nan}
    background = float(np.nanmedian([left_background, right_background]))
    y = y - background
    search = (x >= reference - 5.0) & (x <= reference + 5.0)
    if np.count_nonzero(search) < 2:
        return {'valid': False, 'status': 'NO_PEAK_NEAR_LINE', 'center': np.nan,
                'fwhm': np.nan, 'phase': np.nan}
    try:
        interpolator = PchipInterpolator(x, y, extrapolate=False)
    except (ValueError, TypeError):
        return {'valid': False, 'status': 'INTERPOLATION_FAILED', 'center': np.nan,
                'fwhm': np.nan, 'phase': np.nan}
    sampled_search = np.where(search)[0]
    imax = int(sampled_search[np.argmax(y[sampled_search])])
    if (imax <= 0 or imax >= len(x) - 1 or
            not np.isfinite(x[imax - 1]) or not np.isfinite(x[imax + 1]) or
            not np.isfinite(y[imax - 1]) or not np.isfinite(y[imax + 1])):
        return {'valid': False, 'status': 'INVALID_PEAK_INTERPOLATION',
                'center': np.nan, 'fwhm': np.nan, 'phase': np.nan}
    ym, y0, yp = y[imax - 1], y[imax], y[imax + 1]
    dx_left = x[imax] - x[imax - 1]
    dx_right = x[imax + 1] - x[imax]
    if (not np.isfinite(dx_left) or not np.isfinite(dx_right) or
            dx_left <= 0.0 or not np.isclose(dx_left, dx_right,
                                              rtol=1.e-7, atol=1.e-10)):
        return {'valid': False, 'status': 'INVALID_PEAK_INTERPOLATION',
                'center': np.nan, 'fwhm': np.nan, 'phase': np.nan}
    denom = ym - 2.0 * y0 + yp
    denom_tolerance = (100.0 * np.finfo(float).eps *
                       max(1.0, abs(ym), abs(y0), abs(yp)))
    if not np.isfinite(denom) or denom >= 0.0 or abs(denom) <= denom_tolerance:
        return {'valid': False, 'status': 'INVALID_PEAK_INTERPOLATION',
                'center': np.nan, 'fwhm': np.nan, 'phase': np.nan}
    delta_pix = 0.5 * (ym - yp) / denom
    if not np.isfinite(delta_pix) or abs(delta_pix) > 1.0:
        return {'valid': False, 'status': 'INVALID_PEAK_INTERPOLATION',
                'center': np.nan, 'fwhm': np.nan, 'phase': np.nan}
    dx = 0.5 * (dx_left + dx_right)
    center = float(x[imax] + delta_pix * dx)
    peak_height = float(y0 - (yp - ym) ** 2 / (8.0 * denom))
    if not np.isfinite(peak_height) or peak_height <= 0.0:
        return {'valid': False, 'status': 'NONPOSITIVE_LINE', 'center': np.nan,
                'fwhm': np.nan, 'phase': np.nan}

    dense = np.linspace(x[0], x[-1], max(401, int(np.ceil((x[-1] - x[0]) * 100)) + 1))
    dense_y = np.asarray(interpolator(dense), dtype=float)
    search_dense = (dense >= reference - 5.0) & (dense <= reference + 5.0)
    if not np.any(search_dense) or not np.any(np.isfinite(dense_y[search_dense])):
        return {'valid': False, 'status': 'NO_PEAK_NEAR_LINE', 'center': np.nan,
                'fwhm': np.nan, 'phase': np.nan}
    peak_candidates = np.where(
        search_dense & np.isfinite(dense_y) &
        (dense_y >= np.roll(dense_y, 1)) &
        (dense_y >= np.roll(dense_y, -1)))[0]
    peak_candidates = peak_candidates[(peak_candidates > 0) &
                                      (peak_candidates < len(dense) - 1)]
    strong = peak_candidates[dense_y[peak_candidates] >= 0.80 * peak_height]
    if strong.size > 1:
        separated = [idx for idx in strong if abs(dense[idx] - center) >= 1.0]
        if separated:
            return {'valid': False, 'status': 'AMBIGUOUS_BLEND', 'center': np.nan,
                    'fwhm': np.nan, 'phase': np.nan}
    half = 0.5 * peak_height
    peak_index = int(np.argmin(np.abs(dense - center)))

    def crossing(direction):
        if direction < 0:
            indices = range(peak_index - 1, -1, -1)
        else:
            indices = range(peak_index, len(dense) - 1)
        for idx in indices:
            j = idx + 1 if direction > 0 else idx
            k = idx if direction > 0 else idx + 1
            y0, y1 = dense_y[j] - half, dense_y[k] - half
            if np.isfinite(y0) and np.isfinite(y1) and y0 * y1 <= 0.0:
                lo, hi = min(dense[j], dense[k]), max(dense[j], dense[k])
                flo, fhi = float(interpolator(lo) - half), float(interpolator(hi) - half)
                for _ in range(45):
                    mid = 0.5 * (lo + hi)
                    fmid = float(interpolator(mid) - half)
                    if flo * fmid <= 0.0:
                        hi, fhi = mid, fmid
                    else:
                        lo, flo = mid, fmid
                return 0.5 * (lo + hi)
        return None

    left_cross = crossing(-1)
    right_cross = crossing(1)
    if left_cross is None:
        return {'valid': False, 'status': 'NO_LEFT_CROSSING', 'center': np.nan,
                'fwhm': np.nan, 'phase': np.nan}
    if right_cross is None:
        return {'valid': False, 'status': 'NO_RIGHT_CROSSING', 'center': np.nan,
                'fwhm': np.nan, 'phase': np.nan}
    if left_cross <= x[0] + 1.e-6 or right_cross >= x[-1] - 1.e-6:
        return {'valid': False, 'status': 'CROSSING_AT_BOUNDARY', 'center': np.nan,
                'fwhm': np.nan, 'phase': np.nan}
    center = float(dense[peak_index])
    nearest_grid = 3470. + 2. * np.rint((center - 3470.) / 2.)
    phase = float(center - nearest_grid)
    return {'valid': True, 'status': 'OK', 'center': center,
            'fwhm': float(right_cross - left_cross), 'phase': phase}


def _lsf_spatial_header(tp, n, pixel_scale):
    """Construct the final cube's two-dimensional spatial WCS."""
    spatial = tp.to_header()
    scale = pixel_scale / 3600.
    spatial['WCSAXES'] = 2
    spatial['CD1_1'] = -scale
    spatial['CD1_2'] = 0.0
    spatial['CD2_1'] = 0.0
    spatial['CD2_2'] = scale
    for key in ('CDELT1', 'CDELT2', 'CDELT3', 'CRPIX3', 'CRVAL3',
                'CTYPE3', 'CUNIT3', 'SPECSYS'):
        if key in spatial:
            del spatial[key]
    spatial['CRPIX1'] = (n + 1) / 2.
    spatial['CRPIX2'] = (n + 1) / 2.
    spatial['CTYPE1'] = 'RA---TAN'
    spatial['CTYPE2'] = 'DEC--TAN'
    spatial['CUNIT1'] = 'deg'
    spatial['CUNIT2'] = 'deg'
    return spatial


def _phase_summary(phases, fwhm):
    """Return a transparent binned phase diagnostic, not a correction."""
    finite = np.isfinite(phases) & np.isfinite(fwhm)
    if np.count_nonzero(finite) < 10:
        return np.nan, np.nan, False
    x = phases[finite]
    y = fwhm[finite]
    edges = np.linspace(-1., 1., 9)
    medians = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        selected = (x >= lo) & (x < hi if hi < edges[-1] else x <= hi)
        if np.count_nonzero(selected) >= 3:
            medians.append(float(np.nanmedian(y[selected])))
    phase_range = (float(np.nanmax(medians) - np.nanmin(medians))
                   if len(medians) >= 2 else np.nan)
    correlation = (float(np.corrcoef(x, y)[0, 1])
                   if np.std(x) > 0. and np.std(y) > 0. else np.nan)
    median = float(np.nanmedian(y))
    evidence = bool(np.isfinite(phase_range) and np.isfinite(median) and
                    median != 0. and phase_range > 0.03 * abs(median))
    return phase_range, correlation, evidence


def _run_lsf_measurement(h5files, surname, def_wave, raarray, decarray,
                         specarray, shot_indices, shot_names, tp, xg, yg,
                         xgrid, ygrid, cube, ncontribcube, pixel_scale, log,
                         records=None):
    """Run the optional sparse empirical effective-resolution experiment."""
    if records is None:
        records = []
        for h5file in h5files:
            t = tables.open_file(h5file)
            records.extend(_build_lsf_amplifier_centers(h5file, t, log=log))
            t.close()
    sample_lookup, unique_positions = _map_lsf_centers_to_spaxels(records, tp, xg, yg)
    log.info('LSF amplifier-center sampling: %d centers, %d unique in-cube spaxels.',
             len(records), len(unique_positions))
    if not unique_positions:
        raise RuntimeError('LSF requested but no amplifier centers map into the cube.')

    line_masks = [np.abs(def_wave - reference) <= ARC_WINDOW_ANGSTROM
                  for reference in REFERENCE_ARC_WAVELENGTHS]
    lsf_wave_indices = np.unique(np.concatenate(
        [np.where(mask)[0] for mask in line_masks])).astype(np.int64)
    lsf_def_wave = def_wave[lsf_wave_indices]
    log.info('LSF compact spectral grid: %d of %d science bins.',
             len(lsf_def_wave), len(def_wave))
    arc_rect = _rectify_lsf_arcs(
        h5files, def_wave, lsf_def_wave, lsf_wave_indices, specarray, log=log)

    normalized_by_line = []
    for line_number, reference in enumerate(REFERENCE_ARC_WAVELENGTHS):
        clipping = _native_arc_clipping_diagnostic(
            arc_rect, lsf_def_wave, reference)
        log.info('LSF native arc diagnostic %0.3f A: usable native profiles=%d, '
                 'flat-top profiles=%d, exact plateaus=%d, suspicious=%d. '
                 'These are QA flags only; no saturation model or rejection applied.',
                 reference, clipping['finite'], clipping['flat_top'],
                 clipping['exact_plateau'], clipping['suspicious'])
        normalized, profile_valid, signal = _normalize_lsf_arc_window(
            arc_rect, lsf_def_wave, reference)
        normalized_by_line.append(normalized)
        if line_number in (0, len(REFERENCE_ARC_WAVELENGTHS) // 2,
                           len(REFERENCE_ARC_WAVELENGTHS) - 1):
            finite_signal = signal[np.isfinite(signal) & (signal > 0.)]
            if finite_signal.size:
                p16, p50, p84 = np.nanpercentile(finite_signal, [16, 50, 84])
                log.info('LSF native arc amplitude %0.3f A: positive integrated '
                         'line signal N=%d p16/p50/p84=%0.5g/%0.5g/%0.5g '
                         'before unit-flux normalization.', reference,
                         finite_signal.size, p16, p50, p84)
            else:
                log.warning('LSF native arc amplitude %0.3f A: no positive '
                            'integrated line signals before normalization.', reference)
    log.info('LSF arc normalization: local outer-window background subtraction '
             'and unit integrated line flux; no assumption of pre-normalized '
             'Raw.arcspectrum was made.')

    fwhm_by_line = []
    center_by_line = []
    phase_by_line = []
    status_by_line = []
    ncontrib_by_line = []
    summary = []
    for line_number, reference in enumerate(REFERENCE_ARC_WAVELENGTHS):
        local = np.abs(lsf_def_wave - reference) <= ARC_WINDOW_ANGSTROM
        local_wave = lsf_def_wave[local]
        line_indices = lsf_wave_indices[local]
        normalized_arc = normalized_by_line[line_number][:, local].copy()
        # Availability is deliberately imposed after line normalization.  A
        # missing science bin therefore removes that bin's contribution but
        # cannot renormalize the surviving arc wings.
        for compact_column, full_index in enumerate(line_indices):
            normalized_arc[~np.isfinite(specarray[:, full_index]), compact_column] = np.nan
        propagated, sample_ncontrib = _sparse_reconstruct_arc(
            raarray, decarray, normalized_arc, line_indices,
            sample_lookup, xg, yg, tp, shot_indices)
        fwhm = np.full(len(unique_positions), np.nan, dtype=float)
        center = np.full(len(unique_positions), np.nan, dtype=float)
        phase = np.full(len(unique_positions), np.nan, dtype=float)
        status = np.full(len(unique_positions), 'NO_PROFILE', dtype='U32')
        nnear = np.full(len(unique_positions), -1, dtype=np.int16)
        profile_valid = np.zeros(len(unique_positions), dtype=bool)
        for sample_id in range(len(unique_positions)):
            # ``propagated`` is already limited to this line's local columns.
            profile = propagated[sample_id]
            support = sample_ncontrib[sample_id]
            nfinite = np.isfinite(profile)
            if np.any(nfinite):
                nnear[sample_id] = int(np.nanmedian(support[nfinite]))
            core = np.abs(local_wave - reference) <= 6.0
            if (np.count_nonzero(core) >= 2 and
                    np.all(np.isfinite(profile[core])) and
                    np.all(support[core] >= 2)):
                profile_valid[sample_id] = True
                result = _measure_direct_fwhm(local_wave, profile, reference)
            else:
                result = {'valid': False, 'status': 'INSUFFICIENT_SUPPORT',
                          'center': np.nan, 'fwhm': np.nan, 'phase': np.nan}
            status[sample_id] = result['status']
            if result['valid']:
                fwhm[sample_id] = result['fwhm']
                center[sample_id] = result['center']
                phase[sample_id] = result['phase']

        fwhm_by_line.append(fwhm)
        center_by_line.append(center)
        phase_by_line.append(phase)
        status_by_line.append(status)
        ncontrib_by_line.append(nnear)
        valid = np.isfinite(fwhm)
        phase_range, phase_corr, phase_evidence = _phase_summary(phase, fwhm)
        values = fwhm[valid]
        centers = center[valid]
        if values.size:
            p16, p50, p84 = np.nanpercentile(values, [16, 50, 84])
            scatter = 1.4826 * np.nanmedian(np.abs(values - p50))
            center_offset = centers - reference
            log.info('LSF %0.3f A: centers=%d, propagated profiles=%d/%d '
                     '(%0.4f), valid FWHM=%d/%d (%0.4f), '
                     'FWHM p16/p50/p84=%0.4f/%0.4f/%0.4f, '
                     'robust spatial scatter=%0.4f, center offset median/range='
                     '%0.4f/%0.4f A, phase-bin range=%s A, phase r=%s, '
                     'phase evidence=%s, laboratory blend=%s, usable_for_lsf=%s',
                     reference,
                     len(records), int(np.count_nonzero(profile_valid)),
                     len(unique_positions),
                     np.count_nonzero(profile_valid) / float(max(1, len(unique_positions))),
                     values.size, len(unique_positions),
                     values.size / float(max(1, len(unique_positions))),
                     p16, p50, p84, scatter, np.nanmedian(center_offset),
                     np.nanmax(center_offset) - np.nanmin(center_offset),
                     ('%0.4f' % phase_range if np.isfinite(phase_range) else 'nan'),
                     ('%0.4f' % phase_corr if np.isfinite(phase_corr) else 'nan'),
                     phase_evidence, ARC_BLEND_STATES[line_number],
                     bool(ARC_USABLE_FOR_LSF[line_number]))
            summary.append((reference, p16, p50, p84, scatter, phase_range,
                            phase_corr, phase_evidence))
        else:
            log.warning('LSF %0.3f A: no valid direct FWHM measurements; '
                        'laboratory blend=%s, usable_for_lsf=%s', reference,
                        ARC_BLEND_STATES[line_number],
                        bool(ARC_USABLE_FOR_LSF[line_number]))
            summary.append((reference, np.nan, np.nan, np.nan, np.nan,
                            np.nan, np.nan, False))

        # The validation deliberately exercises the full image estimator only
        # at three compact wavelengths; production values above came only from
        # the sparse evaluator.
        if line_number == 0:
            local_columns = np.where(local)[0]
            validation_columns = np.unique(
                np.linspace(0, len(local_columns) - 1,
                            min(3, len(local_columns)), dtype=int))
            for local_column in validation_columns:
                compact_column = local_columns[local_column]
                full_index = int(lsf_wave_indices[compact_column])
                x, y = tp.wcs_world2pix(raarray[:, full_index],
                                        decarray[:, full_index], 1)
                full = make_image_gaussian(
                    np.column_stack((x, y)), normalized_arc[:, local_column],
                    None, xg, yg, xgrid, ygrid, 1.8 / 2.35, shot_indices)
                sparse, sparse_n = _sparse_reconstruct_one_wave(
                    np.column_stack((x, y)), normalized_arc[:, local_column],
                    xg, yg, sample_lookup, shot_indices)
                for sample_id, (py, px) in enumerate(unique_positions):
                    np.testing.assert_allclose(sparse[sample_id], full[0][py, px],
                                               rtol=2.e-6, atol=2.e-6)
                    assert int(sparse_n[sample_id]) == int(full[3][py, px])
            log.info('LSF sparse/full validation: SCI and NCONTRIB agree at '
                     'representative wavelengths and all selected sample spaxels.')

    # Build the requested one-row-per-amplifier-center x reference-line table.
    nrows = len(records) * len(REFERENCE_ARC_WAVELENGTHS)
    table = Table()
    table['shot'] = np.array([r['shot'] for r in records for _ in REFERENCE_ARC_WAVELENGTHS], dtype='U80')
    table['specid'] = np.array([r['specid'] for r in records for _ in REFERENCE_ARC_WAVELENGTHS], dtype='U16')
    table['ifuslot'] = np.array([r['ifuslot'] for r in records for _ in REFERENCE_ARC_WAVELENGTHS], dtype=np.int32)
    table['ifuid'] = np.array([r['ifuid'] for r in records for _ in REFERENCE_ARC_WAVELENGTHS], dtype='U16')
    table['amp'] = np.array([r['amp'] for r in records for _ in REFERENCE_ARC_WAVELENGTHS], dtype='U8')
    table['identity_ok'] = np.array([r['identity_ok'] for r in records for _ in REFERENCE_ARC_WAVELENGTHS], dtype=bool)
    table['ra'] = np.array([r['ra'] for r in records for _ in REFERENCE_ARC_WAVELENGTHS], dtype=float)
    table['dec'] = np.array([r['dec'] for r in records for _ in REFERENCE_ARC_WAVELENGTHS], dtype=float)
    table['cube_x'] = np.array([r['cube_x'] for r in records for _ in REFERENCE_ARC_WAVELENGTHS], dtype=np.int32)
    table['cube_y'] = np.array([r['cube_y'] for r in records for _ in REFERENCE_ARC_WAVELENGTHS], dtype=np.int32)
    table['sample_id'] = np.array([r['sample_id'] for r in records for _ in REFERENCE_ARC_WAVELENGTHS], dtype=np.int32)
    table['reference_wavelength'] = np.tile(REFERENCE_ARC_WAVELENGTHS, len(records))
    table['measured_center'] = np.array([center_by_line[j][r['sample_id']] if r['sample_id'] >= 0 else np.nan
                                         for r in records for j in range(len(REFERENCE_ARC_WAVELENGTHS))], dtype=float)
    table['fwhm'] = np.array([fwhm_by_line[j][r['sample_id']] if r['sample_id'] >= 0 else np.nan
                              for r in records for j in range(len(REFERENCE_ARC_WAVELENGTHS))], dtype=float)
    table['grid_phase'] = np.array([phase_by_line[j][r['sample_id']] if r['sample_id'] >= 0 else np.nan
                                    for r in records for j in range(len(REFERENCE_ARC_WAVELENGTHS))], dtype=float)
    table['ncontrib'] = np.array([ncontrib_by_line[j][r['sample_id']] if r['sample_id'] >= 0 else -1
                                  for r in records for j in range(len(REFERENCE_ARC_WAVELENGTHS))], dtype=np.int16)
    table['valid'] = np.array([bool(r['sample_id'] >= 0 and np.isfinite(fwhm_by_line[j][r['sample_id']]))
                               for r in records for j in range(len(REFERENCE_ARC_WAVELENGTHS))], dtype=bool)
    table['usable_for_lsf'] = np.array([
        bool(ARC_USABLE_FOR_LSF[j] and r['sample_id'] >= 0 and
             np.isfinite(fwhm_by_line[j][r['sample_id']]))
        for r in records for j in range(len(REFERENCE_ARC_WAVELENGTHS))], dtype=bool)
    table['status'] = np.array([status_by_line[j][r['sample_id']] if r['sample_id'] >= 0 else 'OUTSIDE_CUBE'
                                for r in records for j in range(len(REFERENCE_ARC_WAVELENGTHS))], dtype='U32')
    table['lab_blend'] = np.array([
        ARC_BLEND_STATES[j] == 'KNOWN_BLEND' for _ in records
        for j in range(len(REFERENCE_ARC_WAVELENGTHS))], dtype=bool)
    table['lab_blend_state'] = np.array([
        ARC_BLEND_STATES[j] for _ in records
        for j in range(len(REFERENCE_ARC_WAVELENGTHS))], dtype='U16')
    table['blend_note'] = np.array([ARC_BLEND_NOTES[float(REFERENCE_ARC_WAVELENGTHS[j])]
                                    for _ in records for j in range(len(REFERENCE_ARC_WAVELENGTHS))], dtype='U96')
    sample_path = op.basename('%s_lsf_samples.fits' % surname)
    table.write(sample_path, format='fits', overwrite=True)

    spatial_header = _lsf_spatial_header(tp, len(xg), pixel_scale)
    public_line_indices = np.where(ARC_USABLE_FOR_LSF)[0]
    for line_number in public_line_indices:
        reference = REFERENCE_ARC_WAVELENGTHS[line_number]
        valid = np.isfinite(fwhm_by_line[line_number])
        image = np.full(xgrid.shape, np.nan, dtype=np.float32)
        if np.any(valid):
            points = np.array([(unique_positions[sid][1], unique_positions[sid][0])
                               for sid in np.where(valid)[0]], dtype=float)
            values = fwhm_by_line[line_number][valid]
            image[:, :] = griddata(points, values, (xgrid - 1., ygrid - 1.),
                                   method='nearest').astype(np.float32)
        science_index = int(np.argmin(np.abs(def_wave - reference)))
        image[ncontribcube[science_index] < 2] = np.nan
        header = spatial_header.copy()
        header['REFWAVE'] = (float(reference), 'laboratory air wavelength [Angstrom]')
        header['WAVECONV'] = ('AIR', 'Hg/Cd reference wavelength convention')
        header['METHOD'] = ('DIRECT/NONPARAMETRIC', 'PCHIP peak and half-height crossings')
        header['INTERP'] = ('NEAREST', 'nearest-neighbor spatial sample interpolation')
        header['LSFMODEL'] = ('NONE', 'profile shape not specified')
        header['BUNIT'] = 'Angstrom'
        header.add_history('Effective cube FWHM from propagated master arcs.')
        output = op.basename('%s_lsf_fwhm_%04d.fits' % (surname, int(np.floor(reference + .5))))
        fits.PrimaryHDU(image, header=header).writeto(output, overwrite=True)

    # Compact diagnostics: raw center locations, FWHM values, medians versus
    # wavelength, and FWHM versus phase on the two-Angstrom grid.
    fig, axes = plt.subplots(2, 2, figsize=(13, 10), constrained_layout=True)
    amp_colors = {'LL': 'tab:blue', 'LU': 'tab:orange',
                  'RL': 'tab:green', 'RU': 'tab:red'}
    for amplifier, color in amp_colors.items():
        amplifier_records = [r for r in records
                             if r['sample_id'] >= 0 and r['amp'] == amplifier]
        axes[0, 0].scatter([r['cube_x'] for r in amplifier_records],
                           [r['cube_y'] for r in amplifier_records],
                           s=4, alpha=.45, color=color, label=amplifier)
    axes[0, 0].set_title('Amplifier-center sample spaxels')
    axes[0, 0].set_aspect('equal', adjustable='box')
    axes[0, 0].set_xlabel('cube x pixel (1-based)')
    axes[0, 0].set_ylabel('cube y pixel (1-based)')
    axes[0, 0].legend(title='amplifier', fontsize=8, markerscale=1.5)
    median_sample = np.nanmedian(np.vstack([fwhm_by_line[i] for i in public_line_indices]), axis=0)
    good_sample = np.isfinite(median_sample)
    if np.any(good_sample):
        axes[0, 1].scatter([unique_positions[sid][1] + 1 for sid in np.where(good_sample)[0]],
                           [unique_positions[sid][0] + 1 for sid in np.where(good_sample)[0]],
                           c=median_sample[good_sample], s=5, cmap='viridis')
    axes[0, 1].set_title('Median valid FWHM at sample spaxels')
    axes[0, 1].set_aspect('equal', adjustable='box')
    axes[0, 1].set_xlabel('cube x pixel (1-based)')
    axes[0, 1].set_ylabel('cube y pixel (1-based)')
    # Include the two known blends in these diagnostic plots so their broad
    # QA widths remain visible, while keeping them out of the public summary.
    for line_number, reference in enumerate(REFERENCE_ARC_WAVELENGTHS):
        values = fwhm_by_line[line_number]
        blend = ARC_BLEND_STATES[line_number] == 'KNOWN_BLEND'
        axes[1, 0].plot(
            reference,
            np.nanmedian(values) if np.any(np.isfinite(values)) else np.nan,
            'x' if blend else 'o',
            color='tab:red' if blend else 'tab:blue',
            label='%0.0f blend QA-only' % reference if blend else None)
        axes[1, 1].scatter(phase_by_line[line_number], values, s=3, alpha=.25,
                           label=('%0.0f blend QA-only' % reference)
                           if blend else '%0.0f' % reference)
    axes[1, 0].set_title('Median FWHM versus reference wavelength (blend points QA-only)')
    axes[1, 0].set_xlabel('reference wavelength [A]')
    axes[1, 0].set_ylabel('FWHM [A]')
    axes[1, 0].legend(fontsize=7, ncol=2)
    axes[1, 1].set_title('FWHM versus 2-Angstrom grid phase (blend QA-only)')
    axes[1, 1].set_xlabel('center minus nearest grid bin [A]')
    axes[1, 1].set_ylabel('FWHM [A]')
    axes[1, 1].legend(fontsize=7, ncol=3)
    qa_path = op.basename('%s_lsf_qa.png' % surname)
    fig.savefig(qa_path, dpi=180)
    plt.close(fig)

    public_measured_count = sum(
        np.any(np.isfinite(fwhm_by_line[i])) for i in public_line_indices)
    measured_count = sum(
        np.any(np.isfinite(fwhm_by_line[i]))
        for i in range(len(REFERENCE_ARC_WAVELENGTHS)))
    all_values = np.concatenate([
        fwhm_by_line[i][np.isfinite(fwhm_by_line[i])]
        for i in public_line_indices if np.any(np.isfinite(fwhm_by_line[i]))
    ]) if any(np.any(np.isfinite(fwhm_by_line[i])) for i in public_line_indices) else np.array([])
    if all_values.size:
        median_values = np.array([summary[i][2] for i in public_line_indices
                                  if np.isfinite(summary[i][2])])
        median_change = (np.nanmax(median_values) - np.nanmin(median_values)
                         if median_values.size else np.nan)
        typical_scatter = np.nanmedian(
            [summary[i][4] for i in public_line_indices
             if np.isfinite(summary[i][4])])
        log.info('LSF overall: %d/%d public reference lines measured; unique evaluated '
                 'spaxels=%d; FWHM range=%0.4f-%0.4f A; spatial robust scatter '
                 'per line=%s; peak-to-peak change in median FWHM=%0.4f A; '
                 'typical spatial scatter=%0.4f A. The prior ~5.35-5.39 A '
                 'single-CCD result is treated only as a sanity check, not a '
                 'constraint. Anchor accounting: 9/9 windows propagated, '
                 '%d/9 measured, 7/9 usable isolated-line anchors, 2/9 '
                 'excluded as known blends.',
                 public_measured_count, len(public_line_indices),
                 len(unique_positions), np.nanmin(all_values), np.nanmax(all_values),
                 ', '.join('%0.4f' % summary[i][4] for i in public_line_indices
                           if np.isfinite(summary[i][4])), median_change, typical_scatter,
                 measured_count)
    else:
        log.warning('LSF overall: no valid public-anchor FWHM values; '
                    'anchor accounting: 9/9 windows propagated, %d/9 measured, '
                    '7/9 usable isolated-line anchors, 2/9 excluded as known blends.',
                    measured_count)
    return table, summary, len(unique_positions)

args = parser.parse_args(args=None)
if args.wave_workers < 1:
    parser.error('--wave-workers must be at least 1')
args.log = setup_logging('make_image_from_h5')

def_wave = np.linspace(3470., 5540., 1036)

h5files = sorted(glob.glob(args.h5files))
args.log.info(f"Detected {len(h5files)} input file(s). Assuming each h5 contains 3 interleaved dithers by default.")
if not h5files:
    raise ValueError('no H5 files matched the production input pattern')
if any(op.basename(h5file) == '20200523_0000023.h5' for h5file in h5files):
    raise ValueError('20200523_0000023.h5 is explicitly excluded from the M101 sample')

bounding_box = [float(corner.replace(' ', ''))
                        for corner in args.image_center_size.split(',')]

bounding_box[2] = int(bounding_box[2]*60./args.pixel_scale/2.) * 2 * args.pixel_scale

bb = int(bounding_box[2]/args.pixel_scale/2.)*args.pixel_scale
N = int(bounding_box[2]/args.pixel_scale/2.) * 2 + 1
args.log.info('Image size in pixels: %i' % N)
xg = np.arange(N) + 1
yg = np.arange(N) + 1
xgrid, ygrid = np.meshgrid(xg, yg)

cnt = 0
cnt_array = np.zeros((len(h5files), 2), dtype=int)
# Build per-exposure (shot) index selections assuming 3 interleaved dithers per h5
nexp_default = 3
shot_indices = []  # list of numpy arrays of global indices for each exposure
shot_names = []
lsf_center_records = []
for i, h5file in enumerate(h5files):
    t = tables.open_file(h5file)
    ra = t.root.Info.cols.ra[:]
    n_fib = len(ra)
    # window for this file in the global arrays
    start = cnt
    end = cnt + n_fib
    cnt_array[i, 0] = start
    cnt_array[i, 1] = end
    # Build local indices grouped by 112-fiber blocks, split into 3 exposures
    inds_local = np.arange(n_fib)
    block_ids = (inds_local // 112).astype(int)
    for k in range(nexp_default):
        sel_local = np.where((block_ids % nexp_default) == k)[0]
        if sel_local.size > 0:
            shot_indices.append(sel_local + start)
            shot_names.append('%s exposure %d' % (op.basename(h5file), k + 1))
    if args.make_lsf:
        lsf_center_records.extend(_build_lsf_amplifier_centers(h5file, t, log=args.log))
    cnt = end
    t.close()
args.log.info('Number of total fibers: %i' % cnt)
args.log.info(f'Total shots (exposures) assumed: {len(shot_indices)} (3 per h5file).')
if args.make_lsf:
    args.log.info('LSF QA retained %d amplifier-center records before WCS mapping.',
                  len(lsf_center_records))

# Astrometry with CRPIX at image center (not lower-left)
_x0 = (N + 1) / 2.0
_y0 = (N + 1) / 2.0
A = Astrometry(bounding_box[0], bounding_box[1], 0., _x0, _y0)
# Ensure TP uses same centered CRPIX
tp = A.setup_TP(A.ra0, A.dec0, 0., x0=_x0, y0=_y0)

# Preflight coverage check: ensure some fibers land within requested region
try:
    xmin, xmax = xg.min(), xg.max()
    ymin, ymax = yg.min(), yg.max()
    in_region_total = 0
    for h5file in h5files:
        t = tables.open_file(h5file)
        ra_chk = t.root.Info.cols.ra[:]
        dec_chk = t.root.Info.cols.dec[:]
        t.close()
        x_chk, y_chk = tp.wcs_world2pix(ra_chk, dec_chk, 1)
        mask_in = (x_chk >= xmin) & (x_chk <= xmax) & (y_chk >= ymin) & (y_chk <= ymax)
        n_in = int(mask_in.sum())
        args.log.info(f'Fibers in region for {op.basename(h5file)}: {n_in}')
        in_region_total += n_in
    if in_region_total == 0:
        args.log.error('No fibers found within the requested cube region. '
                       'The output would be all zeros. Aborting. '\
                       f"Center=({bounding_box[0]:.5f},{bounding_box[1]:.5f}), size={bounding_box[2]:.2f} arcsec")
        sys.exit(2)
    if in_region_total < 10:
        args.log.warning(f'Only {in_region_total} fibers fall within the region; output may be very noisy or near zero.')
except Exception as e:
    args.log.warning(f'Coverage precheck failed with {e}; proceeding anyway.')

# One-time fixed external-image setup.  These images are characterized once;
# only temporary per-H5/per-exposure PSF matches are made during PASS 1.
if args.m101_on_image is None or args.m101_on_filter is None:
    parser.error('M101 hierarchy requires --m101-on-image and --m101-on-filter '
                 '(historical -if/-ff aliases are accepted)')
if args.m101_iterations < 1:
    parser.error('--m101-iterations must be positive')
f_global = validated_m101.load_fq(args.m101_fq_template)
filters_global = {
    'ON': validated_m101.read_filter(args.m101_on_filter),
    'OFF': validated_m101.read_filter(args.m101_off_filter),
}
images_global = {
    'ON': validated_m101.load_image(args.m101_on_image),
    'OFF': validated_m101.load_image(args.m101_off_image),
}
for band in ('ON', 'OFF'):
    validated_m101.estimate_external_background(images_global[band])
    validated_m101.characterize_external_image(images_global[band], band)
    args.log.info('M101 %s external image characterized once: FWHM=%.8g arcsec, '
                  'background=%.8g +/- %.8g', band,
                  images_global[band]['psf']['fwhm_arcsec'],
                  images_global[band]['background'],
                  images_global[band]['background_scatter'])

# The residual-sky routine retains only this coarse ON-image binimage as a
# blank-region criterion.  It is not used for external photometry or
# normalization; all calibration photometry uses the exact aperture above.
image_for_sky = images_global['ON']
original_image_data = image_for_sky['data']
original_image_wcs = image_for_sky['wcs']
wc = original_image_wcs
ny, nx = original_image_data.shape
yind, xind = np.indices((ny, nx))
r, d = wc.wcs_pix2world(xind.ravel() + 1.0, yind.ravel() + 1.0, 1)
P = np.zeros((len(r), 2))
px, py = tp.wcs_world2pix(r, d, 1)
P[:, 0], P[:, 1] = px, py
distance = np.sqrt((P[:, 0] - xg[0]) ** 2 + (P[:, 1] - yg[0]) ** 2)
nearest = np.argmin(distance)
yi, xi = np.unravel_index(nearest, yind.shape)
cut_size = 4 * len(xg)
newimage = original_image_data[yi:yi + cut_size, xi:xi + cut_size]
if newimage.shape != (cut_size, cut_size):
    raise ValueError('ON image does not contain the requested residual-sky cutout')
binimage = rebin(newimage, (len(xg), len(xg)))
xcoord_image = P[:, 0].reshape(original_image_data.shape)[yi:yi + cut_size, xi:xi + cut_size]
ycoord_image = P[:, 1].reshape(original_image_data.shape)[yi:yi + cut_size, xi:xi + cut_size]
ximage = rebin(xcoord_image, (len(xg), len(xg)))
yimage = rebin(ycoord_image, (len(xg), len(xg)))
P_small = np.column_stack((ximage.ravel(), yimage.ravel()))
binimage = griddata(P_small, binimage.ravel(), (xgrid, ygrid), method='cubic')

state_templates_global = _m101_load_state_template(args.m101_ifu_state_template, args.log)
calibrations = []
for h5file in h5files:
    if op.basename(h5file) == '20200523_0000023.h5':
        raise ValueError('excluded H5 was passed to the production builder: %s' % h5file)
    production_state = 2 if op.basename(h5file) in M101_SECONDARY_H5 else 1
    args.log.info('M101 PASS 1 calibration %s: production state=%d',
                  op.basename(h5file), production_state)
    calibrations.append(_m101_calibrate_one_h5(
        h5file, images_global, filters_global, f_global, args.m101_iterations,
        state_templates_global, production_state, args.log))
_m101_derive_gray_factors(calibrations, args.log)
_m101_write_calibration_qa(
    calibrations, images_global, 'm101_production_calibration.csv', args.log)
if args.m101_calibration_only:
    args.log.info('M101 calibration-only requested; stopping before PASS 2 cube ingestion.')
    sys.exit(0)
# Allocate cube products only after calibration-only has passed.  PASS 1 has
# retained compact calibration products, not H5 spectra or cube planes.
cube = np.zeros((len(def_wave),) + xgrid.shape, dtype='float32')
variancecube = np.zeros((len(def_wave),) + xgrid.shape, dtype='float32')
weightcube = np.zeros((len(def_wave),) + xgrid.shape, dtype='float32')
ncontribcube = np.zeros((len(def_wave),) + xgrid.shape, dtype='uint8')
dqcube = np.zeros((len(def_wave),) + xgrid.shape, dtype='uint16')
raarray = np.zeros((cnt, len(def_wave)), dtype='float32')
decarray = np.zeros((cnt, len(def_wave)), dtype='float32')
specarray = np.zeros((cnt, len(def_wave)), dtype='float32')
errarray = np.zeros((cnt, len(def_wave)), dtype='float32')
# PASS 2: reopen each H5, apply compact calibration products, and populate the
# existing arrays.  The historical norm_array and external normalization path
# are intentionally absent.
cnt = 0
calibration_by_h5 = {calibration['h5']: calibration for calibration in calibrations}
for h5file in h5files:
    h5name = op.basename(h5file)
    calibration = calibration_by_h5[h5name]
    spectra, error, ra, dec, labels, survey_by_exp = _m101_apply_h5_calibration(
        h5file, calibration, f_global, binimage, xg, yg, tp, args.log)
    cnt1 = cnt + len(ra)

    # Preserve the existing wavelength-dependent ADR cube positions, but use
    # the Survey row belonging to each exposure rather than Survey row 0.
    extractor = Extract(wave=def_wave)
    for exposure in (1, 2, 3):
        selected = labels == exposure
        survey = survey_by_exp[exposure]
        astrometry = Astrometry(float(survey['ra']), float(survey['dec']),
                                float(survey['pa']), 0., 0.)
        extractor.get_ADR_RAdec(astrometry)
        indices = np.flatnonzero(selected)
        raarray[cnt + indices, :] = (
            ra[indices, None] - extractor.ADRra[None, :] / 3600. /
            np.cos(np.deg2rad(float(survey['dec']))))
        decarray[cnt + indices, :] = (
            dec[indices, None] - extractor.ADRdec[None, :] / 3600.)

    Gk = Gaussian1DKernel(1.8)
    for k in np.arange(len(spectra)):
        if np.isfinite(spectra[k]).sum() > 800:
            spectra[k] = interpolate_replace_nans(spectra[k], Gk,
                                                  **{'boundary': 'extend'})
    specarray[cnt:cnt1, :] = spectra
    errarray[cnt:cnt1, :] = error
    args.log.info('PASS 2 %s: assigned calibrated spectra directly to '
                  'specarray/errarray; no norm_array applied.', h5name)
    cnt = cnt1

def render_wavelength(i):
    """Build one wavelength plane using the test Gaussian imaging path."""
    x, y = tp.wcs_world2pix(raarray[:, i], decarray[:, i], 1)
    # Check how many positions fall within xgrid/ygrid bounds
    xmin, xmax = xg.min(), xg.max()
    ymin, ymax = yg.min(), yg.max()
    in_bounds = (x >= xmin) & (x <= xmax) & (y >= ymin) & (y <= ymax)
    n_in = int(np.sum(in_bounds))
    n_tot = int(len(raarray))

    args.log.info('Working on wavelength %0.0f' % def_wave[i])
    # Keep this per-worker; sharing the old serial-loop Pos array would make
    # concurrent wavelength workers overwrite one another's positions.
    Pos = np.column_stack((x, y))
    data = specarray[:, i]
    edata = errarray[:, i]
    provenance = _classify_error_provenance(data, edata)
    shot_provenance = np.zeros((len(shot_indices), 6), dtype=np.int64)
    for k, cnt in enumerate(shot_indices):
        if isinstance(cnt, (tuple, list)) and len(cnt) == 2:
            indices = np.arange(int(cnt[0]), int(cnt[1]), dtype=np.int64)
        else:
            indices = np.asarray(cnt, dtype=np.int64)
        shot_provenance[k, :] = _classify_error_provenance(
            data[indices], edata[indices])
    image, varianceimage, weight, ncontrib, dq, variance_stats = make_image_gaussian(
        Pos, data, edata, xg, yg, xgrid, ygrid, 1.8 / 2.35, shot_indices)
    return i, image, varianceimage, weight, ncontrib, dq, variance_stats, provenance, shot_provenance


def _warm_up_gaussian_splat():
    """Compile the Numba kernel before wavelength workers are started."""
    indices = np.array([0], dtype=np.int64)
    position = np.array([1.0], dtype=np.float64)
    flux = np.array([1.0], dtype=np.float32)
    error = np.array([1.0], dtype=np.float32)
    flux_sum = np.zeros((3, 3), dtype=np.float32)
    weight_sum = np.zeros((3, 3), dtype=np.float32)
    variance_numerator = np.zeros((3, 3), dtype=np.float32)
    error_weight_sum = np.zeros((3, 3), dtype=np.float32)
    support = np.zeros((3, 3), dtype=np.uint8)
    _gaussian_splat_shot_xy(
        indices, position, position, flux, error, 1.0, 1.0, 3, 3,
        1.8 / 2.35, 3, 2, np.pi * 0.75 ** 2, flux_sum, weight_sum,
        variance_numerator, error_weight_sum, support)


def _warm_up_sparse_gaussian_splat():
    """Compile the optional sparse kernel before the LSF reconstruction."""
    indices = np.array([0], dtype=np.int64)
    position = np.array([1.0], dtype=np.float64)
    flux = np.array([1.0], dtype=np.float32)
    lookup = np.arange(9, dtype=np.int32).reshape(3, 3)
    flux_sum = np.zeros(9, dtype=np.float32)
    weight_sum = np.zeros(9, dtype=np.float32)
    support = np.zeros(9, dtype=np.uint8)
    _gaussian_splat_sparse_shot_xy(
        indices, position, position, flux, lookup, 1.0, 1.0, 3, 3,
        1.8 / 2.35, 3, 2, np.pi * 0.75 ** 2, flux_sum, weight_sum, support)


_warm_up_gaussian_splat()
if args.make_lsf:
    _warm_up_sparse_gaussian_splat()
args.log.info('Gaussian test reconstruction enabled: SCI uses subpixel splats and NCONTRIB >= 2.')
args.log.warning(
    'Gaussian reconstruction variance semantics: VAR is the final SCI-median '
    'variance; ERROR is its 1-sigma companion (sqrt(VAR)). Formal shot '
    'variances require complete positive fiber-error support, and empirical '
    'shot scatter is used for NCONTRIB >= 5.')

args.log.info('Using %d wavelength worker(s)' % args.wave_workers)
variance_diagnostics = {
    'valid_sci_voxels': 0,
    'finite_variance_voxels': 0,
    'shot_samples': 0,
    'shot_samples_missing_variance': 0,
    'both_variances': 0,
    'empirical_exceeds_formal': 0,
    'median_ratios': [],
    'median_ncontrib_values': [],
    'variance_unavailable_voxels': 0,
    'insufficient_support_voxels': 0,
    'formal_var_used_voxels': 0,
    'empirical_var_used_voxels': 0,
    'empirical_only_voxels': 0,
}
provenance_by_wave = np.zeros((len(def_wave), 6), dtype=np.int64)
provenance_by_shot = np.zeros((len(shot_indices), 6), dtype=np.int64)
with ThreadPoolExecutor(max_workers=args.wave_workers) as executor:
    # executor.map preserves wavelength order in the returned results.  Each
    # worker only reads the shared input arrays; cube writes remain here.
    for i, image, varianceimage, weight, ncontrib, dq, plane_stats, provenance, shot_provenance in executor.map(
            render_wavelength, range(len(def_wave))):
        cube[i, :, :] = image
        variancecube[i, :, :] = varianceimage
        weightcube[i, :, :] = weight
        ncontribcube[i, :, :] = ncontrib
        dqcube[i, :, :] = dq
        provenance_by_wave[i, :] = provenance
        provenance_by_shot += shot_provenance
        for key in ('valid_sci_voxels', 'finite_variance_voxels',
                    'shot_samples', 'shot_samples_missing_variance',
                    'both_variances', 'empirical_exceeds_formal',
                    'variance_unavailable_voxels', 'insufficient_support_voxels',
                    'formal_var_used_voxels', 'empirical_var_used_voxels',
                    'empirical_only_voxels'):
            variance_diagnostics[key] += plane_stats[key]
        if np.isfinite(plane_stats['median_ratio']):
            variance_diagnostics['median_ratios'].append(plane_stats['median_ratio'])
        if np.isfinite(plane_stats['median_ncontrib']):
            variance_diagnostics['median_ncontrib_values'].append(
                plane_stats['median_ncontrib'])

_valid_sci = variance_diagnostics['valid_sci_voxels']
_finite_var = variance_diagnostics['finite_variance_voxels']
_shots = variance_diagnostics['shot_samples']
_both = variance_diagnostics['both_variances']
args.log.info(
    'Variance diagnostics: finite final VAR fraction among valid SCI=%0.4f; '
    'shot SCI samples lacking complete error support=%0.4f; '
    'empirical variance > formal variance=%0.4f; '
    'median sqrt(Var_empirical/Var_formal)=%s; '
    'median NCONTRIB for valid SCI=%s; valid SCI voxels with VAR unavailable=%d',
    (_finite_var / float(_valid_sci) if _valid_sci else np.nan),
    (variance_diagnostics['shot_samples_missing_variance'] /
     float(_shots) if _shots else np.nan),
    (variance_diagnostics['empirical_exceeds_formal'] /
     float(_both) if _both else np.nan),
    (('%0.4f' % np.median(variance_diagnostics['median_ratios']))
     if variance_diagnostics['median_ratios'] else 'nan'),
    (('%0.2f' % np.median(variance_diagnostics['median_ncontrib_values']))
     if variance_diagnostics['median_ncontrib_values'] else 'nan'),
    variance_diagnostics['variance_unavailable_voxels'])

_prov_names = ('SCI-valid', 'finite positive error', 'nonfinite error',
               'zero error', 'negative error', 'other invalid error')
_prov_total = provenance_by_wave.sum(axis=0)
_prov_sci = int(_prov_total[PROV_NSCI])
args.log.info('Error provenance: SCI-valid fiber samples=%d', _prov_sci)
for _prov_index, _prov_name in enumerate(_prov_names[1:], start=1):
    args.log.info('Error provenance: %-22s=%d (%0.4f)', _prov_name,
                  int(_prov_total[_prov_index]),
                  (_prov_total[_prov_index] / float(_prov_sci)
                   if _prov_sci else np.nan))

# A compact 12-bin wavelength summary makes edge or interval concentration
# visible without retaining per-pixel provenance masks or logging every plane.
_wave_bins = np.linspace(0, len(def_wave), 13, dtype=int)
args.log.info('Error provenance by wavelength bin: wave_start-wave_end, '
              'NSCI, frac_valid, frac_nonfinite, frac_zero, frac_negative, frac_other')
for _bin_start, _bin_stop in zip(_wave_bins[:-1], _wave_bins[1:]):
    _bin_prov = provenance_by_wave[_bin_start:_bin_stop].sum(axis=0)
    _bin_sci = int(_bin_prov[PROV_NSCI])
    _bin_den = float(_bin_sci) if _bin_sci else np.nan
    args.log.info(
        'Error provenance wavelength %0.0f-%0.0f: NSCI=%d, '
        'valid=%0.4f, nonfinite=%0.4f, zero=%0.4f, negative=%0.4f, other=%0.4f',
        def_wave[_bin_start], def_wave[_bin_stop - 1], _bin_sci,
        _bin_prov[PROV_ERROR_VALID] / _bin_den,
        _bin_prov[PROV_ERROR_NONFINITE] / _bin_den,
        _bin_prov[PROV_ERROR_ZERO] / _bin_den,
        _bin_prov[PROV_ERROR_NEGATIVE] / _bin_den,
        _bin_prov[PROV_ERROR_OTHER] / _bin_den)

_invalid_by_wave = (provenance_by_wave[:, PROV_ERROR_NONFINITE] +
                    provenance_by_wave[:, PROV_ERROR_ZERO] +
                    provenance_by_wave[:, PROV_ERROR_NEGATIVE] +
                    provenance_by_wave[:, PROV_ERROR_OTHER])
_invalid_fraction_by_wave = np.divide(
    _invalid_by_wave, provenance_by_wave[:, PROV_NSCI],
    out=np.full(len(def_wave), np.nan, dtype=float),
    where=provenance_by_wave[:, PROV_NSCI] > 0)
_peak_wave = int(np.nanargmax(_invalid_fraction_by_wave)) if np.any(np.isfinite(_invalid_fraction_by_wave)) else None
if _peak_wave is not None:
    args.log.info('Error provenance concentration: peak invalid-error wavelength '
                  '%0.0f A, fraction=%0.4f; first/last-bin fractions=%s/%s.',
                  def_wave[_peak_wave], _invalid_fraction_by_wave[_peak_wave],
                  ('%0.4f' % _invalid_fraction_by_wave[0]),
                  ('%0.4f' % _invalid_fraction_by_wave[-1]))

_invalid_by_shot = provenance_by_shot[:, PROV_ERROR_NONFINITE] + provenance_by_shot[:, PROV_ERROR_ZERO] + provenance_by_shot[:, PROV_ERROR_NEGATIVE] + provenance_by_shot[:, PROV_ERROR_OTHER]
_top_shots = np.argsort(_invalid_by_shot)[::-1]
for _shot_index in _top_shots[:3]:
    if _invalid_by_shot[_shot_index] <= 0:
        break
    args.log.info('Error provenance exposure: %s invalid=%d of SCI-valid=%d',
                  shot_names[_shot_index], int(_invalid_by_shot[_shot_index]),
                  int(provenance_by_shot[_shot_index, PROV_NSCI]))

args.log.info(
    'DQ diagnostics: valid SCI voxels=%d; insufficient-support voxels=%d; '
    'SCI-valid voxels with finite VAR=%d; VAR unavailable=%d; '
    'formal VAR adopted=%d; empirical VAR adopted over formal=%d; '
    'empirical-only VAR=%d',
    variance_diagnostics['valid_sci_voxels'],
    variance_diagnostics['insufficient_support_voxels'],
    variance_diagnostics['finite_variance_voxels'],
    variance_diagnostics['variance_unavailable_voxels'],
    variance_diagnostics['formal_var_used_voxels'],
    variance_diagnostics['empirical_var_used_voxels'],
    variance_diagnostics['empirical_only_voxels'])

if args.make_lsf:
    _run_lsf_measurement(
        h5files, args.surname, def_wave, raarray, decarray, specarray,
        shot_indices, shot_names, tp, xg, yg, xgrid, ygrid, cube,
        ncontribcube, args.pixel_scale, args.log,
        records=lsf_center_records)

name = op.basename('%s_cube.fits' % args.surname)
scale = args.pixel_scale / 3600.
header['WCSAXES'] = 3
header['CD1_1']   = -scale              # minus sign -> RA increases to the left (north up, east left)
header['CD1_2']   = 0.0
header['CD2_1']   = 0.0
header['CD2_2']   = scale
del header['CDELT1'], header['CDELT2']
header['CRPIX1'] = (N+1) / 2
header['CRPIX2'] = (N+1) / 2
# Ensure DS9 recognizes 3D WCS
header['WCSAXES'] = 3
# Spectral axis definition (linear wavelength grid)
header['CDELT3'] = 2.
header['CRPIX3'] = 1.
header['CRVAL3'] = 3470.
header['CTYPE1'] = 'RA---TAN'
header['CTYPE2'] = 'DEC--TAN'
header['CTYPE3'] = 'WAVE'
header['CUNIT1'] = 'deg'
header['CUNIT2'] = 'deg'
header['CUNIT3'] = 'Angstrom'
header['SPECSYS'] = 'TOPOCENT'
header.add_history('Gaussian subpixel reconstruction.')
header.add_history('NCONTRIB >= 2 required for valid SCI and VAR.')
header.add_history('Final VAR = max(formal median variance, empirical median variance).')
F = fits.PrimaryHDU(np.array(cube, 'float32'), header=header)
F.writeto(name, overwrite=True)
name = op.basename('%s_variance_cube.fits' % args.surname)
header['CRPIX1'] = (N+1) / 2
header['CRPIX2'] = (N+1) / 2
# Ensure DS9 recognizes 3D WCS
header['WCSAXES'] = 3
# Spectral axis definition (linear wavelength grid)
header['CDELT3'] = 2.
header['CRPIX3'] = 1.
header['CRVAL3'] = 3470.
header['CTYPE1'] = 'RA---TAN'
header['CTYPE2'] = 'DEC--TAN'
header['CTYPE3'] = 'WAVE'
header['CUNIT1'] = 'deg'
header['CUNIT2'] = 'deg'
header['CUNIT3'] = 'Angstrom'
header['SPECSYS'] = 'TOPOCENT'
variance_header = header.copy()
variance_header['BUNIT'] = 'SCI units squared'
variance_header['EXTNAME'] = 'VARIANCE'
variance_header.add_comment(
    'VAR = variance of final SCI estimator; units are SCI units squared.')
variance_header.add_comment(
    'Formal and empirical estimates are combined by VAR = max(formal, empirical) when both exist.')
F = fits.PrimaryHDU(np.array(variancecube, 'float32'), header=variance_header)
F.writeto(name, overwrite=True)
# Preserve the historical error-cube filename as a 1-sigma companion.  Its
# meaning is now explicitly ERROR = sqrt(VAR), rather than a median of shot
# interpolation errors.
name = op.basename('%s_errorcube.fits' % args.surname)
error_header = header.copy()
error_header['BUNIT'] = 'SCI units'
error_header['EXTNAME'] = 'ERROR'
error_header.add_comment(
    'ERROR = 1-sigma standard deviation = sqrt(VAR). See the variance cube.')
F = fits.PrimaryHDU(np.sqrt(np.array(variancecube, 'float32')), header=error_header)
F.writeto(name, overwrite=True)
name = op.basename('%s_dq_cube.fits' % args.surname)
dq_header = header.copy()
dq_header['BUNIT'] = 'bit mask'
dq_header['EXTNAME'] = 'DQ'
dq_header['DQBIT0'] = 'NCONTRIB < 2'
dq_header['DQBIT1'] = 'SCI valid, VAR unavailable'
dq_header['DQBIT2'] = 'empirical VAR adopted over formal'
dq_header['DQBIT3'] = 'formal VAR adopted'
dq_header['DQBIT4'] = 'empirical-only VAR adopted'
dq_header.add_comment('DQ bits are independent flags; NCONTRIB/COVERAGE describe continuous support.')
F = fits.PrimaryHDU(np.array(dqcube, 'uint16'), header=dq_header)
F.writeto(name, overwrite=True)
# Write weight cube as requested
name = op.basename('%s_weight_cube.fits' % args.surname)
header['CRPIX1'] = (N+1) / 2
header['CRPIX2'] = (N+1) / 2
header['WCSAXES'] = 3
header['CDELT3'] = 2.
header['CRPIX3'] = 1.
header['CRVAL3'] = 3470.
header['CTYPE1'] = 'RA---TAN'
header['CTYPE2'] = 'DEC--TAN'
header['CTYPE3'] = 'WAVE'
header['CUNIT1'] = 'deg'
header['CUNIT2'] = 'deg'
header['CUNIT3'] = 'Angstrom'
header['SPECSYS'] = 'TOPOCENT'
F = fits.PrimaryHDU(np.array(weightcube, 'float32'), header=header)
F.writeto(name, overwrite=True)
name = op.basename('%s_ncontrib_cube.fits' % args.surname)
ncontrib_header = header.copy()
ncontrib_header['BUNIT'] = 'count'
ncontrib_header['EXTNAME'] = 'NCONTRIB'
ncontrib_header['COMMENT'] = 'Number of independent shots with a valid fiber within 2 sigma.'
F = fits.PrimaryHDU(np.array(ncontribcube, 'uint8'), header=ncontrib_header)
F.writeto(name, overwrite=True)
