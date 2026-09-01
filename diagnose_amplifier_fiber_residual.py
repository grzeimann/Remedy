#!/usr/bin/env python3
"""Diagnose amplifier-relative residuals in Remedy M101 H5 files.

The utility is read-only and processes one H5 at a time.  The normal path
uses rectified Fibers products; the opt-in ``--spectral-sky-test`` path uses
one exposure of the native Raw science spectra to test for sky-like edge
structure, and ``--additive-bandaid`` tests an in-memory Fibers correction.
"""

from argparse import ArgumentParser
import csv
import glob
from pathlib import Path
import warnings

import numpy as np
import tables
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS
from scipy.stats import spearmanr, theilslopes

from astrometry import Astrometry
from math_utils import biweight


RECTIFIED_WAVE = np.linspace(3470.0, 5540.0, 1036)
FIBERS_PER_AMPLIFIER = 112
FIBERS_PER_IFU = 448
EXPECTED_EXPOSURES = 3
M101_RA_DEG = 210.800
M101_DEC_DEG = 54.333
M101_SKY_MIN_RADIUS_ARCMIN = 6.0
M101_SKY_MIN_FINITE_FRACTION = 0.8
M101_SKY_MIN_FIBERS = 20
GAUSSIAN_FWHM_ARCSEC = 1.8
GAUSSIAN_SIGMA_ARCSEC = GAUSSIAN_FWHM_ARCSEC / 2.35
ROW_CHUNK = 2048

AMPLIFIERS = ("LL", "LU", "RL", "RU")
REFERENCE_J = np.arange(40, 71)
EDGE_J = {
    "LL": np.arange(0, 20),
    "RU": np.arange(0, 20),
    "LU": np.arange(92, 112),
    "RL": np.arange(92, 112),
}


def resolved_h5_files(pattern):
    """Use the same nonrecursive quoted-glob convention as the cube builder."""
    return sorted(glob.glob(pattern))


def _text(value):
    if isinstance(value, (bytes, np.bytes_)):
        return value.decode("utf-8", errors="replace").strip()
    return str(value).strip()


def exposure_labels(nrows, nslots):
    """Reproduce quick_reduction's 112-row, interleaved exposure grouping."""
    if nslots <= 0:
        raise ValueError("no IFUSLOT values were found")
    if nrows % (FIBERS_PER_IFU * nslots) != 0:
        raise ValueError(
            "rows=%d is not divisible by 448*nslots=%d; cannot infer "
            "M101 exposure grouping" % (nrows, FIBERS_PER_IFU * nslots))
    nexp = int(nrows / float(FIBERS_PER_IFU * nslots))
    if nexp != EXPECTED_EXPOSURES:
        raise ValueError("expected 3 exposures, inferred %d from rows=%d, "
                         "nslots=%d" % (nexp, nrows, nslots))
    rows = np.arange(nrows, dtype=np.int64)
    return ((rows // FIBERS_PER_AMPLIFIER) % nexp + 1).astype(np.int16), nexp


def build_amplifier_groups(info):
    """Build physical exposure x IFUSLOT x AMP groups in stored row order."""
    nrows = int(info.nrows)
    required = {"ifuslot", "amp", "specid", "ifuid"}
    if not required.issubset(info.colnames):
        raise ValueError("Info must contain ifuslot, amp, specid, and ifuid")
    ifuslot = np.asarray(info.cols.ifuslot[:])
    amp = np.asarray([_text(value) for value in info.cols.amp[:]])
    specid = np.asarray(info.cols.specid[:])
    ifuid = np.asarray(info.cols.ifuid[:])
    labels, nexp = exposure_labels(nrows, len(np.unique(ifuslot)))

    groups = []
    keys = sorted(set(zip(labels.tolist(), ifuslot.tolist(), amp.tolist())),
                  key=lambda key: (int(key[0]), int(key[1]), key[2]))
    for exposure, slot, amplifier in keys:
        indices = np.flatnonzero(
            (labels == exposure) & (ifuslot == slot) & (amp == amplifier))
        if indices.size != FIBERS_PER_AMPLIFIER:
            raise ValueError(
                "%s exposure %d IFUSLOT %s AMP %s has %d rows; expected 112"
                % (getattr(info, "_v_pathname", "H5"), exposure, slot,
                   amplifier, indices.size))
        spec_values = np.unique(specid[indices])
        ifuid_values = np.unique(ifuid[indices])
        if spec_values.size != 1 or ifuid_values.size != 1:
            raise ValueError(
                "inconsistent SPECID/IFUID in exposure %d IFUSLOT %s AMP %s"
                % (exposure, slot, amplifier))
        groups.append({
            "exposure": int(exposure),
            "ifuslot": int(slot),
            "ifuid": int(ifuid_values[0]),
            "specid": int(spec_values[0]),
            "amp": amplifier,
            "indices": indices,
            "j": np.arange(FIBERS_PER_AMPLIFIER, dtype=int),
        })
    return groups, labels, nexp


def load_blank_image(path):
    if path is None:
        return None
    hdul = fits.open(path, memmap=True)
    data = np.asarray(hdul[0].data)
    if data.ndim != 2:
        hdul.close()
        raise ValueError("--image must contain a 2D image")
    return {"hdul": hdul, "data": data, "wcs": WCS(hdul[0].header)}


def _image_blank_selection(image, ra, dec):
    if image is None:
        return np.ones(ra.shape, dtype=bool), None
    x, y = image["wcs"].world_to_pixel_values(ra, dec)
    finite = np.isfinite(x) & np.isfinite(y)
    xi = np.zeros(x.shape, dtype=int)
    yi = np.zeros(y.shape, dtype=int)
    xi[finite] = np.rint(x[finite]).astype(int)
    yi[finite] = np.rint(y[finite]).astype(int)
    valid = (finite &
             (xi >= 0) & (xi < image["data"].shape[1]) &
             (yi >= 0) & (yi < image["data"].shape[0]))
    blank = np.zeros(ra.shape, dtype=bool)
    blank[valid] = np.isfinite(image["data"][yi[valid], xi[valid]]) & (
        image["data"][yi[valid], xi[valid]] < 0.01)
    return valid, blank


def _residual_sky_band(fibers, finite_counts, band_spectrum, ra, dec,
                       labels, image):
    """Reproduce cube residual-sky selection for only the selected band."""
    nrows = int(fibers.nrows)
    n_wave = int(fibers.coldtypes["spectrum"].shape[0])
    dra = ((ra - M101_RA_DEG) * np.cos(np.deg2rad(M101_DEC_DEG)) * 60.0)
    ddec = (dec - M101_DEC_DEG) * 60.0
    sky_region = np.hypot(dra, ddec) > M101_SKY_MIN_RADIUS_ARCMIN
    image_valid, image_blank = _image_blank_selection(image, ra, dec)
    sufficient = finite_counts >= int(np.ceil(
        M101_SKY_MIN_FINITE_FRACTION * n_wave))
    residuals = {}
    selected_counts = {}
    for exposure in range(1, EXPECTED_EXPOSURES + 1):
        selected = (
            (labels == exposure) & sky_region & image_valid & sufficient)
        if image_blank is not None:
            selected &= image_blank
        selected_counts[exposure] = int(selected.sum())
        if selected.sum() < M101_SKY_MIN_FIBERS:
            residuals[exposure] = np.full(
                band_spectrum.shape[1], np.nan, dtype=float)
            continue

        # Check the production full-spectrum residual finite-fraction test.
        candidate_indices = np.flatnonzero(selected)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            candidate_spectrum = np.asarray(
                fibers.read_coordinates(candidate_indices, field="spectrum"),
                dtype=float)
            residual_full = np.asarray(
                biweight(candidate_spectrum, axis=0), dtype=float)
        if np.isfinite(residual_full).sum() / float(n_wave) < (
                M101_SKY_MIN_FINITE_FRACTION):
            residuals[exposure] = np.full(
                band_spectrum.shape[1], np.nan, dtype=float)
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            residuals[exposure] = np.asarray(
                biweight(band_spectrum[selected], axis=0), dtype=float)
    return residuals, selected_counts


def _read_h5_file(path, wave_center, half_width, image):
    with tables.open_file(path, mode="r") as h5:
        if "Info" not in h5.root._v_children or "Fibers" not in h5.root._v_children:
            raise ValueError("%s must contain Info and Fibers tables" % path)
        info = h5.root.Info
        fibers = h5.root.Fibers
        if int(info.nrows) != int(fibers.nrows):
            raise ValueError("%s Info/Fibers row mismatch" % path)
        required = {"spectrum", "skyspectrum", "fiber_to_fiber"}
        if not required.issubset(fibers.colnames):
            raise ValueError("%s Fibers table lacks required columns" % path)
        nrows = int(info.nrows)
        ra = np.asarray(info.cols.ra[:], dtype=float)
        dec = np.asarray(info.cols.dec[:], dtype=float)
        groups, labels, nexp = build_amplifier_groups(info)
        if set(group["amp"] for group in groups) - set(AMPLIFIERS):
            raise ValueError("%s contains an unexpected amplifier label" % path)
        n_wave = int(fibers.coldtypes["spectrum"].shape[0])
        if n_wave != RECTIFIED_WAVE.size:
            raise ValueError("%s Fibers.spectrum has %d bins; expected %d" %
                             (path, n_wave, RECTIFIED_WAVE.size))
        band = ((RECTIFIED_WAVE >= wave_center - half_width) &
                (RECTIFIED_WAVE <= wave_center + half_width))
        band_indices = np.flatnonzero(band)
        if band_indices.size == 0:
            raise ValueError("requested continuum band has no wavelength bins")
        # PyTables vector columns cannot select their inner wavelength axis
        # during the table read. Read bounded row chunks, then retain only the
        # requested continuum band in memory.
        band_spectra = []
        band_skies = []
        band_ftfs = []
        finite_counts = []
        for start in range(0, nrows, ROW_CHUNK):
            stop = min(nrows, start + ROW_CHUNK)
            full_spectrum = np.asarray(
                fibers.read(start=start, stop=stop, field="spectrum"),
                dtype=float)
            band_spectra.append(full_spectrum[:, band_indices])
            finite_counts.append(np.isfinite(full_spectrum).sum(axis=1))
            band_skies.append(np.asarray(
                fibers.read(start=start, stop=stop, field="skyspectrum"),
                dtype=float)[:, band_indices])
            band_ftfs.append(np.asarray(
                fibers.read(start=start, stop=stop, field="fiber_to_fiber"),
                dtype=float)[:, band_indices])
        spectra = np.concatenate(band_spectra, axis=0)
        sky = np.concatenate(band_skies, axis=0)
        ftf = np.concatenate(band_ftfs, axis=0)
        finite_counts = np.concatenate(finite_counts, axis=0)

        offsets = np.array([], dtype=float)
        if "Survey" in h5.root._v_children and "offset" in h5.root.Survey.colnames:
            offsets = np.asarray(h5.root.Survey.cols.offset[:], dtype=float)
        finite_offsets = offsets[np.isfinite(offsets)]
        if finite_offsets.size:
            scale = float(finite_offsets[0])
            if not np.allclose(finite_offsets, scale, rtol=1e-5, atol=1e-7):
                raise ValueError("%s Survey.offset is inconsistent by exposure"
                                 % path)
            sky_scale = scale
            scale_note = "sky_in = spectrum + skyspectrum * Survey.offset"
        else:
            sky_scale = 1.0
            scale_note = ("sky_in = spectrum + skyspectrum; invalid/missing "
                          "Survey.offset means no offset was applied upstream")
        sky_in = spectra + sky * sky_scale

        # Full Fibers.spectrum is read only for the production eligibility
        # fraction; residual-sky candidates are reread only when needed.
        residuals, selected_counts = _residual_sky_band(
            fibers, finite_counts, spectra, ra, dec, labels, image)
        cube_spectra = spectra.copy()
        residual_band_by_row = np.full_like(cube_spectra, np.nan)
        for exposure, residual in residuals.items():
            selected = labels == exposure
            finite_residual = np.isfinite(residual)
            if np.any(finite_residual):
                cube_spectra[np.ix_(
                    selected, finite_residual)] -= residual[finite_residual]
                residual_band_by_row[np.ix_(
                    selected, finite_residual)] = residual[finite_residual]

        records = []
        profile_instances = {amp: [] for amp in AMPLIFIERS}
        max_delta_difference = 0.0
        for group in groups:
            indices = group["indices"]
            f = np.nanmedian(ftf[indices], axis=1)
            k = np.nanmedian(sky[indices] * sky_scale, axis=1)
            s_h5 = np.nanmedian(spectra[indices], axis=1)
            s_cube = np.nanmedian(cube_spectra[indices], axis=1)
            o = np.nanmedian(sky_in[indices], axis=1)
            reference = np.nanmedian(f[REFERENCE_J])
            k_reference = np.nanmedian(k[REFERENCE_J])
            h5_reference = np.nanmedian(s_h5[REFERENCE_J])
            cube_reference = np.nanmedian(s_cube[REFERENCE_J])
            o_reference = np.nanmedian(o[REFERENCE_J])
            delta_h5 = s_h5 - h5_reference
            delta_cube = s_cube - cube_reference
            finite_delta = np.isfinite(delta_h5) & np.isfinite(delta_cube)
            if np.any(finite_delta):
                max_delta_difference = max(
                    max_delta_difference,
                    float(np.max(np.abs(
                        delta_h5[finite_delta] - delta_cube[finite_delta]))))
            edge = EDGE_J[group["amp"]]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                residual_level = np.nanmedian(
                    residual_band_by_row[indices])
            record = {
                "h5": Path(path).name,
                "exposure": group["exposure"],
                "ifuslot": group["ifuslot"],
                "ifuid": group["ifuid"],
                "specid": group["specid"],
                "amp": group["amp"],
                "n_usable_fibers": int(np.isfinite(s_cube).sum()),
                "B": float(np.nanmedian(k[REFERENCE_J])),
                "E_h5": float(np.nanmedian(delta_h5[edge])),
                "E_cube": float(np.nanmedian(delta_cube[edge])),
                "F_edge_contrast": float(
                    np.nanmedian(f[edge]) / reference - 1.0),
                "residual_sky_level": float(residual_level),
                "science_reference": float(h5_reference),
                "sky_reference": float(k_reference),
                "science_edge": float(np.nanmedian(s_cube[edge])),
                "sky_edge": float(np.nanmedian(k[edge])),
                "sky_in_reference": float(o_reference),
                "sky_in_edge": float(np.nanmedian(o[edge])),
                "delta_h5": delta_h5,
                "delta_cube": delta_cube,
                "delta_k": k - k_reference,
                "delta_o": o - o_reference,
                "f_rel": f / reference - 1.0,
            }
            records.append(record)
            profile_instances[group["amp"]].append(record)
        return {
            "path": Path(path), "records": records,
            "profiles": profile_instances,
            "nrows": nrows, "nexp": nexp,
            "nslots": len(np.unique(info.cols.ifuslot[:])),
            "unique_group_sizes": sorted(set(
                len(group["indices"]) for group in groups)),
            "amp_counts": {amp: sum(group["amp"] == amp for group in groups)
                           for amp in AMPLIFIERS},
            "max_delta_difference": max_delta_difference,
            "selected_sky_counts": selected_counts,
            "scale_note": scale_note, "sky_scale": sky_scale,
            "n_band": int(band_indices.size),
        }


def _stack_profile(records, field, folded=False):
    arrays = []
    for record in records:
        value = np.asarray(record[field], dtype=float)
        if folded and record["amp"] in ("LU", "RL"):
            value = value[::-1]
        arrays.append(value)
    if not arrays:
        nan = np.full(FIBERS_PER_AMPLIFIER, np.nan)
        return nan, nan, nan, np.zeros(FIBERS_PER_AMPLIFIER, dtype=int)
    values = np.asarray(arrays)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        median = np.nanmedian(values, axis=0)
        p16, p84 = np.nanpercentile(values, [16, 84], axis=0)
    return median, p16, p84, np.isfinite(values).sum(axis=0)


def write_profile_csv(path, all_profiles):
    fields = ["coordinate", "amplifier", "fiber_number", "median",
              "p16", "p84", "n_instances"]
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for coordinate, by_amp in all_profiles.items():
            for amp, fields_by_value in by_amp.items():
                for j in range(FIBERS_PER_AMPLIFIER):
                    writer.writerow({
                        "coordinate": coordinate, "amplifier": amp,
                        "fiber_number": j,
                        "median": fields_by_value["median"][j],
                        "p16": fields_by_value["p16"][j],
                        "p84": fields_by_value["p84"][j],
                        "n_instances": fields_by_value["n"][j],
                    })


def write_summary_csv(path, records):
    fields = ["h5", "exposure", "ifuslot", "ifuid", "specid", "amp",
              "n_usable_fibers", "B", "E_h5", "E_cube", "F_edge_contrast",
              "residual_sky_level", "science_reference", "sky_reference",
              "science_edge", "sky_edge", "sky_in_reference", "sky_in_edge"]
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: record[field] for field in fields}
                         for record in records)


def _fit_rows(records, amp):
    subset = [record for record in records
              if record["amp"] == amp and np.isfinite(record["B"])
              and np.isfinite(record["E_cube"])]
    x = np.asarray([record["B"] for record in subset])
    y = np.asarray([record["E_cube"] for record in subset])
    if x.size < 3 or np.ptp(x) == 0.0:
        return {"n": int(x.size), "alpha": np.nan, "beta": np.nan,
                "spearman": np.nan}
    fit = theilslopes(y, x)
    rho = spearmanr(x, y).statistic
    return {"n": int(x.size), "alpha": float(fit.intercept),
            "beta": float(fit.slope), "spearman": float(rho)}


def make_figures(output_dir, profiles, records):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    quantities = [
        ("f_rel", "F_rel(j)"),
        ("delta_o", "delta_O (sky-in)"),
        ("delta_k", "delta_K"),
        ("delta_cube", "delta_S"),
    ]
    fig, axes = plt.subplots(4, 4, figsize=(16, 12), sharex=True)
    for column, amp in enumerate(AMPLIFIERS):
        for row, (field, ylabel) in enumerate(quantities):
            median, p16, p84, _ = _stack_profile(profiles[amp], field)
            axis = axes[row, column]
            x = np.arange(FIBERS_PER_AMPLIFIER)
            axis.fill_between(x, p16, p84, alpha=.2)
            axis.plot(x, median, lw=1.2)
            if field == "delta_cube":
                h5_median, _, _, _ = _stack_profile(
                    profiles[amp], "delta_h5")
                axis.plot(x, h5_median, "--", lw=1.0,
                          label="h5" if column == 0 else None)
            if row == 0:
                axis.set_title(amp)
            if column == 0:
                axis.set_ylabel(ylabel)
            axis.axvspan(40, 70, color="0.7", alpha=.18)
            edge = EDGE_J[amp]
            axis.axvspan(edge[0] - .5, edge[-1] + .5,
                         color="tab:red", alpha=.10)
            axis.grid(alpha=.2)
    for axis in axes[-1]:
        axis.set_xlabel("Remedy fiber number j")
    fig.suptitle("Amplifier profiles in Remedy fiber order")
    fig.tight_layout()
    fig.savefig(output_dir / "amplifier_profiles_remedy_order.png", dpi=160)
    plt.close(fig)

    folded_quantities = [
        ("f_rel", "F_rel(q)"),
        ("delta_o", "delta_O(q)"),
        ("delta_k", "delta_K(q)"),
        ("delta_cube", "delta_S_cube(q)"),
    ]
    fig, axes = plt.subplots(4, 1, figsize=(9, 12), sharex=True)
    for axis, (field, ylabel) in zip(axes, folded_quantities):
        combined = []
        for amp in AMPLIFIERS:
            median, p16, p84, _ = _stack_profile(
                profiles[amp], field, folded=True)
            x = np.arange(FIBERS_PER_AMPLIFIER)
            axis.plot(x, median, label=amp, lw=1.0)
            axis.fill_between(x, p16, p84, alpha=.08)
            for record in profiles[amp]:
                value = np.asarray(record[field], dtype=float)
                combined.append(value[::-1] if amp in ("LU", "RL") else value)
        if combined:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                combined_median = np.nanmedian(np.asarray(combined), axis=0)
            axis.plot(x, combined_median, "k--", lw=1.5, label="all amps")
        axis.axvspan(0, 19, color="tab:red", alpha=.10)
        axis.set_ylabel(ylabel)
        axis.grid(alpha=.2)
    axes[0].legend(ncol=5, fontsize=8)
    axes[-1].set_xlabel("folded readout distance q")
    fig.suptitle("Amplifier profiles folded to readout distance")
    fig.tight_layout()
    fig.savefig(output_dir / "amplifier_profiles_readout_distance.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for amp in AMPLIFIERS:
        subset = [record for record in records
                  if record["amp"] == amp and np.isfinite(record["B"])
                  and np.isfinite(record["E_cube"])]
        x = np.asarray([record["B"] for record in subset])
        y = np.asarray([record["E_cube"] for record in subset])
        axes[0].scatter(x, y, s=10, alpha=.5, label=amp)
        positive = np.isfinite(x) & (x != 0.0)
        axes[1].scatter(x[positive], (y / x)[positive], s=10, alpha=.5,
                        label=amp)
    axes[0].axhline(0.0, color="0.4", lw=.8)
    axes[1].axhline(0.0, color="0.4", lw=.8)
    axes[0].set(xlabel="B: reference sky level",
                ylabel="E: edge minus center residual")
    axes[1].set(xlabel="B: reference sky level", ylabel="E / B")
    for axis in axes:
        axis.grid(alpha=.2)
        axis.legend(fontsize=8)
    fig.suptitle("Amplifier edge excess versus sky brightness")
    fig.tight_layout()
    fig.savefig(output_dir / "amplifier_edge_excess_vs_sky.png", dpi=160)
    plt.close(fig)


def _candidate_physical_groups(groups):
    keys = sorted(set((group["ifuslot"], group["amp"]) for group in groups),
                  key=lambda key: (int(key[0]), key[1]))
    by_key = {key: [] for key in keys}
    for group in groups:
        by_key[(group["ifuslot"], group["amp"])].append(group)
    if any(len(value) != EXPECTED_EXPOSURES for value in by_key.values()):
        raise ValueError("candidate FTF requires three exposure copies for "
                         "every physical IFUSLOT/amplifier")
    return keys, by_key


def _broad_candidate_correction(master, wave):
    """Make a deliberately broad, positive, center-normalized correction."""
    nphys, nfiber, n_wave = master.shape
    n_bins = 8
    edges = np.linspace(0, n_wave, n_bins + 1, dtype=int)
    reference = np.nanmedian(master[:, REFERENCE_J, :], axis=1)
    c_raw = np.full(master.shape, np.nan, dtype=np.float32)
    np.divide(master, reference[:, None, :], out=c_raw,
              where=np.isfinite(reference[:, None, :])
              & (reference[:, None, :] != 0.0))
    correction = np.ones(master.shape, dtype=np.float32)
    for p in range(nphys):
        for j in range(nfiber):
            values = c_raw[p, j]
            centers = []
            broad = []
            for b in range(n_bins):
                lo, hi = edges[b], edges[b + 1]
                finite = np.isfinite(values[lo:hi]) & (values[lo:hi] > 0.0)
                if finite.sum() >= max(3, (hi - lo) // 4):
                    broad.append(float(np.nanmedian(values[lo:hi][finite])))
                    centers.append(float(np.nanmedian(wave[p, j, lo:hi])))
            if len(broad) >= 2:
                valid = np.isfinite(centers) & np.isfinite(broad)
                if valid.sum() >= 2:
                    correction[p, j] = np.interp(
                        wave[p, j], np.asarray(centers)[valid],
                        np.asarray(broad)[valid],
                        left=broad[np.flatnonzero(valid)[0]],
                        right=broad[np.flatnonzero(valid)[-1]])
            elif len(broad) == 1:
                correction[p, j] = broad[0]
    center_correction = np.nanmedian(
        correction[:, REFERENCE_J, :], axis=1)
    for p in range(nphys):
        good = np.isfinite(center_correction[p]) & (
            center_correction[p] != 0.0)
        normalizer = np.ones(n_wave, dtype=np.float32)
        normalizer[good] = center_correction[p, good]
        correction[p] /= normalizer[None, :]
    correction[~np.isfinite(correction) | (correction <= 0.0)] = 1.0
    return correction


def _candidate_read_h5(path, wave_center, half_width):
    """Derive C from Raw.mscispectrum and test it on Raw.spectrum."""
    with tables.open_file(path, mode="r") as h5:
        if not {"Info", "Raw", "Fibers"}.issubset(h5.root._v_children):
            raise ValueError("%s must contain Info, Raw, and Fibers" % path)
        info = h5.root.Info
        raw = h5.root.Raw
        fibers = h5.root.Fibers
        if int(info.nrows) != int(raw.nrows) or int(info.nrows) != int(fibers.nrows):
            raise ValueError("%s Info/Raw/Fibers row mismatch" % path)
        raw_required = {"mscispectrum", "spectrum", "wave"}
        fiber_required = {"fiber_to_fiber"}
        if not raw_required.issubset(raw.colnames):
            raise ValueError("%s Raw table lacks candidate columns" % path)
        if not fiber_required.issubset(fibers.colnames):
            raise ValueError("%s Fibers table lacks fiber_to_fiber" % path)
        nrows = int(raw.nrows)
        n_wave = int(raw.coldtypes["mscispectrum"].shape[0])
        if n_wave != 1032:
            raise ValueError("%s Raw native spectrum has %d bins; expected 1032"
                             % (path, n_wave))
        if int(raw.coldtypes["wave"].shape[0]) != n_wave:
            raise ValueError("%s Raw wave/spectrum shapes disagree" % path)
        if int(fibers.coldtypes["fiber_to_fiber"].shape[0]) != 1036:
            raise ValueError("%s Fibers FTF has unexpected length" % path)

        info_rows = {
            "ifuslot": np.asarray(info.cols.ifuslot[:]),
            "amp": np.asarray([_text(value) for value in info.cols.amp[:]]),
        }
        groups, labels, nexp = build_amplifier_groups(info)
        physical_keys, groups_by_key = _candidate_physical_groups(groups)
        physical_id = {key: number for number, key
                       in enumerate(physical_keys)}
        row_pid = np.empty(nrows, dtype=np.int32)
        row_j = np.empty(nrows, dtype=np.int16)
        for key, physical_groups in groups_by_key.items():
            pid = physical_id[key]
            for group in physical_groups:
                row_pid[group["indices"]] = pid
                row_j[group["indices"]] = group["j"]
        nphys = len(physical_keys)
        master = np.full((nphys, FIBERS_PER_AMPLIFIER, n_wave),
                         np.nan, dtype=np.float32)
        master_wave = np.full_like(master, np.nan)
        difference_sum = 0.0
        difference_count = 0
        difference_max = 0.0
        for start in range(0, nrows, ROW_CHUNK):
            stop = min(nrows, start + ROW_CHUNK)
            mscispectrum = np.asarray(
                raw.read(start=start, stop=stop, field="mscispectrum"),
                dtype=np.float32)
            wave = np.asarray(
                raw.read(start=start, stop=stop, field="wave"),
                dtype=np.float32)
            ids = np.arange(start, stop)
            first = labels[ids] == 1
            if np.any(first):
                master[row_pid[ids[first]], row_j[ids[first]]] = (
                    mscispectrum[first])
                master_wave[row_pid[ids[first]], row_j[ids[first]]] = (
                    wave[first])
            repeated = ~first
            if np.any(repeated):
                reference = master[row_pid[ids[repeated]],
                                   row_j[ids[repeated]]]
                values = mscispectrum[repeated]
                valid = np.isfinite(values) & np.isfinite(reference)
                differences = np.abs(values - reference)
                differences = differences[valid]
                if differences.size:
                    difference_sum += float(differences.sum())
                    difference_count += int(differences.size)
                    difference_max = max(difference_max,
                                         float(differences.max()))
        band = ((RECTIFIED_WAVE >= wave_center - half_width) &
                (RECTIFIED_WAVE <= wave_center + half_width))
        band_indices = np.flatnonzero(band)
        if band_indices.size == 0:
            raise ValueError("requested continuum band has no wavelength bins")
        correction = _broad_candidate_correction(master, master_wave)
        c_rect = np.ones((nphys, FIBERS_PER_AMPLIFIER,
                          band_indices.size), dtype=np.float32)
        for p in range(nphys):
            for j in range(FIBERS_PER_AMPLIFIER):
                c_rect[p, j] = np.interp(
                    RECTIFIED_WAVE[band_indices], master_wave[p, j],
                    correction[p, j],
                    left=1.0, right=1.0)
        raw_legacy = np.full(nrows, np.nan, dtype=float)
        raw_candidate = np.full(nrows, np.nan, dtype=float)
        ftf_legacy = np.full(nrows, np.nan, dtype=float)
        ftf_candidate = np.full(nrows, np.nan, dtype=float)
        for start in range(0, nrows, ROW_CHUNK):
            stop = min(nrows, start + ROW_CHUNK)
            spectrum = np.asarray(
                raw.read(start=start, stop=stop, field="spectrum"),
                dtype=float)
            wave = np.asarray(
                raw.read(start=start, stop=stop, field="wave"),
                dtype=float)
            ftf = np.asarray(
                fibers.read(start=start, stop=stop,
                            field="fiber_to_fiber"), dtype=float)
            ids = np.arange(start, stop)
            for local, row in enumerate(ids):
                pid = row_pid[row]
                j = row_j[row]
                c_native = np.interp(
                    wave[local], master_wave[pid, j], correction[pid, j],
                    left=1.0, right=1.0)
                native_band = ((wave[local] >= wave_center - half_width) &
                               (wave[local] <= wave_center + half_width))
                raw_legacy[row] = np.nanmedian(
                    spectrum[local, native_band])
                raw_candidate[row] = np.nanmedian(
                    spectrum[local, native_band] /
                    c_native[native_band])
                ftf_legacy[row] = np.nanmedian(ftf[local, band_indices])
                ftf_candidate[row] = np.nanmedian(
                    ftf[local, band_indices] * c_rect[pid, j])

        candidate_records = []
        cal_profiles = {amp: [] for amp in AMPLIFIERS}
        raw_profiles = {amp: [] for amp in AMPLIFIERS}
        for key, physical_groups in groups_by_key.items():
            pid = physical_id[key]
            master_c = np.nanmedian(c_rect[pid], axis=1)
            cal_profiles[key[1]].append({
                "C": master_c,
                "legacy_ftf": ftf_legacy[
                    physical_groups[0]["indices"]].copy(),
                "candidate_ftf": ftf_candidate[
                    physical_groups[0]["indices"]].copy(),
                "C_minus_1": master_c - 1.0,
            })
            for group in physical_groups:
                indices = group["indices"]
                f = ftf_legacy[indices]
                fc = ftf_candidate[indices]
                s = raw_legacy[indices]
                sc = raw_candidate[indices]
                reference_f = np.nanmedian(f[REFERENCE_J])
                reference_fc = np.nanmedian(fc[REFERENCE_J])
                center_s = np.nanmedian(s[REFERENCE_J])
                center_sc = np.nanmedian(sc[REFERENCE_J])
                edge = EDGE_J[group["amp"]]
                e_s = np.nanmedian(s[edge]) - center_s
                e_sc = np.nanmedian(sc[edge]) - center_sc
                candidate_records.append({
                    "h5": Path(path).name,
                    "exposure": group["exposure"],
                    "specid": group["specid"],
                    "ifuslot": group["ifuslot"],
                    "ifuid": group["ifuid"],
                    "amp": group["amp"],
                    "legacy_ftf_edge_contrast": np.nanmedian(f[edge]) /
                    reference_f - 1.0,
                    "candidate_ftf_edge_contrast": np.nanmedian(fc[edge]) /
                    reference_fc - 1.0,
                    "median_C_edge": np.nanmedian(
                        c_rect[pid, edge]),
                    "median_C_center": np.nanmedian(
                        c_rect[pid, REFERENCE_J]),
                    "E_raw_legacy": e_s,
                    "E_raw_candidate": e_sc,
                    "fractional_E_raw_legacy": (
                        e_s / center_s if np.isfinite(center_s)
                        and center_s != 0.0 else np.nan),
                    "fractional_E_raw_candidate": (
                        e_sc / center_sc if np.isfinite(center_sc)
                        and center_sc != 0.0 else np.nan),
                    "B": np.nanmedian(
                        np.asarray(c_rect[pid, REFERENCE_J])),
                    "science_reference": center_s,
                    "science_candidate_reference": center_sc,
                    "science_edge": np.nanmedian(s[edge]),
                    "science_candidate_edge": np.nanmedian(sc[edge]),
                })
                raw_profiles[group["amp"]].append({
                    "legacy": s - center_s,
                    "candidate": sc - center_sc,
                })
        return {
            "records": candidate_records,
            "cal_profiles": cal_profiles,
            "raw_profiles": raw_profiles,
            "repeat_mean_abs": (difference_sum / difference_count
                                if difference_count else np.nan),
            "repeat_max_abs": difference_max,
            "n_repeat_values": difference_count,
            "n_band": int(band_indices.size),
            "nexp": nexp,
            "nphys": nphys,
            "physical_keys": physical_keys,
        }


def _candidate_profile_rows(cal_profiles, raw_profiles):
    fields = ("C", "C_minus_1", "legacy_ftf", "candidate_ftf")
    rows = []
    for amp in AMPLIFIERS:
        for field in fields:
            arrays = [item[field] for item in cal_profiles[amp]]
            values = np.asarray(arrays, dtype=float)
            if field == "legacy_ftf":
                values = values / np.nanmedian(values[:, REFERENCE_J], axis=1)[:, None]
            elif field == "candidate_ftf":
                values = values / np.nanmedian(values[:, REFERENCE_J], axis=1)[:, None]
            if amp in ("LU", "RL"):
                values = values[:, ::-1]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                median = np.nanmedian(values, axis=0)
                p16, p84 = np.nanpercentile(values, [16, 84], axis=0)
            for q in range(FIBERS_PER_AMPLIFIER):
                rows.append({
                    "coordinate": "q", "amplifier": amp,
                    "profile": field, "q": q,
                    "median": median[q], "p16": p16[q], "p84": p84[q],
                    "n_instances": int(np.isfinite(values[:, q]).sum()),
                })
        for field in ("legacy", "candidate"):
            arrays = [item[field] for item in raw_profiles[amp]]
            values = np.asarray(arrays, dtype=float)
            if amp in ("LU", "RL"):
                values = values[:, ::-1]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                median = np.nanmedian(values, axis=0)
                p16, p84 = np.nanpercentile(values, [16, 84], axis=0)
            for q in range(FIBERS_PER_AMPLIFIER):
                rows.append({
                    "coordinate": "q", "amplifier": amp,
                    "profile": "raw_" + field, "q": q,
                    "median": median[q], "p16": p16[q], "p84": p84[q],
                    "n_instances": int(np.isfinite(values[:, q]).sum()),
                })
    return rows


def write_candidate_summary(path, records):
    fields = ["h5", "exposure", "specid", "ifuslot", "ifuid", "amp",
              "legacy_ftf_edge_contrast", "candidate_ftf_edge_contrast",
              "median_C_edge", "median_C_center", "E_raw_legacy",
              "E_raw_candidate", "fractional_E_raw_legacy",
              "fractional_E_raw_candidate", "B", "science_reference",
              "science_candidate_reference", "science_edge",
              "science_candidate_edge"]
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: record[field] for field in fields}
                         for record in records)


def write_candidate_profiles(path, rows):
    fields = ["coordinate", "amplifier", "profile", "q", "median",
              "p16", "p84", "n_instances"]
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def make_candidate_figures(output_dir, cal_profiles, raw_profiles):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = np.arange(FIBERS_PER_AMPLIFIER)
    fig, axes = plt.subplots(3, 1, figsize=(10, 11), sharex=True)
    colors = dict(zip(AMPLIFIERS, ("tab:blue", "tab:orange",
                                   "tab:green", "tab:red")))
    for amp in AMPLIFIERS:
        c = np.asarray([item["C"] for item in cal_profiles[amp]])
        f = np.asarray([item["legacy_ftf"] for item in cal_profiles[amp]])
        fc = np.asarray([item["candidate_ftf"] for item in cal_profiles[amp]])
        c = c[:, ::-1] if amp in ("LU", "RL") else c
        if amp in ("LU", "RL"):
            f = f[:, ::-1]
            fc = fc[:, ::-1]
        f = f / np.nanmedian(f[:, REFERENCE_J], axis=1)[:, None]
        fc = fc / np.nanmedian(fc[:, REFERENCE_J], axis=1)[:, None]
        for axis, values, label in (
                (axes[0], c, amp),
                (axes[1], f, amp + " legacy"),
                (axes[1], fc, amp + " candidate")):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                median = np.nanmedian(values, axis=0)
                p16, p84 = np.nanpercentile(values, [16, 84], axis=0)
            axis.plot(x, median, color=colors[amp],
                      linestyle="--" if "candidate" in label else "-",
                      label=label, lw=1.0)
            if axis is axes[0]:
                axis.fill_between(x, p16, p84, color=colors[amp], alpha=.10)
    c_values = []
    for amp in AMPLIFIERS:
        values = np.asarray([item["C"] for item in cal_profiles[amp]])
        values = values[:, ::-1] if amp in ("LU", "RL") else values
        c_values.append(values)
    if c_values:
        axes[0].plot(x, np.nanmedian(np.vstack(c_values), axis=0),
                     "k--", lw=1.5, label="all amps")
    axes[2].plot(x, np.nanmedian(np.vstack(c_values), axis=0) - 1.0,
                 "k-", lw=1.5, label="combined C-1")
    axes[0].set_ylabel("C(q) at 4600 A")
    axes[1].set_ylabel("relative FTF")
    axes[2].set_ylabel("C(q)-1")
    axes[2].set_xlabel("folded readout distance q")
    for axis in axes:
        axis.axvspan(0, 19, color="tab:red", alpha=.10)
        axis.grid(alpha=.2)
    axes[0].legend(ncol=5, fontsize=7)
    axes[1].legend(ncol=4, fontsize=7)
    fig.suptitle("Master-science residual response and candidate FTF")
    fig.tight_layout()
    fig.savefig(output_dir / "master_science_ftf_correction_readout.png",
                dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 5, figsize=(18, 4), sharey=True)
    all_legacy, all_candidate = [], []
    for column, amp in enumerate(AMPLIFIERS):
        for field, style, label in (("legacy", "-", "legacy"),
                                    ("candidate", "--", "candidate")):
            values = np.asarray([item[field] for item in raw_profiles[amp]])
            values = values[:, ::-1] if amp in ("LU", "RL") else values
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                median = np.nanmedian(values, axis=0)
                p16, p84 = np.nanpercentile(values, [16, 84], axis=0)
            axes[column].fill_between(x, p16, p84, alpha=.15)
            axes[column].plot(x, median, style, label=label)
            (all_legacy if field == "legacy" else all_candidate).extend(values)
        axes[column].set_title(amp)
        axes[column].axvspan(0, 19, color="tab:red", alpha=.10)
        axes[column].grid(alpha=.2)
        axes[column].legend(fontsize=8)
    axes[0].set_ylabel("Raw spectrum minus center")
    axes[0].set_xlabel("q")
    for axis in axes[1:]:
        axis.set_xlabel("q")
    if all_legacy and all_candidate:
        axes[4].cla()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            axes[4].plot(x, np.nanmedian(np.asarray(all_legacy), axis=0),
                         label="legacy")
            axes[4].plot(x, np.nanmedian(np.asarray(all_candidate), axis=0),
                         "--", label="candidate")
        axes[4].set_title("all amps")
        axes[4].axvspan(0, 19, color="tab:red", alpha=.10)
        axes[4].grid(alpha=.2)
        axes[4].legend(fontsize=8)
    fig.suptitle("Independent Raw science before/after candidate FTF")
    fig.tight_layout()
    fig.savefig(output_dir / "raw_science_candidate_ftf_before_after.png",
                dpi=160)
    plt.close(fig)


def run_candidate_ftf(files, output_dir, wave_center, half_width):
    results = [_candidate_read_h5(path, wave_center, half_width)
               for path in files]
    records = [record for result in results for record in result["records"]]
    cal_profiles = {amp: [item for result in results
                          for item in result["cal_profiles"][amp]]
                    for amp in AMPLIFIERS}
    raw_profiles = {amp: [item for result in results
                          for item in result["raw_profiles"][amp]]
                    for amp in AMPLIFIERS}
    profile_rows = _candidate_profile_rows(cal_profiles, raw_profiles)
    write_candidate_summary(output_dir / "candidate_ftf_summary.csv", records)
    write_candidate_profiles(output_dir / "candidate_ftf_profiles.csv",
                             profile_rows)
    make_candidate_figures(output_dir, cal_profiles, raw_profiles)

    print("")
    print("Candidate FTF experiment:")
    print("  repeated master-science copies: mean/max absolute disagreement "
          "%.6g / %.6g over %d finite values" %
          (np.nanmean([result["repeat_mean_abs"] for result in results]),
           np.nanmax([result["repeat_max_abs"] for result in results]),
           sum(result["n_repeat_values"] for result in results)))
    for amp in AMPLIFIERS:
        subset = [record for record in records if record["amp"] == amp]
        print("  %s: median C(edge)=%.6g, C(center)=%.6g, "
              "legacy/candidate FTF edge contrast=%.6g/%.6g" %
              (amp, np.nanmedian([record["median_C_edge"] for record in subset]),
               np.nanmedian([record["median_C_center"] for record in subset]),
               np.nanmedian([record["legacy_ftf_edge_contrast"]
                             for record in subset]),
               np.nanmedian([record["candidate_ftf_edge_contrast"]
                             for record in subset])))
    legacy = np.asarray([record["E_raw_legacy"] for record in records])
    candidate = np.asarray([record["E_raw_candidate"] for record in records])
    finite = np.isfinite(legacy) & np.isfinite(candidate)
    legacy_median = np.nanmedian(legacy)
    candidate_median = np.nanmedian(candidate)
    reduction = (100.0 * (1.0 - abs(candidate_median) / abs(legacy_median))
                 if legacy_median != 0.0 else np.nan)
    print("  actual Raw science: median E legacy/candidate=%.6g/%.6g; "
          "|E| change=%.6g%%" %
          (legacy_median, candidate_median, reduction))
    print("  combined fractional E legacy/candidate=%.6g/%.6g" %
          (np.nanmedian([record["fractional_E_raw_legacy"]
                         for record in records]),
           np.nanmedian([record["fractional_E_raw_candidate"]
                         for record in records])))
    print("  master-science correction positive at expected readout edge: %s"
          % ("YES" if np.nanmedian(
              [record["median_C_edge"] - record["median_C_center"]
               for record in records]) > 0.0 else "NO"))
    print("  M101 readout-edge residual reduced by candidate correction: %s"
          % ("YES" if np.nanmedian(np.abs(candidate[finite])) <
             np.nanmedian(np.abs(legacy[finite])) else "NO"))


SPECTRAL_Q_BINS = ((0, 9), (10, 19), (20, 29), (30, 39),
                   (40, 59), (60, 79), (80, 111))
SPECTRAL_CONTINUUM_BINS = 30
SPECTRAL_RAW_CHUNK = 2048


def _broad_continuum_1d(values, wave, n_bins=SPECTRAL_CONTINUUM_BINS):
    """Robust broad continuum using only a small number of wide bins."""
    values = np.asarray(values, dtype=float)
    wave = np.asarray(wave, dtype=float)
    edges = np.linspace(0, values.size, n_bins + 1, dtype=int)
    centers = []
    medians = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        segment = values[lo:hi]
        finite = np.isfinite(segment)
        if finite.sum() >= max(3, (hi - lo) // 4):
            centers.append(float(np.nanmedian(wave[lo:hi])))
            medians.append(float(np.nanmedian(segment[finite])))
    result = np.full(values.shape, np.nan, dtype=float)
    if len(medians) >= 2:
        result = np.interp(wave, np.asarray(centers), np.asarray(medians))
    elif len(medians) == 1:
        result[:] = medians[0]
    return result


def _robust_zero_slope(x, y):
    finite = np.isfinite(x) & np.isfinite(y)
    x = np.asarray(x)[finite]
    y = np.asarray(y)[finite]
    if x.size < 5 or np.sum(x * x) == 0.0:
        return np.nan, np.full(y.shape, np.nan), finite
    beta = float(np.sum(x * y) / np.sum(x * x))
    for _ in range(3):
        residual = y - beta * x
        center = np.median(residual)
        scale = 1.4826 * np.median(np.abs(residual - center))
        if not np.isfinite(scale) or scale == 0.0:
            break
        keep = np.abs(residual - center) <= 4.0 * scale
        if keep.sum() < 5 or np.sum(x[keep] * x[keep]) == 0.0:
            break
        beta = float(np.sum(x[keep] * y[keep]) /
                     np.sum(x[keep] * x[keep]))
    return beta, y - beta * x, finite


def _robust_rms(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan
    return float(1.4826 * np.median(np.abs(values - np.median(values))))


def _spectral_fit(difference, sky, wave):
    wave = np.asarray(wave, dtype=float)
    d_hp = difference - _broad_continuum_1d(difference, wave)
    s_hp = sky - _broad_continuum_1d(sky, wave)
    beta, _, finite = _robust_zero_slope(s_hp, d_hp)
    if np.sum(finite) >= 3:
        x = s_hp[finite]
        y = d_hp[finite]
        residual = y - beta * x if np.isfinite(beta) else y
        pearson = np.corrcoef(x, y)[0, 1] if np.std(x) and np.std(y) else np.nan
        spearman = spearmanr(x, y).statistic
    else:
        residual = np.array([], dtype=float)
        pearson = np.nan
        spearman = np.nan
    return {
        "D": difference, "D_hp": d_hp, "S": sky, "S_hp": s_hp,
        "beta": float(beta), "pearson": float(pearson),
        "spearman": float(spearman),
        "rms_before": _robust_rms(d_hp),
        "rms_after": _robust_rms(residual),
        "residual": (d_hp - beta * s_hp
                     if np.isfinite(beta) else np.full(d_hp.shape, np.nan)),
    }


def _spectral_native_rows(path, exposure, image):
    """Read one exposure's blank-sky Raw spectra and interpolate once."""
    with tables.open_file(path, mode="r") as h5:
        if not {"Info", "Raw"}.issubset(h5.root._v_children):
            raise ValueError("%s must contain Info and Raw tables" % path)
        info = h5.root.Info
        raw = h5.root.Raw
        if int(info.nrows) != int(raw.nrows):
            raise ValueError("%s Info/Raw row mismatch" % path)
        if not {"spectrum", "wave"}.issubset(raw.colnames):
            raise ValueError("%s Raw lacks spectrum/wave" % path)
        groups, labels, nexp = build_amplifier_groups(info)
        if exposure < 1 or exposure > nexp:
            raise ValueError("requested exposure %d but %s has %d exposures" %
                             (exposure, path, nexp))
        ra = np.asarray(info.cols.ra[:], dtype=float)
        dec = np.asarray(info.cols.dec[:], dtype=float)
        dra = ((ra - M101_RA_DEG) * np.cos(np.deg2rad(M101_DEC_DEG)) * 60.0)
        ddec = (dec - M101_DEC_DEG) * 60.0
        radial_blank = np.hypot(dra, ddec) > M101_SKY_MIN_RADIUS_ARCMIN
        image_valid, image_blank = _image_blank_selection(image, ra, dec)
        if image_blank is None:
            raise ValueError("the spectral sky test requires --image")
        row_j = np.full(int(info.nrows), -1, dtype=np.int16)
        row_amp = np.full(int(info.nrows), "", dtype="U2")
        for group in groups:
            row_j[group["indices"]] = group["j"]
            row_amp[group["indices"]] = group["amp"]
        candidate = ((labels == exposure) & radial_blank & image_valid &
                     image_blank)
        candidate_indices = np.flatnonzero(candidate)
        if candidate_indices.size == 0:
            raise ValueError("no image-selected blank-sky fibers in exposure %d"
                             % exposure)

        native_by_amp = {amp: {"wave": [], "spectrum": [], "j": []}
                         for amp in AMPLIFIERS}
        n_wave = int(raw.coldtypes["spectrum"].shape[0])
        minimum_finite = int(np.ceil(
            M101_SKY_MIN_FINITE_FRACTION * n_wave))
        for start in range(0, candidate_indices.size, SPECTRAL_RAW_CHUNK):
            rows = candidate_indices[start:start + SPECTRAL_RAW_CHUNK]
            spectrum = np.asarray(
                raw.read_coordinates(rows, field="spectrum"), dtype=float)
            wave = np.asarray(
                raw.read_coordinates(rows, field="wave"), dtype=float)
            keep = np.isfinite(spectrum).sum(axis=1) >= minimum_finite
            for amp in AMPLIFIERS:
                selected = keep & (row_amp[rows] == amp)
                if np.any(selected):
                    native_by_amp[amp]["wave"].append(wave[selected])
                    native_by_amp[amp]["spectrum"].append(spectrum[selected])
                    native_by_amp[amp]["j"].append(row_j[rows[selected]])
        all_waves = [value for item in native_by_amp.values()
                     for value in item["wave"]]
        if not all_waves:
            raise ValueError("no blank-sky fibers meet the finite-spectrum criterion")
        sample_waves = np.vstack(all_waves)[:256]
        common_wave = np.nanmedian(sample_waves, axis=0)
        spectra_by_amp = {}
        for amp, item in native_by_amp.items():
            if not item["wave"]:
                spectra_by_amp[amp] = {
                    "spectrum": np.empty((0, n_wave)),
                    "j": np.empty(0, dtype=int)}
                continue
            waves = np.vstack(item["wave"])
            spectra = np.vstack(item["spectrum"])
            rectified = np.full(spectra.shape, np.nan, dtype=float)
            for index in range(spectra.shape[0]):
                rectified[index] = np.interp(
                    common_wave, waves[index], spectra[index],
                    left=np.nan, right=np.nan)
            spectra_by_amp[amp] = {
                "spectrum": rectified,
                "j": np.concatenate(item["j"]).astype(int),
            }
        counts = {
            amp: int(spectra_by_amp[amp]["j"].size) for amp in AMPLIFIERS}
        return {
            "wave": common_wave, "by_amp": spectra_by_amp,
            "counts": counts, "groups": groups, "nexp": nexp,
            "candidate_count": int(candidate_indices.size),
        }


def _spectral_reference_profiles(data):
    global _spectral_wave
    _spectral_wave = np.asarray(data["wave"], dtype=float)
    by_amp = data["by_amp"]
    central_rows = []
    for amp in AMPLIFIERS:
        values = by_amp[amp]["spectrum"]
        j = by_amp[amp]["j"]
        selected = np.isin(j, REFERENCE_J)
        if np.any(selected):
            central_rows.append(values[selected])
    if not central_rows:
        raise ValueError("no blank-sky fibers in central reference region")
    global_sky = np.nanmedian(np.vstack(central_rows), axis=0)
    results = {}
    for amp in AMPLIFIERS:
        values = by_amp[amp]["spectrum"]
        j = by_amp[amp]["j"]
        q = j if amp in ("LL", "RU") else 111 - j
        central = np.nanmedian(values[np.isin(j, REFERENCE_J)], axis=0)
        edge = np.nanmedian(values[(q >= 0) & (q <= 19)], axis=0)
        edge_fit = _spectral_fit(edge - central, central, _spectral_wave)
        edge_global_fit = _spectral_fit(
            edge - central, global_sky, _spectral_wave)
        q_fits = []
        for q_min, q_max in SPECTRAL_Q_BINS:
            selected = (q >= q_min) & (q <= q_max)
            difference = np.nanmedian(values[selected], axis=0) - central
            fit = _spectral_fit(difference, central, _spectral_wave)
            fit_global = _spectral_fit(difference, global_sky, _spectral_wave)
            q_fits.append({
                "q_min": q_min, "q_max": q_max,
                "N_fibers": int(selected.sum()), "fit": fit,
                "fit_global": fit_global,
            })
        results[amp] = {
            "edge": edge_fit, "edge_global": edge_global_fit,
            "q_fits": q_fits, "central": central, "edge_spectrum": edge,
            "global": global_sky,
        }
    return results


def _write_spectral_outputs(output_dir, h5_name, exposure, data, profiles):
    summary_fields = [
        "h5", "exposure", "amplifier", "region", "q_min", "q_max",
        "N_fibers", "beta", "beta_global", "Pearson_r", "Spearman_rho",
        "robust_rms_before", "robust_rms_after",
    ]
    summary_rows = []
    for amp in AMPLIFIERS:
        edge = profiles[amp]["edge"]
        edge_global = profiles[amp]["edge_global"]
        j = data["by_amp"][amp]["j"]
        q = j if amp in ("LL", "RU") else 111 - j
        summary_rows.append({
            "h5": h5_name, "exposure": exposure, "amplifier": amp,
            "region": "edge", "q_min": 0, "q_max": 19,
            "N_fibers": int(((q >= 0) & (q <= 19)).sum()),
            "beta": edge["beta"], "beta_global": edge_global["beta"],
            "Pearson_r": edge["pearson"], "Spearman_rho": edge["spearman"],
            "robust_rms_before": edge["rms_before"],
            "robust_rms_after": edge["rms_after"],
        })
        for qfit in profiles[amp]["q_fits"]:
            fit = qfit["fit"]
            fit_global = qfit["fit_global"]
            summary_rows.append({
                "h5": h5_name, "exposure": exposure, "amplifier": amp,
                "region": "qbin", "q_min": qfit["q_min"],
                "q_max": qfit["q_max"], "N_fibers": qfit["N_fibers"],
                "beta": fit["beta"], "beta_global": fit_global["beta"],
                "Pearson_r": fit["pearson"],
                "Spearman_rho": fit["spearman"],
                "robust_rms_before": fit["rms_before"],
                "robust_rms_after": fit["rms_after"],
            })
    with (output_dir / "sky_spectral_edge_fit_summary.csv").open(
            "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(summary_rows)

    spectral_fields = ["h5", "exposure", "wavelength", "amplifier",
                       "D", "D_hp", "S", "S_hp", "beta_S_hp",
                       "residual_after_fit"]
    with (output_dir / "edge_excess_spectra.csv").open(
            "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=spectral_fields)
        writer.writeheader()
        for amp in AMPLIFIERS:
            fit = profiles[amp]["edge"]
            for wave, values in zip(data["wave"], zip(
                    fit["D"], fit["D_hp"], fit["S"], fit["S_hp"],
                    fit["S_hp"] * fit["beta"], fit["residual"])):
                writer.writerow({
                    "h5": h5_name, "exposure": exposure,
                    "wavelength": wave, "amplifier": amp,
                    "D": values[0], "D_hp": values[1], "S": values[2],
                    "S_hp": values[3], "beta_S_hp": values[4],
                    "residual_after_fit": values[5],
                })
    return summary_rows


def _spectral_feature_centers(wave, spectrum, count=4):
    finite = np.isfinite(spectrum)
    if not np.any(finite):
        return []
    order = np.argsort(np.abs(np.where(finite, spectrum, 0.0)))[::-1]
    selected = []
    for index in order:
        if all(abs(int(index) - previous) > 12 for previous in selected):
            selected.append(int(index))
        if len(selected) == count:
            break
    return [float(wave[index]) for index in sorted(selected)]


def _robust_ylim(*arrays, pad_fraction=0.2):
    values = []
    for array in arrays:
        finite = np.asarray(array, dtype=float)
        values.append(finite[np.isfinite(finite)])
    values = np.concatenate([value for value in values if value.size]) \
        if any(value.size for value in values) else np.array([], dtype=float)
    if values.size == 0:
        return -1.0, 1.0
    p01, p99 = np.percentile(values, [1.0, 99.0])
    span = p99 - p01
    if not np.isfinite(span) or span <= 0.0:
        scale = max(abs(float(p01)), 1.0)
        return float(p01 - 0.05 * scale), float(p99 + 0.05 * scale)
    return (float(p01 - pad_fraction * span),
            float(p99 + pad_fraction * span))


def make_edge_center_blank_sky_figure(output_dir, wave, profiles):
    """Plot the actual blank-sky edge/center spectra used by the test."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    amp_profiles = {}
    for amp in AMPLIFIERS:
        profile = profiles[amp]
        central = np.asarray(profile["central"], dtype=float)
        edge_fit = profile["edge"]
        edge = np.asarray(profile["edge_spectrum"], dtype=float)
        difference = np.asarray(edge_fit["D"], dtype=float)
        # Reuse the exact broad component already made for D_hp in the
        # spectral-sky experiment; this figure adds no new smoothing rule.
        smooth_difference = difference - np.asarray(edge_fit["D_hp"],
                                                     dtype=float)
        safe_scale = max(1.0, np.nanmax(np.abs(central))) \
            if np.any(np.isfinite(central)) else 1.0
        ratio = np.full(central.shape, np.nan, dtype=float)
        valid = (np.isfinite(edge) & np.isfinite(central) &
                 (np.abs(central) > np.finfo(float).eps * safe_scale))
        ratio[valid] = edge[valid] / central[valid] - 1.0
        amp_profiles[amp] = {
            "edge": edge, "center": central, "difference": difference,
            "smooth_difference": smooth_difference, "ratio": ratio,
        }

    combined_edge = np.nanmedian(np.vstack([
        amp_profiles[amp]["edge"] for amp in AMPLIFIERS]), axis=0)
    combined_center = np.nanmedian(np.vstack([
        amp_profiles[amp]["center"] for amp in AMPLIFIERS]), axis=0)
    combined_difference = combined_edge - combined_center
    combined_smooth = _broad_continuum_1d(combined_difference, wave)
    combined_ratio = np.full(combined_center.shape, np.nan, dtype=float)
    safe_scale = max(1.0, np.nanmax(np.abs(combined_center))) \
        if np.any(np.isfinite(combined_center)) else 1.0
    valid = (np.isfinite(combined_edge) & np.isfinite(combined_center) &
             (np.abs(combined_center) > np.finfo(float).eps * safe_scale))
    combined_ratio[valid] = combined_edge[valid] / combined_center[valid] - 1.0
    amp_profiles["all amps"] = {
        "edge": combined_edge, "center": combined_center,
        "difference": combined_difference,
        "smooth_difference": combined_smooth, "ratio": combined_ratio,
    }

    # Keep the x-axis to the common measured overlap rather than endpoint
    # columns with no usable edge/center spectrum.
    coverage = np.sum([
        np.isfinite(amp_profiles[amp]["edge"]) &
        np.isfinite(amp_profiles[amp]["center"])
        for amp in AMPLIFIERS], axis=0)
    usable_wave = coverage >= len(AMPLIFIERS)
    if not np.any(usable_wave):
        usable_wave = np.isfinite(wave)
    x = np.asarray(wave)[usable_wave]

    columns = (*AMPLIFIERS, "all amps")
    fig, axes = plt.subplots(3, 5, figsize=(19, 10), sharex=True)
    for column, name in enumerate(columns):
        values = amp_profiles[name]
        display = {
            "edge": values["edge"][usable_wave],
            "center": values["center"][usable_wave],
            "difference": values["difference"][usable_wave],
            "smooth_difference": values["smooth_difference"][usable_wave],
            "ratio": values["ratio"][usable_wave],
        }
        axes[0, column].plot(x, display["edge"], label="E: edge q=0..19")
        axes[0, column].plot(x, display["center"], label="C: center j=40..70")
        axes[1, column].plot(x, display["difference"], label="E - C")
        axes[1, column].plot(x, display["smooth_difference"],
                             label="broad(E - C)")
        axes[2, column].plot(x, display["ratio"], label="E / C - 1")
        axes[2, column].axhline(0.0, color="k", linewidth=.8, alpha=.6)
        axes[0, column].set_title(name)
        for row in range(3):
            axes[row, column].grid(alpha=.2)
            axes[row, column].set_xlim(float(x[0]), float(x[-1]))
        axes[0, column].set_ylim(*_robust_ylim(
            display["edge"], display["center"]))
        axes[1, column].set_ylim(*_robust_ylim(
            display["difference"], display["smooth_difference"]))
        axes[2, column].set_ylim(*_robust_ylim(display["ratio"]))
        axes[2, column].set_xlabel("native wavelength (A)")
    axes[0, 0].set_ylabel("spectrum")
    axes[1, 0].set_ylabel("E - C")
    axes[2, 0].set_ylabel("E / C - 1")
    axes[0, 0].legend(fontsize=7)
    axes[1, 0].legend(fontsize=7)
    axes[2, 0].legend(fontsize=7)
    fig.suptitle("Blank-sky spectra: readout edge versus amplifier center")
    fig.tight_layout()
    fig.savefig(output_dir / "edge_vs_center_blank_sky_spectra.png", dpi=160)
    plt.close(fig)


def make_spectral_figures(output_dir, wave, profiles):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 4, figsize=(16, 7), sharex=True)
    for column, amp in enumerate(AMPLIFIERS):
        fit = profiles[amp]["edge"]
        axes[0, column].plot(wave, fit["D"], label="D")
        axes[0, column].plot(wave, fit["beta"] * fit["S"], "--",
                             label="beta S")
        axes[1, column].plot(wave, fit["D_hp"], label="D_hp")
        axes[1, column].plot(wave, fit["beta"] * fit["S_hp"], "--",
                             label="beta S_hp")
        axes[0, column].set_title(amp)
        for row in range(2):
            axes[row, column].grid(alpha=.2)
        axes[1, column].set_xlabel("native wavelength (A)")
        axes[0, column].text(
            .02, .96, "beta=%.3g\\nr=%.3g\\nrho=%.3g\\nRMS %.3g -> %.3g" %
            (fit["beta"], fit["pearson"], fit["spearman"],
             fit["rms_before"], fit["rms_after"]),
            transform=axes[0, column].transAxes, va="top", fontsize=7)
    axes[0, 0].set_ylabel("D")
    axes[1, 0].set_ylabel("high-pass")
    axes[0, 0].legend(fontsize=7)
    axes[1, 0].legend(fontsize=7)
    fig.suptitle("Blank-sky edge excess versus sky spectrum")
    fig.tight_layout()
    fig.savefig(output_dir / "edge_excess_spectrum_vs_sky.png", dpi=160)
    plt.close(fig)

    d_hp = np.nanmedian(np.vstack(
        [profiles[amp]["edge"]["D_hp"] for amp in AMPLIFIERS]), axis=0)
    s_hp = np.nanmedian(np.vstack(
        [profiles[amp]["edge"]["S_hp"] for amp in AMPLIFIERS]), axis=0)
    beta, _, _ = _robust_zero_slope(s_hp, d_hp)
    centers = _spectral_feature_centers(wave, s_hp)
    if not centers:
        centers = [float(np.nanmedian(wave))]
    fig, axes = plt.subplots(len(centers), 1,
                             figsize=(9, max(3, 2.4 * len(centers))),
                             squeeze=False)
    for axis, center in zip(axes[:, 0], centers):
        selected = (wave >= center - 12.0) & (wave <= center + 12.0)
        axis.plot(wave[selected], s_hp[selected], label="S_hp")
        axis.plot(wave[selected], d_hp[selected], label="D_hp")
        axis.plot(wave[selected], beta * s_hp[selected], "--",
                  label="beta S_hp")
        axis.set_title("%.1f A" % center)
        axis.grid(alpha=.2)
    if centers:
        axes[0, 0].legend(fontsize=8)
    fig.suptitle("Automatically selected strong sky-feature regions")
    fig.tight_layout()
    fig.savefig(output_dir / "edge_excess_sky_feature_zooms.png", dpi=160)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(9, 5))
    for amp in AMPLIFIERS:
        q = [0.5 * (item["q_min"] + item["q_max"])
             for item in profiles[amp]["q_fits"]]
        beta_q = [item["fit"]["beta"] for item in profiles[amp]["q_fits"]]
        axis.plot(q, beta_q, "o-", label=amp)
    combined = []
    for index in range(len(SPECTRAL_Q_BINS)):
        combined.append(np.nanmedian([
            profiles[amp]["q_fits"][index]["fit"]["beta"]
            for amp in AMPLIFIERS]))
    axis.plot([0.5 * sum(item) for item in SPECTRAL_Q_BINS], combined,
              "k--", label="combined")
    axis.axvspan(0, 19, color="tab:red", alpha=.10)
    axis.set(xlabel="folded readout distance q", ylabel="beta(q)")
    axis.grid(alpha=.2)
    axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "sky_like_residual_vs_readout_distance.png",
                dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    for axis, amp in zip(axes.ravel(), AMPLIFIERS):
        fit = profiles[amp]["edge"]
        axis.plot(wave, fit["D_hp"], label="D_hp")
        axis.plot(wave, fit["residual"], label="D_hp - beta S_hp")
        axis.set_title(amp)
        axis.grid(alpha=.2)
        axis.legend(fontsize=8)
    fig.suptitle("Edge excess after removing fitted sky-like component")
    fig.tight_layout()
    fig.savefig(output_dir / "edge_excess_after_sky_component_removal.png",
                dpi=160)
    plt.close(fig)

    make_edge_center_blank_sky_figure(output_dir, wave, profiles)


def run_spectral_sky_test(path, exposure, image, output_dir):
    data = _spectral_native_rows(path, exposure, image)
    profiles = _spectral_reference_profiles(data)
    rows = _write_spectral_outputs(
        output_dir, Path(path).name, exposure, data, profiles)
    make_spectral_figures(output_dir, data["wave"], profiles)
    print("")
    print("Spectral sky-like residual experiment:")
    print("  selected H5/exposure: %s / %d" % (path, exposure))
    print("  analyzed stream: Raw.spectrum/Raw.wave after legacy FTF, "
          "before rectification and sky subtraction")
    print("  Raw.mscispectrum and the candidate-FTF path were not used")
    print("  blank-sky fibers after finite-spectrum check: %d" %
          sum(data["counts"].values()))
    print("  blank-sky fibers by amplifier: %s" %
          ", ".join("%s=%d" % (amp, data["counts"][amp])
                   for amp in AMPLIFIERS))
    for amp in AMPLIFIERS:
        j = data["by_amp"][amp]["j"]
        print("  %s: edge q<20=%d, center j=40..70=%d" %
              (amp, int(np.sum((j if amp in ("LL", "RU") else 111 - j) < 20)),
               int(np.isin(j, REFERENCE_J).sum())))
        fit = profiles[amp]["edge"]
        reduction = (100.0 * (1.0 - fit["rms_after"] / fit["rms_before"])
                     if fit["rms_before"] else np.nan)
        print("    beta=%.6g, Pearson=%.6g, Spearman=%.6g, "
              "robust RMS %.6g -> %.6g (%.3g%% reduction)" %
              (fit["beta"], fit["pearson"], fit["spearman"],
               fit["rms_before"], fit["rms_after"], reduction))
    print("  beta(q):")
    for amp in AMPLIFIERS:
        print("    %s: %s" % (amp, ", ".join(
            "[%d-%d]=%.5g" % (item["q_min"], item["q_max"],
                               item["fit"]["beta"])
            for item in profiles[amp]["q_fits"])))
    edge_fits = [profiles[amp]["edge"] for amp in AMPLIFIERS]
    positive = [fit["beta"] > 0.0 and fit["pearson"] > 0.0 and
                fit["rms_after"] < fit["rms_before"]
                for fit in edge_fits]
    if all(positive):
        conclusion = "YES"
    elif any(positive):
        conclusion = "AMBIGUOUS"
    else:
        conclusion = "NO"
    print("  detailed edge residual follows sky spectral structure: %s"
          % conclusion)


ADDITIVE_MODELS = ("constant_final", "raw_constant")
ADDITIVE_MODEL_LABELS = {
    "constant_final": "Model A: constant final units",
    "raw_constant": "Model B: constant Raw e-/A propagated through calibration",
}
ADDITIVE_SAFE_WAVE = (3700.0, 5350.0)
ADDITIVE_SMOOTH_SIGMA = 2.5


def _additive_flux_basis(h5, n_wave, exposure):
    """Return the exact Raw-spectrum to Fibers wavelength basis."""
    if n_wave != RECTIFIED_WAVE.size:
        raise ValueError("Fibers spectrum has %d bins; expected %d" %
                         (n_wave, RECTIFIED_WAVE.size))
    if "Survey" not in h5.root._v_children:
        raise ValueError("selected H5 lacks Survey table")
    survey = h5.root.Survey
    required = {"exptime", "millum", "throughput", "offset", "exp"}
    if not required.issubset(survey.colnames):
        raise ValueError("Survey lacks the Raw->Fibers calibration columns")
    exps = np.asarray(survey.cols.exp[:], dtype=int)
    selected = np.flatnonzero(exps == int(exposure))
    if selected.size != 1:
        raise ValueError("Survey must contain one row for selected exposure")
    survey_row = survey[selected[0]]
    exptime = float(survey_row["exptime"])
    millum = float(survey_row["millum"])
    guider_transparency = float(survey_row["throughput"])
    offset_value = float(survey_row["offset"])
    if not np.isfinite(exptime) or exptime == 0.0:
        raise ValueError("selected Survey.exptime is invalid")
    if not np.isfinite(millum) or not np.isfinite(guider_transparency):
        raise ValueError("selected Survey illumination/transparency is invalid")
    gratio = millum * guider_transparency / 5e5
    if not np.isfinite(gratio) or gratio == 0.0:
        raise ValueError("selected Survey guider ratio is invalid")

    throughput_path = Path(__file__).resolve().parent / "CALS" / "throughput.txt"
    table = Table.read(throughput_path, format="ascii.fixed_width_two_line")
    standard_wavelength = np.asarray(table["wavelength"], dtype=float)
    standard_throughput = np.asarray(table["throughput"], dtype=float)
    if (standard_wavelength.size != n_wave or
            not np.allclose(standard_wavelength, RECTIFIED_WAVE,
                            rtol=0.0, atol=1e-6)):
        raise ValueError("CALS/throughput.txt wavelength grid does not match "
                         "the production def_wave")

    # These expressions intentionally mirror quick_reduction.py.  A finite
    # Survey.offset is applied to SCI; a nonfinite offset leaves the factor at
    # unity, matching the production ``if np.isfinite(offset)`` branch.
    offset = offset_value if np.isfinite(offset_value) else 1.0
    mult_fac = (6.626e-27 * (3e18 / RECTIFIED_WAVE) / 360.0 /
                5e5 / 0.92 * 5)
    mult_fac *= 1e29 * RECTIFIED_WAVE**2 / 2.99792e18
    final_norm = 1e-29 * 2.99792e18 / RECTIFIED_WAVE**2 * 1e17
    raw_to_fibers = (mult_fac * (360.0 / exptime) /
                     standard_throughput / gratio * final_norm * offset)
    reference = ((RECTIFIED_WAVE >= 4000.0) &
                 (RECTIFIED_WAVE <= 5000.0))
    scale = float(np.nanmedian(raw_to_fibers[reference]))
    if not np.isfinite(scale) or scale == 0.0:
        raise ValueError("cannot normalize the quick-reduction flux basis")
    note = ("K_raw_to_fibers = mult_fac*(360/exptime)/standard_throughput/"
            "gratio*final_norm*offset; CALS/throughput.txt used")
    metadata = {
        "exptime": exptime, "millum": millum,
        "guider_transparency": guider_transparency, "gratio": gratio,
        "offset": offset, "raw_basis_scale": scale,
        "raw_basis_unscaled": raw_to_fibers,
    }
    return raw_to_fibers, raw_to_fibers / scale, note, metadata


def _additive_median_rows(values, selected, description):
    selected = np.asarray(selected, dtype=bool)
    if not np.any(selected):
        raise ValueError("no fibers available for %s" % description)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        result = np.nanmedian(np.asarray(values)[selected], axis=0)
    if not np.any(np.isfinite(result)):
        raise ValueError("no finite spectrum available for %s" % description)
    return result


def _robust_amplitude_rms(values):
    """RMS amplitude resistant to a small number of extreme samples."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan
    return float(np.sqrt(np.nanmedian(values * values)))


def _additive_bandaid_rows(path, exposure, image):
    """Read one exposure of blank-sky Fibers spectra for the Band-Aid test."""
    with tables.open_file(path, mode="r") as h5:
        required_tables = {"Info", "Fibers"}
        if not required_tables.union({"Raw"}).issubset(h5.root._v_children):
            raise ValueError("%s must contain Info, Raw, and Fibers tables" %
                             path)
        info = h5.root.Info
        fibers = h5.root.Fibers
        raw = h5.root.Raw
        if int(info.nrows) != int(fibers.nrows):
            raise ValueError("%s Info/Fibers row mismatch" % path)
        if int(info.nrows) != int(raw.nrows):
            raise ValueError("%s Info/Raw row mismatch" % path)
        if "spectrum" not in fibers.colnames or \
                not {"spectrum", "wave"}.issubset(raw.colnames):
            raise ValueError("%s lacks required Fibers/Raw spectrum columns" %
                             path)
        groups, labels, nexp = build_amplifier_groups(info)
        if exposure < 1 or exposure > nexp:
            raise ValueError("requested exposure %d but %s has %d exposures" %
                             (exposure, path, nexp))
        nrows = int(info.nrows)
        n_fiber_wave = int(fibers.coldtypes["spectrum"].shape[0])
        n_raw_wave = int(raw.coldtypes["spectrum"].shape[0])
        if n_fiber_wave != RECTIFIED_WAVE.size:
            raise ValueError("%s Fibers.spectrum has %d bins; expected %d" %
                             (path, n_fiber_wave, RECTIFIED_WAVE.size))
        ra = np.asarray(info.cols.ra[:], dtype=float)
        dec = np.asarray(info.cols.dec[:], dtype=float)
        if image is None:
            raise ValueError("the additive Band-Aid test requires --image")
        dra = ((ra - M101_RA_DEG) * np.cos(np.deg2rad(M101_DEC_DEG)) * 60.0)
        ddec = (dec - M101_DEC_DEG) * 60.0
        radial_blank = np.hypot(dra, ddec) > M101_SKY_MIN_RADIUS_ARCMIN
        image_valid, image_blank = _image_blank_selection(image, ra, dec)
        candidate = ((labels == exposure) & radial_blank & image_valid &
                     image_blank)
        candidate_indices = np.flatnonzero(candidate)
        if candidate_indices.size == 0:
            raise ValueError("no image-selected blank-sky fibers in exposure %d"
                             % exposure)

        row_j = np.full(nrows, -1, dtype=np.int16)
        row_amp = np.full(nrows, "", dtype="U2")
        for group in groups:
            row_j[group["indices"]] = group["j"]
            row_amp[group["indices"]] = group["amp"]
        specid = np.asarray(info.cols.specid[:])
        ifuslot = np.asarray(info.cols.ifuslot[:])
        ifuid = np.asarray(info.cols.ifuid[:])
        identity = np.array([
            "%s|%s|%s" % (_text(specid[index]), int(ifuslot[index]),
                           int(ifuid[index]))
            for index in candidate_indices], dtype="U128")
        unique_identity = sorted(set(identity.tolist()))
        partitions = {value: index % 2 for index, value in
                      enumerate(unique_identity)}
        partition = np.asarray([partitions[value] for value in identity],
                               dtype=np.int8)

        minimum_finite = int(np.ceil(
            M101_SKY_MIN_FINITE_FRACTION * n_fiber_wave))
        kept_rows = []
        kept_fiber_spectra = []
        kept_raw_spectra = []
        kept_raw_waves = []
        for start in range(0, candidate_indices.size, ROW_CHUNK):
            rows = candidate_indices[start:start + ROW_CHUNK]
            fiber_spectrum = np.asarray(
                fibers.read_coordinates(rows, field="spectrum"), dtype=float)
            raw_spectrum = np.asarray(
                raw.read_coordinates(rows, field="spectrum"), dtype=float)
            raw_wave = np.asarray(
                raw.read_coordinates(rows, field="wave"), dtype=float)
            keep = np.isfinite(fiber_spectrum).sum(axis=1) >= minimum_finite
            if np.any(keep):
                kept_rows.append(rows[keep])
                kept_fiber_spectra.append(fiber_spectrum[keep])
                kept_raw_spectra.append(raw_spectrum[keep])
                kept_raw_waves.append(raw_wave[keep])
        if not kept_fiber_spectra:
            raise ValueError("no blank-sky fibers meet the finite-spectrum "
                             "criterion")
        kept_rows = np.concatenate(kept_rows)
        spectra = np.concatenate(kept_fiber_spectra, axis=0)
        raw_spectra = np.concatenate(kept_raw_spectra, axis=0)
        raw_waves = np.concatenate(kept_raw_waves, axis=0)
        candidate_position = {int(row): index for index, row in
                              enumerate(candidate_indices)}
        kept_position = np.asarray([candidate_position[int(row)]
                                    for row in kept_rows], dtype=int)
        physical_identity = identity[kept_position]
        partition = partition[kept_position]
        j = row_j[kept_rows].astype(int)
        amp = row_amp[kept_rows]
        common_raw_wave = np.nanmedian(raw_waves[:256], axis=0)
        raw_rectified = np.full(raw_spectra.shape, np.nan, dtype=float)
        for index in range(raw_spectra.shape[0]):
            valid = np.isfinite(raw_waves[index]) & np.isfinite(raw_spectra[index])
            if np.sum(valid) >= 2:
                raw_rectified[index] = np.interp(
                    common_raw_wave, raw_waves[index, valid],
                    raw_spectra[index, valid], left=np.nan, right=np.nan)
        K_absolute, K_normalized, norm_note, calibration = _additive_flux_basis(
            h5, n_fiber_wave, exposure)
        return {
            "path": Path(path), "exposure": int(exposure),
            "wave": RECTIFIED_WAVE.copy(), "spectrum": spectra,
            "raw_wave": common_raw_wave, "raw_spectrum": raw_rectified,
            "j": j, "amp": amp, "partition": partition,
            "identity": physical_identity, "K": {
                "constant_final": np.ones(n_fiber_wave, dtype=float),
                "raw_constant": K_normalized,
            },
            "K_raw_to_fibers_absolute": K_absolute,
            "K_raw_to_fibers_normalized": K_normalized,
            "norm_note": norm_note, "calibration": calibration,
            "candidate_count": int(candidate_indices.size),
            "n_fibers": int(spectra.shape[0]),
            "n_ifus": len(unique_identity),
            "n_ifus_train": sum(index % 2 == 0
                                 for index in range(len(unique_identity))),
            "n_ifus_validation": sum(index % 2 == 1
                                      for index in range(len(unique_identity))),
        }


def _smooth_additive_profile(values, sigma=ADDITIVE_SMOOTH_SIGMA):
    from scipy.ndimage import gaussian_filter1d

    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    if not np.any(finite):
        return np.full(values.shape, np.nan, dtype=float)
    numerator = gaussian_filter1d(np.where(finite, values, 0.0), sigma,
                                  mode="nearest")
    denominator = gaussian_filter1d(finite.astype(float), sigma,
                                    mode="nearest")
    result = np.full(values.shape, np.nan, dtype=float)
    valid = denominator > 0.0
    result[valid] = numerator[valid] / denominator[valid]
    return result


def _fit_additive_amplitude(difference, basis, wave):
    safe = ((wave >= ADDITIVE_SAFE_WAVE[0]) &
            (wave <= ADDITIVE_SAFE_WAVE[1]))
    beta, _, _ = _robust_zero_slope(
        np.asarray(basis)[safe], np.asarray(difference)[safe])
    return float(beta)


def _additive_training_profiles(data):
    wave = data["wave"]
    raw_wave = data["raw_wave"]
    profiles = {}
    q_axis = np.arange(FIBERS_PER_AMPLIFIER, dtype=int)
    for amp in AMPLIFIERS:
        amp_rows = data["amp"] == amp
        train = amp_rows & (data["partition"] == 0)
        central_selected = train & np.isin(data["j"], REFERENCE_J)
        central = _additive_median_rows(
            data["spectrum"], central_selected,
            "%s TRAIN central reference" % amp)
        central_raw = _additive_median_rows(
            data["raw_spectrum"], central_selected,
            "%s TRAIN Raw central reference" % amp)
        amp_models = {}
        if amp in ("LL", "RU"):
            q_for_row = data["j"]
            q_for_reference = REFERENCE_J
        else:
            q_for_row = 111 - data["j"]
            q_for_reference = 111 - REFERENCE_J
        for model in ADDITIVE_MODELS:
            basis = data["K"][model]
            raw = np.full(FIBERS_PER_AMPLIFIER, np.nan, dtype=float)
            scatter = np.full(FIBERS_PER_AMPLIFIER, np.nan, dtype=float)
            train_q_values = {}
            for q in q_axis:
                selected = train & (q_for_row == q)
                if not np.any(selected):
                    continue
                if model == "raw_constant":
                    r_q = _additive_median_rows(
                        data["raw_spectrum"], selected,
                        "%s TRAIN Raw q=%d" % (amp, q))
                    difference = r_q - central_raw
                    finite = ((raw_wave >= ADDITIVE_SAFE_WAVE[0]) &
                              (raw_wave <= ADDITIVE_SAFE_WAVE[1]) &
                              np.isfinite(difference))
                    raw[q] = (float(np.nanmedian(difference[finite]))
                              if np.any(finite) else np.nan)
                    scatter[q] = _robust_rms(
                        difference[finite] - raw[q]) if np.any(finite) else np.nan
                else:
                    r_q = _additive_median_rows(
                        data["spectrum"], selected,
                        "%s TRAIN q=%d" % (amp, q))
                    difference = r_q - central
                    raw[q] = _fit_additive_amplitude(difference, basis, wave)
                train_q_values[q] = difference
            smooth = _smooth_additive_profile(raw)
            center_level = np.nanmedian(smooth[q_for_reference])
            if np.isfinite(center_level):
                smooth = smooth - center_level
            applied = smooth.copy()
            applied[q_axis >= 40] = 0.0
            applied[~np.isfinite(applied)] = 0.0
            amp_models[model] = {
                "basis": basis, "center": central, "raw": raw,
                "smooth": smooth, "applied": applied,
                "train_q_values": train_q_values, "scatter": scatter,
            }
        profiles[amp] = {
            "q": q_axis, "models": amp_models,
            "q_for_row": q_for_row, "q_for_reference": q_for_reference,
            "central_raw": central_raw,
        }
    return profiles


def _additive_validation_profiles(data, training_profiles, wave_center,
                                  half_width):
    wave = data["wave"]
    raw_wave = data["raw_wave"]
    band = ((wave >= wave_center - half_width) &
            (wave <= wave_center + half_width))
    if not np.any(band):
        raise ValueError("additive Band-Aid continuum band has no bins")
    results = {}
    for amp in AMPLIFIERS:
        amp_rows = data["amp"] == amp
        validation = amp_rows & (data["partition"] == 1)
        q_for_row = training_profiles[amp]["q_for_row"]
        central_selected = validation & np.isin(data["j"], REFERENCE_J)
        edge_selected = validation & (q_for_row < 20)
        center = _additive_median_rows(
            data["spectrum"], central_selected,
            "%s VALIDATION central reference" % amp)
        edge = _additive_median_rows(
            data["spectrum"], edge_selected,
            "%s VALIDATION edge" % amp)
        raw_center = _additive_median_rows(
            data["raw_spectrum"], central_selected,
            "%s VALIDATION Raw central reference" % amp)
        raw_edge = _additive_median_rows(
            data["raw_spectrum"], edge_selected,
            "%s VALIDATION Raw edge" % amp)
        raw_model = training_profiles[amp]["models"]["raw_constant"]
        raw_correction = raw_model["applied"][q_for_row]
        raw_corrected_rows = data["raw_spectrum"] - raw_correction[:, None]
        raw_corrected_edge = _additive_median_rows(
            raw_corrected_rows, edge_selected,
            "%s VALIDATION corrected Raw edge" % amp)
        raw_legacy_difference = raw_edge - raw_center
        raw_corrected_difference = raw_corrected_edge - raw_center
        raw_validation = {
            "center": raw_center, "legacy": {
                "edge": raw_edge, "difference": raw_legacy_difference,
                "smooth_difference": _broad_continuum_1d(
                    raw_legacy_difference, raw_wave),
            },
            "corrected": {
                "edge": raw_corrected_edge,
                "difference": raw_corrected_difference,
                "smooth_difference": _broad_continuum_1d(
                    raw_corrected_difference, raw_wave),
            },
        }
        model_results = {}
        for model in ADDITIVE_MODELS:
            model_profile = training_profiles[amp]["models"][model]
            if model == "raw_constant":
                correction = model_profile["applied"][q_for_row]
                corrected_rows = (data["spectrum"] -
                                  correction[:, None] *
                                  data["K_raw_to_fibers_absolute"])
            else:
                correction = model_profile["applied"][q_for_row]
                corrected_rows = data["spectrum"] - correction[:, None]
            corrected_edge = _additive_median_rows(
                corrected_rows, edge_selected,
                "%s VALIDATION corrected edge" % amp)
            difference = edge - center
            corrected_difference = corrected_edge - center
            model_results[model] = {
                "edge": corrected_edge,
                "difference": corrected_difference,
                "smooth_difference": _broad_continuum_1d(
                    corrected_difference, wave),
                "ratio": _safe_spectral_ratio(corrected_edge, center),
            }
        legacy_difference = edge - center
        legacy = {
            "edge": edge, "difference": legacy_difference,
            "smooth_difference": _broad_continuum_1d(
                legacy_difference, wave),
            "ratio": _safe_spectral_ratio(edge, center),
        }
        continuum_values = {}
        legacy_levels = np.nanmedian(data["spectrum"][:, band], axis=1)
        center_level = _additive_median_rows(
            legacy_levels, central_selected, "%s VALIDATION continuum" % amp)
        for model in ("legacy",) + ADDITIVE_MODELS:
            if model == "legacy":
                levels = legacy_levels
            else:
                correction = training_profiles[amp]["models"][model][
                    "applied"][q_for_row]
                if model == "raw_constant":
                    corrected = (data["spectrum"] -
                                 correction[:, None] *
                                 data["K_raw_to_fibers_absolute"])
                else:
                    corrected = data["spectrum"] - correction[:, None]
                levels = np.nanmedian(corrected[:, band], axis=1)
            q_values = np.full(FIBERS_PER_AMPLIFIER, np.nan, dtype=float)
            for q in range(FIBERS_PER_AMPLIFIER):
                selected = validation & (q_for_row == q)
                if np.any(selected):
                    q_values[q] = np.nanmedian(levels[selected])
            continuum_values[model] = {
                "level": q_values, "residual": q_values - center_level,
            }
        results[amp] = {
            "center": center, "legacy": legacy,
            "models": model_results, "continuum": continuum_values,
            "raw_validation": raw_validation,
            "n_validation": int(validation.sum()),
            "n_edge": int(edge_selected.sum()),
            "n_center": int(central_selected.sum()),
        }
    return results


def _safe_spectral_ratio(numerator, denominator):
    numerator = np.asarray(numerator, dtype=float)
    denominator = np.asarray(denominator, dtype=float)
    result = np.full(numerator.shape, np.nan, dtype=float)
    finite = np.isfinite(numerator) & np.isfinite(denominator)
    if np.any(finite):
        scale = max(1.0, float(np.nanmax(np.abs(denominator[finite]))))
        finite &= np.abs(denominator) > np.finfo(float).eps * scale
        result[finite] = numerator[finite] / denominator[finite] - 1.0
    return result


def _additive_common_wave_mask(validation_profiles, wave):
    coverage = []
    for amp in AMPLIFIERS:
        legacy = validation_profiles[amp]["legacy"]
        coverage.append(np.isfinite(legacy["edge"]) &
                        np.isfinite(validation_profiles[amp]["center"]))
    mask = np.sum(coverage, axis=0) == len(AMPLIFIERS)
    if not np.any(mask):
        mask = np.isfinite(wave)
    return mask


def _write_additive_outputs(output_dir, data, training_profiles,
                            validation_profiles):
    summary_fields = [
        "h5", "exposure", "amplifier", "model", "n_train_fibers",
        "n_validation_fibers", "q20_legacy", "q20_corrected",
        "q20_absolute_reduction", "q20_percent_reduction",
        "q0_39_rms_legacy", "q0_39_rms_corrected",
        "q_ge40_median_change", "broad_rms_legacy", "broad_rms_corrected",
        "highpass_rms_legacy", "highpass_rms_corrected",
        "raw_q20_legacy", "raw_q20_corrected", "raw_q20_absolute_reduction",
        "raw_q20_percent_reduction", "raw_broad_rms_legacy",
        "raw_broad_rms_corrected",
        "raw_basis_scale", "exptime", "millum", "guider_throughput",
        "gratio", "survey_offset", "median_A_edge_raw_units",
    ]
    summary_rows = []
    profile_fields = [
        "h5", "exposure", "amplifier", "q",
        "A_raw_unsmoothed_e_per_A", "A_raw_smoothed_e_per_A",
        "A_raw_applied_e_per_A", "wavelength_scatter_e_per_A",
        "n_train_fibers", "n_validation_fibers",
    ]
    profile_rows = []
    for amp in AMPLIFIERS:
        train_count = int(np.sum((data["amp"] == amp) &
                                 (data["partition"] == 0)))
        valid_count = int(np.sum((data["amp"] == amp) &
                                 (data["partition"] == 1)))
        validation = validation_profiles[amp]
        for model in ADDITIVE_MODELS:
            fit = training_profiles[amp]["models"][model]
            corrected = validation["models"][model]
            legacy_residual = validation["continuum"]["legacy"]["residual"]
            corrected_residual = validation["continuum"][model]["residual"]
            q20_legacy = float(np.nanmedian(legacy_residual[:20]))
            q20_corrected = float(np.nanmedian(corrected_residual[:20]))
            reduction = q20_legacy - q20_corrected
            q0_39_legacy = _robust_amplitude_rms(legacy_residual[:40])
            q0_39_corrected = _robust_amplitude_rms(corrected_residual[:40])
            q_ge40_change = float(np.nanmedian(
                corrected_residual[40:] - legacy_residual[40:]))
            broad_legacy = _robust_amplitude_rms(
                validation["legacy"]["smooth_difference"])
            broad_corrected = _robust_amplitude_rms(
                corrected["smooth_difference"])
            highpass_legacy = _robust_rms(
                validation["legacy"]["difference"] -
                validation["legacy"]["smooth_difference"])
            highpass_corrected = _robust_rms(
                corrected["difference"] - corrected["smooth_difference"])
            if model == "raw_constant":
                raw_legacy = validation["raw_validation"]["legacy"]
                raw_corrected = validation["raw_validation"]["corrected"]
                raw_q20_legacy = float(np.nanmedian(
                    raw_legacy["difference"][:20]))
                raw_q20_corrected = float(np.nanmedian(
                    raw_corrected["difference"][:20]))
                raw_reduction = raw_q20_legacy - raw_q20_corrected
                raw_broad_legacy = _robust_amplitude_rms(
                    raw_legacy["smooth_difference"])
                raw_broad_corrected = _robust_amplitude_rms(
                    raw_corrected["smooth_difference"])
            else:
                raw_q20_legacy = raw_q20_corrected = raw_reduction = np.nan
                raw_broad_legacy = raw_broad_corrected = np.nan
            calibration = data["calibration"]
            raw_scale = calibration["raw_basis_scale"]
            median_A_edge_raw = (
                float(np.nanmedian(fit["applied"][:20]) / raw_scale)
                if model == "raw_constant" else np.nan)
            summary_rows.append({
                "h5": data["path"].name, "exposure": data["exposure"],
                "amplifier": amp, "model": model,
                "n_train_fibers": train_count,
                "n_validation_fibers": valid_count,
                "q20_legacy": q20_legacy,
                "q20_corrected": q20_corrected,
                "q20_absolute_reduction": reduction,
                "q20_percent_reduction": (100.0 * reduction / q20_legacy
                                           if q20_legacy != 0.0 else np.nan),
                "q0_39_rms_legacy": q0_39_legacy,
                "q0_39_rms_corrected": q0_39_corrected,
                "q_ge40_median_change": q_ge40_change,
                "broad_rms_legacy": broad_legacy,
                "broad_rms_corrected": broad_corrected,
                "highpass_rms_legacy": highpass_legacy,
                "highpass_rms_corrected": highpass_corrected,
                "raw_q20_legacy": raw_q20_legacy,
                "raw_q20_corrected": raw_q20_corrected,
                "raw_q20_absolute_reduction": raw_reduction,
                "raw_q20_percent_reduction": (
                    100.0 * raw_reduction / raw_q20_legacy
                    if np.isfinite(raw_q20_legacy) and raw_q20_legacy != 0.0
                    else np.nan),
                "raw_broad_rms_legacy": raw_broad_legacy,
                "raw_broad_rms_corrected": raw_broad_corrected,
                "raw_basis_scale": raw_scale,
                "exptime": calibration["exptime"],
                "millum": calibration["millum"],
                "guider_throughput": calibration["guider_transparency"],
                "gratio": calibration["gratio"],
                "survey_offset": calibration["offset"],
                "median_A_edge_raw_units": median_A_edge_raw,
            })
        raw_fit = training_profiles[amp]["models"]["raw_constant"]
        for q in range(FIBERS_PER_AMPLIFIER):
            profile_rows.append({
                "h5": data["path"].name, "exposure": data["exposure"],
                "amplifier": amp, "q": q,
                "A_raw_unsmoothed_e_per_A": raw_fit["raw"][q],
                "A_raw_smoothed_e_per_A": raw_fit["smooth"][q],
                "A_raw_applied_e_per_A": raw_fit["applied"][q],
                "wavelength_scatter_e_per_A": raw_fit["scatter"][q],
                "n_train_fibers": train_count,
                "n_validation_fibers": valid_count,
            })
    with (output_dir / "additive_bandaid_summary.csv").open(
            "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(summary_rows)
    with (output_dir / "additive_bandaid_profiles.csv").open(
            "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=profile_fields)
        writer.writeheader()
        writer.writerows(profile_rows)
    return summary_rows


def _additive_plot_wave(validation_profiles, wave):
    mask = _additive_common_wave_mask(validation_profiles, wave)
    return np.asarray(wave)[mask], mask


def make_additive_basis_figure(output_dir, data):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    wave = data["wave"]
    constant = np.ones(wave.size, dtype=float)
    raw_basis = data["K"]["raw_constant"]
    finite = np.isfinite(raw_basis)
    raw_absolute = data["K_raw_to_fibers_absolute"]
    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    axes[0].plot(wave, constant, label="constant-final basis = 1")
    axes[0].plot(wave[finite], raw_basis[finite],
                 label="K_raw_to_fibers_normalized")
    axes[0].set_ylabel("normalized basis")
    axes[0].set_ylim(*_robust_ylim(constant, raw_basis))
    axes[1].plot(wave, raw_absolute,
                 label="K_raw_to_fibers_absolute")
    axes[1].set_ylabel("Fibers units per Raw e-/A")
    axes[1].set_ylim(*_robust_ylim(raw_absolute))
    axes[1].set_xlabel("wavelength (A)")
    for axis in axes:
        axis.set_xlim(float(wave[0]), float(wave[-1]))
        axis.grid(alpha=.2)
        axis.legend(fontsize=8)
    axes[0].set_title("Additive Band-Aid wavelength bases")
    fig.tight_layout()
    fig.savefig(output_dir / "additive_bandaid_wavelength_basis.png", dpi=160)
    plt.close(fig)


def make_additive_raw_validation_figure(output_dir, data,
                                       validation_profiles):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    wave = data["raw_wave"]
    entries = {}
    for amp in AMPLIFIERS:
        entries[amp] = validation_profiles[amp]["raw_validation"]
    center = np.nanmedian(np.vstack([
        entries[amp]["center"] for amp in AMPLIFIERS]), axis=0)
    legacy_edge = np.nanmedian(np.vstack([
        entries[amp]["legacy"]["edge"] for amp in AMPLIFIERS]), axis=0)
    corrected_edge = np.nanmedian(np.vstack([
        entries[amp]["corrected"]["edge"] for amp in AMPLIFIERS]), axis=0)
    legacy_difference = legacy_edge - center
    corrected_difference = corrected_edge - center
    entries["all amps"] = {
        "center": center,
        "legacy": {
            "edge": legacy_edge, "difference": legacy_difference,
            "smooth_difference": _broad_continuum_1d(
                legacy_difference, wave),
        },
        "corrected": {
            "edge": corrected_edge, "difference": corrected_difference,
            "smooth_difference": _broad_continuum_1d(
                corrected_difference, wave),
        },
    }
    coverage = np.sum([
        np.isfinite(entries[amp]["center"]) &
        np.isfinite(entries[amp]["legacy"]["edge"])
        for amp in AMPLIFIERS], axis=0)
    wave_mask = coverage == len(AMPLIFIERS)
    if not np.any(wave_mask):
        wave_mask = np.isfinite(wave)
    x = wave[wave_mask]

    fig, axes = plt.subplots(2, 5, figsize=(19, 7), sharex=True)
    for column, amp in enumerate((*AMPLIFIERS, "all amps")):
        entry = entries[amp]
        center = entry["center"][wave_mask]
        legacy = entry["legacy"]
        corrected = entry["corrected"]
        axes[0, column].plot(x, legacy["edge"][wave_mask], label="Raw E")
        axes[0, column].plot(x, center, label="Raw C")
        axes[0, column].plot(x, corrected["edge"][wave_mask],
                             "--", label="Raw E corrected")
        axes[1, column].plot(x, legacy["difference"][wave_mask],
                             label="legacy Raw E - C")
        axes[1, column].plot(x, corrected["difference"][wave_mask],
                             "--", label="corrected Raw E - C")
        axes[1, column].plot(x, legacy["smooth_difference"][wave_mask],
                             ":", label="broad legacy E - C")
        axes[1, column].plot(x, corrected["smooth_difference"][wave_mask],
                             "-.", label="broad corrected E - C")
        axes[0, column].set_title(amp)
        axes[0, column].set_ylim(*_robust_ylim(
            legacy["edge"][wave_mask], center,
            corrected["edge"][wave_mask]))
        axes[1, column].set_ylim(*_robust_ylim(
            legacy["difference"][wave_mask],
            corrected["difference"][wave_mask],
            legacy["smooth_difference"][wave_mask],
            corrected["smooth_difference"][wave_mask]))
        for row in range(2):
            axes[row, column].grid(alpha=.2)
        axes[1, column].set_xlabel("native wavelength (A)")
    axes[0, 0].set_ylabel("Raw spectrum (e-/A)")
    axes[1, 0].set_ylabel("Raw E - C (e-/A)")
    axes[0, 0].legend(fontsize=7)
    axes[1, 0].legend(fontsize=7)
    fig.suptitle("Held-out Raw-space additive validation")
    fig.tight_layout()
    fig.savefig(output_dir / "additive_bandaid_raw_validation.png", dpi=160)
    plt.close(fig)


def make_additive_bandaid_figures(output_dir, data, training_profiles,
                                  validation_profiles):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    wave = data["wave"]
    fig, axes = plt.subplots(2, 4, figsize=(16, 7), sharex=True)
    for row, model in enumerate(ADDITIVE_MODELS):
        for column, amp in enumerate(AMPLIFIERS):
            fit = training_profiles[amp]["models"][model]
            axes[row, column].plot(training_profiles[amp]["q"], fit["raw"],
                                   "o", alpha=.35, label="raw A(q)")
            axes[row, column].plot(training_profiles[amp]["q"], fit["smooth"],
                                   linewidth=2, label="smoothed A(q)")
            axes[row, column].axvspan(0, 19, color="tab:red", alpha=.10)
            axes[row, column].axvline(40, color="k", linestyle=":",
                                      linewidth=.8)
            axes[row, column].set_title(amp if row == 0 else "")
            axes[row, column].grid(alpha=.2)
            axes[row, column].set_ylim(*_robust_ylim(fit["raw"], fit["smooth"]))
            if row == 1:
                axes[row, column].set_xlabel("folded readout distance q")
        axes[row, 0].set_ylabel("A(q) [%s]" %
                                 ("final units control" if model ==
                                  "constant_final" else "e-/A"))
    axes[0, 0].legend(fontsize=7)
    axes[1, 0].legend(fontsize=7)
    fig.suptitle("Additive Band-Aid: inferred spatial profile")
    fig.tight_layout()
    fig.savefig(output_dir / "additive_bandaid_profile_q.png", dpi=160)
    plt.close(fig)

    x, wave_mask = _additive_plot_wave(validation_profiles, wave)
    columns = (*AMPLIFIERS, "all amps")
    edge_center = {}
    for amp in AMPLIFIERS:
        edge_center[amp] = {
            "center": validation_profiles[amp]["center"],
            "legacy": validation_profiles[amp]["legacy"],
            "models": validation_profiles[amp]["models"],
        }
    center_all = np.nanmedian(np.vstack([
        edge_center[amp]["center"] for amp in AMPLIFIERS]), axis=0)
    legacy_all_edge = np.nanmedian(np.vstack([
        edge_center[amp]["legacy"]["edge"] for amp in AMPLIFIERS]), axis=0)
    all_entry = {
        "center": center_all,
        "legacy": {
            "edge": legacy_all_edge,
            "difference": legacy_all_edge - center_all,
        },
        "models": {},
    }
    all_entry["legacy"]["smooth_difference"] = _broad_continuum_1d(
        all_entry["legacy"]["difference"], wave)
    all_entry["legacy"]["ratio"] = _safe_spectral_ratio(
        all_entry["legacy"]["edge"], center_all)
    for model in ADDITIVE_MODELS:
        edge = np.nanmedian(np.vstack([
            edge_center[amp]["models"][model]["edge"]
            for amp in AMPLIFIERS]), axis=0)
        difference = edge - center_all
        all_entry["models"][model] = {
            "edge": edge, "difference": difference,
            "smooth_difference": _broad_continuum_1d(difference, wave),
            "ratio": _safe_spectral_ratio(edge, center_all),
        }
    edge_center["all amps"] = all_entry

    fig, axes = plt.subplots(3, 5, figsize=(19, 10), sharex=True)
    for column, name in enumerate(columns):
        entry = edge_center[name]
        center = entry["center"][wave_mask]
        legacy = entry["legacy"]
        edge = legacy["edge"][wave_mask]
        corr_a = entry["models"]["constant_final"]["edge"][wave_mask]
        corr_b = entry["models"]["raw_constant"]["edge"][wave_mask]
        axes[0, column].plot(x, edge, label="E legacy")
        axes[0, column].plot(x, center, label="C")
        axes[0, column].plot(x, corr_a, "--", label="E corrected A")
        axes[0, column].plot(x, corr_b, ":", label="E Raw-derived corrected")
        axes[1, column].plot(x, legacy["difference"][wave_mask],
                             label="legacy E - C")
        axes[1, column].plot(
            x, legacy["smooth_difference"][wave_mask],
            "--", label="broad legacy E - C")
        for model, style in (("constant_final", "-"),
                             ("raw_constant", ":")):
            corrected = entry["models"][model]
            axes[1, column].plot(
                x, corrected["difference"][wave_mask], style,
                label="corrected %s E - C" % ("A" if model ==
                                               "constant_final" else "Raw"))
            axes[1, column].plot(
                x, corrected["smooth_difference"][wave_mask],
                style, alpha=.45)
        axes[2, column].plot(x, legacy["ratio"][wave_mask],
                             label="legacy E / C - 1")
        axes[2, column].plot(
            x, entry["models"]["constant_final"]["ratio"][wave_mask],
            "--", label="corrected A")
        axes[2, column].plot(
            x, entry["models"]["raw_constant"]["ratio"][wave_mask],
            ":", label="Raw-derived corrected")
        axes[2, column].axhline(0.0, color="k", linewidth=.8, alpha=.6)
        axes[0, column].set_title(name)
        for row in range(3):
            axes[row, column].grid(alpha=.2)
            axes[row, column].set_xlim(float(x[0]), float(x[-1]))
        axes[0, column].set_ylim(*_robust_ylim(edge, center, corr_a, corr_b))
        axes[1, column].set_ylim(*_robust_ylim(
            legacy["difference"][wave_mask],
            legacy["smooth_difference"][wave_mask],
            entry["models"]["constant_final"]["difference"][wave_mask],
            entry["models"]["raw_constant"]["difference"][wave_mask]))
        axes[2, column].set_ylim(*_robust_ylim(
            legacy["ratio"][wave_mask],
            entry["models"]["constant_final"]["ratio"][wave_mask],
            entry["models"]["raw_constant"]["ratio"][wave_mask]))
        axes[2, column].set_xlabel("wavelength (A)")
    axes[0, 0].set_ylabel("spectrum")
    axes[1, 0].set_ylabel("E - C")
    axes[2, 0].set_ylabel("E / C - 1")
    axes[0, 0].legend(fontsize=6)
    axes[1, 0].legend(fontsize=6)
    axes[2, 0].legend(fontsize=6)
    fig.suptitle("Held-out blank-sky validation: additive Band-Aid")
    fig.tight_layout()
    fig.savefig(output_dir / "additive_bandaid_validation_spectra.png",
                dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 5, figsize=(19, 4.5), sharey=True)
    for column, name in enumerate(columns):
        if name == "all amps":
            stacks = {}
            for model in ("legacy",) + ADDITIVE_MODELS:
                stacks[model] = np.nanmedian(np.vstack([
                    validation_profiles[amp]["continuum"][model]["residual"]
                    for amp in AMPLIFIERS]), axis=0)
        else:
            stacks = {model: validation_profiles[name]["continuum"][model][
                "residual"] for model in ("legacy",) + ADDITIVE_MODELS}
        axes[column].plot(np.arange(112), stacks["legacy"],
                          label="legacy")
        axes[column].plot(np.arange(112), stacks["constant_final"],
                          "--", label="corrected A")
        axes[column].plot(np.arange(112), stacks["raw_constant"],
                          ":", label="Raw-derived corrected")
        axes[column].axvspan(0, 19, color="tab:red", alpha=.10)
        axes[column].axvline(40, color="k", linestyle=":", linewidth=.8)
        axes[column].set_title(name)
        axes[column].grid(alpha=.2)
        axes[column].set_xlabel("folded readout distance q")
        axes[column].set_ylim(*_robust_ylim(
            stacks["legacy"], stacks["constant_final"],
            stacks["raw_constant"]))
    axes[0].set_ylabel("4600 A continuum residual")
    axes[0].legend(fontsize=7)
    fig.suptitle("Held-out continuum profile before/after additive correction")
    fig.tight_layout()
    fig.savefig(output_dir / "additive_bandaid_folded_profile.png", dpi=160)
    plt.close(fig)
    make_additive_basis_figure(output_dir, data)
    make_additive_raw_validation_figure(output_dir, data, validation_profiles)


def run_additive_bandaid(path, exposure, image, output_dir,
                         wave_center, half_width):
    data = _additive_bandaid_rows(path, exposure, image)
    training = _additive_training_profiles(data)
    validation = _additive_validation_profiles(
        data, training, wave_center, half_width)
    summary_rows = _write_additive_outputs(
        output_dir, data, training, validation)
    make_additive_bandaid_figures(output_dir, data, training, validation)

    print("")
    print("Additive post-processing Band-Aid experiment:")
    print("  selected H5/exposure: %s / %d" % (path, exposure))
    print("  Raw.spectrum used for direct TRAIN/VALIDATION fit/check; "
          "Fibers.spectrum used for propagated application")
    print("  blank-sky candidates: %d; usable fibers: %d" %
          (data["candidate_count"], data["n_fibers"]))
    print("  physical IFUs: %d (TRAIN=%d, VALIDATION=%d)" %
          (data["n_ifus"], data["n_ifus_train"],
           data["n_ifus_validation"]))
    print("  partition fibers: TRAIN=%d, VALIDATION=%d" %
          (int(np.sum(data["partition"] == 0)),
           int(np.sum(data["partition"] == 1))))
    print("  %s" % data["norm_note"])
    print("  K_raw_to_fibers normalized to median 4000--5000 A = 1; "
          "Survey.offset=%.6g; exptime=%.6g; millum=%.6g; "
          "guider throughput=%.6g; gratio=%.6g" %
          (data["calibration"]["offset"], data["calibration"]["exptime"],
           data["calibration"]["millum"],
           data["calibration"]["guider_transparency"],
           data["calibration"]["gratio"]))
    print("  validation metrics by amplifier/model:")
    for row in summary_rows:
        print("    %s %s: q<20 %.6g -> %.6g (%.3g%%); "
              "q<40 RMS %.6g -> %.6g; broad RMS %.6g -> %.6g; "
              "high-pass RMS %.6g -> %.6g; q>=40 change %.3g" %
              (row["amplifier"], row["model"], row["q20_legacy"],
               row["q20_corrected"], row["q20_percent_reduction"],
               row["q0_39_rms_legacy"], row["q0_39_rms_corrected"],
               row["broad_rms_legacy"], row["broad_rms_corrected"],
               row["highpass_rms_legacy"], row["highpass_rms_corrected"],
               row["q_ge40_median_change"]))
    print("  Raw-space validation for raw_constant:")
    for amp in AMPLIFIERS:
        raw_validation = validation[amp]["raw_validation"]
        raw_legacy = raw_validation["legacy"]
        raw_corrected = raw_validation["corrected"]
        raw_q20_legacy = np.nanmedian(raw_legacy["difference"][:20])
        raw_q20_corrected = np.nanmedian(raw_corrected["difference"][:20])
        raw_broad_legacy = _robust_amplitude_rms(
            raw_legacy["smooth_difference"])
        raw_broad_corrected = _robust_amplitude_rms(
            raw_corrected["smooth_difference"])
        scatter = np.nanmedian(
            training[amp]["models"]["raw_constant"]["scatter"][:20])
        print("    %s: q<20 %.6g -> %.6g; broad RMS %.6g -> %.6g; "
              "median wavelength scatter q<20=%.6g" %
              (amp, raw_q20_legacy, raw_q20_corrected,
               raw_broad_legacy, raw_broad_corrected, scatter))
    raw_validation_values = [validation[amp]["raw_validation"]
                             for amp in AMPLIFIERS]
    raw_q20_legacy = np.nanmedian([
        np.nanmedian(item["legacy"]["difference"][:20])
        for item in raw_validation_values])
    raw_q20_corrected = np.nanmedian([
        np.nanmedian(item["corrected"]["difference"][:20])
        for item in raw_validation_values])
    raw_broad_legacy = np.nanmedian([
        _robust_amplitude_rms(item["legacy"]["smooth_difference"])
        for item in raw_validation_values])
    raw_broad_corrected = np.nanmedian([
        _robust_amplitude_rms(item["corrected"]["smooth_difference"])
        for item in raw_validation_values])
    raw_scatter = np.nanmedian([
        np.nanmedian(training[amp]["models"]["raw_constant"]["scatter"][:20])
        for amp in AMPLIFIERS])
    print("    combined: q<20 %.6g -> %.6g; broad RMS %.6g -> %.6g; "
          "median wavelength scatter q<20=%.6g" %
          (raw_q20_legacy, raw_q20_corrected, raw_broad_legacy,
           raw_broad_corrected, raw_scatter))
    print("  combined median comparison:")
    for model in ADDITIVE_MODELS:
        rows = [row for row in summary_rows if row["model"] == model]
        print("    %s: q<20 reduction=%.6g; broad RMS reduction=%.6g; "
              "high-pass RMS change=%.6g" %
              (ADDITIVE_MODEL_LABELS[model],
               np.nanmedian([row["q20_absolute_reduction"] for row in rows]),
               np.nanmedian([row["broad_rms_legacy"] -
                             row["broad_rms_corrected"] for row in rows]),
               np.nanmedian([row["highpass_rms_corrected"] -
                             row["highpass_rms_legacy"] for row in rows])))
    positive = {}
    for model in ADDITIVE_MODELS:
        positive[model] = np.nanmedian([
            np.nanmedian(training[amp]["models"][model]["applied"][:20])
            for amp in AMPLIFIERS]) > 0.0
    print("  inferred correction positive at expected readout edge: %s" %
          ", ".join("%s=%s" % (model, "YES" if positive[model] else "NO")
                    for model in ADDITIVE_MODELS))
    print("  median inferred A(q<20) in Raw-spectrum units (e-/A):")
    for amp in AMPLIFIERS:
        amplitude = np.nanmedian(
            training[amp]["models"]["raw_constant"]["applied"][:20])
        print("    %s: %.6g" % (amp, amplitude))
    print("  no production model was selected by the diagnostic")


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("h5_pattern", nargs="?",
                        help="quoted H5 glob, e.g. '2*.h5'")
    parser.add_argument("--h5", dest="h5_file",
                        help="single H5 file/glob for --spectral-sky-test")
    parser.add_argument("--image", type=Path,
                        help="optional M101 image for binimage < 0.01 selection")
    parser.add_argument("--wave-center", type=float, default=4600.0)
    parser.add_argument("--half-width", type=float, default=20.0)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--exclude", action="append", default=[],
                        help="shot stem or comma-separated shot stems to exclude")
    parser.add_argument("--candidate-ftf", action="store_true",
                        help="run the experimental Raw master-science FTF test")
    parser.add_argument("--spectral-sky-test", action="store_true",
                        help="run only the selected Raw sky spectral test")
    parser.add_argument("--additive-bandaid", action="store_true",
                        help="run only the held-out Fibers additive test")
    parser.add_argument("--exposure", type=int, default=1,
                        help="1-based exposure for spectral/additive tests")
    args = parser.parse_args()
    if args.half_width <= 0.0:
        parser.error("--half-width must be positive")
    input_pattern = args.h5_file or args.h5_pattern
    if not input_pattern:
        parser.error("an H5 glob/path is required")
    files = resolved_h5_files(input_pattern)
    excludes = {token.strip() for value in args.exclude
                for token in value.split(",") if token.strip()}
    files = [path for path in files
             if Path(path).stem not in excludes
             and Path(path).name not in excludes]
    if not files:
        parser.error("no H5 files remain after pattern/exclusion filtering")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print("Supplied H5 pattern: %s" % input_pattern)
    print("H5 files resolved after exclusions: %d" % len(files))
    for path in files:
        print("  %s" % path)
    if excludes:
        print("Excluded shot identifiers: %s" % ", ".join(sorted(excludes)))
    if args.spectral_sky_test:
        if len(files) != 1:
            parser.error("--spectral-sky-test requires exactly one resolved H5; "
                         "select one file with --h5 or a single positional path")
        if args.image is None:
            parser.error("--spectral-sky-test requires --image for blank-sky "
                         "selection")
        image = load_blank_image(args.image)
        try:
            run_spectral_sky_test(files[0], args.exposure, image,
                                  args.output_dir)
        finally:
            image["hdul"].close()
        return
    if args.additive_bandaid:
        if len(files) != 1:
            parser.error("--additive-bandaid requires exactly one resolved H5; "
                         "select one file with --h5 or a single positional path")
        if args.image is None:
            parser.error("--additive-bandaid requires --image for blank-sky "
                         "selection")
        image = load_blank_image(args.image)
        try:
            run_additive_bandaid(
                files[0], args.exposure, image, args.output_dir,
                args.wave_center, args.half_width)
        finally:
            image["hdul"].close()
        return
    image = load_blank_image(args.image)
    results = []
    try:
        for path in files:
            results.append(_read_h5_file(
                path, args.wave_center, args.half_width, image))
    finally:
        if image is not None:
            image["hdul"].close()

    records = [record for result in results for record in result["records"]]
    profiles = {amp: [record for result in results
                      for record in result["profiles"][amp]]
                for amp in AMPLIFIERS}
    all_profiles = {}
    for field in ("f_rel", "delta_o", "delta_k", "delta_cube"):
        all_profiles[field] = {}
        for amp in AMPLIFIERS:
            median, p16, p84, n = _stack_profile(
                profiles[amp], field)
            all_profiles[field][amp] = {
                "median": median, "p16": p16, "p84": p84, "n": n}
    write_summary_csv(args.output_dir / "amplifier_residual_summary.csv",
                      records)
    write_profile_csv(args.output_dir / "amplifier_profiles.csv",
                      all_profiles)
    make_figures(args.output_dir, profiles, records)
    if args.candidate_ftf:
        run_candidate_ftf(files, args.output_dir, args.wave_center,
                          args.half_width)

    print("")
    print("Bookkeeping:")
    print("  H5 files: %d" % len(files))
    print("  exposures: %d per H5" % EXPECTED_EXPOSURES)
    print("  physical amplifier instances: %d" % len(records))
    print("  instances by amplifier: %s" % ", ".join(
        "%s=%d" % (amp, sum(record["amp"] == amp for record in records))
        for amp in AMPLIFIERS))
    print("  unique group sizes: %s" % sorted(set(
        result["unique_group_sizes"][0] for result in results)))
    print("  continuum band: %.3f +/- %.3f A (%d bins)" %
          (args.wave_center, args.half_width, results[0]["n_band"]))
    print("Rectified wavelength convention: Fibers stores no wavelength column; "
          "using quick_reduction's exact def_wave=np.linspace(3470,5540,1036).")
    print("FTF convention: stored fiber_to_fiber is the multiplicative response "
          "divided out of science (scispectra = scispectra / ftf); larger FTF "
          "therefore reduces the stored science spectrum.")
    print("Spectrum/sky convention: %s" % results[0]["scale_note"])
    print("Residual-sky selection follows the cube's >6 arcmin, valid-image, "
          "blank-image (<0.01 when --image is supplied), >=80%% finite, and "
          ">=20-fiber per-exposure criteria.")
    if args.image is not None:
        print("External image blankness is sampled at each fiber RA/Dec with "
              "the image WCS; no cube-sized image is constructed.")
    print("")
    print("Median edge diagnostics by amplifier:")
    for amp in AMPLIFIERS:
        subset = [record for record in records if record["amp"] == amp]
        print("  %s: median E_cube=%.6g, median F_edge_contrast=%.6g" %
              (amp, np.nanmedian([record["E_cube"] for record in subset]),
               np.nanmedian([record["F_edge_contrast"] for record in subset])))
        fit = _fit_rows(records, amp)
        print("      E_cube = %.6g + %.6g B; N=%d; Spearman=%.6g" %
              (fit["alpha"], fit["beta"], fit["n"], fit["spearman"]))
    max_delta = max(result["max_delta_difference"] for result in results)
    print("Maximum centrally referenced |delta_S_h5-delta_S_cube|: %.6g"
          % max_delta)
    print("Expected-edge sign (positive median E_cube):")
    for amp in AMPLIFIERS:
        subset = [record for record in records if record["amp"] == amp]
        median_e = np.nanmedian([record["E_cube"] for record in subset])
        print("  %s: %s (median E=%.6g)" %
              (amp, "YES" if median_e > 0.0 else "NO/INCONCLUSIVE", median_e))


if __name__ == "__main__":
    main()
