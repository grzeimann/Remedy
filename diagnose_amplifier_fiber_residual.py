#!/usr/bin/env python3
"""Diagnose amplifier-relative continuum residuals in Remedy M101 H5 files.

The utility is read-only. It uses only the rectified Fibers products and
processes one H5 at a time; it does not construct an image or inspect
Raw.spectrum.
"""

from argparse import ArgumentParser
import csv
import glob
from pathlib import Path
import warnings

import numpy as np
import tables
from astropy.io import fits
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


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("h5_pattern", help="quoted H5 glob, e.g. '2*.h5'")
    parser.add_argument("--image", type=Path,
                        help="optional M101 image for binimage < 0.01 selection")
    parser.add_argument("--wave-center", type=float, default=4600.0)
    parser.add_argument("--half-width", type=float, default=20.0)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--exclude", action="append", default=[],
                        help="shot stem or comma-separated shot stems to exclude")
    parser.add_argument("--candidate-ftf", action="store_true",
                        help="run the experimental Raw master-science FTF test")
    args = parser.parse_args()
    if args.half_width <= 0.0:
        parser.error("--half-width must be positive")
    files = resolved_h5_files(args.h5_pattern)
    excludes = {token.strip() for value in args.exclude
                for token in value.split(",") if token.strip()}
    files = [path for path in files
             if Path(path).stem not in excludes
             and Path(path).name not in excludes]
    if not files:
        parser.error("no H5 files remain after pattern/exclusion filtering")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print("Supplied H5 pattern: %s" % args.h5_pattern)
    print("H5 files resolved after exclusions: %d" % len(files))
    for path in files:
        print("  %s" % path)
    if excludes:
        print("Excluded shot identifiers: %s" % ", ".join(sorted(excludes)))
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
