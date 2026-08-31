#!/usr/bin/env python3
"""Measure native H5 science-spectrum masked fractions versus wavelength.

This is a read-only diagnostic.  It follows the M101 cube-builder convention
of accepting one quoted glob, resolves it with ``sorted(glob.glob(pattern))``,
and reads each H5 table in bounded row chunks.  It does not regenerate H5
files or alter a cube.
"""

from argparse import ArgumentParser
import csv
import glob
from pathlib import Path
import tempfile

import numpy as np
import tables


WAVE_MIN = 3470.0
WAVE_MAX = 5540.0
BIN_WIDTH = 1.0
M101_VELOCITY_KMS = 140.0
SPEED_OF_LIGHT_KMS = 299792.458
LINE_HALF_WIDTH = 5.0
CONTROL_OFFSETS = ((-30.0, -10.0), (10.0, 30.0))
MIN_PEAK_SAMPLES = 100
N_STRONGEST_PEAKS = 20
ROW_CHUNK = 512
FAST_ROW_CHUNK = 4096
FAST_LINE_HALF_WIDTH = 6.0
FAST_CONTINUUM_OFFSETS = ((-30.0, -15.0), (15.0, 30.0))
FAST_MIN_REGION_SAMPLES = 3
FAST_MIN_FINITE_FRACTION = 0.8
FAST_NATIVE_BIN_HALF_WIDTH = 8.0

CRITICAL_LINES = (
    ("OII_3727", 3727.0),
    ("Hdelta_4102", 4101.74),
    ("Hgamma_4340", 4340.47),
    ("Hbeta_4861", 4861.33),
    ("OIII4959", 4958.92),
    ("OIII5007", 5006.84),
)


def usable_mask(values):
    """Return the finite, nonzero support mask used for optional QA checks."""
    values = np.asarray(values)
    return np.isfinite(values) & (values != 0.0)


def resolved_h5_files(pattern):
    """Resolve exactly the supplied nonrecursive glob, in cube-builder order."""
    return sorted(glob.glob(pattern))


def usable_exposure_numbers(info_table):
    """Infer row exposure numbers using the existing 112-fiber ordering."""
    nrows = int(info_table.nrows)
    if "ifuslot" in info_table.colnames:
        slots = np.asarray(info_table.cols.ifuslot[:])
        nslots = len(np.unique(slots))
    else:
        nslots = 0
    nexp = (int(nrows / float(448 * nslots))
            if nslots and nrows % (448 * nslots) == 0 else 1)
    exposure = (np.arange(nrows, dtype=np.int64) // 112) % nexp + 1
    return exposure.astype(np.int16), nexp


def _text_values(values):
    return np.asarray([
        value.decode("utf-8", errors="replace").strip()
        if isinstance(value, (bytes, np.bytes_)) else str(value).strip()
        for value in values
    ])


def native_spectrum_table(h5):
    """Find the native spectrum/wavelength pair in the existing H5 schema."""
    for name, wavelength_name in (("Raw", "wave"), ("Fibers", "wavelength")):
        if name not in h5.root._v_children:
            continue
        table = getattr(h5.root, name)
        columns = set(table.colnames)
        if "spectrum" in columns and wavelength_name in columns:
            return table, wavelength_name
    raise ValueError("H5 has no aligned native spectrum/wavelength table")


def empty_histogram(nbins):
    return (np.zeros(nbins, dtype=np.int64),
            np.zeros(nbins, dtype=np.int64))


def add_histogram(total, masked, wave, spectrum, wave_min=WAVE_MIN):
    """Accumulate total and nonfinite counts using actual native wavelengths."""
    wave = np.asarray(wave)
    spectrum = np.asarray(spectrum)
    flat_wave = wave.ravel().astype(float, copy=False)
    flat_spectrum = spectrum.ravel()
    valid_wave = np.isfinite(flat_wave)
    bins = np.floor(flat_wave[valid_wave] - wave_min).astype(np.int64)
    in_range = (bins >= 0) & (bins < len(total))
    bins = bins[in_range]
    values = flat_spectrum[valid_wave][in_range]
    total += np.bincount(bins, minlength=len(total)).astype(np.int64)
    masked += np.bincount(
        bins, weights=(~np.isfinite(values)).astype(np.int64),
        minlength=len(masked)).astype(np.int64)


def _group_histograms(group_values, wave, spectrum, histograms):
    """Accumulate optional exposure/amplifier/IFUSLOT subgroup histograms."""
    for group_name in np.unique(group_values):
        selected = group_values == group_name
        total, masked = histograms.setdefault(str(group_name),
                                              empty_histogram(len(next(iter(histograms.values()))[0]))
                                              if histograms else empty_histogram(
                                                  int(round((WAVE_MAX - WAVE_MIN) / BIN_WIDTH))))
        add_histogram(total, masked, wave[selected], spectrum[selected])


def inspect_h5_file(path, nbins):
    """Inspect one H5, retaining only histograms and compact metadata."""
    total, masked = empty_histogram(nbins)
    exposure_hist = {}
    amplifier_hist = {}
    ifuslot_hist = {}
    with tables.open_file(path, mode="r") as h5:
        table, wavelength_name = native_spectrum_table(h5)
        nrows = int(table.nrows)
        if "Info" not in h5.root._v_children:
            raise ValueError("H5 has no Info table: %s" % path)
        info = h5.root.Info
        if int(info.nrows) != nrows:
            raise ValueError("Info/native row mismatch in %s: %d/%d" %
                             (path, int(info.nrows), nrows))
        exposures, nexp = usable_exposure_numbers(info)
        if "amp" in info.colnames:
            amps = _text_values(info.cols.amp[:])
        else:
            amps = np.full(nrows, "UNKNOWN")
        if "ifuslot" in info.colnames:
            slots = np.asarray(info.cols.ifuslot[:]).astype(str)
        else:
            slots = np.full(nrows, "UNKNOWN")
        for start in range(0, nrows, ROW_CHUNK):
            stop = min(nrows, start + ROW_CHUNK)
            spectrum = table.read(start=start, stop=stop, field="spectrum")
            wave = table.read(start=start, stop=stop, field=wavelength_name)
            add_histogram(total, masked, wave, spectrum)
            _group_histograms(exposures[start:stop], wave, spectrum,
                              exposure_hist)
            _group_histograms(amps[start:stop], wave, spectrum, amplifier_hist)
            _group_histograms(slots[start:stop], wave, spectrum, ifuslot_hist)
        schema = {
            "table": "/%s" % table.name,
            "spectrum_column": "spectrum",
            "wavelength_column": wavelength_name,
            "nrows": nrows,
            "nwave": int(table.coldtypes["spectrum"].shape[0]),
            "nexp": nexp,
        }
    return {
        "path": Path(path), "total": total, "masked": masked,
        "exposure": exposure_hist, "amplifier": amplifier_hist,
        "ifuslot": ifuslot_hist, "schema": schema,
    }


def combined_histograms(results):
    total = sum((result["total"] for result in results),
                np.zeros_like(results[0]["total"]))
    masked = sum((result["masked"] for result in results),
                 np.zeros_like(results[0]["masked"]))
    return total, masked


def histogram_rows(total, masked):
    wavelength = WAVE_MIN + (np.arange(len(total)) + 0.5) * BIN_WIDTH
    finite = total - masked
    fraction_nan = np.divide(masked, total, out=np.full(total.shape, np.nan, dtype=float),
                             where=total > 0)
    return [{"wavelength": float(w), "n_total": int(n),
             "n_finite": int(f), "n_nan": int(m),
             "fraction_nan": float(frac)}
            for w, n, f, m, frac in zip(wavelength, total, finite, masked,
                                        fraction_nan)]


def window_stats(rows, lo, hi):
    selected = [row for row in rows if lo <= row["wavelength"] <= hi]
    n_total = sum(row["n_total"] for row in selected)
    n_nan = sum(row["n_nan"] for row in selected)
    return {
        "n_total": int(n_total), "n_nan": int(n_nan),
        "fraction_nan": float(n_nan / n_total) if n_total else np.nan,
    }


def line_summary(rows, name, rest_wave):
    observed = rest_wave * (1.0 + M101_VELOCITY_KMS / SPEED_OF_LIGHT_KMS)
    line = window_stats(rows, observed - LINE_HALF_WIDTH,
                        observed + LINE_HALF_WIDTH)
    controls = []
    for lo, hi in CONTROL_OFFSETS:
        controls.append(window_stats(rows, observed + lo, observed + hi))
    control_total = sum(item["n_total"] for item in controls)
    control_nan = sum(item["n_nan"] for item in controls)
    local_fraction = (control_nan / control_total
                      if control_total else np.nan)
    excess = (line["fraction_nan"] - local_fraction
              if np.isfinite(line["fraction_nan"]) and np.isfinite(local_fraction)
              else np.nan)
    enhancement = (line["fraction_nan"] / local_fraction
                   if local_fraction > 0.0 else np.nan)
    return {
        "line_name": name, "rest_wavelength": rest_wave,
        "expected_observed_wavelength": observed,
        "line_n_total": line["n_total"], "line_n_nan": line["n_nan"],
        "line_nan_fraction": line["fraction_nan"],
        "local_n_total": int(control_total), "local_n_nan": int(control_nan),
        "local_nan_fraction": local_fraction, "excess": excess,
        "enhancement": enhancement,
    }


def local_baseline(fraction_nan, total, half_width=30, exclude=5):
    baseline = np.full(len(fraction_nan), np.nan, dtype=float)
    for index in range(len(fraction_nan)):
        lo = max(0, index - half_width)
        hi = min(len(fraction_nan), index + half_width + 1)
        selected = np.arange(lo, hi)
        selected = selected[np.abs(selected - index) > exclude]
        selected = selected[total[selected] >= MIN_PEAK_SAMPLES]
        values = fraction_nan[selected]
        values = values[np.isfinite(values)]
        if values.size:
            baseline[index] = np.median(values)
    return baseline


def strongest_peaks(rows):
    fraction_nan = np.asarray([row["fraction_nan"] for row in rows], dtype=float)
    total = np.asarray([row["n_total"] for row in rows], dtype=int)
    baseline = local_baseline(fraction_nan, total)
    excess = fraction_nan - baseline
    candidates = np.flatnonzero(np.isfinite(excess) &
                                (total >= MIN_PEAK_SAMPLES) & (excess > 0.0))
    order = candidates[np.argsort(excess[candidates])[::-1]]
    chosen = []
    for index in order:
        if all(abs(int(index) - previous) > 3 for previous in chosen):
            chosen.append(int(index))
        if len(chosen) >= N_STRONGEST_PEAKS:
            break
    peaks = []
    for index in chosen:
        local = baseline[index]
        peaks.append({
            "wavelength": rows[index]["wavelength"],
            "fraction_nan": rows[index]["fraction_nan"],
            "local_baseline": float(local),
            "excess": float(excess[index]),
            "enhancement": (float(rows[index]["fraction_nan"] / local)
                            if local > 0.0 else np.nan),
            "n_total": rows[index]["n_total"],
            "n_nan": rows[index]["n_nan"],
        })
    return peaks, baseline


def write_histogram_csv(path, rows):
    fields = ["wavelength", "n_total", "n_finite", "n_nan", "fraction_nan"]
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_line_summary_csv(path, summaries):
    fields = ["line_name", "rest_wavelength", "expected_observed_wavelength",
              "line_n_total", "line_n_nan", "line_nan_fraction",
              "local_n_total", "local_n_nan", "local_nan_fraction",
              "excess", "enhancement"]
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summaries)


def write_peaks_csv(path, peaks):
    fields = ["wavelength", "fraction_nan", "local_baseline", "excess",
              "enhancement", "n_total", "n_nan"]
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(peaks)


def write_group_csv(path, results):
    fields = ["h5", "group_type", "group", "n_total", "n_finite", "n_nan",
              "fraction_nan"]
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for result in results:
            for group_type in ("exposure", "amplifier", "ifuslot"):
                for group, (total, masked) in result[group_type].items():
                    finite = total - masked
                    writer.writerow({
                        "h5": result["path"].name,
                        "group_type": group_type, "group": group,
                        "n_total": int(total.sum()), "n_finite": int(finite.sum()),
                        "n_nan": int(masked.sum()),
                        "fraction_nan": (float(masked.sum() / total.sum())
                                         if total.sum() else np.nan),
                    })


def write_mask_plot(path, rows, baseline, summaries):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    wave = np.asarray([row["wavelength"] for row in rows])
    fraction_nan = np.asarray([row["fraction_nan"] for row in rows])
    fig, axis = plt.subplots(figsize=(12, 5))
    axis.plot(wave, fraction_nan, ".", ms=2, label="raw 1-A bins")
    axis.plot(wave, baseline, lw=1.0, alpha=.8, label="local median baseline")
    for summary in summaries:
        center = summary["expected_observed_wavelength"]
        axis.axvspan(center - LINE_HALF_WIDTH, center + LINE_HALF_WIDTH,
                     color="tab:red", alpha=.12)
        axis.axvline(center, color="tab:red", lw=.6, alpha=.6)
        axis.text(center, axis.get_ylim()[1], summary["line_name"], rotation=90,
                  va="top", ha="right", fontsize=7)
    axis.set(xlabel="native wavelength (Angstrom)", ylabel="masked / nonfinite fraction",
             title="H5 native science-spectrum masking")
    axis.grid(alpha=.2)
    axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def _fast_expected_centers():
    return (
        4958.92 * (1.0 + M101_VELOCITY_KMS / SPEED_OF_LIGHT_KMS),
        5006.84 * (1.0 + M101_VELOCITY_KMS / SPEED_OF_LIGHT_KMS),
    )


def _fast_native_column_slice(table, wavelength_name, low, high):
    """Find a compact column slice; PyTables still reads vector columns whole."""
    reference = np.asarray(
        table.read(start=0, stop=1, field=wavelength_name)[0], dtype=float)
    candidate = np.flatnonzero(np.isfinite(reference)
                               & (reference >= low) & (reference <= high))
    if candidate.size == 0:
        raise ValueError("native wavelength table does not cover fast-test range")
    return max(0, int(candidate[0]) - 4), min(
        reference.size, int(candidate[-1]) + 5)


def _fast_region_mask(wave, center, offsets):
    low, high = offsets
    return (np.isfinite(wave) &
            (wave >= center + low) & (wave <= center + high))


def _fast_measure_chunk(spectrum, wave, center_4959, center_5007,
                        bin_min, n_native_bins):
    finite = np.isfinite(spectrum)
    line_4959 = _fast_region_mask(
        wave, center_4959, (-FAST_LINE_HALF_WIDTH, FAST_LINE_HALF_WIDTH))
    continuum_4959 = np.zeros(wave.shape, dtype=bool)
    for offsets in FAST_CONTINUUM_OFFSETS:
        continuum_4959 |= _fast_region_mask(wave, center_4959, offsets)

    n_line = line_4959.sum(axis=1)
    n_line_finite = (line_4959 & finite).sum(axis=1)
    n_continuum = continuum_4959.sum(axis=1)
    n_continuum_finite = (continuum_4959 & finite).sum(axis=1)
    continuum_values = np.where(continuum_4959 & finite, spectrum, np.nan)
    continuum = np.nanmedian(continuum_values, axis=1)
    line_values = np.where(line_4959 & finite,
                           spectrum - continuum[:, None], np.nan)
    spacing = np.nanmedian(np.abs(np.diff(wave, axis=1)), axis=1)
    spacing[~np.isfinite(spacing) | (spacing <= 0.0)] = 1.0
    f4959 = np.nansum(line_values, axis=1) * spacing
    valid_4959 = (
        (n_line >= FAST_MIN_REGION_SAMPLES) &
        (n_continuum >= FAST_MIN_REGION_SAMPLES) &
        (n_line_finite / np.maximum(n_line, 1)
         >= FAST_MIN_FINITE_FRACTION) &
        (n_continuum_finite / np.maximum(n_continuum, 1)
         >= FAST_MIN_FINITE_FRACTION) &
        np.isfinite(f4959) & (f4959 > 0.0))

    line_5007 = _fast_region_mask(
        wave, center_5007, (-FAST_LINE_HALF_WIDTH, FAST_LINE_HALF_WIDTH))
    control_5007 = np.zeros(wave.shape, dtype=bool)
    for offsets in FAST_CONTINUUM_OFFSETS:
        control_5007 |= _fast_region_mask(wave, center_5007, offsets)
    n_5007 = line_5007.sum(axis=1)
    nan_5007 = (line_5007 & ~finite).sum(axis=1)
    n_control = control_5007.sum(axis=1)
    nan_control = (control_5007 & ~finite).sum(axis=1)

    valid_indices = np.flatnonzero(valid_4959)
    valid_wave = wave[valid_indices]
    valid_finite = finite[valid_indices]
    finite_wave = np.isfinite(valid_wave)
    native_bin = np.floor(np.where(finite_wave, valid_wave - bin_min, 0.0)
                          ).astype(np.int64)
    native_bin_valid = ((native_bin >= 0) & (native_bin < n_native_bins)
                        & finite_wave)
    native_total = np.zeros((valid_indices.size, n_native_bins),
                            dtype=np.uint16)
    native_nan = np.zeros_like(native_total)
    for bin_number in range(n_native_bins):
        in_bin = native_bin_valid & (native_bin == bin_number)
        native_total[:, bin_number] = in_bin.sum(axis=1)
        native_nan[:, bin_number] = (
            (in_bin & ~valid_finite).sum(axis=1))

    return {
        "f4959": f4959[valid_indices].astype(np.float64),
        "any_5007_nan": (nan_5007[valid_indices] > 0),
        "n_5007": n_5007[valid_indices].astype(np.int32),
        "nan_5007": nan_5007[valid_indices].astype(np.int32),
        "n_control": n_control[valid_indices].astype(np.int32),
        "nan_control": nan_control[valid_indices].astype(np.int32),
        "native_total": native_total,
        "native_nan": native_nan,
    }


def _fast_empty_data():
    return {
        "f4959": [], "any_5007_nan": [], "n_5007": [], "nan_5007": [],
        "n_control": [], "nan_control": [], "native_total": [],
        "native_nan": [],
    }


def _fast_append_data(data, chunk):
    for key, value in chunk.items():
        data[key].append(value)


def _fast_join_data(data, n_native_bins):
    if not data["f4959"]:
        return {
            "f4959": np.empty(0, dtype=float),
            "any_5007_nan": np.empty(0, dtype=bool),
            "n_5007": np.empty(0, dtype=np.int32),
            "nan_5007": np.empty(0, dtype=np.int32),
            "n_control": np.empty(0, dtype=np.int32),
            "nan_control": np.empty(0, dtype=np.int32),
            "native_total": np.empty((0, n_native_bins), dtype=np.uint16),
            "native_nan": np.empty((0, n_native_bins), dtype=np.uint16),
        }
    joined = {}
    for key, values in data.items():
        joined[key] = np.concatenate(values, axis=0)
    return joined


def _fast_collect_file(path, center_4959, center_5007, bin_min,
                       n_native_bins):
    data = _fast_empty_data()
    with tables.open_file(path, mode="r") as h5:
        table, wavelength_name = native_spectrum_table(h5)
        low = min(center_4959 - 30.0, center_5007 - 30.0)
        high = max(center_4959 + 30.0, center_5007 + 30.0)
        column_start, column_stop = _fast_native_column_slice(
            table, wavelength_name, low, high)
        for start in range(0, int(table.nrows), FAST_ROW_CHUNK):
            stop = min(int(table.nrows), start + FAST_ROW_CHUNK)
            # Vector columns cannot be subselected in the PyTables read.
            # Slice immediately after reading and do no unrelated processing.
            spectrum = np.asarray(
                table.read(start=start, stop=stop, field="spectrum")
            )[:, column_start:column_stop]
            wave = np.asarray(
                table.read(start=start, stop=stop, field=wavelength_name)
            )[:, column_start:column_stop]
            _fast_append_data(
                data, _fast_measure_chunk(
                    spectrum, wave, center_4959, center_5007,
                    bin_min, n_native_bins))
        return _fast_join_data(data, n_native_bins), int(table.nrows)


def _fast_summary_row(label, indices, data):
    indices = np.asarray(indices, dtype=np.int64)
    f4959 = data["f4959"][indices]
    n_5007 = int(data["n_5007"][indices].sum())
    nan_5007 = int(data["nan_5007"][indices].sum())
    n_control = int(data["n_control"][indices].sum())
    nan_control = int(data["nan_control"][indices].sum())
    fraction_5007 = (nan_5007 / n_5007 if n_5007 else np.nan)
    fraction_control = (nan_control / n_control if n_control else np.nan)
    return {
        "selection": label,
        "n_fibers": int(indices.size),
        "median_F4959": float(np.median(f4959)) if f4959.size else np.nan,
        "p16_F4959": float(np.percentile(f4959, 16)) if f4959.size else np.nan,
        "p84_F4959": float(np.percentile(f4959, 84)) if f4959.size else np.nan,
        "fraction_5007_any_nan": (
            float(data["any_5007_nan"][indices].mean())
            if indices.size else np.nan),
        "n_5007_samples": n_5007,
        "n_5007_nan": nan_5007,
        "mean_5007_nan_fraction": fraction_5007,
        "n_control_samples": n_control,
        "n_control_nan": nan_control,
        "control_nan_fraction": fraction_control,
        "5007_minus_control_excess": (
            fraction_5007 - fraction_control
            if np.isfinite(fraction_5007) and np.isfinite(fraction_control)
            else np.nan),
    }


def _fast_brightness_groups(data):
    order = np.argsort(data["f4959"], kind="mergesort")
    n = order.size
    if n == 0:
        return [("all_valid", order)], order
    groups = []
    groups.append(("all_valid", order))
    for number, indices in enumerate(np.array_split(order, min(10, n))):
        groups.append(("quantile_%02d" % (number + 1), indices))
    for fraction in (0.50, 0.25, 0.10, 0.05, 0.01):
        count = max(1, int(np.ceil(n * fraction)))
        groups.append(("brightest_%g%%" % (100.0 * fraction),
                       order[-count:]))
    count = max(1, int(np.ceil(n * 0.50)))
    groups.append(("faintest_50%", order[:count]))
    return groups, order


def _fast_native_bin_rows(data, groups, bin_min, n_native_bins):
    rows = []
    for label, indices in groups:
        total = data["native_total"][indices].sum(axis=0)
        masked = data["native_nan"][indices].sum(axis=0)
        for bin_number in range(n_native_bins):
            rows.append({
                "selection": label,
                "wavelength": float(bin_min + bin_number + 0.5),
                "n_fibers": int(indices.size),
                "n_total": int(total[bin_number]),
                "n_nan": int(masked[bin_number]),
                "fraction_nan": (float(masked[bin_number] / total[bin_number])
                                 if total[bin_number] else np.nan),
            })
    return rows


def _write_fast_summary(path, rows):
    fields = ["selection", "n_fibers", "median_F4959", "p16_F4959",
              "p84_F4959", "fraction_5007_any_nan", "n_5007_samples",
              "n_5007_nan", "mean_5007_nan_fraction", "n_control_samples",
              "n_control_nan", "control_nan_fraction",
              "5007_minus_control_excess"]
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_fast_native_bins(path, rows):
    fields = ["selection", "wavelength", "n_fibers", "n_total", "n_nan",
              "fraction_nan"]
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_fast_plot(path, quantile_rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = np.asarray([row["median_F4959"] for row in quantile_rows])
    line = np.asarray([row["mean_5007_nan_fraction"]
                       for row in quantile_rows])
    control = np.asarray([row["control_nan_fraction"]
                          for row in quantile_rows])
    excess = line - control
    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    axes[0].plot(x, line, "o-", label="5007 line")
    axes[0].plot(x, control, "o-", label="5007 nearby continuum")
    axes[0].set_ylabel("masked / nonfinite fraction")
    axes[0].legend(fontsize=8)
    axes[1].plot(x, excess, "o-", color="tab:red")
    axes[1].axhline(0.0, color="0.4", lw=.8)
    axes[1].set(xlabel="median continuum-subtracted F4959 (native units)",
                ylabel="5007 minus control")
    for axis in axes:
        axis.grid(alpha=.2)
    if np.all(x > 0.0):
        axes[1].set_xscale("log")
    fig.suptitle("OIII5007 masking versus independent OIII4959 brightness")
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def _run_oiii_brightness_test(files, output_dir):
    center_4959, center_5007 = _fast_expected_centers()
    bin_min = np.floor(center_5007 - FAST_NATIVE_BIN_HALF_WIDTH)
    bin_max = np.ceil(center_5007 + FAST_NATIVE_BIN_HALF_WIDTH)
    n_native_bins = int(bin_max - bin_min)
    all_data = _fast_empty_data()
    rows_examined = 0
    for path in files:
        data, nrows = _fast_collect_file(
            path, center_4959, center_5007, bin_min, n_native_bins)
        rows_examined += nrows
        _fast_append_data(all_data, data)
    data = _fast_join_data(all_data, n_native_bins)
    groups, order = _fast_brightness_groups(data)
    summary_rows = [
        _fast_summary_row(label, indices, data)
        for label, indices in groups
        if (label == "all_valid" or label.startswith("quantile_")
            or label.startswith("brightest_"))
    ]
    native_rows = _fast_native_bin_rows(
        data, groups, bin_min, n_native_bins)
    _write_fast_summary(
        output_dir / "oiii5007_mask_vs_4959_brightness.csv", summary_rows)
    _write_fast_native_bins(
        output_dir / "oiii5007_native_bins_vs_4959_brightness.csv", native_rows)
    _write_fast_plot(
        output_dir / "oiii5007_mask_vs_4959_brightness.png",
        [row for row in summary_rows if row["selection"].startswith("quantile_")])

    report = []
    report.append("Fast OIII5007 masking-versus-OIII4959 brightness diagnostic")
    report.append("H5 files examined: %d" % len(files))
    report.append("Native rows examined: %d" % rows_examined)
    report.append("Observed centers (140 km/s): 4959=%.3f A, 5007=%.3f A" %
                  (center_4959, center_5007))
    report.append("Fibers with valid positive 4959 measurement: %d" %
                  data["f4959"].size)
    report.append("4959 validity requires >=%d samples and >=%.0f%% finite "
                  "support in both line and continuum regions." %
                  (FAST_MIN_REGION_SAMPLES, 100.0 * FAST_MIN_FINITE_FRACTION))
    report.append("5007 event is any nonfinite native sample in +/-%.1f A; "
                  "continuum is the matched +/-15--30 A sidebands." %
                  FAST_LINE_HALF_WIDTH)
    report.append("Targeted I/O note: PyTables reads each vector column as a "
                  "whole row vector; this path slices the required wavelength "
                  "region immediately and performs no full-spectrum histograms "
                  "or subgroup diagnostics.")
    report.append("")
    report.append("Brightness-conditioned summary:")
    for row in summary_rows:
        report.append(
            "  %-14s N=%d median_F4959=%.6g 5007_any=%.6g "
            "5007_nan=%.6g control_nan=%.6g excess=%.6g" %
            (row["selection"], row["n_fibers"], row["median_F4959"],
             row["fraction_5007_any_nan"], row["mean_5007_nan_fraction"],
             row["control_nan_fraction"], row["5007_minus_control_excess"]))
    bright_bins = [row for row in native_rows
                   if row["selection"] == "brightest_50%"]
    faint_bins = [row for row in native_rows
                  if row["selection"] == "faintest_50%"]
    paired = [(bright, faint) for bright, faint in zip(bright_bins, faint_bins)
              if np.isfinite(bright["fraction_nan"])
              and np.isfinite(faint["fraction_nan"])]
    if paired:
        strongest = max(paired, key=lambda pair: abs(
            pair[0]["fraction_nan"] - pair[1]["fraction_nan"]))
        bright, faint = strongest
        report.append("")
        report.append("Strongest individual 5007-region native bin difference "
                      "(brightest 50%% versus faintest 50%%):")
        report.append("  wavelength=%.1f A faint_fraction=%.6g "
                      "bright_fraction=%.6g bright-minus-faint=%.6g" %
                      (bright["wavelength"], faint["fraction_nan"],
                       bright["fraction_nan"],
                       bright["fraction_nan"] - faint["fraction_nan"]))
    (output_dir / "oiii5007_mask_vs_4959_brightness_report.txt").write_text(
        "\n".join(report) + "\n")
    print("\n".join(report))


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("h5_pattern", help="quoted H5 glob, e.g. '2*.h5'")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--oiii-brightness-test", action="store_true",
                        help="run only the targeted OIII5007-vs-4959 test")
    args = parser.parse_args()
    files = resolved_h5_files(args.h5_pattern)
    if not files:
        parser.error("no H5 files matched pattern: %s" % args.h5_pattern)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print("Supplied H5 pattern: %s" % args.h5_pattern)
    print("Resolved H5 files: %d" % len(files))
    for path in files:
        print("  %s" % path)
    if args.oiii_brightness_test:
        _run_oiii_brightness_test(files, args.output_dir)
        return

    nbins = int(round((WAVE_MAX - WAVE_MIN) / BIN_WIDTH))
    results = [inspect_h5_file(path, nbins) for path in files]
    total, masked = combined_histograms(results)
    rows = histogram_rows(total, masked)
    summaries = [line_summary(rows, name, rest)
                 for name, rest in CRITICAL_LINES]
    peaks, baseline = strongest_peaks(rows)
    write_histogram_csv(args.output_dir / "h5_nan_fraction_vs_wavelength.csv", rows)
    write_line_summary_csv(args.output_dir / "h5_nan_mask_line_summary.csv", summaries)
    write_peaks_csv(args.output_dir / "h5_nan_mask_strongest_peaks.csv", peaks)
    write_group_csv(args.output_dir / "h5_nan_mask_by_group.csv", results)
    write_mask_plot(args.output_dir / "h5_nan_fraction_vs_wavelength.png",
                    rows, baseline, summaries)

    report = []

    def say(message=""):
        print(message)
        report.append(message)

    total_samples = int(total.sum())
    masked_samples = int(masked.sum())
    say("Supplied H5 pattern: %s" % args.h5_pattern)
    say("Resolved H5 files: %d" % len(files))
    say("Exposures examined (row-order inference per file): %s" %
        ", ".join("%s=%d" % (result["path"].name, result["schema"]["nexp"])
                  for result in results))
    say("Exposure labels follow the existing 112-fiber row partition; "
        "Info.exp and Survey.exp are not used because they are not reliable "
        "exposure identifiers in these products.")
    say("Total native samples: %d; finite: %d; masked/nonfinite: %d; "
        "overall masked fraction: %.6g" %
        (total_samples, total_samples - masked_samples, masked_samples,
         masked_samples / total_samples if total_samples else np.nan))
    say("Native schema: %s" % "; ".join(
        "%s uses %s[%s] + %s[%s] (%d rows)" %
        (result["path"].name, result["schema"]["table"],
         result["schema"]["spectrum_column"], result["schema"]["table"],
         result["schema"]["wavelength_column"], result["schema"]["nrows"])
        for result in results))
    say("Expected line centers use an approximate M101 velocity of %.1f km/s; "
        "line windows are +/- %.1f A and controls are offsets %s." %
        (M101_VELOCITY_KMS, LINE_HALF_WIDTH, CONTROL_OFFSETS))
    say("Critical-line masking summary:")
    for summary in summaries:
        say("  %s: observed=%0.3f A line=%0.6g local=%0.6g excess=%0.6g "
            "enhancement=%0.6g" %
            (summary["line_name"], summary["expected_observed_wavelength"],
             summary["line_nan_fraction"], summary["local_nan_fraction"],
             summary["excess"], summary["enhancement"]))
    say("Strongest wavelength-localized masked-fraction peaks:")
    for peak in peaks:
        say("  %0.1f A: fraction=%0.6g baseline=%0.6g excess=%0.6g "
            "enhancement=%0.6g n_total=%d n_nan=%d" %
            (peak["wavelength"], peak["fraction_nan"], peak["local_baseline"],
             peak["excess"], peak["enhancement"], peak["n_total"], peak["n_nan"]))
    say("")
    say("Reduction-path interpretation from quick_reduction.py:")
    say("  clean_data(image, pixelmask) interpolates detector-row samples over "
        "static pixel-mask locations; it is used for dark/flat/arc/twilight "
        "and science detector images and is not itself an astronomical-line "
        "trigger.")
    say("  Dynamic rejection is separate: get_mask flags C1 > 5 profile-chi2 "
        "columns (expanded across neighboring native columns), low fiber-to-fiber "
        "response, bad/nonfinite or zero science samples, bad fibers, and bad "
        "amplifiers before the native Raw.spectrum table is written.")
    say("  Therefore these products are reported only as NaN/masked-sample "
        "fractions, not cosmic-ray fractions. A wavelength-localized excess is "
        "evidence consistent with, but does not prove, dynamic emission-line "
        "rejection.")
    say("  No independent pre-mask signal proxy was found in the native H5 "
        "tables, so no signal-versus-masking diagnostic was attempted.")
    (args.output_dir / "h5_nan_mask_diagnostic_report.txt").write_text(
        "\n".join(report) + "\n")


if __name__ == "__main__":
    main()
