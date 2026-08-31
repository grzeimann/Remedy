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


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("h5_pattern", help="quoted H5 glob, e.g. '2*.h5'")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    files = resolved_h5_files(args.h5_pattern)
    if not files:
        parser.error("no H5 files matched pattern: %s" % args.h5_pattern)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print("Supplied H5 pattern: %s" % args.h5_pattern)
    print("Resolved H5 files: %d" % len(files))
    for path in files:
        print("  %s" % path)

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
