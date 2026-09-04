#!/usr/bin/env python3
"""Broad, streaming comparison of two large VIRUS mosaic cubes.

The cubes are opened with FITS memory mapping and are processed one wavelength
plane at a time.  The script therefore does not load two multi-GB cubes into
RAM or retain a list of changed voxels.  With zero tolerances, the numerical
comparison is exact apart from treating NaN/NaN as unchanged.

Example:
    python compare_mosaic_cubes.py old_cube.fits revised_cube.fits \
        --output-dir cube_comparison

The default collapsed products are a representative continuum plane/window,
[O II] 3727, H-delta, H-gamma, H-beta, and [O III] 4959 and 5007.  Custom
products use:

    --line NAME:CENTER:HALF_WIDTH[:sum|mean]
"""

from argparse import ArgumentParser
import csv
from pathlib import Path

import numpy as np
from astropy.io import fits


DEFAULT_LINES = (
    ("continuum", 4600.0, 20.0, "median"),
    ("OII_3727", 3727.0, 6.0, "sum"),
    ("Hdelta_4102", 4101.74, 6.0, "sum"),
    ("Hgamma_4340", 4340.47, 6.0, "sum"),
    ("Hbeta_4861", 4861.33, 6.0, "sum"),
    ("OIII_4959", 4958.92, 6.0, "sum"),
    ("OIII_5007", 5006.84, 6.0, "sum"),
)

CRITICAL_LINES = DEFAULT_LINES[1:]


def support_state_counts(old, new):
    """Count finite and nonzero usable states without conflating them."""
    old = np.asarray(old)
    new = np.asarray(new)
    old_finite = np.isfinite(old)
    new_finite = np.isfinite(new)
    old_usable = old_finite & (old != 0.0)
    new_usable = new_finite & (new != 0.0)
    return {
        "n_pixels": int(old.size),
        "old_finite": int(old_finite.sum()),
        "new_finite": int(new_finite.sum()),
        "both_finite": int((old_finite & new_finite).sum()),
        "old_only_finite": int((old_finite & ~new_finite).sum()),
        "new_only_finite": int((~old_finite & new_finite).sum()),
        "neither_finite": int((~old_finite & ~new_finite).sum()),
        "old_usable": int(old_usable.sum()),
        "new_usable": int(new_usable.sum()),
        "both_usable": int((old_usable & new_usable).sum()),
        "old_only_usable": int((old_usable & ~new_usable).sum()),
        "new_only_usable": int((~old_usable & new_usable).sum()),
        "neither_usable": int((~old_usable & ~new_usable).sum()),
        "old_zero": int(np.count_nonzero(old == 0.0)),
        "new_zero": int(np.count_nonzero(new == 0.0)),
    }


def fraction(count, total):
    return float(count) / total if total else np.nan


def critical_line_row(line_name, reference, plane_index, wave, old, new,
                      diff, factor, factor_count):
    """Build one per-plane referee support/difference diagnostic row."""
    counts = support_state_counts(old, new)
    row = {
        "line_name": line_name,
        "reference_wavelength": float(reference),
        "wavelength": float(wave),
        "plane_index": int(plane_index),
        **counts,
        "changed_voxels": int(diff["changed"]),
        "changed_fraction": fraction(diff["changed"], counts["n_pixels"]),
        "max_abs_difference": float(diff["max_abs"]),
        "rms_abs_difference": float(diff["rms_abs"]),
        "bright_pixel_new_over_old": float(factor),
        "bright_pixel_ratio_n": int(factor_count),
    }
    for name in ("old_finite", "new_finite", "both_finite",
                 "old_only_finite", "new_only_finite", "neither_finite",
                 "old_usable", "new_usable", "both_usable",
                 "old_only_usable", "new_only_usable", "neither_usable"):
        row[name + "_fraction"] = fraction(row[name], counts["n_pixels"])
    return row


def summarize_critical_line(line_name, reference, rows):
    """Summarize support changes over one critical-line wavelength window."""
    total_pixels = sum(row["n_pixels"] for row in rows)
    summary = {
        "line_name": line_name,
        "reference_wavelength": float(reference),
        "number_of_wavelength_planes": len(rows),
        "n_pixels": total_pixels,
    }
    for name in ("old_finite", "new_finite", "old_usable", "new_usable",
                 "both_usable", "old_only_usable", "new_only_usable",
                 "changed_voxels"):
        total = sum(row[name] for row in rows)
        summary[name + ("_fraction" if name != "changed_voxels" else "")] = (
            fraction(total, total_pixels) if name != "changed_voxels" else int(total))
    summary["changed_fraction"] = fraction(
        sum(row["changed_voxels"] for row in rows), total_pixels)
    factors = np.asarray([row["bright_pixel_new_over_old"] for row in rows],
                         dtype=float)
    factors = factors[np.isfinite(factors)]
    summary["median_bright_pixel_new_over_old"] = (
        float(np.median(factors)) if factors.size else np.nan)
    for state in ("old_only_usable", "new_only_usable"):
        values = np.asarray([fraction(row[state], row["n_pixels"])
                             for row in rows], dtype=float)
        if values.size:
            index = int(np.nanargmax(values))
            summary["max_" + state + "_fraction"] = float(values[index])
            summary["max_" + state + "_wavelength"] = rows[index]["wavelength"]
        else:
            summary["max_" + state + "_fraction"] = np.nan
            summary["max_" + state + "_wavelength"] = np.nan
    return summary


def parse_line(value):
    parts = value.split(":")
    if len(parts) not in (3, 4):
        raise ValueError("line must be NAME:CENTER:HALF_WIDTH[:sum|mean]")
    name = parts[0]
    center = float(parts[1])
    half_width = float(parts[2])
    mode = parts[3].lower() if len(parts) == 4 else "sum"
    if not name or half_width < 0 or mode not in ("sum", "mean"):
        raise ValueError("invalid line specification: %s" % value)
    return name, center, half_width, mode


def wavelengths(header, nplane):
    """Return wavelength for each plane using the cube's spectral WCS."""
    crval = float(header.get("CRVAL3", 0.0))
    cdelt = float(header.get("CDELT3", 1.0))
    crpix = float(header.get("CRPIX3", 1.0))
    pixels = np.arange(nplane, dtype=float) + 1.0
    return crval + (pixels - crpix) * cdelt


def empty_stats():
    return {
        "pixels": 0,
        "finite": 0,
        "nan": 0,
        "inf": 0,
        "zero": 0,
        "sum": 0.0,
        "sum2": 0.0,
        "min": np.inf,
        "max": -np.inf,
    }


def update_stats(total, image):
    image = np.asarray(image)
    finite = np.isfinite(image)
    values = image[finite].astype(np.float64, copy=False)
    total["pixels"] += image.size
    total["finite"] += int(finite.sum())
    total["nan"] += int(np.isnan(image).sum())
    total["inf"] += int(np.isinf(image).sum())
    total["zero"] += int(np.count_nonzero(image == 0.0))
    if values.size:
        total["sum"] += float(values.sum(dtype=np.float64))
        total["sum2"] += float(np.square(values).sum(dtype=np.float64))
        total["min"] = min(total["min"], float(values.min()))
        total["max"] = max(total["max"], float(values.max()))


def finish_stats(stats):
    finite = stats["finite"]
    mean = stats["sum"] / finite if finite else np.nan
    variance = stats["sum2"] / finite - mean * mean if finite else np.nan
    return {
        **stats,
        "finite_fraction": finite / stats["pixels"] if stats["pixels"] else np.nan,
        "mean": mean,
        "std": np.sqrt(max(variance, 0.0)) if np.isfinite(variance) else np.nan,
        "min": stats["min"] if finite else np.nan,
        "max": stats["max"] if finite else np.nan,
    }


def short_stats(image):
    return finish_stats_from_image(image)


def finish_stats_from_image(image):
    stats = empty_stats()
    update_stats(stats, image)
    return finish_stats(stats)


def format_stats(label, stats):
    return (
        f"{label}: finite={stats['finite']}/{stats['pixels']} "
        f"({stats['finite_fraction']:.6g}), zero={stats['zero']}, "
        f"min={stats['min']:.6g}, max={stats['max']:.6g}, "
        f"mean={stats['mean']:.6g}, std={stats['std']:.6g}"
    )


def compare_arrays(old, new, atol, rtol):
    """Return changed count and finite-difference summary for two arrays."""
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        same = np.isclose(old, new, atol=atol, rtol=rtol, equal_nan=True)
    changed = ~same
    both_finite = np.isfinite(old) & np.isfinite(new)
    delta = np.abs(new[both_finite].astype(np.float64) -
                   old[both_finite].astype(np.float64))
    return {
        "changed": int(changed.sum()),
        "max_abs": float(delta.max()) if delta.size else 0.0,
        "rms_abs": float(np.sqrt(np.mean(delta * delta))) if delta.size else 0.0,
    }


def wavelength_factor(old, new):
    """Estimate new/old using bright, positive, mutually finite pixels."""
    valid = np.isfinite(old) & np.isfinite(new) & (old > 0.0) & (new > 0.0)
    if not np.any(valid):
        return np.nan, 0
    threshold = np.percentile(old[valid], 75.0)
    selected = valid & (old >= threshold)
    ratio = (new[selected] / old[selected]).astype(np.float64)
    ratio = ratio[np.isfinite(ratio)]
    if not ratio.size:
        return np.nan, 0
    # The median is a robust average and avoids ratios dominated by faint sky.
    return float(np.median(ratio)), int(ratio.size)


def collapse(data, indices, mode):
    """Collapse selected spectral planes without treating missing data as zero."""
    block = np.asarray(data[indices])
    finite = np.isfinite(block)
    count = finite.sum(axis=0)
    with np.errstate(invalid="ignore"):
        result = np.nansum(block, axis=0, dtype=np.float64)
        if mode == "mean":
            result = result / count
    result[count == 0] = np.nan
    return result.astype(np.float32)


def map_header(header):
    output = header.copy()
    for key in ("NAXIS3", "CRPIX3", "CRVAL3", "CDELT3", "CTYPE3", "CUNIT3"):
        output.remove(key, ignore_missing=True)
    output["NAXIS"] = 2
    output["WCSAXES"] = 2
    return output


def write_map(path, image, header):
    fits.PrimaryHDU(np.asarray(image, dtype=np.float32), header=header).writeto(
        path, overwrite=True
    )


def write_factor_plot(path, rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = np.asarray(rows, dtype=float)
    selected = np.isfinite(rows[:, 1]) & (rows[:, 0] >= 3550.0) & (rows[:, 0] <= 5450.0)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(rows[selected, 0], rows[selected, 1], lw=1.2)
    ax.axhline(1.0, color="k", ls="--", lw=0.8)
    ax.set(xlabel="Wavelength (Angstrom)", ylabel="Median factor (new / old)",
           title="Bright-pixel flux factor: recreated cube / baseline cube")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def write_critical_line_plot(path, rows, line_name, reference):
    """Plot finite/usable support changes across one critical-line window."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = sorted(rows, key=lambda row: row["wavelength"])
    wave = np.asarray([row["wavelength"] for row in rows], dtype=float)
    fig, axes = plt.subplots(2, 1, figsize=(7, 5), sharex=True,
                             constrained_layout=True)
    axes[0].plot(wave, [row["old_usable_fraction"] for row in rows],
                 "o-", ms=3, label="old usable")
    axes[0].plot(wave, [row["new_usable_fraction"] for row in rows],
                 "o-", ms=3, label="new usable")
    axes[1].plot(wave, [row["old_only_usable_fraction"] for row in rows],
                 "o-", ms=3, label="old-only usable")
    axes[1].plot(wave, [row["new_only_usable_fraction"] for row in rows],
                 "o-", ms=3, label="new-only usable")
    axes[0].set_ylabel("fraction")
    axes[1].set_ylabel("fraction")
    axes[1].set_xlabel("wavelength (Angstrom)")
    axes[0].set_title("%s support diagnostic (%.2f A)" %
                      (line_name, reference))
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("old_cube", type=Path)
    parser.add_argument("new_cube", type=Path)
    parser.add_argument("--atol", type=float, default=0.0,
                        help="absolute comparison tolerance (default: exact)")
    parser.add_argument("--rtol", type=float, default=0.0,
                        help="relative comparison tolerance (default: exact)")
    parser.add_argument("--slice", dest="slices", type=float, action="append",
                        help="representative wavelength to report; repeatable")
    parser.add_argument("--line", action="append", type=parse_line,
                        help="NAME:CENTER:HALF_WIDTH[:sum|mean]; repeatable")
    parser.add_argument("--output-dir", type=Path,
                        help="write report, changed-wavelength table, and line maps")
    parser.add_argument("--factor-plot", type=Path,
                        help="factor plot path; defaults to output-dir when supplied")
    args = parser.parse_args()

    if args.atol < 0 or args.rtol < 0:
        parser.error("--atol and --rtol must be non-negative")
    if not args.old_cube.is_file():
        parser.error("old cube does not exist: %s" % args.old_cube)
    if not args.new_cube.is_file():
        parser.error("new cube does not exist: %s" % args.new_cube)

    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)

    report = []
    factor_rows = []

    def say(message=""):
        print(message)
        report.append(message)

    with fits.open(args.old_cube, memmap=True) as old_hdul, \
            fits.open(args.new_cube, memmap=True) as new_hdul:
        old = old_hdul[0].data
        new = new_hdul[0].data
        old_header = old_hdul[0].header
        new_header = new_hdul[0].header

        if old is None or new is None or old.ndim != 3 or new.ndim != 3:
            parser.error("both inputs must contain a 3-D primary HDU")
        if old.shape != new.shape:
            parser.error("cube shapes differ: %s versus %s" % (old.shape, new.shape))

        nplane = old.shape[0]
        old_wave = wavelengths(old_header, nplane)
        new_wave = wavelengths(new_header, nplane)
        if not np.array_equal(old_wave, new_wave):
            parser.error("spectral WCS differs between cubes")

        say("Old cube: %s" % args.old_cube)
        say("New cube: %s" % args.new_cube)
        say("Shape: %s; wavelength range %.2f--%.2f A" %
            (old.shape, old_wave[0], old_wave[-1]))
        say("Comparison tolerances: atol=%g, rtol=%g" % (args.atol, args.rtol))

        old_total = empty_stats()
        new_total = empty_stats()
        changed_rows = []
        critical_rows = {name: [] for name, _, _, _ in CRITICAL_LINES}
        critical_by_plane = {}
        for line_name, reference, half_width, _ in CRITICAL_LINES:
            for plane_index in np.flatnonzero(
                    np.abs(old_wave - reference) <= half_width):
                critical_by_plane.setdefault(int(plane_index), []).append(
                    (line_name, reference))
        for i, wave in enumerate(old_wave):
            old_plane = np.asarray(old[i])
            new_plane = np.asarray(new[i])
            update_stats(old_total, old_plane)
            update_stats(new_total, new_plane)
            diff = compare_arrays(old_plane, new_plane, args.atol, args.rtol)
            factor, factor_count = wavelength_factor(old_plane, new_plane)
            factor_rows.append((wave, factor, factor_count))
            if diff["changed"]:
                changed_rows.append((i, wave, diff["changed"],
                                     diff["max_abs"], diff["rms_abs"]))
            for line_name, reference in critical_by_plane.get(i, ()):
                critical_rows[line_name].append(
                    critical_line_row(line_name, reference, i, wave,
                                      old_plane, new_plane, diff, factor,
                                      factor_count))

        say(format_stats("Old full cube", finish_stats(old_total)))
        say(format_stats("New full cube", finish_stats(new_total)))
        say("Changed voxels: %d/%d (%.6g%%)" %
            (sum(row[2] for row in changed_rows), old.size,
             100.0 * sum(row[2] for row in changed_rows) / old.size))
        say("Wavelengths containing changed voxels: %d/%d" %
            (len(changed_rows), nplane))
        if changed_rows:
            say("Changed wavelength details:")
            for i, wave, count, max_abs, rms_abs in changed_rows:
                say("  index=%4d wave=%8.2f A changed=%d max_abs=%g rms_abs=%g" %
                    (i, wave, count, max_abs, rms_abs))
        else:
            say("No changed voxels found.")

        say("Flux factor is robust median(new / old) over positive pixels above "
            "the old-cube 75th percentile per wavelength.")
        factor_values = np.asarray(factor_rows, dtype=float)
        factor_sel = ((factor_values[:, 0] >= 3550.0) &
                      (factor_values[:, 0] <= 5450.0) &
                      np.isfinite(factor_values[:, 1]))
        if np.any(factor_sel):
            say("Factor 3550--5450 A: median=%g, min=%g, max=%g" %
                (np.median(factor_values[factor_sel, 1]),
                 np.min(factor_values[factor_sel, 1]),
                 np.max(factor_values[factor_sel, 1])))

        critical_summaries = []
        say("Critical-line support diagnostics (descriptive only):")
        for line_name, reference, _, _ in CRITICAL_LINES:
            summary = summarize_critical_line(
                line_name, reference, critical_rows[line_name])
            critical_summaries.append(summary)
            say("  %s: planes=%d old/new finite=%.6g/%.6g, old/new usable="
                "%.6g/%.6g, shared/old-only/new-only usable="
                "%.6g/%.6g/%.6g, changed=%.6g; max old-only=%.6g at %.2f A, "
                "max new-only=%.6g at %.2f A" %
                (line_name, summary["number_of_wavelength_planes"],
                 summary["old_finite_fraction"], summary["new_finite_fraction"],
                 summary["old_usable_fraction"], summary["new_usable_fraction"],
                 summary["both_usable_fraction"],
                 summary["old_only_usable_fraction"],
                 summary["new_only_usable_fraction"],
                 summary["changed_fraction"],
                 summary["max_old_only_usable_fraction"],
                 summary["max_old_only_usable_wavelength"],
                 summary["max_new_only_usable_fraction"],
                 summary["max_new_only_usable_wavelength"]))

        output_header = map_header(old_header)
        slice_waves = args.slices
        if slice_waves is None:
            slice_waves = [line[1] for line in DEFAULT_LINES]
        say("Representative slices:")
        for requested in slice_waves:
            i = int(np.argmin(np.abs(old_wave - requested)))
            diff = compare_arrays(old[i], new[i], args.atol, args.rtol)
            say("  requested=%8.2f actual=%8.2f A index=%4d changed=%d "
                "max_abs=%g rms_abs=%g" %
                (requested, old_wave[i], i, diff["changed"],
                 diff["max_abs"], diff["rms_abs"]))
            if args.output_dir:
                tag = "slice_%07.2f" % old_wave[i]
                write_map(args.output_dir / (tag + "_old.fits"), old[i], output_header)
                write_map(args.output_dir / (tag + "_new.fits"), new[i], output_header)
                write_map(args.output_dir / (tag + "_difference.fits"),
                          new[i] - old[i], output_header)

        lines = args.line if args.line else list(DEFAULT_LINES)
        say("Collapsed products:")
        for name, center, half_width, mode in lines:
            indices = np.flatnonzero(np.abs(old_wave - center) <= half_width)
            if indices.size == 0:
                indices = np.array([int(np.argmin(np.abs(old_wave - center)))])
                say("  %s: no planes in requested window; using %.2f A" %
                    (name, old_wave[indices[0]]))
            old_map = collapse(old, indices, mode)
            new_map = collapse(new, indices, mode)
            diff = compare_arrays(old_map, new_map, args.atol, args.rtol)
            say("  %s: %.2f +/- %.2f A, %d plane(s), %s, changed_pixels=%d, "
                "max_abs=%g rms_abs=%g" %
                (name, center, half_width, indices.size, mode,
                 diff["changed"], diff["max_abs"], diff["rms_abs"]))
            support = support_state_counts(old_map, new_map)
            say("    usable support: shared=%d (%.6g), old-only=%d (%.6g), "
                "new-only=%d (%.6g), neither=%d (%.6g)" %
                (support["both_usable"],
                 fraction(support["both_usable"], support["n_pixels"]),
                 support["old_only_usable"],
                 fraction(support["old_only_usable"], support["n_pixels"]),
                 support["new_only_usable"],
                 fraction(support["new_only_usable"], support["n_pixels"]),
                 support["neither_usable"],
                 fraction(support["neither_usable"], support["n_pixels"])))
            say("    " + format_stats("old", short_stats(old_map)))
            say("    " + format_stats("new", short_stats(new_map)))
            if args.output_dir:
                write_map(args.output_dir / (name + "_old.fits"), old_map, output_header)
                write_map(args.output_dir / (name + "_new.fits"), new_map, output_header)
                write_map(args.output_dir / (name + "_difference.fits"),
                          new_map - old_map, output_header)

        say("Critical-line support maxima (descriptive; compare Hbeta with "
            "OIII_5007 without assigning a cause):")
        for summary in critical_summaries:
            say("  %s: max old-only usable=%.6g at %.2f A; max new-only "
                "usable=%.6g at %.2f A" %
                (summary["line_name"],
                 summary["max_old_only_usable_fraction"],
                 summary["max_old_only_usable_wavelength"],
                 summary["max_new_only_usable_fraction"],
                 summary["max_new_only_usable_wavelength"]))

    if args.output_dir:
        (args.output_dir / "comparison_report.txt").write_text("\n".join(report) + "\n")
        with (args.output_dir / "changed_wavelengths.csv").open("w") as stream:
            stream.write("index,wavelength_angstrom,changed_voxels,max_abs,rms_abs\n")
            for row in changed_rows:
                stream.write("%d,%.6f,%d,%g,%g\n" % row)
        with (args.output_dir / "flux_factor_vs_wavelength.csv").open("w") as stream:
            stream.write("wavelength_angstrom,factor_new_over_old,n_pixels\n")
            for row in factor_rows:
                stream.write("%.6f,%g,%d\n" % row)
        critical_fields = [
            "line_name", "reference_wavelength", "wavelength", "plane_index",
            "n_pixels", "old_finite", "new_finite", "both_finite",
            "old_only_finite", "new_only_finite", "neither_finite",
            "old_finite_fraction", "new_finite_fraction",
            "both_finite_fraction", "old_only_finite_fraction",
            "new_only_finite_fraction", "neither_finite_fraction",
            "old_usable", "new_usable", "both_usable", "old_only_usable",
            "new_only_usable", "neither_usable", "old_usable_fraction",
            "new_usable_fraction", "both_usable_fraction",
            "old_only_usable_fraction", "new_only_usable_fraction",
            "neither_usable_fraction", "old_zero", "new_zero",
            "changed_voxels", "changed_fraction", "max_abs_difference",
            "rms_abs_difference", "bright_pixel_new_over_old",
            "bright_pixel_ratio_n",
        ]
        with (args.output_dir / "critical_line_wavelength_diagnostics.csv").open(
                "w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=critical_fields)
            writer.writeheader()
            for line_name, _, _, _ in CRITICAL_LINES:
                writer.writerows(critical_rows[line_name])
        summary_fields = [
            "line_name", "reference_wavelength", "number_of_wavelength_planes",
            "n_pixels", "old_finite_fraction", "new_finite_fraction",
            "old_usable_fraction", "new_usable_fraction",
            "both_usable_fraction", "old_only_usable_fraction",
            "new_only_usable_fraction", "changed_voxels", "changed_fraction",
            "median_bright_pixel_new_over_old",
            "max_old_only_usable_fraction", "max_old_only_usable_wavelength",
            "max_new_only_usable_fraction", "max_new_only_usable_wavelength",
        ]
        with (args.output_dir / "critical_line_summary.csv").open(
                "w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=summary_fields)
            writer.writeheader()
            writer.writerows(critical_summaries)
        for line_name, reference, _, _ in CRITICAL_LINES:
            write_critical_line_plot(
                args.output_dir / (line_name + "_support_diagnostic.png"),
                critical_rows[line_name], line_name, reference)
        factor_plot = args.factor_plot or (args.output_dir / "flux_factor_vs_wavelength.png")
        write_factor_plot(factor_plot, factor_rows)
        print("Wrote comparison products to %s" % args.output_dir)
    elif args.factor_plot:
        write_factor_plot(args.factor_plot, factor_rows)


if __name__ == "__main__":
    main()
