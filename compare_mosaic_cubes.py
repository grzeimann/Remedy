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
[O II] 3727, H-beta, and [O III] 4959 and 5007.  Custom products use:

    --line NAME:CENTER:HALF_WIDTH[:sum|mean]
"""

from argparse import ArgumentParser
from pathlib import Path

import numpy as np
from astropy.io import fits


DEFAULT_LINES = (
    ("continuum", 4600.0, 20.0, "median"),
    ("OII_3727", 3727.0, 6.0, "sum"),
    ("Hbeta_4861", 4861.33, 6.0, "sum"),
    ("OIII_4959", 4958.92, 6.0, "sum"),
    ("OIII_5007", 5006.84, 6.0, "sum"),
)


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
            say("    " + format_stats("old", short_stats(old_map)))
            say("    " + format_stats("new", short_stats(new_map)))
            if args.output_dir:
                write_map(args.output_dir / (name + "_old.fits"), old_map, output_header)
                write_map(args.output_dir / (name + "_new.fits"), new_map, output_header)
                write_map(args.output_dir / (name + "_difference.fits"),
                          new_map - old_map, output_header)

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
        factor_plot = args.factor_plot or (args.output_dir / "flux_factor_vs_wavelength.png")
        write_factor_plot(factor_plot, factor_rows)
        print("Wrote comparison products to %s" % args.output_dir)
    elif args.factor_plot:
        write_factor_plot(args.factor_plot, factor_rows)


if __name__ == "__main__":
    main()
