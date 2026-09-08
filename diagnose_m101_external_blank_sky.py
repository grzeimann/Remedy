#!/usr/bin/env python3
"""Diagnose an external-image definition of blank VIRUS fibers for M101.

This script is deliberately independent of spectroscopy.  It reads only the
Info table from each H5 (sky coordinates and hardware/exposure bookkeeping),
measures a circular VIRUS-like aperture on an external FITS image, and writes
diagnostic tables, plots, and a JSON summary.  It never reads Fibers.spectrum
or Fibers.skyspectrum, applies a sky subtraction, or modifies an H5 file.
"""

from argparse import ArgumentParser
import csv
import glob
import json
from pathlib import Path
import time
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
import tables
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
from scipy.ndimage import convolve, map_coordinates

import diagnose_m101_hierarchical as validated_m101


M101_RA_DEG = 210.800
M101_DEC_DEG = 54.333
DEFAULT_FIBER_RADIUS_ARCSEC = 0.75
DEFAULT_GUARD_RADIUS_ARCSEC = 0.0
DEFAULT_BLANK_SIGMA = 3.0
DEFAULT_MINIMUM_APERTURE_VALID_FRACTION = 0.9
DEFAULT_REFERENCE_RADIUS_ARCMIN = 6.0
SENSITIVITY_SIGMAS = (2.0, 2.5, 3.0, 3.5, 4.0, 5.0)
RADIAL_CUTS_ARCMIN = (None, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0)


def as_text(value):
    if isinstance(value, (bytes, np.bytes_)):
        return value.decode("utf-8", errors="replace").strip()
    return str(value).strip()


def finite_scale(values):
    """Return the usual robust MAD scale in the native image units."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    if mad > 0.0:
        return 1.4826 * mad
    return float(np.std(values))


def expand_h5_inputs(items):
    paths = []
    for item in items:
        expanded = str(Path(item).expanduser())
        matches = sorted(glob.glob(expanded))
        if not matches and Path(expanded).is_file():
            matches = [expanded]
        paths.extend(matches)
    unique = []
    seen = set()
    for path in paths:
        resolved = str(Path(path).resolve())
        if resolved not in seen:
            seen.add(resolved)
            unique.append(Path(resolved))
    if not unique:
        raise ValueError("no H5 files matched the supplied paths/patterns")
    return unique


def load_external_image(path):
    with fits.open(path, memmap=True) as hdul:
        data = np.asarray(hdul[0].data, dtype=np.float32).copy()
        header = hdul[0].header.copy()
    if data.ndim != 2:
        raise ValueError("external FITS primary HDU must be a 2-D image")
    try:
        wcs = WCS(header).celestial
    except Exception as exc:
        raise ValueError("external FITS does not contain a usable celestial WCS") from exc
    scales = np.asarray(proj_plane_pixel_scales(wcs), dtype=float) * 3600.0
    if scales.size != 2 or not np.all(np.isfinite(scales)) or np.any(scales <= 0.0):
        raise ValueError("external WCS has no usable two-axis pixel scale")
    finite = np.isfinite(data)
    if not finite.any():
        raise ValueError("external image contains no finite pixels")
    median = float(np.median(data[finite]))
    robust = finite_scale(data[finite])
    # Keep a small thumbnail for the spatial plot, so the full image can be
    # reused in-place during convolution without retaining another large copy.
    step = max(1, int(np.ceil(max(data.shape) / 1800.0)))
    display = data[::step, ::step].copy()
    image = {
        "path": str(Path(path).resolve()),
        "data": data,
        "display": display,
        "display_step": step,
        "header": header,
        "wcs": wcs,
        "shape": tuple(int(v) for v in data.shape),
        "pixel_scales_arcsec": [float(v) for v in scales],
        "pixel_scale_arcsec": float(np.median(scales)),
        "bunit": header.get("BUNIT"),
        "finite_fraction": float(np.mean(finite)),
        "median": median,
        "robust_scale": robust,
    }
    return image


def wcs_footprint(wcs, shape):
    ny, nx = shape
    try:
        corners = np.asarray(wcs.calc_footprint(axes=(nx, ny)), dtype=float)
    except Exception:
        corners = np.asarray([], dtype=float)
    if corners.ndim != 2 or corners.shape[1] != 2:
        return []
    return [[float(ra), float(dec)] for ra, dec in corners]


def build_aperture_kernel(pixel_scales_arcsec, radius_arcsec):
    sx, sy = (float(v) for v in pixel_scales_arcsec)
    radius_x = radius_arcsec / sx
    radius_y = radius_arcsec / sy
    half_x = int(np.ceil(radius_x))
    half_y = int(np.ceil(radius_y))
    x = np.arange(-half_x, half_x + 1, dtype=float)
    y = np.arange(-half_y, half_y + 1, dtype=float)
    xx, yy = np.meshgrid(x, y)
    inside = (xx * sx) ** 2 + (yy * sy) ** 2 <= radius_arcsec ** 2
    if not inside.any():
        raise ValueError("requested aperture is smaller than one image pixel")
    kernel = inside.astype(np.float32)
    kernel /= np.sum(kernel)
    return kernel, {
        "radius_arcsec": float(radius_arcsec),
        "radius_x_pixels": float(radius_x),
        "radius_y_pixels": float(radius_y),
        "radius_median_pixels": float(radius_arcsec / np.median([sx, sy])),
        "kernel_shape": [int(v) for v in kernel.shape],
        "kernel_pixel_count": int(inside.sum()),
    }


def make_aperture_map(image, kernel):
    """Return aperture mean and finite-pixel coverage on the image grid."""
    data = image["data"]
    finite = np.isfinite(data)
    # The finite mask remains available while data is reused as the validity
    # input.  convolve() uses zero padding so edge coverage is measured.
    data[~finite] = 0.0
    aperture_sum = convolve(data, kernel, mode="constant", cval=0.0)
    data[:] = finite
    aperture_valid_fraction = convolve(data, kernel, mode="constant", cval=0.0)
    del data, finite
    aperture_mean = np.full(aperture_sum.shape, np.nan, dtype=np.float32)
    good = aperture_valid_fraction > 0.0
    np.divide(aperture_sum, aperture_valid_fraction, out=aperture_mean,
              where=good)
    return aperture_mean, aperture_valid_fraction


def estimate_background(values, bins=512):
    """Estimate the dominant mode and noise scale from the low side.

    The mode is the center of the most populated robust-range histogram bin.
    The scale uses the lower-side median-to-5th-percentile distance, divided
    by the corresponding normal-distribution distance (1.326 sigma).  This
    intentionally uses the background side and does not let the positive
    source tail inflate the estimate.
    """
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size < 100:
        raise ValueError("fewer than 100 valid aperture-image pixels")
    p01, p995 = np.percentile(values, [0.5, 99.5])
    robust = finite_scale(values)
    if not np.isfinite(robust) or robust <= 0.0:
        robust = max(float(np.std(values)), 1e-12)
    span = max(float(p995 - p01), 20.0 * robust, 1e-12)
    low = float(p01 - 0.02 * span)
    high = float(p995 + 0.02 * span)
    edges = np.linspace(low, high, bins + 1)
    counts, edges = np.histogram(values, bins=edges)
    mode_bin = int(np.argmax(counts))
    mode_center = float(0.5 * (edges[mode_bin] + edges[mode_bin + 1]))
    modal_values = values[(values >= edges[mode_bin]) &
                          (values < edges[mode_bin + 1])]
    mu_blank = float(np.median(modal_values)) if modal_values.size else mode_center

    lower = values <= mode_center
    lower_values = values[lower]
    if lower_values.size < 50:
        raise ValueError("too few low-side values for background scale")
    lower_median, lower_p5 = np.percentile(lower_values, [50.0, 5.0])
    sigma_blank = float((lower_median - lower_p5) / 1.326)
    if not np.isfinite(sigma_blank) or sigma_blank <= 0.0:
        sigma_blank = finite_scale(lower_values)
    if not np.isfinite(sigma_blank) or sigma_blank <= 0.0:
        raise ValueError("could not estimate a positive background scale")

    percentiles = {
        str(p): float(np.percentile(values, p))
        for p in (1, 5, 16, 50, 84, 95, 99)
    }
    return {
        "mu_blank": mu_blank,
        "sigma_blank": sigma_blank,
        "mode_center": mode_center,
        "mode_bin_count": int(counts[mode_bin]),
        "mode_bin_fraction": float(counts[mode_bin] / values.size),
        "lower_side_count": int(lower_values.size),
        "median": float(np.median(values)),
        "percentiles": percentiles,
        "n_values": int(values.size),
        "hist_edges": edges,
        "hist_counts": counts,
        "hist_underflow": int(np.sum(values < edges[0])),
        "hist_overflow": int(np.sum(values >= edges[-1])),
    }


def sample_aperture_map(aperture_mean, valid_fraction, wcs, ra, dec,
                        minimum_valid_fraction):
    x, y = wcs.world_to_pixel_values(ra, dec)
    ny, nx = aperture_mean.shape
    image_covered = (np.isfinite(x) & np.isfinite(y) &
                     (x >= 0.0) & (x <= nx - 1) &
                     (y >= 0.0) & (y <= ny - 1))
    sample_x = np.where(np.isfinite(x), x, 0.0)
    sample_y = np.where(np.isfinite(y), y, 0.0)
    aperture_value = map_coordinates(
        aperture_mean, [sample_y, sample_x], order=1, mode="constant",
        cval=np.nan, prefilter=False)
    aperture_valid = map_coordinates(
        valid_fraction, [sample_y, sample_x], order=1, mode="constant",
        cval=0.0, prefilter=False)
    valid_aperture = (image_covered & np.isfinite(aperture_value) &
                      np.isfinite(aperture_valid) &
                      (aperture_valid >= minimum_valid_fraction))
    aperture_value[~valid_aperture] = np.nan
    return x, y, aperture_value, aperture_valid, image_covered, valid_aperture


def read_h5_metadata(paths):
    """Read only Info metadata and validated exposure labels from each H5."""
    chunks = []
    for path in paths:
        with tables.open_file(path, mode="r") as h5:
            if not hasattr(h5.root, "Info"):
                raise ValueError("%s has no Info table" % path)
            info = h5.root.Info
            groups, labels = validated_m101.build_groups(info)
            del groups
            required = {"ra", "dec", "ifuslot", "ifuid", "specid", "amp"}
            if not required.issubset(set(info.colnames)):
                missing = sorted(required - set(info.colnames))
                raise ValueError("%s Info table lacks %s" % (path, ", ".join(missing)))
            chunks.append({
                "h5": path.name,
                "path": str(path),
                "ra": np.asarray(info.cols.ra[:], dtype=float),
                "dec": np.asarray(info.cols.dec[:], dtype=float),
                "ifuslot": np.asarray(info.cols.ifuslot[:]),
                "ifuid": np.asarray(info.cols.ifuid[:]),
                "specid": np.asarray(info.cols.specid[:]),
                "amp": np.asarray([as_text(v) for v in info.cols.amp[:]], dtype=object),
                "exposure": np.asarray(labels, dtype=int),
            })
    return chunks


def concatenate_chunks(chunks):
    keys = ("ra", "dec", "ifuslot", "ifuid", "specid", "amp", "exposure")
    result = {key: np.concatenate([chunk[key] for chunk in chunks]) for key in keys}
    result["h5"] = np.concatenate([
        np.full(chunk["ra"].size, chunk["h5"], dtype=object) for chunk in chunks
    ])
    result["h5_path"] = [chunk["path"] for chunk in chunks]
    result["row_index"] = np.concatenate([
        np.arange(chunk["ra"].size, dtype=int) for chunk in chunks
    ])
    return result


def angular_radius_arcmin(ra, dec):
    dra_arcmin = ((ra - M101_RA_DEG) * np.cos(np.deg2rad(M101_DEC_DEG)) * 60.0)
    ddec_arcmin = (dec - M101_DEC_DEG) * 60.0
    return np.hypot(dra_arcmin, ddec_arcmin)


def row_counts(data, h5_name, exposure, reference_radius):
    mask = data["h5"] == h5_name
    mask &= data["exposure"] == exposure
    valid = data["valid_aperture"]
    selected = mask & valid
    inside = selected & (data["radius_arcmin"] <= reference_radius)
    outside = selected & (data["radius_arcmin"] > reference_radius)
    blank = data["blank"]
    n_inside = int(inside.sum())
    n_outside = int(outside.sum())
    n_image = int((mask & data["image_covered"]).sum())
    n_valid = int(selected.sum())
    n_blank = int((selected & blank).sum())
    n_blank_inside = int((inside & blank).sum())
    n_blank_outside = int((outside & blank).sum())
    return {
        "H5": h5_name,
        "exposure": int(exposure),
        "N_total_fibers": int(mask.sum()),
        "N_image_covered": n_image,
        "N_valid_aperture": n_valid,
        "N_blank": n_blank,
        "blank_fraction": n_blank / n_valid if n_valid else np.nan,
        "N_inside_6arcmin": n_inside,
        "N_outside_6arcmin": n_outside,
        "N_blank_inside_6arcmin": n_blank_inside,
        "N_blank_outside_6arcmin": n_blank_outside,
        "N_nonblank_outside_6arcmin": n_outside - n_blank_outside,
        "blank_fraction_inside_6arcmin": n_blank_inside / n_inside if n_inside else np.nan,
        "blank_fraction_outside_6arcmin": n_blank_outside / n_outside if n_outside else np.nan,
    }


COUNT_FIELDS = (
    "H5", "exposure", "N_total_fibers", "N_image_covered",
    "N_valid_aperture", "N_blank", "blank_fraction",
    "N_inside_6arcmin", "N_outside_6arcmin",
    "N_blank_inside_6arcmin", "N_blank_outside_6arcmin",
    "N_nonblank_outside_6arcmin", "blank_fraction_inside_6arcmin",
    "blank_fraction_outside_6arcmin",
)


def aggregate_count(data, h5_name, exposure_label, reference_radius):
    if h5_name == "ALL_H5":
        mask = np.ones(data["ra"].shape, dtype=bool)
    else:
        mask = data["h5"] == h5_name
    if exposure_label != "ALL":
        mask &= data["exposure"] == int(exposure_label)
    valid = data["valid_aperture"]
    selected = mask & valid
    inside = selected & (data["radius_arcmin"] <= reference_radius)
    outside = selected & (data["radius_arcmin"] > reference_radius)
    n_inside, n_outside = int(inside.sum()), int(outside.sum())
    n_valid = int(selected.sum())
    n_blank = int((selected & data["blank"]).sum())
    n_blank_inside = int((inside & data["blank"]).sum())
    n_blank_outside = int((outside & data["blank"]).sum())
    return {
        "H5": h5_name,
        "exposure": exposure_label,
        "N_total_fibers": int(mask.sum()),
        "N_image_covered": int((mask & data["image_covered"]).sum()),
        "N_valid_aperture": n_valid,
        "N_blank": n_blank,
        "blank_fraction": n_blank / n_valid if n_valid else np.nan,
        "N_inside_6arcmin": n_inside,
        "N_outside_6arcmin": n_outside,
        "N_blank_inside_6arcmin": n_blank_inside,
        "N_blank_outside_6arcmin": n_blank_outside,
        "N_nonblank_outside_6arcmin": n_outside - n_blank_outside,
        "blank_fraction_inside_6arcmin": n_blank_inside / n_inside if n_inside else np.nan,
        "blank_fraction_outside_6arcmin": n_blank_outside / n_outside if n_outside else np.nan,
    }


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def write_csv(path, rows, fields):
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: "" if (isinstance(row.get(field), float) and
                                            not np.isfinite(row[field]))
                             else row.get(field, "") for field in fields})


def plot_histograms(path, field_hist, fiber_values, background, adopted_sigma):
    edges = field_hist["hist_edges"]
    fiber_values = np.asarray(fiber_values, dtype=float)
    fiber_values = fiber_values[np.isfinite(fiber_values)]
    fiber_percentiles = {
        str(p): float(np.percentile(fiber_values, p))
        for p in (1, 5, 16, 50, 84, 95, 99)
    } if fiber_values.size else {}
    combined = np.concatenate((
        np.asarray([field_hist["percentiles"]["1"],
                    field_hist["percentiles"]["99"]]),
        fiber_values if fiber_values.size else np.asarray([], dtype=float)))
    low, high = np.percentile(combined, [0.5, 99.5])
    threshold = background["mu_blank"] + adopted_sigma * background["sigma_blank"]
    low = min(low, background["mu_blank"] - 2.0 * background["sigma_blank"])
    high = max(high, threshold + background["sigma_blank"])
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        low, high = edges[0], edges[-1]
    bins = np.linspace(low, high, 180)
    field_values_counts, _ = np.histogram(
        np.asarray(field_hist["values_for_plot"], dtype=float), bins=bins)
    fiber_counts, _ = np.histogram(fiber_values, bins=bins)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=True, sharey=True)
    distributions = ((axes[0], field_values_counts, "External-image field",
                      field_hist["n_values"], field_hist["percentiles"]),
                     (axes[1], fiber_counts, "VIRUS fiber positions",
                      fiber_values.size, fiber_percentiles))
    for ax, counts, title, n, percentiles in distributions:
        centers = 0.5 * (bins[:-1] + bins[1:])
        width = np.diff(bins)
        threshold = background["mu_blank"] + adopted_sigma * background["sigma_blank"]
        ax.axvspan(bins[0], min(threshold, bins[-1]), color="tab:green",
                   alpha=0.16, label="classified blank")
        if threshold < bins[-1]:
            ax.axvspan(max(threshold, bins[0]), bins[-1], color="tab:red",
                       alpha=0.12, label="positive/source tail")
        ax.bar(centers, counts, width=width, color="0.45", alpha=0.8,
               edgecolor="none", label="aperture means")
        mu, sigma = background["mu_blank"], background["sigma_blank"]
        ax.axvline(mu, color="tab:blue", lw=1.8, label=r"$\mu_\mathrm{blank}$")
        ax.axvline(mu - sigma, color="tab:blue", ls="--", lw=1.1,
                   label=r"$\mu\pm\sigma$")
        ax.axvline(mu + sigma, color="tab:blue", ls="--", lw=1.1)
        ax.axvline(threshold, color="tab:red", lw=1.8,
                   label="adopted threshold")
        for p, style in ((16, ":"), (50, "-"), (84, ":")):
            ax.axvline(percentiles[str(p)], color="black",
                       ls=style, lw=0.8, alpha=0.75)
        if percentiles:
            text = ("N=%d\nmedian=%.4g\nP05/P95=%.4g/%.4g\nP01/P99=%.4g/%.4g" %
                    (n, percentiles["50"], percentiles["5"], percentiles["95"],
                     percentiles["1"], percentiles["99"]))
        else:
            text = "N=0"
        ax.text(0.98, 0.97, text, transform=ax.transAxes, ha="right",
                va="top", fontsize=8,
                bbox={"facecolor": "white", "alpha": 0.78, "edgecolor": "none"})
        ax.set_title(title)
        ax.set_xlabel("External-image aperture mean (native units)")
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("Number of aperture samples")
    fig.suptitle("External-image blank-sky threshold diagnostic")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_threshold_sensitivity(path, rows):
    sigma = np.asarray([row["blank_sigma"] for row in rows], dtype=float)
    count = np.asarray([row["N_blank"] for row in rows], dtype=float)
    fraction = np.asarray([row["blank_fraction"] for row in rows], dtype=float)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(sigma, count, "o-", color="tab:green", label="N blank")
    ax.set_xlabel("Blank threshold (sigma above background mode)")
    ax.set_ylabel("N blank fiber positions", color="tab:green")
    ax.tick_params(axis="y", labelcolor="tab:green")
    ax.grid(alpha=0.25)
    right = ax.twinx()
    right.plot(sigma, fraction, "s--", color="tab:blue", label="blank fraction")
    right.set_ylabel("Blank fraction of valid fibers", color="tab:blue")
    right.tick_params(axis="y", labelcolor="tab:blue")
    ax.set_title("Blank-fiber threshold sensitivity")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_radial(path, radius, valid, blank, reference_radius):
    values = radius[valid & np.isfinite(radius)]
    selected_blank = blank[valid & np.isfinite(radius)]
    if values.size == 0:
        return
    edges = np.arange(0.0, max(1.0, np.ceil(np.max(values)) + 1.0), 1.0)
    if edges.size < 2:
        edges = np.array([0.0, 1.0])
    total, _ = np.histogram(values, bins=edges)
    nblank, _ = np.histogram(values[selected_blank], bins=edges)
    centers = 0.5 * (edges[:-1] + edges[1:])
    fraction = np.divide(nblank, total, out=np.full(total.shape, np.nan, dtype=float),
                         where=total > 0)
    fig, ax = plt.subplots(figsize=(9, 5))
    width = 0.38
    ax.bar(centers - width / 2, total, width=width, color="0.65", label="total fiber count")
    ax.bar(centers + width / 2, nblank, width=width, color="tab:green", label="blank fiber count")
    ax.set_xlabel("Radius from M101 (arcmin), 1 arcmin bins")
    ax.set_ylabel("Fiber count")
    right = ax.twinx()
    right.plot(centers, fraction, "o-", color="tab:blue", label="blank fraction")
    right.set_ylabel("Blank fraction")
    right.set_ylim(-0.02, 1.02)
    ax.axvline(reference_radius, color="tab:red", ls="--",
               label="6 arcmin reference")
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = right.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, fontsize=8, loc="upper left")
    ax.set_title("External-image blank fraction versus radius")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_spatial_map(path, image, data, reference_radius):
    display = image["display"]
    finite_display = display[np.isfinite(display)]
    vmin, vmax = np.percentile(finite_display, [1.0, 99.5])
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        vmin, vmax = float(np.nanmin(finite_display)), float(np.nanmax(finite_display))
    stretch = np.arcsinh((display - vmin) / max(vmax - vmin, 1e-12))
    ny, nx = image["shape"]
    step = image["display_step"]
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(stretch, origin="lower", extent=(0, nx, 0, ny), cmap="gray",
              interpolation="nearest", norm=Normalize(vmin=0.0,
              vmax=float(np.nanpercentile(stretch, 99.8))))
    valid = data["valid_aperture"] & np.isfinite(data["x"]) & np.isfinite(data["y"])
    blank = valid & data["blank"]
    nonblank = valid & ~data["blank"]

    def display_indices(mask, maximum=30000):
        indexes = np.flatnonzero(mask)
        if indexes.size <= maximum:
            return indexes
        return indexes[np.linspace(0, indexes.size - 1, maximum, dtype=int)]

    ib = display_indices(blank)
    ins = display_indices(nonblank)
    ax.scatter(data["x"][ins], data["y"][ins], s=2, c="tab:red", alpha=0.35,
               linewidths=0, label="nonblank")
    ax.scatter(data["x"][ib], data["y"][ib], s=2, c="tab:cyan", alpha=0.45,
               linewidths=0, label="blank")
    theta = np.linspace(0.0, 2.0 * np.pi, 361)
    radius_deg = reference_radius / 60.0
    circle_ra = M101_RA_DEG + radius_deg * np.cos(theta) / np.cos(np.deg2rad(M101_DEC_DEG))
    circle_dec = M101_DEC_DEG + radius_deg * np.sin(theta)
    circle_x, circle_y = image["wcs"].world_to_pixel_values(circle_ra, circle_dec)
    ax.plot(circle_x, circle_y, color="yellow", lw=1.5, ls="--", label="6 arcmin reference")
    center_x, center_y = image["wcs"].world_to_pixel_values(M101_RA_DEG, M101_DEC_DEG)
    ax.plot(center_x, center_y, marker="+", color="yellow", ms=10, mew=1.5)
    ax.set_xlim(0, nx)
    ax.set_ylim(0, ny)
    ax.set_xlabel("External-image pixel x")
    ax.set_ylabel("External-image pixel y")
    ax.set_title("External image and VIRUS fiber blank classification")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_fiber_table(path, data, reference_radius):
    fields = ("H5", "exposure", "row_index", "SPECID", "IFUSLOT", "IFUID", "AMP",
              "RA", "Dec", "radius_arcmin", "image_x", "image_y",
              "aperture_value", "aperture_valid_fraction", "image_covered",
              "blank", "outside_reference_radius")
    with gzip_open_text(path) as stream:
        writer = csv.writer(stream)
        writer.writerow(fields)
        outside = data["radius_arcmin"] > reference_radius
        for i in range(data["ra"].size):
            writer.writerow((data["h5"][i], int(data["exposure"][i]), int(data["row_index"][i]),
                             int(data["specid"][i]), int(data["ifuslot"][i]), int(data["ifuid"][i]),
                             data["amp"][i], data["ra"][i], data["dec"][i], data["radius_arcmin"][i],
                             data["x"][i], data["y"][i], data["aperture_value"][i],
                             data["aperture_valid_fraction"][i], bool(data["image_covered"][i]),
                             bool(data["blank"][i]), bool(outside[i])))


def gzip_open_text(path):
    import gzip
    return gzip.open(path, "wt", newline="")


def radial_cut_rows(data):
    valid = data["valid_aperture"] & np.isfinite(data["radius_arcmin"])
    rows = []
    for cut in RADIAL_CUTS_ARCMIN:
        selected = valid if cut is None else valid & (data["radius_arcmin"] > cut)
        total = int(selected.sum())
        blank = int((selected & data["blank"]).sum())
        rows.append({
            "radial_cut": "none" if cut is None else "> %.1f arcmin" % cut,
            "minimum_radius_arcmin": "" if cut is None else cut,
            "N_valid_aperture": total,
            "N_blank": blank,
            "blank_fraction": blank / total if total else np.nan,
            "N_nonblank": total - blank,
        })
    return rows


def print_startup(args, h5_paths):
    print("External blank-sky diagnostic defaults/configuration")
    print("  H5 inputs:                         %d" % len(h5_paths))
    print("  external image:                    %s" % args.image)
    print("  fiber radius:                      %.3f arcsec" % args.fiber_radius_arcsec)
    print("  guard radius:                      %.3f arcsec" % args.guard_radius_arcsec)
    print("  effective aperture radius:         %.3f arcsec" %
          (args.fiber_radius_arcsec + args.guard_radius_arcsec))
    print("  blank threshold sigma:              %.3f" % args.blank_sigma)
    print("  minimum aperture valid fraction:   %.3f" % args.minimum_aperture_valid_fraction)
    print("  reference radius (diagnostic only): %.3f arcmin" % args.reference_radius_arcmin)
    print("  aperture statistic:                 mean in native image units")
    print("  spectra read:                       no")


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("h5", nargs="+", help="H5 paths or shell-style patterns")
    parser.add_argument("--image", required=True, help="external 2-D FITS image")
    parser.add_argument("--output-dir", default="external_blank_sky_diagnostic")
    parser.add_argument("--fiber-radius-arcsec", type=float, default=DEFAULT_FIBER_RADIUS_ARCSEC)
    parser.add_argument("--guard-radius-arcsec", type=float, default=DEFAULT_GUARD_RADIUS_ARCSEC)
    parser.add_argument("--blank-sigma", type=float, default=DEFAULT_BLANK_SIGMA)
    parser.add_argument("--minimum-aperture-valid-fraction", type=float,
                        default=DEFAULT_MINIMUM_APERTURE_VALID_FRACTION)
    parser.add_argument("--reference-radius-arcmin", type=float,
                        default=DEFAULT_REFERENCE_RADIUS_ARCMIN)
    parser.add_argument("--write-fiber-table", action="store_true")
    args = parser.parse_args()
    if args.fiber_radius_arcsec <= 0.0:
        parser.error("--fiber-radius-arcsec must be positive")
    if args.guard_radius_arcsec < 0.0:
        parser.error("--guard-radius-arcsec must be nonnegative")
    if args.blank_sigma <= 0.0:
        parser.error("--blank-sigma must be positive")
    if not 0.0 < args.minimum_aperture_valid_fraction <= 1.0:
        parser.error("--minimum-aperture-valid-fraction must be in (0, 1]")
    if args.reference_radius_arcmin <= 0.0:
        parser.error("--reference-radius-arcmin must be positive")

    started = time.perf_counter()
    timings = {}
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    h5_paths = expand_h5_inputs(args.h5)
    print_startup(args, h5_paths)

    t0 = time.perf_counter()
    image = load_external_image(args.image)
    timings["image_load"] = time.perf_counter() - t0
    print("External image")
    print("  filename:        %s" % image["path"])
    print("  dimensions:      %d x %d (ny x nx)" % image["shape"])
    print("  BUNIT:           %s" % (image["bunit"] if image["bunit"] is not None else "<not present>"))
    print("  pixel scale:     %.6g x %.6g arcsec/pixel (median %.6g)" %
          (*image["pixel_scales_arcsec"], image["pixel_scale_arcsec"]))
    print("  finite fraction: %.6f" % image["finite_fraction"])
    print("  image median:    %.8g" % image["median"])
    print("  robust scale:    %.8g" % image["robust_scale"])
    print("  WCS footprint:   %s" % image["wcs"].calc_footprint(axes=(image["shape"][1], image["shape"][0])))

    t0 = time.perf_counter()
    effective_radius = args.fiber_radius_arcsec + args.guard_radius_arcsec
    kernel, aperture_info = build_aperture_kernel(image["pixel_scales_arcsec"], effective_radius)
    timings["aperture_kernel_construction"] = time.perf_counter() - t0
    print("Aperture")
    print("  effective radius: %.6g arcsec" % effective_radius)
    print("  radius in image pixels: x=%.6g, y=%.6g, representative=%.6g" %
          (aperture_info["radius_x_pixels"], aperture_info["radius_y_pixels"],
           aperture_info["radius_median_pixels"]))
    print("  top-hat kernel: %s, %d included pixels" %
          (aperture_info["kernel_shape"], aperture_info["kernel_pixel_count"]))

    t0 = time.perf_counter()
    aperture_mean, aperture_valid_fraction = make_aperture_map(image, kernel)
    timings["aperture_map_convolution"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    field_valid = (aperture_valid_fraction >= args.minimum_aperture_valid_fraction) & np.isfinite(aperture_mean)
    field_values = np.asarray(aperture_mean[field_valid], dtype=np.float32)
    background = estimate_background(field_values)
    background["values_for_plot"] = field_values
    timings["threshold_inference"] = time.perf_counter() - t0
    threshold = background["mu_blank"] + args.blank_sigma * background["sigma_blank"]
    print("External-image background model")
    print("  sufficiently valid aperture pixels: %d" % field_values.size)
    print("  mu_blank:                         %.8g" % background["mu_blank"])
    print("  sigma_blank:                      %.8g" % background["sigma_blank"])
    print("  adopted threshold:                %.8g (%.3g sigma)" % (threshold, args.blank_sigma))
    print("  field median / robust scale:      %.8g / %.8g" %
          (background["median"], finite_scale(field_values)))
    print("  mode bin fraction:                %.6f" % background["mode_bin_fraction"])

    t0 = time.perf_counter()
    chunks = read_h5_metadata(h5_paths)
    data = concatenate_chunks(chunks)
    timings["h5_coordinate_reads"] = time.perf_counter() - t0
    n_positions = data["ra"].size
    print("H5 metadata: %d fiber positions from %d H5 files" % (n_positions, len(h5_paths)))

    t0 = time.perf_counter()
    (data["x"], data["y"], data["aperture_value"],
     data["aperture_valid_fraction"], data["image_covered"],
     data["valid_aperture"]) = sample_aperture_map(
         aperture_mean, aperture_valid_fraction, image["wcs"], data["ra"], data["dec"],
         args.minimum_aperture_valid_fraction)
    data["radius_arcmin"] = angular_radius_arcmin(data["ra"], data["dec"])
    timings["wcs_aperture_sampling"] = time.perf_counter() - t0
    del aperture_mean, aperture_valid_fraction

    t0 = time.perf_counter()
    data["blank"] = data["valid_aperture"] & (data["aperture_value"] <= threshold)

    fiber_values = data["aperture_value"][data["valid_aperture"]]
    threshold_rows = []
    for sigma in SENSITIVITY_SIGMAS:
        limit = background["mu_blank"] + sigma * background["sigma_blank"]
        valid = data["valid_aperture"]
        inside = valid & (data["radius_arcmin"] <= args.reference_radius_arcmin)
        outside = valid & (data["radius_arcmin"] > args.reference_radius_arcmin)
        blank_at_sigma = valid & (data["aperture_value"] <= limit)
        n_valid = int(valid.sum())
        n_blank = int(blank_at_sigma.sum())
        threshold_rows.append({
            "blank_sigma": sigma,
            "threshold": limit,
            "total_valid_fiber_positions": n_valid,
            "N_blank": n_blank,
            "blank_fraction": n_blank / n_valid if n_valid else np.nan,
            "N_blank_inside_6arcmin": int((blank_at_sigma & inside).sum()),
            "N_blank_outside_6arcmin": int((blank_at_sigma & outside).sum()),
        })

    count_rows = []
    h5_names = [chunk["h5"] for chunk in chunks]
    for h5_name in h5_names:
        for exposure in (1, 2, 3):
            count_rows.append(row_counts(data, h5_name, exposure, args.reference_radius_arcmin))
        count_rows.append(aggregate_count(data, h5_name, "ALL", args.reference_radius_arcmin))
    count_rows.append(aggregate_count(data, "ALL_H5", "ALL", args.reference_radius_arcmin))
    radial_rows = radial_cut_rows(data)
    valid = data["valid_aperture"]
    inside = valid & (data["radius_arcmin"] <= args.reference_radius_arcmin)
    outside = valid & (data["radius_arcmin"] > args.reference_radius_arcmin)
    n_valid = int(valid.sum())
    n_blank = int(data["blank"].sum())
    n_blank_inside = int((data["blank"] & inside).sum())
    n_blank_outside = int((data["blank"] & outside).sum())
    global_counts = {
        "total_fibers": int(n_positions),
        "image_covered_fibers": int(data["image_covered"].sum()),
        "valid_aperture_fibers": n_valid,
        "blank_fibers": n_blank,
        "blank_inside_6arcmin": n_blank_inside,
        "blank_outside_6arcmin": n_blank_outside,
        "nonblank_outside_6arcmin": int(outside.sum() - n_blank_outside),
        "fibers_inside_6arcmin": int(inside.sum()),
        "fibers_outside_6arcmin": int(outside.sum()),
        "blank_fraction_inside_6arcmin": n_blank_inside / inside.sum() if inside.sum() else np.nan,
        "blank_fraction_outside_6arcmin": n_blank_outside / outside.sum() if outside.sum() else np.nan,
    }
    timings["classification_counting"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    write_csv(output_dir / "external_blank_sky_counts.csv", count_rows, COUNT_FIELDS)
    write_csv(output_dir / "blank_threshold_sensitivity.csv", threshold_rows,
              ("blank_sigma", "threshold", "total_valid_fiber_positions", "N_blank",
               "blank_fraction", "N_blank_inside_6arcmin", "N_blank_outside_6arcmin"))
    write_csv(output_dir / "radial_cut_sensitivity.csv", radial_rows,
              ("radial_cut", "minimum_radius_arcmin", "N_valid_aperture", "N_blank",
               "blank_fraction", "N_nonblank"))
    if args.write_fiber_table:
        write_fiber_table(output_dir / "external_blank_fibers.csv.gz", data,
                          args.reference_radius_arcmin)
    plot_histograms(output_dir / "external_blank_sky_histograms.png", background,
                    fiber_values, background, args.blank_sigma)
    plot_threshold_sensitivity(output_dir / "blank_count_vs_threshold.png", threshold_rows)
    plot_radial(output_dir / "blank_fraction_vs_radius.png", data["radius_arcmin"],
                valid, data["blank"], args.reference_radius_arcmin)
    plot_spatial_map(output_dir / "external_blank_fiber_map.png", image, data,
                     args.reference_radius_arcmin)
    timings["output_plots"] = time.perf_counter() - t0

    exposure_blank = np.asarray([row["N_blank"] for row in count_rows
                                 if row["exposure"] in (1, 2, 3)], dtype=float)
    exposure_support = {
        "min": int(np.min(exposure_blank)) if exposure_blank.size else 0,
        "median": float(np.median(exposure_blank)) if exposure_blank.size else np.nan,
        "max": int(np.max(exposure_blank)) if exposure_blank.size else 0,
        "N_exposures_below_20": int(np.sum(exposure_blank < 20)),
        "N_exposures_below_50": int(np.sum(exposure_blank < 50)),
        "N_exposures_below_100": int(np.sum(exposure_blank < 100)),
    }
    timings["total_runtime"] = time.perf_counter() - started
    sampling_time = timings["wcs_aperture_sampling"] + timings["classification_counting"]
    positions_per_second = n_positions / sampling_time if sampling_time > 0 else np.inf
    timing_report = dict(timings)
    timing_report["fiber_positions"] = n_positions
    timing_report["positions_per_second"] = positions_per_second
    summary = {
        "image": {
            "path": image["path"],
            "filename": Path(image["path"]).name,
            "dimensions_ny_nx": list(image["shape"]),
            "BUNIT": image["bunit"],
            "pixel_scales_arcsec": image["pixel_scales_arcsec"],
            "pixel_scale_arcsec": image["pixel_scale_arcsec"],
            "finite_fraction": image["finite_fraction"],
            "median": image["median"],
            "robust_scale": image["robust_scale"],
            "wcs_footprint_ra_dec": wcs_footprint(image["wcs"], image["shape"]),
        },
        "aperture": {
            "fiber_radius_arcsec": args.fiber_radius_arcsec,
            "guard_radius_arcsec": args.guard_radius_arcsec,
            "effective_radius_arcsec": effective_radius,
            "statistic_used": "aperture mean in native image units",
            **aperture_info,
            "minimum_valid_fraction": args.minimum_aperture_valid_fraction,
        },
        "background_estimator": (
            "dominant histogram mode over sufficiently valid aperture-mean image "
            "pixels; sigma from the low-side median-to-5th-percentile distance "
            "divided by 1.326, with no positive-tail mixture fit"
        ),
        "mu_blank": background["mu_blank"],
        "sigma_blank": background["sigma_blank"],
        "adopted_blank_sigma": args.blank_sigma,
        "adopted_absolute_threshold": threshold,
        "background_qa": {key: value for key, value in background.items()
                          if key not in ("hist_edges", "hist_counts", "values_for_plot")},
        "threshold_sensitivity": threshold_rows,
        "counts": global_counts,
        "per_h5_exposure_counts": count_rows,
        "radial_cut_sensitivity": radial_rows,
        "exposure_support": exposure_support,
        "timing": timing_report,
        "h5_paths": [str(path) for path in h5_paths],
        "reference_radius_arcmin": args.reference_radius_arcmin,
        "reference_center_ra_deg": M101_RA_DEG,
        "reference_center_dec_deg": M101_DEC_DEG,
        "spectroscopy_used_for_classification": False,
    }
    with (output_dir / "external_blank_sky_summary.json").open("w") as stream:
        json.dump(json_safe(summary), stream, indent=2, sort_keys=True)

    print("External blank-sky timing")
    print("  image load:                  %.3f s" % timings["image_load"])
    print("  aperture-kernel construction: %.3f s" % timings["aperture_kernel_construction"])
    print("  aperture-map convolution:   %.3f s" % timings["aperture_map_convolution"])
    print("  threshold inference:        %.3f s" % timings["threshold_inference"])
    print("  H5 coordinate reads:        %.3f s" % timings["h5_coordinate_reads"])
    print("  WCS + aperture sampling:    %.3f s" % timings["wcs_aperture_sampling"])
    print("  classification/counting:    %.3f s" % timings["classification_counting"])
    print("  output/plots:               %.3f s" % timings["output_plots"])
    print("  total:                      %.3f s" % timings["total_runtime"])
    print("  fiber positions:            %d" % n_positions)
    print("  classifications per second: %.6g" % positions_per_second)

    print("\nExternal image background:")
    print("  mu_blank = %.8g" % background["mu_blank"])
    print("  sigma_blank = %.8g" % background["sigma_blank"])
    print("  adopted %.3g-sigma threshold = %.8g" % (args.blank_sigma, threshold))
    print("Fiber classification:")
    print("  image-covered fibers = %d; valid aperture fibers = %d" %
          (global_counts["image_covered_fibers"], global_counts["valid_aperture_fibers"]))
    print("  blank fibers = %d (%.3f%% of valid aperture fibers)" %
          (n_blank, 100.0 * n_blank / n_valid if n_valid else np.nan))
    print("  blank inside 6 arcmin = %d" % n_blank_inside)
    print("  blank outside 6 arcmin = %d" % n_blank_outside)
    print("  NONBLANK outside 6 arcmin = %d" % global_counts["nonblank_outside_6arcmin"])
    print("Exposure support:")
    print("  min / median / max blank fibers per H5 exposure = %d / %.1f / %d" %
          (exposure_support["min"], exposure_support["median"], exposure_support["max"]))
    print("  exposures with N_blank < 20 / 50 / 100 = %d / %d / %d" %
          (exposure_support["N_exposures_below_20"], exposure_support["N_exposures_below_50"],
           exposure_support["N_exposures_below_100"]))
    print("Threshold stability:")
    print("  N_blank at 2, 2.5, 3, 3.5, 4, 5 sigma = %s" %
          ", ".join(str(row["N_blank"]) for row in threshold_rows))

    mode_fraction = background["mode_bin_fraction"]
    clean = mode_fraction >= 0.01 and background["sigma_blank"] > 0.0
    if clean:
        clean_text = "appears cleanly identifiable from a dominant low-flux background mode"
    else:
        clean_text = "is not cleanly identifiable from a dominant low-flux mode"
    if global_counts["nonblank_outside_6arcmin"] > 0:
        radial_text = "the 6 arcmin cut still appears useful as an additional safety criterion"
    elif global_counts["blank_inside_6arcmin"] > 0:
        radial_text = "the 6 arcmin cut appears redundant for this image classification"
    else:
        radial_text = "the value of the 6 arcmin cut is inconclusive in the covered sample"
    if timings["aperture_map_convolution"] < 120.0 and positions_per_second >= 1.0e4:
        speed_text = "the method appears fast enough for later cube-builder integration"
    else:
        speed_text = "the method may need performance work before cube-builder integration"
    print("Diagnostic conclusion:")
    print("  The external-image blank population %s." % clean_text)
    print("  In this covered sample, %s." % radial_text)
    print("  With one-time map construction, %s." % speed_text)


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        main()
