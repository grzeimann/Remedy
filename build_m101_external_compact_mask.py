#!/usr/bin/env python3
"""Build an external-imaging-only compact/high-gradient mask for M101.

The ON and OFF images are analyzed independently with a PSF-scale
difference-of-Gaussians diagnostic.  Accepted compact positive features are
grown to a lower significance threshold and then dilated by the fiber radius
plus the astrometric margin.  The output is a uint8 raster with bits
``1=ON``, ``2=OFF`` and ``3=both``.

The intended future measurement-builder usage is::

    mask_data, mask_wcs = load_compact_mask(mask_path)
    compact_masked, inside_footprint = mask_radec(mask_data, mask_wcs, ra, dec)
    external_valid &= ~compact_masked

This lookup is one vectorized WCS transform followed by one NumPy array
lookup.  It does not match fiber coordinates against individual sources.
This script never reads VIRUS residuals or Bayesian results and does not alter
the input images or the existing measurement HDF5.
"""

from argparse import ArgumentParser
import csv
import json
from time import perf_counter
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
from scipy.ndimage import (binary_dilation, find_objects, gaussian_filter, label,
                           maximum_filter, map_coordinates)


BANDS = ("ON", "OFF")
MASK_BIT_ON = np.uint8(1)
MASK_BIT_OFF = np.uint8(2)


def _robust_location_scale(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan, np.nan
    location = float(np.median(values))
    scale = 1.4826 * float(np.median(np.abs(values - location)))
    if not np.isfinite(scale) or scale <= 0:
        scale = float(np.std(values))
    return location, max(scale, np.finfo(float).eps)


def _disk(radius_pixels):
    radius_pixels = int(max(0, np.ceil(float(radius_pixels))))
    if radius_pixels == 0:
        return np.ones((1, 1), dtype=bool)
    y, x = np.ogrid[-radius_pixels:radius_pixels + 1, -radius_pixels:radius_pixels + 1]
    return (x * x + y * y <= radius_pixels * radius_pixels)


def _read_celestial_image(path):
    path = Path(path).expanduser().resolve()
    with fits.open(path, memmap=True) as hdul:
        selected = None
        for hdu in hdul:
            if hdu.data is not None and np.ndim(hdu.data) == 2:
                selected = hdu
                break
        if selected is None:
            raise ValueError("no 2D image HDU found in %s" % path)
        data = np.array(selected.data, dtype=float, copy=True)
        header = selected.header.copy()
    if data.size == 0:
        raise ValueError("empty image in %s" % path)
    try:
        wcs = WCS(header).celestial
    except Exception as exc:
        raise ValueError("could not construct celestial WCS for %s" % path) from exc
    if not wcs.has_celestial:
        raise ValueError("image does not contain a celestial WCS: %s" % path)
    scales = np.asarray(proj_plane_pixel_scales(wcs), dtype=float) * 3600.0
    if scales.size != 2 or not np.all(np.isfinite(scales)) or np.any(scales <= 0):
        raise ValueError("could not obtain positive celestial pixel scales from %s" % path)
    return data, wcs, float(np.mean(scales)), path


def _pixel_scale_arcsec(wcs):
    scales = np.asarray(proj_plane_pixel_scales(wcs), dtype=float) * 3600.0
    if scales.size != 2 or not np.all(np.isfinite(scales)) or np.any(scales <= 0):
        raise ValueError("WCS has no finite positive pixel scale")
    return float(np.mean(scales))


def detect_compact_features(image, wcs, psf_fwhm_arcsec, fiber_radius_arcsec=0.0,
                            astrometry_margin_arcsec=0.0, peak_sigma=8.0,
                            grow_sigma=3.0, compactness_max=1.5,
                            broad_psf_factor=3.0, minimum_dilation_psf=0.0):
    """Detect compact positive high-pass features in one external image."""
    image = np.asarray(image, dtype=float)
    if image.ndim != 2:
        raise ValueError("image must be 2D")
    if not np.isfinite(psf_fwhm_arcsec) or psf_fwhm_arcsec <= 0:
        raise ValueError("PSF FWHM must be finite and positive")
    if peak_sigma <= grow_sigma or grow_sigma < 0 or compactness_max <= 0:
        raise ValueError("require peak_sigma > grow_sigma >= 0 and compactness_max > 0")
    pixel_scale = _pixel_scale_arcsec(wcs)
    sigma_psf = float(psf_fwhm_arcsec / 2.355 / pixel_scale)
    if sigma_psf <= 0.25:
        raise ValueError("PSF is undersampled on the supplied image grid: %.4g pixels" % sigma_psf)
    broad_sigma = max(float(broad_psf_factor) * sigma_psf, sigma_psf * 1.01)
    finite = np.isfinite(image)
    fill = float(np.nanmedian(image[finite])) if np.any(finite) else 0.0
    work = np.where(finite, image, fill)
    stage_start = perf_counter()
    compact_signal = gaussian_filter(work, sigma=sigma_psf, mode="nearest") - gaussian_filter(
        work, sigma=broad_sigma, mode="nearest")
    gaussian_dog_seconds = perf_counter() - stage_start
    component_stage_start = perf_counter()
    compact_location, compact_noise = _robust_location_scale(compact_signal[finite])
    significance = (compact_signal - compact_location) / compact_noise
    neighborhood = max(3, int(2 * np.ceil(sigma_psf) + 1))
    if neighborhood % 2 == 0:
        neighborhood += 1
    local_max = compact_signal == maximum_filter(compact_signal, size=neighborhood, mode="nearest")
    candidate = local_max & finite & (significance >= float(peak_sigma))
    grown_components, n_components = label(
        finite & (significance >= float(grow_sigma)), structure=np.ones((3, 3), dtype=bool))
    component_slices = find_objects(grown_components)
    core_mask = np.zeros(image.shape, dtype=bool)
    raw_mask = np.zeros(image.shape, dtype=bool)
    final_mask = np.zeros(image.shape, dtype=bool)
    features = []
    used_components = set()
    peak_indices = np.argwhere(candidate)
    peak_indices = sorted(peak_indices.tolist(), key=lambda pos: (-float(significance[tuple(pos)]), pos[0], pos[1]))
    dilation_arcsec = float(fiber_radius_arcsec) + float(astrometry_margin_arcsec) \
        + float(minimum_dilation_psf) * float(psf_fwhm_arcsec)
    dilation_pixels = max(0.0, dilation_arcsec / pixel_scale)
    dilation_structure = _disk(dilation_pixels)
    global_background = float(np.nanmedian(image[finite])) if np.any(finite) else np.nan

    for peak_y, peak_x in peak_indices:
        component_id = int(grown_components[peak_y, peak_x])
        if component_id == 0 or component_id in used_components:
            continue
        used_components.add(component_id)
        component_slice = component_slices[component_id - 1]
        if component_slice is None:
            continue
        component = grown_components[component_slice] == component_id
        component_y, component_x = np.where(component)
        if component_y.size == 0:
            continue
        local_radius = int(max(4, np.ceil(4.0 * sigma_psf)))
        y0, y1 = max(0, peak_y - local_radius), min(image.shape[0], peak_y + local_radius + 1)
        x0, x1 = max(0, peak_x - local_radius), min(image.shape[1], peak_x + local_radius + 1)
        patch = image[y0:y1, x0:x1]
        patch_component = grown_components[y0:y1, x0:x1] == component_id
        background_values = patch[np.isfinite(patch) & ~patch_component]
        local_background = float(np.median(background_values)) if background_values.size else global_background

        compact_cutout = compact_signal[component_slice]
        significance_cutout = significance[component_slice]
        weights = np.maximum(compact_cutout[component] - compact_location, 0.0)
        weight_sum = np.sum(weights)
        if weight_sum <= 0:
            continue
        weighted_y = float(np.sum(component_y * weights) / weight_sum)
        weighted_x = float(np.sum(component_x * weights) / weight_sum)
        variance_y = float(np.sum(weights * (component_y - weighted_y) ** 2) / weight_sum)
        variance_x = float(np.sum(weights * (component_x - weighted_x) ** 2) / weight_sum)
        measured_fwhm = float(2.355 * np.sqrt(max(0.5 * (variance_x + variance_y), 0.0)) * pixel_scale)
        fwhm_ratio = measured_fwhm / float(psf_fwhm_arcsec)
        if not np.isfinite(measured_fwhm) or fwhm_ratio > float(compactness_max):
            continue

        core = component & (significance_cutout >= float(peak_sigma))
        core_mask[component_slice] |= core
        raw_mask[component_slice] |= component

        component_y_slice, component_x_slice = component_slice
        pad_y = dilation_structure.shape[0] // 2
        pad_x = dilation_structure.shape[1] // 2
        padded_slice = (slice(max(0, component_y_slice.start - pad_y),
                              min(image.shape[0], component_y_slice.stop + pad_y)),
                        slice(max(0, component_x_slice.start - pad_x),
                              min(image.shape[1], component_x_slice.stop + pad_x)))
        padded_component = np.zeros((padded_slice[0].stop - padded_slice[0].start,
                                     padded_slice[1].stop - padded_slice[1].start), dtype=bool)
        local_y = slice(component_y_slice.start - padded_slice[0].start,
                        component_y_slice.stop - padded_slice[0].start)
        local_x = slice(component_x_slice.start - padded_slice[1].start,
                        component_x_slice.stop - padded_slice[1].start)
        padded_component[local_y, local_x] = component
        dilated = binary_dilation(padded_component, structure=dilation_structure)
        final_mask[padded_slice] |= dilated
        ra, dec = wcs.pixel_to_world_values(float(peak_x), float(peak_y))
        features.append({
            "RA": float(ra), "Dec": float(dec), "peak_pixel_x": int(peak_x), "peak_pixel_y": int(peak_y),
            "peak_value": float(image[peak_y, peak_x]), "local_background": local_background,
            "compact_signal_peak": float(compact_signal[peak_y, peak_x]),
            "contrast_significance": float(significance[peak_y, peak_x]),
            "measured_FWHM_arcsec": measured_fwhm, "PSF_FWHM_arcsec": float(psf_fwhm_arcsec),
            "FWHM_over_PSF": fwhm_ratio, "raw_footprint_area_pixels": int(np.sum(component)),
            "final_mask_area_pixels": int(np.sum(dilated)),
            "equivalent_final_radius_arcsec": float(np.sqrt(np.sum(dilated) * pixel_scale ** 2 / np.pi)),
        })
    return {"compact_signal": compact_signal, "significance": significance,
            "core_mask": core_mask, "raw_mask": raw_mask, "mask": final_mask,
            "features": features, "pixel_scale_arcsec": pixel_scale,
            "sigma_psf_pixels": sigma_psf, "compact_location": compact_location,
            "compact_noise": compact_noise, "dilation_arcsec": dilation_arcsec,
            "timings": {"gaussian_dog_seconds": gaussian_dog_seconds,
                         "component_processing_seconds": perf_counter() - component_stage_start}}


def _sky_corners(shape, wcs):
    height, width = shape
    x, y = np.meshgrid([0.0, width - 1.0], [0.0, height - 1.0])
    ra, dec = wcs.pixel_to_world_values(x.ravel(), y.ravel())
    return np.asarray(ra, dtype=float), np.asarray(dec, dtype=float)


def _same_grid(shape_a, wcs_a, shape_b, wcs_b):
    if tuple(shape_a) != tuple(shape_b):
        return False
    height, width = shape_a
    x, y = np.meshgrid([0.0, width / 2.0, width - 1.0], [0.0, height / 2.0, height - 1.0])
    ra_a, dec_a = wcs_a.pixel_to_world_values(x.ravel(), y.ravel())
    ra_b, dec_b = wcs_b.pixel_to_world_values(x.ravel(), y.ravel())
    dra = (np.asarray(ra_a) - np.asarray(ra_b) + 180.0) % 360.0 - 180.0
    return bool(np.nanmax(np.hypot(dra, np.asarray(dec_a) - np.asarray(dec_b))) < 1e-8)


def _tangent_offsets(ra, dec, ra0, dec0):
    dra = np.deg2rad((np.asarray(ra) - ra0 + 180.0) % 360.0 - 180.0)
    dec = np.deg2rad(dec); dec0 = np.deg2rad(dec0)
    denominator = np.sin(dec0) * np.sin(dec) + np.cos(dec0) * np.cos(dec) * np.cos(dra)
    xi = np.cos(dec) * np.sin(dra) / denominator
    eta = (np.cos(dec0) * np.sin(dec) - np.sin(dec0) * np.cos(dec) * np.cos(dra)) / denominator
    return np.rad2deg(xi), np.rad2deg(eta)


def _common_tan_wcs(shape_a, wcs_a, shape_b, wcs_b, pixel_scale_arcsec):
    ra_a, dec_a = _sky_corners(shape_a, wcs_a); ra_b, dec_b = _sky_corners(shape_b, wcs_b)
    ra_all = np.deg2rad(np.concatenate((ra_a, ra_b))); dec_all = np.deg2rad(np.concatenate((dec_a, dec_b)))
    x = np.cos(dec_all) * np.cos(ra_all); y = np.cos(dec_all) * np.sin(ra_all); z = np.sin(dec_all)
    ra0 = float(np.rad2deg(np.arctan2(np.sum(y), np.sum(x))) % 360.0)
    dec0 = float(np.rad2deg(np.arctan2(np.sum(z), np.hypot(np.sum(x), np.sum(y)))))
    xi, eta = _tangent_offsets(np.concatenate((ra_a, ra_b)), np.concatenate((dec_a, dec_b)), ra0, dec0)
    scale = float(pixel_scale_arcsec) / 3600.0
    xmin, xmax = float(np.min(xi)) - 2.0 * scale, float(np.max(xi)) + 2.0 * scale
    ymin, ymax = float(np.min(eta)) - 2.0 * scale, float(np.max(eta)) + 2.0 * scale
    width = int(np.ceil((xmax - xmin) / scale)) + 1
    height = int(np.ceil((ymax - ymin) / scale)) + 1
    if width * height > 100_000_000:
        raise ValueError("common WCS grid would be unnecessarily large: %d x %d" % (width, height))
    output_wcs = WCS(naxis=2)
    output_wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    output_wcs.wcs.crval = [ra0, dec0]
    output_wcs.wcs.cdelt = [scale, scale]
    output_wcs.wcs.crpix = [1.0 - xmin / scale, 1.0 - ymin / scale]
    output_wcs.wcs.cunit = ["deg", "deg"]
    return output_wcs, (height, width)


def _resample_mask(mask, input_wcs, output_wcs, output_shape):
    y, x = np.indices(output_shape, dtype=float)
    ra, dec = output_wcs.pixel_to_world_values(x, y)
    input_x, input_y = input_wcs.world_to_pixel_values(ra, dec)
    finite = np.isfinite(input_x) & np.isfinite(input_y)
    result = np.zeros(output_shape, dtype=bool)
    if np.any(finite):
        sampled = map_coordinates(mask.astype(float), [input_y[finite], input_x[finite]],
                                  order=0, mode="constant", cval=0.0)
        result[finite] = sampled != 0
    return result


def load_compact_mask(path):
    """Load the uint8 raster and celestial WCS for later vectorized lookup."""
    path = Path(path).expanduser().resolve()
    with fits.open(path, memmap=True) as hdul:
        data = np.asarray(hdul[0].data, dtype=np.uint8).copy()
        wcs = WCS(hdul[0].header).celestial
    if data.ndim != 2 or not wcs.has_celestial:
        raise ValueError("compact mask must be a 2D celestial raster: %s" % path)
    return data, wcs


def mask_radec(mask_data, wcs, ra, dec):
    """Vectorized mask lookup; returns ``(masked, inside_footprint)``."""
    mask_data = np.asarray(mask_data)
    if mask_data.ndim != 2:
        raise ValueError("mask_data must be 2D")
    ra, dec = np.broadcast_arrays(np.asarray(ra, dtype=float), np.asarray(dec, dtype=float))
    shape = ra.shape
    flat_ra, flat_dec = ra.ravel(), dec.ravel()
    x, y = wcs.world_to_pixel_values(flat_ra, flat_dec)
    finite = np.isfinite(x) & np.isfinite(y)
    ix = np.zeros(x.shape, dtype=np.int64); iy = np.zeros(y.shape, dtype=np.int64)
    ix[finite] = np.rint(x[finite]).astype(np.int64); iy[finite] = np.rint(y[finite]).astype(np.int64)
    height, width = mask_data.shape
    inside = finite & (ix >= 0) & (ix < width) & (iy >= 0) & (iy < height)
    masked = np.zeros(flat_ra.shape, dtype=bool)
    masked[inside] = mask_data[iy[inside], ix[inside]] != 0
    return masked.reshape(shape), inside.reshape(shape)


def _strip_private(feature):
    return {key: value for key, value in feature.items() if not key.startswith("_")}


def _distribution_summary(features, field):
    values = np.asarray([row[field] for row in features], dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"count": 0, "median": None, "p16": None, "p84": None, "min": None, "max": None}
    p16, median, p84 = np.percentile(values, (16, 50, 84))
    return {"count": int(values.size), "median": float(median), "p16": float(p16),
            "p84": float(p84), "min": float(np.min(values)), "max": float(np.max(values))}


def _count_overlapping_detections(on_features, off_features, radius_arcsec):
    if not on_features or not off_features:
        return 0
    on_ra = np.deg2rad([row["RA"] for row in on_features])[:, None]
    on_dec = np.deg2rad([row["Dec"] for row in on_features])[:, None]
    off_ra = np.deg2rad([row["RA"] for row in off_features])[None, :]
    off_dec = np.deg2rad([row["Dec"] for row in off_features])[None, :]
    cos_sep = (np.sin(on_dec) * np.sin(off_dec)
               + np.cos(on_dec) * np.cos(off_dec) * np.cos(on_ra - off_ra))
    separation = np.rad2deg(np.arccos(np.clip(cos_sep, -1.0, 1.0))) * 3600.0
    return int(np.sum(separation <= float(radius_arcsec)))


def _write_sources(path, features):
    fields = ("band", "RA", "Dec", "peak_pixel_x", "peak_pixel_y", "peak_value", "local_background",
              "compact_signal_peak", "contrast_significance", "measured_FWHM_arcsec", "PSF_FWHM_arcsec",
              "FWHM_over_PSF", "raw_footprint_area_pixels", "final_mask_area_pixels",
              "equivalent_final_radius_arcsec")
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for feature in features:
            writer.writerow({field: feature.get(field, "") for field in fields})


def _write_mask(path, mask, wcs, metadata):
    header = wcs.to_header()
    header["BUNIT"] = "bit flags"
    header["MASKBIT"] = "1=ON, 2=OFF, 3=both"
    for key, value in metadata.items():
        fits_key = ("M" + str(key).upper())[:8]
        if isinstance(value, (bool, np.bool_)):
            value = bool(value)
        elif isinstance(value, (np.integer, int)):
            value = int(value)
        elif isinstance(value, (np.floating, float)):
            value = float(value)
        else:
            value = str(value)[:68]
        try:
            header[fits_key] = value
        except (ValueError, TypeError):
            pass
    header.add_history("External compact/high-gradient morphology mask; no VIRUS or Bayesian quantity used.")
    fits.PrimaryHDU(data=np.asarray(mask, dtype=np.uint8), header=header).writeto(path, overwrite=True)


def _stretch(image):
    finite = np.asarray(image)[np.isfinite(image)]
    if finite.size == 0:
        return 0.0, 1.0
    return tuple(np.percentile(finite, (1.0, 99.5)))


def _plot_diagnostic(path, on_image, off_image, on_result, off_result, union_mask):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 5, figsize=(18, 7), constrained_layout=True)
    entries = (("ON", on_image, on_result), ("OFF", off_image, off_result))
    for row, (band, image, result) in enumerate(entries):
        vmin, vmax = _stretch(image)
        axes[row, 0].imshow(image, origin="lower", cmap="gray", vmin=vmin, vmax=vmax)
        axes[row, 0].set_title("%s external image" % band)
        compact = result["compact_signal"]
        limit = float(np.percentile(np.abs(compact[np.isfinite(compact)]), 99.5)) if np.any(np.isfinite(compact)) else 1.0
        axes[row, 1].imshow(compact, origin="lower", cmap="coolwarm", vmin=-limit, vmax=limit)
        axes[row, 1].set_title("%s compact signal" % band)
        axes[row, 2].imshow(result["significance"], origin="lower", cmap="magma", vmin=0, vmax=max(12, np.nanpercentile(result["significance"], 99.5)))
        axes[row, 2].imshow(np.ma.masked_where(~result["core_mask"], result["core_mask"]), origin="lower", cmap="Reds", alpha=.8)
        if result["features"]:
            axes[row, 2].scatter([f["peak_pixel_x"] for f in result["features"]],
                                 [f["peak_pixel_y"] for f in result["features"]],
                                 facecolors="none", edgecolors="cyan", s=35)
        axes[row, 2].set_title("%s detected cores" % band)
        axes[row, 3].imshow(result["mask"], origin="lower", cmap="Reds", vmin=0, vmax=1)
        axes[row, 3].set_title("%s grown + dilated mask" % band)
        for axis in axes[row, :4]:
            axis.set_xlabel("pixel x"); axis.set_ylabel("pixel y")
    axes[0, 4].imshow(union_mask, origin="lower", cmap="viridis", vmin=0, vmax=3)
    axes[0, 4].set_title("ON/OFF union bits")
    axes[0, 4].set_xlabel("pixel x"); axes[0, 4].set_ylabel("pixel y")
    axes[1, 4].axis("off")
    axes[1, 4].text(.02, .98, "bits:\n0 unmasked\n1 ON\n2 OFF\n3 both\n\nFinal lookup: mask != 0",
                    transform=axes[1, 4].transAxes, va="top", family="monospace")
    fig.suptitle("M101 external compact/high-gradient mask diagnostic")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run_synthetic_validation():
    """Small deterministic morphology and vectorized-lookup regression test."""
    rng = np.random.default_rng(20260904)
    shape = (192, 192); pixel_scale = .4; psf_fwhm = 2.0; sigma = psf_fwhm / 2.355 / pixel_scale
    wcs = WCS(naxis=2); wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]; wcs.wcs.crval = [210.0, 54.0]
    wcs.wcs.cdelt = [-pixel_scale / 3600.0, pixel_scale / 3600.0]; wcs.wcs.crpix = [96.0, 96.0]
    yy, xx = np.indices(shape); diffuse = 2.0 * np.exp(-((xx - 70) ** 2 + (yy - 105) ** 2) / (2 * 24 ** 2))

    def gaussian(x0, y0, width, amplitude):
        return amplitude * np.exp(-((xx - x0) ** 2 + (yy - y0) ** 2) / (2 * width ** 2))

    on = diffuse + gaussian(96, 96, sigma, 25.0) + gaussian(145, 58, 2.55, 20.0) \
        + gaussian(120, 140, sigma, 19.0) + gaussian(45, 45, 12.0, 35.0)
    off = diffuse + gaussian(96, 96, sigma, 22.0) + gaussian(145, 58, 2.55, 18.0) \
        + gaussian(72, 140, sigma, 19.0) + gaussian(45, 45, 12.0, 35.0)
    on += rng.normal(0, .025, shape); off += rng.normal(0, .025, shape)
    kwargs = dict(fiber_radius_arcsec=.8, astrometry_margin_arcsec=.4, peak_sigma=8.0,
                  grow_sigma=3.0, compactness_max=1.5)
    on_result = detect_compact_features(on, wcs, psf_fwhm, **kwargs)
    off_result = detect_compact_features(off, wcs, psf_fwhm, **kwargs)

    def nearest(features, x0, y0):
        return min(features, key=lambda f: (f["peak_pixel_x"] - x0) ** 2 + (f["peak_pixel_y"] - y0) ** 2) if features else None

    on_core = nearest(on_result["features"], 96, 96)
    on_extended = nearest(on_result["features"], 145, 58)
    broad = [f for f in on_result["features"] if (f["peak_pixel_x"] - 45) ** 2 + (f["peak_pixel_y"] - 45) ** 2 < 12 ** 2]
    if on_core is None or on_extended is None or on_core["FWHM_over_PSF"] > 1.5 or on_extended["FWHM_over_PSF"] > 1.5 or broad:
        raise AssertionError("synthetic compact morphology classification failed")
    if np.sum(on_result["mask"]) <= np.sum(on_result["core_mask"]):
        raise AssertionError("synthetic grown/dilated footprint did not expand")
    union = on_result["mask"].astype(np.uint8) * MASK_BIT_ON | off_result["mask"].astype(np.uint8) * MASK_BIT_OFF
    if not np.any(union == 3) or not np.any(union == 1) or not np.any(union == 2):
        raise AssertionError("synthetic ON/OFF union bits failed")
    data, inside = mask_radec(union, wcs, *wcs.pixel_to_world_values(
        np.asarray([96.0, 10.0, 180.0]), np.asarray([96.0, 10.0, 180.0])))
    outside, outside_inside = mask_radec(union, wcs, np.asarray([220.0]), np.asarray([54.0]))
    if not data[0] or not inside.all() or outside[0] or outside_inside[0]:
        raise AssertionError("synthetic vectorized WCS lookup failed")
    return {"status": "PASS", "on_detections": len(on_result["features"]),
            "off_detections": len(off_result["features"]),
            "on_compact_fwhm_over_psf": on_core["FWHM_over_PSF"],
            "on_extended_fwhm_over_psf": on_extended["FWHM_over_PSF"],
            "union_bits": {"ON": int(np.sum(union == 1)), "OFF": int(np.sum(union == 2)), "both": int(np.sum(union == 3))},
            "grown_beyond_core": True, "vectorized_lookup": True}


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--on-image", help="external ON image FITS path")
    parser.add_argument("--off-image", help="external OFF image FITS path")
    parser.add_argument("--psf-fwhm-on", type=float, help="ON PSF FWHM in arcsec")
    parser.add_argument("--psf-fwhm-off", type=float, help="OFF PSF FWHM in arcsec")
    parser.add_argument("--fiber-radius-arcsec", type=float,
                        help="fiber radius in arcsec; required for image builds")
    parser.add_argument("--astrometry-margin-arcsec", type=float,
                        help="astrometry safety margin in arcsec; required for image builds")
    parser.add_argument("--peak-sigma", type=float, default=8.0,
                        help="positive compact-signal peak threshold in robust sigma (default: 8)")
    parser.add_argument("--grow-sigma", type=float, default=3.0,
                        help="positive compact-signal footprint threshold in robust sigma (default: 3)")
    parser.add_argument("--compactness-max", type=float, default=1.5,
                        help="maximum measured FWHM / PSF FWHM (default: 1.5)")
    parser.add_argument("--broad-psf-factor", type=float, default=3.0,
                        help="broad Gaussian smoothing scale in PSF sigma units")
    parser.add_argument("--minimum-dilation-psf", type=float, default=0.0,
                        help="optional explicit minimum dilation in PSF FWHM units")
    parser.add_argument("--output-mask", default="m101_external_compact_mask.fits")
    parser.add_argument("--output-sources", default="m101_external_compact_mask_sources.csv")
    parser.add_argument("--output-diagnostic", default="m101_external_compact_mask_diagnostic.png")
    parser.add_argument("--output-summary", default="m101_external_compact_mask_summary.json")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--synthetic-test", action="store_true")
    args = parser.parse_args()
    if args.synthetic_test:
        print(json.dumps(run_synthetic_validation(), indent=2)); return
    required = ("on_image", "off_image", "psf_fwhm_on", "psf_fwhm_off",
                "fiber_radius_arcsec", "astrometry_margin_arcsec")
    missing = ["--" + name.replace("_", "-") for name in required if getattr(args, name) is None]
    if missing:
        parser.error("the following arguments are required for an image build: " + ", ".join(missing))
    if args.psf_fwhm_on <= 0 or args.psf_fwhm_off <= 0:
        raise SystemExit("PSF FWHM values must be positive")
    if args.fiber_radius_arcsec < 0 or args.astrometry_margin_arcsec < 0 or args.minimum_dilation_psf < 0:
        raise SystemExit("fiber radius, astrometry margin, and minimum dilation must be non-negative")
    if args.peak_sigma <= args.grow_sigma or args.grow_sigma < 0 or args.compactness_max <= 0:
        raise SystemExit("require peak-sigma > grow-sigma >= 0 and compactness-max > 0")
    output_mask = Path(args.output_mask).expanduser().resolve()
    output_sources = Path(args.output_sources).expanduser().resolve()
    output_diagnostic = Path(args.output_diagnostic).expanduser().resolve()
    output_summary = Path(args.output_summary).expanduser().resolve()
    input_on = Path(args.on_image).expanduser().resolve(); input_off = Path(args.off_image).expanduser().resolve()
    outputs = (output_mask, output_sources, output_diagnostic, output_summary)
    if any(output == input_path for output in outputs for input_path in (input_on, input_off)):
        raise SystemExit("outputs must not overwrite either input image")
    if not args.overwrite:
        existing = [str(path) for path in outputs if path.exists()]
        if existing:
            raise SystemExit("outputs exist; use --overwrite: %s" % ", ".join(existing))
    for output in outputs:
        output.parent.mkdir(parents=True, exist_ok=True)
    read_start = perf_counter()
    on_image, on_wcs, on_scale, _ = _read_celestial_image(input_on)
    off_image, off_wcs, off_scale, _ = _read_celestial_image(input_off)
    print("timing read images: %.3f s" % (perf_counter() - read_start))
    common = dict(fiber_radius_arcsec=args.fiber_radius_arcsec,
                  astrometry_margin_arcsec=args.astrometry_margin_arcsec,
                  peak_sigma=args.peak_sigma, grow_sigma=args.grow_sigma,
                  compactness_max=args.compactness_max, broad_psf_factor=args.broad_psf_factor,
                  minimum_dilation_psf=args.minimum_dilation_psf)
    on_result = detect_compact_features(on_image, on_wcs, args.psf_fwhm_on, **common)
    print("timing Gaussian/DoG ON: %.3f s" % on_result["timings"]["gaussian_dog_seconds"])
    print("timing component processing ON: %.3f s" % on_result["timings"]["component_processing_seconds"])
    off_result = detect_compact_features(off_image, off_wcs, args.psf_fwhm_off, **common)
    print("timing Gaussian/DoG OFF: %.3f s" % off_result["timings"]["gaussian_dog_seconds"])
    print("timing component processing OFF: %.3f s" % off_result["timings"]["component_processing_seconds"])
    wcs_start = perf_counter()
    same_grid = _same_grid(on_image.shape, on_wcs, off_image.shape, off_wcs)
    if same_grid:
        output_wcs, output_shape = on_wcs, on_image.shape
        on_output_mask, off_output_mask = on_result["mask"], off_result["mask"]
    else:
        output_wcs, output_shape = _common_tan_wcs(on_image.shape, on_wcs, off_image.shape, off_wcs,
                                                    min(on_scale, off_scale))
        on_output_mask = _resample_mask(on_result["mask"], on_wcs, output_wcs, output_shape)
        off_output_mask = _resample_mask(off_result["mask"], off_wcs, output_wcs, output_shape)
    wcs_seconds = perf_counter() - wcs_start
    if same_grid:
        print("timing WCS union/resampling: %.3f s (same WCS/grid; no resampling performed)" % wcs_seconds)
    else:
        print("timing WCS union/resampling: %.3f s (common grid resampling performed)" % wcs_seconds)
    union_mask = on_output_mask.astype(np.uint8) * MASK_BIT_ON | off_output_mask.astype(np.uint8) * MASK_BIT_OFF
    source_rows = []
    for band, result in (("ON", on_result), ("OFF", off_result)):
        source_rows.extend(dict(band=band, **_strip_private(feature)) for feature in result["features"])
    write_start = perf_counter()
    _write_mask(output_mask, union_mask, output_wcs, {
        "BASIS": "external imaging morphology only", "ONFILE": str(input_on), "OFFFILE": str(input_off),
        "PSFON": args.psf_fwhm_on, "PSFOFF": args.psf_fwhm_off, "FIBRAD": args.fiber_radius_arcsec,
        "ASTROM": args.astrometry_margin_arcsec, "PEAKSIG": args.peak_sigma, "GROWSIG": args.grow_sigma,
        "COMPMAX": args.compactness_max, "SAMGRID": same_grid})
    _write_sources(output_sources, source_rows)
    overlap_radius = args.fiber_radius_arcsec + args.astrometry_margin_arcsec
    on_area = int(np.sum(on_output_mask)); off_area = int(np.sum(off_output_mask)); union_area = int(np.sum(union_mask != 0))
    pixel_area_deg2 = (_pixel_scale_arcsec(output_wcs) / 3600.0) ** 2
    summary = {
        "input_files": {"ON": str(input_on), "OFF": str(input_off)},
        "wcs": {"ON": {"shape": list(on_image.shape), "pixel_scale_arcsec": on_scale, "celestial": on_wcs.to_header_string()},
                "OFF": {"shape": list(off_image.shape), "pixel_scale_arcsec": off_scale, "celestial": off_wcs.to_header_string()},
                "output": {"shape": list(output_shape), "pixel_scale_arcsec": _pixel_scale_arcsec(output_wcs),
                           "reused_identical_grid": same_grid}},
        "psf_fwhm_arcsec": {"ON": args.psf_fwhm_on, "OFF": args.psf_fwhm_off},
        "fiber_radius_arcsec": args.fiber_radius_arcsec, "astrometry_margin_arcsec": args.astrometry_margin_arcsec,
        "peak_sigma": args.peak_sigma, "grow_sigma": args.grow_sigma, "compactness_max": args.compactness_max,
        "broad_psf_factor": args.broad_psf_factor, "minimum_dilation_psf": args.minimum_dilation_psf,
        "number_ON_detections": len(on_result["features"]), "number_OFF_detections": len(off_result["features"]),
        "overlapping_detection_pairs": _count_overlapping_detections(on_result["features"], off_result["features"], overlap_radius),
        "measured_FWHM_arcsec": {"ON": _distribution_summary(on_result["features"], "measured_FWHM_arcsec"),
                                  "OFF": _distribution_summary(off_result["features"], "measured_FWHM_arcsec")},
        "measured_FWHM_over_PSF": {"ON": _distribution_summary(on_result["features"], "FWHM_over_PSF"),
                                    "OFF": _distribution_summary(off_result["features"], "FWHM_over_PSF")},
        "masked_sky_area": {"ON_deg2": on_area * pixel_area_deg2, "OFF_deg2": off_area * pixel_area_deg2,
                             "union_deg2": union_area * pixel_area_deg2},
        "fraction_common_mask_grid_masked": float(union_area / union_mask.size),
        "output_mask": str(output_mask), "output_sources": str(output_sources),
        "output_diagnostic": str(output_diagnostic), "output_summary": str(output_summary),
        "mask_basis": "external imaging morphology only; no VIRUS residual or Bayesian quantity used",
        "mask_bits": {"0": "unmasked", "1": "ON", "2": "OFF", "3": "both"},
        "future_ingestion": "external_valid &= ~mask_radec(mask_data, mask_wcs, ra, dec)[0]; outside footprint is returned as unmasked with inside_footprint=False",
        "synthetic_validation": run_synthetic_validation(),
    }
    output_summary.write_text(json.dumps(summary, indent=2, default=str))
    print("timing writing outputs: %.3f s" % (perf_counter() - write_start))
    diagnostic_start = perf_counter()
    _plot_diagnostic(output_diagnostic, on_image, off_image, on_result, off_result, union_mask)
    print("timing diagnostic plot: %.3f s" % (perf_counter() - diagnostic_start))
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
