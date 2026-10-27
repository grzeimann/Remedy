#!/usr/bin/env python3
"""Plot calibrated spectra for the manually selected M101 amp candidates.

This is intentionally a small, report-only diagnostic.  It uses the current
M101 calibration reader from :mod:`make_m101_cube_current_model` through the
normal per-exposure sky subtraction, but it does not read or apply either
reviewed rejection file and it does not reconstruct a cube.

Run from the M101 reduction directory, for example::

    cd /work/03730/gregz/maverick/m101_recon
    python /work/03730/gregz/maverick/Remedy/plot_m101_candidate_amp_spectra.py

The expected production products and candidate H5 files are resolved relative
to the current working directory. Product paths can be overridden for a
different TACC layout with the command-line options below.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    from astropy.stats import biweight_location
except ImportError:  # pragma: no cover - current Remedy environments have astropy
    biweight_location = None

import make_m101_cube_current_model as current_model


WAVE = np.asarray(current_model.DEF_WAVE, dtype=float)
EXPOSURES = (1, 2, 3)
STAT_WAVE_MIN = 3540.0
STAT_WAVE_MAX = 5480.0
PLOT_WAVE_MIN = 3470.0
PLOT_WAVE_MAX = 5540.0
MIN_BIWEIGHT_FIBERS = 3

FIXED_H5_NAMES = (
    "20200430_0000020.h5",
    "20200430_0000021.h5",
    "20200430_0000022.h5",
    "20200517_0000014.h5",
    "20200521_0000021.h5",
    "20200523_0000024.h5",
    "20200625_0000017.h5",
    "20200710_0000013.h5",
    "20200710_0000014.h5",
)

# These seven tables are the fixed scientific input to the original
# diagnostic. The two dynamic locations are added only after their geometry
# query and are never inferred from a rejection file.
FIXED_LOCATIONS = (
    {
        "x": 759,
        "y": 1007,
        "candidates": (
            {"H5": "20200430_0000020.h5", "SPECID": 38,
             "IFUSLOT": 96, "IFUID": 14, "AMP": "LU"},
        ),
    },
    {
        "x": 780,
        "y": 960,
        "candidates": (
            {"H5": "20200430_0000022.h5", "SPECID": 38,
             "IFUSLOT": 96, "IFUID": 14, "AMP": "LU"},
            {"H5": "20200625_0000017.h5", "SPECID": 404,
             "IFUSLOT": 73, "IFUID": 71, "AMP": "LL"},
            {"H5": "20200625_0000017.h5", "SPECID": 404,
             "IFUSLOT": 73, "IFUID": 71, "AMP": "LU"},
        ),
    },
    {
        "x": 745,
        "y": 915,
        "candidates": (
            {"H5": "20200710_0000013.h5", "SPECID": 330,
             "IFUSLOT": 92, "IFUID": 75, "AMP": "RU"},
            {"H5": "20200517_0000014.h5", "SPECID": 203,
             "IFUSLOT": 86, "IFUID": 31, "AMP": "RU"},
        ),
    },
    {
        "x": 14,
        "y": 280,
        "candidates": (
            {"H5": "20200430_0000021.h5", "SPECID": 38,
             "IFUSLOT": 96, "IFUID": 14, "AMP": "LU"},
        ),
    },
    {
        "x": 1121,
        "y": 465,
        "candidates": (
            {"H5": "20200710_0000014.h5", "SPECID": 412,
             "IFUSLOT": 13, "IFUID": 43, "AMP": "LL"},
            {"H5": "20200521_0000021.h5", "SPECID": 202,
             "IFUSLOT": 35, "IFUID": 74, "AMP": "LU"},
        ),
    },
    {
        "x": 1084,
        "y": 449,
        "candidates": (
            {"H5": "20200430_0000022.h5", "SPECID": 302,
             "IFUSLOT": 36, "IFUID": 41, "AMP": "LL"},
            {"H5": "20200625_0000017.h5", "SPECID": 412,
             "IFUSLOT": 13, "IFUID": 43, "AMP": "LL"},
            {"H5": "20200430_0000022.h5", "SPECID": 302,
             "IFUSLOT": 36, "IFUID": 41, "AMP": "LU"},
        ),
    },
    {
        "x": 1071,
        "y": 150,
        "candidates": (
            {"H5": "20200523_0000024.h5", "SPECID": 51,
             "IFUSLOT": 105, "IFUID": 51, "AMP": "LL"},
            {"H5": "20200710_0000013.h5", "SPECID": 412,
             "IFUSLOT": 13, "IFUID": 43, "AMP": "LL"},
        ),
    },
)

DYNAMIC_LOCATION_COORDINATES = ((247, 391), (319, 711))
DYNAMIC_WAVELENGTH = 4882.0
DYNAMIC_RADIUS_ARCSEC = 2.0
DYNAMIC_EXCLUDED_H5 = "20200523_0000023.h5"
VALID_AMPS = {"LL", "LU", "RL", "RU"}

METRIC_FIELDS = (
    "DS9_X", "DS9_Y", "H5", "exposure", "SPECID", "IFUSLOT", "IFUID",
    "AMP", "n_fibers", "median_3540_5480", "sigma_MAD_3540_5480",
    "std_3540_5480", "RMS_3540_5480", "finite_wavelength_samples",
    "exposure_scale_applied",
)


def validate_candidate_table(locations=None, available_h5_names=None):
    """Validate the fixed seven tables and, when supplied, all nine locations."""
    if len(FIXED_LOCATIONS) != 7:
        raise AssertionError("fixed candidate table must contain exactly seven locations")
    fixed_coordinates = {(int(row["x"]), int(row["y"]))
                         for row in FIXED_LOCATIONS}
    if len(fixed_coordinates) != 7:
        raise AssertionError("fixed candidate locations must be unique")
    if fixed_coordinates & set(DYNAMIC_LOCATION_COORDINATES):
        raise AssertionError("dynamic location overlaps a fixed location")

    locations = tuple(FIXED_LOCATIONS if locations is None else locations)
    if len(locations) not in (7, 9):
        raise AssertionError("candidate table must contain seven or nine locations")
    coordinates = {(int(row["x"]), int(row["y"])) for row in locations}
    if len(coordinates) != len(locations):
        raise AssertionError("candidate locations must be unique")
    if len(locations) == 9 and set(DYNAMIC_LOCATION_COORDINATES) - coordinates:
        raise AssertionError("the nine-location table lacks a dynamic location")

    fixed_used = {
        candidate["H5"]
        for location in FIXED_LOCATIONS
        for candidate in location["candidates"]
    }
    if not fixed_used <= set(FIXED_H5_NAMES):
        raise AssertionError("fixed candidate table contains an unexpected H5")

    used = set()
    for location in locations:
        if not location["candidates"]:
            raise AssertionError("each location needs at least one candidate")
        for candidate in location["candidates"]:
            used.add(candidate["H5"])
            if str(candidate["AMP"]).upper() not in VALID_AMPS:
                raise AssertionError("unexpected amplifier orientation")
    if available_h5_names is not None and not used <= set(available_h5_names):
        missing = sorted(used - set(available_h5_names))
        raise AssertionError("candidate H5 files do not exist: %s" % missing)


def _location_with_candidates(x, y, candidates, geometry_candidates=None):
    location = {"x": int(x), "y": int(y), "candidates": tuple(candidates)}
    if geometry_candidates is not None:
        location["geometry_candidates"] = tuple(geometry_candidates)
    return location


def _physical_key(candidate):
    return (int(candidate["SPECID"]), int(candidate["IFUSLOT"]),
            int(candidate["IFUID"]), str(candidate["AMP"]).upper())


def _candidate_label(candidate):
    return "%s | %d/%d/%d/%s" % (
        candidate["H5"], candidate["SPECID"], candidate["IFUSLOT"],
        candidate["IFUID"], str(candidate["AMP"]).upper())


def _resolve_path(data_dir, value):
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = data_dir / path
    return path.resolve()


def _resolve_input_path(data_dir, value):
    """Resolve a product path, accepting a flat m101_recon layout too."""
    path = _resolve_path(data_dir, value)
    if path.exists() or Path(value).expanduser().is_absolute():
        return path
    flat_path = data_dir / Path(value).name
    return flat_path.resolve() if flat_path.exists() else path


def _first_existing(data_dir, names):
    for name in names:
        path = _resolve_path(data_dir, name)
        if path.exists():
            return path
    return None


def _collapse_geometry_matches(matches):
    """Collapse nearby fibers to physical amplifier candidates."""
    grouped = defaultdict(lambda: {
        "exposures": set(), "n_fibers": 0, "min_separation_arcsec": np.inf,
    })
    for row in matches:
        key = (
            Path(row["H5"]).name, int(row["SPECID"]), int(row["IFUSLOT"]),
            int(row["IFUID"]), str(row["AMP"]).upper(),
        )
        state = grouped[key]
        state["exposures"].add(int(row["exposure"]))
        state["n_fibers"] += 1
        state["min_separation_arcsec"] = min(
            state["min_separation_arcsec"], float(row["separation_arcsec"]))

    result = []
    for key, state in grouped.items():
        candidate = {
            "H5": key[0], "SPECID": key[1], "IFUSLOT": key[2],
            "IFUID": key[3], "AMP": key[4],
        }
        result.append({
            "candidate": candidate,
            "exposures": tuple(sorted(state["exposures"])),
            "n_fibers": int(state["n_fibers"]),
            "min_separation_arcsec": float(state["min_separation_arcsec"]),
        })
    return sorted(result, key=lambda row: (
        row["min_separation_arcsec"], row["candidate"]["H5"],
        row["candidate"]["SPECID"], row["candidate"]["IFUSLOT"],
        row["candidate"]["IFUID"], row["candidate"]["AMP"],
    ))


def _discover_dynamic_locations(data_dir, cube_path):
    """Find candidates for only the two new positions using light geometry."""
    import tables
    from astropy.io import fits
    import query_m101_fibers_near_pixel as geometry_query

    astrometry_cls, extract_cls, build_groups, _, _ = (
        geometry_query._load_project_helpers())
    with fits.open(cube_path, memmap=True) as hdul:
        header = hdul[0].header
        targets = {
            coordinates: geometry_query.cube_pixel_to_sky(header, *coordinates)
            for coordinates in DYNAMIC_LOCATION_COORDINATES
        }

    h5_paths = sorted(data_dir.glob("20*.h5"))
    h5_paths = [path for path in h5_paths if path.name != DYNAMIC_EXCLUDED_H5]
    if not h5_paths:
        raise ValueError("no normal 20*.h5 files found in %s" % data_dir)

    locations = []
    for coordinates in DYNAMIC_LOCATION_COORDINATES:
        target_ra, target_dec = targets[coordinates]
        matches = []
        for h5_path in h5_paths:
            with tables.open_file(str(h5_path), mode="r") as h5:
                if not {"Info", "Survey"}.issubset(h5.root._v_children):
                    raise ValueError("%s needs Info and Survey tables" % h5_path.name)
                info = h5.root.Info
                groups, _labels = build_groups(info)
                native_ra = np.asarray(info.cols.ra[:], dtype=float)
                native_dec = np.asarray(info.cols.dec[:], dtype=float)
                if native_ra.shape != native_dec.shape or native_ra.size != info.nrows:
                    raise ValueError("%s Info RA/Dec row alignment is invalid" %
                                     h5_path.name)
                survey_by_exposure = geometry_query._survey_by_exposure(
                    h5.root.Survey)
                adr_by_exposure = {
                    exposure: geometry_query._adr_offset_at_wavelength(
                        survey_row, DYNAMIC_WAVELENGTH, astrometry_cls, extract_cls)
                    for exposure, survey_row in survey_by_exposure.items()
                }

                for group in groups:
                    exposure = int(group["exposure"])
                    if exposure not in survey_by_exposure:
                        raise ValueError("%s Survey has no exposure %d" %
                                         (h5_path.name, exposure))
                    indices = np.asarray(group["indices"], dtype=int)
                    dra, ddec = adr_by_exposure[exposure]
                    ra_at_wave = native_ra[indices] - dra
                    dec_at_wave = native_dec[indices] - ddec
                    separation = geometry_query.angular_separation_arcsec(
                        ra_at_wave, dec_at_wave, target_ra, target_dec)
                    selected = np.isfinite(separation) & (
                        separation <= DYNAMIC_RADIUS_ARCSEC)
                    if not np.any(selected):
                        continue
                    amp = str(group["amp"]).upper()
                    for local in np.flatnonzero(selected):
                        matches.append({
                            "H5": h5_path.name, "exposure": exposure,
                            "SPECID": int(group["specid"]),
                            "IFUSLOT": int(group["ifuslot"]),
                            "IFUID": int(group["ifuid"]), "AMP": amp,
                            "separation_arcsec": float(separation[local]),
                        })

        geometry_candidates = _collapse_geometry_matches(matches)
        if not geometry_candidates:
            raise ValueError(
                "no geometry candidates within %.1f arcsec for DS9 (%d,%d)" %
                (DYNAMIC_RADIUS_ARCSEC, coordinates[0], coordinates[1]))
        locations.append(_location_with_candidates(
            coordinates[0], coordinates[1],
            [row["candidate"] for row in geometry_candidates],
            geometry_candidates=geometry_candidates))
    return tuple(locations)


def _load_exposure_scales(path, required_h5_names):
    """Read existing final relative scales without refitting them.

    The production current-model table stores the two-pass quantity
    ``cumulative_relative_scale``.  A diagnostic table that only stores
    ``exposure_correction`` is also accepted when explicitly supplied.
    """
    if path is None:
        return {}, False, "not found; scalar exposure scaling omitted"
    path = Path(path).resolve()
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        fields = set(reader.fieldnames or ())
        scale_field = None
        for name in ("cumulative_relative_scale", "exposure_correction"):
            if name in fields:
                scale_field = name
                break
        if scale_field is None:
            raise ValueError("exposure scale table lacks a usable scale column: %s" % path)
        result = {}
        for row in reader:
            key = (Path(row["H5"]).name, int(row["exposure"]))
            if key in result:
                raise ValueError("duplicate exposure scale row: %s" % (key,))
            value = float(row[scale_field])
            if not np.isfinite(value) or value == 0.0:
                raise ValueError("invalid exposure scale for %s: %r" % (key, value))
            # cumulative_relative_scale is the divisor applied to the
            # already sky-subtracted calibrated spectra.  A legacy diagnostic
            # exposure_correction is already the multiplicative correction.
            result[key] = (1.0 / value if scale_field == "cumulative_relative_scale"
                           else value)
    expected = {(name, exposure) for name in required_h5_names
                for exposure in EXPOSURES}
    if not expected.issubset(result):
        missing = sorted(expected - set(result))
        raise ValueError("exposure scale table is missing required rows: %s" % missing[:8])
    return result, True, "%s (%s)" % (path, scale_field)


def _robust_location_by_wavelength(values):
    """Robust location over fibers, with median fallback at weak support."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or values.shape[1] != WAVE.size:
        raise ValueError("fiber spectra must have shape (N, %d)" % WAVE.size)
    finite_count = np.sum(np.isfinite(values), axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        if biweight_location is None:
            result = np.full(WAVE.shape, np.nan, dtype=float)
        else:
            try:
                result = np.asarray(
                    biweight_location(values, axis=0, ignore_nan=True), dtype=float)
            except (TypeError, ValueError, FloatingPointError):
                result = np.full(WAVE.shape, np.nan, dtype=float)
    if result.shape != WAVE.shape:
        result = np.full(WAVE.shape, np.nan, dtype=float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        medians = np.nanmedian(values, axis=0)
    fallback = (finite_count < MIN_BIWEIGHT_FIBERS) | ~np.isfinite(result)
    result[fallback] = medians[fallback]
    return result


def _spectral_metrics(spectrum):
    """Calculate the requested wavelength-domain metrics."""
    spectrum = np.asarray(spectrum, dtype=float)
    statistic_mask = ((WAVE >= STAT_WAVE_MIN) & (WAVE <= STAT_WAVE_MAX) &
                      np.isfinite(spectrum))
    values = spectrum[statistic_mask]
    if values.size == 0:
        return {
            "median_3540_5480": np.nan,
            "sigma_MAD_3540_5480": np.nan,
            "std_3540_5480": np.nan,
            "RMS_3540_5480": np.nan,
            "finite_wavelength_samples": 0,
        }
    center = float(np.median(values))
    residual = values - center
    residual_center = float(np.median(residual))
    mad = float(np.median(np.abs(residual - residual_center)))
    return {
        "median_3540_5480": center,
        "sigma_MAD_3540_5480": 1.4826 * mad,
        "std_3540_5480": float(np.std(residual)),
        "RMS_3540_5480": float(np.sqrt(np.mean(residual ** 2))),
        "finite_wavelength_samples": int(values.size),
    }


def _candidate_spectra_for_h5(h5_name, calibrated_spectra, groups, candidates,
                              exposure_scales):
    """Extract and robustly collapse only candidate amplifier groups."""
    groups_by_key = {}
    for group in groups:
        key = (int(group["exposure"]), int(group["specid"]),
               int(group["ifuslot"]), int(group["ifuid"]),
               str(group["amp"]).upper())
        groups_by_key[key] = group

    result = {}
    for candidate in candidates:
        physical = _physical_key(candidate)
        for exposure in EXPOSURES:
            key = (exposure,) + physical
            if key not in groups_by_key:
                raise ValueError("candidate group absent from %s: %s" % (h5_name, key))
            group = groups_by_key[key]
            indices = np.asarray(group["indices"], dtype=int)
            fibers = np.asarray(calibrated_spectra[indices], dtype=float)
            spectrum = _robust_location_by_wavelength(fibers)
            scale_key = (h5_name, exposure)
            scale_applied = scale_key in exposure_scales
            if scale_applied:
                spectrum = spectrum * float(exposure_scales[scale_key])
            metrics = _spectral_metrics(spectrum)
            finite_fiber_rows = np.any(np.isfinite(fibers), axis=1)
            result[(physical, exposure)] = {
                "candidate": dict(candidate),
                "exposure": int(exposure),
                "spectrum": spectrum,
                "n_fibers": int(np.count_nonzero(finite_fiber_rows)),
                "exposure_scale_applied": bool(scale_applied),
                **metrics,
            }
    return result


def _robust_limits(series, lower=0.5, upper=99.5):
    values = np.concatenate([
        np.asarray(item, dtype=float)[
            (WAVE >= PLOT_WAVE_MIN) & (WAVE <= PLOT_WAVE_MAX) &
            np.isfinite(item)]
        for item in series
    ]) if series else np.asarray([], dtype=float)
    if values.size == 0:
        return -1.0, 1.0
    lo, hi = np.percentile(values, (lower, upper))
    if not np.isfinite(lo) or not np.isfinite(hi):
        return -1.0, 1.0
    span = hi - lo
    if span <= 0.0:
        span = max(abs(float(lo)) * 0.05, 1.0)
    padding = 0.08 * span
    return float(lo - padding), float(hi + padding)


def _plot_location(location, rows, output_path, scale_description):
    colors = plt.get_cmap("tab10")(np.linspace(0.05, 0.85, len(location["candidates"])))
    line_styles = {1: "-", 2: "--", 3: ":"}
    fig, axes = plt.subplots(2, 1, figsize=(15.5, 10.0), sharex=True,
                             gridspec_kw={"height_ratios": (1.25, 1.0)})
    top, bottom = axes
    top_series = []
    bottom_series = []
    physical_to_color = {
        _physical_key(candidate): colors[index]
        for index, candidate in enumerate(location["candidates"])
    }

    for candidate in location["candidates"]:
        physical = _physical_key(candidate)
        color = physical_to_color[physical]
        for exposure in EXPOSURES:
            row = rows[(physical, exposure)]
            spectrum = np.asarray(row["spectrum"], dtype=float)
            center = float(row["median_3540_5480"])
            shape = spectrum - center if np.isfinite(center) else spectrum
            label = (
                "%s | e%d | N=%d | sigmaMAD=%.4g | std=%.4g" %
                (_candidate_label(candidate), exposure, row["n_fibers"],
                 row["sigma_MAD_3540_5480"], row["std_3540_5480"])
            )
            top.plot(WAVE, spectrum, color=color, linestyle=line_styles[exposure],
                     linewidth=1.05, alpha=0.92, label=label)
            bottom.plot(WAVE, shape, color=color, linestyle=line_styles[exposure],
                        linewidth=1.05, alpha=0.92)
            top_series.append(spectrum)
            bottom_series.append(shape)

    for axis in axes:
        axis.axvline(STAT_WAVE_MIN, color="0.55", linestyle="--", linewidth=.7)
        axis.axvline(STAT_WAVE_MAX, color="0.55", linestyle="--", linewidth=.7)
        axis.set_xlim(PLOT_WAVE_MIN, PLOT_WAVE_MAX)
        axis.grid(True, alpha=.20)
        axis.tick_params(labelsize=9)
    top.set_ylim(*_robust_limits(top_series))
    bottom.set_ylim(*_robust_limits(bottom_series))
    top.set_ylabel("calibrated sky-subtracted spectrum", fontsize=10)
    bottom.set_ylabel("spectrum - own median", fontsize=10)
    bottom.set_xlabel("wavelength [Angstrom]", fontsize=10)
    top.set_title("robust location across all fibers in each amp; "
                  "scatter measured over 3540-5480 A", fontsize=10)
    top.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0.0,
               fontsize=8.1, frameon=True)
    fig.suptitle("M101 candidate amplifier spectra — DS9 (%d,%d)" %
                 (location["x"], location["y"]), fontsize=15, y=.985)
    fig.text(.01, .012, "Exposure scaling applied: %s. "
             "Vertical lines mark the metric interval; robust percentile y-limits "
             "can clip isolated extremes." % scale_description,
             fontsize=8.5, ha="left")
    fig.subplots_adjust(left=.07, right=.68, top=.91, bottom=.08, hspace=.08)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _write_metrics(path, rows):
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=METRIC_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in METRIC_FIELDS})


def _write_summary(path, location_rows, scale_description):
    lines = [
        "M101 candidate amplifier spectral diagnostic",
        "",
        "This report lists candidates in increasing sigma_MAD only; it does not",
        "classify any candidate as good or bad and does not apply a rejection.",
        "Calibration: current multiplicative model, alpha correction, persistent",
        "mean amplifier residual correction, and normal per-exposure sky subtraction.",
        "Reviewed QA and spectral rejection files were not read.",
        "Exposure scaling: %s" % scale_description,
        "Metric interval: 3540-5480 Angstrom only.",
        "",
    ]
    for location, rows in location_rows:
        lines.append("DS9 (%d,%d)" % (location["x"], location["y"]))
        geometry_candidates = location.get("geometry_candidates")
        if geometry_candidates is not None:
            lines.extend([
                "  Geometry candidates discovered within 2 arcsec at 4882 A",
                "  H5 | SPECID | IFUSLOT | IFUID | AMP | exposures represented | "
                "nearby fibers | minimum separation_arcsec",
            ])
            for geometry_row in geometry_candidates:
                candidate = geometry_row["candidate"]
                lines.append(
                    "  %s | %d | %d | %d | %s | %s | %d | %.8g" % (
                        candidate["H5"], candidate["SPECID"],
                        candidate["IFUSLOT"], candidate["IFUID"],
                        candidate["AMP"],
                        ",".join(str(exposure)
                                 for exposure in geometry_row["exposures"]),
                        geometry_row["n_fibers"],
                        geometry_row["min_separation_arcsec"],
                    )
                )
            lines.append("")
        ordered = sorted(
            rows,
            key=lambda row: (
                not np.isfinite(row["sigma_MAD_3540_5480"]),
                row["sigma_MAD_3540_5480"]
                if np.isfinite(row["sigma_MAD_3540_5480"]) else np.inf,
                row["candidate"]["H5"], row["exposure"],
            ),
        )
        for row in ordered:
            lines.append(
                "  %s | e%d | N=%d | median=%.8g | sigmaMAD=%.8g | "
                "std=%.8g | RMS=%.8g | finite=%d | scale=%s" % (
                    _candidate_label(row["candidate"]), row["exposure"],
                    row["n_fibers"], row["median_3540_5480"],
                    row["sigma_MAD_3540_5480"], row["std_3540_5480"],
                    row["RMS_3540_5480"], row["finite_wavelength_samples"],
                    row["exposure_scale_applied"],
                )
            )
        lines.append("")
    Path(path).write_text("\n".join(lines))


def _synthetic_tests():
    """Small local-only tests for table shape, geometry collapse, and metrics."""
    validate_candidate_table()
    if len(FIXED_LOCATIONS) != 7:
        raise AssertionError("fixed location count is not seven")
    synthetic_dynamic = (
        _location_with_candidates(
            247, 391,
            ({"H5": "dynamic_a.h5", "SPECID": 1, "IFUSLOT": 2,
              "IFUID": 3, "AMP": "LL"},)),
        _location_with_candidates(
            319, 711,
            ({"H5": "dynamic_b.h5", "SPECID": 4, "IFUSLOT": 5,
              "IFUID": 6, "AMP": "RU"},)),
    )
    requested = FIXED_LOCATIONS + synthetic_dynamic
    validate_candidate_table(
        requested, available_h5_names=set(FIXED_H5_NAMES) |
        {"dynamic_a.h5", "dynamic_b.h5"})
    if len(requested) != 9:
        raise AssertionError("total requested location count is not nine")

    collapsed = _collapse_geometry_matches([
        {"H5": "x.h5", "exposure": 1, "SPECID": 1, "IFUSLOT": 2,
         "IFUID": 3, "AMP": "LL", "separation_arcsec": 1.25},
        {"H5": "x.h5", "exposure": 2, "SPECID": 1, "IFUSLOT": 2,
         "IFUID": 3, "AMP": "LL", "separation_arcsec": .75},
        {"H5": "x.h5", "exposure": 1, "SPECID": 1, "IFUSLOT": 2,
         "IFUID": 3, "AMP": "LU", "separation_arcsec": 1.50},
    ])
    if len(collapsed) != 2 or next(
            row for row in collapsed if row["candidate"]["AMP"] == "LU")[
                "n_fibers"] != 1:
        raise AssertionError("2-arcsec candidate collapse is incorrect")
    ll = next(row for row in collapsed if row["candidate"]["AMP"] == "LL")
    if ll["exposures"] != (1, 2) or not np.isclose(
            ll["min_separation_arcsec"], .75):
        raise AssertionError("candidate exposure/separation collapse is incorrect")

    synthetic_wave = np.array([3470.0, 3540.0, 4000.0, 5480.0, 5540.0])
    if not np.array_equal(
            (synthetic_wave >= STAT_WAVE_MIN) & (synthetic_wave <= STAT_WAVE_MAX),
            np.array([False, True, True, True, False])):
        raise AssertionError("3540-5480 wavelength mask is incorrect")

    original_wave = WAVE.copy()
    try:
        # Test the robust collapse against one strong fiber outlier.
        values = np.vstack([
            np.full(WAVE.size, 2.0),
            np.full(WAVE.size, 2.1),
            np.full(WAVE.size, 1.9),
            np.full(WAVE.size, 2.0),
        ])
        values[0, 10] = 1.0e6
        collapsed = _robust_location_by_wavelength(values)
        if not np.isfinite(collapsed).all() or collapsed[10] > 3.0:
            raise AssertionError("synthetic robust amplifier spectrum failed")

        # Test exactly the requested metric definitions and interval.
        metric_wave = np.array([3470.0, 3540.0, 4000.0, 5480.0, 5540.0])
        metric_spectrum = np.array([10.0, 1.0, 3.0, 5.0, 100.0])
        saved_wave = globals()["WAVE"]
        globals()["WAVE"] = metric_wave
        metrics = _spectral_metrics(metric_spectrum)
        globals()["WAVE"] = saved_wave
        if metrics["finite_wavelength_samples"] != 3:
            raise AssertionError("metric wavelength mask selected the wrong samples")
        if not np.isclose(metrics["median_3540_5480"], 3.0):
            raise AssertionError("metric center is incorrect")
        if not np.isclose(metrics["sigma_MAD_3540_5480"], 1.4826 * 2.0):
            raise AssertionError("sigma_MAD is incorrect")
        if not np.isclose(metrics["std_3540_5480"], np.sqrt(8.0 / 3.0)):
            raise AssertionError("standard deviation is incorrect")
        if not np.isclose(metrics["RMS_3540_5480"], np.sqrt(8.0 / 3.0)):
            raise AssertionError("RMS is incorrect")
    finally:
        globals()["WAVE"] = original_wave
    print("synthetic tests: PASS")


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=".",
                        help="TACC m101_recon directory (default: current directory)")
    parser.add_argument("--output-dir", default="m101_candidate_amp_spectra")
    parser.add_argument("--cube", default="m101_virus_v2_rebuild_cube.fits",
                        help="production cube used only for the dynamic-position WCS")
    parser.add_argument("--state", default="m101_onoff_hierarchy/m101_onoff_state.json")
    parser.add_argument("--band-cache", default="m101_band_initializer/m101_band_cache.npz")
    parser.add_argument("--alpha-fits",
                        default="m101_alpha_template_bands/alpha_template_fits.csv")
    parser.add_argument("--alpha-summary",
                        default="m101_alpha_same_amp_loo/alpha_physical_amp_summary.csv")
    parser.add_argument("--amp-residual-models",
                        default="m101-amp-residual-pca-final/amp_residual_pca_models.npz")
    parser.add_argument("--fq-template", default=None,
                        help="fixed f(q) CSV; defaults to common_fq_template.csv")
    parser.add_argument("--external-blank-fibers", default=None,
                        help="external blank-fiber CSV.GZ; defaults to the established path")
    parser.add_argument("--exposure-scale-csv", default=None,
                        help="existing exposure-scale CSV; omitted when not found")
    parser.add_argument("--self-test", action="store_true",
                        help="run synthetic tests without reading H5 files")
    return parser.parse_args()


def main():
    args = _parse_args()
    if args.self_test:
        _synthetic_tests()
        return

    data_dir = Path(args.data_dir).expanduser().resolve()
    output_dir = _resolve_path(data_dir, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cube_path = _resolve_input_path(data_dir, args.cube)
    if not cube_path.is_file() and args.cube == "m101_virus_v2_rebuild_cube.fits":
        cube_path = (_first_existing(
            data_dir, ("m101_virus_v2_rebuild_cube.fits",
                       "m101_plots/m101_virus_v2_rebuild_cube.fits"))
                     or cube_path)
    if not cube_path.is_file():
        raise FileNotFoundError("production cube for dynamic WCS is missing: %s" %
                                cube_path)

    # The original seven tables are retained verbatim. Only the two new
    # locations perform the lightweight 4882-A geometry search.
    dynamic_locations = _discover_dynamic_locations(data_dir, cube_path)
    locations = tuple(FIXED_LOCATIONS) + tuple(dynamic_locations)
    candidate_h5_names = tuple(sorted({candidate["H5"]
                                       for location in locations
                                       for candidate in location["candidates"]}))
    h5_paths = [data_dir / name for name in candidate_h5_names]
    missing_h5 = [str(path) for path in h5_paths if not path.is_file()]
    if missing_h5:
        raise FileNotFoundError("candidate M101 H5 files are missing: %s" % missing_h5)
    validate_candidate_table(locations,
                             available_h5_names={path.name for path in h5_paths})

    fq_path = (_resolve_input_path(data_dir, args.fq_template) if args.fq_template else
               _first_existing(data_dir, ("common_fq_template.csv",
                                          "m101_plots/common_fq_template.csv",
                                          "m101_band_initializer/common_fq_template.csv")))
    if fq_path is None:
        raise FileNotFoundError(
            "fixed f(q) template not found; pass --fq-template common_fq_template.csv")
    blank_path = (_resolve_input_path(data_dir, args.external_blank_fibers)
                  if args.external_blank_fibers else
                  _first_existing(data_dir, (
                      "external_blank_fibers.csv.gz",
                      "external_blank_sky_diagnostic/external_blank_fibers.csv.gz",
                      "m101_plots/external_blank_sky_diagnostic/external_blank_fibers.csv.gz")))
    if blank_path is None:
        raise FileNotFoundError(
            "external blank-fiber table not found; pass --external-blank-fibers")
    scale_path = (_resolve_input_path(data_dir, args.exposure_scale_csv)
                  if args.exposure_scale_csv else
                  _first_existing(data_dir, ("m101_cube_current_model_exposure_scale.csv",)))
    exposure_scales, scale_applied, scale_description = _load_exposure_scales(
        scale_path, candidate_h5_names) if scale_path is not None else (
            {}, False, "not found; scalar exposure scaling omitted")

    state_path = _resolve_input_path(data_dir, args.state)
    band_cache_path = _resolve_input_path(data_dir, args.band_cache)
    alpha_fits_path = _resolve_input_path(data_dir, args.alpha_fits)
    alpha_summary_path = _resolve_input_path(data_dir, args.alpha_summary)
    amp_means_path = _resolve_input_path(data_dir, args.amp_residual_models)

    state, response, composition, _ = current_model._load_frozen_response(state_path)
    identity_map, _, _, _ = current_model._load_identity_map(band_cache_path, h5_paths)
    alpha_direct, _ = current_model._load_alpha_fits(alpha_fits_path)
    alpha_history, _ = current_model._load_alpha_history(alpha_summary_path)
    amp_means, _, _, _ = current_model._load_amp_means(amp_means_path)
    fq_template, historical_f0, _ = current_model._load_fq(fq_path)
    blank_masks, _ = current_model.frozen_parent.m101_blank_fibers.load(
        blank_path, h5_paths)

    # No QA or spectral-rejection CSV is read here.  The current-model helper
    # applies calibration and normal exposure sky subtraction only.  Gather
    # candidates by H5 first so each required H5 is opened/calibrated once.
    candidates_by_h5 = {}
    for location in locations:
        for candidate in location["candidates"]:
            candidates_by_h5.setdefault(candidate["H5"], {})[
                _physical_key(candidate)] = candidate

    candidate_spectra = {}
    for h5_name in candidate_h5_names:
        candidates = list(candidates_by_h5.get(h5_name, {}).values())
        if not candidates:
            continue
        h5_path = data_dir / h5_name
        calibrated = current_model._calibrate_h5_current(
            h5_path, identity_map, response, alpha_direct, alpha_history,
            amp_means, fq_template, historical_f0, blank_masks[h5_name],
            current_model.M101_SKY_MIN_FIBERS)
        spectra, _, _, _, _, _, _, _, _, groups, _, _, _ = calibrated
        candidate_spectra[h5_name] = _candidate_spectra_for_h5(
            h5_name, spectra, groups, candidates, exposure_scales)

    candidate_rows_by_location = []
    all_metric_rows = []
    for location in locations:
        location_rows = []
        for candidate in location["candidates"]:
            physical = _physical_key(candidate)
            for exposure in EXPOSURES:
                row = candidate_spectra[candidate["H5"]][(physical, exposure)]
                location_rows.append(row)
                all_metric_rows.append({
                    "DS9_X": int(location["x"]),
                    "DS9_Y": int(location["y"]),
                    "H5": row["candidate"]["H5"],
                    "exposure": row["exposure"],
                    "SPECID": row["candidate"]["SPECID"],
                    "IFUSLOT": row["candidate"]["IFUSLOT"],
                    "IFUID": row["candidate"]["IFUID"],
                    "AMP": row["candidate"]["AMP"],
                    "n_fibers": row["n_fibers"],
                    "median_3540_5480": row["median_3540_5480"],
                    "sigma_MAD_3540_5480": row["sigma_MAD_3540_5480"],
                    "std_3540_5480": row["std_3540_5480"],
                    "RMS_3540_5480": row["RMS_3540_5480"],
                    "finite_wavelength_samples": row["finite_wavelength_samples"],
                    "exposure_scale_applied": row["exposure_scale_applied"],
                })
        row_lookup = {(_physical_key(row["candidate"]), row["exposure"]): row
                      for row in location_rows}
        candidate_rows_by_location.append((location, row_lookup, location_rows))

    for location, row_lookup, location_rows in candidate_rows_by_location:
        output_path = output_dir / ("amp_spectra_x%d_y%d.png" %
                                    (location["x"], location["y"]))
        _plot_location(location, row_lookup, output_path, scale_description)

    _write_metrics(output_dir / "m101_candidate_amp_spectral_metrics.csv",
                   all_metric_rows)
    _write_summary(output_dir / "m101_candidate_amp_spectral_summary.txt",
                   [(location, rows) for location, _, rows in candidate_rows_by_location],
                   scale_description)
    print("wrote %d diagnostic figures and two reports to %s" %
          (len(candidate_rows_by_location), output_dir))
    print("exposure scaling: %s" % scale_description)
    print("frozen response composition: %s" % composition.get(
        "composition_matches_stored_cumulative", False))
    print("loaded model state: %s" % state.get("schema_version", "unknown"))


if __name__ == "__main__":
    main()
