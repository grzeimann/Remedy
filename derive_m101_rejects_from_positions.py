#!/usr/bin/env python3
"""Derive reviewed M101 input rejection lists from manually inspected pixels.

This is deliberately a position-driven diagnostic, not a cube patcher.  It
imports the current production builder and reruns its frozen calibration,
final sky, and two-pass exposure-scale path.  At each supplied DS9 position it
then evaluates the same per-shot Gaussian deposition used by the production
cube and attributes the signal to exact H5/exposure/physical-amplifier
groups.

The script writes reviewed reject files only for candidates that have a
coherent flux explanation, persistent spectral behavior, and a clear margin
over the next candidate.  Ambiguous candidates are reported but are never
written to a reject file.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import glob
import importlib.util
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS


PRODUCTION_DIR = Path("/work/03730/gregz/maverick/m101_recon")
REMEDY_DIR = Path("/work/03730/gregz/maverick/Remedy")
OUTPUT_DIR_NAME = "m101_final_position_trace"
GROUP_FIELDS = ("H5", "exposure", "SPECID", "IFUSLOT", "IFUID", "AMP")
SPECTRAL_TARGETS = (
    (200, 335), (250, 362), (276, 318), (230, 292),
    (675, 1067), (704, 1026), (663, 1003),
    (976, 1020), (1003, 976), (953, 949),
    (1192, 43),
)
ALL_WAVELENGTH_TARGETS = (
    (759, 1007), (780, 960), (745, 915),
    (14, 280),
    (1121, 465), (1084, 449),
    (1071, 150),
)
SPECTRAL_INSPECTION_WAVELENGTHS = tuple(range(4868, 4898, 2))
SPECTRAL_REJECT_WAVELENGTHS = tuple(range(4874, 4891, 2))
SPECTRAL_CONTROL_WAVELENGTHS = (4868, 4870, 4872, 4892, 4894, 4896)
ALL_WAVELENGTHS = (3800, 4200, 4600, 4882, 5200, 5450)


def say(message):
    print("[position-trace] " + str(message), flush=True)


def _jsonable(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, set)):
        return [_jsonable(item) for item in value]
    return value


def _finite(values):
    values = np.asarray(values, dtype=float)
    return values[np.isfinite(values)]


def _robust_location(values):
    values = _finite(values)
    return float(np.median(values)) if values.size else np.nan


def _robust_scale(values):
    values = _finite(values)
    if values.size < 2:
        return np.nan
    center = np.median(values)
    mad = 1.4826 * np.median(np.abs(values - center))
    if np.isfinite(mad) and mad > 0:
        return float(mad)
    scatter = np.std(values, ddof=1)
    return float(scatter) if np.isfinite(scatter) and scatter > 0 else 0.0


def _text(value):
    if isinstance(value, (bytes, np.bytes_)):
        return value.decode("utf-8", errors="replace").strip()
    return str(value).strip()


def _amp(value):
    value = _text(value)
    if value.startswith("b'") and value.endswith("'"):
        value = value[2:-1]
    return value.upper()


def _group_key(h5, group):
    return (
        Path(h5).name, int(group["exposure"]), int(group["specid"]),
        int(group["ifuslot"]), int(group["ifuid"]), _amp(group["amp"]),
    )


def _group_label(key):
    return "%s/e%d/%d/%d/%d/%s" % key


def _key_dict(key):
    return {field: key[index] for index, field in enumerate(GROUP_FIELDS)}


def _production_gaussian_weight(production, fiber_x, fiber_y, pixel_x,
                                 pixel_y, sigma):
    """Use the production kernel across builder revisions.

    Current local builders expose ``production_gaussian_weight``.  Older
    TACC copies have the same private axis kernel used by
    ``_gaussian_splat_shot_xy`` but do not expose the scalar wrapper.  The
    final expression is the exact production axis formula, not an alternate
    reconstruction approximation.
    """
    public = getattr(production, "production_gaussian_weight", None)
    if public is not None:
        return float(public(fiber_x, fiber_y, pixel_x, pixel_y, sigma))
    axis = getattr(production, "_production_gaussian_axis_weight", None)
    if axis is not None:
        return float(axis(float(fiber_x), float(pixel_x), float(sigma)) *
                     axis(float(fiber_y), float(pixel_y), float(sigma)))
    return float(np.exp(-0.5 * (float(pixel_x) - float(fiber_x)) ** 2 /
                        float(sigma) ** 2) *
                 np.exp(-0.5 * (float(pixel_y) - float(fiber_y)) ** 2 /
                        float(sigma) ** 2))


def _summary_path(summary, *keys):
    value = summary
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    if isinstance(value, dict):
        value = value.get("path")
    return Path(str(value)).expanduser() if value else None


def _resolve_product(path, source_dir, production_dir):
    if path is None:
        return None
    path = Path(path).expanduser()
    options = [path]
    if path.is_absolute():
        options += [source_dir / path.name, production_dir / path.name]
    else:
        options += [source_dir / path, production_dir / path]
    for option in options:
        if option.exists():
            return option.resolve()
    return path.resolve()


def _find_summary(production_dir, explicit=None):
    if explicit is not None:
        path = Path(explicit).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError("production summary does not exist: %s" % path)
        return path
    candidates = []
    for pattern in ("*current_model_summary.json", "**/*current_model_summary.json"):
        candidates.extend(sorted(production_dir.glob(pattern)))
    seen = set()
    for path in candidates:
        path = path.resolve()
        if path in seen:
            continue
        seen.add(path)
        try:
            data = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        if data.get("input_h5_files") or data.get("input_h5_glob"):
            return path
    raise FileNotFoundError(
        "could not find a current-model summary in %s; pass --summary" % production_dir)


def _discover_h5files(summary, production_dir, explicit_glob=None):
    if explicit_glob:
        paths = [Path(path).resolve() for path in glob.glob(explicit_glob)]
    else:
        paths = []
        for value in summary.get("input_h5_files", []):
            original = Path(str(value)).expanduser()
            candidates = [original, production_dir / original.name]
            for candidate in candidates:
                if candidate.exists():
                    paths.append(candidate.resolve())
                    break
        if not paths and summary.get("input_h5_glob"):
            pattern = str(summary["input_h5_glob"])
            paths = [Path(path).resolve() for path in glob.glob(pattern)]
    paths = sorted({path for path in paths if path.is_file()})
    if not paths:
        raise FileNotFoundError("no production H5 inputs were found")
    if any(path.name == "20200523_0000023.h5" for path in paths):
        raise RuntimeError("production H5 list contains excluded 20200523_0000023.h5")
    return paths


def _load_production_module(source_dir):
    path = source_dir / "make_m101_cube_current_model.py"
    if not path.exists():
        raise FileNotFoundError("production builder is unavailable: %s" % path)
    spec = importlib.util.spec_from_file_location("m101_position_production", path)
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(source_dir))
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _model_paths(summary, source_dir, production_dir):
    defaults = {
        "state": source_dir / "m101_onoff_hierarchy/m101_onoff_state.json",
        "band_cache": source_dir / "m101_band_initializer/m101_band_cache.npz",
        "alpha_fits": source_dir / "m101_alpha_template_bands/alpha_template_fits.csv",
        "alpha_history": source_dir / "m101_alpha_same_amp_loo/alpha_physical_amp_summary.csv",
        "amp_means": source_dir / "m101-amp-residual-pca-final/amp_residual_pca_models.npz",
        "fq_template": None,
        "blank_file": None,
        "on_filter": production_dir / "hbeta_on_filter.dat",
        "off_filter": production_dir / "hbeta_off_filter.dat",
        "exposure_scale": None,
    }
    result = {}
    for name, default in defaults.items():
        if name == "exposure_scale":
            path = (_summary_path(summary, "exposure_scale", "exposure_scale_csv") or
                    _summary_path(summary, "exposure_scale_csv"))
        elif name in ("on_filter", "off_filter"):
            path = default if default.exists() else _summary_path(
                summary, "filters", name.upper())
        elif name == "blank_file":
            path = (_summary_path(summary, "provenance", "blank_file") or
                    _summary_path(summary, "blank_loader", "path"))
        else:
            path = _summary_path(summary, "provenance", name)
        result[name] = _resolve_product(path or default, source_dir, production_dir)
    return result


def _load_scale_rows(path, h5files):
    if path is None or not path.exists():
        return None
    rows = {}
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            try:
                h5 = Path(row["H5"]).name
                exposure = int(row["exposure"])
                scale = float(row["cumulative_relative_scale"])
            except (KeyError, TypeError, ValueError):
                continue
            if h5 and np.isfinite(scale) and scale != 0:
                rows[(h5, exposure)] = {"cumulative_relative_scale": scale}
    required = {(path.name, exposure) for path in h5files for exposure in (1, 2, 3)}
    return rows if required.issubset(rows) else None


def _load_model(production, paths, h5files):
    state, response, composition, state_path = production._load_frozen_response(paths["state"])
    identity_map, cache_manifest, cache_manifest_path, cache_path = (
        production._load_identity_map(paths["band_cache"], h5files))
    alpha_direct, alpha_path = production._load_alpha_fits(paths["alpha_fits"])
    alpha_history, alpha_summary_path = production._load_alpha_history(paths["alpha_history"])
    amp_means, mean_mask, mean_scope, amp_means_path = production._load_amp_means(
        paths["amp_means"])
    fq, f0, fq_path = production._load_fq(paths["fq_template"])
    blank_masks, blank_provenance = production.frozen_parent.m101_blank_fibers.load(
        paths["blank_file"], h5files)
    return {
        "state": state, "response": response, "composition": composition,
        "state_path": state_path, "identity_map": identity_map,
        "cache_manifest": cache_manifest, "cache_manifest_path": cache_manifest_path,
        "cache_path": cache_path, "alpha_direct": alpha_direct,
        "alpha_path": alpha_path, "alpha_history": alpha_history,
        "alpha_summary_path": alpha_summary_path, "amp_means": amp_means,
        "mean_mask": mean_mask, "mean_scope": mean_scope,
        "amp_means_path": amp_means_path, "fq": fq, "f0": f0, "fq_path": fq_path,
        "blank_masks": blank_masks, "blank_provenance": blank_provenance,
    }


def _calibrate_records(production, model, h5files, scale_rows, paths,
                       production_dir, pixel_scale, bootstrap_draws):
    records = []
    for h5file in h5files:
        result = production._calibrate_h5_current(
            h5file, model["identity_map"], model["response"],
            model["alpha_direct"], model["alpha_history"], model["amp_means"],
            model["fq"], model["f0"], model["blank_masks"][h5file.name],
            production.M101_SKY_MIN_FIBERS)
        (spectra, errors, ra, dec, labels, surveys, calibration_summary,
         sky_records, sky_by_exposure, groups, q_rows, hardware_bad,
         corrected_total) = result
        records.append({
            "h5file": h5file,
            "spectra": spectra.astype(np.float32),
            "original_spectra": spectra.astype(np.float32).copy(),
            "errors": errors.astype(np.float32),
            "ra": np.asarray(ra, dtype=float), "dec": np.asarray(dec, dtype=float),
            "labels": np.asarray(labels, dtype=int), "surveys": surveys,
            "groups": groups, "q_rows": np.asarray(q_rows, dtype=int),
            "hardware_bad": np.asarray(hardware_bad, dtype=bool),
            "corrected_total": np.asarray(corrected_total, dtype=np.float32),
            "sky_by_exposure": {
                int(key): np.asarray(value, dtype=np.float32)
                for key, value in sky_by_exposure.items()
            },
            "blank_mask": np.asarray(model["blank_masks"][h5file.name], dtype=bool),
            "calibration_summary": calibration_summary,
        })
        say("calibrated %s (%d fibers)" % (h5file.name, len(ra)))

    if scale_rows is None:
        say("production scale CSV unavailable; deriving exact two-pass scales")
        on_filter = paths["on_filter"]
        off_filter = paths["off_filter"]
        if on_filter is None or off_filter is None or not on_filter.exists() or not off_filter.exists():
            raise FileNotFoundError("ON/OFF filters are required to derive exposure scales")
        on_values = production.validated_m101.read_filter(on_filter)
        off_values = production.validated_m101.read_filter(off_filter)
        band_responses, _ = production.native._band_responses(on_values, off_values)
        scale_list = production._apply_two_pass_exposure_scale(
            records, band_responses, pixel_scale, bootstrap_draws=bootstrap_draws)
        scale_rows = {
            (row["H5"], int(row["exposure"])): row for row in scale_list
        }

    for record in records:
        for exposure in production.EXPOSURES:
            scale = float(scale_rows[(record["h5file"].name, exposure)][
                "cumulative_relative_scale"])
            selected = record["labels"] == exposure
            record["spectra"][selected] = record["spectra"][selected] / scale
            record["spectra"][selected] = record["spectra"][selected].astype(np.float32)
        record.pop("original_spectra", None)
    return records, scale_rows


def _wave_index(def_wave, wavelength):
    indices = np.flatnonzero(np.isclose(def_wave, float(wavelength), rtol=0.0, atol=1.e-8))
    if len(indices) != 1:
        raise ValueError("wavelength %.8g A is not uniquely present in DEF_WAVE" % wavelength)
    return int(indices[0])


def _build_geometry(production, summary, cube_header, pixel_scale):
    image = summary.get("image_center_size", {})
    if not image:
        raise ValueError("production summary has no image_center_size geometry")
    ra = float(image["RA"])
    dec = float(image["Dec"])
    size_arcmin = float(image["size_arcsec"]) / 60.0
    _, _, _, n, xg, yg, xgrid, ygrid, tp = production._parse_image_geometry(
        "%.12f,%.12f,%.12f" % (ra, dec, size_arcmin), pixel_scale)
    if cube_header is not None:
        nx = int(cube_header.get("NAXIS1", n))
        ny = int(cube_header.get("NAXIS2", n))
        if (nx, ny) != (n, n):
            raise ValueError("summary geometry and production cube shape disagree")
        cube_wcs = WCS(cube_header).celestial
        cube_x, cube_y = cube_wcs.world_to_pixel_values(ra, dec)
        production_x, production_y = tp.wcs_world2pix([ra], [dec], 1)
        wcs_residual = float(np.hypot(
            float(production_x[0]) - 1.0 - float(cube_x),
            float(production_y[0]) - 1.0 - float(cube_y)))
        if not np.isfinite(wcs_residual) or wcs_residual > 1.e-5:
            raise ValueError("production summary geometry does not reproduce cube WCS")
    else:
        wcs_residual = np.nan
    return {"n": n, "xg": xg, "yg": yg, "xgrid": xgrid,
            "ygrid": ygrid, "tp": tp, "pixel_scale": pixel_scale,
            "wcs_center_residual_pixels": wcs_residual}


def _find_sci_cube(production_dir, summary):
    explicit = []
    for key in ("science_cube", "cube", "sci"):
        path = _summary_path(summary, key)
        if path:
            explicit.append(path)
    preferred = [
        production_dir / "m101_virus_v2_cube.fits",
        production_dir / "m101_v2_cube.fits",
        production_dir / "m101_cube.fits",
    ]
    candidates = explicit + preferred + sorted(production_dir.glob("*_cube.fits"))
    seen = set()
    for path in candidates:
        path = Path(path)
        if not path.is_absolute():
            path = production_dir / path
        path = path.resolve()
        if path in seen or not path.exists():
            continue
        seen.add(path)
        if any(token in path.name.lower() for token in
               ("variance", "error", "dq", "weight", "ncontrib")):
            continue
        try:
            header = fits.getheader(path)
            if int(header.get("NAXIS3", -1)) == 1036:
                return path
        except (OSError, ValueError):
            continue
    raise FileNotFoundError("could not find the production SCI cube in %s" % production_dir)


def _load_cube_context(cube_path, production, summary, targets, all_waves, spectral_waves):
    wave_indices = sorted({_wave_index(production.DEF_WAVE, wave)
                           for wave in all_waves + spectral_waves})
    with fits.open(cube_path, memmap=True) as hdul:
        data = hdul[0].data
        header = hdul[0].header.copy()
        if data.ndim != 3 or data.shape[0] != len(production.DEF_WAVE):
            raise ValueError("production SCI cube has unexpected shape: %s" % (data.shape,))
        shape = data.shape[1:]
        target_cube = {}
        spatial_baseline = {}
        spatial_scale = {}
        for category, points in targets.items():
            for point_index, (x, y) in enumerate(points):
                ix, iy = int(round(x)) - 1, int(round(y)) - 1
                if not (0 <= ix < shape[1] and 0 <= iy < shape[0]):
                    raise ValueError("target DS9 coordinate outside cube: (%s, %s)" % (x, y))
                target_cube[(category, point_index)] = {}
                spatial_baseline[(category, point_index)] = {}
                spatial_scale[(category, point_index)] = {}
                for wave in all_waves + spectral_waves:
                    index = _wave_index(production.DEF_WAVE, wave)
                    value = float(data[index, iy, ix])
                    target_cube[(category, point_index)][wave] = value
                    y0, y1 = max(0, iy - 4), min(shape[0], iy + 5)
                    x0, x1 = max(0, ix - 4), min(shape[1], ix + 5)
                    patch = np.asarray(data[index, y0:y1, x0:x1], dtype=float).ravel()
                    center = (iy - y0) * (x1 - x0) + (ix - x0)
                    patch = np.delete(patch, center)
                    spatial_baseline[(category, point_index)][wave] = _robust_location(patch)
                    spatial_scale[(category, point_index)][wave] = _robust_scale(patch)
        return {
            "path": cube_path, "header": header, "shape": shape,
            "target_cube": target_cube, "spatial_baseline": spatial_baseline,
            "spatial_scale": spatial_scale, "wave_indices": wave_indices,
        }


def _positions_for_wave(production, record, exposure, wave_index, tp):
    survey = record["surveys"][int(exposure)]
    astrometry = production.Astrometry(
        float(survey["ra"]), float(survey["dec"]), float(survey["pa"]), 0.0, 0.0)
    extractor = production.Extract(wave=production.DEF_WAVE)
    extractor.get_ADR_RAdec(astrometry)
    selected = record["labels"] == int(exposure)
    dec0 = float(survey["dec"])
    ra = record["ra"][selected] - extractor.ADRra[wave_index] / 3600.0 / np.cos(np.deg2rad(dec0))
    dec = record["dec"][selected] - extractor.ADRdec[wave_index] / 3600.0
    x, y = tp.wcs_world2pix(ra, dec, 1)
    return np.asarray(x, dtype=float), np.asarray(y, dtype=float), np.flatnonzero(selected)


def _deposits_at_pixel(production, x, y, values, target_ix, target_iy, sigma, area):
    """Return exact production-kernel weights and weighted values at one pixel."""
    target_x, target_y = float(target_ix + 1), float(target_iy + 1)
    finite = np.isfinite(values) & np.isfinite(x) & np.isfinite(y)
    x_center = np.floor(x - 1.0).astype(int)
    y_center = np.floor(y - 1.0).astype(int)
    in_kernel = (finite & (np.abs(target_ix - x_center) <= 3) &
                 (np.abs(target_iy - y_center) <= 3))
    indices = np.flatnonzero(in_kernel)
    if not len(indices):
        return indices, np.empty(0), np.empty(0)
    supported = (finite & ((target_x - x) ** 2 + (target_y - y) ** 2 <=
                          (2.0 * sigma) ** 2))
    if not np.any(supported):
        return np.empty(0, dtype=int), np.empty(0), np.empty(0)
    weights = np.asarray([
        _production_gaussian_weight(
            production, x[index], y[index], target_x, target_y, sigma)
        for index in indices
    ], dtype=float)
    weighted = weights * np.asarray(values[indices], dtype=float) / float(area)
    return indices, weights, weighted


def _target_wave_groups(production, record, positions, target, exposure, wave_index,
                        category, target_value, target_baseline, target_scale,
                        spectral_baseline=None):
    """Attribute one target/wavelength to exact groups and retain fiber rows."""
    sigma = (production.GAUSSIAN_FWHM_ARCSEC / 2.35) / 1.0
    area = production.FIBER_AREA
    x, y, row_indices = positions
    values = record["spectra"][row_indices, wave_index].astype(float)
    deposited, weights, weighted = _deposits_at_pixel(
        production, x, y, values, target["ix"], target["iy"], sigma, area)
    if not len(deposited):
        return [], []
    total_weight = float(np.sum(weights))
    total_weighted = float(np.sum(weighted))
    shot_value = total_weighted / total_weight if total_weight > 0 else np.nan
    group_by_row = {}
    for group in record["groups"]:
        if int(group["exposure"]) != int(exposure):
            continue
        for row in np.asarray(group["indices"], dtype=int):
            group_by_row[int(row)] = group
    group_data = {}
    for local_index, row_index in enumerate(row_indices[deposited]):
        group = group_by_row.get(int(row_index))
        if group is None:
            continue
        key = _group_key(record["h5file"], group)
        group_data.setdefault(key, {"group": group, "rows": [], "weights": [], "weighted": []})
        group_data[key]["rows"].append(int(row_index))
        group_data[key]["weights"].append(float(weights[local_index]))
        group_data[key]["weighted"].append(float(weighted[local_index]))

    aggregates = []
    fiber_rows = []
    total_abs = float(np.sum(np.abs(weighted)))
    for key, item in group_data.items():
        group_weight = float(np.sum(item["weights"]))
        group_weighted = float(np.sum(item["weighted"]))
        group_value = group_weighted / group_weight if group_weight > 0 else np.nan
        leave_weight = total_weight - group_weight
        leave_value = ((total_weighted - group_weighted) / leave_weight
                       if leave_weight > 0 else np.nan)
        group_fraction = (abs(group_weighted) / total_abs if total_abs > 0 else np.nan)
        robust_fiber = _robust_location(
            np.asarray(item["weighted"], dtype=float) /
            np.asarray(item["weights"], dtype=float))
        aggregate = {
            "key": key, "group": item["group"], "fiber_rows": item["rows"],
            "n_fibers": len(item["rows"]), "group_weight": group_weight,
            "group_weighted": group_weighted, "group_value": group_value,
            "robust_fiber_value": robust_fiber, "shot_value": shot_value,
            "leave_group_out_value": leave_value,
            "group_impact": shot_value - leave_value
            if np.isfinite(shot_value) and np.isfinite(leave_value) else np.nan,
            "weight_fraction": group_weight / total_weight if total_weight > 0 else np.nan,
            "contribution_fraction": group_fraction,
            "target_value": target_value, "target_baseline": target_baseline,
            "target_excess": target_value - target_baseline
            if np.isfinite(target_value) and np.isfinite(target_baseline) else np.nan,
            "local_baseline": spectral_baseline if spectral_baseline is not None else target_baseline,
            "spectral_excess": robust_fiber - spectral_baseline
            if spectral_baseline is not None and np.isfinite(robust_fiber)
            and np.isfinite(spectral_baseline) else np.nan,
            "total_weighted": total_weighted,
            "total_abs_weighted": total_abs,
            "wave_index": int(wave_index), "wavelength_A": float(production.DEF_WAVE[wave_index]),
            "category": category,
        }
        aggregates.append(aggregate)
        for row_index, weight, contribution in zip(
                item["rows"], item["weights"], item["weighted"]):
            fiber_rows.append({
                "row_index": int(row_index), "weight": float(weight),
                "weighted_contribution": float(contribution),
                "fiber_value": float(record["spectra"][row_index, wave_index] / area),
                "key": key,
            })
    return aggregates, fiber_rows


def _annotate_candidates(groups_by_target, category, mask_waves, target_count):
    """Rank groups and assign conservative candidate statuses per target."""
    decisions = {}
    for target_key, wave_map in groups_by_target.items():
        per_group = {}
        for wave, aggregates in wave_map.items():
            for aggregate in aggregates:
                item = per_group.setdefault(aggregate["key"], {"key": aggregate["key"], "waves": []})
                item["waves"].append(aggregate)
        ranked = []
        for item in per_group.values():
            rows = item["waves"]
            n_supported = len(rows)
            anomalies = np.asarray([row["target_excess"] for row in rows], dtype=float)
            impacts = np.asarray([row["group_impact"] for row in rows], dtype=float)
            fractions = np.asarray([row["contribution_fraction"] for row in rows], dtype=float)
            finite_anomaly = np.isfinite(anomalies)
            finite_impact = np.isfinite(impacts)
            signs_match = (finite_anomaly & finite_impact &
                           (np.sign(anomalies) == np.sign(impacts)) &
                           (np.abs(anomalies) > 0) & (np.abs(impacts) > 0))
            pathological = signs_match & (np.abs(impacts) >= 0.20 * np.abs(anomalies))
            score = float(np.nanmedian(np.abs(impacts[signs_match]))) if np.any(signs_match) else 0.0
            score *= float(np.nanmedian(fractions[np.isfinite(fractions)])) if np.any(np.isfinite(fractions)) else 0.0
            item.update({
                "supported_wavelengths": n_supported,
                "pathological_wavelengths": int(np.count_nonzero(pathological)),
                "sign_match_fraction": float(np.mean(signs_match)) if n_supported else 0.0,
                "median_contribution_fraction": _robust_location(fractions),
                "robust_residual": _robust_location(impacts),
                "score": score,
                "associated_positions": [target_key[1]],
            })
            ranked.append(item)
        ranked.sort(key=lambda item: item["score"], reverse=True)
        for rank, item in enumerate(ranked, 1):
            item["rank"] = rank
        if ranked:
            top = ranked[0]
            runner_score = ranked[1]["score"] if len(ranked) > 1 else 0.0
            margin = top["score"] / max(runner_score, 1.e-30)
            required_support = 4 if category == "ALL_WAVELENGTH" else 5
            required_path = 3 if category == "ALL_WAVELENGTH" else 4
            unambiguous = (
                top["supported_wavelengths"] >= required_support and
                top["pathological_wavelengths"] >= required_path and
                top["sign_match_fraction"] >= 0.67 and
                top["median_contribution_fraction"] >= 0.10 and
                margin >= 1.5)
            for item in ranked:
                item["status"] = (
                    "UNAMBIGUOUS" if item is top and unambiguous else
                    "AMBIGUOUS" if item is top else "NOT_CULPRIT")
                item["margin_over_runner_up"] = margin if item is top else np.nan
            decisions[target_key] = {"ranked": ranked, "confirmed": top if unambiguous else None,
                                     "top": top, "margin": margin}
        else:
            decisions[target_key] = {"ranked": [], "confirmed": None, "top": None, "margin": np.nan}
    return decisions


def _group_footprint(production, record, aggregate, wave_index, geometry, target):
    """Make a small exact-kernel footprint for the diagnostic plot."""
    group = aggregate["group"]
    x, y, row_indices = _positions_for_wave(
        production, record, int(group["exposure"]), wave_index, geometry["tp"])
    selected_rows = set(int(row) for row in np.asarray(group["indices"], dtype=int))
    local = np.asarray([index for index, row in enumerate(row_indices) if int(row) in selected_rows], dtype=int)
    ny, nx = geometry["n"], geometry["n"]
    flux_sum = np.zeros((ny, nx), dtype=np.float32)
    weight_sum = np.zeros((ny, nx), dtype=np.float32)
    variance_numerator = np.zeros((ny, nx), dtype=np.float32)
    error_weight_sum = np.zeros((ny, nx), dtype=np.float32)
    support = np.zeros((ny, nx), dtype=np.uint8)
    values = record["spectra"][row_indices, wave_index]
    production._gaussian_splat_shot_xy(
        local.astype(np.int64), x, y, values.astype(float),
        np.zeros_like(values, dtype=float), 1.0, 1.0, nx, ny,
        (production.GAUSSIAN_FWHM_ARCSEC / 2.35) / geometry["pixel_scale"],
        3, int(np.ceil(2.0 * (production.GAUSSIAN_FWHM_ARCSEC / 2.35) /
                      geometry["pixel_scale"])), float(production.FIBER_AREA),
        flux_sum, weight_sum, variance_numerator, error_weight_sum, support)
    image = np.full((ny, nx), np.nan, dtype=float)
    good = (support != 0) & (weight_sum > 0)
    image[good] = flux_sum[good] / weight_sum[good]
    x0, x1 = max(0, target["ix"] - 7), min(nx, target["ix"] + 8)
    y0, y1 = max(0, target["iy"] - 7), min(ny, target["iy"] + 8)
    return image[y0:y1, x0:x1], (x0, x1, y0, y1)


def _write_plot(production, records, targets, decisions, groups_by_target,
                geometry, output_path):
    all_targets = [("SPECTRAL_4882", index, point)
                   for index, point in enumerate(SPECTRAL_TARGETS)]
    all_targets += [("ALL_WAVELENGTH", index, point)
                    for index, point in enumerate(ALL_WAVELENGTH_TARGETS)]
    fig, axes = plt.subplots(4, 5, figsize=(15, 12), squeeze=False)
    for axis in axes.ravel():
        axis.axis("off")
    for axis, (category, point_index, (x, y)) in zip(axes.ravel(), all_targets):
        axis.axis("on")
        target = {"x": x, "y": y, "ix": int(round(x)) - 1,
                  "iy": int(round(y)) - 1}
        decision = decisions.get((category, point_index), {})
        top = decision.get("top")
        if top is None:
            axis.text(.5, .5, "no geometric candidate", ha="center", va="center")
            axis.set_title("%s\nDS9 (%d,%d)" % (category, x, y), fontsize=8)
            continue
        aggregate = next((row for row in top["waves"]
                          if np.isclose(row["wavelength_A"], 4882.0)),
                         top["waves"][0])
        record = next(record for record in records
                      if record["h5file"].name == aggregate["key"][0])
        wave_index = aggregate["wave_index"]
        image, extent = _group_footprint(
            production, record, aggregate, wave_index, geometry, target)
        axis.imshow(image, origin="lower", cmap="viridis", interpolation="nearest")
        tx = target["ix"] - extent[0]
        ty = target["iy"] - extent[2]
        axis.plot(tx, ty, "rx", ms=7, mew=1.5)
        status = top["status"]
        axis.set_title("%s %s\n(%d,%d) %s" %
                       (category, status, x, y, _group_label(top["key"])), fontsize=7)
        axis.set_xticks([])
        axis.set_yticks([])
    fig.suptitle("M101 position-driven attribution: top exact Gaussian group footprints\n"
                 "red x = manually supplied DS9 pixel; spectral panels use 4882 A",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _compare_previous(path, confirmed, ambiguous):
    if path is None or not path.exists():
        return {"available": False, "previous_rows": 0, "reconfirmed": [],
                "not_supported": [], "newly_discovered": [], "ambiguous": []}
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    previous = set()
    for row in rows:
        try:
            key = (Path(row["H5"]).name, int(row["exposure"]), int(row["SPECID"]),
                   int(row["IFUSLOT"]), int(row["IFUID"]), _amp(row["AMP"]))
            wave = float(row["wavelength_A"]) if "wavelength_A" in row else None
            previous.add((key, wave))
        except (KeyError, TypeError, ValueError):
            continue
    confirmed_entries = set(confirmed)
    ambiguous_entries = set(ambiguous)
    return {
        "available": True, "previous_rows": len(previous),
        "reconfirmed": sorted(previous & confirmed_entries, key=str),
        "not_supported": sorted(previous - confirmed_entries - ambiguous_entries, key=str),
        "newly_discovered": sorted(confirmed_entries - previous, key=str),
        "ambiguous": sorted(previous & ambiguous_entries, key=str),
    }


def _production_rejection_verification(source_path):
    source = source_path.read_text()
    lines = source.splitlines()
    def line(fragment, occurrence="last"):
        matches = [number for number, text in enumerate(lines, 1) if fragment in text]
        if not matches:
            return None
        return matches[-1] if occurrence == "last" else matches[0]
    qa_line = line("qa_rejection = _apply_qa_rejection")
    spectral_line = line("spectral_rejection = _apply_spectral_rejection")
    concat_line = line("specarray = np.concatenate")
    scale_line = line("exposure_scale_rows = _apply_two_pass_exposure_scale")
    qa_measure_line = line("qa_rows, qa_queue_rows, qa_summary = _evaluate_amplifier_qa")
    return {
        "qa_reject_argument": "--qa-reject-file" in source,
        "spectral_reject_argument": "--spectral-reject-file" in source,
        "qa_rejection_line": qa_line,
        "spectral_rejection_line": spectral_line,
        "specarray_concatenation_line": concat_line,
        "qa_before_specarray": bool(qa_line and concat_line and qa_line < concat_line),
        "spectral_before_specarray": bool(spectral_line and concat_line and spectral_line < concat_line),
        "qa_nans_spectra_and_errors": "record[\"spectra\"][indices] = np.nan" in source and
        "record[\"errors\"][indices] = np.nan" in source,
        "spectral_nans_spectra_and_errors": "record[\"spectra\"][indices, wave_index] = np.nan" in source and
        "record[\"errors\"][indices, wave_index] = np.nan" in source,
        "two_pass_before_qa": bool(scale_line and qa_line and scale_line < qa_line),
        "final_qa_before_rejection": bool(qa_measure_line and qa_line and qa_measure_line < qa_line),
    }


def _write_csv(path, rows):
    fields = [
        "category", "DS9_X", "DS9_Y", "numpy_x", "numpy_y", "fiber_row",
        "wavelength_A", "H5", "exposure", "SPECID", "IFUSLOT", "IFUID", "AMP",
        "fiber_value",
        "reconstruction_weight", "weighted_contribution", "contribution_fraction",
        "local_baseline", "spectral_excess", "candidate_rank", "attribution_status",
        "geometric_support", "dominant_flux_contributor", "anomalous_contributor",
        "robust_fiber_value", "group_reconstructed_value", "shot_reconstructed_value",
        "leave_group_out_value", "group_impact", "weight_fraction", "target_reconstructed_value",
        "target_baseline", "target_excess", "candidate_score", "supported_wavelengths",
        "pathological_wavelengths", "associated_positions",
    ]
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _jsonable(row.get(field, "")) for field in fields})


def _write_summary_text(path, summary):
    lines = [
        "M101 position-driven rejection attribution",
        "created_utc: %s" % summary["created_utc"],
        "production_cube: %s" % summary["production_cube"],
        "",
        "Authoritative coordinate convention:",
        "  DS9 X,Y are 1-based; NumPy ix=round(X)-1, iy=round(Y)-1.",
        "",
        "DEF_WAVE verification:",
    ]
    for row in summary["wavelength_grid"]["requested_indices"]:
        lines.append("  %s A -> zero-based index %s" % (row["wavelength_A"], row["index"]))
    lines.extend([
        "",
        "Final spectral rejection:",
        "  confirmed groups: %d" % summary["final_rejection"]["spectral"]["confirmed_groups"],
        "  first rejected wavelength: 4874 A",
        "  last rejected wavelength: 4890 A",
        "  planes per group: 9",
        "  total group-wavelength entries: %d" % summary["final_rejection"]["spectral"]["entries"],
        "",
        "Final whole-group QA rejection:",
        "  confirmed groups: %d" % summary["final_rejection"]["all_wavelength"]["confirmed_groups"],
        "",
        "Per-position decisions:",
    ])
    for item in summary["position_decisions"]:
        lines.append("  %s DS9=(%s,%s) NumPy=(%s,%s) top=%s status=%s" % (
            item["category"], item["DS9_X"], item["DS9_Y"], item["numpy_x"],
            item["numpy_y"], item.get("top_group", "none"), item.get("status", "NONE")))
    lines.extend([
        "",
        "Previous-list cross-checks:",
    ])
    for name, value in summary["previous_cross_check"].items():
        lines.append("  %s: %s" % (name, value))
    lines.extend([
        "",
        "Production rejection implementation verification:",
    ])
    for name, value in summary["production_rejection_verification"].items():
        lines.append("  %s: %s" % (name, value))
    lines.append("")
    path.write_text("\n".join(lines))


def _synthetic_checks():
    """Small no-data checks for attribution ranking and nine-plane expansion."""
    mask = tuple(range(4874, 4891, 2))
    assert mask == (4874, 4876, 4878, 4880, 4882, 4884, 4886, 4888, 4890)
    assert len(mask) == 9
    assert all((wave - 4874) % 2 == 0 for wave in mask)
    values = {wave: (10.0 if wave in mask else 1.0) for wave in range(4868, 4897, 2)}
    assert _robust_location([values[wave] for wave in SPECTRAL_CONTROL_WAVELENGTHS]) == 1.0
    # A group at the target pixel has the maximum exact Gaussian weight;
    # a group three pixels away still deposits, while a distant group does not.
    weights = [np.exp(-0.5 * (0.0 ** 2 + 0.0 ** 2) / .766 ** 2),
               np.exp(-0.5 * (3.0 ** 2 + 0.0 ** 2) / .766 ** 2), 0.0]
    assert weights[0] > weights[1] > weights[2]
    class Kernel:
        @staticmethod
        def production_gaussian_weight(fiber_x, fiber_y, pixel_x, pixel_y, sigma):
            return float(np.exp(-0.5 * ((fiber_x - pixel_x) ** 2 +
                                        (fiber_y - pixel_y) ** 2) / sigma ** 2))
    deposited, _, _ = _deposits_at_pixel(
        Kernel(), np.asarray([11.0, 14.0, 20.0]),
        np.asarray([11.0, 11.0, 20.0]), np.ones(3), 10, 10, .766, 1.0)
    assert deposited.tolist() == [0, 1]
    target_waves = SPECTRAL_INSPECTION_WAVELENGTHS
    fake_wave_map = {}
    for wave in target_waves:
        fake_wave_map[wave] = [
            {"key": ("bad.h5", 1, 1, 1, 1, "LU"),
             "target_excess": 9.0, "group_impact": 9.0,
             "contribution_fraction": .8},
            {"key": ("support.h5", 1, 1, 1, 1, "RU"),
             "target_excess": 9.0, "group_impact": .1,
             "contribution_fraction": .2},
        ]
    decision = _annotate_candidates({("SPECTRAL_4882", 0): fake_wave_map},
                                    "SPECTRAL_4882", target_waves, 1)[
                                        ("SPECTRAL_4882", 0)]
    assert decision["confirmed"]["key"] == ("bad.h5", 1, 1, 1, 1, "LU")
    assert set(SPECTRAL_REJECT_WAVELENGTHS) == set(mask)
    return {"nine_plane_expansion": list(mask),
            "known_gaussian_group_identified": True,
            "localized_spectral_anomaly_identified": True,
            "synthetic_status": "PASS"}


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--production-dir", type=Path, default=PRODUCTION_DIR)
    parser.add_argument("--source-dir", type=Path, default=REMEDY_DIR)
    parser.add_argument("--summary", type=Path, default=None)
    parser.add_argument("--h5-glob", default=None)
    parser.add_argument("--pixel-scale", type=float, default=1.0)
    parser.add_argument("--bootstrap-draws", type=int, default=300)
    parser.add_argument("--synthetic-test", action="store_true")
    return parser.parse_args()


def main():
    args = _parse_args()
    if args.synthetic_test:
        print(json.dumps(_synthetic_checks(), indent=2))
        return
    if args.pixel_scale <= 0 or args.bootstrap_draws < 1:
        raise SystemExit("--pixel-scale must be positive and --bootstrap-draws must be positive")

    production_dir = args.production_dir.expanduser().resolve()
    source_dir = args.source_dir.expanduser().resolve()
    output_dir = production_dir / OUTPUT_DIR_NAME
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = _find_summary(production_dir, args.summary)
    summary_data = json.loads(summary_path.read_text())
    h5files = _discover_h5files(summary_data, production_dir, args.h5_glob)
    production = _load_production_module(source_dir)
    def_wave = np.asarray(production.DEF_WAVE, dtype=float)
    requested_indices = []
    for wavelength in sorted(set(SPECTRAL_INSPECTION_WAVELENGTHS + ALL_WAVELENGTHS)):
        index = _wave_index(def_wave, wavelength)
        requested_indices.append({"wavelength_A": wavelength, "index": index})
        say("DEF_WAVE %.1f A -> zero-based index %d" % (wavelength, index))
    if _wave_index(def_wave, 4882) != 706:
        raise RuntimeError("DEF_WAVE does not place 4882 A at expected index 706")

    cube_path = _find_sci_cube(production_dir, summary_data)
    with fits.open(cube_path, memmap=True) as hdul:
        cube_header = hdul[0].header.copy()
    geometry = _build_geometry(production, summary_data, cube_header, args.pixel_scale)
    targets = {"SPECTRAL_4882": SPECTRAL_TARGETS,
               "ALL_WAVELENGTH": ALL_WAVELENGTH_TARGETS}
    cube_context = _load_cube_context(
        cube_path, production, summary_data, targets,
        ALL_WAVELENGTHS, SPECTRAL_INSPECTION_WAVELENGTHS)
    paths = _model_paths(summary_data, source_dir, production_dir)
    required = ("state", "band_cache", "alpha_fits", "alpha_history",
                "amp_means", "fq_template", "blank_file")
    for name in required:
        if paths[name] is None or not paths[name].exists():
            raise FileNotFoundError("required production input unavailable: %s=%s" %
                                    (name, paths[name]))
    model = _load_model(production, paths, h5files)
    scale_rows = _load_scale_rows(paths["exposure_scale"], h5files)
    records, scale_rows = _calibrate_records(
        production, model, h5files, scale_rows, paths, production_dir,
        args.pixel_scale, args.bootstrap_draws)

    target_wave_lists = {"SPECTRAL_4882": SPECTRAL_INSPECTION_WAVELENGTHS,
                         "ALL_WAVELENGTH": ALL_WAVELENGTHS}
    positions_cache = {}
    for record_index, record in enumerate(records):
        for exposure in production.EXPOSURES:
            for wavelength in sorted(set(SPECTRAL_INSPECTION_WAVELENGTHS + ALL_WAVELENGTHS)):
                wave_index = _wave_index(def_wave, wavelength)
                positions_cache[(record_index, exposure, wave_index)] = _positions_for_wave(
                    production, record, exposure, wave_index, geometry["tp"])

    groups_by_target = {}
    decision_objects = {}
    fiber_rows = []
    position_decisions = []
    for category, points in targets.items():
        for point_index, (x, y) in enumerate(points):
            target_key = (category, point_index)
            target = {"x": x, "y": y, "ix": int(round(x)) - 1,
                      "iy": int(round(y)) - 1}
            say("%s DS9 X,Y=(%d,%d) NumPy ix,iy=(%d,%d)" %
                (category, x, y, target["ix"], target["iy"]))
            wave_map = {}
            provisional_target_values = {}
            # First pass obtains the exact no-rejection reconstructed target value.
            for wavelength in target_wave_lists[category]:
                wave_index = _wave_index(def_wave, wavelength)
                shot_values = []
                for record_index, record in enumerate(records):
                    for exposure in production.EXPOSURES:
                        positions = positions_cache[(record_index, exposure, wave_index)]
                        values = record["spectra"][record["labels"] == exposure, wave_index]
                        dep, weights, weighted = _deposits_at_pixel(
                            production, positions[0], positions[1], values,
                            target["ix"], target["iy"],
                            (production.GAUSSIAN_FWHM_ARCSEC / 2.35) / args.pixel_scale,
                            production.FIBER_AREA)
                        if len(dep) and np.sum(weights) > 0:
                            shot_values.append(float(np.sum(weighted) / np.sum(weights)))
                provisional_target_values[wavelength] = _robust_location(shot_values)
            spectral_baseline_at_target = _robust_location(
                [provisional_target_values[w] for w in SPECTRAL_CONTROL_WAVELENGTHS
                 if w in provisional_target_values])
            spectral_scale_at_target = _robust_scale(
                [provisional_target_values[w] for w in SPECTRAL_CONTROL_WAVELENGTHS
                 if w in provisional_target_values])
            for wavelength in target_wave_lists[category]:
                wave_index = _wave_index(def_wave, wavelength)
                if category == "SPECTRAL_4882":
                    baseline = spectral_baseline_at_target
                    scale = spectral_scale_at_target
                    spectral_group_baseline = None
                else:
                    baseline = cube_context["spatial_baseline"][target_key][wavelength]
                    scale = cube_context["spatial_scale"][target_key][wavelength]
                    spectral_group_baseline = None
                aggregates = []
                for record_index, record in enumerate(records):
                    for exposure in production.EXPOSURES:
                        positions = positions_cache[(record_index, exposure, wave_index)]
                        one, _ = _target_wave_groups(
                            production, record, positions, target, exposure, wave_index, category,
                            provisional_target_values[wavelength], baseline, scale,
                            spectral_group_baseline)
                        for aggregate in one:
                            aggregate["record_index"] = record_index
                        aggregates.extend(one)
                wave_map[wavelength] = aggregates
            if category == "SPECTRAL_4882":
                # Establish each candidate's local spectral baseline from retained
                # wavelengths outside the intentional +/-8 A rejection window.
                by_group = {}
                for wave, rows in wave_map.items():
                    for row in rows:
                        by_group.setdefault(row["key"], []).append(row)
                for key, rows in by_group.items():
                    controls = [row["robust_fiber_value"] for row in rows
                                if row["wavelength_A"] in SPECTRAL_CONTROL_WAVELENGTHS]
                    group_baseline = _robust_location(controls)
                    for row in rows:
                        row["local_baseline"] = group_baseline
                        row["spectral_excess"] = (
                            row["robust_fiber_value"] - group_baseline
                            if np.isfinite(row["robust_fiber_value"]) and
                            np.isfinite(group_baseline) else np.nan)
            groups_by_target[target_key] = wave_map
            decisions = _annotate_candidates(
                {target_key: wave_map}, category,
                target_wave_lists[category], len(points))[target_key]
            decision_objects[target_key] = decisions
            for wave, rows in wave_map.items():
                ranks = {item["key"]: item for item in decisions["ranked"]}
                dominant = None
                if rows:
                    dominant = max(rows, key=lambda row: abs(row["group_weighted"]))["key"]
                for aggregate in rows:
                    candidate = ranks[aggregate["key"]]
                    aggregate["candidate_rank"] = candidate["rank"]
                    aggregate["candidate_status"] = candidate["status"]
                    aggregate["candidate_score"] = candidate["score"]
                    aggregate["supported_wavelengths"] = candidate["supported_wavelengths"]
                    aggregate["pathological_wavelengths"] = candidate["pathological_wavelengths"]
                    aggregate["associated_positions"] = ";".join(
                        "%s:%s" % (category, position_decisions_index)
                        for position_decisions_index in candidate["associated_positions"])
                    aggregate["dominant_key"] = dominant
                    aggregate["anomalous_contributor"] = (
                        candidate["status"] == "UNAMBIGUOUS" and
                        candidate["rank"] == 1 and
                        np.isfinite(aggregate["group_impact"]) and
                        np.isfinite(aggregate["target_excess"]) and
                        np.sign(aggregate["group_impact"]) == np.sign(aggregate["target_excess"]))
                    for fiber in [item for item in _fiber_rows_for_aggregate(
                            production, records, aggregate, target, wave_index, positions_cache)]:
                        fiber_rows.append({**fiber, **aggregate, **_key_dict(aggregate["key"]),
                                           "DS9_X": x, "DS9_Y": y,
                                           "numpy_x": target["ix"], "numpy_y": target["iy"],
                                           "fiber_row": fiber["row_index"],
                                           "reconstruction_weight": fiber["weight"],
                                           "weighted_contribution": fiber["weighted_contribution"],
                                           "contribution_fraction": (
                                               abs(fiber["weighted_contribution"]) /
                                               aggregate["total_abs_weighted"]
                                               if aggregate["total_abs_weighted"] > 0 else np.nan),
                                           "geometric_support": True,
                                           "dominant_flux_contributor": aggregate["key"] == dominant})
            top = decisions["top"]
            position_decisions.append({
                "category": category, "DS9_X": x, "DS9_Y": y,
                "numpy_x": target["ix"], "numpy_y": target["iy"],
                "top_group": _group_label(top["key"]) if top else None,
                "status": top["status"] if top else "NO_GEOMETRIC_SUPPORT",
                "candidate_count": len(decisions["ranked"]),
                "margin_over_runner_up": decisions["margin"],
                "candidates": [
                    {"group": _group_label(item["key"]),
                     "rank": item["rank"], "status": item["status"],
                     "supported_wavelengths": item["supported_wavelengths"],
                     "pathological_wavelengths": item["pathological_wavelengths"],
                     "median_contribution_fraction": item["median_contribution_fraction"],
                     "robust_residual_across_wavelength": item["robust_residual"],
                     "sign_match_fraction": item["sign_match_fraction"],
                     "score": item["score"]}
                    for item in decisions["ranked"]
                ],
                "target_values": provisional_target_values,
            })

    # Reconstruct decision objects for reject-file writing from per-position status.
    confirmed_spectral_groups = set()
    ambiguous_spectral_groups = set()
    confirmed_qa_groups = set()
    ambiguous_qa_groups = set()
    for target_key, wave_map in groups_by_target.items():
        category, point_index = target_key
        decision = _annotate_candidates({target_key: wave_map}, category,
                                        target_wave_lists[category],
                                        len(targets[category]))[target_key]
        if decision["confirmed"] is not None:
            if category == "SPECTRAL_4882":
                confirmed_spectral_groups.add(decision["confirmed"]["key"])
            else:
                confirmed_qa_groups.add(decision["confirmed"]["key"])
        if decision["top"] is not None and decision["top"]["status"] == "AMBIGUOUS":
            if category == "SPECTRAL_4882":
                ambiguous_spectral_groups.add(decision["top"]["key"])
            else:
                ambiguous_qa_groups.add(decision["top"]["key"])

    spectral_rows = []
    for key in sorted(confirmed_spectral_groups, key=str):
        for wavelength in SPECTRAL_REJECT_WAVELENGTHS:
            spectral_rows.append({**_key_dict(key), "wavelength_A": wavelength})
    qa_rows = [{**_key_dict(key)} for key in sorted(confirmed_qa_groups, key=str)]
    _write_csv(output_dir / "m101_position_attribution.csv", fiber_rows)
    with (output_dir / "m101_final_spectral_reject.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(GROUP_FIELDS) + ["wavelength_A"])
        writer.writeheader()
        writer.writerows(spectral_rows)
    with (output_dir / "m101_final_qa_reject.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(GROUP_FIELDS))
        writer.writeheader()
        writer.writerows(qa_rows)

    previous_spectral = _resolve_product(
        production_dir / "m101_targeted_spectral_reject.csv", source_dir, production_dir)
    previous_qa = _resolve_product(
        production_dir / "m101_targeted_qa_reject.csv", source_dir, production_dir)
    previous_cross_check = {
        "spectral": _compare_previous(
            previous_spectral,
            {(key, wave) for key in confirmed_spectral_groups for wave in SPECTRAL_REJECT_WAVELENGTHS},
            {(key, wave) for key in ambiguous_spectral_groups for wave in SPECTRAL_REJECT_WAVELENGTHS}),
        "qa": _compare_previous(
            previous_qa,
            {((key), None) for key in confirmed_qa_groups},
            {((key), None) for key in ambiguous_qa_groups}),
    }

    verification = _production_rejection_verification(
        source_dir / "make_m101_cube_current_model.py")
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "script": str(Path(__file__).resolve()),
        "production_directory": str(production_dir),
        "production_summary": str(summary_path),
        "production_cube": str(cube_path),
        "source_builder": str(source_dir / "make_m101_cube_current_model.py"),
        "input_h5_files": [str(path) for path in h5files],
        "coordinate_convention": "DS9 X,Y 1-based; numpy ix=round(X)-1, iy=round(Y)-1",
        "wavelength_grid": {
            "start_A": float(def_wave[0]), "stop_A": float(def_wave[-1]),
            "step_A": float(def_wave[1] - def_wave[0]), "planes": int(len(def_wave)),
            "requested_indices": requested_indices,
            "spectral_inspection_A": list(SPECTRAL_INSPECTION_WAVELENGTHS),
            "spectral_rejection_A": list(SPECTRAL_REJECT_WAVELENGTHS),
            "all_wavelength_test_A": list(ALL_WAVELENGTHS),
        },
        "geometry": {"pixels": geometry["n"], "pixel_scale_arcsec": args.pixel_scale,
                     "fwhm_arcsec": production.GAUSSIAN_FWHM_ARCSEC,
                     "summary_to_cube_wcs_center_residual_pixels": geometry[
                         "wcs_center_residual_pixels"],
                     "support_radius_sigma": 2.0, "deposition_radius_pixels": 3},
        "final_rejection": {
            "spectral": {"confirmed_groups": len(confirmed_spectral_groups),
                          "first_rejected_wavelength_A": 4874,
                          "last_rejected_wavelength_A": 4890,
                          "planes_per_group": 9, "entries": len(spectral_rows),
                          "groups": [_key_dict(key) for key in sorted(confirmed_spectral_groups, key=str)],
                          "ambiguous_groups": [_key_dict(key) for key in sorted(ambiguous_spectral_groups, key=str)]},
            "all_wavelength": {"confirmed_groups": len(confirmed_qa_groups),
                               "entries": len(qa_rows),
                               "groups": [_key_dict(key) for key in sorted(confirmed_qa_groups, key=str)],
                               "ambiguous_groups": [_key_dict(key) for key in sorted(ambiguous_qa_groups, key=str)]},
        },
        "position_decisions": position_decisions,
        "previous_cross_check": previous_cross_check,
        "production_rejection_verification": verification,
        "notes": [
            "Candidate identities from earlier automatic/interative tracing were not used.",
            "Manual positions are attribution seeds, not output masks.",
            "No final cube pixels, calibration, registry, sky, scale, QA, or reconstruction code was modified.",
            "contribution_fraction is the absolute weighted signal share at the target; signed contribution is retained separately as weighted_contribution.",
        ],
    }
    summary_json = output_dir / "m101_position_attribution_summary.json"
    summary_json.write_text(json.dumps(_jsonable(summary), indent=2))
    _write_summary_text(output_dir / "m101_position_attribution_summary.txt", _jsonable(summary))
    _write_plot(production, records, targets, decision_objects, groups_by_target,
                geometry, output_dir / "m101_position_attribution.png")
    say("wrote %s" % output_dir)
    say("final spectral groups=%d (%d entries); final whole-group QA groups=%d" %
        (len(confirmed_spectral_groups), len(spectral_rows), len(confirmed_qa_groups)))


def _fiber_rows_for_aggregate(production, records, aggregate, target, wave_index, positions_cache):
    record = records[int(aggregate["record_index"])]
    exposure = aggregate["key"][1]
    positions = positions_cache[(int(aggregate["record_index"]), exposure, wave_index)]
    selected_rows = set(aggregate["fiber_rows"])
    _, _, row_indices = positions
    values = record["spectra"][row_indices, wave_index]
    x, y = positions[0], positions[1]
    sigma = (production.GAUSSIAN_FWHM_ARCSEC / 2.35) / 1.0
    result = []
    for local, row_index in enumerate(row_indices):
        if int(row_index) not in selected_rows:
            continue
        weight = _production_gaussian_weight(
            production, x[local], y[local], target["ix"] + 1,
            target["iy"] + 1, sigma)
        if not np.isfinite(values[local]):
            continue
        result.append({"row_index": int(row_index), "weight": float(weight),
                       "fiber_value": float(values[local] / production.FIBER_AREA),
                       "weighted_contribution": float(weight * values[local] / production.FIBER_AREA)})
    return result


if __name__ == "__main__":
    main()
