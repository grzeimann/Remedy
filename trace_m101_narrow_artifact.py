#!/usr/bin/env python3
"""Trace the narrow M101 VIRUS cube artifact to H5/exposure/amplifier groups.

This is intentionally a diagnostic only.  It never writes a reject file, edits
the hardware registry, or rebuilds the production cube.  When run on TACC it
imports the production ``make_m101_cube_current_model.py`` from the Remedy
source tree and reuses its calibration and ADR conventions.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import importlib.util
from pathlib import Path
import sys
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from scipy import ndimage


PRODUCTION_DIR = Path("/work/03730/gregz/maverick/m101_recon")
REMEDY_DIR = Path("/work/03730/gregz/maverick/Remedy")
NATIVE_START_A = 3470.0
NATIVE_STEP_A = 2.0
NATIVE_PLANES = 1036
TARGET_WAVELENGTHS = (4874, 4876, 4882, 4884, 4886)
CONTROL_WAVELENGTHS = (4868, 4870, 4872, 4878, 4880, 4888, 4890)
GROUP_FIELDS = ("H5", "exposure", "SPECID", "IFUSLOT", "IFUID", "AMP")
OUTPUT_NAMES = {
    "csv": "m101_v2_narrow_artifact_amp_trace.csv",
    "json": "m101_v2_narrow_artifact_summary.json",
    "txt": "m101_v2_narrow_artifact_summary.txt",
    "residual_png": "m101_v2_narrow_artifact_residuals.png",
    "match_png": "m101_v2_narrow_artifact_amp_matches.png",
}


def say(message):
    print("[trace] " + message, flush=True)


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
    std = np.std(values, ddof=1)
    return float(std) if np.isfinite(std) and std > 0 else 0.0


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
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _text(value):
    if isinstance(value, (bytes, np.bytes_)):
        return value.decode("utf-8", errors="replace").strip()
    return str(value).strip()


def _amp(value):
    value = _text(value)
    if value.startswith("b'") and value.endswith("'"):
        value = value[2:-1]
    return value.upper()


def _physical_key(specid, ifuslot, ifuid, amp):
    return (int(specid), int(ifuslot), int(ifuid), _amp(amp))


def _group_key(h5_name, exposure, specid, ifuslot, ifuid, amp):
    return (Path(h5_name).name, int(exposure), int(specid), int(ifuslot),
            int(ifuid), _amp(amp))


def _group_label(key):
    return "%s/e%d/%d/%d/%d/%s" % key


def _wavelength_grid():
    return NATIVE_START_A + NATIVE_STEP_A * np.arange(NATIVE_PLANES, dtype=float)


def _wave_index(wavelength):
    index = int(round((float(wavelength) - NATIVE_START_A) / NATIVE_STEP_A))
    if not np.isclose(NATIVE_START_A + NATIVE_STEP_A * index,
                      float(wavelength), rtol=0, atol=1e-8):
        raise ValueError("wavelength is not on the authoritative 2 A grid: %s" % wavelength)
    return index


def _find_summary(production_dir, explicit=None):
    if explicit:
        path = Path(explicit).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError("production summary does not exist: %s" % path)
        return path
    preferred = (
        production_dir / "m101_virus_v2_current_model_summary.json",
        production_dir / "m101_virus_v2_cube_current_model_summary.json",
        production_dir / "m101_virus_v2_current_model_summary.json",
    )
    candidates = [path for path in preferred if path.exists()]
    candidates.extend(sorted(production_dir.glob("*current_model_summary.json")))
    candidates.extend(sorted(production_dir.glob("**/*current_model_summary.json")))
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


def _summary_path(summary, *keys):
    value = summary
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    if isinstance(value, dict):
        value = value.get("path")
    if not value:
        return None
    return Path(str(value)).expanduser()


def _resolve_product(path, source_dir, production_dir):
    if path is None:
        return None
    options = [path]
    if not path.is_absolute():
        options.extend((source_dir / path, production_dir / path))
    else:
        options.extend((source_dir / path.name, production_dir / path.name))
    for option in options:
        if option.exists():
            return option.resolve()
    return path.resolve()


def _discover_h5files(summary, production_dir, explicit_glob=None):
    if explicit_glob:
        paths = sorted(Path(path).resolve()
                       for path in __import__("glob").glob(explicit_glob))
    else:
        paths = []
        for value in summary.get("input_h5_files", []):
            path = Path(str(value)).expanduser()
            if path.exists():
                paths.append(path.resolve())
            else:
                local = production_dir / path.name
                if local.exists():
                    paths.append(local.resolve())
        if not paths and summary.get("input_h5_glob"):
            pattern = str(summary["input_h5_glob"])
            paths = sorted(Path(path).resolve()
                           for path in __import__("glob").glob(pattern))
    paths = sorted({path for path in paths if path.is_file()})
    if not paths:
        raise FileNotFoundError(
            "no H5 inputs were found; pass --h5-glob or inspect %s" %
            summary.get("script", "the production summary"))
    if any(path.name == "20200523_0000023.h5" for path in paths):
        raise RuntimeError("production H5 list contains explicitly excluded 20200523_0000023.h5")
    return paths


def _load_production_module(source_dir):
    path = source_dir / "make_m101_cube_current_model.py"
    if not path.exists():
        raise FileNotFoundError("production builder is unavailable: %s" % path)
    spec = importlib.util.spec_from_file_location("m101_production_builder", path)
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(source_dir))
    spec.loader.exec_module(module)
    return module


def _load_cube_planes(production_dir, wave_indices):
    sci_path = production_dir / "m101_virus_v2_cube.fits"
    dq_path = production_dir / "m101_virus_v2_dq_cube.fits"
    ncontrib_path = production_dir / "m101_virus_v2_ncontrib_cube.fits"
    say("loading production metadata")
    with fits.open(sci_path, memmap=True) as hdul:
        science = hdul[0].data
        header = hdul[0].header.copy()
        if science.ndim != 3 or science.shape[0] != NATIVE_PLANES:
            raise ValueError("SCI cube must have shape (1036, ny, nx), got %s" %
                             (science.shape,))
        selected = np.asarray(science[wave_indices], dtype=float)
        control_indices = np.asarray([_wave_index(w) for w in CONTROL_WAVELENGTHS])
        controls = np.asarray(science[control_indices], dtype=float)
        shape = science.shape[1:]
    with fits.open(dq_path, memmap=True) as hdul:
        dq = np.asarray(hdul[0].data[wave_indices])
    with fits.open(ncontrib_path, memmap=True) as hdul:
        ncontrib = np.asarray(hdul[0].data[wave_indices])
    baseline = np.nanmedian(controls, axis=0)
    valid = np.isfinite(selected) & np.isfinite(baseline)
    if dq.shape != selected.shape or ncontrib.shape != selected.shape:
        raise ValueError("SCI, DQ, and NCONTRIB selected-plane shapes disagree")
    valid &= ncontrib >= 2
    valid &= (np.asarray(dq, dtype=np.uint16) & np.uint16(1)) == 0
    residuals = selected - baseline[None, :, :]
    residuals[~valid] = np.nan
    return {
        "science_path": sci_path,
        "dq_path": dq_path,
        "ncontrib_path": ncontrib_path,
        "header": header,
        "shape": shape,
        "residuals": residuals,
        "baseline": baseline,
        "valid": valid,
        "wave_indices": np.asarray(wave_indices, dtype=int),
    }


def _artifact_regions(cube, target_wavelengths, min_pixels=4):
    residuals = cube["residuals"]
    all_finite = residuals[np.isfinite(residuals)]
    center = _robust_location(all_finite)
    scale = _robust_scale(all_finite)
    if not np.isfinite(scale) or scale <= 0:
        scale = max(abs(center), 1.0) * 1.0e-6
    thresholds = []
    regions = []
    for plane, wavelength in enumerate(target_wavelengths):
        values = residuals[plane]
        finite = values[np.isfinite(values)]
        local_center = _robust_location(finite)
        local_scale = _robust_scale(finite)
        if not np.isfinite(local_scale) or local_scale <= 0:
            local_scale = scale
        p995 = float(np.percentile(finite, 99.5)) if finite.size else np.nan
        threshold = max(local_center + 8.0 * local_scale,
                        p995 if np.isfinite(p995) else -np.inf)
        if not np.isfinite(threshold):
            threshold = local_center + 8.0 * local_scale
        thresholds.append(float(threshold))
        mask = np.isfinite(values) & (values > threshold)
        mask = ndimage.binary_opening(mask, structure=np.ones((3, 3), dtype=bool))
        labels, count = ndimage.label(mask, structure=np.ones((3, 3), dtype=np.uint8))
        for label in range(1, count + 1):
            yy, xx = np.where(labels == label)
            if yy.size < min_pixels:
                continue
            pixels = values[labels == label]
            positive = pixels[np.isfinite(pixels) & (pixels > 0)]
            y0, y1 = int(yy.min()), int(yy.max())
            x0, x1 = int(xx.min()), int(xx.max())
            regions.append({
                "wavelength": int(wavelength),
                "plane": int(plane),
                "bbox_y0": y0, "bbox_y1": y1,
                "bbox_x0": x0, "bbox_x1": x1,
                "N_pixels": int(yy.size),
                "robust_residual": _robust_location(pixels),
                "peak_residual": float(np.nanmax(pixels)),
                "integrated_residual": float(np.nansum(positive)),
                "threshold": float(threshold),
                "mask": labels == label,
            })
    return regions, thresholds, {"center": center, "scale": scale}


def _plot_residuals(cube, target_wavelengths, thresholds, output_path):
    residuals = cube["residuals"]
    finite = residuals[np.isfinite(residuals)]
    scale = _robust_scale(finite)
    center = _robust_location(finite)
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0
    lo = center - 6.0 * scale
    hi = center + 18.0 * scale
    if not np.isfinite(hi) or hi <= lo:
        lo, hi = np.nanpercentile(finite, [1, 99.5])
    fig, axes = plt.subplots(1, len(target_wavelengths),
                             figsize=(3.2 * len(target_wavelengths), 3.4),
                             squeeze=False)
    for plane, (axis, wavelength) in enumerate(zip(axes[0], target_wavelengths)):
        image = np.ma.masked_invalid(residuals[plane])
        im = axis.imshow(image, origin="lower", cmap="magma", vmin=lo, vmax=hi,
                         interpolation="nearest")
        axis.set_title("%d A\nthreshold %.3g" % (wavelength, thresholds[plane]), fontsize=8)
        axis.set_xticks([])
        axis.set_yticks([])
        fig.colorbar(im, ax=axis, fraction=.046, pad=.03)
    fig.suptitle("M101 v2 narrow-artifact residuals\nfixed robust scale", fontsize=11)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _model_paths(summary, source_dir, production_dir):
    provenance = summary.get("provenance", {})
    defaults = {
        "state": source_dir / "m101_onoff_hierarchy/m101_onoff_state.json",
        "band_cache": source_dir / "m101_band_initializer/m101_band_cache.npz",
        "alpha_fits": source_dir / "m101_alpha_template_bands/alpha_template_fits.csv",
        "alpha_history": source_dir / "m101_alpha_same_amp_loo/alpha_physical_amp_summary.csv",
        "amp_means": source_dir / "m101-amp-residual-pca-final/amp_residual_pca_models.npz",
        "fq_template": None,
        "blank_file": None,
        "on_filter": None,
        "off_filter": None,
        "exposure_scale": None,
    }
    result = {}
    for name, default in defaults.items():
        if name == "exposure_scale":
            path = _summary_path(summary, "exposure_scale", "exposure_scale_csv")
            if path is None:
                path = _summary_path(summary, "exposure_scale_csv")
        elif name in ("on_filter", "off_filter"):
            path = _summary_path(summary, "filters", name.upper())
        elif name == "blank_file":
            path = _summary_path(summary, "provenance", "blank_file")
            if path is None:
                path = _summary_path(summary, "blank_loader", "path")
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
            h5 = Path(row.get("H5", "")).name
            if not h5:
                continue
            try:
                exposure = int(row["exposure"])
                g1 = float(row["g1_rel"])
                g2 = float(row["g2_rel"])
                cumulative = float(row.get("cumulative_relative_scale", g1 * g2))
            except (KeyError, TypeError, ValueError):
                continue
            if not np.isfinite(cumulative) or cumulative == 0:
                continue
            rows[(h5, exposure)] = {
                "g1_rel": g1, "g2_rel": g2,
                "cumulative_relative_scale": cumulative,
            }
    required = {(path.name, exposure) for path in h5files for exposure in (1, 2, 3)}
    if not required.issubset(rows):
        return None
    return rows


def _load_model(production, paths, h5files):
    state, response, composition, state_path = production._load_frozen_response(paths["state"])
    identity_map, cache_manifest, cache_manifest_path, cache_path = production._load_identity_map(
        paths["band_cache"], h5files)
    alpha_direct, alpha_path = production._load_alpha_fits(paths["alpha_fits"])
    alpha_history, alpha_summary_path = production._load_alpha_history(paths["alpha_history"])
    amp_means, mean_mask, mean_scope, amp_means_path = production._load_amp_means(paths["amp_means"])
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
        "amp_means_path": amp_means_path, "fq": fq, "f0": f0,
        "fq_path": fq_path, "blank_masks": blank_masks,
        "blank_provenance": blank_provenance,
    }


def _calibrate_narrow_inputs(production, model, h5files, target_indices,
                             control_indices, scale_rows, production_dir,
                             bootstrap_draws):
    say("loading/calibrating affected fiber wavelengths")
    narrow = []
    full_records = []
    for h5file in h5files:
        result = production._calibrate_h5_current(
            h5file, model["identity_map"], model["response"],
            model["alpha_direct"], model["alpha_history"], model["amp_means"],
            model["fq"], model["f0"], model["blank_masks"][h5file.name],
            production.M101_SKY_MIN_FIBERS)
        (spectra, errors, ra, dec, labels, surveys, calibration_summary,
         sky_records, sky_by_exposure, groups, q_rows, hardware_bad,
         corrected_total) = result
        if scale_rows is None:
            full_records.append({
                "h5file": h5file, "spectra": spectra.astype(np.float32),
                "original_spectra": spectra.astype(np.float32).copy(),
                "errors": errors.astype(np.float32), "ra": ra, "dec": dec,
                "labels": labels, "surveys": surveys, "groups": groups,
                "hardware_bad": hardware_bad,
                "corrected_total": np.asarray(corrected_total, dtype=np.float32),
                "sky_by_exposure": {int(k): np.asarray(v, dtype=np.float32)
                                    for k, v in sky_by_exposure.items()},
                "blank_mask": np.asarray(model["blank_masks"][h5file.name], dtype=bool),
                "calibration_summary": calibration_summary,
            })
            continue
        selected = np.concatenate((target_indices, control_indices))
        order = np.argsort(selected)
        selected_sorted = selected[order]
        selected_spectra = np.asarray(spectra[:, selected_sorted], dtype=float)
        selected_spectra = selected_spectra[:, np.argsort(order)]
        for exposure in (1, 2, 3):
            mask = np.asarray(labels, dtype=int) == exposure
            scale = scale_rows[(h5file.name, exposure)]["cumulative_relative_scale"]
            selected_spectra[mask] /= scale
        narrow.append({
            "H5": h5file.name, "spectra": selected_spectra,
            "ra": np.asarray(ra, dtype=float), "dec": np.asarray(dec, dtype=float),
            "labels": np.asarray(labels, dtype=int), "groups": groups,
            "q_rows": np.asarray(q_rows, dtype=int),
            "calibration_summary": calibration_summary,
        })
        say("  calibrated %s (%d fibers)" % (h5file.name, len(ra)))

    if scale_rows is None:
        say("production exposure-scale CSV was unavailable; deriving the exact two-pass scales")
        on_filter = production.frozen_parent._default_input("on_filter")
        off_filter = production.frozen_parent._default_input("off_filter")
        if not on_filter or not off_filter:
            raise FileNotFoundError("cannot derive exposure scales: ON/OFF filters unavailable")
        on_values = production.validated_m101.read_filter(on_filter)
        off_values = production.validated_m101.read_filter(off_filter)
        band_responses, _ = production.native._band_responses(on_values, off_values)
        scale_rows_list = production._apply_two_pass_exposure_scale(
            full_records, band_responses, 1.0, bootstrap_draws=bootstrap_draws)
        scale_rows = {(row["H5"], int(row["exposure"])): row
                      for row in scale_rows_list}
        for record in full_records:
            target = np.concatenate((target_indices, control_indices))
            order = np.argsort(target)
            selected = np.asarray(record["spectra"][:, target[order]], dtype=float)
            selected = selected[:, np.argsort(order)]
            for exposure in (1, 2, 3):
                selected[record["labels"] == exposure] /= scale_rows[
                    (record["h5file"].name, exposure)]["cumulative_relative_scale"]
            narrow.append({
                "H5": record["h5file"].name, "spectra": selected,
                "ra": np.asarray(record["ra"], dtype=float),
                "dec": np.asarray(record["dec"], dtype=float),
                "labels": np.asarray(record["labels"], dtype=int),
                "groups": record["groups"], "q_rows": np.asarray(record["q_rows"], dtype=int),
                "calibration_summary": record["calibration_summary"],
            })
    return narrow, scale_rows


def _group_rows(narrow, target_wavelengths, control_wavelengths,
                thresholds, regions, target_indices, control_indices):
    say("grouping amplifier candidates")
    rows = []
    per_group = {}
    for record in narrow:
        spectra = record["spectra"]
        baseline = np.nanmedian(spectra[:, len(target_indices):], axis=1)
        target_flux = spectra[:, :len(target_indices)]
        target_residual = target_flux - baseline[:, None]
        control_residual = spectra[:, len(target_indices):] - baseline[:, None]
        for group in record["groups"]:
            exposure = int(group["exposure"])
            indices = np.asarray(group["indices"], dtype=int)
            if indices.size == 0:
                continue
            key = _group_key(record["H5"], exposure, group["specid"],
                             group["ifuslot"], group["ifuid"], group["amp"])
            physical = _physical_key(group["specid"], group["ifuslot"],
                                     group["ifuid"], group["amp"])
            score = 0.0
            for plane, wavelength in enumerate(target_wavelengths):
                values = target_residual[indices, plane]
                threshold = thresholds[plane]
                finite = values[np.isfinite(values)]
                positive = finite[finite > 0]
                extreme = finite > threshold
                integrated = float(np.sum(positive)) if positive.size else 0.0
                robust = _robust_location(finite)
                maximum = float(np.max(finite)) if finite.size else np.nan
                fraction = float(np.mean(extreme)) if finite.size else np.nan
                score += integrated / max(abs(threshold), 1.0)
                cvalues = control_residual[indices].ravel()
                rows.append({
                    "H5": key[0], "exposure": key[1], "SPECID": key[2],
                    "IFUSLOT": key[3], "IFUID": key[4], "AMP": key[5],
                    "physical_amplifier": "%d/%d/%d/%s" % physical,
                    "wavelength": int(wavelength), "N_fibers": int(indices.size),
                    "robust_residual": robust, "max_residual": maximum,
                    "integrated_positive_residual": integrated,
                    "extreme_fraction": fraction,
                    "control_wavelength_residual": _robust_location(cvalues),
                    "extreme_threshold": float(threshold),
                    "spatial_match_fraction": np.nan,
                    "spatial_dice": np.nan,
                    "candidate_rank": np.nan,
                    "candidate_score": np.nan,
                    "_group_key": key, "_fiber_indices": indices,
                    "_record": record,
                })
            per_group[key] = {
                "key": key, "physical": physical, "record": record,
                "score": score,
            }
    ranked = sorted(per_group.values(), key=lambda item: item["score"], reverse=True)
    rank_map = {item["key"]: rank for rank, item in enumerate(ranked, 1)}
    score_map = {item["key"]: item["score"] for item in ranked}
    for row in rows:
        row["candidate_rank"] = rank_map[row["_group_key"]]
        row["candidate_score"] = score_map[row["_group_key"]]
    return rows, ranked


def _spatial_setup(cube, production, header):
    try:
        wcs = WCS(header).celestial
    except Exception as exc:
        raise RuntimeError("could not read the production spatial WCS") from exc
    ny, nx = cube["shape"]
    sigma = (production.GAUSSIAN_FWHM_ARCSEC / 2.35) / 1.0
    return wcs, nx, ny, sigma


def _candidate_footprint(production, record, group, wavelength_index,
                         residual_values, cube, wcs, nx, ny, sigma):
    group_indices = np.asarray(group["indices"], dtype=int)
    selected = record["labels"] == int(group["exposure"])
    group_indices = group_indices[selected[group_indices]]
    if group_indices.size == 0:
        return np.zeros((ny, nx), dtype=float), np.zeros((ny, nx), dtype=bool)
    survey = record["surveys"][int(group["exposure"])] if "surveys" in record else None
    # The narrow record deliberately does not retain surveys; re-read them from
    # the H5 through the production calibration helper's original group record.
    # They are attached below by _attach_surveys before attribution.
    if survey is None:
        survey = record["survey_by_exposure"][int(group["exposure"])]
    astrometry = production.Astrometry(float(survey["ra"]), float(survey["dec"]),
                                       float(survey["pa"]), 0.0, 0.0)
    extractor = production.Extract(wave=production.DEF_WAVE)
    extractor.get_ADR_RAdec(astrometry)
    dec0 = float(survey["dec"])
    ra = record["ra"][group_indices] - extractor.ADRra[wavelength_index] / 3600.0 / np.cos(np.deg2rad(dec0))
    dec = record["dec"][group_indices] - extractor.ADRdec[wavelength_index] / 3600.0
    x, y = wcs.world_to_pixel_values(ra, dec)
    x = np.asarray(x, dtype=float) + 1.0
    y = np.asarray(y, dtype=float) + 1.0
    values = np.asarray(residual_values[group_indices], dtype=float)
    errors = np.zeros_like(values)
    indices = np.arange(group_indices.size, dtype=np.int64)
    flux_sum = np.zeros((ny, nx), dtype=np.float32)
    weight_sum = np.zeros((ny, nx), dtype=np.float32)
    variance_numerator = np.zeros((ny, nx), dtype=np.float32)
    error_weight_sum = np.zeros((ny, nx), dtype=np.float32)
    support = np.zeros((ny, nx), dtype=np.uint8)
    production._gaussian_splat_shot_xy(
        indices, x, y, values, errors, 1.0, 1.0, nx, ny, sigma, 3,
        int(np.ceil(2.0 * sigma)), float(production.FIBER_AREA), flux_sum,
        weight_sum, variance_numerator, error_weight_sum, support)
    image = np.full((ny, nx), np.nan, dtype=float)
    good = support.astype(bool) & (weight_sum > 0)
    image[good] = flux_sum[good] / weight_sum[good]
    return image, good


def _attach_surveys(narrow, h5files):
    import tables
    by_name = {path.name: path for path in h5files}
    for record in narrow:
        survey = {}
        with tables.open_file(by_name[record["H5"]], mode="r") as h5:
            for row in h5.root.Survey:
                survey[int(row["exp"])] = {name: row[name] for name in h5.root.Survey.colnames}
        record["survey_by_exposure"] = survey


def _attribute_top_candidates(production, narrow, rows, ranked, cube,
                              target_wavelengths, target_indices, top_n):
    say("reconstructing top candidate footprints")
    wcs, nx, ny, sigma = _spatial_setup(cube, production, cube["header"])
    row_lookup = {(row["_group_key"], int(row["wavelength"])): row for row in rows}
    affected_masks = {}
    for wavelength, plane in zip(target_wavelengths, range(len(target_wavelengths))):
        mask = np.zeros((ny, nx), dtype=bool)
        for region in cube["regions"]:
            if region["wavelength"] == wavelength:
                mask |= region["mask"]
        affected_masks[wavelength] = mask
    for item in ranked[:top_n]:
        record = item["record"]
        group = next(group for group in record["groups"]
                     if _group_key(record["H5"], group["exposure"], group["specid"],
                                   group["ifuslot"], group["ifuid"], group["amp"]) == item["key"])
        for plane, wavelength in enumerate(target_wavelengths):
            values = record["spectra"][:, plane] - np.nanmedian(
                record["spectra"][:, len(target_indices):], axis=1)
            image, support = _candidate_footprint(
                production, record, group, target_indices[plane], values,
                cube, wcs, nx, ny, sigma)
            detected = affected_masks[wavelength]
            positive = support & np.isfinite(image) & (image > 0)
            intersection = int(np.count_nonzero(positive & detected))
            detected_count = int(np.count_nonzero(detected))
            candidate_count = int(np.count_nonzero(positive))
            match = intersection / detected_count if detected_count else np.nan
            dice = (2.0 * intersection / (detected_count + candidate_count)
                    if detected_count + candidate_count else np.nan)
            row = row_lookup.get((item["key"], int(wavelength)))
            if row is not None:
                row["spatial_match_fraction"] = match
                row["spatial_dice"] = dice
            item.setdefault("footprints", {})[int(wavelength)] = {
                "image": image, "support": support,
                "match_fraction": match, "dice": dice,
            }


def _plot_matches(cube, target_wavelengths, ranked, rows, output_path, top_n):
    selected = ranked[:top_n]
    ncols = len(target_wavelengths)
    nrows = max(1, len(selected))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.0 * ncols, 2.8 * nrows),
                             squeeze=False)
    for row_index, item in enumerate(selected):
        for col, wavelength in enumerate(target_wavelengths):
            axis = axes[row_index, col]
            image = item.get("footprints", {}).get(int(wavelength), {}).get("image")
            if image is None:
                axis.axis("off")
                continue
            finite = image[np.isfinite(image)]
            if finite.size:
                vmax = np.percentile(finite, 99.0)
                vmin = np.percentile(finite, 5.0)
                if vmax <= vmin:
                    vmin, vmax = np.nanmin(finite), np.nanmax(finite)
            else:
                vmin, vmax = 0, 1
            axis.imshow(image, origin="lower", cmap="viridis", vmin=vmin, vmax=vmax,
                        interpolation="nearest")
            for region in cube["regions"]:
                if region["wavelength"] != wavelength:
                    continue
                mask = region["mask"]
                axis.contour(mask.astype(float), levels=[.5], colors="red", linewidths=.7)
            if col == 0:
                axis.set_ylabel("rank %d\n%s" % (rows_for_key(rows, item["key"]),
                                                  _group_label(item["key"])), fontsize=7)
            if row_index == 0:
                axis.set_title("%d A" % wavelength, fontsize=8)
            axis.set_xticks([])
            axis.set_yticks([])
    fig.suptitle("Top amplifier/exposure footprints; red = cube residual regions", fontsize=11)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def rows_for_key(rows, key):
    values = [row["candidate_rank"] for row in rows if row["_group_key"] == key]
    return int(values[0]) if values else -1


def _clean_rows(rows):
    fields = [
        "H5", "exposure", "SPECID", "IFUSLOT", "IFUID", "AMP",
        "physical_amplifier", "wavelength", "N_fibers", "robust_residual",
        "max_residual", "integrated_positive_residual", "extreme_fraction",
        "control_wavelength_residual", "extreme_threshold",
        "spatial_match_fraction", "spatial_dice", "candidate_rank",
        "candidate_score",
    ]
    clean = []
    for row in rows:
        clean.append({field: _jsonable(row.get(field, "")) for field in fields})
    return fields, clean


def _decision_support(rows, ranked, top_n):
    top = ranked[:top_n]
    top_keys = {item["key"] for item in top}
    physical = {}
    for item in top:
        key = item["physical"]
        physical.setdefault(key, {"H5": set(), "instances": [], "score": 0.0})
        physical[key]["H5"].add(item["key"][0])
        physical[key]["instances"].append(item["key"])
        physical[key]["score"] += item["score"]
    repeated = [item for item in physical.values() if len(item["H5"]) >= 2]
    reject_rows = []
    for item in top:
        key = item["key"]
        relevant = [row for row in rows if row["_group_key"] == key]
        if any(np.isfinite(row["extreme_fraction"]) and row["extreme_fraction"] >= .05
               for row in relevant):
            reject_rows.append({field: key[index] for index, field in enumerate(GROUP_FIELDS)})
    return {
        "top_candidates": [
            {"group": _group_label(item["key"]),
             "physical_amplifier": "%d/%d/%d/%s" % item["physical"],
             "score": item["score"]}
            for item in top],
        "repeated_physical_amplifiers": [
            {"physical_amplifier": "%d/%d/%d/%s" % key,
             "distinct_H5": sorted(value["H5"]),
             "group_instances": [_group_label(item) for item in value["instances"]],
             "score": value["score"]}
            for key, value in physical.items() if len(value["H5"]) >= 2],
        "one_physical_amplifier_appears_repeatedly": bool(repeated),
        "failure_pattern": (
            "persistent across multiple H5 files" if repeated else
            "exposure-specific or confined to one H5 among the ranked candidates"),
        "rows_that_would_belong_in_explicit_QA_reject_CSV": reject_rows,
        "hardware_registry_consideration": (
            "consider persistent physical amplifier(s) above" if repeated else
            "no multi-H5 persistence among the top candidates"),
    }


def _write_summary_text(path, summary):
    decision = summary["decision_support"]
    lines = [
        "M101 v2 narrow artifact trace",
        "created_utc: %s" % summary["created_utc"],
        "",
        "Top candidate amplifier/exposure groups:",
    ]
    for index, item in enumerate(decision["top_candidates"], 1):
        lines.append("  %d. %s  physical=%s  score=%.6g" %
                     (index, item["group"], item["physical_amplifier"], item["score"]))
    lines.extend([
        "",
        "One physical amplifier appears repeatedly: %s" %
        decision["one_physical_amplifier_appears_repeatedly"],
        "Failure pattern: %s" % decision["failure_pattern"],
        "Hardware-registry guidance: %s" % decision["hardware_registry_consideration"],
        "",
        "Rows that WOULD belong in an explicit QA reject CSV (not written):",
    ])
    if decision["rows_that_would_belong_in_explicit_QA_reject_CSV"]:
        for row in decision["rows_that_would_belong_in_explicit_QA_reject_CSV"]:
            lines.append("  " + ",".join("%s=%s" % (key, value)
                                         for key, value in row.items()))
    else:
        lines.append("  none selected by the diagnostic ranking threshold")
    lines.extend([
        "",
        "No reject CSV was created. No production product or hardware registry was modified.",
    ])
    path.write_text("\n".join(lines) + "\n")


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--production-dir", type=Path, default=PRODUCTION_DIR)
    parser.add_argument("--source-dir", type=Path, default=REMEDY_DIR)
    parser.add_argument("--summary", type=Path, default=None,
                        help="production current-model summary JSON")
    parser.add_argument("--h5-glob", default=None,
                        help="override H5 discovery if the summary lacks input_h5_files")
    parser.add_argument("--top-groups", type=int, default=12)
    parser.add_argument("--min-region-pixels", type=int, default=4)
    parser.add_argument("--bootstrap-draws", type=int, default=300,
                        help="only used if the production exposure-scale CSV is unavailable")
    return parser.parse_args()


def main():
    args = _parse_args()
    if args.top_groups < 1 or args.min_region_pixels < 1:
        raise SystemExit("--top-groups and --min-region-pixels must be positive")
    production_dir = args.production_dir.expanduser().resolve()
    source_dir = args.source_dir.expanduser().resolve()
    output_dir = production_dir / "m101_v2_narrow_artifact_trace"
    output_dir.mkdir(parents=True, exist_ok=True)

    wavelengths = _wavelength_grid()
    target_indices = np.asarray([_wave_index(w) for w in TARGET_WAVELENGTHS], dtype=int)
    control_indices = np.asarray([_wave_index(w) for w in CONTROL_WAVELENGTHS], dtype=int)
    summary_path = _find_summary(production_dir, args.summary)
    summary_data = json.loads(summary_path.read_text())
    h5files = _discover_h5files(summary_data, production_dir, args.h5_glob)
    production = _load_production_module(source_dir)
    if not np.allclose(production.DEF_WAVE, wavelengths, rtol=0, atol=0):
        raise RuntimeError("production DEF_WAVE is not the required authoritative 3470+2*i grid")

    cube = _load_cube_planes(production_dir, target_indices)
    regions, thresholds, robust_cube = _artifact_regions(
        cube, TARGET_WAVELENGTHS, min_pixels=args.min_region_pixels)
    cube["regions"] = regions
    _plot_residuals(cube, TARGET_WAVELENGTHS, thresholds,
                    output_dir / OUTPUT_NAMES["residual_png"])
    say("locating affected cube regions: %d coherent regions" % len(regions))

    paths = _model_paths(summary_data, source_dir, production_dir)
    required_paths = ("state", "band_cache", "alpha_fits", "alpha_history",
                      "amp_means", "fq_template", "blank_file")
    for name in required_paths:
        path = paths[name]
        if path is None or not path.exists():
            raise FileNotFoundError("required production input %s is unavailable: %s" % (name, path))
    model = _load_model(production, paths, h5files)
    scale_rows = _load_scale_rows(paths["exposure_scale"], h5files)
    narrow, scale_rows = _calibrate_narrow_inputs(
        production, model, h5files, target_indices, control_indices,
        scale_rows, production_dir, args.bootstrap_draws)
    _attach_surveys(narrow, h5files)
    rows, ranked = _group_rows(
        narrow, TARGET_WAVELENGTHS, CONTROL_WAVELENGTHS, thresholds, regions,
        target_indices, control_indices)
    _attribute_top_candidates(
        production, narrow, rows, ranked, cube, TARGET_WAVELENGTHS,
        target_indices, args.top_groups)
    _plot_matches(cube, TARGET_WAVELENGTHS, ranked, rows,
                  output_dir / OUTPUT_NAMES["match_png"], args.top_groups)

    fields, clean_rows = _clean_rows(rows)
    with (output_dir / OUTPUT_NAMES["csv"]).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(clean_rows)
    decision = _decision_support(rows, ranked, args.top_groups)
    result = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "script": str(Path(__file__).resolve()),
        "production_directory": str(production_dir),
        "production_summary": str(summary_path),
        "source_builder": str(source_dir / "make_m101_cube_current_model.py"),
        "input_h5_files": [str(path) for path in h5files],
        "wavelength_grid": {"start_A": NATIVE_START_A, "step_A": NATIVE_STEP_A,
                            "planes": NATIVE_PLANES},
        "target_wavelengths_A": list(TARGET_WAVELENGTHS),
        "control_wavelengths_A": list(CONTROL_WAVELENGTHS),
        "cube_products": {
            "science": str(cube["science_path"]), "dq": str(cube["dq_path"]),
            "ncontrib": str(cube["ncontrib_path"]),
        },
        "cube_robust_center": robust_cube["center"],
        "cube_robust_scale": robust_cube["scale"],
        "regions": [{key: value for key, value in region.items() if key != "mask"}
                    for region in regions],
        "calibration": {
            "reused_production_builder": True,
            "production_model": "make_m101_cube_current_model._calibrate_h5_current",
            "two_pass_exposure_scale_csv": str(paths["exposure_scale"])
            if paths["exposure_scale"] is not None and paths["exposure_scale"].exists()
            else None,
            "no_calibration_fit": True,
            "no_cube_reconstruction": True,
        },
        "decision_support": decision,
        "outputs": {name: str(output_dir / filename)
                    for name, filename in OUTPUT_NAMES.items()},
    }
    (output_dir / OUTPUT_NAMES["json"]).write_text(
        json.dumps(_jsonable(result), indent=2, sort_keys=True) + "\n")
    _write_summary_text(output_dir / OUTPUT_NAMES["txt"], result)
    say("writing outputs")
    for path in (output_dir / name for name in OUTPUT_NAMES.values()):
        print("[trace] output: %s" % path, flush=True)
    print("[trace] top candidates:", flush=True)
    for item in decision["top_candidates"]:
        print("[trace]   %s" % item["group"], flush=True)
    print("[trace] repeated physical amplifier: %s" %
          decision["one_physical_amplifier_appears_repeatedly"], flush=True)
    print("[trace] no reject CSV was created; no registry or production product was modified", flush=True)


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        main()
