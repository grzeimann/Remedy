#!/usr/bin/env python3
"""Read-only verification of the completed M101 current-model rebuild.

The default invocation is intended to run from the TACC production directory.
It reads the rebuild summary and FITS products, reuses the production cube
builder's calibration/ADR/Gaussian helpers, and writes diagnostic products
under ``m101_final_rebuild_check``.  It never writes a rejection CSV, H5 file,
FITS product, or reconstructed cube.

Use ``--self-test`` locally for the data-free checks.  The default mode must
only be run where the production H5 files and cube products are available.
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
import warnings

import numpy as np


PREFIX = "m101_virus_v2_rebuild"
SUMMARY_NAME = PREFIX + "_current_model_summary.json"
CHECK_DIR_NAME = "m101_final_rebuild_check"
WHOLE_GROUP_EXPECTED = 21
SPECTRAL_ENTRY_EXPECTED = 297
WHOLE_GROUP_POSITIONS = (
    (759, 1007), (780, 960), (745, 915), (14, 280),
    (1121, 465), (1084, 449), (1071, 150),
)
POSITION_WAVELENGTHS = (3800, 4200, 4600, 4882, 5200, 5450)
TRACE_POSITION = (637, 701)
TRACE_WAVELENGTHS = tuple(range(4868, 4898, 2))
REJECTION_WAVELENGTHS = tuple(range(4874, 4891, 2))
GROUP_FIELDS = ("H5", "exposure", "SPECID", "IFUSLOT", "IFUID", "AMP")

# These are deliberately explicit and are printed in the report.  They are
# diagnostic thresholds, not masking rules.
SUPPORT_RATIO_THRESHOLD = 0.75
EXTREME_ROBUST_Z_THRESHOLD = 10.0
EXTREME_ABS_SCI_THRESHOLD = 10.0
NONNEGLIGIBLE_WEIGHT_FRACTION = 1.0e-4
AMBIGUOUS_PHYSICAL_FRACTION = 0.10


def _jsonable(value):
    """Convert NumPy/path values and non-finite floats for JSON output."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = value.item()
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _wave_index(wave, wavelength):
    wave = np.asarray(wave, dtype=float)
    index = int(np.argmin(np.abs(wave - float(wavelength))))
    if not np.isclose(wave[index], float(wavelength), rtol=0.0, atol=1.0e-8):
        raise ValueError("%.12g A does not map to DEF_WAVE" % wavelength)
    return index


def _ds9_to_numpy(x, y):
    """Convert 1-based DS9/FITS coordinates to NumPy (iy, ix)."""
    ix = int(x) - 1
    iy = int(y) - 1
    return iy, ix


def _finite(values):
    values = np.asarray(values, dtype=float)
    return values[np.isfinite(values)]


def _robust_center_scale(values):
    values = _finite(values)
    if values.size == 0:
        return np.nan, np.nan
    center = float(np.median(values))
    scale = 1.4826 * float(np.median(np.abs(values - center)))
    if not np.isfinite(scale) or scale <= 0.0:
        scale = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
    return center, scale


def _summary_value(summary, *keys):
    value = summary
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    if isinstance(value, dict):
        return value.get("path", value.get("full_path"))
    return value


def _resolve_product(value, production_dir, source_dir):
    if value is None:
        return None
    path = Path(str(value)).expanduser()
    candidates = []
    if path.is_absolute():
        candidates.append(path)
        candidates.extend((production_dir / path.name, source_dir / path.name))
    else:
        candidates.extend((production_dir / path, source_dir / path, path))
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0].resolve() if candidates else path.resolve()


def _find_summary(production_dir, explicit=None):
    if explicit is not None:
        path = Path(explicit).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError("summary does not exist: %s" % path)
        return path
    preferred = production_dir / SUMMARY_NAME
    if preferred.exists():
        return preferred.resolve()
    candidates = sorted(production_dir.glob("*current_model_summary.json"))
    for path in candidates:
        try:
            data = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        if data.get("input_h5_files") or data.get("input_h5_glob"):
            return path.resolve()
    raise FileNotFoundError(
        "could not find %s or another current-model summary in %s" %
        (SUMMARY_NAME, production_dir))


def _discover_h5files(summary, production_dir):
    paths = []
    for value in summary.get("input_h5_files", []):
        path = Path(str(value)).expanduser()
        if path.exists():
            paths.append(path.resolve())
        elif (production_dir / path.name).exists():
            paths.append((production_dir / path.name).resolve())
    if not paths and summary.get("input_h5_glob"):
        paths = [Path(path).resolve()
                 for path in glob.glob(str(summary["input_h5_glob"]))]
    paths = sorted({path for path in paths if path.is_file()})
    if not paths:
        raise FileNotFoundError("no production H5 files found from summary")
    if any(path.name == "20200523_0000023.h5" for path in paths):
        raise RuntimeError("summary includes explicitly excluded H5 20200523_0000023.h5")
    return paths


def _load_builder(source_dir):
    path = source_dir / "make_m101_cube_current_model.py"
    if not path.exists():
        raise FileNotFoundError("production builder is unavailable: %s" % path)
    spec = importlib.util.spec_from_file_location("m101_rebuild_builder", path)
    if spec is None or spec.loader is None:
        raise ImportError("could not load production builder: %s" % path)
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(source_dir))
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _read_csv_rows(path):
    with Path(path).open(newline="") as stream:
        return list(csv.DictReader(stream))


def _model_paths(summary, production_dir, source_dir):
    names = (
        "state", "band_cache", "alpha_fits", "alpha_history", "amp_means",
        "fq_template", "blank_file",
    )
    paths = {}
    for name in names:
        value = _summary_value(summary, "provenance", name)
        paths[name] = _resolve_product(value, production_dir, source_dir)
    scale = _summary_value(summary, "exposure_scale", "exposure_scale_csv")
    if scale is None:
        scale = production_dir / (PREFIX + "_exposure_scale.csv")
    paths["exposure_scale"] = _resolve_product(scale, production_dir, source_dir)
    paths["qa_reject"] = _resolve_product(
        _summary_value(summary, "qa", "reject_file"), production_dir, source_dir)
    if paths["qa_reject"] is None or not paths["qa_reject"].exists():
        paths["qa_reject"] = production_dir / "m101_final_qa_reject.csv"
    paths["spectral_reject"] = _resolve_product(
        _summary_value(summary, "spectral_rejection", "spectral_reject_file"),
        production_dir, source_dir)
    if paths["spectral_reject"] is None or not paths["spectral_reject"].exists():
        paths["spectral_reject"] = production_dir / "m101_final_spectral_reject.csv"
    return paths


def _load_scale_rows(path, h5files):
    if path is None or not path.exists():
        raise FileNotFoundError("production exposure-scale CSV is unavailable: %s" % path)
    result = {}
    for row in _read_csv_rows(path):
        try:
            key = (Path(row["H5"]).name, int(row["exposure"]))
            g1 = float(row["g1_rel"])
            g2 = float(row["g2_rel"])
            cumulative = float(row.get("cumulative_relative_scale", g1 * g2))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("malformed exposure-scale row: %s" % exc) from exc
        if not np.all(np.isfinite((g1, g2, cumulative))) or cumulative == 0.0:
            raise ValueError("non-finite exposure scale for %s" % (key,))
        result[key] = {"g1_rel": g1, "g2_rel": g2,
                       "cumulative_relative_scale": cumulative}
    required = {(path.name, exposure) for path in h5files for exposure in (1, 2, 3)}
    missing = sorted(required - set(result))
    if missing:
        raise ValueError("exposure-scale CSV lacks rows: %s" % missing)
    return result


def _load_model(builder, paths, h5files):
    required = ("state", "band_cache", "alpha_fits", "alpha_history",
                "amp_means", "fq_template", "blank_file")
    for name in required:
        if paths[name] is None or not paths[name].exists():
            raise FileNotFoundError("required production input %s is unavailable: %s" %
                                    (name, paths[name]))
    state, response, composition, state_path = builder._load_frozen_response(paths["state"])
    identity_map, cache_manifest, cache_manifest_path, cache_path = builder._load_identity_map(
        paths["band_cache"], h5files)
    alpha_direct, alpha_path = builder._load_alpha_fits(paths["alpha_fits"])
    alpha_history, alpha_summary_path = builder._load_alpha_history(paths["alpha_history"])
    amp_means, mean_mask, mean_scope, amp_means_path = builder._load_amp_means(paths["amp_means"])
    fq, f0, fq_path = builder._load_fq(paths["fq_template"])
    blank_masks, blank_provenance = builder.frozen_parent.m101_blank_fibers.load(
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


def _make_records(builder, model, h5files, scale_rows):
    records = []
    for h5file in h5files:
        result = builder._calibrate_h5_current(
            h5file, model["identity_map"], model["response"],
            model["alpha_direct"], model["alpha_history"], model["amp_means"],
            model["fq"], model["f0"], model["blank_masks"][h5file.name],
            builder.M101_SKY_MIN_FIBERS)
        (spectra, errors, ra, dec, labels, surveys, calibration_summary,
         sky_records, sky_by_exposure, groups, q_rows, hardware_bad,
         corrected_total) = result
        records.append({
            "h5file": h5file,
            "spectra": spectra.astype(np.float32),
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
            "calibration_summary": calibration_summary,
        })

    # Apply the exact two recorded exposure-scale passes used by the builder.
    # The helper expects a map keyed by (H5 basename, exposure), which is the
    # same key shape returned by _load_scale_rows.
    builder._apply_exposure_relative_scales(records, scale_rows, "g1_rel")
    builder._apply_exposure_relative_scales(records, scale_rows, "g2_rel")
    return records


def _apply_qa_rejects(builder, records, path):
    keys, _ = builder._load_qa_reject_file(path)
    found = set()
    rejected_rows = 0
    for record in records:
        for group in record["groups"]:
            key = builder._qa_group_key(record["h5file"].name, group)
            if key not in keys:
                continue
            found.add(key)
            indices = np.asarray(group["indices"], dtype=int)
            record["spectra"][indices] = np.nan
            record["errors"][indices] = np.nan
            rejected_rows += int(indices.size)
    missing = keys - found
    if missing:
        raise ValueError("QA rejection file contains unmatched groups: %s" % sorted(missing))
    return {"file": str(path), "groups": int(len(found)),
            "fiber_rows": int(rejected_rows)}


def _apply_spectral_rejects(builder, records, path):
    entries, _ = builder._load_spectral_reject_file(path)
    applied = builder._apply_spectral_rejection(records, entries)
    return {"file": str(path), **applied}


def _load_cube_products(production_dir, wave):
    from astropy.io import fits

    paths = {
        "science": production_dir / (PREFIX + "_cube.fits"),
        "dq": production_dir / (PREFIX + "_dq_cube.fits"),
        "ncontrib": production_dir / (PREFIX + "_ncontrib_cube.fits"),
        "weight": production_dir / (PREFIX + "_weight_cube.fits"),
    }
    for path in paths.values():
        if not path.exists():
            raise FileNotFoundError("required rebuild product is unavailable: %s" % path)
    with fits.open(paths["science"], memmap=True) as hdul:
        science = np.asarray(hdul[0].data, dtype=np.float32)
        header = hdul[0].header.copy()
    with fits.open(paths["dq"], memmap=True) as hdul:
        dq = np.asarray(hdul[0].data)
    with fits.open(paths["ncontrib"], memmap=True) as hdul:
        ncontrib = np.asarray(hdul[0].data)
    with fits.open(paths["weight"], memmap=True) as hdul:
        weight = np.asarray(hdul[0].data)
    for name, data in (("dq", dq), ("ncontrib", ncontrib), ("weight", weight)):
        if data.shape != science.shape:
            raise ValueError("%s shape %s differs from SCI shape %s" %
                             (name, data.shape, science.shape))
    if science.ndim != 3:
        raise ValueError("SCI cube is not three-dimensional: %s" % (science.shape,))
    return {"paths": paths, "science": science, "dq": dq,
            "ncontrib": ncontrib, "weight": weight, "header": header,
            "wave": np.asarray(wave, dtype=float)}


def _position_rows(cube):
    rows = []
    science = cube["science"]
    dq = cube["dq"]
    ncontrib = cube["ncontrib"]
    weight = cube["weight"]
    wave = cube["wave"]
    ny, nx = science.shape[1:]
    for x, y in WHOLE_GROUP_POSITIONS:
        iy, ix = _ds9_to_numpy(x, y)
        if not (0 <= ix < nx and 0 <= iy < ny):
            raise IndexError("DS9 position (%d,%d) is outside cube %dx%d" %
                             (x, y, nx, ny))
        for requested in POSITION_WAVELENGTHS:
            index = _wave_index(wave, requested)
            plane_sci = science[index]
            center, scale = _robust_center_scale(plane_sci)
            threshold_scale = max(1.4826 * (scale if np.isfinite(scale) else 0.0), 1.0e-6)
            sci = float(science[index, iy, ix])
            finite = np.isfinite(sci)
            robust_z = (abs(sci - center) / threshold_scale
                        if finite and np.isfinite(center) else np.nan)
            rows.append({
                "DS9_X": int(x), "DS9_Y": int(y), "ix": int(ix), "iy": int(iy),
                "wavelength_A": float(wave[index]), "plane": int(index),
                "SCI": sci, "NCONTRIB": int(ncontrib[index, iy, ix]),
                "DQ": int(dq[index, iy, ix]),
                "WEIGHT": float(weight[index, iy, ix]),
                "plane_median_SCI": center, "plane_robust_z": robust_z,
            })

    by_position = {}
    for row in rows:
        by_position.setdefault((row["DS9_X"], row["DS9_Y"]), []).append(row)
    summaries = []
    for position, position_rows in by_position.items():
        states = []
        for row in position_rows:
            nbad = row["NCONTRIB"] < 2 or (row["DQ"] & 1) != 0
            if nbad:
                states.append("INSUFFICIENT_SUPPORT")
            elif (np.isfinite(row["plane_robust_z"]) and
                  row["plane_robust_z"] >= EXTREME_ROBUST_Z_THRESHOLD and
                  abs(row["SCI"]) >= EXTREME_ABS_SCI_THRESHOLD):
                states.append("EXTREME_SURVIVING_FLUX")
            else:
                plane_n = cube["ncontrib"][row["plane"]]
                plane_w = cube["weight"][row["plane"]]
                nmed = float(np.nanmedian(plane_n[np.isfinite(plane_n)]))
                wfinite = plane_w[np.isfinite(plane_w) & (plane_w > 0)]
                wmed = float(np.median(wfinite)) if wfinite.size else np.nan
                reduced = (row["NCONTRIB"] < SUPPORT_RATIO_THRESHOLD * nmed or
                           (np.isfinite(wmed) and
                            row["WEIGHT"] < SUPPORT_RATIO_THRESHOLD * wmed))
                states.append("REDUCED_SUPPORT" if reduced else "VALID_FLUX")
        if "EXTREME_SURVIVING_FLUX" in states:
            overall = "EXTREME_SURVIVING_FLUX"
        elif "INSUFFICIENT_SUPPORT" in states:
            overall = "INSUFFICIENT_SUPPORT"
        elif "REDUCED_SUPPORT" in states:
            overall = "REDUCED_SUPPORT"
        else:
            overall = "VALID_FLUX"
        summaries.append({"DS9_X": position[0], "DS9_Y": position[1],
                          "classification": overall, "sample_states": states})
    return rows, summaries


def _build_trace_geometry(builder, summary, cube, wave_index):
    image = summary.get("image_center_size", {})
    try:
        ra = float(image["RA"])
        dec = float(image["Dec"])
        size_arcmin = float(image["size_arcsec"]) / 60.0
        pixel_scale = float(image["pixel_scale_arcsec"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("summary lacks production image geometry") from exc
    _, _, _, n, xg, yg, xgrid, ygrid, tp = builder._parse_image_geometry(
        "%r,%r,%r" % (ra, dec, size_arcmin), pixel_scale)
    if cube["science"].shape[1:] != (n, n):
        raise ValueError("summary geometry gives %dx%d but cube is %s" %
                         (n, n, cube["science"].shape[1:]))
    sigma = (builder.GAUSSIAN_FWHM_ARCSEC / 2.35) / pixel_scale
    return tp, xg, yg, xgrid, ygrid, sigma, pixel_scale


def _trace_group_at_wave(builder, record, group, wave_index, tp, sigma,
                         pixel_scale, target_x, target_y):
    exposure = int(group["exposure"])
    selected = np.asarray(group["indices"], dtype=int)
    survey = record["surveys"][exposure]
    astrometry = builder.Astrometry(float(survey["ra"]), float(survey["dec"]),
                                    float(survey["pa"]), 0.0, 0.0)
    extractor = builder.Extract(wave=builder.DEF_WAVE)
    extractor.get_ADR_RAdec(astrometry)
    ra = (record["ra"][selected] - extractor.ADRra[wave_index] / 3600.0 /
          np.cos(np.deg2rad(float(survey["dec"]))))
    dec = record["dec"][selected] - extractor.ADRdec[wave_index] / 3600.0
    xpos, ypos = tp.wcs_world2pix(ra, dec, 1)
    xpos = np.asarray(xpos, dtype=float)
    ypos = np.asarray(ypos, dtype=float)
    ix_center = np.floor(xpos - 1.0).astype(int)
    iy_center = np.floor(ypos - 1.0).astype(int)
    in_kernel = ((np.abs((target_x - 1) - ix_center) <= 3) &
                 (np.abs((target_y - 1) - iy_center) <= 3))
    finite = np.isfinite(record["spectra"][selected, wave_index])
    use = in_kernel & finite & np.isfinite(xpos) & np.isfinite(ypos)
    if not np.any(use):
        return {"weight": 0.0, "weighted_contribution": 0.0,
                "fiber_count": 0, "shot_value": np.nan}
    weights = np.array([
        builder.production_gaussian_weight(x, y, target_x, target_y, sigma)
        for x, y in zip(xpos[use], ypos[use])
    ], dtype=float)
    values = np.asarray(record["spectra"][selected[use], wave_index], dtype=float)
    area = float(builder.FIBER_AREA)
    weighted = weights * values / area
    total_weight = float(np.sum(weights))
    weighted_contribution = float(np.sum(weighted))
    return {"weight": total_weight,
            "weighted_contribution": weighted_contribution,
            "fiber_count": int(np.count_nonzero(use)),
            "shot_value": (weighted_contribution / total_weight
                           if total_weight > 0.0 else np.nan)}


def _trace_artifact(builder, records, summary, cube, reporter):
    wave = cube["wave"]
    target_index = _wave_index(wave, 4882)
    tp, xg, yg, xgrid, ygrid, sigma, pixel_scale = _build_trace_geometry(
        builder, summary, cube, target_index)
    del xg, yg, xgrid, ygrid  # geometry is retained in tp exactly as in the builder
    all_groups = []
    for record in records:
        for group in record["groups"]:
            result = _trace_group_at_wave(
                builder, record, group, target_index, tp, sigma, pixel_scale,
                TRACE_POSITION[0], TRACE_POSITION[1])
            if result["weight"] > 0.0:
                all_groups.append({"record": record, "group": group, **result})
    total_weight = sum(row["weight"] for row in all_groups)
    for row in all_groups:
        row["weight_fraction"] = row["weight"] / total_weight if total_weight else np.nan
    candidates = [row for row in all_groups
                  if row["weight_fraction"] >= NONNEGLIGIBLE_WEIGHT_FRACTION]
    total_abs_contribution = sum(abs(row["weighted_contribution"]) for row in candidates)
    for row in candidates:
        group = row["group"]
        record = row["record"]
        key = builder._qa_group_key(record["h5file"].name, group)
        row["group_key"] = key
        row["physical_key"] = (int(group["specid"]), int(group["ifuslot"]),
                                int(group["ifuid"]), str(group["amp"]).upper())
        row["fraction_of_reconstructed_contribution"] = (
            row["weighted_contribution"] / total_abs_contribution
            if total_abs_contribution else np.nan)

    shot_sums = {}
    for row in all_groups:
        shot_key = (row["record"]["h5file"].name, int(row["group"]["exposure"]))
        state = shot_sums.setdefault(shot_key, {"weight": 0.0, "numerator": 0.0})
        state["weight"] += row["weight"]
        state["numerator"] += row["weighted_contribution"]
    shot_values = {
        "%s exposure %d" % key: state["numerator"] / state["weight"]
        for key, state in shot_sums.items() if state["weight"] > 0.0
    }
    reconstructed_median = (float(np.median(list(shot_values.values())))
                             if shot_values else np.nan)

    trace_rows = []
    for row in candidates:
        record, group = row["record"], row["group"]
        for wavelength in TRACE_WAVELENGTHS:
            index = _wave_index(wave, wavelength)
            contribution = _trace_group_at_wave(
                builder, record, group, index, tp, sigma, pixel_scale,
                TRACE_POSITION[0], TRACE_POSITION[1])
            value = contribution["shot_value"]
            trace_rows.append({
                "H5": record["h5file"].name, "exposure": int(group["exposure"]),
                "SPECID": int(group["specid"]), "IFUSLOT": int(group["ifuslot"]),
                "IFUID": int(group["ifuid"]), "AMP": str(group["amp"]).upper(),
                "wavelength_A": int(wavelength), "SCIENCE_VALUE": value,
                "gaussian_reconstruction_weight": contribution["weight"],
                "weighted_contribution": contribution["weighted_contribution"],
                "fiber_count": contribution["fiber_count"],
                "weight_fraction_all_groups": (
                    contribution["weight"] / total_weight if total_weight else np.nan),
                "fraction_of_reconstructed_contribution": (
                    contribution["weighted_contribution"] / total_abs_contribution
                    if total_abs_contribution else np.nan),
                "shot_value": value,
            })

    physical_totals = {}
    for row in candidates:
        physical_totals.setdefault(row["physical_key"], 0.0)
        physical_totals[row["physical_key"]] += abs(row["weighted_contribution"])
    max_total = max(physical_totals.values(), default=0.0)
    credible_physical = sorted(
        key for key, value in physical_totals.items()
        if value >= AMBIGUOUS_PHYSICAL_FRACTION * max_total)
    attribution_clear = len(credible_physical) <= 1 and bool(credible_physical)

    reporter("4882-A trace target: DS9 (%d,%d), NumPy (iy=%d, ix=%d); "
             "sigma=%.9g pixels, FWHM=%.3g arcsec, pixel scale=%.3g arcsec" %
             (TRACE_POSITION[0], TRACE_POSITION[1], TRACE_POSITION[1] - 1,
              TRACE_POSITION[0] - 1, sigma, builder.GAUSSIAN_FWHM_ARCSEC,
              pixel_scale))
    iy, ix = _ds9_to_numpy(*TRACE_POSITION)
    reporter("observed cube SCI=%.8g, reconstructed per-shot median=%.8g, "
             "shots with finite support=%d" %
             (cube["science"][target_index, iy, ix], reconstructed_median,
              len(shot_values)))
    reporter("Non-negligible candidate threshold: group Gaussian-weight fraction >= %.3g" %
             NONNEGLIGIBLE_WEIGHT_FRACTION)
    reporter("H5 exposure SPECID IFUSLOT IFUID AMP weight weighted_contribution "
             "weight_fraction contribution_fraction")
    for row in candidates:
        reporter("%s %d %d %d %d %s %.8g %.8g %.6g %.6g" % (
            row["record"]["h5file"].name, int(row["group"]["exposure"]),
            int(row["group"]["specid"]), int(row["group"]["ifuslot"]),
            int(row["group"]["ifuid"]), str(row["group"]["amp"]).upper(),
            row["weight"], row["weighted_contribution"], row["weight_fraction"],
            row["fraction_of_reconstructed_contribution"]))
    reporter("Candidate spectral inspection (same contributors):")
    reporter("H5 exposure SPECID IFUSLOT IFUID AMP wave SCIENCE_VALUE weight "
             "weighted_contribution")
    for row in trace_rows:
        reporter("%s %d %d %d %d %s %d %.8g %.8g %.8g" % (
            row["H5"], row["exposure"], row["SPECID"], row["IFUSLOT"],
            row["IFUID"], row["AMP"], row["wavelength_A"],
            row["SCIENCE_VALUE"], row["gaussian_reconstruction_weight"],
            row["weighted_contribution"]))

    control_waves = {4868, 4870, 4872, 4878, 4880, 4888, 4890, 4892, 4894, 4896}
    defect_signature = False
    for candidate in candidates:
        values = [row["SCIENCE_VALUE"] for row in trace_rows
                  if row["H5"] == candidate["record"]["h5file"].name and
                  row["exposure"] == int(candidate["group"]["exposure"]) and
                  row["SPECID"] == int(candidate["group"]["specid"]) and
                  row["IFUSLOT"] == int(candidate["group"]["ifuslot"]) and
                  row["IFUID"] == int(candidate["group"]["ifuid"]) and
                  row["AMP"] == str(candidate["group"]["amp"]).upper()]
        by_wave = {row["wavelength_A"]: row["SCIENCE_VALUE"] for row in trace_rows
                   if row["H5"] == candidate["record"]["h5file"].name and
                   row["exposure"] == int(candidate["group"]["exposure"]) and
                   row["SPECID"] == int(candidate["group"]["specid"]) and
                   row["IFUSLOT"] == int(candidate["group"]["ifuslot"]) and
                   row["IFUID"] == int(candidate["group"]["ifuid"]) and
                   row["AMP"] == str(candidate["group"]["amp"]).upper()}
        control = np.asarray([by_wave.get(item, np.nan) for item in control_waves])
        narrow = np.asarray([by_wave.get(item, np.nan)
                             for item in range(4874, 4887, 2)])
        ccenter, cscale = _robust_center_scale(control)
        peak = float(np.nanmax(np.abs(narrow - ccenter))) if np.any(np.isfinite(narrow)) else 0.0
        scale = max(1.4826 * (cscale if np.isfinite(cscale) else 0.0), 1.0)
        if peak >= 5.0 * scale:
            defect_signature = True
    result = {
        "target": {"DS9_X": TRACE_POSITION[0], "DS9_Y": TRACE_POSITION[1],
                   "wavelength_A": 4882, "plane": target_index,
                   "cube_SCI": cube["science"][target_index, iy, ix],
                   "reconstructed_per_shot_median": reconstructed_median,
                   "reconstructed_per_shot_values": shot_values},
        "geometry": {"pixel_scale_arcsec": pixel_scale,
                      "fwhm_arcsec": builder.GAUSSIAN_FWHM_ARCSEC,
                      "sigma_pixels": sigma,
                      "gaussian_support_radius_sigma": 2.0,
                      "kernel_radius_pixels": 3},
        "thresholds": {
            "nonnegligible_weight_fraction": NONNEGLIGIBLE_WEIGHT_FRACTION,
            "credible_physical_fraction_of_max": AMBIGUOUS_PHYSICAL_FRACTION,
        },
        "candidates_4882": [{
            "H5": row["record"]["h5file"].name,
            "exposure": int(row["group"]["exposure"]),
            "SPECID": int(row["group"]["specid"]),
            "IFUSLOT": int(row["group"]["ifuslot"]),
            "IFUID": int(row["group"]["ifuid"]),
            "AMP": str(row["group"]["amp"]).upper(),
            "gaussian_reconstruction_weight": row["weight"],
            "weighted_contribution": row["weighted_contribution"],
            "fraction_of_reconstructed_contribution": row[
                "fraction_of_reconstructed_contribution"],
        } for row in candidates],
        "physical_totals": {
            "%d/%d/%d/%s" % key: value
            for key, value in physical_totals.items()},
        "credible_physical_amplifiers": ["%d/%d/%d/%s" % key
                                          for key in credible_physical],
        "attribution_clear": attribution_clear,
        "defect_signature": defect_signature,
        "trace_rows": trace_rows,
    }
    return result, candidates, credible_physical


def _would_be_reject_rows(records, physical_keys):
    rows = []
    physical_keys = set(physical_keys)
    for record in records:
        for group in record["groups"]:
            physical = (int(group["specid"]), int(group["ifuslot"]),
                        int(group["ifuid"]), str(group["amp"]).upper())
            if physical not in physical_keys:
                continue
            for exposure in (1, 2, 3):
                if int(group["exposure"]) != exposure:
                    continue
                for wavelength in REJECTION_WAVELENGTHS:
                    rows.append({"H5": record["h5file"].name,
                                 "exposure": exposure,
                                 "SPECID": physical[0], "IFUSLOT": physical[1],
                                 "IFUID": physical[2], "AMP": physical[3],
                                 "wavelength_A": wavelength})
    unique = {(row["H5"], row["exposure"], row["SPECID"], row["IFUSLOT"],
               row["IFUID"], row["AMP"], row["wavelength_A"]): row
              for row in rows}
    return [unique[key] for key in sorted(unique)]


def _write_csv(path, rows):
    rows = list(rows)
    fields = list(rows[0]) if rows else list(GROUP_FIELDS) + ["wavelength_A"]
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


class _Report:
    def __init__(self):
        self.lines = []

    def __call__(self, message=""):
        message = str(message)
        self.lines.append(message)
        print(message, flush=True)


def _run_self_test(source_dir):
    # This test imports only the local builder module; it does not import data
    # products or open any TACC path.
    builder = _load_builder(source_dir)
    wave = np.asarray(builder.DEF_WAVE, dtype=float)
    expected = 3470.0 + 2.0 * np.arange(1036, dtype=float)
    assert wave.shape == expected.shape
    assert np.array_equal(wave, expected)
    assert _ds9_to_numpy(759, 1007) == (1006, 758)
    assert _ds9_to_numpy(637, 701) == (700, 636)
    assert _wave_index(wave, 4882) == 706
    assert wave[_wave_index(wave, 4882)] == 4882.0
    assert REJECTION_WAVELENGTHS == (4874, 4876, 4878, 4880, 4882,
                                     4884, 4886, 4888, 4890)
    assert len(REJECTION_WAVELENGTHS) == 9
    print("self-test: PASS")
    print("  DS9 (759,1007) -> NumPy (iy=1006, ix=758)")
    print("  DS9 (637,701) -> NumPy (iy=700, ix=636)")
    print("  DEF_WAVE[706] = %.1f A for 4882 A" % wave[706])
    print("  4874-4890 A expansion contains %d wavelengths" %
          len(REJECTION_WAVELENGTHS))


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--production-dir", type=Path, default=Path.cwd(),
                        help="directory containing rebuild products (default: cwd)")
    parser.add_argument("--source-dir", type=Path,
                        default=Path(__file__).resolve().parent,
                        help="directory containing make_m101_cube_current_model.py")
    parser.add_argument("--summary", type=Path, default=None,
                        help="override the current-model summary path")
    parser.add_argument("--self-test", action="store_true",
                        help="run data-free local checks and stop")
    return parser.parse_args()


def main():
    args = _parse_args()
    if args.self_test:
        _run_self_test(args.source_dir.expanduser().resolve())
        return

    production_dir = args.production_dir.expanduser().resolve()
    source_dir = args.source_dir.expanduser().resolve()
    check_dir = production_dir / CHECK_DIR_NAME
    check_dir.mkdir(parents=True, exist_ok=True)
    report = _Report()
    summary_path = _find_summary(production_dir, args.summary)
    summary = json.loads(summary_path.read_text())
    builder = _load_builder(source_dir)
    wave = np.asarray(builder.DEF_WAVE, dtype=float)
    if wave.ndim != 1 or wave.size != 1036:
        raise ValueError("unexpected DEF_WAVE shape: %s" % (wave.shape,))
    cube = _load_cube_products(production_dir, wave)

    qa_info = summary.get("qa", {}).get("rejection", {})
    spectral_info = summary.get("spectral_rejection", {})
    qa_file = _resolve_product(_summary_value(summary, "qa", "reject_file"),
                               production_dir, source_dir)
    spectral_file = _resolve_product(
        _summary_value(summary, "spectral_rejection", "spectral_reject_file"),
        production_dir, source_dir)
    if qa_file is None or not qa_file.exists():
        qa_file = production_dir / "m101_final_qa_reject.csv"
    if spectral_file is None or not spectral_file.exists():
        spectral_file = production_dir / "m101_final_spectral_reject.csv"

    report("M101 final rebuild artifact verification")
    report("summary: %s" % summary_path)
    report("products: %s" % ", ".join(str(path) for path in cube["paths"].values()))
    report("")
    report("1. Rejection provenance")
    report("QA reject file: %s" % qa_file)
    report("whole-group rejected groups: %s" % qa_info.get("rejected_groups", "MISSING"))
    report("whole-group rejected fiber rows: %s" %
           qa_info.get("rejected_fiber_rows", "MISSING"))
    report("spectral reject file: %s" % spectral_file)
    report("spectral group-wavelength entries: %s" %
           spectral_info.get("rejected_group_wavelength_entries", "MISSING"))
    report("spectral rejected fiber-wavelength cells: %s" %
           spectral_info.get("rejected_fiber_wavelength_cells", "MISSING"))
    if qa_info.get("rejected_groups") != WHOLE_GROUP_EXPECTED:
        report("WARNING: expected %d whole-group rejections, summary reports %s" %
               (WHOLE_GROUP_EXPECTED, qa_info.get("rejected_groups", "MISSING")))
    if spectral_info.get("rejected_group_wavelength_entries") != SPECTRAL_ENTRY_EXPECTED:
        report("WARNING: expected %d spectral entries, summary reports %s" %
               (SPECTRAL_ENTRY_EXPECTED,
                spectral_info.get("rejected_group_wavelength_entries", "MISSING")))

    position_rows, position_summaries = _position_rows(cube)
    report("")
    report("2. Seven whole-group artifact positions")
    report("Thresholds: support is valid at NCONTRIB >= 2 and DQ bit 0 clear; "
           "REDUCED_SUPPORT is < %.2f of same-plane median NCONTRIB/WEIGHT; "
           "EXTREME_SURVIVING_FLUX requires robust z >= %.1f and |SCI| >= %.3g." %
           (SUPPORT_RATIO_THRESHOLD, EXTREME_ROBUST_Z_THRESHOLD,
            EXTREME_ABS_SCI_THRESHOLD))
    report("DS9_X DS9_Y wavelength SCI NCONTRIB DQ WEIGHT")
    for row in position_rows:
        report("%d %d %.1f %.8g %d %d %.8g" %
               (row["DS9_X"], row["DS9_Y"], row["wavelength_A"], row["SCI"],
                row["NCONTRIB"], row["DQ"], row["WEIGHT"]))
    for item in position_summaries:
        report("position (%d,%d): %s" %
               (item["DS9_X"], item["DS9_Y"], item["classification"]))

    paths = _model_paths(summary, production_dir, source_dir)
    h5files = _discover_h5files(summary, production_dir)
    model = _load_model(builder, paths, h5files)
    scale_rows = _load_scale_rows(paths["exposure_scale"], h5files)
    records = _make_records(builder, model, h5files, scale_rows)
    applied_qa = _apply_qa_rejects(builder, records, qa_file)
    applied_spectral = _apply_spectral_rejects(builder, records, spectral_file)

    report("")
    report("3. Remaining 4882-A artifact trace")
    trace, candidates, credible_physical = _trace_artifact(
        builder, records, summary, cube, report)
    clear = bool(trace["attribution_clear"])
    recommend = clear and bool(trace["defect_signature"])
    if clear:
        physical_text = ", ".join(trace["credible_physical_amplifiers"])
        report("resolved H5 + physical amplifier attribution: %s" % physical_text)
    else:
        report("AMBIGUOUS: different physical amplifiers remain credible: %s" %
               ", ".join(trace["credible_physical_amplifiers"]))
    report("narrow-defect signature (large 4880-4886 excursion vs controls): %s" %
           trace["defect_signature"])
    report("recommend adding to 4874-4890 A spectral rejection window: %s" %
           ("YES" if recommend else "NO / NEEDS REVIEW"))

    would_be_rows = _would_be_reject_rows(records, credible_physical) if clear else []
    report("")
    report("4. Recommendation only")
    report("Were all 21 whole-group rejections recorded as applied? %s" %
           (qa_info.get("rejected_groups") == WHOLE_GROUP_EXPECTED))
    report("Were all 297 spectral rejection entries recorded as applied? %s" %
           (spectral_info.get("rejected_group_wavelength_entries") ==
            SPECTRAL_ENTRY_EXPECTED))
    report("Seven positions: %s" % ", ".join(
        "%d,%d=%s" % (item["DS9_X"], item["DS9_Y"], item["classification"])
        for item in position_summaries))
    report("What H5 + physical amplifier causes (637,701) at 4882 A? %s" %
           (", ".join(trace["credible_physical_amplifiers"])
            if clear else "ambiguous"))
    report("Should it be added to the existing window? %s" %
           ("YES" if recommend else "NO / NEEDS REVIEW"))
    report("Exact rows that WOULD be appended (not written):")
    if would_be_rows:
        report(",".join(GROUP_FIELDS) + ",wavelength_A")
        for row in would_be_rows:
            report(",".join(str(row[field]) for field in GROUP_FIELDS + ("wavelength_A",)))
    else:
        report("none; attribution is not clear")
    report("")
    report("No rejection CSV, FITS product, H5 file, or cube was modified.")

    trace_csv = check_dir / "m101_remaining_4882_trace.csv"
    _write_csv(trace_csv, trace["trace_rows"])
    result = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "script": str(Path(__file__).resolve()),
        "production_directory": str(production_dir),
        "summary": str(summary_path),
        "cube_products": {key: str(value) for key, value in cube["paths"].items()},
        "rejection_provenance": {
            "qa_file": str(qa_file), "summary": qa_info,
            "spectral_file": str(spectral_file), "spectral_summary": spectral_info,
            "qa_applied_readback": applied_qa,
            "spectral_applied_readback": applied_spectral,
            "whole_group_expected": WHOLE_GROUP_EXPECTED,
            "spectral_entries_expected": SPECTRAL_ENTRY_EXPECTED,
        },
        "whole_group_positions": {
            "thresholds": {
                "support_ratio": SUPPORT_RATIO_THRESHOLD,
                "extreme_robust_z": EXTREME_ROBUST_Z_THRESHOLD,
                "extreme_abs_SCI": EXTREME_ABS_SCI_THRESHOLD,
            },
            "rows": position_rows, "summaries": position_summaries,
        },
        "trace": trace,
        "recommendation": {
            "all_whole_group_rejections_recorded":
                qa_info.get("rejected_groups") == WHOLE_GROUP_EXPECTED,
            "all_spectral_entries_recorded":
                spectral_info.get("rejected_group_wavelength_entries") ==
                SPECTRAL_ENTRY_EXPECTED,
            "attribution_clear": clear,
            "recommend_add_to_window": recommend,
            "would_be_reject_rows": would_be_rows,
        },
        "outputs": {
            "txt": str(check_dir / "m101_final_rebuild_check.txt"),
            "json": str(check_dir / "m101_final_rebuild_check.json"),
            "trace_csv": str(trace_csv),
        },
    }
    (check_dir / "m101_final_rebuild_check.txt").write_text(
        "\n".join(report.lines) + "\n")
    (check_dir / "m101_final_rebuild_check.json").write_text(
        json.dumps(_jsonable(result), indent=2, sort_keys=True) + "\n")
    print("outputs:")
    print("  %s" % (check_dir / "m101_final_rebuild_check.txt"))
    print("  %s" % (check_dir / "m101_final_rebuild_check.json"))
    print("  %s" % trace_csv)


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        main()
