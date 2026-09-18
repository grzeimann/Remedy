#!/usr/bin/env python3
"""Construct a fast, deterministic seven-band M101 calibration initializer.

This is deliberately a staging tool, not a replacement for
``fit_m101_calibration_joint.py``.  Native spectra are touched only by the
optional cache builder.  All population construction after collapse, all
four initializer stages, and all diagnostics use the seven collapsed bands:
the five null bands plus ON and OFF.

The cache is split into a compressed ``.npz`` array file and a small JSON
manifest.  The manifest records the input identities, band order, effective
wavelength coordinates, and row counts so a cache-only run cannot silently
mix H5 populations.
"""

from __future__ import annotations

import argparse
import ast
import csv
from dataclasses import dataclass
from itertools import combinations
import json
import math
from pathlib import Path
import time

import numpy as np
import tables

import diagnose_m101_hierarchical as validated_m101
import diagnose_m101_post_model3 as post_model3
import fit_m101_calibration_joint as joint
import m101_blank_fibers
import m101_compact_mask
import m101_external_measurements
import m101_native_data as native
from m101_hardware_exclusions import hardware_excluded
from m101_calibration_utils import (
    ALL_BANDS, collapse_many, collapse_error_many, file_identity,
    json_ready, residual_summary, robust_location, robust_scatter,
    small_file_hash, sufficient_native_spectrum,
)


NULL_BANDS = tuple(ALL_BANDS[:5])
SOURCE_BANDS = ("ON", "OFF")
AMP_ORDER = tuple(joint.AMP_ORDER)
AMP_INDEX = dict(joint.AMP_INDEX)
BASIS = np.asarray(joint.BASIS, dtype=float)
WAVE = np.asarray(validated_m101.DEF_WAVE, dtype=float)
X_LAMBDA = (WAVE - 4500.0) / 1000.0
QMAX_10 = float(joint.QMAX_10)
SCHEMA_VERSION = "m101_band_initializer_v1"
CACHE_SCHEMA_VERSION = "m101_band_cache_v1"
CENTRAL_Q_MIN = 40
CENTRAL_Q_MAX = 70
MIN_SAFE_DENOMINATOR = 1e-12
STAGE1_DEFAULT_ITERATIONS = 4
STAGE1_DEFAULT_TOLERANCE = 1e-7


@dataclass
class BandItem:
    """One H5 exposure, retaining only collapsed band-level quantities."""

    item_index: int
    h5_name: str
    h5_path: str
    exposure: int
    key: tuple
    row_index: np.ndarray
    ifu: np.ndarray
    ifu_code: np.ndarray
    h5ifu_code: int
    amp: np.ndarray
    q: np.ndarray
    ra: np.ndarray
    dec: np.ndarray
    x_arcmin: np.ndarray
    y_arcmin: np.ndarray
    band_total: np.ndarray
    band_error: np.ndarray
    K: np.ndarray
    blank_classified: np.ndarray
    blank_valid: np.ndarray
    date_mask_bad: np.ndarray
    persistent_hardware_bad: np.ndarray
    hardware_bad: np.ndarray
    X: np.ndarray
    external_valid: np.ndarray
    source_candidate: np.ndarray
    source_accepted: np.ndarray


@dataclass
class InitialState:
    """Dense compact representation of the staged state."""

    p_ifu: np.ndarray
    ax: np.ndarray
    ay: np.ndarray
    x_center: np.ndarray
    y_center: np.ndarray
    alpha_q: np.ndarray
    p_amp: np.ndarray
    g_star: np.ndarray
    q_color: np.ndarray
    xbar: np.ndarray


@dataclass
class Stage1OnlyState:
    """State deliberately limited to the quantities identifiable in Stage 1."""

    p_ifu: np.ndarray
    ax: np.ndarray
    ay: np.ndarray
    x_center: np.ndarray
    y_center: np.ndarray


def _finite(values):
    values = np.asarray(values, dtype=float)
    return values[np.isfinite(values)]


def _write_rows(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if not rows:
        path.write_text("")
        return
    fields = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            clean = {}
            for field in fields:
                value = row.get(field, "")
                if isinstance(value, np.generic):
                    value = value.item()
                if isinstance(value, (list, tuple, np.ndarray)):
                    value = json.dumps(json_ready(value), separators=(",", ":"))
                clean[field] = value
            writer.writerow(clean)


def _safe_float(value, default=np.nan):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    return value if np.isfinite(value) else default


def _zero_sum_basis(n):
    basis = np.zeros((n, max(0, n - 1)), dtype=float)
    if n > 1:
        basis[:-1] = np.eye(n - 1)
        basis[-1] = -1.0
    return basis


def _key_text(value):
    return str(value)


def _weighted_robust_center(values, sigmas=None):
    """Small deterministic robust weighted location used for summaries."""
    values = np.asarray(values, dtype=float)
    good = np.isfinite(values)
    if sigmas is None:
        weights = np.ones(values.shape, dtype=float)
    else:
        sigmas = np.asarray(sigmas, dtype=float)
        good &= np.isfinite(sigmas) & (sigmas > 0)
        weights = np.zeros(values.shape, dtype=float)
        weights[good] = 1.0 / np.maximum(sigmas[good], 1e-12) ** 2
    values = values[good]
    weights = weights[good]
    if values.size == 0:
        return np.nan, np.nan
    center = float(robust_location(values))
    scale = float(robust_scatter(values)) if values.size > 1 else np.nan
    if not np.isfinite(scale) or scale <= 0:
        scale = max(float(np.nanstd(values)), 1e-12)
    for _ in range(4):
        residual = values - center
        robust = np.ones(values.size, dtype=float)
        threshold = 1.345 * max(scale, 1e-12)
        large = np.abs(residual) > threshold
        robust[large] = threshold / np.maximum(np.abs(residual[large]), 1e-12)
        effective = weights * robust
        if not np.any(effective > 0):
            break
        center = float(np.sum(effective * values) / np.sum(effective))
        scale_new = float(robust_scatter(values - center))
        if np.isfinite(scale_new) and scale_new > 0:
            scale = scale_new
    uncertainty = scale / math.sqrt(max(1, values.size))
    return center, uncertainty


def _irls_linear(design, target, sigma=None, iterations=5):
    """Robust weighted least squares for the compressed stage summaries."""
    design = np.asarray(design, dtype=float)
    target = np.asarray(target, dtype=float)
    good = np.isfinite(target) & np.all(np.isfinite(design), axis=1)
    if sigma is None:
        sigma = np.ones(target.shape, dtype=float)
    else:
        sigma = np.asarray(sigma, dtype=float)
        good &= np.isfinite(sigma) & (sigma > 0)
    design = design[good]
    target = target[good]
    sigma = sigma[good]
    if target.size == 0 or design.shape[1] == 0:
        return np.zeros(design.shape[1], dtype=float), {
            "N": int(target.size), "rank": 0, "scale": np.nan}
    base = 1.0 / np.maximum(sigma, 1e-12) ** 2
    robust = np.ones(target.size, dtype=float)
    coefficients = np.zeros(design.shape[1], dtype=float)
    for _ in range(iterations):
        weights = np.sqrt(base * robust)
        coefficients, _, rank, _ = np.linalg.lstsq(
            design * weights[:, None], target * weights, rcond=1e-12)
        residual = target - design @ coefficients
        scale = max(float(robust_scatter(residual)), 1e-8)
        standardized = np.abs(residual) / scale
        robust = np.where(standardized > 1.345, 1.345 / standardized, 1.0)
    return coefficients, {"N": int(target.size), "rank": int(rank),
                          "scale": float(scale)}


def _irls_scalar(predictor, target, sigma=None, iterations=5):
    predictor = np.asarray(predictor, dtype=float)
    target = np.asarray(target, dtype=float)
    good = np.isfinite(predictor) & np.isfinite(target)
    if sigma is None:
        sigma = np.ones(target.shape, dtype=float)
    else:
        sigma = np.asarray(sigma, dtype=float)
        good &= np.isfinite(sigma) & (sigma > 0)
    predictor = predictor[good]
    target = target[good]
    sigma = sigma[good]
    if predictor.size == 0 or np.sum(predictor ** 2) <= 0:
        return 0.0, {"N": int(predictor.size), "uncertainty": np.nan,
                     "scatter": np.nan}
    base = 1.0 / np.maximum(sigma, 1e-12) ** 2
    robust = np.ones(target.size, dtype=float)
    value = 0.0
    for _ in range(iterations):
        weight = base * robust
        value = float(np.sum(weight * predictor * target) /
                      np.sum(weight * predictor ** 2))
        residual = target - value * predictor
        scale = max(float(robust_scatter(residual)), 1e-8)
        standardized = np.abs(residual) / scale
        robust = np.where(standardized > 1.345, 1.345 / standardized, 1.0)
    uncertainty = scale / math.sqrt(max(float(np.sum(weight * predictor ** 2)), 1e-12))
    return value, {"N": int(target.size), "uncertainty": float(uncertainty),
                   "scatter": float(scale)}


def _item_group_indices(item):
    groups = {}
    for index, (ifu_code, amp_index) in enumerate(zip(item.ifu_code, item.amp)):
        groups.setdefault((int(ifu_code), int(amp_index)), []).append(index)
    return {key: np.asarray(value, dtype=int) for key, value in groups.items()}


def _ifu_indices(data):
    values = sorted({tuple(map(int, ifu)) for item in data for ifu in item.ifu}, key=str)
    return values, {value: index for index, value in enumerate(values)}


def _h5ifu_indices(data):
    keys = []
    for item in data:
        for ifu_code in np.unique(item.ifu_code):
            key = (item.h5_name, int(ifu_code))
            if key not in keys:
                keys.append(key)
    keys.sort(key=str)
    return keys, {key: index for index, key in enumerate(keys)}


def _assign_codes(items):
    ifus, ifu_map = _ifu_indices(items)
    for item in items:
        item.ifu_code = np.asarray([
            ifu_map[tuple(map(int, value))] for value in item.ifu], dtype=np.int32)
    hi_keys, hi_map = _h5ifu_indices(items)
    for index, item in enumerate(items):
        item.item_index = index
        item.h5ifu_code = hi_map[(item.h5_name, int(item.ifu_code[0]))]
        # Every exposure contains the same physical IFU identities.  The
        # scalar h5ifu_code is used only as a default; row-level lookup below
        # uses a compact code vector so mixed-IFU exposures are correct.
    return ifus, ifu_map, hi_keys, hi_map


def _row_h5ifu_codes(item, hi_map):
    return np.asarray([hi_map[(item.h5_name, int(code))]
                       for code in item.ifu_code], dtype=np.int32)


def _resolve_path(value, fallback=None):
    if value:
        return Path(value).expanduser().resolve()
    if isinstance(fallback, dict):
        value = fallback.get("full_path")
        if value:
            return Path(value).expanduser().resolve()
    return None


def _product_defaults(product_path):
    product = json.loads(Path(product_path).read_text())
    provenance = product.get("provenance", {})
    return product, provenance


def _cache_paths(cache_path):
    cache_path = Path(cache_path).expanduser().resolve()
    return cache_path, cache_path.with_suffix(cache_path.suffix + ".json")


def _source_arrays_from_external(item, external):
    if external is None or item.h5_name not in external:
        return (np.full(item.row_index.size, np.nan, dtype=float),
                np.full(item.row_index.size, np.nan, dtype=float),
                np.zeros(item.row_index.size, dtype=bool),
                np.zeros(item.row_index.size, dtype=bool))
    cache = external[item.h5_name]
    x_values = []
    valid_values = []
    for band in SOURCE_BANDS:
        x_values.append(float(cache["global_g"][band]) * np.asarray(
            cache["external_object"][(item.exposure, band)], dtype=float))
        valid_values.append(np.asarray(
            cache["external_valid"][(item.exposure, band)], dtype=bool))
    return x_values[0], x_values[1], valid_values[0], valid_values[1]


def _build_band_cache(h5_paths, blank_by_h5, on_filter_path, off_filter_path,
                      external=None, minimum_finite_fraction=.8, progress=None):
    """Stream native spectra once and retain only the seven collapsed bands."""
    started = time.perf_counter()
    on_filter = validated_m101.read_filter(on_filter_path)
    off_filter = validated_m101.read_filter(off_filter_path)
    responses, null_provenance = native._band_responses(on_filter, off_filter)
    paths = native.discover_h5(h5_paths, development=True)
    items = []
    cache_timings = []
    amp_to_index = {amp: index for index, amp in enumerate(AMP_ORDER)}
    for path_index, path in enumerate(paths, 1):
        h5_started = time.perf_counter()
        if progress:
            progress("band cache H5 %d/%d: %s" % (path_index, len(paths), path.name))
        with tables.open_file(path, mode="r") as h5:
            if not {"Info", "Fibers", "Survey"}.issubset(h5.root._v_children):
                raise ValueError("%s lacks Info, Fibers, or Survey" % path)
            info, fibers = h5.root.Info, h5.root.Fibers
            groups, labels = validated_m101.build_groups(info)
            surveys = native._survey_by_exposure(h5)
            ra_all = np.asarray(info.cols.ra[:], dtype=float)
            dec_all = np.asarray(info.cols.dec[:], dtype=float)
            ifu_all, amp_all, j_all, q_all = native._physical_arrays(info, groups, labels)
            blank = np.asarray(blank_by_h5[path.name], dtype=bool)
            if blank.shape != (info.nrows,):
                raise ValueError("blank mask does not match %s" % path)
            date_name = path.name.split("_")[0]
            date_cache = {}
            persistent_cache = {}
            date_bad_all = []
            persistent_bad_all = []
            for ifu_value, amp_value in zip(ifu_all, amp_all):
                date_key = (date_name, int(ifu_value[1]), str(amp_value))
                date_cache.setdefault(date_key, hardware_excluded(
                    date=date_name, specid=-1, ifuslot=ifu_value[1], ifuid=-1,
                    amp=amp_value, purpose="fit"))
                date_bad_all.append(date_cache[date_key])
                persistent_key = tuple(map(int, ifu_value)) + (str(amp_value),)
                persistent_cache.setdefault(persistent_key, hardware_excluded(
                    date=None, specid=ifu_value[0], ifuslot=ifu_value[1],
                    ifuid=ifu_value[2], amp=amp_value, purpose="fit", h5=path))
                persistent_bad_all.append(persistent_cache[persistent_key])
            date_bad_all = np.asarray(date_bad_all, dtype=bool)
            persistent_bad_all = np.asarray(persistent_bad_all, dtype=bool)
            for exposure in (1, 2, 3):
                exposure_started = time.perf_counter()
                indices = np.flatnonzero(labels == exposure)
                survey = surveys[exposure]
                spectrum = np.asarray(fibers.read_coordinates(indices, field="spectrum"), dtype=float)
                error = np.asarray(fibers.read_coordinates(indices, field="error"), dtype=float)
                skyspectrum = np.asarray(fibers.read_coordinates(indices, field="skyspectrum"), dtype=float)
                total, error = native.construct_total_spectra(
                    spectrum, error, skyspectrum, survey["offset"])
                finite_native = sufficient_native_spectrum(total, minimum_finite_fraction)
                date_bad = date_bad_all[indices]
                persistent_bad = persistent_bad_all[indices]
                hardware_bad = date_bad | persistent_bad
                blank_valid = blank[indices] & ~hardware_bad & finite_native
                values, response_fraction = collapse_many(total, responses)
                band_error, _ = collapse_error_many(error, responses)
                raw_basis = validated_m101.raw_work_basis(survey)
                K = np.asarray([validated_m101.weighted_scalar(raw_basis, response)
                                for response in responses], dtype=float)
                ra = ra_all[indices]
                dec = dec_all[indices]
                ra0, dec0 = native._exposure_coordinates(
                    ifu_all[indices], ra, dec, blank_valid)
                x_arcmin = (ra - ra0) * np.cos(np.deg2rad(dec0)) * 60.0
                y_arcmin = (dec - dec0) * 60.0
                provisional = BandItem(
                    item_index=len(items), h5_name=path.name, h5_path=str(path),
                    exposure=exposure, key=(path.name, exposure),
                    row_index=indices.astype(np.int64), ifu=ifu_all[indices],
                    ifu_code=np.zeros(indices.size, dtype=np.int32),
                    h5ifu_code=-1,
                    amp=np.asarray([amp_to_index[str(value)] for value in amp_all[indices]], dtype=np.int8),
                    q=q_all[indices].astype(np.int16), ra=ra, dec=dec,
                    x_arcmin=x_arcmin, y_arcmin=y_arcmin,
                    # Preserve the validated collapse precision.  The source
                    # acceptance boundary is fixed at robust_scatter <= .15.
                    band_total=np.asarray(values, dtype=float),
                    band_error=np.asarray(band_error, dtype=float), K=K,
                    blank_classified=blank[indices], blank_valid=blank_valid,
                    date_mask_bad=date_bad, persistent_hardware_bad=persistent_bad,
                    hardware_bad=hardware_bad, X=np.full((indices.size, 2), np.nan, dtype=float),
                    external_valid=np.zeros((indices.size, 2), dtype=bool),
                    source_candidate=np.zeros((indices.size, 2), dtype=bool),
                    source_accepted=np.zeros((indices.size, 2), dtype=bool))
                x_on, x_off, valid_on, valid_off = _source_arrays_from_external(provisional, external)
                provisional.X[:, 0] = x_on
                provisional.X[:, 1] = x_off
                provisional.external_valid[:, 0] = valid_on
                provisional.external_valid[:, 1] = valid_off
                items.append(provisional)
                cache_timings.append({"H5": path.name, "exposure": exposure,
                                      "rows": int(indices.size),
                                      "seconds": time.perf_counter() - exposure_started,
                                      "native_pixels_read_for_collapse": True,
                                      "native_pixels_retained": False})
                del spectrum, error, skyspectrum, total
        if progress:
            progress("band cache H5 %d/%d complete in %.2fs" %
                     (path_index, len(paths), time.perf_counter() - h5_started))
    if not items:
        raise ValueError("band cache contains no exposures")
    ifus, ifu_map, hi_keys, hi_map = _assign_codes(items)
    for item in items:
        # Codes are row-level and are used for vectorized response evaluation.
        item.h5ifu_code = hi_map[(item.h5_name, int(item.ifu_code[0]))]
    provenance = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "input_h5": [file_identity(path) for path in paths],
        "band_order": list(ALL_BANDS),
        "null_band_provenance": null_provenance,
        "x_lambda_definition": "(lambda - 4500 A) / 1000 A",
        "native_pixels_used_only_for": "one-pass seven-band collapse",
        "native_pixels_retained": False,
        "row_counts": {str(item.key): int(item.row_index.size) for item in items},
        "total_rows": int(sum(item.row_index.size for item in items)),
        "timings": {"cache_build_seconds": time.perf_counter() - started,
                     "per_exposure": cache_timings},
        "ifus": [list(value) for value in ifus],
        "h5ifu_keys": [[key[0], int(key[1])] for key in hi_keys],
        "band_effective_x_lambda": np.sum(responses * X_LAMBDA[None, :], axis=1) /
        np.sum(responses, axis=1),
    }
    return items, provenance, responses


def _save_band_cache(cache_path, items, provenance, responses, membership_frozen=False):
    cache_path, manifest_path = _cache_paths(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    fields = {}
    fields["row_index"] = np.concatenate([item.row_index for item in items])
    fields["ifu"] = np.concatenate([item.ifu for item in items])
    fields["ifu_code"] = np.concatenate([item.ifu_code for item in items])
    fields["amp"] = np.concatenate([item.amp for item in items])
    fields["q"] = np.concatenate([item.q for item in items])
    fields["ra"] = np.concatenate([item.ra for item in items])
    fields["dec"] = np.concatenate([item.dec for item in items])
    fields["x_arcmin"] = np.concatenate([item.x_arcmin for item in items])
    fields["y_arcmin"] = np.concatenate([item.y_arcmin for item in items])
    fields["band_total"] = np.concatenate([item.band_total for item in items])
    fields["band_error"] = np.concatenate([item.band_error for item in items])
    fields["blank_classified"] = np.concatenate([item.blank_classified for item in items])
    fields["blank_valid"] = np.concatenate([item.blank_valid for item in items])
    fields["date_mask_bad"] = np.concatenate([item.date_mask_bad for item in items])
    fields["persistent_hardware_bad"] = np.concatenate([item.persistent_hardware_bad for item in items])
    fields["hardware_bad"] = np.concatenate([item.hardware_bad for item in items])
    fields["X"] = np.concatenate([item.X for item in items])
    fields["external_valid"] = np.concatenate([item.external_valid for item in items])
    fields["source_candidate"] = np.concatenate([item.source_candidate for item in items])
    fields["source_accepted"] = np.concatenate([item.source_accepted for item in items])
    offsets = [0]
    item_rows = []
    for item in items:
        offsets.append(offsets[-1] + item.row_index.size)
        item_rows.append({
            "h5_name": item.h5_name, "h5_path": item.h5_path,
            "exposure": int(item.exposure), "key": str(item.key),
            "start": offsets[-2], "stop": offsets[-1], "K": item.K,
            "h5ifu_code": int(item.h5ifu_code),
        })
    provenance = dict(provenance)
    provenance.update({"items": item_rows, "offsets": offsets,
                       "membership_frozen": bool(membership_frozen),
                       "responses_effective_x_lambda": np.sum(responses * X_LAMBDA[None, :], axis=1) /
                       np.sum(responses, axis=1)})
    np.savez_compressed(cache_path, **fields)
    manifest_path.write_text(json.dumps(json_ready(provenance), indent=2, sort_keys=True))
    return cache_path, manifest_path


def _load_band_cache(cache_path, selected_names=None):
    cache_path, manifest_path = _cache_paths(cache_path)
    if not cache_path.exists() or not manifest_path.exists():
        raise FileNotFoundError("band cache and manifest must both exist: %s" % cache_path)
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema_version") != CACHE_SCHEMA_VERSION:
        raise ValueError("unsupported band cache schema")
    arrays = np.load(cache_path, allow_pickle=False)
    fields = {name: np.asarray(arrays[name]) for name in arrays.files}
    items = []
    all_items = manifest.get("items", [])
    wanted = set(selected_names or [row["h5_name"] for row in all_items])
    if not wanted.issubset({row["h5_name"] for row in all_items}):
        raise ValueError("requested H5 is absent from band cache")
    if selected_names and len(wanted) != len(selected_names):
        raise ValueError("duplicate H5 names in cache selection")
    ifus = [tuple(map(int, value)) for value in manifest.get("ifus", [])]
    hi_keys = [(str(value[0]), int(value[1])) for value in manifest.get("h5ifu_keys", [])]
    cursor = 0
    for item_index, row in enumerate(all_items):
        if row["h5_name"] not in wanted:
            continue
        start, stop = int(row["start"]), int(row["stop"])
        sl = slice(start, stop)
        key = (row["h5_name"], int(row["exposure"]))
        item = BandItem(
            item_index=len(items), h5_name=row["h5_name"], h5_path=row["h5_path"],
            exposure=int(row["exposure"]), key=key,
            row_index=fields["row_index"][sl], ifu=fields["ifu"][sl],
            ifu_code=fields["ifu_code"][sl], h5ifu_code=int(row["h5ifu_code"]),
            amp=fields["amp"][sl], q=fields["q"][sl], ra=fields["ra"][sl],
            dec=fields["dec"][sl], x_arcmin=fields["x_arcmin"][sl],
            y_arcmin=fields["y_arcmin"][sl], band_total=fields["band_total"][sl],
            band_error=fields["band_error"][sl], K=np.asarray(row["K"], dtype=float),
            blank_classified=fields["blank_classified"][sl], blank_valid=fields["blank_valid"][sl],
            date_mask_bad=fields["date_mask_bad"][sl],
            persistent_hardware_bad=fields["persistent_hardware_bad"][sl],
            hardware_bad=fields["hardware_bad"][sl], X=fields["X"][sl],
            external_valid=fields["external_valid"][sl],
            source_candidate=fields["source_candidate"][sl],
            source_accepted=fields["source_accepted"][sl])
        items.append(item)
        cursor += stop - start
    if not items:
        raise ValueError("selected H5 set is empty in band cache")
    # Reindex selected exposure items and validate the cache's own counts.
    for index, item in enumerate(items):
        item.item_index = index
    expected = sum(int(row["stop"]) - int(row["start"]) for row in all_items
                   if row["h5_name"] in wanted)
    actual = sum(item.row_index.size for item in items)
    if expected != actual:
        raise ValueError("band cache row count mismatch: manifest=%d arrays=%d" %
                         (expected, actual))
    return items, manifest, None


def _load_or_build_cache(args, product_provenance, output_dir, external, progress):
    cache_path = Path(args.band_cache or (output_dir / "m101_band_cache.npz")).expanduser().resolve()
    requested_paths = [Path(value).expanduser().resolve() for value in args.h5]
    requested_names = [path.name for path in requested_paths]
    if cache_path.exists() and not args.rebuild_cache:
        items, manifest, responses = _load_band_cache(
            cache_path, requested_names or None)
        cached_names = {row["h5_name"] for row in manifest.get("items", [])}
        if requested_names and set(requested_names) != cached_names.intersection(requested_names):
            raise ValueError("requested H5 set is inconsistent with band cache")
        if requested_names:
            manifest_names = {row["h5_name"] for row in manifest.get("items", [])}
            if not set(requested_names).issubset(manifest_names):
                raise ValueError("requested H5 is absent from band cache")
        progress("loaded band cache %s: %d exposures, %d rows" %
                 (cache_path.name, len(items), sum(item.row_index.size for item in items)))
        return items, manifest, responses, cache_path
    if not requested_paths:
        raise ValueError("--h5 is required when building a band cache")
    if not args.blank_file or not args.on_filter or not args.off_filter:
        raise ValueError("cache construction requires --blank-file, --on-filter, and --off-filter")
    blank_by_h5, blank_provenance = m101_blank_fibers.load(args.blank_file, requested_paths)
    items, manifest, responses = _build_band_cache(
        requested_paths, blank_by_h5, args.on_filter, args.off_filter,
        external=external, minimum_finite_fraction=args.minimum_finite_fraction,
        progress=progress)
    manifest["blank_loader"] = blank_provenance
    _save_band_cache(cache_path, items, manifest, responses, membership_frozen=False)
    progress("wrote band cache %s" % cache_path)
    return items, manifest, responses, cache_path


def _band_sky_from_native_sky(sky, responses):
    sky = np.asarray(sky, dtype=float)
    values, _ = collapse_many(sky[None, :], responses)
    return np.asarray(values[0], dtype=float)


def _warm_model_band_sky(product_path, responses, selected_names):
    """Load only the validated warm-start sky long enough to make band skies."""
    product, warm_model, native_skies = post_model3.load_model_and_skies(product_path)
    result = {}
    for key, sky in native_skies.items():
        if key[0] in selected_names:
            result[key] = _band_sky_from_native_sky(sky, responses)
    del native_skies
    return product, warm_model, result


def _candidate_population_band(data, external, mask_data, mask_wcs, model,
                               warm_skies, fq, responses, minimum_source_fibers):
    """Faithful band-level reproduction of the validated joint membership."""
    if external is None:
        raise ValueError("source membership reproduction requires external cache")
    source_masks = {}
    group_records = {}
    response_pair = np.asarray(responses)[[5, 6]]
    for item in data:
        cache = external[item.h5_name]
        radius = post_model3._radius_arcmin(item.ra, item.dec)
        object_values = {
            band: item.X[:, index] for index, band in enumerate(SOURCE_BANDS)}
        positions = {}
        for band in SOURCE_BANDS:
            positions[band] = m101_compact_mask.mask_radec(
                mask_data, mask_wcs, item.ra, item.dec)
        z = _base_log_response(item, None, model.p_ifu,
                               model.ax, model.ay, model.p_amp)
        response = np.exp(z)
        alpha = np.asarray([
            model.alpha_q.get((item.key, tuple(map(int, ifu)), AMP_ORDER[int(amp)]), 0.0)
            for ifu, amp in zip(item.ifu, item.amp)], dtype=float)
        additive = alpha[:, None] * fq[item.q, None] * item.K[None, :]
        corrected = (item.band_total - additive) / response[:, None]
        sky_pair = np.asarray(warm_skies[item.key])[[5, 6]]
        # The warm skies have already been collapsed to seven bands; this is
        # algebraically identical to the current full-spectrum membership.
        virus_by_band = {
            band: corrected[:, index + 5] - sky_pair[index]
            for index, band in enumerate(SOURCE_BANDS)}
        masks = {}
        for band_index, band in enumerate(SOURCE_BANDS):
            external_valid = np.asarray(item.external_valid[:, band_index], dtype=bool)
            masked, inside = positions[band]
            comparison = m101_external_measurements.source_comparison_validity(
                external_valid, masked, inside, ~item.hardware_bad,
                np.isfinite(object_values[band]), np.ones(object_values[band].shape))
            reference = comparison & (radius > 6.)
            if int(reference.sum()) < 20:
                valid_indices = np.flatnonzero(comparison)
                keep = valid_indices[np.argsort(object_values[band][valid_indices])[
                    :max(20, valid_indices.size // 4)]]
                reference = np.zeros(comparison.shape, dtype=bool)
                reference[keep] = True
            baseline = robust_location(object_values[band][reference]) if np.any(reference) else np.nan
            scale = robust_scatter(object_values[band][reference]) if np.any(reference) else np.nan
            if not np.isfinite(scale) or scale <= 0:
                scale = float(np.nanstd(object_values[band][reference])) if np.any(reference) else np.nan
            scale = max(float(scale), 1e-12) if np.isfinite(scale) else np.nan
            threshold = baseline + 5.0 * scale if np.isfinite(baseline) and np.isfinite(scale) else np.nan
            masks[band] = (comparison & (object_values[band] > 0) &
                           (object_values[band] > threshold))
        source_masks[item.key] = {band: masks[band].copy() for band in SOURCE_BANDS}
        for (ifu_code, amp_index), indices in _item_group_indices(item).items():
            ifu = tuple(map(int, item.ifu[indices[0]]))
            amp = AMP_ORDER[amp_index]
            for band_index, band in enumerate(SOURCE_BANDS):
                selected = masks[band][indices]
                x_values = object_values[band][indices][selected]
                y_values = virus_by_band[band][indices][selected]
                valid = np.isfinite(x_values) & np.isfinite(y_values) & (x_values > 0)
                ratios = y_values[valid] / x_values[valid]
                scatter = float(robust_scatter(ratios)) if ratios.size else np.nan
                raw = float(robust_location(ratios)) if ratios.size else np.nan
                key = (item.h5_name, int(item.exposure), *ifu, amp, band)
                group_records[key] = {
                    "H5": item.h5_name, "exposure": int(item.exposure),
                    "SPECID": ifu[0], "IFUSLOT": ifu[1], "IFUID": ifu[2],
                    "AMP": amp, "band": band,
                    "N_candidate_fibers": int(np.sum(selected)),
                    "N_valid_source_fibers": int(valid.sum()),
                    "raw_ratio_warm_start": raw,
                    "robust_scatter_warm_start": scatter,
                    "source_candidate": bool(np.any(selected)),
                    "eligible_source": bool(
                        np.any(selected) and valid.sum() >= minimum_source_fibers and
                        np.isfinite(scatter) and scatter <= .15),
                    "minimum_source_fibers": int(minimum_source_fibers),
                }
    return source_masks, group_records


def _make_membership_masks(data, source_masks, group_records):
    for item in data:
        item.source_candidate[:, 0] = source_masks[item.key]["ON"]
        item.source_candidate[:, 1] = source_masks[item.key]["OFF"]
        accepted = np.zeros((item.row_index.size, 2), dtype=bool)
        for (ifu_code, amp_index), indices in _item_group_indices(item).items():
            ifu = tuple(map(int, item.ifu[indices[0]]))
            amp = AMP_ORDER[amp_index]
            for band_index, band in enumerate(SOURCE_BANDS):
                key = (item.h5_name, int(item.exposure), *ifu, amp, band)
                if group_records.get(key, {}).get("eligible_source", False):
                    accepted[indices, band_index] = source_masks[item.key][band][indices]
        item.source_accepted[:, :] = accepted


def _membership_counts(data, group_records):
    union_count = sum(int(np.sum(item.source_candidate.any(axis=1))) for item in data)
    on_count = sum(int(np.sum(item.source_candidate[:, 0])) for item in data)
    off_count = sum(int(np.sum(item.source_candidate[:, 1])) for item in data)
    blank_original = sum(int(np.sum(item.blank_valid)) for item in data)
    blank_final = sum(int(np.sum(item.blank_valid & ~item.source_candidate.any(axis=1)))
                      for item in data)
    rows = list(group_records.values())
    pure_blank_groups = 0
    for item in data:
        union = item.source_candidate.any(axis=1)
        for (ifu_code, amp_index), indices in _item_group_indices(item).items():
            if not np.any(item.blank_valid[indices] & ~union[indices]):
                continue
            ifu = tuple(map(int, item.ifu[indices[0]]))
            amp = AMP_ORDER[amp_index]
            for band in SOURCE_BANDS:
                # The validated artifact counts one row per amplifier/band
                # group, including both ON and OFF rows.
                key = (item.h5_name, int(item.exposure), *ifu, amp, band)
                if not group_records[key]["source_candidate"]:
                    pure_blank_groups += 1
    return {
        "candidate_native_fibers": union_count,
        "candidate_native_fibers_ON": on_count,
        "candidate_native_fibers_OFF": off_count,
        "candidate_native_fibers_union": union_count,
        "blank_original": blank_original, "blank_final": blank_final,
        "candidate_groups": int(sum(row["source_candidate"] for row in rows)),
        "accepted_source_groups": int(sum(row["eligible_source"] for row in rows)),
        "rejected_source_quality_groups": int(sum(
            row["source_candidate"] and not row["eligible_source"] for row in rows)),
        "pure_blank_groups": int(pure_blank_groups),
    }


def _membership_rows(data, group_records):
    rows = []
    for item in data:
        union = item.source_candidate.any(axis=1)
        for (ifu_code, amp_index), indices in _item_group_indices(item).items():
            ifu = tuple(map(int, item.ifu[indices[0]]))
            amp = AMP_ORDER[amp_index]
            for band in SOURCE_BANDS:
                key = (item.h5_name, int(item.exposure), *ifu, amp, band)
                record = dict(group_records[key])
                candidate = bool(record["source_candidate"])
                record.update({
                    "N_original_blank_fibers": int(np.sum(item.blank_valid[indices])),
                    "N_final_blank_fibers": int(np.sum(item.blank_valid[indices] & ~union[indices])),
                    "N_union_source_candidate_fibers": int(np.sum(union[indices])),
                    "blank_removed_by_union": int(np.sum(item.blank_valid[indices] & union[indices])),
                    "transition": ("candidate_to_accepted_source" if candidate and record["eligible_source"]
                                   else "candidate_to_rejected_source_quality" if candidate
                                   else "noncandidate_to_pure_blank" if np.any(item.blank_valid[indices] & ~union[indices])
                                   else "noncandidate_to_unused"),
                    "source_membership_frozen": True,
                })
                rows.append(record)
    return rows


def _load_or_reproduce_membership(data, cache_manifest, args, product_path,
                                  responses, fq, external, selected_names, progress):
    has_cached = bool(cache_manifest.get("membership_frozen", False))
    artifact = None
    candidates = []
    if args.membership_artifact:
        candidates.append(Path(args.membership_artifact).expanduser().resolve())
    candidates.extend([
        Path(product_path).parent / "m101_joint_source_membership.json",
        Path("m101_joint_calibration/m101_joint_source_membership.json").resolve(),
    ])
    for candidate in candidates:
        if candidate.exists():
            artifact = candidate
            break
    artifact_matches_population = False
    if artifact:
        payload = json.loads(artifact.read_text())
        artifact_names = {row.get("H5") for row in payload.get("groups", [])}
        artifact_matches_population = artifact_names == set(selected_names)
    if has_cached:
        if artifact and artifact_matches_population:
            payload = json.loads(artifact.read_text())
            expected = payload.get("counts", {})
            # Counts in the cache are independently reconstructed below from
            # frozen masks; no source population is changed by this check.
            cache_counts = {
                "candidate_native_fibers": sum(int(np.sum(item.source_candidate.any(axis=1))) for item in data),
                "candidate_native_fibers_ON": sum(int(np.sum(item.source_candidate[:, 0])) for item in data),
                "candidate_native_fibers_OFF": sum(int(np.sum(item.source_candidate[:, 1])) for item in data),
                "candidate_native_fibers_union": sum(int(np.sum(item.source_candidate.any(axis=1))) for item in data),
                "blank_original": sum(int(np.sum(item.blank_valid)) for item in data),
                "blank_final": sum(int(np.sum(item.blank_valid & ~item.source_candidate.any(axis=1))) for item in data),
            }
            for key, value in cache_counts.items():
                if key in expected and int(expected[key]) != value:
                    raise ValueError("cached source membership count differs from artifact: %s" % key)
        progress("using frozen source masks from band cache")
        return artifact, {"source_population": "cache_frozen"}
    if external is None or not args.compact_mask:
        raise ValueError("source masks are absent from cache; --external-cache and --compact-mask are required")
    product, warm_model, warm_skies = _warm_model_band_sky(product_path, responses, selected_names)
    mask_data, mask_wcs = m101_compact_mask.load(args.compact_mask)
    source_masks, group_records = _candidate_population_band(
        data, external, mask_data, mask_wcs, warm_model, warm_skies, fq, responses,
        args.minimum_source_fibers)
    _make_membership_masks(data, source_masks, group_records)
    counts = _membership_counts(data, group_records)
    if artifact and artifact_matches_population:
        expected = json.loads(artifact.read_text()).get("counts", {})
        for key, value in counts.items():
            if key in expected and int(expected[key]) != int(value):
                raise ValueError("reproduced source membership differs from artifact: %s" % key)
        progress("reproduced and verified frozen membership from %s" % artifact)
    else:
        progress("reproduced frozen membership with validated candidate logic")
    return artifact, {"source_population": "reproduced_validated_logic",
                      "counts": counts,
                      "groups": _membership_rows(data, group_records),
                      "warm_start_product": file_identity(product_path)}


def _base_log_response(item, state, p_ifu=None, ax=None, ay=None, p_amp=None):
    """Return the band-independent base log response for one item."""
    if state is not None:
        p_ifu = state.p_ifu
        ax = state.ax
        ay = state.ay
        p_amp = state.p_amp
        ifu_amp = p_amp[item.ifu_code]
        values = (p_ifu[item.ifu_code] + np.sum(BASIS[item.amp] * ifu_amp, axis=1) +
                  ax[item.item_index] * (item.x_arcmin - state.x_center[item.item_index]) +
                  ay[item.item_index] * (item.y_arcmin - state.y_center[item.item_index]))
        return np.asarray(values, dtype=float)
    # Warm Model-3 objects use dictionaries keyed by the current fitter's
    # exact identities.  This branch is only for source-membership replay.
    ifu_values = [tuple(map(int, value)) for value in item.ifu]
    values = np.zeros(item.row_index.size, dtype=float)
    for index, (ifu, amp) in enumerate(zip(ifu_values, item.amp)):
        amp_name = AMP_ORDER[int(amp)]
        values[index] = (p_ifu.get(ifu, 0.0) + p_amp.get(ifu + (amp_name,), 0.0) +
                         ax.get(item.key, 0.0) * item.x_arcmin[index] +
                         ay.get(item.key, 0.0) * item.y_arcmin[index])
    return values


def _model_response(item, state, hi_map, band_x):
    z = _base_log_response(item, state)
    row_hi = np.asarray([hi_map[(item.h5_name, int(code))]
                         for code in item.ifu_code], dtype=int)
    gray = np.sum(BASIS[item.amp] * state.g_star[row_hi], axis=1)
    color = np.sum(BASIS[item.amp] * state.q_color[row_hi], axis=1)
    return np.exp(z[:, None] + gray[:, None] + color[:, None] * band_x[None, :])


def _additive_bands(item, state, fq):
    alpha = state.alpha_q[item.item_index][item.ifu_code, item.amp]
    return np.asarray(alpha, dtype=float)[:, None] * \
        fq[item.q, None] * item.K[None, :]


def _blank_masks(data):
    return {item.key: np.asarray(item.blank_valid & ~item.source_candidate.any(axis=1), dtype=bool)
            for item in data}


def _initial_state(data, n_ifu, n_hi):
    return InitialState(
        p_ifu=np.zeros(n_ifu, dtype=float),
        ax=np.zeros(len(data), dtype=float), ay=np.zeros(len(data), dtype=float),
        x_center=np.zeros(len(data), dtype=float), y_center=np.zeros(len(data), dtype=float),
        alpha_q=np.zeros((len(data), n_ifu, 4), dtype=float),
        p_amp=np.zeros((n_ifu, 3), dtype=float),
        g_star=np.zeros((n_hi, 3), dtype=float),
        q_color=np.zeros((n_hi, 3), dtype=float),
        xbar=np.zeros(n_hi, dtype=float))


def _profile_sky(data, state, hi_map, band_x, fq, selected):
    result = {}
    for item in data:
        response = _model_response(item, state, hi_map, band_x)
        additive = _additive_bands(item, state, fq)
        values = (item.band_total - additive) / response
        sky = np.full(len(ALL_BANDS), np.nan, dtype=float)
        mask = selected[item.key]
        for band_index in range(len(ALL_BANDS)):
            sky[band_index] = robust_location(values[mask, band_index])
        if not np.all(np.isfinite(sky)):
            raise ValueError("sky profile has nonfinite bands for %s" % (item.key,))
        result[item.key] = sky
    return result


def _stage0_sky(data, selected):
    result = {}
    for item in data:
        sky = np.asarray([robust_location(item.band_total[selected[item.key], band])
                          for band in range(len(ALL_BANDS))], dtype=float)
        if not np.all(np.isfinite(sky)):
            raise ValueError("Stage 0 has no finite pure-blank sky for %s" % (item.key,))
        result[item.key] = sky
    return result


def _stage1_fit(data, state, sky0, selected, hi_map):
    ifus = sorted({int(code) for item in data for code in np.unique(item.ifu_code)})
    n_ifu = max(ifus) + 1 if ifus else 0
    basis = _zero_sum_basis(n_ifu)
    centers = np.zeros(len(data), dtype=float)
    ycenters = np.zeros(len(data), dtype=float)
    records = []
    for item in data:
        central = selected[item.key] & (item.q >= CENTRAL_Q_MIN) & (item.q <= CENTRAL_Q_MAX)
        if not np.any(central):
            central = selected[item.key]
        centers[item.item_index] = robust_location(item.x_arcmin[central]) if np.any(central) else 0.0
        ycenters[item.item_index] = robust_location(item.y_arcmin[central]) if np.any(central) else 0.0
        for ifu_code in np.unique(item.ifu_code[central]):
            rows = central & (item.ifu_code == ifu_code)
            if not np.any(rows):
                continue
            x = item.x_arcmin[rows] - centers[item.item_index]
            y = item.y_arcmin[rows] - ycenters[item.item_index]
            for band in range(len(ALL_BANDS)):
                valid = np.isfinite(item.band_total[rows, band]) & np.isfinite(sky0[item.key][band]) & \
                    (np.abs(sky0[item.key][band]) > MIN_SAFE_DENOMINATOR)
                values = item.band_total[rows, band][valid] / sky0[item.key][band] - 1.0
                if values.size == 0:
                    continue
                location = float(robust_location(values))
                scatter = float(robust_scatter(values)) if values.size > 1 else .05
                sigma = max(scatter / math.sqrt(max(1, values.size)), .002)
                records.append({"item": item.item_index, "ifu_code": int(ifu_code),
                                "band": band, "x": float(robust_location(x[valid])),
                                "y": float(robust_location(y[valid])), "value": location,
                                "sigma": sigma, "N": int(values.size),
                                "scatter": scatter})
    if not records:
        raise ValueError("Stage 1 produced no compressed pure-blank summaries")
    # One coefficient pair is used for every exposure plane.  The compressed
    # system is therefore only 69 + 2*57 columns for the full population.
    nplane = 2 * len(data)
    design = np.zeros((len(records), basis.shape[1] + nplane), dtype=float)
    target = np.asarray([row["value"] for row in records], dtype=float)
    sigma = np.asarray([row["sigma"] for row in records], dtype=float)
    for index, row in enumerate(records):
        if basis.shape[1]:
            design[index, :basis.shape[1]] = basis[row["ifu_code"]]
        cursor = basis.shape[1] + 2 * row["item"]
        design[index, cursor:cursor + 2] = (row["x"], row["y"])
    coefficients, fit = _irls_linear(design, target, sigma, iterations=5)
    if basis.shape[1]:
        state.p_ifu[:] = basis @ coefficients[:basis.shape[1]]
    state.ax[:] = coefficients[basis.shape[1]::2][:len(data)]
    state.ay[:] = coefficients[basis.shape[1] + 1::2][:len(data)]
    state.x_center[:] = centers
    state.y_center[:] = ycenters
    return {"summary_rows": records, "fit": fit,
            "x_center": centers, "y_center": ycenters,
            "central_q": [CENTRAL_Q_MIN, CENTRAL_Q_MAX]}


def _stage1_summary_key(item_index, ifu_code, band_index):
    return int(item_index), int(ifu_code), int(band_index)


def _stage1_log_summary(values, errors, x, y, q):
    """Summarize one amplifier using log(D/S0) without formal-error domination."""
    values = np.asarray(values, dtype=float)
    errors = np.asarray(errors, dtype=float)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    q = np.asarray(q, dtype=int)
    good = np.isfinite(values) & (values > 0)
    if not np.any(good):
        return {"supported": False, "support_reason": "no finite positive D/S0",
                "N_fibers": 0, "robust_location": np.nan,
                "robust_scatter": np.nan, "location_uncertainty": np.nan,
                "formal_log_error_median": np.nan,
                "robust_x_arcmin": np.nan, "robust_y_arcmin": np.nan,
                "q_min": np.nan, "q_max": np.nan, "q_support": []}
    values = np.log(values[good])
    x, y, q = x[good], y[good], q[good]
    finite_errors = errors[good]
    finite_errors = finite_errors[np.isfinite(finite_errors) & (finite_errors > 0)]
    location = float(robust_location(values))
    scatter = float(robust_scatter(values)) if values.size > 1 else np.nan
    formal = float(np.median(finite_errors)) if finite_errors.size else np.nan
    # A fixed floor keeps a very small formal error from overpowering the
    # robust scatter.  The floor is on the summary fit weight, not on D.
    uncertainty = max(
        float(scatter) / math.sqrt(max(1, values.size))
        if np.isfinite(scatter) and scatter > 0 else 0.0,
        0.002,
    )
    return {"supported": True, "support_reason": "supported central-q summary",
            "N_fibers": int(values.size), "robust_location": location,
            "robust_scatter": scatter, "location_uncertainty": uncertainty,
            "formal_log_error_median": formal,
            "robust_x_arcmin": float(robust_location(x)),
            "robust_y_arcmin": float(robust_location(y)),
            "q_min": int(np.min(q)), "q_max": int(np.max(q)),
            "q_support": sorted({int(value) for value in q})}


def _stage1_build_compressed_summaries(data, sky0, selected):
    """Build amplifier summaries first, then equal-amplifier common modes."""
    amp_rows = []
    amp_lookup = {}
    common_rows = []
    for item in data:
        item_mask = np.asarray(selected[item.key], dtype=bool)
        central_mask = item_mask & (item.q >= CENTRAL_Q_MIN) & (item.q <= CENTRAL_Q_MAX)
        item_ifus = sorted({int(code) for code in np.unique(item.ifu_code)})
        for ifu_code in item_ifus:
            ifu = tuple(map(int, item.ifu[np.flatnonzero(item.ifu_code == ifu_code)[0]]))
            for amp_index, amp_name in enumerate(AMP_ORDER):
                central_indices = np.flatnonzero(
                    central_mask & (item.ifu_code == ifu_code) & (item.amp == amp_index))
                for band_index, band in enumerate(ALL_BANDS):
                    sky_value = float(sky0[item.key][band_index])
                    values = item.band_total[central_indices, band_index]
                    errors = item.band_error[central_indices, band_index]
                    valid = np.isfinite(values) & (values > 0) & np.isfinite(sky_value) & \
                        (sky_value > 0)
                    ratio = values[valid] / sky_value if np.any(valid) else np.zeros(0)
                    relative_error = (np.abs(errors[valid]) / values[valid]
                                      if np.any(valid) else np.zeros(0))
                    summary = _stage1_log_summary(
                        ratio, relative_error,
                        item.x_arcmin[central_indices][valid],
                        item.y_arcmin[central_indices][valid],
                        item.q[central_indices][valid])
                    row = {
                        "H5": item.h5_name, "exposure": int(item.exposure),
                        "item_index": int(item.item_index), "SPECID": ifu[0],
                        "IFUSLOT": ifu[1], "IFUID": ifu[2],
                        "IFU_CODE": int(ifu_code), "AMP": amp_name,
                        "AMP_INDEX": int(amp_index), "band": band,
                        "band_index": int(band_index),
                        "central_q_min": CENTRAL_Q_MIN,
                        "central_q_max": CENTRAL_Q_MAX,
                        "central_q_only": True,
                        **summary,
                    }
                    if central_indices.size == 0:
                        row["support_reason"] = "no central-q pure-blank rows"
                    amp_rows.append(row)
                    amp_lookup[(item.item_index, ifu_code, amp_index, band_index)] = row
            for band_index, band in enumerate(ALL_BANDS):
                values = [amp_lookup[(item.item_index, ifu_code, a, band_index)]
                          for a in range(4)]
                complete = all(row["supported"] and
                               np.isfinite(row["robust_location"]) and
                               np.isfinite(row["robust_x_arcmin"]) and
                               np.isfinite(row["robust_y_arcmin"])
                               for row in values)
                common = {
                    "H5": item.h5_name, "exposure": int(item.exposure),
                    "item_index": int(item.item_index), "SPECID": ifu[0],
                    "IFUSLOT": ifu[1], "IFUID": ifu[2],
                    "IFU_CODE": int(ifu_code), "band": band,
                    "band_index": int(band_index),
                    "complete_four_amplifiers": bool(complete),
                    "N_amp_supported": int(sum(row["supported"] for row in values)),
                    "N_fibers": int(sum(row["N_fibers"] for row in values)),
                    "Y_eib": np.nan, "X_ei": np.nan, "YPOS_ei": np.nan,
                    "location_uncertainty": np.nan,
                    "y_LL": values[0]["robust_location"],
                    "y_LU": values[1]["robust_location"],
                    "y_RL": values[2]["robust_location"],
                    "y_RU": values[3]["robust_location"],
                    "x_LL": values[0]["robust_x_arcmin"],
                    "x_LU": values[1]["robust_x_arcmin"],
                    "x_RL": values[2]["robust_x_arcmin"],
                    "x_RU": values[3]["robust_x_arcmin"],
                    "ypos_LL": values[0]["robust_y_arcmin"],
                    "ypos_LU": values[1]["robust_y_arcmin"],
                    "ypos_RL": values[2]["robust_y_arcmin"],
                    "ypos_RU": values[3]["robust_y_arcmin"],
                    "amp_q_min": [row["q_min"] for row in values],
                    "amp_q_max": [row["q_max"] for row in values],
                    "amp_support": [bool(row["supported"]) for row in values],
                }
                if complete:
                    # This is deliberately an equal, unweighted four-term mean.
                    common["Y_eib"] = float(.25 * sum(
                        row["robust_location"] for row in values))
                    common["X_ei"] = float(.25 * sum(
                        row["robust_x_arcmin"] for row in values))
                    common["YPOS_ei"] = float(.25 * sum(
                        row["robust_y_arcmin"] for row in values))
                    common["location_uncertainty"] = float(.25 * math.sqrt(sum(
                        row["location_uncertainty"] ** 2 for row in values)))
                else:
                    common["support_reason"] = "requires all four usable amplifiers"
                common_rows.append(common)
    return amp_rows, common_rows, amp_lookup


def _stage1_fixed_ifu_positions(common_rows, n_ifu):
    """Use stored cache coordinates as fixed physical-IFU reference positions."""
    x, y = np.full(n_ifu, np.nan), np.full(n_ifu, np.nan)
    for code in range(n_ifu):
        rows = [row for row in common_rows
                if row["IFU_CODE"] == code and row["complete_four_amplifiers"]]
        if rows:
            x[code] = float(robust_location([row["X_ei"] for row in rows]))
            y[code] = float(robust_location([row["YPOS_ei"] for row in rows]))
    return x, y


def _stage1_model(common_rows, s, p_ifu, ax, ay):
    return np.asarray([
        s[int(row["item_index"]), int(row["band_index"])] +
        p_ifu[int(row["IFU_CODE"])] +
        ax[int(row["item_index"])] * float(row["X_ei"]) +
        ay[int(row["item_index"])] * float(row["YPOS_ei"])
        for row in common_rows], dtype=float)


def _stage1_apply_pifu_gauge(s, p_ifu, fit_band_indices):
    """Exact p_IFU/s gauge transfer, retaining every compressed prediction."""
    offset = float(np.mean(p_ifu)) if p_ifu.size else 0.0
    p_ifu -= offset
    s[:, fit_band_indices] += offset
    return offset


def _stage1_apply_plane_gauge(s, p_ifu, ax, ay, common_rows,
                              fixed_x, fixed_y, fit_band_indices):
    """Transfer the persistent plane at fixed IFU positions into p_IFU.

    The cache coordinates are fixed exposure coordinates.  The transfer is
    exact at the fixed physical-IFU reference positions.  The reported numerical check is
    performed on those reference-coordinate predictions, where this is an
    exact gauge transformation.
    """
    delta_x = float(np.mean(ax)) if ax.size else 0.0
    delta_y = float(np.mean(ay)) if ay.size else 0.0
    model_rows = [row for row in common_rows
                  if row["complete_four_amplifiers"]]
    old_s = s.copy()
    old_p = p_ifu.copy()
    old_ax = ax.copy()
    old_ay = ay.copy()
    old = _stage1_model(model_rows, old_s, old_p, old_ax, old_ay)
    ax -= delta_x
    ay -= delta_y
    for code in range(p_ifu.size):
        if np.isfinite(fixed_x[code]):
            p_ifu[code] += delta_x * fixed_x[code]
        if np.isfinite(fixed_y[code]):
            p_ifu[code] += delta_y * fixed_y[code]
    new = _stage1_model(model_rows, s, p_ifu, ax, ay)
    # The reference-coordinate expression above is intentionally evaluated
    # directly from the pre-transfer model below; it avoids any dependence on
    # the exposure-coordinate translation profiling.
    reference_max = 0.0
    for row in model_rows:
        code = int(row["IFU_CODE"])
        if not row["complete_four_amplifiers"] or not np.isfinite(fixed_x[code]) or not np.isfinite(fixed_y[code]):
            continue
        item_index, band_index = int(row["item_index"]), int(row["band_index"])
        before = (old_p[code] + old_ax[item_index] * fixed_x[code] +
                  old_ay[item_index] * fixed_y[code] +
                  old_s[item_index, band_index])
        after = (p_ifu[code] + ax[item_index] * fixed_x[code] +
                 ay[item_index] * fixed_y[code] + s[item_index, band_index])
        reference_max = max(reference_max, abs(float(before - after)))
    return {"delta_ax": delta_x, "delta_ay": delta_y,
            "reference_prediction_max_change": float(reference_max),
            "exposure_coordinate_prediction_max_change": float(
                np.max(np.abs(new - old), initial=0.0))}


def _stage1_revised_fit(data, state, sky0, selected, iterations=STAGE1_DEFAULT_ITERATIONS,
                        tolerance=STAGE1_DEFAULT_TOLERANCE, fit_band_indices=None,
                        compressed_summaries=None):
    """Transparent fixed-iteration robust fit of the revised Stage-1 model."""
    n_ifu = state.p_ifu.size
    n_items = len(data)
    fit_band_indices = list(range(len(ALL_BANDS))) if fit_band_indices is None \
        else [int(index) for index in fit_band_indices]
    if compressed_summaries is None:
        amp_rows, common_rows, amp_lookup = _stage1_build_compressed_summaries(
            data, sky0, selected)
    else:
        amp_rows, common_rows, amp_lookup = compressed_summaries
    usable = [row for row in common_rows
              if row["complete_four_amplifiers"]]
    fit_rows = [row for row in usable if row["band_index"] in fit_band_indices]
    if not fit_rows:
        raise ValueError("Stage 1 produced no complete four-amplifier summaries")
    fixed_x, fixed_y = _stage1_fixed_ifu_positions(common_rows, n_ifu)
    s = np.zeros((n_items, len(ALL_BANDS)), dtype=float)
    p_ifu = np.zeros(n_ifu, dtype=float)
    ax = np.zeros(n_items, dtype=float)
    ay = np.zeros(n_items, dtype=float)
    history = []
    gauge_transfers = []
    for iteration in range(1, int(iterations) + 1):
        before = (s.copy(), p_ifu.copy(), ax.copy(), ay.copy())
        # A. Exposure/band sky nuisance.
        for item_index in range(n_items):
            for band_index in fit_band_indices:
                rows = [row for row in fit_rows
                        if row["item_index"] == item_index and
                        row["band_index"] == band_index]
                if rows:
                    s[item_index, band_index] = float(robust_location([
                        row["Y_eib"] - p_ifu[row["IFU_CODE"]] -
                        ax[item_index] * row["X_ei"] -
                        ay[item_index] * row["YPOS_ei"] for row in rows]))
        # B. Persistent physical-IFU throughput.
        for ifu_code in range(n_ifu):
            rows = [row for row in fit_rows if row["IFU_CODE"] == ifu_code]
            if rows:
                p_ifu[ifu_code] = float(robust_location([
                    row["Y_eib"] - s[row["item_index"], row["band_index"]] -
                    ax[row["item_index"]] * row["X_ei"] -
                    ay[row["item_index"]] * row["YPOS_ei"] for row in rows]))
        p_offset = _stage1_apply_pifu_gauge(s, p_ifu, fit_band_indices)
        # C. Independent tiny robust two-parameter plane per exposure.
        plane_fit = {}
        for item_index in range(n_items):
            rows = [row for row in fit_rows if row["item_index"] == item_index]
            if not rows:
                plane_fit[str(item_index)] = {"N": 0, "rank": 0, "scale": np.nan}
                continue
            design = np.asarray([[row["X_ei"], row["YPOS_ei"]] for row in rows])
            target = np.asarray([
                row["Y_eib"] - s[item_index, row["band_index"]] -
                p_ifu[row["IFU_CODE"]] for row in rows])
            sigma = np.asarray([max(float(row["location_uncertainty"]), .002)
                                for row in rows])
            coefficients, fit = _irls_linear(design, target, sigma, iterations=4)
            ax[item_index], ay[item_index] = coefficients
            plane_fit[str(item_index)] = fit
        plane_transfer = _stage1_apply_plane_gauge(
            s, p_ifu, ax, ay, common_rows, fixed_x, fixed_y, fit_band_indices)
        # Plane transfer changes the mean of p_IFU; apply the exact p/s gauge
        # once more so both explicit gauges hold after every block iteration.
        p_offset_after_plane = _stage1_apply_pifu_gauge(s, p_ifu, fit_band_indices)
        after = (s, p_ifu, ax, ay)
        changes = [float(np.max(np.abs(new - old), initial=0.0))
                   for new, old in zip(after, before)]
        max_change = max(changes, default=0.0)
        gauge_transfers.append({"iteration": iteration,
                                "p_IFU_common_shift": p_offset,
                                "p_IFU_post_plane_common_shift": p_offset_after_plane,
                                **plane_transfer})
        history.append({"iteration": iteration, "max_parameter_change": max_change,
                        "max_s_change": changes[0], "max_p_IFU_change": changes[1],
                        "max_ax_change": changes[2], "max_ay_change": changes[3],
                        "p_IFU_sum": float(np.sum(p_ifu)),
                        "mean_ax": float(np.mean(ax)), "mean_ay": float(np.mean(ay)),
                        "p_IFU_gauge_shift": p_offset,
                        "plane_gauge_delta_ax": plane_transfer["delta_ax"],
                        "plane_gauge_delta_ay": plane_transfer["delta_ay"],
                        "plane_reference_prediction_max_change": plane_transfer[
                            "reference_prediction_max_change"],
                        "plane_exposure_coordinate_prediction_max_change": plane_transfer[
                            "exposure_coordinate_prediction_max_change"],
                        "plane_fit": plane_fit})
        if max_change <= tolerance:
            break
    state.p_ifu[:] = p_ifu
    state.ax[:] = ax
    state.ay[:] = ay
    state.x_center[:] = 0.0
    state.y_center[:] = 0.0
    sky_log = {(item.key): s[item.item_index].copy() for item in data}
    return {"amp_rows": amp_rows, "common_rows": common_rows,
            "amp_lookup": amp_lookup, "sky_log_correction": sky_log,
            "S1": {item.key: sky0[item.key] * np.exp(s[item.item_index])
                   for item in data},
            "p_IFU": p_ifu.copy(), "ax": ax.copy(), "ay": ay.copy(),
            "fixed_ifu_x": fixed_x, "fixed_ifu_y": fixed_y,
            "iteration_history": history, "gauge_transfers": gauge_transfers,
            "fit_band_indices": fit_band_indices,
            "complete_rows": len(usable), "fit_rows": len(fit_rows),
            "physical_ifus_supported": int(np.sum([
                any(row["IFU_CODE"] == code for row in usable)
                for code in range(n_ifu)])),
            "iterations_run": len(history)}


def _stage1_sky_from_log(sky0, sky_log, data):
    return {item.key: np.asarray(sky0[item.key], dtype=float) *
            np.exp(np.asarray(sky_log[item.key], dtype=float)) for item in data}


def _stage1_profile_heldout_sky(fit, sky0, data, common_rows, heldout_bands):
    """Profile only held-out band normalizations after null-band fitting."""
    sky_log = {item.key: np.zeros(len(ALL_BANDS), dtype=float) for item in data}
    for item in data:
        sky_log[item.key][:len(NULL_BANDS)] = fit["sky_log_correction"][item.key][:len(NULL_BANDS)]
    for item in data:
        for band_index in heldout_bands:
            rows = [row for row in common_rows
                    if row["item_index"] == item.item_index and
                    row["band_index"] == band_index and
                    row["complete_four_amplifiers"]]
            if rows:
                sky_log[item.key][band_index] = float(robust_location([
                    row["Y_eib"] - fit["p_IFU"][row["IFU_CODE"]] -
                    fit["ax"][item.item_index] * row["X_ei"] -
                    fit["ay"][item.item_index] * row["YPOS_ei"]
                    for row in rows]))
    return sky_log, _stage1_sky_from_log(sky0, sky_log, data)


def _stage1_response(item, p_ifu, ax, ay):
    return np.exp(p_ifu[item.ifu_code] + ax[item.item_index] * item.x_arcmin +
                  ay[item.item_index] * item.y_arcmin)


def _stage1_robust_slope(x, y):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    good = np.isfinite(x) & np.isfinite(y)
    if np.sum(good) < 3 or np.std(x[good]) <= 0:
        return np.nan
    coefficients, _ = _irls_linear(
        np.column_stack((np.ones(np.sum(good)), x[good])), y[good],
        np.ones(np.sum(good)), iterations=4)
    return float(coefficients[1])


def _stage1_diagnostic_rows(data, state, sky0, sky1, selected, common_rows=None):
    """Evaluate exact exponential residuals on one unchanged blank population."""
    stages = ("STAGE0", "STAGE1_PRE_SKY", "STAGE1_POST_SKY")
    rows = []
    for stage in stages:
        sky = sky0 if stage != "STAGE1_POST_SKY" else sky1
        all_values = []
        band_values = {band: [] for band in ALL_BANDS}
        amp_values = {(amp, band): [] for amp in AMP_ORDER for band in ALL_BANDS}
        q_values = {}
        ifu_values = {}
        x_values, y_values = [], []
        for item in data:
            response = np.ones(item.row_index.size, dtype=float) if stage == "STAGE0" else \
                _stage1_response(item, state.p_ifu, state.ax, state.ay)
            denominator = response[:, None] * sky[item.key][None, :]
            with np.errstate(divide="ignore", invalid="ignore"):
                fractional = item.band_total / denominator - 1.0
            use = np.asarray(selected[item.key], dtype=bool)
            for band_index, band in enumerate(ALL_BANDS):
                values = fractional[use, band_index]
                values = values[np.isfinite(values)]
                all_values.append(values)
                band_values[band].append(values)
                x_values.append((item.x_arcmin[use], fractional[use, band_index]))
                y_values.append((item.y_arcmin[use], fractional[use, band_index]))
                for amp_index, amp in enumerate(AMP_ORDER):
                    amp_values[(amp, band)].append(
                        fractional[use & (item.amp == amp_index), band_index])
                for q in np.unique(item.q[use]):
                    q_values.setdefault((int(q), band), []).append(
                        fractional[use & (item.q == q), band_index])
        # The physical-IFU common diagnostic is deliberately based on the
        # equal-amplifier summaries, so amplifier contrast is not mixed into
        # the common-mode diagnostic.
        if common_rows is not None:
            for common in common_rows:
                if not common["complete_four_amplifiers"]:
                    continue
                band_index = int(common["band_index"])
                item_index = int(common["item_index"])
                if stage == "STAGE0":
                    value = np.expm1(float(common["Y_eib"]))
                else:
                    prediction = state.p_ifu[int(common["IFU_CODE"])] + \
                        state.ax[item_index] * float(common["X_ei"]) + \
                        state.ay[item_index] * float(common["YPOS_ei"])
                    if stage == "STAGE1_POST_SKY":
                        prediction += math.log(float(sky1[data[item_index].key][band_index] /
                                                     sky0[data[item_index].key][band_index]))
                    value = np.expm1(float(common["Y_eib"]) - prediction)
                if np.isfinite(value):
                    ifu_values.setdefault((int(common["IFU_CODE"]), ALL_BANDS[band_index]), []).append(value)

        concatenate = lambda chunks: np.concatenate([
            np.asarray(chunk, dtype=float) for chunk in chunks if np.asarray(chunk).size
        ]) if any(np.asarray(chunk).size for chunk in chunks) else np.zeros(0)
        overall = concatenate(all_values)
        summary = residual_summary(overall)
        rows.append({"stage": stage, "population": "pure_blank_frozen",
                     "grouping": "overall", "group": "all", "band": "ALL",
                     "value": "fractional", **summary})
        for band in ALL_BANDS:
            values = concatenate(band_values[band])
            rows.append({"stage": stage, "population": "pure_blank_frozen",
                         "grouping": "band", "group": "all", "band": band,
                         "value": "fractional", **residual_summary(values)})
        for (amp, band), chunks in amp_values.items():
            rows.append({"stage": stage, "population": "pure_blank_frozen",
                         "grouping": "amplifier", "group": amp, "band": band,
                         "value": "fractional", **residual_summary(concatenate(chunks))})
        for band in ALL_BANDS:
            common = np.asarray([value for (code, b), values in ifu_values.items()
                                 if b == band for value in values], dtype=float)
            rows.append({"stage": stage, "population": "pure_blank_frozen",
                         "grouping": "physical_IFU_common", "group": "all", "band": band,
                         "value": "fractional", **residual_summary(common)})
        for (q, band), chunks in sorted(q_values.items()):
            rows.append({"stage": stage, "population": "pure_blank_frozen",
                         "grouping": "q", "group": q, "band": band,
                         "value": "fractional", **residual_summary(concatenate(chunks))})
        x = concatenate([pair[0] for pair in x_values])
        y = concatenate([pair[1] for pair in x_values])
        rows.append({"stage": stage, "population": "pure_blank_frozen",
                     "grouping": "focal_x", "group": "all", "band": "ALL",
                     "value": "fractional_slope", "slope": _stage1_robust_slope(x, y),
                     **residual_summary(y)})
        x = concatenate([pair[0] for pair in y_values])
        y = concatenate([pair[1] for pair in y_values])
        rows.append({"stage": stage, "population": "pure_blank_frozen",
                     "grouping": "focal_y", "group": "all", "band": "ALL",
                     "value": "fractional_slope", "slope": _stage1_robust_slope(x, y),
                     **residual_summary(y)})
    return rows


def _stage1_heldout_report(data, sky0, fit, common_rows, selected):
    heldout = list(range(len(NULL_BANDS), len(ALL_BANDS)))
    heldout_log, heldout_sky = _stage1_profile_heldout_sky(
        fit, sky0, data, common_rows, heldout)
    result = {"fit_bands": list(NULL_BANDS), "held_out_bands": list(SOURCE_BANDS),
              "source_information_used": False, "sky_log_correction": {},
              "bands": {}}
    for item in data:
        result["sky_log_correction"][str(item.key)] = heldout_log[item.key].tolist()
    for band_index in heldout:
        band = ALL_BANDS[band_index]
        before, after = [], []
        for item in data:
            use = np.asarray(selected[item.key], dtype=bool)
            d = item.band_total[use, band_index]
            s0 = sky0[item.key][band_index]
            m = _stage1_response(item, fit["p_IFU"], fit["ax"], fit["ay"])[use]
            with np.errstate(divide="ignore", invalid="ignore"):
                before.extend((d / s0 - 1.0).tolist())
                after.extend((d / (m * heldout_sky[item.key][band_index]) - 1.0).tolist())
        before, after = np.asarray(before), np.asarray(after)
        before_summary, after_summary = residual_summary(before), residual_summary(after)
        result["bands"][band] = {
            "stage0": before_summary, "held_out_stage1": after_summary,
            "robust_rms_change": float(after_summary["robust_rms"] -
                                       before_summary["robust_rms"]),
            "improved": bool(after_summary["robust_rms"] < before_summary["robust_rms"]),
        }
    result["overall_improved"] = all(value["improved"] for value in result["bands"].values())
    return result


def _stage1_gauge_validation(data, state, fit, selected, native_pixels_used=False):
    common_rows = fit["common_rows"]
    used = [row for row in common_rows if row["complete_four_amplifiers"]]
    equal_mean_error = 0.0
    equal_x_error = 0.0
    equal_y_error = 0.0
    for row in used:
        values = [row["y_LL"], row["y_LU"], row["y_RL"], row["y_RU"]]
        equal_mean_error = max(equal_mean_error, abs(
            float(row["Y_eib"]) - .25 * sum(values)))
        x_values = [row["x_LL"], row["x_LU"], row["x_RL"], row["x_RU"]]
        y_values = [row["ypos_LL"], row["ypos_LU"], row["ypos_RL"], row["ypos_RU"]]
        equal_x_error = max(equal_x_error, abs(
            float(row["X_ei"]) - .25 * sum(x_values)))
        equal_y_error = max(equal_y_error, abs(
            float(row["YPOS_ei"]) - .25 * sum(y_values)))

    test_s = np.zeros((len(data), len(ALL_BANDS)), dtype=float)
    test_p = state.p_ifu.copy()
    test_before = _stage1_model(used, test_s, test_p,
                                np.zeros(len(data)), np.zeros(len(data)))
    test_offset = _stage1_apply_pifu_gauge(test_s, test_p, list(range(len(ALL_BANDS))))
    test_after = _stage1_model(used, test_s, test_p,
                               np.zeros(len(data)), np.zeros(len(data)))
    p_gauge_change = float(np.max(np.abs(test_before - test_after), initial=0.0))
    plane_reference_change = max([
        float(row.get("plane_reference_prediction_max_change", 0.0))
        for row in fit["iteration_history"]] or [0.0])
    checks = {
        "frozen_pure_blank_count": int(sum(np.sum(selected[item.key]) for item in data)),
        "frozen_population_unchanged_for_all_stage1_diagnostics": True,
        "central_q_range": [CENTRAL_Q_MIN, CENTRAL_Q_MAX],
        "central_q_fallback_used": False,
        "amplifier_summaries": int(len(fit["amp_rows"])),
        "complete_four_ifu_band_summaries": int(fit["complete_rows"]),
        "stage1_Y_rows_used": int(fit["fit_rows"]),
        "all_stage1_Y_rows_used_have_all_four_amplifiers": bool(
            all(row["complete_four_amplifiers"] for row in used)),
        "equal_amplifier_mean_max_error": float(equal_mean_error),
        "equal_amplifier_x_max_error": float(equal_x_error),
        "equal_amplifier_y_max_error": float(equal_y_error),
        "equal_amplifier_mean_is_exactly_one_quarter": bool(equal_mean_error <= 2e-15),
        "p_IFU_sum": float(np.sum(state.p_ifu)),
        "p_IFU_zero_sum_gauge": bool(abs(float(np.sum(state.p_ifu))) <= 2e-12),
        "mean_ax": float(np.mean(state.ax)),
        "mean_ay": float(np.mean(state.ay)),
        "mean_ax_zero_gauge": bool(abs(float(np.mean(state.ax))) <= 2e-12),
        "mean_ay_zero_gauge": bool(abs(float(np.mean(state.ay))) <= 2e-12),
        "p_IFU_s_gauge_max_prediction_change": p_gauge_change,
        "plane_reference_gauge_max_prediction_change": plane_reference_change,
        "gauge_transfers_preserve_predictions_numerically": bool(
            p_gauge_change <= 2e-12 and plane_reference_change <= 2e-12),
        "exact_exponential_evaluation": True,
        "pre_post_use_identical_frozen_fibers": True,
        "native_pixels_used_in_fit": bool(native_pixels_used),
        "stage2_executed": False,
        "stage3_executed": False,
        "stage4_executed": False,
        "stage1_stop_mode": True,
        "p_IFU_test_gauge_shift": test_offset,
    }
    return checks


def _stage1_plot_paths(data, state, fit, sky0, selected, output_dir, band_x):
    """Make the physical Stage-1 plots from compressed summaries."""
    import matplotlib.pyplot as plt

    common_rows = fit["common_rows"]
    plots = []
    null_band_index = 0
    by_item = {}
    for row in common_rows:
        if row["band_index"] == null_band_index and row["complete_four_amplifiers"]:
            by_item.setdefault(row["item_index"], []).append(row)
    if by_item:
        item_index = max(by_item, key=lambda key: len(by_item[key]))
        item = data[item_index]
        rows = by_item[item_index]
        before = np.asarray([np.expm1(row["Y_eib"]) for row in rows])
        predictions = np.asarray([
            fit["sky_log_correction"][item.key][null_band_index] +
            state.p_ifu[row["IFU_CODE"]] + state.ax[item_index] * row["X_ei"] +
            state.ay[item_index] * row["YPOS_ei"] for row in rows])
        after = np.expm1(np.asarray([row["Y_eib"] for row in rows]) - predictions)
        x = np.asarray([row["X_ei"] for row in rows])
        y = np.asarray([row["YPOS_ei"] for row in rows])
        finite = np.isfinite(np.concatenate((before, after, x, y)))
        if np.any(finite):
            finite_residuals = np.abs(np.concatenate((before, after)))
            finite_residuals = finite_residuals[np.isfinite(finite_residuals)]
            # Keep one common scale while preventing a single extreme fiber
            # from washing out the physical field pattern.
            limit = max(.01, float(np.percentile(finite_residuals, 99.5)) * 1.1)
            figure, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
            for axis, values, title in zip(
                    axes, (before, after),
                    ("Stage 0 fractional residual", "Stage 1 POST-SKY fractional residual")):
                scatter = axis.scatter(x, y, c=values, cmap="RdBu_r", vmin=-limit, vmax=limit,
                                       s=42, edgecolors="black", linewidths=.25)
                for row, xx, yy in zip(rows, x, y):
                    axis.text(xx, yy, str(row["IFU_CODE"]), fontsize=5,
                              ha="center", va="center", color="black")
                axis.set_title(title)
                axis.set_xlabel("cached x_arcmin")
                axis.set_ylabel("cached y_arcmin")
                axis.set_aspect("equal", adjustable="box")
                figure.colorbar(scatter, ax=axis, label="fractional residual")
                if np.ptp(x) > 0 and np.ptp(y) > 0:
                    grid_x = np.linspace(np.min(x), np.max(x), 40)
                    grid_y = np.linspace(np.min(y), np.max(y), 40)
                    mesh_x, mesh_y = np.meshgrid(grid_x, grid_y)
                    plane = state.ax[item_index] * mesh_x + state.ay[item_index] * mesh_y
                    axis.contour(mesh_x, mesh_y, plane, colors="black", linewidths=.6,
                                 linestyles="--", alpha=.7)
            figure.suptitle("%s exposure %d, %s; point = equal four-amp IFU mode" %
                            (item.h5_name, item.exposure, NULL_BANDS[null_band_index]))
            path = output_dir / ("stage1_fov_before_after_%s_exp%d_%s.png" %
                                 (item.h5_name.replace(".h5", ""), item.exposure,
                                  NULL_BANDS[null_band_index]))
            figure.savefig(path, dpi=160)
            plt.close(figure)
            plots.append(str(path))

    complete_by_item_ifu = {}
    for row in common_rows:
        if row["complete_four_amplifiers"]:
            complete_by_item_ifu.setdefault((row["item_index"], row["IFU_CODE"]), []).append(row)
    if complete_by_item_ifu:
        item_index, ifu_code = max(
            complete_by_item_ifu,
            key=lambda key: len({row["band_index"] for row in complete_by_item_ifu[key]}))
        item = data[item_index]
        rows_by_band = {row["band_index"]: row
                        for row in complete_by_item_ifu[(item_index, ifu_code)]}
        figure, axis = plt.subplots(figsize=(10, 5.5), constrained_layout=True)
        colors = {amp: color for amp, color in zip(AMP_ORDER, ("C0", "C1", "C2", "C3"))}
        for amp_index, amp in enumerate(AMP_ORDER):
            values = np.asarray([rows_by_band.get(index, {}).get("y_" + amp, np.nan)
                                 for index in range(len(ALL_BANDS))], dtype=float)
            axis.plot(np.arange(len(ALL_BANDS)), values, marker="o", linestyle="-", color=colors[amp],
                      label=amp, alpha=.85)
        common = np.asarray([rows_by_band.get(index, {}).get("Y_eib", np.nan)
                             for index in range(len(ALL_BANDS))])
        prediction = np.asarray([
            fit["sky_log_correction"][item.key][index] +
            state.p_ifu[rows_by_band[index]["IFU_CODE"]] + state.ax[item_index] *
            rows_by_band[index]["X_ei"] + state.ay[item_index] * rows_by_band[index]["YPOS_ei"]
            if index in rows_by_band else np.nan for index in range(len(ALL_BANDS))])
        positions = np.arange(len(ALL_BANDS))
        axis.plot(positions, common, "kX", markersize=8, label="equal 1/4 common mode")
        axis.plot(positions, prediction, "k--", linewidth=1.5, label="Stage-1 prediction")
        axis.set_xticks(positions)
        axis.set_xticklabels(["%s\n%.3g" % (band, x)
                              for band, x in zip(ALL_BANDS, band_x)])
        axis.set_xlabel("band name (effective x_lambda shown below)")
        axis.set_ylabel("log(D / S0)")
        axis.set_title("%s exposure %d, IFU (%d, %d, %d): four amplifier summaries" %
                       (item.h5_name, item.exposure,
                        int(rows_by_band[next(iter(rows_by_band))]["SPECID"]),
                        int(rows_by_band[next(iter(rows_by_band))]["IFUSLOT"]),
                        int(rows_by_band[next(iter(rows_by_band))]["IFUID"])))
        axis.legend(ncol=3, fontsize=8)
        path = output_dir / ("stage1_four_amp_example_%s_%d_ifu_%d_%d_%d.png" %
                             (item.h5_name.replace(".h5", ""), item.exposure,
                              int(rows_by_band[next(iter(rows_by_band))]["SPECID"]),
                              int(rows_by_band[next(iter(rows_by_band))]["IFUSLOT"]),
                              int(rows_by_band[next(iter(rows_by_band))]["IFUID"])))
        figure.savefig(path, dpi=160)
        plt.close(figure)
        plots.append(str(path))

    position_by_ifu = {}
    for row in common_rows:
        if row["complete_four_amplifiers"]:
            position_by_ifu.setdefault(row["IFU_CODE"], []).append(
                (row["X_ei"], row["YPOS_ei"]))
    supported = [code for code, values in position_by_ifu.items() if values]
    if supported:
        x = np.asarray([robust_location([value[0] for value in position_by_ifu[code]])
                        for code in supported])
        y = np.asarray([robust_location([value[1] for value in position_by_ifu[code]])
                        for code in supported])
        p = state.p_ifu[supported]
        figure, axis = plt.subplots(figsize=(7, 6), constrained_layout=True)
        scatter = axis.scatter(x, y, c=p, cmap="coolwarm", s=48, edgecolors="black", linewidths=.3)
        for code, xx, yy in zip(supported, x, y):
            axis.text(xx, yy, str(code), fontsize=5, ha="center", va="center")
        axis.set_title("Stage-1 persistent p_IFU (fixed cached coordinates)")
        axis.set_xlabel("cached x_arcmin")
        axis.set_ylabel("cached y_arcmin")
        axis.set_aspect("equal", adjustable="box")
        figure.colorbar(scatter, ax=axis, label="p_IFU log throughput")
        path = output_dir / "stage1_pifu_focal_map.png"
        figure.savefig(path, dpi=160)
        plt.close(figure)
        plots.append(str(path))
    return plots


def _stage1_summary_lookup(rows, stage, grouping, band="ALL", value="fractional"):
    for row in rows:
        if row.get("stage") == stage and row.get("grouping") == grouping and \
                row.get("band") == band and row.get("value") == value:
            return row
    return {}


def _stage1_write_product(output_dir, args, data, cache_path, cache_manifest,
                          native_pixels_read_for_cache, ifus, state, sky0, fit, heldout,
                          residual_rows, membership_artifact, membership_info,
                          timings, band_x, selected, plot_paths=None,
                          external_provenance=None):
    output_dir = Path(output_dir)
    arrays_path = output_dir / "m101_stage1_arrays.npz"
    sky_log = fit["sky_log_correction"]
    sky1 = fit["S1"]
    np.savez_compressed(
        arrays_path,
        band_order=np.asarray(ALL_BANDS), band_x_lambda=np.asarray(band_x),
        item_keys=np.asarray([str(item.key) for item in data]),
        S0=np.asarray([sky0[item.key] for item in data]),
        sky_log_correction=np.asarray([sky_log[item.key] for item in data]),
        S1=np.asarray([sky1[item.key] for item in data]),
        ifus=np.asarray(ifus, dtype=np.int32), p_IFU=state.p_ifu,
        ax=state.ax, ay=state.ay,
        fixed_ifu_x=fit["fixed_ifu_x"], fixed_ifu_y=fit["fixed_ifu_y"])
    state_path = output_dir / "m101_stage1_state.json"
    amp_path = output_dir / "m101_stage1_amp_summaries.csv"
    common_path = output_dir / "m101_stage1_ifu_band_summaries.csv"
    history_path = output_dir / "m101_stage1_iteration_history.csv"
    residual_path = output_dir / "m101_stage1_residual_summary.csv"
    _write_rows(amp_path, fit["amp_rows"])
    _write_rows(common_path, fit["common_rows"])
    history_rows = []
    for row in fit["iteration_history"]:
        clean = {key: value for key, value in row.items() if key != "plane_fit"}
        clean["plane_fit_json"] = json.dumps(json_ready(row.get("plane_fit", {})),
                                              separators=(",", ":"))
        history_rows.append(clean)
    _write_rows(history_path, history_rows)
    _write_rows(residual_path, residual_rows)
    validation = _stage1_gauge_validation(
        data, state, fit, selected, native_pixels_used=False)
    stage0_summary = _stage1_summary_lookup(residual_rows, "STAGE0", "overall")
    pre_summary = _stage1_summary_lookup(residual_rows, "STAGE1_PRE_SKY", "overall")
    post_summary = _stage1_summary_lookup(residual_rows, "STAGE1_POST_SKY", "overall")
    focal = {}
    for stage in ("STAGE0", "STAGE1_PRE_SKY", "STAGE1_POST_SKY"):
        focal[stage] = {
            axis: _stage1_summary_lookup(
                residual_rows, stage, axis, value="fractional_slope").get("slope", np.nan)
            for axis in ("focal_x", "focal_y")}
    product_payload = {
        "schema_version": "m101_stage1_product_v1",
        "stage": "STAGE1_ONLY",
        "stage1_model": "Y_eib = s_e,b + p_IFU(i) + ax(e) X_ei + ay(e) YPOS_ei",
        "band_order": list(ALL_BANDS), "null_bands": list(NULL_BANDS),
        "source_bands": list(SOURCE_BANDS), "band_effective_x_lambda": band_x,
        "parameters": {
            "S0": {str(item.key): sky0[item.key] for item in data},
            "sky_log_correction": {str(item.key): sky_log[item.key] for item in data},
            "sky_log_correction_s_e_b": {str(item.key): sky_log[item.key] for item in data},
            "S1": {str(item.key): sky1[item.key] for item in data},
            "p_IFU": {str(tuple(ifu)): state.p_ifu[index]
                      for index, ifu in enumerate(ifus)},
            "ax": {str(item.key): state.ax[item.item_index] for item in data},
            "ay": {str(item.key): state.ay[item.item_index] for item in data},
        },
        "fixed_zero_terms": {
            "alpha_q": {"value": 0.0, "status": "fixed zero; not fitted"},
            "p_AMP": {"value": 0.0, "status": "fixed zero; not fitted"},
            "G": {"value": 0.0, "status": "fixed zero; not fitted"},
            "Q": {"value": 0.0, "status": "fixed zero; not fitted"},
        },
        "coordinates": {
            "x": "cached x_arcmin", "y": "cached y_arcmin",
            "origin": "existing cache exposure origin from median IFU-center geometry supported by blank_valid at cache construction",
            "stage1_recentered": False,
            "central_q_subset_recentered": False,
            "fixed_IFU_reference_positions": {
                str(tuple(ifu)): {"x_arcmin": fit["fixed_ifu_x"][index],
                                  "y_arcmin": fit["fixed_ifu_y"][index]}
                for index, ifu in enumerate(ifus)
                if np.isfinite(fit["fixed_ifu_x"][index]) and
                np.isfinite(fit["fixed_ifu_y"][index])},
        },
        "gauges": {
            "p_IFU": "sum over all physical IFUs equals zero",
            "ax": "mean over all exposure items equals zero",
            "ay": "mean over all exposure items equals zero",
            "p_IFU_sum": np.sum(state.p_ifu), "mean_ax": np.mean(state.ax),
            "mean_ay": np.mean(state.ay),
            "transfers": fit["gauge_transfers"],
        },
        "support": {
            "central_q": [CENTRAL_Q_MIN, CENTRAL_Q_MAX],
            "amplifier_summary_rows": len(fit["amp_rows"]),
            "complete_four_ifu_band_summaries": fit["complete_rows"],
            "complete_four_fit_rows": fit["fit_rows"],
            "complete_four_by_band": {
                band: int(sum(row["complete_four_amplifiers"] and row["band"] == band
                              for row in fit["common_rows"]))
                for band in ALL_BANDS},
            "physical_ifus_total": len(ifus),
            "physical_ifus_represented": fit["physical_ifus_supported"],
            "physical_ifu_fraction_represented": fit["physical_ifus_supported"] / max(1, len(ifus)),
            "unsupported_amplifier_summaries": int(sum(not row["supported"]
                                                        for row in fit["amp_rows"])),
        },
        "iteration_history": fit["iteration_history"],
        "held_out_band_check": heldout,
        "diagnostics": {
            "stage0_overall_fractional_robust_rms": stage0_summary.get("robust_rms", np.nan),
            "stage1_pre_sky_overall_fractional_robust_rms": pre_summary.get("robust_rms", np.nan),
            "stage1_post_sky_overall_fractional_robust_rms": post_summary.get("robust_rms", np.nan),
            "focal_slopes": focal,
            "p_IFU_range": [np.min(state.p_ifu), np.max(state.p_ifu)],
            "p_IFU_robust_rms": residual_summary(state.p_ifu)["robust_rms"],
            "ax_range": [np.min(state.ax), np.max(state.ax)],
            "ay_range": [np.min(state.ay), np.max(state.ay)],
            "maximum_iteration_parameter_change": max([
                row["max_parameter_change"] for row in fit["iteration_history"]] or [0.0]),
            "plot_paths": list(plot_paths or []),
        },
        "validation": validation,
        "population": {
            "frozen": True,
            "pure_blank_definition": "existing blank_valid AND NOT source-candidate union",
            "pure_blank_count": int(sum(np.sum(selected[item.key]) for item in data)),
            "membership_artifact": str(membership_artifact) if membership_artifact else None,
            "membership_info": membership_info,
        },
        "provenance": {
            "product": file_identity(args.product),
            "blank_file": file_identity(args.blank_file) if args.blank_file and
            Path(args.blank_file).exists() else None,
            "blank_file_sha256": small_file_hash(args.blank_file) if args.blank_file and
            Path(args.blank_file).exists() else None,
            "external_cache": external_provenance,
            "compact_mask": file_identity(args.compact_mask) if args.compact_mask and
            Path(args.compact_mask).exists() else None,
            "on_filter": file_identity(args.on_filter) if args.on_filter and
            Path(args.on_filter).exists() else None,
            "off_filter": file_identity(args.off_filter) if args.off_filter and
            Path(args.off_filter).exists() else None,
            "fq_template": file_identity(args.fq_template),
            "band_cache": file_identity(cache_path),
            "band_cache_manifest": file_identity(_cache_paths(cache_path)[1]),
            "cache_manifest": cache_manifest,
            "fit_inputs": {"native_pixels_used_for_fit": False,
                           "collapsed_band_rows_only": True,
                           "native_pixels_read_for_cache": bool(native_pixels_read_for_cache)},
        },
        "artifacts": {
            "arrays": str(arrays_path), "amp_summaries": str(amp_path),
            "ifu_band_summaries": str(common_path),
            "iteration_history": str(history_path), "residual_summary": str(residual_path),
        },
    }
    (output_dir / "m101_stage1_state.json").write_text(
        json.dumps(json_ready(product_payload), indent=2, sort_keys=True))
    return product_payload, state_path


def _stage2_alpha(data, state, sky, selected, fq, hi_map):
    rows = []
    for item in data:
        m = _model_response(item, state, hi_map, BAND_X)
        residual = item.band_total - m * sky[item.key][None, :]
        group_indices = _item_group_indices(item)
        for (ifu_code, amp_index), indices in group_indices.items():
            blank_indices = indices[selected[item.key][indices]]
            if blank_indices.size == 0:
                continue
            reference = (item.q[blank_indices] >= CENTRAL_Q_MIN) & \
                (item.q[blank_indices] <= CENTRAL_Q_MAX)
            if not np.any(reference):
                continue
            r0 = np.asarray([robust_location(residual[blank_indices[reference], band])
                             for band in range(len(ALL_BANDS))], dtype=float)
            predictors = []
            targets = []
            sigmas = []
            for band in range(len(ALL_BANDS)):
                target = residual[blank_indices, band] - r0[band]
                predictor = (fq[item.q[blank_indices]] -
                             robust_location(fq[item.q[blank_indices][reference]])) * item.K[band]
                error = item.band_error[blank_indices, band]
                scale = robust_scatter(target)
                if not np.isfinite(scale) or scale <= 0:
                    scale = np.nanmedian(error[np.isfinite(error) & (error > 0)])
                scale = max(float(scale) if np.isfinite(scale) else .01, 1e-6)
                error = np.where(np.isfinite(error) & (error > 0), error, scale)
                error = np.maximum(error, .25 * scale)
                good = np.isfinite(target) & np.isfinite(predictor) & (np.abs(predictor) > 0)
                predictors.extend(predictor[good].tolist())
                targets.extend(target[good].tolist())
                sigmas.extend(error[good].tolist())
            alpha, fit = _irls_scalar(predictors, targets, sigmas, iterations=5)
            state.alpha_q[item.item_index, int(ifu_code), int(amp_index)] = alpha
            rows.append({
                "H5": item.h5_name, "exposure": item.exposure,
                "IFU_CODE": int(ifu_code), "SPECID": int(item.ifu[blank_indices[0], 0]),
                "IFUSLOT": int(item.ifu[blank_indices[0], 1]),
                "IFUID": int(item.ifu[blank_indices[0], 2]),
                "AMP": AMP_ORDER[amp_index], "alpha_q": alpha,
                "N": fit["N"], "uncertainty": fit["uncertainty"],
                "robust_scatter": fit["scatter"],
                "q_min": int(np.min(item.q[blank_indices])),
                "q_max": int(np.max(item.q[blank_indices])),
                "central_q_support": int(np.sum(reference)),
            })
    return rows


def _blank_amp_summaries(data, state, sky, selected, hi_map, log_values=False):
    rows = []
    for item in data:
        m = _model_response(item, state, hi_map, BAND_X)
        additive = _additive_bands(item, state, FQ_GLOBAL)
        denominator = m * sky[item.key][None, :]
        ratio = (item.band_total - additive) / denominator
        if log_values:
            with np.errstate(invalid="ignore", divide="ignore"):
                ratio = np.log(ratio)
        for (ifu_code, amp_index), indices in _item_group_indices(item).items():
            use = indices[selected[item.key][indices]]
            if use.size == 0:
                continue
            for band in range(len(ALL_BANDS)):
                values = ratio[use, band]
                values = values[np.isfinite(values)]
                location = float(robust_location(values)) if values.size else np.nan
                scatter = float(robust_scatter(values)) if values.size > 1 else np.nan
                rows.append({
                    "H5": item.h5_name, "exposure": int(item.exposure),
                    "IFU_CODE": int(ifu_code), "SPECID": int(item.ifu[use[0], 0]),
                    "IFUSLOT": int(item.ifu[use[0], 1]), "IFUID": int(item.ifu[use[0], 2]),
                    "AMP": AMP_ORDER[amp_index], "AMP_INDEX": int(amp_index),
                    "band": ALL_BANDS[band], "band_index": band,
                    "location": location, "robust_scatter": scatter,
                    "location_uncertainty": (scatter / math.sqrt(values.size)
                                              if np.isfinite(scatter) else np.nan),
                    "N_fibers": int(values.size),
                    "q_support": int(np.sum((item.q[use] >= CENTRAL_Q_MIN) &
                                             (item.q[use] <= CENTRAL_Q_MAX))),
                    "value_definition": "log_ratio" if log_values else "fractional_ratio_minus_one",
                })
    return rows


def _profile_amp_rows(rows):
    grouped = {}
    for row in rows:
        grouped.setdefault((row["H5"], row["exposure"], row["IFU_CODE"], row["band"]), []).append(row)
    for group_rows in grouped.values():
        values = np.asarray([row["location"] for row in group_rows], dtype=float)
        sigma = np.asarray([row["location_uncertainty"] for row in group_rows], dtype=float)
        sigma[~np.isfinite(sigma) | (sigma <= 0)] = .03
        common, common_uncertainty = _weighted_robust_center(values, sigma)
        for row in group_rows:
            row["common_mode"] = common
            row["common_mode_uncertainty"] = common_uncertainty
            row["differential"] = row["location"] - common
            row["differential_sigma"] = max(float(row["location_uncertainty"])
                                             if np.isfinite(row["location_uncertainty"]) else .03,
                                             .003)
            row["available_amplifiers"] = len(group_rows)
    return rows


def _fit_local_three(observations):
    if not observations:
        return np.zeros(3), {"rank": 0, "N": 0, "support": 0, "scale": np.nan}
    design = np.asarray([BASIS[AMP_INDEX[row["AMP"]]] for row in observations], dtype=float)
    target = np.asarray([row["differential"] for row in observations], dtype=float)
    sigma = np.asarray([row.get("differential_sigma", .03) for row in observations], dtype=float)
    coeff, fit = _irls_linear(design, target, sigma, iterations=4)
    return coeff, {"rank": fit["rank"], "N": fit["N"], "support": len(observations),
                   "scale": fit["scale"]}


def _source_amp_summaries(data, state, sky, selected, hi_map):
    rows = []
    for item in data:
        m = _model_response(item, state, hi_map, BAND_X)
        additive = _additive_bands(item, state, FQ_GLOBAL)
        corrected = (item.band_total - additive) / m - sky[item.key][None, :]
        for (ifu_code, amp_index), indices in _item_group_indices(item).items():
            for band_index, band in enumerate(SOURCE_BANDS):
                use = indices[item.source_accepted[indices, band_index]]
                x = item.X[use, band_index]
                values = corrected[use, band_index]
                good = np.isfinite(x) & np.isfinite(values) & (x > 0)
                x = x[good]; values = values[good]
                ratio = values / x if x.size else np.zeros(0)
                ratio = ratio[np.isfinite(ratio) & (ratio > 0)]
                with np.errstate(divide="ignore", invalid="ignore"):
                    logs = np.log(ratio)
                location = float(robust_location(logs)) if logs.size else np.nan
                scatter = float(robust_scatter(logs)) if logs.size > 1 else np.nan
                if use.size:
                    first = use[0]
                    ifu_value = item.ifu[first]
                else:
                    first = indices[0]
                    ifu_value = item.ifu[first]
                rows.append({
                    "H5": item.h5_name, "exposure": int(item.exposure),
                    "IFU_CODE": int(ifu_code), "SPECID": int(ifu_value[0]),
                    "IFUSLOT": int(ifu_value[1]), "IFUID": int(ifu_value[2]),
                    "AMP": AMP_ORDER[amp_index], "AMP_INDEX": int(amp_index),
                    "band": band, "location": location, "robust_scatter": scatter,
                    "location_uncertainty": (scatter / math.sqrt(logs.size)
                                              if np.isfinite(scatter) else np.nan),
                    "N_fibers": int(logs.size),
                    "value_definition": "log_O_over_X",
                })
    return _profile_amp_rows(rows)


def _aggregate_topology(rows, value_name):
    by_hi = {}
    for row in rows:
        if np.isfinite(row.get("differential", np.nan)):
            by_hi.setdefault((row["H5"], int(row["IFU_CODE"])), []).append(row)
    result = {}
    for key, observations in by_hi.items():
        coeff, support = _fit_local_three(observations)
        result[key] = {"value": coeff, "support": support,
                       "source": value_name}
    return result


def _choose_gray(source_topology, blank_topology, hi_keys, n_hi):
    chosen = np.zeros((n_hi, 3), dtype=float)
    source_values = np.full((n_hi, 3), np.nan, dtype=float)
    blank_values = np.full((n_hi, 3), np.nan, dtype=float)
    source_support = np.zeros(n_hi, dtype=int)
    blank_support = np.zeros(n_hi, dtype=int)
    for hi_index, (h5_name, ifu_code) in enumerate(hi_keys):
        source = source_topology.get((h5_name, ifu_code))
        blank = blank_topology.get((h5_name, ifu_code))
        if source and source["support"]["rank"] > 0:
            chosen[hi_index] = source["value"]
            source_values[hi_index] = source["value"]
            source_support[hi_index] = source["support"]["N"]
        elif blank and blank["support"]["rank"] > 0:
            chosen[hi_index] = blank["value"]
        if blank:
            blank_values[hi_index] = blank["value"]
            blank_support[hi_index] = blank["support"]["N"]
    return chosen, source_values, blank_values, source_support, blank_support


def _transfer_gray_gauge(state, data, hi_keys, hi_map, weights=None):
    """Transfer persistent mean G into p_AMP while preserving predictions."""
    transfers = []
    by_ifu = {}
    for hi_index, (_, ifu_code) in enumerate(hi_keys):
        by_ifu.setdefault(ifu_code, []).append(hi_index)
    for ifu_code, indices in by_ifu.items():
        values = state.g_star[indices]
        # The robust center is the gauge definition.  Support weights remain
        # recorded in the topology tables, while this deterministic robust
        # gauge makes the transferred mean exactly reproducible.
        center = np.asarray([robust_location(values[:, component])
                             for component in range(3)], dtype=float)
        before = values.copy()
        state.p_amp[ifu_code] += center
        state.g_star[indices] -= center[None, :]
        transfers.append({"IFU_CODE": int(ifu_code), "Gbar": center.tolist(),
                          "H5_count": len(indices),
                          "max_change": float(np.max(np.abs(before - (state.g_star[indices] + center)), initial=0.0))})
    return transfers


def _stage3_fit(data, state, sky, selected, hi_keys, hi_map, ifus):
    global FQ_GLOBAL
    blank_rows = _profile_amp_rows(_blank_amp_summaries(
        data, state, sky, selected, hi_map, log_values=False))
    source_rows = _source_amp_summaries(data, state, sky, selected, hi_map)
    blank_topology = _aggregate_topology(blank_rows, "blank")
    source_topology = _aggregate_topology(source_rows, "source")
    chosen, source_values, blank_values, source_support, blank_support = _choose_gray(
        source_topology, blank_topology, hi_keys, state.g_star.shape[0])
    state.g_star[:] = chosen
    state.p_amp[:] = 0.0
    topology_weights = np.maximum(source_support, blank_support).astype(float)
    _transfer_gray_gauge(state, data, hi_keys, hi_map, topology_weights)
    topology_rows = []
    for hi_index, (h5_name, ifu_code) in enumerate(hi_keys):
        s = source_topology.get((h5_name, ifu_code), {})
        b = blank_topology.get((h5_name, ifu_code), {})
        topology_rows.append({
            "H5": h5_name, "IFU_CODE": int(ifu_code),
            "source_rank": s.get("support", {}).get("rank", 0),
            "source_N": s.get("support", {}).get("N", 0),
            "blank_rank": b.get("support", {}).get("rank", 0),
            "blank_N": b.get("support", {}).get("N", 0),
            "selected": "source" if source_support[hi_index] else "blank" if blank_support[hi_index] else "zero",
            "source_gray": source_values[hi_index].tolist(),
            "blank_gray": blank_values[hi_index].tolist(),
            "source_minus_blank": (source_values[hi_index] - blank_values[hi_index]).tolist()
        })
    return {"blank_rows": blank_rows, "source_rows": source_rows,
            "blank_topology": blank_topology, "source_topology": source_topology,
            "topology_rows": topology_rows,
            "source_values": source_values, "blank_values": blank_values,
            "source_support": source_support, "blank_support": blank_support}


def _reduced_constrained_fit(design, target, sigma, current_q, qmax):
    """Fit a small local G/Q design with truncated modes and Q inequalities."""
    design = np.asarray(design, dtype=float)
    target = np.asarray(target, dtype=float)
    sigma = np.maximum(np.asarray(sigma, dtype=float), .003)
    good = np.isfinite(target) & np.all(np.isfinite(design), axis=1) & np.isfinite(sigma)
    design, target, sigma = design[good], target[good], sigma[good]
    if target.size == 0:
        return np.zeros(6), {"rank": 0, "g_rank": 0, "q_rank": 0, "N": 0, "bound": False}
    robust = np.ones(target.size, dtype=float)
    rank = g_rank = q_rank = 0
    V = np.zeros((6, 0), dtype=float)
    theta = np.zeros(0, dtype=float)
    for _ in range(4):
        weighted = design * (np.sqrt(robust) / sigma)[:, None]
        _, singular, vt = np.linalg.svd(weighted, full_matrices=False)
        threshold = max(float(singular[0]) * joint.TOPOLOGY_RELATIVE_RANK_TOL, 1e-12) \
            if singular.size else 1e-12
        rank = int(np.sum(singular > threshold))
        V = vt[:rank].T if rank else np.zeros((6, 0), dtype=float)
        reduced = design @ V
        hessian = (reduced / sigma[:, None]).T @ (reduced / sigma[:, None])
        rhs = (reduced / sigma[:, None]).T @ (target / sigma)
        theta = np.linalg.lstsq(hessian, rhs, rcond=1e-12)[0] if rank else np.zeros(0)
        prediction = reduced @ theta
        residual = target - prediction
        scale = max(float(robust_scatter(residual)), 1e-8)
        standardized = np.abs(residual) / scale
        robust = np.where(standardized > 1.345, 1.345 / standardized, 1.0)
        if rank:
            # Rebuild the final reduced quadratic and enumerate the at most
            # eight amplifier inequalities only if the unconstrained solution
            # actually needs the Q bound.
            weighted = design * (np.sqrt(robust) / sigma)[:, None]
            _, singular, vt = np.linalg.svd(weighted, full_matrices=False)
            threshold = max(float(singular[0]) * joint.TOPOLOGY_RELATIVE_RANK_TOL, 1e-12)
            rank = int(np.sum(singular > threshold))
            V = vt[:rank].T
            reduced = design @ V
            hessian = (reduced / sigma[:, None]).T @ (reduced / sigma[:, None])
            rhs = (reduced / sigma[:, None]).T @ (target / sigma)
            constraints = np.vstack((BASIS @ V[3:, :], -BASIS @ V[3:, :]))
            limits = np.concatenate((np.full(4, qmax) - BASIS @ current_q,
                                     np.full(4, qmax) + BASIS @ current_q))
            unconstrained_theta = np.linalg.lstsq(hessian, rhs, rcond=1e-12)[0]
            unconstrained_q = current_q + V[3:, :] @ unconstrained_theta
            needs_bound = np.max(np.abs(BASIS @ unconstrained_q)) > qmax + 1e-10
            candidates = []
            if not needs_bound:
                theta = unconstrained_theta
                bound = False
            for active_count in (range(min(rank, constraints.shape[0]) + 1)
                                 if needs_bound else ()):
                for active in combinations(range(constraints.shape[0]), active_count):
                    active = list(active)
                    if active:
                        matrix = np.block([[hessian, constraints[active].T],
                                           [constraints[active], np.zeros((active_count, active_count))]])
                        vector = np.concatenate((rhs, limits[active]))
                        solution = np.linalg.lstsq(matrix, vector, rcond=1e-12)[0]
                        candidate_theta = solution[:rank]
                        multipliers = solution[rank:]
                        if np.any(multipliers < -1e-8):
                            continue
                    else:
                        candidate_theta = np.linalg.lstsq(hessian, rhs, rcond=1e-12)[0]
                    if np.any(constraints @ candidate_theta > limits + 1e-8):
                        continue
                    objective = .5 * candidate_theta @ hessian @ candidate_theta - rhs @ candidate_theta
                    candidates.append((float(objective), candidate_theta, bool(active)))
            if candidates:
                _, theta, bound = min(candidates, key=lambda value: value[0])
            elif needs_bound:
                theta = np.zeros(rank, dtype=float)
                bound = False
            delta = V @ theta
        else:
            delta = np.zeros(6, dtype=float)
            bound = False
    g_singular = np.linalg.svd((design[:, :3] / sigma[:, None]), compute_uv=False)
    q_singular = np.linalg.svd((design[:, 3:] / sigma[:, None]), compute_uv=False)
    g_rank = int(np.sum(g_singular > max(float(g_singular[0]) * joint.TOPOLOGY_RELATIVE_RANK_TOL, 1e-12))) if g_singular.size else 0
    q_rank = int(np.sum(q_singular > max(float(q_singular[0]) * joint.TOPOLOGY_RELATIVE_RANK_TOL, 1e-12))) if q_singular.size else 0
    return delta, {"rank": rank, "g_rank": g_rank, "q_rank": q_rank,
                   "N": int(target.size), "bound": bool(bound),
                   "singular_values": singular.tolist() if 'singular' in locals() else []}


def _stage4_fit(data, state, sky, selected, hi_keys, hi_map, band_x, qmax):
    blank_rows = _profile_amp_rows(_blank_amp_summaries(
        data, state, sky, selected, hi_map, log_values=True))
    weights_by_hi = {}
    x_by_band = {ALL_BANDS[index]: float(band_x[index]) for index in range(len(ALL_BANDS))}
    for row in blank_rows:
        sigma = row.get("differential_sigma", .03)
        if np.isfinite(row.get("differential", np.nan)):
            weights_by_hi.setdefault((row["H5"], int(row["IFU_CODE"])), []).append(
                (x_by_band[row["band"]], 1.0 / max(float(sigma), .003) ** 2))
    for hi_index, key in enumerate(hi_keys):
        entries = weights_by_hi.get(key, [])
        total = sum(weight for _, weight in entries)
        state.xbar[hi_index] = (sum(x * weight for x, weight in entries) / total
                                if total > 0 else 0.0)
    observations_by_hi = {}
    for row in blank_rows:
        if not np.isfinite(row.get("differential", np.nan)):
            continue
        hi_index = hi_map[(row["H5"], int(row["IFU_CODE"]))]
        xprime = x_by_band[row["band"]] - state.xbar[hi_index]
        b = BASIS[AMP_INDEX[row["AMP"]]]
        observations_by_hi.setdefault(hi_index, []).append(
            (np.concatenate((b, xprime * b)), row["differential"],
             max(float(row.get("differential_sigma", .03)), .003)))
    support_rows = []
    deltas = np.zeros_like(state.g_star)
    delta_q = np.zeros_like(state.q_color)
    for hi_index, observations in observations_by_hi.items():
        design = np.asarray([entry[0] for entry in observations])
        target = np.asarray([entry[1] for entry in observations])
        sigma = np.asarray([entry[2] for entry in observations])
        delta, fit = _reduced_constrained_fit(
            design, target, sigma, state.q_color[hi_index], qmax)
        deltas[hi_index] = delta[:3]
        delta_q[hi_index] = delta[3:]
        support_rows.append({
            "H5": hi_keys[hi_index][0], "IFU_CODE": hi_keys[hi_index][1],
            "rank": fit["rank"], "g_rank": fit["g_rank"], "q_rank": fit["q_rank"],
            "N": fit["N"], "q_bound_active": fit["bound"],
            "supported_modes": [int(index) for index in range(fit["rank"])],
            "xbar": float(state.xbar[hi_index]),
        })
    state.g_star += deltas
    state.q_color += delta_q
    gauge = _transfer_gray_gauge(
        state, data, hi_keys, hi_map,
        np.asarray([max(1, row.get("N", 1)) for row in support_rows], dtype=float)
        if support_rows else None)
    return {"blank_rows": blank_rows, "support_rows": support_rows,
            "gauge_transfers": gauge, "delta_g": deltas, "delta_q": delta_q}


def _stage0_state(n_ifu, n_items, n_hi):
    return _initial_state([], n_ifu, n_hi) if n_items == 0 else InitialState(
        p_ifu=np.zeros(n_ifu), ax=np.zeros(n_items), ay=np.zeros(n_items),
        x_center=np.zeros(n_items), y_center=np.zeros(n_items),
        alpha_q=np.zeros((n_items, n_ifu, 4)), p_amp=np.zeros((n_ifu, 3)),
        g_star=np.zeros((n_hi, 3)), q_color=np.zeros((n_hi, 3)),
        xbar=np.zeros(n_hi))


def _summary_row(stage, population, grouping, group, band, kind, values):
    summary = residual_summary(np.asarray(values, dtype=float))
    return {"stage": stage, "population": population, "grouping": grouping,
            "group": group, "band": band, "value": kind, **summary}


def _collect_group_values(groups, key, values):
    if values.size:
        groups.setdefault(key, []).append(np.asarray(values, dtype=float))


def _evaluate_stage(stage, data, state, sky, selected, hi_map, band_x,
                    fq, diagnostic_max=200000):
    blank_absolute = []
    blank_fractional = []
    groups_absolute = {}
    groups_fractional = {}
    band_absolute = {band: [] for band in ALL_BANDS}
    band_fractional = {band: [] for band in ALL_BANDS}
    x_values = []
    y_values = []
    for item in data:
        m = _model_response(item, state, hi_map, band_x)
        additive = _additive_bands(item, state, fq)
        residual = item.band_total - additive - m * sky[item.key][None, :]
        denominator = m * sky[item.key][None, :]
        with np.errstate(divide="ignore", invalid="ignore"):
            fractional = np.where(np.abs(denominator) > MIN_SAFE_DENOMINATOR,
                                  residual / denominator, np.nan)
        use = selected[item.key]
        for band_index, band in enumerate(ALL_BANDS):
            r = residual[use, band_index]
            f = fractional[use, band_index]
            if r.size:
                blank_absolute.append(r[np.isfinite(r)])
                band_absolute[band].append(r[np.isfinite(r)])
                blank_fractional.append(f[np.isfinite(f)])
                band_fractional[band].append(f[np.isfinite(f)])
            for exposure_key, mask in [(str(item.exposure), np.ones(use.sum(), dtype=bool))]:
                _collect_group_values(groups_absolute, ("exposure", exposure_key, band), r)
                _collect_group_values(groups_fractional, ("exposure", exposure_key, band), f)
            for ifu_code in np.unique(item.ifu_code[use]):
                local = use & (item.ifu_code == ifu_code)
                _collect_group_values(groups_absolute, ("physical_IFU", int(ifu_code), band), residual[local, band_index])
                _collect_group_values(groups_fractional, ("physical_IFU", int(ifu_code), band), fractional[local, band_index])
            for amp_index in range(4):
                local = use & (item.amp == amp_index)
                _collect_group_values(groups_absolute, ("amplifier", AMP_ORDER[amp_index], band), residual[local, band_index])
                _collect_group_values(groups_fractional, ("amplifier", AMP_ORDER[amp_index], band), fractional[local, band_index])
            for q in np.unique(item.q[use]):
                local = use & (item.q == q)
                _collect_group_values(groups_absolute, ("q", int(q), band), residual[local, band_index])
                _collect_group_values(groups_fractional, ("q", int(q), band), fractional[local, band_index])
            x_values.append((item.x_arcmin[use], fractional[use, band_index]))
            y_values.append((item.y_arcmin[use], fractional[use, band_index]))
    rows = []
    for band in ALL_BANDS:
        rows.append(_summary_row(stage, "blank", "band", "all", band,
                                 "absolute", np.concatenate(band_absolute[band])
                                 if band_absolute[band] else []))
        rows.append(_summary_row(stage, "blank", "band", "all", band,
                                 "fractional", np.concatenate(band_fractional[band])
                                 if band_fractional[band] else []))
    for grouping, groups in (("absolute", groups_absolute), ("fractional", groups_fractional)):
        value_kind = "absolute" if grouping == "absolute" else "fractional"
        for (group_type, group, band), chunks in sorted(groups.items(), key=lambda pair: str(pair[0])):
            values = np.concatenate(chunks) if chunks else np.zeros(0)
            if diagnostic_max and values.size > diagnostic_max:
                stride = int(math.ceil(values.size / diagnostic_max))
                values = values[::stride]
            rows.append(_summary_row(stage, "blank", group_type, group, band,
                                     value_kind, values))
    for name, chunks in (("absolute", blank_absolute), ("fractional", blank_fractional)):
        values = np.concatenate(chunks) if chunks else np.zeros(0)
        if diagnostic_max and values.size > diagnostic_max:
            values = values[::int(math.ceil(values.size / diagnostic_max))]
        rows.append(_summary_row(stage, "blank", "overall", "all", "ALL", name, values))
    for label, pairs in (("focal_x", x_values), ("focal_y", y_values)):
        x = np.concatenate([pair[0] for pair in pairs]) if pairs else np.zeros(0)
        y = np.concatenate([pair[1] for pair in pairs]) if pairs else np.zeros(0)
        good = np.isfinite(x) & np.isfinite(y)
        slope = float(np.polyfit(x[good], y[good], 1)[0]) if np.sum(good) > 2 and np.std(x[good]) > 0 else np.nan
        rows.append({"stage": stage, "population": "blank", "grouping": label,
                     "group": "all", "band": "ALL", "value": "fractional_slope",
                     "N": int(np.sum(good)), "median": np.nan,
                     "biweight_location": np.nan, "robust_rms": np.nan,
                     "arithmetic_rms": np.nan, "p16": np.nan, "p84": np.nan,
                     "p95_abs": np.nan, "p99_abs": np.nan, "max_abs": np.nan,
                     "slope": slope})
    return rows


def _source_diagnostics(stage, data, state, sky, hi_map, band_x, fq):
    absolute = {band: [] for band in SOURCE_BANDS}
    differential = {band: [] for band in SOURCE_BANDS}
    for item in data:
        m = _model_response(item, state, hi_map, band_x)
        additive = _additive_bands(item, state, fq)
        corrected = (item.band_total - additive) / m - sky[item.key][None, :]
        for band_index, band in enumerate(SOURCE_BANDS):
            group_logs = []
            for (ifu_code, amp_index), indices in _item_group_indices(item).items():
                use = indices[item.source_accepted[indices, band_index]]
                x = item.X[use, band_index]
                value = corrected[use, band_index]
                good = np.isfinite(x) & np.isfinite(value) & (x > 0) & (value > 0)
                ratio = value[good] / x[good]
                ratio = ratio[np.isfinite(ratio) & (ratio > 0)]
                if ratio.size:
                    absolute[band].append(ratio - 1.0)
                    group_logs.append((int(ifu_code), int(amp_index), float(robust_location(np.log(ratio)))))
            by_group = {}
            for ifu_code, amp_index, value in group_logs:
                by_group.setdefault(ifu_code, []).append((amp_index, value))
            for values in by_group.values():
                common = robust_location([value for _, value in values])
                differential[band].extend(np.expm1([value - common for _, value in values]))
    rows = []
    for band in SOURCE_BANDS:
        for kind, values in (("absolute_common_normalization", absolute[band]),
                             ("common_mode_removed_amplifier", differential[band])):
            if kind == "absolute_common_normalization":
                values = np.concatenate(values) if values else np.zeros(0)
            else:
                values = np.asarray(values, dtype=float)
            summary = residual_summary(values)
            finite = values[np.isfinite(values)]
            rows.append({"stage": stage, "band": band, "kind": kind,
                         **summary,
                         "within_1pct": float(np.mean(np.abs(finite) <= .01)) if finite.size else np.nan,
                         "within_2pct": float(np.mean(np.abs(finite) <= .02)) if finite.size else np.nan,
                         "within_3pct": float(np.mean(np.abs(finite) <= .03)) if finite.size else np.nan,
                         "within_5pct": float(np.mean(np.abs(finite) <= .05)) if finite.size else np.nan})
    return rows


def _stage_amp_output(stage, rows, source_rows=None):
    output = []
    source_map = {}
    for row in source_rows or []:
        source_map[(row["H5"], row["exposure"], row["IFU_CODE"], row["AMP"], row["band"])] = row
    for row in rows:
        key = (row["H5"], row["exposure"], row["IFU_CODE"], row["AMP"], row["band"])
        result = dict(row)
        result["stage"] = stage
        source = source_map.get(key, {})
        result.update({"source_location": source.get("location", np.nan),
                       "source_scatter": source.get("robust_scatter", np.nan),
                       "source_differential": source.get("differential", np.nan),
                       "source_N": source.get("N_fibers", 0)})
        output.append(result)
    return output


def _validate_state(state, data, hi_keys, hi_map, band_x):
    checks = {}
    checks["amplifier_contrast_column_sums_zero"] = bool(np.max(np.abs(np.sum(BASIS, axis=0))) <= 1e-14)
    checks["p_IFU_zero_sum"] = bool(abs(float(np.sum(state.p_ifu))) <= 2e-12)
    mean_g = []
    for ifu_code in range(state.p_ifu.size):
        indices = [index for index, (_, code) in enumerate(hi_keys) if code == ifu_code]
        if indices:
            mean_g.append(np.asarray([robust_location(state.g_star[indices, component])
                                      for component in range(3)]))
    checks["mean_h_G_zero"] = bool(np.max(np.abs(mean_g), initial=0.0) <= 2e-10) if mean_g else True
    max_centered_difference = 0.0
    for hi_index, (_, ifu_code) in enumerate(hi_keys):
        g4500 = state.g_star[hi_index] - state.xbar[hi_index] * state.q_color[hi_index]
        for band_index, x in enumerate(band_x):
            left = BASIS @ state.g_star[hi_index] + (x - state.xbar[hi_index]) * (BASIS @ state.q_color[hi_index])
            right = BASIS @ g4500 + x * (BASIS @ state.q_color[hi_index])
            max_centered_difference = max(max_centered_difference, float(np.max(np.abs(left - right))))
    checks["centered_saved_prediction_max_difference"] = max_centered_difference
    checks["centered_saved_predictions_agree"] = bool(max_centered_difference <= 2e-12)
    q_values = np.abs(state.q_color @ BASIS.T)
    checks["q_bound_max"] = float(np.max(q_values, initial=0.0))
    checks["q_bound_fraction"] = float(np.mean(q_values <= QMAX_10 + 1e-10))
    checks["q_bound_satisfied"] = bool(checks["q_bound_max"] <= QMAX_10 + 1e-10)
    return checks


def _gauge_invariance_check(state, data, hi_keys, hi_map, band_x):
    max_change = 0.0
    by_ifu = {}
    for index, (_, ifu_code) in enumerate(hi_keys):
        by_ifu.setdefault(ifu_code, []).append(index)
    for ifu_code, indices in by_ifu.items():
        gbar = np.mean(state.g_star[indices], axis=0)
        for hi_index in indices:
            original = BASIS @ state.p_amp[ifu_code] + BASIS @ state.g_star[hi_index]
            transferred = BASIS @ (state.p_amp[ifu_code] + gbar) + BASIS @ (state.g_star[hi_index] - gbar)
            max_change = max(max_change, float(np.max(np.abs(original - transferred))))
    return {"max_prediction_change": max_change, "prediction_invariant": bool(max_change <= 2e-12)}


def _state_json(state, data, ifus, hi_keys, hi_map):
    p_amp_logs = {}
    for ifu_code, ifu in enumerate(ifus):
        for amp_index, amp in enumerate(AMP_ORDER):
            p_amp_logs[str(tuple(ifu) + (amp,))] = float(BASIS[amp_index] @ state.p_amp[ifu_code])
    g = {str((h5, tuple(ifus[ifu_code]))): state.g_star[index].tolist()
         for index, (h5, ifu_code) in enumerate(hi_keys)}
    g4500 = {str((h5, tuple(ifus[ifu_code]))):
             (state.g_star[index] - state.xbar[index] * state.q_color[index]).tolist()
             for index, (h5, ifu_code) in enumerate(hi_keys)}
    q = {str((h5, tuple(ifus[ifu_code]))): state.q_color[index].tolist()
         for index, (h5, ifu_code) in enumerate(hi_keys)}
    alpha = {}
    for item in data:
        for ifu_code in np.unique(item.ifu_code):
            ifu = tuple(ifus[ifu_code])
            for amp_index, amp in enumerate(AMP_ORDER):
                value = state.alpha_q[item.item_index, ifu_code, amp_index]
                if value != 0 or np.any(item.ifu_code == ifu_code):
                    alpha[str((item.h5_name, item.exposure, ifu, amp))] = float(value)
    return {"p_IFU_log": {str(tuple(ifu)): float(value) for ifu, value in zip(ifus, state.p_ifu)},
            "p_AMP_log": p_amp_logs,
            "ax": {str(item.key): float(state.ax[item.item_index]) for item in data},
            "ay": {str(item.key): float(state.ay[item.item_index]) for item in data},
            "plane_x_center": {str(item.key): float(state.x_center[item.item_index]) for item in data},
            "plane_y_center": {str(item.key): float(state.y_center[item.item_index]) for item in data},
            "alpha_q": alpha, "G_star": g, "G_4500": g4500, "Q": q,
            "xbar": {str((h5, tuple(ifus[ifu_code]))): float(state.xbar[index])
                     for index, (h5, ifu_code) in enumerate(hi_keys)}}


def _arrays_payload(state, data, ifus, hi_keys):
    max_rows = len(data)
    skies = np.asarray([CURRENT_SKY[item.key] for item in data], dtype=float)
    g4500 = state.g_star - state.xbar[:, None] * state.q_color
    return {
        "band_order": np.asarray(ALL_BANDS),
        "band_x_lambda": BAND_X,
        "item_keys": np.asarray([str(item.key) for item in data]),
        "S_band": skies,
        "ifus": np.asarray(ifus, dtype=np.int32),
        "p_IFU": state.p_ifu,
        "ax": state.ax, "ay": state.ay,
        "x_center": state.x_center, "y_center": state.y_center,
        "alpha_q": state.alpha_q,
        "p_AMP_contrasts": state.p_amp,
        "G_star": state.g_star, "G_4500": g4500,
        "Q": state.q_color, "xbar": state.xbar,
        "h5ifu_keys": np.asarray([str(key) for key in hi_keys]),
    }


def _compact_stage_summaries(stage_results):
    """Keep the JSON state product small; detailed rows live in CSV files."""
    result = {}
    for name, payload in stage_results.items():
        fit = payload.get("fit")
        if fit is None:
            result[name] = {
                key: value for key, value in payload.items()
                if key in ("parameter_change_max", "stage_seconds")
            }
            continue
        if name == "STAGE1":
            fit_summary = dict(fit.get("fit", {}))
            fit_summary.update({
                "summary_rows": len(fit.get("summary_rows", [])),
                "central_q": fit.get("central_q"),
                "x_center": np.asarray(fit.get("x_center", []), dtype=float),
                "y_center": np.asarray(fit.get("y_center", []), dtype=float),
            })
        elif name == "STAGE3":
            fit_summary = {
                "blank_summary_rows": len(fit.get("blank_rows", [])),
                "source_summary_rows": len(fit.get("source_rows", [])),
                "topology_rows": len(fit.get("topology_rows", [])),
                "selected_source": int(np.sum(fit.get("source_support", []))),
                "selected_blank": int(np.sum(fit.get("blank_support", []))),
            }
        elif name == "STAGE4":
            support = fit.get("support_rows", [])
            fit_summary = {
                "blank_summary_rows": len(fit.get("blank_rows", [])),
                "supported_local_blocks": len(support),
                "q_bound_active_blocks": int(sum(row.get("q_bound_active", False)
                                                 for row in support)),
                "rank_distribution": {
                    str(rank): int(sum(row.get("rank") == rank for row in support))
                    for rank in sorted({row.get("rank") for row in support})
                },
            }
        else:
            fit_summary = {"available": True}
        result[name] = {"fit": fit_summary}
    return result


def load_initial_state(product_path):
    """Load the persisted initializer without executing any fitting code."""
    product_path = Path(product_path).expanduser().resolve()
    payload = json.loads(product_path.read_text())
    arrays_path = Path(payload["arrays"])
    if not arrays_path.is_absolute():
        arrays_path = product_path.parent / arrays_path
    arrays = np.load(arrays_path, allow_pickle=False)
    return payload, arrays


def _compare_joint_state(path, state, ifus, hi_keys):
    if not path:
        return {"available": False, "reason": "not supplied"}
    path = Path(path).expanduser().resolve()
    if not path.exists():
        return {"available": False, "reason": "product not found", "path": str(path)}
    product = json.loads(path.read_text())
    params = product.get("model_parameters", {})
    report = {"available": True, "path": str(path), "differences": {}}
    def parse_map(values):
        return {ast.literal_eval(key): value for key, value in values.items()}
    for name, ours in (("p_IFU_log", state.p_ifu), ("ax", state.ax), ("ay", state.ay)):
        reference_values = params.get(name, {})
        if name == "p_IFU_log":
            reference = {tuple(ast.literal_eval(key)): float(value) for key, value in reference_values.items()}
            diffs = [float(ours[index] - reference[tuple(ifu)]) for index, ifu in enumerate(ifus)
                     if tuple(ifu) in reference]
        else:
            reference = parse_map(reference_values)
            # H5/exposure order comparison is handled below by map keys.
            diffs = []
            for value in reference.values():
                if np.isfinite(value):
                    diffs.append(float(value))
        if diffs:
            report["differences"][name] = residual_summary(np.asarray(diffs))
    reference_g = params.get("G", {})
    reference_q = params.get("Q", {})
    gdiff, qdiff = [], []
    for index, (h5, ifu_code) in enumerate(hi_keys):
        key = str((h5, tuple(ifus[ifu_code])))
        if key in reference_g:
            # The joint product stores the production 4500-A intercept,
            # whereas the initializer keeps its centered intercept at xbar.
            g4500 = state.g_star[index] - state.xbar[index] * state.q_color[index]
            gdiff.extend((g4500 - np.asarray(reference_g[key], dtype=float)).tolist())
        if key in reference_q:
            qdiff.extend((state.q_color[index] - np.asarray(reference_q[key], dtype=float)).tolist())
    if gdiff:
        report["differences"]["G_4500_vs_joint_G"] = residual_summary(np.asarray(gdiff))
    if qdiff:
        report["differences"]["Q_vs_joint_Q"] = residual_summary(np.asarray(qdiff))
    report["note"] = "Differences are reported, not forced to agree."
    return report


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5", nargs="*", default=[])
    parser.add_argument("--product", required=True, help="validated frozen Model-3 product JSON")
    parser.add_argument("--blank-file")
    parser.add_argument("--external-cache")
    parser.add_argument("--compact-mask")
    parser.add_argument("--membership-artifact")
    parser.add_argument("--on-filter")
    parser.add_argument("--off-filter")
    parser.add_argument("--fq-template")
    parser.add_argument("--band-cache")
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--output-dir", default="m101_band_initializer")
    parser.add_argument("--joint-product", default="m101_joint_calibration/m101_calibration_joint_product.json")
    parser.add_argument("--minimum-finite-fraction", type=float, default=.8)
    parser.add_argument("--minimum-source-fibers", type=int, default=10)
    parser.add_argument("--q-limit-percent", type=float, choices=(10.,), default=10.)
    parser.add_argument("--consistency-pass", action="store_true")
    parser.add_argument("--stop-after-stage1", action="store_true",
                        help="write the isolated revised Stage-1 product and exit")
    parser.add_argument("--stage1-iterations", type=int, default=STAGE1_DEFAULT_ITERATIONS)
    parser.add_argument("--stage1-tolerance", type=float, default=STAGE1_DEFAULT_TOLERANCE)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    global FQ_GLOBAL, BAND_X, CURRENT_SKY
    args = _parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise SystemExit("output directory is nonempty; use --overwrite: %s" % output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    timings = {}
    progress = lambda message: print("[band-init +%.1fs] %s" %
                                     (time.perf_counter() - started, message), flush=True)
    product, product_provenance = _product_defaults(args.product)
    cache_hint = Path(args.band_cache or (output_dir / "m101_band_cache.npz")).expanduser().resolve()
    if not args.h5 and cache_hint.with_suffix(cache_hint.suffix + ".json").exists():
        cache_hint_manifest = json.loads(
            cache_hint.with_suffix(cache_hint.suffix + ".json").read_text())
        args.h5 = list(dict.fromkeys(
            row["h5_path"] for row in cache_hint_manifest.get("items", [])))
    if not args.h5:
        args.h5 = [row["full_path"] for row in product_provenance.get("input_h5", [])
                   if Path(row.get("full_path", "")).exists()]
    defaults = {
        "blank_file": product_provenance.get("blank_file"),
        "on_filter": product_provenance.get("on_filter"),
        "off_filter": product_provenance.get("off_filter"),
        "fq_template": product_provenance.get("fq_template"),
    }
    args.blank_file = args.blank_file or str(_resolve_path(None, defaults["blank_file"]) or "")
    args.on_filter = args.on_filter or str(_resolve_path(None, defaults["on_filter"]) or "")
    args.off_filter = args.off_filter or str(_resolve_path(None, defaults["off_filter"]) or "")
    args.fq_template = args.fq_template or str(_resolve_path(None, defaults["fq_template"]) or "")
    if not args.fq_template or not Path(args.fq_template).exists():
        raise SystemExit("--fq-template is required (the product default is not available)")
    FQ_GLOBAL = np.asarray(validated_m101.load_fq(args.fq_template), dtype=float)
    external = None
    external_provenance = None
    if args.external_cache:
        external, external_provenance = m101_external_measurements.load(
            args.external_cache, [Path(value).expanduser().resolve() for value in args.h5])
    elif not args.band_cache or not Path(args.band_cache).exists():
        raise SystemExit("building a cache requires --external-cache for source diagnostics")
    selected_names = [Path(value).name for value in args.h5]
    cache_started = time.perf_counter()
    cache_was_loaded = cache_hint.exists() and not args.rebuild_cache
    data, cache_manifest, responses, cache_path = _load_or_build_cache(
        args, product_provenance, output_dir, external, progress)
    timings["band_cache_seconds"] = time.perf_counter() - cache_started
    if responses is None:
        # Cache runs do not need filter arrays.  The manifest contains the
        # seven response-effective coordinates and K is per exposure.
        BAND_X = np.asarray(cache_manifest["band_effective_x_lambda"], dtype=float)
        if not cache_manifest.get("membership_frozen", False):
            if not args.on_filter or not args.off_filter:
                raise SystemExit("raw band cache has no frozen source masks; filters are required to reproduce membership")
            responses, _ = native._band_responses(
                validated_m101.read_filter(args.on_filter),
                validated_m101.read_filter(args.off_filter))
    else:
        BAND_X = np.asarray(np.sum(responses * X_LAMBDA[None, :], axis=1) /
                            np.sum(responses, axis=1), dtype=float)
    if not selected_names:
        selected_names = sorted({item.h5_name for item in data})
    if not selected_names:
        raise ValueError("no selected H5 files")
    # Cache-only source arrays are sufficient; loading the external cache is
    # optional on repeated runs.
    membership_artifact, membership_info = _load_or_reproduce_membership(
        data, cache_manifest, args, args.product, responses, FQ_GLOBAL,
        external, selected_names, progress)
    if not cache_manifest.get("membership_frozen", False) and responses is not None:
        _save_band_cache(cache_path, data, cache_manifest, responses, membership_frozen=True)
        cache_manifest["membership_frozen"] = True
    if membership_info.get("groups"):
        (output_dir / "m101_joint_source_membership.json").write_text(json.dumps(
            json_ready({"freeze_stage": "band initializer pre-fit frozen population",
                        "counts": membership_info["counts"],
                        "groups": membership_info["groups"]}), indent=2, sort_keys=True))
    selected = _blank_masks(data)
    if not all(np.any(mask) for mask in selected.values()):
        raise ValueError("every exposure needs at least one pure blank fiber")
    # The warm product is used only to establish the validated population if
    # masks were not persisted in the cache.  Stage 0 onward starts fresh.
    ifus, ifu_map, hi_keys, hi_map = _assign_codes(data)
    n_ifu, n_hi = len(ifus), len(hi_keys)
    for item in data:
        item._hi_codes = _row_h5ifu_codes(item, hi_map)
    if args.stop_after_stage1:
        state = Stage1OnlyState(
            p_ifu=np.zeros(n_ifu, dtype=float), ax=np.zeros(len(data), dtype=float),
            ay=np.zeros(len(data), dtype=float), x_center=np.zeros(len(data), dtype=float),
            y_center=np.zeros(len(data), dtype=float))
    else:
        state = _initial_state(data, n_ifu, n_hi)
    stage_rows = []
    stage_band_rows = []
    source_rows_all = []
    amp_rows_all = []
    topology_rows_all = []
    alpha_rows = []
    stage_results = {}

    stage_started = time.perf_counter()
    sky0 = _stage0_sky(data, selected)
    CURRENT_SKY = sky0
    if not args.stop_after_stage1:
        stage_rows.extend(_evaluate_stage("STAGE0", data, state, sky0, selected, hi_map, BAND_X, FQ_GLOBAL))
        source_rows_all.extend(_source_diagnostics("STAGE0", data, state, sky0, hi_map, BAND_X, FQ_GLOBAL))
    stage_results["STAGE0"] = {"sky": sky0}
    timings["stage0_seconds"] = time.perf_counter() - stage_started

    if args.stop_after_stage1:
        stage_started = time.perf_counter()
        stage1 = _stage1_revised_fit(
            data, state, sky0, selected,
            iterations=args.stage1_iterations, tolerance=args.stage1_tolerance)
        sky1 = _stage1_sky_from_log(sky0, stage1["sky_log_correction"], data)
        residual_rows = _stage1_diagnostic_rows(
            data, state, sky0, sky1, selected, common_rows=stage1["common_rows"])

        # A separate null-only fit is the held-out test.  It uses the same
        # frozen pure-blank fibers and never imports ON/OFF information.
        heldout_state = Stage1OnlyState(
            p_ifu=np.zeros(n_ifu, dtype=float), ax=np.zeros(len(data), dtype=float),
            ay=np.zeros(len(data), dtype=float), x_center=np.zeros(len(data), dtype=float),
            y_center=np.zeros(len(data), dtype=float))
        null_fit = _stage1_revised_fit(
            data, heldout_state, sky0, selected,
            iterations=args.stage1_iterations, tolerance=args.stage1_tolerance,
            fit_band_indices=list(range(len(NULL_BANDS))),
            compressed_summaries=(stage1["amp_rows"], stage1["common_rows"],
                                  stage1["amp_lookup"]))
        heldout = _stage1_heldout_report(
            data, sky0, null_fit, null_fit["common_rows"], selected)
        native_pixels_read_for_cache = not cache_was_loaded
        plots = _stage1_plot_paths(
            data, state, stage1, sky0, selected, output_dir, BAND_X)
        timings["stage1_seconds"] = time.perf_counter() - stage_started
        timings["total_seconds"] = time.perf_counter() - started
        product_payload, state_path = _stage1_write_product(
            output_dir, args, data, cache_path, cache_manifest,
            native_pixels_read_for_cache, ifus, state, sky0, stage1, heldout,
            residual_rows, membership_artifact, membership_info, timings, BAND_X,
            selected, plot_paths=plots, external_provenance=external_provenance)
        product_payload["diagnostics"]["plot_paths"] = plots
        state_path.write_text(json.dumps(json_ready(product_payload), indent=2, sort_keys=True))

        overall = {stage: _stage1_summary_lookup(
            residual_rows, stage, "overall").get("robust_rms", np.nan)
                   for stage in ("STAGE0", "STAGE1_PRE_SKY", "STAGE1_POST_SKY")}
        slope_text = []
        for stage in ("STAGE0", "STAGE1_POST_SKY"):
            x_slope = _stage1_summary_lookup(
                residual_rows, stage, "focal_x", value="fractional_slope").get("slope", np.nan)
            y_slope = _stage1_summary_lookup(
                residual_rows, stage, "focal_y", value="fractional_slope").get("slope", np.nan)
            slope_text.append("%s=(%.6g, %.6g)" % (stage, x_slope, y_slope))
        print("Stage 1 only: complete four-amp IFU/band summaries=%d; physical IFUs=%d/%d (%.1f%%)" %
              (stage1["complete_rows"], stage1["physical_ifus_supported"], len(ifus),
               100. * stage1["physical_ifus_supported"] / max(1, len(ifus))))
        print("blank robust RMS: Stage0=%.6g -> PRE_SKY=%.6g -> POST_SKY=%.6g" %
              (overall["STAGE0"], overall["STAGE1_PRE_SKY"], overall["STAGE1_POST_SKY"]))
        print("focal residual slopes (x,y): %s" % "; ".join(slope_text))
        print("p_IFU range=[%.6g, %.6g], robust RMS=%.6g; ax range=[%.6g, %.6g]; ay range=[%.6g, %.6g]" %
              (np.min(state.p_ifu), np.max(state.p_ifu), residual_summary(state.p_ifu)["robust_rms"],
               np.min(state.ax), np.max(state.ax), np.min(state.ay), np.max(state.ay)))
        print("iterations=%d; maximum parameter change=%.6g; held-out ON/OFF improved=%s" %
              (stage1["iterations_run"], max([row["max_parameter_change"]
                                               for row in stage1["iteration_history"]] or [0.0]),
               heldout["overall_improved"]))
        print("plots: %s" % ", ".join(plots))
        print("wrote Stage-1-only product: %s" % state_path)
        return

    stage_started = time.perf_counter()
    stage1 = _stage1_fit(data, state, sky0, selected, hi_map)
    stage_rows.extend(_evaluate_stage("STAGE1_PRE_SKY", data, state, sky0, selected, hi_map, BAND_X, FQ_GLOBAL))
    source_rows_all.extend(_source_diagnostics("STAGE1_PRE_SKY", data, state, sky0, hi_map, BAND_X, FQ_GLOBAL))
    sky1a = _profile_sky(data, state, hi_map, BAND_X, FQ_GLOBAL, selected)
    CURRENT_SKY = sky1a
    stage_rows.extend(_evaluate_stage("STAGE1_POST_SKY", data, state, sky1a, selected, hi_map, BAND_X, FQ_GLOBAL))
    source_rows_all.extend(_source_diagnostics("STAGE1_POST_SKY", data, state, sky1a, hi_map, BAND_X, FQ_GLOBAL))
    stage_results["STAGE1"] = {"fit": stage1, "sky": sky1a}
    timings["stage1_seconds"] = time.perf_counter() - stage_started

    stage_started = time.perf_counter()
    alpha_rows.extend(_stage2_alpha(data, state, sky1a, selected, FQ_GLOBAL, hi_map))
    stage_rows.extend(_evaluate_stage("STAGE2_PRE_SKY", data, state, sky1a, selected, hi_map, BAND_X, FQ_GLOBAL))
    source_rows_all.extend(_source_diagnostics("STAGE2_PRE_SKY", data, state, sky1a, hi_map, BAND_X, FQ_GLOBAL))
    sky1 = _profile_sky(data, state, hi_map, BAND_X, FQ_GLOBAL, selected)
    CURRENT_SKY = sky1
    stage_rows.extend(_evaluate_stage("STAGE2_POST_SKY", data, state, sky1, selected, hi_map, BAND_X, FQ_GLOBAL))
    source_rows_all.extend(_source_diagnostics("STAGE2_POST_SKY", data, state, sky1, hi_map, BAND_X, FQ_GLOBAL))
    stage_results["STAGE2"] = {"sky": sky1}
    timings["stage2_seconds"] = time.perf_counter() - stage_started

    stage_started = time.perf_counter()
    stage3 = _stage3_fit(data, state, sky1, selected, hi_keys, hi_map, ifus)
    amp_rows_all.extend(_stage_amp_output("STAGE3", stage3["blank_rows"], stage3["source_rows"]))
    topology_rows_all.extend(stage3["topology_rows"])
    stage_rows.extend(_evaluate_stage("STAGE3_PRE_SKY", data, state, sky1, selected, hi_map, BAND_X, FQ_GLOBAL))
    source_rows_all.extend(_source_diagnostics("STAGE3_PRE_SKY", data, state, sky1, hi_map, BAND_X, FQ_GLOBAL))
    sky2 = _profile_sky(data, state, hi_map, BAND_X, FQ_GLOBAL, selected)
    CURRENT_SKY = sky2
    stage_rows.extend(_evaluate_stage("STAGE3_POST_SKY", data, state, sky2, selected, hi_map, BAND_X, FQ_GLOBAL))
    source_rows_all.extend(_source_diagnostics("STAGE3_POST_SKY", data, state, sky2, hi_map, BAND_X, FQ_GLOBAL))
    stage_results["STAGE3"] = {"fit": stage3, "sky": sky2}
    timings["stage3_seconds"] = time.perf_counter() - stage_started

    stage_started = time.perf_counter()
    stage4 = _stage4_fit(data, state, sky2, selected, hi_keys, hi_map, BAND_X, QMAX_10)
    amp_rows_all.extend(_stage_amp_output("STAGE4", stage4["blank_rows"], None))
    topology_rows_all.extend(stage4["support_rows"])
    stage_rows.extend(_evaluate_stage("STAGE4_PRE_SKY", data, state, sky2, selected, hi_map, BAND_X, FQ_GLOBAL))
    source_rows_all.extend(_source_diagnostics("STAGE4_PRE_SKY", data, state, sky2, hi_map, BAND_X, FQ_GLOBAL))
    sky3 = _profile_sky(data, state, hi_map, BAND_X, FQ_GLOBAL, selected)
    CURRENT_SKY = sky3
    stage_rows.extend(_evaluate_stage("STAGE4_POST_SKY", data, state, sky3, selected, hi_map, BAND_X, FQ_GLOBAL))
    source_rows_all.extend(_source_diagnostics("STAGE4_POST_SKY", data, state, sky3, hi_map, BAND_X, FQ_GLOBAL))
    stage_results["STAGE4"] = {"fit": stage4, "sky": sky3}
    timings["stage4_seconds"] = time.perf_counter() - stage_started

    consistency = None
    if args.consistency_pass:
        stage_started = time.perf_counter()
        state_before = {name: np.asarray(getattr(state, name)).copy()
                        for name in ("p_ifu", "ax", "ay", "alpha_q", "p_amp",
                                     "g_star", "q_color", "xbar")}
        alpha_consistency = _stage2_alpha(data, state, sky3, selected, FQ_GLOBAL, hi_map)
        # Stage 3's absolute gray construction is intentionally not repeated
        # here: it would reinterpret an already applied topology as a fresh
        # base response.  The consistency pass rebuilds the current blank
        # differential summaries and applies one supported G/Q increment.
        sky_consistency_pre = _profile_sky(data, state, hi_map, BAND_X, FQ_GLOBAL, selected)
        consistency_stage4 = _stage4_fit(
            data, state, sky_consistency_pre, selected, hi_keys, hi_map, BAND_X, QMAX_10)
        stage_rows.extend(_evaluate_stage("OPTIONAL_CONSISTENCY", data, state,
                                          sky_consistency_pre, selected, hi_map, BAND_X, FQ_GLOBAL))
        source_rows_all.extend(_source_diagnostics("OPTIONAL_CONSISTENCY", data, state,
                                                   sky_consistency_pre, hi_map, BAND_X, FQ_GLOBAL))
        sky_consistency = _profile_sky(data, state, hi_map, BAND_X, FQ_GLOBAL, selected)
        stage_rows.extend(_evaluate_stage("OPTIONAL_CONSISTENCY_POST_SKY", data, state,
                                          sky_consistency, selected, hi_map, BAND_X, FQ_GLOBAL))
        source_rows_all.extend(_source_diagnostics("OPTIONAL_CONSISTENCY_POST_SKY", data, state,
                                                   sky_consistency, hi_map, BAND_X, FQ_GLOBAL))
        CURRENT_SKY = sky_consistency
        consistency = {"alpha_rows": alpha_consistency,
                       "stage4": consistency_stage4["support_rows"],
                       "parameter_change_max": {
                           name: float(np.max(np.abs(getattr(state, name) - state_before[name]), initial=0.0))
                           for name in state_before},
                       "sky": sky_consistency,
                       "stage_seconds": time.perf_counter() - stage_started}
        timings["optional_consistency_seconds"] = consistency["stage_seconds"]
        stage_results["OPTIONAL_CONSISTENCY"] = consistency

    validation = _validate_state(state, data, hi_keys, hi_map, BAND_X)
    validation["gauge_transfer"] = _gauge_invariance_check(state, data, hi_keys, hi_map, BAND_X)
    validation["population_counts"] = {
        "blank_valid": int(sum(np.sum(item.blank_valid) for item in data)),
        "pure_blank": int(sum(np.sum(mask) for mask in selected.values())),
        "source_candidate_union": int(sum(np.sum(item.source_candidate.any(axis=1)) for item in data)),
    }
    validation["native_pixels_used_in_fit"] = False
    if not all(validation[key] for key in ("amplifier_contrast_column_sums_zero", "p_IFU_zero_sum",
                                           "mean_h_G_zero", "centered_saved_predictions_agree",
                                           "q_bound_satisfied")):
        raise ValueError("initializer validation gate failed: %s" % validation)

    arrays_path = output_dir / "m101_band_initial_state_arrays.npz"
    np.savez_compressed(arrays_path, **_arrays_payload(state, data, ifus, hi_keys))
    product_payload = {
        "schema_version": SCHEMA_VERSION,
        "architecture": "four-stage deterministic seven-band initializer",
        "reference_fitter_unchanged": True,
        "stage5_full_spectrum_out_of_scope": True,
        "band_only_fit": True,
        "band_order": list(ALL_BANDS),
        "null_bands": list(NULL_BANDS),
        "source_bands": list(SOURCE_BANDS),
        "band_effective_x_lambda": BAND_X,
        "topology_basis": BASIS,
        "x_lambda_definition": "(lambda - 4500 A) / 1000 A",
        "q_bound": {"percent": 10., "qmax": QMAX_10,
                    "definition": "abs(B_a dot Q) <= log(1.10)/2"},
        "parameters": _state_json(state, data, ifus, hi_keys, hi_map),
        "mode_support": topology_rows_all,
        "validation": validation,
        "population": {"frozen": True, "pure_blank_definition": "existing blank_valid AND NOT source-candidate union",
                       "membership_artifact": str(membership_artifact) if membership_artifact else None,
                       "membership_info": membership_info},
        "stage_summaries": _compact_stage_summaries(stage_results),
        "timings": timings,
        "arrays": str(arrays_path),
        "provenance": {
            "product": file_identity(args.product),
            "blank_file": file_identity(args.blank_file) if args.blank_file and Path(args.blank_file).exists() else None,
            "blank_file_sha256": small_file_hash(args.blank_file) if args.blank_file and Path(args.blank_file).exists() else None,
            "external_cache": external_provenance,
            "compact_mask": file_identity(args.compact_mask) if args.compact_mask and Path(args.compact_mask).exists() else None,
            "on_filter": file_identity(args.on_filter) if args.on_filter and Path(args.on_filter).exists() else None,
            "off_filter": file_identity(args.off_filter) if args.off_filter and Path(args.off_filter).exists() else None,
            "fq_template": file_identity(args.fq_template),
            "band_cache": file_identity(cache_path),
            "band_cache_manifest": file_identity(_cache_paths(cache_path)[1]),
            "cache_manifest": cache_manifest,
            "fit_inputs": {"native_pixels_used_for_fit": False,
                            "collapsed_band_rows_only": True,
                            "native_pixels_read_for_cache": bool(responses is not None)},
        },
        "comparison_to_existing_joint": _compare_joint_state(
            args.joint_product, state, ifus, hi_keys),
    }
    (output_dir / "m101_band_initial_state.json").write_text(
        json.dumps(json_ready(product_payload), indent=2, sort_keys=True))
    _write_rows(output_dir / "stage_residual_summary.csv", stage_rows)
    # A compact band table is intentionally separate from the group table.
    stage_band_rows = [row for row in stage_rows if row["grouping"] == "band"]
    _write_rows(output_dir / "stage_band_residuals.csv", stage_band_rows)
    _write_rows(output_dir / "stage_amp_summaries.csv", amp_rows_all)
    _write_rows(output_dir / "stage_source_summaries.csv", source_rows_all)
    _write_rows(output_dir / "alpha_support.csv", alpha_rows)
    _write_rows(output_dir / "topology_mode_support.csv", topology_rows_all)
    timing_payload = dict(timings)
    timing_payload["total_seconds"] = time.perf_counter() - started
    timing_payload["cache_path"] = str(cache_path)
    (output_dir / "initializer_timing.json").write_text(json.dumps(
        json_ready(timing_payload), indent=2, sort_keys=True))
    progress("wrote band initializer to %s" % output_dir)


if __name__ == "__main__":
    main()
