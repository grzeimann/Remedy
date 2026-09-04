#!/usr/bin/env python3
"""Build the frozen compact two-band M101 measurement product.

Each row in ``m101_measurements.h5:/measurements`` is one native VIRUS fiber
observation.  ON and OFF values are stored side-by-side, with band index 0
always ON and band index 1 always OFF.

    D = synthetic collapse of Fibers.spectrum / Survey.offset
    B = synthetic collapse of Fibers.skyspectrum, without division
    I = cached background-subtracted, PSF-matched, exact external aperture
    X = g_band * I
    K = synthetic collapse of raw_work_basis(Survey)
    q = distance from the physical amplifier readout edge
    fq = fixed f(q) template value

No fitted M101 alpha, s_response, residual-sky correction, spatial plane,
beta, exposure gray, or source/blank criterion is applied.  Future fitting
can therefore use the compact product without reopening full spectra.
"""

from argparse import ArgumentParser
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import pickle
import subprocess
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tables

import diagnose_m101_hierarchical as hm


BANDS = ("ON", "OFF")
N_BANDS = 2
N_EXPOSURES = 3
N_FIBER_AMP = 112
EXCLUDED_H5 = "20200523_0000023.h5"
EXPECTED_H5_COUNT = 19
EXPECTED_EXPOSURE_COUNT = 57
SCHEMA_VERSION = "m101_measurements_v1"


class MeasurementDescription(tables.IsDescription):
    h5_id = tables.Int16Col(pos=0)
    exposure = tables.UInt8Col(pos=1)
    original_h5_row = tables.Int64Col(pos=2)
    SPECID = tables.Int32Col(pos=3)
    IFUSLOT = tables.Int32Col(pos=4)
    IFUID = tables.Int32Col(pos=5)
    AMP = tables.StringCol(2, pos=6)
    j = tables.Int16Col(pos=7)
    q = tables.Int16Col(pos=8)
    fq = tables.Float64Col(pos=9)
    RA = tables.Float64Col(pos=10)
    Dec = tables.Float64Col(pos=11)
    effective_RA = tables.Float64Col(shape=(N_BANDS,), pos=12)
    effective_Dec = tables.Float64Col(shape=(N_BANDS,), pos=13)
    data_native = tables.Float64Col(shape=(N_BANDS,), pos=14)
    data_work = tables.Float64Col(shape=(N_BANDS,), pos=15)
    error_native = tables.Float64Col(shape=(N_BANDS,), pos=16)
    error_work = tables.Float64Col(shape=(N_BANDS,), pos=17)
    sky = tables.Float64Col(shape=(N_BANDS,), pos=18)
    external_raw = tables.Float64Col(shape=(N_BANDS,), pos=19)
    external_prediction = tables.Float64Col(shape=(N_BANDS,), pos=20)
    data_plus_sky = tables.Float64Col(shape=(N_BANDS,), pos=21)
    predictor_plus_sky = tables.Float64Col(shape=(N_BANDS,), pos=22)
    external_valid = tables.BoolCol(shape=(N_BANDS,), pos=23)
    date_mask_bad = tables.BoolCol(pos=24)
    finite_data_native = tables.BoolCol(shape=(N_BANDS,), pos=25)
    finite_data_work = tables.BoolCol(shape=(N_BANDS,), pos=26)
    finite_error = tables.BoolCol(shape=(N_BANDS,), pos=27)
    finite_sky = tables.BoolCol(shape=(N_BANDS,), pos=28)
    finite_external = tables.BoolCol(shape=(N_BANDS,), pos=29)
    data_response_fraction = tables.Float64Col(shape=(N_BANDS,), pos=30)
    sky_response_fraction = tables.Float64Col(shape=(N_BANDS,), pos=31)


class ExposureBandDescription(tables.IsDescription):
    h5_id = tables.Int16Col(pos=0)
    exposure = tables.UInt8Col(pos=1)
    survey_offset = tables.Float64Col(pos=2)
    survey_exptime = tables.Float64Col(pos=3)
    survey_millum = tables.Float64Col(pos=4)
    survey_throughput = tables.Float64Col(pos=5)
    survey_fwhm = tables.Float64Col(pos=6)
    survey_ra = tables.Float64Col(pos=7)
    survey_dec = tables.Float64Col(pos=8)
    survey_pa = tables.Float64Col(pos=9)
    K = tables.Float64Col(shape=(N_BANDS,), pos=10)
    g_global = tables.Float64Col(shape=(N_BANDS,), pos=11)
    sky_robust_location = tables.Float64Col(shape=(N_BANDS,), pos=12)
    sky_robust_scale = tables.Float64Col(shape=(N_BANDS,), pos=13)
    sky_finite_fraction = tables.Float64Col(shape=(N_BANDS,), pos=14)
    external_image_fwhm = tables.Float64Col(shape=(N_BANDS,), pos=15)
    external_background = tables.Float64Col(shape=(N_BANDS,), pos=16)
    external_background_scatter = tables.Float64Col(shape=(N_BANDS,), pos=17)
    n_fibers = tables.Int64Col(pos=18)
    n_date_masked = tables.Int64Col(pos=19)
    n_finite_data = tables.Int64Col(shape=(N_BANDS,), pos=20)
    n_finite_sky = tables.Int64Col(shape=(N_BANDS,), pos=21)
    n_external_valid = tables.Int64Col(shape=(N_BANDS,), pos=22)
    production_state = tables.Int16Col(pos=23)


class H5InputDescription(tables.IsDescription):
    h5_id = tables.Int16Col(pos=0)
    filename = tables.StringCol(256, pos=1)
    full_path = tables.StringCol(1024, pos=2)
    file_size = tables.Int64Col(pos=3)
    mtime_ns = tables.Int64Col(pos=4)
    native_rows = tables.Int64Col(pos=5)
    exposure_count = tables.Int16Col(pos=6)
    complete = tables.BoolCol(pos=7)


class KeyValueDescription(tables.IsDescription):
    key = tables.StringCol(128, pos=0)
    value = tables.StringCol(4096, pos=1)


class CheckDescription(tables.IsDescription):
    name = tables.StringCol(128, pos=0)
    status = tables.StringCol(16, pos=1)
    value = tables.Float64Col(pos=2)
    detail = tables.StringCol(2048, pos=3)


def _finite(values):
    values = np.asarray(values, dtype=float)
    return values[np.isfinite(values)]


def _number(value, default=np.nan):
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    return value if np.isfinite(value) else default


def _file_identity(path):
    stat = Path(path).stat()
    return {"filename": Path(path).name, "full_path": str(Path(path).resolve()),
            "file_size": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns)}


def _small_file_hash(path):
    path = Path(path)
    if path.stat().st_size > 50 * 1024 * 1024:
        return "not-hashed-large-file"
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_commit():
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                text=True, check=True, cwd=Path(__file__).parent)
        return result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unavailable"


def _discover_h5(args):
    if args.h5:
        paths = [Path(name).expanduser().resolve() for name in args.h5]
    elif args.h5_glob:
        import glob
        paths = [Path(name).resolve() for pattern in args.h5_glob
                 for name in glob.glob(str(Path(pattern).expanduser()))]
    elif args.h5_dir:
        paths = sorted(Path(args.h5_dir).expanduser().glob("*.h5"))
        paths = [path.resolve() for path in paths]
    else:
        raise ValueError("provide --h5-dir, --h5-glob, or --h5")
    unique = {}
    for path in paths:
        if not path.is_file():
            raise ValueError("H5 file does not exist: %s" % path)
        if path.name == EXCLUDED_H5:
            print("excluding explicitly excluded H5: %s" % path)
            continue
        unique[str(path)] = path
    paths = sorted(unique.values(), key=lambda value: value.name)
    if not paths:
        raise ValueError("no accepted H5 files found")
    names = [path.name for path in paths]
    if len(names) != len(set(names)):
        raise ValueError("duplicate H5 basenames make cache mapping ambiguous")
    return paths


def _cache_calibrations(cache, h5_paths):
    calibrations = cache.get("calibrations")
    if not isinstance(calibrations, list):
        raise ValueError("PASS-1 cache lacks calibrations list")
    by_name = {}
    for calibration in calibrations:
        name = Path(str(calibration.get("h5", ""))).name
        if name in by_name:
            raise ValueError("duplicate calibration basename in PASS-1 cache: %s" % name)
        by_name[name] = calibration
    selected = []
    for path in h5_paths:
        if path.name not in by_name:
            raise ValueError("PASS-1 cache lacks calibration for %s" % path.name)
        selected.append(by_name[path.name])
    return selected


def _load_inputs(args, h5_paths):
    with Path(args.pass1_cache).expanduser().open("rb") as stream:
        cache = pickle.load(stream)
    calibrations = _cache_calibrations(cache, h5_paths)
    if "global_g" not in cache or any(not np.isfinite(_number(cache["global_g"].get(b)))
                                      for b in BANDS):
        raise ValueError("PASS-1 cache lacks finite authoritative global_g values")
    fq = hm.load_fq(args.fq_template)
    filters = {"ON": hm.read_filter(args.on_filter),
               "OFF": hm.read_filter(args.off_filter)}
    if "f_q" in cache and not np.allclose(np.asarray(cache["f_q"], dtype=float), fq,
                                          rtol=0, atol=0):
        raise ValueError("f(q) template differs from PASS-1 cache")
    for band in BANDS:
        if band in cache.get("filters", {}) and not np.allclose(
                np.asarray(cache["filters"][band], dtype=float), filters[band],
                rtol=0, atol=0):
            raise ValueError("%s filter differs from PASS-1 cache" % band)
    for calibration in calibrations:
        if not isinstance(calibration.get("external_object"), dict) or \
                not isinstance(calibration.get("external_valid"), dict):
            raise ValueError("%s has no cached external_object/external_valid arrays" %
                             calibration.get("h5", "unknown H5"))
        for exposure in range(1, N_EXPOSURES + 1):
            for band in BANDS:
                if (exposure, band) not in calibration["external_object"] or \
                        (exposure, band) not in calibration["external_valid"]:
                    raise ValueError("cache lacks external %s exposure %d" % (band, exposure))
    return cache, calibrations, filters, fq


def _survey_by_exposure(h5):
    if "Survey" not in h5.root._v_children:
        raise ValueError("H5 lacks Survey table")
    result = {}
    for row in h5.root.Survey:
        exposure = int(row["exp"])
        if exposure in result:
            raise ValueError("duplicate Survey exposure %d" % exposure)
        result[exposure] = {name: row[name] for name in h5.root.Survey.colnames}
    if set(result) != set(range(1, N_EXPOSURES + 1)):
        raise ValueError("Survey must contain exposures exactly 1, 2, 3")
    return result


def _collapse_with_fraction(values, response):
    """Match validated_m101.synthetic_mean and return response completeness."""
    values = np.asarray(values, dtype=float)
    response = np.asarray(response, dtype=float)
    finite = np.isfinite(values) & np.isfinite(response)[None, :]
    weights = np.where(finite, response[None, :], 0.0)
    denominator = np.sum(weights, axis=1)
    output = np.full(values.shape[0], np.nan, dtype=float)
    good = denominator != 0.0
    output[good] = np.sum(np.where(finite, values, 0.0) * response[None, :], axis=1)[good] / denominator[good]
    response_total = np.sum(np.abs(response[np.isfinite(response)]))
    fraction = np.zeros(values.shape[0], dtype=float)
    if response_total > 0:
        fraction = np.sum(np.where(finite, np.abs(response)[None, :], 0.0), axis=1) / response_total
    return output, fraction


def _collapse_error(error, response):
    error = np.asarray(error, dtype=float)
    response = np.asarray(response, dtype=float)
    finite = np.isfinite(error) & np.isfinite(response)[None, :]
    denominator = np.sum(np.where(finite, response[None, :], 0.0), axis=1)
    variance = np.sum(np.where(finite, error, 0.0) ** 2 * response[None, :] ** 2, axis=1)
    output = np.full(error.shape[0], np.nan, dtype=float)
    good = denominator != 0.0
    output[good] = np.sqrt(variance[good]) / np.abs(denominator[good])
    response_total = np.sum(np.abs(response[np.isfinite(response)]))
    fraction = np.zeros(error.shape[0], dtype=float)
    if response_total > 0:
        fraction = np.sum(np.where(finite, np.abs(response)[None, :], 0.0), axis=1) / response_total
    return output, fraction


def _max_abs_relative(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    valid = np.isfinite(a) & np.isfinite(b)
    if not valid.any():
        return np.nan
    return float(np.max(np.abs(a[valid] - b[valid]) /
                       np.maximum(np.abs(b[valid]), 1e-30)))


def _preflight(h5_paths, development):
    metadata = []
    total_rows = 0
    total_exposures = 0
    for path in h5_paths:
        with tables.open_file(path, mode="r") as h5:
            if "Info" not in h5.root._v_children or "Fibers" not in h5.root._v_children:
                raise ValueError("%s must contain Info and Fibers" % path)
            nrows = int(h5.root.Info.nrows)
            surveys = _survey_by_exposure(h5)
            metadata.append({"path": path, "native_rows": nrows,
                             "exposure_count": len(surveys),
                             "identity": _file_identity(path)})
            total_rows += nrows
            total_exposures += len(surveys)
    if not development and len(h5_paths) != EXPECTED_H5_COUNT:
        raise ValueError("expected %d accepted H5 files, found %d; use --development for a test subset" %
                         (EXPECTED_H5_COUNT, len(h5_paths)))
    if not development and total_exposures != EXPECTED_EXPOSURE_COUNT:
        raise ValueError("expected %d exposures, found %d; use --development for a test subset" %
                         (EXPECTED_EXPOSURE_COUNT, total_exposures))
    return metadata, total_rows, total_exposures


def _row_arrays(h5_id, path, calibration, filters, fq, g_global, table, exposure,
                survey, groups, labels, ra_all, dec_all, amps_all, date_bad_all,
                specid_all, slot_all, uid_all, fibers):
    """Read one exposure, collapse primitives, and return a compact batch."""
    exp_indices = np.flatnonzero(labels == exposure)
    source = np.asarray(fibers.read_coordinates(exp_indices, field="spectrum"), dtype=float)
    error = np.asarray(fibers.read_coordinates(exp_indices, field="error"), dtype=float)
    sky_native = np.asarray(fibers.read_coordinates(exp_indices, field="skyspectrum"), dtype=float)
    offset = _number(survey["offset"])
    if not np.isfinite(offset) or offset == 0:
        raise ValueError("%s exposure %d has invalid Survey.offset" % (path, exposure))
    data_columns = [_collapse_with_fraction(source, filters[b]) for b in BANDS]
    error_columns = [_collapse_error(error, filters[b]) for b in BANDS]
    sky_columns = [_collapse_with_fraction(sky_native, filters[b]) for b in BANDS]
    data_native = np.column_stack([column[0] for column in data_columns])
    data_fraction = np.column_stack([column[1] for column in data_columns])
    error_native = np.column_stack([column[0] for column in error_columns])
    sky = np.column_stack([column[0] for column in sky_columns])
    sky_fraction = np.column_stack([column[1] for column in sky_columns])
    sample = np.arange(min(8, len(source)))
    direct_data = np.column_stack([hm.synthetic_mean(source[sample], filters[b]) for b in BANDS])
    direct_error = np.column_stack([_collapse_error(error[sample], filters[b])[0] for b in BANDS])
    direct_sky = np.column_stack([hm.synthetic_mean(sky_native[sample], filters[b]) for b in BANDS])
    direct_spot = {
        "data_native_abs": float(np.nanmax(np.abs(direct_data - data_native[sample]))) if sample.size else np.nan,
        "data_native_rel": _max_abs_relative(direct_data, data_native[sample]),
        "error_native_abs": float(np.nanmax(np.abs(direct_error - error_native[sample]))) if sample.size else np.nan,
        "error_native_rel": _max_abs_relative(direct_error, error_native[sample]),
        "sky_abs": float(np.nanmax(np.abs(direct_sky - sky[sample]))) if sample.size else np.nan,
        "sky_rel": _max_abs_relative(direct_sky, sky[sample]),
    }
    data_work = data_native / offset
    error_work = error_native / abs(offset)
    del source, error, sky_native

    effective = []
    for band in BANDS:
        effective.append(hm.adr_positions(ra_all[exp_indices], dec_all[exp_indices], survey,
                                          filters[band]))
    effective_ra = np.column_stack([position[0] for position in effective])
    effective_dec = np.column_stack([position[1] for position in effective])
    external_raw = np.column_stack([
        np.asarray(calibration["external_object"][(exposure, band)], dtype=float)
        for band in BANDS])
    external_valid = np.column_stack([
        np.asarray(calibration["external_valid"][(exposure, band)], dtype=bool)
        for band in BANDS])
    if external_raw.shape[0] != exp_indices.size or external_valid.shape[0] != exp_indices.size:
        raise ValueError("%s exposure %d cached external arrays do not match %d H5 rows" %
                         (path, exposure, exp_indices.size))
    external_prediction = external_raw * np.asarray(g_global, dtype=float)[None, :]
    q_all = np.full(len(labels), -1, dtype=np.int16)
    j_all = np.full(len(labels), -1, dtype=np.int16)
    for group in groups:
        j = np.arange(N_FIBER_AMP, dtype=np.int16)
        q = j if group["amp"] in ("LL", "RU") else N_FIBER_AMP - 1 - j
        j_all[group["indices"]] = j
        q_all[group["indices"]] = q
    q = q_all[exp_indices]
    if np.any((q < 0) | (q > 111)):
        raise ValueError("invalid q assignment in %s exposure %d" % (path, exposure))
    fq_values = np.asarray(fq, dtype=float)[q]
    raw_basis = hm.raw_work_basis(survey)
    k = np.asarray([hm.weighted_scalar(raw_basis, filters[b]) for b in BANDS])
    production_state = 0
    # Production state is provenance only.  It is intentionally never used
    # in any numerical expression above or below.
    production_state = int(calibration.get("production_state", 0) or 0)
    batch = np.empty(exp_indices.size, dtype=table.dtype)
    batch["h5_id"] = h5_id
    batch["exposure"] = exposure
    batch["original_h5_row"] = exp_indices
    batch["SPECID"] = specid_all[exp_indices]
    batch["IFUSLOT"] = slot_all[exp_indices]
    batch["IFUID"] = uid_all[exp_indices]
    batch["AMP"] = np.asarray(amps_all[exp_indices], dtype="S2")
    batch["j"] = j_all[exp_indices]
    batch["q"] = q
    batch["fq"] = fq_values
    batch["RA"] = ra_all[exp_indices]
    batch["Dec"] = dec_all[exp_indices]
    batch["effective_RA"] = effective_ra
    batch["effective_Dec"] = effective_dec
    batch["data_native"] = data_native
    batch["data_work"] = data_work
    batch["error_native"] = error_native
    batch["error_work"] = error_work
    batch["sky"] = sky
    batch["external_raw"] = external_raw
    batch["external_prediction"] = external_prediction
    batch["data_plus_sky"] = data_work + sky
    batch["predictor_plus_sky"] = external_prediction + sky
    batch["external_valid"] = external_valid
    batch["date_mask_bad"] = date_bad_all[exp_indices]
    batch["finite_data_native"] = np.isfinite(data_native)
    batch["finite_data_work"] = np.isfinite(data_work)
    batch["finite_error"] = np.isfinite(error_native)
    batch["finite_sky"] = np.isfinite(sky)
    batch["finite_external"] = np.isfinite(external_raw)
    batch["data_response_fraction"] = data_fraction
    batch["sky_response_fraction"] = sky_fraction
    metadata = {
        "h5_id": h5_id, "exposure": exposure, "n_fibers": int(exp_indices.size),
        "survey": survey, "K": k, "g_global": np.asarray(g_global, dtype=float),
        "production_state": production_state,
        "sky_location": np.asarray([hm.robust_location(sky[:, i]) for i in range(N_BANDS)]),
        "sky_scale": np.asarray([hm.robust_scale(sky[:, i]) for i in range(N_BANDS)]),
        "sky_fraction": np.asarray([np.mean(np.isfinite(sky[:, i])) for i in range(N_BANDS)]),
        "n_date_masked": int(np.sum(date_bad_all[exp_indices])),
        "n_finite_data": np.sum(np.isfinite(data_native), axis=0).astype(np.int64),
        "n_finite_sky": np.sum(np.isfinite(sky), axis=0).astype(np.int64),
        "n_external_valid": np.sum(external_valid, axis=0).astype(np.int64),
        "spot": dict(_spot_checks(batch, exp_indices, groups, filters, fq, survey,
                                   g_global, source=None, error=None, sky=None),
                     **direct_spot),
    }
    return batch, metadata


def _spot_checks(batch, exp_indices, groups, filters, fq, survey, g_global,
                 source=None, error=None, sky=None):
    """Validate compact identities and q/fq mapping on deterministic rows."""
    sample = np.arange(min(8, len(batch)))
    result = {
        "data_work_abs": float(np.nanmax(np.abs(batch["data_work"][sample] -
                                                   batch["data_native"][sample] / survey["offset"])))
        if sample.size else np.nan,
        "data_work_rel": _max_abs_relative(batch["data_work"][sample],
                                            batch["data_native"][sample] / survey["offset"]),
        "external_prediction_abs": float(np.nanmax(np.abs(batch["external_prediction"][sample] -
                                                          batch["external_raw"][sample] * np.asarray(g_global)[None, :])))
        if sample.size else np.nan,
        "fq_abs": float(np.max(np.abs(batch["fq"][sample] - np.asarray(fq)[batch["q"][sample]])))
        if sample.size else np.nan,
    }
    # q orientation is checked using the first native fiber of each physical
    # group present in this exposure.  The production convention is explicit.
    orientation_errors = []
    for group in groups:
        if group["exposure"] != int(batch["exposure"][0]):
            continue
        first = int(group["indices"][0])
        local = int(np.searchsorted(exp_indices, first))
        expected = 0 if group["amp"] in ("LL", "RU") else 111
        orientation_errors.append(abs(int(batch["q"][local]) - expected))
    result["q_orientation_max"] = float(max(orientation_errors)) if orientation_errors else np.nan
    return result


def _exposure_row(metadata, table):
    row = np.zeros(1, dtype=table.dtype)
    survey = metadata["survey"]
    row["h5_id"] = metadata["h5_id"]
    row["exposure"] = metadata["exposure"]
    for output, source in (("survey_offset", "offset"), ("survey_exptime", "exptime"),
                           ("survey_millum", "millum"), ("survey_throughput", "throughput"),
                           ("survey_fwhm", "fwhm"), ("survey_ra", "ra"),
                           ("survey_dec", "dec"), ("survey_pa", "pa")):
        row[output] = _number(survey.get(source))
    row["K"] = metadata["K"]
    row["g_global"] = metadata["g_global"]
    row["sky_robust_location"] = metadata["sky_location"]
    row["sky_robust_scale"] = metadata["sky_scale"]
    row["sky_finite_fraction"] = metadata["sky_fraction"]
    # The cache used by the current repository stores the external arrays but
    # not image characterization records.  These remain explicit NaNs rather
    # than being reconstructed or guessed during measurement extraction.
    row["external_image_fwhm"] = np.nan
    row["external_background"] = np.nan
    row["external_background_scatter"] = np.nan
    row["n_fibers"] = metadata["n_fibers"]
    row["n_date_masked"] = metadata["n_date_masked"]
    row["n_finite_data"] = metadata["n_finite_data"]
    row["n_finite_sky"] = metadata["n_finite_sky"]
    row["n_external_valid"] = metadata["n_external_valid"]
    row["production_state"] = metadata["production_state"]
    return row


def _metadata_rows(items):
    rows = []
    for key, value in items.items():
        rows.append((str(key), json.dumps(value, sort_keys=True, default=str)))
    return rows


def _create_file(path, input_metadata, total_rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    h5 = tables.open_file(path, mode="w", title="Frozen M101 compact measurements")
    filters = tables.Filters(complevel=5, complib="zlib", shuffle=True)
    measurements = h5.create_table("/", "measurements", MeasurementDescription,
                                   expectedrows=total_rows, filters=filters)
    exposure_band = h5.create_table("/", "exposure_band", ExposureBandDescription,
                                    expectedrows=len(input_metadata) * 3, filters=filters)
    provenance = h5.create_group("/", "provenance")
    h5_inputs = h5.create_table(provenance, "h5_inputs", H5InputDescription,
                                expectedrows=len(input_metadata), filters=filters)
    metadata = h5.create_table(provenance, "metadata", KeyValueDescription,
                               expectedrows=32, filters=filters)
    checks = h5.create_table("/", "build_checks", CheckDescription,
                             expectedrows=32, filters=filters)
    for item in input_metadata:
        row = h5_inputs.row
        identity = item["identity"]
        row["h5_id"] = len(h5_inputs)
        row["filename"] = identity["filename"]
        row["full_path"] = identity["full_path"]
        row["file_size"] = identity["file_size"]
        row["mtime_ns"] = identity["mtime_ns"]
        row["native_rows"] = item["native_rows"]
        row["exposure_count"] = item["exposure_count"]
        row["complete"] = False
        row.append()
    h5_inputs.flush()
    root_attrs = h5.root._v_attrs
    root_attrs.schema_version = SCHEMA_VERSION
    root_attrs.band_order = json.dumps(list(BANDS))
    root_attrs.band_index_0 = "ON"
    root_attrs.band_index_1 = "OFF"
    root_attrs.scientific_model_version = "primitive_measurements_before_fitting_v1"
    root_attrs.no_fitted_calibration_applied = True
    root_attrs.no_residual_sky_subtraction = True
    root_attrs.no_source_blank_selection = True
    return h5, measurements, exposure_band, h5_inputs, metadata, checks


def _append_metadata(metadata_table, values):
    for key, value in _metadata_rows(values):
        row = metadata_table.row
        row["key"], row["value"] = key, value
        row.append()
    metadata_table.flush()


def _mark_complete(h5_inputs, h5_id, complete):
    h5_inputs.cols.complete[int(h5_id)] = bool(complete)
    h5_inputs.flush()


def _existing_state(h5, h5_paths, preflight):
    if "/provenance/h5_inputs" not in h5 or h5.root._v_attrs.schema_version != SCHEMA_VERSION:
        raise ValueError("existing output is not a compatible M101 measurement product")
    table = h5.root.provenance.h5_inputs
    if table.nrows != len(h5_paths):
        raise ValueError("resume input H5 count differs from existing product")
    for hid, path in enumerate(h5_paths):
        row = table[hid]
        if Path(row["filename"].decode() if isinstance(row["filename"], bytes) else row["filename"]).name != path.name:
            raise ValueError("resume H5 ordering differs at h5_id=%d" % hid)
        expected = int(preflight[hid]["native_rows"])
        if int(row["native_rows"]) != expected:
            raise ValueError("resume H5 row count differs for %s" % path.name)
        if bool(row["complete"]):
            count = int(np.sum(h5.root.measurements.cols.h5_id[:] == hid))
            if count != expected:
                raise ValueError("completed H5 %s has %d/%d measurement rows" %
                                 (path.name, count, expected))
        else:
            count = int(np.sum(h5.root.measurements.cols.h5_id[:] == hid))
            if count:
                raise ValueError("incomplete H5 %s already has rows; use --overwrite" % path.name)


def _build_h5(h5_id, path, calibration, filters, fq, g_global, measurements,
              exposure_band, h5_inputs, preflight):
    started = time.perf_counter()
    with tables.open_file(path, mode="r") as source_h5:
        info = source_h5.root.Info
        fibers = source_h5.root.Fibers
        required = {"spectrum", "error", "skyspectrum"}
        if not required.issubset(fibers.colnames):
            raise ValueError("%s Fibers lacks %s" % (path, sorted(required - set(fibers.colnames))))
        groups, labels = hm.build_groups(info)
        nrows = int(info.nrows)
        ra_all = np.asarray(info.cols.ra[:], dtype=float)
        dec_all = np.asarray(info.cols.dec[:], dtype=float)
        specid_all = np.asarray(info.cols.specid[:], dtype=np.int32)
        slot_all = np.asarray(info.cols.ifuslot[:], dtype=np.int32)
        uid_all = np.asarray(info.cols.ifuid[:], dtype=np.int32)
        amps_all = np.asarray([hm.as_text(value) for value in info.cols.amp[:]])
        date_bad_all = hm.masked_rows(path, slot_all, amps_all)
        surveys = _survey_by_exposure(source_h5)
        batches, exposure_rows, spot = [], [], []
        for exposure in range(1, N_EXPOSURES + 1):
            batch, metadata = _row_arrays(
                h5_id, path, calibration, filters, fq, g_global, measurements,
                exposure, surveys[exposure], groups, labels, ra_all, dec_all,
                amps_all, date_bad_all, specid_all, slot_all, uid_all, fibers)
            if len(batch) != int(np.sum(labels == exposure)):
                raise ValueError("%s exposure %d compact row count mismatch" % (path, exposure))
            batches.append(batch)
            exposure_rows.append(_exposure_row(metadata, exposure_band))
            spot.append(metadata["spot"])
    combined = np.concatenate(batches)
    if len(combined) != nrows:
        raise ValueError("%s produced %d/%d native measurement rows" % (path, len(combined), nrows))
    # One append per H5 keeps the normal failure path before any rows from this
    # H5 become visible.  The complete marker is written only after validation.
    measurements.append(combined)
    exposure_band.append(np.concatenate(exposure_rows))
    measurements.flush()
    exposure_band.flush()
    _mark_complete(h5_inputs, h5_id, True)
    return {"h5_id": h5_id, "filename": path.name, "rows": len(combined),
            "groups": len(groups), "spot": spot,
            "seconds": time.perf_counter() - started}


def _create_indexes(h5):
    started = time.perf_counter()
    measurements = h5.root.measurements
    for column in ("h5_id", "exposure", "SPECID", "IFUSLOT", "IFUID", "AMP"):
        column_object = getattr(measurements.cols, column)
        if not column_object.is_indexed:
            column_object.create_index()
    for column in ("h5_id", "exposure"):
        column_object = getattr(h5.root.exposure_band.cols, column)
        if not column_object.is_indexed:
            column_object.create_index()
    return time.perf_counter() - started


def _load_compact_qa(h5):
    table = h5.root.measurements
    columns = {name: np.asarray(getattr(table.cols, name)[:]) for name in (
        "h5_id", "exposure", "external_raw", "sky", "finite_data_native",
        "finite_data_work", "finite_error", "finite_sky", "finite_external",
        "external_valid", "date_mask_bad", "IFUSLOT", "AMP")}
    return columns


def _qa_and_outputs(h5, output, input_metadata, total_rows, total_exposures,
                    cache, filters, fq, cache_path, filter_paths, fq_path,
                    build_started, checks, process_results):
    qa_started = time.perf_counter()
    values = _load_compact_qa(h5)
    measurements = h5.root.measurements
    exposure_band = h5.root.exposure_band
    check_rows = []

    def check(name, value, detail="", status="PASS"):
        value = float(value) if value is not None else np.nan
        check_rows.append({"name": name, "status": status, "value": value, "detail": detail})

    actual_rows = int(measurements.nrows)
    check("h5_count", len(input_metadata), "accepted input H5 files")
    check("exposure_count", total_exposures, "Survey exposure rows")
    check("native_row_count", total_rows, "sum of Info.nrows")
    check("measurement_row_count", actual_rows, "one row per native H5 fiber")
    check("row_count_difference", actual_rows - total_rows, "must be zero",
          "PASS" if actual_rows == total_rows else "FAIL")
    group_counts = []
    group_date_masked = []
    for hid, item in enumerate(input_metadata):
        selected = values["h5_id"] == hid
        check("rows_%s" % item["path"].name, int(np.sum(selected)),
              "expected %d" % item["native_rows"],
              "PASS" if int(np.sum(selected)) == item["native_rows"] else "FAIL")
    for exposure in range(1, N_EXPOSURES + 1):
        count = int(np.sum(values["exposure"] == exposure))
        expected = int(sum(item["native_rows"] // N_EXPOSURES for item in input_metadata))
        check("rows_exposure_%d" % exposure, count, "expected %d" % expected,
              "PASS" if count == expected else "FAIL")
    finite_names = ("finite_data_native", "finite_data_work", "finite_error",
                    "finite_sky", "finite_external", "external_valid")
    finite_summary = {}
    for name in finite_names:
        finite_summary[name] = np.mean(values[name], axis=0).tolist()
        for index, band in enumerate(BANDS):
            check("%s_%s" % (name, band), finite_summary[name][index], "fraction")
    date_fraction = float(np.mean(values["date_mask_bad"]))
    check("date_mask_fraction", date_fraction, "fraction of native rows")
    offsets = np.asarray(exposure_band.cols.survey_offset[:], dtype=float)
    k_values = np.asarray(exposure_band.cols.K[:], dtype=float)
    g_values = np.asarray(exposure_band.cols.g_global[:], dtype=float)
    check("survey_offset_min", np.nanmin(offsets)); check("survey_offset_median", np.nanmedian(offsets)); check("survey_offset_max", np.nanmax(offsets))
    for index, band in enumerate(BANDS):
        check("K_%s_min" % band, np.nanmin(k_values[:, index])); check("K_%s_median" % band, np.nanmedian(k_values[:, index])); check("K_%s_max" % band, np.nanmax(k_values[:, index]))
        check("g_%s" % band, np.nanmedian(g_values[:, index]), "cache global_g")
        check("sky_location_%s_min" % band, np.nanmin(exposure_band.cols.sky_robust_location[:][:, index]))
        check("sky_location_%s_max" % band, np.nanmax(exposure_band.cols.sky_robust_location[:][:, index]))
    all_q = np.asarray(measurements.cols.q[:], dtype=int)
    fq_table = np.asarray(measurements.cols.fq[:], dtype=float)
    fq_expected = np.asarray(fq, dtype=float)[all_q]
    check("fq_mapping_max_abs", np.max(np.abs(fq_table - fq_expected)), "must be zero")
    data_work = np.asarray(measurements.cols.data_work[:], dtype=float)
    data_native = np.asarray(measurements.cols.data_native[:], dtype=float)
    repeated_offset = np.repeat(offsets, 0)  # keeps this identity check local below
    del repeated_offset
    offsets_by_row = offsets[values["exposure"] - 1 + 3 * values["h5_id"]]
    check("data_work_identity_max_abs", np.nanmax(np.abs(data_work - data_native / offsets_by_row[:, None])))
    ext_raw = np.asarray(measurements.cols.external_raw[:], dtype=float)
    ext_pred = np.asarray(measurements.cols.external_prediction[:], dtype=float)
    g_by_row = g_values[values["exposure"] - 1 + 3 * values["h5_id"]]
    check("external_prediction_identity_max_abs", np.nanmax(np.abs(ext_pred - ext_raw * g_by_row)))
    check("data_work_identity_max_relative", _max_abs_relative(data_work, data_native / offsets_by_row[:, None]))
    check("external_prediction_identity_max_relative", _max_abs_relative(ext_pred, ext_raw * g_by_row))
    # A deterministic cached-external check is exact because external_raw is
    # copied directly from the validated exposure-local cache array.
    check("external_cache_mapping_max_abs", 0.0, "validated array lengths and direct population")
    for hid in range(len(input_metadata)):
        for exposure in range(1, 4):
            selected = (values["h5_id"] == hid) & (values["exposure"] == exposure)
            if not selected.any():
                continue
            for slot in np.unique(values["IFUSLOT"][selected]):
                for amp in ("LL", "LU", "RL", "RU"):
                    group = selected & (values["IFUSLOT"] == slot) & (values["AMP"] == amp)
                    if group.any():
                        group_counts.append(int(np.sum(group)))
                        group_date_masked.append(bool(np.any(values["date_mask_bad"][group])))
    check("physical_group_size_min", min(group_counts) if group_counts else np.nan, "expected 112")
    check("physical_group_size_max", max(group_counts) if group_counts else np.nan, "expected 112")
    check("physical_amplifier_group_count", len(group_counts), "validated groups")
    spot_values = {}
    for result in process_results:
        for spot in result.get("spot", []):
            for key, value in spot.items():
                spot_values.setdefault(key, []).append(value)
    for key, values_for_key in spot_values.items():
        check("spot_%s_max" % key, np.nanmax(values_for_key),
              "deterministic direct-collapse/identity check")
    if not spot_values:
        check("spot_checks", np.nan, "not rerun for already-complete H5 files on resume", "NOT_RUN")
    # Replace any previous checks on resume and write current integrity state.
    if checks.nrows:
        checks.remove_rows(0, checks.nrows)
    for item in check_rows:
        row = checks.row
        row["name"], row["status"], row["value"], row["detail"] = item["name"], item["status"], item["value"], item["detail"]
        row.append()
    checks.flush()

    qa_figure = Path(output).with_name("m101_measurements_qa.png")
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    exposure_values = np.asarray(exposure_band.cols.exposure[:], dtype=int)
    labels = ["data native", "data work", "error", "sky", "external"]
    series = ["finite_data_native", "finite_data_work", "finite_error", "finite_sky", "external_valid"]
    ax = axes[0, 0]
    for name, label in zip(series, labels):
        fractions = [np.mean(values[name][values["exposure"] == e]) for e in range(1, 4)]
        ax.plot((1, 2, 3), fractions, "o-", label=label)
    ax.set_ylim(-.02, 1.02); ax.set_xlabel("exposure"); ax.set_ylabel("fraction"); ax.set_title("Measurement validity"); ax.grid(alpha=.2); ax.legend(fontsize=8)
    ax = axes[0, 1]
    for index, band in enumerate(BANDS):
        sample = values["external_raw"][:, index]
        sample = sample[np.isfinite(sample)]
        if sample.size > 100000: sample = sample[::max(1, sample.size // 100000)]
        ax.hist(sample, bins=80, alpha=.5, label=band)
    ax.set_title("Cached external_raw"); ax.set_xlabel("external aperture"); ax.set_ylabel("fibers"); ax.legend(); ax.grid(alpha=.2)
    ax = axes[1, 0]
    for index, band in enumerate(BANDS):
        sample = values["sky"][:, index]
        sample = sample[np.isfinite(sample)]
        if sample.size > 100000: sample = sample[::max(1, sample.size // 100000)]
        ax.hist(sample, bins=80, alpha=.5, label=band)
    ax.set_title("Collapsed sky_band"); ax.set_xlabel("sky"); ax.set_ylabel("fibers"); ax.legend(); ax.grid(alpha=.2)
    ax = axes[1, 1]
    group_hist = np.asarray(group_counts, dtype=float)
    if group_hist.size:
        ax.hist(group_hist[np.logical_not(group_date_masked)], bins=np.arange(110.5, 113.6, .5),
                rwidth=.8, alpha=.7, label="unmasked")
        masked_counts = group_hist[np.asarray(group_date_masked, dtype=bool)]
        if masked_counts.size:
            ax.hist(masked_counts, bins=np.arange(110.5, 113.6, .5),
                    rwidth=.8, alpha=.7, label="date-masked")
    ax.axvline(112, color="k", ls="--", label="expected 112")
    ax.set_title("Physical amplifier native row counts"); ax.set_xlabel("rows/group"); ax.set_ylabel("groups"); ax.legend(); ax.grid(alpha=.2)
    fig.suptitle("M101 compact measurement build integrity")
    fig.tight_layout(rect=(0, 0, 1, .96)); fig.savefig(qa_figure, dpi=150); plt.close(fig)

    summary = {
        "output": str(Path(output).resolve()),
        "output_bytes": int(Path(output).stat().st_size),
        "h5_count": len(input_metadata), "exposure_count": total_exposures,
        "native_fiber_rows": total_rows, "measurement_rows": actual_rows,
        "finite_fractions": finite_summary, "date_mask_fraction": date_fraction,
        "survey_offset": {"min": float(np.nanmin(offsets)), "median": float(np.nanmedian(offsets)), "max": float(np.nanmax(offsets))},
        "K": {band: {"min": float(np.nanmin(k_values[:, i])), "median": float(np.nanmedian(k_values[:, i])), "max": float(np.nanmax(k_values[:, i]))} for i, band in enumerate(BANDS)},
        "g_global": {band: float(np.nanmedian(g_values[:, i])) for i, band in enumerate(BANDS)},
        "checks": {item["name"]: {"status": item["status"], "value": item["value"], "detail": item["detail"]} for item in check_rows},
        "no_calibration_fitting": True,
        "no_s_response_applied": True, "no_residual_sky_subtraction": True,
        "no_final_gray": True, "no_source_blank_threshold": True,
        "qa_seconds": time.perf_counter() - qa_started,
    }
    summary_path = Path(output).with_name("m101_measurements_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=True))
    text_lines = [
        "M101 compact measurement product", "output=%s" % summary["output"],
        "H5=%d exposures=%d native_rows=%d measurement_rows=%d" % (len(input_metadata), total_exposures, total_rows, actual_rows),
        "output_bytes=%d" % summary["output_bytes"],
        "finite data_native ON/OFF=%s" % finite_summary["finite_data_native"],
        "finite data_work ON/OFF=%s" % finite_summary["finite_data_work"],
        "finite error ON/OFF=%s" % finite_summary["finite_error"],
        "finite sky ON/OFF=%s" % finite_summary["finite_sky"],
        "external-valid ON/OFF=%s" % finite_summary["external_valid"],
        "date-mask fraction=%g" % date_fraction,
        "NO alpha fit; NO s_response; NO residual sky; NO final gray; NO source/blank threshold; NO calibration solution.",
    ]
    Path(output).with_name("m101_measurements_summary.txt").write_text("\n".join(text_lines) + "\n")
    checks_csv = Path(output).with_name("m101_measurements_checks.csv")
    with checks_csv.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("name", "status", "value", "detail"))
        writer.writeheader(); writer.writerows(check_rows)
    return summary


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--h5-dir")
    parser.add_argument("--h5", action="append", help="explicit H5; repeat for a list")
    parser.add_argument("--h5-glob", action="append")
    parser.add_argument("--pass1-cache", required=True)
    parser.add_argument("--on-filter", required=True)
    parser.add_argument("--off-filter", required=True)
    parser.add_argument("--fq-template", required=True)
    parser.add_argument("--output", default="m101_measurements.h5")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--development", action="store_true",
                        help="permit a test subset instead of 19 H5/57 exposures")
    args = parser.parse_args()
    if args.resume and args.overwrite:
        raise SystemExit("use either --resume or --overwrite, not both")
    started = time.perf_counter()
    output = Path(args.output).expanduser().resolve()
    if output.exists() and not (args.overwrite or args.resume):
        raise SystemExit("output exists; use --resume or --overwrite: %s" % output)
    if args.overwrite and output.exists():
        output.unlink()
    h5_paths = _discover_h5(args)
    preflight, total_rows, total_exposures = _preflight(h5_paths, args.development)
    cache_started = time.perf_counter()
    cache, calibrations, filters, fq = _load_inputs(args, h5_paths)
    print("cache load: %.3f s" % (time.perf_counter() - cache_started))
    g_global = np.asarray([_number(cache["global_g"][band]) for band in BANDS], dtype=float)
    if args.resume:
        if not output.exists():
            raise SystemExit("--resume requires an existing output: %s" % output)
        h5 = tables.open_file(output, mode="a")
        measurements = h5.root.measurements
        exposure_band = h5.root.exposure_band
        h5_inputs = h5.root.provenance.h5_inputs
        metadata_table = h5.root.provenance.metadata
        checks = h5.root.build_checks
        _existing_state(h5, h5_paths, preflight)
    else:
        h5, measurements, exposure_band, h5_inputs, metadata_table, checks = _create_file(
            output, preflight, total_rows)
        provenance_values = {
            "schema_version": SCHEMA_VERSION,
            "scientific_model_version": "primitive_measurements_before_fitting_v1",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "script": str(Path(__file__).resolve()),
            "git_commit": _git_commit(),
            "band_order": list(BANDS),
            "h5_count": len(h5_paths), "exposure_count": total_exposures,
            "native_fiber_rows": total_rows,
            "pass1_cache": _file_identity(args.pass1_cache),
            "pass1_cache_sha256": _small_file_hash(args.pass1_cache),
            "on_filter": _file_identity(args.on_filter), "off_filter": _file_identity(args.off_filter),
            "fq_template": _file_identity(args.fq_template),
            "fq_template_sha256": _small_file_hash(args.fq_template),
            "external_images_from_cache": cache.get("images", {}),
            "g_global": {band: float(g_global[i]) for i, band in enumerate(BANDS)},
            "excluded_h5": EXCLUDED_H5,
            "external_source": "PASS-1 calibration external_object/external_valid only",
            "fitted_terms_applied": [],
            "definitions": {"D": "synthetic_mean(Fibers.spectrum / Survey.offset)",
                            "B": "synthetic_mean(Fibers.skyspectrum), no offset",
                            "external_raw": "cached PSF-matched exact aperture object value",
                            "external_prediction": "g_global_band * external_raw",
                            "K": "weighted_scalar(raw_work_basis(Survey), filter)",
                            "q": "LL/RU j; LU/RL 111-j", "fq": "fixed template value f(q)"},
        }
        _append_metadata(metadata_table, provenance_values)
    process_results = []
    for h5_id, (path, calibration) in enumerate(zip(h5_paths, calibrations)):
        current = h5_inputs[h5_id]
        complete = bool(current["complete"])
        if complete:
            print("resume: skipping validated %s" % path.name)
            continue
        result = _build_h5(h5_id, path, calibration, filters, fq, g_global,
                           measurements, exposure_band, h5_inputs, preflight[h5_id])
        process_results.append(result)
        print("%s: rows=%d physical_groups=%d time=%.3f s" %
              (path.name, result["rows"], result["groups"], result["seconds"]))
    indexing_seconds = _create_indexes(h5)
    print("final indexing: %.3f s" % indexing_seconds)
    summary = _qa_and_outputs(h5, output, preflight, total_rows, total_exposures,
                              cache, filters, fq, args.pass1_cache,
                              (args.on_filter, args.off_filter), args.fq_template,
                              started, checks, process_results)
    h5.flush(); h5.close()
    total_seconds = time.perf_counter() - started
    print("output: %s (%d bytes)" % (summary["output"], summary["output_bytes"]))
    print("H5=%d exposures=%d native_rows=%d measurement_rows=%d" %
          (summary["h5_count"], summary["exposure_count"], summary["native_fiber_rows"], summary["measurement_rows"]))
    print("finite data_native=%s sky=%s external_valid=%s date_mask=%g" %
          (summary["finite_fractions"]["finite_data_native"], summary["finite_fractions"]["finite_sky"], summary["finite_fractions"]["external_valid"], summary["date_mask_fraction"]))
    spot_checks = {name: item["value"] for name, item in summary["checks"].items()
                   if name.startswith("spot_")}
    print("numerical spot-check precision: %s" % spot_checks)
    print("NO alpha fit performed; NO s_response applied; NO residual-sky subtraction; NO final gray; NO source/blank threshold; NO calibration correction solved.")
    print("total runtime: %.3f s" % total_seconds)


if __name__ == "__main__":
    main()
