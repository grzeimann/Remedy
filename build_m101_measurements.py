#!/usr/bin/env python3
"""Build the frozen compact M101 measurement product.

Each row in ``m101_measurements.h5:/measurements`` is one native VIRUS fiber
observation.  The product contains two ON/OFF external-source measurement
bands, five internally defined sky-null spectral bands, and an independently
derived external blank-fiber flag.  ON and OFF values are stored side-by-side,
with band index 0 always ON and band index 1 always OFF.

    D = synthetic collapse of Fibers.spectrum / Survey.offset
    B = synthetic collapse of Fibers.skyspectrum, without division
    I = cached background-subtracted, PSF-matched, exact external aperture
    X = g_band * I
    K = synthetic collapse of raw_work_basis(Survey)
    q = distance from the physical amplifier readout edge
    fq = fixed f(q) template value

The five sky-null bands are deterministic top-hat means of the native VIRUS
wavelength grid.  Their primitive data, errors, Quick Reduction sky values,
response completeness, and exposure-level blank-fiber QA are frozen for
future blank-fiber likelihood constraints.  They are not fitted or zeroed by
this builder.  The persisted external blank classification is consumed as an
input and does not remove measurement rows.

No fitted M101 alpha, s_response, residual-sky correction, spatial plane,
beta, exposure gray, or source threshold is applied.  Future fitting
can therefore use the compact product without reopening full spectra.

For an externally blank row, the future null constraint is represented by the
stored primitives ``D_null = p + alpha H_null`` with ``P_null = 0``.  This
builder only stores ``D_null``, its propagated error, the primitive Quick
Reduction sky, ``K_null``, and the external ``sky_blank`` flag; it does not
fit those terms or subtract, center, or zero the null measurements.
"""

from argparse import ArgumentParser
import csv
from datetime import datetime, timezone
import gzip
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
SKY_NULL_BANDS = (
    "NULL_HIGH1",
    "NULL_LOW1",
    "NULL_HIGH2",
    "NULL_LOW2",
    "NULL_HIGH3",
)
SKY_NULL_WINDOWS_A = (
    (3538.0, 3550.0),
    (3602.0, 3614.0),
    (3628.0, 3640.0),
    (3676.0, 3688.0),
    (3902.0, 3914.0),
)
N_SKY_NULL_BANDS = len(SKY_NULL_BANDS)
EXPECTED_VIRUS_WAVE = np.linspace(3470.0, 5540.0, 1036)
SCHEMA_VERSION = "m101_measurements_v2"
SCIENTIFIC_MODEL_VERSION = "primitive_measurements_with_external_blank_null_bands_v2"


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
    sky_blank = tables.BoolCol(pos=32)
    sky_null_data_native = tables.Float64Col(shape=(N_SKY_NULL_BANDS,), pos=33)
    sky_null_data_work = tables.Float64Col(shape=(N_SKY_NULL_BANDS,), pos=34)
    sky_null_error_native = tables.Float64Col(shape=(N_SKY_NULL_BANDS,), pos=35)
    sky_null_error_work = tables.Float64Col(shape=(N_SKY_NULL_BANDS,), pos=36)
    sky_null_sky = tables.Float64Col(shape=(N_SKY_NULL_BANDS,), pos=37)
    sky_null_finite_data_native = tables.BoolCol(shape=(N_SKY_NULL_BANDS,), pos=38)
    sky_null_finite_data_work = tables.BoolCol(shape=(N_SKY_NULL_BANDS,), pos=39)
    sky_null_finite_error = tables.BoolCol(shape=(N_SKY_NULL_BANDS,), pos=40)
    sky_null_finite_sky = tables.BoolCol(shape=(N_SKY_NULL_BANDS,), pos=41)
    sky_null_data_response_fraction = tables.Float64Col(shape=(N_SKY_NULL_BANDS,), pos=42)
    sky_null_sky_response_fraction = tables.Float64Col(shape=(N_SKY_NULL_BANDS,), pos=43)


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


class SkyNullExposureDescription(tables.IsDescription):
    h5_id = tables.Int16Col(pos=0)
    exposure = tables.UInt8Col(pos=1)
    K = tables.Float64Col(shape=(N_SKY_NULL_BANDS,), pos=2)
    blank_data_work_location = tables.Float64Col(shape=(N_SKY_NULL_BANDS,), pos=3)
    blank_data_work_scale = tables.Float64Col(shape=(N_SKY_NULL_BANDS,), pos=4)
    n_blank_external = tables.Int64Col(pos=5)
    n_blank_usable = tables.Int64Col(shape=(N_SKY_NULL_BANDS,), pos=6)
    data_finite_fraction = tables.Float64Col(shape=(N_SKY_NULL_BANDS,), pos=7)
    error_finite_fraction = tables.Float64Col(shape=(N_SKY_NULL_BANDS,), pos=8)
    sky_finite_fraction = tables.Float64Col(shape=(N_SKY_NULL_BANDS,), pos=9)


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


def _validate_sky_null_responses():
    """Construct the frozen inclusive top-hats on the canonical VIRUS grid."""
    wave = np.asarray(hm.DEF_WAVE, dtype=float)
    if wave.shape != EXPECTED_VIRUS_WAVE.shape or not np.array_equal(
            wave, EXPECTED_VIRUS_WAVE):
        raise ValueError("hm.DEF_WAVE is not the expected 1036-sample VIRUS grid "
                         "from 3470.0 to 5540.0 A")
    responses = []
    actual = {}
    for name, (lower, upper) in zip(SKY_NULL_BANDS, SKY_NULL_WINDOWS_A):
        if not np.isfinite(lower) or not np.isfinite(upper) or lower > upper:
            raise ValueError("invalid sky-null window for %s" % name)
        response = np.where((wave >= lower) & (wave <= upper), 1.0, 0.0)
        samples = wave[response != 0.0]
        if samples.size < 3 or np.sum(response) <= 0.0:
            raise ValueError("sky-null window %s contains too few VIRUS samples" % name)
        if np.any(samples < lower) or np.any(samples > upper):
            raise ValueError("sky-null response escaped requested bounds for %s" % name)
        responses.append(response)
        actual[name] = {
            "requested_bounds_A": [float(lower), float(upper)],
            "samples_A": [float(value) for value in samples],
            "minimum_A": float(samples.min()),
            "maximum_A": float(samples.max()),
            "center_A": float(np.mean(samples)),
            "n_samples": int(samples.size),
            "total_weight": float(np.sum(response)),
        }
    provenance = {
        "wave_grid": {"array": "hm.DEF_WAVE", "start_A": float(wave[0]),
                      "stop_A": float(wave[-1]), "n_samples": int(wave.size),
                      "step_A": float(wave[1] - wave[0])},
        "band_order": list(SKY_NULL_BANDS),
        "requested_windows_A": [list(window) for window in SKY_NULL_WINDOWS_A],
        "bands": actual,
        "response_definition": "1 inside inclusive requested limits, 0 outside",
        "blank_usable_definition": "sky_blank AND not date_mask_bad AND finite sky_null_data_work AND finite sky_null_error_work AND sky_null_error_work > 0",
        "blank_finite_fraction_definition": "fractions are among sky_blank AND not date_mask_bad rows for each exposure and null band",
    }
    return np.asarray(responses, dtype=float), provenance


def _parse_external_blank_bool(value):
    value = str(value).strip().lower()
    if value in {"true", "1"}:
        return True
    if value in {"false", "0"}:
        return False
    raise ValueError("external blank table has ambiguous blank value: %r" % value)


def _load_external_blank_fibers(path, h5_paths):
    """Load a complete external blank mask per requested H5 basename/row."""
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise ValueError("external blank-fiber table does not exist: %s" % path)
    input_by_name = {}
    masks = {}
    labels_by_name = {}
    for h5_path in h5_paths:
        name = h5_path.name
        if name in input_by_name:
            raise ValueError("input H5 basenames are ambiguous: %s" % name)
        with tables.open_file(h5_path, mode="r") as h5:
            if "Info" not in h5.root._v_children:
                raise ValueError("input H5 has no Info table: %s" % h5_path)
            info = h5.root.Info
            groups, labels = hm.build_groups(info)
            del groups
            nrows = int(info.nrows)
        input_by_name[name] = h5_path
        masks[name] = {"seen": np.zeros(nrows, dtype=bool),
                       "blank": np.zeros(nrows, dtype=bool)}
        labels_by_name[name] = np.asarray(labels, dtype=int)

    opener = gzip.open if path.suffix == ".gz" else open
    rows_parsed = 0
    matching_rows = 0
    extra_h5_names = set()
    extra_seen = set()
    with opener(path, "rt", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"H5", "exposure", "row_index", "blank"}
        fields = set(reader.fieldnames or ())
        missing = required - fields
        if missing:
            raise ValueError("external blank table lacks columns: %s" % sorted(missing))
        for row in reader:
            rows_parsed += 1
            if row.get(None) is not None:
                raise ValueError("malformed external blank CSV row %d has extra fields" % rows_parsed)
            raw_h5 = str(row.get("H5", "")).strip()
            if not raw_h5:
                raise ValueError("external blank CSV row %d has an empty H5" % rows_parsed)
            h5_name = Path(raw_h5).name
            try:
                row_index = int(str(row.get("row_index", "")).strip())
                exposure = int(str(row.get("exposure", "")).strip())
            except (TypeError, ValueError) as exc:
                raise ValueError("invalid row_index/exposure for %s row %d" %
                                 (h5_name, rows_parsed)) from exc
            if row_index < 0 or exposure < 1 or exposure > N_EXPOSURES:
                raise ValueError("invalid row_index/exposure for %s row %d" %
                                 (h5_name, rows_parsed))
            blank = _parse_external_blank_bool(row.get("blank"))
            if h5_name not in input_by_name:
                extra_key = (h5_name, row_index)
                if extra_key in extra_seen:
                    raise ValueError("duplicate external blank row for %s row_index %d" %
                                     (h5_name, row_index))
                extra_seen.add(extra_key)
                extra_h5_names.add(h5_name)
                continue
            matching_rows += 1
            state = masks[h5_name]
            if row_index >= state["seen"].size:
                raise ValueError("external blank row_index %d is out of range for %s" %
                                 (row_index, h5_name))
            if state["seen"][row_index]:
                raise ValueError("duplicate external blank row for %s row_index %d" %
                                 (h5_name, row_index))
            expected_exposure = labels_by_name[h5_name][row_index]
            if exposure != expected_exposure:
                raise ValueError("external blank exposure mismatch for %s row_index %d: "
                                 "CSV=%d H5=%d" %
                                 (h5_name, row_index, exposure, expected_exposure))
            state["blank"][row_index] = blank
            state["seen"][row_index] = True

    for h5_name, state in masks.items():
        if not np.all(state["seen"]):
            missing_rows = int(np.sum(~state["seen"]))
            raise ValueError("external blank table is incomplete for %s: %d Info rows missing" %
                             (h5_name, missing_rows))

    result = {name: state["blank"] for name, state in masks.items()}
    matched_blank = int(sum(np.sum(mask) for mask in result.values()))
    matched_total = int(sum(mask.size for mask in result.values()))
    provenance = {
        "path": str(path),
        "identity": _file_identity(path),
        "sha256": _small_file_hash(path),
        "rows": rows_parsed,
        "matched_rows": matching_rows,
        "blank_rows": matched_blank,
        "nonblank_rows": matched_total - matched_blank,
        "h5_files_represented": len(input_by_name) + len(extra_h5_names),
        "input_h5_files_matched": len(input_by_name),
        "ignored_extra_h5_files": len(extra_h5_names),
        "ignored_extra_h5_names": sorted(extra_h5_names),
        "matching_identity": "H5 basename + row_index; CSV exposure cross-checked against validated labels",
        "classification_semantics": "externally derived classification consumed as input; no VIRUS spectral quantity used to define blankness",
    }
    return result, provenance


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
                specid_all, slot_all, uid_all, fibers, null_responses,
                external_blank):
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
    null_data_columns = [_collapse_with_fraction(source, response)
                         for response in null_responses]
    null_error_columns = [_collapse_error(error, response)
                          for response in null_responses]
    null_sky_columns = [_collapse_with_fraction(sky_native, response)
                        for response in null_responses]
    sky_null_data_native = np.column_stack([column[0] for column in null_data_columns])
    sky_null_data_fraction = np.column_stack([column[1] for column in null_data_columns])
    sky_null_error_native = np.column_stack([column[0] for column in null_error_columns])
    sky_null_sky = np.column_stack([column[0] for column in null_sky_columns])
    sky_null_sky_fraction = np.column_stack([column[1] for column in null_sky_columns])
    sky_null_blank = np.asarray(external_blank[exp_indices], dtype=bool)
    if sky_null_blank.shape != (exp_indices.size,):
        raise ValueError("external blank mask does not match %s exposure %d" %
                         (path, exposure))
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
    direct_null_data = np.column_stack([
        hm.synthetic_mean(source[sample], response) for response in null_responses])
    direct_null_error = np.column_stack([
        _collapse_error(error[sample], response)[0] for response in null_responses])
    direct_null_sky = np.column_stack([
        hm.synthetic_mean(sky_native[sample], response) for response in null_responses])
    direct_null_spot = {
        "null_data_native_abs": float(np.nanmax(
            np.abs(direct_null_data - sky_null_data_native[sample]))) if sample.size else np.nan,
        "null_data_native_rel": _max_abs_relative(
            direct_null_data, sky_null_data_native[sample]),
        "null_error_native_abs": float(np.nanmax(
            np.abs(direct_null_error - sky_null_error_native[sample]))) if sample.size else np.nan,
        "null_error_native_rel": _max_abs_relative(
            direct_null_error, sky_null_error_native[sample]),
        "null_sky_abs": float(np.nanmax(
            np.abs(direct_null_sky - sky_null_sky[sample]))) if sample.size else np.nan,
        "null_sky_rel": _max_abs_relative(direct_null_sky, sky_null_sky[sample]),
    }
    data_work = data_native / offset
    error_work = error_native / abs(offset)
    sky_null_data_work = sky_null_data_native / offset
    sky_null_error_work = sky_null_error_native / abs(offset)
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
    batch["sky_blank"] = sky_null_blank
    batch["sky_null_data_native"] = sky_null_data_native
    batch["sky_null_data_work"] = sky_null_data_work
    batch["sky_null_error_native"] = sky_null_error_native
    batch["sky_null_error_work"] = sky_null_error_work
    batch["sky_null_sky"] = sky_null_sky
    batch["sky_null_finite_data_native"] = np.isfinite(sky_null_data_native)
    batch["sky_null_finite_data_work"] = np.isfinite(sky_null_data_work)
    batch["sky_null_finite_error"] = np.isfinite(sky_null_error_native)
    batch["sky_null_finite_sky"] = np.isfinite(sky_null_sky)
    batch["sky_null_data_response_fraction"] = sky_null_data_fraction
    batch["sky_null_sky_response_fraction"] = sky_null_sky_fraction
    null_candidate = sky_null_blank & ~date_bad_all[exp_indices]
    null_usable = (null_candidate[:, None] &
                   np.isfinite(sky_null_data_work) &
                   np.isfinite(sky_null_error_work) &
                   (sky_null_error_work > 0.0))
    null_data_finite_fraction = np.asarray([
        np.mean(np.isfinite(sky_null_data_work[null_candidate, index]))
        if np.any(null_candidate) else np.nan
        for index in range(N_SKY_NULL_BANDS)])
    null_error_finite_fraction = np.asarray([
        np.mean(np.isfinite(sky_null_error_work[null_candidate, index]))
        if np.any(null_candidate) else np.nan
        for index in range(N_SKY_NULL_BANDS)])
    null_sky_finite_fraction = np.asarray([
        np.mean(np.isfinite(sky_null_sky[null_candidate, index]))
        if np.any(null_candidate) else np.nan
        for index in range(N_SKY_NULL_BANDS)])
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
        "K_null": np.asarray([hm.weighted_scalar(raw_basis, response)
                              for response in null_responses]),
        "n_blank_external": int(np.sum(sky_null_blank)),
        "n_blank_usable": np.sum(null_usable, axis=0).astype(np.int64),
        "null_data_finite_fraction": null_data_finite_fraction,
        "null_error_finite_fraction": null_error_finite_fraction,
        "null_sky_finite_fraction": null_sky_finite_fraction,
        "null_location": np.asarray([
            hm.robust_location(sky_null_data_work[null_usable[:, index], index])
            for index in range(N_SKY_NULL_BANDS)]),
        "null_scale": np.asarray([
            hm.robust_scale(sky_null_data_work[null_usable[:, index], index])
            for index in range(N_SKY_NULL_BANDS)]),
        "spot": dict(_spot_checks(batch, exp_indices, groups, filters, fq, survey,
                                   g_global, source=None, error=None, sky=None),
                     **direct_spot, **direct_null_spot),
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


def _sky_null_exposure_row(metadata, table):
    row = np.zeros(1, dtype=table.dtype)
    row["h5_id"] = metadata["h5_id"]
    row["exposure"] = metadata["exposure"]
    row["K"] = metadata["K_null"]
    row["blank_data_work_location"] = metadata["null_location"]
    row["blank_data_work_scale"] = metadata["null_scale"]
    row["n_blank_external"] = metadata["n_blank_external"]
    row["n_blank_usable"] = metadata["n_blank_usable"]
    row["data_finite_fraction"] = metadata["null_data_finite_fraction"]
    row["error_finite_fraction"] = metadata["null_error_finite_fraction"]
    row["sky_finite_fraction"] = metadata["null_sky_finite_fraction"]
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
    sky_null_exposure = h5.create_table("/", "sky_null_exposure",
                                        SkyNullExposureDescription,
                                        expectedrows=len(input_metadata) * 3,
                                        filters=filters)
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
    root_attrs.sky_null_band_order = json.dumps(list(SKY_NULL_BANDS))
    root_attrs.sky_null_requested_windows_A = json.dumps(
        [list(window) for window in SKY_NULL_WINDOWS_A])
    root_attrs.scientific_model_version = SCIENTIFIC_MODEL_VERSION
    root_attrs.no_fitted_calibration_applied = True
    root_attrs.no_residual_sky_subtraction = True
    root_attrs.no_null_band_centering = True
    root_attrs.external_blank_classification_ingested = True
    root_attrs.external_blank_classification_filters_rows = False
    return h5, measurements, exposure_band, sky_null_exposure, h5_inputs, metadata, checks


def _append_metadata(metadata_table, values):
    for key, value in _metadata_rows(values):
        row = metadata_table.row
        row["key"], row["value"] = key, value
        row.append()
    metadata_table.flush()


def _mark_complete(h5_inputs, h5_id, complete):
    h5_inputs.cols.complete[int(h5_id)] = bool(complete)
    h5_inputs.flush()


def _metadata_dict(metadata_table):
    result = {}
    for row in metadata_table:
        key = row["key"]
        value = row["value"]
        if isinstance(key, bytes):
            key = key.decode()
        if isinstance(value, bytes):
            value = value.decode()
        try:
            result[str(key)] = json.loads(value)
        except (TypeError, ValueError):
            result[str(key)] = value
    return result


def _existing_state(h5, h5_paths, preflight, blank_provenance,
                    on_filter, off_filter, fq_template):
    if ("/provenance/h5_inputs" not in h5 or
            h5.root._v_attrs.schema_version != SCHEMA_VERSION or
            "/sky_null_exposure" not in h5):
        raise ValueError("existing output is not a compatible M101 measurement product")
    attrs = h5.root._v_attrs
    saved_null_bands = json.loads(str(attrs.sky_null_band_order))
    saved_null_windows = json.loads(str(attrs.sky_null_requested_windows_A))
    if saved_null_bands != list(SKY_NULL_BANDS) or saved_null_windows != [
            list(window) for window in SKY_NULL_WINDOWS_A]:
        raise ValueError("resume sky-null definitions differ from this builder")
    table = h5.root.provenance.h5_inputs
    if table.nrows != len(h5_paths):
        raise ValueError("resume input H5 count differs from existing product")
    for hid, path in enumerate(h5_paths):
        row = table[hid]
        saved_identity = {
            "filename": row["filename"].decode() if isinstance(row["filename"], bytes)
            else str(row["filename"]),
            "full_path": row["full_path"].decode() if isinstance(row["full_path"], bytes)
            else str(row["full_path"]),
            "file_size": int(row["file_size"]),
            "mtime_ns": int(row["mtime_ns"]),
        }
        if saved_identity != preflight[hid]["identity"]:
            raise ValueError("resume H5 identity differs for %s" % path.name)
        if Path(saved_identity["filename"]).name != path.name:
            raise ValueError("resume H5 ordering differs at h5_id=%d" % hid)
        expected = int(preflight[hid]["native_rows"])
        if int(row["native_rows"]) != expected:
            raise ValueError("resume H5 row count differs for %s" % path.name)
        if bool(row["complete"]):
            count = int(np.sum(h5.root.measurements.cols.h5_id[:] == hid))
            if count != expected:
                raise ValueError("completed H5 %s has %d/%d measurement rows" %
                                 (path.name, count, expected))
            null_count = int(np.sum(h5.root.sky_null_exposure.cols.h5_id[:] == hid))
            if null_count != N_EXPOSURES:
                raise ValueError("completed H5 %s has %d/%d sky-null exposure rows" %
                                 (path.name, null_count, N_EXPOSURES))
        else:
            count = int(np.sum(h5.root.measurements.cols.h5_id[:] == hid))
            if count:
                raise ValueError("incomplete H5 %s already has rows; use --overwrite" % path.name)
            null_count = int(np.sum(h5.root.sky_null_exposure.cols.h5_id[:] == hid))
            if null_count:
                raise ValueError("incomplete H5 %s already has sky-null exposure rows; use --overwrite" % path.name)
    metadata = _metadata_dict(h5.root.provenance.metadata)
    expected_files = {
        "on_filter": _file_identity(on_filter),
        "off_filter": _file_identity(off_filter),
        "fq_template": _file_identity(fq_template),
        "external_blank_fibers": blank_provenance["identity"],
    }
    expected_hashes = {
        "on_filter_sha256": _small_file_hash(on_filter),
        "off_filter_sha256": _small_file_hash(off_filter),
        "fq_template_sha256": _small_file_hash(fq_template),
        "external_blank_fibers_sha256": blank_provenance["sha256"],
    }
    for key, expected in expected_files.items():
        if metadata.get(key) != expected:
            raise ValueError("resume %s identity differs" % key)
    for key, expected in expected_hashes.items():
        if metadata.get(key) != expected:
            raise ValueError("resume %s differs" % key)


def _build_h5(h5_id, path, calibration, filters, fq, g_global, measurements,
              exposure_band, sky_null_exposure, h5_inputs, preflight,
              null_responses, external_blank_by_h5):
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
        if path.name not in external_blank_by_h5 or \
                external_blank_by_h5[path.name].size != nrows:
            raise ValueError("external blank mask does not match %s" % path)
        batches, exposure_rows, spot = [], [], []
        for exposure in range(1, N_EXPOSURES + 1):
            batch, metadata = _row_arrays(
                h5_id, path, calibration, filters, fq, g_global, measurements,
                exposure, surveys[exposure], groups, labels, ra_all, dec_all,
                amps_all, date_bad_all, specid_all, slot_all, uid_all, fibers,
                null_responses, external_blank_by_h5[path.name])
            if len(batch) != int(np.sum(labels == exposure)):
                raise ValueError("%s exposure %d compact row count mismatch" % (path, exposure))
            batches.append(batch)
            exposure_rows.append(_exposure_row(metadata, exposure_band))
            sky_null_exposure.append(_sky_null_exposure_row(metadata, sky_null_exposure))
            spot.append(metadata["spot"])
    combined = np.concatenate(batches)
    if len(combined) != nrows:
        raise ValueError("%s produced %d/%d native measurement rows" % (path, len(combined), nrows))
    # One append per H5 keeps the normal failure path before any rows from this
    # H5 become visible.  The complete marker is written only after validation.
    measurements.append(combined)
    exposure_band.append(np.concatenate(exposure_rows))
    sky_null_exposure.flush()
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
        column_object = getattr(h5.root.sky_null_exposure.cols, column)
        if not column_object.is_indexed:
            column_object.create_index()
    return time.perf_counter() - started


def _load_compact_qa(h5):
    table = h5.root.measurements
    columns = {name: np.asarray(getattr(table.cols, name)[:]) for name in (
        "h5_id", "exposure", "SPECID", "IFUSLOT", "IFUID", "external_raw", "sky",
        "sky_blank", "sky_null_data_native", "sky_null_data_work",
        "sky_null_error_native", "sky_null_error_work", "sky_null_sky",
        "sky_null_finite_data_native", "sky_null_finite_data_work",
        "sky_null_finite_error", "sky_null_finite_sky",
        "sky_null_data_response_fraction", "sky_null_sky_response_fraction",
        "finite_data_native",
        "finite_data_work", "finite_error", "finite_sky", "finite_external",
        "external_valid", "date_mask_bad")}
    columns["AMP"] = np.asarray([
        value.decode() if isinstance(value, (bytes, np.bytes_)) else str(value)
        for value in table.cols.AMP[:]])
    return columns


def _distribution(values):
    values = _finite(values)
    if not values.size:
        return {name: np.nan for name in
                ("minimum", "p05", "p16", "median", "p84", "p95", "maximum")}
    percentiles = np.percentile(values, [0, 5, 16, 50, 84, 95, 100])
    return {name: float(value) for name, value in zip(
        ("minimum", "p05", "p16", "median", "p84", "p95", "maximum"),
        percentiles)}


def _support_summary(rows):
    counts = np.asarray([row["N_blank_external"] for row in rows], dtype=float)
    if not counts.size:
        return {"n_amplifier_observations": 0, "distribution": _distribution(counts),
                "threshold_counts": {"zero": 0, "less_than_5": 0,
                                     "less_than_10": 0, "less_than_20": 0},
                "threshold_fractions": {"zero": np.nan, "less_than_5": np.nan,
                                         "less_than_10": np.nan, "less_than_20": np.nan}}
    threshold_counts = {
        "zero": int(np.sum(counts == 0)),
        "less_than_5": int(np.sum(counts < 5)),
        "less_than_10": int(np.sum(counts < 10)),
        "less_than_20": int(np.sum(counts < 20)),
    }
    return {
        "n_amplifier_observations": int(counts.size),
        "distribution": _distribution(counts),
        "threshold_counts": threshold_counts,
        "threshold_fractions": {key: float(value / counts.size)
                                 for key, value in threshold_counts.items()},
    }


def _amplifier_support_rows(values, input_metadata):
    rows = []
    for h5_id, item in enumerate(input_metadata):
        for exposure in range(1, N_EXPOSURES + 1):
            exposure_selected = ((values["h5_id"] == h5_id) &
                                 (values["exposure"] == exposure))
            for slot in np.unique(values["IFUSLOT"][exposure_selected]):
                for amp in ("LL", "LU", "RL", "RU"):
                    selected = (exposure_selected &
                                (values["IFUSLOT"] == slot) &
                                (values["AMP"] == amp))
                    if not selected.any():
                        continue
                    indices = np.flatnonzero(selected)
                    if indices.size != N_FIBER_AMP:
                        raise ValueError("%s exposure %d %s/%s has %d rows, not 112" %
                                         (item["path"].name, exposure, slot, amp,
                                          indices.size))
                    specids = np.unique(values["SPECID"][indices])
                    ifuids = np.unique(values["IFUID"][indices])
                    if specids.size != 1 or ifuids.size != 1:
                        raise ValueError("inconsistent amplifier identity for %s exposure %d %s/%s" %
                                         (item["path"].name, exposure, slot, amp))
                    blank = values["sky_blank"][indices]
                    candidate = blank & ~values["date_mask_bad"][indices]
                    usable = (candidate[:, None] &
                              np.isfinite(values["sky_null_data_work"][indices]) &
                              np.isfinite(values["sky_null_error_work"][indices]) &
                              (values["sky_null_error_work"][indices] > 0.0))
                    result = {
                        "H5": item["path"].name,
                        "h5_id": int(h5_id),
                        "exposure": int(exposure),
                        "SPECID": int(specids[0]),
                        "IFUSLOT": int(slot),
                        "IFUID": int(ifuids[0]),
                        "AMP": amp,
                        "N_native": int(indices.size),
                        "N_blank_external": int(np.sum(blank)),
                    }
                    for band_index, band in enumerate(SKY_NULL_BANDS):
                        result["N_blank_usable_%s" % band] = int(np.sum(usable[:, band_index]))
                    rows.append(result)
    return rows


def _qa_and_outputs(h5, output, input_metadata, total_rows, total_exposures,
                    cache, filters, fq, cache_path, filter_paths, fq_path,
                    build_started, checks, process_results, blank_provenance,
                    null_provenance):
    qa_started = time.perf_counter()
    values = _load_compact_qa(h5)
    measurements = h5.root.measurements
    exposure_band = h5.root.exposure_band
    sky_null_exposure = h5.root.sky_null_exposure
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
    null_finite_names = {
        "sky_null_finite_data_native": "data_native",
        "sky_null_finite_data_work": "data_work",
        "sky_null_finite_error": "error",
        "sky_null_finite_sky": "QR_sky",
    }
    null_finite_summary = {}
    for name, label in null_finite_names.items():
        null_finite_summary[name] = np.mean(values[name], axis=0).tolist()
        for index, band in enumerate(SKY_NULL_BANDS):
            check("%s_%s" % (name, band), null_finite_summary[name][index],
                  "fraction")
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
    null_data_native = np.asarray(measurements.cols.sky_null_data_native[:], dtype=float)
    null_data_work = np.asarray(measurements.cols.sky_null_data_work[:], dtype=float)
    null_error_native = np.asarray(measurements.cols.sky_null_error_native[:], dtype=float)
    null_error_work = np.asarray(measurements.cols.sky_null_error_work[:], dtype=float)
    check("sky_null_data_work_identity_max_abs", np.nanmax(
        np.abs(null_data_work - null_data_native / offsets_by_row[:, None])))
    check("sky_null_error_work_identity_max_abs", np.nanmax(
        np.abs(null_error_work - null_error_native /
               np.abs(offsets_by_row[:, None]))))
    check("sky_null_data_work_identity_max_relative", _max_abs_relative(
        null_data_work, null_data_native / offsets_by_row[:, None]))
    check("sky_null_error_work_identity_max_relative", _max_abs_relative(
        null_error_work, null_error_native / np.abs(offsets_by_row[:, None])))
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
    amplifier_rows = _amplifier_support_rows(values, input_metadata)
    check("physical_group_size_min", min(group_counts) if group_counts else np.nan, "expected 112")
    check("physical_group_size_max", max(group_counts) if group_counts else np.nan, "expected 112")
    check("physical_amplifier_group_count", len(group_counts), "validated groups")
    check("sky_null_exposure_row_count", int(sky_null_exposure.nrows),
          "one row per H5 exposure")
    null_k_values = np.asarray(sky_null_exposure.cols.K[:], dtype=float)
    null_locations = np.asarray(
        sky_null_exposure.cols.blank_data_work_location[:], dtype=float)
    null_scales = np.asarray(
        sky_null_exposure.cols.blank_data_work_scale[:], dtype=float)
    for index, band in enumerate(SKY_NULL_BANDS):
        check("K_%s_min" % band, np.nanmin(null_k_values[:, index]))
        check("K_%s_median" % band, np.nanmedian(null_k_values[:, index]))
        check("K_%s_max" % band, np.nanmax(null_k_values[:, index]))
        check("blank_location_%s_finite_fraction" % band,
              np.mean(np.isfinite(null_locations[:, index])))
        check("blank_scale_%s_finite_fraction" % band,
              np.mean(np.isfinite(null_scales[:, index])))
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

    blank_csv = Path(output).with_name("m101_measurements_blank_amplifier_counts.csv")
    blank_fields = ["H5", "h5_id", "exposure", "SPECID", "IFUSLOT", "IFUID", "AMP",
                    "N_native", "N_blank_external"] + [
                        "N_blank_usable_%s" % band for band in SKY_NULL_BANDS]
    with blank_csv.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=blank_fields)
        writer.writeheader()
        writer.writerows(amplifier_rows)

    support = _support_summary(amplifier_rows)
    support_by_h5 = {
        item["path"].name: _support_summary([
            row for row in amplifier_rows if row["H5"] == item["path"].name])
        for item in input_metadata}
    support_by_exposure = {
        str(exposure): _support_summary([
            row for row in amplifier_rows if row["exposure"] == exposure])
        for exposure in range(1, N_EXPOSURES + 1)}
    support_by_h5_exposure = {
        "%s/exposure_%d" % (item["path"].name, exposure): _support_summary([
            row for row in amplifier_rows if row["H5"] == item["path"].name and
            row["exposure"] == exposure])
        for item in input_metadata for exposure in range(1, N_EXPOSURES + 1)}

    qa_figure = Path(output).with_name("m101_measurements_qa.png")
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    del exposure_band
    labels = ["data native", "data work", "error", "sky", "external"]
    series = ["finite_data_native", "finite_data_work", "finite_error", "finite_sky", "external_valid"]
    ax = axes[0, 0]
    for name, label in zip(series, labels):
        fractions = [np.mean(values[name][values["exposure"] == e]) for e in range(1, 4)]
        ax.plot((1, 2, 3), fractions, "o-", label=label)
    ax.set_ylim(-.02, 1.02); ax.set_xlabel("exposure"); ax.set_ylabel("fraction"); ax.set_title("Measurement validity"); ax.grid(alpha=.2); ax.legend(fontsize=8)
    ax = axes[0, 1]
    blank_counts = np.asarray([row["N_blank_external"] for row in amplifier_rows], dtype=float)
    if blank_counts.size:
        ax.hist(blank_counts, bins=min(40, max(5, int(np.ptp(blank_counts) + 1))),
                alpha=.8, color="tab:purple")
    ax.set_title("External blank fibers per amplifier")
    ax.set_xlabel("N_blank_external"); ax.set_ylabel("amplifiers"); ax.grid(alpha=.2)
    ax = axes[1, 0]
    for index, band in enumerate(SKY_NULL_BANDS):
        sample = null_locations[:, index]
        sample = sample[np.isfinite(sample)]
        if sample.size:
            ax.hist(sample, bins=30, alpha=.45, label=band)
    ax.set_title("Blank null data robust location")
    ax.set_xlabel("sky_null_data_work"); ax.set_ylabel("H5 exposures")
    ax.legend(fontsize=8); ax.grid(alpha=.2)
    ax = axes[1, 1]
    for index, band in enumerate(SKY_NULL_BANDS):
        sample = null_scales[:, index]
        sample = sample[np.isfinite(sample)]
        if sample.size:
            ax.hist(sample, bins=30, alpha=.45, label=band)
    ax.set_title("Blank null data robust scale")
    ax.set_xlabel("sky_null_data_work"); ax.set_ylabel("H5 exposures")
    ax.legend(fontsize=8); ax.grid(alpha=.2)
    fig.suptitle("M101 compact measurement build integrity")
    fig.tight_layout(rect=(0, 0, 1, .96)); fig.savefig(qa_figure, dpi=150); plt.close(fig)

    summary_path = Path(output).with_name("m101_measurements_summary.json")
    checks_csv = Path(output).with_name("m101_measurements_checks.csv")
    summary = {
        "output": str(Path(output).resolve()),
        "output_bytes": int(Path(output).stat().st_size),
        "h5_count": len(input_metadata), "exposure_count": total_exposures,
        "native_fiber_rows": total_rows, "measurement_rows": actual_rows,
        "finite_fractions": finite_summary, "date_mask_fraction": date_fraction,
        "sky_null_band_order": list(SKY_NULL_BANDS),
        "sky_null_definitions": null_provenance,
        "sky_null_finite_fractions": null_finite_summary,
        "sky_null_response_completeness_distribution": {
            band: {"data": _distribution(values["sky_null_data_response_fraction"][:, index]),
                   "sky": _distribution(values["sky_null_sky_response_fraction"][:, index])}
            for index, band in enumerate(SKY_NULL_BANDS)},
        "sky_null_K": {band: {"min": float(np.nanmin(null_k_values[:, i])),
                              "median": float(np.nanmedian(null_k_values[:, i])),
                              "max": float(np.nanmax(null_k_values[:, i]))}
                       for i, band in enumerate(SKY_NULL_BANDS)},
        "sky_null_blank_robust_location": {
            band: _distribution(null_locations[:, index])
            for index, band in enumerate(SKY_NULL_BANDS)},
        "sky_null_blank_robust_scale": {
            band: _distribution(null_scales[:, index])
            for index, band in enumerate(SKY_NULL_BANDS)},
        "external_blank_provenance": blank_provenance,
        "blank_amplifier_counts_csv": str(blank_csv.resolve()),
        "blank_support": support,
        "blank_support_by_h5": support_by_h5,
        "blank_support_by_exposure": support_by_exposure,
        "blank_support_by_h5_exposure": support_by_h5_exposure,
        "generated_qa_files": [
            str(qa_figure.resolve()), str(blank_csv.resolve()),
            str(summary_path.resolve()), str(checks_csv.resolve()),
            str(Path(output).with_name("m101_measurements_summary.txt").resolve()),
        ],
        "timing": {
            "h5_build_seconds": float(sum(item["seconds"] for item in process_results)),
            "per_h5_seconds": {item["filename"]: float(item["seconds"])
                               for item in process_results},
            "qa_seconds": time.perf_counter() - qa_started,
        },
        "survey_offset": {"min": float(np.nanmin(offsets)), "median": float(np.nanmedian(offsets)), "max": float(np.nanmax(offsets))},
        "K": {band: {"min": float(np.nanmin(k_values[:, i])), "median": float(np.nanmedian(k_values[:, i])), "max": float(np.nanmax(k_values[:, i]))} for i, band in enumerate(BANDS)},
        "g_global": {band: float(np.nanmedian(g_values[:, i])) for i, band in enumerate(BANDS)},
        "checks": {item["name"]: {"status": item["status"], "value": item["value"], "detail": item["detail"]} for item in check_rows},
        "no_calibration_fitting": True,
        "no_s_response_applied": True, "no_residual_sky_subtraction": True,
        "no_final_gray": True, "no_null_band_centering": True,
        "external_blank_classification_filters_rows": False,
        "qa_seconds": time.perf_counter() - qa_started,
    }
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
        "sky-null bands=%s" % list(SKY_NULL_BANDS),
        "sky-null windows=%s" % null_provenance["requested_windows_A"],
        "sky-null finite data_native=%s" % null_finite_summary["sky_null_finite_data_native"],
        "sky-null finite data_work=%s" % null_finite_summary["sky_null_finite_data_work"],
        "sky-null finite error=%s" % null_finite_summary["sky_null_finite_error"],
        "sky-null finite QR-sky=%s" % null_finite_summary["sky_null_finite_sky"],
        "sky-null K min/median/max=%s" % summary["sky_null_K"],
        "blank support distribution=%s" % support["distribution"],
        "blank support threshold counts=%s" % support["threshold_counts"],
        "blank support threshold fractions=%s" % support["threshold_fractions"],
        "blank support by H5=%s" % support_by_h5,
        "blank support by exposure=%s" % support_by_exposure,
        "blank robust location=%s" % summary["sky_null_blank_robust_location"],
        "blank robust scale=%s" % summary["sky_null_blank_robust_scale"],
        "blank amplifier QA=%s" % str(blank_csv.resolve()),
        "date-mask fraction=%g" % date_fraction,
        "NO calibration fitting; NO posterior z; NO p; NO alpha; NO residual sky subtraction; NO null centering; blank classification stored without removing rows.",
    ]
    Path(output).with_name("m101_measurements_summary.txt").write_text("\n".join(text_lines) + "\n")
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
    parser.add_argument("--external-blank-fibers", required=True,
                        help="persisted external blank-fiber CSV or CSV.GZ")
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
    null_responses, null_provenance = _validate_sky_null_responses()
    blank_started = time.perf_counter()
    external_blank_by_h5, blank_provenance = _load_external_blank_fibers(
        args.external_blank_fibers, h5_paths)
    print("external blank CSV load: %.3f s; matched=%d blank=%d nonblank=%d ignored_extra_h5=%d" %
          (time.perf_counter() - blank_started, blank_provenance["matched_rows"],
           blank_provenance["blank_rows"], blank_provenance["nonblank_rows"],
           blank_provenance["ignored_extra_h5_files"]))
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
        sky_null_exposure = h5.root.sky_null_exposure
        h5_inputs = h5.root.provenance.h5_inputs
        metadata_table = h5.root.provenance.metadata
        checks = h5.root.build_checks
        _existing_state(h5, h5_paths, preflight, blank_provenance,
                        args.on_filter, args.off_filter, args.fq_template)
    else:
        h5, measurements, exposure_band, sky_null_exposure, h5_inputs, metadata_table, checks = _create_file(
            output, preflight, total_rows)
        provenance_values = {
            "schema_version": SCHEMA_VERSION,
            "scientific_model_version": SCIENTIFIC_MODEL_VERSION,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "script": str(Path(__file__).resolve()),
            "git_commit": _git_commit(),
            "band_order": list(BANDS),
            "source_band_order": list(BANDS),
            "sky_null_band_order": list(SKY_NULL_BANDS),
            "sky_null_definitions": null_provenance,
            "h5_count": len(h5_paths), "exposure_count": total_exposures,
            "native_fiber_rows": total_rows,
            "pass1_cache": _file_identity(args.pass1_cache),
            "pass1_cache_sha256": _small_file_hash(args.pass1_cache),
            "on_filter": _file_identity(args.on_filter), "off_filter": _file_identity(args.off_filter),
            "on_filter_sha256": _small_file_hash(args.on_filter),
            "off_filter_sha256": _small_file_hash(args.off_filter),
            "fq_template": _file_identity(args.fq_template),
            "fq_template_sha256": _small_file_hash(args.fq_template),
            "external_blank_fibers": blank_provenance["identity"],
            "external_blank_fibers_sha256": blank_provenance["sha256"],
            "external_blank_provenance": blank_provenance,
            "h5_input_identities": [item["identity"] for item in preflight],
            "external_images_from_cache": cache.get("images", {}),
            "g_global": {band: float(g_global[i]) for i, band in enumerate(BANDS)},
            "excluded_h5": EXCLUDED_H5,
            "external_source": "PASS-1 calibration external_object/external_valid only",
            "fitted_terms_applied": [],
            "blank_classification_ingested": True,
            "blank_classification_filters_rows": False,
            "no_null_band_centering": True,
            "definitions": {"D": "synthetic_mean(Fibers.spectrum / Survey.offset)",
                            "B": "synthetic_mean(Fibers.skyspectrum), no offset",
                            "external_raw": "cached PSF-matched exact aperture object value",
                            "external_prediction": "g_global_band * external_raw",
                            "K": "weighted_scalar(raw_work_basis(Survey), filter)",
                            "K_null": "weighted_scalar(raw_work_basis(Survey), sky-null top-hat)",
                            "sky_null_data": "synthetic_mean(Fibers.spectrum)",
                            "sky_null_data_work": "sky_null_data_native / Survey.offset",
                            "sky_null_error": "propagated collapse of Fibers.error",
                            "sky_null_error_work": "sky_null_error_native / abs(Survey.offset)",
                            "sky_null_sky": "synthetic_mean(Fibers.skyspectrum), primitive only",
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
                           measurements, exposure_band, sky_null_exposure, h5_inputs,
                           preflight[h5_id], null_responses, external_blank_by_h5)
        process_results.append(result)
        print("%s: rows=%d physical_groups=%d time=%.3f s" %
              (path.name, result["rows"], result["groups"], result["seconds"]))
    indexing_seconds = _create_indexes(h5)
    print("final indexing: %.3f s" % indexing_seconds)
    summary = _qa_and_outputs(h5, output, preflight, total_rows, total_exposures,
                              cache, filters, fq, args.pass1_cache,
                              (args.on_filter, args.off_filter), args.fq_template,
                              started, checks, process_results, blank_provenance,
                              null_provenance)
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
    print("NO calibration fit performed; NO posterior z; NO p; NO alpha; NO residual-sky subtraction; NO null centering; blank classification stored without filtering rows.")
    print("total runtime: %.3f s" % total_seconds)


if __name__ == "__main__":
    main()
