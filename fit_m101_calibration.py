#!/usr/bin/env python3
"""Fit the clean M101 additive plus hierarchical multiplicative calibration.

The implementation has two deliberately separate data paths.  Externally
classified blank native spectra estimate ``R + cU + dV``.  The compact source
measurements, after that additive prediction is subtracted, estimate
``z = log(m)``.  Blank classification is never used as the complement of a
source mask. ``sky_blank`` is used only to select native spectra for fitting
``R+cU+dV``. The z likelihood uses ordinary external-valid, compact, date,
finite, and error validity; rows with X approximately zero have zero site
information.
"""

from argparse import ArgumentParser
import csv
import gzip
import hashlib
import glob
import json
from pathlib import Path
import time
from types import MappingProxyType
import warnings

import numpy as np
import tables
from astropy.table import Table
from scipy import linalg, sparse
from scipy.special import logsumexp
from scipy.sparse.linalg import lsqr, splu

from math_utils import biweight


WAVE = np.linspace(3470.0, 5540.0, 1036)
N_WAVE = WAVE.size
AMPS = ("LL", "LU", "RL", "RU")
N_AMP_FIBERS = 112
N_EXPOSURES = 3
NULL_WINDOWS = ((3538.0, 3550.0), (3602.0, 3614.0), (3628.0, 3640.0),
                (3676.0, 3688.0), (3902.0, 3914.0))
NULL_NAMES = ("NULL_HIGH1", "NULL_LOW1", "NULL_HIGH2", "NULL_LOW2", "NULL_HIGH3")
Z_GRID = np.linspace(-1.2, 1.2, 161)
SCHEMA = "m101_calibration_clean_v1"

# Existing date-dependent hardware exclusions are retained as bookkeeping for
# the validated native grouping.  They are not inferred from residuals.
DATE_MASKS = {
    "20200430-20200501": ("057RL", "057RU", "057LL", "057LU", "058RU", "058RL", "021RL", "021RU", "021LL", "021LU"),
    "20200430-20200715": ("092LU", "094RU", "046RU", "104LL", "104LU", "028RU", "028RL", "027LL", "027LU", "067LL", "067LU", "025LU", "106RU", "106RL", "026LL", "026LU", "026RL", "026RU", "103LL", "103LU"),
    "20200525-20200526": ("057RL", "057LL", "089RU", "089RL"),
    "20200517-20200522": ("030RL", "030RU", "030LL", "030LU"),
    "20200523-20200526": ("039RL", "039RU", "039LL", "039LU"),
    "20200622-20200715": ("089RL", "089RU", "096RL", "096RU"),
}


def text(value):
    return value.decode("utf-8", errors="replace").strip() if isinstance(value, (bytes, np.bytes_)) else str(value).strip()


def identity(path):
    path = Path(path).expanduser().resolve()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    stat = path.stat()
    return {"path": str(path), "filename": path.name, "size": int(stat.st_size),
            "mtime_ns": int(stat.st_mtime_ns), "sha256": digest.hexdigest()}


def robust_location(values):
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if not x.size:
        return np.nan
    if x.size < 5:
        return float(np.median(x))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        answer = float(biweight(x))
    return answer if np.isfinite(answer) else float(np.median(x))


def robust_scale(values):
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if not x.size:
        return np.nan
    mad = np.median(np.abs(x - np.median(x)))
    return float(1.4826 * mad if mad > 0 else np.std(x))


def robust_spectrum(values):
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or not values.shape[0]:
        return np.full(N_WAVE, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        if values.shape[0] < 5:
            result = np.nanmedian(values, axis=0)
        else:
            result = np.asarray(biweight(values, axis=0), dtype=float)
            fallback = np.nanmedian(values, axis=0)
            result[~np.isfinite(result)] = fallback[~np.isfinite(result)]
    return result


def robust_rms(values):
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    return float(np.sqrt(np.median(x * x))) if x.size else np.nan


def weighted_scalar(values, response):
    values = np.asarray(values, dtype=float)
    response = np.asarray(response, dtype=float)
    use = np.isfinite(values) & np.isfinite(response) & (response != 0)
    den = np.sum(np.where(use, response, 0.0))
    return float(np.sum(np.where(use, values * response, 0.0)) / den) if den else np.nan


def read_filter(path):
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise ValueError("filter does not exist: %s" % path)
    table = Table.read(path, format="ascii")
    if not {"Wavelength", "R"}.issubset(table.colnames):
        raise ValueError("filter must contain Wavelength and R columns: %s" % path)
    response = np.interp(WAVE, np.asarray(table["Wavelength"], float),
                         np.asarray(table["R"], float), left=0.0, right=0.0)
    if not np.any(np.isfinite(response) & (response != 0)):
        raise ValueError("filter has no response on the native wavelength grid")
    return response


def discover_h5(args):
    values = list(args.h5_positional or [])
    for group in args.h5 or []:
        values.extend(group)
    for pattern in args.h5_glob or []:
        values.extend(glob.glob(str(Path(pattern).expanduser())))
    unique = {}
    for value in values:
        path = Path(value).expanduser().resolve()
        if not path.is_file():
            raise ValueError("H5 input does not exist: %s" % path)
        if path.name in unique and unique[path.name] != path:
            raise ValueError("ambiguous H5 basename: %s" % path.name)
        unique[path.name] = path
    if not unique:
        raise ValueError("no original H5 files supplied")
    return [unique[name] for name in sorted(unique)]


def exposure_groups(info):
    required = {"ifuslot", "amp", "specid", "ifuid"}
    if not required.issubset(info.colnames):
        raise ValueError("Info lacks physical identity columns")
    nrows = int(info.nrows)
    slots = np.asarray(info.cols.ifuslot[:])
    nslots = np.unique(slots).size
    if nrows % (448 * nslots) or nrows // (448 * nslots) != 3:
        raise ValueError("cannot infer exactly three validated exposures")
    rows = np.arange(nrows, dtype=np.int64)
    labels = ((rows // N_AMP_FIBERS) % N_EXPOSURES + 1).astype(np.int16)
    amps = np.asarray([text(value) for value in info.cols.amp[:]])
    specid = np.asarray(info.cols.specid[:], int)
    ifuid = np.asarray(info.cols.ifuid[:], int)
    groups = []
    for exposure, slot, amp in sorted(set(zip(labels.tolist(), slots.tolist(), amps.tolist()))):
        indices = np.flatnonzero((labels == exposure) & (slots == slot) & (amps == amp))
        if indices.size != N_AMP_FIBERS:
            raise ValueError("%s exposure %s has %d rows, expected 112" % (slot, amp, indices.size))
        if np.unique(specid[indices]).size != 1 or np.unique(ifuid[indices]).size != 1:
            raise ValueError("physical identity changes inside an amplifier observation")
        groups.append({"exposure": int(exposure), "indices": indices,
                       "SPECID": int(specid[indices[0]]), "IFUSLOT": int(slot),
                       "IFUID": int(ifuid[indices[0]]), "AMP": amp})
    return groups, labels


def date_mask(path, ifuslot, amp):
    try:
        date = int(Path(path).name[:8])
    except ValueError:
        return np.zeros(np.asarray(ifuslot).shape, dtype=bool)
    answer = np.zeros(np.asarray(ifuslot).shape, dtype=bool)
    for interval, names in DATE_MASKS.items():
        start, stop = (int(value) for value in interval.split("-"))
        if start <= date < stop:
            for name in names:
                answer |= (np.asarray(ifuslot) == int(name[:3])) & (np.asarray(amp) == name[3:])
    return answer


def load_blank(path, h5_paths):
    state = {}
    labels = {}
    for h5_path in h5_paths:
        with tables.open_file(h5_path) as h5:
            _, labels[h5_path.name] = exposure_groups(h5.root.Info)
            state[h5_path.name] = {"seen": np.zeros(h5.root.Info.nrows, bool),
                                   "blank": np.zeros(h5.root.Info.nrows, bool)}
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(Path(path).expanduser(), "rt", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"H5", "exposure", "row_index", "blank"}
        if not required.issubset(reader.fieldnames or ()):
            raise ValueError("external blank CSV lacks required columns")
        for number, row in enumerate(reader, 1):
            name = Path(text(row["H5"])).name
            if name not in state:
                continue
            index = int(row["row_index"])
            if index < 0 or index >= state[name]["seen"].size or state[name]["seen"][index]:
                raise ValueError("invalid or duplicate blank row %d" % number)
            exposure = int(row["exposure"])
            if exposure != int(labels[name][index]):
                raise ValueError("blank CSV exposure mismatch at %s row %d" % (name, index))
            value = text(row["blank"]).lower()
            if value not in ("true", "false", "1", "0"):
                raise ValueError("invalid blank flag at CSV row %d" % number)
            state[name]["seen"][index] = True
            state[name]["blank"][index] = value in ("true", "1")
    for name, values in state.items():
        if not np.all(values["seen"]):
            raise ValueError("blank CSV omits rows for %s" % name)
    return {name: values["blank"] for name, values in state.items()}, identity(path)


def load_measurement_h5(path):
    with tables.open_file(Path(path).expanduser()) as h5:
        if "/measurements" not in h5 or "/exposure_band" not in h5:
            raise ValueError("measurement product lacks required tables")
        schema = text(getattr(h5.root._v_attrs, "schema_version", ""))
        if schema != "m101_measurements_v2":
            raise ValueError("clean fitter requires m101_measurements_v2, got %s" % schema)
        if json.loads(text(getattr(h5.root._v_attrs, "band_order", "[]"))) != ["ON", "OFF"]:
            raise ValueError("measurement band order is not exactly [ON, OFF]")
        rows = h5.root.measurements.read()
        input_rows = h5.root.provenance.h5_inputs.read()
        measurement_ids = sorted(set(int(value) for value in rows["h5_id"]))
        input_names = [text(row["filename"]) for row in input_rows]
        if len(input_names) != len(measurement_ids):
            raise ValueError("measurement H5 input provenance count does not match h5_id values")
        # Some frozen development products repeat h5_id=0 in provenance while
        # the measurement rows correctly use consecutive IDs.  The persisted
        # input-table order is the authoritative correspondence in that case.
        inputs = dict(zip(measurement_ids, input_names))
        if not rows.size:
            raise ValueError("measurement product is empty")
        metadata = {text(row["key"]): json.loads(text(row["value"])) for row in h5.root.provenance.metadata}
    return rows, inputs, metadata


def collect_blank_groups(h5_paths, blank_masks, minimum_finite_fraction,
                         min_blank_ifus, min_blank_per_amp):
    exposure_data = {}
    ifu_obs, amp_obs = {}, {}
    all_ifus, all_amps = set(), set()
    for h5_path in h5_paths:
        with tables.open_file(h5_path) as h5:
            groups, labels = exposure_groups(h5.root.Info)
            for group in groups:
                if not group["AMP"] in AMPS:
                    raise ValueError("unknown physical amplifier label %s" % group["AMP"])
                all_ifus.add((group["SPECID"], group["IFUSLOT"], group["IFUID"]))
                all_amps.add((group["SPECID"], group["IFUSLOT"], group["IFUID"], group["AMP"]))
            slots = np.asarray(h5.root.Info.cols.ifuslot[:])
            amps = np.asarray([text(v) for v in h5.root.Info.cols.amp[:]])
            blocked = date_mask(h5_path, slots, amps)
            surveys = {int(row["exp"]): row for row in h5.root.Survey}
            if set(surveys) != {1, 2, 3}:
                raise ValueError("Survey exposures are not exactly 1,2,3")
            row_to_group = np.full(h5.root.Info.nrows, -1, int)
            for index, group in enumerate(groups):
                row_to_group[group["indices"]] = index
            for exposure in range(1, 4):
                candidates = np.flatnonzero((labels == exposure) & blank_masks[h5_path.name] & ~blocked)
                spectra = np.asarray(h5.root.Fibers.read_coordinates(candidates, field="spectrum"), float)
                offset = float(surveys[exposure]["offset"])
                if not np.isfinite(offset) or offset == 0:
                    raise ValueError("invalid Survey.offset in %s" % h5_path)
                if spectra.size:
                    good = np.mean(np.isfinite(spectra), axis=1) >= minimum_finite_fraction
                    candidates, spectra = candidates[good], spectra[good] / offset
                by_ifu, by_amp = {}, {}
                for group_index, group in enumerate(groups):
                    if group["exposure"] != exposure:
                        continue
                    local = np.flatnonzero(row_to_group[candidates] == group_index)
                    if not local.size:
                        continue
                    ifu = (group["SPECID"], group["IFUSLOT"], group["IFUID"])
                    amp = ifu + (group["AMP"],)
                    by_amp[amp] = (spectra[local], int(local.size))
                    by_ifu.setdefault(ifu, []).append(spectra[local])
                n_amp = 0
                for amp, (values, count) in by_amp.items():
                    if count < min_blank_per_amp:
                        continue
                    amp_obs[((h5_path.name, exposure), amp)] = {"y": robust_spectrum(values), "n": count}
                    n_amp += 1
                usable_ifus = 0
                for ifu, chunks in by_ifu.items():
                    values = np.concatenate(chunks, axis=0)
                    if values.shape[0] < min_blank_ifus * min_blank_per_amp:
                        continue
                    ifu_obs[((h5_path.name, exposure), ifu)] = {"y": robust_spectrum(values), "n": int(values.shape[0])}
                    usable_ifus += 1
                exposure_data[(h5_path.name, exposure)] = {"h5_id": h5_paths.index(h5_path),
                    "n_blank": int(candidates.size), "n_ifu": usable_ifus, "n_amp": n_amp}
    return exposure_data, ifu_obs, amp_obs, sorted(all_ifus), sorted(all_amps)


def robust_slope(y, x, counts=None):
    y, x = np.asarray(y, float), np.asarray(x, float)
    use = np.isfinite(y) & np.isfinite(x) & (np.abs(x) > 1e-14)
    if np.sum(use) < 2:
        return 0.0
    y, x = y[use], x[use]
    weights = np.ones(y.size) if counts is None else np.clip(np.asarray(counts, float)[use], 1, 112)
    slope = np.sum(weights * x * y) / max(np.sum(weights * x * x), 1e-30)
    for _ in range(5):
        residual = y - slope * x
        scale = robust_scale(residual)
        if not np.isfinite(scale) or scale <= 0:
            break
        weights_now = weights * np.minimum(1.0, 1.5 * scale / np.maximum(np.abs(residual), 1e-30))
        slope = np.sum(weights_now * x * y) / max(np.sum(weights_now * x * x), 1e-30)
    return float(slope) if np.isfinite(slope) else 0.0


def build_record_indices(records):
    by_hardware, by_exposure = {}, {}
    for index, record in enumerate(records):
        by_hardware.setdefault(record["hardware"], []).append(index)
        by_exposure.setdefault(record["exposure"], []).append(index)
    def freeze(mapping):
        values = {}
        for key, indices in mapping.items():
            array = np.asarray(indices, dtype=np.int64)
            array.setflags(write=False)
            values[key] = array
        return MappingProxyType(values)
    exposures = np.empty(len(records), dtype=object)
    for index, record in enumerate(records):
        exposures[index] = record["exposure"]
    counts = np.asarray([record["n"] for record in records], dtype=float)
    exposures.setflags(write=False); counts.setflags(write=False)
    return freeze(by_hardware), freeze(by_exposure), exposures, counts


def fit_template(records, coefficients, residuals, indices=None):
    if indices is None:
        if residuals is None or len(residuals) == 0:
            return np.full(N_WAVE, np.nan)
        indices = np.arange(len(records), dtype=np.int64)
        y = np.asarray(residuals, dtype=float)
    else:
        indices = np.asarray(indices, dtype=np.int64)
        y = np.asarray(residuals, dtype=float)[indices]
    if not indices.size:
        return np.full(N_WAVE, np.nan)
    x = np.asarray([coefficients[records[index]["hardware"]] for index in indices], dtype=float)
    counts = np.clip(np.asarray([records[index]["n"] for index in indices], dtype=float), 1.0, 112.0)
    use = np.isfinite(y) & np.isfinite(x)[:, None]
    x2 = x[:, None]
    weights = counts[:, None] * use
    denominator = np.sum(weights * x2 * x2, axis=0)
    answer = np.divide(np.sum(weights * x2 * y, axis=0), denominator,
                       out=np.zeros(N_WAVE), where=denominator > 1e-30)
    for _ in range(4):
        residual = np.where(use, y - x2 * answer[None, :], np.nan)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            center = np.nanmedian(residual, axis=0)
            scale = 1.4826 * np.nanmedian(np.abs(residual - center[None, :]), axis=0)
            fallback_scale = np.nanstd(residual, axis=0)
        scale = np.where(np.isfinite(scale) & (scale > 1e-12), scale, fallback_scale)
        scale = np.where(np.isfinite(scale) & (scale > 1e-12), scale, 1.0)
        robust_weight = np.minimum(1.0, 1.5 * scale[None, :] / np.maximum(np.abs(residual), 1e-30))
        weights_now = weights * robust_weight
        denominator = np.sum(weights_now * x2 * x2, axis=0)
        answer = np.divide(np.nansum(weights_now * x2 * y, axis=0), denominator,
                           out=answer, where=denominator > 1e-30)
    answer[denominator <= 1e-30] = np.nan
    return answer


def fit_coefficients(records, templates, residuals, keys, indices_by_hardware=None,
                     record_exposures=None, record_counts=None):
    answer = {key: 0.0 for key in keys}
    if indices_by_hardware is None:
        indices_by_hardware, _, record_exposures, record_counts = build_record_indices(records)
    if record_exposures is None:
        _, _, record_exposures, _ = build_record_indices(records)
    if record_counts is None:
        record_counts = np.asarray([record["n"] for record in records], dtype=float)
    residuals = np.asarray(residuals, dtype=float)
    for key in keys:
        indices = indices_by_hardware.get(key, np.empty(0, dtype=np.int64))
        if not indices.size:
            continue
        y = residuals[indices]
        x = np.asarray([templates[record_exposures[index]] for index in indices], dtype=float)
        use = np.isfinite(y) & np.isfinite(x)
        counts = np.broadcast_to(np.clip(record_counts[indices], 1.0, 112.0)[:, None], y.shape)
        answer[key] = robust_slope(y[use], x[use], counts[use])
    return answer


def normalize_ifu(c, U, R, supported):
    values = np.asarray([c[key] for key in supported], float)
    location = robust_location(values) if values.size else 0.0
    scale = robust_scale(values) if values.size else 1.0
    if not np.isfinite(scale) or scale <= 1e-12:
        scale = 1.0
    old = {key: np.asarray(value, float).copy() for key, value in U.items()}
    for exposure in U:
        R[exposure] = R[exposure] + location * old[exposure]
        U[exposure] = old[exposure] * scale
    supported_set = set(supported)
    return {key: ((c[key] - location) / scale if key in supported_set else 0.0) for key in c}, U, R, {"location": float(location), "scale": float(scale)}


def normalize_amp(d, V, ifu_keys, amp_keys, supported):
    for ifu in ifu_keys:
        keys = [ifu + (amp,) for amp in AMPS if ifu + (amp,) in supported]
        if keys:
            center = float(np.mean([d[key] for key in keys]))
            for key in keys:
                d[key] -= center
    values = np.asarray([d[key] for key in supported], float)
    scale = robust_scale(values) if values.size else 1.0
    if not np.isfinite(scale) or scale <= 1e-12:
        scale = 1.0
    for exposure in V:
        V[exposure] *= scale
    return {key: d[key] / scale if key in supported else 0.0 for key in amp_keys}, V, {"scale": float(scale)}


def pearson_correlation(first, second):
    first, second = np.asarray(first, dtype=float), np.asarray(second, dtype=float)
    use = np.isfinite(first) & np.isfinite(second)
    if np.sum(use) < 2:
        return np.nan
    first, second = first[use], second[use]
    first -= np.mean(first); second -= np.mean(second)
    denominator = np.sqrt(np.sum(first * first) * np.sum(second * second))
    return float(np.sum(first * second) / denominator) if denominator > 0 else np.nan


def solve_additive(exposure_data, ifu_obs, amp_obs, ifu_keys, amp_keys, max_iterations, tolerance):
    solve_started = time.perf_counter()
    exposures = sorted(exposure_data)
    ifu_records = [{"exposure": e, "hardware": key, "n": rec["n"]} for (e, key), rec in ifu_obs.items()]
    amp_records = [{"exposure": e, "hardware": key, "n": rec["n"]} for (e, key), rec in amp_obs.items()]
    ifu_indices_by_hardware, ifu_indices_by_exposure, ifu_record_exposures, ifu_record_counts = build_record_indices(ifu_records)
    amp_indices_by_hardware, amp_indices_by_exposure, amp_record_exposures, amp_record_counts = build_record_indices(amp_records)
    ifu_y = np.asarray([record["y"] for record in ifu_obs.values()], dtype=float)
    amp_y = np.asarray([record["y"] for record in amp_obs.values()], dtype=float)
    amp_ifus = [record["hardware"][:3] for record in amp_records]
    supported_c = sorted({key for _, key in ifu_obs})
    supported_d = sorted({key for _, key in amp_obs})
    initialization_started = time.perf_counter()
    R = {e: robust_spectrum(ifu_y[ifu_indices_by_exposure.get(e, np.empty(0, dtype=np.int64))]) for e in exposures}
    c = {key: 0.0 for key in ifu_keys}; d = {key: 0.0 for key in amp_keys}
    U = {}
    for e in exposures:
        indices = ifu_indices_by_exposure.get(e, np.empty(0, dtype=np.int64))
        candidates = ifu_y[indices] - R[e]
        U[e] = max(candidates, key=robust_rms, default=np.zeros(N_WAVE)).copy()
    residuals = np.asarray([y - R[e] for y, e in zip(ifu_y, ifu_record_exposures)])
    c = fit_coefficients(ifu_records, U, residuals, ifu_keys, ifu_indices_by_hardware,
                         ifu_record_exposures, ifu_record_counts)
    c, U, R, c_norm = normalize_ifu(c, U, R, supported_c)
    V = {}
    for e in exposures:
        indices = amp_indices_by_exposure.get(e, np.empty(0, dtype=np.int64))
        amp_start = amp_y[indices] - R[e] - np.asarray([c[amp_ifus[index]] for index in indices])[:, None] * U[e]
        V[e] = max(amp_start, key=robust_rms, default=np.zeros(N_WAVE)).copy()
    initialization_seconds = time.perf_counter() - initialization_started
    history = []
    previous_prediction = None
    previous_stage3_robust_rms = None
    outer_timings = []
    for iteration in range(1, max_iterations + 1):
        iteration_started = time.perf_counter()
        ifu_residuals = np.asarray([y - R[e] for y, e in zip(ifu_y, ifu_record_exposures)])
        stage_started = time.perf_counter()
        c = fit_coefficients(ifu_records, U, ifu_residuals, ifu_keys, ifu_indices_by_hardware,
                             ifu_record_exposures, ifu_record_counts)
        c_fit_seconds = time.perf_counter() - stage_started
        stage_started = time.perf_counter()
        for e in exposures:
            indices = ifu_indices_by_exposure.get(e, np.empty(0, dtype=np.int64))
            if indices.size:
                U[e] = fit_template(ifu_records, c, ifu_residuals, indices=indices)
        c, U, R, c_norm = normalize_ifu(c, U, R, supported_c)
        U_fit_seconds = time.perf_counter() - stage_started
        amp_residuals = np.asarray([y - R[e] - c[ifu] * U[e]
                                    for y, e, ifu in zip(amp_y, amp_record_exposures, amp_ifus)])
        stage_started = time.perf_counter()
        d = fit_coefficients(amp_records, V, amp_residuals, amp_keys, amp_indices_by_hardware,
                             amp_record_exposures, amp_record_counts)
        d_fit_seconds = time.perf_counter() - stage_started
        stage_started = time.perf_counter()
        for e in exposures:
            indices = amp_indices_by_exposure.get(e, np.empty(0, dtype=np.int64))
            if indices.size:
                V[e] = fit_template(amp_records, d, amp_residuals, indices=indices)
        d, V, d_norm = normalize_amp(d, V, ifu_keys, amp_keys, supported_d)
        V_fit_seconds = time.perf_counter() - stage_started
        stage_started = time.perf_counter()
        for e in exposures:
            indices = amp_indices_by_exposure.get(e, np.empty(0, dtype=np.int64))
            if indices.size:
                values = amp_y[indices] - np.asarray([c[amp_ifus[index]] for index in indices])[:, None] * U[e]
                values -= np.asarray([d[amp_records[index]["hardware"]] for index in indices])[:, None] * V[e]
                R[e] = robust_spectrum(values)
        R_update_seconds = time.perf_counter() - stage_started
        stage_started = time.perf_counter()
        prediction = np.asarray([R[e] + c[ifu] * U[e] + d[key] * V[e]
                                 for (e, key), ifu in zip(amp_obs, amp_ifus)])
        if previous_prediction is None:
            delta = np.inf
            change_rms = np.inf
            maximum_abs_prediction_change = np.inf
        else:
            difference = prediction - previous_prediction
            finite = np.isfinite(difference)
            change_rms = float(np.sqrt(np.mean(difference[finite] ** 2))) if np.any(finite) else np.inf
            scale = float(np.sqrt(np.mean(prediction[np.isfinite(prediction)] ** 2))) if np.any(np.isfinite(prediction)) else 1.0
            delta = change_rms / max(scale, 1e-12)
            maximum_abs_prediction_change = float(np.nanmax(np.abs(difference))) if np.any(finite) else np.inf
        previous_prediction = prediction.copy()
        stage3 = amp_y - prediction
        prediction_finite = prediction[np.isfinite(prediction)]
        additive_prediction_rms = float(np.sqrt(np.mean(prediction_finite ** 2))) if prediction_finite.size else np.nan
        stage3_robust = robust_rms(stage3)
        stage3_finite = stage3[np.isfinite(stage3)]
        stage3_arithmetic = float(np.sqrt(np.mean(stage3_finite ** 2))) if stage3_finite.size else np.nan
        if previous_stage3_robust_rms is None:
            fractional_stage3_change = np.inf
        elif np.isfinite(previous_stage3_robust_rms) and np.isfinite(stage3_robust):
            fractional_stage3_change = (float(abs(stage3_robust - previous_stage3_robust_rms)) /
                                         abs(previous_stage3_robust_rms)
                                         if previous_stage3_robust_rms != 0 else
                                         (0.0 if stage3_robust == 0 else np.inf))
        else:
            fractional_stage3_change = np.inf
        previous_stage3_robust_rms = stage3_robust
        corr_records = [{"H5": e[0], "exposure": e[1], "correlation": pearson_correlation(U[e], V[e])}
                        for e in exposures]
        corr_values = np.asarray([record["correlation"] for record in corr_records], dtype=float)
        finite_corr = corr_values[np.isfinite(corr_values)]
        median_corr = float(np.median(finite_corr)) if finite_corr.size else np.nan
        max_abs_corr = float(np.max(np.abs(finite_corr))) if finite_corr.size else np.nan
        prediction_over_residual = (change_rms / stage3_robust
                                    if np.isfinite(change_rms) and np.isfinite(stage3_robust) and stage3_robust > 0
                                    else np.inf)
        history.append({"iteration": iteration, "invariant_prediction_change": delta,
                        "invariant_prediction_change_rms": change_rms,
                        "additive_prediction_rms": additive_prediction_rms,
                        "raw_prediction_scale": float(np.nanmax(np.abs(prediction))) if np.any(np.isfinite(prediction)) else np.nan,
                        "stage3_robust_rms": stage3_robust, "stage3_arithmetic_rms": stage3_arithmetic,
                        "maximum_abs_prediction_change": maximum_abs_prediction_change,
                        "fractional_change_in_stage3_robust_rms": fractional_stage3_change,
                        "prediction_change_over_residual": prediction_over_residual,
                        "c_location": c_norm["location"], "c_scale": c_norm["scale"],
                        "d_scale": d_norm["scale"], "corr_U_V": corr_records,
                        "median_corr_U_V": median_corr, "max_abs_corr_U_V": max_abs_corr})
        qa_seconds = time.perf_counter() - stage_started
        iteration_seconds = time.perf_counter() - iteration_started
        outer_timings.append({"iteration": iteration, "total_seconds": iteration_seconds,
                              "c_fit_seconds": c_fit_seconds, "U_fit_seconds": U_fit_seconds,
                              "d_fit_seconds": d_fit_seconds, "V_fit_seconds": V_fit_seconds,
                              "R_update_seconds": R_update_seconds,
                              "convergence_QA_seconds": qa_seconds})
        current = history[-1]
        print("additive iteration %d: change=%.6g change_rms=%.6g residual=%.6g ratio=%.6g corr=%.4g/%.4g time=%.3fs" %
              (iteration, current["invariant_prediction_change"], current["invariant_prediction_change_rms"],
               current["stage3_robust_rms"], current["prediction_change_over_residual"],
               current["median_corr_U_V"], current["max_abs_corr_U_V"], iteration_seconds))
        if np.isfinite(delta) and delta < tolerance:
            converged = True
            break
    else:
        converged = False
    timings = {"additive_initialization_seconds": initialization_seconds,
               "additive_solve_total_seconds": time.perf_counter() - solve_started,
               "mean_outer_iteration_seconds": (float(np.mean([item["total_seconds"] for item in outer_timings]))
                                                 if outer_timings else 0.0),
               "outer_iterations": outer_timings}
    return R, U, V, c, d, history, converged, timings


HISTORY_CSV_FIELDS = (
    "iteration", "invariant_prediction_change", "invariant_prediction_change_rms",
    "additive_prediction_rms", "stage3_robust_rms", "stage3_arithmetic_rms",
    "maximum_abs_prediction_change", "fractional_change_in_stage3_robust_rms",
    "prediction_change_over_residual", "c_location", "c_scale", "d_scale",
    "median_corr_U_V", "max_abs_corr_U_V", "corr_U_V", "outer_iteration_seconds",
    "c_fit_seconds", "U_fit_seconds", "d_fit_seconds", "V_fit_seconds",
    "R_update_seconds", "convergence_QA_seconds")


def write_additive_history_csv(path, history, timings):
    path = Path(path).expanduser().resolve()
    timing_rows = timings["outer_iterations"]
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=HISTORY_CSV_FIELDS)
        writer.writeheader()
        for entry, timing in zip(history, timing_rows):
            row = {field: entry.get(field, "") for field in HISTORY_CSV_FIELDS}
            row["corr_U_V"] = json.dumps(entry["corr_U_V"], separators=(",", ":"), default=str)
            row.update({"outer_iteration_seconds": timing["total_seconds"],
                        "c_fit_seconds": timing["c_fit_seconds"],
                        "U_fit_seconds": timing["U_fit_seconds"],
                        "d_fit_seconds": timing["d_fit_seconds"],
                        "V_fit_seconds": timing["V_fit_seconds"],
                        "R_update_seconds": timing["R_update_seconds"],
                        "convergence_QA_seconds": timing["convergence_QA_seconds"]})
            writer.writerow(row)
    return path


def null_responses():
    return np.asarray([((WAVE >= low) & (WAVE <= high)).astype(float) for low, high in NULL_WINDOWS])


def collapse_components(R, U, V, filters):
    result = {}
    for exposure, terms in R.items():
        result[(exposure, "R")] = np.asarray([weighted_scalar(terms, filters[name]) for name in ("ON", "OFF")])
        result[(exposure, "U")] = np.asarray([weighted_scalar(U[exposure], filters[name]) for name in ("ON", "OFF")])
        result[(exposure, "V")] = np.asarray([weighted_scalar(V[exposure], filters[name]) for name in ("ON", "OFF")])
    return result


def apply_compact_mask(rows, mask_path):
    if not mask_path:
        return np.ones(len(rows), bool)
    from build_m101_external_compact_mask import load_compact_mask, mask_radec
    data, wcs = load_compact_mask(mask_path)
    ra = np.asarray(rows["effective_RA"], float); dec = np.asarray(rows["effective_Dec"], float)
    masked_on, inside_on = mask_radec(data, wcs, ra[:, 0], dec[:, 0])
    masked_off, inside_off = mask_radec(data, wcs, ra[:, 1], dec[:, 1])
    external = np.asarray(rows["external_valid"], bool)
    outside = (external[:, 0] & ~inside_on) | (external[:, 1] & ~inside_off)
    if np.any(outside):
        raise ValueError("compact-mask footprint mismatch: external-valid coordinate outside mask footprint for %d rows" % int(np.sum(outside)))
    return ~(masked_on | masked_off)


def local_z(row_block, additive_band, compact_ok, z0_sigma=.5, data_error_scales=(np.sqrt(2.58), np.sqrt(4.26)), model_fractions=(.0590, .0307)):
    rows = row_block
    additive_band = np.asarray(additive_band, dtype=float)
    if additive_band.shape != (2,):
        raise ValueError("additive_band must have shape (2,), got %s" % (additive_band.shape,))
    if len(rows) != len(compact_ok):
        raise ValueError("compact mask and measurement block length differ")
    valid = (np.asarray(compact_ok, bool) & ~np.asarray(rows["date_mask_bad"], bool))[:, None]
    data = np.asarray(rows["data_work"], float); error = np.asarray(rows["error_work"], float); X = np.asarray(rows["external_prediction"], float)
    valid = valid & np.asarray(rows["external_valid"], bool) & np.isfinite(data) & np.isfinite(error) & (error > 0) & np.isfinite(X)
    Dcorr = data - additive_band[None, :]
    assert Dcorr.shape == data.shape
    information = np.zeros(2); log_like = np.zeros(Z_GRID.size)
    for band in range(2):
        use = valid[:, band]
        # Near-zero source rows are retained in QA but have exactly zero z weight.
        use &= np.abs(X[:, band]) > 1e-12
        if not np.any(use):
            continue
        x_band = X[use, band]; error_band = error[use, band]
        sigma = np.sqrt((data_error_scales[band] * error_band) ** 2 + (model_fractions[band] * x_band) ** 2)
        information[band] = float(np.sum((x_band / sigma) ** 2))
        prediction = np.exp(Z_GRID[:, None]) * x_band[None, :]
        log_like += np.sum(-0.5 * ((Dcorr[use, band][None, :] - prediction) / sigma[None, :]) ** 2 - np.log(sigma[None, :]), axis=1)
    log_post = log_like - 0.5 * (Z_GRID / z0_sigma) ** 2
    log_post -= logsumexp(log_post)
    probability = np.exp(log_post)
    mean = float(np.sum(probability * Z_GRID)); variance = float(np.sum(probability * (Z_GRID - mean) ** 2))
    sigma = max(np.sqrt(max(variance, 0)), 1e-6)
    site_tau = 1.0 / (sigma * sigma) - 1.0 / (z0_sigma * z0_sigma)
    site_nu = mean / (sigma * sigma)
    # The finite grid slightly truncates the standalone prior.  With no
    # nonzero-X likelihood rows, remove that numerical truncation artifact.
    if not np.any(information > 0):
        site_tau = 0.0; site_nu = 0.0
    if not np.isfinite(site_tau) or site_tau <= 0 or not np.isfinite(site_nu):
        site_tau = 0.0; site_nu = 0.0; site_z_hat = np.nan; site_sigma = np.inf; noninformative = True
    else:
        site_z_hat = site_nu / site_tau; site_sigma = np.sqrt(1.0 / site_tau); noninformative = False
    return {"z": mean, "sigma": sigma, "site_z_hat": site_z_hat, "site_sigma": site_sigma,
            "site_tau": site_tau, "site_nu": site_nu, "noninformative_site": noninformative, "information": information,
            "n_valid": np.sum(valid, axis=0).astype(int), "n_source": np.sum(valid & (np.abs(X) > 1e-12), axis=0).astype(int),
            "edge_mass": float(probability[0] + probability[-1]), "Dcorr": Dcorr, "valid": valid}


def build_hierarchy(obs):
    exposures = sorted({o["exposure_key"] for o in obs})
    ifus_by_exp = {e: sorted({o["ifu"] for o in obs if o["exposure_key"] == e}) for e in exposures}
    amps_by_ifu = {ifu: sorted({o["amp"] for o in obs if o["ifu"] == ifu})
                   for ifu in sorted({o["ifu"] for o in obs})}
    g_index = {e: i for i, e in enumerate(exposures)}
    e_index, a_index, cursor = {}, {}, len(exposures)
    for e in exposures:
        keys = ifus_by_exp[e]
        H = linalg.helmert(len(keys), full=False).T if len(keys) > 1 else np.zeros((1, 0))
        assert H.shape == (len(keys), max(len(keys) - 1, 0))
        e_index[e] = (cursor, keys, H); cursor += H.shape[1]
    for ifu in sorted(amps_by_ifu):
        keys = amps_by_ifu[ifu]
        H = linalg.helmert(len(keys), full=False).T if len(keys) > 1 else np.zeros((1, 0))
        assert H.shape == (len(keys), max(len(keys) - 1, 0))
        a_index[ifu] = (cursor, keys, H); cursor += H.shape[1]
    design = sparse.lil_matrix((len(obs), cursor), dtype=float)
    site_tau = np.asarray([o.get("site_tau", 0.0) for o in obs], dtype=float)
    site_nu = np.asarray([o.get("site_nu", 0.0) for o in obs], dtype=float)
    for number, o in enumerate(obs):
        design[number, g_index[o["exposure_key"]]] = 1.0
        start, keys, H = e_index[o["exposure_key"]]
        if H.shape[1]: design[number, start:start + H.shape[1]] = H[keys.index(o["ifu"])]
        start, keys, H = a_index[o["ifu"]]
        if H.shape[1]: design[number, start:start + H.shape[1]] = H[keys.index(o["amp"])]
    prior_precision = np.r_[np.full(len(exposures), 1.0 / .25**2),
                            np.full(cursor - len(exposures), 1.0 / .1**2)]
    design = design.tocsr()
    Q = sparse.diags(prior_precision, format="csc") + design.T @ sparse.diags(site_tau, format="csc") @ design
    h = np.asarray(design.T @ site_nu).ravel()
    factor = splu(Q.tocsc())
    theta = factor.solve(h)

    def linear_variance(forms):
        forms = np.asarray(forms, dtype=float)
        answer = np.empty(forms.shape[0], dtype=float)
        for start in range(0, forms.shape[0], 256):
            block = forms[start:start + 256]
            solved = factor.solve(block.T)
            answer[start:start + len(block)] = np.sum(block * solved.T, axis=1)
        return np.maximum(answer, 0.0)

    gamma_forms = np.zeros((len(exposures), cursor));
    for e in exposures: gamma_forms[g_index[e], g_index[e]] = 1.0
    gamma_variances = linear_variance(gamma_forms)
    gamma = {e: float(theta[g_index[e]]) for e in exposures}
    gamma_sigma = {e: float(np.sqrt(gamma_variances[g_index[e]])) for e in exposures}
    iota, iota_sigma = {}, {}
    for e in exposures:
        start, keys, H = e_index[e]
        values = H @ theta[start:start + H.shape[1]] if H.shape[1] else np.zeros(len(keys))
        forms = np.zeros((len(keys), cursor));
        if H.shape[1]: forms[:, start:start + H.shape[1]] = H
        variances = linear_variance(forms)
        for key, value, variance in zip(keys, values, variances):
            iota[(e, key)] = float(value); iota_sigma[(e, key)] = float(np.sqrt(max(variance, 0.0)))
    eta, eta_sigma = {}, {}
    for ifu, (start, keys, H) in a_index.items():
        values = H @ theta[start:start + H.shape[1]] if H.shape[1] else np.zeros(len(keys))
        forms = np.zeros((len(keys), cursor));
        if H.shape[1]: forms[:, start:start + H.shape[1]] = H
        variances = linear_variance(forms)
        for key, value, variance in zip(keys, values, variances):
            eta[(ifu, key)] = float(value); eta_sigma[(ifu, key)] = float(np.sqrt(max(variance, 0.0)))
    for number, o in enumerate(obs):
        form = design.getrow(number).toarray().ravel()
        o["posterior_z"] = float(form @ theta)
        o["posterior_sigma"] = float(np.sqrt(linear_variance(form[None, :])[0]))
    return gamma, iota, eta, gamma_sigma, iota_sigma, eta_sigma, {"Q": Q, "h": h, "design": design}


def synthetic_test():
    rng = np.random.default_rng(17)
    ekeys = [("a.h5", 1), ("b.h5", 1), ("c.h5", 1)]
    ifus = [(1, 2, 3), (1, 2, 4), (1, 2, 5), (1, 2, 6)]
    amps = [ifu + (amp,) for ifu in ifus for amp in AMPS]
    unsupported_ifu = (1, 2, 99); unsupported_amp = unsupported_ifu + ("RU",)
    R_true = {e: rng.normal(0, .2, N_WAVE) for e in ekeys}
    U_true = {e: rng.normal(0, .08, N_WAVE) for e in ekeys}
    V_true = {e: rng.normal(0, .04, N_WAVE) for e in ekeys}
    c_true = dict(zip(ifus, (-1.5, -.5, .5, 1.5)))
    d_pattern = dict(zip(AMPS, (-1.5, -.5, .5, 1.5)))
    d_true = {ifu + (amp,): d_pattern[amp] for ifu in ifus for amp in AMPS}
    ifu_obs, amp_obs, exp_data = {}, {}, {}
    for e in ekeys:
        exp_data[e] = {"n_blank": 4 * 4, "n_ifu": 4, "n_amp": 16}
        for ifu in ifus:
            amp_values = []
            for amp in AMPS:
                key = ifu + (amp,); y = R_true[e] + c_true[ifu] * U_true[e] + d_true[key] * V_true[e]
                amp_values.append(y)
                amp_obs[(e, key)] = {"y": y.copy(), "n": 20}
            ifu_obs[(e, ifu)] = {"y": robust_spectrum(np.asarray(amp_values)), "n": 80}
    all_ifus = ifus + [unsupported_ifu]; all_amps = amps + [unsupported_amp]
    R, U, V, c, d, history, converged, _ = solve_additive(exp_data, ifu_obs, amp_obs, all_ifus, all_amps, 20, 1e-7)
    assert converged and history[-1]["invariant_prediction_change"] < 1e-7
    predicted = np.asarray([R[e] + c[key[:3]] * U[e] + d[key] * V[e] for e, key in amp_obs])
    truth = np.asarray([amp_obs[(e, key)]["y"] for e, key in amp_obs])
    assert np.nanmax(np.abs(predicted - truth)) < 2e-8
    assert c[unsupported_ifu] == 0.0 and d[unsupported_amp] == 0.0
    assert np.allclose(np.mean([d[ifu + (amp,)] for amp in AMPS]), 0.0)
    assert all(ifu + ("RU",) in d for ifu in ifus)
    reversed_result = solve_additive(dict(reversed(exp_data.items())), dict(reversed(ifu_obs.items())),
                                     dict(reversed(amp_obs.items())), list(reversed(all_ifus)),
                                     list(reversed(all_amps)), 20, 1e-7)
    R_reversed, U_reversed, V_reversed, c_reversed, d_reversed = reversed_result[:5]
    for e, key in amp_obs:
        assert np.allclose(R[e] + c[key[:3]] * U[e] + d[key] * V[e],
                           R_reversed[e] + c_reversed[key[:3]] * U_reversed[e] + d_reversed[key] * V_reversed[e], atol=2e-8)
    filters = {"ON": np.ones(N_WAVE), "OFF": np.ones(N_WAVE)}
    bands = collapse_components(R, U, V, filters)
    assert all(np.allclose(bands[(e, "R")], weighted_scalar(R[e], filters["ON"])) for e in ekeys)
    theta_iota = {e: np.asarray((-.03, -.01, .01, .03)) for e in ekeys}
    theta_eta = np.asarray((-.02, -.007, .007, .02))
    rows_dtype = np.dtype([("date_mask_bad", "?") , ("data_work", "f8", (2,)), ("error_work", "f8", (2,)), ("external_prediction", "f8", (2,)), ("external_valid", "?", (2,)), ("sky_blank", "?")])
    hierarchy_obs = []
    for e_index, e in enumerate(ekeys):
        for ifu_index, ifu in enumerate(ifus):
            for amp_index, amp in enumerate(AMPS):
                iota = theta_iota[e][ifu_index]
                eta = theta_eta[amp_index]
                z_true = .04 * (e_index - 1) + iota + eta
                X = np.tile(np.asarray((1.0, .7)), (112, 1)); rows = np.zeros(112, dtype=rows_dtype)
                rows["data_work"] = np.exp(z_true) * X; rows["error_work"] = .001; rows["external_prediction"] = X; rows["external_valid"] = True
                local = local_z(rows, np.zeros(2), np.ones(112, bool))
                assert abs(local["z"] - z_true) < .02
                hierarchy_obs.append({"exposure_key": e, "ifu": ifu, "amp": ifu + (amp,), "z": local["z"], "sigma": local["sigma"], "site_z_hat": local["site_z_hat"], "site_sigma": local["site_sigma"], "site_tau": local["site_tau"], "site_nu": local["site_nu"]})
    zero_rows = np.zeros(112, dtype=rows_dtype); zero_rows["error_work"] = .001; zero_rows["external_prediction"] = 0.; zero_rows["external_valid"] = True
    zero_local = local_z(zero_rows, np.zeros(2), np.ones(112, bool)); assert zero_local["site_tau"] <= 1e-8 and zero_local["noninformative_site"]
    result = build_hierarchy(hierarchy_obs)
    gamma, iota, eta, gamma_sigma, iota_sigma, eta_sigma, global_qa = result
    for e in ekeys:
        assert abs(gamma[e] - .04 * (ekeys.index(e) - 1)) < .01
        assert abs(sum(iota[(e, ifu)] for ifu in ifus)) < 1e-10
    for ifu in ifus:
        assert abs(sum(eta[(ifu, ifu + (amp,))] for amp in AMPS)) < 1e-10
        assert all(np.isfinite(eta_sigma[(ifu, ifu + (amp,))]) for amp in AMPS)
    assert all(abs(o["posterior_z"] - (.04 * (ekeys.index(o["exposure_key"]) - 1) + theta_iota[o["exposure_key"]][ifus.index(o["ifu"])] + theta_eta[AMPS.index(o["amp"][-1])])) < .01 for o in hierarchy_obs)
    shuffled = list(reversed(hierarchy_obs)); build_hierarchy(shuffled)
    assert np.allclose([o["posterior_z"] for o in hierarchy_obs], [o["posterior_z"] for o in shuffled[::-1]])
    sky_changed = zero_rows.copy(); sky_changed["sky_blank"] = True; zero_local_changed = local_z(sky_changed, np.zeros(2), np.ones(112, bool)); assert zero_local_changed["site_tau"] == zero_local["site_tau"]
    print(json.dumps({"status": "PASS", "tests": ["solve_additive", "additive prediction recovery", "unsupported hardware fallback", "all four amplifier positions", "ON/OFF collapse", "local_z posterior and site", "zero-X noninformative site", "build_hierarchy Helmert orientation", "global parameter recovery", "posterior uncertainty", "row-order invariance", "sky_blank has no z effect"]}))


def write_product(path, wavelength, exp_rows, ifu_rows, amp_rows, obs_rows, gamma, iota, eta, gamma_sigma, iota_sigma, eta_sigma, metadata, overwrite=False):
    class Exposure(tables.IsDescription):
        H5=tables.StringCol(256); exposure=tables.UInt8Col(); R_lambda=tables.Float64Col(shape=(N_WAVE,)); U_lambda=tables.Float64Col(shape=(N_WAVE,)); V_lambda=tables.Float64Col(shape=(N_WAVE,)); R_ON_OFF=tables.Float64Col(shape=(2,)); U_ON_OFF=tables.Float64Col(shape=(2,)); V_ON_OFF=tables.Float64Col(shape=(2,)); R_null=tables.Float64Col(shape=(5,)); U_null=tables.Float64Col(shape=(5,)); V_null=tables.Float64Col(shape=(5,)); iterations=tables.Int16Col(); converged=tables.BoolCol(); invariant_prediction_change=tables.Float64Col(); invariant_prediction_change_rms=tables.Float64Col(); n_blank_fibers=tables.Int32Col(); n_blank_ifus=tables.Int16Col(); n_blank_amplifiers=tables.Int16Col()
    class IFU(tables.IsDescription):
        SPECID=tables.Int32Col(); IFUSLOT=tables.Int32Col(); IFUID=tables.Int32Col(); c=tables.Float64Col(); supported=tables.BoolCol(); support_exposures=tables.Int16Col(); descriptive_scatter=tables.Float64Col()
    class Amp(tables.IsDescription):
        SPECID=tables.Int32Col(); IFUSLOT=tables.Int32Col(); IFUID=tables.Int32Col(); AMP=tables.StringCol(2); d=tables.Float64Col(); supported=tables.BoolCol(); support_exposures=tables.Int16Col(); descriptive_scatter=tables.Float64Col()
    class Obs(tables.IsDescription):
        H5=tables.StringCol(256); exposure=tables.UInt8Col(); SPECID=tables.Int32Col(); IFUSLOT=tables.Int32Col(); IFUID=tables.Int32Col(); AMP=tables.StringCol(2); local_z_mean=tables.Float64Col(); local_z_sigma=tables.Float64Col(); site_z_hat=tables.Float64Col(); site_sigma=tables.Float64Col(); site_tau=tables.Float64Col(); site_nu=tables.Float64Col(); noninformative_site=tables.BoolCol(); posterior_z_mean=tables.Float64Col(); posterior_z_sigma=tables.Float64Col(); source_counts=tables.Int32Col(shape=(2,)); information=tables.Float64Col(shape=(2,)); edge_mass=tables.Float64Col(); p_good=tables.Float64Col(); additive_R_ON_OFF=tables.Float64Col(shape=(2,)); additive_IFU_ON_OFF=tables.Float64Col(shape=(2,)); additive_AMP_ON_OFF=tables.Float64Col(shape=(2,));
    class Metadata(tables.IsDescription): key=tables.StringCol(128); value=tables.StringCol(1000000)
    filt=tables.Filters(complevel=5, complib="zlib", shuffle=True); path=Path(path).expanduser().resolve()
    if path.exists():
        if not overwrite:
            raise FileExistsError("output exists; use --overwrite: %s" % path)
        path.unlink()
    with tables.open_file(path, "w", title="M101 clean calibration") as h5:
        h5.root._v_attrs.schema_version=SCHEMA; h5.create_array("/", "wavelength", np.asarray(wavelength,float)); h5.create_group("/", "provenance"); h5.create_group("/", "exposures"); h5.create_group("/", "physical_ifus"); h5.create_group("/", "physical_amplifiers"); h5.create_group("/", "amplifier_observations"); h5.create_group("/", "global_parameters")
        for group, name, desc, rows in [(h5.root.exposures,"rows",Exposure,exp_rows),(h5.root.physical_ifus,"rows",IFU,ifu_rows),(h5.root.physical_amplifiers,"rows",Amp,amp_rows),(h5.root.amplifier_observations,"rows",Obs,obs_rows),(h5.root.provenance,"metadata",Metadata,[])]:
            table=h5.create_table(group,name,desc,expectedrows=max(len(rows),1),filters=filt)
            for source in rows:
                row=table.row
                for field in table.colnames:
                    if field in source: row[field]=source[field]
                row.append()
            table.flush()
        for name, values in (("gamma",gamma),("iota",iota),("eta",eta),("gamma_sigma",gamma_sigma),("iota_sigma",iota_sigma),("eta_sigma",eta_sigma)):
            keys=sorted(values,key=str); arr=np.asarray([values[k] for k in keys],float); h5.create_array("/global_parameters",name,arr); h5.root.global_parameters._v_attrs[name+"_keys"]=json.dumps([list(k) if isinstance(k,tuple) else k for k in keys])
        meta=h5.root.provenance.metadata
        for key,value in metadata.items(): row=meta.row; row["key"]=key; row["value"]=json.dumps(value,default=str); row.append()
        meta.flush()
    return path


def main():
    parser=ArgumentParser(description=__doc__); parser.add_argument("h5_positional",nargs="*"); parser.add_argument("--h5",nargs="+",action="append"); parser.add_argument("--h5-glob",action="append"); parser.add_argument("--measurements",required=False,default="m101_measurements.h5"); parser.add_argument("--external-blank-fibers",required=False); parser.add_argument("--on-filter",required=False); parser.add_argument("--off-filter",required=False); parser.add_argument("--compact-mask"); parser.add_argument("--output",default="m101_calibration.h5"); parser.add_argument("--max-additive-iterations",type=int,default=30); parser.add_argument("--additive-tolerance",type=float,default=1e-5); parser.add_argument("--minimum-finite-fraction",type=float,default=.8); parser.add_argument("--min-blank-ifus",type=int,default=3); parser.add_argument("--min-blank-per-amp",type=int,default=10); parser.add_argument("--overwrite",action="store_true"); parser.add_argument("--synthetic-test",action="store_true"); args=parser.parse_args()
    if args.synthetic_test: return synthetic_test()
    if not args.external_blank_fibers or not args.on_filter or not args.off_filter: raise SystemExit("--external-blank-fibers, --on-filter, and --off-filter are required")
    output=Path(args.output).expanduser().resolve()
    if not 0 < args.minimum_finite_fraction <= 1 or args.max_additive_iterations < 1 or args.additive_tolerance <= 0: raise SystemExit("invalid fitting controls")
    started=time.perf_counter()
    if output.exists() and not args.overwrite:
        raise SystemExit("output exists; use --overwrite: %s" % output)
    timings={}
    stage_started=time.perf_counter(); h5_paths=discover_h5(args); blank,blank_id=load_blank(args.external_blank_fibers,h5_paths); timings["blank_csv_loading_seconds"]=time.perf_counter()-stage_started
    filters={"ON":read_filter(args.on_filter),"OFF":read_filter(args.off_filter)}; filter_ids={key:identity(value) for key,value in (("ON",args.on_filter),("OFF",args.off_filter))}
    stage_started=time.perf_counter(); exp_data,ifu_obs,amp_obs,ifu_keys,amp_keys=collect_blank_groups(h5_paths,blank,args.minimum_finite_fraction,args.min_blank_ifus,args.min_blank_per_amp); timings["native_h5_blank_aggregation_seconds"]=time.perf_counter()-stage_started
    R,U,V,c,d,history,converged,additive_timings=solve_additive(exp_data,ifu_obs,amp_obs,ifu_keys,amp_keys,args.max_additive_iterations,args.additive_tolerance); timings.update(additive_timings)
    stage_started=time.perf_counter(); band=collapse_components(R,U,V,filters); timings["on_off_collapse_seconds"]=time.perf_counter()-stage_started
    stage_started=time.perf_counter(); rows,ids,measurement_meta=load_measurement_h5(args.measurements); timings["measurement_h5_loading_seconds"]=time.perf_counter()-stage_started
    stage_started=time.perf_counter(); compact=apply_compact_mask(rows,args.compact_mask); timings["compact_mask_application_seconds"]=time.perf_counter()-stage_started; h5_name={int(k):v for k,v in ids.items()}
    if set(h5_name.values()) != {path.name for path in h5_paths}:
        raise ValueError("measurement H5 basename set does not match supplied original H5 files")
    for band_name, key in (("ON", "on_filter_sha256"), ("OFF", "off_filter_sha256")):
        persisted = measurement_meta.get(key)
        if persisted and text(persisted) != filter_ids[band_name]["sha256"]:
            raise ValueError("%s filter identity disagrees with measurement provenance" % band_name)
    groups={}
    for index,row in enumerate(rows):
        key=(h5_name[int(row["h5_id"])],int(row["exposure"]),int(row["SPECID"]),int(row["IFUSLOT"]),int(row["IFUID"]),text(row["AMP"])); groups.setdefault(key,[]).append(index)
    expected = set()
    for path in h5_paths:
        with tables.open_file(path, "r") as raw:
            raw_groups, _ = exposure_groups(raw.root.Info)
            expected.update((path.name, g["exposure"], g["SPECID"], g["IFUSLOT"], g["IFUID"], g["AMP"]) for g in raw_groups)
    if set(groups) != expected:
        raise ValueError("measurement amplifier identity set does not exactly match original H5 groups")
    obs=[]; obs_rows=[]
    stage_started=time.perf_counter()
    for key,indices in sorted(groups.items()):
        if len(indices)!=112: raise ValueError("measurement amplifier observation %s has %d rows, expected 112" % (key,len(indices)))
        h5,e,spec,slot,uid,amp=key; ifu=(spec,slot,uid); ak=ifu+(amp,); exp_key=(h5,e); indices=np.asarray(indices); terms=band[(exp_key,"R")]+c.get(ifu,0)*band[(exp_key,"U")]+d.get(ak,0)*band[(exp_key,"V")]; local=local_z(rows[indices],terms,compact[indices]);
        obs.append({"exposure_key":exp_key,"ifu":ifu,"amp":ak,"z":local["z"],"sigma":local["sigma"],"site_z_hat":local["site_z_hat"],"site_sigma":local["site_sigma"],"site_tau":local["site_tau"],"site_nu":local["site_nu"]}); obs_rows.append({"H5":h5,"exposure":e,"SPECID":spec,"IFUSLOT":slot,"IFUID":uid,"AMP":amp,"local_z_mean":local["z"],"local_z_sigma":local["sigma"],"site_z_hat":local["site_z_hat"],"site_sigma":local["site_sigma"],"site_tau":local["site_tau"],"site_nu":local["site_nu"],"noninformative_site":local["noninformative_site"],"source_counts":local["n_source"],"information":local["information"],"edge_mass":local["edge_mass"],"p_good":1.0,"additive_R_ON_OFF":band[(exp_key,"R")],"additive_IFU_ON_OFF":c.get(ifu,0)*band[(exp_key,"U")],"additive_AMP_ON_OFF":d.get(ak,0)*band[(exp_key,"V")]})
    timings["local_z_calculations_seconds"]=time.perf_counter()-stage_started
    stage_started=time.perf_counter(); gamma,iota,eta,gamma_sigma,iota_sigma,eta_sigma,global_qa=build_hierarchy(obs); timings["global_hierarchy_seconds"]=time.perf_counter()-stage_started
    for source,o in zip(obs_rows,obs): source["posterior_z_mean"]=o["posterior_z"]; source["posterior_z_sigma"]=o["posterior_sigma"]
    nulls=null_responses(); exp_rows=[]
    for e in sorted(exp_data):
        terms=exp_data[e]; exp_rows.append({"H5":e[0],"exposure":e[1],"R_lambda":R[e],"U_lambda":U[e],"V_lambda":V[e],"R_ON_OFF":band[(e,"R")],"U_ON_OFF":band[(e,"U")],"V_ON_OFF":band[(e,"V")],"R_null":[weighted_scalar(R[e],x) for x in nulls],"U_null":[weighted_scalar(U[e],x) for x in nulls],"V_null":[weighted_scalar(V[e],x) for x in nulls],"iterations":len(history),"converged":converged,"invariant_prediction_change":history[-1]["invariant_prediction_change"],"invariant_prediction_change_rms":history[-1]["invariant_prediction_change_rms"],"n_blank_fibers":terms["n_blank"],"n_blank_ifus":terms["n_ifu"],"n_blank_amplifiers":terms["n_amp"]})
    ifu_rows=[]
    for key in ifu_keys:
        values=[rec["y"]-R[e] for (e,k),rec in ifu_obs.items() if k==key]; ifu_rows.append({"SPECID":key[0],"IFUSLOT":key[1],"IFUID":key[2],"c":c.get(key,0.),"supported":key in {k for _,k in ifu_obs},"support_exposures":len({e for (e,k) in ifu_obs if k==key}),"descriptive_scatter":robust_scale(np.asarray(values)) if values else np.nan})
    amp_rows=[]
    for key in amp_keys:
        values=[rec["y"]-R[e]-c.get(key[:3],0)*U[e] for (e,k),rec in amp_obs.items() if k==key]; amp_rows.append({"SPECID":key[0],"IFUSLOT":key[1],"IFUID":key[2],"AMP":key[3],"d":d.get(key,0.),"supported":key in {k for _,k in amp_obs},"support_exposures":len({e for (e,k) in amp_obs if k==key}),"descriptive_scatter":robust_scale(np.asarray(values)) if values else np.nan})
    runtime_candidates={key:timings[key] for key in ("blank_csv_loading_seconds","native_h5_blank_aggregation_seconds","additive_solve_total_seconds","on_off_collapse_seconds","measurement_h5_loading_seconds","compact_mask_application_seconds","local_z_calculations_seconds","global_hierarchy_seconds")}
    timings["top_three_runtime_contributors"]=[{"stage":key,"seconds":value} for key,value in sorted(runtime_candidates.items(),key=lambda item:item[1],reverse=True)[:3]]
    history_csv=output.parent / "m101_calibration_additive_history.csv"
    output_started=time.perf_counter(); write_additive_history_csv(history_csv,history,additive_timings); metadata={"schema_version":SCHEMA,"created_utc":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),"script":str(Path(__file__).resolve()),"measurement":identity(args.measurements),"input_h5":[identity(p) for p in h5_paths],"external_blank_fibers":blank_id,"filters":filter_ids,"measurement_metadata":measurement_meta,"model_equation":"W = exp(z) S + R + cU + dV + noise","native_final_equation":"[Fibers.spectrum/Survey.offset - R - cU - dV] / exp(posterior_z_mean)","uses_Fibers_skyspectrum":False,"uses_sky_blank_for_z":False,"fits_p":False,"fits_alpha":False,"fits_q_mode":False,"source_likelihood":"external_valid + compact_mask + date/finite/error; |X|<=1e-12 has zero z information","delta_z_band":0.0,"error_model":"sqrt((s_b*sigma_data)^2 + (f_b*X)^2)","normalization":{"c":"robust location zero, robust scale one; amplitude in U","d":"arithmetic centered within physical IFU, pooled robust scale one; amplitude in V"},"convergence":{"history":history,"iterations":len(history),"converged":converged,"invariant_prediction_change":history[-1]["invariant_prediction_change"]},"timing":timings,"runtime_seconds":time.perf_counter()-started,"unsupported_ifus":[list(k) for k in ifu_keys if k not in {x[1] for x in ifu_obs}],"unsupported_amplifiers":[list(k) for k in amp_keys if k not in {x[1] for x in amp_obs}],"additive_history_csv":str(history_csv)}; write_product(output,WAVE,exp_rows,ifu_rows,amp_rows,obs_rows,gamma,iota,eta,gamma_sigma,iota_sigma,eta_sigma,metadata,overwrite=args.overwrite); timings["output_provenance_writing_seconds"]=time.perf_counter()-output_started
    timings["total_runtime_seconds"]=time.perf_counter()-started
    metadata["timing"]=timings; write_product(output,WAVE,exp_rows,ifu_rows,amp_rows,obs_rows,gamma,iota,eta,gamma_sigma,iota_sigma,eta_sigma,metadata,overwrite=True)
    print("timings: %s" % json.dumps({key:timings[key] for key in ("blank_csv_loading_seconds","native_h5_blank_aggregation_seconds","additive_initialization_seconds","additive_solve_total_seconds","mean_outer_iteration_seconds","on_off_collapse_seconds","measurement_h5_loading_seconds","compact_mask_application_seconds","local_z_calculations_seconds","global_hierarchy_seconds","output_provenance_writing_seconds","total_runtime_seconds")},default=str))
    print("top three runtime contributors: %s" % json.dumps(timings["top_three_runtime_contributors"],default=str))
    print(json.dumps({"output":str(output),"history_csv":str(history_csv),"runtime_seconds":timings["total_runtime_seconds"],"iterations":len(history),"converged":converged,"convergence_metric":history[-1]["invariant_prediction_change"],"supported_ifus":sum(x["supported"] for x in ifu_rows),"unsupported_ifus":[x for x in ifu_rows if not x["supported"]],"supported_amplifiers":sum(x["supported"] for x in amp_rows if x["supported"]),"unsupported_amplifiers":[x for x in amp_rows if not x["supported"]],"amplifier_observations":len(obs_rows),"delta_z_band":0.0},indent=2,default=str))


if __name__ == "__main__": main()
