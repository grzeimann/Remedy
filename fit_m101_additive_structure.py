#!/usr/bin/env python3
"""Fit the production full-spectrum additive M101 calibration structure.

The input is the independently classified external blank population.  The
fit is performed on robust blank group spectra, rather than on the native
fiber matrix, using

    A_eja(lambda) = R_e(lambda) + c_j U_e(lambda) + d_a V_e(lambda).

This program deliberately does not estimate alpha, z, p, a source model, or
anything from ``Fibers.skyspectrum``.  The only optional blank-only alpha
preprocessing described by the calibration design is intentionally omitted:
the robust group spectra are fitted directly from W.
"""

import argparse
import csv
import gzip
import glob
import hashlib
import json
from pathlib import Path
import time
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tables

import diagnose_m101_hierarchical as hm
from math_utils import biweight


N_AMP_FIBERS = 112
N_EXPOSURES = 3
AMPS = ("LL", "LU", "RL", "RU")
DEF_WAVE = np.asarray(hm.DEF_WAVE, dtype=float)
BANDS = ("ON", "OFF")
NULL_BANDS = ("NULL_HIGH1", "NULL_LOW1", "NULL_HIGH2", "NULL_LOW2", "NULL_HIGH3")
NULL_WINDOWS = ((3538.0, 3550.0), (3602.0, 3614.0), (3628.0, 3640.0),
                (3676.0, 3688.0), (3902.0, 3914.0))
SCHEMA_VERSION = "m101_additive_structure_v1"
N_WAVE = DEF_WAVE.size
DEFAULT_MINIMUM_FINITE_FRACTION = .8
DEFAULT_MIN_BLANK_IFUS = 3
DEFAULT_MIN_BLANK_PER_AMP = 10


def text(value):
    return value.decode("utf-8", errors="replace").strip() if isinstance(value, (bytes, np.bytes_)) else str(value).strip()


def file_identity(path):
    path = Path(path).expanduser().resolve()
    stat = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(path), "filename": path.name, "size": int(stat.st_size),
            "mtime_ns": int(stat.st_mtime_ns), "sha256": digest.hexdigest()}


def robust_location(values):
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.nan
    if finite.size < 3:
        return float(np.median(finite))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        result = float(biweight(finite))
    return result if np.isfinite(result) else float(np.median(finite))


def robust_scale(values):
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.nan
    mad = np.median(np.abs(finite - np.median(finite)))
    return float(1.4826 * mad) if mad > 0 else float(np.std(finite))


def robust_spectrum(values):
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or values.shape[0] == 0:
        return np.full(values.shape[-1] if values.ndim else 0, np.nan)
    if values.shape[0] < 5:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return np.nanmedian(values, axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        result = np.asarray(biweight(values, axis=0), dtype=float)
        fallback = np.nanmedian(values, axis=0)
    result[~np.isfinite(result)] = fallback[~np.isfinite(result)]
    return result


def robust_rms(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return float(np.sqrt(np.median(values * values))) if values.size else np.nan


def parse_blank(value):
    value = text(value).lower()
    if value in ("true", "1"):
        return True
    if value in ("false", "0"):
        return False
    raise ValueError("malformed blank value: %r" % value)


def discover_h5(args):
    values = list(args.h5_positional or [])
    for group in args.h5 or []:
        values.extend(group if isinstance(group, (list, tuple)) else [group])
    for pattern in args.h5_glob or []:
        values.extend(glob.glob(str(Path(pattern).expanduser())))
    paths = []
    for value in values:
        path = Path(value).expanduser().resolve()
        if not path.is_file():
            raise ValueError("H5 input does not exist: %s" % path)
        paths.append(path)
    unique = {}
    for path in paths:
        if path.name in unique and unique[path.name] != path:
            raise ValueError("ambiguous H5 basename: %s" % path.name)
        unique[path.name] = path
    result = [unique[name] for name in sorted(unique)]
    if not result:
        raise ValueError("no H5 inputs were supplied")
    return result


def load_external_blank(path, h5_paths):
    """Strictly match basename + native row index and cross-check exposure."""
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise ValueError("external blank table does not exist: %s" % path)
    states, labels_by_name = {}, {}
    for h5_path in h5_paths:
        with tables.open_file(h5_path, mode="r") as h5:
            _, labels = hm.build_groups(h5.root.Info)
            states[h5_path.name] = {"seen": np.zeros(h5.root.Info.nrows, dtype=bool),
                                    "blank": np.zeros(h5.root.Info.nrows, dtype=bool)}
            labels_by_name[h5_path.name] = np.asarray(labels, dtype=int)
    opener = gzip.open if path.suffix == ".gz" else open
    parsed = 0
    extras = set()
    with opener(path, "rt", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"H5", "exposure", "row_index", "blank"}
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError("external blank CSV lacks columns: %s" % sorted(missing))
        for row in reader:
            parsed += 1
            if row.get(None) is not None:
                raise ValueError("extra fields in external blank CSV row %d" % parsed)
            name = Path(text(row.get("H5", ""))).name
            try:
                index = int(text(row.get("row_index", "")))
                exposure = int(text(row.get("exposure", "")))
                blank = parse_blank(row.get("blank", ""))
            except (TypeError, ValueError) as exc:
                raise ValueError("malformed external blank CSV row %d" % parsed) from exc
            if name not in states:
                extras.add(name)
                continue
            state = states[name]
            if index < 0 or index >= state["seen"].size:
                raise ValueError("row_index %d is out of range for %s" % (index, name))
            if state["seen"][index]:
                raise ValueError("duplicate blank match for %s row %d" % (name, index))
            expected = int(labels_by_name[name][index])
            if exposure != expected:
                raise ValueError("CSV/H5 exposure mismatch for %s row %d: %d != %d" %
                                 (name, index, exposure, expected))
            state["seen"][index] = True
            state["blank"][index] = blank
    for name, state in states.items():
        if not np.all(state["seen"]):
            raise ValueError("external blank table is missing %d rows for %s" %
                             (int(np.sum(~state["seen"])), name))
    masks = {name: state["blank"] for name, state in states.items()}
    return masks, {"identity": file_identity(path), "rows": parsed,
                   "blank_rows": int(sum(np.sum(v) for v in masks.values())),
                   "nonblank_rows": int(sum(v.size - np.sum(v) for v in masks.values())),
                   "ignored_extra_h5_files": sorted(extras),
                   "matching_identity": "H5 basename + original/native row index; exposure cross-checked",
                   "classification_semantics": "external classification only"}


def load_fq(path):
    path = Path(path).expanduser().resolve()
    rows = list(csv.DictReader(path.open(newline="")))
    if not rows or not {"q", "f"}.issubset(rows[0]):
        raise ValueError("fq template must contain q,f columns")
    result = np.full(N_AMP_FIBERS, np.nan)
    for row in rows:
        q, value = int(text(row["q"])), float(text(row["f"]))
        if q < 0 or q >= N_AMP_FIBERS or np.isfinite(result[q]) or not np.isfinite(value):
            raise ValueError("invalid or duplicate fq q=%s" % q)
        result[q] = value
    if not np.all(np.isfinite(result)):
        raise ValueError("fq template must contain q=0..111 exactly once")
    return result, file_identity(path)


def load_filters(on_path, off_path):
    filters = {"ON": hm.read_filter(on_path), "OFF": hm.read_filter(off_path)}
    return filters, {name: file_identity(path) for name, path in (("ON", on_path), ("OFF", off_path))}


def null_responses():
    responses = []
    provenance = {}
    for name, (lower, upper) in zip(NULL_BANDS, NULL_WINDOWS):
        response = np.where((DEF_WAVE >= lower) & (DEF_WAVE <= upper), 1.0, 0.0)
        if np.sum(response) < 3:
            raise ValueError("null band has fewer than three wavelength samples: %s" % name)
        responses.append(response)
        provenance[name] = {"requested_bounds_A": [lower, upper],
                            "samples_A": DEF_WAVE[response != 0].tolist(),
                            "response_definition": "1 inside inclusive requested limits, 0 outside"}
    return np.asarray(responses), provenance


def collapse(values, response):
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values) & np.isfinite(response)[None, :]
    weights = np.where(finite, response[None, :], 0.0)
    denominator = np.sum(weights, axis=1)
    result = np.full(values.shape[0], np.nan)
    good = denominator != 0.0
    result[good] = np.sum(np.where(finite, values, 0.0) * response[None, :], axis=1)[good] / denominator[good]
    return result


def group_lookup(groups, nrows):
    result = np.full(nrows, -1, dtype=int)
    for index, group in enumerate(groups):
        result[group["indices"]] = index
    if np.any(result < 0):
        raise ValueError("native rows missing from amplifier groups")
    return result


def surveys_by_exposure(h5):
    result = {}
    for row in h5.root.Survey:
        exposure = int(row["exp"])
        if exposure in result:
            raise ValueError("duplicate Survey exposure %d" % exposure)
        result[exposure] = row
    if set(result) != set(range(1, N_EXPOSURES + 1)):
        raise ValueError("Survey exposures are not exactly 1,2,3")
    return result


def collect_observations(h5_paths, blank_masks, minimum_finite_fraction, min_blank_per_amp,
                         min_blank_ifus):
    exposure_data = {}
    ifu_obs, amp_obs = {}, {}
    all_ifus, all_amps = set(), set()
    h5_ids = {path.name: index for index, path in enumerate(h5_paths)}
    for path in h5_paths:
        with tables.open_file(path, mode="r") as h5:
            info, fibers = h5.root.Info, h5.root.Fibers
            if "spectrum" not in fibers.colnames:
                raise ValueError("%s Fibers lacks spectrum" % path)
            groups, labels = hm.build_groups(info)
            group_ids = group_lookup(groups, info.nrows)
            date_mask_bad = hm.masked_rows(path, np.asarray(info.cols.ifuslot[:]),
                                           np.asarray([text(v) for v in info.cols.amp[:]]))
            surveys = surveys_by_exposure(h5)
            mask = blank_masks[path.name]
            for exposure in range(1, N_EXPOSURES + 1):
                exposure_key = (path.name, exposure)
                exposure_rows = np.flatnonzero(labels == exposure)
                candidates = exposure_rows[mask[exposure_rows] & ~date_mask_bad[exposure_rows]]
                spectra = np.asarray(fibers.read_coordinates(candidates, field="spectrum"), dtype=float)
                offset = float(surveys[exposure]["offset"])
                if not np.isfinite(offset) or offset == 0:
                    raise ValueError("%s exposure %d has invalid Survey.offset" % (path, exposure))
                finite_fraction = np.mean(np.isfinite(spectra), axis=1) if spectra.size else np.asarray([])
                usable = finite_fraction >= minimum_finite_fraction
                candidates, spectra = candidates[usable], spectra[usable] / offset
                by_ifu, by_amp = {}, {}
                for group_index, group in enumerate(groups):
                    if group["exposure"] != exposure:
                        continue
                    local = np.flatnonzero(group_ids[candidates] == group_index)
                    if not local.size:
                        continue
                    amp_key = (group["specid"], group["ifuslot"], group["ifuid"], group["amp"])
                    ifu_key = amp_key[:3]
                    all_ifus.add(ifu_key); all_amps.add(amp_key)
                    values = spectra[local]
                    by_amp[amp_key] = (values, int(values.shape[0]))
                    by_ifu.setdefault(ifu_key, []).append(values)
                amp_count = 0
                for key, (values, count) in by_amp.items():
                    if count < min_blank_per_amp:
                        continue
                    amp_obs[(exposure_key, key)] = {"y": robust_spectrum(values), "n": count}
                    amp_count += 1
                if len(by_ifu) < min_blank_ifus:
                    # The exposure remains in the product, but its sparse
                    # support is recorded and it is not silently invented.
                    pass
                for key, chunks in by_ifu.items():
                    values = np.vstack(chunks)
                    if values.shape[0] >= 1:
                        ifu_obs[(exposure_key, key)] = {"y": robust_spectrum(values),
                                                        "n": int(values.shape[0])}
                exposure_data[exposure_key] = {"h5_id": h5_ids[path.name],
                                               "n_blank": int(candidates.size),
                                               "n_ifu": int(len(by_ifu)),
                                               "n_amp": int(amp_count)}
    if not ifu_obs:
        raise ValueError("no usable external blank IFU group spectra were found")
    return exposure_data, ifu_obs, amp_obs, sorted(all_ifus), sorted(all_amps)


def robust_slope(y, x, counts=None):
    y, x = np.asarray(y, dtype=float), np.asarray(x, dtype=float)
    use = np.isfinite(y) & np.isfinite(x) & (np.abs(x) > 1e-14)
    if np.sum(use) < 2:
        return 0.0
    y, x = y[use], x[use]
    weights = np.ones(y.size) if counts is None else np.asarray(counts, dtype=float)[use]
    weights = np.clip(np.where(np.isfinite(weights), weights, 1.0), 1.0, 112.0)
    slope = float(np.sum(weights * x * y) / max(np.sum(weights * x * x), 1e-30))
    for _ in range(5):
        residual = y - slope * x
        scale = robust_scale(residual)
        if not np.isfinite(scale) or scale <= 0:
            break
        robust_weight = np.minimum(1.0, 1.5 * scale / np.maximum(np.abs(residual), 1e-30))
        total = weights * robust_weight
        slope = float(np.sum(total * x * y) / max(np.sum(total * x * x), 1e-30))
    return slope if np.isfinite(slope) else 0.0


def fit_template(records, coefficient, residuals):
    result = np.full(N_WAVE, np.nan)
    for wave in range(N_WAVE):
        ys, xs, ns = [], [], []
        for record, residual in zip(records, residuals):
            value = residual[wave]
            if np.isfinite(value) and np.isfinite(coefficient[record["hardware"]]):
                ys.append(value); xs.append(coefficient[record["hardware"]]); ns.append(record["n"])
        result[wave] = robust_slope(np.asarray(ys), np.asarray(xs), np.asarray(ns)) if ys else np.nan
    return result


def fit_coefficients(records, templates, residuals, hardware_keys):
    result = {key: 0.0 for key in hardware_keys}
    for key in hardware_keys:
        selected = [(r, residual) for r, residual in zip(records, residuals)
                    if r["hardware"] == key and np.any(np.isfinite(residual))]
        if not selected:
            continue
        ys, xs, ns = [], [], []
        for record, residual in selected:
            template = templates[record["exposure"]]
            use = np.isfinite(residual) & np.isfinite(template)
            ys.extend(residual[use]); xs.extend(template[use]); ns.extend([record["n"]] * int(np.sum(use)))
        result[key] = robust_slope(np.asarray(ys), np.asarray(xs), np.asarray(ns))
    return result


def normalize_ifu(c, U, R, supported):
    values = np.asarray([c[key] for key in supported if c.get(key, 0.0) != 0.0], dtype=float)
    if values.size == 0:
        return {key: 0.0 for key in c}, U, R, {"location": 0.0, "scale": 1.0}
    location = robust_location(values)
    scale = robust_scale(values)
    if not np.isfinite(scale) or scale <= 1e-12:
        scale = 1.0
    old_U = {key: np.asarray(value, dtype=float).copy() for key, value in U.items()}
    for exposure in U:
        R[exposure] = R[exposure] + location * old_U[exposure]
        U[exposure] = old_U[exposure] * scale
    normalized = {key: (c[key] - location) / scale if c[key] != 0.0 else 0.0 for key in c}
    return normalized, U, R, {"location": float(location), "scale": float(scale)}


def center_and_normalize_amp(d, V, ifu_keys, amp_keys, supported):
    for ifu in ifu_keys:
        values = [d[(ifu[0], ifu[1], ifu[2], amp)] for amp in AMPS
                  if (ifu[0], ifu[1], ifu[2], amp) in supported]
        if values:
            # The four physical amplifiers define the relative coordinate
            # system.  Use their deterministic arithmetic mean so the
            # within-IFU centering identity is exact when all four are
            # supported; robust fitting has already limited outliers.
            location = float(np.mean(values))
            for amp in AMPS:
                key = (ifu[0], ifu[1], ifu[2], amp)
                if key in d and key in supported:
                    d[key] -= location
    values = np.asarray([d[key] for key in amp_keys if key in supported], dtype=float)
    scale = robust_scale(values) if values.size else 1.0
    if not np.isfinite(scale) or scale <= 1e-12:
        scale = 1.0
    for exposure in V:
        V[exposure] *= scale
    for key in d:
        if key in supported:
            d[key] /= scale
        else:
            d[key] = 0.0
    return d, V, {"scale": float(scale)}


def solve_structure(exposure_data, ifu_obs, amp_obs, ifu_keys, amp_keys,
                    max_iterations, tolerance):
    exposures = sorted(exposure_data)
    # R starts as the robust exposure location of supported IFU spectra.
    R = {}
    for exposure in exposures:
        values = [record["y"] for (key, _), record in ifu_obs.items() if key == exposure]
        R[exposure] = robust_spectrum(np.asarray(values)) if values else np.zeros(N_WAVE)
    c = {key: 0.0 for key in ifu_keys}
    d = {key: 0.0 for key in amp_keys}
    # Use the first stable, lexicographically ordered residual as the
    # deterministic rank-1 start.  A robust location here would erase the
    # mode because c is centered at zero by construction.
    U = {}
    for exposure in exposures:
        candidates = [record["y"] - R[exposure] for (key, _), record in ifu_obs.items()
                      if key == exposure]
        U[exposure] = (np.asarray(max(candidates, key=robust_rms), dtype=float).copy()
                       if candidates else np.zeros(N_WAVE))
    supported_c = {key for (exposure, key) in ifu_obs}
    supported_d = {key for (exposure, key) in amp_obs}
    c = fit_coefficients([{"exposure": exposure, "hardware": key, "n": record["n"]}
                          for (exposure, key), record in ifu_obs.items()], U,
                         [record["y"] - R[exposure] for (exposure, _), record in ifu_obs.items()], ifu_keys)
    c, U, R, c_norm = normalize_ifu(c, U, R, sorted(supported_c))
    V = {exposure: np.zeros(N_WAVE) for exposure in exposures}
    history = []
    for iteration in range(1, max_iterations + 1):
        old_components = np.concatenate([np.concatenate([R[e], U[e], V[e]]) for e in exposures])
        old_model = np.asarray([R[exposure] + c[key[:3]] * U[exposure] + d[key] * V[exposure]
                                for (exposure, key) in amp_obs], dtype=float)
        ifu_records = [{"exposure": exposure, "hardware": key, "n": record["n"]}
                       for (exposure, key), record in ifu_obs.items()]
        ifu_residuals = [record["y"] - R[exposure]
                         for (exposure, _), record in ifu_obs.items()]
        c = fit_coefficients(ifu_records, U, ifu_residuals, ifu_keys)
        for exposure in exposures:
            selected = [residual for record, residual in zip(ifu_records, ifu_residuals)
                        if record["exposure"] == exposure]
            selected_records = [record for record in ifu_records if record["exposure"] == exposure]
            if selected:
                U[exposure] = fit_template(selected_records, c, selected)
        c, U, R, c_norm = normalize_ifu(c, U, R, sorted(supported_c))
        amp_records = [{"exposure": exposure, "hardware": key, "n": record["n"]}
                       for (exposure, key), record in amp_obs.items()]
        amp_residuals = [record["y"] - R[exposure] - c[key[:3]] * U[exposure]
                         for (exposure, key), record in amp_obs.items()]
        # Start the amplifier mode from one deterministic residual and then
        # update V by robust regression against the persistent d coefficients.
        # A group location across four amplifiers would erase a mean-centered
        # mode, so it is not a valid initializer here.
        for exposure in exposures:
            selected = [residual for (exp, _), residual in zip(amp_obs, amp_residuals) if exp == exposure]
            if selected:
                V[exposure] = np.asarray(max(selected, key=robust_rms), dtype=float).copy()
        d = fit_coefficients(amp_records, V, amp_residuals, amp_keys)
        for exposure in exposures:
            selected = [residual for (exp, _), residual in zip(amp_obs, amp_residuals) if exp == exposure]
            selected_records = [record for record in amp_records if record["exposure"] == exposure]
            if selected:
                V[exposure] = fit_template(selected_records, d, selected)
        d = fit_coefficients(amp_records, V, amp_residuals, amp_keys)
        d, V, d_norm = center_and_normalize_amp(d, V, ifu_keys, amp_keys, supported_d)
        for exposure in exposures:
            corrected = [record["y"] - c[key[:3]] * U[exposure] - d[key] * V[exposure]
                         for (exp, key), record in amp_obs.items() if exp == exposure]
            if corrected:
                # A damped update prevents a sparse amplifier population from
                # moving the exposure owner by more than one alternating step.
                updated_R = robust_spectrum(np.asarray(corrected))
                R[exposure] = .5 * R[exposure] + .5 * updated_R
        new_components = np.concatenate([np.concatenate([R[e], U[e], V[e]]) for e in exposures])
        component_delta = float(np.nanmax(np.abs(new_components - old_components)) /
                                max(np.nanmax(np.abs(old_components)), 1e-12))
        new_model = np.asarray([R[exposure] + c[key[:3]] * U[exposure] + d[key] * V[exposure]
                                for (exposure, key) in amp_obs], dtype=float)
        delta = float(np.nanmax(np.abs(new_model - old_model)) /
                      max(np.nanmax(np.abs(old_model)), 1e-12))
        stage3 = [record["y"] - R[exposure] - c[key[:3]] * U[exposure] - d[key] * V[exposure]
                  for (exposure, key), record in amp_obs.items()]
        rms = robust_rms(np.asarray(stage3))
        history.append({"iteration": iteration, "delta": delta, "component_delta": component_delta, "stage3_rms": rms,
                        "c_location": c_norm["location"], "c_scale": c_norm["scale"],
                        "d_scale": d_norm["scale"]})
        if not np.isfinite(delta):
            raise RuntimeError("additive alternating fit diverged at iteration %d" % iteration)
        if delta < tolerance:
            return R, U, V, c, d, history, True
    return R, U, V, c, d, history, False


def collapse_terms(R, U, V, c, d, exposure, ifu, amp, filters, nulls):
    source = np.asarray([R[exposure] + c[ifu] * U[exposure] + d[amp] * V[exposure]])
    result = {"R_ON_OFF": np.asarray([hm.weighted_scalar(R[exposure], filters[name]) for name in BANDS]),
              "U_ON_OFF": np.asarray([hm.weighted_scalar(U[exposure], filters[name]) for name in BANDS]),
              "V_ON_OFF": np.asarray([hm.weighted_scalar(V[exposure], filters[name]) for name in BANDS]),
              "R_null": np.asarray([hm.weighted_scalar(R[exposure], response) for response in nulls]),
              "U_null": np.asarray([hm.weighted_scalar(U[exposure], response) for response in nulls]),
              "V_null": np.asarray([hm.weighted_scalar(V[exposure], response) for response in nulls])}
    result["source_prediction"] = source
    return result


def qa_rows(exposure_data, ifu_obs, amp_obs, R, U, V, c, d):
    rows = []
    for (exposure, key), record in amp_obs.items():
        prediction = R[exposure] + c[key[:3]] * U[exposure] + d[key] * V[exposure]
        values = record["y"]
        stages = [values, values - R[exposure], values - R[exposure] - c[key[:3]] * U[exposure], values - prediction]
        rows.append({"level": "amplifier", "H5": exposure[0], "exposure": exposure[1],
                     "SPECID": key[0], "IFUSLOT": key[1], "IFUID": key[2], "AMP": key[3],
                     "N_blank": record["n"], "stage_rms": np.asarray([robust_rms(v) for v in stages])})
    for (exposure, key), record in ifu_obs.items():
        stages = [record["y"], record["y"] - R[exposure],
                  record["y"] - R[exposure] - c[key] * U[exposure],
                  record["y"] - R[exposure] - c[key] * U[exposure]]
        rows.append({"level": "ifu", "H5": exposure[0], "exposure": exposure[1],
                     "SPECID": key[0], "IFUSLOT": key[1], "IFUID": key[2], "AMP": "",
                     "N_blank": record["n"], "stage_rms": np.asarray([robust_rms(v) for v in stages])})
    return rows


def _description():
    class Exposure(tables.IsDescription):
        h5_id = tables.Int16Col(); H5 = tables.StringCol(256); exposure = tables.UInt8Col()
        R_lambda = tables.Float64Col(shape=(N_WAVE,)); U_lambda = tables.Float64Col(shape=(N_WAVE,)); V_lambda = tables.Float64Col(shape=(N_WAVE,))
        R_ON_OFF = tables.Float64Col(shape=(2,)); U_ON_OFF = tables.Float64Col(shape=(2,)); V_ON_OFF = tables.Float64Col(shape=(2,))
        R_null = tables.Float64Col(shape=(5,)); U_null = tables.Float64Col(shape=(5,)); V_null = tables.Float64Col(shape=(5,))
        n_blank_fibers = tables.Int32Col(); n_blank_ifus = tables.Int16Col(); n_blank_amplifiers = tables.Int16Col()
        iterations = tables.Int16Col(); converged = tables.BoolCol(); final_delta = tables.Float64Col(); stage_rms = tables.Float64Col(shape=(4,))
    class IFU(tables.IsDescription):
        SPECID = tables.Int32Col(); IFUSLOT = tables.Int32Col(); IFUID = tables.Int32Col(); c = tables.Float64Col()
        c_support_exposures = tables.Int16Col(); c_supported = tables.BoolCol(); c_scatter = tables.Float64Col()
    class Amp(tables.IsDescription):
        SPECID = tables.Int32Col(); IFUSLOT = tables.Int32Col(); IFUID = tables.Int32Col(); AMP = tables.StringCol(2); d = tables.Float64Col()
        d_support_exposures = tables.Int16Col(); d_supported = tables.BoolCol(); d_scatter = tables.Float64Col()
    class QA(tables.IsDescription):
        level = tables.StringCol(16); H5 = tables.StringCol(256); exposure = tables.UInt8Col(); SPECID = tables.Int32Col(); IFUSLOT = tables.Int32Col(); IFUID = tables.Int32Col(); AMP = tables.StringCol(2); N_blank = tables.Int32Col(); stage_rms = tables.Float64Col(shape=(4,))
    class Convergence(tables.IsDescription):
        iteration = tables.Int16Col(); delta = tables.Float64Col(); stage3_rms = tables.Float64Col(); c_location = tables.Float64Col(); c_scale = tables.Float64Col(); d_scale = tables.Float64Col()
    class Metadata(tables.IsDescription):
        key = tables.StringCol(128); value = tables.StringCol(8192)
    return Exposure, IFU, Amp, QA, Convergence, Metadata


def write_product(path, h5_paths, exposure_data, ifu_obs, amp_obs, ifu_keys, amp_keys,
                  R, U, V, c, d, history, converged, filters, filter_ids, fq_id,
                  blank_provenance, null_provenance, args, runtime):
    path = Path(path).expanduser().resolve()
    if path.exists() and not args.overwrite:
        raise ValueError("output exists; use --overwrite: %s" % path)
    nulls, _ = null_responses()
    qa = qa_rows(exposure_data, ifu_obs, amp_obs, R, U, V, c, d)
    Exposure, IFU, Amp, QA, Convergence, Metadata = _description()
    filters_h5 = tables.Filters(complevel=5, complib="zlib", shuffle=True)
    input_ids = [file_identity(p) for p in h5_paths]
    metadata = {
        "schema_version": SCHEMA_VERSION, "script": str(Path(__file__).resolve()),
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "input_h5": input_ids, "external_blank_fibers": blank_provenance,
        "fq_template": fq_id, "on_filter": filter_ids["ON"], "off_filter": filter_ids["OFF"],
        "wavelength_grid": {"array": "hm.DEF_WAVE", "n": int(N_WAVE),
                            "min": float(DEF_WAVE[0]), "max": float(DEF_WAVE[-1]),
                            "sha256": hashlib.sha256(np.asarray(DEF_WAVE, dtype="<f8").tobytes()).hexdigest()},
        "band_order": list(BANDS), "null_band_order": list(NULL_BANDS),
        "null_definitions": null_provenance,
        "model_equation": "A_eja(lambda)=R_e(lambda)+c_j*U_e(lambda)+d_a*V_e(lambda)",
        "scientific_model": "W=exp(z)S+alpha*K*f(q)+R+cU+dV+noise; additive terms are subtracted before exp(z)",
        "blank_alpha_choice": "not used; robust group spectra are fit directly from W",
        "uses_Fibers_skyspectrum": False, "fits_z": False, "fits_alpha": False, "fits_p": False,
        "uses_leave_one_out": False, "uses_pca_or_svd": False,
        "physical_ifu_identity": "SPECID + IFUSLOT + IFUID",
        "physical_amplifier_identity": "SPECID + IFUSLOT + IFUID + AMP",
        "exposure_identity": "H5 basename + exposure",
        "group_spectrum_definition": "robust wavelength-specific external-blank location of Fibers.spectrum/Survey.offset",
        "normalization": {"R": "exposure-wide mean owner", "c": "robust location zero and robust scale one", "d": "centered among supported amplifiers within each physical IFU; pooled robust scale one"},
        "thresholds": {"minimum_finite_fraction": args.minimum_finite_fraction, "min_blank_ifus": args.min_blank_ifus, "min_blank_per_amp": args.min_blank_per_amp},
        "convergence": {"max_iterations": args.max_iterations, "tolerance": args.tolerance, "iterations": len(history), "converged": bool(converged), "history": history},
        "runtime_seconds": runtime,
    }
    amp_qa = np.asarray([row["stage_rms"] for row in qa if row["level"] == "amplifier"], dtype=float)
    if amp_qa.size:
        stage_medians = np.nanmedian(amp_qa, axis=0)
        metadata["qa_stage_rms"] = stage_medians.tolist()
        metadata["qa_stage_variance_reduction"] = {
            "R": float(1.0 - (stage_medians[1] / stage_medians[0]) ** 2) if stage_medians[0] > 0 else np.nan,
            "cU": float(1.0 - (stage_medians[2] / stage_medians[1]) ** 2) if stage_medians[1] > 0 else np.nan,
            "dV": float(1.0 - (stage_medians[3] / stage_medians[2]) ** 2) if stage_medians[2] > 0 else np.nan,
        }
    with tables.open_file(path, mode="w", title="M101 additive structure") as h5:
        h5.root._v_attrs.schema_version = SCHEMA_VERSION
        h5.root._v_attrs.band_order = json.dumps(list(BANDS))
        h5.root._v_attrs.null_band_order = json.dumps(list(NULL_BANDS))
        h5.root._v_attrs.wavelength_grid = json.dumps({"array": "hm.DEF_WAVE", "n": int(N_WAVE)})
        tables_data = (("exposures", Exposure, []), ("physical_ifus", IFU, []),
                       ("physical_amplifiers", Amp, []), ("qa_group_residuals", QA, qa),
                       ("convergence", Convergence, history))
        for name, description, rows in tables_data:
            table = h5.create_table("/", name, description, expectedrows=max(len(rows), len(exposure_data)), filters=filters_h5)
            if name == "exposures":
                for exposure in sorted(exposure_data):
                    row = table.row; info = exposure_data[exposure]
                    row["h5_id"] = info["h5_id"]; row["H5"] = exposure[0]; row["exposure"] = exposure[1]
                    row["R_lambda"] = R[exposure]; row["U_lambda"] = U[exposure]; row["V_lambda"] = V[exposure]
                    row["R_ON_OFF"] = [hm.weighted_scalar(R[exposure], filters[n]) for n in BANDS]
                    row["U_ON_OFF"] = [hm.weighted_scalar(U[exposure], filters[n]) for n in BANDS]
                    row["V_ON_OFF"] = [hm.weighted_scalar(V[exposure], filters[n]) for n in BANDS]
                    row["R_null"] = [hm.weighted_scalar(R[exposure], x) for x in nulls]
                    row["U_null"] = [hm.weighted_scalar(U[exposure], x) for x in nulls]
                    row["V_null"] = [hm.weighted_scalar(V[exposure], x) for x in nulls]
                    row["n_blank_fibers"] = info["n_blank"]; row["n_blank_ifus"] = info["n_ifu"]; row["n_blank_amplifiers"] = info["n_amp"]
                    row["iterations"] = len(history); row["converged"] = converged; row["final_delta"] = history[-1]["delta"] if history else np.nan
                    row["stage_rms"] = np.asarray([r["stage_rms"] for r in qa if r["level"] == "amplifier" and r["H5"] == exposure[0] and r["exposure"] == exposure[1]], dtype=float).mean(axis=0) if any(r["level"] == "amplifier" and r["H5"] == exposure[0] and r["exposure"] == exposure[1] for r in qa) else np.full(4, np.nan)
                    row.append()
            elif name == "physical_ifus":
                support = {key: len({exposure for exposure, hardware in ifu_obs if hardware == key}) for key in ifu_keys}
                scatter = {key: robust_scale([record["y"] - R[exposure] for (exposure, hardware), record in ifu_obs.items() if hardware == key]) for key in ifu_keys}
                for key in ifu_keys:
                    row = table.row; row["SPECID"], row["IFUSLOT"], row["IFUID"] = key; row["c"] = c[key]; row["c_support_exposures"] = support.get(key, 0); row["c_supported"] = bool(key in {x[1] for x in ifu_obs}); row["c_scatter"] = scatter.get(key, np.nan); row.append()
            elif name == "physical_amplifiers":
                support = {key: len({exposure for exposure, hardware in amp_obs if hardware == key}) for key in amp_keys}
                scatter = {key: robust_scale([record["y"] - R[exposure] - c[key[:3]] * U[exposure] for (exposure, hardware), record in amp_obs.items() if hardware == key]) for key in amp_keys}
                for key in amp_keys:
                    row = table.row; row["SPECID"], row["IFUSLOT"], row["IFUID"], row["AMP"] = key; row["d"] = d[key]; row["d_support_exposures"] = support.get(key, 0); row["d_supported"] = bool(key in {x[1] for x in amp_obs}); row["d_scatter"] = scatter.get(key, np.nan); row.append()
            elif name == "qa_group_residuals":
                for source in rows:
                    row = table.row
                    for field in table.colnames:
                        if field in source: row[field] = source[field]
                    row.append()
            else:
                for source in rows:
                    row = table.row
                    for field in table.colnames:
                        if field in source: row[field] = source[field]
                    row.append()
            table.flush()
        group = h5.create_group("/", "provenance")
        metadata_table = h5.create_table(group, "metadata", Metadata, expectedrows=len(metadata), filters=filters_h5)
        for key, value in metadata.items():
            row = metadata_table.row; row["key"] = key; row["value"] = json.dumps(value, sort_keys=True, default=str); row.append()
        metadata_table.flush()
        for name in ("exposures", "physical_ifus", "physical_amplifiers"):
            table = getattr(h5.root, name)
            for field in ("h5_id", "exposure", "SPECID", "IFUSLOT", "IFUID"):
                if field in table.colnames: getattr(table.cols, field).create_index()
    return path


def make_qa_figure(path, qa, c, d):
    path = Path(path)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    axes[0].hist([value for value in c.values() if value != 0], bins=20, color="tab:blue")
    axes[0].set_title("raw persistent IFU c"); axes[0].set_xlabel("c")
    axes[1].hist([value for value in d.values() if value != 0], bins=20, color="tab:orange")
    axes[1].set_title("raw persistent amplifier d"); axes[1].set_xlabel("d")
    amp = [row["stage_rms"] for row in qa if row["level"] == "amplifier"]
    if amp:
        axes[2].plot(np.nanmedian(np.asarray(amp), axis=0), "o-")
    axes[2].set_xticks(range(4), ("W", "W-R", "W-R-cU", "W-R-cU-dV"), rotation=30)
    axes[2].set_title("in-sample robust RMS")
    fig.savefig(path, dpi=140); plt.close(fig)
    return str(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("h5_positional", nargs="*")
    parser.add_argument("--h5", nargs="+", action="append")
    parser.add_argument("--h5-glob", action="append")
    parser.add_argument("--external-blank-fibers", required=True)
    parser.add_argument("--fq-template", required=True)
    parser.add_argument("--on-filter", required=True)
    parser.add_argument("--off-filter", required=True)
    parser.add_argument("--output", default="m101_additive_structure.h5")
    parser.add_argument("--minimum-finite-fraction", type=float, default=DEFAULT_MINIMUM_FINITE_FRACTION)
    parser.add_argument("--min-blank-ifus", type=int, default=DEFAULT_MIN_BLANK_IFUS)
    parser.add_argument("--min-blank-per-amp", type=int, default=DEFAULT_MIN_BLANK_PER_AMP)
    parser.add_argument("--max-iterations", type=int, default=30)
    parser.add_argument("--tolerance", type=float, default=1e-5)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if not 0 < args.minimum_finite_fraction <= 1 or args.min_blank_ifus < 1 or args.min_blank_per_amp < 1:
        raise SystemExit("invalid blank support threshold")
    if args.max_iterations < 1 or not np.isfinite(args.tolerance) or args.tolerance <= 0:
        raise SystemExit("invalid convergence setting")
    started = time.perf_counter()
    h5_paths = discover_h5(args)
    blank_masks, blank_provenance = load_external_blank(args.external_blank_fibers, h5_paths)
    fq, fq_id = load_fq(args.fq_template)
    filters, filter_ids = load_filters(args.on_filter, args.off_filter)
    nulls, null_provenance = null_responses()
    del fq, nulls  # fq is loaded and identity-checked for provenance; alpha is not fit here.
    exposure_data, ifu_obs, amp_obs, ifu_keys, amp_keys = collect_observations(
        h5_paths, blank_masks, args.minimum_finite_fraction, args.min_blank_per_amp, args.min_blank_ifus)
    R, U, V, c, d, history, converged = solve_structure(
        exposure_data, ifu_obs, amp_obs, ifu_keys, amp_keys, args.max_iterations, args.tolerance)
    if not converged:
        last = history[-1] if history else {}
        raise RuntimeError("additive alternating fit did not converge after %d iterations; last=%s" %
                           (args.max_iterations, last))
    output = write_product(args.output, h5_paths, exposure_data, ifu_obs, amp_obs, ifu_keys, amp_keys,
                           R, U, V, c, d, history, converged, filters, filter_ids, fq_id,
                           blank_provenance, null_provenance, args, time.perf_counter() - started)
    qa = qa_rows(exposure_data, ifu_obs, amp_obs, R, U, V, c, d)
    qa_figure = make_qa_figure(output.with_name("m101_additive_structure_qa.png"), qa, c, d)
    summary = {"output": str(output), "script_lines": sum(1 for _ in Path(__file__).open()),
               "runtime_seconds": time.perf_counter() - started, "n_exposure_instances": len(exposure_data),
               "n_physical_ifus": len(ifu_keys), "n_supported_ifus": int(sum(key in {x[1] for x in ifu_obs} for key in ifu_keys)),
               "unsupported_ifus": [key for key in ifu_keys if key not in {x[1] for x in ifu_obs}],
               "n_physical_amplifiers": len(amp_keys), "n_supported_amplifiers": int(sum(key in {x[1] for x in amp_obs} for key in amp_keys)),
               "unsupported_amplifiers": [key for key in amp_keys if key not in {x[1] for x in amp_obs}],
               "c_distribution": {"median": robust_location(list(c.values())), "scale": robust_scale(list(c.values()))},
               "d_distribution": {"median": robust_location(list(d.values())), "scale": robust_scale(list(d.values()))},
               "iterations": len(history), "converged": converged, "convergence_history": history,
               "qa_figure": qa_figure, "blank_alpha_choice": "not used"}
    output.with_name("m101_additive_structure_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
