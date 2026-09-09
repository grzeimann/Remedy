#!/usr/bin/env python3
"""Inspect full-spectrum transfer to externally blank M101 IFUs.

This diagnostic consumes native VIRUS spectra, an alpha-only Bayesian solution,
the frozen external blank-fiber classification, and the fitted ``f(q)``
template.  It applies only

    C(lambda) = [Fibers.spectrum / Survey.offset
                 - alpha_mean * raw_work_basis(lambda) * f(q)]
                / exp(posterior_z_mean)

and compares selected blank IFUs with leave-one-IFU-out blank spectra from the
same exposure.  It also characterizes every sufficiently blank physical IFU
and its within-IFU amplifier deviations in the pre-m ``Y`` basis.  The
residual diagnostic removes the common blank reference in the pre-m basis and
then divides each fiber by its own fitted multiplier:

    r_corrected_i(lambda) = (Y_i(lambda) - R_Y,e,-j(lambda)) / m_i

It performs no fitting, cube reconstruction, or use of ``Fibers.skyspectrum``.
"""

import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path
import glob
import time
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tables

import diagnose_m101_hierarchical as hm
from math_utils import biweight


N_FIBER_AMP = 112
N_EXPOSURES = 3
AMPS = ("LL", "LU", "RL", "RU")
DEF_WAVE = np.asarray(hm.DEF_WAVE, dtype=float)
N_WAVE = DEF_WAVE.size
NULL_BANDS = ("NULL_HIGH1", "NULL_LOW1", "NULL_HIGH2", "NULL_LOW2", "NULL_HIGH3")
NULL_WINDOWS = ((3538.0, 3550.0), (3602.0, 3614.0), (3628.0, 3640.0),
                (3676.0, 3688.0), (3902.0, 3914.0))
DEFAULT_MINIMUM_FINITE_FRACTION = 0.8
DEFAULT_IFUS_PER_EXPOSURE = 3


def text(value):
    return value.decode("utf-8", errors="replace").strip() if isinstance(value, (bytes, np.bytes_)) else str(value).strip()


def file_identity(path):
    path = Path(path).resolve()
    stat = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(path), "size": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns),
            "sha256": digest.hexdigest()}


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


def robust_rms(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return float(np.sqrt(np.median(values * values))) if values.size else np.nan


def robust_spectrum(values):
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or values.shape[0] == 0:
        return np.full(values.shape[-1] if values.ndim else 0, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        result = np.asarray(biweight(values, axis=0), dtype=float)
        fallback = np.nanmedian(values, axis=0)
    result[~np.isfinite(result)] = fallback[~np.isfinite(result)]
    return result


def robust_spectrum_with_count(values):
    values = np.asarray(values, dtype=float)
    return robust_spectrum(values), np.sum(np.isfinite(values), axis=0).astype(int)


def per_fiber_window_median(values, indexes):
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or values.shape[0] == 0:
        return np.asarray([], dtype=float)
    selected = values[:, indexes]
    result = np.full(selected.shape[0], np.nan)
    for index in range(selected.shape[0]):
        finite = selected[index][np.isfinite(selected[index])]
        if finite.size:
            result[index] = np.median(finite)
    return result


def write_csv(path, rows, fields=None):
    path = Path(path)
    fields = fields or (list(rows[0]) if rows else [])
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for source in rows:
            clean = {}
            for field in fields:
                value = source.get(field, "")
                if isinstance(value, (float, np.floating)) and not np.isfinite(value):
                    value = ""
                elif isinstance(value, np.generic):
                    value = value.item()
                clean[field] = value
            writer.writerow(clean)


def parse_blank(value):
    value = text(value).lower()
    if value in ("true", "1"):
        return True
    if value in ("false", "0"):
        return False
    raise ValueError("malformed blank value: %r" % value)


def load_external_blank(path, h5_paths):
    """Load every native row using H5 basename + row_index matching."""
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise ValueError("external blank table does not exist: %s" % path)
    by_name, states, labels_by_name = {}, {}, {}
    for h5_path in h5_paths:
        name = h5_path.name
        if name in by_name:
            raise ValueError("ambiguous H5 basename: %s" % name)
        with tables.open_file(h5_path, mode="r") as h5:
            groups, labels = hm.build_groups(h5.root.Info)
            del groups
            by_name[name] = h5_path
            labels_by_name[name] = np.asarray(labels, dtype=int)
            states[name] = {"seen": np.zeros(h5.root.Info.nrows, dtype=bool),
                            "blank": np.zeros(h5.root.Info.nrows, dtype=bool)}
    opener = gzip.open if path.suffix == ".gz" else open
    parsed = matched = 0
    extras, extra_seen = set(), set()
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
            if index < 0 or exposure < 1 or exposure > N_EXPOSURES:
                raise ValueError("invalid row_index/exposure in CSV row %d" % parsed)
            if name not in by_name:
                key = (name, index)
                if key in extra_seen:
                    raise ValueError("duplicate ignored extra %s row %d" % key)
                extra_seen.add(key); extras.add(name)
                continue
            state = states[name]
            if index >= state["seen"].size:
                raise ValueError("row_index %d is out of range for %s" % (index, name))
            if state["seen"][index]:
                raise ValueError("duplicate blank match for %s row %d" % (name, index))
            expected = int(labels_by_name[name][index])
            if exposure != expected:
                raise ValueError("CSV/H5 exposure mismatch for %s row %d: %d != %d" %
                                 (name, index, exposure, expected))
            state["seen"][index] = True; state["blank"][index] = blank; matched += 1
    for name, state in states.items():
        if not np.all(state["seen"]):
            raise ValueError("external blank table is missing %d rows for %s" %
                             (int(np.sum(~state["seen"])), name))
    masks = {name: state["blank"] for name, state in states.items()}
    return masks, {"identity": file_identity(path), "rows": parsed, "matched_rows": matched,
                   "blank_rows": int(sum(np.sum(v) for v in masks.values())),
                   "nonblank_rows": int(sum(v.size - np.sum(v) for v in masks.values())),
                   "h5_files_represented": len(by_name) + len(extras),
                   "ignored_extra_h5_files": sorted(extras),
                   "matching_identity": "H5 basename + row_index; CSV exposure cross-checked against validated labels",
                   "classification_semantics": "externally derived classification; no VIRUS spectral quantity used to define blankness"}


def load_fq(path):
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise ValueError("fq template does not exist: %s" % path)
    rows = list(csv.DictReader(path.open(newline="")))
    if not rows or not {"q", "f"}.issubset(rows[0]):
        raise ValueError("fq template must contain q,f columns")
    result = np.full(N_FIBER_AMP, np.nan)
    for row in rows:
        try:
            q, value = int(text(row["q"])), float(text(row["f"]))
        except (TypeError, ValueError) as exc:
            raise ValueError("malformed fq template row: %r" % row) from exc
        if q < 0 or q >= N_FIBER_AMP or np.isfinite(result[q]) or not np.isfinite(value):
            raise ValueError("invalid or duplicate fq q=%s" % q)
        result[q] = value
    if not np.all(np.isfinite(result)):
        raise ValueError("fq template must contain q=0..111 exactly once")
    return result, file_identity(path)


def load_fit(path):
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise ValueError("fit H5 does not exist: %s" % path)
    metadata = {}
    with tables.open_file(path, mode="r") as h5:
        if "/amplifier_observations" not in h5:
            raise ValueError("fit H5 lacks /amplifier_observations")
        if "/provenance/metadata" not in h5:
            raise ValueError("fit H5 lacks provenance metadata")
        for row in h5.root.provenance.metadata:
            metadata[text(row["key"])] = json.loads(text(row["value"]))
        if metadata.get("additive_model") != "alpha-only" or metadata.get("p_fixed_zero") is not True:
            raise ValueError("fit is not explicitly the alpha-only/p-fixed-zero model")
        table = h5.root.amplifier_observations
        required = {"H5", "exposure", "SPECID", "IFUSLOT", "IFUID", "AMP",
                     "posterior_z_mean", "alpha_mean", "p_mean", "p_sigma"}
        missing = required - set(table.colnames)
        if missing:
            raise ValueError("fit amplifier table lacks %s" % sorted(missing))
        fits = {}
        for row in table:
            key = (Path(text(row["H5"])).name, int(row["exposure"]), int(row["SPECID"]),
                   int(row["IFUSLOT"]), int(row["IFUID"]), text(row["AMP"]))
            if key in fits:
                raise ValueError("duplicate fit amplifier row: %s" % (key,))
            if abs(float(row["p_mean"])) > 0.0 or abs(float(row["p_sigma"])) > 0.0:
                raise ValueError("fit contains nonzero p for %s" % (key,))
            fits[key] = {"z": float(row["posterior_z_mean"]), "alpha": float(row["alpha_mean"])}
    if not fits:
        raise ValueError("fit amplifier table is empty")
    return fits, metadata, file_identity(path)


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


def select_ifus(groups, labels, blank, ra, dec, per_exposure):
    by_key = {}
    for group in groups:
        ifu = (group["specid"], group["ifuslot"], group["ifuid"])
        item = by_key.setdefault((group["exposure"], ifu), {"groups": [], "rows": []})
        item["groups"].append(group); item["rows"].extend(group["indices"].tolist())
    candidates = []
    for (exposure, ifu), item in by_key.items():
        rows = np.asarray(item["rows"], dtype=int)
        candidates.append({"ifu": ifu, "exposure": exposure, "rows": rows,
                           "n_total": int(rows.size), "n_blank": int(np.sum(blank[rows])),
                           "blank_fraction": float(np.mean(blank[rows])),
                           "ra": float(np.nanmedian(ra[rows])), "dec": float(np.nanmedian(dec[rows]))})
    selected = []
    rationale = {}
    for exposure in range(1, N_EXPOSURES + 1):
        pool = sorted([x for x in candidates if x["exposure"] == exposure],
                      key=lambda x: (-x["blank_fraction"], x["ifu"]))
        eligible = [x for x in pool if x["blank_fraction"] >= .8]
        chosen = []
        if pool:
            chosen.append(pool[0])
        if eligible and len(chosen) < per_exposure:
            remaining = [x for x in eligible if x not in chosen]
            while remaining and len(chosen) < per_exposure:
                if len(chosen) == 1:
                    ref = chosen[0]
                    item = max(remaining, key=lambda x: ((x["ra"] - ref["ra"]) ** 2 + (x["dec"] - ref["dec"]) ** 2, x["ifu"]))
                else:
                    item = min(remaining, key=lambda x: (x["blank_fraction"], x["ifu"]))
                chosen.append(item); remaining.remove(item)
        if len(chosen) < per_exposure:
            for item in pool:
                if item not in chosen:
                    chosen.append(item)
                if len(chosen) == per_exposure:
                    break
        if len(chosen) < per_exposure:
            raise ValueError("exposure %d has only %d physical IFUs" % (exposure, len(chosen)))
        rationale[exposure] = {"algorithm": "highest blank fraction; spatially separated eligible IFU; lowest eligible blank fraction; fallback ranked pool",
                               "threshold": .8, "threshold_eligible": len(eligible),
                               "exception": len(eligible) < per_exposure,
                               "selected": [x["ifu"] for x in chosen]}
        selected.extend(chosen[:per_exposure])
    return selected, rationale


def group_lookup(groups, nrows):
    result = np.full(nrows, -1, dtype=int)
    for index, group in enumerate(groups):
        if np.any(result[group["indices"]] >= 0):
            raise ValueError("native group overlap")
        result[group["indices"]] = index
    if np.any(result < 0):
        raise ValueError("native rows missing from amplifier groups")
    return result


def process_h5(path, blank, fit, fq, selected, minimum_finite_fraction,
               population_min_blank_fraction, population_min_blank_fibers,
               population_min_blank_per_amp):
    started = time.perf_counter()
    with tables.open_file(path, mode="r") as h5:
        info, fibers = h5.root.Info, h5.root.Fibers
        groups, labels = hm.build_groups(info)
        if "spectrum" not in fibers.colnames:
            raise ValueError("%s Fibers lacks spectrum" % path)
        ra, dec = np.asarray(info.cols.ra[:], float), np.asarray(info.cols.dec[:], float)
        specid = np.asarray(info.cols.specid[:], dtype=int)
        ifuid = np.asarray(info.cols.ifuid[:], dtype=int)
        ifuslot = np.asarray(info.cols.ifuslot[:])
        amp = np.asarray([text(v) for v in info.cols.amp[:]])
        date_mask_bad = hm.masked_rows(path, ifuslot, amp)
        group_ids = group_lookup(groups, info.nrows)
        surveys = surveys_by_exposure(h5)
        selected_keys = {(x["exposure"], x["ifu"]) for x in selected}
        by_ifu = {}
        for group in groups:
            key = (group["exposure"], (group["specid"], group["ifuslot"], group["ifuid"]))
            item = by_ifu.setdefault(key, {"groups": [], "rows": []})
            item["groups"].append(group)
            item["rows"].extend(group["indices"].tolist())
        population = []
        detailed = []
        for exposure in range(1, N_EXPOSURES + 1):
            exposure_rows = np.flatnonzero(labels == exposure)
            candidates = exposure_rows[blank[exposure_rows] & ~date_mask_bad[exposure_rows]]
            spectra = np.asarray(fibers.read_coordinates(candidates, field="spectrum"), dtype=float)
            offset = float(surveys[exposure]["offset"])
            if not np.isfinite(offset) or offset == 0:
                raise ValueError("%s exposure %d has invalid Survey.offset" % (path, exposure))
            finite_fraction = np.mean(np.isfinite(spectra), axis=1)
            usable = finite_fraction >= minimum_finite_fraction
            candidates = candidates[usable]
            spectra = spectra[usable]
            raw = spectra / offset
            K = hm.raw_work_basis(surveys[exposure])
            alpha_rows = np.full(candidates.size, np.nan)
            multiplier_rows = np.full(candidates.size, np.nan)
            q_rows = np.full(candidates.size, -1, dtype=int)
            for group_index, group in enumerate(groups):
                if group["exposure"] != exposure:
                    continue
                local = np.flatnonzero(group_ids[candidates] == group_index)
                if not local.size:
                    continue
                key = (path.name, exposure, group["specid"], group["ifuslot"], group["ifuid"], group["amp"])
                if key not in fit:
                    raise ValueError("missing amplifier fit row: %s" % (key,))
                q = np.arange(N_FIBER_AMP) if group["amp"] in ("LL", "RU") else np.arange(N_FIBER_AMP - 1, -1, -1)
                position = {int(row): index for index, row in enumerate(group["indices"])}
                j = np.asarray([position[int(row)] for row in candidates[local]], dtype=int)
                alpha_rows[local] = fit[key]["alpha"]
                multiplier_rows[local] = np.exp(fit[key]["z"])
                q_rows[local] = q[j]
            if np.any(~np.isfinite(alpha_rows)) or np.any(~np.isfinite(multiplier_rows)) or np.any(q_rows < 0):
                raise ValueError("missing calibration assignment for blank spectrum rows in %s exposure %d" % (path, exposure))
            y = raw - alpha_rows[:, None] * K[None, :] * fq[q_rows, None]
            for (item_exposure, ifu), item in by_ifu.items():
                if item_exposure != exposure:
                    continue
                native_rows = np.asarray(item["rows"], dtype=int)
                own = np.isin(candidates, native_rows)
                reference = ~own
                if np.any(np.isin(candidates[reference], native_rows)):
                    raise AssertionError("leave-one-IFU-out reference includes the tested IFU")
                n_blank_external = int(np.sum(blank[native_rows]))
                n_blank_usable = int(np.sum(own))
                blank_fraction = float(n_blank_external / native_rows.size) if native_rows.size else np.nan
                ra0 = float(np.nanmedian(ra[native_rows]))
                dec0 = float(np.nanmedian(dec[native_rows]))
                chosen = {"ifu": ifu, "exposure": exposure, "rows": native_rows,
                          "n_total": int(native_rows.size), "n_blank": n_blank_external,
                          "blank_fraction": blank_fraction, "ra": ra0, "dec": dec0}
                if not np.any(own) or not np.any(reference):
                    continue
                R_y, n_r_y = robust_spectrum_with_count(y[reference])
                own_pre_m = y[own] - R_y[None, :]
                F_y, n_f_y = robust_spectrum_with_count(y[own])
                qualifies = blank_fraction >= population_min_blank_fraction and n_blank_usable >= population_min_blank_fibers
                common = {"path": path, "h5": path.name, "exposure": exposure, "ifu": ifu,
                          "chosen": chosen, "F_y": F_y, "R_y": R_y, "own_pre_m": own_pre_m,
                          "n_f_y": n_f_y, "n_r_y": n_r_y, "usable_rows": candidates,
                          "own_mask": own, "reference_mask": reference, "y": y,
                          "raw": raw, "multipliers": multiplier_rows, "q": q_rows,
                          "ra": ra, "dec": dec, "x_arcsec": (ra - ra0) * np.cos(np.deg2rad(dec0)) * 3600.0,
                          "y_arcsec": (dec - dec0) * 3600.0, "specid": specid,
                          "ifuslot": np.asarray(ifuslot), "ifuid": ifuid, "amp": amp, "groups": groups,
                          "K": K, "selected_rows": native_rows, "population_qualifies": qualifies,
                          "n_blank_usable": n_blank_usable, "elapsed": time.perf_counter() - started}
                if qualifies:
                    population.append(common)
                if (exposure, ifu) in selected_keys:
                    R_w, n_r_w = robust_spectrum_with_count(raw[reference])
                    own_raw = raw[own] - R_w[None, :]
                    own_monly = own_raw / multiplier_rows[own, None]
                    own_corrected = own_pre_m / multiplier_rows[own, None]
                    common.update({"R_w": R_w, "delta": robust_spectrum(own_corrected),
                                   "delta_raw": robust_spectrum(own_raw), "delta_monly": robust_spectrum(own_monly),
                                   "n_f": n_f_y, "n_r": n_r_y, "n_r_w": n_r_w,
                                   "own_corrected": own_corrected, "own_monly": own_monly, "own_raw": own_raw,
                                   "multipliers": multiplier_rows, "selected_rows": native_rows})
                    detailed.append(common)
            # A complete fit table is required for every physical amplifier.
            for group in groups:
                key = (path.name, group["exposure"], group["specid"], group["ifuslot"], group["ifuid"], group["amp"])
                if key not in fit:
                    raise ValueError("missing amplifier fit row: %s" % (key,))
    return {"detailed": detailed, "population": population}

def window_indices():
    result = []
    for low, high in NULL_WINDOWS:
        result.append(np.flatnonzero((DEF_WAVE >= low) & (DEF_WAVE <= high)))
    return result


def spectrum_metrics(record):
    delta = record["delta"]; raw = record["delta_raw"]; monly = record["delta_monly"]
    valid = np.isfinite(delta); values = delta[valid]
    raw_values = raw[np.isfinite(raw)]
    monly_values = monly[np.isfinite(monly)]
    robust_rms = float(np.sqrt(np.median(values * values))) if values.size else np.nan
    raw_rms = float(np.sqrt(np.median(raw_values * raw_values))) if raw_values.size else np.nan
    monly_rms = float(np.sqrt(np.median(monly_values * monly_values))) if monly_values.size else np.nan
    summary = {"median_abs_residual_full_spectrum": float(np.median(np.abs(values))) if values.size else np.nan,
               "p90_abs_residual_full_spectrum": float(np.percentile(np.abs(values), 90)) if values.size else np.nan,
               "corrected_robust_RMS_full_spectrum": robust_rms,
               "monly_robust_RMS_full_spectrum": monly_rms,
               "raw_robust_RMS_full_spectrum": raw_rms,
               "calibrated_over_raw_RMS": robust_rms / raw_rms if np.isfinite(raw_rms) and raw_rms > 0 else np.nan,
               "raw_over_calibrated_RMS": raw_rms / robust_rms if np.isfinite(robust_rms) and robust_rms > 0 else np.nan,
               "improvement_factor": raw_rms / robust_rms if np.isfinite(robust_rms) and robust_rms > 0 else np.nan,
               "correlation_with_leave_one_out_sky": np.nan}
    corr = valid & np.isfinite(record["R_y"])
    if np.sum(corr) >= 3 and np.std(delta[corr]) > 0 and np.std(record["R_y"][corr]) > 0:
        summary["correlation_with_leave_one_out_sky"] = float(np.corrcoef(delta[corr], record["R_y"][corr])[0, 1])
    for low, high in ((3470., 3800.), (3800., 4500.), (4500., 5540.)):
        use = valid & (DEF_WAVE >= low) & (DEF_WAVE <= high)
        summary["corrected_robust_RMS_%d_%d" % (low, high)] = float(np.sqrt(np.median(delta[use] ** 2))) if np.any(use) else np.nan
    for name, indexes in zip(NULL_BANDS, window_indices()):
        use = np.isfinite(delta[indexes])
        summary["residual_%s" % name] = robust_location(delta[indexes][use]) if np.any(use) else np.nan
    null_values = np.asarray([summary["residual_%s" % name] for name in NULL_BANDS])
    summary["robust_scatter_five_null_bands"] = robust_scale(null_values)
    if values.size:
        index = int(np.nanargmax(np.abs(delta)))
        summary["maximum_absolute_residual_wavelength"] = float(DEF_WAVE[index])
        summary["maximum_absolute_residual_value"] = float(delta[index])
    else:
        summary["maximum_absolute_residual_wavelength"] = np.nan; summary["maximum_absolute_residual_value"] = np.nan
    return summary


def residual_rows(record):
    chosen = record["chosen"]; rows = []
    own = np.array([(record["specid"][row], int(record["ifuslot"][row]), record["ifuid"][row]) == chosen["ifu"]
                    for row in record["usable_rows"]])
    reference = ~own
    for index, wavelength in enumerate(DEF_WAVE):
        rows.append({"H5": record["h5"], "exposure": record["exposure"], "SPECID": chosen["ifu"][0],
                     "IFUSLOT": chosen["ifu"][1], "IFUID": chosen["ifu"][2], "wavelength": float(wavelength),
                     "N_blank_IFU_finite": int(record["n_f"][index]), "N_blank_reference_finite": int(record["n_r"][index]),
                     "IFU_pre_m_Y": record["F_y"][index],
                     "leave_one_IFU_out_pre_m_Y_sky": record["R_y"][index],
                     "residual": record["delta"][index],
                     "corrected_residual": record["delta"][index],
                     "monly_residual": record["delta_monly"][index],
                     "raw_residual": record["delta_raw"][index],
                     "robust_scatter_IFU": robust_scale(record["own_corrected"][:, index]),
                     "robust_scatter_reference": robust_scale(record["y"][reference, index])})
    return rows


def amp_rows(record):
    chosen = record["chosen"]; rows = []
    own = np.array([(record["specid"][row], int(record["ifuslot"][row]), record["ifuid"][row]) == chosen["ifu"]
                    for row in record["usable_rows"]])
    own_amp = record["amp"][record["usable_rows"]][own]
    for name, indexes in zip(NULL_BANDS, window_indices()):
        for amp in AMPS:
            mask = own_amp == amp
            vals = record["own_corrected"][mask][:, indexes] if np.any(mask) else np.empty((0, indexes.size))
            per_fiber = per_fiber_window_median(vals, np.arange(vals.shape[1])) if vals.size else np.asarray([])
            rows.append({"H5": record["h5"], "exposure": record["exposure"], "SPECID": chosen["ifu"][0],
                         "IFUSLOT": chosen["ifu"][1], "IFUID": chosen["ifu"][2], "AMP": amp,
                         "null_band": name, "N_blank": int(np.sum(np.isfinite(per_fiber))),
                         "robust_location": robust_location(per_fiber), "robust_scale": robust_scale(per_fiber)})
    return rows


def plot_record(record, output_dir):
    chosen = record["chosen"]
    label = "%s_exp%d_SPECID%d_SLOT%d_UID%d" % (record["h5"].replace(".h5", ""), record["exposure"], *chosen["ifu"])
    safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in label)
    own = np.array([(record["specid"][row], int(record["ifuslot"][row]), record["ifuid"][row]) == chosen["ifu"]
                    for row in record["usable_rows"]])
    own_rows = record["usable_rows"][own]
    amp = record["amp"][own_rows]
    colors = dict(zip(AMPS, ("tab:blue", "tab:orange", "tab:green", "tab:red")))
    all_residuals = np.concatenate([x[np.isfinite(x)] for x in (record["delta"], record["delta_raw"], record["delta_monly"])
                                    if np.any(np.isfinite(x))])
    spectral_limit = max(float(np.percentile(np.abs(all_residuals), 99) * 1.25), 1e-8) if all_residuals.size else 1.0
    high2 = window_indices()[2]
    representative = per_fiber_window_median(record["own_corrected"], high2)
    spatial_limit = max(float(np.percentile(np.abs(representative[np.isfinite(representative)]), 99) * 1.15), 1e-8) if np.any(np.isfinite(representative)) else 1.0

    fig = plt.figure(figsize=(12, 11), constrained_layout=True)
    grid = fig.add_gridspec(3, 1)
    ax_spectrum = fig.add_subplot(grid[0])
    ax_residual = fig.add_subplot(grid[1], sharex=ax_spectrum)
    ax_spatial = fig.add_subplot(grid[2])
    if ax_spatial.get_shared_x_axes().joined(ax_spatial, ax_spectrum) or ax_spatial.get_shared_x_axes().joined(ax_spatial, ax_residual):
        raise AssertionError("primary spatial axis unexpectedly shares its wavelength x-axis")
    ax_spectrum.plot(DEF_WAVE, record["R_y"], label="pre-m leave-one-IFU-out R_Y", lw=.9)
    ax_spectrum.plot(DEF_WAVE, record["F_y"], label="pre-m selected IFU Y", lw=.8, alpha=.8)
    ax_spectrum.set_ylabel("Y (working flux)")
    ax_spectrum.legend(fontsize=8)
    ax_residual.plot(DEF_WAVE, record["delta_raw"], label="raw: W - R_W", lw=.65, alpha=.55)
    ax_residual.plot(DEF_WAVE, record["delta_monly"], label="multiplication only", lw=.7, alpha=.75)
    ax_residual.plot(DEF_WAVE, record["delta"], label="corrected: (Y - R_Y) / m", lw=.8)
    ax_residual.axhline(0, color="k", lw=.7)
    ax_residual.set_ylim(-spectral_limit, spectral_limit)
    ax_residual.set_ylabel("residual")
    ax_residual.legend(fontsize=8)
    for low, high in NULL_WINDOWS:
        ax_residual.axvspan(low, high, color="tab:orange", alpha=.18)
    for amp_name in AMPS:
        mask = amp == amp_name
        if np.any(mask):
            ax_spatial.scatter(record["x_arcsec"][own_rows][mask], record["y_arcsec"][own_rows][mask],
                               c=representative[mask], cmap="coolwarm", vmin=-spatial_limit, vmax=spatial_limit,
                               s=24, edgecolors=colors[amp_name], linewidths=.6, label=amp_name)
    ax_spatial.set_xlabel("Delta RA cos(Dec) [arcsec]")
    ax_spatial.set_ylabel("Delta Dec [arcsec]")
    ax_spatial.set_title("corrected NULL_HIGH2 residual; nonblank fibers omitted; AMP edge colors")
    ax_spatial.set_aspect("equal", adjustable="box")
    ax_spatial.legend(fontsize=7, ncol=4)
    ax_spectrum.set_title("%s; blank fraction %.3f; N_blank=%d" % (label, chosen["blank_fraction"], chosen["n_blank"]))
    ax_residual.set_xlim(3470, 5540)
    ax_residual.set_xlabel("wavelength [A]")
    for axis in (ax_spectrum, ax_residual, ax_spatial):
        axis.grid(alpha=.18)
    path = output_dir / (safe + "_diagnostic.png")
    fig.savefig(path, dpi=140)
    plt.close(fig)

    band_values = [per_fiber_window_median(record["own_corrected"], indexes) for indexes in window_indices()]
    finite_band_values = np.concatenate([x[np.isfinite(x)] for x in band_values if np.any(np.isfinite(x))]) if any(np.any(np.isfinite(x)) for x in band_values) else np.asarray([1.])
    common_scale = max(float(np.percentile(np.abs(finite_band_values), 99)), 1e-8)
    finite_x = record["x_arcsec"][own_rows][np.isfinite(record["x_arcsec"][own_rows])]
    finite_y = record["y_arcsec"][own_rows][np.isfinite(record["y_arcsec"][own_rows])]
    xlim = _plot_limits(finite_x)
    ylim = _plot_limits(finite_y)
    fig, axes = plt.subplots(1, 5, figsize=(17, 4), sharex=True, sharey=True, constrained_layout=True)
    for axis, name, values in zip(axes, NULL_BANDS, band_values):
        axis.scatter(record["x_arcsec"][own_rows], record["y_arcsec"][own_rows], c=values, cmap="coolwarm",
                     vmin=-common_scale, vmax=common_scale, s=18)
        axis.set_title(name)
        axis.set_xlabel("Delta RA cos(Dec) [arcsec]")
        axis.set_xlim(*xlim)
        axis.set_ylim(*ylim)
        axis.set_aspect("equal", adjustable="box")
        axis.grid(alpha=.15)
    axes[0].set_ylabel("Delta Dec [arcsec]")
    fig.suptitle("%s: corrected five null-band spatial residuals; common color limit %.4g" % (label, common_scale))
    path2 = output_dir / (safe + "_null_spatial.png")
    fig.savefig(path2, dpi=140)
    plt.close(fig)

    q_values = band_values
    q_finite = np.concatenate([x[np.isfinite(x)] for x in q_values if np.any(np.isfinite(x))]) if any(np.any(np.isfinite(x)) for x in q_values) else np.asarray([1.])
    q_limit = max(float(np.percentile(np.abs(q_finite), 99) * 1.15), 1e-8)
    fig, axes = plt.subplots(1, 5, figsize=(17, 4), sharex=True, sharey=True, constrained_layout=True)
    for axis, name, values in zip(axes, NULL_BANDS, q_values):
        for amp_name in AMPS:
            mask = amp == amp_name
            if np.any(mask):
                axis.scatter(record["q"][own][mask], values[mask], s=7, alpha=.55, label=amp_name)
        axis.axhline(0, color="k", lw=.5)
        axis.set_title(name)
        axis.set_xlabel("q")
        axis.set_xlim(-2, 113)
        axis.set_ylim(-q_limit, q_limit)
        axis.grid(alpha=.15)
    axes[0].set_ylabel("corrected residual")
    axes[0].legend(fontsize=7)
    path3 = output_dir / (safe + "_residual_vs_q.png")
    fig.savefig(path3, dpi=140)
    plt.close(fig)
    return [str(path), str(path2), str(path3)], {"label": label, "five_band_common_symmetric_limit": common_scale,
                                                "representative_spatial_symmetric_limit": spatial_limit,
                                                "spatial_xlim_arcsec": list(xlim), "spatial_ylim_arcsec": list(ylim),
                                                "spectral_xlim_A": [3470.0, 5540.0], "q_xlim": [-2.0, 113.0],
                                                "q_symmetric_limit": q_limit}


def _plot_limits(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return (-1.0, 1.0)
    low, high = float(np.min(values)), float(np.max(values))
    span = max(high - low, 1.0)
    pad = .08 * span
    return low - pad, high + pad

def plot_overview(output_dir, records):
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), constrained_layout=True)
    x = np.arange(len(records)); labels = ["%s/e%d/%s-%d-%d" % (r["h5"], r["exposure"], r["ifu"][0], r["ifu"][1], r["ifu"][2]) for r in records]
    axes[0].bar(x, [spectrum_metrics(r)["median_abs_residual_full_spectrum"] for r in records]); axes[0].set_ylabel("median |residual|"); axes[0].set_xticks(x, labels, rotation=75, ha="right")
    for j, name in enumerate(NULL_BANDS): axes[1].plot(x, [spectrum_metrics(r)["residual_%s" % name] for r in records], "o-", label=name)
    axes[1].axhline(0, color="k", lw=.7); axes[1].set_ylabel("null residual"); axes[1].set_xticks(x, labels, rotation=75, ha="right"); axes[1].legend(fontsize=7, ncol=3)
    path = output_dir / "blank_ifu_transfer_overview.png"; fig.savefig(path, dpi=140); plt.close(fig); return str(path)


def ordering_sanity():
    """Verify that pre-m subtraction removes a common Y exactly."""
    reference = np.asarray([1.0, -0.5, 2.0, 4.0])
    multipliers = np.asarray([0.5, 2.0, 4.0])
    y = np.repeat(reference[None, :], multipliers.size, axis=0)
    common = robust_spectrum(y)
    correct = (y - common[None, :]) / multipliers[:, None]
    old_reference = robust_spectrum(y / multipliers[:, None])
    old = y / multipliers[:, None] - old_reference[None, :]
    return {"correct_order_residual_max_abs": float(np.nanmax(np.abs(correct))),
            "old_order_residual_max_abs": float(np.nanmax(np.abs(old))),
            "different_amplifier_m": multipliers.tolist()}


def aggregate_summary(summary_rows):
    def median(rows, key):
        values = np.asarray([row[key] for row in rows], dtype=float)
        return float(np.nanmedian(values)) if np.any(np.isfinite(values)) else np.nan
    result = {}
    for label, rows in [("all", summary_rows)] + [(name, [row for row in summary_rows if row["H5"] == name])
                                                  for name in sorted({row["H5"] for row in summary_rows})]:
        result[label] = {"n_ifus": len(rows),
                         "median_raw_robust_RMS": median(rows, "raw_robust_RMS_full_spectrum"),
                         "median_corrected_robust_RMS": median(rows, "corrected_robust_RMS_full_spectrum"),
                         "median_calibrated_over_raw_RMS": median(rows, "calibrated_over_raw_RMS"),
                         "median_raw_over_calibrated_RMS": median(rows, "raw_over_calibrated_RMS"),
                         "n_improved_corrected_over_raw_lt_1": int(sum(row["calibrated_over_raw_RMS"] < 1 for row in rows)),
                         "n_improved_raw_over_corrected_gt_1": int(sum(row["raw_over_calibrated_RMS"] > 1 for row in rows))}
    return result


def population_band_values(record):
    """Return IFU-band coefficients and a direct fiber/band cross-check."""
    values = []
    direct = []
    for indexes in window_indices():
        full_spectrum_value = robust_location(record["pop_delta"][indexes])
        fiber_values = per_fiber_window_median(record["own_pre_m"], indexes)
        direct_value = robust_location(fiber_values)
        values.append(full_spectrum_value)
        direct.append(direct_value)
    return np.asarray(values, dtype=float), np.asarray(direct, dtype=float)


def population_amp_values(record, minimum_blank_per_amp):
    own_rows = record["usable_rows"][record["own_mask"]]
    own_amp = record["amp"][own_rows]
    coefficients = {}
    for amp_name in AMPS:
        mask = own_amp == amp_name
        n_blank = int(np.sum(mask))
        values = []
        if n_blank >= minimum_blank_per_amp:
            for indexes in window_indices():
                per_fiber = per_fiber_window_median(record["own_pre_m"][mask], indexes)
                values.append(robust_location(per_fiber))
        else:
            values = [np.nan] * len(NULL_BANDS)
        coefficients[amp_name] = {"N_blank_usable": n_blank, "values": np.asarray(values, dtype=float)}
    return coefficients


def population_products(population, minimum_blank_per_amp):
    band_rows = []
    amp_rows_out = []
    decomposition = []
    orientation = []
    correlation_rows = []
    matrices = {}
    direct_checks = []
    by_exposure = {}
    for record in population:
        record["pop_delta"] = robust_spectrum(record["own_pre_m"])
        record["pop_scale"] = np.asarray([robust_scale(record["own_pre_m"][:, i]) for i in range(DEF_WAVE.size)])
        coeff, direct = population_band_values(record)
        record["pop_coeff"] = coeff
        record["pop_direct_coeff"] = direct
        direct_checks.append((record["h5"], record["exposure"], record["ifu"], coeff, direct))
        chosen = record["chosen"]
        band_rows.append({"H5": record["h5"], "exposure": record["exposure"], "SPECID": chosen["ifu"][0],
                          "IFUSLOT": chosen["ifu"][1], "IFUID": chosen["ifu"][2], "RA_center": chosen["ra"],
                          "Dec_center": chosen["dec"], "N_total": chosen["n_total"],
                          "N_blank_external": chosen["n_blank"], "N_blank_usable": record["n_blank_usable"],
                          "blank_fraction": chosen["blank_fraction"],
                          **{"residual_%s" % name: coeff[i] for i, name in enumerate(NULL_BANDS)}})
        amp_coeff = population_amp_values(record, minimum_blank_per_amp)
        record["amp_coeff"] = amp_coeff
        c_ifu = np.full(len(NULL_BANDS), np.nan)
        for band_index, name in enumerate(NULL_BANDS):
            valid_amp = np.asarray([amp_coeff[a]["values"][band_index] for a in AMPS], dtype=float)
            valid_amp = valid_amp[np.isfinite(valid_amp)]
            c_ifu[band_index] = robust_location(valid_amp)
            for amp_name in AMPS:
                amp_value = amp_coeff[amp_name]["values"][band_index]
                deviation = amp_value - c_ifu[band_index] if np.isfinite(amp_value) and np.isfinite(c_ifu[band_index]) else np.nan
                amp_rows_out.append({"H5": record["h5"], "exposure": record["exposure"], "SPECID": chosen["ifu"][0],
                                     "IFUSLOT": chosen["ifu"][1], "IFUID": chosen["ifu"][2], "AMP": amp_name,
                                     "N_blank_usable": amp_coeff[amp_name]["N_blank_usable"], "null_band": name,
                                     "residual": amp_value, "IFU_common_level": c_ifu[band_index],
                                     "within_IFU_deviation": deviation})
                if np.isfinite(deviation):
                    orientation.append({"H5": record["h5"], "exposure": record["exposure"], "AMP": amp_name,
                                        "null_band": name, "within_IFU_deviation": deviation})
        record["c_ifu"] = c_ifu
        by_exposure.setdefault((record["h5"], record["exposure"]), []).append(record)
    for (h5_name, exposure), records in by_exposure.items():
        for band_index, name in enumerate(NULL_BANDS):
            c_values = np.asarray([r["c_ifu"][band_index] for r in records], dtype=float)
            deviations = np.asarray([row["within_IFU_deviation"] for row in amp_rows_out
                                     if row["H5"] == h5_name and row["exposure"] == exposure and row["null_band"] == name], dtype=float)
            c_values = c_values[np.isfinite(c_values)]
            deviations = deviations[np.isfinite(deviations)]
            sigma_ifu = robust_scale(c_values)
            sigma_amp = robust_scale(deviations)
            decomposition.append({"H5": h5_name, "exposure": exposure, "null_band": name,
                                  "N_IFU": int(c_values.size), "N_AMP": int(deviations.size),
                                  "median_IFU_level": robust_location(c_values), "sigma_IFU": sigma_ifu,
                                  "median_AMP_deviation": robust_location(deviations), "sigma_AMP_within_IFU": sigma_amp,
                                  "sigma_IFU_over_sigma_AMP": sigma_ifu / sigma_amp if np.isfinite(sigma_ifu) and np.isfinite(sigma_amp) and sigma_amp > 0 else np.nan})
        matrix = np.full((len(NULL_BANDS), len(NULL_BANDS)), np.nan)
        counts = np.zeros_like(matrix, dtype=int)
        values = np.asarray([r["pop_coeff"] for r in records], dtype=float)
        for i, row_name in enumerate(NULL_BANDS):
            for j, col_name in enumerate(NULL_BANDS):
                use = np.isfinite(values[:, i]) & np.isfinite(values[:, j])
                counts[i, j] = int(np.sum(use))
                if counts[i, j] >= 3 and np.std(values[use, i]) > 0 and np.std(values[use, j]) > 0:
                    matrix[i, j] = float(np.corrcoef(values[use, i], values[use, j])[0, 1])
                correlation_rows.append({"H5": h5_name, "exposure": exposure, "row_band": row_name,
                                         "column_band": col_name, "correlation": matrix[i, j], "N_IFU": counts[i, j]})
        matrices["%s|%d" % (h5_name, exposure)] = {"bands": list(NULL_BANDS), "correlation": matrix.tolist(), "N_IFU": counts.tolist()}
    orientation_summary = []
    for h5_name in sorted({row["H5"] for row in orientation}):
        for exposure in sorted({row["exposure"] for row in orientation if row["H5"] == h5_name}):
            for name in NULL_BANDS:
                for amp_name in AMPS:
                    vals = np.asarray([row["within_IFU_deviation"] for row in orientation
                                       if row["H5"] == h5_name and row["exposure"] == exposure and row["null_band"] == name and row["AMP"] == amp_name], dtype=float)
                    orientation_summary.append({"H5": h5_name, "exposure": exposure, "AMP": amp_name,
                                                "null_band": name, "median_d_AMP": robust_location(vals),
                                                "scale_d_AMP": robust_scale(vals), "N": int(np.sum(np.isfinite(vals)))})
    return {"band_rows": band_rows, "amp_rows": amp_rows_out, "decomposition": decomposition,
            "orientation_summary": orientation_summary, "correlation_rows": correlation_rows,
            "correlation_matrices": matrices, "direct_checks": direct_checks}


def population_spectrum_rows(record):
    chosen = record["chosen"]
    return [{"H5": record["h5"], "exposure": record["exposure"], "SPECID": chosen["ifu"][0],
             "IFUSLOT": chosen["ifu"][1], "IFUID": chosen["ifu"][2], "wavelength": float(wave),
             "N_blank_finite": int(record["n_f_y"][i]), "IFU_residual": record["pop_delta"][i],
             "IFU_residual_scale": record["pop_scale"][i]} for i, wave in enumerate(DEF_WAVE)]


def write_csv_gz(path, rows, fields=None):
    path = Path(path)
    fields = fields or (list(rows[0]) if rows else [])
    with gzip.open(path, "wt", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for source in rows:
            clean = {}
            for field in fields:
                value = source.get(field, "")
                if isinstance(value, (float, np.floating)) and not np.isfinite(value):
                    value = ""
                elif isinstance(value, np.generic):
                    value = value.item()
                clean[field] = value
            writer.writerow(clean)


def population_figures(output_dir, population, products):
    paths = []
    metadata = {"exposure": {}, "summary": {}}
    keys = sorted({(r["h5"], r["exposure"]) for r in population})
    for key in keys:
        h5_name, exposure = key
        records = [r for r in population if (r["h5"], r["exposure"]) == key]
        safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in "%s_exp%d" % (h5_name.replace(".h5", ""), exposure))
        spectra = np.asarray([r["pop_delta"] for r in records], dtype=float)
        median = np.nanmedian(spectra, axis=0)
        p16 = np.nanpercentile(spectra, 16, axis=0)
        p84 = np.nanpercentile(spectra, 84, axis=0)
        finite = spectra[np.isfinite(spectra)]
        limit = max(float(np.percentile(np.abs(finite), 99) * 1.15), 1e-8) if finite.size else 1.0
        fig, axis = plt.subplots(figsize=(12, 5), constrained_layout=True)
        for spectrum in spectra:
            axis.plot(DEF_WAVE, spectrum, lw=.45, alpha=.30, color="tab:blue")
        axis.plot(DEF_WAVE, median, color="k", lw=1.4, label="population median")
        axis.plot(DEF_WAVE, p16, color="k", lw=.8, ls="--", label="p16/p84")
        axis.plot(DEF_WAVE, p84, color="k", lw=.8, ls="--")
        for low, high in NULL_WINDOWS:
            axis.axvspan(low, high, color="tab:orange", alpha=.15)
        axis.set_xlim(3470, 5540); axis.set_ylim(-limit, limit); axis.set_xlabel("wavelength [A]")
        axis.set_ylabel("pre-m IFU residual")
        axis.set_title("%s exposure %d: %d qualifying blank IFUs" % (h5_name, exposure, len(records)))
        axis.legend(fontsize=8); axis.grid(alpha=.18)
        path = output_dir / ("population_ifu_spectra_%s.png" % safe); fig.savefig(path, dpi=140); plt.close(fig); paths.append(str(path))
        coeff = np.asarray([r["pop_coeff"] for r in records], dtype=float)
        fig, axis = plt.subplots(figsize=(10, 5), constrained_layout=True)
        x = np.arange(len(NULL_BANDS))
        for row in coeff:
            axis.plot(x, row, color="tab:blue", alpha=.28, lw=.7)
            axis.scatter(x, row, color="tab:blue", s=9, alpha=.45)
        med = np.nanmedian(coeff, axis=0)
        lo = np.nanpercentile(coeff, 16, axis=0)
        hi = np.nanpercentile(coeff, 84, axis=0)
        axis.errorbar(x, med, yerr=np.vstack((med - lo, hi - med)), fmt="ko-", lw=1.5, capsize=3, label="median with p16/p84")
        axis.axhline(0, color="k", lw=.7); axis.set_xticks(x, [n.replace("NULL_", "") for n in NULL_BANDS]); axis.set_ylabel("pre-m IFU coefficient"); axis.set_title("%s exposure %d: connected IFU null coefficients" % (h5_name, exposure)); axis.grid(alpha=.18); axis.legend(fontsize=8)
        path = output_dir / ("population_ifu_bands_%s.png" % safe); fig.savefig(path, dpi=140); plt.close(fig); paths.append(str(path))
        matrix_key = "%s|%d" % key
        matrix = np.asarray(products["correlation_matrices"][matrix_key]["correlation"], dtype=float)
        fig, axis = plt.subplots(figsize=(5, 4.5), constrained_layout=True)
        image = axis.imshow(matrix, vmin=-1, vmax=1, cmap="coolwarm")
        axis.set_xticks(x, [n.replace("NULL_", "") for n in NULL_BANDS], rotation=45, ha="right"); axis.set_yticks(x, [n.replace("NULL_", "") for n in NULL_BANDS]); axis.set_title("%s exposure %d IFU correlations" % (h5_name, exposure))
        for i in range(len(NULL_BANDS)):
            for j in range(len(NULL_BANDS)):
                if np.isfinite(matrix[i, j]): axis.text(j, i, "%.2f" % matrix[i, j], ha="center", va="center", fontsize=8)
        fig.colorbar(image, ax=axis, label="Pearson r"); path = output_dir / ("population_ifu_correlations_%s.png" % safe); fig.savefig(path, dpi=140); plt.close(fig); paths.append(str(path))
        centers_ra = np.asarray([r["chosen"]["ra"] for r in records]); centers_dec = np.asarray([r["chosen"]["dec"] for r in records])
        ra0, dec0 = float(np.nanmedian(centers_ra)), float(np.nanmedian(centers_dec))
        xs = (centers_ra - ra0) * np.cos(np.deg2rad(dec0)) * 3600.; ys = (centers_dec - dec0) * 3600.
        finite_coeff = coeff[np.isfinite(coeff)]
        spatial_scale = max(float(np.percentile(np.abs(finite_coeff), 99)), 1e-8) if finite_coeff.size else 1.0
        fig, axes = plt.subplots(1, 5, figsize=(17, 4), sharex=True, sharey=True, constrained_layout=True)
        for axis, name, vals in zip(axes, NULL_BANDS, coeff.T):
            axis.scatter(xs, ys, c=vals, cmap="coolwarm", vmin=-spatial_scale, vmax=spatial_scale, s=35, edgecolors="k", linewidths=.25)
            axis.set_title(name); axis.set_xlabel("Delta RA cos(Dec) [arcsec]"); axis.set_aspect("equal", adjustable="box"); axis.grid(alpha=.15)
        axes[0].set_ylabel("Delta Dec [arcsec]"); fig.suptitle("%s exposure %d: IFU coefficient focal-plane distribution; color limit %.4g" % (h5_name, exposure, spatial_scale)); path = output_dir / ("population_ifu_spatial_%s.png" % safe); fig.savefig(path, dpi=140); plt.close(fig); paths.append(str(path))
        metadata["exposure"][matrix_key] = {"n_ifu": len(records), "spatial_common_symmetric_limit": spatial_scale, "spatial_center": [ra0, dec0], "spectral_limit": limit}
    for key in sorted({(r["H5"], r["exposure"]) for r in products["amp_rows"]}):
        h5_name, exposure = key
        safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in "%s_exp%d" % (h5_name.replace(".h5", ""), exposure))
        rows = [r for r in products["amp_rows"] if (r["H5"], r["exposure"]) == key]
        finite_amp = np.asarray([r["within_IFU_deviation"] for r in rows], dtype=float)
        finite_amp = finite_amp[np.isfinite(finite_amp)]
        amp_limit = max(float(np.percentile(np.abs(finite_amp), 99) * 1.15), 1e-8) if finite_amp.size else 1.0
        fig, axis = plt.subplots(figsize=(10, 5), constrained_layout=True)
        cmap = dict(zip(AMPS, ("tab:blue", "tab:orange", "tab:green", "tab:red")))
        for amp_name in AMPS:
            grouped = {}
            for row in rows:
                if row["AMP"] == amp_name and np.isfinite(row["within_IFU_deviation"]):
                    grouped.setdefault((row["SPECID"], row["IFUSLOT"], row["IFUID"]), {})[row["null_band"]] = row["within_IFU_deviation"]
            for values_by_band in grouped.values():
                ordered = np.asarray([values_by_band.get(name, np.nan) for name in NULL_BANDS], dtype=float)
                axis.plot(np.arange(len(NULL_BANDS)), ordered, color=cmap[amp_name], alpha=.25, lw=.7)
                axis.scatter(np.arange(len(NULL_BANDS)), ordered, color=cmap[amp_name], s=8, alpha=.45, label=amp_name)
        handles, labels = axis.get_legend_handles_labels()
        unique = dict(zip(labels, handles))
        axis.axhline(0, color="k", lw=.7); axis.set_ylim(-amp_limit, amp_limit)
        axis.set_xticks(np.arange(len(NULL_BANDS)), [n.replace("NULL_", "") for n in NULL_BANDS])
        axis.set_ylabel("within-IFU AMP deviation"); axis.set_title("%s exposure %d: connected amplifier deviations" % (h5_name, exposure))
        axis.legend(unique.values(), unique.keys(), fontsize=8); axis.grid(alpha=.18)
        path = output_dir / ("population_amp_connected_%s.png" % safe); fig.savefig(path, dpi=140); plt.close(fig); paths.append(str(path))
        fig, axis = plt.subplots(figsize=(10, 5), constrained_layout=True)
        x = np.arange(len(NULL_BANDS)); width = .2
        for index, amp_name in enumerate(AMPS):
            values = np.asarray([r["median_d_AMP"] for r in products["orientation_summary"] if r["H5"] == h5_name and r["exposure"] == exposure and r["AMP"] == amp_name], dtype=float)
            axis.scatter(x + (index - 1.5) * width, values, label=amp_name, color=cmap[amp_name])
        axis.axhline(0, color="k", lw=.7); axis.set_xticks(x, [n.replace("NULL_", "") for n in NULL_BANDS]); axis.set_ylabel("median within-IFU AMP deviation"); axis.set_title("%s exposure %d: AMP orientation summaries" % (h5_name, exposure)); axis.legend(fontsize=8); axis.grid(alpha=.18)
        path = output_dir / ("population_amp_orientation_%s.png" % safe); fig.savefig(path, dpi=140); plt.close(fig); paths.append(str(path))
    fig, axis = plt.subplots(figsize=(12, 6), constrained_layout=True)
    x = np.arange(len(NULL_BANDS)); offset = np.linspace(-.18, .18, 6)
    keys = sorted({(r["H5"], r["exposure"]) for r in products["decomposition"]})
    for index, key in enumerate(keys):
        rows = [r for r in products["decomposition"] if (r["H5"], r["exposure"]) == key]
        axis.plot(x + offset[index], [r["sigma_IFU"] for r in rows], "o-", label="%s e%d IFU" % (key[0].replace(".h5", ""), key[1]))
        axis.plot(x + offset[index], [r["sigma_AMP_within_IFU"] for r in rows], "s--", alpha=.65)
    axis.set_xticks(x, [n.replace("NULL_", "") for n in NULL_BANDS]); axis.set_ylabel("robust scale"); axis.set_title("IFU versus within-IFU amplifier residual scale"); axis.legend(fontsize=7, ncol=2); axis.grid(alpha=.18)
    path = output_dir / "population_spatial_scale_comparison.png"; fig.savefig(path, dpi=140); plt.close(fig); paths.append(str(path))
    return paths, metadata

def exposure_instance_id(h5_name, exposure):
    return "%s_e%d" % (h5_name.replace(".h5", ""), int(exposure))


def physical_ifu_key(record):
    return tuple(int(x) for x in record["ifu"])


def physical_amp_key(row):
    return (int(row["SPECID"]), int(row["IFUSLOT"]), int(row["IFUID"]), text(row["AMP"]))


def rank_values(values):
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=float)
    sorted_values = values[order]
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = .5 * (start + end - 1)
        start = end
    return ranks


def pearson_r(x, y):
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    use = np.isfinite(x) & np.isfinite(y)
    if np.sum(use) < 3 or np.std(x[use]) == 0 or np.std(y[use]) == 0:
        return np.nan
    return float(np.corrcoef(x[use], y[use])[0, 1])


def spearman_r(x, y):
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    use = np.isfinite(x) & np.isfinite(y)
    return pearson_r(rank_values(x[use]), rank_values(y[use])) if np.sum(use) >= 3 else np.nan


def robust_linear_stats(x, y):
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    use = np.isfinite(x) & np.isfinite(y)
    x = x[use]; y = y[use]
    if x.size < 2:
        return np.nan, np.nan, np.nan
    slopes = []
    for i in range(x.size - 1):
        dx = x[i + 1:] - x[i]
        good = dx != 0
        if np.any(good):
            slopes.extend(((y[i + 1:][good] - y[i]) / dx[good]).tolist())
    slope = robust_location(np.asarray(slopes, dtype=float)) if slopes else np.nan
    intercept = robust_location(y - slope * x) if np.isfinite(slope) else np.nan
    ratios = y[np.abs(x) > 1e-12] / x[np.abs(x) > 1e-12]
    slope_zero = robust_location(ratios) if ratios.size else np.nan
    return slope, intercept, slope_zero


def pair_statistics(x, y):
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    use = np.isfinite(x) & np.isfinite(y)
    x = x[use]; y = y[use]
    slope, intercept, slope_zero = robust_linear_stats(x, y)
    return {"N_matched": int(x.size), "Pearson_r": pearson_r(x, y), "Spearman_r": spearman_r(x, y),
            "robust_slope": slope, "robust_intercept": intercept, "slope_through_zero": slope_zero,
            "normalized_RMS_difference": float(np.sqrt(np.median((y - x) ** 2))) if x.size else np.nan}


def build_persistence_inputs(population, products):
    ifu = {}
    centers = {}
    seen_amp = set()
    for record in population:
        instance = exposure_instance_id(record["h5"], record["exposure"])
        key = physical_ifu_key(record)
        if instance in ifu and key in ifu[instance]:
            raise ValueError("duplicate physical IFU identity within %s: %s" % (instance, key))
        ifu.setdefault(instance, {})[key] = np.asarray(record["pop_coeff"], dtype=float)
        centers[(instance, key)] = (float(record["chosen"]["ra"]), float(record["chosen"]["dec"]))
    amp = {}
    for row in products["amp_rows"]:
        instance = exposure_instance_id(row["H5"], row["exposure"])
        key = physical_amp_key(row)
        band = NULL_BANDS.index(text(row["null_band"]))
        seen_key = (instance, key, band)
        if seen_key in seen_amp:
            raise ValueError("duplicate physical amplifier identity within %s: %s %s" % (instance, key, text(row["null_band"])))
        seen_amp.add(seen_key)
        target = amp.setdefault(instance, {}).setdefault(key, np.full(len(NULL_BANDS), np.nan))
        target[band] = float(row["within_IFU_deviation"]) if row["within_IFU_deviation"] != "" else np.nan
    if len(ifu) != 6:
        raise ValueError("persistence analysis requires six exposure instances; found %d" % len(ifu))
    return ifu, amp, centers


def normalize_persistence(data):
    normalized = {}
    stats = {}
    for instance, values in data.items():
        matrix = np.asarray(list(values.values()), dtype=float)
        mu = np.asarray([robust_location(matrix[:, k]) for k in range(len(NULL_BANDS))])
        scale = np.asarray([robust_scale(matrix[:, k]) for k in range(len(NULL_BANDS))])
        stats[instance] = {"mu": mu, "sigma": scale}
        normalized[instance] = {key: (value - mu) / scale for key, value in values.items()}
    return normalized, stats


def normalization_qa(normalized):
    locations = []
    scales = []
    for values in normalized.values():
        matrix = np.asarray(list(values.values()), dtype=float)
        for band_index in range(len(NULL_BANDS)):
            locations.append(robust_location(matrix[:, band_index]))
            scales.append(robust_scale(matrix[:, band_index]))
    locations = np.asarray(locations, dtype=float)
    scales = np.asarray(scales, dtype=float)
    return {"max_abs_robust_location": float(np.nanmax(np.abs(locations))),
            "median_robust_location": float(np.nanmedian(locations)),
            "median_robust_scale": float(np.nanmedian(scales)),
            "N": int(locations.size)}


def persistence_pairs(instances):
    instances = sorted(instances)
    pairs = []
    for i, first in enumerate(instances):
        for second in instances[i + 1:]:
            first_h5 = first.rsplit("_e", 1)[0]
            second_h5 = second.rsplit("_e", 1)[0]
            first_e = int(first.rsplit("_e", 1)[1])
            second_e = int(second.rsplit("_e", 1)[1])
            if first_h5 == second_h5 or first_e == second_e:
                category = "within_H5" if first_h5 == second_h5 else "cross_H5_same_exposure_index"
                pairs.append((first, second, category))
    return pairs


def persistence_pair_rows(data, normalized, level):
    rows = []
    instances = sorted(data)
    pairs = persistence_pairs(instances)
    for first, second, category in pairs:
        first_keys = set(data[first]); second_keys = set(data[second]); matched = sorted(first_keys & second_keys)
        raw = np.asarray([[data[first][key][k], data[second][key][k]] for key in matched for k in range(len(NULL_BANDS))], dtype=float)
        del raw
        for band_index, name in enumerate(NULL_BANDS):
            x = np.asarray([data[first][key][band_index] for key in matched])
            y = np.asarray([data[second][key][band_index] for key in matched])
            raw_stats = pair_statistics(x, y)
            nx = np.asarray([normalized[first].get(key, np.full(len(NULL_BANDS), np.nan))[band_index] for key in matched])
            ny = np.asarray([normalized[second].get(key, np.full(len(NULL_BANDS), np.nan))[band_index] for key in matched])
            norm_stats = pair_statistics(nx, ny)
            rows.append({"level": level, "exposure_A": first, "exposure_B": second, "comparison": category,
                         "null_band": name, "N_matched": raw_stats["N_matched"],
                         "raw_Pearson_r": raw_stats["Pearson_r"], "raw_Spearman_r": raw_stats["Spearman_r"],
                         "raw_robust_slope": raw_stats["robust_slope"], "raw_robust_intercept": raw_stats["robust_intercept"],
                         "raw_slope_through_zero": raw_stats["slope_through_zero"],
                         "normalized_Pearson_r": norm_stats["Pearson_r"], "normalized_Spearman_r": norm_stats["Spearman_r"],
                         "normalized_robust_slope": norm_stats["robust_slope"], "normalized_robust_intercept": norm_stats["robust_intercept"],
                         "normalized_slope_through_zero": norm_stats["slope_through_zero"],
                         "normalized_RMS_difference": norm_stats["normalized_RMS_difference"]})
    return rows


def persistence_matrices(data, normalized, level):
    matrices = {}
    rows = []
    instances = sorted(data)
    for band_index, name in enumerate(NULL_BANDS):
        for kind, source in (("raw", data), ("normalized", normalized)):
            corr = np.full((len(instances), len(instances)), np.nan)
            counts = np.zeros((len(instances), len(instances)), dtype=int)
            for i, first in enumerate(instances):
                for j, second in enumerate(instances):
                    keys = sorted(set(source[first]) & set(source[second]))
                    x = np.asarray([source[first][key][band_index] for key in keys])
                    y = np.asarray([source[second][key][band_index] for key in keys])
                    use = np.isfinite(x) & np.isfinite(y)
                    counts[i, j] = int(np.sum(use))
                    corr[i, j] = pearson_r(x, y)
                    rows.append({"level": level, "matrix_type": kind, "null_band": name,
                                 "row_instance": first, "column_instance": second,
                                 "correlation": corr[i, j], "N_matched": counts[i, j]})
            matrices["%s|%s|%s" % (level, kind, name)] = {"instances": instances, "correlation": corr.tolist(), "N_matched": counts.tolist()}
    return rows, matrices


def template_records(data, normalized, minimum_exposures, level):
    keys = sorted(set().union(*(set(values) for values in normalized.values())))
    templates = {}
    rows = []
    for key in keys:
        for band_index, name in enumerate(NULL_BANDS):
            observations = [(instance, normalized[instance][key][band_index]) for instance in normalized if key in normalized[instance] and np.isfinite(normalized[instance][key][band_index])]
            if len(observations) < minimum_exposures:
                continue
            values = np.asarray([value for _, value in observations])
            template = robust_location(values)
            scatter = robust_scale(values)
            templates[(key, band_index)] = template
            row = {"null_band": name, "N_exposures": len(observations),
                   "template_normalized_coefficient": template, "across_exposure_scatter": scatter}
            if level == "IFU":
                row.update({"SPECID": key[0], "IFUSLOT": key[1], "IFUID": key[2]})
            else:
                row.update({"SPECID": key[0], "IFUSLOT": key[1], "IFUID": key[2], "AMP": key[3]})
            rows.append(row)
    return templates, rows


def template_performance(normalized, templates, level):
    rows = []
    for instance, values in normalized.items():
        for band_index, name in enumerate(NULL_BANDS):
            matched = [(key, values[key][band_index], templates[(key, band_index)]) for key in values if (key, band_index) in templates and np.isfinite(values[key][band_index])]
            y = np.asarray([item[1] for item in matched]); prediction = np.asarray([item[2] for item in matched])
            residual = y - prediction
            variance = float(np.var(y)) if y.size else np.nan
            rows.append({"level": level, "exposure": instance, "null_band": name, "N_matched": int(y.size),
                         "correlation": pearson_r(y, prediction),
                         "RMS_residual": float(np.sqrt(np.median(residual ** 2))) if y.size else np.nan,
                         "robust_residual_scale": robust_scale(residual),
                         "fraction_variance_explained": 1.0 - float(np.var(residual)) / variance if y.size and variance > 0 else np.nan})
    return rows


def exposure_amplitudes(normalized, templates, level):
    rows = []
    for instance, values in normalized.items():
        for band_index, name in enumerate(NULL_BANDS):
            matched = [(key, values[key][band_index], templates[(key, band_index)]) for key in values if (key, band_index) in templates and np.isfinite(values[key][band_index])]
            y = np.asarray([item[1] for item in matched]); x = np.asarray([item[2] for item in matched])
            slope = robust_linear_stats(x, y)[2]
            rows.append({"level": level, "exposure": instance, "null_band": name,
                         "amplitude": slope, "N": int(y.size), "correlation": pearson_r(x, y)})
    return rows


def persistent_amp_orientation(amp_template_rows, minimum_exposures):
    orientation_medians = {}
    for band in NULL_BANDS:
        for amp_name in AMPS:
            vals = np.asarray([row["template_normalized_coefficient"] for row in amp_template_rows
                               if row["null_band"] == band and row["AMP"] == amp_name
                               and row["N_exposures"] >= minimum_exposures], dtype=float)
            vals = vals[np.isfinite(vals)]
            orientation_medians[(band, amp_name)] = robust_location(vals)
    result = []
    for band in NULL_BANDS:
        all_values = np.asarray([row["template_normalized_coefficient"] for row in amp_template_rows
                                 if row["null_band"] == band and row["N_exposures"] >= minimum_exposures], dtype=float)
        all_values = all_values[np.isfinite(all_values)]
        centered_values = []
        for row in amp_template_rows:
            if row["null_band"] != band or row["N_exposures"] < minimum_exposures:
                continue
            value = float(row["template_normalized_coefficient"])
            center = orientation_medians[(band, row["AMP"])]
            if np.isfinite(value) and np.isfinite(center):
                centered_values.append(value - center)
        centered_values = np.asarray(centered_values, dtype=float)
        pooled_before = robust_scale(all_values)
        pooled_after = robust_scale(centered_values)
        fractional_reduction = 1.0 - pooled_after / pooled_before if np.isfinite(pooled_before) and pooled_before > 0 and np.isfinite(pooled_after) else np.nan
        for amp_name in AMPS:
            values = np.asarray([row["template_normalized_coefficient"] for row in amp_template_rows
                                 if row["null_band"] == band and row["AMP"] == amp_name
                                 and row["N_exposures"] >= minimum_exposures], dtype=float)
            values = values[np.isfinite(values)]
            result.append({"null_band": band, "AMP": amp_name, "N": int(values.size),
                           "orientation_median": orientation_medians[(band, amp_name)],
                           "orientation_scale": robust_scale(values),
                           "pooled_template_scale_before": pooled_before,
                           "pooled_orientation_residual_scale_after": pooled_after,
                           "pooled_fractional_reduction": fractional_reduction})
    return result

def persistence_figures(output_dir, ifu_data, ifu_normalized, amp_data, amp_normalized,
                        ifu_matrices, amp_matrices, ifu_templates, amp_template_rows,
                        amplitude_rows, centers, minimum_exposures):
    paths = []
    instances = sorted(ifu_data)
    pairs = persistence_pairs(instances)
    for level, data, normalized, matrices in (("IFU", ifu_data, ifu_normalized, ifu_matrices), ("AMP", amp_data, amp_normalized, amp_matrices)):
        for band_index, name in enumerate(NULL_BANDS):
            for kind, source in (("raw", data), ("normalized", normalized)):
                key = "%s|%s|%s" % (level, kind, name)
                matrix = np.asarray(matrices[key]["correlation"], dtype=float)
                fig, axis = plt.subplots(figsize=(6, 5), constrained_layout=True)
                image = axis.imshow(matrix, vmin=-1, vmax=1, cmap="coolwarm")
                axis.set_xticks(np.arange(len(instances)), instances, rotation=75, ha="right")
                axis.set_yticks(np.arange(len(instances)), instances)
                axis.set_title("%s %s persistence %s" % (level, kind, name))
                for i in range(len(instances)):
                    for j in range(len(instances)):
                        if np.isfinite(matrix[i, j]): axis.text(j, i, "%.2f" % matrix[i, j], ha="center", va="center", fontsize=7)
                fig.colorbar(image, ax=axis, label="Pearson r")
                path = output_dir / ("%s_persistence_%s_%s.png" % (level.lower(), kind, name)); fig.savefig(path, dpi=140); plt.close(fig); paths.append(str(path))
            if level == "IFU":
                fig, axes = plt.subplots(3, 3, figsize=(13, 12), constrained_layout=True)
                for axis, (first, second, category) in zip(axes.flat, pairs):
                    keys = sorted(set(data[first]) & set(data[second]))
                    x = np.asarray([data[first][key][band_index] for key in keys]); y = np.asarray([data[second][key][band_index] for key in keys])
                    stats = pair_statistics(x, y); finite = np.concatenate((x[np.isfinite(x)], y[np.isfinite(y)]))
                    lim = max(float(np.percentile(np.abs(finite), 99) * 1.1), 1e-8) if finite.size else 1.
                    axis.scatter(x, y, s=12, alpha=.65); axis.plot([-lim, lim], [-lim, lim], "k--", lw=.7); axis.axhline(0, color="k", lw=.4); axis.axvline(0, color="k", lw=.4)
                    slope, intercept = stats["robust_slope"], stats["robust_intercept"]
                    if np.isfinite(slope): axis.plot([-lim, lim], intercept + slope * np.asarray([-lim, lim]), color="tab:red", lw=1)
                    axis.set_xlim(-lim, lim); axis.set_ylim(-lim, lim); axis.set_title("%s vs %s" % (first, second), fontsize=8)
                    axis.text(.03, .97, "N=%d\nr=%.2f\nrs=%.2f\ns0=%.2f" % (stats["N_matched"], stats["Pearson_r"], stats["Spearman_r"], stats["slope_through_zero"]), transform=axis.transAxes, va="top", fontsize=7)
                    axis.grid(alpha=.15)
                fig.suptitle("IFU %s persistence scatter: %s" % (kind, name)); path = output_dir / ("ifu_persistence_scatter_%s_%s.png" % (kind, name)); fig.savefig(path, dpi=140); plt.close(fig); paths.append(str(path))
    # Persistent IFU template focal-plane map.
    template_values = {(row["SPECID"], row["IFUSLOT"], row["IFUID"], row["null_band"]): row["template_normalized_coefficient"] for row in ifu_templates}
    unique_ifus = sorted({(int(row["SPECID"]), int(row["IFUSLOT"]), int(row["IFUID"])) for row in ifu_templates})
    if unique_ifus:
        ras = np.asarray([np.nanmedian([centers[(instance, key)][0] for instance in centers if (instance, key) in centers]) for key in unique_ifus])
        decs = np.asarray([np.nanmedian([centers[(instance, key)][1] for instance in centers if (instance, key) in centers]) for key in unique_ifus])
        ra0, dec0 = float(np.nanmedian(ras)), float(np.nanmedian(decs)); xs = (ras - ra0) * np.cos(np.deg2rad(dec0)) * 3600.; ys = (decs - dec0) * 3600.
        vals = np.asarray([v for v in template_values.values() if np.isfinite(v)]); limit = max(float(np.percentile(np.abs(vals), 99)), 1e-8) if vals.size else 1.
        fig, axes = plt.subplots(1, 5, figsize=(17, 4), sharex=True, sharey=True, constrained_layout=True)
        for axis, band in zip(axes, NULL_BANDS):
            colors = np.asarray([template_values.get((key[0], key[1], key[2], band), np.nan) for key in unique_ifus])
            axis.scatter(xs, ys, c=colors, cmap="coolwarm", vmin=-limit, vmax=limit, s=30, edgecolors="k", linewidths=.25); axis.set_title(band); axis.set_xlabel("Delta RA cos(Dec) [arcsec]"); axis.set_aspect("equal", adjustable="box"); axis.grid(alpha=.15)
        axes[0].set_ylabel("Delta Dec [arcsec]"); fig.suptitle("Persistent IFU template; common color limit %.4g" % limit); path = output_dir / "persistent_ifu_template_spatial.png"; fig.savefig(path, dpi=140); plt.close(fig); paths.append(str(path))
    for level in ("IFU", "AMP"):
        rows = [row for row in amplitude_rows if row["level"] == level]
        fig, axis = plt.subplots(figsize=(10, 5), constrained_layout=True)
        for instance in instances:
            values = [next((row["amplitude"] for row in rows if row["exposure"] == instance and row["null_band"] == band), np.nan) for band in NULL_BANDS]
            axis.plot(np.arange(len(NULL_BANDS)), values, "o-", label=instance)
        axis.axhline(0, color="k", lw=.7); axis.set_xticks(np.arange(len(NULL_BANDS)), [band.replace("NULL_", "") for band in NULL_BANDS]); axis.set_ylabel("descriptive exposure amplitude"); axis.set_title("%s exposure amplitudes across null bands" % level); axis.legend(fontsize=7, ncol=2); axis.grid(alpha=.18)
        path = output_dir / ("persistence_%s_exposure_amplitudes.png" % level.lower()); fig.savefig(path, dpi=140); plt.close(fig); paths.append(str(path))
    return paths

def discover_h5(args):
    sources = []
    if args.h5_positional: sources.extend(args.h5_positional)
    if args.h5: sources.extend(args.h5)
    if args.h5_glob: sources.extend(glob.glob(args.h5_glob))
    if not sources:
        raise ValueError("supply positional H5 files, --h5, or --h5-glob")
    paths = [Path(x).expanduser().resolve() for x in sources]
    if any(not x.is_file() for x in paths): raise ValueError("an input H5 does not exist")
    if len({x.name for x in paths}) != len(paths): raise ValueError("ambiguous input H5 basenames")
    return sorted(paths)


def selection_signature(rows):
    return sorted((row["H5"], int(row["exposure"]), tuple(row["ifu"])) for row in rows)


def validate_selection_reference(path, selected_all):
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise ValueError("selection reference does not exist: %s" % path)
    data = json.loads(path.read_text())
    expected = data.get("selected_ifus")
    if expected is None:
        raise ValueError("selection reference lacks selected_ifus: %s" % path)
    expected_signature = sorted((text(row["H5"]), int(row["exposure"]), tuple(int(x) for x in row["ifu"])) for row in expected)
    actual_signature = selection_signature(selected_all)
    if expected_signature != actual_signature:
        raise ValueError("current deterministic IFU selection differs from reference %s" % path)
    return {"path": str(path), "selection_unchanged": True, "n_selected": len(actual_signature)}


def load_additive_full(path, h5_names):
    path = Path(path).expanduser().resolve()
    with tables.open_file(path, mode="r") as h5:
        if text(getattr(h5.root._v_attrs, "schema_version", "")) != "m101_additive_structure_v1":
            raise ValueError("unsupported additive structure schema")
        if "/provenance/metadata" not in h5:
            raise ValueError("additive product lacks provenance metadata")
        metadata = {text(row["key"]): json.loads(text(row["value"]))
                    for row in h5.root.provenance.metadata}
        product_names = {Path(item.get("filename", item.get("path", ""))).name
                         for item in metadata.get("input_h5", [])}
        if product_names != set(h5_names):
            raise ValueError("additive and diagnostic H5 basenames do not agree")
        exposures, ifus, amps = {}, {}, {}
        for row in h5.root.exposures:
            key = (Path(text(row["H5"])).name, int(row["exposure"]))
            exposures[key] = {name: np.asarray(row[name], dtype=float)
                              for name in ("R_lambda", "U_lambda", "V_lambda")}
        for row in h5.root.physical_ifus:
            key = (int(row["SPECID"]), int(row["IFUSLOT"]), int(row["IFUID"]))
            ifus[key] = (float(row["c"]), bool(row["c_supported"]))
        for row in h5.root.physical_amplifiers:
            key = (int(row["SPECID"]), int(row["IFUSLOT"]), int(row["IFUID"]), text(row["AMP"]))
            amps[key] = (float(row["d"]), bool(row["d_supported"]))
    return exposures, ifus, amps, metadata, file_identity(path)


def production_validation(h5_paths, additive_path, final_fit_path, fq_path, blank_path,
                          output_dir, minimum_finite_fraction):
    fit, fit_metadata, fit_identity = load_fit(final_fit_path)
    expected_additive = fit_metadata.get("additive_structure")
    if expected_additive:
        actual_additive = file_identity(additive_path)
        if (expected_additive.get("size") != actual_additive["size"] or
                expected_additive.get("mtime_ns") != actual_additive["mtime_ns"] or
                Path(expected_additive.get("path", "")).name != Path(actual_additive["path"]).name):
            raise ValueError("final fit additive identity does not match supplied additive product")
    fq, fq_identity = load_fq(fq_path)
    blank_masks, blank_provenance = load_external_blank(blank_path, h5_paths)
    h5_names = {path.name for path in h5_paths}
    additive_exp, additive_ifus, additive_amps, additive_metadata, additive_identity = load_additive_full(additive_path, h5_names)
    if additive_metadata.get("uses_leave_one_out") is True:
        raise ValueError("production additive product unexpectedly records leave-one-out fitting")
    records = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for path in h5_paths:
            with tables.open_file(path, mode="r") as h5:
                info, fibers = h5.root.Info, h5.root.Fibers
                groups, labels = hm.build_groups(info)
                group_ids = group_lookup(groups, info.nrows)
                if "spectrum" not in fibers.colnames:
                    raise ValueError("%s Fibers lacks spectrum" % path)
                ifuslot = np.asarray(info.cols.ifuslot[:]); amp = np.asarray([text(v) for v in info.cols.amp[:]])
                specid = np.asarray(info.cols.specid[:], dtype=int); ifuid = np.asarray(info.cols.ifuid[:], dtype=int)
                ra = np.asarray(info.cols.ra[:], dtype=float); dec = np.asarray(info.cols.dec[:], dtype=float)
                date_bad = hm.masked_rows(path, ifuslot, amp); surveys = surveys_by_exposure(h5)
                for exposure in range(1, N_EXPOSURES + 1):
                    rows = np.flatnonzero((labels == exposure) & blank_masks[path.name] & ~date_bad)
                    spectra = np.asarray(fibers.read_coordinates(rows, field="spectrum"), dtype=float)
                    finite = np.mean(np.isfinite(spectra), axis=1) >= minimum_finite_fraction if spectra.size else np.asarray([], dtype=bool)
                    rows, spectra = rows[finite], spectra[finite]
                    offset = float(surveys[exposure]["offset"])
                    if not np.isfinite(offset) or offset == 0:
                        raise ValueError("invalid Survey.offset for %s exposure %d" % (path, exposure))
                    raw = spectra / offset
                    K = hm.raw_work_basis(surveys[exposure])
                    by_group = {}
                    for group_index, group in enumerate(groups):
                        if group["exposure"] != exposure:
                            continue
                        local = np.flatnonzero(group_ids[rows] == group_index)
                        if not local.size:
                            continue
                        ifu = (group["specid"], group["ifuslot"], group["ifuid"]); amp_key = ifu + (group["amp"],)
                        fit_key = (path.name, exposure, *amp_key)
                        if fit_key not in fit:
                            raise ValueError("missing final fit amplifier row: %s" % (fit_key,))
                        if (path.name, exposure) not in additive_exp:
                            raise ValueError("missing additive exposure row: %s" % ((path.name, exposure),))
                        c, c_supported = additive_ifus.get(ifu, (0.0, False)); d, d_supported = additive_amps.get(amp_key, (0.0, False))
                        terms = additive_exp[(path.name, exposure)]
                        q = np.arange(N_FIBER_AMP) if group["amp"] in ("LL", "RU") else np.arange(N_FIBER_AMP - 1, -1, -1)
                        positions = {int(row): index for index, row in enumerate(group["indices"])}
                        q_rows = np.asarray([q[positions[int(row)]] for row in rows[local]], dtype=int)
                        alpha, z = fit[fit_key]["alpha"], fit[fit_key]["z"]
                        additive = terms["R_lambda"] + c * terms["U_lambda"] + d * terms["V_lambda"]
                        H = K[None, :] * fq[q_rows, None]
                        native = raw[local]
                        stages = np.asarray([native, native - terms["R_lambda"],
                                             native - terms["R_lambda"] - c * terms["U_lambda"],
                                             native - additive, native - additive - alpha * H], dtype=float)
                        stages = np.concatenate((stages, (stages[-1] / np.exp(z))[None, :, :]), axis=0)
                        group_native = np.asarray([robust_spectrum(stage) for stage in stages])
                        group_null = np.asarray([[robust_location(stage[indexes]) for indexes in window_indices()]
                                                 for stage in group_native])
                        records.append({"H5": path.name, "exposure": exposure, "ifu": ifu, "AMP": group["amp"],
                                        "N_blank": int(local.size), "RA": float(np.nanmedian(ra[rows[local]])),
                                        "Dec": float(np.nanmedian(dec[rows[local]])), "stages": group_native,
                                        "null": group_null, "c": c, "d": d, "c_supported": c_supported,
                                        "d_supported": d_supported})
    if not records:
        raise ValueError("production validation found no usable externally blank fibers")
    stage_names = ("stage0_raw", "stage1_exposure", "stage2_ifu", "stage3_amp", "stage4_alpha", "stage5_final")
    spectra = np.asarray([record["stages"] for record in records], dtype=float)
    nulls = np.asarray([record["null"] for record in records], dtype=float)
    full_rows, null_rows = [], []
    for index, name in enumerate(stage_names):
        values = spectra[:, index]
        median = np.nanmedian(values, axis=0)
        scale = np.asarray([robust_scale(values[:, wave]) for wave in range(N_WAVE)])
        for wave, wavelength in enumerate(DEF_WAVE):
            full_rows.append({"stage": name, "wavelength": float(wavelength), "median": median[wave], "robust_scatter": scale[wave], "N": int(np.sum(np.isfinite(values[:, wave])))})
        for band_index, band in enumerate(NULL_BANDS):
            band_values = nulls[:, index, band_index]
            null_rows.append({"stage": name, "null_band": band, "median": robust_location(band_values), "robust_scatter": robust_scale(band_values), "N": int(np.sum(np.isfinite(band_values)))})
    write_csv(output_dir / "production_blank_full_spectrum.csv", full_rows)
    write_csv(output_dir / "production_blank_null_bands.csv", null_rows)
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), constrained_layout=True)
    colors = ("0.45", "tab:blue", "tab:orange", "tab:green", "tab:red", "k")
    for index, name in enumerate(stage_names):
        rows = [row for row in full_rows if row["stage"] == name]
        axes[0].plot(DEF_WAVE, [row["median"] for row in rows], color=colors[index], label=name)
        axes[0].fill_between(DEF_WAVE, [row["median"] - row["robust_scatter"] for row in rows], [row["median"] + row["robust_scatter"] for row in rows], color=colors[index], alpha=.08)
        rows = [row for row in null_rows if row["stage"] == name]
        axes[1].plot(np.arange(len(NULL_BANDS)), [row["robust_scatter"] for row in rows], "o-", color=colors[index], label=name)
    axes[0].set(xlabel="wavelength [A]", ylabel="blank residual", title="complete-product blank residual stages"); axes[0].legend(fontsize=8, ncol=3); axes[0].grid(alpha=.2)
    axes[1].set_xticks(np.arange(len(NULL_BANDS)), [x.replace("NULL_", "") for x in NULL_BANDS], rotation=30); axes[1].set_ylabel("robust scatter"); axes[1].grid(alpha=.2); axes[1].legend(fontsize=8, ncol=3)
    figure = output_dir / "production_blank_validation.png"; fig.savefig(figure, dpi=140); plt.close(fig)
    map_figure, map_axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    wave_index = int(np.argmin(np.abs(DEF_WAVE - 3908.0)))
    map_stages = (0, 3, 5)
    ra_values = np.asarray([record["RA"] for record in records]); dec_values = np.asarray([record["Dec"] for record in records])
    ra0, dec0 = np.nanmedian(ra_values), np.nanmedian(dec_values)
    x_values = (ra_values - ra0) * np.cos(np.deg2rad(dec0)) * 3600.0; y_values = (dec_values - dec0) * 3600.0
    map_limits = max(float(np.nanpercentile(np.abs(spectra[:, map_stages, wave_index]), 99)), 1e-8)
    null_limits = max(float(np.nanpercentile(np.abs(nulls[:, map_stages, 0]), 99)), 1e-8)
    for column, stage_index in enumerate(map_stages):
        map_axes[0, column].scatter(x_values, y_values, c=spectra[:, stage_index, wave_index],
                                    cmap="coolwarm", vmin=-map_limits, vmax=map_limits, s=8)
        map_axes[0, column].set_title("%s at %.1f A" % (stage_names[stage_index], DEF_WAVE[wave_index]))
        map_axes[1, column].scatter(x_values, y_values, c=nulls[:, stage_index, 0],
                                    cmap="coolwarm", vmin=-null_limits, vmax=null_limits, s=8)
        map_axes[1, column].set_title("%s %s" % (stage_names[stage_index], NULL_BANDS[0]))
        for row_index in range(2):
            map_axes[row_index, column].set_xlabel("Delta RA cos(Dec) [arcsec]")
            map_axes[row_index, column].set_aspect("equal", adjustable="box")
    map_axes[0, 0].set_ylabel("Delta Dec [arcsec]"); map_axes[1, 0].set_ylabel("Delta Dec [arcsec]")
    map_figure_path = output_dir / "production_blank_spatial_maps.png"; map_figure.savefig(map_figure_path, dpi=140); plt.close(map_figure)
    stage_rms = [robust_rms(spectra[:, index]) for index in range(len(stage_names))]
    summary = {"mode": "production-validation", "h5_files": [str(path) for path in h5_paths],
               "final_fit": fit_identity, "additive_structure": additive_identity, "fq_template": fq_identity,
               "external_blank_fibers": blank_provenance, "n_blank_group_records": len(records),
               "stage_names": list(stage_names), "stage_full_spectrum_rms": stage_rms,
               "stage_improvement_factor_from_raw": [stage_rms[0] / x if np.isfinite(x) and x > 0 else np.nan for x in stage_rms],
               "figure": str(figure), "spatial_maps": str(map_figure_path), "uses_leave_one_out": False, "uses_Fibers_skyspectrum": False,
               "uses_p": False, "native_equation": "[W-R-cU-dV-alpha*K*f(q)]/exp(posterior_z)"}
    (output_dir / "production_blank_validation_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps(summary, indent=2, default=str))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("h5_positional", nargs="*")
    parser.add_argument("--h5", nargs="+")
    parser.add_argument("--h5-glob")
    parser.add_argument("--fit-h5")
    parser.add_argument("--final-fit-h5")
    parser.add_argument("--additive-structure")
    parser.add_argument("--production-validation", action="store_true")
    parser.add_argument("--on-filter")
    parser.add_argument("--off-filter")
    parser.add_argument("--fq-template", required=True)
    parser.add_argument("--external-blank-fibers", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--minimum-finite-fraction", type=float, default=DEFAULT_MINIMUM_FINITE_FRACTION)
    parser.add_argument("--ifus-per-exposure", type=int, default=DEFAULT_IFUS_PER_EXPOSURE)
    parser.add_argument("--population-min-blank-fraction", type=float, default=.8)
    parser.add_argument("--population-min-blank-fibers", type=int, default=50)
    parser.add_argument("--population-min-blank-per-amp", type=int, default=10)
    parser.add_argument("--persistence-min-exposures", type=int, default=3)
    parser.add_argument("--selection-reference", help="prior provenance JSON used to verify unchanged detailed IFU selections")
    args = parser.parse_args()
    started = time.perf_counter()
    if not 0 < args.minimum_finite_fraction <= 1:
        raise SystemExit("minimum finite fraction must be in (0,1]")
    if not 0 < args.population_min_blank_fraction <= 1:
        raise SystemExit("population minimum blank fraction must be in (0,1]")
    if args.ifus_per_exposure < 1 or args.population_min_blank_fibers < 1 or args.population_min_blank_per_amp < 1 or args.persistence_min_exposures < 1:
        raise SystemExit("IFU and blank-fiber thresholds must be positive")
    if args.production_validation:
        if not args.final_fit_h5 or not args.additive_structure:
            raise SystemExit("--production-validation requires --final-fit-h5 and --additive-structure")
        Path(args.output_dir).expanduser().resolve().mkdir(parents=True, exist_ok=True)
        return production_validation(
            discover_h5(args), args.additive_structure, args.final_fit_h5,
            args.fq_template, args.external_blank_fibers,
            Path(args.output_dir).expanduser().resolve(), args.minimum_finite_fraction)
    if not args.fit_h5:
        raise SystemExit("legacy diagnostic mode requires --fit-h5")
    h5_paths = discover_h5(args)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    fit, fit_metadata, fit_identity = load_fit(args.fit_h5)
    fq, fq_identity = load_fq(args.fq_template)
    blank_masks, blank_provenance = load_external_blank(args.external_blank_fibers, h5_paths)
    selected_all, selection_rationale = [], {}
    records, population = [], []
    for path in h5_paths:
        with tables.open_file(path, mode="r") as h5:
            groups, labels = hm.build_groups(h5.root.Info)
            ra = np.asarray(h5.root.Info.cols.ra[:], float)
            dec = np.asarray(h5.root.Info.cols.dec[:], float)
            selected, rationale = select_ifus(groups, labels, blank_masks[path.name], ra, dec, args.ifus_per_exposure)
        selected_all.extend([{**x, "H5": path.name} for x in selected])
        selection_rationale[path.name] = rationale
        processed = process_h5(path, blank_masks[path.name], fit, fq, selected,
                               args.minimum_finite_fraction, args.population_min_blank_fraction,
                               args.population_min_blank_fibers, args.population_min_blank_per_amp)
        records.extend(processed["detailed"])
        population.extend(processed["population"])
    selection_reference = validate_selection_reference(args.selection_reference, selected_all) if args.selection_reference else None
    if args.selection_reference and len(records) != len(selected_all):
        raise ValueError("not every selected IFU produced a detailed record")
    all_rows = []
    summary_rows = []
    amp_summary = []
    figure_paths = []
    plot_metadata = []
    for record in records:
        metrics = spectrum_metrics(record)
        chosen = record["chosen"]
        base = {"H5": record["h5"], "exposure": record["exposure"], "SPECID": chosen["ifu"][0],
                "IFUSLOT": chosen["ifu"][1], "IFUID": chosen["ifu"][2], "N_total": chosen["n_total"],
                "N_blank": chosen["n_blank"], "N_blank_usable": record["n_blank_usable"],
                "blank_fraction": chosen["blank_fraction"], **metrics}
        summary_rows.append(base)
        all_rows.extend(residual_rows(record))
        amp_summary.extend(amp_rows(record))
        paths, plot_meta = plot_record(record, output_dir)
        figure_paths.extend(paths)
        plot_metadata.append(plot_meta)
    write_csv(output_dir / "blank_ifu_transfer_summary.csv", summary_rows)
    write_csv(output_dir / "blank_ifu_amp_residuals.csv", amp_summary)
    spectral_fields = ["H5", "exposure", "SPECID", "IFUSLOT", "IFUID", "wavelength",
                       "N_blank_IFU_finite", "N_blank_reference_finite", "IFU_pre_m_Y",
                       "leave_one_IFU_out_pre_m_Y_sky", "residual", "corrected_residual",
                       "monly_residual", "raw_residual", "robust_scatter_IFU",
                       "robust_scatter_reference"]
    for record in records:
        key = "%s_exp%d_SPECID%d_SLOT%d_UID%d" % (record["h5"].replace(".h5", ""), record["exposure"], *record["ifu"])
        safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in key)
        write_csv(output_dir / (safe + ".csv"), residual_rows(record), spectral_fields)
    overview = plot_overview(output_dir, records)
    figure_paths.append(overview)

    products = population_products(population, args.population_min_blank_per_amp)
    population_spectrum_fields = ["H5", "exposure", "SPECID", "IFUSLOT", "IFUID", "wavelength",
                                  "N_blank_finite", "IFU_residual", "IFU_residual_scale"]
    population_spectrum_rows_all = []
    for record in population:
        population_spectrum_rows_all.extend(population_spectrum_rows(record))
    write_csv_gz(output_dir / "blank_ifu_population_spectra.csv.gz", population_spectrum_rows_all,
                 population_spectrum_fields)
    write_csv(output_dir / "blank_ifu_band_coefficients.csv", products["band_rows"])
    write_csv(output_dir / "blank_amp_band_coefficients.csv", products["amp_rows"])
    write_csv(output_dir / "blank_spatial_scale_decomposition.csv", products["decomposition"])
    write_csv(output_dir / "blank_amp_orientation_summary.csv", products["orientation_summary"])
    write_csv(output_dir / "blank_ifu_band_correlations.csv", products["correlation_rows"])
    population_paths, population_plot_metadata = population_figures(output_dir, population, products)
    figure_paths.extend(population_paths)

    direct_diffs = []
    for h5_name, exposure, ifu, coeff, direct in products["direct_checks"]:
        use = np.isfinite(coeff) & np.isfinite(direct)
        direct_diffs.extend((coeff[use] - direct[use]).tolist())
    direct_diffs = np.asarray(direct_diffs, dtype=float)
    direct_qa = {"N": int(direct_diffs.size),
                 "max_abs": float(np.max(np.abs(direct_diffs))) if direct_diffs.size else np.nan,
                 "median_abs": float(np.median(np.abs(direct_diffs))) if direct_diffs.size else np.nan}
    amp_common_qa = []
    for record in population:
        for band_index in range(len(NULL_BANDS)):
            vals = np.asarray([record["amp_coeff"][a]["values"][band_index] for a in AMPS], dtype=float)
            vals = vals[np.isfinite(vals)]
            if vals.size:
                center = robust_location(vals)
                deviations = vals - center
                amp_common_qa.append(robust_location(deviations))
    amp_common_qa = np.asarray(amp_common_qa, dtype=float)
    hierarchy_qa = {"IFU_band_direct_collapse": direct_qa,
                    "median_robust_location_of_amp_deviations": robust_location(amp_common_qa),
                    "max_abs_robust_location_of_amp_deviations": float(np.max(np.abs(amp_common_qa))) if amp_common_qa.size else np.nan,
                    "population_reference_excludes_current_IFU": True,
                    "population_does_not_use_posterior_z": True,
                    "population_does_not_use_Fibers_skyspectrum": True}
    ordering = ordering_sanity()
    persistence_dir = output_dir
    ifu_data, amp_data, centers = build_persistence_inputs(population, products)
    ifu_normalized, ifu_norm_stats = normalize_persistence(ifu_data)
    amp_normalized, amp_norm_stats = normalize_persistence(amp_data)
    ifu_norm_qa = normalization_qa(ifu_normalized)
    amp_norm_qa = normalization_qa(amp_normalized)
    ifu_pair_rows = persistence_pair_rows(ifu_data, ifu_normalized, "IFU")
    amp_pair_rows = persistence_pair_rows(amp_data, amp_normalized, "AMP")
    ifu_matrix_rows, ifu_matrices = persistence_matrices(ifu_data, ifu_normalized, "IFU")
    amp_matrix_rows, amp_matrices = persistence_matrices(amp_data, amp_normalized, "AMP")
    ifu_templates, ifu_template_rows = template_records(ifu_data, ifu_normalized, args.persistence_min_exposures, "IFU")
    amp_templates, amp_template_rows = template_records(amp_data, amp_normalized, args.persistence_min_exposures, "AMP")
    ifu_performance = template_performance(ifu_normalized, ifu_templates, "IFU")
    amp_performance = template_performance(amp_normalized, amp_templates, "AMP")
    ifu_amplitudes = exposure_amplitudes(ifu_normalized, ifu_templates, "IFU")
    amp_amplitudes = exposure_amplitudes(amp_normalized, amp_templates, "AMP")
    orientation_decomposition = persistent_amp_orientation(amp_template_rows, args.persistence_min_exposures)
    template_band_rows = []
    template_keys = sorted({key for (key, band_index) in ifu_templates})
    template_matrix = np.full((len(NULL_BANDS), len(NULL_BANDS)), np.nan)
    template_counts = np.zeros((len(NULL_BANDS), len(NULL_BANDS)), dtype=int)
    for i, row_band in enumerate(NULL_BANDS):
        for j, column_band in enumerate(NULL_BANDS):
            values = [(ifu_templates[(key, i)], ifu_templates[(key, j)]) for key in template_keys
                      if (key, i) in ifu_templates and (key, j) in ifu_templates]
            x = np.asarray([pair[0] for pair in values]); y = np.asarray([pair[1] for pair in values])
            template_counts[i, j] = int(len(values))
            template_matrix[i, j] = pearson_r(x, y)
            template_band_rows.append({"row_band": row_band, "column_band": column_band,
                                       "correlation": template_matrix[i, j], "N_IFU": template_counts[i, j]})
    persistence_paths = persistence_figures(persistence_dir, ifu_data, ifu_normalized, amp_data, amp_normalized,
                                            ifu_matrices, amp_matrices, ifu_template_rows, amp_template_rows,
                                            ifu_amplitudes + amp_amplitudes, centers, args.persistence_min_exposures)
    figure_paths.extend(persistence_paths)
    write_csv(persistence_dir / "persistence_pairwise_summary.csv", ifu_pair_rows + amp_pair_rows)
    write_csv(persistence_dir / "amplifier_persistence_pairwise.csv", amp_pair_rows)
    write_csv(persistence_dir / "ifu_persistence_matrices.csv", ifu_matrix_rows)
    write_csv(persistence_dir / "amplifier_persistence_matrices.csv", amp_matrix_rows)
    write_csv(persistence_dir / "persistent_ifu_template.csv", ifu_template_rows)
    write_csv(persistence_dir / "persistent_amplifier_template.csv", amp_template_rows)
    write_csv(persistence_dir / "persistent_ifu_template_performance.csv", ifu_performance)
    write_csv(persistence_dir / "persistent_amplifier_template_performance.csv", amp_performance)
    write_csv(persistence_dir / "persistent_amp_orientation_decomposition.csv", orientation_decomposition)
    write_csv(persistence_dir / "persistent_ifu_band_correlation.csv", template_band_rows)
    amplitude_rows = []
    for row in ifu_amplitudes + amp_amplitudes:
        amplitude_rows.append({"level": row["level"], "H5": row["exposure"].rsplit("_e", 1)[0] + ".h5",
                               "exposure": int(row["exposure"].rsplit("_e", 1)[1]),
                               "exposure_instance": row["exposure"], "null_band": row["null_band"],
                               "amplitude": row["amplitude"], "N": row["N"], "correlation": row["correlation"]})
    amplitude_by_key = {(row["exposure_instance"], row["null_band"], row["level"]): row for row in amplitude_rows}
    exposure_amplitude_rows = []
    for instance in sorted(ifu_data):
        for name in NULL_BANDS:
            u = amplitude_by_key.get((instance, name, "IFU"), {})
            a = amplitude_by_key.get((instance, name, "AMP"), {})
            exposure_amplitude_rows.append({"H5": u.get("H5", a.get("H5", "")), "exposure": u.get("exposure", a.get("exposure", "")),
                                            "exposure_instance": instance, "null_band": name,
                                            "A_IFU": u.get("amplitude", np.nan), "A_AMP": a.get("amplitude", np.nan),
                                            "N_IFU": u.get("N", 0), "N_AMP": a.get("N", 0),
                                            "correlation_IFU": u.get("correlation", np.nan), "correlation_AMP": a.get("correlation", np.nan)})
    write_csv(persistence_dir / "persistence_exposure_amplitudes.csv", exposure_amplitude_rows)
    fig, axis = plt.subplots(figsize=(6, 5), constrained_layout=True)
    image = axis.imshow(template_matrix, vmin=-1, vmax=1, cmap="coolwarm")
    axis.set_xticks(np.arange(len(NULL_BANDS)), [n.replace("NULL_", "") for n in NULL_BANDS], rotation=45, ha="right")
    axis.set_yticks(np.arange(len(NULL_BANDS)), [n.replace("NULL_", "") for n in NULL_BANDS])
    for i in range(len(NULL_BANDS)):
        for j in range(len(NULL_BANDS)):
            if np.isfinite(template_matrix[i, j]): axis.text(j, i, "%.2f" % template_matrix[i, j], ha="center", va="center", fontsize=8)
    axis.set_title("persistent IFU template band correlation"); fig.colorbar(image, ax=axis, label="Pearson r")
    template_band_figure = persistence_dir / "persistent_ifu_band_correlation.png"; fig.savefig(template_band_figure, dpi=140); plt.close(fig); figure_paths.append(str(template_band_figure))
    def persistence_aggregate(rows):
        result = {}
        for level in ("IFU", "AMP"):
            for comparison in ("within_H5", "cross_H5_same_exposure_index"):
                for band in NULL_BANDS:
                    selected_rows = [row for row in rows if row["level"] == level and row["comparison"] == comparison and row["null_band"] == band]
                    result["%s|%s|%s" % (level, comparison, band)] = {
                        "median_raw_Pearson_r": float(np.nanmedian([row["raw_Pearson_r"] for row in selected_rows])) if selected_rows else np.nan,
                        "median_normalized_Pearson_r": float(np.nanmedian([row["normalized_Pearson_r"] for row in selected_rows])) if selected_rows else np.nan,
                        "median_raw_Spearman_r": float(np.nanmedian([row["raw_Spearman_r"] for row in selected_rows])) if selected_rows else np.nan,
                        "median_normalized_Spearman_r": float(np.nanmedian([row["normalized_Spearman_r"] for row in selected_rows])) if selected_rows else np.nan,
                        "N_pair_comparisons": len(selected_rows)}
        return result
    within_h5_by_file = {}
    for h5_name in sorted({row["exposure_A"].rsplit("_e", 1)[0] for row in ifu_pair_rows + amp_pair_rows}):
        for level in ("IFU", "AMP"):
            for band in NULL_BANDS:
                selected_rows = [row for row in ifu_pair_rows + amp_pair_rows
                                 if row["level"] == level and row["comparison"] == "within_H5"
                                 and row["null_band"] == band and row["exposure_A"].startswith(h5_name + "_e")]
                within_h5_by_file["%s|%s|%s" % (h5_name, level, band)] = {
                    "median_raw_Pearson_r": float(np.nanmedian([row["raw_Pearson_r"] for row in selected_rows])) if selected_rows else np.nan,
                    "median_normalized_Pearson_r": float(np.nanmedian([row["normalized_Pearson_r"] for row in selected_rows])) if selected_rows else np.nan,
                    "median_raw_Spearman_r": float(np.nanmedian([row["raw_Spearman_r"] for row in selected_rows])) if selected_rows else np.nan,
                    "median_normalized_Spearman_r": float(np.nanmedian([row["normalized_Spearman_r"] for row in selected_rows])) if selected_rows else np.nan,
                    "N_pair_comparisons": len(selected_rows)}
    template_explained = {}
    for level, rows in (("IFU", ifu_performance), ("AMP", amp_performance)):
        for band in NULL_BANDS:
            values = [row["fraction_variance_explained"] for row in rows if row["null_band"] == band and np.isfinite(row["fraction_variance_explained"])]
            template_explained["%s|%s" % (level, band)] = float(np.nanmedian(values)) if values else np.nan
    orientation_reduction = {band: float(np.nanmedian([row["pooled_fractional_reduction"] for row in orientation_decomposition if row["null_band"] == band and np.isfinite(row["pooled_fractional_reduction"])])) for band in NULL_BANDS}
    persistence_summary = {"pairwise": persistence_aggregate(ifu_pair_rows + amp_pair_rows),
                           "within_h5_by_file": within_h5_by_file,
                           "template_explained_variance_median": template_explained,
                           "orientation_variance_reduction_median": orientation_reduction,
                           "persistent_ifu_band_correlation": template_matrix.tolist(),
                           "persistent_ifu_band_N": template_counts.tolist(),
                           "normalization_qa": {"IFU": ifu_norm_qa, "AMP": amp_norm_qa},
                           "n_persistent_ifu_template_rows": len(ifu_template_rows),
                           "n_persistent_amplifier_template_rows": len(amp_template_rows)}
    population_counts = {}
    for key in sorted({(r["h5"], r["exposure"]) for r in population}):
        population_counts["%s|%d" % key] = sum(1 for r in population if (r["h5"], r["exposure"]) == key)
    provenance = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "script": str(Path(__file__).resolve()),
        "input_h5": [file_identity(p) for p in h5_paths],
        "fit_h5": fit_identity,
        "external_blank_fibers": blank_provenance,
        "fq_template": fq_identity,
        "fit_additive_model": fit_metadata["additive_model"],
        "p_fixed_zero": fit_metadata["p_fixed_zero"],
        "wavelength_grid": {"array": "hm.DEF_WAVE", "n": int(DEF_WAVE.size), "min": float(DEF_WAVE[0]), "max": float(DEF_WAVE[-1])},
        "null_windows_A": {name: list(window) for name, window in zip(NULL_BANDS, NULL_WINDOWS)},
        "minimum_finite_fraction": args.minimum_finite_fraction,
        "population_min_blank_fraction": args.population_min_blank_fraction,
        "population_min_blank_fibers": args.population_min_blank_fibers,
        "population_min_blank_per_amp": args.population_min_blank_per_amp,
        "persistence_min_exposures": args.persistence_min_exposures,
        "persistence_exposure_instance_definition": "H5 basename + _e1/_e2/_e3",
        "persistent_IFU_identity": "SPECID + IFUSLOT + IFUID",
        "persistent_amplifier_identity": "SPECID + IFUSLOT + IFUID + AMP",
        "persistence_analysis": "descriptive exposure-to-exposure matching; no PCA/SVD or calibration correction",
        "population_analysis_basis": "pre-m Y = Fibers.spectrum/Survey.offset - alpha*K*f(q)",
        "IFU_reference": "leave-one-IFU-out robust exposure blank reference",
        "IFU_band_coefficient_definition": "robust_location of the pre-m IFU residual over each frozen null window",
        "amplifier_band_coefficient_definition": "robust_location of pre-m residual over usable blank fibers within one amplifier and each null window",
        "IFU_common_definition": "robust_location over valid amplifier coefficients within one IFU",
        "within_IFU_amp_deviation_definition": "a_AMP - c_IFU",
        "spatial_scale_definition": "1.4826*MAD, with standard deviation fallback when MAD is zero",
        "ifu_selection_algorithm": "highest blank fraction; spatially separated eligible IFU; lowest eligible blank fraction; ranked fallback",
        "spatial_map_nonblank_convention": "nonblank fibers are omitted; only usable externally blank fibers are colored",
        "spatial_coordinate_convention": "x=(RA-ra0)*cos(dec0)*3600 arcsec; y=(Dec-dec0)*3600 arcsec; no RA flip",
        "selected_ifus": selected_all,
        "selection_rationale": selection_rationale,
        "selection_reference": selection_reference,
        "calibration_equation": "C(lambda) = [Fibers.spectrum/Survey.offset - alpha_mean * raw_work_basis(lambda) * f(q)] / exp(posterior_z_mean)",
        "pre_m_quantity": "Y = Fibers.spectrum/Survey.offset - alpha_mean * raw_work_basis(lambda) * f(q)",
        "leave_one_IFU_out_reference": "robust_location(Y from external blank fibers outside selected IFU)",
        "corrected_fiber_residual": "(Y - reference) / exp(posterior_z_mean)",
        "raw_residual": "W - robust_location(W from external blank fibers outside selected IFU)",
        "multiplication_only_residual": "[W - robust_location(W)] / exp(posterior_z_mean)",
        "residual_sky_removed_before_multiplicative_calibration": True,
        "Fibers.skyspectrum_not_used": True,
        "no_new_fit": True,
        "no_cube_reconstruction": True,
        "no_pca_svd": True,
        "population_counts": population_counts,
        "population_plot_metadata": population_plot_metadata,
        "persistence_summary": persistence_summary,
        "hierarchy_qa": hierarchy_qa,
        "ordering_sanity": ordering
    }
    provenance_path = output_dir / "blank_ifu_transfer_provenance.json"
    provenance_path.write_text(json.dumps(provenance, indent=2, default=str))
    summary = {"script_lines": sum(1 for _ in Path(__file__).open()),
               "runtime_seconds": time.perf_counter() - started,
               "n_input_h5": len(h5_paths), "n_selected_ifus": len(records),
               "n_population_ifus": len(population), "population_counts": population_counts,
               "selected_summary": summary_rows, "aggregate": aggregate_summary(summary_rows),
               "generated_figures": figure_paths, "generated_spectral_csvs": len(records),
               "population_products": {"n_population_spectra_rows": len(population_spectrum_rows_all),
                                       "correlation_matrices": products["correlation_matrices"]},
               "persistence_summary": persistence_summary,
               "hierarchy_qa": hierarchy_qa, "ordering_sanity": ordering,
               "selection_reference": selection_reference,
               "provenance": str(provenance_path), "raw_comparison": True,
               "multiplication_only_comparison": True}
    (output_dir / "blank_ifu_transfer_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
