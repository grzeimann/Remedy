#!/usr/bin/env python3
"""Census classified blank-fiber support per M101 amplifier and exposure.

This is deliberately a cache-only information-gathering diagnostic.  It reads
the compact M101 band cache and its manifest, counts existing blank and
hardware-valid rows in 40 <= q <= 75, and writes small descriptive tables and
plots.  It does not open native H5 files, fit a model, or change calibration
state.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


CACHE_SCHEMA = "m101_band_cache_v1"
Q_MIN = 40
Q_MAX = 75
AMP_ORDER = ("LL", "LU", "RL", "RU")
AMP_INDEX = {name: index for index, name in enumerate(AMP_ORDER)}
THRESHOLDS = (1, 2, 3, 5, 10, 15, 20, 25, 30)


def _json_ready(value):
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_json_ready(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _json_ready(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _file_identity(path):
    path = Path(path).expanduser().resolve()
    stat = path.stat()
    return {
        "filename": path.name,
        "full_path": str(path),
        "exists": True,
        "file_size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _write_rows(path, rows):
    path = Path(path)
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
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
                if isinstance(value, (tuple, list, np.ndarray)):
                    value = json.dumps(_json_ready(value), separators=(",", ":"))
                clean[field] = value
            writer.writerow(clean)


def _finite(values):
    values = np.asarray(values, dtype=float)
    return values[np.isfinite(values)]


def _distribution(values):
    values = _finite(values)
    if not values.size:
        return {
            "min": np.nan,
            "p10": np.nan,
            "p25": np.nan,
            "median": np.nan,
            "p75": np.nan,
            "p90": np.nan,
            "max": np.nan,
        }
    return {
        "min": float(np.min(values)),
        "p10": float(np.percentile(values, 10)),
        "p25": float(np.percentile(values, 25)),
        "median": float(np.median(values)),
        "p75": float(np.percentile(values, 75)),
        "p90": float(np.percentile(values, 90)),
        "max": float(np.max(values)),
    }


def _load_manifest(cache_path):
    manifest_path = cache_path.with_suffix(cache_path.suffix + ".json")
    if not cache_path.exists() or not manifest_path.exists():
        raise FileNotFoundError(
            "band cache and adjacent manifest are required: %s" % cache_path)
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema_version") != CACHE_SCHEMA:
        raise ValueError("unsupported cache schema: %s" % manifest.get("schema_version"))
    items = list(manifest.get("items", ()))
    if not items:
        raise ValueError("band-cache manifest has no exposure items")
    return manifest, manifest_path


def _load_census_rows(cache_path, manifest):
    """Count cache rows without importing or opening any native-data module."""
    required = (
        "ifu", "ifu_code", "amp", "q", "blank_classified", "blank_valid",
        "hardware_bad", "source_candidate",
    )
    with np.load(cache_path, allow_pickle=False) as archive:
        missing = [name for name in required if name not in archive.files]
        if missing:
            raise ValueError("band cache is missing required fields: %s" % missing)
        arrays = {name: np.asarray(archive[name]) for name in required}

    n_rows = int(arrays["q"].size)
    if arrays["ifu"].shape != (n_rows, 3):
        raise ValueError("cache ifu shape is inconsistent with q")
    for name in ("ifu_code", "amp", "q", "blank_classified", "blank_valid", "hardware_bad"):
        if arrays[name].shape != (n_rows,):
            raise ValueError("cache field %s has an inconsistent shape" % name)
    if arrays["source_candidate"].shape[0] != n_rows:
        raise ValueError("cache source_candidate shape is inconsistent with q")

    rows = []
    hardware_scope_failures = []
    blank_subset_failure_count = 0
    blank_hardware_failure_count = 0
    observed_items = []

    for item in manifest["items"]:
        h5 = str(item["h5_name"])
        exposure = int(item["exposure"])
        start = int(item["start"])
        stop = int(item["stop"])
        if start < 0 or stop > n_rows or stop <= start:
            raise ValueError("invalid cache block for %s exposure %d" % (h5, exposure))
        sl = slice(start, stop)
        ifu = arrays["ifu"][sl]
        ifu_code = arrays["ifu_code"][sl]
        amp = arrays["amp"][sl]
        q = arrays["q"][sl]
        blank_classified = arrays["blank_classified"][sl].astype(bool, copy=False)
        blank_valid = arrays["blank_valid"][sl].astype(bool, copy=False)
        hardware_bad = arrays["hardware_bad"][sl].astype(bool, copy=False)
        source_candidate = arrays["source_candidate"][sl]
        central = (q >= Q_MIN) & (q <= Q_MAX)

        if np.any(blank_valid & ~blank_classified):
            blank_subset_failure_count += int(np.sum(blank_valid & ~blank_classified))
        if np.any(blank_valid & hardware_bad):
            blank_hardware_failure_count += int(np.sum(blank_valid & hardware_bad))

        physical_values = np.unique(np.column_stack((ifu, ifu_code)), axis=0)
        for physical_value in physical_values:
            physical = tuple(int(value) for value in physical_value[:3])
            code = int(physical_value[3])
            physical_mask = np.all(ifu == np.asarray(physical), axis=1) & (ifu_code == code)
            for amp_index, amp_name in enumerate(AMP_ORDER):
                group = physical_mask & (amp == amp_index)
                amplifier_exists = bool(np.any(group))
                if not amplifier_exists:
                    rows.append({
                        "H5": h5,
                        "exposure": exposure,
                        "SPECID": physical[0],
                        "IFUSLOT": physical[1],
                        "IFUID": physical[2],
                        "IFU_CODE": code,
                        "AMP": amp_name,
                        "N_total_q40_75": 0,
                        "N_hardware_valid_q40_75": 0,
                        "N_hardware_invalid_q40_75": 0,
                        "N_blank_classified_q40_75": 0,
                        "N_blank_valid_q40_75": 0,
                        "N_blank": 0,
                        "blank_fraction_of_valid": np.nan,
                        "N_source_candidate_q40_75": 0,
                        "amplifier_exists_in_cache": False,
                        "amplifier_status": "unavailable",
                    })
                    continue

                if amp_index < 0 or amp_index >= len(AMP_ORDER):
                    raise ValueError("unexpected cache amplifier index: %d" % amp_index)
                hardware_values = np.unique(hardware_bad[group])
                if hardware_values.size > 1:
                    hardware_scope_failures.append({
                        "H5": h5,
                        "exposure": exposure,
                        "SPECID": physical[0],
                        "IFUSLOT": physical[1],
                        "IFUID": physical[2],
                        "AMP": AMP_ORDER[amp_index],
                    })

                selected = group & central
                n_total = int(np.sum(selected))
                n_hardware_valid = int(np.sum(selected & ~hardware_bad))
                n_blank_classified = int(np.sum(selected & blank_classified))
                n_blank_valid = int(np.sum(selected & blank_valid))
                n_source_candidate = int(np.sum(
                    selected & np.any(np.asarray(source_candidate, dtype=bool), axis=1)))

                if n_total == 0:
                    status = "no_central_q_rows"
                elif n_hardware_valid == 0:
                    status = "hardware_excluded"
                else:
                    status = "usable"

                rows.append({
                    "H5": h5,
                    "exposure": exposure,
                    "SPECID": physical[0],
                    "IFUSLOT": physical[1],
                    "IFUID": physical[2],
                    "IFU_CODE": code,
                    "AMP": AMP_ORDER[amp_index],
                    "N_total_q40_75": n_total,
                    "N_hardware_valid_q40_75": n_hardware_valid,
                    "N_hardware_invalid_q40_75": n_total - n_hardware_valid,
                    "N_blank_classified_q40_75": n_blank_classified,
                    "N_blank_valid_q40_75": n_blank_valid,
                    "N_blank": n_blank_valid,
                    "blank_fraction_of_valid": (
                        float(n_blank_valid / n_hardware_valid)
                        if n_hardware_valid else np.nan),
                    "N_source_candidate_q40_75": n_source_candidate,
                    "amplifier_exists_in_cache": True,
                    "amplifier_status": status,
                })
        observed_items.append((h5, exposure))

    if hardware_scope_failures:
        sample = hardware_scope_failures[:3]
        raise ValueError(
            "hardware_bad is not amplifier-scoped in cache; examples: %s" % sample)
    if blank_subset_failure_count:
        raise ValueError(
            "blank_valid is not a subset of blank_classified (%d rows)" %
            blank_subset_failure_count)
    if blank_hardware_failure_count:
        raise ValueError(
            "blank_valid includes hardware_bad rows (%d rows)" %
            blank_hardware_failure_count)

    expected_total_rows = int(manifest.get("total_rows", n_rows))
    if expected_total_rows != n_rows:
        raise ValueError(
            "manifest total_rows=%d disagrees with cache=%d" %
            (expected_total_rows, n_rows))
    return rows, {
        "cache_rows": n_rows,
        "observed_items": observed_items,
        "hardware_bad_amplifier_scoped": True,
        "blank_valid_subset_of_classified": True,
        "blank_valid_hardware_clean": True,
    }


def _usable_rows(rows):
    return [row for row in rows if row["amplifier_status"] == "usable"]


def _threshold_rows(usable):
    values = np.asarray([row["N_blank"] for row in usable], dtype=int)
    denominator = int(values.size)
    result = []
    zero_count = int(np.sum(values == 0))
    result.append({
        "threshold": 0,
        "criterion": "N_blank == 0",
        "count": zero_count,
        "fraction_of_usable": zero_count / denominator if denominator else np.nan,
    })
    for threshold in THRESHOLDS:
        count = int(np.sum(values >= threshold))
        result.append({
            "threshold": threshold,
            "criterion": "N_blank >= %d" % threshold,
            "count": count,
            "fraction_of_usable": count / denominator if denominator else np.nan,
        })
    return result


def _exposure_summary(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["H5"], int(row["exposure"]))].append(row)
    result = []
    for (h5, exposure), group in sorted(grouped.items()):
        usable = [row for row in group if row["amplifier_status"] == "usable"]
        values = np.asarray([row["N_blank"] for row in usable], dtype=float)
        stats = _distribution(values)
        result.append({
            "H5": h5,
            "exposure": exposure,
            "N_amplifier_exposure_rows": len(group),
            "N_usable_amplifiers": len(usable),
            "N_hardware_excluded_amplifiers": sum(
                row["amplifier_status"] == "hardware_excluded" for row in group),
            "N_no_central_q_rows_amplifiers": sum(
                row["amplifier_status"] == "no_central_q_rows" for row in group),
            "median_N_blank": stats["median"],
            "p10_N_blank": stats["p10"],
            "p25_N_blank": stats["p25"],
            "p75_N_blank": stats["p75"],
            "p90_N_blank": stats["p90"],
            "N_blank_zero": int(np.sum(values == 0)),
            "N_blank_ge5": int(np.sum(values >= 5)),
            "N_blank_ge10": int(np.sum(values >= 10)),
            "N_blank_ge20": int(np.sum(values >= 20)),
            "N_blank_ge30": int(np.sum(values >= 30)),
        })
    return result


def _physical_summary(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(int(row["SPECID"]), int(row["IFUSLOT"]),
                 int(row["IFUID"]), row["AMP"])].append(row)
    result = []
    for identity, group in sorted(grouped.items()):
        specid, ifuslot, ifuid, amp = identity
        usable = [row for row in group if row["amplifier_status"] == "usable"]
        values = np.asarray([row["N_blank"] for row in usable], dtype=float)
        stats = _distribution(values)
        codes = sorted({int(row["IFU_CODE"]) for row in group})
        denominator = len(usable)
        result.append({
            "SPECID": specid,
            "IFUSLOT": ifuslot,
            "IFUID": ifuid,
            "IFU_CODE": codes[0] if len(codes) == 1 else ";".join(map(str, codes)),
            "IFU_CODE_values": ";".join(map(str, codes)),
            "AMP": amp,
            "N_exposures_observed": len(group),
            "N_exposures_hardware_valid": len(usable),
            "N_exposures_hardware_excluded": sum(
                row["amplifier_status"] == "hardware_excluded" for row in group),
            "N_exposures_no_central_q_rows": sum(
                row["amplifier_status"] == "no_central_q_rows" for row in group),
            "median_N_blank": stats["median"],
            "min_N_blank": stats["min"],
            "max_N_blank": stats["max"],
            "p10_N_blank": stats["p10"],
            "p90_N_blank": stats["p90"],
            "fraction_valid_exposures_ge1": (
                sum(row["N_blank"] >= 1 for row in usable) / denominator
                if denominator else np.nan),
            "fraction_valid_exposures_ge5": (
                sum(row["N_blank"] >= 5 for row in usable) / denominator
                if denominator else np.nan),
            "fraction_valid_exposures_ge10": (
                sum(row["N_blank"] >= 10 for row in usable) / denominator
                if denominator else np.nan),
            "fraction_valid_exposures_ge20": (
                sum(row["N_blank"] >= 20 for row in usable) / denominator
                if denominator else np.nan),
            "fraction_valid_exposures_ge30": (
                sum(row["N_blank"] >= 30 for row in usable) / denominator
                if denominator else np.nan),
        })
    return result


def _make_plots(output_dir, usable, exposure_rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    values = np.asarray([row["N_blank"] for row in usable], dtype=int)
    if values.size:
        maximum = int(np.max(values))
        bins = np.arange(-0.5, maximum + 1.5, 1.0)
        figure, axis = plt.subplots(figsize=(9, 5))
        axis.hist(values, bins=bins, color="#4472a8", edgecolor="white", linewidth=.4)
        axis.set_xlabel("N blank valid fibers, q=40--75")
        axis.set_ylabel("usable amplifier/exposure count")
        axis.set_title("M101 blank-fiber support per usable amplifier/exposure")
        axis.grid(axis="y", alpha=.2)
        figure.tight_layout()
        figure.savefig(output_dir / "blank_fiber_histogram.png", dpi=130)
        plt.close(figure)

        support = np.arange(0, maximum + 1, dtype=int)
        survival = np.asarray([np.mean(values >= threshold) for threshold in support])
        figure, axis = plt.subplots(figsize=(9, 5))
        axis.step(support, survival, where="post", color="#b23a48")
        axis.scatter(support, survival, s=10, color="#b23a48")
        axis.set_xlabel("N blank valid fibers threshold")
        axis.set_ylabel("fraction with N_blank >= threshold")
        axis.set_ylim(0, 1.02)
        axis.set_title("Cumulative blank-fiber support")
        axis.grid(alpha=.2)
        figure.tight_layout()
        figure.savefig(output_dir / "blank_fiber_survival.png", dpi=130)
        plt.close(figure)

    by_exposure = []
    labels = []
    for row in exposure_rows:
        matching = [item for item in usable
                    if item["H5"] == row["H5"] and item["exposure"] == row["exposure"]]
        if not matching:
            continue
        by_exposure.append([item["N_blank"] for item in matching])
        labels.append("%s/e%d" % (Path(row["H5"]).stem, row["exposure"]))
    if by_exposure:
        figure_width = max(12.0, 0.28 * len(by_exposure))
        figure, axis = plt.subplots(figsize=(figure_width, 6))
        axis.boxplot(by_exposure, positions=np.arange(1, len(by_exposure) + 1),
                     widths=.7, showfliers=False)
        axis.set_xticks(np.arange(1, len(labels) + 1))
        axis.set_xticklabels(labels, rotation=90, fontsize=6)
        axis.set_xlabel("H5 / exposure")
        axis.set_ylabel("N blank valid fibers, q=40--75")
        axis.set_title("Blank-fiber support by exposure")
        axis.grid(axis="y", alpha=.2)
        figure.tight_layout()
        figure.savefig(output_dir / "blank_fiber_by_exposure.png", dpi=120)
        plt.close(figure)


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--band-cache", default="m101_band_initializer/m101_band_cache.npz")
    parser.add_argument("--output-dir", default="m101_blank_fiber_census")
    parser.add_argument("--no-plots", action="store_true",
                        help="write tables/state without PNG plots")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = _parse_args()
    cache_path = Path(args.band_cache).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise SystemExit("output directory is nonempty; use --overwrite: %s" % output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest, manifest_path = _load_manifest(cache_path)
    rows, load_validation = _load_census_rows(cache_path, manifest)
    usable = _usable_rows(rows)
    threshold_rows = _threshold_rows(usable)
    exposure_rows = _exposure_summary(rows)
    physical_rows = _physical_summary(rows)

    rows.sort(key=lambda row: (
        row["H5"], int(row["exposure"]), int(row["SPECID"]),
        int(row["IFUSLOT"]), int(row["IFUID"]), row["AMP"]))
    _write_rows(output_dir / "blank_fiber_amp_census.csv", rows)
    _write_rows(output_dir / "blank_fiber_threshold_summary.csv", threshold_rows)
    _write_rows(output_dir / "blank_fiber_exposure_summary.csv", exposure_rows)
    _write_rows(output_dir / "blank_fiber_physical_amp_summary.csv", physical_rows)
    if not args.no_plots:
        _make_plots(output_dir, usable, exposure_rows)

    values = np.asarray([row["N_blank"] for row in usable], dtype=float)
    overall_distribution = _distribution(values)
    h5_names = sorted({row["H5"] for row in rows})
    exposure_keys = sorted({(row["H5"], int(row["exposure"])) for row in rows})
    manifest_h5_names = sorted({str(item["h5_name"]) for item in manifest["items"]})
    expected_exposure_keys = sorted(
        (str(item["h5_name"]), int(item["exposure"])) for item in manifest["items"])
    population_matches_manifest = (
        h5_names == manifest_h5_names and exposure_keys == expected_exposure_keys)

    validation = {
        "cache_only": True,
        "native_spectra_read": False,
        "q_range_exact_inclusive": [Q_MIN, Q_MAX],
        "blank_classification_reused": True,
        "blank_fields_used": ["blank_classified", "blank_valid"],
        "hardware_field_used": "hardware_bad",
        "hardware_exclusions_amplifier_scoped": load_validation[
            "hardware_bad_amplifier_scoped"],
        "blank_valid_subset_of_blank_classified": load_validation[
            "blank_valid_subset_of_classified"],
        "blank_valid_hardware_clean": load_validation["blank_valid_hardware_clean"],
        "zero_blank_usable_rows_included": all(
            row in rows for row in rows
            if row["amplifier_status"] == "usable" and row["N_blank"] == 0),
        "counts_within_central_q_rows": all(
            0 <= row["N_blank"] <= row["N_hardware_valid_q40_75"]
            <= row["N_total_q40_75"] for row in rows),
        "population_matches_cache_manifest": population_matches_manifest,
        "no_calibration_state_changed": True,
    }
    validation_passed = all(
        value for key, value in validation.items()
        if key != "native_spectra_read"
    ) and not validation["native_spectra_read"]
    if not validation_passed:
        raise RuntimeError("cache census validation failed: %s" % validation)

    state = {
        "schema_version": "m101_blank_fiber_census_v1",
        "script": str(Path(__file__).resolve()),
        "purpose": "cache-only descriptive census of classified blank fibers per amplifier/exposure",
        "cache_provenance": _file_identity(cache_path),
        "manifest_provenance": _file_identity(manifest_path),
        "q_range_inclusive": [Q_MIN, Q_MAX],
        "blank_fields_used": ["blank_classified", "blank_valid"],
        "hardware_field_used": "hardware_bad",
        "source_candidate_field_used": "source_candidate",
        "native_spectra_read": False,
        "counts": {
            "H5_count": len(h5_names),
            "exposure_count": len(exposure_keys),
            "amplifier_exposure_row_count": len(rows),
            "usable_amplifier_exposure_count": len(usable),
            "hardware_excluded_amplifier_exposure_count": sum(
                row["amplifier_status"] == "hardware_excluded" for row in rows),
            "no_central_q_rows_amplifier_exposure_count": sum(
                row["amplifier_status"] == "no_central_q_rows" for row in rows),
            "cache_row_count": load_validation["cache_rows"],
        },
        "overall_distribution_usable_N_blank": overall_distribution,
        "threshold_summary": threshold_rows,
        "validation": validation,
        "plots": {
            "written": not args.no_plots,
            "histogram": str((output_dir / "blank_fiber_histogram.png").resolve())
            if not args.no_plots else None,
            "survival": str((output_dir / "blank_fiber_survival.png").resolve())
            if not args.no_plots else None,
            "by_exposure": str((output_dir / "blank_fiber_by_exposure.png").resolve())
            if not args.no_plots else None,
            "focal_plane_map": None,
        },
        "artifacts": {
            "amp_census": str((output_dir / "blank_fiber_amp_census.csv").resolve()),
            "threshold_summary": str((output_dir / "blank_fiber_threshold_summary.csv").resolve()),
            "exposure_summary": str((output_dir / "blank_fiber_exposure_summary.csv").resolve()),
            "physical_amp_summary": str((output_dir / "blank_fiber_physical_amp_summary.csv").resolve()),
        },
    }
    (output_dir / "blank_fiber_census_state.json").write_text(
        json.dumps(_json_ready(state), indent=2, sort_keys=True))

    zero_row = next(row for row in threshold_rows if row["criterion"] == "N_blank == 0")
    threshold_lookup = {row["threshold"]: row for row in threshold_rows
                        if row["criterion"].startswith("N_blank >=")}
    print("H5 count: %d" % len(h5_names))
    print("exposure count: %d" % len(exposure_keys))
    print("usable amplifier/exposure combinations: %d" % len(usable))
    print("blank fibers per amp (N_blank_valid_q40_75):")
    for name in ("min", "p10", "p25", "median", "p75", "p90", "max"):
        print("  %-6s %s" % (name, "%.6g" % overall_distribution[name]))
    print("threshold summary:")
    for threshold in (1, 5, 10, 20, 30):
        row = threshold_lookup[threshold]
        print("  N_blank >= %-2d: %d (%.2f%%)" % (
            threshold, row["count"], 100.0 * row["fraction_of_usable"]))
    print("  N_blank == 0 : %d (%.2f%%)" % (
        zero_row["count"], 100.0 * zero_row["fraction_of_usable"]))

    ordered = sorted(usable, key=lambda row: (
        row["N_blank"], row["H5"], int(row["exposure"]),
        int(row["SPECID"]), int(row["IFUSLOT"]), int(row["IFUID"]), row["AMP"]))
    print("10 amplifier/exposure rows with fewest blank fibers:")
    for row in ordered[:10]:
        print("  %s e%d SPECID=%d IFUSLOT=%d IFUID=%d %s: N_blank=%d" % (
            row["H5"], row["exposure"], row["SPECID"], row["IFUSLOT"],
            row["IFUID"], row["AMP"], row["N_blank"]))
    print("10 amplifier/exposure rows with most blank fibers:")
    for row in ordered[-10:][::-1]:
        print("  %s e%d SPECID=%d IFUSLOT=%d IFUID=%d %s: N_blank=%d" % (
            row["H5"], row["exposure"], row["SPECID"], row["IFUSLOT"],
            row["IFUID"], row["AMP"], row["N_blank"]))
    print("wrote cache-only census products to %s" % output_dir)


if __name__ == "__main__":
    main()
