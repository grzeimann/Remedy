#!/usr/bin/env python3
"""Leave-one-H5-out test of historical same-amplifier alpha prediction.

This is intentionally a standalone analysis of the existing
``alpha_template_fits.csv`` product.  It does not read the band cache or
native spectra and does not refit any alpha values.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from math_utils import biweight


PREDICTORS = ("zero", "global", "same_amp")
ERROR_THRESHOLDS = (0.148, 0.296, 0.444)
DEFAULT_GALLERY_SIZE = 12


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


def _robust_location(values):
    values = _finite(values)
    if not values.size:
        return np.nan
    if values.size == 1 or np.all(values == values[0]):
        return float(values[0])
    result = biweight(values)
    if np.isfinite(result):
        return float(result)
    return float(np.nanmedian(values))


def _robust_scatter(values):
    values = _finite(values)
    if values.size < 2:
        return np.nan
    if np.all(values == values[0]):
        return 0.0
    _, scale = biweight(values, calc_std=True)
    if np.isfinite(scale) and scale > 0:
        return float(scale)
    return float(np.nanstd(values))


def _correlation(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    good = np.isfinite(x) & np.isfinite(y)
    if np.sum(good) < 3:
        return np.nan
    x = x[good]
    y = y[good]
    if np.std(x) <= 0 or np.std(y) <= 0:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def _distribution(values):
    values = _finite(values)
    if not values.size:
        return {name: np.nan for name in (
            "N", "median", "robust_scatter", "p01", "p05", "p10", "p16",
            "p25", "p75", "p84", "p90", "p95", "p99", "min", "max")}
    return {
        "N": int(values.size),
        "median": float(np.median(values)),
        "robust_scatter": _robust_scatter(values),
        "p01": float(np.percentile(values, 1)),
        "p05": float(np.percentile(values, 5)),
        "p10": float(np.percentile(values, 10)),
        "p16": float(np.percentile(values, 16)),
        "p25": float(np.percentile(values, 25)),
        "p75": float(np.percentile(values, 75)),
        "p84": float(np.percentile(values, 84)),
        "p90": float(np.percentile(values, 90)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def _physical_amp_key(row):
    return (int(row["SPECID"]), int(row["IFUSLOT"]), int(row["IFUID"]),
            str(row["AMP"]))


def _physical_amp_label(key):
    return "%d/%d/%d/%s" % key


def _read_fit_rows(path):
    required = {"H5", "exposure", "SPECID", "IFUSLOT", "IFUID", "IFU_CODE",
                "AMP", "alpha_joint"}
    rows = []
    total_rows = 0
    finite_rows = 0
    with Path(path).open(newline="") as stream:
        reader = csv.DictReader(stream)
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError("alpha fit CSV is missing fields: %s" % sorted(missing))
        for raw in reader:
            total_rows += 1
            try:
                alpha = float(raw["alpha_joint"])
            except (TypeError, ValueError):
                continue
            if not np.isfinite(alpha):
                continue
            rows.append({
                "H5": str(raw["H5"]),
                "exposure": int(raw["exposure"]),
                "SPECID": int(raw["SPECID"]),
                "IFUSLOT": int(raw["IFUSLOT"]),
                "IFUID": int(raw["IFUID"]),
                "IFU_CODE": str(raw["IFU_CODE"]),
                "AMP": str(raw["AMP"]),
                "alpha_exposure": alpha,
            })
            finite_rows += 1
    if not rows:
        raise ValueError("no finite alpha_joint rows found in %s" % path)
    return rows, {"total_rows": total_rows, "finite_rows": finite_rows}


def _collapse_h5(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(_physical_amp_key(row), row["H5"])].append(row)

    collapsed = []
    for (amp_key, h5), entries in sorted(grouped.items(),
                                         key=lambda item: (item[0][0], item[0][1])):
        entries = sorted(entries, key=lambda row: row["exposure"])
        alpha_values = np.asarray([row["alpha_exposure"] for row in entries],
                                  dtype=float)
        metadata = entries[0]
        if len({row["IFU_CODE"] for row in entries}) != 1:
            raise ValueError("inconsistent IFU_CODE within %s/%s" % (amp_key, h5))
        collapsed.append({
            "H5": h5,
            "SPECID": amp_key[0],
            "IFUSLOT": amp_key[1],
            "IFUID": amp_key[2],
            "IFU_CODE": metadata["IFU_CODE"],
            "AMP": amp_key[3],
            "alpha_h5": _robust_location(alpha_values),
            "within_h5_alpha_scatter": _robust_scatter(alpha_values),
            "N_exposures": int(alpha_values.size),
            "exposures": ",".join(str(row["exposure"]) for row in entries),
        })
    return collapsed


def _make_loo_rows(collapsed):
    by_amp = defaultdict(list)
    for row in collapsed:
        by_amp[_physical_amp_key(row)].append(row)
    all_h5 = sorted({row["H5"] for row in collapsed})
    loo_rows = []
    for row in collapsed:
        amp_key = _physical_amp_key(row)
        training_same_amp = [
            other["alpha_h5"] for other in by_amp[amp_key]
            if other["H5"] != row["H5"] and np.isfinite(other["alpha_h5"])
        ]
        if len(training_same_amp) < 2:
            continue
        global_training = [
            other["alpha_h5"] for other in collapsed
            if other["H5"] != row["H5"] and np.isfinite(other["alpha_h5"])
        ]
        if not global_training:
            continue
        alpha_pred_same_amp = _robust_location(training_same_amp)
        alpha_pred_global = float(np.median(global_training))
        measured = row["alpha_h5"]
        predictions = {
            "zero": 0.0,
            "global": alpha_pred_global,
            "same_amp": alpha_pred_same_amp,
        }
        result = dict(row)
        result.update({
            "N_training_H5": int(len(training_same_amp)),
            "N_global_training_H5": int(len(set(
                other["H5"] for other in collapsed if other["H5"] != row["H5"]))),
            "alpha_pred_zero": predictions["zero"],
            "alpha_pred_global": predictions["global"],
            "alpha_pred_same_amp": predictions["same_amp"],
        })
        for predictor, predicted in predictions.items():
            result["error_" + predictor] = measured - predicted
        loo_rows.append(result)
    return loo_rows, by_amp, all_h5


def _summary_rows(loo_rows, physical_summary):
    rows = []
    subsets = {
        "all_finite": lambda measured: np.isfinite(measured),
        "alpha_gt_0.30": lambda measured: measured > 0.30,
        "alpha_gt_0.44": lambda measured: measured > 0.44,
        "alpha_gt_0.80": lambda measured: measured > 0.80,
    }
    measured = np.asarray([row["alpha_h5"] for row in loo_rows], dtype=float)
    for subset_name, selector in subsets.items():
        subset_mask = selector(measured)
        for predictor in PREDICTORS:
            predicted = np.asarray([
                row["alpha_pred_" + predictor] for row in loo_rows], dtype=float)
            errors = measured - predicted
            good = subset_mask & np.isfinite(errors) & np.isfinite(predicted)
            error_values = errors[good]
            abs_error = np.abs(error_values)
            row = {
                "summary_type": "prediction_performance",
                "subset": subset_name,
                "predictor": predictor,
                "threshold": ("all" if subset_name == "all_finite"
                              else subset_name.replace("alpha_gt_", ">")),
                "N": int(error_values.size),
                "bias": _robust_location(error_values),
                "robust_scatter": _robust_scatter(error_values),
                "median_absolute_error": (
                    float(np.median(abs_error)) if abs_error.size else np.nan),
                "correlation_predicted_measured": _correlation(
                    predicted[good], measured[good]),
            }
            for threshold in ERROR_THRESHOLDS:
                row["fraction_abs_error_lt_%.3f" % threshold] = (
                    float(np.mean(abs_error < threshold)) if abs_error.size else np.nan)
            rows.append(row)

    for metric, values in (
            ("within_h5_alpha_scatter",
             [row["within_h5_alpha_scatter"] for row in physical_summary["h5_rows"]]),
            ("between_h5_alpha_scatter",
             [row["between_h5_alpha_scatter"] for row in physical_summary["amp_rows"]]),
            ("historical_alpha_robust_location",
             [row["historical_alpha_robust_location"]
              for row in physical_summary["amp_rows"]])):
        row = {"summary_type": "population_distribution", "metric": metric}
        row.update(_distribution(values))
        rows.append(row)
    return rows


def _physical_summary(collapsed):
    by_amp = defaultdict(list)
    for row in collapsed:
        by_amp[_physical_amp_key(row)].append(row)
    amp_rows = []
    for amp_key, records in sorted(by_amp.items()):
        alpha_h5 = np.asarray([row["alpha_h5"] for row in records], dtype=float)
        within = np.asarray([row["within_h5_alpha_scatter"] for row in records],
                            dtype=float)
        metadata = records[0]
        amp_rows.append({
            "SPECID": amp_key[0], "IFUSLOT": amp_key[1], "IFUID": amp_key[2],
            "IFU_CODE": metadata["IFU_CODE"], "AMP": amp_key[3],
            "N_H5": int(alpha_h5.size),
            "N_exposure_rows": int(np.sum([row["N_exposures"] for row in records])),
            "historical_alpha_robust_location": _robust_location(alpha_h5),
            "historical_alpha_median": (float(np.median(alpha_h5))
                                         if alpha_h5.size else np.nan),
            "historical_alpha_p10": (float(np.percentile(alpha_h5, 10))
                                      if alpha_h5.size else np.nan),
            "historical_alpha_p90": (float(np.percentile(alpha_h5, 90))
                                      if alpha_h5.size else np.nan),
            "between_h5_alpha_scatter": _robust_scatter(alpha_h5),
            "N_within_h5_scatter": int(np.sum(np.isfinite(within))),
            "within_h5_scatter_median": (
                float(np.nanmedian(within)) if np.any(np.isfinite(within)) else np.nan),
            "within_h5_scatter_robust_scatter": _robust_scatter(within),
            "within_h5_scatter_p90": (
                float(np.nanpercentile(within, 90))
                if np.any(np.isfinite(within)) else np.nan),
        })
    return {
        "amp_rows": amp_rows,
        "h5_rows": collapsed,
    }


def _make_plots(output_dir, loo_rows, physical_summary, gallery_size):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    measured = np.asarray([row["alpha_h5"] for row in loo_rows], dtype=float)
    same_amp = np.asarray([row["alpha_pred_same_amp"] for row in loo_rows], dtype=float)
    figure, axis = plt.subplots(figsize=(6, 6))
    axis.scatter(same_amp, measured, s=12, alpha=.45, color="#4472a8",
                 edgecolors="none", rasterized=True)
    finite = np.isfinite(same_amp) & np.isfinite(measured)
    if np.any(finite):
        lo = min(np.min(same_amp[finite]), np.min(measured[finite]))
        hi = max(np.max(same_amp[finite]), np.max(measured[finite]))
        pad = max(.03, .05 * (hi - lo))
        axis.plot([lo - pad, hi + pad], [lo - pad, hi + pad],
                  color="black", ls="--", lw=.9, label="1:1")
    axis.set_xlabel("same-amplifier LOO predicted alpha")
    axis.set_ylabel("held-out measured alpha")
    axis.set_title("Held-out alpha versus same-amplifier prediction")
    axis.grid(alpha=.2)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "alpha_same_amp_loo_measured_vs_predicted.png",
                   dpi=130)
    plt.close(figure)

    figure, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
    all_errors = np.concatenate([
        np.asarray([row["error_" + predictor] for row in loo_rows], dtype=float)
        for predictor in PREDICTORS])
    finite_errors = _finite(all_errors)
    if finite_errors.size:
        lo = float(np.percentile(finite_errors, .5))
        hi = float(np.percentile(finite_errors, 99.5))
        if hi <= lo:
            lo, hi = float(np.min(finite_errors)), float(np.max(finite_errors))
        if hi <= lo:
            lo, hi = lo - .1, hi + .1
        bins = np.linspace(lo, hi, 31)
    else:
        bins = 30
    for axis, predictor in zip(axes, PREDICTORS):
        errors = _finite([row["error_" + predictor] for row in loo_rows])
        axis.hist(errors, bins=bins, color="#4c72b0", edgecolor="white")
        axis.axvline(0, color="black", lw=.8)
        axis.set_title(predictor)
        axis.set_xlabel("measured - predicted")
        axis.grid(axis="y", alpha=.2)
    axes[0].set_ylabel("count")
    figure.suptitle("LOO prediction errors")
    figure.tight_layout()
    figure.savefig(output_dir / "alpha_same_amp_loo_error_histograms.png", dpi=130)
    plt.close(figure)

    amp_rows = physical_summary["amp_rows"]
    historical = _finite([
        row["historical_alpha_robust_location"] for row in amp_rows])
    figure, axis = plt.subplots(figsize=(7, 4.5))
    axis.hist(historical, bins=max(15, min(40, int(np.sqrt(historical.size))))
              if historical.size else 15, color="#55a868", edgecolor="white")
    axis.axvline(np.median(historical), color="black", lw=.9, label="median")
    axis.set_xlabel("historical robust alpha per physical amplifier")
    axis.set_ylabel("physical amplifier count")
    axis.set_title("Distribution of historical physical-amplifier alpha")
    axis.grid(axis="y", alpha=.2)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "alpha_physical_amp_historical_alpha.png", dpi=130)
    plt.close(figure)

    between = _finite([row["between_h5_alpha_scatter"] for row in amp_rows])
    figure, axis = plt.subplots(figsize=(7, 4.5))
    axis.hist(between, bins=max(15, min(40, int(np.sqrt(between.size))))
              if between.size else 15, color="#c44e52", edgecolor="white")
    axis.set_xlabel("robust alpha scatter across H5s")
    axis.set_ylabel("physical amplifier count")
    axis.set_title("Distribution of between-H5 alpha scatter")
    axis.grid(axis="y", alpha=.2)
    figure.tight_layout()
    figure.savefig(output_dir / "alpha_physical_amp_between_h5_scatter.png", dpi=130)
    plt.close(figure)

    # Deterministic gallery spanning the historical-alpha distribution.
    candidates = [row for row in amp_rows if row["N_H5"] >= 2]
    candidates.sort(key=lambda row: (
        row["historical_alpha_robust_location"],
        row["SPECID"], row["IFUSLOT"], row["IFUID"], row["AMP"]))
    chosen = []
    if candidates:
        for fraction in np.linspace(0.0, 1.0, min(gallery_size, len(candidates))):
            index = min(len(candidates) - 1,
                        int(round(fraction * (len(candidates) - 1))))
            candidate = candidates[index]
            if candidate not in chosen:
                chosen.append(candidate)
    if chosen:
        ncols = 3
        nrows = int(math.ceil(len(chosen) / ncols))
        figure, axes = plt.subplots(nrows, ncols,
                                    figsize=(14, 3.3 * nrows),
                                    squeeze=False, sharex=True)
        by_amp_h5 = defaultdict(list)
        for row in physical_summary["h5_rows"]:
            by_amp_h5[_physical_amp_key(row)].append(row)
        h5_order = sorted({row["H5"] for row in physical_summary["h5_rows"]})
        h5_index = {h5: index for index, h5 in enumerate(h5_order)}
        for panel_index, summary in enumerate(chosen):
            axis = axes.flat[panel_index]
            key = (summary["SPECID"], summary["IFUSLOT"],
                   summary["IFUID"], summary["AMP"])
            records = sorted(by_amp_h5[key], key=lambda row: row["H5"])
            x = np.asarray([h5_index[row["H5"]] for row in records], dtype=float)
            y = np.asarray([row["alpha_h5"] for row in records], dtype=float)
            axis.plot(x, y, "o-", color="#4472a8", ms=3, lw=.8)
            axis.axhline(0, color="black", lw=.6)
            axis.set_title("%s; median=%.3g" % (
                _physical_amp_label(key), summary["historical_alpha_robust_location"]),
                fontsize=8)
            axis.grid(alpha=.18)
            axis.tick_params(labelsize=7)
        for panel_index in range(len(chosen), nrows * ncols):
            axes.flat[panel_index].set_visible(False)
        for axis in axes[-1, :]:
            axis.set_xlabel("chronological H5 index")
        for axis in axes[:, 0]:
            axis.set_ylabel("alpha per H5")
        figure.suptitle("Representative physical-amplifier alpha histories", y=.995)
        figure.tight_layout()
        figure.savefig(output_dir / "alpha_same_amp_history_gallery.png",
                       dpi=130, bbox_inches="tight")
        plt.close(figure)
    return len(chosen)


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fits", default="m101_alpha_template_bands/alpha_template_fits.csv",
                        help="existing alpha_template_fits.csv; no spectra are read")
    parser.add_argument("--output-dir", default="m101_alpha_same_amp_loo")
    parser.add_argument("--gallery-size", type=int, default=DEFAULT_GALLERY_SIZE)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = _parse_args()
    if args.gallery_size <= 0:
        raise SystemExit("--gallery-size must be positive")
    fits_path = Path(args.fits).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise SystemExit("output directory is nonempty; use --overwrite: %s" % output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fit_rows, input_counts = _read_fit_rows(fits_path)
    collapsed = _collapse_h5(fit_rows)
    loo_rows, by_amp, all_h5 = _make_loo_rows(collapsed)
    physical_summary = _physical_summary(collapsed)
    summary_rows = _summary_rows(loo_rows, physical_summary)

    fit_output_rows = []
    for row in loo_rows:
        fit_output_rows.append({
            "H5": row["H5"], "SPECID": row["SPECID"],
            "IFUSLOT": row["IFUSLOT"], "IFUID": row["IFUID"],
            "IFU_CODE": row["IFU_CODE"], "AMP": row["AMP"],
            "N_exposures": row["N_exposures"],
            "exposures": row["exposures"],
            "alpha_measured_h5": row["alpha_h5"],
            "within_h5_alpha_scatter": row["within_h5_alpha_scatter"],
            "N_training_H5": row["N_training_H5"],
            "N_global_training_H5": row["N_global_training_H5"],
            "alpha_pred_zero": row["alpha_pred_zero"],
            "alpha_pred_global": row["alpha_pred_global"],
            "alpha_pred_same_amp": row["alpha_pred_same_amp"],
            "error_zero": row["error_zero"],
            "error_global": row["error_global"],
            "error_same_amp": row["error_same_amp"],
        })
    loo_path = output_dir / "alpha_same_amp_loo.csv"
    summary_path = output_dir / "alpha_same_amp_loo_summary.csv"
    physical_path = output_dir / "alpha_physical_amp_summary.csv"
    _write_rows(loo_path, fit_output_rows)
    _write_rows(summary_path, summary_rows)
    _write_rows(physical_path, physical_summary["amp_rows"])

    gallery_count = 0
    if not args.no_plots:
        gallery_count = _make_plots(output_dir, loo_rows, physical_summary,
                                    args.gallery_size)

    all_h5_within = [row["within_h5_alpha_scatter"] for row in collapsed]
    between = [row["between_h5_alpha_scatter"]
               for row in physical_summary["amp_rows"]]
    state = {
        "schema_version": "m101_alpha_same_amp_loo_v1",
        "script": str(Path(__file__).resolve()),
        "purpose": "diagnostic-only leave-one-H5-out same physical amplifier alpha prediction",
        "input_provenance": _file_identity(fits_path),
        "native_spectra_read": False,
        "band_fits_rerun": False,
        "selection": {
            "physical_amplifier_identity": ["SPECID", "IFUSLOT", "IFUID", "AMP"],
            "h5_collapse": "robust location of finite exposure alpha_joint values",
            "same_amp_training": "all other H5s for the same physical amplifier",
            "minimum_training_H5": 2,
            "global_baseline": "median alpha_h5 from all other H5s, excluding held-out H5",
            "zero_baseline": 0.0,
        },
        "counts": {
            "input_rows_total": input_counts["total_rows"],
            "input_rows_finite_alpha": input_counts["finite_rows"],
            "collapsed_amp_h5_rows": len(collapsed),
            "physical_amplifiers": len(physical_summary["amp_rows"]),
            "H5_count": len(all_h5),
            "LOO_rows_with_at_least_two_training_H5": len(loo_rows),
            "gallery_examples": gallery_count,
        },
        "population_distributions": {
            "within_h5_alpha_scatter": _distribution(all_h5_within),
            "between_h5_alpha_scatter": _distribution(between),
        },
        "validation": {
            "fits_csv_only": True,
            "native_spectra_read": False,
            "band_fits_rerun": False,
            "same_amp_only": True,
            "no_regression_or_shrinkage": True,
            "no_neighboring_amplifier_information": True,
        },
        "plots": {
            "written": not args.no_plots,
            "gallery_size_requested": int(args.gallery_size),
            "physical_amp_gallery_y": "automatic",
        },
        "artifacts": {
            "loo": str(loo_path.resolve()),
            "summary": str(summary_path.resolve()),
            "physical_amp_summary": str(physical_path.resolve()),
        },
    }
    (output_dir / "alpha_same_amp_loo_state.json").write_text(
        json.dumps(_json_ready(state), indent=2, sort_keys=True))

    print("input alpha fit rows: %d finite of %d" % (
        input_counts["finite_rows"], input_counts["total_rows"]))
    print("collapsed physical-amplifier/H5 rows: %d" % len(collapsed))
    print("physical amplifiers: %d" % len(physical_summary["amp_rows"]))
    print("LOO rows with >=2 training H5s: %d" % len(loo_rows))
    for row in summary_rows:
        if row.get("summary_type") != "prediction_performance":
            continue
        print("%s / %s: N=%d bias=%.6g scatter=%.6g med|err|=%.6g "
              "corr=%.6g frac<0.148=%.6g frac<0.296=%.6g frac<0.444=%.6g" % (
                  row["subset"], row["predictor"], row["N"], row["bias"],
                  row["robust_scatter"], row["median_absolute_error"],
                  row["correlation_predicted_measured"],
                  row["fraction_abs_error_lt_0.148"],
                  row["fraction_abs_error_lt_0.296"],
                  row["fraction_abs_error_lt_0.444"]))
    print("within-H5 alpha scatter: %s" % _distribution(all_h5_within))
    print("between-H5 alpha scatter: %s" % _distribution(between))
    print("wrote diagnostic products to %s" % output_dir)


if __name__ == "__main__":
    main()
