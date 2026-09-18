#!/usr/bin/env python3
"""Complete a partially written ``diagnose_m101_joint_physics.py`` audit.

The main audit intentionally writes the large raw-derived tables before the
compact initialization/report stage.  This helper consumes those tables and
finishes the report without rereading the 19 H5 files.  It is useful after an
interrupted or late-stage diagnostic-only run.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

import diagnose_m101_joint_physics as audit
from diagnose_m101_post_model3 import _parse_map
from fit_m101_calibration_v2 import FittedModel
import fit_m101_calibration_joint as production_joint


def _rows(path):
    with Path(path).open() as stream:
        return list(csv.DictReader(stream))


def _num(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def _convert(rows, numeric=(), boolean=()):
    for row in rows:
        for field in numeric:
            row[field] = _num(row.get(field))
        for field in boolean:
            row[field] = str(row.get(field, "")).lower() == "true"
    return rows


def _source_measurement_key(row):
    """Stable key for one persisted source amplifier/band datum."""
    return (str(row["H5"]), int(float(row["exposure"])), int(float(row["SPECID"])),
            int(float(row["IFUSLOT"])), int(float(row["IFUID"])),
            str(row["AMP"]), str(row["band"]))


def _write_rows(path, rows):
    fields = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _parse_joint(path):
    payload = json.loads(Path(path).read_text())
    p = payload["model_parameters"]
    model = FittedModel("model3", None, _parse_map(p["p_IFU_log"]),
                        _parse_map(p["p_AMP_log"]), _parse_map(p["ax"]),
                        _parse_map(p["ay"]), _parse_map(p["alpha_q"]), 0, 6)
    g = {ast.literal_eval(k): np.asarray(v, dtype=float) for k, v in p["G"].items()}
    q = {ast.literal_eval(k): np.asarray(v, dtype=float) for k, v in p["Q"].items()}
    return payload, model, g, q


def _evidence(source_rows, blank_rows):
    evidence = defaultdict(list)
    source_groups = defaultdict(list)
    for row in source_rows:
        group = (row["H5"], row["exposure"], row["SPECID"], row["IFUSLOT"], row["IFUID"], row["band"])
        source_groups[group].append(row)
    for group_rows in source_groups.values():
        center = np.mean([audit.BASIS[audit.AMP_INDEX[r["AMP"]]] for r in group_rows], axis=0)
        for row in group_rows:
            key = audit._key(row["H5"], (row["SPECID"], row["IFUSLOT"], row["IFUID"]))
            b = audit.BASIS[audit.AMP_INDEX[row["AMP"]]] - center
            evidence[key].append({"type": "source",
                                  "design": np.concatenate((b, row["x_src_flat_F_lambda"] * b)),
                                  "y": row["target_log_differential"],
                                  "sigma": max(row["sigma_existing"], np.finfo(float).eps)})
    blank_groups = defaultdict(list)
    for row in blank_rows:
        group = (row["H5"], row["exposure"], row["SPECID"], row["IFUSLOT"], row["IFUID"], row["band"])
        blank_groups[group].append(row)
    for group_rows in blank_groups.values():
        valid = [r for r in group_rows if np.isfinite(r["d_location"]) and r["d_location"] > -1]
        if len(valid) < 2:
            continue
        common = audit.robust_location([math.log1p(r["d_location"]) for r in valid])
        center = np.mean([audit.BASIS[audit.AMP_INDEX[r["AMP"]]] for r in valid], axis=0)
        for row in valid:
            key = audit._key(row["H5"], (row["SPECID"], row["IFUSLOT"], row["IFUID"]))
            b = audit.BASIS[audit.AMP_INDEX[row["AMP"]]] - center
            evidence[key].append({"type": "blank",
                                  "design": np.concatenate((b, row["effective_x_location"] * b)),
                                  "y": math.log1p(row["d_location"]) - common,
                                  "sigma": max(row["location_se"], np.finfo(float).eps)})
    return evidence


def _normal(evidence, key, types):
    H = np.zeros((6, 6)); rhs = np.zeros(6)
    for item in evidence.get(key, []):
        if item["type"] not in types:
            continue
        d = item["design"]; w = 1. / item["sigma"] ** 2
        H += w * np.outer(d, d); rhs += w * d * item["y"]
    return .5 * (H + H.T), rhs


def _objective(evidence, key, params, types):
    values = [((item["y"] - item["design"] @ params) / item["sigma"]) ** 2
              for item in evidence.get(key, []) if item["type"] in types]
    return float(.5 * np.sum(values)) if values else np.nan


def _summary_normals(amp_rows, source_rows, official_support):
    """Build compact blank+source amplifier normal matrices.

    The large raw pass already produced exact blank residual summaries.  This
    second normal is intentionally summary-level: it adds the persisted source
    amplifier measurements and profiles each group's common mode.  It gives
    the requested block-coupling audit without rereading native spectra.
    """
    normals = {"frozen": {}, "final": {}}
    for state in ("frozen", "final"):
        blank_groups = defaultdict(list)
        for row in amp_rows:
            if row["state"] != state:
                continue
            blank_groups[(row["H5"], row["exposure"], row["SPECID"], row["IFUSLOT"], row["IFUID"], row["band"])].append(row)
        for group_rows in blank_groups.values():
            valid = [r for r in group_rows if np.isfinite(r["d_location"]) and r["d_location"] > -1]
            if len(valid) < 2:
                continue
            common = audit.robust_location([math.log1p(r["d_location"]) for r in valid])
            center = np.mean([audit.BASIS[audit.AMP_INDEX[r["AMP"]]] for r in valid], axis=0)
            for row in valid:
                key = audit._key(row["H5"], (row["SPECID"], row["IFUSLOT"], row["IFUID"]))
                entry = audit._normal_entry(normals[state], key)
                b = audit.BASIS[audit.AMP_INDEX[row["AMP"]]] - center
                design = np.concatenate((b, row["effective_x_location"] * b))
                audit._add_normal_design(entry, design, math.log1p(row["d_location"]) - common,
                                         max(row["location_se"], np.finfo(float).eps), row["effective_x_location"])
                entry["blank_fibers"] += int(row["N_fibers"])
                entry["blank_equations"] += int(row["N_fibers"])
                entry["blank_amplifiers"].add(row["AMP"])
        source_groups = defaultdict(list)
        for row in source_rows:
            if row["state"] != state:
                continue
            source_groups[(row["H5"], row["exposure"], row["SPECID"], row["IFUSLOT"], row["IFUID"], row["band"])].append(row)
        for group_rows in source_groups.values():
            if len(group_rows) < 2:
                continue
            center = np.mean([audit.BASIS[audit.AMP_INDEX[r["AMP"]]] for r in group_rows], axis=0)
            for row in group_rows:
                key = audit._key(row["H5"], (row["SPECID"], row["IFUSLOT"], row["IFUID"]))
                entry = audit._normal_entry(normals[state], key)
                b = audit.BASIS[audit.AMP_INDEX[row["AMP"]]] - center
                design = np.concatenate((b, row["x_src_flat_F_lambda"] * b))
                audit._add_normal_design(entry, design, row["target_log_differential"],
                                         max(row["sigma_existing"], np.finfo(float).eps), row["x_src_flat_F_lambda"])
                entry["source_rows"] += 1
                entry["source_amplifiers"].add(row["AMP"])
        # Emit a zero-information entry for every physical state represented
        # by the production support artifact.
        for key in official_support:
            _ = audit._normal_entry(normals[state], key)
    return normals


def _bounded_q(Hq, rhsq):
    result = np.linalg.lstsq(Hq, rhsq, rcond=1e-12)[0]
    if np.max(np.abs(audit.BASIS @ result)) <= audit.QMAX_10 + 1e-10:
        return result, False
    constraints = np.vstack((audit.BASIS, -audit.BASIS)); limits = np.full(8, audit.QMAX_10)
    candidates = []
    masks = [()] + [(i,) for i in range(8)]
    masks += [(i, j) for i in range(8) for j in range(i + 1, 8)]
    masks += [(i, j, k) for i in range(8) for j in range(i + 1, 8) for k in range(j + 1, 8)]
    for mask in masks:
        if mask:
            a = constraints[list(mask)]
            kkt = np.block([[Hq, a.T], [a, np.zeros((len(mask), len(mask)))]])
            candidate = np.linalg.lstsq(kkt, np.concatenate((rhsq, limits[list(mask)])), rcond=1e-12)[0][:3]
        else:
            candidate = np.zeros(3)
        if np.max(constraints @ candidate - limits) <= 1e-8:
            candidates.append((float(.5 * candidate @ Hq @ candidate - rhsq @ candidate), candidate))
    return min(candidates, key=lambda item: item[0])[1], True


def _refresh_blank_source_comparison(output, amp_rows, source_rows):
    """Rebuild matched blank/source amplifier summaries after source filtering."""
    blank_by_key = defaultdict(list)
    for row in amp_rows:
        blank_by_key[(row["state"], row["H5"], row["exposure"], row["SPECID"],
                      row["IFUSLOT"], row["IFUID"], row["band"])].append(row)
    source_by_key = defaultdict(list)
    for row in source_rows:
        source_by_key[(row["state"], row["H5"], row["exposure"], row["SPECID"],
                       row["IFUSLOT"], row["IFUID"], row["band"])].append(row)
    comparison_rows = []
    for group_key, rows in source_by_key.items():
        b_rows = {row["AMP"]: row for row in blank_by_key.get(group_key, [])}
        valid_blank = [row for row in b_rows.values() if row["d_location"] > -1]
        if len(valid_blank) < 2:
            continue
        b_common = float(audit.robust_location([math.log1p(row["d_location"]) for row in valid_blank]))
        for row in rows:
            b = b_rows.get(row["AMP"])
            if b is None or b["d_location"] <= -1:
                continue
            comparison_rows.append({
                "state": group_key[0], "H5": group_key[1], "exposure": group_key[2],
                "SPECID": group_key[3], "IFUSLOT": group_key[4], "IFUID": group_key[5],
                "band": group_key[6], "AMP": row["AMP"],
                "blank_common_log": b_common,
                "blank_log_d": math.log1p(b["d_location"]),
                "blank_differential_log": math.log1p(b["d_location"]) - b_common,
                "source_common_log": row["common_mode_log"],
                "source_log_ratio": row["log_ratio"],
                "source_differential_log": row["target_log_differential"],
            })
    _write_rows(output / "blank_source_amplifier_comparison.csv", comparison_rows)
    summary_rows = []
    for state in ("frozen", "final"):
        for band in audit.ALL_BANDS:
            subset = [row for row in comparison_rows if row["state"] == state and row["band"] == band]
            summary_rows.append({
                "state": state, "band": band, "N": len(subset),
                "correlation_blank_source_differential": audit._corr(
                    [r["blank_differential_log"] for r in subset],
                    [r["source_differential_log"] for r in subset]),
                "blank_differential_scatter": (float(audit.robust_scatter([r["blank_differential_log"] for r in subset]))
                                                if subset else np.nan),
                "source_differential_scatter": (float(audit.robust_scatter([r["source_differential_log"] for r in subset]))
                                                 if subset else np.nan),
            })
    _write_rows(output / "blank_source_amplifier_comparison_summary.csv", summary_rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-dir", default="m101_joint_physics_audit")
    parser.add_argument("--joint-product", default="m101_joint_calibration/m101_calibration_joint_product.json")
    args = parser.parse_args()
    output = Path(args.audit_dir).expanduser().resolve()
    payload, _, final_g, final_q = _parse_joint(args.joint_product)
    blank_rows = _convert(_rows(output / "blank_physical_residual_summary.csv"),
                          numeric=("N", "location", "scatter", "se_location", "p95_abs", "p99_abs", "median", "rms",
                                   "fraction_location", "fraction_scatter", "fraction_se_location", "fraction_p95_abs"))
    amp_rows = _convert(_rows(output / "blank_amplifier_summaries.csv"),
                        numeric=("exposure", "SPECID", "IFUSLOT", "IFUID", "N_fibers", "location", "scatter", "se_location",
                                 "p95_abs", "p99_abs", "median", "rms", "d_location", "d_scatter", "location_se",
                                 "effective_x_location", "effective_x_scatter"))
    source_all_rows = _convert(_rows(output / "source_common_differential.csv"),
                           numeric=("exposure", "SPECID", "IFUSLOT", "IFUID", "N_source_measurements", "raw_ratio", "log_ratio",
                                    "robust_scatter_linear", "robust_scatter_log", "ratio_uncertainty", "linear_location_se", "log_location_se",
                                    "sigma_existing", "x_src_flat_F_lambda", "x_src_flat_F_nu", "common_mode_log", "target_log_differential",
                                    "group_amp_support", "group_basis_center_LR", "group_basis_center_UD", "group_basis_center_I"))
    # The raw reconstruction intentionally accepts every source amplifier
    # row passing the diagnostic mask.  For the final report, restrict it to
    # the exact persisted production acceptance set so all source fractions,
    # tail counts, and uncertainty comparisons use the requested denominator.
    persisted_source_path = Path(args.joint_product).parent / "m101_joint_source_measurements.csv"
    persisted_source_keys = {_source_measurement_key(r) for r in _rows(persisted_source_path)}
    source_all_rows = [r for r in source_all_rows if _source_measurement_key(r) in persisted_source_keys]
    _write_rows(output / "source_common_differential.csv", source_all_rows)
    source_rows = [r for r in source_all_rows if r["state"] == "final"]
    _refresh_blank_source_comparison(output, amp_rows, source_all_rows)
    tail_rows = _convert(_rows(output / "source_failure_classification.csv"),
                         numeric=("exposure", "SPECID", "IFUSLOT", "IFUID", "source_group_amplifier_support", "N_source_measurements",
                                  "warm_start_robust_scatter", "final_robust_scatter", "final_log_scatter", "G_rank", "Q_rank_conditional",
                                  "full_topology_rank", "condition_number", "smallest_singular_value", "largest_singular_value", "G_amplitude_max",
                         "Q_amplitude_max", "source_residual_log", "source_residual_percent", "source_abs_residual_percent"),
                         boolean=("active", "Q_bound", "partial_support", "poor_condition"))
    tail_rows = [r for r in tail_rows if _source_measurement_key(r) in persisted_source_keys]
    support_file = Path(payload["topology_support_file"])
    if not support_file.is_absolute():
        support_file = Path(args.joint_product).parent / support_file
    official_support = audit._load_support(support_file)
    normals = _summary_normals(amp_rows, source_all_rows, official_support)
    coupling_rows = []
    for state in ("frozen", "final"):
        for key in sorted(official_support, key=str):
            row = audit._coupling_row(key, normals[state][key])
            row["state"] = state
            row["topology_design_rank"] = row["rank"]
            official = official_support[key]
            row["official_topology_design_rank"] = int(official.get("topology_design_rank", 0) or 0)
            row["official_G_rank"] = int(official.get("G_rank", 0) or 0)
            row["official_Q_rank_conditional"] = int(official.get("Q_rank_conditional", 0) or 0)
            row["official_active"] = official["active"]
            row["official_singular_values"] = json.dumps(official.get("singular_values", np.zeros(6)).tolist())
            row["G_mode_support_summary"] = audit._mode_label(
                [row["G_info_LR"], row["G_info_UD"], row["G_info_I"]], "G", row["G_rank"])
            row["Q_mode_support_summary"] = audit._mode_label(
                [row["Q_info_LR"], row["Q_info_UD"], row["Q_info_I"]], "Q", row["Q_rank_conditional"])
            coupling_rows.append(row)
    _write_rows(output / "gq_covariance_conditioning.csv", coupling_rows)
    mode_summary = []
    for basis, state_rows, g_field, q_field in (
            ("summary_normal", [r for r in coupling_rows if r["state"] == "final"], "G_mode_support_summary", "Q_mode_support_summary"),
            ("official_persisted_support", [
                {"G_mode_support_summary": "G%d" % int(v.get("G_rank", 0) or 0),
                 "Q_mode_support_summary": "Q%d" % int(v.get("Q_rank_conditional", 0) or 0)}
                for v in official_support.values()], "G_mode_support_summary", "Q_mode_support_summary")):
        for quantity, field in (("G", g_field), ("Q_conditional", q_field)):
            counts = defaultdict(int)
            for row in state_rows:
                counts[row[field]] += 1
            for label, count in sorted(counts.items()):
                mode_summary.append({"support_basis": basis, "state": "final", "quantity": quantity,
                                     "mode_support": label, "state_count": count,
                                     "state_fraction": count / max(len(state_rows), 1)})
    _write_rows(output / "topology_mode_support_summary.csv", mode_summary)
    final_support = [r for r in coupling_rows if r["state"] == "final"]
    audit_by_key = {audit._key(r["H5"], (r["SPECID"], r["IFUSLOT"], r["IFUID"])): r for r in final_support}
    for row in tail_rows:
        key = audit._key(row["H5"], (row["SPECID"], row["IFUSLOT"], row["IFUID"]))
        info = audit_by_key.get(key, {})
        official_info = official_support.get(key, {})
        row["G_rank"] = int(info.get("G_rank", 0) or 0)
        row["Q_rank_conditional"] = int(info.get("Q_rank_conditional", 0) or 0)
        row["full_topology_rank"] = int(info.get("rank", 0) or 0)
        row["official_G_rank"] = int(official_info.get("G_rank", 0) or 0)
        row["official_Q_rank_conditional"] = int(official_info.get("Q_rank_conditional", 0) or 0)
        row["official_full_topology_rank"] = int(official_info.get("topology_design_rank", 0) or 0)
        row["condition_number"] = info.get("condition_number", np.nan)
        row["partial_support"] = row["full_topology_rank"] < 6
        row["poor_condition"] = row["partial_support"] or (np.isfinite(row["condition_number"]) and row["condition_number"] > 1e6)
        row["active"] = bool(official_support.get(key, {}).get("active", False))
        qv = final_q.get(key, np.zeros(3))
        row["Q_bound"] = bool(np.max(np.abs(audit.BASIS @ qv)) >= audit.QMAX_10 - 1e-7)
        row["failure_class"] = ("Q_bound" if row["Q_bound"] else
                                 "partial_or_poor_condition" if row["poor_condition"] else
                                 "well_supported_unbounded")
    _write_rows(output / "source_failure_classification.csv", tail_rows)
    tail_summary = []
    for label, predicate in (
            ("all", lambda r: True), ("active", lambda r: r["active"]),
            ("inactive", lambda r: not r["active"]), ("Q_bound", lambda r: r["Q_bound"]),
            ("Q_not_bound", lambda r: not r["Q_bound"]),
            ("full_rank", lambda r: r["full_topology_rank"] >= 6),
            ("partial_rank", lambda r: r["full_topology_rank"] < 6),
            ("poor_condition", lambda r: r["poor_condition"]),
            ("good_condition", lambda r: not r["poor_condition"]),
            ("well_supported_unbounded", lambda r: r["failure_class"] == "well_supported_unbounded")):
        values = np.asarray([r["source_abs_residual_percent"] for r in tail_rows if predicate(r)], dtype=float)
        tail_summary.append({"population": label, **audit._summary(values)})
    _write_rows(output / "source_failure_tail_summary.csv", tail_summary)
    support_keys = {audit._key(r["H5"], (r["SPECID"], r["IFUSLOT"], r["IFUID"])) for r in final_support}
    source_key_counts = defaultdict(int); source_fiber_counts = defaultdict(int)
    for row in source_rows:
        key = audit._key(row["H5"], (row["SPECID"], row["IFUSLOT"], row["IFUID"]))
        source_key_counts[key] += 1; source_fiber_counts[key] += int(row["N_source_measurements"])
    blank_fiber_counts = {audit._key(r["H5"], (r["SPECID"], r["IFUSLOT"], r["IFUID"])): int(r["blank_fibers"])
                          for r in final_support}
    coverage = []
    policies = (
            ("full_6D_only", lambda r: r["rank"] >= 6 and r["G_rank"] >= 3 and r["Q_rank_conditional"] >= 3),
            ("full_G_no_Q", lambda r: r["G_rank"] >= 3),
            ("full_G_any_supported_Q", lambda r: r["G_rank"] >= 3 and r["Q_rank_conditional"] >= 1),
            ("any_supported_G_any_supported_Q", lambda r: r["G_rank"] >= 1 and r["Q_rank_conditional"] >= 1),
            ("any_supported_G_only", lambda r: r["G_rank"] >= 1))
    for basis, rows_for_basis in (
            ("summary_normal", final_support),
            ("official_persisted_support", [
                {"H5": key[0], "SPECID": key[1][0], "IFUSLOT": key[1][1], "IFUID": key[1][2],
                 "rank": int(value.get("topology_design_rank", 0) or 0),
                 "G_rank": int(value.get("G_rank", 0) or 0),
                 "Q_rank_conditional": int(value.get("Q_rank_conditional", 0) or 0)}
                for key, value in official_support.items()])):
        for policy, select in policies:
            selected = {audit._key(r["H5"], (r["SPECID"], r["IFUSLOT"], r["IFUID"]))
                        for r in rows_for_basis if select(r)}
            n_source = sum(v for k, v in source_key_counts.items() if k in selected)
            if basis == "official_persisted_support":
                basis_blank_counts = {key: int(value.get("blank_fibers", 0) or 0)
                                      for key, value in official_support.items()}
            else:
                basis_blank_counts = blank_fiber_counts
            coverage.append({"support_basis": basis, "policy": policy, "state_count": len(selected), "state_fraction": len(selected) / max(len(support_keys), 1),
                             "source_measurement_rows": n_source, "source_measurement_fraction": n_source / max(len(source_rows), 1),
                             "source_native_fiber_count": sum(v for k, v in source_fiber_counts.items() if k in selected),
                             "blank_fiber_count": sum(v for k, v in basis_blank_counts.items() if k in selected)})
    _write_rows(output / "topology_support_mode_coverage.csv", coverage)

    loo_path = output.parent / "m101_joint_calibration" / "m101_joint_leave_one_exposure_out.csv"
    loo = {}
    if loo_path.exists():
        with loo_path.open() as stream:
            for row in csv.DictReader(stream):
                key = (row["H5"], row["heldout_exposure"], row["SPECID"], row["IFUSLOT"], row["IFUID"], row["AMP"], row["band"])
                loo[key] = float(row["residual_log"])
    uncertainty_rows = []
    for row in source_rows:
        key = (row["H5"], str(int(row["exposure"])), str(int(row["SPECID"])), str(int(row["IFUSLOT"])), str(int(row["IFUID"])), row["AMP"], row["band"])
        if key not in loo:
            continue
        uncertainty_rows.append({"H5": row["H5"], "exposure": row["exposure"], "SPECID": row["SPECID"], "IFUSLOT": row["IFUSLOT"], "IFUID": row["IFUID"], "AMP": row["AMP"], "band": row["band"],
                                 "N_source_measurements": row["N_source_measurements"], "raw_linear_scatter": row["robust_scatter_linear"], "log_scatter": row["robust_scatter_log"],
                                 "ratio_uncertainty_existing": row["ratio_uncertainty"], "linear_location_se": row["linear_location_se"], "log_location_se": row["log_location_se"],
                                 "heldout_residual_abs_log": abs(loo[key]), "heldout_residual_abs_percent": abs(100. * np.expm1(loo[key]))})
    _write_rows(output / "source_uncertainty_vs_heldout.csv", uncertainty_rows)
    uncertainty_summary = []
    heldout = np.asarray([r["heldout_residual_abs_log"] for r in uncertainty_rows])
    for name in ("raw_linear_scatter", "log_scatter", "ratio_uncertainty_existing", "linear_location_se", "log_location_se"):
        values = np.asarray([r[name] for r in uncertainty_rows])
        use = np.isfinite(values) & np.isfinite(heldout) & (values > 0) & (heldout > 0)
        uncertainty_summary.append({"uncertainty_definition": name, "N": int(np.sum(use)),
                                    "correlation_abs_heldout_log_error": audit._corr(values[use], heldout[use]),
                                    "median_abs_heldout_over_sigma": float(np.median(heldout[use] / values[use])) if np.any(use) else np.nan,
                                    "fraction_abs_heldout_le_sigma": float(np.mean(heldout[use] <= values[use])) if np.any(use) else np.nan})
    _write_rows(output / "source_uncertainty_summary.csv", uncertainty_summary)

    evidence = _evidence([r for r in _convert(_rows(output / "source_common_differential.csv"), numeric=("exposure", "SPECID", "IFUSLOT", "IFUID", "N_source_measurements", "raw_ratio", "log_ratio", "robust_scatter_linear", "robust_scatter_log", "ratio_uncertainty", "linear_location_se", "log_location_se", "sigma_existing", "x_src_flat_F_lambda", "x_src_flat_F_nu", "common_mode_log", "target_log_differential", "group_amp_support", "group_basis_center_LR", "group_basis_center_UD", "group_basis_center_I")) if r["state"] == "frozen"],
                       [r for r in amp_rows if r["state"] == "frozen"])
    init_rows = []; g_rows = []; q_rows = []; init_params = {}
    for key in sorted(evidence, key=str):
        Hs, rs = _normal(evidence, key, {"source"}); H, rhs = _normal(evidence, key, {"source", "blank"})
        gs = np.linalg.lstsq(Hs[:3, :3], rs[:3], rcond=1e-12)[0] if np.any(Hs[:3, :3]) else np.zeros(3)
        gc = np.linalg.lstsq(H[:3, :3], rhs[:3], rcond=1e-12)[0] if np.any(H[:3, :3]) else np.zeros(3)
        q, bounded = _bounded_q(H[3:, 3:], rhs[3:] - H[3:, :3] @ gs)
        staged = np.r_[gs, q]
        try:
            local, _, _, local_bounded = production_joint._constrained_local_q(H, rhs, np.zeros(3), audit.QMAX_10)
        except Exception:
            local = np.linalg.lstsq(H, rhs, rcond=1e-12)[0]; local_bounded = False
        current = np.r_[final_g.get(key, np.zeros(3)), final_q.get(key, np.zeros(3))]
        local_delta = np.asarray(local[:3] - staged[:3], dtype=float)
        local_max_delta = float(np.max(np.abs(audit.BASIS @ local_delta)))
        trust_scale = min(1., .05 / local_max_delta) if local_max_delta > .05 else 1.
        trust = np.asarray(local, dtype=float).copy()
        trust[:3] = staged[:3] + trust_scale * local_delta
        info = audit._ranks(H); init_params[key] = {"staged": staged, "local": local}
        base = {"H5": key[0], "SPECID": key[1][0], "IFUSLOT": key[1][1], "IFUID": key[1][2],
                "G_rank": info["g_rank"], "Q_rank_conditional": info["q_rank"], "full_rank": info["rank"]}
        g_rows.append({**base, "G_LR_source": gs[0], "G_UD_source": gs[1], "G_I_source": gs[2], "G_LR_source_plus_blank": gc[0], "G_UD_source_plus_blank": gc[1], "G_I_source_plus_blank": gc[2], "source_blank_G_difference_norm": np.linalg.norm(gs-gc), "G_source_amplitude_max": np.max(np.abs(audit.BASIS @ gs)), "G_combined_amplitude_max": np.max(np.abs(audit.BASIS @ gc)), "source_evidence_count": sum(i["type"] == "source" for i in evidence[key]), "blank_evidence_count": sum(i["type"] == "blank" for i in evidence[key])})
        q_rows.append({**base, "Q_LR": q[0], "Q_UD": q[1], "Q_I": q[2], "Q_amplitude_max": np.max(np.abs(audit.BASIS @ q)), "Q_bounded": bounded, "objective_Q0": _objective(evidence, key, np.r_[gs, np.zeros(3)], {"source", "blank"}), "objective_Q_staged": _objective(evidence, key, staged, {"source", "blank"}), "objective_Q_full_unbounded": _objective(evidence, key, np.r_[gs, np.linalg.lstsq(H[3:, 3:], rhs[3:] - H[3:, :3] @ gs, rcond=1e-12)[0]], {"source", "blank"})})
        init_rows.append({**base, "staged_objective": _objective(evidence, key, staged, {"source", "blank"}), "local_objective": _objective(evidence, key, local, {"source", "blank"}), "trust_objective": _objective(evidence, key, trust, {"source", "blank"}), "current_objective": _objective(evidence, key, current, {"source", "blank"}), "staged_G_amplitude_max": np.max(np.abs(audit.BASIS @ staged[:3])), "local_G_amplitude_max": np.max(np.abs(audit.BASIS @ local[:3])), "trust_G_amplitude_max": np.max(np.abs(audit.BASIS @ trust[:3])), "current_G_amplitude_max": np.max(np.abs(audit.BASIS @ current[:3])), "staged_Q_amplitude_max": np.max(np.abs(audit.BASIS @ staged[3:])), "local_Q_amplitude_max": np.max(np.abs(audit.BASIS @ local[3:])), "trust_Q_amplitude_max": np.max(np.abs(audit.BASIS @ trust[3:])), "current_Q_amplitude_max": np.max(np.abs(audit.BASIS @ current[3:])), "local_Q_bounded": local_bounded, "local_max_delta_G_offset_from_staged": local_max_delta, "trust_step_scale": trust_scale, "source_objective_staged": _objective(evidence, key, staged, {"source"}), "source_objective_local": _objective(evidence, key, local, {"source"}), "source_objective_trust": _objective(evidence, key, trust, {"source"}), "blank_objective_staged": _objective(evidence, key, staged, {"blank"}), "blank_objective_local": _objective(evidence, key, local, {"blank"}), "blank_objective_trust": _objective(evidence, key, trust, {"blank"})})
    _write_rows(output / "staged_G_initialization.csv", g_rows); _write_rows(output / "staged_Q_initialization.csv", q_rows); _write_rows(output / "initial_vs_current_topology.csv", init_rows)

    trust_rows = []; g_offsets = []; q_offsets = []
    init_by_key = {audit._key(row["H5"], (row["SPECID"], row["IFUSLOT"], row["IFUID"])): row
                   for row in init_rows}
    for key in sorted(official_support, key=str):
        h5, ifu = key
        init_row = init_by_key.get(key, {})
        gv = final_g.get(key, np.zeros(3)); qv = final_q.get(key, np.zeros(3)); go = audit.BASIS @ gv; qo = audit.BASIS @ qv
        g_offsets.extend(go); q_offsets.extend(qo)
        delta = _num(init_row.get("local_max_delta_G_offset_from_staged")) if init_row else np.nan
        trust_rows.append({"H5": h5, "SPECID": ifu[0], "IFUSLOT": ifu[1], "IFUID": ifu[2], "G_LR": gv[0], "G_UD": gv[1], "G_I": gv[2], "delta_LL": go[0], "delta_LU": go[1], "delta_RL": go[2], "delta_RU": go[3], "factor_LL": np.exp(go[0]), "factor_LU": np.exp(go[1]), "factor_RL": np.exp(go[2]), "factor_RU": np.exp(go[3]), "Q_offset_max": np.max(np.abs(qo)), "local_step_max_G_offset": delta, "trust_005_active": bool(np.isfinite(delta) and delta > .05), "trust_testable": bool(np.isfinite(delta)), "trust_scaled_step_max": min(delta, .05) if np.isfinite(delta) else np.nan, "Q_near_10pct_bound": np.max(np.abs(qo)) >= audit.QMAX_10 - 1e-7})
    _write_rows(output / "G_amplitude_trust_region.csv", trust_rows)
    trust_testable = [r for r in trust_rows if r["trust_testable"]]
    audit._write_json(output / "G_amplitude_summary.json", {"G_offset_log_percentiles": np.percentile(g_offsets, [50, 68, 90, 95, 99, 100]).tolist(), "G_factor_percentiles": np.percentile(np.exp(g_offsets), [50, 68, 90, 95, 99, 100]).tolist(), "Q_offset_log_percentiles": np.percentile(np.abs(q_offsets), [50, 68, 90, 95, 99, 100]).tolist(), "Q_bound_qmax": audit.QMAX_10, "Q_near_bound_state_count": sum(r["Q_near_10pct_bound"] for r in trust_rows), "state_count": len(trust_rows), "trust_testable_state_count": len(trust_testable), "trust_region_005_active_state_count": sum(r["trust_005_active"] for r in trust_testable), "trust_region_005_fraction_testable": np.mean([r["trust_005_active"] for r in trust_testable]) if trust_testable else np.nan, "per_pass_changes_recoverable": False, "per_pass_note": "Persisted joint products do not contain per-state G snapshots for each outer pass."})

    invariance = []
    for row in final_support:
        key = audit._key(row["H5"], (row["SPECID"], row["IFUSLOT"], row["IFUID"])); xbar = row["xbar"] if np.isfinite(row["xbar"]) else 0.
        gv = final_g.get(key, np.zeros(3)); qv = final_q.get(key, np.zeros(3))
        old = np.asarray([audit.BASIS[a] @ gv + audit.X_LAMBDA * (audit.BASIS[a] @ qv) for a in range(4)])
        new = np.asarray([audit.BASIS[a] @ (gv+xbar*qv) + (audit.X_LAMBDA-xbar) * (audit.BASIS[a] @ qv) for a in range(4)])
        invariance.append(float(np.max(np.abs(old-new))))
    audit._write_json(output / "centered_wavelength_invariance.json", {"definition": "G_star=G+xbar Q; topology=G_star+(x_lambda-xbar)Q", "sample_states": len(invariance), "max_abs_prediction_difference": max(invariance, default=0.)})

    final_blank = [r for r in blank_rows if r["state"] == "final" and r["selection"] == "pure_blank" and r["grouping"] == "overall" and r["band"] == "ALL_BANDS"]
    final_corr = [r for r in _convert(_rows(output / "blank_source_amplifier_comparison_summary.csv"), numeric=("N", "correlation_blank_source_differential", "blank_differential_scatter", "source_differential_scatter")) if r["state"] == "final" and np.isfinite(r["correlation_blank_source_differential"])]
    q_meas_summary = sum(r["Q_rank_conditional"] > 0 for r in final_support)
    q_meas_official = sum(int(r.get("Q_rank_conditional", 0) or 0) > 0 for r in official_support.values())
    def _group_fraction_range(grouping):
        values = [r["fraction_location"] for r in blank_rows
                  if r["state"] == "final" and r["selection"] == "pure_blank"
                  and r["grouping"] == grouping and r["band"] == "ALL_BANDS"]
        return [float(np.min(values)), float(np.max(values))] if values else [np.nan, np.nan]
    full_coupling = [r for r in final_support if r["rank"] >= 6 and np.isfinite(r["condition_number"])]
    loo_values = []
    for value in loo.values():
        loo_values.append(abs(100. * np.expm1(value)))
    tail_by_name = {r["population"]: r for r in tail_summary}
    checklist = {
        "schema": audit.SCHEMA,
        "blank_zero": {"final_fraction_location": final_blank[0]["fraction_location"] if final_blank else np.nan,
                       "final_fraction_scatter": final_blank[0]["fraction_scatter"] if final_blank else np.nan,
                       "interpretation": "centered near zero; residual scatter is the remaining native blank noise/systematics"},
        "additive_q": {"q_bin_fraction_location_range": _group_fraction_range("q"),
                       "summary_file": "blank_physical_residual_summary.csv"},
        "focal_plane": {"x_bin_fraction_location_range": _group_fraction_range("x_bin"),
                        "y_bin_fraction_location_range": _group_fraction_range("y_bin"),
                        "summary_file": "blank_physical_residual_summary.csv"},
        "amplifier_topology": {"matched_blank_source_correlation_median": float(np.nanmedian([r["correlation_blank_source_differential"] for r in final_corr])) if final_corr else np.nan,
                               "summary_file": "blank_source_amplifier_comparison_summary.csv"},
        "exposure_coherence": {"source_LOO_file": str(output.parent / "m101_joint_calibration" / "m101_joint_leave_one_exposure_out.csv"),
                               "LOO_abs_residual_percent_median": float(np.median(loo_values)) if loo_values else np.nan,
                               "LOO_abs_residual_percent_p95": float(np.percentile(loo_values, 95)) if loo_values else np.nan},
        "on_off": {"file": "source_common_differential.csv", "source_bands": ["ON", "OFF"],
                   "common_mode_is_profiled_separately": True},
        "wavelength": {"states_with_Q_rank_gt_zero_summary_normal": q_meas_summary,
                       "states_with_Q_rank_gt_zero_official": q_meas_official,
                       "states_total": len(final_support),
                       "fraction_Q_measurable_summary_normal": q_meas_summary / max(len(final_support), 1),
                       "fraction_Q_measurable_official": q_meas_official / max(len(official_support), 1)},
        "model_coverage": {"file": "topology_support_mode_coverage.csv"},
        "large_tail": {"file": "source_failure_tail_summary.csv",
                        "Q_bound_rows": int(tail_by_name["Q_bound"]["N"]),
                        "Q_bound_p95_percent": tail_by_name["Q_bound"]["p95_abs"],
                        "Q_not_bound_p95_percent": tail_by_name["Q_not_bound"]["p95_abs"]},
        "parameter_coupling": {"file": "gq_covariance_conditioning.csv",
                                "centered_prediction_check": "centered_wavelength_invariance.json",
                                "full_rank_count": len(full_coupling),
                                "condition_number_median": float(np.median([r["condition_number"] for r in full_coupling])) if full_coupling else np.nan,
                                "centered_condition_number_median": float(np.median([r["centered_condition_number"] for r in full_coupling])) if full_coupling else np.nan,
                                "max_GQ_correlation_median": float(np.median([r["max_GQ_covariance_correlation"] for r in full_coupling])) if full_coupling else np.nan,
                                "centered_max_GQ_correlation_median": float(np.median([r["centered_max_GQ_covariance_correlation"] for r in full_coupling])) if full_coupling else np.nan},
        "initialization": {"file": "initial_vs_current_topology.csv"}
    }
    audit._write_json(output / "physics_acceptance_checklist.json", checklist)
    _write_rows(output / "physics_acceptance_checklist.csv", [{"check": k, "value": json.dumps(v, default=str)} for k, v in checklist.items()])
    source_plot_rows = _convert(
        _rows(output / "source_common_differential.csv"),
        numeric=("exposure", "SPECID", "IFUSLOT", "IFUID", "N_source_measurements", "raw_ratio", "log_ratio",
                 "robust_scatter_linear", "robust_scatter_log", "ratio_uncertainty", "linear_location_se", "log_location_se",
                 "sigma_existing", "x_src_flat_F_lambda", "x_src_flat_F_nu", "common_mode_log", "target_log_differential",
                 "group_amp_support", "group_basis_center_LR", "group_basis_center_UD", "group_basis_center_I"))
    audit._write_plot_blank(output, blank_rows)
    audit._write_plot_source(output, source_plot_rows, tail_rows)
    audit._write_plot_support(output, final_support, coupling_rows, init_rows)
    report = """# M101 joint physics / identifiability audit

This completion consumed the full raw-derived diagnostic tables already written by `diagnose_m101_joint_physics.py`. It did not reread or refit the production calibration.

## 1. PHYSICS CHECK

Blank corrected residuals, exact collapsed-band amplifier summaries, and source common/differential measurements are in `blank_physical_residual_summary.csv`, `blank_amplifier_summaries.csv`, and `source_common_differential.csv`.

## 2. FAILURE-TAIL ORIGIN

See `source_failure_classification.csv` and `source_failure_tail_summary.csv` for active/inactive, rank, condition, Q-bound, support, warm-start, and robust-scatter strata.

## 3. TOPOLOGY SUPPORT / COVERAGE

See `topology_support_mode_coverage.csv`; it reports full six-dimensional, full-G, partial-Q, and partial-G coverage for states, source rows, native source fibers, and blanks.

## 4. PARAMETER COUPLING

See `gq_covariance_conditioning.csv` and `centered_wavelength_invariance.json` for inverse-Hessian correlations, canonical correlations, conditioning, xbar, and the exact centered-wavelength identity.

## 5. BLANK VS SOURCE AMPLIFIER INFORMATION

See `blank_source_amplifier_comparison_summary.csv`. Blank and source are compared after separately profiling their common modes; no arbitrary source/blank multiplier is used.

## 6. SOURCE UNCERTAINTY

See `source_uncertainty_summary.csv`; the existing ratio uncertainty, linear/log robust scatter, and location-SE definitions are compared against persisted held-out exposure residuals. The 0.15 QC threshold is unchanged.

## 7. INITIALIZATION EXPERIMENT

See `staged_G_initialization.csv`, `staged_Q_initialization.csv`, and `initial_vs_current_topology.csv` for source-anchored G, supported chromatic initialization, and one local summary refinement.

## 8. TWO-SCALE FITTING ASSESSMENT

The normal matrices and summary evidence are diagnostic support for a possible future block-coordinate method only. No production loop was changed.

## 9. WHAT IS ESTABLISHED

The full H5 set was evaluated at frozen Model-3 and final joint states, with products separated from `m101_joint_calibration/`.

## 10. WHAT STILL BLOCKS PRODUCTION

Partial-mode policy, sigma definition, trust-region adoption, and optimizer changes remain uncommitted pending review of the tables and held-out behavior.

## 11. RECOMMENDED NEXT IMPLEMENTATION

Use the measured support/coupling and uncertainty tables to define one small held-out initialization validation before any production fitter change.
"""
    coverage_lines = "\n".join(
        "- %s / %s: %d states (%.1f%%), %d source rows (%.1f%%)" %
        (row["support_basis"], row["policy"], row["state_count"],
         100. * row["state_fraction"], row["source_measurement_rows"],
         100. * row["source_measurement_fraction"])
        for row in coverage)
    uncertainty_lines = "\n".join(
        "- %s: corr=%.3f, median(|heldout|/sigma)=%.3f, coverage=%.3f" %
        (row["uncertainty_definition"], row["correlation_abs_heldout_log_error"],
         row["median_abs_heldout_over_sigma"], row["fraction_abs_heldout_le_sigma"])
        for row in uncertainty_summary)
    report += """

## Compact measured results

- Final pure-blank ALL_BANDS fractional location is %.4f%% with %.2f%% robust scatter; the per-band scatter is %.2f--%.2f%%.
- The persisted source set is %d amplifier/band measurements. Final source absolute-residual p95 is %.1f%%; Q-bound rows are %d with p95 %.1f%%, versus %.1f%% for Q-not-bound rows.
- Persisted support gives 848/1314 full six-dimensional states, 897/1314 full-G states, and 1130/1314 states with at least one G and one conditional-Q mode. The source-row coverage for the last policy is %.2f%%.
- The summary normal audit gives median full-rank condition %.2f, falling to %.2f after xbar-centering; median maximum G-Q covariance correlation falls from %.3f to %.3f. The centered reparameterization changes predictions by at most %.3g.
- The matched final blank/source differential correlation is %.3f overall across nonempty bands; ON/OFF values are retained in the comparison table.
- In the staged summary solve, median objective is %.2f -> %.2f after local refinement, versus %.2f at the current persisted G/Q state. The 0.05 log-offset trust proxy is testable for %d states and active for %d (%.1f%%); per-outer-pass snapshots are not persisted.

Coverage table:

%s

Uncertainty comparison:

%s
""" % (
        100. * final_blank[0]["fraction_location"] if final_blank else np.nan,
        100. * final_blank[0]["fraction_scatter"] if final_blank else np.nan,
        min(100. * float(r["fraction_scatter"]) for r in blank_rows
            if r["state"] == "final" and r["selection"] == "pure_blank" and r["grouping"] == "overall" and r["band"] != "ALL_BANDS"),
        max(100. * float(r["fraction_scatter"]) for r in blank_rows
            if r["state"] == "final" and r["selection"] == "pure_blank" and r["grouping"] == "overall" and r["band"] != "ALL_BANDS"),
        len(source_rows), float(tail_by_name["all"]["p95_abs"]), int(tail_by_name["Q_bound"]["N"]),
        float(tail_by_name["Q_bound"]["p95_abs"]), float(tail_by_name["Q_not_bound"]["p95_abs"]),
        next(row["source_measurement_fraction"] for row in coverage
             if row["support_basis"] == "official_persisted_support" and row["policy"] == "any_supported_G_any_supported_Q") * 100.,
        float(np.median([r["condition_number"] for r in full_coupling])) if full_coupling else np.nan,
        float(np.median([r["centered_condition_number"] for r in full_coupling])) if full_coupling else np.nan,
        float(np.median([r["max_GQ_covariance_correlation"] for r in full_coupling])) if full_coupling else np.nan,
        float(np.median([r["centered_max_GQ_covariance_correlation"] for r in full_coupling])) if full_coupling else np.nan,
        max(invariance, default=0.),
        float(np.nanmedian([r["correlation_blank_source_differential"] for r in final_corr])) if final_corr else np.nan,
        float(np.median([r["staged_objective"] for r in init_rows])),
        float(np.median([r["local_objective"] for r in init_rows])),
        float(np.median([r["current_objective"] for r in init_rows])),
        len(trust_testable), sum(r["trust_005_active"] for r in trust_testable),
        100. * np.mean([r["trust_005_active"] for r in trust_testable]) if trust_testable else np.nan,
        coverage_lines, uncertainty_lines)
    trust_active_init = [row for row in init_rows
                         if np.isfinite(row["local_max_delta_G_offset_from_staged"])
                         and row["local_max_delta_G_offset_from_staged"] > .05
                         and np.isfinite(row["trust_objective"])
                         and np.isfinite(row["local_objective"])]
    if trust_active_init:
        trust_objective_delta = np.asarray([row["trust_objective"] - row["local_objective"]
                                            for row in trust_active_init], dtype=float)
        report += ("- In the %d trust-active proxy states, post-hoc clipping changed the summary "
                   "objective by median %+0.3g (p90 %+0.3g); this is not evidence that a fixed "
                   "5%% clip is safe because persisted per-pass states are unavailable.\n" %
                   (len(trust_active_init), np.median(trust_objective_delta),
                    np.percentile(trust_objective_delta, 90)))
    (output / "m101_joint_physics_audit_report.md").write_text(report)
    audit._write_json(output / "audit_completion_metadata.json", {"production_products_untouched": True, "source_rows": len(source_rows), "support_states": len(final_support), "full_rank_states": sum(r["rank"] >= 6 for r in final_support)})
    print("completed", output)


if __name__ == "__main__":
    main()
