#!/usr/bin/env python3
"""Validate a clean M101 calibration product without fitting anything.

The blank path uses ``sky_blank`` only to select the external blank spectra
that train the additive model.  The source path uses the measurement product's
external-valid, compact, date, finite, and positive-error fields; it never
uses ``~sky_blank`` as a source mask.
"""

from argparse import ArgumentParser
import csv
import json
from pathlib import Path
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tables

from fit_m101_calibration import (N_WAVE, NULL_NAMES, WAVE, apply_compact_mask,
                                  date_mask, discover_h5, exposure_groups,
                                  identity, load_blank, read_filter, robust_location,
                                  robust_rms, robust_scale, robust_spectrum, text,
                                  weighted_scalar)


N_AMP_FIBERS = 112
STAGES = ("raw", "after_R", "after_cU", "after_dV")


def write_csv(path, rows, fields):
    path = Path(path)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def product(path):
    with tables.open_file(path, "r") as h5:
        if getattr(h5.root._v_attrs, "schema_version", "") != "m101_calibration_clean_v1":
            raise ValueError("unsupported clean calibration schema")
        exp = {}
        for row in h5.root.exposures.rows:
            key = (text(row["H5"]), int(row["exposure"]))
            exp[key] = {name: row[name] for name in h5.root.exposures.rows.colnames}
        ifus = {}
        for row in h5.root.physical_ifus.rows:
            key = (int(row["SPECID"]), int(row["IFUSLOT"]), int(row["IFUID"]))
            ifus[key] = {name: row[name] for name in h5.root.physical_ifus.rows.colnames}
        amps = {}
        for row in h5.root.physical_amplifiers.rows:
            ifu = (int(row["SPECID"]), int(row["IFUSLOT"]), int(row["IFUID"]))
            key = ifu + (text(row["AMP"]),)
            amps[key] = {name: row[name] for name in h5.root.physical_amplifiers.rows.colnames}
        observations = {}
        for row in h5.root.amplifier_observations.rows:
            key = (text(row["H5"]), int(row["exposure"]), int(row["SPECID"]),
                   int(row["IFUSLOT"]), int(row["IFUID"]), text(row["AMP"]))
            observations[key] = {name: row[name] for name in h5.root.amplifier_observations.rows.colnames}
        metadata = {}
        for row in h5.root.provenance.metadata:
            metadata[text(row["key"])] = json.loads(text(row["value"]))
        return exp, ifus, amps, observations, metadata


def source_measurements(path, h5_paths, calibration, compact_path):
    exp, ifus, amps, observations, metadata = calibration
    with tables.open_file(path, "r") as h5:
        rows = h5.root.measurements.read()
        input_rows = h5.root.provenance.h5_inputs.read()
        measurement_ids = sorted(set(int(value) for value in rows["h5_id"]))
        input_names = [Path(text(row["filename"])).name for row in input_rows]
        if len(input_names) != len(measurement_ids):
            raise ValueError("measurement H5 input provenance count does not match h5_id values")
        names = dict(zip(measurement_ids, input_names))
    compact = apply_compact_mask(rows, compact_path)
    grouped = {}
    for index, row in enumerate(rows):
        key = (names[int(row["h5_id"])], int(row["exposure"]), int(row["SPECID"]),
               int(row["IFUSLOT"]), int(row["IFUID"]), text(row["AMP"]))
        grouped.setdefault(key, []).append(index)
    records = []
    for key, indices in sorted(grouped.items()):
        if len(indices) != N_AMP_FIBERS:
            raise ValueError("measurement amplifier observation %s has %d rows, expected 112" % (key, len(indices)))
        if key not in observations:
            raise ValueError("measurement observation missing from calibration: %s" % (key,))
        obs = observations[key]
        index = np.asarray(indices, dtype=int)
        data = np.asarray(rows["data_work"][index], float)
        error = np.asarray(rows["error_work"][index], float)
        X = np.asarray(rows["external_prediction"][index], float)
        external = np.asarray(rows["external_valid"][index], bool)
        valid = external & np.isfinite(data) & np.isfinite(error) & (error > 0) & np.isfinite(X)
        valid &= ~np.asarray(rows["date_mask_bad"][index], bool)[:, None]
        valid &= compact[index, None]
        additive = np.asarray(obs["additive_R_ON_OFF"], float)[None, :] + np.asarray(obs["additive_IFU_ON_OFF"], float)[None, :] + np.asarray(obs["additive_AMP_ON_OFF"], float)[None, :]
        Dcorr = data - additive
        z = float(obs["posterior_z_mean"])
        m = np.exp(z)
        z_pred = m * X
        sigma_eff = np.sqrt((np.sqrt(2.58) * error) ** 2 + (np.asarray([.0590, .0307])[None, :] * X) ** 2)
        for band, band_name in enumerate(("ON", "OFF")):
            use = valid[:, band]
            source_use = use & (np.abs(X[:, band]) > 1e-12)
            for local, row_index in enumerate(index):
                if not use[local]:
                    continue
                records.append({"H5": key[0], "exposure": key[1], "SPECID": key[2],
                                "IFUSLOT": key[3], "IFUID": key[4], "AMP": key[5],
                                "band": band_name, "source_information": float((X[local, band] / sigma_eff[local, band]) ** 2) if source_use[local] else 0.0,
                                "near_zero_X": bool(not source_use[local]),
                                "r_z0": float(Dcorr[local, band] - X[local, band]),
                                "r_z": float(Dcorr[local, band] - z_pred[local, band]),
                                "r_cal": float(Dcorr[local, band] / m - X[local, band]),
                                "normalized_z0": float((Dcorr[local, band] - X[local, band]) / sigma_eff[local, band]),
                                "normalized_z": float((Dcorr[local, band] - z_pred[local, band]) / sigma_eff[local, band]),
                                "normalized_cal": float((Dcorr[local, band] / m - X[local, band]) / (sigma_eff[local, band] / m))})
    return records


def summarize_records(records, group_fields):
    groups = {}
    for record in records:
        key = tuple(record[field] for field in group_fields)
        groups.setdefault(key, []).append(record)
    output = []
    for key, values in sorted(groups.items(), key=lambda pair: str(pair[0])):
        row = {field: value for field, value in zip(group_fields, key)}
        for metric in ("r_z0", "r_z", "r_cal", "normalized_z0", "normalized_z", "normalized_cal"):
            array = np.asarray([item[metric] for item in values], float)
            row[metric + "_rms"] = float(np.sqrt(np.mean(array * array)))
            row[metric + "_robust_rms"] = robust_rms(array)
            row[metric + "_p16"] = float(np.percentile(array, 16))
            row[metric + "_median"] = float(np.median(array))
            row[metric + "_p84"] = float(np.percentile(array, 84))
        row["n"] = len(values)
        row["n_near_zero_X"] = int(sum(item["near_zero_X"] for item in values))
        output.append(row)
    return output


def blank_qa(h5_paths, blank_masks, calibration, minimum_fraction):
    exp, ifus, amps, observations, metadata = calibration
    stages = {stage: [] for stage in STAGES}
    group_rows = []
    native_max = 0.0
    native_rel = 0.0
    q_max = 0.0
    null_rows = []
    for h5_path in h5_paths:
        with tables.open_file(h5_path, "r") as h5:
            groups, labels = exposure_groups(h5.root.Info)
            slots = np.asarray(h5.root.Info.cols.ifuslot[:])
            amps_col = np.asarray([text(v) for v in h5.root.Info.cols.amp[:]])
            blocked = date_mask(h5_path, slots, amps_col)
            surveys = {int(row["exp"]): row for row in h5.root.Survey}
            blank = blank_masks[h5_path.name]
            for group in groups:
                key = (h5_path.name, group["exposure"], group["SPECID"], group["IFUSLOT"], group["IFUID"], group["AMP"])
                if key not in observations:
                    raise ValueError("blank group missing from calibration: %s" % (key,))
                candidates = group["indices"][blank[group["indices"]] & ~blocked[group["indices"]]]
                if not candidates.size:
                    continue
                spectrum = np.asarray(h5.root.Fibers.read_coordinates(candidates, field="spectrum"), float) / float(surveys[group["exposure"]]["offset"])
                usable = np.mean(np.isfinite(spectrum), axis=1) >= minimum_fraction
                spectrum = spectrum[usable]
                if not spectrum.size:
                    continue
                row = observations[key]; ifu = (group["SPECID"], group["IFUSLOT"], group["IFUID"]); amp = ifu + (group["AMP"],)
                e = exp[(key[0], key[1])]
                c = float(ifus.get(ifu, {"c": 0.0})["c"]); d = float(amps.get(amp, {"d": 0.0})["d"])
                R = np.asarray(e["R_lambda"], float); U = np.asarray(e["U_lambda"], float); V = np.asarray(e["V_lambda"], float)
                values = (spectrum, spectrum - R, spectrum - R - c * U, spectrum - R - c * U - d * V)
                group_row = {"H5": key[0], "exposure": key[1], "SPECID": key[2], "IFUSLOT": key[3], "IFUID": key[4], "AMP": key[5], "n_fibers": int(spectrum.shape[0])}
                for stage, value in zip(STAGES, values):
                    robust_group = robust_spectrum(value)
                    stages[stage].append(robust_group)
                    group_row[stage + "_robust_rms"] = robust_rms(robust_group)
                    group_row[stage + "_location"] = robust_location(robust_group)
                m = np.exp(float(row["posterior_z_mean"]))
                sigma3 = np.asarray(h5.root.Fibers.cols.error[candidates[usable]], float) / np.abs(float(surveys[group["exposure"]]["offset"]))
                finite = np.isfinite(values[3]) & np.isfinite(sigma3) & (sigma3 > 0)
                direct = np.where(finite, values[3] / m, np.nan)
                predicted = np.where(finite, values[3] / m, np.nan)
                if np.any(finite):
                    native_max = max(native_max, float(np.nanmax(np.abs(direct - predicted))))
                    native_rel = max(native_rel, float(np.nanmax(np.abs(direct - predicted) / np.maximum(np.abs(predicted), 1e-30))))
                    q3 = values[3] / sigma3; q4 = direct / (sigma3 / m)
                    q_max = max(q_max, float(np.nanmax(np.abs(q4[finite] - q3[finite]))))
                group_rows.append(group_row)
                null_values = np.asarray([weighted_scalar(robust_spectrum(values[3]), ((WAVE >= lo) & (WAVE <= hi)).astype(float)) for lo, hi in ((3538,3550),(3602,3614),(3628,3640),(3676,3688),(3902,3914))])
                null_rows.append({"H5": key[0], "exposure": key[1], "IFUID": key[4], "AMP": key[5], **{name: float(value) for name, value in zip(NULL_NAMES, null_values)}})
    matrices = {stage: np.asarray(values) for stage, values in stages.items()}
    stage_rows = []
    for stage, matrix in matrices.items():
        for wave, wavelength in enumerate(WAVE):
            values = matrix[:, wave] if matrix.size else np.asarray([])
            finite = values[np.isfinite(values)]
            stage_rows.append({"stage": stage, "wavelength_index": wave, "wavelength": float(wavelength), "n": int(finite.size), "location": robust_location(finite), "robust_scatter": robust_scale(finite), "rms": float(np.sqrt(np.mean(finite * finite))) if finite.size else np.nan})
    return stage_rows, group_rows, null_rows, {"max_abs_r4_minus_r3_over_m": native_max, "max_relative_r4_identity": native_rel, "max_abs_normalized_invariance": q_max}, matrices


def figures(output_dir, stage_rows, matrices, source_records):
    output_dir = Path(output_dir)
    for stage in STAGES:
        values = [row["location"] for row in stage_rows if row["stage"] == stage]
        scatter = [row["robust_scatter"] for row in stage_rows if row["stage"] == stage]
        plt.plot(WAVE, values, label=stage)
        plt.fill_between(WAVE, np.asarray(values) - scatter, np.asarray(values) + scatter, alpha=.15)
    plt.xlabel("wavelength"); plt.ylabel("blank residual in W"); plt.legend(); plt.tight_layout(); plt.savefig(output_dir / "blank_full_spectrum.png", dpi=130); plt.close()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for band, name in enumerate(("ON", "OFF")):
        values = [row["r_z" if band == 0 else "r_z"] for row in source_records if row["band"] == name]
        axes[band].hist(values, bins=60, histtype="step", label=name)
        axes[band].set_title(name); axes[band].set_xlabel("source residual after z")
    fig.tight_layout(); fig.savefig(output_dir / "source_residuals.png", dpi=130); plt.close(fig)
    # The group-index maps are compact deterministic QA maps in focal-plane order.
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for axis, stage in zip(axes, STAGES[1:]):
        values = np.asarray([robust_rms(matrices[stage][i]) for i in range(len(matrices[stage]))])
        axis.scatter(np.arange(values.size), values, s=5)
        axis.set_title(stage); axis.set_xlabel("deterministic blank group"); axis.set_ylabel("robust RMS")
    fig.tight_layout(); fig.savefig(output_dir / "blank_group_spatial_qa.png", dpi=130); plt.close(fig)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("h5_positional", nargs="*"); parser.add_argument("--h5", nargs="+", action="append"); parser.add_argument("--h5-glob", action="append")
    parser.add_argument("--measurements", required=True); parser.add_argument("--calibration", required=True); parser.add_argument("--external-blank-fibers", required=True); parser.add_argument("--on-filter", required=True); parser.add_argument("--off-filter", required=True); parser.add_argument("--compact-mask"); parser.add_argument("--output-dir", default="m101_calibration_validation"); parser.add_argument("--minimum-finite-fraction", type=float, default=.8); parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise SystemExit("output directory is nonempty; use --overwrite: %s" % output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    h5_paths = discover_h5(args)
    blank_masks, blank_id = load_blank(args.external_blank_fibers, h5_paths)
    filters = {"ON": read_filter(args.on_filter), "OFF": read_filter(args.off_filter)}
    calibration = product(args.calibration)
    stage_rows, group_rows, null_rows, blank_identity, matrices = blank_qa(h5_paths, blank_masks, calibration, args.minimum_finite_fraction)
    source_records = source_measurements(args.measurements, h5_paths, calibration, args.compact_mask)
    source_summaries = summarize_records(source_records, ["band"])
    for fields, rows, name in ((["stage","wavelength_index","wavelength","n","location","robust_scatter","rms"], stage_rows, "additive_stage_statistics.csv"),
                               (list(group_rows[0]) if group_rows else [], group_rows, "per_group_qa.csv"),
                               (list(null_rows[0]) if null_rows else [], null_rows, "null_band_statistics.csv"),
                               (list(source_summaries[0]) if source_summaries else [], source_summaries, "source_residual_statistics.csv"),
                               (list(source_records[0]) if source_records else [], source_records, "source_rows.csv")):
        if rows: write_csv(output_dir / name, rows, fields)
    figures(output_dir, stage_rows, matrices, source_records)
    stage_rms = {stage: float(np.sqrt(np.nanmean(matrix * matrix))) for stage, matrix in matrices.items()}
    posterior = np.asarray([float(row["posterior_z_mean"]) for row in calibration[3].values()])
    summary = {"schema_version": "m101_calibration_validation_v1", "calibration": identity(args.calibration), "measurements": identity(args.measurements), "input_h5": [identity(path) for path in h5_paths], "external_blank_fibers": blank_id, "filters": {name: identity(path) for name, path in (("ON", args.on_filter), ("OFF", args.off_filter))}, "source_likelihood_mask": "external_valid + compact_mask + date_mask + finite/error; sky_blank is not used as a complement", "near_zero_X_rows": int(sum(row["near_zero_X"] for row in source_records)), "blank_identity": blank_identity, "blank_stage_robust_rms": {stage: robust_rms(matrix) for stage, matrix in matrices.items()}, "blank_stage_rms": stage_rms, "blank_identity_checks": blank_identity, "source_summaries": source_summaries, "posterior_z": {"p16": float(np.percentile(posterior,16)), "median": float(np.median(posterior)), "p84": float(np.percentile(posterior,84))}, "files": sorted(path.name for path in output_dir.iterdir())}
    (output_dir / "m101_calibration_validation_summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    print(json.dumps({"output_dir": str(output_dir), "summary": str(output_dir / "m101_calibration_validation_summary.json"), "blank_identity": blank_identity, "source_summaries": source_summaries}, indent=2, default=str))


if __name__ == "__main__":
    main()
