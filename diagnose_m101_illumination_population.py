#!/usr/bin/env python3
"""Run and aggregate the validated M101 illumination diagnostic population.

This driver deliberately treats ``diagnose_m101_hierarchical.py`` as a black
box.  It runs one independent diagnostic directory per accepted H5, then uses
only the resulting illumination CSVs for population-level diagnostics.  No
illumination correction is applied to spectra, H5 files, or production cubes.
"""

from argparse import ArgumentParser
import csv
from pathlib import Path
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tables

import diagnose_m101_hierarchical as single


H5_NAMES = (
    "20200430_0000020.h5", "20200430_0000021.h5", "20200430_0000022.h5",
    "20200517_0000013.h5", "20200517_0000014.h5", "20200517_0000015.h5",
    "20200521_0000019.h5", "20200521_0000020.h5", "20200521_0000021.h5",
    "20200523_0000022.h5", "20200523_0000024.h5",
    "20200525_0000020.h5", "20200525_0000021.h5", "20200525_0000022.h5",
    "20200528_0000015.h5", "20200622_0000015.h5", "20200625_0000017.h5",
    "20200710_0000013.h5", "20200710_0000014.h5",
)
EXCLUDED_H5 = "20200523_0000023.h5"
REQUIRED_OUTPUTS = ("illumination_ifu_scalars.csv",
                    "illumination_plane_parameters.csv",
                    "illumination_global_offsets.csv")


def as_bool(value):
    return str(value).strip().lower() in {"true", "1", "yes"}


def as_float(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return np.nan
    return value if np.isfinite(value) else np.nan


def read_csv(path):
    with Path(path).open(newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows, fields):
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field, np.nan) for field in fields}
                         for row in rows)


def read_track_states(h5_paths):
    """Read Survey.name for every exposure; never infer track from date/order."""
    states = {}
    for h5_path in h5_paths:
        with tables.open_file(h5_path, mode="r") as h5:
            if "Survey" not in h5.root._v_children:
                raise ValueError("%s has no Survey table for track classification" % h5_path)
            seen = set()
            for survey_row in h5.root.Survey:
                exposure = int(survey_row["exp"])
                survey_name = single.as_text(survey_row["name"])
                has_e, has_w = "_E" in survey_name, "_W" in survey_name
                if has_e == has_w:
                    raise ValueError("%s exposure %d Survey.name must contain exactly one of _E/_W: %r" %
                                     (h5_path.name, exposure, survey_name))
                key = (h5_path.name, exposure)
                if key in seen:
                    raise ValueError("%s has duplicate Survey exposure %d" % (h5_path.name, exposure))
                seen.add(key)
                states[key] = {"survey_name": survey_name, "track": "E" if has_e else "W"}
            if seen != {(h5_path.name, exposure) for exposure in (1, 2, 3)}:
                raise ValueError("%s Survey must contain exactly exposures 1, 2, 3" % h5_path.name)
    print("track summary: E=%d W=%d" %
          (sum(value["track"] == "E" for value in states.values()),
           sum(value["track"] == "W" for value in states.values())))
    for h5_path in h5_paths:
        for exposure in (1, 2, 3):
            state = states[(h5_path.name, exposure)]
            print("  %s exposure %d %s %s" %
                  (h5_path.name, exposure, state["survey_name"], state["track"]))
    return states


def resolve_h5_paths(h5_dir):
    h5_dir = Path(h5_dir)
    paths = []
    missing = []
    for name in H5_NAMES:
        direct = h5_dir / name
        matches = [direct] if direct.is_file() else list(h5_dir.rglob(name))
        if len(matches) != 1:
            missing.append("%s (%s matches)" % (name, len(matches)))
        else:
            paths.append(matches[0])
    if missing:
        raise FileNotFoundError("sample resolution failed; expected exactly the 19 allowlisted H5s:\n  " +
                                "\n  ".join(missing))
    if EXCLUDED_H5 in {path.name for path in paths}:
        raise RuntimeError("excluded H5 was resolved: %s" % EXCLUDED_H5)
    return paths


def run_single_h5(path, args, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.skip_existing and all((output_dir / name).is_file() for name in REQUIRED_OUTPUTS):
        print("skip existing %s" % path.name)
        return
    command = [sys.executable, str(args.single_script),
               "--h5", str(path),
               "--on-image", str(args.on_image), "--off-image", str(args.off_image),
               "--on-filter", str(args.on_filter), "--off-filter", str(args.off_filter),
               "--fq-template", str(args.fq_template),
               "--output-dir", str(output_dir)]
    if args.iterations is not None:
        command.extend(["--iterations", str(args.iterations)])
    print("run %s" % path.name, flush=True)
    subprocess.run(command, check=True)


def load_population_rows(output_dir, h5_paths, track_states):
    rows = []
    for h5_index, h5_path in enumerate(h5_paths):
        directory = output_dir / h5_path.stem
        scalar_rows = read_csv(directory / "illumination_ifu_scalars.csv")
        plane_rows = {int(row["exposure"]): row for row in
                      read_csv(directory / "illumination_plane_parameters.csv")}
        offset_rows = {(int(row["exposure"]), row["band"]): row for row in
                       read_csv(directory / "illumination_global_offsets.csv")}
        date, shot = h5_path.stem.split("_")
        for scalar in scalar_rows:
            exposure = int(scalar["exposure"])
            plane = plane_rows[exposure]
            base = dict(scalar)
            base.update({"h5": h5_path.name, "date": date, "shot": shot,
                         "h5_index": h5_index, "exposure_order": h5_index * 3 + exposure - 1})
            state = track_states[(h5_path.name, exposure)]
            base.update({"survey_name": state["survey_name"], "track": state["track"]})
            for field in ("cx", "cy", "ra0", "dec0", "robust_RMS_before",
                          "robust_RMS_after", "n_IFU_used", "n_IFU_rejected"):
                base[field] = plane.get(field, np.nan)
            for band, suffix in (("ON", "ON"), ("OFF", "OFF")):
                offset = offset_rows[(exposure, band)]
                base["delta_%s" % suffix] = offset["delta_illumination"]
            ra = as_float(base["mean_RA"]); dec = as_float(base["mean_Dec"])
            ra0 = as_float(base["ra0"]); dec0 = as_float(base["dec0"])
            cx = as_float(base["cx"]); cy = as_float(base["cy"])
            x = (ra - ra0) * np.cos(np.deg2rad(dec0)) * 60.0
            y = (dec - dec0) * 60.0
            s_common = as_float(base["s_common_normalized"])
            s_plane = 1.0 + cx * x + cy * y
            base.update({"x_arcmin": x, "y_arcmin": y, "s_plane": s_plane,
                         "r_raw": s_common - 1.0,
                         "r_IFU": s_common - s_plane,
                         "r_ON": as_float(base["s_ON_normalized"]) - s_plane,
                         "r_OFF": as_float(base["s_OFF_normalized"]) - s_plane})
            if not as_bool(base["well_constrained_common"]):
                base["r_IFU"] = np.nan
            rows.append(base)
    rows.sort(key=lambda row: int(row["exposure_order"]))
    return rows


def population_fields():
    return ["h5", "date", "shot", "exposure", "survey_name", "track",
            "SPECID", "IFUSLOT", "IFUID",
            "s_ON_raw", "s_OFF_raw", "s_common_raw", "s_ON_normalized",
            "s_OFF_normalized", "s_common_normalized", "well_constrained_ON",
            "well_constrained_OFF", "well_constrained_common", "mean_RA", "mean_Dec",
            "cx", "cy", "ra0", "dec0", "delta_ON", "delta_OFF", "x_arcmin",
            "y_arcmin", "s_plane", "r_IFU", "r_ON", "r_OFF", "n_good_amps",
            "n_fibers_ON", "n_fibers_OFF", "scalar_RMS_ON", "scalar_RMS_OFF",
            "scalar_RMS_common", "scalar_uncertainty_ON", "scalar_uncertainty_OFF",
            "scalar_uncertainty_common", "leverage_ON", "leverage_OFF",
            "median_source_ON", "median_source_OFF", "median_sky_ON", "median_sky_OFF",
            "median_total_ON", "median_total_OFF", "sky_fraction_ON", "sky_fraction_OFF"]


def valid_common(row):
    return as_bool(row["well_constrained_common"]) and np.isfinite(as_float(row["r_IFU"]))


def ifu_key(row):
    return (int(row["SPECID"]), int(row["IFUSLOT"]), int(row["IFUID"]))


def exposure_key(row):
    return (row["h5"], int(row["exposure"]))


def common_rows(rows):
    return [row for row in rows if valid_common(row)]


def robust_correlation(x, y):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if x.size < 3 or np.ptp(x) == 0.0 or np.ptp(y) == 0.0:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def make_residual_matrix(rows, output_path):
    ordered_exposures = sorted({exposure_key(row): row for row in rows},
                               key=lambda key: (next(row["h5_index"] for row in rows if exposure_key(row) == key), key[1]))
    identities = sorted({ifu_key(row) for row in rows}, key=lambda key: (key[1], key[0], key[2]))
    index = {key: i for i, key in enumerate(identities)}
    matrix = np.full((len(ordered_exposures), len(identities)), np.nan)
    labels = []
    for i, key in enumerate(ordered_exposures):
        h5, exposure = key
        labels.append("%s\ne%d" % (h5[:8], exposure))
        for row in rows:
            if exposure_key(row) == key and valid_common(row):
                matrix[i, index[ifu_key(row)]] = as_float(row["r_IFU"])
    values = np.abs(matrix[np.isfinite(matrix)])
    scale = max(float(np.percentile(values, 98)) if values.size else .01, .01)
    fig, axis = plt.subplots(figsize=(max(12, len(identities) * .18), 12))
    masked = np.ma.masked_invalid(matrix)
    image = axis.imshow(masked, aspect="auto", interpolation="none", cmap="coolwarm",
                        vmin=-scale, vmax=scale)
    axis.set_xlabel("physical IFU: IFUSLOT (SPECID, IFUID tie-breakers)")
    axis.set_ylabel("H5 / exposure")
    axis.set_yticks(np.arange(len(labels))); axis.set_yticklabels(labels, fontsize=6)
    axis.set_xticks(np.arange(len(identities)))
    axis.set_xticklabels(["%d:%d:%d" % key for key in identities], rotation=90, fontsize=5)
    last_h5 = None
    for i, key in enumerate(ordered_exposures):
        if last_h5 is not None and key[0] != last_h5:
            axis.axhline(i - .5, color="k", lw=1.0)
        last_h5 = key[0]
    fig.colorbar(image, ax=axis, label="r_IFU = s_common_normalized - s_plane")
    fig.suptitle("57-exposure plane-subtracted physical-IFU residual matrix")
    fig.tight_layout(); fig.savefig(output_path, dpi=170); plt.close(fig)
    return matrix, ordered_exposures, identities


def make_template(rows, output_dir):
    grouped = {}
    for row in common_rows(rows):
        grouped.setdefault(ifu_key(row), []).append(as_float(row["r_IFU"]))
    locations = {}
    template_rows = []
    for key, values in sorted(grouped.items(), key=lambda item: (item[0][1], item[0][0], item[0][2])):
        values = np.asarray(values)
        positions = [row for row in rows if ifu_key(row) == key and valid_common(row)]
        locations[key] = (single.robust_location([as_float(row["mean_RA"]) for row in positions]),
                          single.robust_location([as_float(row["mean_Dec"]) for row in positions]))
        template_rows.append({"SPECID": key[0], "IFUSLOT": key[1], "IFUID": key[2],
                              "C_IFU": single.robust_location(values),
                              "N_exposures": values.size,
                              "robust_scatter": single.robust_scale(values),
                              "p16": np.percentile(values, 16), "p84": np.percentile(values, 84)})
    fields = ["SPECID", "IFUSLOT", "IFUID", "C_IFU", "N_exposures",
              "robust_scatter", "p16", "p84"]
    write_csv(output_dir / "physical_ifu_response_template.csv", template_rows, fields)
    c = np.asarray([as_float(row["C_IFU"]) for row in template_rows])
    scatter = np.asarray([as_float(row["robust_scatter"]) for row in template_rows])
    counts = np.asarray([as_float(row["N_exposures"]) for row in template_rows])
    ra = np.asarray([locations[(int(row["SPECID"]), int(row["IFUSLOT"]), int(row["IFUID"]))][0]
                     for row in template_rows])
    dec = np.asarray([locations[(int(row["SPECID"]), int(row["IFUSLOT"]), int(row["IFUID"]))][1]
                      for row in template_rows])
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), squeeze=False)
    axes = axes[0]
    cscale = max(float(np.percentile(np.abs(c[np.isfinite(c)]), 95)) if np.isfinite(c).any() else .01, .01)
    axes[0].scatter(ra, dec, c=c, cmap="coolwarm", vmin=-cscale, vmax=cscale, s=45)
    axes[0].set_title("C_IFU"); axes[0].set_xlabel("RA"); axes[0].set_ylabel("Dec")
    axes[1].scatter(ra, dec, c=scatter, cmap="viridis", s=45)
    axes[1].set_title("robust exposure scatter"); axes[1].set_xlabel("RA"); axes[1].set_ylabel("Dec")
    axes[2].scatter(ra, dec, c=counts, cmap="plasma", s=45)
    axes[2].set_title("N measurements"); axes[2].set_xlabel("RA"); axes[2].set_ylabel("Dec")
    for axis in axes: axis.grid(alpha=.2)
    fig.suptitle("Unsmooth physical-IFU response template")
    fig.tight_layout(rect=(0, 0, 1, .95)); fig.savefig(output_dir / "physical_ifu_response_template.png", dpi=170); plt.close(fig)
    return {key: as_float(row["C_IFU"]) for key, row in zip(
        [(int(row["SPECID"]), int(row["IFUSLOT"]), int(row["IFUID"])) for row in template_rows], template_rows)}


def track_correlation_diagnostics(rows, output_dir, value_field="r_IFU", suffix=""):
    """Compare E/E, W/W, and E/W correlations in the requested residual basis."""
    exposure_keys = sorted({exposure_key(row) for row in rows},
                           key=lambda key: next(row["exposure_order"] for row in rows if exposure_key(row) == key))
    track_by_exposure = {key: next(row["track"] for row in rows if exposure_key(row) == key)
                         for key in exposure_keys}
    ordered = sorted(exposure_keys, key=lambda key: (track_by_exposure[key] != "E",
                                                      next(row["exposure_order"] for row in rows if exposure_key(row) == key)))
    values_by_exposure = {}
    for key in ordered:
        values_by_exposure[key] = {ifu_key(row): as_float(row[value_field]) for row in rows
                                   if exposure_key(row) == key and
                                   as_bool(row["well_constrained_common"]) and
                                   np.isfinite(as_float(row[value_field]))}
    matrix = np.full((len(ordered), len(ordered)), np.nan)
    distributions = []
    for i, left in enumerate(ordered):
        matrix[i, i] = 1.0
        for j in range(i + 1, len(ordered)):
            right = ordered[j]
            common = sorted(set(values_by_exposure[left]) & set(values_by_exposure[right]))
            x = np.asarray([values_by_exposure[left][key] for key in common])
            y = np.asarray([values_by_exposure[right][key] for key in common])
            correlation = robust_correlation(x, y)
            matrix[i, j] = matrix[j, i] = correlation
            pair_type = track_by_exposure[left] + "-" + track_by_exposure[right]
            distributions.append((pair_type, correlation))
    summary_rows = []
    for pair_type in ("E-E", "W-W", "E-W"):
        values = np.asarray([value for kind, value in distributions if kind == pair_type and np.isfinite(value)])
        summary_rows.append({"quantity": value_field, "pair_type": pair_type, "N_pairs": values.size,
                             "median": np.median(values) if values.size else np.nan,
                             "p16": np.percentile(values, 16) if values.size else np.nan,
                             "p84": np.percentile(values, 84) if values.size else np.nan})
        print("%s %s correlations: N=%d median=%.6g p16/p84=%.6g/%.6g" %
              (value_field, pair_type, summary_rows[-1]["N_pairs"], summary_rows[-1]["median"],
               summary_rows[-1]["p16"], summary_rows[-1]["p84"]))
    return ordered, matrix, summary_rows


def plot_track_correlation_matrix(rows, output_dir):
    ordered, matrix, summary = track_correlation_diagnostics(rows, output_dir, "r_IFU")
    _, _, raw_summary = track_correlation_diagnostics(rows, output_dir, "r_raw")
    write_csv(output_dir / "illumination_track_correlation_summary.csv",
              summary + raw_summary, ["quantity", "pair_type", "N_pairs", "median", "p16", "p84"])
    scale = max(float(np.nanpercentile(np.abs(matrix), 98)) if np.isfinite(matrix).any() else 1.0, .05)
    fig, axis = plt.subplots(figsize=(12, 10))
    image = axis.imshow(matrix, cmap="coolwarm", vmin=-scale, vmax=scale, interpolation="none")
    axis.axhline(sum(next(row["track"] for row in rows if exposure_key(row) == key) == "E" for key in ordered) - .5,
                 color="k", lw=2.0)
    axis.axvline(sum(next(row["track"] for row in rows if exposure_key(row) == key) == "E" for key in ordered) - .5,
                 color="k", lw=2.0)
    axis.set_title("Exposure correlations reordered by Survey track (r_IFU)")
    axis.set_xlabel("E exposures, then W exposures"); axis.set_ylabel("E exposures, then W exposures")
    fig.colorbar(image, ax=axis, label="Pearson correlation")
    fig.tight_layout(); fig.savefig(output_dir / "illumination_exposure_correlation_matrix_by_track.png", dpi=170); plt.close(fig)
    return summary, raw_summary


def make_track_templates(rows, output_dir):
    grouped = {}
    locations = {}
    for track in ("E", "W"):
        grouped[track] = {}
        selected = [row for row in rows if row["track"] == track and valid_common(row)]
        for row in selected:
            grouped[track].setdefault(ifu_key(row), []).append(as_float(row["r_IFU"]))
    identities = sorted(set(grouped["E"]) | set(grouped["W"]), key=lambda key: (key[1], key[0], key[2]))
    template_rows = []
    templates = {"E": {}, "W": {}}
    for key in identities:
        output = {"SPECID": key[0], "IFUSLOT": key[1], "IFUID": key[2]}
        for track in ("E", "W"):
            values = np.asarray(grouped[track].get(key, []), dtype=float)
            templates[track][key] = single.robust_location(values) if values.size else np.nan
            output.update({"C_%s" % track: templates[track][key],
                           "scatter_%s" % track: single.robust_scale(values) if values.size else np.nan,
                           "N_%s" % track: values.size,
                           "p16_%s" % track: np.percentile(values, 16) if values.size else np.nan,
                           "p84_%s" % track: np.percentile(values, 84) if values.size else np.nan})
        output["C_W_minus_C_E"] = output["C_W"] - output["C_E"] \
            if np.isfinite(output["C_W"]) and np.isfinite(output["C_E"]) else np.nan
        positions = [row for row in rows if ifu_key(row) == key and valid_common(row)]
        locations[key] = (single.robust_location([as_float(row["mean_RA"]) for row in positions]),
                          single.robust_location([as_float(row["mean_Dec"]) for row in positions]))
        template_rows.append(output)
    fields = ["SPECID", "IFUSLOT", "IFUID", "C_E", "scatter_E", "N_E", "p16_E", "p84_E",
              "C_W", "scatter_W", "N_W", "p16_W", "p84_W", "C_W_minus_C_E"]
    write_csv(output_dir / "physical_ifu_response_template_by_track.csv", template_rows, fields)
    return templates, locations, template_rows


def plot_track_templates(rows, templates, locations, template_rows, output_dir):
    ra = np.asarray([locations[(int(row["SPECID"]), int(row["IFUSLOT"]), int(row["IFUID"]))][0]
                     for row in template_rows]); dec = np.asarray([locations[(int(row["SPECID"]), int(row["IFUSLOT"]), int(row["IFUID"]))][1]
                                                                      for row in template_rows])
    data = {name: np.asarray([as_float(row[name]) for row in template_rows])
            for name in ("C_E", "C_W", "C_W_minus_C_E", "scatter_E", "scatter_W", "N_E", "N_W")}
    cscale = max(float(np.nanpercentile(np.abs(np.concatenate((data["C_E"], data["C_W"]))), 95)), .01)
    dscale = max(float(np.nanpercentile(np.abs(data["C_W_minus_C_E"]), 95)), .01)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for axis, name, scale, title in zip(axes, ("C_E", "C_W", "C_W_minus_C_E"), (cscale, cscale, dscale), ("C_E", "C_W", "C_W - C_E")):
        finite = np.isfinite(data[name])
        axis.scatter(ra[finite], dec[finite], c=data[name][finite], cmap="coolwarm", vmin=-scale, vmax=scale, s=42)
        axis.set_title(title); axis.set_xlabel("RA"); axis.set_ylabel("Dec"); axis.grid(alpha=.2)
    fig.suptitle("Track-specific physical-IFU response templates (identity is SPECID/IFUSLOT/IFUID)")
    fig.tight_layout(rect=(0, 0, 1, .95)); fig.savefig(output_dir / "physical_ifu_response_templates_E_W.png", dpi=170); plt.close(fig)


def plot_track_template_comparison(template_rows, output_dir):
    selected = [row for row in template_rows if np.isfinite(as_float(row["C_E"])) and np.isfinite(as_float(row["C_W"]))]
    x = np.asarray([as_float(row["C_E"]) for row in selected]); y = np.asarray([as_float(row["C_W"]) for row in selected])
    fit = single.fit_line(x, y); residual = y - (fit["g"] * x + fit["z"])
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].scatter(x, y, s=24, alpha=.75)
    if x.size >= 3:
        grid = np.linspace(min(x.min(), y.min()), max(x.max(), y.max()), 100)
        axes[0].plot(grid, grid, "k:", label="identity"); axes[0].plot(grid, fit["g"] * grid + fit["z"], "b-", label="robust fit")
    axes[0].set_xlabel("C_E"); axes[0].set_ylabel("C_W"); axes[0].legend(fontsize=8)
    axes[0].text(.03, .97, "r=%.5g\nrobust scatter=%.5g\nN=%d" % (robust_correlation(x, y), single.robust_rms(residual), x.size), transform=axes[0].transAxes, va="top")
    order = np.argsort([int(row["IFUSLOT"]) for row in selected])
    axes[1].plot(np.arange(len(selected)), (y - x)[order], "o-", ms=3)
    axes[1].axhline(0, color="k", lw=.7); axes[1].set_xlabel("IFU identity sorted by IFUSLOT"); axes[1].set_ylabel("C_W - C_E"); axes[1].grid(alpha=.2)
    fig.suptitle("East versus West physical-IFU response")
    fig.tight_layout(rect=(0, 0, 1, .95)); fig.savefig(output_dir / "physical_ifu_response_E_vs_W.png", dpi=170); plt.close(fig)
    return {"N": x.size, "correlation": robust_correlation(x, y), "robust_scatter": single.robust_rms(residual),
            "slope": fit["g"], "intercept": fit["z"]}


def raw_track_template_summary(rows, output_dir):
    grouped = {"E": {}, "W": {}}
    for row in rows:
        if not as_bool(row["well_constrained_common"]):
            continue
        value = as_float(row["s_common_normalized"])
        if np.isfinite(value):
            grouped[row["track"]].setdefault(ifu_key(row), []).append(value - 1.0)
    east = {key: single.robust_location(values) for key, values in grouped["E"].items()}
    west = {key: single.robust_location(values) for key, values in grouped["W"].items()}
    common = sorted(set(east) & set(west))
    x = np.asarray([east[key] for key in common]); y = np.asarray([west[key] for key in common])
    result = {"N_common_IFU": len(common), "C_E_raw_C_W_raw_correlation": robust_correlation(x, y),
              "C_E_raw_robust_scatter": single.robust_scale(x),
              "C_W_raw_robust_scatter": single.robust_scale(y)}
    write_csv(output_dir / "illumination_track_no_plane_summary.csv", [result],
              ["N_common_IFU", "C_E_raw_C_W_raw_correlation",
               "C_E_raw_robust_scatter", "C_W_raw_robust_scatter"])
    print("raw C_E versus C_W correlation: N=%d r=%.6g" %
          (result["N_common_IFU"], result["C_E_raw_C_W_raw_correlation"]))
    return result


def template_for_rows(rows, track=None, excluded_h5=None, value_field="r_IFU"):
    grouped = {}
    for row in rows:
        if excluded_h5 is not None and row["h5"] == excluded_h5:
            continue
        if track is not None and row["track"] != track:
            continue
        if value_field == "r_IFU" and not valid_common(row):
            continue
        value = as_float(row[value_field])
        if np.isfinite(value):
            grouped.setdefault(ifu_key(row), []).append(value)
    return {key: single.robust_location(values) for key, values in grouped.items()}


def evaluate_prediction(template, test_rows):
    selected = [row for row in test_rows if ifu_key(row) in template and np.isfinite(template[ifu_key(row)])]
    x = np.asarray([template[ifu_key(row)] for row in selected]); y = np.asarray([as_float(row["r_IFU"]) for row in selected])
    fit = single.robust_zero_slope(x, y)
    before = single.robust_rms(y)
    after = single.robust_rms(y - fit["slope"] * x) if np.isfinite(fit["slope"]) else np.nan
    return {"beta": fit["slope"], "correlation": robust_correlation(x, y),
            "rms_before": before, "rms_after": after,
            "improvement": ((before - after) / before if np.isfinite(before) and before > 0 and np.isfinite(after) else np.nan),
            "N": y.size}


def track_leave_one_h5_out(rows, floor_by_exposure, output_dir):
    results = []
    for held_out in H5_NAMES:
        test_h5 = [row for row in rows if row["h5"] == held_out]
        for exposure in (1, 2, 3):
            test_rows = [row for row in test_h5 if int(row["exposure"]) == exposure and valid_common(row)]
            if not test_rows:
                continue
            track = test_rows[0]["track"]
            templates = {"universal": template_for_rows(rows, excluded_h5=held_out),
                         "matched": template_for_rows(rows, track=track, excluded_h5=held_out),
                         "opposite": template_for_rows(rows, track="W" if track == "E" else "E", excluded_h5=held_out)}
            common_keys = set(templates["universal"]) & set(templates["matched"]) & set(templates["opposite"])
            same_test = [row for row in test_rows if ifu_key(row) in common_keys]
            predictions = {name: evaluate_prediction(template, same_test)
                           for name, template in templates.items()}
            floor = floor_by_exposure.get((held_out, exposure), {})
            output = {"h5": held_out, "date": held_out[:8], "shot": held_out[9:16],
                      "exposure": exposure, "track": track,
                      "N_common_IFU": predictions["matched"]["N"],
                      "sigma_measurement_floor": floor.get("sigma_measurement", np.nan)}
            for name in ("universal", "matched", "opposite"):
                output.update({"beta_%s" % name: predictions[name]["beta"],
                               "correlation_%s" % name: predictions[name]["correlation"],
                               "RMS_before_%s" % name: predictions[name]["rms_before"],
                               "RMS_after_%s" % name: predictions[name]["rms_after"],
                               "improvement_%s" % name: predictions[name]["improvement"]})
            results.append(output)
    fields = ["h5", "date", "shot", "exposure", "track", "N_common_IFU", "sigma_measurement_floor"]
    for name in ("universal", "matched", "opposite"):
        fields.extend(["beta_%s" % name, "correlation_%s" % name,
                       "RMS_before_%s" % name, "RMS_after_%s" % name,
                       "improvement_%s" % name])
    write_csv(output_dir / "illumination_leave_one_h5_out_by_track.csv", results, fields)
    x = np.arange(len(results)); fig, axes = plt.subplots(2, 2, figsize=(15, 9))
    axes[0, 0].plot(x, [as_float(row["RMS_before_matched"]) for row in results], "k-", lw=1, label="before")
    for name, color in (("universal", "tab:blue"), ("matched", "tab:green"), ("opposite", "tab:red")):
        axes[0, 0].plot(x, [as_float(row["RMS_after_%s" % name]) for row in results], "o-", ms=3, color=color, label="after %s" % name)
    axes[0, 0].plot(x, [as_float(row["sigma_measurement_floor"]) for row in results], "--", color="0.4", label="ON/OFF floor")
    axes[0, 0].set_title("Held-out RMS"); axes[0, 0].legend(fontsize=7)
    for name, color in (("universal", "tab:blue"), ("matched", "tab:green"), ("opposite", "tab:red")):
        values = [as_float(row["improvement_%s" % name]) for row in results]
        axes[0, 1].hist([value for value in values if np.isfinite(value)], bins=20, alpha=.45, color=color, label=name)
    axes[0, 1].set_title("improvement fraction"); axes[0, 1].legend(fontsize=7)
    for name, color in (("universal", "tab:blue"), ("matched", "tab:green"), ("opposite", "tab:red")):
        values = [as_float(row["correlation_%s" % name]) for row in results]
        axes[1, 0].hist([value for value in values if np.isfinite(value)], bins=20, alpha=.45, color=color, label=name)
    axes[1, 0].set_title("held-out correlation"); axes[1, 0].legend(fontsize=7)
    axes[1, 1].plot(x, [as_float(row["beta_matched"]) for row in results], "o-", ms=3)
    axes[1, 1].axhline(0, color="k", lw=.7); axes[1, 1].set_title("matched-track beta")
    for axis in axes.flat: axis.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(output_dir / "illumination_track_template_validation.png", dpi=170); plt.close(fig)
    return results


def plot_track_beta(results, output_dir):
    fig, axis = plt.subplots(figsize=(14, 5))
    for track, color, marker in (("E", "tab:orange", "o"), ("W", "tab:blue", "s")):
        selected = [row for row in results if row["track"] == track]
        positions = [i for i, row in enumerate(results) if row["track"] == track]
        axis.plot(positions, [as_float(row["beta_matched"]) for row in selected], marker=marker, ls="-", color=color, ms=4, label=track)
        values = np.asarray([as_float(row["beta_matched"]) for row in selected]); values = values[np.isfinite(values)]
        print("matched-track beta %s: N=%d median=%.6g p16/p84=%.6g/%.6g scatter=%.6g" %
              (track, values.size, np.median(values) if values.size else np.nan,
               np.percentile(values, 16) if values.size else np.nan,
               np.percentile(values, 84) if values.size else np.nan,
               single.robust_scale(values)))
    axis.axhline(0, color="k", lw=.7); axis.set_xlabel("chronological held-out H5/exposure")
    axis.set_ylabel("matched-track beta"); axis.set_title("Matched E/W template amplitude"); axis.legend(); axis.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(output_dir / "illumination_track_template_beta.png", dpi=170); plt.close(fig)


def fit_template_by_exposure(rows, template, output_dir):
    results = []
    for key in sorted({exposure_key(row) for row in rows},
                      key=lambda key: next(row["exposure_order"] for row in rows if exposure_key(row) == key)):
        selected = [row for row in rows if exposure_key(row) == key and valid_common(row) and ifu_key(row) in template]
        x = np.asarray([template[ifu_key(row)] for row in selected]); y = np.asarray([as_float(row["r_IFU"]) for row in selected])
        fit = single.robust_zero_slope(x, y)
        after = y - fit["slope"] * x if np.isfinite(fit["slope"]) else np.full(y.shape, np.nan)
        results.append({"h5": key[0], "exposure": key[1], "beta": fit["slope"],
                        "correlation": robust_correlation(x, y),
                        "robust_RMS_before": single.robust_rms(y),
                        "robust_RMS_after": single.robust_rms(after), "N": y.size})
    fields = ["h5", "exposure", "beta", "correlation", "robust_RMS_before", "robust_RMS_after", "N"]
    write_csv(output_dir / "illumination_template_fit_by_exposure.csv", results, fields)
    x = np.arange(len(results))
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    axes[0].plot(x, [as_float(row["beta"]) for row in results], "o-", ms=3)
    axes[0].axhline(0, color="k", lw=.7); axes[0].set_ylabel("beta"); axes[0].set_title("Descriptive template amplitude")
    axes[1].plot(x, [as_float(row["robust_RMS_before"]) for row in results], "o-", ms=3, label="before")
    axes[1].plot(x, [as_float(row["robust_RMS_after"]) for row in results], "o-", ms=3, label="after beta*C_IFU")
    axes[1].set_ylabel("robust RMS"); axes[1].set_xlabel("chronological H5/exposure"); axes[1].legend()
    for axis in axes: axis.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(output_dir / "illumination_template_amplitude_by_exposure.png", dpi=170); plt.close(fig)
    return results


def exposure_correlations(rows, output_dir):
    keys = sorted({exposure_key(row) for row in rows},
                  key=lambda key: next(row["exposure_order"] for row in rows if exposure_key(row) == key))
    maps = {key: {ifu_key(row): as_float(row["r_IFU"]) for row in rows
                  if exposure_key(row) == key and valid_common(row)} for key in keys}
    matrix = np.full((len(keys), len(keys)), np.nan)
    pair_rows = []
    for i, left in enumerate(keys):
        for j, right in enumerate(keys[i:], i):
            common = sorted(set(maps[left]) & set(maps[right]))
            x = np.asarray([maps[left][key] for key in common]); y = np.asarray([maps[right][key] for key in common])
            correlation = robust_correlation(x, y)
            matrix[i, j] = matrix[j, i] = correlation
            if i != j:
                pair_rows.append({"h5_a": left[0], "exposure_a": left[1], "h5_b": right[0],
                                  "exposure_b": right[1], "correlation": correlation, "N": len(common)})
    fields = ["h5_a", "exposure_a", "h5_b", "exposure_b", "correlation", "N"]
    write_csv(output_dir / "illumination_exposure_correlations.csv", pair_rows, fields)
    finite = np.abs(matrix[np.isfinite(matrix) & ~np.eye(len(keys), dtype=bool)])
    scale = max(float(np.percentile(finite, 98)) if finite.size else 1.0, .05)
    fig, axis = plt.subplots(figsize=(12, 10))
    image = axis.imshow(matrix, cmap="coolwarm", vmin=-scale, vmax=scale, interpolation="none")
    axis.set_title("Pairwise plane-subtracted physical-IFU correlations")
    axis.set_xlabel("chronological H5/exposure"); axis.set_ylabel("chronological H5/exposure")
    transitions = [i for i in range(1, len(keys)) if keys[i][0] != keys[i - 1][0]]
    for i in transitions:
        axis.axhline(i - .5, color="k", lw=1); axis.axvline(i - .5, color="k", lw=1)
    fig.colorbar(image, ax=axis, label="Pearson correlation of r_IFU")
    fig.tight_layout(); fig.savefig(output_dir / "illumination_exposure_correlation_matrix.png", dpi=170); plt.close(fig)
    return keys


def plot_template_stability(rows, template, output_dir):
    template_rows = []
    for key, c in template.items():
        values = np.asarray([as_float(row["r_IFU"]) for row in rows if ifu_key(row) == key and valid_common(row)])
        if values.size:
            template_rows.append((key, c, single.robust_scale(values)))
    x = np.asarray([abs(c) for _, c, _ in template_rows]); y = np.asarray([scatter for _, _, scatter in template_rows])
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].scatter(x, y, s=20, alpha=.7); axes[0].set_xlabel("|C_IFU|"); axes[0].set_ylabel("robust exposure scatter")
    candidates = sorted(template_rows, key=lambda item: item[1])
    chosen = [candidates[0], candidates[-1], min(candidates, key=lambda item: abs(item[1])), max(template_rows, key=lambda item: item[2])]
    seen = set()
    for key, c, scatter in chosen:
        if key in seen: continue
        seen.add(key)
        selected = sorted([row for row in rows if ifu_key(row) == key and valid_common(row)], key=lambda row: int(row["exposure_order"]))
        axes[1].plot([int(row["exposure_order"]) for row in selected], [as_float(row["r_IFU"]) for row in selected], "o-", ms=3,
                     label="%d:%d:%d" % key)
    axes[1].set_xlabel("chronological H5/exposure"); axes[1].set_ylabel("r_IFU"); axes[1].legend(fontsize=7)
    for axis in axes: axis.grid(alpha=.2)
    fig.suptitle("Physical-IFU response stability")
    fig.tight_layout(rect=(0, 0, 1, .95)); fig.savefig(output_dir / "physical_ifu_response_stability.png", dpi=170); plt.close(fig)


def measurement_floor(rows, output_dir):
    results = []
    keys = sorted({exposure_key(row) for row in rows}, key=lambda key: next(row["exposure_order"] for row in rows if exposure_key(row) == key))
    for key in keys:
        selected = [row for row in rows if exposure_key(row) == key and
                    as_bool(row["well_constrained_ON"]) and as_bool(row["well_constrained_OFF"])]
        d = np.asarray([np.log(as_float(row["s_ON_raw"])) - np.log(as_float(row["s_OFF_raw"]))
                        for row in selected if as_float(row["s_ON_raw"]) > 0 and as_float(row["s_OFF_raw"]) > 0])
        r = np.asarray([as_float(row["r_IFU"]) for row in rows if exposure_key(row) == key and valid_common(row)])
        sigma = single.robust_scale(d)
        results.append({"h5": key[0], "exposure": key[1], "intrinsic_robust_scatter": single.robust_scale(r),
                        "sigma_ON_OFF": sigma, "sigma_measurement": sigma / np.sqrt(2.0), "N": d.size})
    fields = ["h5", "exposure", "intrinsic_robust_scatter", "sigma_ON_OFF", "sigma_measurement", "N"]
    write_csv(output_dir / "illumination_measurement_floor.csv", results, fields)
    x = np.arange(len(results)); fig, axis = plt.subplots(figsize=(14, 5))
    axis.plot(x, [as_float(row["intrinsic_robust_scatter"]) for row in results], "o-", ms=3, label="intrinsic scatter r_IFU")
    axis.plot(x, [as_float(row["sigma_measurement"]) for row in results], "o-", ms=3, label="ON/OFF floor / sqrt(2)")
    axis.plot(x, [as_float(row["sigma_ON_OFF"]) for row in results], "o-", ms=3, label="sigma ON/OFF")
    axis.set_xlabel("chronological H5/exposure"); axis.set_ylabel("robust fractional scatter"); axis.legend(); axis.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(output_dir / "illumination_dynamic_range_vs_measurement_floor.png", dpi=170); plt.close(fig)
    return {(row["h5"], int(row["exposure"])): row for row in results}


def leave_one_h5_out(rows, floor_by_exposure, output_dir):
    h5s = [name for name in H5_NAMES]
    results = []
    for held_out in h5s:
        train = [row for row in rows if row["h5"] != held_out and valid_common(row)]
        grouped = {}
        for row in train: grouped.setdefault(ifu_key(row), []).append(as_float(row["r_IFU"]))
        train_template = {key: single.robust_location(values) for key, values in grouped.items()}
        for exposure in (1, 2, 3):
            common = [r for r in rows if r["h5"] == held_out and int(r["exposure"]) == exposure and
                      valid_common(r) and ifu_key(r) in train_template]
            x = np.asarray([train_template[ifu_key(r)] for r in common]); y = np.asarray([as_float(r["r_IFU"]) for r in common])
            if not common:
                continue
            fit = single.robust_zero_slope(x, y)
            if not np.isfinite(fit["slope"]): continue
            residual = y - fit["slope"] * x
            floor = floor_by_exposure.get((held_out, exposure), {})
            results.append({"h5": held_out, "date": held_out[:8], "shot": held_out[9:16],
                            "exposure": exposure, "beta": fit["slope"],
                            "heldout_correlation": robust_correlation(x, y),
                            "robust_RMS_before": single.robust_rms(y),
                            "robust_RMS_after": single.robust_rms(residual),
                            "improvement_fraction": ((single.robust_rms(y) - single.robust_rms(residual)) /
                                                      single.robust_rms(y)
                                                      if single.robust_rms(y) > 0 else np.nan),
                            "N_common_IFU": y.size,
                            "sigma_measurement_floor": floor.get("sigma_measurement", np.nan)})
    fields = ["h5", "date", "shot", "exposure", "beta", "heldout_correlation",
              "robust_RMS_before", "robust_RMS_after", "improvement_fraction",
              "N_common_IFU", "sigma_measurement_floor"]
    write_csv(output_dir / "illumination_leave_one_h5_out.csv", results, fields)
    x = np.arange(len(results)); fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    axes[0, 0].plot(x, [as_float(r["robust_RMS_before"]) for r in results], "o-", ms=3, label="before")
    axes[0, 0].plot(x, [as_float(r["robust_RMS_after"]) for r in results], "o-", ms=3, label="after")
    axes[0, 0].plot(x, [as_float(r["sigma_measurement_floor"]) for r in results], "o-", ms=3, label="ON/OFF floor")
    axes[0, 0].set_title("Held-out RMS"); axes[0, 0].legend(fontsize=7)
    axes[0, 1].hist([as_float(r["improvement_fraction"]) for r in results if np.isfinite(as_float(r["improvement_fraction"]))], bins=20)
    axes[0, 1].set_title("improvement fraction")
    axes[1, 0].plot(x, [as_float(r["beta"]) for r in results], "o-", ms=3); axes[1, 0].set_title("held-out beta")
    axes[1, 1].plot(x, [as_float(r["heldout_correlation"]) for r in results], "o-", ms=3); axes[1, 1].set_title("held-out correlation")
    for axis in axes.flat: axis.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(output_dir / "illumination_leave_one_h5_out_summary.png", dpi=170); plt.close(fig)
    return results


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--h5-dir", required=True)
    parser.add_argument("--on-image", required=True); parser.add_argument("--off-image", required=True)
    parser.add_argument("--on-filter", required=True); parser.add_argument("--off-filter", required=True)
    parser.add_argument("--fq-template", required=True)
    parser.add_argument("--output-dir", default="hierarchical_population")
    parser.add_argument("--single-script", default=str(Path(__file__).with_name("diagnose_m101_hierarchical.py")))
    parser.add_argument("--iterations", type=int)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()
    args.output_dir = Path(args.output_dir); args.output_dir.mkdir(parents=True, exist_ok=True)
    args.single_script = Path(args.single_script)
    h5_paths = resolve_h5_paths(args.h5_dir)
    track_states = read_track_states(h5_paths)
    for path in h5_paths:
        run_single_h5(path, args, args.output_dir / path.stem)
    rows = load_population_rows(args.output_dir, h5_paths, track_states)
    write_csv(args.output_dir / "illumination_population.csv", rows, population_fields())
    make_residual_matrix(rows, args.output_dir / "illumination_population_matrix.png")
    template = make_template(rows, args.output_dir)
    fit_template_by_exposure(rows, template, args.output_dir)
    exposure_correlations(rows, args.output_dir)
    plot_template_stability(rows, template, args.output_dir)
    floor = measurement_floor(rows, args.output_dir)
    leave_one_h5_out(rows, floor, args.output_dir)
    track_summary, raw_summary = plot_track_correlation_matrix(rows, args.output_dir)
    track_templates, locations, track_template_rows = make_track_templates(rows, args.output_dir)
    plot_track_templates(rows, track_templates, locations, track_template_rows, args.output_dir)
    comparison = plot_track_template_comparison(track_template_rows, args.output_dir)
    raw_track_template_summary(rows, args.output_dir)
    track_validation = track_leave_one_h5_out(rows, floor, args.output_dir)
    plot_track_beta(track_validation, args.output_dir)
    print("track-template comparison: N=%d C_E/C_W correlation=%.6g robust scatter=%.6g" %
          (comparison["N"], comparison["correlation"], comparison["robust_scatter"]))
    print("illumination population diagnostic complete: %s" % args.output_dir)


if __name__ == "__main__":
    main()
