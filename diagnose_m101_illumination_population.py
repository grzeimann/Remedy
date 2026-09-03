#!/usr/bin/env python3
"""Run and aggregate the validated M101 illumination diagnostic population.

This driver deliberately treats ``diagnose_m101_hierarchical.py`` as a black
box.  It runs one independent diagnostic directory per accepted H5, then uses
only the resulting illumination CSVs for population-level diagnostics.  No
illumination correction is applied to spectra, H5 files, or production cubes.
"""

from argparse import ArgumentParser
import csv
import json
from pathlib import Path
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tables
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

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
MIN_COMMON_IFUS_FOR_CORRELATION = 10
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


def load_population_rows(output_dir, h5_paths):
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
                         "r_IFU": s_common - s_plane,
                         "r_ON": as_float(base["s_ON_normalized"]) - s_plane,
                         "r_OFF": as_float(base["s_OFF_normalized"]) - s_plane})
            if not as_bool(base["well_constrained_common"]):
                base["r_IFU"] = np.nan
            rows.append(base)
    rows.sort(key=lambda row: int(row["exposure_order"]))
    return rows


def population_fields():
    return ["h5", "date", "shot", "exposure", "inferred_response_state",
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


def exposure_order(rows):
    return sorted({exposure_key(row) for row in rows},
                  key=lambda key: next(row["exposure_order"] for row in rows
                                       if exposure_key(row) == key))


def exposure_maps(rows):
    return {key: {ifu_key(row): as_float(row["r_IFU"]) for row in rows
                  if exposure_key(row) == key and valid_common(row)}
            for key in exposure_order(rows)}


def robust_correlation(x, y):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if x.size < 3 or np.ptp(x) == 0.0 or np.ptp(y) == 0.0:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def cluster_response_states(rows, output_dir):
    """Cluster exposures using only the existing plane-subtracted r_IFU.

    Pairwise correlations with fewer than MIN_COMMON_IFUS_FOR_CORRELATION
    shared physical IFUs are left missing.  Missing distances are imputed
    with the median observed distance only for the linkage calculation; the
    displayed matrix remains marked as missing.
    """
    keys = exposure_order(rows)
    maps = exposure_maps(rows)
    n = len(keys)
    correlations = np.full((n, n), np.nan)
    common_counts = np.zeros((n, n), dtype=int)
    for i, left in enumerate(keys):
        correlations[i, i] = 1.0
        for j in range(i + 1, n):
            right = keys[j]
            common = sorted(set(maps[left]) & set(maps[right]))
            common_counts[i, j] = common_counts[j, i] = len(common)
            if len(common) >= MIN_COMMON_IFUS_FOR_CORRELATION:
                x = np.asarray([maps[left][key] for key in common])
                y = np.asarray([maps[right][key] for key in common])
                correlations[i, j] = correlations[j, i] = robust_correlation(x, y)

    distances = 1.0 - np.clip(correlations, -1.0, 1.0)
    finite_off_diagonal = np.isfinite(distances) & ~np.eye(n, dtype=bool)
    if n < 2 or not finite_off_diagonal.any():
        raise RuntimeError("cannot cluster response states: no usable exposure pairs")
    fill_distance = float(np.median(distances[finite_off_diagonal]))
    distances[~np.isfinite(distances)] = fill_distance
    np.fill_diagonal(distances, 0.0)
    linkage_result = linkage(squareform(distances, checks=False), method="average")
    raw_labels = fcluster(linkage_result, 2, criterion="maxclust")
    if len(set(raw_labels)) != 2:
        raise RuntimeError("two-cluster diagnostic returned %d clusters" % len(set(raw_labels)))

    # Cluster labels have no intrinsic ordering.  Chronological order is used
    # only to name the earlier cluster state1 and the later cluster state2.
    cluster_order = sorted(set(raw_labels),
                           key=lambda label: np.median([i for i, value in enumerate(raw_labels)
                                                        if value == label]))
    label_map = {cluster_order[0]: 1, cluster_order[1]: 2}
    state_by_exposure = {key: label_map[label] for key, label in zip(keys, raw_labels)}
    print("response-state clustering: %d usable pairs, %d pairs below N=%d (distance fill=%g)" %
          (int(np.sum(finite_off_diagonal) // 2),
           int(np.sum(~finite_off_diagonal & ~np.eye(n, dtype=bool)) // 2),
           MIN_COMMON_IFUS_FOR_CORRELATION, fill_distance))
    correlation_summary = []
    for pair_type in ("state1-state1", "state2-state2", "state1-state2"):
        values = []
        for i in range(n):
            for j in range(i + 1, n):
                type_for_pair = ("state%d-state%d" %
                                 (state_by_exposure[keys[i]], state_by_exposure[keys[j]]))
                if type_for_pair in {"state2-state1", "state1-state2"}:
                    type_for_pair = "state1-state2"
                if type_for_pair == pair_type and np.isfinite(correlations[i, j]):
                    values.append(correlations[i, j])
        values = np.asarray(values)
        summary = {"pair_type": pair_type, "N_pairs": values.size,
                   "median": np.median(values) if values.size else np.nan,
                   "p16": np.percentile(values, 16) if values.size else np.nan,
                   "p84": np.percentile(values, 84) if values.size else np.nan}
        correlation_summary.append(summary)
        print("%s correlations: N=%d median=%g p16/p84=%g/%g" %
              (pair_type, summary["N_pairs"], summary["median"], summary["p16"], summary["p84"]))
    write_csv(output_dir / "illumination_response_state_correlation_summary.csv",
              correlation_summary, ["pair_type", "N_pairs", "median", "p16", "p84"])
    for state in (1, 2):
        print("response state %d chronological members:" % state)
        for key in keys:
            if state_by_exposure[key] == state:
                print("  %s exposure=%d" % key)

    order = sorted(range(n), key=lambda i: (state_by_exposure[keys[i]], i))
    reordered = correlations[np.ix_(order, order)]
    labels = ["%s\\ne%d" % (keys[i][0][:8], keys[i][1]) for i in order]
    fig, axis = plt.subplots(figsize=(12, 10))
    image = axis.imshow(reordered, cmap="coolwarm", vmin=-1, vmax=1,
                        interpolation="none")
    axis.set_title("Response-state clustering from physical-IFU correlations")
    axis.set_xlabel("exposures reordered by inferred state")
    axis.set_ylabel("exposures reordered by inferred state")
    axis.set_xticks(np.arange(n)); axis.set_xticklabels(labels, rotation=90, fontsize=5)
    axis.set_yticks(np.arange(n)); axis.set_yticklabels(labels, fontsize=5)
    boundary = sum(state_by_exposure[key] == 1 for key in keys)
    axis.axhline(boundary - .5, color="k", lw=2)
    axis.axvline(boundary - .5, color="k", lw=2)
    fig.colorbar(image, ax=axis, label="Pearson correlation of r_IFU")
    fig.tight_layout(); fig.savefig(output_dir / "illumination_response_state_clustering.png",
                                    dpi=170); plt.close(fig)
    return state_by_exposure, correlations, common_counts


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


def template_from_rows(rows, state=None, excluded_h5=None):
    grouped = {}
    for row in rows:
        if not valid_common(row) or (state is not None and int(row["inferred_response_state"]) != state):
            continue
        if excluded_h5 is not None and row["h5"] == excluded_h5:
            continue
        grouped.setdefault(ifu_key(row), []).append(as_float(row["r_IFU"]))
    return {key: single.robust_location(values) for key, values in grouped.items()
            if np.isfinite(single.robust_location(values))}


def state_templates(rows, output_dir):
    grouped = {(state, ifu_key(row)): [] for state in (1, 2) for row in rows
               if valid_common(row) and int(row["inferred_response_state"]) == state}
    locations = {}
    for row in rows:
        if valid_common(row):
            locations.setdefault(ifu_key(row), []).append(
                (as_float(row["mean_RA"]), as_float(row["mean_Dec"])))
    for row in rows:
        if valid_common(row):
            grouped.setdefault((int(row["inferred_response_state"]), ifu_key(row)), []).append(
                as_float(row["r_IFU"]))
    identities = sorted({key for _, key in grouped}, key=lambda key: (key[1], key[0], key[2]))
    template_rows = []
    templates = {1: {}, 2: {}}
    for key in identities:
        output = {"SPECID": key[0], "IFUSLOT": key[1], "IFUID": key[2]}
        for state in (1, 2):
            values = np.asarray(grouped.get((state, key), []), dtype=float)
            values = values[np.isfinite(values)]
            c = single.robust_location(values)
            templates[state][key] = c
            output.update({"C_state%d" % state: c,
                           "scatter_state%d" % state: single.robust_scale(values),
                           "N_state%d" % state: values.size})
        output["C_state2_minus_state1"] = (output["C_state2"] - output["C_state1"]
                                            if np.isfinite(as_float(output["C_state1"])) and
                                            np.isfinite(as_float(output["C_state2"])) else np.nan)
        template_rows.append(output)
    fields = ["SPECID", "IFUSLOT", "IFUID", "C_state1", "scatter_state1", "N_state1",
              "C_state2", "scatter_state2", "N_state2", "C_state2_minus_state1"]
    write_csv(output_dir / "physical_ifu_response_template_by_state.csv", template_rows, fields)

    ra = np.asarray([single.robust_location([value[0] for value in locations[key]])
                     for key in identities])
    dec = np.asarray([single.robust_location([value[1] for value in locations[key]])
                      for key in identities])
    c1 = np.asarray([templates[1].get(key, np.nan) for key in identities])
    c2 = np.asarray([templates[2].get(key, np.nan) for key in identities])
    difference = c2 - c1
    finite_c = np.concatenate((c1[np.isfinite(c1)], c2[np.isfinite(c2)]))
    cscale = max(float(np.percentile(np.abs(finite_c), 95)) if finite_c.size else .01, .01)
    finite_both = np.isfinite(c1) & np.isfinite(c2)
    correlation = robust_correlation(c1[finite_both], c2[finite_both])
    line = single.fit_line(c1[finite_both], c2[finite_both])
    relation_scatter = single.robust_scale(c2[finite_both] -
                                           (line["g"] * c1[finite_both] + line["z"])
                                           if np.isfinite(line["g"]) else [])
    print("state template comparison: N=%d, correlation=%g, C2=%g*C1+%g, robust scatter=%g" %
          (int(np.sum(finite_both)), correlation, line["g"], line["z"], relation_scatter))
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for axis, values, title in ((axes[0, 0], c1, "C_state1"),
                                (axes[0, 1], c2, "C_state2"),
                                (axes[1, 0], difference, "C_state2 - C_state1")):
        axis.scatter(ra, dec, c=values, cmap="coolwarm", vmin=-cscale, vmax=cscale, s=35)
        axis.set_title(title); axis.set_xlabel("RA"); axis.set_ylabel("Dec"); axis.grid(alpha=.2)
    axis = axes[1, 1]
    axis.scatter(c1[finite_both], c2[finite_both], s=22, alpha=.75)
    limits = np.asarray([np.nanmin(np.r_[c1[finite_both], c2[finite_both]]),
                         np.nanmax(np.r_[c1[finite_both], c2[finite_both]])]) if finite_both.any() else np.array([-.01, .01])
    axis.plot(limits, limits, "k--", label="identity")
    if np.isfinite(line["g"]):
        axis.plot(limits, line["g"] * limits + line["z"], label="robust fit")
    axis.set_title("state template comparison (r=%.3g)" % correlation)
    axis.set_xlabel("C_state1"); axis.set_ylabel("C_state2"); axis.legend(fontsize=8); axis.grid(alpha=.2)
    fig.suptitle("Unsmoothed physical-IFU response states")
    fig.tight_layout(rect=(0, 0, 1, .95)); fig.savefig(output_dir / "physical_ifu_response_state_comparison.png",
                                                       dpi=170); plt.close(fig)
    return templates


def evaluate_template_prediction(test_rows, template):
    selected = [row for row in test_rows if ifu_key(row) in template and
                np.isfinite(as_float(template[ifu_key(row)]))]
    if len(selected) < 3:
        return {"beta": np.nan, "correlation": np.nan, "robust_RMS_before": np.nan,
                "robust_RMS_after": np.nan, "improvement_fraction": np.nan, "N": len(selected)}
    x = np.asarray([template[ifu_key(row)] for row in selected])
    y = np.asarray([as_float(row["r_IFU"]) for row in selected])
    fit = single.robust_zero_slope(x, y)
    before = single.robust_rms(y)
    after = single.robust_rms(y - fit["slope"] * x) if np.isfinite(fit["slope"]) else np.nan
    return {"beta": fit["slope"], "correlation": robust_correlation(x, y),
            "robust_RMS_before": before, "robust_RMS_after": after,
            "improvement_fraction": ((before - after) / before if np.isfinite(before) and before > 0 else np.nan),
            "N": len(selected)}


def plot_state_templates(rows, templates, output_dir):
    identities = sorted(set(templates[1]) | set(templates[2]), key=lambda key: (key[1], key[0], key[2]))
    locations = {}
    for key in identities:
        positions = [row for row in rows if ifu_key(row) == key and valid_common(row)]
        locations[key] = (single.robust_location([as_float(row["mean_RA"]) for row in positions]),
                          single.robust_location([as_float(row["mean_Dec"]) for row in positions]))
    ra = np.asarray([locations[key][0] for key in identities]); dec = np.asarray([locations[key][1] for key in identities])
    values = [np.asarray([templates[state].get(key, np.nan) for key in identities]) for state in (1, 2)]
    values.append(values[1] - values[0])
    finite = np.concatenate([v[np.isfinite(v)] for v in values[:2]])
    scale = max(float(np.percentile(np.abs(finite), 95)) if finite.size else .01, .01)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for axis, value, title in zip(axes, values, ("C_state1", "C_state2", "C_state2 - C_state1")):
        axis.scatter(ra, dec, c=value, cmap="coolwarm", vmin=-scale, vmax=scale, s=35)
        axis.set_title(title); axis.set_xlabel("RA"); axis.set_ylabel("Dec"); axis.grid(alpha=.2)
    fig.suptitle("Response-state physical-IFU templates")
    fig.tight_layout(rect=(0, 0, 1, .95)); fig.savefig(output_dir / "physical_ifu_response_templates_by_state.png",
                                                       dpi=170); plt.close(fig)


def leave_one_h5_out_by_state(rows, floor_by_exposure, output_dir):
    results = []
    for held_out in H5_NAMES:
        held_rows = [row for row in rows if row["h5"] == held_out]
        for exposure in (1, 2, 3):
            test = [row for row in held_rows if int(row["exposure"]) == exposure and valid_common(row)]
            if not test:
                continue
            state = int(test[0]["inferred_response_state"])
            universal = template_from_rows(rows, excluded_h5=held_out)
            matching = template_from_rows(rows, state=state, excluded_h5=held_out)
            opposite = template_from_rows(rows, state=3 - state, excluded_h5=held_out)
            available = set(universal) & set(matching) & set(opposite) & {ifu_key(row) for row in test}
            test = [row for row in test if ifu_key(row) in available]
            if len(test) < 3:
                continue
            no_template = {"beta": np.nan, "correlation": np.nan,
                           "robust_RMS_before": single.robust_rms([as_float(row["r_IFU"]) for row in test]),
                           "robust_RMS_after": np.nan, "improvement_fraction": np.nan,
                           "N": len(test)}
            no_template["robust_RMS_after"] = no_template["robust_RMS_before"]
            no_template["improvement_fraction"] = 0.0
            evaluations = {"universal": evaluate_template_prediction(test, universal),
                           "matching_state": evaluate_template_prediction(test, matching),
                           "opposite_state": evaluate_template_prediction(test, opposite)}
            output = {"h5": held_out, "date": held_out[:8], "shot": held_out[9:16],
                      "exposure": exposure, "inferred_response_state": state,
                      "sigma_measurement_floor": floor_by_exposure.get((held_out, exposure), {}).get("sigma_measurement", np.nan)}
            output.update({"N_common_IFU": len(test)})
            for name, values in (("no_template", no_template),) + tuple(evaluations.items()):
                for field in ("beta", "correlation", "robust_RMS_before", "robust_RMS_after", "improvement_fraction"):
                    output["%s_%s" % (name, field)] = values[field]
                output["%s_N" % name] = values["N"]
            results.append(output)
    fields = ["h5", "date", "shot", "exposure", "inferred_response_state", "N_common_IFU",
              "sigma_measurement_floor"]
    for name in ("no_template", "universal", "matching_state", "opposite_state"):
        fields.extend(["%s_%s" % (name, field) for field in
                       ("beta", "correlation", "robust_RMS_before", "robust_RMS_after", "improvement_fraction", "N")])
    write_csv(output_dir / "illumination_leave_one_h5_out_by_state.csv", results, fields)
    x = np.arange(len(results)); fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True)
    for name, label, style in (("no_template", "before", "k--"),
                               ("universal", "universal", "o-"),
                               ("matching_state", "matching state", "o-"),
                               ("opposite_state", "opposite state", "o-")):
        field = "%s_robust_RMS_after" % name
        values = [as_float(row[field]) for row in results]
        if name == "no_template":
            values = [as_float(row["no_template_robust_RMS_before"]) for row in results]
        axes[0, 0].plot(x, values, style, ms=3, label=label)
    axes[0, 0].plot(x, [as_float(row["sigma_measurement_floor"]) for row in results], "-", lw=1, label="ON/OFF floor")
    axes[0, 0].set_title("held-out RMS"); axes[0, 0].legend(fontsize=7)
    for name, label in (("universal", "universal"), ("matching_state", "matching state"), ("opposite_state", "opposite state")):
        axes[0, 1].hist([as_float(row["%s_improvement_fraction" % name]) for row in results
                         if np.isfinite(as_float(row["%s_improvement_fraction" % name]))], bins=15, alpha=.5, label=label)
        axes[1, 0].hist([as_float(row["%s_correlation" % name]) for row in results
                         if np.isfinite(as_float(row["%s_correlation" % name]))], bins=15, alpha=.5, label=label)
    axes[0, 1].set_title("improvement fraction"); axes[1, 0].set_title("held-out correlation")
    axes[0, 1].legend(fontsize=7); axes[1, 0].legend(fontsize=7)
    axes[1, 1].plot(x, [as_float(row["matching_state_beta"]) for row in results], "o-", ms=3)
    axes[1, 1].set_title("matching-state beta"); axes[1, 1].set_xlabel("chronological held-out exposure")
    for axis in axes.flat: axis.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(output_dir / "illumination_response_state_validation.png", dpi=170); plt.close(fig)
    return results


def scalar_metadata_value(value):
    array = np.asarray(value)
    if array.size != 1:
        return None
    value = array.reshape(-1)[0]
    if isinstance(value, (bytes, np.bytes_)):
        return single.as_text(value)
    if isinstance(value, (str, np.str_)):
        return str(value)
    if np.isscalar(value):
        try:
            number = float(value)
        except (TypeError, ValueError):
            return str(value)
        return number if np.isfinite(number) else np.nan
    return None


def metadata_column_is_scalar(table, field):
    column = table.colinstances[field]
    shape = getattr(column, "shape", ())
    # Column.shape includes the table-row dimension.  Any remaining
    # dimensions describe an array-valued cell and are intentionally skipped.
    return len(shape) <= 1 or int(np.prod(shape[1:])) == 1


def read_exposure_metadata(h5_paths, state_by_exposure):
    """Collect scalar Survey fields and compact numeric Info summaries only."""
    output = []
    for h5_path in h5_paths:
        date, shot = h5_path.stem.split("_")
        with tables.open_file(h5_path, mode="r") as h5:
            survey = h5.root.Survey
            survey_fields = [field for field in survey.colnames if field != "exp" and
                             metadata_column_is_scalar(survey, field)]
            survey_rows = {}
            for survey_row in survey:
                exposure = int(survey_row["exp"])
                if exposure in survey_rows:
                    raise ValueError("%s Survey has duplicate exposure %d" % (h5_path.name, exposure))
                survey_rows[exposure] = {field: scalar_metadata_value(survey_row[field])
                                         for field in survey_fields}
            if set(survey_rows) != {1, 2, 3}:
                raise ValueError("%s Survey must contain exactly exposures 1, 2, 3" % h5_path.name)

            info_summaries = {}
            if hasattr(h5.root, "Info"):
                info = h5.root.Info
                info_exp_field = "exp" if "exp" in info.colnames else None
                info_identity_fields = {"specid", "ifuslot", "ifuid", "amp"}
                for field in info.colnames:
                    if field == info_exp_field or field in info_identity_fields or not metadata_column_is_scalar(info, field):
                        continue
                    column = info.colinstances[field]
                    if np.dtype(column.dtype).kind not in "iufb":
                        continue
                    try:
                        values = np.asarray(info.cols[field][:], dtype=float)
                    except (TypeError, ValueError):
                        continue
                    if values.ndim != 1:
                        continue
                    if info_exp_field is not None:
                        info_exposures = np.asarray(info.cols[info_exp_field][:], dtype=int)
                        has_requested_exposures = np.isin(info_exposures, (1, 2, 3)).any()
                    else:
                        info_exposures = np.full(values.size, -1, dtype=int)
                        has_requested_exposures = False
                    for exposure in (1, 2, 3):
                        selected = (values[info_exposures == exposure]
                                    if has_requested_exposures else values)
                        selected = selected[np.isfinite(selected)]
                        if selected.size:
                            info_summaries.setdefault(exposure, {}).update({
                                "Info_%s_median" % field: np.median(selected),
                                "Info_%s_min" % field: np.min(selected),
                                "Info_%s_max" % field: np.max(selected)})
            for exposure in (1, 2, 3):
                row = {"h5": h5_path.name, "date": date, "shot": shot,
                       "exposure": exposure,
                       "inferred_response_state": state_by_exposure[(h5_path.name, exposure)]}
                row.update({"Survey_%s" % field: value for field, value in survey_rows.get(exposure, {}).items()})
                if "Survey_name" in row:
                    row["survey_name"] = row["Survey_name"]
                row.update(info_summaries.get(exposure, {}))
                output.append(row)
    return output


def metadata_summary(metadata_rows, output_dir):
    base_fields = {"h5", "date", "shot", "exposure", "inferred_response_state", "survey_name"}
    fields = sorted({field for row in metadata_rows for field in row if field not in base_fields})
    summary = []
    for field in fields:
        numeric_values = {state: np.asarray([as_float(row.get(field)) for row in metadata_rows
                                             if int(row["inferred_response_state"]) == state], dtype=float)
                          for state in (1, 2)}
        finite = {state: values[np.isfinite(values)] for state, values in numeric_values.items()}
        if all(values.size >= 3 for values in finite.values()):
            medians = {state: float(np.median(finite[state])) for state in (1, 2)}
            scatters = {state: single.robust_scale(finite[state]) for state in (1, 2)}
            denominator = np.sqrt(.5 * (scatters[1] ** 2 + scatters[2] ** 2))
            separation = abs(medians[1] - medians[2]) / denominator if denominator > 0 else (
                np.inf if medians[1] != medians[2] else 0.0)
            summary.append({"kind": "numeric", "field": field,
                            "median_state1": medians[1], "median_state2": medians[2],
                            "robust_scatter_state1": scatters[1], "robust_scatter_state2": scatters[2],
                            "standardized_separation": separation,
                            "N_state1": finite[1].size, "N_state2": finite[2].size,
                            "value_counts_state1": "", "value_counts_state2": ""})
        else:
            values_by_state = {}
            for state in (1, 2):
                values = [str(row.get(field, "")).strip() for row in metadata_rows
                          if int(row["inferred_response_state"]) == state and
                          str(row.get(field, "")).strip() not in {"", "nan", "None"}]
                counts = {}
                for value in values: counts[value] = counts.get(value, 0) + 1
                values_by_state[state] = counts
            if not any(values_by_state.values()):
                continue
            summary.append({"kind": "categorical", "field": field,
                            "median_state1": "", "median_state2": "",
                            "robust_scatter_state1": "", "robust_scatter_state2": "",
                            "standardized_separation": "",
                            "N_state1": sum(values_by_state[1].values()),
                            "N_state2": sum(values_by_state[2].values()),
                            "value_counts_state1": json.dumps(values_by_state[1], sort_keys=True),
                            "value_counts_state2": json.dumps(values_by_state[2], sort_keys=True)})
    summary.sort(key=lambda row: (row["kind"] != "numeric",
                                  -as_float(row["standardized_separation"])
                                  if row["kind"] == "numeric" else row["field"]))
    fields = ["kind", "field", "median_state1", "median_state2", "robust_scatter_state1",
              "robust_scatter_state2", "standardized_separation", "N_state1", "N_state2",
              "value_counts_state1", "value_counts_state2"]
    write_csv(output_dir / "illumination_response_state_metadata_summary.csv", summary, fields)
    print("metadata state-separation candidates:")
    for row in summary[:15]:
        print("  %s: separation=%s state1=%s state2=%s counts=(%s, %s)" %
              (row["field"], row["standardized_separation"], row["median_state1"], row["median_state2"],
               row["value_counts_state1"], row["value_counts_state2"]))
    return summary


def state_exposure_fits(rows, templates, state_by_exposure, output_dir):
    results = []
    for key in exposure_order(rows):
        selected = [row for row in rows if exposure_key(row) == key and valid_common(row)]
        if not selected:
            continue
        state = state_by_exposure[key]
        fits = {"state1": evaluate_template_prediction(selected, templates[1]),
                "state2": evaluate_template_prediction(selected, templates[2])}
        matching = fits["state%d" % state]
        results.append({"h5": key[0], "date": key[0][:8], "shot": key[0][9:16],
                        "exposure": key[1], "inferred_response_state": state,
                        "matching_beta": matching["beta"],
                        "correlation_state1": fits["state1"]["correlation"],
                        "correlation_state2": fits["state2"]["correlation"],
                        "matching_RMS_before": matching["robust_RMS_before"],
                        "matching_RMS_after": matching["robust_RMS_after"], "N": matching["N"]})
    fields = ["h5", "date", "shot", "exposure", "inferred_response_state", "matching_beta",
              "correlation_state1", "correlation_state2", "matching_RMS_before",
              "matching_RMS_after", "N"]
    write_csv(output_dir / "illumination_response_state_exposure_fits.csv", results, fields)
    x = np.arange(len(results)); states = np.asarray([int(row["inferred_response_state"]) for row in results])
    fig, axes = plt.subplots(4, 1, figsize=(15, 12), sharex=True)
    axes[0].scatter(x, states, c=states, cmap="Set1", vmin=1, vmax=2, s=28)
    axes[0].set_ylabel("state"); axes[0].set_yticks((1, 2))
    axes[1].plot(x, [as_float(row["matching_beta"]) for row in results], "o-", ms=3)
    axes[1].axhline(1, color="k", lw=.7); axes[1].set_ylabel("matching beta")
    axes[2].plot(x, [as_float(row["correlation_state1"]) for row in results], "o-", ms=3, label="state1")
    axes[2].plot(x, [as_float(row["correlation_state2"]) for row in results], "o-", ms=3, label="state2")
    axes[2].set_ylabel("correlation"); axes[2].legend(fontsize=8)
    axes[3].plot(x, [as_float(row["matching_RMS_before"]) for row in results], "o-", ms=3, label="before")
    axes[3].plot(x, [as_float(row["matching_RMS_after"]) for row in results], "o-", ms=3, label="after")
    axes[3].set_ylabel("robust RMS"); axes[3].set_xlabel("chronological exposure"); axes[3].legend(fontsize=8)
    boundaries = [i for i in range(1, len(results)) if results[i]["h5"] != results[i - 1]["h5"]]
    for axis in axes:
        for boundary in boundaries: axis.axvline(boundary - .5, color="k", lw=.8)
        axis.grid(alpha=.2)
    fig.suptitle("Inferred response state timeline")
    fig.tight_layout(rect=(0, 0, 1, .95)); fig.savefig(output_dir / "illumination_response_state_timeline.png",
                                                       dpi=170); plt.close(fig)
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
    for path in h5_paths:
        run_single_h5(path, args, args.output_dir / path.stem)
    rows = load_population_rows(args.output_dir, h5_paths)
    state_by_exposure, _, _ = cluster_response_states(rows, args.output_dir)
    for row in rows:
        row["inferred_response_state"] = state_by_exposure[exposure_key(row)]
    write_csv(args.output_dir / "illumination_population.csv", rows, population_fields())
    make_residual_matrix(rows, args.output_dir / "illumination_population_matrix.png")
    template = make_template(rows, args.output_dir)
    fit_template_by_exposure(rows, template, args.output_dir)
    exposure_correlations(rows, args.output_dir)
    plot_template_stability(rows, template, args.output_dir)
    floor = measurement_floor(rows, args.output_dir)
    leave_one_h5_out(rows, floor, args.output_dir)
    state_template_values = state_templates(rows, args.output_dir)
    plot_state_templates(rows, state_template_values, args.output_dir)
    leave_one_h5_out_by_state(rows, floor, args.output_dir)
    metadata_rows = read_exposure_metadata(h5_paths, state_by_exposure)
    metadata_base_fields = ["h5", "date", "shot", "exposure", "inferred_response_state",
                            "survey_name"]
    metadata_fields = metadata_base_fields + sorted(
        {field for row in metadata_rows for field in row if field not in metadata_base_fields})
    write_csv(args.output_dir / "illumination_response_state_metadata.csv", metadata_rows, metadata_fields)
    metadata_summary(metadata_rows, args.output_dir)
    state_exposure_fits(rows, state_template_values, state_by_exposure, args.output_dir)
    print("illumination population diagnostic complete: %s" % args.output_dir)


if __name__ == "__main__":
    main()
