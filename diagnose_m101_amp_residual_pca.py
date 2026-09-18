#!/usr/bin/env python3
"""PCA diagnostic for full-resolution M101 amplifier residual spectra.

This script deliberately inherits the spectral calibration semantics from
``diagnose_m101_frozenM_spectral_residuals.py`` rather than rebuilding them.
The parent diagnostic is the authority for:

    M(h,i,a) = exp(R_total_h_i_a)
    S_e(lambda) = robust location of M-corrected classified blank q=40..75 fibers
    E_f(lambda) = D_f(lambda) - M_f * S_e(lambda)

The only new operation here is to collapse E_f(lambda) to one robust spectrum
per physical amplifier/exposure and study exposure-to-exposure departures from
that amplifier's robust mean with a 3-component PCA.

Important: alpha*f(q)*K(lambda) is NOT applied here.  The established alpha
fit is centered on the central-q robust level via f(q)-f0, so the central-q
collapsed residual spectrum studied here is intentionally orthogonal to that
q-dependent additive term.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Single authoritative parent for the full-resolution preprocessing.
import diagnose_m101_frozenM_spectral_residuals as parent


N_COMPONENTS = 3
DEFAULT_MIN_BLANK = 3
DEFAULT_MIN_EXPOSURES = 5
DEFAULT_GALLERY_SIZE = 10
MAD_TO_SIGMA = 1.4826
MAD_FLOOR_FRACTION = 0.05
MIN_WAVE_SUPPORT_FRACTION = 0.80


def _json_ready(value):
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _write_rows(path, rows):
    rows = list(rows)
    path = Path(path)
    if not rows:
        path.write_text("")
        return
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})


def _robust_scatter_flat(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size < 2:
        return np.nan
    return parent._safe_scatter(values)


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", default="m101_onoff_hierarchy/m101_onoff_state.json")
    parser.add_argument("--band-cache", default="m101_band_initializer/m101_band_cache.npz")
    parser.add_argument("--h5", nargs="*",
                        help="optional H5 paths/basenames; otherwise use band-cache provenance")
    parser.add_argument("--blank-file", default=None)
    parser.add_argument("--on-filter", default=None)
    parser.add_argument("--off-filter", default=None)
    parser.add_argument("--output-dir", default="m101_amp_residual_pca")
    parser.add_argument("--min-blank", type=int, default=DEFAULT_MIN_BLANK,
                        help="minimum central-q blank fibers for an amp/exposure collapse")
    parser.add_argument("--min-exposures", type=int, default=DEFAULT_MIN_EXPOSURES,
                        help="minimum usable exposure spectra required to fit 3 PCs")
    parser.add_argument("--gallery-size", type=int, default=DEFAULT_GALLERY_SIZE)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _resolve_inputs(args):
    """Resolve exactly the same provenance inputs as the parent diagnostic."""
    state_path = Path(args.state).expanduser().resolve()
    state, response = parent._load_state(state_path)
    composition = parent._verify_response_composition(state)
    if not composition.get("composition_matches_stored_cumulative", False):
        raise RuntimeError(
            "pass-1 plus refinement response does not reproduce cumulative R_total")

    cache_path = Path(args.band_cache).expanduser().resolve()
    manifest, manifest_path = parent._load_band_manifest(cache_path)
    h5_paths = parent._resolve_h5_paths(args, manifest)
    identity_map = parent._load_cache_identity_map(cache_path, manifest, h5_paths)

    blank_file = args.blank_file or parent._default_input("blank_file")
    on_filter = args.on_filter or parent._default_input("on_filter")
    off_filter = args.off_filter or parent._default_input("off_filter")
    for label, value in (("blank file", blank_file),
                         ("ON filter", on_filter),
                         ("OFF filter", off_filter)):
        if not value or not Path(value).exists():
            raise FileNotFoundError("%s is unavailable; pass it explicitly" % label)

    return {
        "state_path": state_path,
        "state": state,
        "response": response,
        "composition": composition,
        "cache_path": cache_path,
        "manifest": manifest,
        "manifest_path": manifest_path,
        "h5_paths": h5_paths,
        "identity_map": identity_map,
        "blank_file": str(Path(blank_file).expanduser().resolve()),
        "on_filter": str(Path(on_filter).expanduser().resolve()),
        "off_filter": str(Path(off_filter).expanduser().resolve()),
    }


def _print_provenance(inputs):
    print("[pca] preprocessing provenance", flush=True)
    print("  ON/OFF state : %s" % inputs["state_path"], flush=True)
    print("  response     : hierarchy.cumulative_final.R_total_h_i_a", flush=True)
    print("  M            : exp(R_total_h_i_a), wavelength independent", flush=True)
    print("  band cache   : %s" % inputs["cache_path"], flush=True)
    print("  manifest     : %s" % inputs["manifest_path"], flush=True)
    print("  blank file   : %s" % inputs["blank_file"], flush=True)
    print("  ON filter    : %s" % inputs["on_filter"], flush=True)
    print("  OFF filter   : %s" % inputs["off_filter"], flush=True)
    print("  H5 files     : %d (%d exposure instances expected if all have 3 exposures)" %
          (len(inputs["h5_paths"]), 3 * len(inputs["h5_paths"])), flush=True)
    print("  q range      : %d..%d inclusive" % (parent.Q_MIN, parent.Q_MAX), flush=True)
    print("  wavelength   : %.1f..%.1f A on native grid" %
          (parent.WAVE_MIN, parent.WAVE_MAX), flush=True)
    print("  alpha term   : NOT APPLIED (intentional central-q gauge separation)", flush=True)
    print("  interpolation: none", flush=True)


def _validate_parent_semantics():
    if (parent.Q_MIN, parent.Q_MAX) != (40, 75):
        raise RuntimeError("parent q-range changed; expected inclusive 40..75")
    if not np.array_equal(parent.SCIENCE_MASK,
                          (parent.WAVE >= parent.WAVE_MIN) &
                          (parent.WAVE <= parent.WAVE_MAX)):
        raise RuntimeError("parent science wavelength mask semantics changed")


def _collapse_amp_exposure(item, calculation, physical, amp, codes, min_blank):
    """Return one robust E(lambda) spectrum for one physical amp/exposure."""
    physical = tuple(int(value) for value in physical)
    physical_mask = np.all(
        np.asarray(item.ifu, dtype=int) == np.asarray(physical, dtype=int), axis=1)
    amp_mask = np.asarray(item.amp, dtype=object) == str(amp)
    selected = physical_mask & amp_mask & np.asarray(calculation["base"], dtype=bool)
    n_blank = int(np.sum(selected))
    if n_blank < min_blank:
        return None

    code_values = np.unique(np.asarray(codes, dtype=int)[physical_mask])
    code_values = code_values[code_values >= 0]
    if code_values.size != 1:
        raise ValueError(
            "physical IFU does not map to exactly one usable IFU_CODE: %s %s" %
            (item.h5_name, physical))

    residual = np.asarray(calculation["residual"], dtype=float)[selected]
    spectrum = parent._safe_location(residual, axis=0)
    spectrum = np.asarray(spectrum, dtype=float)
    spectrum[~parent.SCIENCE_MASK] = np.nan

    if not np.any(np.isfinite(spectrum[parent.SCIENCE_MASK])):
        return None

    return {
        "H5": str(item.h5_name),
        "exposure": int(item.exposure),
        "SPECID": physical[0],
        "IFUSLOT": physical[1],
        "IFUID": physical[2],
        "IFU_CODE": int(code_values[0]),
        "AMP": str(amp),
        "N_blank_central": n_blank,
        # float32 is ample here and keeps the compact collection genuinely compact.
        "residual": spectrum.astype(np.float32),
    }


def _collect_amp_exposure_spectra(inputs, min_blank):
    """One native-data pass; retain only one collapsed spectrum per amp/exposure."""
    blank_masks, blank_provenance = parent.m101_blank_fibers.load(
        inputs["blank_file"], inputs["h5_paths"])

    records = []
    loader_provenance = []
    exposure_instances = 0
    for h5_index, h5_path in enumerate(inputs["h5_paths"], 1):
        print("[pca] native pass %d/%d %s" %
              (h5_index, len(inputs["h5_paths"]), h5_path.name), flush=True)
        native_items, provenance = parent.native.load(
            [h5_path], blank_masks, inputs["on_filter"], inputs["off_filter"],
            include_native_errors=False,
            include_band_errors=False,
            collapse_band_indices=(),
            progress=None)
        loader_provenance.append(provenance)
        exposure_instances += len(native_items)

        for item in sorted(native_items, key=lambda value: value.exposure):
            codes = []
            for physical in np.asarray(item.ifu, dtype=int):
                key = (item.h5_name, parent._physical_key(physical))
                codes.append(inputs["identity_map"].get(key, -1))

            # This function is the authoritative frozen-M + full-resolution sky
            # calculation from diagnose_m101_frozenM_spectral_residuals.py.
            calculation = parent._sky_and_residuals(
                item, codes, inputs["response"], parent.Q_MAX)

            for physical, amp in parent._physical_groups(item, calculation):
                record = _collapse_amp_exposure(
                    item, calculation, physical, amp, codes, min_blank)
                if record is not None:
                    records.append(record)

        del native_items

    if not records:
        raise RuntimeError("no usable amplifier/exposure residual spectra were collected")
    return records, blank_provenance, loader_provenance, exposure_instances


def _amp_key(record):
    return (int(record["SPECID"]), int(record["IFUSLOT"]),
            int(record["IFUID"]), str(record["AMP"]))


def _amp_label(key):
    return "%d/%d/%d/%s" % key


def _mad_scale(departure):
    """Wavelength-by-wavelength MAD sigma around the already robust-centered data."""
    departure = np.asarray(departure, dtype=float)
    # The compact spectra intentionally carry NaN outside the science range.
    # Avoid calling nanmedian on those entirely-NaN columns: the result is
    # already unused there, but NumPy emits an avoidable RuntimeWarning.
    mad = np.full(departure.shape[1], np.nan, dtype=float)
    supported = np.any(np.isfinite(departure), axis=0)
    if np.any(supported):
        with np.errstate(all="ignore"):
            mad[supported] = np.nanmedian(
                np.abs(departure[:, supported]), axis=0)
    sigma = MAD_TO_SIGMA * mad
    positive = sigma[np.isfinite(sigma) & (sigma > 0)]
    if positive.size:
        floor = max(1.0e-12, MAD_FLOOR_FRACTION * float(np.nanmedian(positive)))
    else:
        floor = 1.0e-12
    sigma = np.where(np.isfinite(sigma) & (sigma >= floor), sigma, floor)
    return sigma, floor


def _fit_one_amp(records, min_exposures):
    records = sorted(records, key=lambda row: (row["H5"], row["exposure"]))
    key = _amp_key(records[0])
    matrix = np.asarray([row["residual"] for row in records], dtype=float)
    science = np.asarray(parent.SCIENCE_MASK, dtype=bool)
    # This robust mean is a valid diagnostic product as soon as one usable
    # residual spectrum exists, even when the exposure support is insufficient
    # for PCA.  Keep PCA-specific support checks below it.
    mu = parent._safe_location(matrix, axis=0)
    mu[~science] = np.nan
    ifu_codes = sorted({int(row["IFU_CODE"]) for row in records})
    if len(ifu_codes) != 1:
        raise ValueError("physical amplifier changed IFU_CODE across exposures: %s" %
                         (key,))
    ifu_code = ifu_codes[0]
    required_exposures = max(min_exposures, N_COMPONENTS + 1)
    if len(records) < required_exposures:
        return None, {
            "key": key,
            "IFU_CODE": ifu_code,
            "mean": mu,
            "N_usable_exposures": len(records),
            "minimum_exposures": required_exposures,
            "N_supported_wavelengths": "",
            "minimum_supported_wavelengths": 10,
            "reason": "fewer_than_min_exposures",
            "detail": "%d usable exposure spectra < %d required" %
                      (len(records), required_exposures),
        }

    finite_fraction = np.mean(np.isfinite(matrix), axis=0)
    wave_support = science & (finite_fraction >= MIN_WAVE_SUPPORT_FRACTION)
    n_supported_wavelengths = int(np.sum(wave_support))
    if n_supported_wavelengths < 10:
        return None, {
            "key": key,
            "IFU_CODE": ifu_code,
            "mean": mu,
            "N_usable_exposures": len(records),
            "minimum_exposures": required_exposures,
            "N_supported_wavelengths": n_supported_wavelengths,
            "minimum_supported_wavelengths": 10,
            "reason": "fewer_than_min_wavelength_support",
            "detail": "%d supported wavelengths < 10 required" %
                      n_supported_wavelengths,
        }

    departure = matrix - mu[None, :]
    sigma, sigma_floor = _mad_scale(departure)

    z = departure / sigma[None, :]
    # Missing samples are set to zero AFTER robust centering, i.e. to the amp's
    # robust mean, and unsupported wavelengths are excluded from the SVD.
    zw = np.asarray(z[:, wave_support], dtype=float)
    zw[~np.isfinite(zw)] = 0.0

    # Direct SVD deliberately avoids an additional arithmetic recentering step;
    # the robust wavelength-by-wavelength centering above defines the PCA gauge.
    u, singular, vt = np.linalg.svd(zw, full_matrices=False)
    n_components = min(N_COMPONENTS, singular.size)
    if n_components < N_COMPONENTS:
        return None, {
            "key": key,
            "IFU_CODE": ifu_code,
            "mean": mu,
            "N_usable_exposures": len(records),
            "minimum_exposures": required_exposures,
            "N_supported_wavelengths": n_supported_wavelengths,
            "minimum_supported_wavelengths": 10,
            "reason": "insufficient_svd_rank",
            "detail": "%d available singular vectors < %d required" %
                      (n_components, N_COMPONENTS),
        }
    components_supported = vt[:N_COMPONENTS]
    scores = u[:, :N_COMPONENTS] * singular[:N_COMPONENTS]
    zhat_supported = scores @ components_supported

    zhat = np.zeros_like(z)
    zhat[:, wave_support] = zhat_supported
    reconstruction = mu[None, :] + sigma[None, :] * zhat
    reconstruction[:, ~science] = np.nan

    power = singular ** 2
    explained = (power[:N_COMPONENTS] / np.sum(power)
                 if np.sum(power) > 0 else np.full(N_COMPONENTS, np.nan))

    components = np.full((N_COMPONENTS, matrix.shape[1]), np.nan, dtype=float)
    components[:, wave_support] = components_supported

    before = _robust_scatter_flat(departure[:, science])
    after = _robust_scatter_flat((matrix - reconstruction)[:, science])
    exposure_error = np.asarray([
        _robust_scatter_flat((matrix[index] - reconstruction[index])[science])
        for index in range(len(records))
    ], dtype=float)

    return {
        "key": key,
        "IFU_CODE": ifu_code,
        "records": records,
        "matrix": matrix,
        "mean": mu,
        "scale": sigma,
        "sigma_floor": sigma_floor,
        "wave_support": wave_support,
        "components": components,
        "scores": scores,
        "reconstruction": reconstruction,
        "explained": explained,
        "robust_before": before,
        "robust_after": after,
        "exposure_error": exposure_error,
    }, None


def _fit_all(records, min_exposures):
    grouped = defaultdict(list)
    for record in records:
        grouped[_amp_key(record)].append(record)

    fits = []
    skipped = []
    robust_means = []
    for key in sorted(grouped):
        fit, skip = _fit_one_amp(grouped[key], min_exposures)
        source = fit if fit is not None else skip
        robust_means.append({
            "key": source["key"],
            "IFU_CODE": source["IFU_CODE"],
            "mean": source["mean"],
            "N_usable_exposures": source["N_usable_exposures"]
            if "N_usable_exposures" in source else len(source["records"]),
        })
        if fit is None:
            skipped.append(skip)
        else:
            fits.append(fit)
    return fits, skipped, robust_means


def _skipped_rows(skipped):
    rows = []
    for item in skipped:
        key = item["key"]
        rows.append({
            "SPECID": key[0],
            "IFUSLOT": key[1],
            "IFUID": key[2],
            "AMP": key[3],
            "physical_amplifier": _amp_label(key),
            "N_usable_exposures": item["N_usable_exposures"],
            "minimum_exposures": item["minimum_exposures"],
            "N_supported_wavelengths": item["N_supported_wavelengths"],
            "minimum_supported_wavelengths": item["minimum_supported_wavelengths"],
            "reason": item["reason"],
            "detail": item["detail"],
        })
    return rows


def _summary_rows(fits):
    rows = []
    for fit in fits:
        key = fit["key"]
        n_blank = np.asarray([row["N_blank_central"] for row in fit["records"]], dtype=float)
        rows.append({
            "SPECID": key[0],
            "IFUSLOT": key[1],
            "IFUID": key[2],
            "IFU_CODE": fit["IFU_CODE"],
            "AMP": key[3],
            "N_usable_exposures": len(fit["records"]),
            "median_N_blank_central": float(np.median(n_blank)),
            "min_N_blank_central": int(np.min(n_blank)),
            "PC1_explained_variance": float(fit["explained"][0]),
            "PC2_explained_variance": float(fit["explained"][1]),
            "PC3_explained_variance": float(fit["explained"][2]),
            "cumulative_3PC_explained_variance": float(np.sum(fit["explained"])),
            "robust_scatter_before_PCA": fit["robust_before"],
            "robust_scatter_after_3PC": fit["robust_after"],
            "MAD_sigma_floor": fit["sigma_floor"],
            "N_supported_wavelengths": int(np.sum(fit["wave_support"])),
        })
    return rows


def _save_models(path, fits, robust_means):
    labels = np.asarray([_amp_label(fit["key"]) for fit in fits], dtype="U32")
    ifu_code = np.asarray([fit["IFU_CODE"] for fit in fits], dtype=np.int32)
    robust_labels = np.asarray(
        [_amp_label(item["key"]) for item in robust_means], dtype="U32")
    robust_ifu_code = np.asarray(
        [item["IFU_CODE"] for item in robust_means], dtype=np.int32)
    robust_mean_exposures = np.asarray(
        [item["N_usable_exposures"] for item in robust_means], dtype=np.int32)
    robust_means_array = np.asarray(
        [item["mean"] for item in robust_means], dtype=np.float32)
    n_wave = parent.WAVE.size
    scales = (np.asarray([fit["scale"] for fit in fits], dtype=np.float32)
              if fits else np.empty((0, n_wave), dtype=np.float32))
    components = (np.asarray([fit["components"] for fit in fits], dtype=np.float32)
                  if fits else np.empty((0, N_COMPONENTS, n_wave), dtype=np.float32))
    explained = (np.asarray([fit["explained"] for fit in fits], dtype=np.float32)
                 if fits else np.empty((0, N_COMPONENTS), dtype=np.float32))
    support = (np.asarray([fit["wave_support"] for fit in fits], dtype=bool)
               if fits else np.empty((0, n_wave), dtype=bool))
    np.savez_compressed(
        path,
        wavelength=np.asarray(parent.WAVE, dtype=np.float32),
        science_mask=np.asarray(parent.SCIENCE_MASK, dtype=bool),
        # These arrays cover every physical amplifier with at least one
        # retained amplifier/exposure residual spectrum.
        robust_mean_amp_label=robust_labels,
        robust_mean_IFU_CODE=robust_ifu_code,
        robust_mean_N_usable_exposures=robust_mean_exposures,
        robust_mean=robust_means_array,
        # These arrays are PCA-only and therefore cover only support-passing
        # physical amplifiers.  The historical names are retained for API
        # compatibility with the first PCA output.
        amp_label=labels,
        IFU_CODE=ifu_code,
        pca_amp_label=labels,
        pca_IFU_CODE=ifu_code,
        mad_sigma=scales,
        components=components,
        explained_variance_ratio=explained,
        wavelength_support=support,
    )


def _select_gallery(fits, maximum):
    """Deterministically span the cumulative-variance distribution."""
    candidates = [fit for fit in fits if np.all(np.isfinite(fit["explained"]))]
    candidates.sort(key=lambda fit: (
        float(np.sum(fit["explained"])), -len(fit["records"]), fit["key"]))
    if len(candidates) <= maximum:
        return candidates
    indices = np.linspace(0, len(candidates) - 1, maximum).round().astype(int)
    return [candidates[index] for index in np.unique(indices)]


def _select_exposure_examples(fit, maximum=4):
    order = np.argsort(np.where(np.isfinite(fit["exposure_error"]),
                                fit["exposure_error"], np.inf))
    if order.size <= maximum:
        return order.tolist()
    picks = np.linspace(0, order.size - 1, maximum).round().astype(int)
    return [int(order[index]) for index in np.unique(picks)]


def _plot_fit(path, fit):
    examples = _select_exposure_examples(fit, 4)
    nrows = 1 + len(examples)
    figure, axes = plt.subplots(
        nrows, 1, figsize=(12, 3.0 + 2.2 * len(examples)), sharex=True,
        squeeze=False)
    axes = axes[:, 0]
    wave = np.asarray(parent.WAVE, dtype=float)
    science = np.asarray(parent.SCIENCE_MASK, dtype=bool)

    top = axes[0]
    top.plot(wave[science], fit["mean"][science], lw=1.5,
             label="robust mean residual")
    top.set_ylabel("mean residual")
    top.grid(alpha=.15)
    twin = top.twinx()
    for index in range(N_COMPONENTS):
        twin.plot(wave[science], fit["components"][index, science],
                  lw=1.0, alpha=.8, label="PC%d" % (index + 1))
    twin.set_ylabel("standardized PC shape")
    handles1, labels1 = top.get_legend_handles_labels()
    handles2, labels2 = twin.get_legend_handles_labels()
    top.legend(handles1 + handles2, labels1 + labels2,
               fontsize=7, ncol=4, loc="upper right")
    ev = fit["explained"]
    top.set_title(
        "%s; IFU_CODE=%d; Nexp=%d; EV=%.3f/%.3f/%.3f; cumulative=%.3f" %
        (_amp_label(fit["key"]), fit["IFU_CODE"], len(fit["records"]),
         ev[0], ev[1], ev[2], np.sum(ev)))

    for axis, index in zip(axes[1:], examples):
        record = fit["records"][index]
        axis.plot(wave[science], fit["matrix"][index, science],
                  lw=1.0, label="actual residual")
        axis.plot(wave[science], fit["reconstruction"][index, science],
                  lw=1.1, ls="--", label="3-PC reconstruction")
        axis.plot(wave[science], fit["mean"][science],
                  lw=.7, alpha=.45, label="robust mean")
        coeff = fit["scores"][index]
        axis.set_ylabel("residual")
        axis.set_title(
            "%s e%d; Nblank=%d; coeff=(%.2f, %.2f, %.2f); reconstruction scatter=%.5g" %
            (record["H5"], record["exposure"], record["N_blank_central"],
             coeff[0], coeff[1], coeff[2], fit["exposure_error"][index]),
            fontsize=8)
        axis.grid(alpha=.13)
        axis.legend(fontsize=7, loc="upper right", ncol=3)

    axes[-1].set_xlabel("wavelength [Angstrom]")
    axes[-1].set_xlim(parent.WAVE_MIN, parent.WAVE_MAX)
    figure.tight_layout()
    figure.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(figure)


def _write_gallery(output_dir, fits, maximum):
    gallery = _select_gallery(fits, maximum)
    gallery_dir = output_dir / "gallery"
    gallery_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, fit in enumerate(gallery, 1):
        name = "%02d_%s.png" % (index, _amp_label(fit["key"]).replace("/", "_"))
        path = gallery_dir / name
        _plot_fit(path, fit)
        rows.append({
            "gallery_index": index,
            "physical_amplifier": _amp_label(fit["key"]),
            "IFU_CODE": fit["IFU_CODE"],
            "N_usable_exposures": len(fit["records"]),
            "cumulative_3PC_explained_variance": float(np.sum(fit["explained"])),
            "plot": str(path.resolve()),
        })
    _write_rows(output_dir / "gallery_manifest.csv", rows)
    return rows


def main():
    args = _parse_args()
    if args.min_blank < 1:
        raise SystemExit("--min-blank must be >= 1")
    if args.min_exposures < N_COMPONENTS + 1:
        raise SystemExit("--min-exposures must be >= %d" % (N_COMPONENTS + 1))
    if args.gallery_size < 1:
        raise SystemExit("--gallery-size must be >= 1")

    started = time.perf_counter()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise SystemExit("output directory is nonempty; use --overwrite: %s" % output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    _validate_parent_semantics()
    inputs = _resolve_inputs(args)
    _print_provenance(inputs)

    records, blank_provenance, loader_provenance, exposure_instances = _collect_amp_exposure_spectra(
        inputs, args.min_blank)
    print("[pca] processed %d exposure instances" % exposure_instances, flush=True)
    print("[pca] retained %d compact amplifier/exposure spectra" % len(records), flush=True)

    fits, skipped, robust_means = _fit_all(records, args.min_exposures)
    skipped_rows = _skipped_rows(skipped)
    skipped_by_reason = defaultdict(int)
    for row in skipped_rows:
        skipped_by_reason[row["reason"]] += 1
    reason_summary = ", ".join(
        "%s=%d" % (reason, skipped_by_reason[reason])
    for reason in sorted(skipped_by_reason)) or "none"
    print("[pca] fitted %d physical amplifiers; skipped %d (%s)" %
          (len(fits), len(skipped), reason_summary), flush=True)

    summary_rows = _summary_rows(fits)
    _write_rows(output_dir / "amp_residual_pca_summary.csv", summary_rows)
    _write_rows(output_dir / "amp_residual_pca_skipped.csv", skipped_rows)
    _save_models(output_dir / "amp_residual_pca_models.npz", fits, robust_means)
    if not fits:
        raise RuntimeError("no physical amplifier has enough support for 3-component PCA")
    gallery_rows = [] if args.no_plots else _write_gallery(
        output_dir, fits, args.gallery_size)

    state = {
        "schema_version": "m101_amp_residual_pca_v2",
        "script": str(Path(__file__).resolve()),
        "purpose": (
            "diagnostic-only 3-PC description of exposure-to-exposure central-q "
            "amplifier residual spectra after the current frozen ON/OFF-derived M and "
            "full-resolution exposure sky"),
        "authoritative_parent": str(Path(parent.__file__).resolve()),
        "model_semantics": {
            "response_field": "hierarchy.cumulative_final.R_total_h_i_a",
            "M_definition": "exp(R_total_h_i_a)",
            "M_wavelength_independent": True,
            "residual_definition": "E_f(lambda) = D_f(lambda) - M_f*S_e(lambda)",
            "sky_definition": (
                "robust full-resolution sky from classified blank, hardware-valid "
                "q=40..75 fibers after D/M"),
            "alpha_f_q_K_applied": False,
            "alpha_reason": (
                "central-q residual is intentionally the spectral level centered away "
                "by the established alpha fit using f(q)-f0"),
            "q_range_inclusive": [parent.Q_MIN, parent.Q_MAX],
            "wavelength_range_A": [parent.WAVE_MIN, parent.WAVE_MAX],
            "native_wavelength_grid_used": True,
            "interpolation_used": False,
        },
        "pca_semantics": {
            "N_components": N_COMPONENTS,
            "mean": "wavelength-by-wavelength biweight robust location across exposures",
            "scale": "1.4826 * MAD of mean-subtracted exposure residuals at each wavelength",
            "MAD_floor_fraction_of_median_positive_sigma": MAD_FLOOR_FRACTION,
            "minimum_wavelength_support_fraction": MIN_WAVE_SUPPORT_FRACTION,
            "additional_arithmetic_recentering_before_SVD": False,
        },
        "saved_model_semantics": {
            "robust_mean": (
                "one wavelength-by-wavelength biweight robust mean for every physical "
                "amplifier with at least one retained amplifier/exposure residual spectrum"),
            "PCA_arrays": (
                "saved only for physical amplifiers passing minimum exposure, wavelength "
                "support, and 3-component SVD-rank checks"),
            "robust_mean_amp_label": "labels aligned with robust_mean",
            "amp_label": "labels aligned with PCA-only arrays",
        },
        "selection": {
            "minimum_blank_central_fibers_per_amp_exposure": args.min_blank,
            "minimum_exposures_per_physical_amp": args.min_exposures,
        },
        "counts": {
            "H5": len(inputs["h5_paths"]),
            "exposure_instances": exposure_instances,
            "amp_exposure_spectra": len(records),
            "physical_amps_with_robust_mean": len(robust_means),
            "physical_amps_fit": len(fits),
            "physical_amps_skipped_for_support": len(skipped),
            "skipped_by_reason": dict(sorted(skipped_by_reason.items())),
            "gallery": len(gallery_rows),
        },
        "provenance": {
            "state": parent._file_identity(inputs["state_path"]),
            "band_cache": parent._file_identity(inputs["cache_path"]),
            "band_cache_manifest": parent._file_identity(inputs["manifest_path"]),
            "blank_file": parent._file_identity(inputs["blank_file"]),
            "on_filter": parent._file_identity(inputs["on_filter"]),
            "off_filter": parent._file_identity(inputs["off_filter"]),
            "H5": [parent._file_identity(path) for path in inputs["h5_paths"]],
            "blank_loader": blank_provenance,
            "native_loader": loader_provenance,
        },
        "response_composition_validation": inputs["composition"],
        "outputs": {
            "summary": str((output_dir / "amp_residual_pca_summary.csv").resolve()),
            "skipped": str((output_dir / "amp_residual_pca_skipped.csv").resolve()),
            "models": str((output_dir / "amp_residual_pca_models.npz").resolve()),
            "gallery_manifest": (
                str((output_dir / "gallery_manifest.csv").resolve())
                if not args.no_plots else None),
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    (output_dir / "amp_residual_pca_state.json").write_text(
        json.dumps(_json_ready(state), indent=2, sort_keys=True))

    cumulative = np.asarray([
        row["cumulative_3PC_explained_variance"] for row in summary_rows], dtype=float)
    before = np.asarray([row["robust_scatter_before_PCA"] for row in summary_rows], dtype=float)
    after = np.asarray([row["robust_scatter_after_3PC"] for row in summary_rows], dtype=float)
    print("[pca] cumulative 3-PC variance median=%.4f p10=%.4f p90=%.4f" %
          (np.nanmedian(cumulative), np.nanpercentile(cumulative, 10),
           np.nanpercentile(cumulative, 90)), flush=True)
    print("[pca] median robust scatter before/after: %.6g -> %.6g" %
          (np.nanmedian(before), np.nanmedian(after)), flush=True)
    print("[pca] wrote %s" % output_dir, flush=True)


if __name__ == "__main__":
    main()
