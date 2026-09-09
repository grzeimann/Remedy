"""Load or explicitly regenerate the validated M101 external measurements."""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import tables

import diagnose_m101_hierarchical as validated_m101
import build_m101_measurements as builder


SCHEMA_VERSION = "m101_external_measurements_v1"


def _identity(path):
    return builder._file_identity(Path(path))


def load(path, h5_paths):
    """Load external ``I`` and validity arrays from the existing PASS-1 cache."""
    path = Path(path).expanduser().resolve()
    with path.open("rb") as stream:
        cache = pickle.load(stream)
    by_name = {Path(item["h5"]).name: item for item in cache.get("calibrations", [])}
    expected = {Path(item).name for item in h5_paths}
    if set(by_name) != expected:
        raise ValueError("external measurement cache H5 set does not match inputs")
    result = {}
    for h5_path in h5_paths:
        item = by_name[h5_path.name]
        result[h5_path.name] = {
            "external_object": {tuple(key): np.asarray(value, dtype=float)
                                 for key, value in item["external_object"].items()},
            "external_valid": {tuple(key): np.asarray(value, dtype=bool)
                                for key, value in item["external_valid"].items()},
            "global_g": {key: float(value) for key, value in cache["global_g"].items()},
        }
    provenance = {
        "schema_version": SCHEMA_VERSION,
        "cache": _identity(path),
        "cache_version": cache.get("version"),
        "calibration_model": cache.get("calibration_model"),
        "input_h5": [_identity(item) for item in h5_paths],
        "images": cache.get("images"),
        "filters_in_cache": list(cache.get("filters", {})),
        "X_definition": "X = g_global * I; I is matched external image exact-aperture measurement",
    }
    return result, provenance


def source_comparison_validity(external_valid, compact_masked,
                               compact_inside, date_hardware_valid,
                               object_finite, object_error):
    """Apply the existing source-reference validity semantics.

    The compact mask excludes external source comparisons only where its
    raster has coverage and a compact/high-gradient feature is present.  The
    function returns a comparison selector; it never mutates or returns a
    VIRUS-spectrum validity mask.
    """
    arrays = [np.asarray(value, dtype=bool) for value in
              (external_valid, compact_masked, compact_inside, date_hardware_valid,
               object_finite)]
    error = np.asarray(object_error, dtype=float)
    if any(value.shape != arrays[0].shape for value in arrays[1:]) or error.shape != arrays[0].shape:
        raise ValueError("source validity arrays have different shapes")
    compact_excluded = arrays[1] & arrays[2]
    return (arrays[0] & ~compact_excluded & arrays[3] & arrays[4]
            & np.isfinite(error) & (error > 0))


def build(h5_paths, on_image, off_image, on_filter, off_filter, output, global_g):
    """Regenerate only the external ``I`` measurements using validated primitives."""
    images = {"ON": validated_m101.load_image(on_image),
              "OFF": validated_m101.load_image(off_image)}
    responses = {"ON": validated_m101.read_filter(on_filter),
                 "OFF": validated_m101.read_filter(off_filter)}
    if not isinstance(global_g, dict) or any(band not in global_g for band in ("ON", "OFF")):
        raise ValueError("external build requires validated global_g values for ON and OFF")
    global_g = {band: float(global_g[band]) for band in ("ON", "OFF")}
    output = Path(output).expanduser().resolve()
    calibrations = []
    for h5_path in h5_paths:
        with tables.open_file(h5_path, mode="r") as h5:
            groups, labels = validated_m101.build_groups(h5.root.Info)
            ra = np.asarray(h5.root.Info.cols.ra[:], dtype=float)
            dec = np.asarray(h5.root.Info.cols.dec[:], dtype=float)
            surveys = {int(row["exp"]): {name: row[name] for name in h5.root.Survey.colnames}
                       for row in h5.root.Survey}
            item = {"h5": h5_path.name, "external_object": {}, "external_valid": {}}
            for exposure in (1, 2, 3):
                indices = np.flatnonzero(labels == exposure)
                survey = surveys[exposure]
                for band in ("ON", "OFF"):
                    eff_ra, eff_dec = validated_m101.adr_positions(
                        ra[indices], dec[indices], survey, responses[band])
                    matched = validated_m101.matched_external_image(
                        images[band], exposure, float(survey["fwhm"]), band)
                    value, valid = validated_m101.sample_image_exact(
                        images[band], matched, eff_ra, eff_dec)
                    item["external_object"][(exposure, band)] = value
                    item["external_valid"][(exposure, band)] = valid
            calibrations.append(item)
    payload = {"version": 1, "schema_version": SCHEMA_VERSION,
               "h5_files": [str(Path(path).expanduser().resolve()) for path in h5_paths],
               "images": {"ON": str(Path(on_image).expanduser().resolve()),
                          "OFF": str(Path(off_image).expanduser().resolve())},
               "filters": responses, "global_g": global_g,
               "calibrations": calibrations}
    with output.open("wb") as stream:
        pickle.dump(payload, stream, protocol=pickle.HIGHEST_PROTOCOL)
    summary = output.with_suffix(output.suffix + ".summary.json")
    summary.write_text(json.dumps({"schema_version": SCHEMA_VERSION,
                                   "output": str(output),
                                   "input_h5": [_identity(path) for path in h5_paths],
                                   "images": {band: _identity(path) for band, path in
                                              (("ON", on_image), ("OFF", off_image))},
                                   "X_definition": "X = g_global * I"},
                                  indent=2, sort_keys=True))
    return output, payload


def load_or_build(path, h5_paths, build_kwargs=None):
    if path is not None:
        return load(path, h5_paths)
    if not build_kwargs:
        raise ValueError("external measurements require --external-cache or explicit build inputs")
    kwargs = dict(build_kwargs)
    output = kwargs.pop("output")
    return load(build(output=output, h5_paths=h5_paths, **kwargs)[0], h5_paths)
