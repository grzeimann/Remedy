"""Importable access to the validated M101 external blank classification."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import build_m101_measurements as builder


SCHEMA_VERSION = "m101_external_blank_fibers_v1"


def load(path, h5_paths):
    masks, provenance = builder._load_external_blank_fibers(path, h5_paths)
    return {Path(name).name: value.copy() for name, value in masks.items()}, provenance


def build(h5_paths, image, output, **kwargs):
    """Run the existing spectroscopy-independent blank producer explicitly."""
    import diagnose_m101_external_blank_sky as producer

    output = Path(output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    chunks = producer.read_h5_metadata([Path(path).expanduser().resolve() for path in h5_paths])
    data = producer.concatenate_chunks(chunks)
    image_data = producer.load_external_image(image)
    radius = float(kwargs.get("fiber_radius_arcsec", producer.DEFAULT_FIBER_RADIUS_ARCSEC))
    guard = float(kwargs.get("guard_radius_arcsec", producer.DEFAULT_GUARD_RADIUS_ARCSEC))
    minimum_valid = float(kwargs.get("minimum_aperture_valid_fraction",
                                     producer.DEFAULT_MINIMUM_APERTURE_VALID_FRACTION))
    sigma = float(kwargs.get("blank_sigma", producer.DEFAULT_BLANK_SIGMA))
    reference_radius = float(kwargs.get("reference_radius_arcmin",
                                         producer.DEFAULT_REFERENCE_RADIUS_ARCMIN))
    kernel, aperture_info = producer.build_aperture_kernel(
        image_data["pixel_scales_arcsec"], radius + guard)
    aperture_mean, aperture_fraction = producer.make_aperture_map(image_data, kernel)
    valid_field = (aperture_fraction >= minimum_valid) & np.isfinite(aperture_mean)
    background = producer.estimate_background(aperture_mean[valid_field])
    threshold = background["mu_blank"] + sigma * background["sigma_blank"]
    (data["x"], data["y"], data["aperture_value"], data["aperture_valid_fraction"],
     data["image_covered"], data["valid_aperture"]) = producer.sample_aperture_map(
         aperture_mean, aperture_fraction, image_data["wcs"], data["ra"], data["dec"],
         minimum_valid)
    data["blank"] = data["valid_aperture"] & (data["aperture_value"] <= threshold)
    producer.write_fiber_table(output, data, reference_radius)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "producer": str(Path(producer.__file__).resolve()),
        "input_h5": [builder._file_identity(path) for path in h5_paths],
        "external_image": builder._file_identity(image),
        "configuration": {"aperture": aperture_info, "blank_sigma": sigma,
                           "minimum_aperture_valid_fraction": minimum_valid,
                           "reference_radius_arcmin": reference_radius},
        "rows": int(data["ra"].size), "blank_rows": int(data["blank"].sum()),
        "output": str(output),
    }
    summary_path = output.with_suffix(output.suffix + ".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    return output, summary


def load_or_build(path, h5_paths, build_path=None, build_kwargs=None):
    if path is not None:
        return load(path, h5_paths)
    if build_path is None:
        raise ValueError("blank classification requires --blank-file or explicit --build-blank")
    output, _ = build(h5_paths=h5_paths, output=build_path, **(build_kwargs or {}))
    return load(output, h5_paths)
