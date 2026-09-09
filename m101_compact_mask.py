"""Importable wrapper for the validated external compact mask."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from build_m101_external_compact_mask import load_compact_mask, mask_radec


def load(path):
    return load_compact_mask(path)


def build(on_image, off_image, psf_fwhm_on, psf_fwhm_off,
          fiber_radius_arcsec, astrometry_margin_arcsec, output_mask,
          output_sources=None, output_diagnostic=None, output_summary=None,
          peak_sigma=8.0, grow_sigma=3.0, compactness_max=1.5,
          broad_psf_factor=3.0, minimum_dilation_psf=0.0, overwrite=False):
    """Regenerate the validated mask using the producer's exact primitives.

    The command-line producer owns the morphology implementation.  This
    function keeps its orchestration importable without invoking argparse or
    changing the producer script.
    """
    import build_m101_external_compact_mask as producer

    output_mask = Path(output_mask).expanduser().resolve()
    output_sources = Path(output_sources or output_mask.with_name(
        output_mask.stem + "_sources.csv")).expanduser().resolve()
    output_diagnostic = Path(output_diagnostic or output_mask.with_name(
        output_mask.stem + "_diagnostic.png")).expanduser().resolve()
    output_summary = Path(output_summary or output_mask.with_name(
        output_mask.stem + "_summary.json")).expanduser().resolve()
    outputs = (output_mask, output_sources, output_diagnostic, output_summary)
    if not overwrite:
        existing = [str(path) for path in outputs if path.exists()]
        if existing:
            raise FileExistsError("compact-mask outputs exist: %s" % ", ".join(existing))

    on, on_wcs, on_scale, on_path = producer._read_celestial_image(on_image)
    off, off_wcs, off_scale, off_path = producer._read_celestial_image(off_image)
    common = dict(fiber_radius_arcsec=fiber_radius_arcsec,
                  astrometry_margin_arcsec=astrometry_margin_arcsec,
                  peak_sigma=peak_sigma, grow_sigma=grow_sigma,
                  compactness_max=compactness_max,
                  broad_psf_factor=broad_psf_factor,
                  minimum_dilation_psf=minimum_dilation_psf)
    on_result = producer.detect_compact_features(on, on_wcs, psf_fwhm_on, **common)
    off_result = producer.detect_compact_features(off, off_wcs, psf_fwhm_off, **common)
    same_grid = producer._same_grid(on.shape, on_wcs, off.shape, off_wcs)
    if same_grid:
        output_wcs, output_shape = on_wcs, on.shape
        on_mask, off_mask = on_result["mask"], off_result["mask"]
    else:
        output_wcs, output_shape = producer._common_tan_wcs(
            on.shape, on_wcs, off.shape, off_wcs, min(on_scale, off_scale))
        on_mask = producer._resample_mask(on_result["mask"], on_wcs, output_wcs, output_shape)
        off_mask = producer._resample_mask(off_result["mask"], off_wcs, output_wcs, output_shape)
    union = (on_mask.astype(np.uint8) * producer.MASK_BIT_ON
             | off_mask.astype(np.uint8) * producer.MASK_BIT_OFF)
    for path in outputs:
        path.parent.mkdir(parents=True, exist_ok=True)
    producer._write_mask(output_mask, union, output_wcs, {
        "BASIS": "external imaging morphology only", "ONFILE": str(on_path),
        "OFFFILE": str(off_path), "PSFON": psf_fwhm_on, "PSFOFF": psf_fwhm_off,
        "FIBRAD": fiber_radius_arcsec, "ASTROM": astrometry_margin_arcsec,
        "PEAKSIG": peak_sigma, "GROWSIG": grow_sigma, "COMPMAX": compactness_max,
        "SAMGRID": same_grid})
    source_rows = []
    for band, result in (("ON", on_result), ("OFF", off_result)):
        source_rows.extend(dict(band=band, **producer._strip_private(feature))
                           for feature in result["features"])
    producer._write_sources(output_sources, source_rows)
    summary = {
        "schema_version": "m101_external_compact_mask_v1",
        "input_files": {"ON": str(on_path), "OFF": str(off_path)},
        "same_grid": bool(same_grid), "output_mask": str(output_mask),
        "output_sources": str(output_sources), "output_diagnostic": str(output_diagnostic),
        "output_summary": str(output_summary),
        "number_ON_detections": len(on_result["features"]),
        "number_OFF_detections": len(off_result["features"]),
        "mask_bits": {"0": "unmasked", "1": "ON", "2": "OFF", "3": "both"},
        "mask_basis": "external imaging morphology only; no VIRUS residual or Bayesian quantity used",
    }
    output_summary.write_text(json.dumps(summary, indent=2, sort_keys=True))
    producer._plot_diagnostic(output_diagnostic, on, off, on_result, off_result, union)
    return output_mask, summary


def load_or_build(path, build_kwargs=None):
    if path is not None:
        return load(path)
    if not build_kwargs:
        raise ValueError("compact mask requires --compact-mask or explicit --build-compact-mask")
    build(**build_kwargs)
    output = build_kwargs.get("output_mask")
    if output is None:
        raise ValueError("compact mask build arguments must include output_mask")
    return load(Path(output))
