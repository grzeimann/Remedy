"""Small utilities for the new M101 total-spectrum calibration.

The numerical primitives in this module are deliberately thin wrappers around
the validated Remedy implementations.  The new scientific model is built in
``fit_m101_calibration_v2.py``; this module only holds reusable bookkeeping,
statistics, and exact collapse calls.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from math_utils import biweight

import build_m101_measurements as validated_measurements
import diagnose_m101_hierarchical as validated_m101


BANDS = ("HIGH1", "LOW1", "HIGH2", "LOW2", "HIGH3", "ON", "OFF")
ALL_BANDS = BANDS


def file_identity(path: str | Path) -> dict:
    """Use the repository's small-file/large-file identity convention."""
    return validated_measurements._file_identity(Path(path))


def small_file_hash(path: str | Path) -> str:
    return validated_measurements._small_file_hash(Path(path))


def collapse(values: np.ndarray, response: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Use the measurement builder's finite-response weighted mean exactly."""
    return validated_measurements._collapse_with_fraction(values, response)


def collapse_error(error: np.ndarray, response: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Use the measurement builder's propagated error exactly."""
    return validated_measurements._collapse_error(error, response)


def robust_location(values: np.ndarray, axis=None):
    """Use the project biweight implementation, with its existing fallback."""
    result = biweight(np.asarray(values, dtype=float), axis=axis)
    if axis is None:
        return float(result) if np.isfinite(result) else float(np.nanmedian(values))
    result = np.asarray(result, dtype=float)
    fallback = np.nanmedian(np.asarray(values, dtype=float), axis=axis)
    return np.where(np.isfinite(result), result, fallback)


def robust_scatter(values: np.ndarray, axis=None):
    """Return the project biweight scale where possible."""
    _, scale = biweight(np.asarray(values, dtype=float), axis=axis, calc_std=True)
    scale = np.asarray(scale, dtype=float)
    if axis is None:
        if np.isfinite(scale) and scale > 0:
            return float(scale)
        return float(validated_m101.robust_scale(values))
    fallback = np.nanstd(np.asarray(values, dtype=float), axis=axis)
    return np.where(np.isfinite(scale) & (scale > 0), scale, fallback)


def residual_summary(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"N": 0, "median": np.nan, "biweight_location": np.nan,
                "robust_rms": np.nan, "arithmetic_rms": np.nan,
                "p16": np.nan, "p84": np.nan, "p95_abs": np.nan,
                "p99_abs": np.nan, "max_abs": np.nan}
    absolute = np.abs(values)
    return {
        "N": int(values.size),
        "median": float(np.median(values)),
        "biweight_location": float(robust_location(values)),
        "robust_rms": float(validated_m101.robust_scale(values)),
        "arithmetic_rms": float(np.sqrt(np.mean(values ** 2))),
        "p16": float(np.percentile(values, 16)),
        "p84": float(np.percentile(values, 84)),
        "p95_abs": float(np.percentile(absolute, 95)),
        "p99_abs": float(np.percentile(absolute, 99)),
        "max_abs": float(np.max(absolute)),
    }


def json_ready(value):
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_ready(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def write_json(path: str | Path, value: dict) -> None:
    Path(path).write_text(json.dumps(json_ready(value), indent=2, sort_keys=True))


def contrast_basis(n: int) -> np.ndarray:
    """Use the same scipy Helmert convention as the existing hierarchy."""
    from scipy import linalg
    if n <= 1:
        return np.zeros((n, 0), dtype=float)
    return np.asarray(linalg.helmert(n, full=False).T, dtype=float)


def contrast_values(coefficients: np.ndarray, basis: np.ndarray) -> np.ndarray:
    coefficients = np.asarray(coefficients, dtype=float)
    return np.asarray(basis @ coefficients, dtype=float)


def sufficient_native_spectrum(spectrum: np.ndarray, minimum_fraction: float) -> np.ndarray:
    spectrum = np.asarray(spectrum, dtype=float)
    threshold = int(np.ceil(float(minimum_fraction) * spectrum.shape[1]))
    return np.isfinite(spectrum).sum(axis=1) >= threshold
