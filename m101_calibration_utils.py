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


def _collapse_many(values: np.ndarray, responses: np.ndarray, *, error: bool,
                   block_size: int = 4096) -> tuple[np.ndarray, np.ndarray]:
    """Collapse one native array against several responses in BLAS-sized blocks.

    The scalar implementation is called once per band by the older code.  The
    formulas here are the same finite-response weighted sums, but the band
    dimension is moved into a matrix multiply.  Blocking keeps the temporary
    arrays bounded for a full VIRUS exposure.
    """
    values = np.asarray(values, dtype=float)
    responses = np.asarray(responses, dtype=float)
    if values.ndim != 2 or responses.ndim != 2:
        raise ValueError("values and responses must both be two-dimensional")
    if values.shape[1] != responses.shape[1]:
        raise ValueError("values and responses have incompatible wavelength axes")
    if block_size <= 0:
        raise ValueError("block_size must be positive")

    finite_values = np.isfinite(values)
    finite_responses = np.isfinite(responses)
    weighted_responses = np.where(finite_responses, responses, 0.0)
    result = np.full((values.shape[0], responses.shape[0]), np.nan, dtype=float)
    response_total = np.sum(np.abs(weighted_responses), axis=1)
    fractions = np.zeros_like(result)
    valid_response_total = response_total > 0
    squared_responses = weighted_responses ** 2 if error else None
    absolute_responses = np.abs(weighted_responses[valid_response_total])
    for start in range(0, values.shape[0], block_size):
        stop = min(start + block_size, values.shape[0])
        finite_block = finite_values[start:stop]
        weighted_block = finite_block.astype(float)
        denominators = weighted_block @ weighted_responses.T
        good = denominators != 0
        if error:
            cleaned = np.where(finite_block, values[start:stop], 0.0)
            numerators = (cleaned ** 2) @ squared_responses.T
            block_result = np.full(denominators.shape, np.nan, dtype=float)
            block_result[good] = (np.sqrt(numerators[good])
                                  / np.abs(denominators[good]))
        else:
            cleaned = np.where(finite_block, values[start:stop], 0.0)
            numerators = cleaned @ weighted_responses.T
            block_result = np.full(denominators.shape, np.nan, dtype=float)
            block_result[good] = numerators[good] / denominators[good]
        result[start:stop] = block_result
        if np.any(valid_response_total):
            fractions[start:stop, valid_response_total] = (
                weighted_block @ absolute_responses.T
                / response_total[valid_response_total]
            )
    return result, fractions


def collapse_many(values: np.ndarray, responses: np.ndarray, *,
                  block_size: int = 4096) -> tuple[np.ndarray, np.ndarray]:
    """Return weighted collapsed values and finite-response fractions."""
    return _collapse_many(values, responses, error=False, block_size=block_size)


def collapse_error_many(error: np.ndarray, responses: np.ndarray, *,
                        block_size: int = 4096) -> tuple[np.ndarray, np.ndarray]:
    """Return propagated errors and finite-response fractions for many bands."""
    return _collapse_many(error, responses, error=True, block_size=block_size)


def robust_location(values: np.ndarray, axis=None):
    """Use the project biweight implementation, with its existing fallback."""
    result = biweight(np.asarray(values, dtype=float), axis=axis)
    if axis is None:
        return float(result) if np.isfinite(result) else float(np.nanmedian(values))
    result = np.asarray(result, dtype=float)
    invalid = ~np.isfinite(result)
    if not np.any(invalid):
        return result
    fallback = np.nanmedian(np.asarray(values, dtype=float), axis=axis)
    return np.where(invalid, fallback, result)


def robust_scatter(values: np.ndarray, axis=None):
    """Return the project biweight scale where possible."""
    _, scale = biweight(np.asarray(values, dtype=float), axis=axis, calc_std=True)
    scale = np.asarray(scale, dtype=float)
    if axis is None:
        if np.isfinite(scale) and scale > 0:
            return float(scale)
        return float(validated_m101.robust_scale(values))
    invalid = ~np.isfinite(scale) | (scale <= 0)
    if not np.any(invalid):
        return scale
    fallback = np.nanstd(np.asarray(values, dtype=float), axis=axis)
    return np.where(invalid, fallback, scale)


def robust_location_scatter(values: np.ndarray, axis=None):
    """Compute location and scatter from one shared biweight evaluation."""
    values = np.asarray(values, dtype=float)
    location, scale = biweight(values, axis=axis, calc_std=True)
    if axis is None:
        location = float(location) if np.isfinite(location) else float(np.nanmedian(values))
        scale = (float(scale) if np.isfinite(scale) and scale > 0
                 else float(validated_m101.robust_scale(values)))
        return location, scale

    location = np.asarray(location, dtype=float)
    scale = np.asarray(scale, dtype=float)
    invalid_location = ~np.isfinite(location)
    invalid_scale = ~np.isfinite(scale) | (scale <= 0)
    if np.any(invalid_location) or np.any(invalid_scale):
        fallback_median = (np.nanmedian(values, axis=axis)
                           if np.any(invalid_location) else None)
        fallback_std = (np.nanstd(values, axis=axis)
                        if np.any(invalid_scale) else None)
        if fallback_median is not None:
            location = np.where(invalid_location, fallback_median, location)
        if fallback_std is not None:
            scale = np.where(invalid_scale, fallback_std, scale)
    return location, scale


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
