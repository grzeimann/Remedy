# -*- coding: utf-8 -*-
"""
Mask utilities for identifying bad pixels on VIRUS-like detector images.

This module factors out the bad-pixel identification logic that was embedded in
remedy_get_wave_v2 (lines ~832–879), so it can be reused elsewhere (e.g., during
calibration file creation).

Primary entry point
- identify_bad_pixels(image, amp_rows=112, very_bad_thresh=10.0)
  Given a 2D image, it robustly normalizes residual-like structure, then applies
  a sequence of row/column rejection and significance tests per 112-row block
  (amplifier block) to flag bad pixels. Returns a boolean/integer mask with the
  same shape as the input (1=bad, 0=good).

Notes
- The original algorithm operated on a residual image (data - model), then
  standardized by per-row and per-column robust scatter. If you pass a raw
  image here, the standardization step still makes the tests reasonably
  scale-free, but results are optimal when the input is a residual.
"""
from __future__ import annotations

import numpy as np
from astropy.stats import mad_std
from astropy.stats import biweight_location as biweight
from scipy.interpolate import interp1d


def _robust_standardize(residual: np.ndarray) -> np.ndarray:
    """Robustly standardize a residual-like image by row and column scatter.

    This mirrors the scaling performed in remedy_get_wave_v2 prior to
    bad-pixel detection.
    """
    residual = np.array(residual, dtype=float)
    # Compute robust std per row, then for columns after row-normalization
    mstd_rows = mad_std(residual, ignore_nan=True, axis=1)
    # Avoid division by zero
    mstd_rows[~np.isfinite(mstd_rows) | (mstd_rows == 0)] = np.nanmedian(mstd_rows[np.isfinite(mstd_rows) & (mstd_rows > 0)]) or 1.0
    norm_by_rows = residual / mstd_rows[:, np.newaxis]
    mstd_cols = mad_std(norm_by_rows, ignore_nan=True, axis=0)
    mstd_cols[~np.isfinite(mstd_cols) | (mstd_cols == 0)] = np.nanmedian(mstd_cols[np.isfinite(mstd_cols) & (mstd_cols > 0)]) or 1.0
    mstd = mstd_rows[:, np.newaxis] * mstd_cols[np.newaxis, :]
    scaled = residual / mstd
    # Clip extreme values to stabilize subsequent logic (as in original code)
    scaled = np.clip(scaled, -50.0, 50.0)
    # Replace NaNs with 0 for downstream counting logic
    scaled[~np.isfinite(scaled)] = 0.0
    return scaled


def identify_bad_pixels(
    image: np.ndarray,
    amp_rows: int = 112,
    very_bad_thresh: float = 10.0,
) -> np.ndarray:
    """Identify bad pixels in an image using robust residual heuristics.

    The logic mirrors remedy_get_wave_v2's per-amplifier screening:
    - Standardize the image by robust row/column scatter.
    - Zero out pixels with |standardized residual| > very_bad_thresh.
    - For each amplifier block of ``amp_rows`` rows:
      * Zero out rows with >200 zero-valued pixels (already-failed rows).
      * Zero out columns with >30 zero-valued pixels (already-failed columns).
      * Perform a column-sum significance test: |sum| > 5 * sqrt(N_nonzero)
        marks the entire column bad in that block.

    Args:
        image: 2D array (Ny x Nx) detector image or residual.
        amp_rows: Number of rows per amplifier block (default 112 for VIRUS).
        very_bad_thresh: Threshold on standardized residual to zero immediately.

    Returns:
        np.ndarray: Integer mask array with 1=bad pixel, 0=good pixel.
    """
    if image is None:
        return None  # type: ignore

    residual = _robust_standardize(image)

    # Step 1: zero out very bad pixels
    very_bad = np.abs(residual) > float(very_bad_thresh)
    residual[very_bad] = 0.0

    ny, nx = residual.shape
    # Step 2: per-amplifier block logic
    for start in range(0, ny, amp_rows):
        end = min(start + amp_rows, ny)
        block = residual[start:end, :]

        # Initial bad row and bad column mask thresholds
        row_stat = (block == 0.0).sum(axis=1)
        block[row_stat > 200, :] = 0.0

        col_stat = (block == 0.0).sum(axis=0)
        block[:, col_stat > 30] = 0.0

        # Column significance test within the block
        num = np.sum(block != 0.0, axis=0)
        col_sum = np.sum(block, axis=0)
        with np.errstate(invalid='ignore'):
            col_test = np.abs(col_sum) > 5.0 * np.sqrt(num)
        block[:, col_test] = 0.0

        residual[start:end, :] = block

    # Build and return final bad-pixel mask: 1 where residual was zeroed
    mask = (residual == 0.0).astype(np.int32)
    return mask


def robust_poly6_tukey(m, y, x, max_iter=30, c=4.685, tol=1e-6):
    """
    Robustly fit y ≈ (c0 + c1 x + c2 x^2) * m + (d0 + d1 x + d2 x^2) via IRLS with Tukey weights.

    Args:
        m (array-like): predictor values (model flux), flattened.
        y (array-like): response values (image flux), flattened.
        x (array-like): pixel coordinate (normalized), same shape as m and y.
        max_iter (int): maximum IRLS iterations.
        c (float): Tukey bisquare tuning constant.
        tol (float): relative convergence tolerance for parameter updates.

    Returns:
        tuple[np.ndarray, np.ndarray]:
            - mult_coeffs (3,): coefficients [c0, c1, c2] for multiplicative poly.
            - add_coeffs (3,): coefficients [d0, d1, d2] for additive poly.
    """
    m = np.asarray(m, dtype=float)
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    mask = np.isfinite(m) & np.isfinite(y) & np.isfinite(x)
    m = m[mask]
    y = y[mask]
    x = x[mask]
    if m.size < 6 or np.nanstd(m) == 0.0:
        return np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0])

    # Design matrix: [m, m*x, m*x^2, 1, x, x^2]
    X = np.column_stack([m, m * x, m * (x ** 2), np.ones_like(x), x, x ** 2])

    # Initial unweighted LS
    try:
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    except Exception:
        return np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0])

    for _ in range(max_iter):
        yhat = X @ beta
        r = y - yhat
        med = np.median(r)
        mad = np.median(np.abs(r - med))
        s = 1.4826 * mad
        if not np.isfinite(s) or s <= 1e-12:
            s = np.nanstd(r)
            if not np.isfinite(s) or s <= 1e-12:
                break
        u = r / (c * s)
        w = (1.0 - u ** 2) ** 2
        w[np.abs(u) >= 1.0] = 0.0
        if np.all(w <= 0):
            break
        Wsqrt = np.sqrt(w)
        Xw = X * Wsqrt[:, None]
        yw = y * Wsqrt
        try:
            beta_new, *_ = np.linalg.lstsq(Xw, yw, rcond=None)
        except Exception:
            break
        if np.linalg.norm(beta_new - beta) < tol * max(1.0, np.linalg.norm(beta)):
            beta = beta_new
            break
        beta = beta_new

    mult = beta[:3]
    add = beta[3:]
    return mult, add

def build_model_spectra(twi_spectra, wave, good_solutions, nbins=2000):
    """
    Build a high-resolution master twilight template and per-row model.

    Args:
        twi_spectra (np.ndarray): Twilight spectra (Nrows, Npix).
        wave (np.ndarray): Wavelength solutions (Nrows, Npix).
        good_solutions (np.ndarray): Boolean (Nrows,) of rows to use.
        nbins (int): Number of wavelength bins for the template.

    Returns:
        tuple: (twi_interp, twi_model, twi_image, residual)
    """
    twi_image = twi_spectra.copy()
    wave_unravel = wave[good_solutions].ravel()
    ind_unravel = np.argsort(wave_unravel)
    twi_unravel = twi_spectra[good_solutions].ravel()
    # Split sorted wavelengths into bins and take robust biweight of flux in each bin
    twi_highres = [biweight(chunk, ignore_nan=True) for chunk in np.array_split(twi_unravel[ind_unravel], nbins)]
    wave_highres = [np.mean(w) for w in np.array_split(wave_unravel[ind_unravel], nbins)]
    twi_interp = interp1d(wave_highres, twi_highres, bounds_error=False, fill_value=np.nan, kind='quadratic')
    twi_model = np.zeros_like(twi_spectra)
    some_obj = twi_interp(wave[good_solutions])
    twi_model[good_solutions] = some_obj
    twi_image[~good_solutions] = 0.0
    residual = twi_image - twi_model
    return twi_interp, twi_model, twi_image, residual


def make_spectral_mask(msci_spectrum: np.ndarray,
                       msci_model_spectrum: np.ndarray,
                       amp_rows: int = 112,
                       very_bad_thresh: float = 10.0) -> np.ndarray:
    """
    Identify outlier spectral regions per amplifier using master SCI spectra and a model.

    This follows the residual standardization and block-masking logic used in
    remedy_get_wave_v2 (lines ~855–877), but works purely in spectral space.

    Args:
        msci_spectrum: Observed master science spectra (Nrows x Npix).
        msci_model_spectrum: Model spectra evaluated on each row's wavelength grid (Nrows x Npix).
        amp_rows: Rows per amplifier block (default 112).
        very_bad_thresh: Threshold on standardized residual to zero immediately (default 10).

    Returns:
        np.ndarray: Integer mask (Nrows x Npix), 1=bad spectral pixel, 0=good.
    """
    if msci_spectrum is None or msci_model_spectrum is None:
        return None  # type: ignore
    msci_image = np.array(msci_spectrum, dtype=float)
    msci_model = np.array(msci_model_spectrum, dtype=float)
    # Compute residuals
    residual = msci_image - msci_model
    # Determine good rows similar to good_fibers in original: rows where model is finite and non-zero
    good_rows = np.isfinite(msci_model).sum(axis=1) > (msci_model.shape[1] * 0.5)
    if not np.any(good_rows):
        # Fallback: use all rows
        good_rows = np.ones(msci_model.shape[0], dtype=bool)
    # Robust row/column standardization on good rows
    mstd_rows = mad_std(residual[good_rows], ignore_nan=True, axis=1)
    with np.errstate(invalid='ignore'):
        mstd_rows[~np.isfinite(mstd_rows) | (mstd_rows == 0)] = np.nanmedian(mstd_rows[np.isfinite(mstd_rows) & (mstd_rows > 0)]) if np.any(np.isfinite(mstd_rows) & (mstd_rows > 0)) else 1.0
    norm_by_rows = residual[good_rows] / mstd_rows[:, np.newaxis]
    mstd_cols = mad_std(norm_by_rows, ignore_nan=True, axis=0)
    with np.errstate(invalid='ignore'):
        mstd_cols[~np.isfinite(mstd_cols) | (mstd_cols == 0)] = np.nanmedian(mstd_cols[np.isfinite(mstd_cols) & (mstd_cols > 0)]) if np.any(np.isfinite(mstd_cols) & (mstd_cols > 0)) else 1.0
    mstd = mstd_rows[:, np.newaxis] * mstd_cols[np.newaxis, :]
    residual[good_rows] = residual[good_rows] / mstd
    residual = np.clip(residual, -50.0, 50.0)

    # Zero very bad residuals and NaNs
    very_bad = np.abs(residual) > float(very_bad_thresh)
    residual[very_bad] = 0.0
    residual[np.isnan(residual)] = 0.0

    nrows = residual.shape[0]
    for start in range(0, nrows, amp_rows):
        end = min(start + amp_rows, nrows)
        block = residual[start:end, :]
        # Initial bad row and column mask thresholds
        row_stat = (block == 0.0).sum(axis=1)
        block[row_stat > 200, :] = 0.0
        col_stat = (block == 0.0).sum(axis=0)
        block[:, col_stat > 30] = 0.0
        # Bad column significance test
        num = np.sum(block != 0.0, axis=0)
        col_sum = np.sum(block, axis=0)
        with np.errstate(invalid='ignore'):
            col_test = np.abs(col_sum) > 5.0 * np.sqrt(num)
        block[:, col_test] = 0.0
        residual[start:end, :] = block

    mask = (residual == 0.0).astype(np.int32)
    return mask
