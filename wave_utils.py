# -*- coding: utf-8 -*-
"""
Utility functions for deriving wavelength solutions.

This module provides a wrapper around the robust per-fiber wavelength
solution implemented in remedy_get_wave_v2.get_wave_single, and expands it
into a full 2D wavelength map across all fibers/rows following the approach
used previously in fiber_utils.get_wave.

The primary entry point is get_wave(spec, trace, ...), which estimates a
wavelength vector for selected rows using per-fiber arc spectra and then fits
smooth polynomials across rows and along dispersion to deliver a complete
(Nrows x Npix) wavelength array.
"""
from __future__ import annotations

import numpy as np
from typing import Optional

# We rely on the robust single-spectrum solver from remedy_get_wave_v2
from remedy_get_wave_v2 import get_wave_single as _get_wave_single


def get_wave(spec: np.ndarray,
             trace: np.ndarray,
             T_array: Optional[np.ndarray] = None,
             res_lim: float = 1.0,
             order: int = 4) -> Optional[np.ndarray]:
    """
    Build a 2D wavelength solution using robust per-row arc fitting.

    Args:
        spec: 2D array (Nrows x Npix) of extracted arc spectra.
        trace: 2D array (Nrows x Npix) of trace positions (same shape as spec).
        T_array: Unused legacy argument kept for compatibility. Ignored.
        res_lim: Maximum acceptable RMS (wavelength units) for a row to be
            considered a reliable solution seed.
        order: Polynomial order used for cross-row and along-dispersion fits.

    Returns:
        2D array of shape (Nrows x Npix) with wavelength at each pixel, or
        None if insufficient good rows are found.
    """
    if spec is None or trace is None:
        return None

    nrows, npix = spec.shape
    w_seed = np.zeros_like(spec, dtype=float)
    rms = np.zeros((nrows,), dtype=float)

    # Choose a sparse set of rows to attempt per-fiber solutions (similar to prior code)
    try_rows = np.hstack([np.arange(2, nrows, 8), nrows - 3])
    try_rows = np.unique(np.clip(try_rows, 0, nrows - 1))

    # Solve per selected row using robust single-fiber routine
    for j in try_rows:
        try:
            # Robustify by median-combining a small window of rows, as before
            j0 = max(0, j - 2)
            j1 = min(nrows, j + 3)
            S = np.nanmedian(spec[j0:j1], axis=0)
            wave_j, res = _get_wave_single(S, final_order=order)
            w_seed[j] = wave_j
            rms[j] = res
        except Exception:
            # Leave defaults (zeros), continue
            pass

    good = (rms > 0) & (rms < res_lim)
    if np.count_nonzero(good) < 7:
        return None

    # Initialize final wavelength array
    wave = np.zeros_like(spec, dtype=float)

    # Fit across rows at a sparse set of columns, then evaluate on all rows
    # This mirrors the strategy used in fiber_utils.get_wave
    xi = np.hstack([np.arange(0, npix, 24), npix - 1])
    xi = np.unique(np.clip(xi, 0, npix - 1))

    for i in xi:
        x = trace[:, i]
        y = w_seed[:, i]
        # Use only good rows for cross-row fit
        coeff = np.polyfit(x[good], y[good], order)
        wave[:, i] = np.polyval(coeff, x)

    # Smooth along dispersion for each row by fitting through valid columns
    xpix = np.arange(npix)
    for j in range(nrows):
        sel = wave[j] > 0
        if np.count_nonzero(sel) >= (order + 1):
            coeff = np.polyfit(xpix[sel], wave[j][sel], order)
            wave[j] = np.polyval(coeff, xpix)
        else:
            # Fallback: copy nearest good neighbor row if available
            # Find nearest row with a solution
            if np.count_nonzero(good) > 0:
                jj = int(np.argmin(np.abs(np.where(good)[0] - j)))
                wave[j] = wave[np.where(good)[0][jj]]
            else:
                return None

    return wave
