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
import logging
from typing import Optional, Tuple
from itertools import combinations
from fiber_utils import find_peaks
from qa_utils import plot_ref_profile_quarters, plot_identify_arc_summary
from pathlib import Path

# Default pixel length assumption for VIRUS spectra, used in some helpers
PIXELS = 1032

# Reference arc line list (Hg, Cd, etc.) used for matching
LINE_LIST = [3610.508, 3650.153, 4046.565, 4358.335, 4678.149, 4799.912, 4916.068, 5085.822, 5460.750]


def suppress_close_peaks(peak_x, peak_h, min_sep=10, max_keep=25):
    peak_x = np.asarray(peak_x, dtype=float)
    peak_h = np.asarray(peak_h, dtype=float)
    order = np.argsort(peak_h)[::-1]
    keep_x = []
    keep_h = []
    for idx in order:
        x = peak_x[idx]
        h = peak_h[idx]
        if not any(np.abs(x - xk) <= min_sep for xk in keep_x):
            keep_x.append(x)
            keep_h.append(h)
        if len(keep_x) >= max_keep:
            break
    keep_x = np.array(keep_x)
    keep_h = np.array(keep_h)
    s = np.argsort(keep_x)
    return keep_x[s], keep_h[s]


def match_with_model(peak_x, line_list, coeff, x_tol=8.0):
    peak_x = np.asarray(peak_x, dtype=float)
    line_list = np.asarray(line_list, dtype=float)
    wave_at_peaks = np.polyval(coeff, peak_x)
    used = set()
    matches = []
    for w in line_list:
        dw = np.abs(wave_at_peaks - w)
        p = np.argmin(dw)
        # local slope of wavelength solution
        if len(coeff) >= 2:
            local_slope = np.polyval(np.polyder(coeff), peak_x[p])
        else:
            local_slope = coeff[0]
        local_slope = np.abs(local_slope) if np.isfinite(local_slope) else 0.0
        if local_slope < 1e-6:
            local_slope = 1.95
        x_resid = dw[p] / local_slope
        if x_resid <= x_tol and p not in used:
            matches.append({
                "x_obs": peak_x[p],
                "wave_ref": w,
                "wave_fit": wave_at_peaks[p],
                "wave_resid": wave_at_peaks[p] - w,
                "x_resid": x_resid,
                "peak_index": p,
            })
            used.add(p)
    return matches


def sigma_clip_matches(matches, nsig=3.0):
    if not matches:
        return matches
    import numpy as _np
    resid = _np.array([m["wave_resid"] for m in matches], dtype=float)
    med = _np.median(resid)
    mad = _np.median(_np.abs(resid - med))
    s = 1.4826 * mad if mad > 0 else _np.std(resid) if resid.size > 1 else 1.0
    if not _np.isfinite(s) or s <= 0:
        return matches
    good = _np.abs(resid - med) <= nsig * s
    return [m for m, g in zip(matches, good) if g]


def refit_from_matches(matches, degree=4):
    if not matches:
        # fallback linear
        return np.array([2.0, 0.0]), 0
    x_obs = np.array([m["x_obs"] for m in matches], dtype=float)
    wave_ref = np.array([m["wave_ref"] for m in matches], dtype=float)
    coeff = np.polyfit(x_obs, wave_ref, degree)
    return coeff, len(matches)


def identify_arc(
    peak_x,
    peak_h,
    line_list,
    min_sep=10,
    max_keep=12,
    anchor_tol_pix=12.0,
    Nbrightest=6,
    final_order=4,
):
    peak_x, peak_h = suppress_close_peaks(peak_x, peak_h, min_sep=min_sep, max_keep=max_keep)
    line_list = np.sort(np.asarray(line_list, dtype=float))
    top_peaks = np.argsort(peak_h)[::-1][:Nbrightest]
    top_peaks_x = np.sort(peak_x[top_peaks])
    best_rms = np.inf
    best_coeff = None
    for combo in combinations(line_list, Nbrightest):
        combo = np.sort(combo)
        coeff = np.polyfit(top_peaks_x, combo, 2)
        wave_pred = np.polyval(coeff, top_peaks_x)
        resid = np.empty_like(wave_pred)
        for i, w in enumerate(wave_pred):
            j = np.argmin(np.abs(line_list - w))
            resid[i] = line_list[j] - w
        rms = np.sqrt(np.mean(resid ** 2))
        if rms < best_rms:
            best_coeff = coeff
            best_rms = rms
    coeff = best_coeff
    matches = match_with_model(peak_x, line_list, coeff, x_tol=anchor_tol_pix)
    matches = sigma_clip_matches(matches, nsig=3.0)
    coeff_fit, _ = refit_from_matches(matches, degree=final_order)
    matches_final = match_with_model(peak_x, line_list, coeff_fit, x_tol=anchor_tol_pix)
    resid = np.array([m["wave_resid"] for m in matches_final], dtype=float)
    rms = np.sqrt(np.mean(resid ** 2))
    detected_x = np.array([m["x_obs"] for m in matches_final], dtype=float)
    best = {
        "coeff": coeff_fit,
        "matches": matches_final,
        "nmatch": len(matches_final),
        "rms": rms,
        "top_peaks_x": top_peaks_x,
        "peak_x_all": np.array(peak_x, dtype=float),
        "detected_x": detected_x,
    }
    return best, len(matches)


def get_wave_single(fiber_arc_spectrum, final_order=4, qa: Optional[dict] = None):
    """
    Derive a per-fiber wavelength solution from an arc spectrum.

    Detected arc peaks are matched to a fixed reference line list (``LINE_LIST``)
    and a polynomial wavelength model is fit in pixel space.

    Args:
        fiber_arc_spectrum (array-like): 1D arc spectrum for a single fiber
            with length equal to the number of spectral pixels (e.g., 1032).
        final_order (int, optional): Final polynomial degree to fit for the
            wavelength model. Defaults to 4.

    Returns:
        tuple[np.ndarray, float]:
            - ``wave``: 1D array giving the wavelength (same units as
              ``LINE_LIST``) for each pixel index.
            - ``rms``: RMS of residuals (in wavelength units) across matched
              lines.
    """
    x_indices = np.arange(len(fiber_arc_spectrum))
    fiber_arc_spectrum_clean = np.array(fiber_arc_spectrum, dtype=float).copy()
    fiber_arc_spectrum_clean[~np.isfinite(fiber_arc_spectrum_clean)] = 0.0
    peak_loc, peaks = find_peaks(fiber_arc_spectrum_clean, thresh=1)
    best, _ = identify_arc(
        peak_loc,
        peaks,
        LINE_LIST,
        min_sep=10,
        max_keep=15,
        anchor_tol_pix=12.0,
        final_order=final_order,
    )
    wave = np.polyval(best["coeff"], x_indices)
    return wave, best["rms"]


def get_wave(spec: np.ndarray,
             trace: np.ndarray,
             T_array: Optional[np.ndarray] = None,
             res_lim: float = 1.0,
             order: int = 4,
             qa: Optional[dict] = None) -> Tuple[Optional[np.ndarray], Optional[Path], Optional[np.ndarray]]:
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
        Tuple (wave, ref_img, rms_rows):
        - wave: 2D array (Nrows x Npix) with wavelength at each pixel, or None if insufficient good rows are found.
        - ref_img: Path to the QA ref_profile_quarters image produced (if any), otherwise None.
        - rms_rows: 1D array (Nrows,) with per-row RMS residuals from seed fits (0 for rows not attempted), or None if unavailable.
    """
    if spec is None or trace is None:
        return None, None, None

    nrows, npix = spec.shape
    w_seed = np.zeros_like(spec, dtype=float)
    rms = np.zeros((nrows,), dtype=float)

    # Choose a sparse set of rows to attempt per-fiber solutions (similar to prior code)
    try_rows = np.hstack([np.arange(2, nrows, 8), nrows - 3])
    try_rows = np.unique(np.clip(try_rows, 0, nrows - 1))

    # Solve per selected row using robust single-fiber routine
    qa_plotted = False
    ref_img: Optional[Path] = None
    for j in try_rows:
        try:
            # Robustify by median-combining a small window of rows, as before
            j0 = max(0, j - 2)
            j1 = min(nrows, j + 3)
            S = np.nanmedian(spec[j0:j1], axis=0)
            wave_j, res = get_wave_single(S, final_order=order)
            w_seed[j] = wave_j
            rms[j] = res

            # Optionally create a QA plot for one representative fiber arc spectrum
            if (not qa_plotted) and (qa is not None):
                try:
                    out_folder = qa.get("out_folder") if isinstance(qa, dict) else None
                    if out_folder is not None:
                        # Detect peaks and run identification to gather candidates and matches
                        peak_loc, peaks = find_peaks(np.asarray(S, dtype=float), thresh=1)
                        best, _ = identify_arc(
                            peak_loc,
                            peaks,
                            LINE_LIST,
                            min_sep=10,
                            max_keep=15,
                            anchor_tol_pix=12.0,
                            final_order=order,
                        )
                        # Compose filename to indicate which row window was used
                        fname = qa.get("ref_plot_name", f"identify_arc_summary_row{int(j)}.png") if isinstance(qa, dict) else f"identify_arc_summary_row{int(j)}.png"
                        ref_img = plot_identify_arc_summary(
                            out_folder=out_folder,
                            ref_profile=S,
                            best=best,
                            filename=fname,
                        )
                        # Mark that we've produced one plot
                        qa_plotted = True
                except Exception as e:
                    logging.getLogger('build_master_bias').warning(f"[QA] Error producing identify_arc_summary for row {j}: {e}")
                    # Do not let QA plotting affect core wavelength solution
                    pass
        except Exception:
            # Leave defaults (zeros), continue
            pass

    good = (rms > 0) & (rms < res_lim)
    if np.count_nonzero(good) < 7:
        return None, ref_img, rms

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
                return None, ref_img, rms

    return wave, ref_img, rms
