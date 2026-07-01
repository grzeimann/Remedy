# -*- coding: utf-8 -*-
"""
QA utilities for per-amplifier summary pages and basic metrics.

This module provides:
- summarize_amp_metrics: derive scalar QA metrics from calibration arrays.
- plot_ref_profile_quarters: diagnostic plot similar to antigen's helper.
- save_amp_qa_page: compose a one-page PNG summary with metrics and plots.

Design goals:
- Be resilient to missing/NaN inputs; return NaN or 0 with clear labels.
- No side effects beyond writing images/files to the specified out folder.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

# Matplotlib lazy import to allow headless environments
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt


def _mad(a: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    med = np.nanmedian(a)
    return 1.4826 * np.nanmedian(np.abs(a - med))


def summarize_amp_metrics(
    amp_id: str,
    masterbias: np.ndarray,
    masterdark: np.ndarray,
    masterflt: np.ndarray,
    mastercmp: Optional[np.ndarray],
    mastersci: Optional[np.ndarray],
    readnoise: float,
    wave: np.ndarray,
    trace: np.ndarray,
    lampspec: Optional[np.ndarray],
    maskspec: Optional[np.ndarray],
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Compute scalar QA metrics for an amplifier.

    Returns a dict with numeric values and small counts; values may be NaN when
    inputs are insufficient.
    """
    metrics: Dict[str, Any] = {}
    metrics["amp"] = amp_id
    # Readnoise
    try:
        metrics["readnoise_e"] = float(readnoise)
    except Exception:
        metrics["readnoise_e"] = np.nan

    # Bias/Dark/Flat central tendency and scatter
    for name, arr in (
        ("bias", masterbias),
        ("dark", masterdark),
        ("flat", masterflt),
    ):
        arrf = np.asarray(arr, dtype=float) if arr is not None else np.array([np.nan])
        metrics[f"{name}_median"] = float(np.nanmedian(arrf))
        metrics[f"{name}_mad"] = float(_mad(arrf))

    # Masked spectral pixels percent
    if maskspec is not None:
        ms = np.asarray(maskspec)
        total = ms.size if ms.size else 1
        frac = float(np.nansum(ms > 0) / total)
        metrics["masked_spectral_frac"] = frac
    else:
        metrics["masked_spectral_frac"] = np.nan

    # Bad wavelength pixels percent (non-finite or zero)
    w = np.asarray(wave, dtype=float)
    bad_w = ~np.isfinite(w) | (w == 0)
    total_w = bad_w.size if bad_w.size else 1
    metrics["bad_wavelength_frac"] = float(np.sum(bad_w) / total_w)

    # Failed traces: rows with too few finite entries
    tr = np.asarray(trace, dtype=float)
    row_good = np.isfinite(tr).sum(axis=1) > (0.5 * tr.shape[1]) if tr.ndim == 2 else np.array([])
    metrics["failed_traces"] = int((~row_good).sum()) if row_good.size else int(112)

    # Failed wavelength fibers: rows with insufficient finite wavelengths
    if w.ndim == 2 and w.size:
        w_row_good = np.isfinite(w).sum(axis=1) > (0.8 * w.shape[1])
        metrics["failed_wave_fibers"] = int((~w_row_good).sum())
    else:
        metrics["failed_wave_fibers"] = int(112)

    # Median arc RMS (placeholder): if residuals of wavelength fit aren't available,
    # we estimate a proxy as robust std of high-pass-filtered lampspec across pixels,
    # then take the median over rows. If lampspec missing, set NaN.
    arc_rms = np.nan
    if lampspec is not None:
        try:
            ls = np.asarray(lampspec, dtype=float)
            # High-pass via median filter approximation using rolling window if available
            # Fallback: subtract per-row median
            if ls.ndim == 2 and ls.shape[1] > 5:
                med = np.nanmedian(ls, axis=1)[:, None]
                hp = ls - med
                row_rms = 1.4826 * np.nanmedian(np.abs(hp), axis=1)
                arc_rms = float(np.nanmedian(row_rms))
        except Exception:
            arc_rms = np.nan
    metrics["median_arc_rms"] = arc_rms

    # Percent of masked detector pixels (if mastersci available)
    if mastersci is not None:
        sc = np.asarray(mastersci, dtype=float)
        # Placeholder: not a real detector mask; leave NaN to avoid confusion
        metrics["masked_detector_frac"] = np.nan
    else:
        metrics["masked_detector_frac"] = np.nan

    if extra:
        metrics.update(extra)
    return metrics


def plot_ref_profile_quarters(
    out_folder: Path,
    ref_profile: Optional[np.ndarray],
    arc_pixel_guesses: Optional[np.ndarray] = None,
    ref_detected: Optional[np.ndarray] = None,
    filename: str = "ref_profile_quarters.png",
) -> Optional[Path]:
    """Four-row diagnostic: ref_profile split into X-quarters with guesses vs detections.

    Standalone version of the provided plotting helper. Inputs may be None; in
    that case, the function returns without creating a file.
    """
    if ref_profile is None:
        return None
    prof = np.asarray(ref_profile, dtype=float).ravel()
    n = int(prof.size)
    if n == 0:
        return None
    x = np.arange(n)
    idx_edges = np.linspace(0, n, 5, dtype=int)
    qs = [(idx_edges[i], idx_edges[i + 1]) for i in range(4)]

    guesses = np.asarray(arc_pixel_guesses, dtype=float).ravel() if arc_pixel_guesses is not None else np.array([])
    det = np.asarray(ref_detected, dtype=float).ravel() if ref_detected is not None else np.array([])
    if guesses.size == 0:
        guesses = np.full(0, np.nan)
    if det.size == 0:
        det = np.full(0, np.nan)
    missing = ~np.isfinite(det) & np.isfinite(guesses)

    finite = prof[np.isfinite(prof)]
    if finite.size:
        lo, hi = np.percentile(finite, [1, 99])
        if hi <= lo:
            hi = lo + 1e-6
    else:
        lo, hi = 0.0, 1.0

    fig, axes = plt.subplots(4, 1, figsize=(10, 6))
    for r, (lo_i, hi_i) in enumerate(qs):
        ax = axes[r]
        slice_y = prof[lo_i:hi_i]
        finite_slice = slice_y[np.isfinite(slice_y)]
        if finite_slice.size:
            lo_p, hi_p = np.percentile(finite_slice, [1, 97])
            if hi_p <= lo_p:
                hi_p = lo_p + 1e-6
        else:
            lo_p, hi_p = 0.0, 1.0

        ax.plot(x[lo_i:hi_i], slice_y, color='0.2', lw=0.8)
        m_q = np.isfinite(guesses) & (guesses >= lo_i) & (guesses < hi_i)
        if np.any(m_q):
            ax.vlines(guesses[m_q], lo_p, hi_p, colors='tab:blue', linestyles='--', alpha=0.5, lw=1.0, label='guess')
        m_d = np.isfinite(det) & (det >= lo_i) & (det < hi_i)
        if np.any(m_d):
            ax.vlines(det[m_d], lo_p, hi_p, colors='tab:green', linestyles='-', alpha=0.7, lw=1.2, label='detected')
        m_miss = missing & (guesses >= lo_i) & (guesses < hi_i)
        if np.any(m_miss):
            gx = guesses[m_miss]
            ax.plot(gx, np.full_like(gx, hi_p), 'x', color='tab:red', ms=6, mew=1.2, label='missing')
        ax.set_xlim(lo_i, hi_i - 1)
        ax.set_ylim(lo_p, hi_p)
        ax.grid(alpha=0.15)
        if r == 0:
            ax.legend(ncol=3, frameon=False, loc='upper right')
    axes[-1].set_xlabel('Pixel (dispersion axis)')
    fig.supylabel('Ref profile (arb)')
    fig.tight_layout(rect=(0.02, 0.02, 1, 1))
    out_folder = Path(out_folder)
    out_folder.mkdir(parents=True, exist_ok=True)
    out = out_folder / filename
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def save_amp_qa_page(
    out_folder: Path,
    amp_id: str,
    metrics: Dict[str, Any],
    ref_profile_img: Optional[Path] = None,
) -> Path:
    """Create a compact PNG QA page with key metrics and an optional image.

    Also writes a JSON sidecar with the metrics.
    """
    out_folder = Path(out_folder)
    out_folder.mkdir(parents=True, exist_ok=True)

    # Prepare figure
    fig = plt.figure(figsize=(10, 6), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 2])

    # Title and metrics panel
    ax0 = fig.add_subplot(gs[0, :])
    ax0.axis('off')
    title = f"QA Summary — Amp: {metrics.get('amp', amp_id)}"
    ax0.text(0.01, 0.85, title, fontsize=14, fontweight='bold')

    # Build multiline metrics text
    lines = [
        f"readnoise: {metrics.get('readnoise_e', np.nan):.2f} e-",
        f"bias median/MAD: {metrics.get('bias_median', np.nan):.2f} / {metrics.get('bias_mad', np.nan):.2f}",
        f"dark median/MAD: {metrics.get('dark_median', np.nan):.2f} / {metrics.get('dark_mad', np.nan):.2f}",
        f"flat median/MAD: {metrics.get('flat_median', np.nan):.2f} / {metrics.get('flat_mad', np.nan):.2f}",
        f"masked spectral %: {100.0 * metrics.get('masked_spectral_frac', np.nan):.2f}",
        f"bad wavelength %: {100.0 * metrics.get('bad_wavelength_frac', np.nan):.2f}",
        f"failed traces: {metrics.get('failed_traces', 0)}",
        f"failed wavelength fibers: {metrics.get('failed_wave_fibers', 0)}",
        f"median arc RMS: {metrics.get('median_arc_rms', np.nan):.3f}",
    ]
    ax0.text(0.01, 0.10, "\n".join(lines), family='monospace', fontsize=11, va='bottom')

    # Image panel(s)
    ax_img = fig.add_subplot(gs[1, :])
    ax_img.axis('off')
    if ref_profile_img and Path(ref_profile_img).exists():
        img = plt.imread(ref_profile_img)
        ax_img.imshow(img)
        ax_img.set_title('Ref profile quarters')
    else:
        ax_img.text(0.5, 0.5, 'No diagnostic image available', ha='center', va='center', color='0.5')

    # Save
    png_path = out_folder / f"QA_{amp_id}.png"
    fig.savefig(png_path, dpi=150)
    plt.close(fig)

    # Write JSON sidecar
    sidecar = out_folder / f"QA_{amp_id}.json"
    try:
        with open(sidecar, 'w') as f:
            json.dump(metrics, f, indent=2, default=lambda o: float(o) if isinstance(o, (np.floating,)) else o)
    except Exception:
        pass

    return png_path
