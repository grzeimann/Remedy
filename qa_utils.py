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
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch


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
    arc_rms_array: Optional[np.ndarray] = None,
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

    # Median arc RMS: prefer provided per-row residuals from wavelength fit.
    arc_rms = np.nan
    if arc_rms_array is not None:
        try:
            arr = np.asarray(arc_rms_array, dtype=float)
            fin = arr[np.isfinite(arr) & (arr > 0)]
            if fin.size:
                arc_rms = float(np.nanmedian(fin))
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

def plot_identify_arc_summary(
    out_folder: Path,
    ref_profile: Optional[np.ndarray],
    best: Optional[dict],
    filename: str = "identify_arc_summary.png",
) -> Optional[Path]:
    """Diagnostic for arc identification using identify_arc outputs.

    Panels:
    - Top: spectrum with candidate peaks (blue dashed), matched peaks (green), and unmatched candidates (red x).
    - Bottom: residuals (wave_fit - wave_ref) vs wave_ref for matched peaks, annotated with RMS and nmatch.

    Args:
        out_folder: directory to write the image.
        ref_profile: reference 1D spectrum used to detect peaks (S).
        best: dict returned by identify_arc containing keys like
              'peak_x_all', 'detected_x', 'matches', 'rms', 'nmatch'.
        filename: output filename.
    """
    if ref_profile is None or best is None:
        return None

    prof = np.asarray(ref_profile, dtype=float).ravel()
    n = int(prof.size)
    if n == 0:
        return None
    x = np.arange(n)

    guesses = np.asarray(best.get("peak_x_all", []), dtype=float).ravel()
    det = np.asarray(best.get("detected_x", []), dtype=float).ravel()

    # Determine unmatched candidate peaks
    tol_pix = 2.5
    if guesses.size == 0:
        guesses = np.full(0, np.nan)
    if det.size == 0:
        det = np.full(0, np.nan)
    if guesses.size > 0 and det.size > 0 and np.isfinite(det).any():
        gfin = np.isfinite(guesses)
        dfin = np.isfinite(det)
        if np.any(gfin) and np.any(dfin):
            dmin = np.full(guesses.shape, np.inf, dtype=float)
            dmin[gfin] = np.min(np.abs(guesses[gfin, None] - det[dfin][None, :]), axis=1)
            matched_mask = dmin <= tol_pix
        else:
            matched_mask = np.zeros(guesses.shape, dtype=bool)
    else:
        matched_mask = np.zeros(guesses.shape, dtype=bool)
    missing = (~matched_mask) & np.isfinite(guesses)

    # Build figure
    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(11, 6.5), gridspec_kw=dict(height_ratios=[2, 1]))

    # Top panel: spectrum and lines
    finite_prof = prof[np.isfinite(prof)]
    if finite_prof.size:
        lo, hi = np.percentile(finite_prof, [1, 99])
        if hi <= lo:
            hi = lo + 1e-6
    else:
        lo, hi = 0.0, 1.0
    ax_top.plot(x, prof, color='0.2', lw=0.8)
    if np.isfinite(guesses).any():
        ax_top.vlines(guesses[np.isfinite(guesses)], lo, hi, colors='tab:blue', linestyles='--', alpha=0.5, lw=1.0, label='candidates')
    if np.isfinite(det).any():
        ax_top.vlines(det[np.isfinite(det)], lo, hi, colors='tab:green', linestyles='-', alpha=0.7, lw=1.2, label='matched')
    if np.any(missing):
        gx = guesses[missing]
        ax_top.plot(gx, np.full_like(gx, hi), 'x', color='tab:red', ms=6, mew=1.2, label='missing')
    ax_top.set_xlim(0, n - 1)
    ax_top.set_ylim(lo, hi)
    ax_top.set_ylabel('Ref profile (arb)')
    ax_top.legend(ncol=3, frameon=False, loc='upper right')
    info = f"nmatch={int(best.get('nmatch', 0))}, rms={float(best.get('rms', np.nan)):.3f}"
    ax_top.text(0.01, 0.95, info, transform=ax_top.transAxes, ha='left', va='top', fontsize=10, bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))

    # Bottom panel: residuals vs. reference wavelength
    matches = best.get('matches', []) or []
    try:
        wave_ref = np.array([m.get('wave_ref', np.nan) for m in matches], dtype=float)
        wave_resid = np.array([m.get('wave_resid', np.nan) for m in matches], dtype=float)
    except Exception:
        wave_ref = np.array([], dtype=float)
        wave_resid = np.array([], dtype=float)
    mfin = np.isfinite(wave_ref) & np.isfinite(wave_resid)
    if np.any(mfin):
        ax_bot.axhline(0.0, color='0.5', lw=1.0)
        ax_bot.scatter(wave_ref[mfin], wave_resid[mfin], s=18, c='tab:purple', alpha=0.8)
        rms = float(best.get('rms', np.nan))
        ax_bot.text(0.02, 0.90, f"RMS={rms:.3f}", transform=ax_bot.transAxes, ha='left', va='top', fontsize=10)
    else:
        ax_bot.text(0.5, 0.5, 'No matched lines', ha='center', va='center', color='0.5')
    ax_bot.set_xlabel('Reference wavelength')
    ax_bot.set_ylabel('Residual (fit - ref)')
    ax_bot.grid(alpha=0.2)

    fig.tight_layout(rect=(0.02, 0.02, 1, 1))
    out_folder = Path(out_folder)
    out_folder.mkdir(parents=True, exist_ok=True)
    out = out_folder / filename
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def plot_specmask_overlay(
    out_folder: Path,
    spec2d: Optional[np.ndarray],
    maskspec: Optional[np.ndarray],
    filename: str = "specmask_overlay.png",
    title: str = "Spectral mask overlay",
) -> Optional[Path]:
    """Render a diagnostic image of the 2D spectra with a mask overlay.

    The spectrum is shown with a robust grayscale colormap, and pixels flagged
    by the spectral mask are overlaid in semi-transparent red so masked regions
    are visually obvious.

    Args:
        out_folder: Directory to save the image.
        spec2d: 2D array (Nrows x Npix) of extracted spectra.
        maskspec: 2D mask array (same shape). Nonzero/True indicates masked.
        filename: Output file name.
        title: Title for the plot.

    Returns:
        Path to the saved PNG, or None if inputs are invalid.
    """
    if spec2d is None or maskspec is None:
        return None
    A = np.asarray(spec2d, dtype=float)
    M = np.asarray(maskspec)
    if A.ndim != 2 or M.ndim != 2 or A.shape != M.shape:
        return None
    nrows, npix = A.shape
    if nrows == 0 or npix == 0:
        return None

    # Robust display range ignoring NaNs and infs
    finite = np.isfinite(A)
    if not np.any(finite):
        vmin, vmax = 0.0, 1.0
    else:
        vals = A[finite]
        lo, hi = np.percentile(vals, [5, 99.5])
        if not np.isfinite(lo):
            lo = np.nanmin(vals)
        if not np.isfinite(hi):
            hi = np.nanmax(vals)
        if hi <= lo:
            hi = lo + 1e-6
        vmin, vmax = float(lo), float(hi)

    fig, ax = plt.subplots(1, 1, figsize=(10, 4.2))
    ax.imshow(A, aspect='auto', origin='lower', cmap='gray', vmin=vmin, vmax=vmax)

    # Build a transparent/solid red colormap for mask overlay
    mask_bool = np.array(M > 0)
    overlay = np.zeros_like(mask_bool, dtype=float)
    overlay[mask_bool] = 1.0
    cmap_overlay = ListedColormap([
        (1, 1, 1, 0.0),   # fully transparent for unmasked
        (1, 0.0, 0.0, 0.45),  # semi-transparent red for masked
    ])
    ax.imshow(overlay, aspect='auto', origin='lower', cmap=cmap_overlay, interpolation='nearest')

    ax.set_xlabel('Pixel')
    ax.set_ylabel('Fiber/row')
    ax.set_title(title)

    # Add simple legend proxies
    handles = [
        Patch(facecolor='gray', edgecolor='none', alpha=0.6, label='spectrum'),
        Patch(facecolor=(1, 0, 0, 0.45), edgecolor='none', label='masked'),
    ]
    ax.legend(handles=handles, loc='upper right', frameon=False)

    fig.tight_layout()
    out_folder = Path(out_folder)
    out_folder.mkdir(parents=True, exist_ok=True)
    out = out_folder / filename
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def plot_trace_overlay(
    out_folder: Path,
    masterflt: Optional[np.ndarray],
    trace: Optional[np.ndarray],
    filename: str = "trace_overlay.png",
    title: str = "Trace overlay on master flat",
    step: int = 8,
) -> Optional[Path]:
    """Render master flat with overlaid fiber traces.

    Overlays a subset of fiber traces as polylines on top of the flat-field
    image to visually confirm the trace solution aligns with the illumination.

    Args:
        out_folder: Directory to save the image.
        masterflt: 2D flat image (Ny x Nx), typically 1032x1032.
        trace: 2D array (Nfibers x Npix) giving detector row (y) vs spectral pixel (x).
        filename: Output filename.
        title: Title for the plot.
        step: Plot every Nth fiber to reduce clutter (default 8).

    Returns:
        Path to the saved PNG, or None if inputs are invalid.
    """
    if masterflt is None or trace is None:
        return None
    A = np.asarray(masterflt, dtype=float)
    Tr = np.asarray(trace, dtype=float)
    if A.ndim != 2 or Tr.ndim != 2:
        return None
    Ny, Nx = A.shape
    # Expect trace shape (Nfibers, Npix) with Npix ~ Nx
    Nfib, Npix = Tr.shape
    if Nx <= 1 or Ny <= 1 or Npix <= 1:
        return None

    # Robust display scaling
    finite = np.isfinite(A)
    if not np.any(finite):
        vmin, vmax = 0.0, 1.0
    else:
        vals = A[finite]
        lo, hi = np.percentile(vals, [5, 99.5])
        if not np.isfinite(lo):
            lo = np.nanmin(vals)
        if not np.isfinite(hi):
            hi = np.nanmax(vals)
        if hi <= lo:
            hi = lo + 1e-6
        vmin, vmax = float(lo), float(hi)

    fig, ax = plt.subplots(1, 1, figsize=(10.5, 4.2))
    ax.imshow(A, aspect='auto', origin='lower', cmap='gray', vmin=vmin, vmax=vmax)

    x = np.arange(Npix, dtype=float)
    # Choose a subset of fibers evenly spaced
    jj = np.arange(0, Nfib, max(1, int(step)))
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(jj)))
    for k, j in enumerate(jj):
        y = Tr[j]
        if not np.isfinite(y).any():
            continue
        # mask non-finite
        m = np.isfinite(y)
        if np.count_nonzero(m) < 5:
            continue
        ax.plot(x[m], y[m], '-', lw=0.8, alpha=0.8, color=colors[k])

    ax.set_xlim(0, Nx - 1)
    ax.set_ylim(0, Ny - 1)
    ax.set_xlabel('Spectral pixel (x)')
    ax.set_ylabel('Detector row (y)')
    ax.set_title(title)

    fig.tight_layout()
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
    specmask_img: Optional[Path] = None,
    trace_img: Optional[Path] = None,
) -> Path:
    """Create a compact PNG QA page with key metrics and optional images.

    Also writes a JSON sidecar with the metrics.
    """
    out_folder = Path(out_folder)
    out_folder.mkdir(parents=True, exist_ok=True)

    # Determine which images exist and build a list of (path, title)
    items = []
    if ref_profile_img and Path(ref_profile_img).exists():
        items.append((Path(ref_profile_img), 'Arc identification summary'))
    if specmask_img and Path(specmask_img).exists():
        items.append((Path(specmask_img), 'Spectral mask overlay'))
    if trace_img and Path(trace_img).exists():
        items.append((Path(trace_img), 'Trace overlay on master flat'))

    # Prepare figure with dynamic columns based on available images
    ncols = max(1, len(items))
    fig = plt.figure(figsize=(11, 6.5), constrained_layout=True)
    gs = fig.add_gridspec(2, ncols, height_ratios=[1, 2])

    # Title and metrics panel
    ax0 = fig.add_subplot(gs[0, :])
    ax0.axis('off')

    # Build multiline metrics text
    lines = [
        f"QA Summary — Amp: {metrics.get('amp', amp_id)}",
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
    if ncols == 1:
        ax_img = fig.add_subplot(gs[1, :])
        ax_img.axis('off')
        if items:
            try:
                img = plt.imread(items[0][0])
                ax_img.imshow(img)
                ax_img.set_title(items[0][1])
            except Exception:
                ax_img.text(0.5, 0.5, 'No diagnostic image available', ha='center', va='center', color='0.5')
        else:
            ax_img.text(0.5, 0.5, 'No diagnostic image available', ha='center', va='center', color='0.5')
    else:
        for i in range(ncols):
            ax = fig.add_subplot(gs[1, i])
            ax.axis('off')
            if i < len(items):
                path_i, title_i = items[i]
                try:
                    img = plt.imread(path_i)
                    ax.imshow(img)
                    ax.set_title(title_i)
                except Exception:
                    ax.text(0.5, 0.5, 'Image unavailable', ha='center', va='center', color='0.5')
            else:
                ax.text(0.5, 0.5, 'No image', ha='center', va='center', color='0.5')

    # Save
    png_path = out_folder / f"QA_{amp_id}.png"
    fig.savefig(png_path, dpi=150)
    # Write JSON sidecar
    sidecar = out_folder / f"QA_{amp_id}.json"
    try:
        with open(sidecar, 'w') as f:
            json.dump(metrics, f, indent=2, default=lambda o: float(o) if isinstance(o, (np.floating,)) else o)
    except Exception:
        pass

    return png_path
