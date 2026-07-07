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
    fibernorm_bands: Optional[Dict[str, Dict[str, float]]] = None,
    trace_rms_per_fiber: Optional[np.ndarray] = None,
    wave_rms_per_fiber: Optional[np.ndarray] = None,
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
    # Flag amps that likely have all-zero frames (e.g., readnoise computed as 0)
    try:
        rn = metrics["readnoise_e"]
        metrics["zero_frames_suspected"] = bool(np.isfinite(rn) and rn <= 1e-6)
    except Exception:
        metrics["zero_frames_suspected"] = False

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

    # Trace and wavelength failure counts based on per-fiber RMS when available
    # Trace failures: count fibers with RMS > 0.5 (fail), warn at > 0.2
    tr = np.asarray(trace, dtype=float)
    fail_thr = 0.5
    warn_thr = 0.2
    if trace_rms_per_fiber is not None:
        try:
            trr = np.asarray(trace_rms_per_fiber, dtype=float).ravel()
            trr_fin = trr[np.isfinite(trr)]
            metrics["trace_rms_median"] = float(np.nanmedian(trr_fin)) if trr_fin.size else np.nan
            metrics["trace_rms_warn_count"] = int(np.sum(trr_fin > warn_thr)) if trr_fin.size else 112
            metrics["failed_traces"] = int(np.sum(trr_fin > fail_thr)) if trr_fin.size else 112
        except Exception:
            # Fallback to finite fraction method if RMS not usable
            row_good = np.isfinite(tr).sum(axis=1) > (0.5 * tr.shape[1]) if tr.ndim == 2 else np.array([])
            metrics["failed_traces"] = int((~row_good).sum()) if row_good.size else int(112)
            metrics["trace_rms_median"] = np.nan
            metrics["trace_rms_warn_count"] = 112
    else:
        # Legacy behavior fallback
        row_good = np.isfinite(tr).sum(axis=1) > (0.5 * tr.shape[1]) if tr.ndim == 2 else np.array([])
        metrics["failed_traces"] = int((~row_good).sum()) if row_good.size else int(112)
        metrics["trace_rms_median"] = np.nan
        metrics["trace_rms_warn_count"] = 112

    # Wavelength failures: prefer per-fiber RMS array (wave_rms_per_fiber),
    # else use arc_rms_array if provided, else fallback to finite count.
    if wave_rms_per_fiber is not None or arc_rms_array is not None:
        try:
            wr = wave_rms_per_fiber if wave_rms_per_fiber is not None else arc_rms_array
            wr = np.asarray(wr, dtype=float).ravel()
            wr_fin = wr[np.isfinite(wr) & (wr > 0)]
            metrics["wave_rms_median"] = float(np.nanmedian(wr_fin)) if wr_fin.size else np.nan
            metrics["wave_rms_warn_count"] = int(np.sum(wr_fin > warn_thr)) if wr_fin.size else 112
            metrics["failed_wave_fibers"] = int(np.sum(wr_fin > fail_thr)) if wr_fin.size else 112
        except Exception:
            if w.ndim == 2 and w.size:
                w_row_good = np.isfinite(w).sum(axis=1) > (0.8 * w.shape[1])
                metrics["failed_wave_fibers"] = int((~w_row_good).sum())
            else:
                metrics["failed_wave_fibers"] = int(112)
            metrics["wave_rms_median"] = np.nan
            metrics["wave_rms_warn_count"] = 112
    else:
        if w.ndim == 2 and w.size:
            w_row_good = np.isfinite(w).sum(axis=1) > (0.8 * w.shape[1])
            metrics["failed_wave_fibers"] = int((~w_row_good).sum())
        else:
            metrics["failed_wave_fibers"] = int(112)
        metrics["wave_rms_median"] = np.nan
        metrics["wave_rms_warn_count"] = 112

    # Median arc RMS (for display): prefer provided per-row residuals.
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

    # Include fibernorm band residual summaries, if provided
    if fibernorm_bands:
        for band_name, vals in fibernorm_bands.items():
            # sanitize key
            key = band_name.lower().replace(' ', '_')
            for stat_name, v in vals.items():
                metrics[f"fibernorm_{key}_{stat_name}"] = float(v) if v is not None and np.isfinite(v) else np.nan

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


def plot_fibernorm_diagnostic(
    out_folder: Path,
    ratio: Optional[np.ndarray],
    fibernorm: Optional[np.ndarray],
    fibers: Optional[list] = None,
    nbins: int = 25,
    filename: str = "fiber_normalization.png",
    title: str = "Fiber normalization diagnostic",
) -> Optional[Path]:
    """Diagnostic plot for fiber-to-fiber normalization.

    For selected fibers, show:
    - raw ratio (twi_spectra / twi_model) as a faint line
    - binned robust averages as scatter points
    - smooth interpolated normalization curve as a solid line

    The y-axis is normalized per fiber by its median for readability.
    """
    if ratio is None or fibernorm is None:
        return None
    R = np.asarray(ratio, dtype=float)
    Nfib, Npix = R.shape if R.ndim == 2 else (0, 0)
    F = np.asarray(fibernorm, dtype=float)
    if R.ndim != 2 or F.ndim != 2 or R.shape != F.shape or Nfib == 0:
        return None
    xpix = np.arange(Npix, dtype=float)
    if fibers is None:
        fibers = [3, 55, 110]
    fibers = sorted(set(int(max(0, min(f, Nfib - 1))) for f in fibers))
    if len(fibers) == 0:
        fibers = [min(0, Nfib - 1)]

    # Prepare bin edges and centers for reference scatter positions
    nbins = int(max(5, nbins))
    edges = np.linspace(-0.5, Npix - 0.5, nbins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 1, figsize=(11, 4.2))
    colors = plt.cm.tab10(np.linspace(0, 1, max(10, len(fibers))))

    for k, f in enumerate(fibers):
        y_raw = R[f].copy()
        y_mod = F[f].copy()
        # Normalize by median for clarity
        med = np.nanmedian(y_raw[np.isfinite(y_raw)]) if np.isfinite(y_raw).any() else 1.0
        if not np.isfinite(med) or med == 0:
            med = 1.0
        y_raw_n = y_raw / med
        y_mod_n = y_mod / med
        c = colors[k % colors.shape[0]]
        # raw curve
        mraw = np.isfinite(y_raw_n)
        if np.count_nonzero(mraw) > 3:
            ax.plot(xpix[mraw], y_raw_n[mraw], '-', color=c, alpha=0.25, lw=0.8, label=f"fiber {f} ratio")
        # binned robust means
        # compute within bins ignoring NaNs
        yb = []
        xb = []
        for i in range(nbins):
            lo, hi = edges[i], edges[i + 1]
            m = (xpix >= lo) & (xpix < hi) & np.isfinite(y_raw_n)
            if np.count_nonzero(m) == 0:
                yb.append(np.nan)
                xb.append(centers[i])
                continue
            vals = y_raw_n[m]
            # robust center via median if biweight unavailable here
            try:
                from math_utils import biweight as _biw
                yb.append(float(_biw(vals)))
            except Exception:
                yb.append(float(np.nanmedian(vals)))
            xb.append(centers[i])
        xb = np.array(xb, dtype=float)
        yb = np.array(yb, dtype=float)
        mb = np.isfinite(xb) & np.isfinite(yb)
        if np.count_nonzero(mb) > 0:
            ax.scatter(xb[mb], yb[mb], s=26, facecolors='none', edgecolors=c, alpha=0.9, label=f"fiber {f} bins")
        # smooth curve from model
        mmod = np.isfinite(y_mod_n)
        if np.count_nonzero(mmod) > 3:
            ax.plot(xpix[mmod], y_mod_n[mmod], '-', color=c, lw=1.3, label=f"fiber {f} smooth")

    ax.set_xlim(0, max(1, Npix - 1))
    # Y-limits around 1
    ycat = []
    for f in fibers:
        y = R[f]
        mu = np.nanmedian(y[np.isfinite(y)]) if np.isfinite(y).any() else 1.0
        if not np.isfinite(mu) or mu == 0:
            mu = 1.0
        if np.isfinite(y).any():
            ycat.append(y / mu)
        if np.isfinite(F[f]).any():
            ycat.append(F[f] / mu)
    if len(ycat):
        allv = np.concatenate([a[np.isfinite(a)] for a in ycat if a is not None])
        if allv.size:
            lo, hi = np.nanpercentile(allv, [2, 98])
            if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
                pad = 0.1 * (hi - lo)
                ax.set_ylim(lo - pad, hi + pad)
    ax.set_xlabel('Spectral pixel (x)')
    ax.set_ylabel('Ratio / median (unitless)')
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend(loc='upper right', ncol=2, fontsize=9, frameon=False)

    fig.tight_layout()
    out_folder = Path(out_folder)
    out_folder.mkdir(parents=True, exist_ok=True)
    out = out_folder / filename
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def compute_fibernorm_band_residuals(
    fibernorm_twi: Optional[np.ndarray],
    fibernorm_sci: Optional[np.ndarray],
    band_halfwidth: int = 100,
) -> Optional[Dict[str, Dict[str, float]]]:
    """Compute residual summary stats between sci and twi fibernorm in three bands.

    Returns a dict keyed by band name (Blue edge, Central, Red edge) with per-band
    summary statistics over fibers for Δ = (sci_norm − twi_norm):
    - median, std, mad
    The per-band normalization follows plot_fibernorm_compare.
    """
    if fibernorm_twi is None or fibernorm_sci is None:
        return None
    Ft = np.asarray(fibernorm_twi, dtype=float)
    Fs = np.asarray(fibernorm_sci, dtype=float)
    if Ft.ndim != 2 or Fs.ndim != 2 or Ft.shape != Fs.shape:
        return None
    Nfib, Npix = Ft.shape
    if Nfib == 0 or Npix == 0:
        return None
    hw = int(max(1, band_halfwidth))
    width = 2 * hw + 1
    c = int(Npix // 2)
    c_lo = max(0, c - hw)
    c_hi = min(Npix, c + hw + 1)
    b_lo = 0
    b_hi = min(Npix, width)
    r_hi = Npix
    r_lo = max(0, Npix - width)
    bands = {
        'Blue edge': (slice(None), slice(b_lo, b_hi)),
        'Central': (slice(None), slice(c_lo, c_hi)),
        'Red edge': (slice(None), slice(r_lo, r_hi)),
    }
    from math_utils import biweight as _biw
    def band_biweight(Fregion: np.ndarray) -> np.ndarray:
        vals = np.full(Nfib, np.nan, dtype=float)
        for f in range(Nfib):
            y = Fregion[f]
            if np.isfinite(y).any():
                try:
                    vals[f] = float(_biw(y[np.isfinite(y)]))
                except Exception:
                    vals[f] = float(np.nanmedian(y))
        return vals
    def _norm(v: np.ndarray) -> np.ndarray:
        med = np.nanmedian(v[np.isfinite(v)]) if np.isfinite(v).any() else 1.0
        if not np.isfinite(med) or med == 0:
            med = 1.0
        return v / med
    out: Dict[str, Dict[str, float]] = {}
    for name, sl in bands.items():
        row_sl, col_sl = sl
        Bt = Ft[row_sl, col_sl]
        Bs = Fs[row_sl, col_sl]
        vt = band_biweight(Bt)
        vs = band_biweight(Bs)
        vn_t = _norm(vt)
        vn_s = _norm(vs)
        mdiff = vn_s - vn_t
        md = mdiff[np.isfinite(mdiff)]
        if md.size:
            out[name] = {
                'median': float(np.nanmedian(md)),
                'std': float(np.nanstd(md)),
                'mad': float(_mad(md)),
            }
        else:
            out[name] = {'median': np.nan, 'std': np.nan, 'mad': np.nan}
    return out


def plot_fibernorm_compare(
    out_folder: Path,
    fibernorm_twi: Optional[np.ndarray],
    fibernorm_sci: Optional[np.ndarray],
    band_halfwidth: int = 100,
    filename: str = "fiber_norm_compare.png",
    title: str = "Fiber normalization: twilight vs science",
) -> Optional[Path]:
    """Compare fiber-normalization between twilight and science.

    Extends the comparison to three spectral regions to check for color terms:
    - Blue edge (low columns)
    - Central band (existing behavior)
    - Red edge (high columns)

    For each region and for each fiber, compute a robust scalar by taking the
    biweight location over a band of width ~2*band_halfwidth+1 columns, then
    plot the twilight vs science normalization (each normalized to its median)
    as a function of fiber index. Simple Δ(sci−twi) stats are annotated.
    """
    if fibernorm_twi is None or fibernorm_sci is None:
        return None
    Ft = np.asarray(fibernorm_twi, dtype=float)
    Fs = np.asarray(fibernorm_sci, dtype=float)
    if Ft.ndim != 2 or Fs.ndim != 2 or Ft.shape != Fs.shape:
        return None
    Nfib, Npix = Ft.shape
    if Nfib == 0 or Npix == 0:
        return None

    hw = int(max(1, band_halfwidth))
    width = 2 * hw + 1

    # Define slices for blue, central, red bands of equal width where possible
    c = int(Npix // 2)
    # Central band
    c_lo = max(0, c - hw)
    c_hi = min(Npix, c + hw + 1)
    # Blue band near start
    b_lo = 0
    b_hi = min(Npix, width)
    # Red band near end
    r_hi = Npix
    r_lo = max(0, Npix - width)

    bands = {
        'Blue edge': (slice(None), slice(b_lo, b_hi)),
        'Central': (slice(None), slice(c_lo, c_hi)),
        'Red edge': (slice(None), slice(r_lo, r_hi)),
    }

    from math_utils import biweight as _biw

    def band_biweight(Fregion: np.ndarray) -> np.ndarray:
        vals = np.full(Nfib, np.nan, dtype=float)
        for f in range(Nfib):
            y = Fregion[f]
            if np.isfinite(y).any():
                try:
                    vals[f] = float(_biw(y[np.isfinite(y)]))
                except Exception:
                    vals[f] = float(np.nanmedian(y))
        return vals

    def _norm(v: np.ndarray) -> np.ndarray:
        med = np.nanmedian(v[np.isfinite(v)]) if np.isfinite(v).any() else 1.0
        if not np.isfinite(med) or med == 0:
            med = 1.0
        return v / med

    # Compute per-band normalized curves for twi and sci
    per_band = {}
    for name, sl in bands.items():
        row_sl, col_sl = sl
        Bt = Ft[row_sl, col_sl]
        Bs = Fs[row_sl, col_sl]
        vt = band_biweight(Bt)
        vs = band_biweight(Bs)
        per_band[name] = (_norm(vt), _norm(vs))

    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 3.8), sharey=True)
    if not isinstance(axes, (list, tuple, np.ndarray)):
        axes = [axes]
    x = np.arange(Nfib, dtype=int)

    # Determine a common y-limit across panels for consistent comparison
    all_vals = []
    for name in ['Blue edge', 'Central', 'Red edge']:
        vn_t, vn_s = per_band[name]
        all_vals.append(vn_t[np.isfinite(vn_t)])
        all_vals.append(vn_s[np.isfinite(vn_s)])
    if any(v.size for v in all_vals):
        cat = np.concatenate([v for v in all_vals if v.size])
        if cat.size:
            lo, hi = np.nanpercentile(cat, [2, 98])
            if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
                ylim = (lo - 0.1 * (hi - lo), hi + 0.1 * (hi - lo))
            else:
                ylim = None
        else:
            ylim = None
    else:
        ylim = None

    for ax, name in zip(axes, ['Blue edge', 'Central', 'Red edge']):
        vn_t, vn_s = per_band[name]
        mt = np.isfinite(vn_t)
        ms = np.isfinite(vn_s)
        if np.count_nonzero(mt) > 1:
            ax.plot(x[mt], vn_t[mt], '-', lw=1.3, alpha=0.9, label='twilight')
        if np.count_nonzero(ms) > 1:
            ax.plot(x[ms], vn_s[ms], '-', lw=1.3, alpha=0.9, label='science')
        mdiff = vn_s - vn_t
        md = mdiff[np.isfinite(mdiff)]
        if md.size:
            mu = float(np.nanmedian(md))
            sc = float(np.nanstd(md))
            ax.text(0.02, 0.97, f"Δ: med={mu:.4f}, σ={sc:.4f}", transform=ax.transAxes, va='top', ha='left', fontsize=9)
        ax.set_title(name)
        ax.grid(alpha=0.25)
        ax.set_xlabel('Fiber index')
        if ylim is not None:
            ax.set_ylim(*ylim)

    axes[0].set_ylabel('Normalized fibernorm (unitless)')
    # Only one legend to avoid clutter
    handles, labels = axes[1].get_legend_handles_labels()
    if handles:
        axes[1].legend(loc='best', frameon=False)

    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out_folder = Path(out_folder)
    out_folder.mkdir(parents=True, exist_ok=True)
    out = out_folder / filename
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def plot_bias_dark_profile(
    out_folder: Path,
    masterbias: Optional[np.ndarray],
    masterdark: Optional[np.ndarray],
    ncenter: int = 100,
    filename: str = "bias_dark_profile.png",
    title: str = "Master bias/dark central collapse",
) -> Optional[Path]:
    """Plot central-column collapsed row profiles for master bias and dark.

    Takes approximately the central ncenter columns (or as many as available)
    from each 2D image, computes the median across columns to obtain a 1D
    profile vs. detector row, and plots bias and dark on the same axes.
    """
    if masterbias is None and masterdark is None:
        return None
    mb = np.asarray(masterbias) if masterbias is not None else None
    md = np.asarray(masterdark) if masterdark is not None else None
    # Determine central slice based on available shape (assume square ~1032)
    def center_collapse(img: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if img is None or img.ndim != 2 or img.size == 0:
            return None
        ny, nx = img.shape
        if nx <= 0 or ny <= 0:
            return None
        w = int(max(1, min(ncenter, nx)))
        c = nx // 2
        x0 = max(0, c - w // 2)
        x1 = min(nx, x0 + w)
        sl = img[:, x0:x1]
        with np.errstate(invalid='ignore'):
            prof = np.nanmedian(sl, axis=1)
        return prof.astype(float)
    pb = center_collapse(mb)
    pd = center_collapse(md)
    if (pb is None or not np.isfinite(pb).any()) and (pd is None or not np.isfinite(pd).any()):
        return None
    # Build plot
    fig, ax = plt.subplots(1, 1, figsize=(10.0, 4.0))
    if pb is not None and np.isfinite(pb).any():
        ax.plot(np.arange(pb.size), pb, label='Bias', color='tab:blue', lw=1.0)
    if pd is not None and np.isfinite(pd).any():
        ax.plot(np.arange(pd.size), pd, label='Dark', color='tab:orange', lw=1.0)
    # Scale
    vals = []
    if pb is not None: vals.append(pb[np.isfinite(pb)])
    if pd is not None: vals.append(pd[np.isfinite(pd)])
    if len(vals):
        allv = np.concatenate(vals) if len(vals) > 1 else vals[0]
        if allv.size:
            lo, hi = np.nanpercentile(allv, [1, 99])
            if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
                pad = 0.1 * (hi - lo)
                ax.set_ylim(lo - pad, hi + pad)
    ax.set_xlabel('Detector row (y)')
    ax.set_ylabel('Collapsed counts (median of central columns)')
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend(loc='best', frameon=False)

    fig.tight_layout()
    out_folder = Path(out_folder)
    out_folder.mkdir(parents=True, exist_ok=True)
    out = out_folder / filename
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def plot_trace_overlay(
    out_folder: Path,
    xchunks: Optional[np.ndarray],
    trace_chunks: Optional[np.ndarray],
    trace: Optional[np.ndarray],
    filename: str = "trace_chunks.png",
    title: str = "Trace chunks diagnostic",
    fibers: Optional[list] = None,
) -> Optional[Path]:
    """Diagnostic plot of trace chunk positions vs. full trace for selected fibers.

    Uses the outputs from fiber_utils.get_trace (xchunks and trace_chunks) and the
    final interpolated trace to visualize agreement. For each selected fiber, we:
    - subtract the mean of that fiber's full trace from both datasets
    - scatter-plot (xchunks, trace_chunks[f]) and overlay the full trace line
      (xpix, trace[f]) on the same axes so multiple fibers share a common y-scale.

    Args:
        out_folder: Directory to save the image.
        xchunks: 1D array of representative x pixel positions per chunk (length Nchunks).
        trace_chunks: 2D array (Nfibers x Nchunks) of per-chunk trace estimates.
        trace: 2D array (Nfibers x Npix) of final trace vs. pixel.
        filename: Output filename (PNG).
        title: Title for the plot.
        fibers: Optional list of fiber indices to plot; defaults to [3, 55, 110]
            (clipped to the available range).

    Returns:
        Path to the saved PNG, or None if inputs are invalid.
    """
    if xchunks is None or trace_chunks is None or trace is None:
        return None
    xc = np.asarray(xchunks, dtype=float).ravel()
    Trc = np.asarray(trace_chunks, dtype=float)
    Tr = np.asarray(trace, dtype=float)
    if Trc.ndim != 2 or Tr.ndim != 2 or xc.ndim != 1:
        return None
    Nfib_c, Nch = Trc.shape
    Nfib, Npix = Tr.shape
    if Nfib == 0 or Npix == 0 or Nch == 0:
        return None
    # Ensure sizes are consistent
    if xc.size != Nch:
        # try to coerce; otherwise fail
        try:
            xc = xc[:Nch]
        except Exception:
            return None
    # Default fibers to plot
    if fibers is None:
        fibers = [3, 55, 110]
    # Clip to available range and unique
    fibers = sorted(set(int(max(0, min(f, Nfib - 1))) for f in fibers))
    if len(fibers) == 0:
        fibers = [min(0, Nfib - 1)]

    xpix = np.arange(Npix, dtype=float)

    # Build the figure
    fig, ax = plt.subplots(1, 1, figsize=(11.0, 4.2))

    colors = plt.cm.tab10(np.linspace(0, 1, max(10, len(fibers))))
    for k, f in enumerate(fibers):
        y_full = Tr[f]
        if not np.isfinite(y_full).any():
            continue
        mu = float(np.nanmean(y_full)) if np.isfinite(y_full).any() else 0.0
        y_off = y_full - mu
        # chunk samples for this fiber
        y_chunks = Trc[f]
        # subtract same mean from chunks
        y_chunks_off = y_chunks - mu
        c = colors[k % colors.shape[0]]
        # line of full trace
        m_full = np.isfinite(y_off)
        if np.count_nonzero(m_full) > 3:
            ax.plot(xpix[m_full], y_off[m_full], '-', lw=1.0, alpha=0.9, color=c, label=f"fiber {f} trace")
        # scatter of chunk measurements
        m_chunks = np.isfinite(y_chunks_off) & np.isfinite(xc)
        if np.count_nonzero(m_chunks) > 0:
            ax.scatter(xc[m_chunks], y_chunks_off[m_chunks], s=22, marker='o', facecolors='none', edgecolors=c, alpha=0.9, label=f"fiber {f} chunks")

    ax.set_xlim(0, max(1, Npix - 1))
    # Y-limits: symmetric around 0 to compare fibers
    y_all = []
    for f in fibers:
        if 0 <= f < Nfib:
            y_full = Tr[f]
            if np.isfinite(y_full).any():
                mu = float(np.nanmean(y_full))
                y_all.append(y_full - mu)
            y_chunks = Trc[f] if f < Nfib_c else None
            if y_chunks is not None:
                y_all.append(y_chunks - mu)
    if len(y_all):
        ycat = np.concatenate([a[np.isfinite(a)] for a in y_all if a is not None])
        if ycat.size:
            lim = np.nanpercentile(np.abs(ycat), 98)
            if np.isfinite(lim) and lim > 0:
                ax.set_ylim(-1.1 * lim, 1.1 * lim)
    ax.set_xlabel('Spectral pixel (x)')
    ax.set_ylabel('Row offset (pixels) [mean-subtracted]')
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend(loc='upper right', ncol=2, fontsize=9, frameon=False)

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
    fibernorm_img: Optional[Path] = None,
    fibernorm_cmp_img: Optional[Path] = None,
    biasdark_img: Optional[Path] = None,
) -> Path:
    """Create a compact PNG QA page with key metrics and optional images.

    Layout: metrics panel on the left; diagnostic plots stacked vertically on the right.
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
        items.append((Path(trace_img), 'Trace chunks diagnostic'))
    if fibernorm_img and Path(fibernorm_img).exists():
        items.append((Path(fibernorm_img), 'Fiber normalization diagnostic'))
    if fibernorm_cmp_img and Path(fibernorm_cmp_img).exists():
        items.append((Path(fibernorm_cmp_img), 'Fiber normalization compare (twi vs sci)'))
    if biasdark_img and Path(biasdark_img).exists():
        items.append((Path(biasdark_img), 'Bias/Dark central collapse'))

    # Ensure at least one row on the right (placeholder if none)
    nimgs = max(1, len(items))

    # Figure size scales with number of images to keep each readable
    fig_height = max(6.5, 3.0 * nimgs + 1.0)
    fig = plt.figure(figsize=(12, fig_height), constrained_layout=True)
    # 2 columns: left for metrics (spans all rows), right stacks images
    gs = fig.add_gridspec(nrows=nimgs, ncols=2, width_ratios=[1, 2])

    # Left: metrics panel spanning all rows
    ax_text = fig.add_subplot(gs[:, 0])
    ax_text.axis('off')

    # Build multiline metrics text
    zero_flag = bool(metrics.get('zero_frames_suspected', False))
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
        f"median trace RMS: {metrics.get('trace_rms_median', np.nan):.3f}",
        f"median arc RMS: {metrics.get('median_arc_rms', np.nan):.3f}",
    ]
    if zero_flag:
        lines.append("WARNING: Suspected all-zero frames (readnoise≈0). Metrics suppressed.")
    ax_text.text(0.02, 0.98, "\n".join(lines), family='monospace', fontsize=11, va='top')

    # Right: stacked images
    if len(items) == 0:
        ax = fig.add_subplot(gs[0, 1])
        ax.axis('off')
        ax.text(0.5, 0.5, 'No diagnostic image available', ha='center', va='center', color='0.5')
    else:
        for i in range(nimgs):
            ax = fig.add_subplot(gs[i, 1])
            ax.axis('off')
            path_i, title_i = items[i]
            try:
                img = plt.imread(path_i)
                ax.imshow(img)
                ax.set_title(title_i)
            except Exception:
                ax.text(0.5, 0.5, 'Image unavailable', ha='center', va='center', color='0.5')

    # Save
    png_path = out_folder / f"QA_{amp_id}.png"
    fig.savefig(png_path, dpi=150)
    
    # Build structured QA record (augment metrics in-place for backward-compatibility)
    def _status_from_thresholds(value: float, thr: dict) -> str:
        try:
            v = float(value)
        except Exception:
            return "fail"
        warn = thr.get('warn', None)
        fail = thr.get('fail', None)
        mode = thr.get('mode', 'high_is_bad')  # or 'low_is_bad'
        if not np.isfinite(v):
            return "fail"
        if mode == 'high_is_bad':
            if fail is not None and v >= fail:
                return 'fail'
            if warn is not None and v >= warn:
                return 'warn'
            return 'pass'
        else:  # low_is_bad
            if fail is not None and v <= fail:
                return 'fail'
            if warn is not None and v <= warn:
                return 'warn'
            return 'pass'
    
    # Default thresholds; can be tuned later or overridden downstream
    thresholds = {
        'readnoise_e': {'warn': 4.5, 'fail': 6.0, 'mode': 'high_is_bad'},
        'bad_wavelength_frac': {'warn': 0.05, 'fail': 0.20, 'mode': 'high_is_bad'},
        'failed_traces': {'warn': 5, 'fail': 20, 'mode': 'high_is_bad'},
        'median_arc_rms': {'warn': 0.20, 'fail': 0.50, 'mode': 'high_is_bad'},
        'trace_rms_median': {'warn': 0.20, 'fail': 0.50, 'mode': 'high_is_bad'},
        'masked_spectral_frac': {'warn': 0.05, 'fail': 0.20, 'mode': 'high_is_bad'},
        # Fiber norm band residuals (absolute median difference)
        'fibernorm_blue_edge_std': {'warn': 0.05, 'fail': 0.10, 'mode': 'high_is_bad'},
        'fibernorm_central_std': {'warn': 0.05, 'fail': 0.10, 'mode': 'high_is_bad'},
        'fibernorm_red_edge_std': {'warn': 0.05, 'fail': 0.10, 'mode': 'high_is_bad'},
    }

    # Plot paths to embed in JSON
    plots = {
        'qa_page': str(png_path),
        'arc_identify': str(ref_profile_img) if ref_profile_img else None,
        'specmask_overlay': str(specmask_img) if specmask_img else None,
        'trace_overlay': str(trace_img) if trace_img else None,
        'fibernorm_diagnostic': str(fibernorm_img) if fibernorm_img else None,
        'fibernorm_compare': str(fibernorm_cmp_img) if fibernorm_cmp_img else None,
        'biasdark_profile': str(biasdark_img) if biasdark_img else None,
    }

    # Compute checks and overall status
    check_keys = [
        'readnoise_e',
        'bad_wavelength_frac',
        'failed_traces',
        'trace_rms_median',
        'median_arc_rms',
        'masked_spectral_frac',
    ]
    # Include fibernorm keys if present
    for fk in ('fibernorm_blue_edge_std', 'fibernorm_central_std', 'fibernorm_red_edge_std'):
        if fk in metrics and np.isfinite(metrics.get(fk, np.nan)):
            check_keys.append(fk)
    checks = {}
    worst = 'pass'
    order = {'pass': 0, 'warn': 1, 'fail': 2}
    zero_flag = bool(metrics.get('zero_frames_suspected', False))
    if zero_flag:
        # Suppress misleading numbers: mark all checks (except readnoise) as fail with no value
        for key in check_keys:
            thr = thresholds.get(key, {})
            if key == 'readnoise_e':
                val = metrics.get(key, np.nan)
                status = _status_from_thresholds(val, thr)
                checks[key] = {
                    'value': float(val) if np.isfinite(val) else None,
                    'status': status,
                    'thresholds': thr,
                }
            else:
                checks[key] = {
                    'value': None,
                    'status': 'fail',
                    'thresholds': thr,
                }
        worst = 'fail'
        metrics['__notes__'] = 'suspected_zero_frames'
    else:
        for key in check_keys:
            val = metrics.get(key, np.nan)
            thr = thresholds.get(key, {})
            # use absolute for fibernorm residual medians
            if 'fibernorm_' in key and np.isfinite(val):
                val = abs(float(val))
            status = _status_from_thresholds(val, thr)
            checks[key] = {
                'value': float(val) if np.isfinite(val) else None,
                'status': status,
                'thresholds': thr,
            }
            if order[status] > order[worst]:
                worst = status

    # Augment metrics dict with structured info
    metrics['__plots__'] = plots
    metrics['__checks__'] = checks
    metrics['__overall_status__'] = worst
    metrics['__thresholds__'] = thresholds

    # Write JSON sidecar
    sidecar = out_folder / f"QA_{amp_id}.json"
    try:
        with open(sidecar, 'w') as f:
            json.dump(metrics, f, indent=2, default=lambda o: float(o) if isinstance(o, (np.floating,)) else o)
    except Exception:
        pass

    return png_path
