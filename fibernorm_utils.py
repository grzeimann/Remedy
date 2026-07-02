import numpy as np
from typing import Optional, Tuple

from scipy.interpolate import interp1d

# Local imports
from math_utils import biweight
from mask_utils import build_model_spectra


def compute_fiber_normalization(
    spectra: Optional[np.ndarray],
    wave2d: Optional[np.ndarray],
    nbins: int = 25,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Compute per-fiber smooth normalization from input spectra.

    This factors the core algorithm out of build_calibration_h5file.step_fiber_normalization
    so it can be reused for both twilight and science spectra.

    Args:
        spectra: 2D array (Nfib x Npix) of extracted spectra (e.g., twilight or science).
        wave2d: 2D array (Nfib x Npix) wavelength solution on same sampling as spectra.
        nbins: Number of x-pixel bins for robust averaging before interpolation.

    Returns:
        fibernorm: 2D array (Nfib x Npix) smooth per-fiber normalization curves, or None.
        ratio: 2D array (Nfib x Npix) raw ratio spectra / model(same grid), or None.
    """
    try:
        if spectra is None or wave2d is None:
            return None, None
        S = np.asarray(spectra, dtype=float)
        W = np.asarray(wave2d, dtype=float)
        if S.ndim != 2 or W.ndim != 2 or S.shape != W.shape:
            return None, None
        nrows, npix = S.shape
        # Build a model on the current wavelength grid
        good_solutions = np.isfinite(W).sum(axis=1) > (0.8 * W.shape[1])
        twi_interp, twi_model_spec, twi_image, _ = build_model_spectra(S, W, good_solutions)
        M = twi_interp(W)  # model sampled on same grid
        # Compute ratio safely
        denom = np.where(np.isfinite(M) & (M != 0), M, np.nan)
        R = S / denom
        R[~np.isfinite(R)] = np.nan
        # Fit smooth per-fiber model via binned robust averages + interpolation
        xpix = np.arange(npix, dtype=float)
        nbins = int(max(5, nbins))
        edges = np.linspace(-0.5, npix - 0.5, nbins + 1)
        centers = 0.5 * (edges[:-1] + edges[1:])
        F = np.full_like(S, np.nan, dtype=float)
        for f in range(nrows):
            y = R[f]
            if not np.isfinite(y).any():
                continue
            yb = np.full(nbins, np.nan, dtype=float)
            for i in range(nbins):
                lo, hi = edges[i], edges[i + 1]
                m = (xpix >= lo) & (xpix < hi) & np.isfinite(y)
                if np.count_nonzero(m) == 0:
                    continue
                vals = y[m]
                try:
                    yb[i] = float(biweight(vals))
                except Exception:
                    yb[i] = float(np.nanmedian(vals))
            mb = np.isfinite(centers) & np.isfinite(yb)
            if np.count_nonzero(mb) >= 4:
                try:
                    fi = interp1d(
                        centers[mb], yb[mb], kind="cubic", bounds_error=False,
                        fill_value="extrapolate", assume_sorted=True,
                    )
                except Exception:
                    fi = interp1d(
                        centers[mb], yb[mb], kind="linear", bounds_error=False,
                        fill_value="extrapolate", assume_sorted=True,
                    )
                F[f] = fi(xpix)
            elif np.count_nonzero(mb) >= 2:
                fi = interp1d(
                    centers[mb], yb[mb], kind="linear", bounds_error=False,
                    fill_value="extrapolate", assume_sorted=True,
                )
                F[f] = fi(xpix)
            else:
                mu = np.nanmedian(y[np.isfinite(y)])
                if np.isfinite(mu):
                    F[f] = np.full(npix, mu, dtype=float)
        return F, R
    except Exception:
        # Caller should handle logging
        return None, None
