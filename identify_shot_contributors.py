#!/usr/bin/env python3
"""Identify M101 H5 shots that can contribute at one sky coordinate.

This is a read-only, one-coordinate diagnostic.  It follows the mosaic
builder's quoted-glob convention and uses the same one-arcsecond tangent-plane
coordinate convention and Gaussian support radius as the reconstruction.
"""

from argparse import ArgumentParser
import glob
from pathlib import Path

import numpy as np
import tables

from astrometry import Astrometry


GAUSSIAN_FWHM_ARCSEC = 1.8
GAUSSIAN_SIGMA_ARCSEC = GAUSSIAN_FWHM_ARCSEC / 2.35
GAUSSIAN_SUPPORT_ARCSEC = 2.0 * GAUSSIAN_SIGMA_ARCSEC
NATIVE_FIBERS_PER_IFU = 448
FIBERS_PER_BLOCK = 112
SCIENCE_WAVE = np.linspace(3470.0, 5540.0, 1036)


def resolved_h5_files(pattern):
    """Resolve only the supplied nonrecursive glob, in cube-builder order."""
    return sorted(glob.glob(pattern))


def exposure_labels(nrows, nslots):
    """Reproduce the existing 112-fiber interleaved exposure partition."""
    nexp = int(nrows / float(NATIVE_FIBERS_PER_IFU * nslots))
    if nexp < 1:
        raise ValueError("cannot infer exposure count from rows=%d, nslots=%d"
                         % (nrows, nslots))
    indices = np.arange(nrows, dtype=np.int64)
    labels = (indices // FIBERS_PER_BLOCK) % nexp + 1
    return labels.astype(np.int16), nexp


def target_tangent_plane(ra, dec):
    """Build the same 1-arcsec, unrotated TP convention as the cube builder."""
    astrometry = Astrometry(ra, dec, 0.0, 1.0, 1.0)
    return astrometry.setup_TP(ra, dec, 0.0, x0=1.0, y0=1.0)


def fiber_distances_arcsec(ra, dec, target_ra, target_dec):
    """Return TP distances in arcsec, with FITS origin=1 explicitly used."""
    tp = target_tangent_plane(target_ra, target_dec)
    x, y = tp.wcs_world2pix(np.asarray(ra, dtype=float),
                            np.asarray(dec, dtype=float), 1)
    finite = np.isfinite(x) & np.isfinite(y)
    distances = np.full(x.shape, np.inf, dtype=float)
    distances[finite] = np.hypot(x[finite] - 1.0, y[finite] - 1.0)
    return distances


def _native_info(h5):
    if "Info" not in h5.root._v_children:
        raise ValueError("H5 has no Info table")
    info = h5.root.Info
    nrows = int(info.nrows)
    if "ra" not in info.colnames or "dec" not in info.colnames:
        raise ValueError("Info table lacks ra/dec columns")
    ra = np.asarray(info.cols.ra[:], dtype=float)
    dec = np.asarray(info.cols.dec[:], dtype=float)
    if "ifuslot" not in info.colnames:
        raise ValueError("Info table lacks ifuslot column")
    nslots = len(np.unique(info.cols.ifuslot[:]))
    labels, nexp = exposure_labels(nrows, nslots)
    return info, ra, dec, labels, nexp


def _science_at_wavelength(h5, row_indices, wavelength):
    """Read only relevant Fibers rows and interpolate on the science grid."""
    if "Fibers" not in h5.root._v_children:
        raise ValueError("H5 has no Fibers table for wavelength check")
    fibers = h5.root.Fibers
    info_rows = int(h5.root.Info.nrows)
    if int(fibers.nrows) != info_rows:
        raise ValueError("Info/Fibers row mismatch: %d/%d" %
                         (info_rows, int(fibers.nrows)))
    row_indices = np.asarray(row_indices, dtype=np.int64)
    if row_indices.size == 0:
        return np.empty(0, dtype=float)
    spectra = np.asarray(
        fibers.read_coordinates(row_indices, field="spectrum"), dtype=float)
    return np.asarray([
        np.interp(float(wavelength), SCIENCE_WAVE, spectrum,
                  left=np.nan, right=np.nan)
        for spectrum in spectra
    ], dtype=float)


def inspect_h5(path, target_ra, target_dec, wavelength=None):
    with tables.open_file(path, mode="r") as h5:
        info, ra, dec, labels, nexp = _native_info(h5)
        distances = fiber_distances_arcsec(ra, dec, target_ra, target_dec)
        finite_coordinates = np.isfinite(ra) & np.isfinite(dec)
        relevant = finite_coordinates & (
            distances <= GAUSSIAN_SUPPORT_ARCSEC)
        relevant_indices = np.flatnonzero(relevant)
        exposure_values = np.unique(labels[relevant])
        row = {
            "h5": Path(path).name,
            "min_distance_arcsec": (
                float(np.min(distances[finite_coordinates]))
                if np.any(finite_coordinates) else np.inf),
            "geometric_fibers": int(relevant.sum()),
            "exposures": ",".join(str(int(value))
                                  for value in exposure_values),
            "geometric_contributor": bool(relevant_indices.size),
            "usable_fibers": None,
            "actual_contributor": None,
            "nexp": nexp,
        }
        if wavelength is not None:
            values = _science_at_wavelength(h5, relevant_indices, wavelength)
            usable = np.isfinite(values)
            row["usable_fibers"] = int(usable.sum())
            row["actual_contributor"] = bool(usable.any())
    return row


def _format_distance(value):
    return "%.3f" % value if np.isfinite(value) else "inf"


def print_results(rows, wavelength):
    print("H5                         min_dist  fibers  exposures  usable  contributor")
    for row in rows:
        contributor = ("YES" if (
            row["actual_contributor"] if wavelength is not None
            else row["geometric_contributor"]) else "NO")
        usable = ("-" if wavelength is None else str(row["usable_fibers"]))
        print("%-26s %8s  %6d  %-9s %7s  %s" %
              (row["h5"], _format_distance(row["min_distance_arcsec"]),
               row["geometric_fibers"], row["exposures"] or "-",
               usable, contributor))

    geometric = [row for row in rows if row["geometric_contributor"]]
    print("")
    print("Geometric contributors: %d" % len(geometric))
    if wavelength is not None:
        actual = [row for row in rows if row["actual_contributor"]]
        print("Actual contributors at %.3f A: %d" %
              (wavelength, len(actual)))
        if len(actual) == 1:
            print("Exactly one H5 contributes at %.3f A: %s" %
                  (wavelength, actual[0]["h5"]))
        elif actual:
            print("Contributing H5 files at %.3f A: %s" %
                  (wavelength, ", ".join(row["h5"] for row in actual)))
    elif geometric:
        print("Geometrically contributing H5 files: %s" %
              ", ".join(row["h5"] for row in geometric))


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("h5_pattern", help="quoted H5 glob, e.g. '2*.h5'")
    parser.add_argument("--ra", type=float, required=True)
    parser.add_argument("--dec", type=float, required=True)
    parser.add_argument("--wavelength", type=float,
                        help="optional science wavelength in Angstrom")
    args = parser.parse_args()

    files = resolved_h5_files(args.h5_pattern)
    if not files:
        parser.error("no H5 files matched pattern: %s" % args.h5_pattern)
    print("Supplied H5 pattern: %s" % args.h5_pattern)
    print("Resolved H5 files: %d" % len(files))
    for path in files:
        print("  %s" % path)
    print("Target: RA=%.8f Dec=%.8f" % (args.ra, args.dec))
    print("Gaussian support: %.6f arcsec (sigma=%.6f arcsec)" %
          (GAUSSIAN_SUPPORT_ARCSEC, GAUSSIAN_SIGMA_ARCSEC))
    if args.wavelength is not None:
        print("Science wavelength: %.6f A; usable means finite interpolated "
              "Fibers.spectrum, matching the cube kernel's finite-value rule."
              % args.wavelength)

    rows = [inspect_h5(path, args.ra, args.dec, args.wavelength)
            for path in files]
    rows.sort(key=lambda row: (
        not (row["actual_contributor"] if args.wavelength is not None
             else row["geometric_contributor"]),
        not row["geometric_contributor"],
        row["min_distance_arcsec"]))
    print_results(rows, args.wavelength)


if __name__ == "__main__":
    main()
