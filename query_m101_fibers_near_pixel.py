#!/usr/bin/env python3
"""Find VIRUS fibers near one rebuilt-cube pixel at one wavelength.

This is deliberately a geometry/identity-only diagnostic.  It reads only the
Info and Survey tables from each M101 H5 file, applies the production ADR
convention at one wavelength, and writes the fibers within the requested
angular radius.  It does not read spectra or reconstruct a cube.

The ``fiber_row`` column is the zero-based row number shared by Info and
Fibers in the H5 file.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


EXCLUDED_H5 = "20200523_0000023.h5"
DEFAULT_CUBE = "m101_virus_v2_rebuild_cube.fits"
DEFAULT_X = 637.0
DEFAULT_Y = 701.0
DEFAULT_WAVELENGTH = 4882.0
DEFAULT_RADIUS = 2.0

FIBER_FIELDS = (
    "H5", "exposure", "fiber_row", "SPECID", "IFUSLOT", "IFUID", "AMP",
    "IFU_CODE", "RA_4882", "Dec_4882", "separation_arcsec",
    "hardware_excluded", "hardware_exclusion_reason",
)


def parse_args(argv=None):
    """Parse the small public command line interface."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--x", type=float, default=DEFAULT_X,
                        help="DS9/FITS 1-based X pixel (default: %(default)s)")
    parser.add_argument("--y", type=float, default=DEFAULT_Y,
                        help="DS9/FITS 1-based Y pixel (default: %(default)s)")
    parser.add_argument("--wavelength", type=float, default=DEFAULT_WAVELENGTH,
                        help="wavelength in Angstrom (default: %(default)s)")
    parser.add_argument("--radius", type=float, default=DEFAULT_RADIUS,
                        help="matching radius in arcsec (default: %(default)s)")
    parser.add_argument("--cube", default=DEFAULT_CUBE,
                        help="rebuilt cube used only for its WCS")
    return parser.parse_args(argv)


def angular_separation_arcsec(ra_deg, dec_deg, target_ra_deg, target_dec_deg):
    """Return spherical separations in arcseconds for scalar or array inputs."""
    ra = np.deg2rad(np.asarray(ra_deg, dtype=float))
    dec = np.deg2rad(np.asarray(dec_deg, dtype=float))
    target_ra = np.deg2rad(float(target_ra_deg))
    target_dec = np.deg2rad(float(target_dec_deg))

    # Haversine form is stable for the small separations used here and handles
    # RA wrap naturally after reducing the longitude difference to [-pi, pi].
    dra = (ra - target_ra + np.pi) % (2.0 * np.pi) - np.pi
    ddec = dec - target_dec
    hav = (np.sin(ddec / 2.0) ** 2
           + np.cos(dec) * np.cos(target_dec) * np.sin(dra / 2.0) ** 2)
    angle = 2.0 * np.arctan2(np.sqrt(np.maximum(hav, 0.0)),
                             np.sqrt(np.maximum(1.0 - hav, 0.0)))
    return np.rad2deg(angle) * 3600.0


def cube_pixel_to_sky(header, x, y):
    """Convert DS9/FITS 1-based spatial coordinates with the cube WCS."""
    from astropy.wcs import WCS

    wcs = WCS(header).celestial
    world = np.asarray(wcs.wcs_pix2world([[float(x), float(y)]], 1), dtype=float)
    if world.shape != (1, 2) or not np.all(np.isfinite(world[0])):
        raise ValueError("cube WCS returned a non-finite target coordinate")
    return float(world[0, 0]), float(world[0, 1])


def _as_text(value):
    if isinstance(value, (bytes, np.bytes_)):
        return value.decode("utf-8", errors="replace").strip()
    return str(value).strip()


def _survey_by_exposure(survey):
    result = {}
    for row in survey:
        exposure = int(row["exp"])
        if exposure in result:
            raise ValueError("Survey has duplicate exposure %d" % exposure)
        result[exposure] = {name: row[name] for name in survey.colnames}
    if not result:
        raise ValueError("Survey table is empty")
    return result


def _optional_ifu_code(info, indices):
    """Use an H5-provided code if present; do not build a cache just for it."""
    for name in ("ifu_code", "IFU_CODE", "ifucode"):
        if name in info.colnames:
            values = np.asarray(info.cols._f_col(name)[indices])
            unique = np.unique(values)
            if unique.size == 1:
                return _as_text(unique[0])
    return ""


def _load_project_helpers():
    """Import the established helpers only when the data query is run."""
    script_dir = Path(__file__).resolve().parent
    source_roots = [script_dir, script_dir / "Remedy"]
    for root in source_roots:
        if (root / "diagnose_m101_hierarchical.py").exists():
            root_text = str(root)
            if root_text not in sys.path:
                sys.path.insert(0, root_text)
            break
    from astrometry import Astrometry
    from extract import Extract
    from diagnose_m101_hierarchical import build_groups
    from m101_hardware_exclusions import exclusion_reason, hardware_excluded
    return Astrometry, Extract, build_groups, exclusion_reason, hardware_excluded


def _adr_offset_at_wavelength(survey_row, wavelength, astrometry_cls, extract_cls):
    """Return the production ADR offset at one wavelength for one exposure."""
    survey_ra = float(survey_row["ra"])
    survey_dec = float(survey_row["dec"])
    pa = float(survey_row["pa"])
    if not np.all(np.isfinite([survey_ra, survey_dec, pa])):
        raise ValueError("Survey RA/Dec/PA is invalid")

    astrometry = astrometry_cls(survey_ra, survey_dec, pa, 0.0, 0.0)
    extractor = extract_cls(wave=np.asarray([float(wavelength)], dtype=float))
    extractor.get_ADR_RAdec(astrometry)
    dra = float(extractor.ADRra[0]) / 3600.0 / np.cos(np.deg2rad(survey_dec))
    ddec = float(extractor.ADRdec[0]) / 3600.0
    return dra, ddec


def _group_key(row):
    return (row["H5"], int(row["exposure"]), int(row["SPECID"]),
            int(row["IFUSLOT"]), int(row["IFUID"]), row["AMP"])


def _physical_key(row):
    return (row["H5"], int(row["SPECID"]), int(row["IFUSLOT"]),
            int(row["IFUID"]), row["AMP"])


def _summary_rows(matches):
    grouped = defaultdict(list)
    for row in matches:
        grouped[_group_key(row)].append(row)
    rows = []
    for key, values in grouped.items():
        distances = np.asarray([float(row["separation_arcsec"]) for row in values])
        rows.append({
            "H5": key[0], "exposure": key[1], "SPECID": key[2],
            "IFUSLOT": key[3], "IFUID": key[4], "AMP": key[5],
            "n_fibers": len(values), "min_separation_arcsec": float(np.min(distances)),
            "median_separation_arcsec": float(np.median(distances)),
            "hardware_excluded": values[0]["hardware_excluded"],
        })
    return sorted(rows, key=lambda row: row["min_separation_arcsec"])


def _physical_summary_rows(matches):
    grouped = defaultdict(list)
    for row in matches:
        grouped[_physical_key(row)].append(row)
    rows = []
    for key, values in grouped.items():
        distances = np.asarray([float(row["separation_arcsec"]) for row in values])
        counts = defaultdict(int)
        for row in values:
            counts[int(row["exposure"])] += 1
        rows.append({
            "H5": key[0], "SPECID": key[1], "IFUSLOT": key[2],
            "IFUID": key[3], "AMP": key[4], "exposures": ",".join(
                str(exposure) for exposure in sorted(counts)),
            "counts_by_exposure": ",".join(
                "%d:%d" % item for item in sorted(counts.items())),
            "n_fibers": len(values), "min_separation_arcsec": float(np.min(distances)),
            "median_separation_arcsec": float(np.median(distances)),
            "hardware_excluded": values[0]["hardware_excluded"],
        })
    return sorted(rows, key=lambda row: row["min_separation_arcsec"])


def _fmt(value):
    if isinstance(value, float):
        return "%.6f" % value
    return str(value)


def _write_summary(path, args, target_ra, target_dec, matches,
                   group_rows, physical_rows):
    with path.open("w", encoding="utf-8") as stream:
        stream.write("M101 fiber geometry query\n")
        stream.write("DS9 X,Y (1-based): %.6f, %.6f\n" % (args.x, args.y))
        stream.write("RA: %.10f\nDec: %.10f\n" % (target_ra, target_dec))
        stream.write("wavelength_A: %.6f\nradius_arcsec: %.6f\n" %
                     (args.wavelength, args.radius))
        stream.write("matching fibers: %d\n\n" % len(matches))

        stream.write("GROUP SUMMARY: H5 + exposure + SPECID + IFUSLOT + IFUID + AMP\n")
        stream.write("H5\texposure\tSPECID\tIFUSLOT\tIFUID\tAMP\tn_fibers\t"
                     "min_separation_arcsec\tmedian_separation_arcsec\t"
                     "hardware_excluded\n")
        for row in group_rows:
            stream.write("\t".join(_fmt(row[name]) for name in (
                "H5", "exposure", "SPECID", "IFUSLOT", "IFUID", "AMP",
                "n_fibers", "min_separation_arcsec",
                "median_separation_arcsec", "hardware_excluded")) + "\n")

        stream.write("\nPHYSICAL AMPLIFIER SUMMARY: H5 + SPECID + IFUSLOT + IFUID + AMP\n")
        stream.write("H5\tSPECID\tIFUSLOT\tIFUID\tAMP\texposures\t"
                     "counts_by_exposure\tn_fibers\tmin_separation_arcsec\t"
                     "median_separation_arcsec\thardware_excluded\n")
        for row in physical_rows:
            stream.write("\t".join(_fmt(row[name]) for name in (
                "H5", "SPECID", "IFUSLOT", "IFUID", "AMP", "exposures",
                "counts_by_exposure", "n_fibers", "min_separation_arcsec",
                "median_separation_arcsec", "hardware_excluded")) + "\n")


def _query(args):
    from astropy.io import fits
    import tables

    astrometry_cls, extract_cls, build_groups, exclusion_reason, hardware_excluded = (
        _load_project_helpers())

    cube_path = Path(args.cube).expanduser().resolve()
    with fits.open(cube_path, memmap=True) as hdul:
        target_ra, target_dec = cube_pixel_to_sky(hdul[0].header, args.x, args.y)

    h5_paths = sorted(Path.cwd().glob("20*.h5"))
    excluded = [path for path in h5_paths if path.name == EXCLUDED_H5]
    h5_paths = [path for path in h5_paths if path.name != EXCLUDED_H5]
    if not h5_paths:
        raise ValueError("no 20*.h5 files found in %s" % Path.cwd())
    if excluded:
        print("Skipping explicitly excluded H5: %s" % EXCLUDED_H5, flush=True)

    matches = []
    for number, h5_path in enumerate(h5_paths, start=1):
        with tables.open_file(str(h5_path), mode="r") as h5:
            if not {"Info", "Survey"}.issubset(h5.root._v_children):
                raise ValueError("%s needs Info and Survey tables" % h5_path.name)
            info = h5.root.Info
            groups, _labels = build_groups(info)
            native_ra = np.asarray(info.cols.ra[:], dtype=float)
            native_dec = np.asarray(info.cols.dec[:], dtype=float)
            if native_ra.shape != native_dec.shape or native_ra.size != info.nrows:
                raise ValueError("%s Info RA/Dec row alignment is invalid" % h5_path.name)
            survey_by_exposure = _survey_by_exposure(h5.root.Survey)
            adr_by_exposure = {
                exposure: _adr_offset_at_wavelength(
                    survey_row, args.wavelength, astrometry_cls, extract_cls)
                for exposure, survey_row in survey_by_exposure.items()
            }
            date = h5_path.name[:8]

            h5_matches = 0
            for group in groups:
                exposure = int(group["exposure"])
                if exposure not in survey_by_exposure:
                    raise ValueError("%s Survey has no exposure %d" %
                                     (h5_path.name, exposure))
                indices = np.asarray(group["indices"], dtype=int)
                dra, ddec = adr_by_exposure[exposure]
                survey_dec = float(survey_by_exposure[exposure]["dec"])
                ra_4882 = native_ra[indices] - dra
                dec_4882 = native_dec[indices] - ddec
                separation = angular_separation_arcsec(
                    ra_4882, dec_4882, target_ra, target_dec)
                selected = np.isfinite(separation) & (separation <= args.radius)
                if not np.any(selected):
                    continue

                specid = int(group["specid"])
                ifuslot = int(group["ifuslot"])
                ifuid = int(group["ifuid"])
                amp = str(group["amp"]).upper()
                excluded_now = hardware_excluded(
                    date=date, h5=h5_path.name, specid=specid,
                    ifuslot=ifuslot, ifuid=ifuid, amp=amp, purpose="cube")
                reason = exclusion_reason(
                    date=date, h5=h5_path.name, specid=specid,
                    ifuslot=ifuslot, ifuid=ifuid, amp=amp, purpose="cube") or ""
                ifu_code = _optional_ifu_code(info, indices)
                for local in np.flatnonzero(selected):
                    row_index = int(indices[local])
                    matches.append({
                        "H5": h5_path.name, "exposure": exposure,
                        "fiber_row": row_index, "SPECID": specid,
                        "IFUSLOT": ifuslot, "IFUID": ifuid, "AMP": amp,
                        "IFU_CODE": ifu_code, "RA_4882": float(ra_4882[local]),
                        "Dec_4882": float(dec_4882[local]),
                        "separation_arcsec": float(separation[local]),
                        "hardware_excluded": bool(excluded_now),
                        "hardware_exclusion_reason": reason,
                    })
                h5_matches += int(np.sum(selected))

        print("[%d/%d] %s: %d matching fibers" %
              (number, len(h5_paths), h5_path.name, h5_matches), flush=True)

    matches.sort(key=lambda row: float(row["separation_arcsec"]))
    group_rows = _summary_rows(matches)
    physical_rows = _physical_summary_rows(matches)
    csv_path = Path.cwd() / "m101_fibers_near_637_701.csv"
    summary_path = Path.cwd() / "m101_fibers_near_637_701_summary.txt"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIBER_FIELDS)
        writer.writeheader()
        writer.writerows(matches)
    _write_summary(summary_path, args, target_ra, target_dec, matches,
                   group_rows, physical_rows)

    print("DS9 X,Y: %.6f, %.6f" % (args.x, args.y))
    print("RA, Dec: %.10f, %.10f" % (target_ra, target_dec))
    print("wavelength: %.6f A; radius: %.6f arcsec" %
          (args.wavelength, args.radius))
    print("matching fibers: %d" % len(matches))
    print("wrote %s" % csv_path)
    print("wrote %s" % summary_path)


def main(argv=None):
    args = parse_args(argv)
    if not np.isfinite(args.x) or not np.isfinite(args.y):
        raise SystemExit("--x and --y must be finite")
    if not np.isfinite(args.wavelength):
        raise SystemExit("--wavelength must be finite")
    if not np.isfinite(args.radius) or args.radius < 0.0:
        raise SystemExit("--radius must be finite and non-negative")
    _query(args)


if __name__ == "__main__":
    main()
