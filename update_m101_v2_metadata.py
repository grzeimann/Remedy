#!/usr/bin/env python3
"""Metadata-only repair for the six M101 VIRUS v2 primary FITS products."""

import argparse
import hashlib
import os
from pathlib import Path
import tempfile

import astropy.units as u
from astropy.io import fits
from astropy.wcs import WCS
import numpy as np


FILES = (
    "m101_virus_v2_cube.fits",
    "m101_virus_v2_variance_cube.fits",
    "m101_virus_v2_errorcube.fits",
    "m101_virus_v2_dq_cube.fits",
    "m101_virus_v2_ncontrib_cube.fits",
    "m101_virus_v2_weight_cube.fits",
)
SCI_UNIT = "1e-17 erg s-1 cm-2 Angstrom-1 arcsec-2"
VAR_UNIT = "(1e-17 erg s-1 cm-2 Angstrom-1 arcsec-2)^2"
HISTORY = (
    "Metadata audit: calibrated spectral flux density divided by fiber area during reconstruction.",
    "Metadata audit: fiber radius=0.75 arcsec; fiber area=1.76714586764 arcsec^2.",
    "Metadata audit: output spatial pixel scale=1 arcsec/pixel; Gaussian reconstruction FWHM=1.8 arcsec.",
    "Metadata audit: ERROR = sqrt(VAR).",
    "Metadata audit: VAR is diagonal per-voxel variance only; spatial covariance is not encoded.",
    "Metadata audit: sqrt(sum(VAR)) for large spatial sums neglects covariance.",
    "Metadata audit: one Gaussian beam is approximately 3.67 arcsec^2 = 3.7 pixels; sqrt(3.7)=1.9 is a rule of thumb, not covariance propagation.",
    "Metadata audit: VAR omits additive calibration-model, exposure-sky, exposure-scale, and reconstruction-covariance uncertainties.",
)


def digest(array):
    h = hashlib.sha256()
    for start in range(0, array.shape[0], 4):
        h.update(np.ascontiguousarray(array[start:start + 4]).tobytes())
    return h.hexdigest()


def open_raw(path):
    """Keep unsigned DQ/BZERO images memory-mappable and hash raw array bytes."""
    hdul = fits.open(path, mode="readonly", memmap=True,
                     do_not_scale_image_data=True)
    if len(hdul) != 1 or hdul[0].data is None:
        hdul.close()
        raise ValueError(f"expected one primary image HDU: {path}")
    return hdul


def wavelength(header, n):
    wcs = WCS(header, relax=True)
    pix = np.arange(1, n + 1, dtype=float)
    raw = np.asarray(wcs.wcs_pix2world(np.ones(n), np.ones(n), pix, 1)[2])
    unit = u.Unit(str(wcs.wcs.cunit[2]))
    return raw * unit.to(u.AA)


def repaired_header(header):
    header = header.copy()
    header["CD3_3"] = (2.0, "linear wavelength increment [Angstrom]")
    header.pop("CDELT3", None)
    for text in HISTORY:
        header.add_history(text)
    return header


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", type=Path,
                        default=Path("/work/03730/gregz/maverick/m101_recon"))
    parser.add_argument("--apply", action="store_true",
                        help="rewrite the six files in place; without this, only validate and report")
    args = parser.parse_args()
    base = args.base_dir.expanduser().resolve()

    records = []
    for name in FILES:
        path = base / name
        if not path.exists():
            raise FileNotFoundError(path)
        with open_raw(path) as hdul:
            data = hdul[0].data
            header = hdul[0].header.copy()
            if data.ndim != 3 or data.shape[0] != 1036:
                raise ValueError(f"unexpected shape for {name}: {data.shape}")
            fixed = repaired_header(header)
            test = wavelength(fixed, 1036)
            if not (np.isclose(test[0], 3470.0, atol=1e-8) and
                    np.isclose(test[-1], 5540.0, atol=1e-8) and
                    np.allclose(np.diff(test), 2.0, atol=1e-8)):
                raise ValueError(f"repaired WCS failed for {name}: {test[[0, -1]]}")
            if name == "m101_virus_v2_cube.fits":
                fixed["BUNIT"] = SCI_UNIT
            elif name == "m101_virus_v2_variance_cube.fits":
                fixed["BUNIT"] = VAR_UNIT
            elif name == "m101_virus_v2_errorcube.fits":
                fixed["BUNIT"] = SCI_UNIT
            records.append({"name": name, "path": path, "header": header,
                            "fixed": fixed, "shape": data.shape,
                            "dtype": str(data.dtype), "hash": digest(data),
                            "finite": digest(np.isfinite(data))})

    if args.apply:
        for record in records:
            path = record["path"]
            temp_path = None
            with open_raw(path) as source:
                data = source[0].data
                output = fits.PrimaryHDU(data=data, header=record["fixed"])
                with tempfile.NamedTemporaryFile(prefix=f".{path.name}.",
                                                  suffix=".tmp", dir=path.parent,
                                                  delete=False) as temp:
                    temp_path = Path(temp.name)
                output.writeto(temp_path, overwrite=True)
            try:
                with open_raw(temp_path) as check:
                    data = check[0].data
                    unchanged = (data.shape == record["shape"] and
                                 str(data.dtype) == record["dtype"] and
                                 digest(data) == record["hash"] and
                                 digest(np.isfinite(data)) == record["finite"])
                if not unchanged:
                    raise RuntimeError(f"data verification failed for temporary copy of {path}")
                os.replace(temp_path, path)
            finally:
                if temp_path is not None and temp_path.exists():
                    temp_path.unlink()
            with open_raw(path) as check:
                data = check[0].data
                final_unchanged = (data.shape == record["shape"] and
                                   str(data.dtype) == record["dtype"] and
                                   digest(data) == record["hash"] and
                                   digest(np.isfinite(data)) == record["finite"])
            if not final_unchanged:
                raise RuntimeError(f"data verification failed after editing {path}")

    print("mode:", "APPLIED in place" if args.apply else "DRY RUN (use --apply to edit)")
    print("filename | BUNIT | CRVAL3 | CRPIX3 | CD3_3 | CDELT3? | first_A | last_A | step_A | data unchanged")
    for record in records:
        header = record["fixed"] if args.apply else record["header"]
        wave = wavelength(header, 1036)
        unchanged = ("yes" if args.apply else "not written")
        print(f"{record['name']} | {header.get('BUNIT', '')} | {header.get('CRVAL3')} | "
              f"{header.get('CRPIX3')} | {header.get('CD3_3')} | "
              f"{'yes' if 'CDELT3' in header else 'no'} | "
              f"{wave[0]:.10g} | {wave[-1]:.10g} | {np.median(np.diff(wave)):.10g} | {unchanged}")
    print("\nFinal unit metadata:")
    print("SCI   :", SCI_UNIT)
    print("VAR   :", VAR_UNIT)
    print("ERROR :", SCI_UNIT)


if __name__ == "__main__":
    main()
