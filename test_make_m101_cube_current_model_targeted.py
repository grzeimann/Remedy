import csv
from types import SimpleNamespace

import numpy as np
from astropy import units as u
from astropy.io import fits
from astropy.units import Unit
from astropy.wcs import WCS

import make_m101_cube_current_model as cube


def _summary():
    return {
        "preflight": {
            "groups_with_frozen_M": 0,
            "groups_without_frozen_M_masked": 0,
        },
        "model": {"historical_f0": 1.0},
        "qa": {
            "reject_file_used": False,
            "rejection": {"rejected_groups": 0, "rejected_fiber_rows": 0},
        },
        "spectral_rejection": {
            "spectral_reject_file_used": True,
            "rejected_group_wavelength_entries": 1,
            "rejected_fiber_wavelength_cells": 2,
        },
        "lsf": {"enabled": False},
    }


def _provenance():
    return {
        name: {"path": "%s.dat" % name}
        for name in ("state", "alpha_fits", "alpha_history", "amp_means")
    }


def _group(exposure=1, specid=322, ifuslot=84, ifuid=62, amp="LU",
           indices=(0, 1)):
    return {
        "exposure": exposure,
        "specid": specid,
        "ifuslot": ifuslot,
        "ifuid": ifuid,
        "amp": amp,
        "indices": np.asarray(indices, dtype=int),
    }


def test_reviewed_csvs_parse_and_4882_maps_to_index_706(tmp_path):
    spectral_path = tmp_path / "m101_targeted_spectral_reject.csv"
    spectral_path.write_text(
        "H5,exposure,SPECID,IFUSLOT,IFUID,AMP,wavelength_A\n"
        "20200525_0000022.h5,1,322,84,62,LU,4882\n"
        "20200523_0000024.h5,1,322,84,62,LU,4882\n"
        "20200521_0000019.h5,1,322,84,62,LU,4882\n"
        "20200430_0000021.h5,2,322,84,62,LU,4882\n"
        "20200521_0000019.h5,2,322,84,62,LU,4882\n"
        "20200622_0000015.h5,3,322,84,62,LU,4882\n")
    with spectral_path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 6
    keys = {(row["H5"], int(row["exposure"]), int(row["SPECID"]),
             int(row["IFUSLOT"]), int(row["IFUID"]), row["AMP"].upper(),
             float(row["wavelength_A"])) for row in rows}
    assert len(keys) == 6
    entries, loaded_path = cube._load_spectral_reject_file(spectral_path)
    assert loaded_path == spectral_path.resolve()
    assert len(entries) == 6
    assert {entry[5] for entry in entries} == {"LU"}
    assert {entry[6] for entry in entries} == {706}
    assert cube.DEF_WAVE[706] == 4882.0

    qa_path = tmp_path / "m101_targeted_qa_reject.csv"
    qa_path.write_text(
        "H5,exposure,SPECID,IFUSLOT,IFUID,AMP\n"
        "20200710_0000013.h5,1,401,79,68,RL\n"
        "20200710_0000013.h5,2,401,79,68,RL\n"
        "20200710_0000013.h5,3,401,79,68,RL\n"
        "20200710_0000014.h5,1,401,79,68,RL\n"
        "20200710_0000014.h5,2,401,79,68,RL\n"
        "20200710_0000014.h5,3,401,79,68,RL\n")
    qa_keys, loaded_qa_path = cube._load_qa_reject_file(qa_path)
    assert loaded_qa_path == qa_path.resolve()
    assert len(qa_keys) == 6
    assert len({key for key in qa_keys}) == 6
    assert {key[-1] for key in qa_keys} == {"RL"}


def test_spectral_rejection_changes_only_requested_cells():
    spectra = np.arange(12, dtype=float).reshape(3, 4)
    errors = spectra + 100.0
    original_spectra = spectra.copy()
    original_errors = errors.copy()
    record = {
        "h5file": SimpleNamespace(name="example.h5"),
        "spectra": spectra,
        "errors": errors,
        "groups": [_group(indices=(0, 1)), _group(ifuid=63, indices=(2,))],
    }
    entry = ("example.h5", 1, 322, 84, 62, "LU", 2)
    result = cube._apply_spectral_rejection([record], {entry})
    second_result = cube._apply_spectral_rejection([record], {entry})

    assert result == {
        "rejected_group_wavelength_entries": 1,
        "rejected_fiber_wavelength_cells": 2,
        "unique_physical_amplifiers": 1,
        "wavelengths_A": [float(cube.DEF_WAVE[2])],
    }
    assert second_result == result
    assert np.isnan(spectra[[0, 1], 2]).all()
    assert np.isnan(errors[[0, 1], 2]).all()
    assert np.array_equal(spectra[:, [0, 1, 3]], original_spectra[:, [0, 1, 3]])
    assert np.array_equal(errors[:, [0, 1, 3]], original_errors[:, [0, 1, 3]])
    assert np.array_equal(spectra[2], original_spectra[2])
    assert np.array_equal(errors[2], original_errors[2])


def test_whole_group_qa_rejection_remains_unchanged():
    spectra = np.arange(8, dtype=float).reshape(2, 4)
    errors = spectra + 10.0
    record = {
        "h5file": SimpleNamespace(name="example.h5"),
        "spectra": spectra,
        "errors": errors,
        "groups": [_group(indices=(0, 1))],
    }
    key = ("example.h5", 1, 322, 84, 62, "LU")
    qa_rows = [{
        "H5": "example.h5", "exposure": 1, "SPECID": 322,
        "IFUSLOT": 84, "IFUID": 62, "AMP": "LU",
    }]
    result = cube._apply_qa_rejection([record], qa_rows, {key})
    assert result == {"rejected_groups": 1, "rejected_fiber_rows": 2}
    assert np.isnan(spectra).all()
    assert np.isnan(errors).all()
    assert qa_rows[0]["QA_STATUS"] == "REJECT"


def test_output_wcs_and_physical_units(tmp_path, monkeypatch):
    class SpatialWCS:
        def to_header(self):
            return fits.Header({
                "WCSAXES": 2,
                "CRPIX1": 1.0, "CRPIX2": 1.0,
                "CRVAL1": 210.8, "CRVAL2": 54.333,
                "CDELT1": -1.0 / 3600.0, "CDELT2": 1.0 / 3600.0,
            })

    monkeypatch.chdir(tmp_path)
    shape = (len(cube.DEF_WAVE), 2, 2)
    cube._write_cube_products(
        "targeted", np.zeros(shape, dtype=np.float32),
        np.ones(shape, dtype=np.float32),
        np.zeros(shape, dtype=np.uint16), np.ones(shape, dtype=np.float32),
        np.ones(shape, dtype=np.uint8), SpatialWCS(), 2, 1.0,
        _provenance(), _summary())

    header = fits.getheader("targeted_cube.fits")
    assert "CDELT3" not in header
    assert header["CD3_3"] == 2.0
    assert header["CTYPE3"] == "WAVE"
    assert header["CUNIT3"] == "Angstrom"
    assert header["SPECSYS"] == "TOPOCENT"
    wcs = WCS(header)
    wavelengths = wcs.all_pix2world(
        [[1.0, 1.0, 1.0], [1.0, 1.0, 2.0], [1.0, 1.0, 1036.0]], 1)[:, 2]
    wavelengths = (wavelengths * wcs.wcs.cunit[2]).to_value(u.Angstrom)
    np.testing.assert_allclose(wavelengths, [3470.0, 3472.0, 5540.0],
                               rtol=0.0, atol=1.e-10)
    assert len(cube.DEF_WAVE) == 1036
    assert header["BUNIT"] == cube.SCI_BUNIT
    assert Unit(header["BUNIT"]) == Unit(cube.SCI_BUNIT)
    assert Unit(fits.getheader("targeted_variance_cube.fits")["BUNIT"]) == Unit(cube.VAR_BUNIT)
    assert Unit(fits.getheader("targeted_errorcube.fits")["BUNIT"]) == Unit(cube.SCI_BUNIT)
    for filename in ("targeted_cube.fits", "targeted_variance_cube.fits",
                     "targeted_errorcube.fits", "targeted_dq_cube.fits",
                     "targeted_weight_cube.fits", "targeted_ncontrib_cube.fits"):
        product_header = fits.getheader(filename)
        assert product_header["CD3_3"] == 2.0
        assert any("Spatial covariance introduced by reconstruction" in text
                   for text in product_header.get("HISTORY", []))
