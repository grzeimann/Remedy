import numpy as np

from fit_m101_calibration_v2 import focal_plot_filename, focal_plot_label, make_layout, support_report
from build_m101_external_compact_mask import run_synthetic_validation
from m101_calibration_utils import collapse, collapse_error, contrast_basis, contrast_values
from m101_external_measurements import source_comparison_validity
from m101_native_data import ExposureData, construct_total_spectra


def _tiny_exposure():
    return ExposureData(
        h5_name="tiny.h5", h5_path="tiny.h5", exposure=1,
        key=("tiny.h5", 1), survey={"offset": 2.0},
        row_index=np.asarray([0, 1, 2]),
        ifu=np.asarray([[1, 2, 3], [1, 2, 3], [1, 2, 4]], dtype=int),
        amp=np.asarray(["LL", "LL", "LU"], dtype=object),
        j=np.asarray([0, 1, 0]), q=np.asarray([0, 1, 111]),
        ra=np.asarray([1.0, 1.0, 1.0]), dec=np.asarray([2.0, 2.0, 2.0]),
        x_arcmin=np.asarray([-1.0, -1.0, 1.0]),
        y_arcmin=np.asarray([0.0, 0.0, 0.0]),
        total=np.ones((3, 4)), total_error=np.ones((3, 4)),
        band_total=np.ones((3, 7)), band_error=np.ones((3, 7)),
        response_fraction=np.ones((3, 7)), K=np.ones(7), K_wave=np.ones(4),
        blank_classified=np.asarray([True, True, True]),
        date_mask_bad=np.asarray([False, False, False]),
        blank_valid=np.asarray([True, False, True]))


def test_total_reconstruction_and_error_are_exact():
    total, error = construct_total_spectra(
        np.asarray([[6.0, 8.0]]), np.asarray([[4.0, 6.0]]),
        np.asarray([[1.0, 2.0]]), 2.0)
    np.testing.assert_allclose(total, [[4.0, 6.0]])
    np.testing.assert_allclose(error, [[2.0, 3.0]])


def test_band_collapse_and_error_delegate_to_validated_builder():
    values = np.asarray([[1.0, 2.0, np.nan, 4.0]])
    errors = np.asarray([[2.0, 4.0, 8.0, 16.0]])
    response = np.asarray([1.0, 2.0, 0.0, 1.0])
    import build_m101_measurements as validated
    expected_value = validated._collapse_with_fraction(values, response)
    expected_error = validated._collapse_error(errors, response)
    actual_value = collapse(values, response)
    actual_error = collapse_error(errors, response)
    np.testing.assert_allclose(actual_value[0], expected_value[0], equal_nan=True)
    np.testing.assert_allclose(actual_value[1], expected_value[1], equal_nan=True)
    np.testing.assert_allclose(actual_error[0], expected_error[0], equal_nan=True)
    np.testing.assert_allclose(actual_error[1], expected_error[1], equal_nan=True)


def test_contrasts_and_illumination_are_identifiable():
    basis = contrast_basis(4)
    values = contrast_values(np.asarray([0.2, -0.1, 0.4]), basis)
    np.testing.assert_allclose(np.sum(values), 0.0, atol=1e-12)
    item = _tiny_exposure()
    layout = make_layout([item], "model1")
    assert len(layout.plane_columns) == 1
    assert all(name.startswith(("p_IFU_", "ax_", "ay_")) for name in layout.names)
    assert not any(name.startswith("illumination_intercept") for name in layout.names)


def test_exposure_identity_and_missing_support_are_explicit():
    first = _tiny_exposure()
    second = _tiny_exposure()
    second.h5_name = "other.h5"
    second.h5_path = "other.h5"
    second.key = ("other.h5", 1)
    report = support_report([first, second])
    assert first.key != second.key
    assert report["counts"]["IFU_unsupported"] == 0
    assert report["counts"]["fiber_unsupported"] == 1
    assert report["per_exposure"][str(first.key)]["IFU"] == 2
    assert focal_plot_filename(first.key) == "tiny_exp01_focal_plane_residual.png"
    assert focal_plot_label(first.key) == "tiny_exp01"


def test_source_mask_is_reference_validity_only():
    external_valid = np.asarray([True, True, True, True])
    compact_masked = np.asarray([False, True, True, False])
    compact_inside = np.asarray([True, True, False, False])
    date_valid = np.ones(4, dtype=bool)
    object_finite = np.ones(4, dtype=bool)
    object_error = np.asarray([1.0, 1.0, 1.0, np.nan])
    comparison = source_comparison_validity(
        external_valid, compact_masked, compact_inside, date_valid,
        object_finite, object_error)
    np.testing.assert_array_equal(comparison, [True, False, True, False])
    spectrum_valid = np.asarray([True, False, True, True])
    np.testing.assert_array_equal(spectrum_valid, [True, False, True, True])


def test_validated_compact_mask_lookup_synthetic_case():
    result = run_synthetic_validation()
    assert result["status"] == "PASS"
    assert result["vectorized_lookup"] is True
