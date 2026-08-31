"""Focused tests for the variance-only Gaussian cube reconstruction change."""

import ast
import warnings
from pathlib import Path

import numpy as np
from numba import njit


def _load_variance_functions():
    """Load target functions without executing the script's CLI pipeline."""
    source = Path(__file__).with_name('make_mosaic_cube_org.py').read_text()
    tree = ast.parse(source)
    wanted = {
        '_gaussian_splat_shot_xy',
        '_classify_error_provenance',
        '_compute_final_variance',
        'make_image_gaussian',
    }
    body = []
    for node in tree.body:
        if (isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and
                node.name in wanted):
            body.append(node)
        elif (isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id.startswith('DQ_')
                for target in node.targets)):
            body.append(node)
    def test_njit(*args, **kwargs):
        # The extracted function has no importable module for Numba's cache.
        kwargs.pop('cache', None)
        return njit(*args, **kwargs)

    namespace = {'np': np, 'njit': test_njit}
    exec(compile(ast.Module(body=body, type_ignores=[]), str(Path(__file__)), 'exec'),
         namespace)
    return (namespace['_classify_error_provenance'],
            namespace['_compute_final_variance'],
            namespace['make_image_gaussian'],
            namespace['DQ_INSUFFICIENT_SUPPORT'],
            namespace['DQ_VAR_INCOMPLETE'],
            namespace['DQ_EMPIRICAL_VAR_USED'],
            namespace['DQ_FORMAL_VAR_USED'],
            namespace['DQ_VAR_EMPIRICAL_ONLY'])


(_classify_error_provenance, _compute_final_variance,
 make_image_gaussian, DQ_INSUFFICIENT_SUPPORT, DQ_VAR_INCOMPLETE,
 DQ_EMPIRICAL_VAR_USED, DQ_FORMAL_VAR_USED,
 DQ_VAR_EMPIRICAL_ONLY) = _load_variance_functions()


def _variance_for(values, variances):
    values = np.asarray(values, dtype=np.float32)[:, None, None]
    variances = np.asarray(variances, dtype=np.float32)[:, None, None]
    ncontrib = np.full((1, 1), len(values), dtype=np.uint8)
    result, dq, stats = _compute_final_variance(values, variances, ncontrib)
    return float(result[0, 0]), int(dq[0, 0]), stats


def test_equal_variance_median_matches_gaussian_approximation():
    sigma = 2.0
    for n in (3, 5, 9):
        # Keep the measurements coincident so the optional empirical term is
        # zero and the formal median expression is the adopted result.
        result, _, _ = _variance_for(np.zeros(n), np.full(n, sigma**2))
        expected = np.pi / 2.0 * sigma**2 / n
        assert np.isclose(result, expected, rtol=1e-6)


def test_two_shot_median_is_exact_mean_variance():
    result, _, _ = _variance_for([1.0, 2.0], [4.0, 9.0])
    assert result == 13.0 / 4.0


def test_heteroscedastic_median_matches_analytic_expression():
    variances = np.array([1.0, 4.0, 9.0])
    sigmas = np.sqrt(variances)
    expected = 3.0 * np.pi / (2.0 * np.sum(1.0 / sigmas) ** 2)
    result, _, _ = _variance_for([1.0, 2.0, 3.0], variances)
    assert np.isclose(result, expected, rtol=1e-6)


def test_excess_shot_scatter_is_adopted_over_formal_variance():
    result, dq, stats = _variance_for([-10.0, -5.0, 0.0, 5.0, 10.0], np.ones(5))
    assert stats['empirical_exceeds_formal'] == 1
    assert result > 1.0
    assert dq & int(DQ_EMPIRICAL_VAR_USED)
    assert not dq & int(DQ_FORMAL_VAR_USED)


def test_formal_variance_provenance_is_exclusive():
    result, dq, _ = _variance_for(np.zeros(5), np.ones(5))
    assert result > 0.0
    assert dq & int(DQ_FORMAL_VAR_USED)
    assert not dq & int(DQ_EMPIRICAL_VAR_USED)
    assert not dq & int(DQ_VAR_EMPIRICAL_ONLY)
    assert not dq & int(DQ_VAR_INCOMPLETE)


def test_empirical_only_variance_provenance():
    result, dq, _ = _variance_for([-10.0, -5.0, 0.0, 5.0, 10.0], [np.nan] * 5)
    assert result > 0.0
    assert dq & int(DQ_VAR_EMPIRICAL_ONLY)
    assert not dq & int(DQ_FORMAL_VAR_USED)
    assert not dq & int(DQ_VAR_INCOMPLETE)


def test_no_variance_keeps_valid_sci_nan_and_sets_incomplete():
    result, dq, _ = _variance_for([1.0, 2.0], [np.nan, np.nan])
    assert np.isnan(result)
    assert dq & int(DQ_VAR_INCOMPLETE)
    assert not dq & int(DQ_INSUFFICIENT_SUPPORT)


def test_insufficient_support_is_invalid_and_exclusive():
    xg = np.arange(1.0, 7.0)
    yg = np.arange(1.0, 7.0)
    xgrid, ygrid = np.meshgrid(xg, yg)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        image, variance, _, ncontrib, dq, _ = make_image_gaussian(
            np.array([[3.1, 3.1]]), np.array([1.0], dtype=np.float32),
            np.array([1.0], dtype=np.float32), xg, yg, xgrid, ygrid,
            1.8 / 2.35, [(0, 1)])
    assert image[3, 3] == 0.0
    assert variance[3, 3] == 0.0
    assert ncontrib[3, 3] == 1
    assert dq[3, 3] & int(DQ_INSUFFICIENT_SUPPORT)
    assert not dq[3, 3] & int(DQ_VAR_INCOMPLETE)


def test_missing_fiber_error_preserves_sci_and_marks_shot_variance_incomplete():
    xg = np.arange(1.0, 7.0)
    yg = np.arange(1.0, 7.0)
    xgrid, ygrid = np.meshgrid(xg, yg)
    positions = np.array([
        [3.1, 3.1], [3.2, 3.2],
        [3.1, 3.1], [3.2, 3.2],
    ])
    flux = np.array([10.0, 12.0, 10.0, 12.0], dtype=np.float32)
    complete_errors = np.ones(4, dtype=np.float32)
    incomplete_errors = complete_errors.copy()
    incomplete_errors[0] = np.nan
    shots = [(0, 2), (2, 4)]

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        complete = make_image_gaussian(
            positions, flux, complete_errors, xg, yg, xgrid, ygrid,
            1.8 / 2.35, shots)
        incomplete = make_image_gaussian(
            positions, flux, incomplete_errors, xg, yg, xgrid, ygrid,
            1.8 / 2.35, shots)

    # This is the before/after variance-only comparison: changing only fiber
    # error validity cannot change the reconstructed SCI or support count.
    np.testing.assert_array_equal(complete[0], incomplete[0])
    np.testing.assert_array_equal(complete[3], incomplete[3])
    assert np.isnan(incomplete[1][3, 3])
    assert np.isfinite(complete[1][3, 3])
    assert incomplete[4][3, 3] & int(DQ_VAR_INCOMPLETE)
    assert not incomplete[4][3, 3] & int(DQ_FORMAL_VAR_USED)


def test_zero_and_negative_errors_preserve_sci_and_are_provenanced():
    xg = np.arange(1.0, 7.0)
    yg = np.arange(1.0, 7.0)
    xgrid, ygrid = np.meshgrid(xg, yg)
    positions = np.array([
        [3.1, 3.1], [3.2, 3.2],
        [3.1, 3.1], [3.2, 3.2],
    ])
    flux = np.array([10.0, 12.0, 10.0, 12.0], dtype=np.float32)
    shots = [(0, 2), (2, 4)]
    for bad_error, category in ((0.0, 3), (-1.0, 4)):
        errors = np.ones(4, dtype=np.float32)
        errors[0] = bad_error
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            result = make_image_gaussian(
                positions, flux, errors, xg, yg, xgrid, ygrid,
                1.8 / 2.35, shots)
        assert result[0][3, 3] > 0.0
        assert result[3][3, 3] == 2
        assert np.isnan(result[1][3, 3])
        assert result[4][3, 3] & int(DQ_VAR_INCOMPLETE)
        counts = _classify_error_provenance(flux, errors)
        assert counts[category] == 1


def test_error_provenance_categories_are_distinct_and_exhaustive():
    science = np.ones(4, dtype=np.float32)
    errors = np.array([1.0, np.nan, 0.0, -1.0], dtype=np.float32)
    counts = _classify_error_provenance(science, errors)
    assert counts.tolist() == [4, 1, 1, 1, 1, 0]
