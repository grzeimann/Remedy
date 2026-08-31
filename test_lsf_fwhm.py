"""Focused validation for the direct empirical FWHM estimator."""

import ast
from pathlib import Path

import numpy as np
from scipy.interpolate import PchipInterpolator


def _load_measure_direct_fwhm():
    source = Path(__file__).with_name('make_mosaic_cube_org.py').read_text()
    tree = ast.parse(source)
    function = next(node for node in tree.body
                    if isinstance(node, ast.FunctionDef) and
                    node.name == '_measure_direct_fwhm')
    namespace = {'np': np, 'PchipInterpolator': PchipInterpolator}
    exec(compile(ast.Module(body=[function], type_ignores=[]),
                 str(Path(__file__)), 'exec'), namespace)
    return namespace['_measure_direct_fwhm']


_measure_direct_fwhm = _load_measure_direct_fwhm()


def test_quadratic_peak_recovers_subpixel_phase_and_reports_fwhm_variation():
    grid = np.arange(3980.0, 4042.0, 2.0)
    reference = 4010.0
    true_fwhm = 5.4
    true_sigma = true_fwhm / 2.354820045
    phases = np.array([-0.8, -0.4, 0.0, 0.4, 0.8])
    centers = []
    widths = []
    for phase in phases:
        true_center = reference + phase
        profile = 0.25 + np.exp(-0.5 * ((grid - true_center) / true_sigma) ** 2)
        result = _measure_direct_fwhm(grid, profile, reference)
        assert result['valid'], result
        centers.append(result['center'])
        widths.append(result['fwhm'])

    centers = np.asarray(centers)
    widths = np.asarray(widths)
    np.testing.assert_allclose(centers, reference + phases, atol=0.08)
    assert np.ptp(centers) > 1.0
    # This is a diagnostic bound for this smooth synthetic profile, not a
    # correction or a physical VIRUS LSF assumption.
    assert np.ptp(widths) < 0.35
    print('synthetic center recovery:', centers.tolist())
    print('synthetic FWHM values:', widths.tolist())
    print('synthetic FWHM peak-to-peak phase variation:', float(np.ptp(widths)))
