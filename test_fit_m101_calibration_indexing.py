import numpy as np

import fit_m101_calibration as calibration


def test_indexed_coefficient_and_exposure_selection_matches_bruteforce():
    rng = np.random.default_rng(41)
    exposures = [("a.h5", 1), ("b.h5", 2), ("c.h5", 3)]
    hardware = [(1, 2, 3), (1, 2, 4), (1, 2, 5)]
    records = []
    residuals = []
    templates = {exposure: rng.normal(size=calibration.N_WAVE) for exposure in exposures}
    slopes = dict(zip(hardware, (-0.7, 0.25, 1.3)))
    for exposure_index, exposure in enumerate(exposures):
        for hardware_index, key in enumerate(hardware):
            records.append({"exposure": exposure, "hardware": key,
                            "n": 15 + exposure_index + hardware_index})
            residual = slopes[key] * templates[exposure] + rng.normal(0, .01, calibration.N_WAVE)
            residual[::113] = np.nan
            residuals.append(residual)

    by_hardware, by_exposure, record_exposures, record_counts = calibration.build_record_indices(records)
    indexed = calibration.fit_coefficients(records, templates, residuals, hardware,
                                           by_hardware, record_exposures, record_counts)

    brute_force = {}
    for key in hardware:
        y, x, counts = [], [], []
        for record, residual in zip(records, residuals):
            if record["hardware"] != key:
                continue
            template = templates[record["exposure"]]
            use = np.isfinite(residual) & np.isfinite(template)
            y.extend(residual[use]); x.extend(template[use])
            counts.extend([record["n"]] * int(np.sum(use)))
        brute_force[key] = calibration.robust_slope(y, x, counts)

    assert np.allclose([indexed[key] for key in hardware],
                       [brute_force[key] for key in hardware], rtol=0, atol=1e-14)

    coefficients = {key: float(index + 1) for index, key in enumerate(hardware)}
    for exposure in exposures:
        indices = by_exposure[exposure]
        indexed_template = calibration.fit_template(records, coefficients, residuals, indices=indices)
        selected_records = [records[index] for index in indices]
        selected_residuals = [residuals[index] for index in indices]
        brute_template = calibration.fit_template(selected_records, coefficients, selected_residuals)
        assert np.allclose(indexed_template, brute_template, rtol=0, atol=1e-14, equal_nan=True)
