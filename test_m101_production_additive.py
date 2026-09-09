import numpy as np

import fit_m101_additive_structure as additive


def _synthetic_groups(seed=19):
    rng = np.random.default_rng(seed)
    exposures = [("one.h5", 1), ("two.h5", 1)]
    ifus = [(412, 13, 43), (412, 13, 44), (412, 13, 45)]
    amps = [ifu + (amp,) for ifu in ifus for amp in additive.AMPS]
    R = {key: rng.normal(size=additive.N_WAVE) for key in exposures}
    U = {key: rng.normal(size=additive.N_WAVE) for key in exposures}
    V = {key: rng.normal(size=additive.N_WAVE) for key in exposures}
    c = {key: rng.normal() for key in ifus}
    d = {key: rng.normal() for key in amps}
    for ifu in ifus:
        center = np.mean([d[ifu + (amp,)] for amp in additive.AMPS])
        for amp in additive.AMPS:
            d[ifu + (amp,)] -= center
    ifu_obs, amp_obs = {}, {}
    for exposure in exposures:
        for ifu in ifus:
            y = R[exposure] + c[ifu] * U[exposure]
            ifu_obs[(exposure, ifu)] = {"y": y, "n": 40}
            for amp in additive.AMPS:
                key = ifu + (amp,)
                amp_obs[(exposure, key)] = {"y": y + d[key] * V[exposure], "n": 20}
    data = {key: {"h5_id": index, "n_blank": 100, "n_ifu": len(ifus), "n_amp": len(amps)}
            for index, key in enumerate(exposures)}
    return exposures, ifus, amps, data, ifu_obs, amp_obs


def test_rank_one_additive_solution_and_unsupported_fallback():
    exposures, ifus, amps, data, ifu_obs, amp_obs = _synthetic_groups()
    unsupported_ifu = (412, 13, 99)
    unsupported_amp = unsupported_ifu + ("LL",)
    R, U, V, c, d, history, converged = additive.solve_structure(
        data, ifu_obs, amp_obs, ifus + [unsupported_ifu], amps + [unsupported_amp], 8, 1e-8)
    assert converged
    assert c[unsupported_ifu] == 0.0
    assert d[unsupported_amp] == 0.0
    for ifu in ifus:
        assert abs(np.mean([d[ifu + (amp,)] for amp in additive.AMPS])) < 1e-12
    residuals = []
    for (exposure, key), record in amp_obs.items():
        residuals.append(record["y"] - R[exposure] - c[key[:3]] * U[exposure] - d[key] * V[exposure])
    assert np.nanmax(np.abs(residuals)) < 1e-8
    assert history[-1]["delta"] < 1e-8


def test_band_collapse_is_linear_and_exact():
    rng = np.random.default_rng(3)
    spectrum_r = rng.normal(size=additive.N_WAVE)
    spectrum_u = rng.normal(size=additive.N_WAVE)
    spectrum_v = rng.normal(size=additive.N_WAVE)
    c, d = .37, -.22
    filters = {"ON": np.linspace(.2, 1.0, additive.N_WAVE),
               "OFF": np.linspace(1.0, .2, additive.N_WAVE)}
    for response in filters.values():
        direct = additive.hm.weighted_scalar(spectrum_r + c * spectrum_u + d * spectrum_v, response)
        separated = (additive.hm.weighted_scalar(spectrum_r, response)
                     + c * additive.hm.weighted_scalar(spectrum_u, response)
                     + d * additive.hm.weighted_scalar(spectrum_v, response))
        assert np.isclose(direct, separated, rtol=0, atol=1e-12)
