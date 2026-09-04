#!/usr/bin/env python3
"""First hierarchical Bayesian inference from the frozen M101 measurements.

The frozen HDF5 measurements are not globally regressed fiber-by-fiber.  For
each physical amplifier observation, its 112 fibers and two bands define the
local model

    T = exp(z) P + p + alpha H,
    T = D + B,  P = X + B,  H = K*f(q).

The two additive parameters are marginalized locally with a 2x2 Bayesian
linear-regression calculation.  Each amplifier is then compressed to a
probabilistic message for ``z = ln(m)``.  Those messages update the sparse,
contrast-parameterized hierarchy

    z = gamma_exposure + iota_exposure_IFU + eta_persistent_amplifier.

Quality probability and multiplicative information are deliberately kept
separate.  This script reads the compact measurement product only; it never
opens original spectra, reconstructs a cube, changes masks, or applies a
production calibration solution.
"""

from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
import csv
import json
from pathlib import Path
import pickle
import subprocess
import time
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tables
from scipy import linalg, sparse
from scipy.optimize import least_squares
from scipy.special import logsumexp
from scipy.sparse import linalg as sparse_linalg
from scipy.stats import norm, multivariate_normal, spearmanr

import diagnose_m101_hierarchical as hm


BANDS = ("ON", "OFF")
N_BANDS = 2
BAND_SIGN = np.array([1.0, -1.0])
N_AMP_FIBERS = 112
N_Z_DEFAULT = 161
Z_MIN_DEFAULT = -1.2
Z_MAX_DEFAULT = 1.2
PI_GOOD_DEFAULT = 0.98
BAD_SCALE_DEFAULT = 10.0
Z0_SIGMA_DEFAULT = 0.50
P_SIGMA_FRACTION_DEFAULT = 0.10
ALPHA_MEAN_DEFAULT = 0.40
ALPHA_SIGMA_DEFAULT = 0.20
GAMMA_SIGMA_DEFAULT = 0.25
IFU_SIGMA_DEFAULT = 0.10
ETA_SIGMA_DEFAULT = 0.10
EDGE_MASS_LIMIT = 1e-3
HUTCHINSON_PROBES_DEFAULT = 64
ERROR_BOOTSTRAP_DEFAULT = 300
EVIDENCE_SCHEMA = "m101_amplifier_evidence_v3_split_alpha"


@dataclass(frozen=True)
class PhysicalIFUKey:
    specid: int
    ifuslot: int
    ifuid: int


@dataclass(frozen=True)
class PhysicalAmplifierKey:
    ifu: PhysicalIFUKey
    amp: str


@dataclass(frozen=True)
class AmplifierObservationKey:
    h5_id: int
    exposure: int
    ifu: PhysicalIFUKey
    amp: str


@dataclass
class AmplifierBlock:
    key: AmplifierObservationKey
    h5_name: str
    ra: float
    dec: float
    effective_ra: np.ndarray
    effective_dec: np.ndarray
    D: np.ndarray
    B: np.ndarray
    X: np.ndarray
    error: np.ndarray
    external_valid: np.ndarray
    date_mask_bad: bool
    fq: np.ndarray
    K: np.ndarray
    q: np.ndarray
    sky_scale: float

    @property
    def T(self):
        return self.D + self.B

    @property
    def P(self):
        return self.X + self.B

    @property
    def H(self):
        return self.fq[:, None] * self.K[None, :]

    @property
    def primitive_valid(self):
        return ((not self.date_mask_bad) & self.external_valid &
                np.isfinite(self.D) & np.isfinite(self.B) &
                np.isfinite(self.X) & np.isfinite(self.error))

    @property
    def likelihood_valid(self):
        return self.primitive_valid & (self.error > 0.0)


@dataclass
class MarginalGrid:
    log_m: np.ndarray
    beta_mean: np.ndarray
    beta_cov: np.ndarray
    log_evidence: float


@dataclass
class AmplifierEvidence:
    key: AmplifierObservationKey
    h5_name: str
    ra: float
    dec: float
    x_amp: np.ndarray
    median_x: np.ndarray
    n_valid: np.ndarray
    local_z_mean: float
    local_z_sigma: float
    local_z_skew: float
    local_m_mean: float
    p_mean: float
    p_sigma: float
    alpha_mean: float
    alpha_sigma: float
    rho_z_p: float
    rho_z_alpha: float
    rho_p_alpha: float
    p_information: float
    alpha_information: float
    I_m: float
    p_good: float
    site_tau: float
    site_nu: float
    site_z_hat: float
    site_sigma: float
    noninformative_site: bool
    grid_edge_flag: bool
    split_minus_joint_log_evidence: float
    split_z_mean: np.ndarray
    split_z_sigma: np.ndarray
    split_p_mean: np.ndarray
    split_p_sigma: np.ndarray
    split_alpha_mean: np.ndarray
    split_alpha_sigma: np.ndarray
    split_grid_edge: np.ndarray
    split_delta_z: float
    split_delta_p: float
    split_delta_alpha: float
    split_delta_alpha_sigma: float
    split_delta_alpha_significance: float
    log_m_good: np.ndarray
    log_m_bad: np.ndarray
    log_m_total: np.ndarray
    beta_mean_z: np.ndarray
    beta_cov_integrated: np.ndarray
    p_sigma_prior: float
    error_floor_used: float


@dataclass
class GlobalLayout:
    gamma_index: dict
    ifu_index: dict
    eta_index: dict
    ifu_order: dict
    parameter_count: int

    def design(self, block_key):
        return _layout_design(self, block_key)


@dataclass
class CalibrationPosterior:
    layout: GlobalLayout
    mean: np.ndarray
    variance: np.ndarray
    factor: object
    Q: sparse.csr_matrix
    h: np.ndarray
    variance_method: str


def hm_helmert(n):
    """Return deterministic orthonormal zero-sum basis columns."""
    return linalg.helmert(int(n), full=False).T


def _text(value):
    if isinstance(value, (bytes, np.bytes_)):
        return value.decode("utf-8", errors="replace").strip()
    return str(value).strip()


def _finite(values):
    values = np.asarray(values, dtype=float)
    return values[np.isfinite(values)]


def _safe_float(value, default=np.nan):
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    return value if np.isfinite(value) else default


def _fmt(value):
    if isinstance(value, (np.bool_, bool)):
        return int(value)
    if isinstance(value, np.generic):
        return value.item()
    return value


def _write_csv(path, rows, fields=None):
    path = Path(path)
    if fields is None:
        fields = list(rows[0].keys()) if rows else []
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _fmt(row.get(field, "")) for field in fields})


def _h5_identity(path):
    path = Path(path)
    stat = path.stat()
    return {"path": str(path.resolve()), "size": int(stat.st_size),
            "mtime_ns": int(stat.st_mtime_ns)}


def _git_commit():
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent,
                                capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unavailable"


class MeasurementStore:
    """Load the compact measurement columns once and group amplifier blocks."""

    REQUIRED = ("h5_id", "exposure", "original_h5_row", "SPECID", "IFUSLOT",
                "IFUID", "AMP", "j", "q", "fq", "RA", "Dec", "effective_RA",
                "effective_Dec", "data_work", "sky", "external_prediction",
                "error_work", "external_valid", "date_mask_bad")

    def __init__(self, path):
        self.path = Path(path).expanduser().resolve()
        if not self.path.exists():
            raise ValueError("measurement product does not exist: %s" % self.path)
        self.blocks = []
        self.h5_names = {}
        self.exposure_rows = {}
        self.exposure_scales = {}
        self._load()

    def _load(self):
        with tables.open_file(self.path, mode="r") as h5:
            version = _text(getattr(h5.root._v_attrs, "schema_version", ""))
            if not version.startswith("m101_measurements_v1"):
                raise ValueError("unsupported measurement schema: %s" % version)
            band_order = _text(getattr(h5.root._v_attrs, "band_order", ""))
            if json.loads(band_order) != list(BANDS):
                raise ValueError("measurement band_order is not [ON, OFF]")
            if "/measurements" not in h5 or "/exposure_band" not in h5:
                raise ValueError("measurement H5 lacks /measurements or /exposure_band")
            table = h5.root.measurements
            missing = set(self.REQUIRED) - set(table.colnames)
            if missing:
                raise ValueError("measurement table lacks columns: %s" % sorted(missing))
            if "/provenance/h5_inputs" in h5:
                # Materialize the table: PyTables Row iteration reuses a
                # mutable row buffer, which is unsafe to retain in a list.
                input_rows = h5.root.provenance.h5_inputs.read()
                measurement_h5_ids = np.unique(np.asarray(table.cols.h5_id[:], dtype=int))
                input_ids = [int(row["h5_id"]) for row in input_rows]
                if len(set(input_ids)) == len(input_rows):
                    self.h5_names = {int(row["h5_id"]): _text(row["filename"])
                                     for row in input_rows}
                elif (len(input_rows) == len(measurement_h5_ids)
                      and len(set(input_ids)) == 1):
                    # Older/current builder products can contain the correct
                    # ordered filename lookup but leave the h5_id field at
                    # its default zero value.  The measurement and
                    # exposure_band tables provide the authoritative ID set;
                    # map the validated lookup rows by their stored order.
                    self.h5_names = {int(h5_id): _text(row["filename"])
                                     for h5_id, row in zip(sorted(measurement_h5_ids), input_rows)}
                else:
                    raise ValueError("/provenance/h5_inputs has inconsistent h5_id values")
            if not self.h5_names:
                raise ValueError("measurement H5 lacks /provenance/h5_inputs")
            eb = h5.root.exposure_band
            for row in eb:
                key = (int(row["h5_id"]), int(row["exposure"]))
                self.exposure_rows[key] = {
                    "K": np.asarray(row["K"], dtype=float),
                    "g": np.asarray(row["g_global"], dtype=float),
                }
            if self.exposure_rows:
                self.g_global = np.nanmedian(
                    np.vstack([value["g"] for value in self.exposure_rows.values()]), axis=0)
            else:
                raise ValueError("measurement H5 has no exposure_band metadata")
            arrays = {name: np.asarray(getattr(table.cols, name)[:])
                      for name in self.REQUIRED}
            arrays["AMP"] = np.asarray([_text(value) for value in arrays["AMP"]])
        self.nrows = len(arrays["h5_id"])
        if self.nrows == 0:
            raise ValueError("measurement table is empty")
        self._group_arrays(arrays)

    def _group_arrays(self, a):
        h5_id = np.asarray(a["h5_id"], dtype=int)
        exposure = np.asarray(a["exposure"], dtype=int)
        specid = np.asarray(a["SPECID"], dtype=int)
        slot = np.asarray(a["IFUSLOT"], dtype=int)
        uid = np.asarray(a["IFUID"], dtype=int)
        amp = np.asarray(a["AMP"])
        j = np.asarray(a["j"], dtype=int)
        amp_code = np.asarray([{"LL": 0, "LU": 1, "RL": 2, "RU": 3}.get(v, 99)
                               for v in amp], dtype=int)
        order = np.lexsort((j, amp_code, uid, slot, specid, exposure, h5_id))
        sorted_key = np.column_stack((h5_id[order], exposure[order], specid[order],
                                       slot[order], uid[order], amp_code[order]))
        boundaries = np.flatnonzero(np.any(sorted_key[1:] != sorted_key[:-1], axis=1)) + 1
        starts = np.r_[0, boundaries]
        stops = np.r_[boundaries, len(order)]
        amplifier_group_count = 0
        for start, stop in zip(starts, stops):
            indices = order[start:stop]
            if len(indices) != N_AMP_FIBERS:
                raise ValueError("physical exposure/IFUSLOT/AMP group has %d rows, expected 112" % len(indices))
            j_values = j[indices]
            if not np.array_equal(j_values, np.arange(N_AMP_FIBERS)):
                raise ValueError("native amplifier group does not contain j=0..111")
            amp_name = amp[indices[0]]
            if amp_name not in ("LL", "LU", "RL", "RU"):
                raise ValueError("unexpected amplifier orientation: %s" % amp_name)
            if len(set(a["q"][indices].tolist())) != N_AMP_FIBERS:
                raise ValueError("q is not one-to-one in amplifier group")
            q_expected = np.arange(N_AMP_FIBERS) if amp_name in ("LL", "RU") \
                else np.arange(N_AMP_FIBERS - 1, -1, -1)
            if not np.array_equal(np.asarray(a["q"][indices], dtype=int), q_expected):
                raise ValueError("q orientation mismatch in %s" % amp_name)
            if len(set(specid[indices].tolist())) != 1 or len(set(uid[indices].tolist())) != 1:
                raise ValueError("SPECID/IFUID is inconsistent within amplifier group")
            h5_value, exp_value = int(h5_id[indices[0]]), int(exposure[indices[0]])
            if (h5_value, exp_value) not in self.exposure_rows:
                raise ValueError("missing exposure_band metadata for h5=%d exposure=%d" %
                                 (h5_value, exp_value))
            if arrays_shape(a["effective_RA"][indices]) != (N_AMP_FIBERS, N_BANDS):
                raise ValueError("effective_RA does not have shape (112,2)")
            key = AmplifierObservationKey(
                h5_value, exp_value,
                PhysicalIFUKey(int(specid[indices[0]]), int(slot[indices[0]]), int(uid[indices[0]])),
                amp_name)
            block = AmplifierBlock(
                key=key, h5_name=self.h5_names[h5_value],
                ra=float(np.nanmean(a["RA"][indices])),
                dec=float(np.nanmean(a["Dec"][indices])),
                effective_ra=np.asarray(a["effective_RA"][indices][0], dtype=float),
                effective_dec=np.asarray(a["effective_Dec"][indices][0], dtype=float),
                D=np.asarray(a["data_work"][indices], dtype=float),
                B=np.asarray(a["sky"][indices], dtype=float),
                X=np.asarray(a["external_prediction"][indices], dtype=float),
                error=np.asarray(a["error_work"][indices], dtype=float),
                external_valid=np.asarray(a["external_valid"][indices], dtype=bool),
                date_mask_bad=bool(np.any(a["date_mask_bad"][indices])),
                fq=np.asarray(a["fq"][indices], dtype=float),
                K=np.asarray(self.exposure_rows[(h5_value, exp_value)]["K"], dtype=float),
                q=np.asarray(a["q"][indices], dtype=int),
                sky_scale=np.nan,
            )
            self.blocks.append(block)
            amplifier_group_count += 1
        self.amplifier_group_count = amplifier_group_count
        self._calculate_exposure_scales()

    def _calculate_exposure_scales(self):
        grouped = {}
        for block in self.blocks:
            grouped.setdefault((block.key.h5_id, block.key.exposure), []).append(block)
        for key, blocks in grouped.items():
            values = np.concatenate([block.B[block.primitive_valid] for block in blocks])
            absolute = np.abs(values)
            # The median is the robust location used here; unlike the
            # biweight helper it is well-defined for an all-zero sky sample.
            scale = float(np.nanmedian(absolute)) if absolute.size else np.nan
            if not np.isfinite(scale) or scale <= 0:
                scale = np.nanmedian(absolute) if absolute.size else np.nan
            scale = max(float(scale), 1e-12) if np.isfinite(scale) else 1e-12
            self.exposure_scales[key] = scale
            for block in blocks:
                block.sky_scale = scale
        self.block_count = len(self.blocks)


def arrays_shape(value):
    return tuple(np.asarray(value).shape)


def _trapezoid_log_integral(log_density, z_grid):
    dz = float(z_grid[1] - z_grid[0])
    weights = np.ones(len(z_grid), dtype=float) * dz
    weights[[0, -1]] *= 0.5
    return float(logsumexp(np.asarray(log_density) + np.log(weights)))


def _normal_logpdf_grid(z_grid, mean, sigma):
    return -0.5 * ((z_grid - mean) / sigma) ** 2 - np.log(sigma * np.sqrt(2 * np.pi))


def _marginal_grid(T, P, H, sigma, beta_mean, beta_sigma, z_grid):
    """Marginalized log likelihood and beta|z using only 2x2 operations."""
    T, P, H, sigma = [np.asarray(value, dtype=float) for value in (T, P, H, sigma)]
    if H.ndim == 1:
        H = H[:, None]
    A = np.column_stack((np.ones(len(T)), H))
    prior_var = np.asarray(beta_sigma, dtype=float) ** 2
    prior_precision = np.diag(1.0 / prior_var)
    beta_mean = np.asarray(beta_mean, dtype=float)
    if len(T) == 0:
        cov = np.diag(prior_var)
        beta = np.repeat(beta_mean[None, :], len(z_grid), axis=0)
        return MarginalGrid(np.zeros(len(z_grid)), beta, cov, 0.0)
    w = 1.0 / (sigma * sigma)
    Q = prior_precision + A.T @ (w[:, None] * A)
    cov = np.linalg.inv(Q)
    sign, logdet_q = np.linalg.slogdet(Q)
    if sign <= 0:
        raise ValueError("local beta posterior precision is not positive definite")
    logdet_c = 2.0 * np.sum(np.log(sigma)) + np.sum(np.log(prior_var)) + logdet_q
    r0 = T - A @ beta_mean
    s00 = float(np.sum(w * r0 * r0))
    s01 = float(np.sum(w * r0 * P))
    s11 = float(np.sum(w * P * P))
    b0 = A.T @ (w * r0)
    bp = A.T @ (w * P)
    ez = np.exp(z_grid)
    u_tw_u = s00 - 2.0 * ez * s01 + ez * ez * s11
    b = b0[:, None] - bp[:, None] * ez[None, :]
    correction = np.einsum("iz,ij,jz->z", b, cov, b)
    quad = np.maximum(u_tw_u - correction, 0.0)
    log_m = -0.5 * (len(T) * np.log(2.0 * np.pi) + logdet_c + quad)
    beta_posterior = beta_mean[None, :] + (cov @ b).T
    return MarginalGrid(log_m, beta_posterior, cov,
                        _trapezoid_log_integral(log_m, z_grid))


def _conditional_beta(T, P, H, sigma, z, beta_mean, beta_sigma):
    """Return beta|z,data for the nominal local model."""
    T, P, H, sigma = [np.asarray(value, dtype=float) for value in (T, P, H, sigma)]
    A = np.column_stack((np.ones(len(T)), H))
    prior_var = np.asarray(beta_sigma, dtype=float) ** 2
    Q = np.diag(1.0 / prior_var) + A.T @ ((1.0 / sigma**2)[:, None] * A)
    cov = np.linalg.inv(Q)
    residual = T - np.exp(float(z)) * P - A @ np.asarray(beta_mean, dtype=float)
    mean = np.asarray(beta_mean, dtype=float) + cov @ (A.T @ (residual / sigma**2))
    return mean, cov


def _band_contrast_transform(block, delta_z_band=0.0, delta_p_band=0.0):
    """Return effective T/P plus the symmetric ON/OFF contrast terms."""
    band_scale = np.exp(0.5 * float(delta_z_band) * BAND_SIGN)
    band_offset = 0.5 * float(delta_p_band) * BAND_SIGN
    return (block.T - band_offset[None, :],
            block.P * band_scale[None, :], band_scale, band_offset)


def _split_band_fits(block, z_grid, alpha_mean, alpha_sigma, p_sigma_prior,
                     error_floor=None, error_floor_factor=None,
                     delta_z_band=0.0, delta_p_band=0.0,
                     z0_sigma=Z0_SIGMA_DEFAULT):
    """Fit the existing local model separately in each band for QA."""
    T_effective, P_effective, _, _ = _band_contrast_transform(
        block, delta_z_band, delta_p_band)
    beta_prior_mean = np.asarray([0.0, alpha_mean])
    beta_prior_sigma = np.asarray([p_sigma_prior, alpha_sigma])
    z_means, z_sigmas, p_means, p_sigmas = [], [], [], []
    alpha_means, alpha_sigmas, edges, log_evidence = [], [], [], 0.0
    for band_index in range(N_BANDS):
        selected = block.likelihood_valid[:, band_index]
        sigma = block.error[:, band_index][selected]
        if error_floor is not None:
            sigma = np.maximum(sigma, float(error_floor))
        if error_floor_factor is not None and sigma.size:
            sigma = np.maximum(sigma, float(error_floor_factor) * float(np.median(sigma)))
        split = _marginal_grid(T_effective[:, band_index][selected],
                               P_effective[:, band_index][selected],
                               block.H[:, band_index][selected], sigma,
                               beta_prior_mean, beta_prior_sigma, z_grid)
        log_q0 = _normal_logpdf_grid(z_grid, 0.0, z0_sigma)
        log_z = _trapezoid_log_integral(split.log_m + log_q0, z_grid)
        log_evidence += log_z
        dz = float(z_grid[1] - z_grid[0])
        weights = np.ones(len(z_grid), dtype=float) * dz
        weights[[0, -1]] *= 0.5
        mass = np.exp(split.log_m + log_q0 + np.log(weights) - log_z)
        mass /= np.sum(mass)
        mean_z = float(np.sum(mass * z_grid))
        variance_z = max(float(np.sum(mass * (z_grid - mean_z) ** 2)), dz * dz / 12.0)
        beta_mean = np.sum(mass[:, None] * split.beta_mean, axis=0)
        beta_second = split.beta_cov + np.einsum("zi,zj->zij", split.beta_mean, split.beta_mean)
        beta_cov = np.sum(mass[:, None, None] * beta_second, axis=0) - np.outer(beta_mean, beta_mean)
        z_means.append(mean_z); z_sigmas.append(np.sqrt(variance_z))
        p_means.append(float(beta_mean[0])); p_sigmas.append(np.sqrt(max(float(beta_cov[0, 0]), 0.0)))
        alpha_means.append(float(beta_mean[1])); alpha_sigmas.append(np.sqrt(max(float(beta_cov[1, 1]), 0.0)))
        edges.append(bool(mass[0] + mass[-1] > EDGE_MASS_LIMIT))
    return {"z_mean": np.asarray(z_means), "z_sigma": np.asarray(z_sigmas),
            "p_mean": np.asarray(p_means), "p_sigma": np.asarray(p_sigmas),
            "alpha_mean": np.asarray(alpha_means), "alpha_sigma": np.asarray(alpha_sigmas),
            "grid_edge": np.asarray(edges, dtype=bool),
            "log_evidence": float(log_evidence)}


def _moment_information(mean, variance, prior_mean, prior_variance):
    if not np.isfinite(mean) or not np.isfinite(variance) or variance <= 0:
        return 0.0
    return float(0.5 * (np.log(prior_variance / variance) +
                        (variance + (mean - prior_mean) ** 2) / prior_variance - 1.0))


def _local_evidence(block, z_grid, pi_good=PI_GOOD_DEFAULT,
                    bad_scale=BAD_SCALE_DEFAULT,
                    p_sigma_fraction=P_SIGMA_FRACTION_DEFAULT,
                    alpha_mean=ALPHA_MEAN_DEFAULT,
                    alpha_sigma=ALPHA_SIGMA_DEFAULT,
                    z0_sigma=Z0_SIGMA_DEFAULT,
                    error_floor=None, error_floor_factor=None,
                    delta_z_band=0.0, delta_p_band=0.0,
                    split_summary=None):
    primitive = block.primitive_valid
    likelihood = block.likelihood_valid
    valid_by_band = np.sum(likelihood, axis=0).astype(int)
    x_amp = np.asarray([np.sum(block.X[:, i][primitive[:, i]]) for i in range(N_BANDS)])
    median_x = np.asarray([np.nanmedian(block.X[:, i][primitive[:, i]]) if np.any(primitive[:, i]) else np.nan
                           for i in range(N_BANDS)])
    p_sigma_prior = max(float(p_sigma_fraction) * float(block.sky_scale), 1e-12)
    sigma = block.error[likelihood]
    if error_floor is not None:
        sigma = np.maximum(sigma, float(error_floor))
    floor_used = float(error_floor) if error_floor is not None else 0.0
    if error_floor_factor is not None and sigma.size:
        factor_floor = float(error_floor_factor) * float(np.median(sigma))
        sigma = np.maximum(sigma, factor_floor)
        floor_used = factor_floor
    T_effective, P_effective, _, _ = _band_contrast_transform(
        block, delta_z_band, delta_p_band)
    T = T_effective[likelihood]
    P = P_effective[likelihood]
    H = block.H[likelihood]
    beta_prior_mean = np.asarray([0.0, alpha_mean])
    beta_prior_sigma = np.asarray([p_sigma_prior, alpha_sigma])
    good = _marginal_grid(T, P, H, sigma, beta_prior_mean, beta_prior_sigma, z_grid)
    bad = _marginal_grid(T, P, H, sigma * float(bad_scale), beta_prior_mean,
                         beta_prior_sigma, z_grid)
    log_q0 = _normal_logpdf_grid(z_grid, 0.0, z0_sigma)
    log_total = logsumexp(np.vstack((np.log(pi_good) + good.log_m,
                                     np.log(1.0 - pi_good) + bad.log_m)), axis=0)
    log_z_total = _trapezoid_log_integral(log_total + log_q0, z_grid)
    log_z_good = _trapezoid_log_integral(good.log_m + log_q0, z_grid)
    log_z_bad = _trapezoid_log_integral(bad.log_m + log_q0, z_grid)
    log_p_good_model = np.log(pi_good) + log_z_good
    log_p_bad_model = np.log(1.0 - pi_good) + log_z_bad
    p_good = float(np.exp(log_p_good_model - logsumexp((log_p_good_model, log_p_bad_model))))
    dz = float(z_grid[1] - z_grid[0])
    grid_weights = np.ones(len(z_grid)) * dz
    grid_weights[[0, -1]] *= 0.5
    log_mass = log_total + log_q0 + np.log(grid_weights) - log_z_total
    mass = np.exp(log_mass)
    mass /= np.sum(mass)
    mean_z = float(np.sum(mass * z_grid))
    centered = z_grid - mean_z
    variance_z = float(np.sum(mass * centered * centered))
    # A grid posterior concentrated in one cell has zero discrete variance,
    # even though the unresolved continuous posterior has finite width.  Use
    # the uniform-within-cell variance as a transparent numerical floor; this
    # also prevents an unresolved site from making the sparse hierarchy
    # ill-conditioned.
    variance_z = max(variance_z, dz * dz / 12.0)
    sigma_z = np.sqrt(max(variance_z, 0.0))
    skew = float(np.sum(mass * centered**3) / sigma_z**3) if sigma_z > 0 else 0.0
    q_density_log = log_total + log_q0 - log_z_total
    I_m = float(np.sum(mass * (q_density_log - log_q0)))
    edge = bool(mass[0] + mass[-1] > EDGE_MASS_LIMIT)
    quality_weight = np.exp(np.log(pi_good) + good.log_m - log_total)
    beta_mean_z = quality_weight[:, None] * good.beta_mean + (1.0 - quality_weight[:, None]) * bad.beta_mean
    beta_second = np.zeros((len(z_grid), 2, 2), dtype=float)
    for index in range(len(z_grid)):
        mg, mb = good.beta_mean[index], bad.beta_mean[index]
        beta_second[index] = (
            quality_weight[index] * (good.beta_cov + np.outer(mg, mg)) +
            (1.0 - quality_weight[index]) * (bad.beta_cov + np.outer(mb, mb)))
    beta_mean = np.sum(mass[:, None] * beta_mean_z, axis=0)
    beta_cov_integrated = np.sum(mass[:, None, None] * beta_second, axis=0) - np.outer(beta_mean, beta_mean)
    beta_cov_integrated = 0.5 * (beta_cov_integrated + beta_cov_integrated.T)
    p_sigma = np.sqrt(max(float(beta_cov_integrated[0, 0]), 0.0))
    alpha_sigma_post = np.sqrt(max(float(beta_cov_integrated[1, 1]), 0.0))
    rho_z_p = float(np.sum(mass * (z_grid - mean_z) * (beta_mean_z[:, 0] - beta_mean[0])) /
                    (sigma_z * p_sigma)) if sigma_z > 0 and p_sigma > 0 else 0.0
    rho_z_alpha = float(np.sum(mass * (z_grid - mean_z) * (beta_mean_z[:, 1] - beta_mean[1])) /
                        (sigma_z * alpha_sigma_post)) if sigma_z > 0 and alpha_sigma_post > 0 else 0.0
    rho_p_alpha = float(beta_cov_integrated[0, 1] / (p_sigma * alpha_sigma_post)) \
        if p_sigma > 0 and alpha_sigma_post > 0 else 0.0
    # Preserve the preliminary uncorrected split-band QA, while evaluating
    # the split-versus-joint evidence for the current (possibly corrected)
    # effective T/P convention.
    if split_summary is None:
        preliminary_split = _split_band_fits(
            block, z_grid, alpha_mean, alpha_sigma, p_sigma_prior,
            error_floor, error_floor_factor, z0_sigma=z0_sigma)
    else:
        preliminary_split = split_summary
    split_delta_alpha = float(preliminary_split["alpha_mean"][0] - preliminary_split["alpha_mean"][1])
    split_delta_alpha_sigma = float(np.hypot(preliminary_split["alpha_sigma"][0],
                                             preliminary_split["alpha_sigma"][1]))
    split_delta_alpha_significance = (
        split_delta_alpha / split_delta_alpha_sigma
        if np.isfinite(split_delta_alpha_sigma) and split_delta_alpha_sigma > 0
        else np.nan)
    final_split = _split_band_fits(
        block, z_grid, alpha_mean, alpha_sigma, p_sigma_prior,
        error_floor, error_floor_factor, delta_z_band, delta_p_band,
        z0_sigma)
    split_minus_joint = float(final_split["log_evidence"] - log_z_good)
    site_tau = 1.0 / variance_z - 1.0 / (z0_sigma * z0_sigma) if variance_z > 0 else 0.0
    site_nu = mean_z / variance_z if variance_z > 0 else 0.0
    noninformative = bool(not np.isfinite(site_tau) or site_tau <= 0)
    if noninformative:
        site_tau, site_nu, site_z_hat, site_sigma = 0.0, 0.0, np.nan, np.inf
    else:
        site_z_hat, site_sigma = site_nu / site_tau, np.sqrt(1.0 / site_tau)
    return AmplifierEvidence(
        key=block.key, h5_name=block.h5_name, ra=block.ra, dec=block.dec,
        x_amp=x_amp, median_x=median_x, n_valid=valid_by_band,
        local_z_mean=mean_z, local_z_sigma=sigma_z, local_z_skew=skew,
        local_m_mean=float(np.sum(mass * np.exp(z_grid))),
        p_mean=float(beta_mean[0]), p_sigma=p_sigma,
        alpha_mean=float(beta_mean[1]), alpha_sigma=alpha_sigma_post,
        rho_z_p=rho_z_p, rho_z_alpha=rho_z_alpha, rho_p_alpha=rho_p_alpha,
        p_information=_moment_information(beta_mean[0], beta_cov_integrated[0, 0], 0.0, p_sigma_prior**2),
        alpha_information=_moment_information(beta_mean[1], beta_cov_integrated[1, 1], alpha_mean, alpha_sigma**2),
        I_m=I_m, p_good=p_good, site_tau=float(site_tau), site_nu=float(site_nu),
        site_z_hat=float(site_z_hat), site_sigma=float(site_sigma),
        noninformative_site=noninformative, grid_edge_flag=edge,
        split_minus_joint_log_evidence=split_minus_joint,
        split_z_mean=np.asarray(preliminary_split["z_mean"]),
        split_z_sigma=np.asarray(preliminary_split["z_sigma"]),
        split_p_mean=np.asarray(preliminary_split["p_mean"]),
        split_p_sigma=np.asarray(preliminary_split["p_sigma"]),
        split_alpha_mean=np.asarray(preliminary_split["alpha_mean"]),
        split_alpha_sigma=np.asarray(preliminary_split["alpha_sigma"]),
        split_grid_edge=np.asarray(preliminary_split["grid_edge"], dtype=bool),
        split_delta_z=float(preliminary_split["z_mean"][0] - preliminary_split["z_mean"][1]),
        split_delta_p=float(preliminary_split["p_mean"][0] - preliminary_split["p_mean"][1]),
        split_delta_alpha=split_delta_alpha,
        split_delta_alpha_sigma=split_delta_alpha_sigma,
        split_delta_alpha_significance=split_delta_alpha_significance,
        log_m_good=good.log_m, log_m_bad=bad.log_m, log_m_total=log_total,
        beta_mean_z=beta_mean_z, beta_cov_integrated=beta_cov_integrated,
        p_sigma_prior=p_sigma_prior, error_floor_used=floor_used)


def _local_evidence_worker(task):
    """Pickleable process-pool wrapper; the scientific calculation stays above."""
    block, z_grid, settings = task
    return _local_evidence(block, z_grid, *settings)


def _robust_location_scale(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan, np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        location = float(hm.robust_location(values))
        scale = float(hm.robust_scale(values)) if values.size > 1 else np.nan
    if not np.isfinite(location):
        location = float(np.median(values))
    if not np.isfinite(scale):
        scale = 1.4826 * float(np.median(np.abs(values - np.median(values))))
    return location, scale


def _preliminary_qa(evidences):
    total = len(evidences)
    preference = np.asarray([e.split_minus_joint_log_evidence > 5.0 for e in evidences], dtype=bool)
    low_half = np.asarray([e.p_good < 0.5 for e in evidences], dtype=bool)
    low_tenth = np.asarray([e.p_good < 0.1 for e in evidences], dtype=bool)
    return {"n": total,
            "split_preference_count": int(np.sum(preference)),
            "split_preference_fraction": float(np.mean(preference)) if total else np.nan,
            "p_good_lt_0p5_count": int(np.sum(low_half)),
            "p_good_lt_0p5_fraction": float(np.mean(low_half)) if total else np.nan,
            "p_good_lt_0p1_count": int(np.sum(low_tenth)),
            "p_good_lt_0p1_fraction": float(np.mean(low_tenth)) if total else np.nan}


def _estimate_band_contrast(evidences):
    usable = np.asarray([
        np.isfinite(e.split_delta_z) and np.isfinite(e.split_delta_p)
        and np.all(np.isfinite(e.split_z_mean)) and np.all(np.isfinite(e.split_p_mean))
        and np.all(np.isfinite(e.split_z_sigma)) and np.all(np.isfinite(e.split_p_sigma))
        and np.all(e.n_valid > 0) and not np.any(e.split_grid_edge)
        for e in evidences], dtype=bool)
    delta_z_values = np.asarray([e.split_delta_z for e in evidences])[usable]
    delta_p_values = np.asarray([e.split_delta_p for e in evidences])[usable]
    delta_z, delta_z_scale = _robust_location_scale(delta_z_values)
    delta_p, delta_p_scale = _robust_location_scale(delta_p_values)
    group_locations = []
    for key in sorted(set((e.key.h5_id, e.key.exposure) for e in evidences)):
        values = [e.split_delta_z for e, use in zip(evidences, usable)
                  if use and (e.key.h5_id, e.key.exposure) == key]
        location, _ = _robust_location_scale(values)
        if np.isfinite(location):
            group_locations.append(location)
    drift_range = float(np.max(group_locations) - np.min(group_locations)) if group_locations else np.nan
    drift_reference = max(float(delta_z_scale), 1e-12) if np.isfinite(delta_z_scale) else np.nan
    return {"delta_z_band": delta_z, "delta_z_robust_scatter": delta_z_scale,
            "delta_z_n_used": int(delta_z_values.size), "delta_p_band": delta_p,
            "delta_p_robust_scatter": delta_p_scale, "delta_p_n_used": int(delta_p_values.size),
            "median_split_delta_z": float(np.median(delta_z_values)) if delta_z_values.size else np.nan,
            "median_split_delta_p": float(np.median(delta_p_values)) if delta_p_values.size else np.nan,
            "split_delta_z_group_range": drift_range,
            "split_delta_z_obvious_drift": bool(np.isfinite(drift_range) and drift_range > 2.0 * drift_reference),
            "n_split_usable": int(np.sum(usable)),
            "n_split_total": len(evidences)}


def _complete_band_contrast(estimate, store):
    delta_z = float(estimate["delta_z_band"])
    delta_p = float(estimate["delta_p_band"])
    scale = np.exp(0.5 * delta_z * BAND_SIGN)
    offset = 0.5 * delta_p * BAND_SIGN
    result = dict(estimate)
    result.update({"baseline_g_ON": float(store.g_global[0]),
                   "baseline_g_OFF": float(store.g_global[1]),
                   "band_scale_ON": float(scale[0]), "band_scale_OFF": float(scale[1]),
                   "band_offset_ON": float(offset[0]), "band_offset_OFF": float(offset[1]),
                   "multiplicative_factor_ON": float(scale[0]),
                   "multiplicative_factor_OFF": float(scale[1]),
                   "effective_g_ON": float(store.g_global[0] * scale[0]),
                   "effective_g_OFF": float(store.g_global[1] * scale[1])})
    return result


def _distribution_summary(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"N": 0, "median": np.nan, "robust_location": np.nan, "robust_scale": np.nan,
                "p16": np.nan, "p50": np.nan, "p84": np.nan, "p05": np.nan, "p95": np.nan}
    location, scale = _robust_location_scale(values)
    p05, p16, p50, p84, p95 = np.percentile(values, (5, 16, 50, 84, 95))
    return {"N": int(values.size), "median": float(np.median(values)),
            "robust_location": location, "robust_scale": scale,
            "p16": float(p16), "p50": float(p50), "p84": float(p84),
            "p05": float(p05), "p95": float(p95)}


def _spearman(x, y):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    if np.sum(valid) < 3 or np.ptp(x[valid]) == 0 or np.ptp(y[valid]) == 0:
        return np.nan
    return float(spearmanr(x[valid], y[valid]).statistic)


def _split_alpha_diagnostic(evidences):
    delta = np.asarray([e.split_delta_alpha for e in evidences], dtype=float)
    absolute = np.abs(delta)
    significance = np.asarray([e.split_delta_alpha_significance for e in evidences], dtype=float)
    split_evidence = np.asarray([e.split_minus_joint_log_evidence for e in evidences], dtype=float)
    strong = split_evidence > 5.0
    finite = np.isfinite(delta) & np.isfinite(significance)
    summaries = {}
    for label, mask in (("all", finite), ("strong_split", finite & strong),
                        ("nonstrong_split", finite & ~strong)):
        summaries[label] = {"delta_alpha": _distribution_summary(delta[mask]),
                            "abs_delta_alpha": _distribution_summary(absolute[mask]),
                            "significance": _distribution_summary(significance[mask]),
                            "fraction_abs_significance_gt_2": float(np.mean(np.abs(significance[mask]) > 2)) if np.any(mask) else np.nan,
                            "fraction_abs_significance_gt_3": float(np.mean(np.abs(significance[mask]) > 3)) if np.any(mask) else np.nan,
                            "fraction_abs_significance_gt_5": float(np.mean(np.abs(significance[mask]) > 5)) if np.any(mask) else np.nan}
    amp_summary = {}
    for amp in ("LL", "LU", "RL", "RU"):
        mask = finite & np.asarray([e.key.amp == amp for e in evidences])
        amp_summary[amp] = _distribution_summary(delta[mask])
    ifu_groups = {}
    for evidence in evidences:
        if np.isfinite(evidence.split_delta_alpha):
            ifu_groups.setdefault(evidence.key.ifu, []).append(evidence.split_delta_alpha)
    ifu_medians = [(key, float(np.median(values))) for key, values in ifu_groups.items() if len(values) >= 2]
    ifu_medians.sort(key=lambda item: abs(item[1]), reverse=True)
    ifu_values = np.asarray([value for _, value in ifu_medians])
    extreme_ifus = [{"SPECID": key.specid, "IFUSLOT": key.ifuslot, "IFUID": key.ifuid,
                     "median_split_delta_alpha": value, "N": len(ifu_groups[key])}
                    for key, value in ifu_medians[:8]]
    exposure_groups = {}
    for evidence in evidences:
        if np.isfinite(evidence.split_delta_alpha):
            exposure_groups.setdefault((evidence.key.h5_id, evidence.key.exposure), []).append(evidence.split_delta_alpha)
    exposure_summary = [{"h5_id": key[0], "exposure": key[1], "N": len(values),
                         "median_split_delta_alpha": float(np.median(values))}
                        for key, values in sorted(exposure_groups.items())]
    exposure_values = np.asarray([row["median_split_delta_alpha"] for row in exposure_summary])
    _, overall_scale = _robust_location_scale(delta[finite])
    exposure_range = float(np.ptp(exposure_values)) if exposure_values.size else np.nan
    amp_locations = [value["robust_location"] for value in amp_summary.values() if np.isfinite(value["robust_location"])]
    amp_range = float(np.ptp(amp_locations)) if amp_locations else np.nan
    strong_location = summaries["strong_split"]["delta_alpha"]["robust_location"]
    nonstrong_location = summaries["nonstrong_split"]["delta_alpha"]["robust_location"]
    global_shift = abs(strong_location - nonstrong_location) if np.isfinite(strong_location) and np.isfinite(nonstrong_location) else np.nan
    structured = ((np.isfinite(exposure_range) and np.isfinite(overall_scale) and exposure_range > 2 * max(overall_scale, 1e-12))
                  or (np.isfinite(amp_range) and np.isfinite(overall_scale) and amp_range > 2 * max(overall_scale, 1e-12)))
    coherent_global = (np.isfinite(global_shift) and np.isfinite(overall_scale)
                       and global_shift > max(0.5 * overall_scale, 0.02)
                       and summaries["strong_split"]["delta_alpha"]["robust_scale"] < max(2 * overall_scale, 0.05))
    if coherent_global and not structured:
        interpretation = "global alpha contrast"
    elif structured:
        interpretation = "structured alpha contrast"
    else:
        interpretation = "no simple alpha explanation"
    source_support = np.asarray([np.hypot(e.x_amp[0], e.x_amp[1]) for e in evidences])
    return {"finite_count": int(np.sum(finite)), "median_delta_alpha": summaries["all"]["delta_alpha"]["median"],
            "robust_delta_alpha_location": summaries["all"]["delta_alpha"]["robust_location"],
            "robust_delta_alpha_scale": summaries["all"]["delta_alpha"]["robust_scale"],
            "strong_split_count": int(np.sum(strong)),
            "strong_split_median_delta_alpha": summaries["strong_split"]["delta_alpha"]["median"],
            "strong_split_robust_scale": summaries["strong_split"]["delta_alpha"]["robust_scale"],
            "nonstrong_split_median_delta_alpha": summaries["nonstrong_split"]["delta_alpha"]["median"],
            "nonstrong_split_robust_scale": summaries["nonstrong_split"]["delta_alpha"]["robust_scale"],
            "fraction_abs_significance_gt_2": summaries["all"]["fraction_abs_significance_gt_2"],
            "fraction_abs_significance_gt_3": summaries["all"]["fraction_abs_significance_gt_3"],
            "fraction_abs_significance_gt_5": summaries["all"]["fraction_abs_significance_gt_5"],
            "strong_split_fraction_abs_significance_gt_3": summaries["strong_split"]["fraction_abs_significance_gt_3"],
            "nonstrong_split_fraction_abs_significance_gt_3": summaries["nonstrong_split"]["fraction_abs_significance_gt_3"],
            "spearman_delta_alpha_vs_split_evidence": _spearman(delta, split_evidence),
            "spearman_abs_delta_alpha_vs_split_evidence": _spearman(absolute, split_evidence),
            "spearman_delta_alpha_vs_source_support": _spearman(delta, source_support),
            "spearman_delta_alpha_vs_I_m": _spearman(delta, np.asarray([e.I_m for e in evidences])),
            "by_population": summaries, "by_amp": amp_summary,
            "physical_ifu_sufficient_count": int(len(ifu_medians)),
            "physical_ifu_median_distribution": _distribution_summary(ifu_values),
            "extreme_physical_ifus": extreme_ifus,
            "exposure_h5_summary": exposure_summary,
            "exposure_h5_median_range": exposure_range,
            "exposure_h5_obvious_drift": bool(np.isfinite(exposure_range) and np.isfinite(overall_scale)
                                               and exposure_range > 2 * max(overall_scale, 1e-12)),
            "interpretation": interpretation,
            "interpretation_rule": "structured if AMP or H5/exposure median range exceeds 2 robust scales; otherwise global only if strong/non-strong shift exceeds max(0.5 scale, 0.02) with concentrated strong population; else no simple explanation"}


def _error_binned_stats(x, u, n_bins=16, edges=None):
    """Return transparent equal-count-bin summaries for normalized residuals."""
    x, u = np.asarray(x, dtype=float), np.asarray(u, dtype=float)
    valid = np.isfinite(x) & np.isfinite(u) & (x >= 0)
    x, u = x[valid], u[valid]
    if x.size == 0:
        return [], np.asarray([])
    if edges is None:
        edges = np.unique(np.quantile(x, np.linspace(0.0, 1.0, int(n_bins) + 1)))
        if edges.size < 2:
            value = float(edges[0])
            edges = np.asarray([value - 0.5, value + 0.5])
    else:
        edges = np.asarray(edges, dtype=float)
    bin_index = np.searchsorted(edges, x, side="right") - 1
    bin_index = np.clip(bin_index, 0, len(edges) - 2)
    result = []
    for index in range(len(edges) - 1):
        selected = bin_index == index
        if not np.any(selected):
            continue
        values = u[selected]
        result.append({"N": int(values.size), "median_x": float(np.median(x[selected])),
                       "median_x2": float(np.median(x[selected] ** 2)),
                       "RMS_u": float(np.sqrt(np.mean(values ** 2))),
                       "robust_scale_u": float(hm.robust_scale(values)),
                       "median_u": float(np.median(values)),
                       "p16_u": float(np.percentile(values, 16)),
                       "p50_u": float(np.percentile(values, 50)),
                       "p84_u": float(np.percentile(values, 84)),
                       "x_low": float(edges[index]), "x_high": float(edges[index + 1])})
    return result, edges


def _error_variance_fit(stats, fixed_intercept=False):
    """Robustly fit variance versus source-to-noise squared using binned robust scale."""
    usable = [row for row in stats if row["N"] > 0 and np.isfinite(row["median_x2"])
              and np.isfinite(row["RMS_u"]) and row["RMS_u"] >= 0]
    if len(usable) < 2:
        return {"c0": 1.0 if fixed_intercept else np.nan,
                "sqrt_c0": 1.0 if fixed_intercept else np.nan,
                "f_model": np.nan, "n_bins": len(usable)}
    x2 = np.asarray([row["median_x2"] for row in usable])
    y = np.asarray([row["robust_scale_u"] ** 2 if np.isfinite(row.get("robust_scale_u", np.nan))
                    else row["RMS_u"] ** 2 for row in usable])
    slope = (y[-1] - y[0]) / (x2[-1] - x2[0]) if x2[-1] > x2[0] else 0.0
    initial_f = np.sqrt(max(float(slope), 0.0))
    if fixed_intercept:
        def residual(parameters):
            return 1.0 + parameters[0] ** 2 * x2 - y
        result = least_squares(residual, [max(initial_f, 1e-4)], bounds=([0.0], [10.0]),
                               loss="soft_l1", f_scale=max(float(np.median(np.abs(y - 1.0))), 0.1))
        c0, f_model = 1.0, float(result.x[0])
    else:
        def residual(parameters):
            return parameters[0] + parameters[1] ** 2 * x2 - y
        initial_c0 = max(float(np.percentile(y, 10)), 1e-6)
        result = least_squares(residual, [initial_c0, max(initial_f, 1e-4)],
                               bounds=([0.0, 0.0], [100.0, 10.0]), loss="soft_l1",
                               f_scale=max(float(np.median(np.abs(y - np.median(y)))), 0.1))
        c0, f_model = float(result.x[0]), float(result.x[1])
    return {"c0": c0, "sqrt_c0": float(np.sqrt(c0)), "f_model": f_model,
            "n_bins": len(usable)}


def _error_bootstrap_fit(clusters, edges, bin_x2, fixed_intercept, n_bootstrap, rng):
    """Cluster bootstrap of binned robust variance with fixed full-sample centers."""
    n_clusters = len(clusters)
    n_bins = max(len(edges) - 1, 1)
    if n_clusters == 0:
        return {"c0": np.asarray([]), "f_model": np.asarray([])}
    bin_values, bin_cluster_ids = [[] for _ in range(n_bins)], [[] for _ in range(n_bins)]
    for cluster_index, (x, u) in enumerate(clusters):
        x, u = np.asarray(x), np.asarray(u)
        valid = np.isfinite(x) & np.isfinite(u) & (x >= 0)
        if not np.any(valid):
            continue
        bin_index = np.clip(np.searchsorted(edges, x[valid], side="right") - 1, 0, n_bins - 1)
        for index in np.unique(bin_index):
            selected = bin_index == index
            bin_values[index].append(u[valid][selected])
            bin_cluster_ids[index].append(np.full(np.sum(selected), cluster_index, dtype=int))
    sorted_bins = []
    for values, cluster_ids in zip(bin_values, bin_cluster_ids):
        if values:
            values = np.concatenate(values); cluster_ids = np.concatenate(cluster_ids)
            center = float(np.median(values))
            order = np.argsort(np.abs(values - center))
            sorted_bins.append((np.abs(values[order] - center), values[order], cluster_ids[order]))
        else:
            sorted_bins.append((np.asarray([]), np.asarray([]), np.asarray([], dtype=int)))
    c0_values, f_values = [], []
    for _ in range(int(n_bootstrap)):
        selected = rng.integers(0, n_clusters, size=n_clusters)
        draw_counts = np.bincount(selected, minlength=n_clusters).astype(float)
        stats = []
        for index, (deviations, values, cluster_ids) in enumerate(sorted_bins):
            if not values.size:
                continue
            weights = draw_counts[cluster_ids]
            total = float(np.sum(weights))
            if total <= 0:
                continue
            cumulative = np.cumsum(weights)
            mad = float(deviations[np.searchsorted(cumulative, 0.5 * total, side="left")])
            stats.append({"N": int(total), "bin_index": index, "median_x2": float(bin_x2[index]),
                          "RMS_u": float(np.sqrt(np.sum(weights * values ** 2) / total)),
                          "robust_scale_u": float(1.4826 * mad) if mad > 0 else 1e-12})
        # The x-bin locations are fixed from the full sample; bootstrap only
        # resamples amplifier observations and all of their fiber residuals.
        fit = _error_variance_fit(stats, fixed_intercept)
        if np.isfinite(fit["f_model"]):
            f_values.append(fit["f_model"])
            if not fixed_intercept:
                c0_values.append(fit["c0"])
    return {"c0": np.asarray(c0_values), "f_model": np.asarray(f_values)}


def _error_fit_bootstrap_summary(values):
    return _distribution_summary(values)


def _error_model_scan(dataset, candidate_factors):
    result = []
    x, u = dataset["x"], dataset["u"]
    for factor in candidate_factors:
        effective = u / np.sqrt(1.0 + (float(factor) * x) ** 2)
        stats, _ = _error_binned_stats(x, effective, edges=dataset["edges"])
        valid = np.isfinite(effective)
        result.append({"f_model": float(factor), "RMS_u_eff": float(np.sqrt(np.mean(effective[valid] ** 2))),
                       "robust_scale_u_eff": float(hm.robust_scale(effective[valid])),
                       "median_abs_u_eff": float(np.median(np.abs(effective[valid]))),
                       "fraction_abs_gt_2": float(np.mean(np.abs(effective[valid]) > 2)),
                       "fraction_abs_gt_3": float(np.mean(np.abs(effective[valid]) > 3)),
                       "fraction_abs_gt_5": float(np.mean(np.abs(effective[valid]) > 5)),
                       "spearman_abs_u_eff_vs_x": _spearman(np.abs(effective), x),
                       "binned": stats})
    return result


def _error_model_diagnostic(store, evidences, obs_rows, band_contrast,
                            n_bootstrap=ERROR_BOOTSTRAP_DEFAULT):
    """QA-only residual/error analysis using the already computed final posterior."""
    clusters = {"ON": [], "OFF": [], "combined": []}
    rows = []
    for evidence, block, posterior_row in zip(evidences, store.blocks, obs_rows):
        _, P_effective, band_scale, band_offset = _band_contrast_transform(
            block, band_contrast["delta_z_band"], band_contrast["delta_p_band"])
        prediction = (np.exp(posterior_row["posterior_z_mean"]) * P_effective
                      + posterior_row["p_mean"] + posterior_row["alpha_mean"] * block.H
                      + band_offset[None, :])
        residual = block.T - prediction
        source = np.exp(posterior_row["posterior_z_mean"]) * band_scale[None, :] * block.X
        sigma = np.asarray(block.error, dtype=float).copy()
        if evidence.error_floor_used > 0:
            sigma = np.maximum(sigma, evidence.error_floor_used)
        per_band = []
        source_amp, source_to_noise = [], []
        for band in range(N_BANDS):
            valid = block.likelihood_valid[:, band] & np.isfinite(residual[:, band]) & np.isfinite(source[:, band])
            valid &= np.isfinite(sigma[:, band]) & (sigma[:, band] > 0)
            u = residual[:, band][valid] / sigma[:, band][valid]
            x = np.abs(source[:, band][valid]) / sigma[:, band][valid]
            clusters[BANDS[band]].append((x, u))
            source_amp.append(float(np.sum(source[:, band][valid])) if np.any(valid) else np.nan)
            source_to_noise.append(float(abs(np.sum(source[:, band][valid])) /
                                       np.sqrt(np.sum(sigma[:, band][valid] ** 2))) if np.any(valid) else np.nan)
            if u.size:
                order = np.argsort(block.q[valid]); ordered = u[order]
                lag1 = (float(np.corrcoef(ordered[:-1], ordered[1:])[0, 1])
                        if ordered.size >= 3 and np.std(ordered[:-1]) > 0 and np.std(ordered[1:]) > 0
                        else np.nan)
                variance = float(np.var(ordered))
                if ordered.size >= 5 and variance > 0:
                    smooth = np.convolve(ordered, np.asarray([.25, .5, .25]), mode="valid")
                    coherence = float(np.var(smooth) / variance)
                else:
                    coherence = np.nan
                band_result = {"u": u, "x": x, "n": int(u.size),
                               "reduced_chi2": float(np.mean(u ** 2)),
                               "robust_scale": float(hm.robust_scale(u)),
                               "median_abs": float(np.median(np.abs(u))),
                               "lag1": lag1, "coherence": coherence}
            else:
                band_result = {"u": u, "x": x, "n": 0, "reduced_chi2": np.nan,
                               "robust_scale": np.nan, "median_abs": np.nan,
                               "lag1": np.nan, "coherence": np.nan}
            per_band.append(band_result)
        combined_x = np.concatenate([per_band[0]["x"], per_band[1]["x"]])
        combined_u = np.concatenate([per_band[0]["u"], per_band[1]["u"]])
        clusters["combined"].append((combined_x, combined_u))
        row = {"H5": posterior_row["H5"], "h5_id": posterior_row["h5_id"], "exposure": posterior_row["exposure"],
               "SPECID": posterior_row["SPECID"], "IFUSLOT": posterior_row["IFUSLOT"], "IFUID": posterior_row["IFUID"],
               "AMP": posterior_row["AMP"], "p_good": evidence.p_good, "I_m": evidence.I_m,
               "split_minus_joint_log_evidence": evidence.split_minus_joint_log_evidence,
               "X_amp_ON": evidence.x_amp[0], "X_amp_OFF": evidence.x_amp[1],
               "source_amp_ON": source_amp[0], "source_amp_OFF": source_amp[1],
               "source_to_noise_ON": source_to_noise[0], "source_to_noise_OFF": source_to_noise[1],
               "source_to_noise_joint": float(np.hypot(source_to_noise[0], source_to_noise[1])),
               "nominal_reduced_chi2_ON": per_band[0]["reduced_chi2"],
               "nominal_reduced_chi2_OFF": per_band[1]["reduced_chi2"],
               "nominal_reduced_chi2_joint": float(np.mean(combined_u ** 2)) if combined_u.size else np.nan,
               "robust_normalized_scale_ON": per_band[0]["robust_scale"],
               "robust_normalized_scale_OFF": per_band[1]["robust_scale"],
               "median_abs_u_ON": per_band[0]["median_abs"], "median_abs_u_OFF": per_band[1]["median_abs"],
               "lag1_residual_correlation_ON": per_band[0]["lag1"],
               "lag1_residual_correlation_OFF": per_band[1]["lag1"],
               "residual_coherence_ON": per_band[0]["coherence"],
               "residual_coherence_OFF": per_band[1]["coherence"]}
        rows.append(row)
    datasets = {}
    for name, values in clusters.items():
        x = np.concatenate([value[0] for value in values if value[0].size]) if any(value[0].size for value in values) else np.asarray([])
        u = np.concatenate([value[1] for value in values if value[1].size]) if any(value[1].size for value in values) else np.asarray([])
        stats, edges = _error_binned_stats(x, u)
        # Identical deterministic cluster draws keep ON/OFF/combined
        # bootstrap realizations tied to the same amplifier observations.
        rng = np.random.default_rng(20260904)
        bin_x2 = np.asarray([row["median_x2"] for row in stats], dtype=float)
        bootstrap_free = _error_bootstrap_fit(values, edges, bin_x2, False, n_bootstrap, rng)
        bootstrap_fixed = _error_bootstrap_fit(values, edges, bin_x2, True, n_bootstrap, rng)
        datasets[name] = {"x": x, "u": u, "clusters": values, "stats": stats, "edges": edges,
                          "free": _error_variance_fit(stats, False), "fixed": _error_variance_fit(stats, True),
                          "bootstrap_free": bootstrap_free, "bootstrap_fixed": bootstrap_fixed,
                          "scan": _error_model_scan({"x": x, "u": u, "edges": edges},
                                                     (0.0, .005, .010, .015, .020, .025, .030, .040, .050, .075, .100))}
    def bootstrap_section(dataset, fixed):
        boot = dataset["bootstrap_fixed" if fixed else "bootstrap_free"]
        answer = {"f_model": _error_fit_bootstrap_summary(boot["f_model"])}
        if not fixed:
            answer["c0"] = _error_fit_bootstrap_summary(boot["c0"])
        return answer
    summary = {"free_intercept": {}, "fixed_intercept": {}, "candidate_scan": {},
               "p_good_correlations": {}, "residual_coherence": {}}
    for name, dataset in datasets.items():
        summary["free_intercept"][name] = {**dataset["free"], "bootstrap": bootstrap_section(dataset, False)}
        summary["fixed_intercept"][name] = {**dataset["fixed"], "bootstrap": bootstrap_section(dataset, True)}
        summary["candidate_scan"][name] = dataset["scan"]
    row_values = {key: np.asarray([row[key] for row in rows], dtype=float)
                  for key in ("p_good", "nominal_reduced_chi2_joint", "source_to_noise_joint",
                              "split_minus_joint_log_evidence")}
    def finite_mean(values):
        values = np.asarray(values, dtype=float)
        values = values[np.isfinite(values)]
        return float(np.mean(values)) if values.size else np.nan
    row_values["robust_normalized_scale_joint"] = np.asarray([
        finite_mean((row["robust_normalized_scale_ON"], row["robust_normalized_scale_OFF"]))
        for row in rows])
    row_values["external_support"] = np.asarray([
        np.hypot(row["X_amp_ON"], row["X_amp_OFF"]) for row in rows])
    summary["p_good_correlations"] = {
        "p_good_vs_reduced_chi2": _spearman(row_values["p_good"], row_values["nominal_reduced_chi2_joint"]),
        "p_good_vs_robust_normalized_scale": _spearman(row_values["p_good"], row_values["robust_normalized_scale_joint"]),
        "p_good_vs_external_support": _spearman(row_values["p_good"], row_values["external_support"]),
        "p_good_vs_source_to_noise": _spearman(row_values["p_good"], row_values["source_to_noise_joint"]),
        "p_good_vs_split_evidence": _spearman(row_values["p_good"], row_values["split_minus_joint_log_evidence"])}
    coherence = np.concatenate([np.asarray([row["residual_coherence_ON"] for row in rows]),
                                np.asarray([row["residual_coherence_OFF"] for row in rows])])
    support = row_values["source_to_noise_joint"]
    summary["residual_coherence"] = {
        "summary": _distribution_summary(coherence),
        "spearman_with_p_good": _spearman(coherence, np.tile(row_values["p_good"], 2)),
        "spearman_with_source_to_noise": _spearman(coherence, np.tile(support, 2)),
        "median_low_p_good": float(np.nanmedian(coherence[np.tile(row_values["p_good"] < .5, 2)])),
        "median_other": float(np.nanmedian(coherence[np.tile(row_values["p_good"] >= .5, 2)]))}
    summary["n_amplifier_observations"] = len(rows)
    summary["n_bootstrap"] = int(n_bootstrap)
    summary["bootstrap_seed"] = 20260904
    summary["definition"] = "u=r/sigma_data, x=abs(exp(z_post)*band_scale*X)/sigma_data; cluster bootstrap resamples whole amplifier observations"
    return summary, rows, datasets


def _weighted_remove_nuisance(vector, design, sigma):
    """Remove a small weighted linear nuisance space without forming a dense matrix."""
    vector = np.asarray(vector, dtype=float)
    design = np.asarray(design, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    weight = 1.0 / sigma ** 2
    normal = design.T @ (weight[:, None] * design)
    rhs = design.T @ (weight * vector)
    try:
        coefficients = np.linalg.solve(normal, rhs)
    except np.linalg.LinAlgError:
        coefficients = np.linalg.lstsq(normal, rhs, rcond=None)[0]
    return vector - design @ coefficients


def _weighted_multiplicative_projection(residual, direction, H, sigma):
    """Project residual and direction orthogonally to [constant, H] in W."""
    design = np.column_stack((np.ones(len(residual)), H))
    residual_perp = _weighted_remove_nuisance(residual, design, sigma)
    direction_perp = _weighted_remove_nuisance(direction, design, sigma)
    weight = 1.0 / sigma ** 2
    denominator = float(np.sum(weight * direction_perp ** 2))
    if not np.isfinite(denominator) or denominator <= 1e-12:
        return residual_perp, direction_perp, np.nan, np.nan, np.nan
    numerator = float(np.sum(weight * direction_perp * residual_perp))
    delta = numerator / denominator
    return residual_perp, direction_perp, delta, 1.0 / np.sqrt(denominator), denominator


def _residual_coherence(q, normalized_residual):
    values = np.asarray(normalized_residual, dtype=float)
    q = np.asarray(q, dtype=float)
    if values.size < 3:
        return np.nan, np.nan
    ordered = values[np.argsort(q)]
    lag1 = (float(np.corrcoef(ordered[:-1], ordered[1:])[0, 1])
            if np.std(ordered[:-1]) > 0 and np.std(ordered[1:]) > 0 else np.nan)
    variance = float(np.var(ordered))
    if ordered.size < 5 or variance <= 0:
        return lag1, np.nan
    smooth = np.convolve(ordered, np.asarray([.25, .5, .25]), mode="valid")
    return lag1, float(np.var(smooth) / variance)


def _multiplicative_residual_decomposition(store, evidences, obs_rows,
                                           band_contrast, original_datasets):
    """QA-only decomposition of fiber residuals into amplifier-scale and shape modes."""
    rows = []
    projected_clusters = {band: {"source": [], "z": []} for band in BANDS}
    coherence = {"before": {band: [] for band in BANDS},
                 "after_source": {band: [] for band in BANDS},
                 "after_z": {band: [] for band in BANDS}}
    for evidence, block, posterior_row in zip(evidences, store.blocks, obs_rows):
        _, P_effective, band_scale, band_offset = _band_contrast_transform(
            block, band_contrast["delta_z_band"], band_contrast["delta_p_band"])
        prediction = (np.exp(posterior_row["posterior_z_mean"]) * P_effective
                      + posterior_row["p_mean"] + posterior_row["alpha_mean"] * block.H
                      + band_offset[None, :])
        residual = block.T - prediction
        source = np.exp(posterior_row["posterior_z_mean"]) * band_scale[None, :] * block.X
        tangent = np.exp(posterior_row["posterior_z_mean"]) * P_effective
        sigma_all = np.asarray(block.error, dtype=float).copy()
        if evidence.error_floor_used > 0:
            sigma_all = np.maximum(sigma_all, evidence.error_floor_used)
        values = {"delta_m_source": [], "sigma_delta_m_source_nominal": [],
                  "delta_z_linear": [], "sigma_delta_z_nominal": [],
                  "projection_leverage_source": [], "projection_leverage_z": [],
                  "source_support": []}
        before_scales, source_scales, z_scales = [], [], []
        before_coherence, source_coherence, z_coherence = [], [], []
        for band in range(N_BANDS):
            valid = block.likelihood_valid[:, band] & np.isfinite(residual[:, band])
            valid &= np.isfinite(source[:, band]) & np.isfinite(tangent[:, band])
            valid &= np.isfinite(sigma_all[:, band]) & (sigma_all[:, band] > 0)
            r = residual[:, band][valid]; s = source[:, band][valid]
            g = tangent[:, band][valid]; sigma = sigma_all[:, band][valid]
            q = block.q[valid]
            r_perp, s_perp, delta_source, sigma_source, leverage_source = _weighted_multiplicative_projection(
                r, s, block.H[:, band][valid], sigma)
            _, g_perp, delta_z, sigma_z, leverage_z = _weighted_multiplicative_projection(
                r, g, block.H[:, band][valid], sigma)
            after_source = r_perp - delta_source * s_perp if np.isfinite(delta_source) else r_perp
            after_z = r_perp - delta_z * g_perp if np.isfinite(delta_z) else r_perp
            u = r / sigma
            u_source = after_source / sigma
            u_z = after_z / sigma
            x = np.abs(s) / sigma
            band_name = BANDS[band]
            projected_clusters[band_name]["source"].append((x, u_source))
            projected_clusters[band_name]["z"].append((x, u_z))
            for target, statistic in ((before_coherence, _residual_coherence(q, u)),
                                      (source_coherence, _residual_coherence(q, u_source)),
                                      (z_coherence, _residual_coherence(q, u_z))):
                target.append(statistic)
            before_scales.append((float(hm.robust_scale(u)), float(np.sqrt(np.mean(u ** 2))) if u.size else np.nan))
            source_scales.append((float(hm.robust_scale(u_source)), float(np.sqrt(np.mean(u_source ** 2))) if u_source.size else np.nan))
            z_scales.append((float(hm.robust_scale(u_z)), float(np.sqrt(np.mean(u_z ** 2))) if u_z.size else np.nan))
            values["delta_m_source"].append(delta_source); values["sigma_delta_m_source_nominal"].append(sigma_source)
            values["delta_z_linear"].append(delta_z); values["sigma_delta_z_nominal"].append(sigma_z)
            values["projection_leverage_source"].append(leverage_source); values["projection_leverage_z"].append(leverage_z)
            values["source_support"].append(float(np.sqrt(np.sum(x ** 2))) if x.size else np.nan)
        row = {"H5": posterior_row["H5"], "h5_id": posterior_row["h5_id"], "exposure": posterior_row["exposure"],
               "SPECID": posterior_row["SPECID"], "IFUSLOT": posterior_row["IFUSLOT"], "IFUID": posterior_row["IFUID"],
               "AMP": posterior_row["AMP"], "p_good": evidence.p_good, "I_m": evidence.I_m,
               "split_minus_joint_log_evidence": evidence.split_minus_joint_log_evidence,
               "source_support_ON": values["source_support"][0], "source_support_OFF": values["source_support"][1],
               "delta_m_source_ON": values["delta_m_source"][0], "delta_m_source_OFF": values["delta_m_source"][1],
               "sigma_delta_m_source_nominal_ON": values["sigma_delta_m_source_nominal"][0],
               "sigma_delta_m_source_nominal_OFF": values["sigma_delta_m_source_nominal"][1],
               "delta_z_linear_ON": values["delta_z_linear"][0], "delta_z_linear_OFF": values["delta_z_linear"][1],
               "sigma_delta_z_nominal_ON": values["sigma_delta_z_nominal"][0],
               "sigma_delta_z_nominal_OFF": values["sigma_delta_z_nominal"][1],
               "normalized_residual_scale_ON": before_scales[0][0], "normalized_residual_scale_OFF": before_scales[1][0],
               "normalized_scale_after_source_ON": source_scales[0][0], "normalized_scale_after_source_OFF": source_scales[1][0],
               "normalized_scale_after_z_ON": z_scales[0][0], "normalized_scale_after_z_OFF": z_scales[1][0],
               "normalized_RMS_ON": before_scales[0][1], "normalized_RMS_OFF": before_scales[1][1],
               "RMS_after_source_ON": source_scales[0][1], "RMS_after_source_OFF": source_scales[1][1],
               "RMS_after_z_ON": z_scales[0][1], "RMS_after_z_OFF": z_scales[1][1],
               "coherence_before_ON": before_coherence[0][1], "coherence_before_OFF": before_coherence[1][1],
               "coherence_after_source_ON": source_coherence[0][1], "coherence_after_source_OFF": source_coherence[1][1],
               "coherence_after_z_ON": z_coherence[0][1], "coherence_after_z_OFF": z_coherence[1][1],
               "lag1_before_ON": before_coherence[0][0], "lag1_before_OFF": before_coherence[1][0],
               "lag1_after_source_ON": source_coherence[0][0], "lag1_after_source_OFF": source_coherence[1][0],
               "lag1_after_z_ON": z_coherence[0][0], "lag1_after_z_OFF": z_coherence[1][0],
               "projection_leverage_source_ON": values["projection_leverage_source"][0],
               "projection_leverage_source_OFF": values["projection_leverage_source"][1],
               "projection_leverage_z_ON": values["projection_leverage_z"][0],
               "projection_leverage_z_OFF": values["projection_leverage_z"][1]}
        rows.append(row)
        coherence["before"]["ON"].append(before_coherence[0][1]); coherence["before"]["OFF"].append(before_coherence[1][1])
        coherence["after_source"]["ON"].append(source_coherence[0][1]); coherence["after_source"]["OFF"].append(source_coherence[1][1])
        coherence["after_z"]["ON"].append(z_coherence[0][1]); coherence["after_z"]["OFF"].append(z_coherence[1][1])
    high_support = np.asarray([np.hypot(row["source_support_ON"], row["source_support_OFF"]) for row in rows])
    high_support &= np.isfinite(high_support)
    high_cut = float(np.nanmedian(high_support)) if np.any(high_support) else np.nan
    high_mask = np.isfinite(high_support) & (high_support >= high_cut)

    def population(values, mask=None):
        values = np.asarray(values, dtype=float)
        if mask is not None:
            values = values[mask]
        return _distribution_summary(values)

    delta_m = {band: np.asarray([row["delta_m_source_%s" % band] for row in rows]) for band in BANDS}
    delta_z = {band: np.asarray([row["delta_z_linear_%s" % band] for row in rows]) for band in BANDS}
    def paired_summary(values):
        valid = np.isfinite(values["ON"]) & np.isfinite(values["OFF"])
        difference = values["ON"][valid] - values["OFF"][valid]
        average = 0.5 * (values["ON"][valid] + values["OFF"][valid])
        return {"spearman_ON_vs_OFF": _spearman(values["ON"], values["OFF"]),
                "ON_minus_OFF": _distribution_summary(difference),
                "average": _distribution_summary(average)}
    def projection_section(values):
        return {"ON": {"all": population(values["ON"]), "high_support": population(values["ON"], high_mask)},
                "OFF": {"all": population(values["OFF"]), "high_support": population(values["OFF"], high_mask)},
                "high_support_definition": "combined source-support upper half; cut=%.6g" % high_cut,
                "paired": paired_summary(values)}
    def projected_fits(kind):
        fits = {}
        for band in BANDS:
            x = np.concatenate([cluster[0] for cluster in projected_clusters[band][kind] if cluster[0].size])
            u = np.concatenate([cluster[1] for cluster in projected_clusters[band][kind] if cluster[1].size])
            stats, edges = _error_binned_stats(x, u)
            fits[band] = {"fit": _error_variance_fit(stats, False), "binned": stats}
        return fits
    after_source = projected_fits("source")
    after_z = projected_fits("z")
    original_f = {band: original_datasets[band]["free"] for band in BANDS}
    after_source_f = {band: after_source[band]["fit"] for band in BANDS}
    after_z_f = {band: after_z[band]["fit"] for band in BANDS}
    pgood = np.asarray([row["p_good"] for row in rows])
    original_scale = np.asarray([np.nanmean((row["normalized_residual_scale_ON"], row["normalized_residual_scale_OFF"])) for row in rows])
    after_scale = np.asarray([np.nanmean((row["normalized_scale_after_source_ON"], row["normalized_scale_after_source_OFF"])) for row in rows])
    abs_dm = np.nanmean(np.abs(np.column_stack((delta_m["ON"], delta_m["OFF"]))), axis=1)
    abs_dz = np.nanmean(np.abs(np.column_stack((delta_z["ON"], delta_z["OFF"]))), axis=1)
    summary = {"source_projection": projection_section(delta_m), "z_projection": projection_section(delta_z),
               "fractional_variance": {"original_f_total": original_f,
                                        "f_after_source_projection": after_source_f,
                                        "f_after_z_projection": after_z_f},
               "residual_coherence": {mode: {band: _distribution_summary(values[band]) for band in BANDS}
                                      for mode, values in coherence.items()},
               "p_good_correlations": {
                   "p_good_vs_abs_delta_m_source": _spearman(pgood, abs_dm),
                   "p_good_vs_abs_delta_z_linear": _spearman(pgood, abs_dz),
                   "p_good_vs_post_source_projection_scale": _spearman(pgood, after_scale),
                   "p_good_vs_original_residual_scale": _spearman(pgood, original_scale),
                   "p_good_vs_source_support": _spearman(pgood, high_support),
                   "p_good_vs_split_evidence": _spearman(pgood, np.asarray([row["split_minus_joint_log_evidence"] for row in rows]))},
               "high_support_count": int(np.sum(high_mask)),
               "high_support_cut": high_cut,
               "fractional_scale_reduction_source": {band: float(1.0 - after_source_f[band]["f_model"] / original_f[band]["f_model"])
                                                      if np.isfinite(after_source_f[band]["f_model"]) and np.isfinite(original_f[band]["f_model"]) and original_f[band]["f_model"] != 0 else np.nan
                                                      for band in BANDS},
               "fractional_scale_reduction_z": {band: float(1.0 - after_z_f[band]["f_model"] / original_f[band]["f_model"])
                                                 if np.isfinite(after_z_f[band]["f_model"]) and np.isfinite(original_f[band]["f_model"]) and original_f[band]["f_model"] != 0 else np.nan
                                                 for band in BANDS}}
    plot_data = {"rows": rows, "original": original_datasets, "after_source": after_source,
                 "after_z": after_z, "high_mask": high_mask}
    return summary, rows, plot_data


def _calculate_local_evidences(blocks, z_grid, settings, workers):
    settings_by_block = settings if isinstance(settings, list) else [settings] * len(blocks)
    if workers == 1:
        return [_local_evidence(block, z_grid, *block_settings)
                for block, block_settings in zip(blocks, settings_by_block)]
    tasks = ((block, z_grid, block_settings)
             for block, block_settings in zip(blocks, settings_by_block))
    with ProcessPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(_local_evidence_worker, tasks, chunksize=8))


def _evidence_config(args, measurement_path):
    return {
        "schema": EVIDENCE_SCHEMA,
        "measurement_identity": _h5_identity(measurement_path),
        "z_grid": [float(args.z_min), float(args.z_max), int(args.n_z)],
        "pi_good": float(args.pi_good), "bad_scale": float(args.bad_scale),
        "p_sigma_fraction": float(args.p_sigma_fraction),
        "alpha_mean": float(args.alpha_mean), "alpha_sigma": float(args.alpha_sigma),
        "z0_sigma": float(args.z0_sigma), "error_floor": args.error_floor,
        "error_floor_factor": args.error_floor_factor,
    }


def _evidence_table_description(nz):
    class EvidenceDescription(tables.IsDescription):
        h5_id = tables.Int16Col()
        exposure = tables.UInt8Col()
        SPECID = tables.Int32Col()
        IFUSLOT = tables.Int32Col()
        IFUID = tables.Int32Col()
        AMP = tables.StringCol(2)
        H5 = tables.StringCol(256)
        RA = tables.Float64Col()
        Dec = tables.Float64Col()
        X_amp = tables.Float64Col(shape=(2,))
        median_x = tables.Float64Col(shape=(2,))
        n_valid = tables.Int16Col(shape=(2,))
        local_z_mean = tables.Float64Col()
        local_z_sigma = tables.Float64Col()
        local_z_skew = tables.Float64Col()
        local_m_mean = tables.Float64Col()
        p_mean = tables.Float64Col()
        p_sigma = tables.Float64Col()
        alpha_mean = tables.Float64Col()
        alpha_sigma = tables.Float64Col()
        rho_z_p = tables.Float64Col()
        rho_z_alpha = tables.Float64Col()
        rho_p_alpha = tables.Float64Col()
        p_information = tables.Float64Col()
        alpha_information = tables.Float64Col()
        I_m = tables.Float64Col()
        p_good = tables.Float64Col()
        site_tau = tables.Float64Col()
        site_nu = tables.Float64Col()
        site_z_hat = tables.Float64Col()
        site_sigma = tables.Float64Col()
        noninformative_site = tables.BoolCol()
        grid_edge_flag = tables.BoolCol()
        split_minus_joint_log_evidence = tables.Float64Col()
        split_z_mean = tables.Float64Col(shape=(2,))
        split_z_sigma = tables.Float64Col(shape=(2,))
        split_p_mean = tables.Float64Col(shape=(2,))
        split_p_sigma = tables.Float64Col(shape=(2,))
        split_alpha_mean = tables.Float64Col(shape=(2,))
        split_alpha_sigma = tables.Float64Col(shape=(2,))
        split_grid_edge = tables.BoolCol(shape=(2,))
        split_delta_z = tables.Float64Col()
        split_delta_p = tables.Float64Col()
        split_delta_alpha = tables.Float64Col()
        split_delta_alpha_sigma = tables.Float64Col()
        split_delta_alpha_significance = tables.Float64Col()
        log_m_good = tables.Float64Col(shape=(nz,))
        log_m_bad = tables.Float64Col(shape=(nz,))
        log_m_total = tables.Float64Col(shape=(nz,))
        beta_mean_z = tables.Float64Col(shape=(nz, 2))
        beta_cov_integrated = tables.Float64Col(shape=(2, 2))
        p_sigma_prior = tables.Float64Col()
        error_floor_used = tables.Float64Col()
    return EvidenceDescription


def _write_evidence_cache(path, evidences, z_grid, config, band_contrast, pre_qa):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    if temporary.exists():
        temporary.unlink()
    filters = tables.Filters(complevel=5, complib="zlib", shuffle=True)
    with tables.open_file(temporary, mode="w", title="M101 local amplifier evidence") as h5:
        h5.root._v_attrs.schema_version = EVIDENCE_SCHEMA
        h5.root._v_attrs.band_order = json.dumps(list(BANDS))
        h5.root._v_attrs.z_grid = json.dumps([float(value) for value in z_grid])
        h5.root._v_attrs.config_json = json.dumps(config, sort_keys=True, default=str)
        h5.root._v_attrs.band_contrast_json = json.dumps(band_contrast, sort_keys=True, default=str)
        h5.root._v_attrs.pre_qa_json = json.dumps(pre_qa, sort_keys=True, default=str)
        table = h5.create_table("/", "evidence", _evidence_table_description(len(z_grid)),
                                expectedrows=len(evidences), filters=filters)
        for evidence in evidences:
            row = table.row
            key = evidence.key
            row["h5_id"] = key.h5_id; row["exposure"] = key.exposure
            row["SPECID"] = key.ifu.specid; row["IFUSLOT"] = key.ifu.ifuslot; row["IFUID"] = key.ifu.ifuid
            row["AMP"] = key.amp; row["H5"] = evidence.h5_name; row["RA"] = evidence.ra; row["Dec"] = evidence.dec
            row["X_amp"] = evidence.x_amp
            for field in ("median_x", "n_valid"):
                row[field] = getattr(evidence, field)
            for field in ("local_z_mean", "local_z_sigma", "local_z_skew", "local_m_mean", "p_mean", "p_sigma",
                          "alpha_mean", "alpha_sigma", "rho_z_p", "rho_z_alpha", "rho_p_alpha",
                          "p_information", "alpha_information", "I_m", "p_good", "site_tau", "site_nu",
                          "site_z_hat", "site_sigma", "split_minus_joint_log_evidence", "p_sigma_prior",
                          "error_floor_used"):
                row[field] = getattr(evidence, field)
            for field in ("split_z_mean", "split_z_sigma", "split_p_mean", "split_p_sigma",
                          "split_alpha_mean", "split_alpha_sigma", "split_grid_edge", "split_delta_z",
                          "split_delta_p", "split_delta_alpha", "split_delta_alpha_sigma",
                          "split_delta_alpha_significance"):
                row[field] = getattr(evidence, field)
            row["noninformative_site"] = evidence.noninformative_site; row["grid_edge_flag"] = evidence.grid_edge_flag
            row["log_m_good"] = evidence.log_m_good; row["log_m_bad"] = evidence.log_m_bad; row["log_m_total"] = evidence.log_m_total
            row["beta_mean_z"] = evidence.beta_mean_z; row["beta_cov_integrated"] = evidence.beta_cov_integrated
            row.append()
        table.flush()
    temporary.replace(path)


def _load_evidence_cache(path, store, z_grid, config):
    path = Path(path)
    if not path.exists():
        return None
    try:
        with tables.open_file(path, mode="r") as h5:
            if _text(getattr(h5.root._v_attrs, "schema_version", "")) != EVIDENCE_SCHEMA:
                return None
            cached_config = json.loads(_text(h5.root._v_attrs.config_json))
            if cached_config != config:
                return None
            cached_grid = np.asarray(json.loads(_text(h5.root._v_attrs.z_grid)), dtype=float)
            if not np.array_equal(cached_grid, z_grid):
                return None
            table = h5.root.evidence
            if table.nrows != len(store.blocks):
                return None
            evidences = []
            band_contrast = json.loads(_text(h5.root._v_attrs.band_contrast_json))
            pre_qa = json.loads(_text(h5.root._v_attrs.pre_qa_json))
            for block, row in zip(store.blocks, table):
                key = block.key
                if (int(row["h5_id"]), int(row["exposure"]), int(row["SPECID"]), int(row["IFUSLOT"]),
                        int(row["IFUID"]), _text(row["AMP"])) != (key.h5_id, key.exposure, key.ifu.specid,
                                                                     key.ifu.ifuslot, key.ifu.ifuid, key.amp):
                    return None
                values = {field: row[field] for field in (
                    "median_x", "n_valid", "local_z_mean", "local_z_sigma", "local_z_skew",
                    "local_m_mean", "p_mean", "p_sigma", "alpha_mean", "alpha_sigma", "rho_z_p",
                    "rho_z_alpha", "rho_p_alpha", "p_information", "alpha_information", "I_m", "p_good",
                    "site_tau", "site_nu", "site_z_hat", "site_sigma", "noninformative_site", "grid_edge_flag",
                    "split_minus_joint_log_evidence", "log_m_good", "log_m_bad", "log_m_total", "beta_mean_z",
                    "beta_cov_integrated", "p_sigma_prior", "error_floor_used", "split_z_mean", "split_z_sigma",
                    "split_p_mean", "split_p_sigma", "split_alpha_mean", "split_alpha_sigma", "split_grid_edge",
                    "split_delta_z", "split_delta_p", "split_delta_alpha", "split_delta_alpha_sigma",
                    "split_delta_alpha_significance")}
                values["x_amp"] = np.asarray(row["X_amp"])
                evidences.append(AmplifierEvidence(
                    key=key, h5_name=block.h5_name, ra=float(row["RA"]), dec=float(row["Dec"]),
                    **{name: np.asarray(value) if name in ("X_amp", "median_x", "n_valid", "split_z_mean", "split_z_sigma", "split_p_mean", "split_p_sigma", "split_alpha_mean", "split_alpha_sigma", "split_grid_edge", "log_m_good", "log_m_bad", "log_m_total", "beta_mean_z", "beta_cov_integrated") else value
                       for name, value in values.items()}))
            return evidences, band_contrast, pre_qa
    except (OSError, tables.HDF5ExtError, KeyError, ValueError, json.JSONDecodeError):
        return None


def build_global_layout(blocks):
    exposures = sorted(set((b.key.h5_id, b.key.exposure) for b in blocks))
    gamma_index = {key: index for index, key in enumerate(exposures)}
    next_index = len(gamma_index)
    ifu_by_exposure = {}
    for block in blocks:
        key = (block.key.h5_id, block.key.exposure)
        ifu_by_exposure.setdefault(key, set()).add(block.key.ifu)
    ifu_index, ifu_order = {}, {}
    for exposure in exposures:
        physical_ifus = sorted(ifu_by_exposure[exposure], key=lambda x: (x.specid, x.ifuslot, x.ifuid))
        basis = hm_helmert(len(physical_ifus))
        shared_indices = np.arange(next_index, next_index + basis.shape[1], dtype=int)
        next_index += basis.shape[1]
        for position, ifu in enumerate(physical_ifus):
            key = (exposure[0], exposure[1], ifu)
            ifu_index[key] = shared_indices
            ifu_order[("iota", key)] = (basis, position)
    physical_ifus = sorted(set(b.key.ifu for b in blocks), key=lambda x: (x.specid, x.ifuslot, x.ifuid))
    eta_index = {}
    for ifu in physical_ifus:
        eta_index[ifu] = np.arange(next_index, next_index + 3, dtype=int)
        next_index += 3
        ifu_order[("eta", ifu)] = (hm_helmert(4), 0)
    # Replace the amplifier position in the eta tuple with a lookup performed
    # by a wrapper below; the raw basis remains an explicit Helmert matrix.
    layout = GlobalLayout(gamma_index, ifu_index, eta_index, ifu_order, next_index)
    layout._amp_positions = {key: position for key, position in
                             (((b.key.ifu, b.key.amp), ("LL", "LU", "RL", "RU").index(b.key.amp))
                              for b in blocks)}
    layout._ifu_amp_order = {ifu: (hm_helmert(4),) for ifu in physical_ifus}
    return layout


def _layout_design(layout, key):
    indices = [layout.gamma_index[(key.h5_id, key.exposure)]]
    values = [1.0]
    iota_key = (key.h5_id, key.exposure, key.ifu)
    basis, position = layout.ifu_order[("iota", iota_key)]
    for coefficient, value in enumerate(basis[position, :]):
        if value:
            indices.append(layout.ifu_index[iota_key][coefficient]); values.append(float(value))
    eta_basis = layout.ifu_order[("eta", key.ifu)][0]
    amp_position = layout._amp_positions[(key.ifu, key.amp)]
    for coefficient, value in enumerate(eta_basis[amp_position, :]):
        if value:
            indices.append(layout.eta_index[key.ifu][coefficient]); values.append(float(value))
    return np.asarray(indices, dtype=int), np.asarray(values, dtype=float)


def _assemble_global(layout, evidences, selected, gamma_sigma, ifu_sigma, eta_sigma):
    n = layout.parameter_count
    prior_precision = np.zeros(n, dtype=float)
    for index in layout.gamma_index.values(): prior_precision[index] = 1.0 / gamma_sigma**2
    for values in layout.ifu_index.values(): prior_precision[values] = 1.0 / ifu_sigma**2
    for values in layout.eta_index.values(): prior_precision[values] = 1.0 / eta_sigma**2
    rows = [np.arange(n)]; cols = [np.arange(n)]; data = [prior_precision]
    h = np.zeros(n, dtype=float)
    for evidence, use in zip(evidences, selected):
        if not use or evidence.site_tau <= 0 or not np.isfinite(evidence.site_tau):
            continue
        indices, values = _layout_design(layout, evidence.key)
        rows.append(np.repeat(indices, len(indices))); cols.append(np.tile(indices, len(indices)))
        data.append(evidence.site_tau * np.outer(values, values).ravel())
        h[indices] += evidence.site_nu * values
    Q = sparse.coo_matrix((np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))),
                          shape=(n, n)).tocsr()
    Q.sum_duplicates()
    return Q, h


def _solve_mean(Q, h):
    factor = sparse_linalg.factorized(Q.tocsc())
    return np.asarray(factor(h), dtype=float), factor


def solve_global(layout, evidences, selected, gamma_sigma, ifu_sigma, eta_sigma,
                 hutchinson_probes=HUTCHINSON_PROBES_DEFAULT, exact_indices=()):
    Q, h = _assemble_global(layout, evidences, selected, gamma_sigma, ifu_sigma, eta_sigma)
    mean, factor = _solve_mean(Q, h)
    n = layout.parameter_count
    variance = np.zeros(n, dtype=float)
    exact_indices = sorted(set(int(i) for i in exact_indices))
    exact_values = {}
    for index in exact_indices:
        unit = np.zeros(n, dtype=float); unit[index] = 1.0
        solution = factor(unit)
        exact_values[index] = max(float(solution[index]), 0.0)
    rng = np.random.default_rng(271828)
    probes = max(int(hutchinson_probes), 1)
    probe_matrix = rng.choice(np.array([-1.0, 1.0]), size=(n, probes))
    solved = np.column_stack([factor(probe_matrix[:, i]) for i in range(probes)])
    variance += np.mean(probe_matrix * solved, axis=1)
    for index, value in exact_values.items():
        variance[index] = value
    variance = np.maximum(variance, 0.0)
    method = "exact selected diagonals + Hutchinson diagonal (%d probes, seed=271828)" % probes
    return CalibrationPosterior(layout, mean, variance, factor, Q, h, method)


def _site_order(evidences):
    return sorted(range(len(evidences)), key=lambda i: (
        evidences[i].key.h5_id, evidences[i].key.exposure, evidences[i].key.ifu.specid,
        evidences[i].key.ifu.ifuslot, evidences[i].key.ifu.ifuid, evidences[i].key.amp))


def _history(evidences, layout, gamma_sigma, ifu_sigma, eta_sigma, probes):
    order = _site_order(evidences)
    first_gamma = sorted(layout.gamma_index.values())[:3]
    first_iota = [values[0] for _, values in sorted(layout.ifu_index.items(), key=lambda x: repr(x[0])) if len(values)][:3]
    first_eta = [values[0] for _, values in sorted(layout.eta_index.items(), key=lambda x: repr(x[0]))][:3]
    stages = []
    exposure_stops = []
    previous = None
    for position, index in enumerate(order):
        stage_key = (evidences[index].key.h5_id, evidences[index].key.exposure)
        if stage_key != previous:
            if previous is not None:
                exposure_stops.append(position)
            previous = stage_key
    exposure_stops.append(len(order))
    for stage, stop in enumerate(exposure_stops):
        selected_indices = set(order[:stop])
        selected = np.asarray([i in selected_indices for i in range(len(evidences))], dtype=bool)
        posterior = solve_global(layout, evidences, selected, gamma_sigma, ifu_sigma, eta_sigma,
                                 hutchinson_probes=max(8, probes // 4), exact_indices=first_gamma + first_eta)
        current = evidences[order[stop - 1]].key
        stages.append({"stage": stage + 1, "h5_id": current.h5_id, "exposure": current.exposure,
                       "cumulative_sites": int(np.sum(selected)),
                       "cumulative_tau": float(np.sum([e.site_tau for e, use in zip(evidences, selected) if use])),
                       "cumulative_I_m": float(np.sum([e.I_m for e, use in zip(evidences, selected) if use])),
                       "gamma_mean": posterior.mean[first_gamma], "gamma_sigma": np.sqrt(posterior.variance[first_gamma]),
                       "iota_mean": posterior.mean[first_iota], "iota_sigma": np.sqrt(posterior.variance[first_iota]),
                       "eta_mean": posterior.mean[first_eta], "eta_sigma": np.sqrt(posterior.variance[first_eta])})
    return stages


def _normal_tail_abs_probability(mean, sigma, threshold):
    if not np.isfinite(sigma) or sigma <= 0:
        return float(abs(mean) > threshold)
    return float(norm.cdf((-threshold - mean) / sigma) + 1.0 - norm.cdf((threshold - mean) / sigma))


def _posterior_rows(store, evidences, posterior, alpha_prior_mean=ALPHA_MEAN_DEFAULT,
                    alpha_prior_sigma=ALPHA_SIGMA_DEFAULT,
                    delta_z_band=0.0, delta_p_band=0.0):
    obs_rows, ifu_accum, amp_accum, exp_accum = [], {}, {}, {}
    for evidence, block in zip(evidences, store.blocks):
        indices, values = _layout_design(posterior.layout, evidence.key)
        z_mean = float(np.dot(values, posterior.mean[indices]))
        z_variance = float(np.sum(values * values * posterior.variance[indices]))
        z_sigma = np.sqrt(max(z_variance, 0.0))
        conditional_valid = block.likelihood_valid
        T_effective, P_effective, _, band_offset = _band_contrast_transform(
            block, delta_z_band, delta_p_band)
        p_values, alpha_values = [], []
        if conditional_valid.any():
            sigma = block.error[conditional_valid]
            if evidence.error_floor_used > 0:
                sigma = np.maximum(sigma, evidence.error_floor_used)
            mean_beta, cov_beta = _conditional_beta(T_effective[conditional_valid], P_effective[conditional_valid],
                                                    block.H[conditional_valid], sigma, z_mean,
                                                    np.asarray([0.0, alpha_prior_mean]),
                                                    np.asarray([evidence.p_sigma_prior, alpha_prior_sigma]))
            p_values.append(mean_beta[0]); alpha_values.append(mean_beta[1])
            p_mean, alpha_mean = float(mean_beta[0]), float(mean_beta[1])
            p_sigma, alpha_sigma = np.sqrt(max(cov_beta[0, 0], 0.0)), np.sqrt(max(cov_beta[1, 1], 0.0))
        else:
            p_mean, alpha_mean = evidence.p_mean, evidence.alpha_mean
            p_sigma, alpha_sigma = evidence.p_sigma, evidence.alpha_sigma
        prediction = (np.exp(z_mean) * P_effective + p_mean + alpha_mean * block.H
                      + band_offset[None, :])
        residual = block.T - prediction
        rms, robust = [], []
        mean_residual, median_residual = [], []
        for band in range(N_BANDS):
            selected = conditional_valid[:, band]
            values_residual = residual[:, band][selected]
            rms.append(float(np.sqrt(np.mean(values_residual**2))) if values_residual.size else np.nan)
            robust.append(float(hm.robust_scale(values_residual)) if values_residual.size else np.nan)
            mean_residual.append(float(np.mean(values_residual)) if values_residual.size else np.nan)
            median_residual.append(float(np.median(values_residual)) if values_residual.size else np.nan)
        flags = []
        if evidence.noninformative_site: flags.append("noninformative_site")
        if evidence.grid_edge_flag: flags.append("grid_edge")
        if evidence.p_good < 0.5: flags.append("low_p_good")
        if evidence.split_minus_joint_log_evidence > 5.0: flags.append("split_band_preferred")
        if evidence.I_m < 0.01: flags.append("low_information")
        row = {"H5": evidence.h5_name, "h5_id": evidence.key.h5_id, "exposure": evidence.key.exposure,
               "SPECID": evidence.key.ifu.specid, "IFUSLOT": evidence.key.ifu.ifuslot,
               "IFUID": evidence.key.ifu.ifuid, "AMP": evidence.key.amp, "RA": evidence.ra, "Dec": evidence.dec,
               "X_amp_ON": evidence.x_amp[0], "X_amp_OFF": evidence.x_amp[1],
               "n_valid_ON": evidence.n_valid[0], "n_valid_OFF": evidence.n_valid[1],
               "local_z_mean": evidence.local_z_mean, "local_z_sigma": evidence.local_z_sigma,
               "local_m_mean": evidence.local_m_mean, "p_mean": p_mean, "p_sigma": p_sigma,
               "alpha_mean": alpha_mean, "alpha_sigma": alpha_sigma, "rho_z_p": evidence.rho_z_p,
               "rho_z_alpha": evidence.rho_z_alpha, "rho_p_alpha": evidence.rho_p_alpha,
               "split_z_mean_ON": evidence.split_z_mean[0], "split_z_sigma_ON": evidence.split_z_sigma[0],
               "split_z_mean_OFF": evidence.split_z_mean[1], "split_z_sigma_OFF": evidence.split_z_sigma[1],
               "split_delta_z": evidence.split_delta_z,
               "split_p_mean_ON": evidence.split_p_mean[0], "split_p_sigma_ON": evidence.split_p_sigma[0],
               "split_p_mean_OFF": evidence.split_p_mean[1], "split_p_sigma_OFF": evidence.split_p_sigma[1],
               "split_delta_p": evidence.split_delta_p,
               "split_alpha_mean_ON": evidence.split_alpha_mean[0], "split_alpha_sigma_ON": evidence.split_alpha_sigma[0],
               "split_alpha_mean_OFF": evidence.split_alpha_mean[1], "split_alpha_sigma_OFF": evidence.split_alpha_sigma[1],
               "split_delta_alpha": evidence.split_delta_alpha,
               "split_delta_alpha_sigma": evidence.split_delta_alpha_sigma,
               "split_delta_alpha_significance": evidence.split_delta_alpha_significance,
               "P_alpha_lt_0p2": float(norm.cdf((0.2 - alpha_mean) / alpha_sigma)) if alpha_sigma > 0 else float(alpha_mean < 0.2),
               "P_alpha_gt_0p6": float(1.0 - norm.cdf((0.6 - alpha_mean) / alpha_sigma)) if alpha_sigma > 0 else float(alpha_mean > 0.6),
               "I_m": evidence.I_m, "p_good": evidence.p_good, "site_tau": evidence.site_tau,
               "site_nu": evidence.site_nu, "site_z_hat": evidence.site_z_hat, "site_sigma": evidence.site_sigma,
               "grid_edge_flag": evidence.grid_edge_flag,
               "joint_vs_split_band_log_evidence": evidence.split_minus_joint_log_evidence,
               "posterior_z_mean": z_mean, "posterior_z_sigma": z_sigma,
               "posterior_m_mean": float(np.exp(z_mean + .5 * z_variance)),
               "residual_RMS_ON": rms[0], "residual_RMS_OFF": rms[1],
               "robust_RMS_ON": robust[0], "robust_RMS_OFF": robust[1],
               "mean_residual_ON": mean_residual[0], "mean_residual_OFF": mean_residual[1],
               "median_residual_ON": median_residual[0], "median_residual_OFF": median_residual[1],
               "model_flags": ";".join(flags)}
        obs_rows.append(row)
        physical_key = evidence.key.ifu
        amp_key = PhysicalAmplifierKey(physical_key, evidence.key.amp)
        amp_accum.setdefault(amp_key, []).append(evidence)
        ifu_key = (evidence.key.h5_id, evidence.key.exposure, physical_key)
        ifu_accum.setdefault(ifu_key, []).append(evidence)
        exp_key = (evidence.key.h5_id, evidence.key.exposure)
        exp_accum.setdefault(exp_key, []).append(evidence)
    return obs_rows, amp_accum, ifu_accum, exp_accum


def _population_rows(amp_accum, posterior):
    rows = []
    for key, values in sorted(amp_accum.items(), key=lambda item: (item[0].ifu.specid, item[0].ifu.ifuslot, item[0].ifu.ifuid, item[0].amp)):
        eta_indices = posterior.layout.eta_index[key.ifu]
        basis = hm_helmert(4)
        amp_position = ("LL", "LU", "RL", "RU").index(key.amp)
        weights = basis[amp_position, :]
        eta_mean = float(np.dot(weights, posterior.mean[eta_indices]))
        eta_variance = float(np.sum(weights * weights * posterior.variance[eta_indices]))
        eta_sigma = np.sqrt(max(eta_variance, 0.0))
        rows.append({"SPECID": key.ifu.specid, "IFUSLOT": key.ifu.ifuslot, "IFUID": key.ifu.ifuid, "AMP": key.amp,
                     "eta_mean": eta_mean, "eta_sigma": eta_sigma,
                     "P_abs_eta_gt_2pct": _normal_tail_abs_probability(eta_mean, eta_sigma, np.log(1.02)),
                     "P_abs_eta_gt_5pct": _normal_tail_abs_probability(eta_mean, eta_sigma, np.log(1.05)),
                     "P_abs_eta_gt_10pct": _normal_tail_abs_probability(eta_mean, eta_sigma, np.log(1.10)),
                     "n_informative_observations": int(sum(v.site_tau > 0 for v in values)),
                     "cumulative_I_m": float(sum(v.I_m for v in values)),
                     "median_p_good": float(np.median([v.p_good for v in values])),
                     "minimum_p_good": float(np.min([v.p_good for v in values])),
                     "X_support_min": float(np.nanmin([np.nanmin(np.abs(v.x_amp)) for v in values])),
                     "X_support_max": float(np.nanmax([np.nanmax(np.abs(v.x_amp)) for v in values]))})
    return rows


def _ifu_rows(ifu_accum, posterior, store):
    rows = []
    for key, values in sorted(ifu_accum.items(), key=lambda item: (item[0][0], item[0][1], item[0][2].specid, item[0][2].ifuslot, item[0][2].ifuid)):
        h5_id, exposure, ifu = key
        iota_indices = posterior.layout.ifu_index[(h5_id, exposure, ifu)]
        physical_ifus = sorted({b.key.ifu for b in store.blocks if b.key.h5_id == h5_id and b.key.exposure == exposure}, key=lambda x: (x.specid, x.ifuslot, x.ifuid))
        basis = hm_helmert(len(physical_ifus))
        position = physical_ifus.index(ifu)
        weights = basis[position, :]
        iota_mean = float(np.dot(weights, posterior.mean[iota_indices])) if len(weights) else 0.0
        iota_sigma = np.sqrt(max(float(np.sum(weights * weights * posterior.variance[iota_indices])), 0.0)) if len(weights) else 0.0
        block = next(b for b in store.blocks if b.key.h5_id == h5_id and b.key.exposure == exposure and b.key.ifu == ifu)
        site_values = [v.site_z_hat for v in values if np.isfinite(v.site_z_hat)]
        rows.append({"H5": block.h5_name, "h5_id": h5_id, "exposure": exposure,
                     "SPECID": ifu.specid, "IFUSLOT": ifu.ifuslot, "IFUID": ifu.ifuid,
                     "RA": block.ra, "Dec": block.dec, "iota_mean": iota_mean, "iota_sigma": iota_sigma,
                     "n_amplifier_sites": len(values), "n_informative_sites": int(sum(v.site_tau > 0 for v in values)),
                     "cumulative_information": float(sum(v.I_m for v in values)),
                     "amplifier_site_z_scatter": hm.robust_scale(site_values),
                     "amplifier_consistency_median_p_good": float(np.median([v.p_good for v in values]))})
    return rows


def _exposure_rows(exp_accum, posterior):
    rows = []
    for key, values in sorted(exp_accum.items()):
        index = posterior.layout.gamma_index[key]
        rows.append({"h5_id": key[0], "exposure": key[1], "gamma_mean": posterior.mean[index],
                     "gamma_sigma": np.sqrt(max(posterior.variance[index], 0.0)),
                     "multiplicative_mean": np.exp(posterior.mean[index] + .5 * posterior.variance[index]),
                     "n_amplifier_sites": len(values),
                     "n_informative_sites": int(sum(v.site_tau > 0 for v in values)),
                     "cumulative_information": float(sum(v.I_m for v in values)),
                     "H5": values[0].h5_name})
    return rows


def _solution_descriptions():
    class Obs(tables.IsDescription):
        h5_id = tables.Int16Col(); exposure = tables.UInt8Col(); H5 = tables.StringCol(256)
        SPECID = tables.Int32Col(); IFUSLOT = tables.Int32Col(); IFUID = tables.Int32Col(); AMP = tables.StringCol(2)
        RA = tables.Float64Col(); Dec = tables.Float64Col(); X_amp_ON = tables.Float64Col(); X_amp_OFF = tables.Float64Col()
        n_valid_ON = tables.Int16Col(); n_valid_OFF = tables.Int16Col(); local_z_mean = tables.Float64Col(); local_z_sigma = tables.Float64Col(); local_m_mean = tables.Float64Col()
        p_mean = tables.Float64Col(); p_sigma = tables.Float64Col(); alpha_mean = tables.Float64Col(); alpha_sigma = tables.Float64Col(); rho_z_p = tables.Float64Col(); rho_z_alpha = tables.Float64Col(); rho_p_alpha = tables.Float64Col(); split_z_mean_ON = tables.Float64Col(); split_z_sigma_ON = tables.Float64Col(); split_z_mean_OFF = tables.Float64Col(); split_z_sigma_OFF = tables.Float64Col(); split_delta_z = tables.Float64Col(); split_p_mean_ON = tables.Float64Col(); split_p_sigma_ON = tables.Float64Col(); split_p_mean_OFF = tables.Float64Col(); split_p_sigma_OFF = tables.Float64Col(); split_delta_p = tables.Float64Col(); split_alpha_mean_ON = tables.Float64Col(); split_alpha_sigma_ON = tables.Float64Col(); split_alpha_mean_OFF = tables.Float64Col(); split_alpha_sigma_OFF = tables.Float64Col(); split_delta_alpha = tables.Float64Col(); split_delta_alpha_sigma = tables.Float64Col(); split_delta_alpha_significance = tables.Float64Col(); P_alpha_lt_0p2 = tables.Float64Col(); P_alpha_gt_0p6 = tables.Float64Col()
        I_m = tables.Float64Col(); p_good = tables.Float64Col(); site_tau = tables.Float64Col(); site_nu = tables.Float64Col(); site_z_hat = tables.Float64Col(); site_sigma = tables.Float64Col(); grid_edge_flag = tables.BoolCol(); joint_vs_split_band_log_evidence = tables.Float64Col()
        posterior_z_mean = tables.Float64Col(); posterior_z_sigma = tables.Float64Col(); posterior_m_mean = tables.Float64Col(); residual_RMS_ON = tables.Float64Col(); residual_RMS_OFF = tables.Float64Col(); robust_RMS_ON = tables.Float64Col(); robust_RMS_OFF = tables.Float64Col(); mean_residual_ON = tables.Float64Col(); mean_residual_OFF = tables.Float64Col(); median_residual_ON = tables.Float64Col(); median_residual_OFF = tables.Float64Col(); model_flags = tables.StringCol(256)
    class Physical(tables.IsDescription):
        SPECID = tables.Int32Col(); IFUSLOT = tables.Int32Col(); IFUID = tables.Int32Col(); AMP = tables.StringCol(2); eta_mean = tables.Float64Col(); eta_sigma = tables.Float64Col(); P_abs_eta_gt_2pct = tables.Float64Col(); P_abs_eta_gt_5pct = tables.Float64Col(); P_abs_eta_gt_10pct = tables.Float64Col(); n_informative_observations = tables.Int32Col(); cumulative_I_m = tables.Float64Col(); median_p_good = tables.Float64Col(); minimum_p_good = tables.Float64Col(); X_support_min = tables.Float64Col(); X_support_max = tables.Float64Col()
    class IFU(tables.IsDescription):
        h5_id = tables.Int16Col(); exposure = tables.UInt8Col(); H5 = tables.StringCol(256); SPECID = tables.Int32Col(); IFUSLOT = tables.Int32Col(); IFUID = tables.Int32Col(); RA = tables.Float64Col(); Dec = tables.Float64Col(); iota_mean = tables.Float64Col(); iota_sigma = tables.Float64Col(); n_amplifier_sites = tables.Int16Col(); n_informative_sites = tables.Int16Col(); cumulative_information = tables.Float64Col(); amplifier_site_z_scatter = tables.Float64Col(); amplifier_consistency_median_p_good = tables.Float64Col()
    class Exposure(tables.IsDescription):
        h5_id = tables.Int16Col(); exposure = tables.UInt8Col(); H5 = tables.StringCol(256); gamma_mean = tables.Float64Col(); gamma_sigma = tables.Float64Col(); multiplicative_mean = tables.Float64Col(); n_amplifier_sites = tables.Int32Col(); n_informative_sites = tables.Int32Col(); cumulative_information = tables.Float64Col()
    class History(tables.IsDescription):
        stage = tables.Int16Col(); h5_id = tables.Int16Col(); exposure = tables.UInt8Col(); cumulative_sites = tables.Int32Col(); cumulative_tau = tables.Float64Col(); cumulative_I_m = tables.Float64Col(); gamma_mean = tables.Float64Col(shape=(3,)); gamma_sigma = tables.Float64Col(shape=(3,)); iota_mean = tables.Float64Col(shape=(3,)); iota_sigma = tables.Float64Col(shape=(3,)); eta_mean = tables.Float64Col(shape=(3,)); eta_sigma = tables.Float64Col(shape=(3,))
    return Obs, Physical, IFU, Exposure, History


def _write_solution(path, obs_rows, physical_rows, ifu_rows, exposure_rows, history_rows, metadata):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    filters = tables.Filters(complevel=5, complib="zlib", shuffle=True)
    Obs, Physical, IFU, Exposure, History = _solution_descriptions()
    with tables.open_file(path, mode="w", title="M101 Bayesian calibration inference") as h5:
        h5.root._v_attrs.schema_version = "m101_bayesian_calibration_v1"
        h5.root._v_attrs.band_order = json.dumps(list(BANDS))
        h5.root._v_attrs.no_cube_reconstruction = True
        h5.root._v_attrs.no_production_calibration_applied = True
        tables_by_rows = (("amplifier_observations", Obs, obs_rows), ("physical_amplifiers", Physical, physical_rows),
                          ("ifu_posteriors", IFU, ifu_rows), ("exposure_posteriors", Exposure, exposure_rows), ("history", History, history_rows))
        for name, description, rows in tables_by_rows:
            table = h5.create_table("/", name, description, expectedrows=len(rows), filters=filters)
            for source in rows:
                row = table.row
                for field in table.colnames:
                    if field in source: row[field] = source[field]
                row.append()
            table.flush()
        group = h5.create_group("/", "provenance")
        metadata_table = h5.create_table(group, "metadata", _metadata_description(), expectedrows=len(metadata), filters=filters)
        for key, value in metadata.items():
            row = metadata_table.row; row["key"] = str(key); row["value"] = json.dumps(value, sort_keys=True, default=str); row.append()
        metadata_table.flush()
        for name in ("amplifier_observations", "physical_amplifiers", "ifu_posteriors", "exposure_posteriors"):
            table = getattr(h5.root, name)
            for column in ("h5_id", "exposure", "SPECID", "IFUSLOT", "IFUID"):
                if column in table.colnames: getattr(table.cols, column).create_index()


def _metadata_description():
    class Metadata(tables.IsDescription):
        key = tables.StringCol(128); value = tables.StringCol(8192)
    return Metadata


def _history_rows(stages):
    return stages


def _set_robust_ylim(axis, values):
    """Set limits from the finite 1--99% range with 20% padding."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size < 2:
        return
    lower, upper = np.percentile(values, (1.0, 99.0))
    span = upper - lower
    if not np.isfinite(span) or span <= 0:
        span = max(abs(lower), abs(upper), 1.0) * 1e-6
    axis.set_ylim(lower - 0.2 * span, upper + 0.2 * span)


def _plot_outputs(output_dir, store, evidences, obs_rows, physical_rows, ifu_rows, exposure_rows, stages, posterior, z_grid, band_contrast, split_alpha_diagnostic):
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    support = np.asarray([np.hypot(e.x_amp[0], e.x_amp[1]) for e in evidences])
    info = np.asarray([e.I_m for e in evidences]); quality = np.asarray([e.p_good for e in evidences])
    fig, axes = plt.subplots(1, 2, figsize=(12, 5)); axes[0].scatter(support, info, c=quality, s=8, cmap="viridis"); axes[0].set(xlabel="external support |X_amp|", ylabel="I_m", title="Information versus source support"); axes[1].scatter(support, quality, c=info, s=8, cmap="plasma"); axes[1].set(xlabel="external support |X_amp|", ylabel="p_good", title="Quality versus source support"); fig.tight_layout(); fig.savefig(output_dir / "m101_bayes_information_vs_source.png", dpi=150); plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8));
    for index in range(min(3, stages[0]["gamma_mean"].size if stages else 0)):
        axes[0, 0].plot([s["cumulative_sites"] for s in stages], [s["gamma_mean"][index] for s in stages], "o-", label="gamma %d" % index)
        axes[0, 1].plot([s["cumulative_sites"] for s in stages], [s["gamma_sigma"][index] for s in stages], "o-", label="gamma %d" % index)
    for index in range(min(3, stages[0]["eta_mean"].size if stages else 0)):
        axes[1, 0].plot([s["cumulative_sites"] for s in stages], [s["eta_mean"][index] for s in stages], "o-", label="eta %d" % index)
        axes[1, 1].plot([s["cumulative_sites"] for s in stages], [s["cumulative_I_m"] for s in stages], "o-", label="cumulative I_m")
    for ax in axes.flat: ax.grid(alpha=.2); ax.legend(fontsize=8)
    axes[0, 0].set_ylabel("posterior mean"); axes[0, 1].set_ylabel("posterior sigma"); axes[1, 0].set_ylabel("posterior mean"); axes[1, 1].set_ylabel("information");
    for ax in axes.flat: ax.set_xlabel("cumulative amplifier sites")
    fig.suptitle("Collapse of ignorance"); fig.tight_layout(rect=(0, 0, 1, .95)); fig.savefig(output_dir / "m101_bayes_collapse_of_ignorance.png", dpi=150); plt.close(fig)
    fig, ax = plt.subplots(figsize=(12, 5)); x = np.arange(len(exposure_rows)); ax.errorbar(x, [r["gamma_mean"] for r in exposure_rows], yerr=[r["gamma_sigma"] for r in exposure_rows], fmt="o", ms=3); ax.axhline(0, color="k", lw=.8); ax.set(xlabel="chronological exposure", ylabel="gamma", title="Exposure posterior"); ax.grid(alpha=.2); fig.tight_layout(); fig.savefig(output_dir / "m101_bayes_gamma_exposure.png", dpi=150); plt.close(fig)
    fig, ax = plt.subplots(figsize=(12, 6)); colors = {"LL": "tab:blue", "LU": "tab:orange", "RL": "tab:green", "RU": "tab:red"};
    for amp, color in colors.items():
        rows = [r for r in physical_rows if r["AMP"] == amp]; pos = np.arange(len(rows)); ax.errorbar(pos, [r["eta_mean"] for r in rows], yerr=[r["eta_sigma"] for r in rows], fmt="o", ms=3, color=color, label=amp)
    ax.axhline(0, color="k", lw=.8); ax.set(xlabel="persistent physical-amplifier identity (within orientation)", ylabel="eta", title="Persistent amplifier posterior"); ax.legend(); ax.grid(alpha=.2); fig.tight_layout(); fig.savefig(output_dir / "m101_bayes_eta_amplifiers.png", dpi=150); plt.close(fig)
    representative = sorted(set((r["h5_id"], r["exposure"]) for r in ifu_rows))[:3]; fig, axes = plt.subplots(1, max(len(representative), 1), figsize=(5 * max(len(representative), 1), 4), squeeze=False)
    for ax, key in zip(axes.flat, representative):
        rows = [r for r in ifu_rows if (r["h5_id"], r["exposure"]) == key]; sc = ax.scatter([r["RA"] for r in rows], [r["Dec"] for r in rows], c=[r["iota_mean"] for r in rows], cmap="coolwarm", s=30); ax.set_title("h5=%d e=%d" % key); ax.set_xlabel("RA"); ax.set_ylabel("Dec"); fig.colorbar(sc, ax=ax, label="iota")
    fig.suptitle("Representative IFU response maps"); fig.tight_layout(rect=(0, 0, 1, .94)); fig.savefig(output_dir / "m101_bayes_iota_maps.png", dpi=150); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5)); colors = {"LL": "tab:blue", "LU": "tab:orange", "RL": "tab:green", "RU": "tab:red"}
    for amp, color in colors.items():
        selected_rows = [r for r in obs_rows if r["AMP"] == amp]
        axes[0].errorbar([r["exposure"] for r in selected_rows], [r["p_mean"] for r in selected_rows], yerr=[r["p_sigma"] for r in selected_rows], fmt=".", ms=3, alpha=.35, color=color, label=amp)
        axes[1].errorbar([r["exposure"] for r in selected_rows], [r["alpha_mean"] for r in selected_rows], yerr=[r["alpha_sigma"] for r in selected_rows], fmt=".", ms=3, alpha=.35, color=color, label=amp)
    axes[0].set(xlabel="exposure", ylabel="p", title="Pedestal posterior"); axes[1].set(xlabel="exposure", ylabel="alpha", title="Template amplitude posterior"); axes[1].legend(fontsize=8); fig.tight_layout(); fig.savefig(output_dir / "m101_bayes_additive_population.png", dpi=150); plt.close(fig)
    split_z = np.asarray([e.split_delta_z for e in evidences], dtype=float)
    split_p = np.asarray([e.split_delta_p for e in evidences], dtype=float)
    split_ok = np.isfinite(split_z) & np.isfinite(split_p)
    split_ok &= np.asarray([np.all(np.isfinite(e.split_z_mean)) and not np.any(e.split_grid_edge) for e in evidences])
    chronology_keys = sorted(set((e.key.h5_id, e.key.exposure) for e in evidences))
    chronology = {key: index for index, key in enumerate(chronology_keys)}
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes[0, 0].hist(split_z[split_ok], bins=30, color="tab:blue", alpha=.75)
    axes[0, 0].axvline(band_contrast["delta_z_band"], color="k", ls="--", label="adopted")
    axes[0, 0].set(xlabel="preliminary split delta_z", ylabel="count"); axes[0, 0].legend()
    axes[0, 1].hist(split_p[split_ok], bins=30, color="tab:orange", alpha=.75)
    axes[0, 1].axvline(band_contrast["delta_p_band"], color="k", ls="--", label="adopted")
    axes[0, 1].set(xlabel="preliminary split delta_p", ylabel="count"); axes[0, 1].legend()
    axes[1, 0].scatter([chronology[(e.key.h5_id, e.key.exposure)] for e, use in zip(evidences, split_ok) if use], split_z[split_ok], s=8, alpha=.5)
    axes[1, 0].axhline(band_contrast["delta_z_band"], color="k", ls="--")
    axes[1, 0].set(xlabel="chronological H5/exposure", ylabel="split delta_z")
    support = np.asarray([np.hypot(e.x_amp[0], e.x_amp[1]) for e in evidences])
    residual_contrast = np.asarray([r["mean_residual_ON"] - r["mean_residual_OFF"] for r in obs_rows])
    axes[1, 1].scatter(support, residual_contrast, s=8, alpha=.35)
    axes[1, 1].axhline(0.0, color="k", lw=.8)
    axes[1, 1].set(xlabel="external support |X_amp|", ylabel="final mean residual ON - OFF")
    for axis in axes.flat: axis.grid(alpha=.2)
    fig.suptitle("Global ON/OFF band contrast diagnostic"); fig.tight_layout(rect=(0, 0, 1, .95)); fig.savefig(output_dir / "m101_bayes_band_contrast.png", dpi=150); plt.close(fig)
    # Split-alpha QA is diagnostic only; it does not alter the shared-alpha model.
    split_alpha = np.asarray([e.split_delta_alpha for e in evidences], dtype=float)
    split_alpha_evidence = np.asarray([e.split_minus_joint_log_evidence for e in evidences], dtype=float)
    split_alpha_support = support
    split_alpha_ok = np.isfinite(split_alpha) & np.isfinite(split_alpha_evidence)
    split_alpha_strong = split_alpha_ok & (split_alpha_evidence > 5.0)
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    if np.any(split_alpha_ok):
        axes[0, 0].hist(split_alpha[split_alpha_ok & ~split_alpha_strong], bins=30,
                        alpha=.65, color="0.55", label="not strong split")
        axes[0, 0].hist(split_alpha[split_alpha_strong], bins=30,
                        alpha=.70, color="tab:red", label="split evidence > 5")
    axes[0, 0].axvline(0.0, color="k", lw=.8)
    axes[0, 0].set(xlabel="split delta_alpha (ON - OFF)", ylabel="count",
                   title="Split-alpha distribution (N=%d)" % split_alpha_diagnostic["finite_count"])
    axes[0, 0].legend(fontsize=8)
    axes[0, 1].scatter(split_alpha_evidence[split_alpha_ok & ~split_alpha_strong],
                       split_alpha[split_alpha_ok & ~split_alpha_strong],
                       s=7, alpha=.35, color="0.45", label="not strong split")
    axes[0, 1].scatter(split_alpha_evidence[split_alpha_strong], split_alpha[split_alpha_strong],
                       s=8, alpha=.45, color="tab:red", label="split evidence > 5")
    axes[0, 1].axhline(0.0, color="k", lw=.8); axes[0, 1].axvline(5.0, color="k", ls="--", lw=.8)
    axes[0, 1].set(xlabel="final split minus joint log evidence", ylabel="split delta_alpha",
                   title="Alpha contrast versus remaining split preference")
    axes[0, 1].legend(fontsize=8)
    chronology_keys = sorted(set((e.key.h5_id, e.key.exposure) for e in evidences))
    chronology = {key: index for index, key in enumerate(chronology_keys)}
    chrono_x = np.asarray([chronology[(e.key.h5_id, e.key.exposure)] for e in evidences])
    amp_colors = {"LL": "tab:blue", "LU": "tab:orange", "RL": "tab:green", "RU": "tab:red"}
    for amp, color in amp_colors.items():
        use = split_alpha_ok & np.asarray([e.key.amp == amp for e in evidences])
        axes[1, 0].scatter(chrono_x[use], split_alpha[use], s=7, alpha=.35, color=color, label=amp)
    axes[1, 0].axhline(0.0, color="k", lw=.8)
    axes[1, 0].set(xlabel="chronological H5/exposure", ylabel="split delta_alpha",
                   title="Temporal/H5 dependence")
    axes[1, 0].legend(fontsize=8, ncol=2)
    axes[1, 1].scatter(split_alpha_support[split_alpha_ok], split_alpha[split_alpha_ok],
                       s=7, alpha=.35, color="tab:purple")
    axes[1, 1].axhline(0.0, color="k", lw=.8)
    axes[1, 1].set(xlabel="external support |X_amp|", ylabel="split delta_alpha",
                   title="Alpha contrast versus source support")
    for axis in axes.flat:
        axis.grid(alpha=.2)
    fig.suptitle("Split-alpha diagnostic (QA; shared-alpha model unchanged)")
    fig.tight_layout(rect=(0, 0, 1, .95)); fig.savefig(output_dir / "m101_bayes_split_alpha_diagnostic.png", dpi=150); plt.close(fig)
    # Posterior predictive fiber-level QA uses a deterministic subset of blocks.
    fig, axes = plt.subplots(2, 3, figsize=(15, 8)); sample_blocks = store.blocks[::max(1, len(store.blocks) // 500)]; sample_rows = obs_rows[::max(1, len(obs_rows) // 500)]; residual_values = [[], [], []]
    for block, row in zip(sample_blocks, sample_rows):
        _, P_effective, _, band_offset = _band_contrast_transform(
            block, band_contrast["delta_z_band"], band_contrast["delta_p_band"])
        prediction = (np.exp(row["posterior_z_mean"]) * P_effective + row["p_mean"]
                      + row["alpha_mean"] * block.H + band_offset[None, :])
        residual = block.T - prediction; valid = block.likelihood_valid
        for band, color in enumerate(("tab:blue", "tab:orange")):
            axes[0, band].scatter(block.q[valid[:, band]], residual[:, band][valid[:, band]], s=2, alpha=.15, color=color)
            axes[1, band].scatter(block.X[:, band][valid[:, band]], residual[:, band][valid[:, band]], s=2, alpha=.15, color=color)
            axes[0, band].set_xlabel("q"); axes[1, band].set_xlabel("X"); axes[0, band].set_ylabel("residual"); axes[1, band].set_ylabel("residual")
            axes[0, 2].scatter(block.B[:, band][valid[:, band]], residual[:, band][valid[:, band]], s=2, alpha=.15, color=color)
            axes[0, 2].set_xlabel("B"); axes[0, 2].set_ylabel("residual")
            residual_values[band].extend(residual[:, band][valid[:, band]])
            residual_values[2].extend(residual[:, band][valid[:, band]])
    for band in range(N_BANDS):
        _set_robust_ylim(axes[0, band], residual_values[band])
        _set_robust_ylim(axes[1, band], residual_values[band])
    _set_robust_ylim(axes[0, 2], residual_values[2])
    axes[1, 2].axis("off"); fig.suptitle("Posterior predictive residuals (blue ON, orange OFF)"); fig.tight_layout(rect=(0, 0, 1, .95)); fig.savefig(output_dir / "m101_bayes_posterior_predictive.png", dpi=150); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 6)); ax.scatter([e.I_m for e in evidences], [e.split_minus_joint_log_evidence for e in evidences], c=[colors[e.key.amp] for e in evidences], s=8); ax.axhline(0, color="k", lw=.8); ax.set(xlabel="I_m", ylabel="split minus joint log evidence", title="Joint versus split-band QA"); ax.grid(alpha=.2); fig.tight_layout(); fig.savefig(output_dir / "m101_bayes_on_off_model_test.png", dpi=150); plt.close(fig)
    # Flagged gallery: transparent, deliberately broad criteria.
    ranking = sorted(range(len(obs_rows)), key=lambda i: (obs_rows[i]["p_good"] < .5, obs_rows[i]["I_m"] * abs(obs_rows[i]["posterior_z_mean"])), reverse=True)[:12]
    n = len(ranking); fig, axes = plt.subplots(max(1, n), 3, figsize=(12, 2.8 * max(1, n)), squeeze=False)
    for axis_row, index in enumerate(ranking):
        block, row, evidence = store.blocks[index], obs_rows[index], evidences[index]
        valid = block.likelihood_valid
        _, P_effective, _, band_offset = _band_contrast_transform(
            block, band_contrast["delta_z_band"], band_contrast["delta_p_band"])
        prediction = (np.exp(row["posterior_z_mean"]) * P_effective + row["p_mean"]
                      + row["alpha_mean"] * block.H + band_offset[None, :])
        axes[axis_row, 0].scatter(block.T[valid], prediction[valid], s=3, alpha=.3); axes[axis_row, 0].plot([np.nanmin(block.T[valid]), np.nanmax(block.T[valid])], [np.nanmin(block.T[valid]), np.nanmax(block.T[valid])], "k--"); axes[axis_row, 0].set_ylabel("T/pred");
        log_post = evidence.log_m_total + _normal_logpdf_grid(z_grid, 0, Z0_SIGMA_DEFAULT); log_post -= logsumexp(log_post); axes[axis_row, 1].plot(z_grid, np.exp(log_post)); axes[axis_row, 1].axvline(row["posterior_z_mean"], color="r"); axes[axis_row, 1].set_ylabel("local posterior")
        mean_residual = np.nanmean((block.T - prediction), axis=1)
        axes[axis_row, 2].scatter(block.q, mean_residual, s=4); _set_robust_ylim(axes[axis_row, 2], mean_residual); axes[axis_row, 2].axhline(0, color="k", lw=.8); axes[axis_row, 2].set_ylabel("mean residual"); axes[axis_row, 2].set_xlabel("q"); axes[axis_row, 2].set_title("%s/%s/%s %s pgood=%.3g I=%.3g da=%.3g sig=%.3g" % (row["SPECID"], row["IFUSLOT"], row["IFUID"], row["AMP"], row["p_good"], row["I_m"], row["split_delta_alpha"], row["split_delta_alpha_significance"]), fontsize=8)
    for ax in axes.flat: ax.grid(alpha=.2)
    fig.suptitle("Flagged amplifier gallery"); fig.tight_layout(rect=(0, 0, 1, .98)); fig.savefig(output_dir / "m101_bayes_flagged_gallery.png", dpi=130); plt.close(fig)


def _plot_error_model_diagnostic(output_dir, datasets, rows):
    """Write the compact QA figure for source-dependent residual variance."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    for axis, name, color in zip((axes[0, 0], axes[0, 1]), ("ON", "OFF"), ("tab:blue", "tab:orange")):
        stats = datasets[name]["stats"]
        x2 = np.asarray([row["median_x2"] for row in stats])
        observed = np.asarray([row["robust_scale_u"] ** 2 for row in stats])
        axis.scatter(x2, observed, s=20, color=color, label=name)
        if x2.size:
            line_x = np.linspace(0.0, max(float(np.max(x2)), 1.0), 200)
            free = datasets[name]["free"]
            fixed = datasets[name]["fixed"]
            axis.plot(line_x, free["c0"] + free["f_model"] ** 2 * line_x, "k-", label="free intercept")
            axis.plot(line_x, 1.0 + fixed["f_model"] ** 2 * line_x, "k--", label="fixed intercept")
        axis.set(xlabel="median x^2", ylabel="robust scale(u)^2", title="%s variance relation" % name)
        axis.legend(fontsize=8); axis.grid(alpha=.2)
    axes[0, 2].hist(datasets["ON"]["bootstrap_free"]["f_model"], bins=30, alpha=.55, label="ON free", color="tab:blue")
    axes[0, 2].hist(datasets["OFF"]["bootstrap_free"]["f_model"], bins=30, alpha=.55, label="OFF free", color="tab:orange")
    axes[0, 2].hist(datasets["combined"]["bootstrap_free"]["f_model"], bins=30, alpha=.45, label="combined free", color="tab:green")
    axes[0, 2].set(xlabel="f_model", ylabel="bootstrap count", title="Cluster-bootstrap fractional error")
    axes[0, 2].legend(fontsize=8); axes[0, 2].grid(alpha=.2)
    representative_f = (0.0, .020, .050, .100)
    for factor in representative_f:
        effective = datasets["combined"]["u"] / np.sqrt(1.0 + (factor * datasets["combined"]["x"]) ** 2)
        stats, _ = _error_binned_stats(datasets["combined"]["x"], effective, edges=datasets["combined"]["edges"])
        axes[1, 0].plot([row["median_x"] for row in stats], [row["robust_scale_u"] for row in stats], "o-", ms=3, label="f=%.3g" % factor)
    axes[1, 0].axhline(1.0, color="k", lw=.8); axes[1, 0].set(xlabel="median x", ylabel="robust scale(u_eff)", title="Candidate fractional error scan")
    axes[1, 0].legend(fontsize=8); axes[1, 0].grid(alpha=.2)
    pgood = np.asarray([row["p_good"] for row in rows])
    source_noise = np.asarray([row["source_to_noise_joint"] for row in rows])
    chi2 = np.asarray([row["nominal_reduced_chi2_joint"] for row in rows])
    axes[1, 1].scatter(source_noise, pgood, c=chi2, s=8, alpha=.35, cmap="viridis")
    axes[1, 1].set(xlabel="source-to-noise (joint)", ylabel="p_good", title="Current quality versus source strength")
    axes[1, 1].grid(alpha=.2)
    def finite_mean(values):
        values = np.asarray(values, dtype=float); values = values[np.isfinite(values)]
        return float(np.mean(values)) if values.size else np.nan
    coherence = np.asarray([finite_mean((row["residual_coherence_ON"], row["residual_coherence_OFF"]))
                            for row in rows])
    support = np.asarray([np.hypot(row["X_amp_ON"], row["X_amp_OFF"]) for row in rows])
    axes[1, 2].scatter(support, coherence, c=pgood, s=8, alpha=.35, cmap="plasma")
    axes[1, 2].set(xlabel="external support |X_amp|", ylabel="residual coherence", title="Within-amplifier coherence")
    axes[1, 2].grid(alpha=.2)
    fig.suptitle("Error-model diagnostic (QA only; current likelihood unchanged)")
    fig.tight_layout(rect=(0, 0, 1, .95)); fig.savefig(Path(output_dir) / "m101_bayes_error_model_diagnostic.png", dpi=150); plt.close(fig)


def run_synthetic_validation():
    rng = np.random.default_rng(12345); z_grid = np.linspace(-1.2, 1.2, 161)
    T = rng.normal(size=7); P = rng.normal(size=7); H = rng.normal(size=7); sigma = np.full(7, .2); beta = np.array([.1, .4]); beta_sigma = np.array([.3, .2]); z = np.linspace(-.5, .5, 5)
    fast = _marginal_grid(T, P, H, sigma, beta, beta_sigma, z)
    A = np.column_stack((np.ones(7), H)); C = np.diag(sigma**2) + A @ np.diag(beta_sigma**2) @ A.T
    dense = np.asarray([multivariate_normal.logpdf(T - np.exp(value) * P, mean=A @ beta, cov=C) for value in z])
    dense_error = float(np.max(np.abs(fast.log_m - dense)))
    if dense_error > 1e-9:
        raise AssertionError("2x2 local marginal likelihood differs from dense calculation: %.4g" % dense_error)
    site_tau = np.array([1.0, 2.0, .5]); site_nu = np.array([.2, -.1, .3]); prior_precision = 4.0
    chronological = 0.0; reverse = 0.0
    for tau, nu in zip(site_tau, site_nu): chronological = (prior_precision * chronological + nu) / (prior_precision + tau); prior_precision += tau
    prior_precision = 4.0
    for tau, nu in zip(site_tau[::-1], site_nu[::-1]): reverse = (prior_precision * reverse + nu) / (prior_precision + tau); prior_precision += tau
    order_error = abs(chronological - reverse)
    if order_error > 1e-12:
        raise AssertionError("synthetic Gaussian site order test failed")
    # Bright/blank/bad behavior is a smoke check, not a production decision.
    def synthetic_block(scale, noise, mismatch=0.0):
        n = 40; p = np.linspace(-scale, scale, n) if scale else np.zeros(n); h = np.linspace(.1, 1.0, n)
        b = np.full(n, .2) if scale else np.zeros(n)
        d = (1.15 * (p + b) + .08 + .4 * h if scale else -b)
        d = d + rng.normal(0, noise, n) + mismatch * np.sin(np.arange(n))
        err = np.full(n, noise)
        return (d[:, None] * np.ones((1, 2)), b[:, None] * np.ones((1, 2)),
                p[:, None] * np.ones((1, 2)), err[:, None] * np.ones((1, 2)), h)
    smoke = []
    for scale, noise, mismatch, label in ((10, .03, 0.0, "bright"), (0, .03, 0.0, "blank"), (10, .03, 10.0, "bad")):
        d, b, x, err, h = synthetic_block(scale, noise, mismatch)
        block = AmplifierBlock(AmplifierObservationKey(0, 1, PhysicalIFUKey(1, 1, 1), "LL"), "synthetic", 0, 0, np.zeros(2), np.zeros(2), d, b, x, err, np.ones_like(d, dtype=bool), False, np.linspace(0, 1, 40), np.ones(2), np.arange(40), .2)
        evidence = _local_evidence(block, z_grid)
        smoke.append((label, evidence.local_z_sigma, evidence.p_good, evidence.I_m))
    if smoke[0][2] < 0.5 or smoke[1][2] < 0.5 or smoke[2][2] >= 0.5:
        raise AssertionError("synthetic quality-mixture behavior failed: %r" % (smoke,))
    if smoke[1][3] >= smoke[0][3]:
        raise AssertionError("synthetic blank/source information distinction failed: %r" % (smoke,))
    # Targeted ON/OFF contrast test: the injected contrast is deliberately
    # small and is recovered only from the preliminary split-band fits.
    n = 40; x = np.linspace(2.0, 20.0, n); b = np.full(n, 0.2); h = np.linspace(.1, 1.0, n)
    injected_delta_z, injected_delta_p = 0.08, 0.04
    P = (x + b)[:, None] * np.ones((1, 2))
    T = (np.exp(0.10 + 0.5 * injected_delta_z * BAND_SIGN)[None, :] * P
         + 0.02 + 0.5 * injected_delta_p * BAND_SIGN[None, :]
         + 0.4 * h[:, None] + rng.normal(0, .03, (n, 2)))
    contrast_block = AmplifierBlock(
        AmplifierObservationKey(0, 1, PhysicalIFUKey(1, 1, 1), "LL"), "synthetic",
        0, 0, np.zeros(2), np.zeros(2), T - b[:, None],
        b[:, None] * np.ones((1, 2)), x[:, None] * np.ones((1, 2)),
        np.full((n, 2), .03), np.ones((n, 2), dtype=bool), False,
        np.linspace(0, 1, n), np.ones(2), np.arange(n), .2)
    preliminary_contrast = _local_evidence(contrast_block, z_grid)
    split_summary = {"z_mean": preliminary_contrast.split_z_mean,
                     "z_sigma": preliminary_contrast.split_z_sigma,
                     "p_mean": preliminary_contrast.split_p_mean,
                     "p_sigma": preliminary_contrast.split_p_sigma,
                     "alpha_mean": preliminary_contrast.split_alpha_mean,
                     "alpha_sigma": preliminary_contrast.split_alpha_sigma,
                     "grid_edge": preliminary_contrast.split_grid_edge}
    corrected_contrast = _local_evidence(
        contrast_block, z_grid, delta_z_band=preliminary_contrast.split_delta_z,
        delta_p_band=preliminary_contrast.split_delta_p, split_summary=split_summary)
    zero_T, zero_P, _, _ = _band_contrast_transform(contrast_block, 0.0, 0.0)
    if np.max(np.abs(zero_T - contrast_block.T)) > 1e-14 or np.max(np.abs(zero_P - contrast_block.P)) > 1e-14:
        raise AssertionError("zero band contrast changed the synthetic block")
    if (np.sign(preliminary_contrast.split_delta_z) != np.sign(injected_delta_z)
            or np.sign(preliminary_contrast.split_delta_p) != np.sign(injected_delta_p)
            or abs(preliminary_contrast.split_delta_z - injected_delta_z) > 0.04
            or abs(preliminary_contrast.split_delta_p - injected_delta_p) > 0.03
            or corrected_contrast.split_minus_joint_log_evidence
            >= 0.1 * preliminary_contrast.split_minus_joint_log_evidence):
        raise AssertionError("synthetic band-contrast refinement failed")
    band_contrast_smoke = {"injected_delta_z": injected_delta_z,
                           "injected_delta_p": injected_delta_p,
                           "recovered_delta_z": preliminary_contrast.split_delta_z,
                           "recovered_delta_p": preliminary_contrast.split_delta_p,
                           "pre_log_evidence_gap": preliminary_contrast.split_minus_joint_log_evidence,
                           "post_log_evidence_gap": corrected_contrast.split_minus_joint_log_evidence}
    # Targeted split-alpha QA: the final model remains shared-alpha, while
    # the independent split fits should recover an injected alpha contrast.
    alpha_delta = 0.12
    alpha_x = 10.0 + 2.0 * np.cos(np.linspace(0.0, 5.0 * np.pi, n))
    alpha_h = np.sin(np.linspace(0.0, 8.0 * np.pi, n))
    alpha_b = np.full(n, 0.2)
    alpha_P = (alpha_x + alpha_b)[:, None] * np.ones((1, 2))
    alpha_values = 0.4 + 0.5 * alpha_delta * BAND_SIGN
    alpha_T = (np.exp(0.10) * alpha_P + 0.02
               + alpha_h[:, None] * alpha_values[None, :]
               + rng.normal(0.0, 0.01, (n, 2)))
    alpha_block = AmplifierBlock(
        AmplifierObservationKey(0, 1, PhysicalIFUKey(1, 1, 1), "LL"), "synthetic",
        0, 0, np.zeros(2), np.zeros(2), alpha_T - alpha_b[:, None],
        alpha_b[:, None] * np.ones((1, 2)), alpha_x[:, None] * np.ones((1, 2)),
        np.full((n, 2), 0.01), np.ones((n, 2), dtype=bool), False,
        alpha_h, np.ones(2), np.arange(n), 0.2)
    alpha_evidence = _local_evidence(alpha_block, z_grid)
    zero_alpha_T = np.exp(0.10) * alpha_P + 0.02 + 0.4 * alpha_h[:, None]
    zero_alpha_block = AmplifierBlock(
        AmplifierObservationKey(0, 1, PhysicalIFUKey(1, 1, 1), "LL"), "synthetic",
        0, 0, np.zeros(2), np.zeros(2), zero_alpha_T - alpha_b[:, None],
        alpha_b[:, None] * np.ones((1, 2)), alpha_x[:, None] * np.ones((1, 2)),
        np.full((n, 2), 0.01), np.ones((n, 2), dtype=bool), False,
        alpha_h, np.ones(2), np.arange(n), 0.2)
    zero_alpha_evidence = _local_evidence(zero_alpha_block, z_grid)
    recovered_alpha_delta = alpha_evidence.split_delta_alpha
    zero_alpha_delta = zero_alpha_evidence.split_delta_alpha
    if (np.sign(recovered_alpha_delta) != np.sign(alpha_delta)
            or abs(recovered_alpha_delta - alpha_delta) > 0.04
            or abs(zero_alpha_delta) > 1e-10):
        raise AssertionError("synthetic split-alpha diagnostic failed: injected=%.4g recovered=%.4g zero=%.4g"
                             % (alpha_delta, recovered_alpha_delta, zero_alpha_delta))
    split_alpha_smoke = {"injected_delta_alpha": alpha_delta,
                         "recovered_delta_alpha": recovered_alpha_delta,
                         "zero_delta_alpha_recovered": zero_alpha_delta}
    return {"status": "PASS", "dense_max_abs_error": dense_error, "order_independence_error": order_error, "local_smoke": smoke, "band_contrast_smoke": band_contrast_smoke, "split_alpha_smoke": split_alpha_smoke}


def _solution_csvs(output_dir, obs_rows, physical_rows, ifu_rows, exposure_rows):
    _write_csv(output_dir / "m101_bayes_amplifier_observations.csv", obs_rows)
    _write_csv(output_dir / "m101_bayes_physical_amplifiers.csv", physical_rows)
    _write_csv(output_dir / "m101_bayes_ifu_posteriors.csv", ifu_rows)
    _write_csv(output_dir / "m101_bayes_exposure_posteriors.csv", exposure_rows)
    additive = [{"H5": row["H5"], "exposure": row["exposure"], "SPECID": row["SPECID"], "IFUSLOT": row["IFUSLOT"], "IFUID": row["IFUID"], "AMP": row["AMP"], "p_mean": row["p_mean"], "p_sigma": row["p_sigma"], "alpha_mean": row["alpha_mean"], "alpha_sigma": row["alpha_sigma"]} for row in obs_rows]
    _write_csv(output_dir / "m101_bayes_additive_population.csv", additive)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--measurements", default="m101_measurements.h5")
    parser.add_argument("--output", default="m101_bayesian_calibration.h5")
    parser.add_argument("--evidence-cache", default="m101_amplifier_evidence.h5")
    parser.add_argument("--workers", type=int, default=1,
                        help="number of worker processes for local amplifier evidence calculations")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--synthetic-test", action="store_true")
    parser.add_argument("--z-min", type=float, default=Z_MIN_DEFAULT); parser.add_argument("--z-max", type=float, default=Z_MAX_DEFAULT); parser.add_argument("--n-z", type=int, default=N_Z_DEFAULT)
    parser.add_argument("--pi-good", type=float, default=PI_GOOD_DEFAULT); parser.add_argument("--bad-scale", type=float, default=BAD_SCALE_DEFAULT)
    parser.add_argument("--p-sigma-fraction", type=float, default=P_SIGMA_FRACTION_DEFAULT); parser.add_argument("--alpha-mean", type=float, default=ALPHA_MEAN_DEFAULT); parser.add_argument("--alpha-sigma", type=float, default=ALPHA_SIGMA_DEFAULT); parser.add_argument("--z0-sigma", type=float, default=Z0_SIGMA_DEFAULT)
    parser.add_argument("--gamma-sigma", type=float, default=GAMMA_SIGMA_DEFAULT); parser.add_argument("--ifu-sigma", type=float, default=IFU_SIGMA_DEFAULT); parser.add_argument("--eta-sigma", type=float, default=ETA_SIGMA_DEFAULT)
    parser.add_argument("--error-floor", type=float, default=None); parser.add_argument("--error-floor-factor", type=float, default=None); parser.add_argument("--hutchinson-probes", type=int, default=HUTCHINSON_PROBES_DEFAULT)
    parser.add_argument("--error-bootstrap", type=int, default=ERROR_BOOTSTRAP_DEFAULT,
                        help="cluster-bootstrap realizations for the error-model diagnostic")
    args = parser.parse_args()
    if args.workers < 1:
        raise SystemExit("--workers must be at least 1")
    if args.synthetic_test:
        print(json.dumps(run_synthetic_validation(), indent=2, default=str)); return
    if args.error_floor is not None and args.error_floor <= 0: raise SystemExit("--error-floor must be positive")
    if args.error_floor_factor is not None and args.error_floor_factor <= 0: raise SystemExit("--error-floor-factor must be positive")
    if args.error_bootstrap < 1: raise SystemExit("--error-bootstrap must be at least 1")
    if args.pi_good <= 0 or args.pi_good >= 1: raise SystemExit("--pi-good must be between 0 and 1")
    output = Path(args.output).expanduser().resolve(); output_dir = output.parent
    if output.exists() and not args.overwrite: raise SystemExit("output exists; use --overwrite: %s" % output)
    started = time.perf_counter(); z_grid = np.linspace(args.z_min, args.z_max, args.n_z)
    if args.n_z < 3 or args.z_max <= args.z_min: raise SystemExit("invalid z grid")
    store_started = time.perf_counter(); store = MeasurementStore(args.measurements); print("measurement load/grouping: %.3f s" % (time.perf_counter() - store_started))
    config = _evidence_config(args, store.path); evidence_started = time.perf_counter()
    cached = _load_evidence_cache(args.evidence_cache, store, z_grid, config)
    if cached is not None:
        evidences, band_contrast, pre_qa = cached
        print("reused compatible evidence cache: %s" % args.evidence_cache)
    else:
        local_settings = (args.pi_good, args.bad_scale, args.p_sigma_fraction,
                          args.alpha_mean, args.alpha_sigma, args.z0_sigma,
                          args.error_floor, args.error_floor_factor, 0.0, 0.0, None)
        preliminary = _calculate_local_evidences(store.blocks, z_grid, local_settings, args.workers)
        contrast_estimate = _estimate_band_contrast(preliminary)
        band_contrast = _complete_band_contrast(contrast_estimate, store)
        pre_qa = _preliminary_qa(preliminary)
        final_settings = (args.pi_good, args.bad_scale, args.p_sigma_fraction,
                          args.alpha_mean, args.alpha_sigma, args.z0_sigma,
                          args.error_floor, args.error_floor_factor,
                          band_contrast["delta_z_band"], band_contrast["delta_p_band"])
        split_summaries = [{"z_mean": evidence.split_z_mean,
                            "z_sigma": evidence.split_z_sigma,
                            "p_mean": evidence.split_p_mean,
                            "p_sigma": evidence.split_p_sigma,
                            "alpha_mean": evidence.split_alpha_mean,
                            "alpha_sigma": evidence.split_alpha_sigma,
                            "grid_edge": evidence.split_grid_edge}
                           for evidence in preliminary]
        evidences = _calculate_local_evidences(
            store.blocks, z_grid,
            [final_settings + (split_summary,)
             for split_summary in split_summaries], args.workers)
        _write_evidence_cache(args.evidence_cache, evidences, z_grid, config,
                              band_contrast, pre_qa)
        print("local evidence calculation/cache: %.3f s" % (time.perf_counter() - evidence_started))
    layout = build_global_layout(store.blocks); exact_indices = list(layout.gamma_index.values())
    posterior_started = time.perf_counter(); selected = np.ones(len(evidences), dtype=bool)
    posterior = solve_global(layout, evidences, selected, args.gamma_sigma, args.ifu_sigma, args.eta_sigma, args.hutchinson_probes, exact_indices)
    reverse_selected = selected[::-1]
    reverse_posterior = solve_global(layout, list(reversed(evidences)), reverse_selected, args.gamma_sigma, args.ifu_sigma, args.eta_sigma, args.hutchinson_probes, exact_indices)
    order_error = float(np.max(np.abs(posterior.mean - reverse_posterior.mean)))
    print("global sparse solve: %.3f s" % (time.perf_counter() - posterior_started))
    history_started = time.perf_counter(); stages = _history(evidences, layout, args.gamma_sigma, args.ifu_sigma, args.eta_sigma, args.hutchinson_probes); print("chronological history: %.3f s" % (time.perf_counter() - history_started))
    obs_rows, amp_accum, ifu_accum, exp_accum = _posterior_rows(
        store, evidences, posterior, args.alpha_mean, args.alpha_sigma,
        band_contrast["delta_z_band"], band_contrast["delta_p_band"])
    physical_rows = _population_rows(amp_accum, posterior); ifu_rows = _ifu_rows(ifu_accum, posterior, store); exposure_rows = _exposure_rows(exp_accum, posterior)
    split_alpha_diagnostic = _split_alpha_diagnostic(evidences)
    error_started = time.perf_counter()
    error_model_diagnostic, error_model_rows, error_model_plot_data = _error_model_diagnostic(
        store, evidences, obs_rows, band_contrast, args.error_bootstrap)
    print("error-model diagnostic: %.3f s" % (time.perf_counter() - error_started))
    post_qa = _preliminary_qa(evidences)
    post_qa.update({"median_mean_residual_ON": float(np.nanmedian([r["mean_residual_ON"] for r in obs_rows])),
                    "median_mean_residual_OFF": float(np.nanmedian([r["mean_residual_OFF"] for r in obs_rows])),
                    "median_mean_residual_ON_minus_OFF": float(np.nanmedian([r["mean_residual_ON"] - r["mean_residual_OFF"] for r in obs_rows]))})
    metadata = {"schema_version": "m101_bayesian_calibration_v1", "created_utc": datetime.now(timezone.utc).isoformat(), "script": str(Path(__file__).resolve()), "git_commit": _git_commit(), "measurement_identity": _h5_identity(store.path), "evidence_cache": _h5_identity(args.evidence_cache), "config": config, "global_priors": {"gamma_sigma": args.gamma_sigma, "ifu_sigma": args.ifu_sigma, "eta_sigma": args.eta_sigma}, "variance_method": posterior.variance_method, "order_independence_max_abs_difference": order_error, "synthetic_validation": run_synthetic_validation(), "contrast_definition": "scipy.linalg.helmert(full=False), iota sums zero per exposure and eta sums zero within each physical IFU", "additive_posterior_conditioning": "p and alpha are conditional on the hierarchy posterior mean z; local marginal moments remain in the evidence cache", "band_contrast": band_contrast, "split_alpha_diagnostic": split_alpha_diagnostic, "qa_pre": pre_qa, "qa_post": post_qa, "no_production_calibration_applied": True}
    output_started = time.perf_counter(); _write_solution(output, obs_rows, physical_rows, ifu_rows, exposure_rows, _history_rows(stages), metadata); _solution_csvs(output_dir, obs_rows, physical_rows, ifu_rows, exposure_rows); _write_csv(output_dir / "m101_bayes_error_model_diagnostic.csv", error_model_rows); _plot_outputs(output_dir, store, evidences, obs_rows, physical_rows, ifu_rows, exposure_rows, stages, posterior, z_grid, band_contrast, split_alpha_diagnostic); _plot_error_model_diagnostic(output_dir, error_model_plot_data, error_model_rows); print("solution/CSV/plots: %.3f s" % (time.perf_counter() - output_started))
    p_values = np.asarray([e.p_good for e in evidences]); info_values = np.asarray([e.I_m for e in evidences]); eta5 = sum(row["P_abs_eta_gt_5pct"] > .95 for row in physical_rows); eta10 = sum(row["P_abs_eta_gt_10pct"] > .95 for row in physical_rows)
    gamma_values = np.asarray([row["gamma_mean"] for row in exposure_rows]); alpha_values = np.asarray([row["alpha_mean"] for row in obs_rows]);
    summary = {"measurement_h5": str(store.path), "output": str(output), "amplifier_observations": len(evidences), "physical_ifus": len(ifu_rows), "persistent_physical_amplifiers": len(physical_rows), "exposures": len(exposure_rows), "evidence_cache": str(Path(args.evidence_cache).resolve()), "median_I_m": float(np.median(info_values)), "I_m_range": [float(np.min(info_values)), float(np.max(info_values))], "median_p_good": float(np.median(p_values)), "p_good_lt_0.5": int(np.sum(p_values < .5)), "p_good_lt_0.1": int(np.sum(p_values < .1)), "gamma_range": [float(np.min(gamma_values)), float(np.max(gamma_values))], "eta_P_gt_5pct_gt_0.95": int(eta5), "eta_P_gt_10pct_gt_0.95": int(eta10), "median_eta_sigma": float(np.median([r["eta_sigma"] for r in physical_rows])), "median_p": float(np.median([r["p_mean"] for r in obs_rows])), "median_alpha": float(np.median(alpha_values)), "alpha_range": [float(np.min(alpha_values)), float(np.max(alpha_values))], "grid_edge_flags": int(sum(e.grid_edge_flag for e in evidences)), "strong_split_preferences": int(sum(e.split_minus_joint_log_evidence > 5 for e in evidences)), "order_independence_max_abs_difference": order_error, "synthetic_validation": metadata["synthetic_validation"], "band_contrast": band_contrast, "split_alpha_diagnostic": split_alpha_diagnostic, "error_model_diagnostic": error_model_diagnostic, "qa_pre": pre_qa, "qa_post": post_qa, "total_runtime_seconds": time.perf_counter() - started}
    (output_dir / "m101_bayes_summary.json").write_text(json.dumps(summary, indent=2, default=str)); print(json.dumps(summary, indent=2, default=str)); print("NO production files modified; no calibration correction was applied to measurements.")


if __name__ == "__main__":
    main()
