#!/usr/bin/env python3
"""Trace the production M101 cube contributors at marked cube positions.

The engine in this file is deliberately a contributor diagnostic, not a cube
builder.  It reads the production H5 metadata and the one requested native
spectral plane, reuses the current-model finite/mask rules, and evaluates the
same 1.8-arcsec Gaussian stencil used by ``make_m101_cube_current_model.py``.

The final median across shots is not a linear operation.  ``effective_weight``
therefore means the exact linear allocation into each production shot image,
with each supported shot receiving 1/N_shots of the allocation.  The reported
fractions are normalized sums of those allocations over valid production
pixels.  ``geometric_weight`` and zero-fraction rows preserve the distinction
between footprint overlap and actual finite production input.
"""

from __future__ import annotations

import argparse
import atexit
import csv
from dataclasses import dataclass, field
from datetime import datetime, timezone
import glob
import json
from pathlib import Path
import sys
import time

import numpy as np
import tables
from astropy.io import fits
import astropy.units as u
from astropy.wcs import WCS
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import diagnose_m101_hierarchical as validated_m101
import diagnose_m101_frozenM_spectral_residuals as frozen_parent
from astrometry import Astrometry
from extract import Extract
import make_m101_cube_current_model as production


WAVE = np.asarray(production.DEF_WAVE, dtype=float)
FWHM_ARCSEC = float(production.GAUSSIAN_FWHM_ARCSEC)
FIBER_RADIUS_ARCSEC = float(production.FIBER_RADIUS_ARCSEC)
N_FIBER_AMP = int(production.N_FIBER_AMP)
N_EXPOSURES = int(production.N_EXPOSURES)
Q_MIN = int(production.Q_MIN)
Q_MAX = int(production.Q_MAX)
MIN_FINITE_FRACTION = float(production.M101_SKY_MIN_FINITE_FRACTION)
DEFAULT_MIN_SKY_FIBERS = int(production.M101_SKY_MIN_FIBERS)
STENCIL_RADIUS = 3
PRODUCTION_SIGMA = FWHM_ARCSEC / 2.35


def _identity(path):
    path = Path(path).expanduser().resolve()
    stat = path.stat()
    return {"path": str(path), "size": int(stat.st_size),
            "mtime_ns": int(stat.st_mtime_ns)}


def _as_text(value):
    if isinstance(value, (bytes, np.bytes_)):
        return value.decode("utf-8", errors="replace").strip()
    return str(value).strip()


def _write_csv(path, rows, fields):
    path = Path(path)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _json_path(value, base=None):
    if value is None:
        return None
    path = Path(str(value)).expanduser()
    if path.exists():
        return path.resolve()
    if base is not None:
        candidate = (Path(base) / path).resolve()
        if candidate.exists():
            return candidate
    candidate = (ROOT / path.name).resolve()
    return candidate if candidate.exists() else path.resolve()


def _find_summary(cube, explicit):
    if explicit:
        path = Path(explicit).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError("production summary not found: %s" % path)
        return path
    stem = Path(cube).stem
    guesses = [
        Path(cube).with_name(stem.replace("_cube", "") + "_current_model_summary.json"),
        Path(cube).with_name(stem + "_current_model_summary.json"),
        ROOT / "m101_virus_v2_rebuild_current_model_summary.json",
        ROOT / "m101_cube_current_model_preflight.json",
    ]
    for guess in guesses:
        if guess.is_file():
            return guess.resolve()
    return None


def _model_value(args, summary, section, option, default=None, required=False):
    explicit = getattr(args, option)
    if explicit:
        return Path(explicit).expanduser().resolve()
    if summary:
        value = summary.get("provenance", {}).get(section, {}).get("path")
        if not value:
            value = summary.get("model_products", {}).get(section, {}).get("path")
        if value:
            path = _json_path(value, getattr(summary, "parent", ROOT))
            if path.is_file():
                return path
    if default is not None:
        path = _json_path(default, ROOT)
        if path.is_file():
            return path
    if required:
        raise FileNotFoundError(
            "could not resolve %s; pass --%s (or --summary with provenance)" %
            (section, option.replace("_", "-")))
    return None


def _resolve_h5(args, summary):
    paths = []
    for value in args.h5:
        paths.extend(Path(item).expanduser().resolve()
                     for item in glob.glob(value))
    for value in args.h5_glob:
        paths.extend(Path(item).expanduser().resolve()
                     for item in glob.glob(value))
    if not paths and summary:
        paths = [Path(item).expanduser().resolve()
                 for item in summary.get("input_h5_files", [])]
    unique = {path.name: path for path in paths if path.is_file()}
    if not unique:
        raise FileNotFoundError(
            "no source H5 files resolved; pass --h5/--h5-glob or --summary "
            "whose input_h5_files are accessible")
    return [unique[name] for name in sorted(unique)]


def _resolve_reject(explicit, candidates, summary=None, summary_key=None):
    if explicit:
        path = Path(explicit).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        return path
    if summary and summary_key:
        value = summary.get("qa", {}).get(summary_key)
        if not value:
            value = summary.get("spectral_rejection", {}).get(summary_key)
        if isinstance(value, dict):
            value = value.get("path")
        if value:
            path = _json_path(value, getattr(summary, "parent", ROOT))
            if path.is_file():
                return path
    for value in candidates:
        path = ROOT / value
        if path.is_file():
            return path.resolve()
    return None


def _parse_rejects(qa_path, spectral_path):
    qa_keys, spectral_entries = set(), set()
    if qa_path:
        qa_keys, _ = production._load_qa_reject_file(qa_path)
    if spectral_path:
        spectral_entries, _ = production._load_spectral_reject_file(spectral_path)
    return qa_keys, spectral_entries


@dataclass
class GroupMeta:
    h5: str
    exposure: int
    specid: int
    ifuslot: int
    ifuid: int
    amp: str
    ifu_code: int
    indices: np.ndarray
    q: np.ndarray
    group_id: str
    physical_amp_id: str
    qa_rejected: bool = False


@dataclass
class Shot:
    h5_path: Path
    h5_name: str
    exposure: int
    survey: dict
    ra: np.ndarray
    dec: np.ndarray
    rows: np.ndarray
    groups: list
    group_by_row: np.ndarray
    physical_by_row: np.ndarray
    ifu_code_by_row: np.ndarray
    q_by_row: np.ndarray
    hardware_bad: np.ndarray
    native_sufficient: np.ndarray
    blank_mask: np.ndarray
    base_xy: np.ndarray
    tree: cKDTree
    adr_ra: np.ndarray
    adr_dec: np.ndarray
    wave_validity_cache: dict = field(default_factory=dict)
    h5_handle: object = None


def _physical_amp_id(specid, ifuslot, ifuid, amp):
    return "%d/%d/%d/%s" % (int(specid), int(ifuslot), int(ifuid),
                             str(amp).upper())


def _group_id(h5, exposure, specid, ifuslot, ifuid, amp):
    return "%s/e%d/%s" % (
        Path(h5).name, int(exposure),
        _physical_amp_id(specid, ifuslot, ifuid, amp))


def _pixel_scale_from_wcs(wcs):
    matrix = np.asarray(wcs.pixel_scale_matrix[:2, :2], dtype=float)
    scales = np.sqrt(np.sum(matrix * matrix, axis=0)) * 3600.0
    finite = scales[np.isfinite(scales) & (scales > 0)]
    return float(np.median(finite)) if finite.size else 1.0


class ContributorEngine:
    """One initialized geometry/model session shared by all query modes."""

    def __init__(self, args):
        started = time.perf_counter()
        self.cube_path = Path(args.cube).expanduser().resolve()
        self.header = fits.getheader(self.cube_path, 0)
        self.cube_shape = tuple(int(v) for v in fits.getdata(
            self.cube_path, 0, memmap=True).shape)
        self.wcs = WCS(self.header)
        self.pixel_scale = _pixel_scale_from_wcs(self.wcs)
        self.sigma = PRODUCTION_SIGMA / self.pixel_scale
        self.support_radius = int(np.ceil(2.0 * self.sigma))
        self.summary_path = _find_summary(self.cube_path, args.summary)
        self.summary = (json.loads(self.summary_path.read_text())
                        if self.summary_path else {})
        self.h5_paths = _resolve_h5(args, self.summary)
        self.qa_path = _resolve_reject(
            args.qa_reject_file,
            ("m101_cube_current_model_qa_reject.csv",), self.summary,
            "reject_file")
        self.spectral_path = _resolve_reject(
            args.spectral_reject_file,
            ("m101_targeted_spectral_reject.csv",), self.summary,
            "spectral_reject_file")
        self.qa_rejects, self.spectral_rejects = _parse_rejects(
            self.qa_path, self.spectral_path)
        self.min_sky_fibers = int(args.min_sky_fibers)

        self._load_model_products(args)
        self.shots = []
        self._load_geometry()
        self.initialization_seconds = time.perf_counter() - started

    def close(self):
        for shot in self.shots:
            if shot.h5_handle is not None:
                shot.h5_handle.close()
                shot.h5_handle = None

    def _load_model_products(self, args):
        state = _model_value(args, self.summary, "state", "state",
                             "m101_onoff_hierarchy/m101_onoff_state.json", True)
        cache = _model_value(args, self.summary, "band_cache", "band_cache",
                             "m101_band_initializer/m101_band_cache.npz", True)
        alpha_fits = _model_value(args, self.summary, "alpha_fits", "alpha_fits",
                                  "m101_alpha_template_bands/alpha_template_fits.csv", True)
        alpha_history = _model_value(args, self.summary, "alpha_history", "alpha_summary",
                                     "m101_alpha_same_amp_loo/alpha_physical_amp_summary.csv", True)
        amp_means = _model_value(args, self.summary, "amp_means", "amp_residual_models",
                                 "m101-amp-residual-pca-final/amp_residual_pca_models.npz", True)
        fq = _model_value(args, self.summary, "fq_template", "fq_template", required=True)
        blank = _model_value(args, self.summary, "blank_file", "external_blank_fibers", required=True)
        self.model_paths = {
            "state": state, "band_cache": cache, "alpha_fits": alpha_fits,
            "alpha_history": alpha_history, "amp_means": amp_means,
            "fq_template": fq, "blank_file": blank,
        }
        self.state, self.response, self.composition, _ = production._load_frozen_response(state)
        self.identity_map, self.cache_manifest, _, _ = production._load_identity_map(
            cache, self.h5_paths)
        self.alpha_direct, _ = production._load_alpha_fits(alpha_fits)
        self.alpha_history, _ = production._load_alpha_history(alpha_history)
        self.amp_means, _, _, _ = production._load_amp_means(amp_means)
        self.fq, self.global_f0, _ = production._load_fq(fq)
        self.blank_masks, self.blank_provenance = frozen_parent.m101_blank_fibers.load(
            blank, self.h5_paths)

    def _load_geometry(self):
        extractor = Extract(wave=WAVE)
        for h5_path in self.h5_paths:
            with tables.open_file(h5_path, mode="r") as h5:
                info = h5.root.Info
                groups, labels = validated_m101.build_groups(info)
                ra = np.asarray(info.cols.ra[:], dtype=float)
                dec = np.asarray(info.cols.dec[:], dtype=float)
                surveys = production._survey_by_exposure(h5)
                hardware = np.zeros(info.nrows, dtype=bool)
                group_by_row = np.full(info.nrows, -1, dtype=np.int32)
                physical_by_row = np.empty(info.nrows, dtype=object)
                ifu_code_by_row = np.full(info.nrows, -1, dtype=np.int32)
                q_by_row = np.full(info.nrows, -1, dtype=np.int16)
                meta_groups = []
                for group_number, group in enumerate(groups):
                    physical = (int(group["specid"]), int(group["ifuslot"]),
                                int(group["ifuid"]), str(group["amp"]).upper())
                    physical3 = physical[:3]
                    ifu_code = int(self.identity_map[(h5_path.name, physical3)])
                    indices = np.asarray(group["indices"], dtype=np.int64)
                    q = production._group_q(group)
                    bad = production.hardware_excluded(
                        date=h5_path.name[:8], h5=h5_path.name,
                        specid=physical[0], ifuslot=physical[1], ifuid=physical[2],
                        amp=physical[3], purpose="cube")
                    hardware[indices] = bool(bad)
                    group_by_row[indices] = group_number
                    physical_by_row[indices] = _physical_amp_id(*physical)
                    ifu_code_by_row[indices] = ifu_code
                    q_by_row[indices] = q
                    meta_groups.append(GroupMeta(
                        h5=h5_path.name, exposure=int(group["exposure"]),
                        specid=physical[0], ifuslot=physical[1], ifuid=physical[2],
                        amp=physical[3], ifu_code=ifu_code, indices=indices,
                        q=q, group_id=_group_id(h5_path.name, group["exposure"], *physical),
                        physical_amp_id=_physical_amp_id(*physical),
                        qa_rejected=(h5_path.name, int(group["exposure"]), *physical) in self.qa_rejects))

                sufficient = self._native_sufficient(h5, surveys, info.nrows)
                blank = np.asarray(self.blank_masks[h5_path.name], dtype=bool)
                for exposure in range(1, N_EXPOSURES + 1):
                    selected = np.flatnonzero(labels == exposure)
                    survey = surveys[exposure]
                    survey_ra = float(survey["ra"])
                    survey_dec = float(survey["dec"])
                    astrometry = Astrometry(survey_ra, survey_dec,
                                            float(survey["pa"]), 0.0, 0.0)
                    extractor.get_ADR_RAdec(astrometry)
                    adr_ra = np.asarray(extractor.ADRra, dtype=float)
                    adr_dec = np.asarray(extractor.ADRdec, dtype=float)
                    # The midpoint plane makes a compact spatial index.  The
                    # query adds the exact ADR excursion as a conservative
                    # candidate margin and then recomputes exact positions.
                    mid = len(WAVE) // 2
                    corrected_ra = (ra[selected] - adr_ra[mid] / 3600.0 /
                                    np.cos(np.deg2rad(survey_dec)))
                    corrected_dec = dec[selected] - adr_dec[mid] / 3600.0
                    bx, by = self.wcs.celestial.world_to_pixel_values(
                        corrected_ra, corrected_dec)
                    base_xy = np.column_stack((bx + 1.0, by + 1.0))
                    finite = np.isfinite(base_xy).all(axis=1)
                    if not np.any(finite):
                        continue
                    # Entries are grouped by exposure, matching production's
                    # shot_indices exactly.
                    global_group = group_by_row[selected]
                    global_group_numbers = [int(i) for i in
                                            sorted(set(global_group.tolist())) if i >= 0]
                    shot_groups = [meta_groups[i] for i in global_group_numbers]
                    global_to_local = {value: index for index, value in
                                       enumerate(global_group_numbers)}
                    local_group = np.asarray(
                        [global_to_local.get(int(value), -1)
                         for value in global_group], dtype=np.int32)
                    self.shots.append(Shot(
                        h5_path=h5_path, h5_name=h5_path.name, exposure=exposure,
                        survey=survey, ra=ra[selected], dec=dec[selected],
                        rows=selected, groups=shot_groups,
                        group_by_row=local_group,
                        physical_by_row=physical_by_row[selected],
                        ifu_code_by_row=ifu_code_by_row[selected],
                        q_by_row=q_by_row[selected], hardware_bad=hardware[selected],
                        native_sufficient=sufficient[selected], blank_mask=blank[selected],
                        base_xy=base_xy, tree=cKDTree(base_xy[finite]),
                        adr_ra=adr_ra, adr_dec=adr_dec))
                    # cKDTree has only finite rows; remember mapping through a
                    # compact field attached after construction.
                    self.shots[-1].tree_indices = np.flatnonzero(finite)

    @staticmethod
    def _native_sufficient(h5, surveys, nrows, chunk=2048):
        fibers = h5.root.Fibers
        result = np.zeros(nrows, dtype=bool)
        for start in range(0, nrows, chunk):
            stop = min(nrows, start + chunk)
            source = np.asarray(fibers.read_coordinates(
                np.arange(start, stop), field="spectrum"), dtype=float)
            sky = np.asarray(fibers.read_coordinates(
                np.arange(start, stop), field="skyspectrum"), dtype=float)
            offsets = np.asarray([
                float(surveys[int((i // N_FIBER_AMP) % N_EXPOSURES) + 1]["offset"])
                for i in range(start, stop)], dtype=float)
            total = source / offsets[:, None] + sky
            result[start:stop] = (
                np.mean(np.isfinite(total), axis=1) >= MIN_FINITE_FRACTION)
        return result

    def _open(self, shot):
        if shot.h5_handle is None:
            shot.h5_handle = tables.open_file(shot.h5_path, mode="r")
        return shot.h5_handle

    def _exact_positions(self, shot, wave_index, row_indices):
        survey_dec = float(shot.survey["dec"])
        ra = shot.ra[row_indices] - shot.adr_ra[wave_index] / 3600.0 / np.cos(np.deg2rad(survey_dec))
        dec = shot.dec[row_indices] - shot.adr_dec[wave_index] / 3600.0
        x, y = self.wcs.celestial.world_to_pixel_values(ra, dec)
        return np.column_stack((x + 1.0, y + 1.0))

    def _wavelength_from_z(self, z):
        value = float(self.wcs.spectral.pixel_to_world_values(float(z) - 1.0))
        units = self.wcs.spectral.world_axis_units
        unit = u.Unit(units[0]) if units and units[0] else u.AA
        return float((value * unit).to_value(u.AA))

    def _z_from_wavelength(self, wavelength):
        units = self.wcs.spectral.world_axis_units
        unit = u.Unit(units[0]) if units and units[0] else u.AA
        value = (float(wavelength) * u.AA).to_value(unit)
        return float(self.wcs.spectral.world_to_pixel_values(value) + 1.0)

    def _validity(self, shot, wave_index):
        if wave_index in shot.wave_validity_cache:
            return shot.wave_validity_cache[wave_index]
        h5 = self._open(shot)
        fibers = h5.root.Fibers
        all_rows = shot.rows
        try:
            source_wave = np.asarray(
                fibers.cols.spectrum[all_rows, wave_index], dtype=float)
            sky_wave = np.asarray(
                fibers.cols.skyspectrum[all_rows, wave_index], dtype=float)
        except Exception:
            source = np.asarray(fibers.read_coordinates(
                all_rows, field="spectrum"), dtype=float)
            native_sky = np.asarray(fibers.read_coordinates(
                all_rows, field="skyspectrum"), dtype=float)
            source_wave = source[:, wave_index]
            sky_wave = native_sky[:, wave_index]
        offset = float(shot.survey["offset"])
        valid = np.zeros(all_rows.size, dtype=bool)
        corrected = np.full(all_rows.size, np.nan, dtype=float)
        k_wave = float(validated_m101.raw_work_basis(shot.survey)[wave_index])
        # Read the row's model scalar/group once.  This follows the current
        # production equation but does not materialize or write a cube.
        for group in shot.groups:
            local = np.flatnonzero(shot.group_by_row == shot.groups.index(group))
            if not local.size:
                continue
            physical = (group.specid, group.ifuslot, group.ifuid, group.amp)
            m_key = (shot.h5_name, group.ifu_code, group.amp)
            if group.qa_rejected or m_key not in self.response:
                continue
            mu_entry = self.amp_means.get(physical)
            if mu_entry is None:
                continue
            alpha, f0, _ = production._alpha_for_group(
                self.alpha_direct, self.alpha_history, shot.h5_name,
                group.exposure, physical, group.ifu_code, self.global_f0)
            m = float(np.exp(float(self.response[m_key])))
            q = shot.q_by_row[local].astype(int)
            mu = float(mu_entry["mean"][wave_index])
            mu = 0.0 if not np.isfinite(mu) else mu
            corrected[local] = ((source_wave[local] / offset + sky_wave[local])
                                - alpha * (self.fq[q] - f0) * k_wave - mu) / m
        # Production sky selection is based on full-spectrum sufficiency,
        # then a robust location at this wavelength.
        for group in shot.groups:
            local = np.flatnonzero(shot.group_by_row == shot.groups.index(group))
            if not local.size:
                continue
            if group.qa_rejected:
                continue
        sky_mask = (shot.blank_mask & ~shot.hardware_bad &
                    shot.native_sufficient & (shot.q_by_row >= Q_MIN) &
                    (shot.q_by_row <= Q_MAX) & np.isfinite(corrected))
        if np.count_nonzero(sky_mask) >= self.min_sky_fibers:
            sky = frozen_parent._safe_location(corrected[sky_mask])
            if np.isfinite(sky):
                valid = np.isfinite(corrected - sky)
        # Production's splat accepts a finite per-plane SCI value; the
        # >=80%-finite criterion is used for blank-sky selection above, not
        # as an extra per-fiber rejection in make_image_gaussian.
        valid &= ~shot.hardware_bad
        valid &= np.isfinite(corrected)
        # Exact wavelength-specific reviewed masks are production masks.
        for group in shot.groups:
            if (shot.h5_name, group.exposure, group.specid, group.ifuslot,
                    group.ifuid, group.amp, wave_index) in self.spectral_rejects:
                valid[shot.group_by_row == shot.groups.index(group)] = False
        shot.wave_validity_cache[wave_index] = valid
        return valid

    def _region_pixels(self, x, y, radius):
        nx, ny = self.cube_shape[-1], self.cube_shape[-2]
        cx, cy = int(np.rint(x)), int(np.rint(y))
        r = float(radius)
        if r <= 0:
            return [(max(1, min(nx, cx)), max(1, min(ny, cy)))]
        pixels = []
        for py in range(max(1, int(np.floor(y - r))),
                       min(ny, int(np.ceil(y + r))) + 1):
            for px in range(max(1, int(np.floor(x - r))),
                           min(nx, int(np.ceil(x + r))) + 1):
                if (px - x) ** 2 + (py - y) ** 2 <= r * r + 1.0e-12:
                    pixels.append((px, py))
        return pixels

    def _candidate_rows(self, shot, pixels):
        margin = STENCIL_RADIUS + self.support_radius + 2.0
        # ADR is a wavelength-dependent displacement; the midpoint tree is
        # expanded by the maximum observed excursion for this exposure.
        survey_dec = float(shot.survey["dec"])
        mid_ra = shot.adr_ra[len(WAVE) // 2]
        mid_dec = shot.adr_dec[len(WAVE) // 2]
        max_shift = np.hypot(
            np.max(np.abs(shot.adr_ra - mid_ra)) / 3600.0 /
            np.cos(np.deg2rad(survey_dec)) / (self.pixel_scale / 3600.0),
            np.max(np.abs(shot.adr_dec - mid_dec)) / 3600.0 /
            (self.pixel_scale / 3600.0))
        found = set()
        for px, py in pixels:
            for tree_index in shot.tree.query_ball_point(
                    [px, py], float(margin + max_shift)):
                found.add(int(shot.tree_indices[tree_index]))
        return np.asarray(sorted(found), dtype=np.int64)

    def query(self, x, y, z=None, wavelength=None, radius=0.0, label=""):
        if (z is None) == (wavelength is None):
            raise ValueError("provide exactly one of z or wavelength")
        if z is not None:
            z_float = float(z)
            wave = self._wavelength_from_z(z_float)
        else:
            wave = float(wavelength)
            z_float = self._z_from_wavelength(wave)
        wave_index = int(np.rint(z_float - 1.0))
        if not 0 <= wave_index < len(WAVE):
            raise ValueError("wavelength/plane is outside cube spectral range")
        plane_z = float(wave_index + 1)
        plane_wave = float(WAVE[wave_index])
        pixels = self._region_pixels(float(x), float(y), float(radius))
        group_stats = {}
        for shot in self.shots:
            candidate = self._candidate_rows(shot, pixels)
            if not candidate.size:
                continue
            positions = self._exact_positions(shot, wave_index, candidate)
            validity = self._validity(shot, wave_index)
            # Per-pixel group weight sums.  The arrays are small because the
            # spatial index has already removed distant fibers.
            shot_pixel = {}
            for (px, py) in pixels:
                dx = positions[:, 0] - px
                dy = positions[:, 1] - py
                # Match the production loop's integer stencil exactly.  It
                # is centered on floor(fiber_position - x_origin), not on a
                # symmetric distance threshold around the subpixel position.
                ix_center = np.floor(positions[:, 0] - 1.0).astype(np.int64) + 1
                iy_center = np.floor(positions[:, 1] - 1.0).astype(np.int64) + 1
                geometric = (np.abs(px - ix_center) <= STENCIL_RADIUS) & \
                            (np.abs(py - iy_center) <= STENCIL_RADIUS)
                if not np.any(geometric):
                    continue
                actual = geometric & validity[candidate]
                support = (
                    (np.abs(px - ix_center) <= self.support_radius) &
                    (np.abs(py - iy_center) <= self.support_radius) &
                    (np.hypot(dx, dy) <= (2.0 * self.sigma + 1e-12)))
                actual_support = support & validity[candidate]
                geo_weights = np.asarray([
                    production.production_gaussian_weight(px0, py0, px, py, self.sigma)
                    if good else 0.0
                    for (px0, py0), good in zip(positions, geometric)], dtype=float)
                # Retain geometric-only groups so a masked/rejected footprint
                # remains visible in the evidence table with zero effective
                # contribution.
                for local_i, row_i in enumerate(candidate):
                    if not geometric[local_i]:
                        continue
                    group_no = int(shot.group_by_row[row_i])
                    group = shot.groups[group_no]
                    item = group_stats.setdefault(group.group_id, {
                        "group": group, "geometric_weight": 0.0,
                        "raw_effective": 0.0, "geo_rows": set(),
                        "actual_rows": set(), "pixel_values": {}})
                    item["geometric_weight"] += float(geo_weights[local_i])
                    item["geo_rows"].add(int(shot.rows[row_i]))
                if not np.any(actual_support) or not np.any(actual):
                    continue
                weights = np.asarray([
                    production.production_gaussian_weight(px0, py0, px, py, self.sigma)
                    if good else 0.0
                    for (px0, py0), good in zip(positions, actual)], dtype=float)
                total = float(np.sum(weights))
                if total <= 0:
                    continue
                # Production's ncontrib is independent-shot support, and the
                # final SCI is valid only for at least two supported shots.
                shot_pixel[(px, py)] = (candidate, geometric, actual,
                                        geo_weights, weights, total)
            # Accumulate provisional per-shot allocations.  We normalize by
            # the number of shots supporting each pixel after all shots are
            # known below, so retain pixel-local records first.
            for key, values in shot_pixel.items():
                candidate, geometric, actual, geo_weights, weights, total = values
                for local_i, row_i in enumerate(candidate):
                    group_no = int(shot.group_by_row[row_i])
                    group = shot.groups[group_no]
                    item = group_stats.setdefault(group.group_id, {
                        "group": group, "geometric_weight": 0.0,
                        "raw_effective": 0.0, "geo_rows": set(),
                        "actual_rows": set(), "pixel_values": {}})
                    if actual[local_i]:
                        item["actual_rows"].add(int(shot.rows[row_i]))
                        item["pixel_values"].setdefault(key, []).append(
                            float(weights[local_i] / total))

        # Apply the production median-support requirement and equal shot share.
        # Count support by H5/exposure shot, not by amplifier group: one shot
        # contains many groups but contributes one production median sample.
        pixel_shot_ids = {}
        for item in group_stats.values():
            for pixel in item["pixel_values"]:
                pixel_shot_ids.setdefault(pixel, set()).add(
                    (item["group"].h5, item["group"].exposure))
        for item in group_stats.values():
            raw = 0.0
            for pixel, values in item["pixel_values"].items():
                nshots = len(pixel_shot_ids[pixel])
                if nshots >= 2:
                    raw += float(np.sum(values)) / nshots
            item["raw_effective"] = raw
        total_effective = float(sum(item["raw_effective"]
                                    for item in group_stats.values()))
        valid_pixels = sum(1 for shots in pixel_shot_ids.values()
                           if len(shots) >= 2)
        rows = []
        ordered = sorted(group_stats.values(),
                         key=lambda item: (item["raw_effective"] > 0,
                                           item["raw_effective"],
                                           item["geometric_weight"]), reverse=True)
        for rank, item in enumerate(ordered, 1):
            group = item["group"]
            fraction = (item["raw_effective"] / total_effective
                        if total_effective > 0 else 0.0)
            geo_count = len(item["geo_rows"])
            actual_count = len(item["actual_rows"])
            rows.append({
                "rank": rank, "fractional_contribution": fraction,
                "effective_weight": item["raw_effective"],
                "geometric_weight": item["geometric_weight"],
                "file": group.h5, "exposure": group.exposure,
                "SPECID": group.specid, "IFUSLOT": group.ifuslot,
                "IFUID": group.ifuid, "AMP": group.amp,
                "IFU_CODE": group.ifu_code,
                "physical_amp_id": group.physical_amp_id,
                "group_id": group.group_id,
                "fiber_rows": ",".join(str(v) for v in sorted(item["actual_rows"])),
                "fiber_row_count": actual_count,
                "n_contributing_fibers": actual_count,
                "masked_fraction": (1.0 - actual_count / geo_count
                                    if geo_count else np.nan),
                "masked": bool(geo_count and not actual_count),
                "geometric_contributor": bool(geo_count),
                "effective_contributor": bool(item["raw_effective"] > 0),
                "geometric_fiber_count": geo_count,
                "qa_rejected": group.qa_rejected,
            })
        amp_totals = {}
        for row in rows:
            amp = row["physical_amp_id"]
            amp_totals.setdefault(amp, 0.0)
            amp_totals[amp] += float(row["fractional_contribution"])
        summary = {
            "input_x": float(x), "input_y": float(y), "input_z": float(z_float),
            "wavelength": plane_wave, "requested_wavelength": wave,
            "radius": float(radius), "label": label,
            "plane_z": plane_z, "n_region_pixels": len(pixels),
            "n_valid_production_pixels": int(valid_pixels),
            "n_contributing_fibers": int(sum(row["n_contributing_fibers"] for row in rows)),
            "n_contributing_group_instances": int(sum(row["effective_weight"] > 0 for row in rows)),
            "n_unique_physical_amplifiers": int(sum(v > 0 for v in amp_totals.values())),
            "dominant_physical_amplifier": (max(amp_totals, key=amp_totals.get)
                                             if amp_totals else ""),
            "dominant_amplifier_fraction": (max(amp_totals.values())
                                             if amp_totals else 0.0),
            "top3_contribution_fraction": float(sum(
                sorted(amp_totals.values(), reverse=True)[:3])),
            "fraction_sum": float(sum(row["fractional_contribution"] for row in rows)),
            "rows": rows,
        }
        return summary

    def provenance(self):
        return {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "cube": _identity(self.cube_path),
            "cube_header": {key: self.header[key] for key in
                            ("NAXIS1", "NAXIS2", "NAXIS3", "CRVAL1", "CRPIX1",
                             "CRVAL2", "CRPIX2", "CRVAL3", "CRPIX3", "CDELT3",
                             "CTYPE1", "CTYPE2", "CTYPE3") if key in self.header},
            "source_h5_files": [_identity(path) for path in self.h5_paths],
            "reconstruction_code": _identity(ROOT / "make_m101_cube_current_model.py"),
            "production_summary": (_identity(self.summary_path)
                                    if self.summary_path else None),
            "model_products": {key: _identity(value)
                               for key, value in self.model_paths.items()},
            "rejection_inputs": {
                "qa_reject_file": _identity(self.qa_path) if self.qa_path else None,
                "spectral_reject_file": _identity(self.spectral_path) if self.spectral_path else None,
            },
            "coordinate_convention": "FITS cube pixel coordinates, 1-based; x/y are pixel centers; z is 1-based spectral plane",
            "weighting_method": "production 1.8 arcsec Gaussian, sigma=FWHM/2.35, radius-3 stencil, per-shot normalized weight, equal allocation across supported shots, median support >= 2",
            "finite_mask_method": "current-model native total, finite-sufficient row, frozen-M/alpha/mu correction, central-q blank sky, hardware/QA/spectral masks",
            "initialization_seconds": self.initialization_seconds,
            "shots": len(self.shots),
            "pixel_scale_arcsec": self.pixel_scale,
        }


LOCATION_FIELDS = [
    "location_id", "input_x", "input_y", "input_z", "wavelength", "radius", "label", "notes",
    "rank", "fractional_contribution", "effective_weight", "geometric_weight",
    "file", "exposure", "SPECID", "IFUSLOT", "IFUID", "AMP", "IFU_CODE",
    "physical_amp_id", "group_id", "fiber_rows", "fiber_row_count",
    "n_contributing_fibers", "masked_fraction", "masked", "geometric_fiber_count",
    "geometric_contributor", "effective_contributor", "qa_rejected",
]

AMPLIFIER_FIELDS = [
    "physical_amp_id", "SPECID", "IFUSLOT", "IFUID", "AMP",
    "n_marked_locations_contributed", "n_marked_locations_dominant",
    "n_unique_wavelengths", "n_unique_exposures", "n_unique_shots",
    "median_fractional_contribution", "mean_fractional_contribution",
    "max_fractional_contribution", "median_fraction_when_dominant",
    "max_fraction_when_dominant", "location_ids", "wavelengths",
    "fraction_marked_locations_appears", "fraction_marked_locations_dominant",
]


def _amplifier_summary(results):
    locations = len(results)
    by_amp = {}
    for result in results:
        per_amp = {}
        for row in result["rows"]:
            if row["fractional_contribution"] <= 0:
                continue
            per_amp.setdefault(row["physical_amp_id"], []).append(row)
        for amp, rows in per_amp.items():
            first = rows[0]
            fraction = float(sum(row["fractional_contribution"] for row in rows))
            item = by_amp.setdefault(amp, {"fractions": [], "dominant": [],
                                          "locations": [], "waves": [], "exposures": set(),
                                          "shots": set(), "first": first})
            item["fractions"].append(fraction)
            item["locations"].append(result["location_id"])
            item["waves"].append(result["wavelength"])
            for row in rows:
                item["exposures"].add(str(row["exposure"]))
                item["shots"].add("%s/e%s" % (row["file"], row["exposure"]))
            max_fraction = max(
                sum(r["fractional_contribution"] for r in result["rows"]
                    if r["physical_amp_id"] == candidate)
                for candidate in {r["physical_amp_id"] for r in result["rows"]})
            if np.isclose(fraction, max_fraction, rtol=0, atol=1e-14):
                item["dominant"].append(fraction)
    output = []
    for amp, item in by_amp.items():
        first = item["first"]
        fractions = np.asarray(item["fractions"], dtype=float)
        dominant = np.asarray(item["dominant"], dtype=float)
        output.append({
            "physical_amp_id": amp, "SPECID": first["SPECID"],
            "IFUSLOT": first["IFUSLOT"], "IFUID": first["IFUID"], "AMP": first["AMP"],
            "n_marked_locations_contributed": len(item["locations"]),
            "n_marked_locations_dominant": len(item["dominant"]),
            "n_unique_wavelengths": len(set(item["waves"])),
            "n_unique_exposures": len(item["exposures"]),
            "n_unique_shots": len(item["shots"]),
            "median_fractional_contribution": float(np.median(fractions)),
            "mean_fractional_contribution": float(np.mean(fractions)),
            "max_fractional_contribution": float(np.max(fractions)),
            "median_fraction_when_dominant": float(np.median(dominant)) if dominant.size else np.nan,
            "max_fraction_when_dominant": float(np.max(dominant)) if dominant.size else np.nan,
            "location_ids": ",".join(str(v) for v in item["locations"]),
            "wavelengths": ",".join("%.6g" % v for v in sorted(set(item["waves"]))),
            "fraction_marked_locations_appears": len(item["locations"]) / locations if locations else 0.0,
            "fraction_marked_locations_dominant": len(item["dominant"]) / locations if locations else 0.0,
        })
    return sorted(output, key=lambda row: (
        row["n_marked_locations_dominant"],
        row["n_marked_locations_contributed"],
        row["median_fractional_contribution"]), reverse=True)


def _print_single(result):
    print("location: x=%.6f y=%.6f z=%.6f wave=%.6f A radius=%.3f" % (
        result["input_x"], result["input_y"], result["plane_z"],
        result["wavelength"], result["radius"]))
    print("rank fraction effective group file exposure SPECID/IFUSLOT/IFUID/AMP fibers masked")
    for row in result["rows"][:30]:
        print("%4d %.6f %.6f %-36s %-26s %d %s/%s/%s/%s %4d %.3f" % (
            row["rank"], row["fractional_contribution"], row["effective_weight"],
            row["group_id"], row["file"], row["exposure"], row["SPECID"],
            row["IFUSLOT"], row["IFUID"], row["AMP"],
            row["n_contributing_fibers"], row["masked_fraction"]))
    for key in ("dominant_physical_amplifier", "dominant_amplifier_fraction",
                "top3_contribution_fraction", "n_contributing_fibers",
                "n_contributing_group_instances", "n_unique_physical_amplifiers",
                "fraction_sum", "n_valid_production_pixels"):
        print("%-36s %s" % (key, result[key]))


def _load_locations(path):
    with Path(path).open(newline="") as stream:
        reader = csv.DictReader(stream)
        fields = set(reader.fieldnames or ())
        if not {"x", "y"}.issubset(fields):
            raise ValueError("input CSV must contain x,y and z or wavelength/wave")
        spectral = "z" if "z" in fields else "wavelength" if "wavelength" in fields else "wave" if "wave" in fields else None
        if spectral is None:
            raise ValueError("input CSV must contain z, wavelength, or wave")
        rows = []
        for index, row in enumerate(reader):
            value = {"x": float(row["x"]), "y": float(row["y"]),
                     "radius": float(row.get("radius") or 0),
                     "label": row.get("label", ""), "location_id": index}
            value[spectral] = float(row[spectral])
            value["notes"] = row.get("notes", "")
            rows.append((value, spectral))
    return rows


def _query_args(parser):
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--z", type=float)
    group.add_argument("--wave", "--wavelength", dest="wave", type=float)
    parser.add_argument("--x", type=float)
    parser.add_argument("--y", type=float)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cube", required=True)
    parser.add_argument("--input", help="CSV with x,y,z or x,y,wavelength")
    parser.add_argument("--output-dir", default="m101_visual_qa_trace")
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--radius", type=float, default=0.0)
    parser.add_argument("--summary")
    parser.add_argument("--h5", action="append", default=[])
    parser.add_argument("--h5-glob", action="append", default=[])
    parser.add_argument("--state")
    parser.add_argument("--band-cache", dest="band_cache")
    parser.add_argument("--alpha-fits", dest="alpha_fits")
    parser.add_argument("--alpha-summary", dest="alpha_summary")
    parser.add_argument("--amp-residual-models", dest="amp_residual_models")
    parser.add_argument("--fq-template", dest="fq_template")
    parser.add_argument("--external-blank-fibers", dest="external_blank_fibers")
    parser.add_argument("--qa-reject-file")
    parser.add_argument("--spectral-reject-file")
    parser.add_argument("--min-sky-fibers", type=int, default=DEFAULT_MIN_SKY_FIBERS)
    _query_args(parser)
    args = parser.parse_args()
    if args.radius < 0:
        parser.error("--radius must be non-negative")
    if not args.input and not args.interactive:
        if args.x is None or args.y is None or (args.z is None and args.wave is None):
            parser.error("single mode needs --x --y and exactly one of --z/--wave")
    if args.input and (args.x is not None or args.y is not None or args.z is not None or args.wave is not None):
        parser.error("do not mix --input with single-query coordinates")

    engine = ContributorEngine(args)
    atexit.register(engine.close)
    print("initialized %d shots in %.3fs; query positions are 1-based cube pixels" %
          (len(engine.shots), engine.initialization_seconds))
    if args.interactive:
        print("Enter: x y z [radius], or x y wave [radius]; Ctrl-D to exit")
        while True:
            try:
                line = input("x y z > ")
            except EOFError:
                break
            if not line.strip():
                continue
            values = line.replace(",", " ").split()
            if len(values) not in (3, 4):
                print("expected three or four numbers")
                continue
            x, y, spectral = map(float, values[:3])
            radius = float(values[3]) if len(values) == 4 else args.radius
            try:
                result = engine.query(x, y, z=spectral, radius=radius)
                _print_single(result)
            except Exception as exc:
                print("query failed: %s" % exc)
        return

    if args.input:
        locations = _load_locations(args.input)
        results = []
        for index, (location, spectral) in enumerate(locations):
            result = engine.query(location["x"], location["y"],
                                  z=location.get("z") if spectral == "z" else None,
                                  wavelength=location.get(spectral) if spectral != "z" else None,
                                  radius=location["radius"], label=location["label"])
            result["location_id"] = index
            result["notes"] = location["notes"]
            results.append(result)
        out = Path(args.output_dir).expanduser().resolve()
        out.mkdir(parents=True, exist_ok=True)
        detail_rows = []
        for result in results:
            for row in result["rows"]:
                detail = dict(row)
                detail.update({key: result[key] for key in
                               ("input_x", "input_y", "input_z", "wavelength", "radius", "label")})
                detail["location_id"] = result["location_id"]
                detail["notes"] = result["notes"]
                detail_rows.append(detail)
        _write_csv(out / "location_contributors.csv", detail_rows, [
            "location_id", "input_x", "input_y", "input_z", "wavelength", "radius", "label", "notes",
            "rank", "fractional_contribution", "effective_weight", "geometric_weight", "file", "exposure",
            "SPECID", "IFUSLOT", "IFUID", "AMP", "IFU_CODE", "physical_amp_id", "group_id",
            "fiber_rows", "fiber_row_count", "n_contributing_fibers", "masked_fraction", "masked",
            "geometric_fiber_count", "geometric_contributor", "effective_contributor", "qa_rejected"])
        amps = _amplifier_summary(results)
        _write_csv(out / "amplifier_summary.csv", amps, AMPLIFIER_FIELDS)
        (out / "provenance.json").write_text(json.dumps(engine.provenance(), indent=2, default=str) + "\n")
        print("wrote %s" % (out / "location_contributors.csv"))
        print("wrote %s" % (out / "amplifier_summary.csv"))
        print("rank dominant/contributing amplifier median/max fraction")
        for index, row in enumerate(amps[:20], 1):
            print("%4d %-22s %7d/%-7d %.4f/%.4f" % (
                index, row["physical_amp_id"], row["n_marked_locations_dominant"],
                row["n_marked_locations_contributed"],
                row["median_fractional_contribution"], row["max_fractional_contribution"]))
        return

    result = engine.query(args.x, args.y, z=args.z, wavelength=args.wave,
                          radius=args.radius)
    _print_single(result)
    print("provenance: %s" % (engine.summary_path or "CLI/model arguments"))


if __name__ == "__main__":
    main()
