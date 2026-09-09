"""Validated native H5 loading and total-spectrum reconstruction for M101."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tables

import diagnose_m101_hierarchical as validated_m101
import build_m101_measurements as validated_measurements
from m101_hardware_exclusions import hardware_excluded

from m101_calibration_utils import BANDS, collapse, collapse_error, sufficient_native_spectrum


WAVE = np.asarray(validated_m101.DEF_WAVE, dtype=float)
NULL_BANDS = ("HIGH1", "LOW1", "HIGH2", "LOW2", "HIGH3")
ALL_BANDS = NULL_BANDS + ("ON", "OFF")


@dataclass
class ExposureData:
    h5_name: str
    h5_path: str
    exposure: int
    key: tuple
    survey: dict
    row_index: np.ndarray
    ifu: np.ndarray
    amp: np.ndarray
    j: np.ndarray
    q: np.ndarray
    ra: np.ndarray
    dec: np.ndarray
    x_arcmin: np.ndarray
    y_arcmin: np.ndarray
    total: np.ndarray
    total_error: np.ndarray
    band_total: np.ndarray
    band_error: np.ndarray
    response_fraction: np.ndarray
    K: np.ndarray
    K_wave: np.ndarray
    blank_classified: np.ndarray
    date_mask_bad: np.ndarray
    blank_valid: np.ndarray
    persistent_hardware_bad: np.ndarray = None
    hardware_bad: np.ndarray = None


def discover_h5(paths, development=True):
    result = []
    seen = set()
    for raw in paths:
        path = Path(raw).expanduser().resolve()
        if path.name == validated_measurements.EXCLUDED_H5:
            continue
        if not path.is_file():
            raise ValueError("H5 file does not exist: %s" % path)
        if path.name in seen:
            raise ValueError("duplicate H5 basename: %s" % path.name)
        seen.add(path.name)
        result.append(path)
    result.sort(key=lambda value: value.name)
    if not result:
        raise ValueError("no accepted H5 files")
    if not development and len(result) != validated_measurements.EXPECTED_H5_COUNT:
        raise ValueError("expected 19 accepted H5 files")
    return result


def _survey_by_exposure(h5):
    result = {}
    for row in h5.root.Survey:
        exposure = int(row["exp"])
        if exposure in result:
            raise ValueError("duplicate Survey exposure %d" % exposure)
        result[exposure] = {name: row[name] for name in h5.root.Survey.colnames}
    if set(result) != {1, 2, 3}:
        raise ValueError("Survey must contain exposures 1, 2, 3")
    return result


def _physical_arrays(info, groups, labels):
    ifuslot = np.asarray(info.cols.ifuslot[:])
    specid = np.asarray(info.cols.specid[:], dtype=int)
    ifuid = np.asarray(info.cols.ifuid[:], dtype=int)
    amp = np.asarray([validated_m101.as_text(value) for value in info.cols.amp[:]], dtype=object)
    j = np.full(info.nrows, -1, dtype=np.int16)
    q = np.full(info.nrows, -1, dtype=np.int16)
    for group in groups:
        native_j = np.arange(validated_m101.N_FIBER_AMP, dtype=np.int16)
        native_q = (native_j if group["amp"] in ("LL", "RU")
                    else validated_m101.N_FIBER_AMP - 1 - native_j)
        j[group["indices"]] = native_j
        q[group["indices"]] = native_q
    if np.any(j < 0) or np.any(q < 0):
        raise ValueError("physical j/q assignment is incomplete")
    ifu = np.column_stack((specid, ifuslot.astype(int), ifuid))
    return ifu, amp, j, q


def _band_responses(on_filter, off_filter):
    null_responses, null_provenance = validated_measurements._validate_sky_null_responses()
    responses = [np.asarray(response, dtype=float) for response in null_responses]
    responses.extend((np.asarray(on_filter, dtype=float), np.asarray(off_filter, dtype=float)))
    return np.asarray(responses), null_provenance


def construct_total_spectra(spectrum, error, skyspectrum, offset):
    """Undo the historical QR subtraction exactly once.

    ``skyspectrum`` is used only in this expression and is never returned as a
    sky-model input to the fitter.
    """
    spectrum = np.asarray(spectrum, dtype=float)
    error = np.asarray(error, dtype=float)
    skyspectrum = np.asarray(skyspectrum, dtype=float)
    offset = float(offset)
    if not np.isfinite(offset) or offset == 0.0:
        raise ValueError("Survey.offset must be finite and nonzero")
    if spectrum.shape != error.shape or spectrum.shape != skyspectrum.shape:
        raise ValueError("spectrum/error/skyspectrum shapes differ")
    total = spectrum / offset + skyspectrum
    total_error = error / abs(offset)
    return total, total_error


def load(h5_paths, blank_by_h5, on_filter_path, off_filter_path,
         minimum_finite_fraction=0.8):
    on_filter = validated_m101.read_filter(on_filter_path)
    off_filter = validated_m101.read_filter(off_filter_path)
    responses, null_provenance = _band_responses(on_filter, off_filter)
    output = []
    for path in discover_h5(h5_paths, development=True):
        with tables.open_file(path, mode="r") as h5:
            if not {"Info", "Fibers", "Survey"}.issubset(h5.root._v_children):
                raise ValueError("%s lacks Info, Fibers, or Survey" % path)
            info, fibers = h5.root.Info, h5.root.Fibers
            groups, labels = validated_m101.build_groups(info)
            surveys = _survey_by_exposure(h5)
            ra = np.asarray(info.cols.ra[:], dtype=float)
            dec = np.asarray(info.cols.dec[:], dtype=float)
            ifu_all, amp_all, j_all, q_all = _physical_arrays(info, groups, labels)
            blank = np.asarray(blank_by_h5[path.name], dtype=bool)
            if blank.shape != (info.nrows,):
                raise ValueError("blank mask does not match %s" % path)
            date_bad_all = np.asarray([
                hardware_excluded(
                    date=path.name.split("_")[0], specid=-1,
                    ifuslot=ifu[1], ifuid=-1, amp=amplifier, purpose="fit")
                for ifu, amplifier in zip(ifu_all, amp_all)
            ], dtype=bool)
            persistent_bad_all = np.asarray([
                hardware_excluded(
                    date=None, specid=ifu[0], ifuslot=ifu[1], ifuid=ifu[2],
                    amp=amplifier, purpose="fit")
                for ifu, amplifier in zip(ifu_all, amp_all)
            ], dtype=bool)
            hardware_bad_all = date_bad_all | persistent_bad_all
            for exposure in (1, 2, 3):
                indices = np.flatnonzero(labels == exposure)
                survey = surveys[exposure]
                total_e, error_e = construct_total_spectra(
                    np.asarray(fibers.read_coordinates(indices, field="spectrum"), dtype=float),
                    np.asarray(fibers.read_coordinates(indices, field="error"), dtype=float),
                    np.asarray(fibers.read_coordinates(indices, field="skyspectrum"), dtype=float),
                    survey["offset"])
                finite_native = sufficient_native_spectrum(total_e, minimum_finite_fraction)
                date_bad = date_bad_all[indices]
                blank_e = blank[indices]
                persistent_bad = persistent_bad_all[indices]
                hardware_bad = hardware_bad_all[indices]
                blank_valid = blank_e & ~hardware_bad & finite_native
                band_values = []
                band_errors = []
                fractions = []
                for response in responses:
                    value, fraction = collapse(total_e, response)
                    band_error, _ = collapse_error(error_e, response)
                    band_values.append(value)
                    band_errors.append(band_error)
                    fractions.append(fraction)
                band_total = np.column_stack(band_values)
                band_error = np.column_stack(band_errors)
                response_fraction = np.column_stack(fractions)
                raw_basis = validated_m101.raw_work_basis(survey)
                K = np.asarray([validated_m101.weighted_scalar(raw_basis, response)
                                for response in responses], dtype=float)
                # The IFU center and plane origin follow the existing M101
                # illumination convention: median valid IFU center per exposure.
                centers = {}
                for ifu_key in sorted(set(map(tuple, ifu_all[indices]))):
                    selected = indices[np.all(ifu_all[indices] == np.asarray(ifu_key), axis=1)]
                    centers[ifu_key] = (float(np.nanmean(ra[selected])),
                                        float(np.nanmean(dec[selected])))
                supported_centers = [centers[key] for key in centers
                                     if np.any(blank_valid[np.all(ifu_all[indices] == np.asarray(key), axis=1)])]
                if not supported_centers:
                    supported_centers = list(centers.values())
                ra0 = float(np.nanmedian([item[0] for item in supported_centers]))
                dec0 = float(np.nanmedian([item[1] for item in supported_centers]))
                x_arcmin = (ra[indices] - ra0) * np.cos(np.deg2rad(dec0)) * 60.0
                y_arcmin = (dec[indices] - dec0) * 60.0
                output.append(ExposureData(
                    h5_name=path.name, h5_path=str(path), exposure=exposure,
                    key=(path.name, exposure), survey=survey,
                    row_index=indices.astype(np.int64), ifu=ifu_all[indices],
                    amp=amp_all[indices], j=j_all[indices], q=q_all[indices],
                    ra=ra[indices], dec=dec[indices], x_arcmin=x_arcmin,
                    y_arcmin=y_arcmin, total=total_e.astype(np.float32),
                    total_error=error_e.astype(np.float32), band_total=band_total,
                    band_error=band_error, response_fraction=response_fraction,
                    K=K, K_wave=raw_basis, blank_classified=blank_e,
                    date_mask_bad=date_bad, blank_valid=blank_valid,
                    persistent_hardware_bad=persistent_bad,
                    hardware_bad=hardware_bad))
    return output, {"responses": responses, "null_provenance": null_provenance,
                    "wave": WAVE, "band_order": ALL_BANDS,
                    "hardware_exclusions": {
                        "legacy_date_slot_rows": int(sum(np.sum(item.date_mask_bad) for item in output)),
                        "persistent_physical_rows": int(sum(np.sum(item.persistent_hardware_bad) for item in output)),
                        "overlap_rows": int(sum(np.sum(item.date_mask_bad & item.persistent_hardware_bad) for item in output)),
                        "total_rows": int(sum(item.row_index.size for item in output)),
                    }}
