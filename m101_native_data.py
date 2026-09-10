"""Validated native H5 loading and total-spectrum reconstruction for M101."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time

import numpy as np
import tables

import diagnose_m101_hierarchical as validated_m101
import build_m101_measurements as validated_measurements
from m101_hardware_exclusions import hardware_excluded

from m101_calibration_utils import (
    BANDS, collapse_many, collapse_error_many, sufficient_native_spectrum,
)


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


def _grouped_nanmean(values, inverse, size):
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    totals = np.bincount(inverse, weights=np.where(finite, values, 0.0),
                         minlength=size)
    counts = np.bincount(inverse, weights=finite.astype(float), minlength=size)
    return np.divide(totals, counts, out=np.full(size, np.nan), where=counts > 0)


def _exposure_coordinates(ifu, ra, dec, blank_valid):
    """Compute IFU centers with one grouping pass instead of one mask per IFU."""
    unique_ifus, inverse = np.unique(ifu, axis=0, return_inverse=True)
    ra_centers = _grouped_nanmean(ra, inverse, unique_ifus.shape[0])
    dec_centers = _grouped_nanmean(dec, inverse, unique_ifus.shape[0])
    supported = np.zeros(unique_ifus.shape[0], dtype=bool)
    np.logical_or.at(supported, inverse, blank_valid)
    center_mask = supported if np.any(supported) else np.ones(supported.shape, dtype=bool)
    ra0 = float(np.nanmedian(ra_centers[center_mask]))
    dec0 = float(np.nanmedian(dec_centers[center_mask]))
    return ra0, dec0


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


def _emit_progress(progress, message):
    if progress is not None:
        progress(message)


def load(h5_paths, blank_by_h5, on_filter_path, off_filter_path,
         minimum_finite_fraction=0.8, timings=None, include_band_errors=True,
         collapse_band_indices=None, include_native_errors=True, progress=None):
    load_started = time.perf_counter()
    filter_started = time.perf_counter()
    _emit_progress(progress, "loading ON/OFF filters and validated null responses")
    on_filter = validated_m101.read_filter(on_filter_path)
    off_filter = validated_m101.read_filter(off_filter_path)
    responses, null_provenance = _band_responses(on_filter, off_filter)
    collapse_band_indices = (tuple(range(len(responses))) if collapse_band_indices is None
                             else tuple(collapse_band_indices))
    if any(index < 0 or index >= len(responses) for index in collapse_band_indices):
        raise ValueError("collapse_band_indices contains an invalid band index")
    if timings is not None:
        timings["native_filter_setup_seconds"] = time.perf_counter() - filter_started
    output = []
    h5_shared_timings = []
    h5_total_timings = []
    exposure_timings = []
    paths = discover_h5(h5_paths, development=True)
    _emit_progress(progress, "accepted %d H5 files; beginning native reconstruction" % len(paths))
    for path_index, path in enumerate(paths, 1):
        h5_started = time.perf_counter()
        _emit_progress(progress, "H5 %d/%d: opening %s" % (path_index, len(paths), path.name))
        with tables.open_file(path, mode="r") as h5:
            shared_started = time.perf_counter()
            if not {"Info", "Fibers", "Survey"}.issubset(h5.root._v_children):
                raise ValueError("%s lacks Info, Fibers, or Survey" % path)
            info, fibers = h5.root.Info, h5.root.Fibers
            _emit_progress(progress, "H5 %d/%d: opened; reading Info/Survey metadata (%d rows)"
                           % (path_index, len(paths), info.nrows))
            metadata_started = time.perf_counter()
            groups, labels = validated_m101.build_groups(info)
            surveys = _survey_by_exposure(h5)
            ra = np.asarray(info.cols.ra[:], dtype=float)
            dec = np.asarray(info.cols.dec[:], dtype=float)
            metadata_seconds = time.perf_counter() - metadata_started
            physical_started = time.perf_counter()
            ifu_all, amp_all, j_all, q_all = _physical_arrays(info, groups, labels)
            physical_seconds = time.perf_counter() - physical_started
            blank = np.asarray(blank_by_h5[path.name], dtype=bool)
            if blank.shape != (info.nrows,):
                raise ValueError("blank mask does not match %s" % path)
            masks_started = time.perf_counter()
            date_cache = {}
            persistent_cache = {}
            date_bad_values, persistent_bad_values = [], []
            date_name = path.name.split("_")[0]
            for ifu, amplifier in zip(ifu_all, amp_all):
                date_key = (date_name, int(ifu[1]), str(amplifier))
                if date_key not in date_cache:
                    date_cache[date_key] = hardware_excluded(
                        date=date_name, specid=-1, ifuslot=ifu[1], ifuid=-1,
                        amp=amplifier, purpose="fit")
                date_bad_values.append(date_cache[date_key])
                persistent_key = (int(ifu[0]), int(ifu[1]), int(ifu[2]), str(amplifier))
                if persistent_key not in persistent_cache:
                    persistent_cache[persistent_key] = hardware_excluded(
                        date=None, specid=ifu[0], ifuslot=ifu[1], ifuid=ifu[2],
                        amp=amplifier, purpose="fit", h5=path)
                persistent_bad_values.append(persistent_cache[persistent_key])
            date_bad_all = np.asarray(date_bad_values, dtype=bool)
            persistent_bad_all = np.asarray(persistent_bad_values, dtype=bool)
            hardware_bad_all = date_bad_all | persistent_bad_all
            mask_seconds = time.perf_counter() - masks_started
            shared_seconds = time.perf_counter() - shared_started
            _emit_progress(progress, "H5 %d/%d: metadata/physical labels/masks done in %.3fs "
                           "(metadata=%.3fs, physical=%.3fs, masks=%.3fs)"
                           % (path_index, len(paths), shared_seconds,
                              metadata_seconds, physical_seconds, mask_seconds))
            if timings is not None:
                h5_shared_timings.append({"H5": path.name,
                                          "seconds": shared_seconds,
                                          "metadata_seconds": metadata_seconds,
                                          "physical_arrays_seconds": physical_seconds,
                                          "hardware_masks_seconds": mask_seconds})
            for exposure in (1, 2, 3):
                exposure_started = time.perf_counter()
                indices = np.flatnonzero(labels == exposure)
                survey = surveys[exposure]
                _emit_progress(progress, "H5 %d/%d %s exposure %d/3: reading spectrum/error/skyspectrum (%d fibers)"
                               % (path_index, len(paths), path.name, exposure, indices.size))
                read_started = time.perf_counter()
                spectrum_e = np.asarray(fibers.read_coordinates(indices, field="spectrum"), dtype=float)
                error_e = (np.asarray(fibers.read_coordinates(indices, field="error"), dtype=float)
                           if include_native_errors else np.zeros_like(spectrum_e))
                skyspectrum_e = np.asarray(fibers.read_coordinates(indices, field="skyspectrum"), dtype=float)
                total_e, error_e = construct_total_spectra(
                    spectrum_e, error_e, skyspectrum_e, survey["offset"])
                read_seconds = time.perf_counter() - read_started
                support_started = time.perf_counter()
                finite_native = sufficient_native_spectrum(total_e, minimum_finite_fraction)
                date_bad = date_bad_all[indices]
                blank_e = blank[indices]
                persistent_bad = persistent_bad_all[indices]
                hardware_bad = hardware_bad_all[indices]
                blank_valid = blank_e & ~hardware_bad & finite_native
                support_seconds = time.perf_counter() - support_started
                collapse_started = time.perf_counter()
                band_total = np.full((indices.size, len(responses)), np.nan, dtype=float)
                band_error = np.full((indices.size, len(responses)), np.nan, dtype=float)
                response_fraction = np.full((indices.size, len(responses)), np.nan, dtype=float)
                selected_responses = responses[list(collapse_band_indices)]
                values, fractions = collapse_many(total_e, selected_responses)
                band_total[:, list(collapse_band_indices)] = values
                response_fraction[:, list(collapse_band_indices)] = fractions
                if include_band_errors:
                    band_errors, _ = collapse_error_many(error_e, selected_responses)
                    band_error[:, list(collapse_band_indices)] = band_errors
                collapse_seconds = time.perf_counter() - collapse_started
                coordinates_started = time.perf_counter()
                raw_basis = validated_m101.raw_work_basis(survey)
                K = np.asarray([validated_m101.weighted_scalar(raw_basis, response)
                                for response in responses], dtype=float)
                # The IFU center and plane origin follow the existing M101
                # illumination convention: median valid IFU center per exposure.
                ra0, dec0 = _exposure_coordinates(
                    ifu_all[indices], ra[indices], dec[indices], blank_valid)
                x_arcmin = (ra[indices] - ra0) * np.cos(np.deg2rad(dec0)) * 60.0
                y_arcmin = (dec[indices] - dec0) * 60.0
                coordinates_seconds = time.perf_counter() - coordinates_started
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
                if timings is not None:
                    exposure_timings.append({
                        "H5": path.name, "exposure": int(exposure),
                        "rows": int(indices.size),
                        "supported_blank_fibers": int(np.sum(blank_valid)),
                        "read_reconstruct_seconds": read_seconds,
                        "support_mask_seconds": support_seconds,
                        "band_collapse_seconds": collapse_seconds,
                        "coordinates_support_seconds": coordinates_seconds,
                        "total_seconds": time.perf_counter() - exposure_started,
                    })
                _emit_progress(progress, "H5 %d/%d %s exposure %d/3: done in %.3fs "
                               "(read=%.3fs, collapse=%.3fs, coords=%.3fs; blank-valid=%d)"
                               % (path_index, len(paths), path.name, exposure,
                                  time.perf_counter() - exposure_started, read_seconds,
                                  collapse_seconds, coordinates_seconds, int(np.sum(blank_valid))))
        if timings is not None:
            h5_total_timings.append({"H5": path.name,
                                     "seconds": time.perf_counter() - h5_started})
        _emit_progress(progress, "H5 %d/%d %s: complete in %.3fs"
                       % (path_index, len(paths), path.name,
                          time.perf_counter() - h5_started))
    if timings is not None:
        timings["native_h5_shared_seconds"] = h5_shared_timings
        timings["native_h5_total_seconds"] = h5_total_timings
        timings["native_per_exposure"] = exposure_timings
        timings["native_load_total_seconds"] = time.perf_counter() - load_started
    return output, {"responses": responses, "null_provenance": null_provenance,
                    "wave": WAVE, "band_order": ALL_BANDS,
                    "hardware_exclusions": {
                        "legacy_date_slot_rows": int(sum(np.sum(item.date_mask_bad) for item in output)),
                        "persistent_physical_rows": int(sum(np.sum(item.persistent_hardware_bad) for item in output)),
                        "overlap_rows": int(sum(np.sum(item.date_mask_bad & item.persistent_hardware_bad) for item in output)),
                        "total_rows": int(sum(item.row_index.size for item in output)),
                    }}
