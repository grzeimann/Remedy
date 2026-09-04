#!/usr/bin/env python3
"""Rapid, standalone M101 physical-amplifier photometry diagnostic.

This is deliberately downstream of the validated PASS-1 cache.  It does not
fit any calibration parameter and never writes an H5 or a cube.  The central
measurements are intentionally visible here::

    X_amp = sum(cached exact external aperture flux)
    Y_amp = sum(response-corrected, residual-sky-subtracted VIRUS flux)
    G_amp = Y_amp / X_amp

The local residual-sky routine below is a small, import-safe reproduction of
``subtract_m101_residual_sky`` in make_mosaic_cube_revised.py.  Its radial,
nearest-pixel external blank criterion is the same frozen criterion; unlike
the cube builder, this diagnostic does not construct a cube grid merely to
make the blank-image interpolation grid.
"""

from argparse import ArgumentParser
import csv
import json
from pathlib import Path
import pickle
import time
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tables
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales

import diagnose_m101_hierarchical as hm


N_EXP = 3
N_AMP_FIBERS = 112
AMPS = ("LL", "LU", "RL", "RU")
BANDS = ("ON", "OFF")
M101_RA = hm.M101_CENTER_RA
M101_DEC = hm.M101_CENTER_DEC
SKY_RADIUS_ARCMIN = 6.0
SKY_IMAGE_LIMIT = 0.01
MIN_SKY_FIBERS = 20
MIN_AMP_FIBERS = 10
SOURCE_CUTS = (3, 5, 10)
DEFAULT_CUT = 5


def _finite(values):
    return np.asarray(values, dtype=float)[np.isfinite(values)]


def _fmt(value):
    if isinstance(value, (np.bool_, bool)):
        return int(value)
    if isinstance(value, np.generic):
        return value.item()
    return value


def _write_csv(path, rows, fields):
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _fmt(row.get(key, "")) for key in fields})


def _percentiles(values):
    values = _finite(values)
    if not values.size:
        return (np.nan, np.nan, np.nan, np.nan, np.nan)
    p16, p50, p84 = np.percentile(values, [16, 50, 84])
    return (float(np.mean(values)), float(np.median(values)),
            float(p16), float(p50), float(p84))


def _radius_arcmin(ra, dec):
    dra = (np.asarray(ra) - M101_RA) * np.cos(np.deg2rad(M101_DEC)) * 60.0
    ddec = (np.asarray(dec) - M101_DEC) * 60.0
    return np.hypot(dra, ddec)


def _load_image_meta(path, keep_data=False):
    with fits.open(path, memmap=True) as hdul:
        data = np.asarray(hdul[0].data, dtype=float)
        if data.ndim != 2:
            raise ValueError("%s must contain a 2-D primary image" % path)
        header = hdul[0].header.copy()
        if keep_data:
            data = data.copy()
    wcs = WCS(header).celestial
    scales = np.asarray(proj_plane_pixel_scales(wcs), dtype=float) * 3600.0
    scales = scales[np.isfinite(scales) & (scales > 0)]
    if not scales.size:
        raise ValueError("%s has no usable WCS pixel scale" % path)
    return {"path": Path(path).resolve(), "wcs": wcs,
            "pixel_scale_arcsec": float(np.median(scales)),
            "data": data if keep_data else None}


def _load_cache(path, h5_path, on_image, off_image, filters, fq):
    with Path(path).open("rb") as stream:
        cache = pickle.load(stream)
    if not isinstance(cache, dict) or "calibrations" not in cache:
        raise ValueError("PASS-1 cache is not a calibration dictionary")
    wanted = Path(h5_path).resolve()
    # PASS-1 caches made on the reduction host contain its absolute paths;
    # permit a local copy while requiring an unambiguous matching basename.
    cached_h5 = [Path(name).name for name in cache.get("h5_files", [])]
    if cached_h5 and wanted.name not in cached_h5:
        raise ValueError("H5 is not one of the PASS-1 cache inputs: %s" % wanted)
    calibration = None
    for candidate in cache["calibrations"]:
        name = str(candidate.get("h5", ""))
        if name == wanted.name or Path(name).name == wanted.name:
            calibration = candidate
            break
    if calibration is None:
        raise ValueError("PASS-1 cache has no calibration for %s" % wanted.name)
    cached_images = cache.get("images", {})
    for band, image in (("ON", on_image), ("OFF", off_image)):
        expected = cached_images.get(band)
        if expected is not None and Path(expected).name != image.name:
            raise ValueError("%s image does not match the PASS-1 cache" % band)
    cached_filters = cache.get("filters", {})
    for band in BANDS:
        if band in cached_filters and not np.allclose(
                np.asarray(cached_filters[band], dtype=float), filters[band],
                rtol=0, atol=0):
            raise ValueError("%s filter does not match the PASS-1 cache" % band)
    if "f_q" in cache and not np.allclose(np.asarray(cache["f_q"], dtype=float), fq,
                                          rtol=0, atol=0):
        raise ValueError("f(q) template does not match the PASS-1 cache")
    return cache, calibration


def _group_good_map(calibration):
    good = np.asarray(calibration.get("good_groups", []), dtype=bool)
    cached_groups = calibration.get("groups", [])
    result = {}
    if len(good) != len(cached_groups):
        raise ValueError("cache good_groups/groups length mismatch")
    for group, is_good in zip(cached_groups, good):
        key = (int(group["exposure"]), int(group["specid"]),
               int(group["ifuslot"]), int(group["ifuid"]), str(group["amp"]))
        result[key] = bool(is_good)
    return result


def _survey_rows(h5):
    if "Survey" not in h5.root._v_children:
        raise ValueError("H5 lacks Survey table")
    result = {}
    for row in h5.root.Survey:
        result[int(row["exp"])] = {name: row[name] for name in h5.root.Survey.colnames}
    if set(result) != set(range(1, N_EXP + 1)):
        raise ValueError("Survey must contain exactly exposures 1, 2, 3")
    return result


def _local_residual_sky(spectra, ra, dec, on_image, h5_name, exposure):
    """Apply the frozen production M101 residual-sky subtraction in memory.

    Source: ``make_mosaic_cube_revised.subtract_m101_residual_sky``.  The
    diagnostic has one exposure block in memory, so the production block
    selection is already implicit.  The coarse cube-grid blank lookup is
    replaced by the equivalent nearest pixel lookup in the original ON image.
    This lookup is used only to select sky fibers, never to calibrate X_amp.
    """
    result = np.asarray(spectra, dtype=float).copy()
    radius_blank = _radius_arcmin(ra, dec) > SKY_RADIUS_ARCMIN
    x, y = on_image["wcs"].world_to_pixel_values(ra, dec)
    finite_xy = np.isfinite(x) & np.isfinite(y)
    xi = np.zeros(x.shape, dtype=int)
    yi = np.zeros(y.shape, dtype=int)
    xi[finite_xy] = np.rint(x[finite_xy]).astype(int)
    yi[finite_xy] = np.rint(y[finite_xy]).astype(int)
    data = on_image["data"]
    in_image = (finite_xy & (xi >= 0) & (yi >= 0) &
                (xi < data.shape[1]) & (yi < data.shape[0]))
    blank = np.zeros(x.shape, dtype=bool)
    blank[in_image] = np.isfinite(data[yi[in_image], xi[in_image]]) & (
        data[yi[in_image], xi[in_image]] < SKY_IMAGE_LIMIT)
    sufficient = np.isfinite(result).sum(axis=1) >= int(np.ceil(.8 * result.shape[1]))
    selected = radius_blank & in_image & blank & sufficient
    if int(selected.sum()) < MIN_SKY_FIBERS:
        print("  residual sky e%d: %d candidates; skipped (minimum %d)" %
              (exposure, int(selected.sum()), MIN_SKY_FIBERS))
        return result
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        residual = np.asarray(hm.biweight(result[selected], axis=0), dtype=float)
    finite = np.isfinite(residual)
    if finite.sum() < int(np.ceil(.8 * result.shape[1])):
        print("  residual sky e%d: residual finite at %d/%d bins; skipped" %
              (exposure, int(finite.sum()), result.shape[1]))
        return result
    result[:, finite] -= residual[finite]
    print("  residual sky e%d: selected %d fibers, median=%+.5g" %
          (exposure, int(selected.sum()), float(np.nanmedian(residual[finite]))))
    return result


def _correct_exposure(h5_path, calibration, fq, filters, on_image, exposure):
    """Read and locally correct one exposure without constructing a cube."""
    started = time.perf_counter()
    with tables.open_file(h5_path, mode="r") as h5:
        info, fibers = h5.root.Info, h5.root.Fibers
        required = {"spectrum", "error", "skyspectrum"}
        if not required.issubset(fibers.colnames):
            raise ValueError("H5 Fibers lacks %s" % sorted(required - set(fibers.colnames)))
        groups, labels = hm.build_groups(info)
        ra_all = np.asarray(info.cols.ra[:], dtype=float)
        dec_all = np.asarray(info.cols.dec[:], dtype=float)
        ifuslot_all = np.asarray(info.cols.ifuslot[:])
        amp_all = np.asarray([hm.as_text(v) for v in info.cols.amp[:]])
        date_bad_all = hm.masked_rows(h5_path, ifuslot_all, amp_all)
        survey = _survey_rows(h5)[exposure]
        offset = float(survey["offset"])
        if not np.isfinite(offset) or offset == 0:
            raise ValueError("invalid Survey.offset in exposure %d" % exposure)
        exp_indices = np.flatnonzero(labels == exposure)
        ra, dec = ra_all[exp_indices], dec_all[exp_indices]
        source = np.asarray(fibers.read_coordinates(exp_indices, field="spectrum"), dtype=float)
        error = np.asarray(fibers.read_coordinates(exp_indices, field="error"), dtype=float)
        sky = np.asarray(fibers.read_coordinates(exp_indices, field="skyspectrum"), dtype=float)
        # Keep the production error basis explicit even though this diagnostic
        # does not use a variance-weighted estimator.
        error_work = error / offset
        del error, error_work
        working = source / offset
        corrected = np.full_like(working, np.nan, dtype=float)
        local_position = {int(row): i for i, row in enumerate(exp_indices)}
        response = calibration["response"]
        alpha = calibration["alpha"]
        k_work = hm.raw_work_basis(survey)
        for group in groups:
            if group["exposure"] != exposure:
                continue
            local = np.asarray([local_position[int(row)] for row in group["indices"]], dtype=int)
            key = (exposure, int(group["specid"]), int(group["ifuslot"]), int(group["ifuid"]))
            s_response = float(response[key])
            if not np.isfinite(s_response) or s_response == 0:
                raise ValueError("invalid s_response for %s" % (key,))
            q = np.arange(N_AMP_FIBERS) if group["amp"] in ("LL", "RU") \
                else np.arange(N_AMP_FIBERS - 1, -1, -1)
            additive = (k_work[None, :] * float(alpha[(exposure, group["amp"])]) *
                        fq[q, None])
            corrected[local] = (working[local] - additive + sky[local]) / s_response - sky[local]
        bad = date_bad_all[exp_indices]
        corrected[bad] = np.nan
        correction_done = time.perf_counter()
        residual_started = correction_done
        corrected = _local_residual_sky(corrected, ra, dec, on_image,
                                        Path(h5_path).name, exposure)
        residual_done = time.perf_counter()
        collapse_started = residual_done
        y = {band: hm.synthetic_mean(corrected, filters[band]) for band in BANDS}
        x_cache = {band: np.asarray(calibration["external_object"][(exposure, band)], dtype=float)
                   for band in BANDS}
        valid_cache = {band: np.asarray(calibration["external_valid"][(exposure, band)], dtype=bool)
                       for band in BANDS}
        for band in BANDS:
            if len(x_cache[band]) != len(exp_indices):
                raise ValueError("cached %s external array length does not match exposure" % band)
            if len(valid_cache[band]) != len(exp_indices):
                raise ValueError("cached %s external-valid array length does not match exposure" % band)
        return {"groups": groups, "exp_indices": exp_indices, "ra": ra, "dec": dec,
                "date_bad": bad, "y": y, "x": x_cache, "external_valid": valid_cache,
                "survey": survey, "corrected": corrected,
                "timings": {"read_correction": correction_done - started,
                            "residual_sky": residual_done - residual_started,
                            "synthetic_collapse": time.perf_counter() - collapse_started}}


def _make_records(calibration, exposure_data, exposure, image_meta):
    group_good = _group_good_map(calibration)
    records = []
    bundles = {}
    exp_indices = exposure_data["exp_indices"]
    local_position = {int(row): i for i, row in enumerate(exp_indices)}
    for group in exposure_data["groups"]:
        if group["exposure"] != exposure:
            continue
        local = np.asarray([local_position[int(row)] for row in group["indices"]], dtype=int)
        key = (exposure, int(group["specid"]), int(group["ifuslot"]),
               int(group["ifuid"]), str(group["amp"]))
        good = group_good.get(key, False)
        mean_ra = float(np.nanmean(exposure_data["ra"][local]))
        mean_dec = float(np.nanmean(exposure_data["dec"][local]))
        date_masked = bool(np.any(exposure_data["date_bad"][local]))
        base = {"exposure": exposure, "SPECID": key[1], "IFUSLOT": key[2],
                "IFUID": key[3], "AMP": key[4], "mean_RA": mean_ra,
                "mean_Dec": mean_dec, "N_total": N_AMP_FIBERS,
                "good_group": good, "date_masked": date_masked,
                "alpha": float(calibration["alpha"][(exposure, key[4])]),
                "s_response": float(calibration["response"][(exposure,) + key[1:4]]),
                "radius_arcmin": float(_radius_arcmin(mean_ra, mean_dec)),
                "mean_surface_brightness_ON": np.nan,
                "mean_surface_brightness_OFF": np.nan}
        for band in BANDS:
            x = exposure_data["x"][band][local]
            y = exposure_data["y"][band][local]
            valid = (good & ~date_masked & exposure_data["external_valid"][band][local] &
                     np.isfinite(x) & np.isfinite(y))
            xv, yv = x[valid], y[valid]
            mean_x, median_x, p16_x, p50_x, p84_x = _percentiles(xv)
            mean_y, median_y, p16_y, p50_y, p84_y = _percentiles(yv)
            X = float(np.sum(xv)) if xv.size else np.nan
            Y = float(np.sum(yv)) if yv.size else np.nan
            G = Y / X if xv.size >= MIN_AMP_FIBERS and np.isfinite(X) and X > 0 else np.nan
            row = dict(base)
            row.update({"band": band, "N_mutually_valid": int(xv.size),
                        "fraction_mutually_valid": float(xv.size / N_AMP_FIBERS),
                        "X_amp": X, "Y_amp": Y, "G_amp": G,
                        "mean_x": mean_x, "median_x": median_x, "p16_x": p16_x,
                        "p50_x": p50_x, "p84_x": p84_x,
                        "mean_y": mean_y, "median_y": median_y, "p16_y": p16_y,
                        "p50_y": p50_y, "p84_y": p84_y,
                        "N_x_positive": int(np.sum(xv > 0)),
                        "N_x_negative": int(np.sum(xv < 0)),
                        "X_per_N_valid": X / xv.size if xv.size else np.nan,
                        "Y_per_N_valid": Y / xv.size if xv.size else np.nan,
                        "good_group": good, "date_masked": date_masked})
            records.append(row)
            bundles[(exposure, key[1], key[2], key[3], key[4], band)] = {
                "x": x, "y": y, "valid": exposure_data["external_valid"][band][local],
                "good": good, "date_bad": date_masked}
            area = np.pi * (0.75 / image_meta[band]["pixel_scale_arcsec"]) ** 2
            row["mean_surface_brightness_%s" % band] = mean_x / area \
                if np.isfinite(mean_x) else np.nan
        # The two base rows share these coordinates and are indexed by key.
    return records, bundles


def _assign_source_support(records):
    refs = {}
    by_eb = {}
    for row in records:
        by_eb.setdefault((row["exposure"], row["band"]), []).append(row)
    for eb, rows in by_eb.items():
        usable = [r for r in rows if np.isfinite(r["X_amp"]) and
                  r["N_mutually_valid"] >= MIN_AMP_FIBERS]
        blank = [r for r in usable if r["radius_arcmin"] > SKY_RADIUS_ARCMIN and
                 np.isfinite(r["mean_surface_brightness_%s" % eb[1]]) and
                 r["mean_surface_brightness_%s" % eb[1]] < SKY_IMAGE_LIMIT]
        method = "r>6 arcmin and mean external SB<0.01"
        if len(blank) < 5:
            # A transparent external-only fallback prevents a sparse H5 from
            # silently receiving an arbitrary zero source significance.
            blank = sorted(usable, key=lambda r: r["mean_surface_brightness_%s" % eb[1]]
                           if np.isfinite(r["mean_surface_brightness_%s" % eb[1]]) else np.inf)
            blank = blank[:max(5, len(blank) // 4)]
            method = "lowest external mean SB quartile fallback"
        values = np.asarray([r["X_amp"] for r in blank], dtype=float)
        location = hm.robust_location(values)
        scale = hm.robust_scale(values)
        if not np.isfinite(scale) or scale <= 0:
            scale = float(np.std(values)) if values.size else np.nan
        scale = max(scale, 1e-12) if np.isfinite(scale) else np.nan
        refs[eb] = {"location": location, "scale": scale, "n_blank": len(blank),
                    "method": method}
        for row in rows:
            z = ((row["X_amp"] - location) / scale
                 if np.isfinite(row["X_amp"]) and np.isfinite(location) and np.isfinite(scale)
                 else np.nan)
            row["blank_X_location"] = location
            row["blank_X_scale"] = scale
            row["N_blank_amplifiers"] = len(blank)
            row["blank_reference_method"] = method
            row["X_source_significance"] = z
            for cut in SOURCE_CUTS:
                row["source_Z%d" % cut] = bool(np.isfinite(z) and z > cut)
    by_identity_band = {(r["exposure"], r["SPECID"], r["IFUSLOT"], r["IFUID"], r["AMP"], r["band"]): r
                        for r in records}
    for row in records:
        other = by_identity_band.get((row["exposure"], row["SPECID"], row["IFUSLOT"],
                                      row["IFUID"], row["AMP"],
                                      "OFF" if row["band"] == "ON" else "ON"))
        for cut in SOURCE_CUTS:
            row["source_joint_Z%d" % cut] = bool(row["source_Z%d" % cut] and other and
                                                   other["source_Z%d" % cut])
    return refs


def _estimators(records, bundles):
    by_eb = {(e, b): [r for r in records if r["exposure"] == e and r["band"] == b]
             for e in range(1, N_EXP + 1) for b in BANDS}
    result = {}
    for (exposure, band), rows in by_eb.items():
        for cut in SOURCE_CUTS:
            selected = [r for r in rows if r["source_Z%d" % cut] and np.isfinite(r["G_amp"])]
            direct = hm.biweight(np.asarray([r["G_amp"] for r in selected], dtype=float)) \
                if selected else np.nan
            by_ifu = {}
            for row in selected:
                by_ifu.setdefault((row["SPECID"], row["IFUSLOT"], row["IFUID"]), []).append(row["G_amp"])
            ifus = [hm.biweight(np.asarray(values, dtype=float)) for values in by_ifu.values()]
            ifus = np.asarray([v for v in ifus if np.isfinite(v)], dtype=float)
            ifu_value = hm.biweight(ifus) if ifus.size else np.nan
            result[(exposure, band, cut)] = {"direct": float(direct),
                                             "ifu": float(ifu_value),
                                             "n_amp": len(selected), "n_ifu": int(ifus.size)}
    return result


def _fiber_estimators(exposure_data, records, bundles, filters, area_by_band):
    result = {}
    for exposure in range(1, N_EXP + 1):
        for band in BANDS:
            xs, ys = [], []
            for row in records:
                if row["exposure"] != exposure or row["band"] != band:
                    continue
                key = (exposure, row["SPECID"], row["IFUSLOT"], row["IFUID"], row["AMP"], band)
                bundle = bundles[key]
                valid = (bundle["good"] & ~bundle["date_bad"] & bundle["valid"] &
                         np.isfinite(bundle["x"]) & np.isfinite(bundle["y"]) &
                         (bundle["x"] / area_by_band[band] > 0.05))
                xs.extend(bundle["x"][valid].tolist())
                ys.extend(bundle["y"][valid].tolist())
            x, y = np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)
            fit = hm.robust_zero_slope(x, y)
            ratio = y / x if x.size else np.asarray([], dtype=float)
            ratio = ratio[np.isfinite(ratio)]
            result[(exposure, band)] = {
                "G_fiber_Huber": float(fit["slope"]),
                "G_fiber_ratio_BW": float(hm.biweight(ratio)) if ratio.size else np.nan,
                "n_fiber_old_selection": int(x.size)}
    return result


def _add_references(records, estimators):
    for row in records:
        e, b = row["exposure"], row["band"]
        for cut in SOURCE_CUTS:
            row["G_reference_direct_amp_Z%d" % cut] = estimators[(e, b, cut)]["direct"]
            row["G_reference_amp_IFU_Z%d" % cut] = estimators[(e, b, cut)]["ifu"]
        ref = estimators[(e, b, DEFAULT_CUT)]["direct"]
        row["G_reference_direct_amp"] = ref
        row["G_reference_amp_IFU"] = estimators[(e, b, DEFAULT_CUT)]["ifu"]
        row["relative_G_amp"] = row["G_amp"] / ref if np.isfinite(row["G_amp"]) and np.isfinite(ref) and ref != 0 else np.nan
        row["log_relative_G_amp"] = (np.log(row["relative_G_amp"])
                                      if np.isfinite(row["relative_G_amp"]) and row["relative_G_amp"] > 0 else np.nan)
        row["D_amp"] = (row["Y_amp"] - ref * row["X_amp"]
                         if np.isfinite(row["Y_amp"]) and np.isfinite(row["X_amp"]) and np.isfinite(ref) else np.nan)


def _ifu_summary(records):
    rows = []
    keys = sorted(set((r["exposure"], r["SPECID"], r["IFUSLOT"], r["IFUID"]) for r in records))
    for key in keys:
        exposure, specid, slot, uid = key
        group = [r for r in records if (r["exposure"], r["SPECID"], r["IFUSLOT"], r["IFUID"]) == key]
        out = {"exposure": exposure, "SPECID": specid, "IFUSLOT": slot, "IFUID": uid}
        for band in BANDS:
            selected = [r for r in group if r["band"] == band and r["source_Z5"] and np.isfinite(r["G_amp"])]
            values = np.asarray([r["G_amp"] for r in selected], dtype=float)
            gifu = float(hm.biweight(values)) if values.size else np.nan
            out["N_source_amps_%s" % band] = int(values.size)
            out["G_IFU_%s" % band] = gifu
            out["robust_scatter_amp_G_%s" % band] = hm.robust_scale(values)
            deviations = np.abs(np.log(values / gifu)) if values.size and np.isfinite(gifu) and gifu > 0 else np.asarray([])
            out["max_amp_deviation_%s" % band] = float(np.max(deviations)) if deviations.size else np.nan
            discrepant = sorted(selected, key=lambda r: abs(r["G_amp"] / gifu - 1) if np.isfinite(gifu) and gifu else -1, reverse=True)
            out["most_discrepant_%s" % band] = ";".join(r["AMP"] for r in discrepant[:2])
            for r in group:
                if r["band"] == band:
                    r["G_IFU_%s" % band] = gifu
                    r["within_ifu_log_%s" % band] = (np.log(r["G_amp"] / gifu)
                        if np.isfinite(r["G_amp"]) and np.isfinite(gifu) and r["G_amp"] > 0 and gifu > 0 else np.nan)
        out["N_joint_source_amps"] = int(sum(r["source_joint_Z5"] and np.isfinite(r["G_amp"])
                                              for r in group if r["band"] == "ON"))
        rows.append(out)
    return rows


def _on_off_rows(records):
    indexed = {(r["exposure"], r["SPECID"], r["IFUSLOT"], r["IFUID"], r["AMP"], r["band"]): r for r in records}
    rows = []
    for key in sorted(set(k[:-1] for k in indexed)):
        e, specid, slot, uid, amp = key
        on, off = indexed.get(key + ("ON",)), indexed.get(key + ("OFF",))
        if on is None or off is None:
            continue
        row = {"exposure": e, "SPECID": specid, "IFUSLOT": slot, "IFUID": uid, "AMP": amp,
               "X_ON": on["X_amp"], "X_OFF": off["X_amp"], "G_ON": on["G_amp"], "G_OFF": off["G_amp"],
               "source_ON": on["source_Z5"], "source_OFF": off["source_Z5"],
               "source_joint": on["source_joint_Z5"],
               "relative_G_ON": on["relative_G_amp"], "relative_G_OFF": off["relative_G_amp"],
               "delta_log_ON_OFF": (on["log_relative_G_amp"] - off["log_relative_G_amp"]
                                    if np.isfinite(on["log_relative_G_amp"]) and np.isfinite(off["log_relative_G_amp"]) else np.nan),
               "D_ON": on["D_amp"], "D_OFF": off["D_amp"], "alpha": on["alpha"],
               "s_response": on["s_response"]}
        for cut in SOURCE_CUTS:
            row["source_ON_Z%d" % cut] = on["source_Z%d" % cut]
            row["source_OFF_Z%d" % cut] = off["source_Z%d" % cut]
            row["source_joint_Z%d" % cut] = on["source_joint_Z%d" % cut]
        rows.append(row)
    return rows


def _candidate_rows(records, ifu_rows):
    by_id = {}
    for row in records:
        by_id.setdefault((row["SPECID"], row["IFUSLOT"], row["IFUID"], row["AMP"]), []).append(row)
    candidates = []
    for key, values in by_id.items():
        logs = _finite([r["log_relative_G_amp"] for r in values])
        on = _finite([r["log_relative_G_amp"] for r in values if r["band"] == "ON"])
        off = _finite([r["log_relative_G_amp"] for r in values if r["band"] == "OFF"])
        deltas = _finite([r["delta_log_ON_OFF"] for r in _on_off_rows(values)])
        nexp = len(set(r["exposure"] for r in values if np.isfinite(r["log_relative_G_amp"])))
        within = []
        significance = []
        for r in values:
            within_value = r.get("within_ifu_log_%s" % r["band"], np.nan)
            if np.isfinite(within_value):
                within.append(within_value)
            if np.isfinite(r["X_source_significance"]):
                significance.append(r["X_source_significance"])
        median_abs = float(np.median(np.abs(logs))) if logs.size else np.nan
        max_abs = float(np.max(np.abs(logs))) if logs.size else np.nan
        within_abs = float(np.median(np.abs(within))) if within else np.nan
        persistence = nexp / 3.0
        score = (median_abs * (1.0 + 0.5 * persistence) +
                 0.25 * (within_abs if np.isfinite(within_abs) else 0) +
                 0.10 * (max_abs if np.isfinite(max_abs) else 0)) if np.isfinite(median_abs) else np.nan
        candidates.append({"H5": values[0].get("H5", ""), "SPECID": key[0], "IFUSLOT": key[1], "IFUID": key[2], "AMP": key[3],
                           "n_usable_exposure_band_measurements": int(logs.size),
                           "n_usable_exposures": nexp, "median_abs_log_relative_G": median_abs,
                           "maximum_abs_log_relative_G": max_abs, "median_log_relative_G_ON": hm.robust_location(on),
                           "median_log_relative_G_OFF": hm.robust_location(off),
                           "n_usable_exposures_ON": len(set(r["exposure"] for r in values if r["band"] == "ON" and np.isfinite(r["log_relative_G_amp"]))),
                           "n_usable_exposures_OFF": len(set(r["exposure"] for r in values if r["band"] == "OFF" and np.isfinite(r["log_relative_G_amp"]))),
                           "robust_scatter_log_relative_G_ON": hm.robust_scale(on),
                           "robust_scatter_log_relative_G_OFF": hm.robust_scale(off),
                           "robust_scatter_log_relative_G": hm.robust_scale(logs),
                           "median_abs_delta_log_ON_OFF": float(np.median(np.abs(deltas))) if deltas.size else np.nan,
                           "sign_repeats_ON": int(np.sum(on > 0)) if on.size else 0,
                           "sign_repeats_OFF": int(np.sum(off > 0)) if off.size else 0,
                           "sign_consistent_ON": bool(on.size and (np.all(on > 0) or np.all(on < 0))),
                           "sign_consistent_OFF": bool(off.size and (np.all(off > 0) or np.all(off < 0))),
                           "median_within_IFU_abs_log": within_abs,
                           "median_X_source_significance": float(np.median(significance)) if significance else np.nan,
                           "median_D_amp": hm.robust_location([r["D_amp"] for r in values]),
                           "robust_scatter_D_amp": hm.robust_scale([r["D_amp"] for r in values]),
                           "rank_score": score,
                           "ranking_definition": "median_abs_log*(1+0.5*n_exposures/3)+0.25*median_abs_within_IFU+0.10*max_abs_log"})
    return sorted(candidates, key=lambda r: -(r["rank_score"] if np.isfinite(r["rank_score"]) else -np.inf))


def _base_fields():
    return ["H5", "exposure", "production_state", "SPECID", "IFUSLOT", "IFUID", "AMP",
            "mean_RA", "mean_Dec", "N_total", "N_mutually_valid", "fraction_mutually_valid",
            "X_amp", "Y_amp", "G_amp", "mean_x", "median_x", "p16_x", "p50_x", "p84_x",
            "N_x_positive", "N_x_negative", "X_per_N_valid", "Y_per_N_valid", "mean_y",
            "median_y", "p16_y", "p50_y", "p84_y", "blank_X_location", "blank_X_scale",
            "X_source_significance", "source_Z3", "source_Z5", "source_Z10", "source_joint_Z3",
            "source_joint_Z5", "source_joint_Z10", "G_reference_direct_amp", "G_reference_amp_IFU",
            "relative_G_amp", "log_relative_G_amp", "D_amp", "alpha", "s_response",
            "good_group", "date_masked", "N_blank_amplifiers", "blank_reference_method",
            "mean_surface_brightness_ON", "mean_surface_brightness_OFF",
            "cache_global_g_ON", "cache_global_g_OFF", "band"]


def _plot_outputs(records, on_off, candidates, ifu_rows, refs, estimators, output_dir):
    colors = dict(zip(AMPS, ("tab:blue", "tab:orange", "tab:green", "tab:red")))
    markers = dict(zip(AMPS, ("o", "s", "^", "D")))
    analyzed_exposures = sorted(set(r["exposure"] for r in records))
    if not analyzed_exposures:
        return
    # X distribution.
    fig, axes = plt.subplots(len(analyzed_exposures), 2, figsize=(12, 4 * len(analyzed_exposures)), squeeze=False)
    for i, e in enumerate(analyzed_exposures):
        for j, band in enumerate(BANDS):
            ax = axes[i, j]
            rows = [r for r in records if r["exposure"] == e and r["band"] == band and np.isfinite(r["X_amp"])]
            for amp in AMPS:
                vals = [r["X_amp"] for r in rows if r["AMP"] == amp]
                if vals:
                    ax.hist(vals, bins=25, alpha=.35, color=colors[amp], label=amp)
            ref = refs[(e, band)]
            ax.axvline(ref["location"], color="k", lw=1, label="blank loc")
            for cut, ls in zip(SOURCE_CUTS, ("--", ":", "-")):
                ax.axvline(ref["location"] + cut * ref["scale"], color="k", ls=ls, lw=.8,
                           label="Z=%d" % cut)
            ax.set_title("exposure %d %s; blank N=%d" % (e, band, ref["n_blank"]))
            ax.set_xlabel("X_amp"); ax.set_ylabel("amplifiers"); ax.grid(alpha=.2)
            if e == 1 and j == 0: ax.legend(fontsize=7, ncol=2)
    fig.suptitle("M101 amplifier external source-support distribution")
    fig.tight_layout(rect=(0, 0, 1, .97)); fig.savefig(output_dir / "m101_amp_X_distribution.png", dpi=150); plt.close(fig)

    # X versus Y.
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, band in zip(axes, BANDS):
        rows = [r for r in records if r["band"] == band and np.isfinite(r["X_amp"]) and np.isfinite(r["Y_amp"])]
        for amp in AMPS:
            all_r = [r for r in rows if r["AMP"] == amp]
            src = [r for r in all_r if r["source_Z5"]]
            ax.scatter([r["X_amp"] for r in all_r], [r["Y_amp"] for r in all_r],
                       marker=markers[amp], color=colors[amp], alpha=.35, label=amp)
            ax.scatter([r["X_amp"] for r in src], [r["Y_amp"] for r in src],
                       marker=markers[amp], color=colors[amp], edgecolor="k", s=42)
        values = [r for r in rows if r["source_Z5"]]
        ref = hm.robust_location([r["G_amp"] for r in values])
        if np.isfinite(ref):
            xx = np.linspace(min(r["X_amp"] for r in rows), max(r["X_amp"] for r in rows), 100)
            ax.plot(xx, ref * xx, "k--", label="Y=G_ref X")
        dev = sorted([r for r in values if np.isfinite(r["log_relative_G_amp"])],
                     key=lambda r: abs(r["log_relative_G_amp"]), reverse=True)[:5]
        for r in dev:
            ax.annotate("%s-%s" % (r["IFUSLOT"], r["AMP"]), (r["X_amp"], r["Y_amp"]), fontsize=7)
        ax.set_title(band); ax.set_xlabel("X_amp"); ax.set_ylabel("Y_amp"); ax.grid(alpha=.2); ax.legend(fontsize=8)
    fig.suptitle("Amplifier-integrated external versus VIRUS photometry")
    fig.tight_layout(rect=(0, 0, 1, .95)); fig.savefig(output_dir / "m101_amp_X_vs_Y.png", dpi=160); plt.close(fig)

    # G versus support.
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, band in zip(axes, BANDS):
        rows = [r for r in records if r["band"] == band and np.isfinite(r["G_amp"]) and np.isfinite(r["relative_G_amp"])]
        for amp in AMPS:
            rr = [r for r in rows if r["AMP"] == amp]
            src = [r for r in rr if r["source_Z5"]]
            ax.scatter([r["X_source_significance"] for r in rr], [r["relative_G_amp"] for r in rr],
                       marker=markers[amp], color=colors[amp], alpha=.35, label=amp)
            ax.scatter([r["X_source_significance"] for r in src], [r["relative_G_amp"] for r in src],
                       marker=markers[amp], color=colors[amp], edgecolor="k", s=42)
        for cut, ls in zip(SOURCE_CUTS, ("--", ":", "-")):
            ax.axvline(cut, color="k", ls=ls, lw=.8)
        ax.axhline(1, color="k", lw=.8); ax.set_title(band); ax.set_xlabel("X source significance Z_X"); ax.set_ylabel("G_amp/G_ref")
        ax.grid(alpha=.2); ax.legend(fontsize=8)
    fig.suptitle("Amplifier ratio versus external source support")
    fig.tight_layout(rect=(0, 0, 1, .95)); fig.savefig(output_dir / "m101_amp_G_vs_X.png", dpi=160); plt.close(fig)

    # ON/OFF agreement.
    fig, ax = plt.subplots(figsize=(6, 6)); joint = [r for r in on_off if r["source_joint"] and np.isfinite(r["relative_G_ON"]) and np.isfinite(r["relative_G_OFF"])]
    for amp in AMPS:
        rr = [r for r in joint if r["AMP"] == amp]
        ax.scatter([r["relative_G_ON"] for r in rr], [r["relative_G_OFF"] for r in rr], marker=markers[amp], color=colors[amp], label=amp)
    if joint:
        lo = min(min(r["relative_G_ON"], r["relative_G_OFF"]) for r in joint); hi = max(max(r["relative_G_ON"], r["relative_G_OFF"]) for r in joint)
        ax.plot([lo, hi], [lo, hi], "k--", label="identity")
        for r in sorted(joint, key=lambda q: abs(q["delta_log_ON_OFF"]), reverse=True)[:6]:
            ax.annotate("%s-%s-e%d" % (r["IFUSLOT"], r["AMP"], r["exposure"]), (r["relative_G_ON"], r["relative_G_OFF"]), fontsize=7)
        scatter = hm.robust_scale([r["delta_log_ON_OFF"] for r in joint])
        corr = np.corrcoef([r["relative_G_ON"] for r in joint], [r["relative_G_OFF"] for r in joint])[0, 1] if len(joint) > 1 else np.nan
        ax.text(.03, .97, "robust scatter ln(ON/OFF)=%.4g\ncorrelation=%.4g" % (scatter, corr), transform=ax.transAxes, va="top")
    ax.set_xlabel("relative G ON"); ax.set_ylabel("relative G OFF"); ax.grid(alpha=.2); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(output_dir / "m101_amp_ON_vs_OFF.png", dpi=160); plt.close(fig)

    # Focal-plane map, one ON/OFF pair per exposure.
    fig, axes = plt.subplots(len(analyzed_exposures), 2, figsize=(12, 4.3 * len(analyzed_exposures)), squeeze=False)
    for i, e in enumerate(analyzed_exposures):
        for j, band in enumerate(BANDS):
            ax = axes[i, j]
            rows = [r for r in records if r["exposure"] == e and r["band"] == band and r["source_Z5"] and np.isfinite(r["log_relative_G_amp"])]
            sc = ax.scatter([r["mean_RA"] for r in rows], [r["mean_Dec"] for r in rows], c=[r["log_relative_G_amp"] for r in rows], cmap="coolwarm", vmin=-.2, vmax=.2, s=55)
            for r in sorted(rows, key=lambda q: abs(q["log_relative_G_amp"]), reverse=True)[:4]:
                ax.annotate("%s-%s" % (r["IFUSLOT"], r["AMP"]), (r["mean_RA"], r["mean_Dec"]), fontsize=7)
            ax.set_title("e%d %s" % (e, band)); ax.set_xlabel("RA"); ax.set_ylabel("Dec"); ax.grid(alpha=.2)
            fig.colorbar(sc, ax=ax, label="log(G/G_ref)")
    fig.suptitle("Source-bearing physical amplifier focal-plane residuals")
    fig.tight_layout(rect=(0, 0, 1, .97)); fig.savefig(output_dir / "m101_amp_focal_map.png", dpi=150); plt.close(fig)

    # Within-IFU population and orientation diagnostic.
    fig, ax = plt.subplots(figsize=(8, 5)); data = [[r.get("within_ifu_log_%s" % r["band"], np.nan) for r in records if r["AMP"] == amp and r.get("within_ifu_log_%s" % r["band"], np.nan) == r.get("within_ifu_log_%s" % r["band"], np.nan)] for amp in AMPS]
    ax.boxplot([values if values else [np.nan] for values in data], labels=AMPS, showfliers=False); ax.axhline(0, color="k", lw=.8); ax.set_ylabel("log(G_amp/G_IFU)"); ax.set_title("Amplifier residual relative to own physical IFU")
    ax.grid(axis="y", alpha=.2); fig.tight_layout(); fig.savefig(output_dir / "m101_amp_within_ifu.png", dpi=160); plt.close(fig)

    # Persistence by physical identity.
    fig, ax = plt.subplots(figsize=(12, 8)); top = candidates[:12]
    for i, candidate in enumerate(top):
        key = tuple(candidate[k] for k in ("SPECID", "IFUSLOT", "IFUID", "AMP"))
        for band, color in (("ON", "tab:blue"), ("OFF", "tab:orange")):
            rr = [r for r in records if (r["SPECID"], r["IFUSLOT"], r["IFUID"], r["AMP"]) == key and r["band"] == band and np.isfinite(r["log_relative_G_amp"])]
            ax.plot([r["exposure"] for r in rr], [r["log_relative_G_amp"] for r in rr], "o-", color=color, alpha=.8)
    ax.axhline(0, color="k", lw=.8); ax.set_xticks((1, 2, 3)); ax.set_xlabel("exposure"); ax.set_ylabel("log(G_amp/G_ref)"); ax.set_title("Persistence of strongest physical amplifier candidates"); ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(output_dir / "m101_amp_persistence.png", dpi=160); plt.close(fig)

    # Compact gallery: one panel per physical amplifier, ON/OFF points over exposures.
    fig, axes = plt.subplots(3, 4, figsize=(14, 9), squeeze=False)
    for ax, candidate in zip(axes.flat, candidates[:12]):
        key = tuple(candidate[k] for k in ("SPECID", "IFUSLOT", "IFUID", "AMP"))
        for band, color in (("ON", "tab:blue"), ("OFF", "tab:orange")):
            rr = [r for r in records if (r["SPECID"], r["IFUSLOT"], r["IFUID"], r["AMP"]) == key and r["band"] == band]
            ax.plot([r["exposure"] for r in rr], [r["relative_G_amp"] for r in rr], "o-", color=color, label=band)
        siblings = [r for r in records if r["SPECID"] == key[0] and r["IFUSLOT"] == key[1] and r["IFUID"] == key[2] and r["source_Z5"]]
        ax.axhline(1, color="k", lw=.7); ax.set_xticks((1, 2, 3)); ax.set_title("%s %s/%s/%s\nscore %.3g" % (key[3], key[0], key[1], key[2], candidate["rank_score"]), fontsize=8)
        ax.set_ylabel("G/Gref"); ax.grid(alpha=.2); ax.legend(fontsize=7)
        if siblings:
            ax.text(.03, .03, "siblings source=%d" % len(siblings), transform=ax.transAxes, fontsize=7)
    for ax in axes.flat[len(candidates[:12]):]: ax.set_visible(False)
    fig.suptitle("Worst physical-amplifier diagnostic gallery (blue ON, orange OFF)")
    fig.tight_layout(rect=(0, 0, 1, .95)); fig.savefig(output_dir / "m101_amp_worst_gallery.png", dpi=150); plt.close(fig)


def _summary(records, on_off, candidates, estimators, fiber, refs):
    lines = []
    lines.append("M101 amplifier-integrated photometry diagnostic")
    lines.append("Source cuts are fixed Z_X > 3, 5, 10; default comparison is Z_X > 5.")
    analyzed_exposures = sorted(set(r["exposure"] for r in records))
    for e in analyzed_exposures:
        counts = [sum(r["exposure"] == e and r["band"] == b and r["source_Z5"] for r in records) for b in BANDS]
        joint = sum(r["exposure"] == e and r["band"] == "ON" and r["source_joint_Z5"] for r in records)
        deltas = [r["delta_log_ON_OFF"] for r in on_off if r["exposure"] == e and r["source_joint"]]
        lines.append("e%d: source amps ON/OFF/joint=%d/%d/%d; joint ON/OFF robust scatter ln=%.5g" % (e, counts[0], counts[1], joint, hm.robust_scale(deltas)))
    scatters = {cut: [] for cut in SOURCE_CUTS}
    for e in analyzed_exposures:
        for cut in SOURCE_CUTS:
            values = [estimators[(e, b, cut)]["direct"] for b in BANDS if np.isfinite(estimators[(e, b, cut)]["direct"])]
            if len(values) == 2: scatters[cut].append(abs(np.log(values[0] / values[1])))
    stable = [(cut, hm.robust_location(v)) for cut, v in scatters.items() if v]
    lines.append("ON/OFF estimator log disagreement by cut: %s" % ", ".join("Z%d=%.5g" % (c, v) for c, v in stable))
    changes = {}
    for left, right in ((3, 5), (5, 10)):
        values = []
        for e in range(1, N_EXP + 1):
            for band in BANDS:
                a = estimators[(e, band, left)]["direct"]
                b = estimators[(e, band, right)]["direct"]
                if np.isfinite(a) and np.isfinite(b) and a > 0 and b > 0:
                    values.append(abs(np.log(b / a)))
        changes[(left, right)] = hm.robust_location(values)
    lines.append("Threshold sensitivity median |delta ln G_direct|: Z3->Z5=%.5g, Z5->Z10=%.5g; "
                 "inspect these before choosing where G becomes stable." %
                 (changes[(3, 5)], changes[(5, 10)]))
    lines.append("Orientation residual centers log(G/Gref): " + ", ".join("%s=%.5g" % (a, hm.robust_location([r["log_relative_G_amp"] for r in records if r["AMP"] == a])) for a in AMPS))
    for e in sorted(set(r["exposure"] for r in records)):
        for band in BANDS:
            values = [estimators[(e, band, cut)]["direct"] for cut in SOURCE_CUTS]
            lines.append("e%d %s estimators: fiber_Huber=%.6g direct(Z3,Z5,Z10)=%s IFU(Z3,Z5,Z10)=%s" %
                         (e, band, fiber[(e, band)]["G_fiber_Huber"],
                          "/".join("%.6g" % v if np.isfinite(v) else "nan" for v in values),
                          "/".join("%.6g" % estimators[(e, band, cut)]["ifu"] if np.isfinite(estimators[(e, band, cut)]["ifu"]) else "nan" for cut in SOURCE_CUTS)))
            lines.append("e%d %s fiber-Huber minus direct/IFU Z5: %.5g/%.5g" %
                         (e, band,
                          fiber[(e, band)]["G_fiber_Huber"] - estimators[(e, band, 5)]["direct"],
                          fiber[(e, band)]["G_fiber_Huber"] - estimators[(e, band, 5)]["ifu"]))
    repeated = [c for c in candidates if c["n_usable_exposures"] >= 3]
    lines.append("Persistent candidates with usable residuals in all 3 exposures: %d" % len(repeated))
    if candidates:
        lines.append("Top candidates: " + ", ".join("%s/%s/%s/%s" % tuple(c[k] for k in ("SPECID", "IFUSLOT", "IFUID", "AMP")) for c in candidates[:10]))
        top = candidates[0]
        lines.append("Top candidate persistence/contrast: usable exposures=%d, median log ON/OFF=%+.5g/%+.5g, "
                     "median |ON-OFF|=%+.5g, median within-IFU |log|=%.5g, median D=%.5g" %
                     (top["n_usable_exposures"], top["median_log_relative_G_ON"],
                      top["median_log_relative_G_OFF"], top["median_abs_delta_log_ON_OFF"],
                      top["median_within_IFU_abs_log"], top["median_D_amp"]))
    lines.append("Stability is assessed by the change in direct/IFU estimates across Z=3,5,10; inspect estimator CSV before choosing a threshold.")
    lines.append("Fiber baseline uses the old individual-fiber surface cut x/aperture_area > 0.05; it is comparison-only.")
    return lines


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--h5", required=True)
    parser.add_argument("--pass1-cache", required=True)
    parser.add_argument("--on-image", required=True)
    parser.add_argument("--off-image", required=True)
    parser.add_argument("--on-filter", required=True)
    parser.add_argument("--off-filter", required=True)
    parser.add_argument("--fq-template", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--exposure", type=int, choices=(1, 2, 3), action="append",
                        help="analyze only this exposure; repeat to select several")
    args = parser.parse_args()
    started = time.perf_counter()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    fq = hm.load_fq(args.fq_template)
    filters = {"ON": hm.read_filter(args.on_filter), "OFF": hm.read_filter(args.off_filter)}
    cache, calibration = _load_cache(args.pass1_cache, args.h5, Path(args.on_image).resolve(),
                                     Path(args.off_image).resolve(), filters, fq)
    print("cache load: %.3f s" % (time.perf_counter() - t0))
    on_image = _load_image_meta(args.on_image, keep_data=True)
    off_image = _load_image_meta(args.off_image, keep_data=False)
    image_meta = {"ON": on_image, "OFF": off_image}
    chosen = sorted(set(args.exposure or [1, 2, 3]))
    records, bundles = [], {}
    for exposure in chosen:
        t = time.perf_counter()
        data = _correct_exposure(args.h5, calibration, fq, filters, on_image, exposure)
        print("H5 read + local spectral correction e%d: %.3f s" % (exposure, data["timings"]["read_correction"]))
        print("residual sky e%d: %.3f s; synthetic collapse: %.3f s" %
              (exposure, data["timings"]["residual_sky"], data["timings"]["synthetic_collapse"]))
        t = time.perf_counter()
        new_records, new_bundles = _make_records(calibration, data, exposure, image_meta)
        records.extend(new_records); bundles.update(new_bundles)
        print("amplifier aggregation e%d: %.3f s" % (exposure, time.perf_counter() - t))
    for row in records:
        row["H5"] = Path(args.h5).name
        row["production_state"] = calibration.get("production_state", "")
        row["cache_global_g_ON"] = cache.get("global_g", {}).get("ON", np.nan)
        row["cache_global_g_OFF"] = cache.get("global_g", {}).get("OFF", np.nan)
    refs = _assign_source_support(records)
    estimators = _estimators(records, bundles)
    _add_references(records, estimators)
    area_by_band = {b: np.pi * (0.75 / image_meta[b]["pixel_scale_arcsec"]) ** 2 for b in BANDS}
    # The exposure bundles are already retained in ``bundles``; the first
    # argument is unused by the comparison helper and kept for clarity.
    fiber = _fiber_estimators({}, records, bundles, filters, area_by_band)
    ifu_rows = _ifu_summary(records)
    on_off = _on_off_rows(records)
    candidates = _candidate_rows(records, ifu_rows)
    fields = _base_fields()
    for cut in SOURCE_CUTS:
        fields += ["G_reference_direct_amp_Z%d" % cut, "G_reference_amp_IFU_Z%d" % cut]
    _write_csv(output_dir / "m101_amplifier_photometry.csv", records, fields)
    off_fields = ["H5", "exposure", "SPECID", "IFUSLOT", "IFUID", "AMP", "X_ON", "X_OFF", "G_ON", "G_OFF",
                  "source_ON", "source_OFF", "source_joint", "relative_G_ON", "relative_G_OFF", "delta_log_ON_OFF",
                  "D_ON", "D_OFF", "alpha", "s_response"] + ["source_%s_Z%d" % (side, cut) for cut in SOURCE_CUTS for side in ("ON", "OFF", "joint")]
    for row in on_off: row["H5"] = Path(args.h5).name
    _write_csv(output_dir / "m101_amplifier_on_off.csv", on_off, off_fields)
    ifu_fields = ["H5", "exposure", "SPECID", "IFUSLOT", "IFUID", "N_source_amps_ON", "N_source_amps_OFF", "N_joint_source_amps",
                  "G_IFU_ON", "G_IFU_OFF", "robust_scatter_amp_G_ON", "robust_scatter_amp_G_OFF", "max_amp_deviation_ON", "max_amp_deviation_OFF", "most_discrepant_ON", "most_discrepant_OFF"]
    for row in ifu_rows: row["H5"] = Path(args.h5).name
    _write_csv(output_dir / "m101_amplifier_ifu_summary.csv", ifu_rows, ifu_fields)
    candidate_fields = list(candidates[0].keys()) if candidates else ["SPECID", "IFUSLOT", "IFUID", "AMP", "rank_score"]
    _write_csv(output_dir / "m101_amplifier_candidates.csv", candidates, candidate_fields)
    estimator_rows = []
    for e in chosen:
        for b in BANDS:
            row = {"H5": Path(args.h5).name, "exposure": e, "band": b,
                   "G_fiber_Huber": fiber[(e, b)]["G_fiber_Huber"],
                   "G_fiber_ratio_BW": fiber[(e, b)]["G_fiber_ratio_BW"],
                   "n_fiber_old_selection": fiber[(e, b)]["n_fiber_old_selection"]}
            for cut in SOURCE_CUTS:
                row.update({"G_amp_direct_Z%d" % cut: estimators[(e, b, cut)]["direct"],
                            "G_amp_IFU_Z%d" % cut: estimators[(e, b, cut)]["ifu"],
                            "N_amp_Z%d" % cut: estimators[(e, b, cut)]["n_amp"],
                            "N_IFU_Z%d" % cut: estimators[(e, b, cut)]["n_ifu"]})
            for left, right in ((3, 5), (5, 10)):
                row["log_change_direct_Z%d_Z%d" % (left, right)] = (
                    np.log(estimators[(e, b, right)]["direct"] /
                           estimators[(e, b, left)]["direct"])
                    if np.isfinite(estimators[(e, b, left)]["direct"]) and
                    np.isfinite(estimators[(e, b, right)]["direct"]) and
                    estimators[(e, b, left)]["direct"] > 0 and
                    estimators[(e, b, right)]["direct"] > 0 else np.nan)
                row["log_change_IFU_Z%d_Z%d" % (left, right)] = (
                    np.log(estimators[(e, b, right)]["ifu"] /
                           estimators[(e, b, left)]["ifu"])
                    if np.isfinite(estimators[(e, b, left)]["ifu"]) and
                    np.isfinite(estimators[(e, b, right)]["ifu"]) and
                    estimators[(e, b, left)]["ifu"] > 0 and
                    estimators[(e, b, right)]["ifu"] > 0 else np.nan)
            row["fiber_Huber_minus_direct_Z5"] = (fiber[(e, b)]["G_fiber_Huber"] - estimators[(e, b, 5)]["direct"])
            row["fiber_Huber_minus_IFU_Z5"] = (fiber[(e, b)]["G_fiber_Huber"] - estimators[(e, b, 5)]["ifu"])
            for cut in SOURCE_CUTS:
                other = estimators[(e, "OFF" if b == "ON" else "ON", cut)]["direct"]
                direct = estimators[(e, b, cut)]["direct"]
                row["ON_OFF_log_difference_direct_Z%d" % cut] = (
                    np.log(direct / other) if b == "ON" and np.isfinite(direct) and np.isfinite(other) and direct > 0 and other > 0 else np.nan)
            estimator_rows.append(row)
    estimator_fields = list(estimator_rows[0].keys()) if estimator_rows else ["H5", "exposure", "band"]
    _write_csv(output_dir / "m101_amplifier_estimators.csv", estimator_rows, estimator_fields)
    _plot_outputs(records, on_off, candidates, ifu_rows, refs, estimators, output_dir)
    summary = _summary(records, on_off, candidates, estimators, fiber, refs)
    (output_dir / "m101_amplifier_summary.txt").write_text("\n".join(summary) + "\n")
    with (output_dir / "m101_amplifier_summary.json").open("w") as stream:
        json.dump({"summary": summary, "blank_references": {"%d_%s" % k: v for k, v in refs.items()}}, stream, indent=2)
    print("\n".join(summary))
    print("plots/output: %.3f s" % (time.perf_counter() - started))
    print("total: %.3f s" % (time.perf_counter() - started))


if __name__ == "__main__":
    main()
