#!/usr/bin/env python3
"""Diagnostic comparison of the M101 [O III] 5007/4959 doublet.

This script deliberately does not modify either input cube or construct a
corrected product.  FITS data are memory mapped and consumed one wavelength
plane at a time.  Small line-profile intermediates are kept in temporary
on-disk memmaps so the full cubes are never loaded into RAM.
"""

from argparse import ArgumentParser
import csv
from pathlib import Path
import tempfile

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS


OIII_4959_WAVELENGTH = 4958.92
OIII_5007_WAVELENGTH = 5006.84
OIII_5007_4959_RATIO = 2.98
SPEED_OF_LIGHT_KMS = 299792.458
LINE_HALF_WIDTH_ANGSTROM = 9.0
CONTINUUM_OFFSETS_ANGSTROM = ((-30.0, -15.0), (15.0, 30.0))
SUPPORT_THRESHOLD = 0.80
SNR_THRESHOLDS = (5.0, 10.0)
PROFILE_RESIDUAL_THRESHOLD = 0.05
STRONG_RATIO_RESIDUAL_THRESHOLD = 0.20
STRONG_PROFILE_RESIDUAL_THRESHOLD = 0.10
CHUNK_ROWS = 128


def wavelengths(header, nplane):
    """Return the spectral WCS coordinate for every primary-HDU plane."""
    crval = float(header.get("CRVAL3", 0.0))
    cdelt = float(header.get("CDELT3", 1.0))
    crpix = float(header.get("CRPIX3", 1.0))
    pixels = np.arange(nplane, dtype=float) + 1.0
    return crval + (pixels - crpix) * cdelt


def usable_mask(values):
    """SCI support convention: finite and not exactly zero."""
    values = np.asarray(values)
    return np.isfinite(values) & (values != 0.0)


def robust_scatter(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not values.size:
        return np.nan
    median = np.median(values)
    return float(1.4826 * np.median(np.abs(values - median)))


def integrated_flux(wave, profile):
    """Integrate a sampled continuum-subtracted profile over wavelength."""
    wave = np.asarray(wave, dtype=float)
    profile = np.asarray(profile, dtype=float)
    finite = np.isfinite(profile) & np.isfinite(wave)
    if not np.any(finite):
        return np.nan
    # The production grid is uniformly sampled, but gradients preserve the
    # same transparent flux-density-to-integrated-flux operation if it is not.
    weights = np.gradient(wave)
    return float(np.sum(profile[finite] * weights[finite]))


def interp_profile_at_velocity(profile, velocity, target_velocity):
    """Linearly interpolate a 1-D measured profile at one velocity."""
    profile = np.asarray(profile, dtype=float)
    velocity = np.asarray(velocity, dtype=float)
    if target_velocity < velocity[0] or target_velocity > velocity[-1]:
        return np.nan
    right = int(np.searchsorted(velocity, target_velocity, side="left"))
    if right == 0:
        return float(profile[0]) if np.isfinite(profile[0]) else np.nan
    if right >= len(velocity):
        return float(profile[-1]) if np.isfinite(profile[-1]) else np.nan
    if velocity[right] == target_velocity:
        return float(profile[right]) if np.isfinite(profile[right]) else np.nan
    left = right - 1
    if not np.isfinite(profile[left]) or not np.isfinite(profile[right]):
        return np.nan
    weight = ((target_velocity - velocity[left]) /
              (velocity[right] - velocity[left]))
    return float(profile[left] + weight * (profile[right] - profile[left]))


def normalized_profile_residual(profile5007, profile4959, peak4959):
    """Compare profiles using 2.98 times the 4959 peak as the scale."""
    denominator = OIII_5007_4959_RATIO * np.asarray(peak4959, dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        return ((np.asarray(profile5007, dtype=float) -
                 OIII_5007_4959_RATIO * np.asarray(profile4959, dtype=float)) /
                denominator)


def select_4959_sample(snr4959, support4959, support5007, threshold):
    """Select using 4959 S/N; 5007 contributes support validity only."""
    # +inf is a valid result when the empirical continuum MAD is exactly
    # zero; NaN remains the invalid/no-information state.
    return (support4959 & support5007 & ~np.isnan(snr4959) &
            (snr4959 >= threshold))


def spatial_header(header):
    output = header.copy()
    for key in ("NAXIS3", "CRPIX3", "CRVAL3", "CDELT3", "CTYPE3",
                "CUNIT3", "PC1_3", "PC2_3", "PC3_1", "PC3_2", "PC3_3",
                "CD1_3", "CD2_3", "CD3_1", "CD3_2", "CD3_3"):
        output.remove(key, ignore_missing=True)
    output["NAXIS"] = 2
    output["WCSAXES"] = 2
    return output


def write_map(path, image, header):
    fits.PrimaryHDU(np.asarray(image, dtype=np.float32), header=header).writeto(
        path, overwrite=True)


def _plane_indices(wave, center, half_width):
    return np.flatnonzero(np.abs(wave - center) <= half_width).astype(int)


def _continuum_indices(wave, center, velocity_kms):
    """Return blue/red sideband indices around a velocity-shifted line."""
    observed = center * (1.0 + velocity_kms / SPEED_OF_LIGHT_KMS)
    windows = []
    for lo, hi in CONTINUUM_OFFSETS_ANGSTROM:
        windows.append(np.flatnonzero(
            (wave >= observed + lo) & (wave <= observed + hi)).astype(int))
    return windows


def estimate_4959_velocity(data, wave, rest_wave):
    """Estimate a global velocity from a robust bright-region 4959 spectrum."""
    indices = _plane_indices(wave, rest_wave, 15.0)
    spectrum = np.full(indices.size, np.nan, dtype=float)
    for k, index in enumerate(indices):
        plane = np.asarray(data[index])
        valid = usable_mask(plane) & (plane > 0.0)
        values = plane[valid].astype(float, copy=False)
        if values.size:
            # The high percentile suppresses the large unsupported/continuum
            # area while retaining a robust bright emission-line statistic.
            cutoff = np.percentile(values, 90.0)
            spectrum[k] = np.median(values[values >= cutoff])
    finite = np.isfinite(spectrum)
    if np.count_nonzero(finite) < 3:
        return np.nan, np.nan, spectrum, wave[indices]
    local_wave = wave[indices]
    outer = finite & ((local_wave <= rest_wave - 10.0) |
                      (local_wave >= rest_wave + 10.0))
    background = np.median(spectrum[outer]) if np.any(outer) else np.median(spectrum[finite])
    excess = np.where(finite, spectrum - background, np.nan)
    positive = np.isfinite(excess) & (excess > 0.0)
    if not np.any(positive):
        return np.nan, np.nan, spectrum, local_wave
    center = float(np.sum(local_wave[positive] * excess[positive]) /
                   np.sum(excess[positive]))
    velocity = SPEED_OF_LIGHT_KMS * (center / rest_wave - 1.0)
    return velocity, center, spectrum, local_wave


def _chunked_continuum_statistics(cont_values, chunk_rows):
    """Compute per-spaxel median and 1.4826*MAD from a temp memmap."""
    n_cont, ny, nx = cont_values.shape
    median = np.full((ny, nx), np.nan, dtype=np.float32)
    rms = np.full((ny, nx), np.nan, dtype=np.float32)
    for y0 in range(0, ny, chunk_rows):
        y1 = min(ny, y0 + chunk_rows)
        chunk = np.asarray(cont_values[:, y0:y1, :])
        with np.errstate(all="ignore"):
            med = np.nanmedian(chunk, axis=0)
            mad = np.nanmedian(np.abs(chunk - med[None, :, :]), axis=0)
        median[y0:y1] = med.astype(np.float32)
        rms[y0:y1] = (1.4826 * mad).astype(np.float32)
    return median, rms


def _support_mask(line_count, line_size, blue_count, blue_size,
                  red_count, red_size, threshold):
    return ((line_count >= threshold * line_size) &
            (blue_count >= threshold * blue_size) &
            (red_count >= threshold * red_size))


def analyze_cube(path, temporary_dir, support_threshold):
    """Analyze one cube and return maps plus compact profile diagnostics."""
    with fits.open(path, memmap=True) as hdul:
        data = hdul[0].data
        header = hdul[0].header.copy()
        if data is None or data.ndim != 3:
            raise ValueError("%s does not contain a 3-D primary cube" % path)
        nplane, ny, nx = data.shape
        wave = wavelengths(header, nplane)
        velocity, observed_4959, global_spectrum, global_wave = (
            estimate_4959_velocity(data, wave, OIII_4959_WAVELENGTH))
        if not np.isfinite(velocity):
            raise ValueError("could not estimate a global 4959 velocity for %s" % path)

        observed_5007 = OIII_5007_WAVELENGTH * (
            1.0 + velocity / SPEED_OF_LIGHT_KMS)
        vhalf = SPEED_OF_LIGHT_KMS * LINE_HALF_WIDTH_ANGSTROM / OIII_4959_WAVELENGTH
        line4959_indices = np.flatnonzero(
            np.abs(SPEED_OF_LIGHT_KMS * (wave / OIII_4959_WAVELENGTH - 1.0) -
                   velocity) <= vhalf).astype(int)
        line5007_indices = np.flatnonzero(
            np.abs(SPEED_OF_LIGHT_KMS * (wave / OIII_5007_WAVELENGTH - 1.0) -
                   velocity) <= vhalf).astype(int)
        continuum_windows = _continuum_indices(
            wave, OIII_4959_WAVELENGTH, velocity)
        continuum_windows_5007 = _continuum_indices(
            wave, OIII_5007_WAVELENGTH, velocity)
        if (not line4959_indices.size or not line5007_indices.size or
                len(continuum_windows[0]) == 0 or len(continuum_windows[1]) == 0):
            raise ValueError("insufficient O III line/continuum wavelength coverage in %s" % path)

        continuum_indices_by_line = {
            "4959": np.unique(np.concatenate(continuum_windows)).astype(int),
            "5007": np.unique(np.concatenate(continuum_windows_5007)).astype(int),
        }
        cont_values_by_line = {}
        for label, indices in continuum_indices_by_line.items():
            cont_path = Path(temporary_dir) / (
                Path(path).stem + "_continuum_" + label + ".dat")
            cont_values_by_line[label] = np.memmap(
                cont_path, mode="w+", dtype="float32",
                shape=(indices.size, ny, nx))
        blue4959_count = np.zeros((ny, nx), dtype=np.uint16)
        red4959_count = np.zeros((ny, nx), dtype=np.uint16)
        blue5007_count = np.zeros((ny, nx), dtype=np.uint16)
        red5007_count = np.zeros((ny, nx), dtype=np.uint16)
        blue4959 = set(continuum_windows[0].tolist())
        red4959 = set(continuum_windows[1].tolist())
        blue5007 = set(continuum_windows_5007[0].tolist())
        red5007 = set(continuum_windows_5007[1].tolist())

        continuum_by_line = {}
        continuum_rms_by_line = {}
        for label, indices in continuum_indices_by_line.items():
            cont_values = cont_values_by_line[label]
            cont_position = {int(index): k for k, index in enumerate(indices)}
            for index in indices:
                plane = np.asarray(data[index])
                valid = usable_mask(plane)
                cont_values[cont_position[int(index)]] = np.where(
                    valid, plane, np.nan).astype(np.float32)
                if label == "4959":
                    if int(index) in blue4959:
                        blue4959_count += valid
                    if int(index) in red4959:
                        red4959_count += valid
                else:
                    if int(index) in blue5007:
                        blue5007_count += valid
                    if int(index) in red5007:
                        red5007_count += valid
            cont_values.flush()
            continuum_by_line[label], continuum_rms_by_line[label] = (
                _chunked_continuum_statistics(cont_values, CHUNK_ROWS))

        profile_paths = {}
        profiles = {}
        line_sums = {}
        line_counts = {}
        line_specs = (("4959", line4959_indices), ("5007", line5007_indices))
        for label, indices in line_specs:
            profile_path = Path(temporary_dir) / (
                Path(path).stem + "_profile_" + label + ".dat")
            profile = np.memmap(
                profile_path, mode="w+", dtype="float32",
                shape=(indices.size, ny, nx))
            line_sum = np.zeros((ny, nx), dtype=np.float64)
            line_count = np.zeros((ny, nx), dtype=np.uint16)
            weights = np.gradient(wave[indices])
            continuum = continuum_by_line[label]
            for k, index in enumerate(indices):
                plane = np.asarray(data[index])
                valid = usable_mask(plane)
                line_count += valid
                values = np.where(valid, plane - continuum, np.nan).astype(np.float32)
                profile[k] = values
                finite = np.isfinite(values)
                line_sum += np.where(finite, values * weights[k], 0.0)
            profile.flush()
            profile_paths[label] = profile_path
            profiles[label] = profile
            line_sums[label] = line_sum
            line_counts[label] = line_count

        support4959 = _support_mask(
            line_counts["4959"], line4959_indices.size,
            blue4959_count, len(continuum_windows[0]),
            red4959_count, len(continuum_windows[1]), support_threshold)
        support5007 = _support_mask(
            line_counts["5007"], line5007_indices.size,
            blue5007_count, len(continuum_windows_5007[0]),
            red5007_count, len(continuum_windows_5007[1]), support_threshold)
        f4959 = line_sums["4959"].astype(np.float32)
        f5007 = line_sums["5007"].astype(np.float32)
        f4959[~support4959] = np.nan
        f5007[~support5007] = np.nan
        line_step = float(np.median(np.gradient(wave[line4959_indices])))
        noise4959 = (continuum_rms_by_line["4959"] *
                     np.sqrt(line_counts["4959"]) * line_step)
        snr4959 = np.full((ny, nx), np.nan, dtype=np.float32)
        finite_noise = np.isfinite(noise4959) & (noise4959 > 0.0)
        np.divide(f4959, noise4959, out=snr4959, where=finite_noise)
        zero_noise_positive_line = ((noise4959 == 0.0) &
                                    np.isfinite(f4959) & (f4959 > 0.0))
        snr4959[zero_noise_positive_line] = np.inf
        snr4959[~support4959] = np.nan
        ratio_valid = (support4959 & support5007 & np.isfinite(f4959) &
                       (f4959 > 0.0) & np.isfinite(f5007))
        ratio = np.full((ny, nx), np.nan, dtype=np.float32)
        ratio[ratio_valid] = f5007[ratio_valid] / f4959[ratio_valid]
        ratio_residual = np.full((ny, nx), np.nan, dtype=np.float32)
        ratio_residual[ratio_valid] = (
            ratio[ratio_valid] / OIII_5007_4959_RATIO - 1.0)

        v4959 = SPEED_OF_LIGHT_KMS * (
            wave[line4959_indices] / OIII_4959_WAVELENGTH - 1.0)
        v5007 = SPEED_OF_LIGHT_KMS * (
            wave[line5007_indices] / OIII_5007_WAVELENGTH - 1.0)
        profile_v = v5007 - velocity
        peak4959 = np.full((ny, nx), np.nan, dtype=np.float32)
        for k in range(len(line4959_indices)):
            profile_plane = np.asarray(profiles["4959"][k], dtype=float)
            finite = np.isfinite(profile_plane)
            magnitude = np.where(finite, np.abs(profile_plane), np.nan)
            if k == 0:
                peak4959[:] = magnitude.astype(np.float32)
            else:
                peak4959[:] = np.fmax(peak4959, magnitude).astype(np.float32)
        profile_max = np.full((ny, nx), np.nan, dtype=np.float32)
        profile_worst_v = np.full((ny, nx), np.nan, dtype=np.float32)
        profile_rows = []
        quality5 = select_4959_sample(
            snr4959, support4959, support5007, SNR_THRESHOLDS[0])
        for k, target_velocity in enumerate(v5007):
            p5007 = np.asarray(profiles["5007"][k], dtype=float)
            p4959_left = np.asarray(profiles["4959"][max(0, np.searchsorted(
                v4959, target_velocity, side="left") - 1)], dtype=float)
            right_index = min(len(v4959) - 1,
                             np.searchsorted(v4959, target_velocity, side="left"))
            p4959_right = np.asarray(profiles["4959"][right_index], dtype=float)
            right_v = v4959[right_index]
            left_index = max(0, right_index - 1)
            left_v = v4959[left_index]
            if right_index == left_index or right_v == left_v:
                p4959_interp = p4959_right
            else:
                weight = (target_velocity - left_v) / (right_v - left_v)
                p4959_interp = p4959_left + weight * (p4959_right - p4959_left)
            residual = normalized_profile_residual(
                p5007, p4959_interp, peak4959)
            selected = quality5 & np.isfinite(residual)
            residual[~selected] = np.nan
            absolute = np.abs(residual)
            update = selected & (
                ~np.isfinite(profile_max) | (absolute > profile_max))
            profile_max[update] = absolute[update].astype(np.float32)
            profile_worst_v[update] = (target_velocity - velocity)
            selected_values = residual[selected]
            if selected_values.size:
                p16, p50, p84 = np.percentile(selected_values, [16, 50, 84])
                profile_rows.append({
                    "cube": Path(path).name,
                    "velocity_kms": float(target_velocity - velocity),
                    "wavelength_4959": float(OIII_4959_WAVELENGTH *
                                              (1.0 + target_velocity /
                                               SPEED_OF_LIGHT_KMS)),
                    "wavelength_5007": float(wave[line5007_indices[k]]),
                    "n_pixels": int(selected_values.size),
                    "median_residual": float(np.median(selected_values)),
                    "robust_scatter": robust_scatter(selected_values),
                    "p16": float(p16), "p50": float(p50), "p84": float(p84),
                    "fraction_large_negative": float(np.mean(
                        selected_values < -PROFILE_RESIDUAL_THRESHOLD)),
                    "fraction_large_positive": float(np.mean(
                        selected_values > PROFILE_RESIDUAL_THRESHOLD)),
                })

        median_profile_5007 = []
        median_profile_expected = []
        for k, target_velocity in enumerate(v5007):
            p5007 = np.asarray(profiles["5007"][k], dtype=float)
            right_index = min(len(v4959) - 1,
                             np.searchsorted(v4959, target_velocity, side="left"))
            left_index = max(0, right_index - 1)
            if v4959[right_index] == v4959[left_index]:
                p4959_interp = np.asarray(profiles["4959"][right_index], dtype=float)
            else:
                weight = ((target_velocity - v4959[left_index]) /
                          (v4959[right_index] - v4959[left_index]))
                p4959_interp = (
                    np.asarray(profiles["4959"][left_index], dtype=float) * (1.0 - weight) +
                    np.asarray(profiles["4959"][right_index], dtype=float) * weight)
            selected = quality5 & np.isfinite(f4959) & (f4959 > 0.0)
            scaled4959 = OIII_5007_4959_RATIO * p4959_interp[selected] / f4959[selected]
            scaled5007 = p5007[selected] / f4959[selected]
            finite = np.isfinite(scaled4959) & np.isfinite(scaled5007)
            median_profile_expected.append(
                float(np.median(scaled4959[finite])) if np.any(finite) else np.nan)
            median_profile_5007.append(
                float(np.median(scaled5007[finite])) if np.any(finite) else np.nan)

        return {
            "path": Path(path), "header": header, "shape": (ny, nx),
            "wave": wave, "velocity": float(velocity),
            "observed_4959": float(observed_4959),
            "observed_5007": float(observed_5007),
            "global_spectrum": global_spectrum, "global_wave": global_wave,
            "f4959": f4959, "f5007": f5007, "snr4959": snr4959,
            "peak4959": peak4959,
            "support4959": support4959, "support5007": support5007,
            "ratio": ratio, "ratio_residual": ratio_residual,
            "ratio_valid": ratio_valid, "profile_max": profile_max,
            "profile_worst_v": profile_worst_v,
            "profile_velocity": profile_v,
            "profile_rows": profile_rows,
            "median_profile_expected": np.asarray(median_profile_expected),
            "median_profile_5007": np.asarray(median_profile_5007),
            "profile_paths": profile_paths,
            "spatial_header": spatial_header(header),
        }


def ratio_statistics(result, threshold):
    selected = select_4959_sample(
        result["snr4959"], result["support4959"],
        result["support5007"], threshold)
    selected &= np.isfinite(result["ratio"])
    values = result["ratio"][selected].astype(float)
    residual = values / OIII_5007_4959_RATIO - 1.0
    return {
        "selected": int(values.size),
        "median": float(np.median(values)) if values.size else np.nan,
        "scatter": robust_scatter(values),
        "within5": float(np.mean(np.abs(residual) <= 0.05)) if values.size else np.nan,
        "within10": float(np.mean(np.abs(residual) <= 0.10)) if values.size else np.nan,
        "outside20": float(np.mean(np.abs(residual) > 0.20)) if values.size else np.nan,
        "mask": selected,
    }


def _sample_indices(mask, limit=120000, seed=17):
    indices = np.flatnonzero(mask.ravel())
    if indices.size <= limit:
        return indices
    rng = np.random.default_rng(seed)
    return rng.choice(indices, size=limit, replace=False)


def write_cube_qa(path, result):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    selected = ratio_statistics(result, SNR_THRESHOLDS[0])["mask"]
    indices = _sample_indices(selected)
    f4959 = result["f4959"].ravel()[indices]
    f5007 = result["f5007"].ravel()[indices]
    ratio = result["ratio"].ravel()[indices]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    axes[0, 0].scatter(f4959, f5007, s=2, alpha=.25)
    if f4959.size:
        upper = np.percentile(f4959, 99.5)
        line_x = np.array([0.0, upper])
        axes[0, 0].plot(line_x, OIII_5007_4959_RATIO * line_x, "r--", lw=1)
    axes[0, 0].set(xlabel="F4959", ylabel="F5007", title="Doublet flux")
    finite_ratio = ratio[np.isfinite(ratio)]
    axes[0, 1].hist(finite_ratio, bins=80, range=(0.0, 6.0), histtype="step")
    axes[0, 1].axvline(OIII_5007_4959_RATIO, color="r", ls="--")
    axes[0, 1].set(xlabel="F5007 / F4959", ylabel="pixels", title="Ratio")
    image = result["ratio_residual"]
    finite = np.isfinite(image)
    limits = np.nanpercentile(image[finite], [2, 98]) if np.any(finite) else (-1, 1)
    axes[0, 2].imshow(image, origin="lower", cmap="coolwarm",
                       vmin=limits[0], vmax=limits[1])
    axes[0, 2].set_title("Ratio residual")
    velocity = result["profile_velocity"]
    axes[1, 0].plot(velocity, result["median_profile_expected"], label="2.98 x 4959")
    axes[1, 0].plot(velocity, result["median_profile_5007"], label="5007")
    axes[1, 0].set(xlabel="velocity relative to global shift (km/s)",
                   ylabel="profile / F4959", title="Median profiles")
    axes[1, 0].legend(fontsize=8)
    profile_rows = result["profile_rows"]
    if profile_rows:
        axes[1, 1].plot([row["velocity_kms"] for row in profile_rows],
                        [row["median_residual"] for row in profile_rows], "o-")
        axes[1, 1].axhline(0.0, color="k", lw=.7)
    axes[1, 1].set(xlabel="velocity relative to global shift (km/s)",
                   ylabel="median normalized residual", title="Profile residual")
    profile_image = result["profile_max"]
    finite = np.isfinite(profile_image)
    limits = np.nanpercentile(profile_image[finite], [2, 98]) if np.any(finite) else (0, 1)
    axes[1, 2].imshow(profile_image, origin="lower", cmap="magma",
                       vmin=limits[0], vmax=limits[1])
    axes[1, 2].set_title("Maximum profile residual")
    fig.suptitle(Path(path).name)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_old_new_scatter(path, old, new):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for axis, threshold in zip(axes, SNR_THRESHOLDS):
        old_stats = ratio_statistics(old, threshold)
        new_stats = ratio_statistics(new, threshold)
        mask = old_stats["mask"] & new_stats["mask"]
        x = old["ratio_residual"][mask]
        y = new["ratio_residual"][mask]
        selected = _sample_indices(mask)
        if selected.size:
            # The flat indices are selected from the common mask, then applied
            # to the corresponding flattened residual arrays.
            x = old["ratio_residual"].ravel()[selected]
            y = new["ratio_residual"].ravel()[selected]
            axis.scatter(x, y, s=2, alpha=.25)
        axis.axline((0, 0), slope=1, color="r", ls="--", lw=.8)
        axis.set(xlabel="old ratio residual", ylabel="new ratio residual",
                 title="S/N >= %g" % threshold)
        axis.grid(alpha=.2)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def write_candidates(path, results, candidate_limit):
    fields = ["cube", "snr_threshold", "x", "y", "ra", "dec", "F4959",
              "F5007", "SNR4959", "ratio", "ratio_residual",
              "maximum_profile_residual", "worst_velocity_kms",
              "candidate_score"]
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for result in results:
            for threshold in SNR_THRESHOLDS:
                stats = ratio_statistics(result, threshold)
                mask = stats["mask"] & np.isfinite(result["profile_max"])
                score = np.maximum(np.abs(result["ratio_residual"]),
                                   result["profile_max"])
                mask &= np.isfinite(score)
                flat = np.flatnonzero(mask.ravel())
                if flat.size > candidate_limit:
                    order = np.argpartition(score.ravel()[flat], -candidate_limit)[-candidate_limit:]
                    flat = flat[order]
                flat = flat[np.argsort(score.ravel()[flat])[::-1]]
                yy, xx = np.unravel_index(flat, result["shape"])
                wcs = WCS(result["spatial_header"])
                ra, dec = wcs.wcs_pix2world(xx + 1, yy + 1, 1)
                for x, y, r_a, d_e, flat_index in zip(xx, yy, ra, dec, flat):
                    writer.writerow({
                        "cube": result["path"].name,
                        "snr_threshold": threshold,
                        "x": int(x + 1), "y": int(y + 1),
                        "ra": float(r_a), "dec": float(d_e),
                        "F4959": float(result["f4959"].ravel()[flat_index]),
                        "F5007": float(result["f5007"].ravel()[flat_index]),
                        "SNR4959": float(result["snr4959"].ravel()[flat_index]),
                        "ratio": float(result["ratio"].ravel()[flat_index]),
                        "ratio_residual": float(result["ratio_residual"].ravel()[flat_index]),
                        "maximum_profile_residual": float(result["profile_max"].ravel()[flat_index]),
                        "worst_velocity_kms": float(result["profile_worst_v"].ravel()[flat_index]),
                        "candidate_score": float(score.ravel()[flat_index]),
                    })


def write_profile_csv(path, results):
    fields = ["cube", "velocity_kms", "wavelength_4959", "wavelength_5007",
              "n_pixels", "median_residual", "robust_scatter", "p16", "p50",
              "p84", "fraction_large_negative", "fraction_large_positive"]
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for result in results:
            writer.writerows(result["profile_rows"])


def write_summary_csv(path, results):
    fields = ["cube", "global_4959_velocity_kms", "adequate_4959_support",
              "snr_threshold", "selected_pixels", "median_ratio",
              "robust_ratio_scatter", "fraction_within_5pct",
              "fraction_within_10pct", "fraction_outside_20pct",
              "max_coherent_profile_residual", "max_coherent_residual_velocity_kms"]
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for result in results:
            profile_rows = result["profile_rows"]
            if profile_rows:
                strongest = max(profile_rows,
                                key=lambda row: abs(row["median_residual"]))
                max_residual = strongest["median_residual"]
                max_velocity = strongest["velocity_kms"]
            else:
                max_residual = np.nan
                max_velocity = np.nan
            for threshold in SNR_THRESHOLDS:
                stats = ratio_statistics(result, threshold)
                writer.writerow({
                    "cube": result["path"].name,
                    "global_4959_velocity_kms": result["velocity"],
                    "adequate_4959_support": int(result["support4959"].sum()),
                    "snr_threshold": threshold,
                    "selected_pixels": stats["selected"],
                    "median_ratio": stats["median"],
                    "robust_ratio_scatter": stats["scatter"],
                    "fraction_within_5pct": stats["within5"],
                    "fraction_within_10pct": stats["within10"],
                    "fraction_outside_20pct": stats["outside20"],
                    "max_coherent_profile_residual": max_residual,
                    "max_coherent_residual_velocity_kms": max_velocity,
                })


def describe_results(results, old, new, say):
    for result in results:
        say("%s: global 4959 velocity=%0.3f km/s; observed 4959=%0.3f A; "
            "adequate-support pixels=%d" %
            (result["path"].name, result["velocity"], result["observed_4959"],
             int(result["support4959"].sum())))
        for threshold in SNR_THRESHOLDS:
            stats = ratio_statistics(result, threshold)
            say("  S/N >= %g: selected=%d, median F5007/F4959=%0.6g, "
                "robust scatter=%0.6g, within 5/10%%=%0.4f/%0.4f, "
                "outside 20%%=%0.4f" %
                (threshold, stats["selected"], stats["median"], stats["scatter"],
                 stats["within5"], stats["within10"], stats["outside20"]))
        strong = (~np.isnan(result["snr4959"]) & (result["snr4959"] >= 5.0) &
                  np.isfinite(result["ratio_residual"]) &
                  np.isfinite(result["profile_max"]) &
                  ((np.abs(result["ratio_residual"]) > STRONG_RATIO_RESIDUAL_THRESHOLD) |
                   (result["profile_max"] > STRONG_PROFILE_RESIDUAL_THRESHOLD)))
        profile_rows = result["profile_rows"]
        if profile_rows:
            strongest = max(profile_rows,
                            key=lambda row: abs(row["median_residual"]))
            say("  largest coherent profile residual=%0.6g at %0.3f km/s "
                "(%0.3f A at 5007); strong candidate pixels=%d" %
                (strongest["median_residual"], strongest["velocity_kms"],
                 strongest["wavelength_5007"], int(strong.sum())))
        else:
            say("  no high-S/N profile residual bin; strong candidate pixels=%d" %
                int(strong.sum()))

    say("Common old/new ratio-residual comparison (same support and 4959 S/N selection):")
    for threshold in SNR_THRESHOLDS:
        old_stats = ratio_statistics(old, threshold)
        new_stats = ratio_statistics(new, threshold)
        common = old_stats["mask"] & new_stats["mask"]
        old_values = old["ratio_residual"][common].astype(float)
        new_values = new["ratio_residual"][common].astype(float)
        finite = np.isfinite(old_values) & np.isfinite(new_values)
        old_values = old_values[finite]
        new_values = new_values[finite]
        say("  S/N >= %g: common=%d, old median/scatter=%0.6g/%0.6g, "
            "new median/scatter=%0.6g/%0.6g" %
            (threshold, old_values.size,
             np.median(old_values) if old_values.size else np.nan,
             robust_scatter(old_values), np.median(new_values) if new_values.size else np.nan,
             robust_scatter(new_values)))


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("old_cube", type=Path)
    parser.add_argument("new_cube", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--support-threshold", type=float, default=SUPPORT_THRESHOLD)
    parser.add_argument("--candidate-limit", type=int, default=200)
    args = parser.parse_args()
    if not args.old_cube.is_file() or not args.new_cube.is_file():
        parser.error("both input cubes must exist")
    if not 0.0 < args.support_threshold <= 1.0:
        parser.error("--support-threshold must be in (0, 1]")
    if args.candidate_limit < 1:
        parser.error("--candidate-limit must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with fits.open(args.old_cube, memmap=True) as old_hdul, \
            fits.open(args.new_cube, memmap=True) as new_hdul:
        old_shape = old_hdul[0].data.shape
        new_shape = new_hdul[0].data.shape
        if old_shape != new_shape:
            parser.error("cube shapes differ: %s versus %s" % (old_shape, new_shape))

    report = []

    def say(message=""):
        print(message)
        report.append(message)

    with tempfile.TemporaryDirectory(prefix="oiii-doublet-") as temporary_dir:
        old = analyze_cube(args.old_cube, temporary_dir, args.support_threshold)
        new = analyze_cube(args.new_cube, temporary_dir, args.support_threshold)
        say("Old cube: %s" % args.old_cube)
        say("New cube: %s" % args.new_cube)
        say("O III ratio constant: F5007/F4959=%g; support threshold=%g" %
            (OIII_5007_4959_RATIO, args.support_threshold))
        say("Profile residual definition: [F5007(v) - 2.98 F4959(v)] / "
            "[2.98 peak(|F4959|)]; 4959 is interpolated onto 5007 velocity bins.")
        describe_results((old, new), old, new, say)

        spatial = old["spatial_header"]
        for prefix, result in (("old", old), ("new", new)):
            write_map(args.output_dir / (prefix + "_oiii4959_flux.fits"),
                      result["f4959"], spatial)
            write_map(args.output_dir / (prefix + "_oiii5007_flux.fits"),
                      result["f5007"], spatial)
            write_map(args.output_dir / (prefix + "_oiii_snr4959.fits"),
                      result["snr4959"], spatial)
            write_map(args.output_dir / (prefix + "_oiii_ratio.fits"),
                      result["ratio"], spatial)
            write_map(args.output_dir / (prefix + "_oiii_ratio_residual.fits"),
                      result["ratio_residual"], spatial)
            write_map(args.output_dir / (prefix + "_oiii_profile_max_residual.fits"),
                      result["profile_max"], spatial)
            write_map(args.output_dir / (prefix + "_oiii_profile_worst_velocity.fits"),
                      result["profile_worst_v"], spatial)
            write_cube_qa(args.output_dir / (prefix + "_oiii_qa.png"), result)

        for threshold, tag in ((5.0, "snr5"), (10.0, "snr10")):
            common = (ratio_statistics(old, threshold)["mask"] &
                      ratio_statistics(new, threshold)["mask"])
            difference = np.full(old["shape"], np.nan, dtype=np.float32)
            difference[common] = (new["ratio_residual"][common] -
                                  old["ratio_residual"][common])
            write_map(args.output_dir / ("new_minus_old_oiii_ratio_residual_" + tag + ".fits"),
                      difference, spatial)
        write_old_new_scatter(
            args.output_dir / "oiii_old_vs_new_ratio_residual.png", old, new)
        write_profile_csv(args.output_dir / "oiii_profile_residual_vs_velocity.csv",
                          (old, new))
        write_summary_csv(args.output_dir / "oiii_doublet_summary.csv", (old, new))
        write_candidates(args.output_dir / "oiii_candidate_anomalies.csv",
                         (old, new), args.candidate_limit)

        (args.output_dir / "oiii_doublet_report.txt").write_text(
            "\n".join(report) + "\n")
    say("Wrote diagnostic products to %s" % args.output_dir)


if __name__ == "__main__":
    main()
