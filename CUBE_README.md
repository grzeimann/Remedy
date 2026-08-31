Cube quick reference (make_mosaic_cube.py output)

What these cubes are
- Wavelength-rectified: spectra are resampled onto a common linear wavelength grid before imaging.
  - Wavelength axis: 3470–5540 Å, 2 Å per slice (1036 planes).
- Flux-calibrated: values trace calibrated flux density.
  - Units: 1e-17 erg s^-1 cm^-2 Å^-1.
  - Note on surface brightness: during imaging each fiber’s flux density is divided by the fiber area (π·0.75^2 arcsec^2) to create a surface-brightness-like map per slice. Thus pixels effectively represent flux density per square arcsecond after the spatial resampling/smoothing step.
- Sky-subtracted: a sky-residual spectrum is estimated from low-background regions and subtracted from all fibers for the selected filter response.
  - Residuals may remain, especially if the field lacks clean sky.
- ADR corrected: differential atmospheric refraction (DAR/ADR) is modeled and applied to shift RA/Dec as a function of wavelength prior to imaging.

Known artifacts and caveats
- Fiber normalization artifacts can persist at the amplifier level, often visible as quadrant-like patterns (~1/4 of the IFU field of view).
- Regions with no fiber coverage (or masked/flagged data) are filled with zeros in the final cube.
- A weight cube is written for support diagnostics; uncertainty products are
  described below.

Outputs written by make_mosaic_cube.py
- <surname>_cube.fits: the data cube.
- <surname>_variance_cube.fits: the variance of the final SCI estimator, in
  SCI-units squared. Formal shot variances are propagated through the final
  shot median; for five or more supported shots, robust shot-to-shot scatter
  may increase the adopted variance.
- <surname>_errorcube.fits: backwards-compatible 1σ companion equal to
  `sqrt(variance_cube)`. `VAR` means variance; `ERROR` means standard
  deviation.
- <surname>_dq_cube.fits: uint16 data-quality bit cube. Bits are independent:
  bit 0 = `NCONTRIB < 2`; bit 1 = SCI valid but VAR unavailable; bit 2 =
  empirical VAR adopted over formal; bit 3 = formal VAR adopted; bit 4 =
  empirical-only VAR adopted.

Axis definitions and WCS (DS9-friendly)
- 3D WCS with WCSAXES = 3.
- Spatial axes (1, 2): RA/Dec in degrees with TAN (gnomonic) projection.
  - CTYPE1 = 'RA---TAN', CUNIT1 = 'deg'
  - CTYPE2 = 'DEC--TAN', CUNIT2 = 'deg'
  - CRPIX1, CRPIX2: set to image center.
- Spectral axis (3): linear wavelength grid in Angstroms.
  - CTYPE3 = 'WAVE', CUNIT3 = 'Angstrom'
  - CRVAL3 = 3470., CRPIX3 = 1., CDELT3 = 2.
  - SPECSYS = 'TOPOCENT'.

Imaging details (how slices are formed)
- Fibers are projected to the tangent-plane WCS; positions are averaged over in-band wavelengths if a filter response is provided.
- Per-slice imaging uses a Gaussian kernel and median combines exposures/dithers where applicable.
- Each fiber’s contribution is divided by the fiber area (π·0.75^2 arcsec^2) to yield a surface-brightness representation.
- The active Gaussian-splat reconstruction uses the same full science weight
  sum for SCI and formal variance propagation. A formal shot variance is left
  undefined when any SCI-contributing fiber lacks a finite positive error.

Units recap
- Flux cube: effectively surface-brightness flux density in 1e-17 erg s^-1 cm^-2 Å^-1 arcsec^-2 after imaging.
- Error cube: 1σ uncertainties in the same units as the flux cube.

Typical spatial setup
- Default pixel scale is 1.0 arcsec unless overridden with --pixel_scale.
- Image size is set from the requested center and size; CRPIX is at the image center for both axes.

How to view in DS9
- Open <surname>_cube.fits. DS9 should recognize it as a 3D cube (WCSAXES=3).
- Use Frame → 3D or the slice slider to navigate the wavelength axis.
- You can overlay regions in RA/Dec thanks to the TAN WCS.

Troubleshooting tips
- If the cube looks empty: check the logs for the “Fibers in region” preflight count and the “specarray global stats” line; verify that your target region actually contains fibers.
- If you see quadrant-like patterns: these are likely amplifier-level fiber normalization residuals; consider additional flat-fielding or masking strategies.
- If units look off: remember the surface-brightness division by fiber area during imaging. The underlying spectral calibration is in 1e-17 erg s^-1 cm^-2 Å^-1, but maps are expressed per arcsec^2.
