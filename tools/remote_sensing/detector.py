# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "scipy", "OpenEXR"]
# ///
"""Images a render of a scene that build_scene.py built through a telescope onto a detector: digital numbers.

The render records band-averaged spectral radiance (W/m²/sr/µm; toa.py's output, or a render without an
atmosphere) on a grid finer than the detector's: its pixel pitch is the detector's divided by an integer k (render
the scene with build_scene.py's --pixel-pitch and --gsd divided by k, and --pixels times k). Per band:
  1. the telescope and the line of sight's jitter blur the radiance: its Fourier transform is multiplied by the
     modulation transfer functions of
       - diffraction by the aperture, a circle with a central obscuration, averaged over the band's wavelengths
         (weighted by their photons);
       - the wavefront error (Shannon's approximation of a random error: 1 - (W/0.18)² (1 - 4(ν - 1/2)²), W the RMS
         error in waves, ν the frequency over the cut-off; for W up to ~0.15 waves);
       - jitter, Gaussian, of a given RMS angle per axis during the integration;
     and divided by the render's own beyond its pixels' area: the reconstruction filter, Gaussian of σ a quarter of
     a render pixel. The image is extended beyond its edges by reflection;
  2. each detector pixel integrates the k×k render pixels on it (its area: its own MTF, and aliasing);
  3. the radiance becomes electrons: L Δλ (λ̄/hc) A Ω τ η t, with Δλ and λ̄ the band's width and centre (a box band),
     A the pixel's area, Ω the projected solid angle of the aperture (an annulus) seen from the focal plane, τ the
     optics' transmittance, η the quantum efficiency and t the integration time;
  4. the detector adds its noise and quantizes: photo-response non-uniformity (PRNU) multiplies the signal, and dark
     current (with its non-uniformity, DSNU) adds to it, both fixed patterns of the instrument's seed; shot noise
     (Poisson) of the signal and dark current; the full well clips the electrons; read noise (Gaussian) adds to them;
     the converter divides by the gain, adds the offset, rounds and clips to its bits.
OUT.exr has one channel of digital numbers (unsigned integers) per band, named after it.

The instrument (INSTRUMENT.json; see instrument.json for an example) gives, with each value a number or, per band,
an object of numbers:
  telescope:  aperture_diameter_m, obscuration (diameter ratio), transmittance, wavefront_error_nm_rms
  jitter_urad_rms (per axis)
  detector:   pixel_pitch_m, integration_time_s, quantum_efficiency, full_well_e, read_noise_e,
              dark_current_e_per_s, prnu (RMS, relative), dsnu (RMS, relative to the dark current),
              gain_e_per_dn, offset_dn, adc_bits, fixed_pattern_seed
The focal length is the render's (SITE_DIR/scene.json).

    uv run tools/remote_sensing/detector.py SITE_DIR/scene.json RADIANCE.exr INSTRUMENT.json OUT.exr
        [--electrons ELECTRONS.exr] [--seed N]

--electrons also writes the signal's electrons per pixel, without noise, non-uniformity or dark current.
"""

import argparse
import json
import math
import pathlib

import numpy as np
import OpenEXR
import scipy.fft

H, C = 6.62607015e-34, 299792458.0
RENDER_FILTER_SIGMA = 0.25      # wave_tracer's perspective sensors' reconstruction filter, in render pixels


def per_band(value, band, name):
    """`value`, or its entry for `band` if it is per band."""
    if isinstance(value, dict):
        if band not in value:
            raise SystemExit(f"instrument: {name} has no value for band '{band}'")
        return float(value[band])
    return float(value)


def annular_mtf(nu, eps):
    """The diffraction MTF of a circular aperture with a central obscuration of diameter ratio `eps`, at `nu` the
    frequency over the cut-off (D/λf). The circle's, A(ν), the obscuration's, ε² A(ν/ε), and their overlap's."""
    nu = np.abs(nu)

    def circle(x):
        x = np.clip(x, 0, 1)
        return 2 / np.pi * (np.arccos(x) - x * np.sqrt(1 - x * x))

    mtf = circle(nu)
    if eps > 0:
        mtf = mtf + np.where(nu < eps, eps**2 * circle(nu / eps), 0)
        phi = np.arccos(np.clip((1 + eps**2 - 4 * nu**2) / (2 * eps), -1, 1))
        overlap = (2 * eps / np.pi * np.sin(phi) + (1 + eps**2) / np.pi * phi
                   - 2 * (1 - eps**2) / np.pi * np.arctan((1 + eps) / (1 - eps) * np.tan(phi / 2)) - 2 * eps**2)
        mtf = mtf + np.where(nu <= (1 - eps) / 2, -2 * eps**2, np.where(nu <= (1 + eps) / 2, overlap, 0))
        mtf = mtf / (1 - eps**2)
    return mtf


def optics_mtf(f, band_nm, f_number, eps, wfe_nm, steps=16):
    """The telescope's MTF at focal-plane frequencies `f` (cycles/m), averaged over the band's wavelengths weighted
    by their photons (∝ λ, for a radiance and response flat over the band)."""
    lo, hi = band_nm
    lam = lo + (np.arange(steps) + 0.5) * (hi - lo) / steps
    total = np.zeros_like(f)
    for l_nm in lam:
        nu = f * l_nm * 1e-9 * f_number
        mtf = annular_mtf(nu, eps)
        if wfe_nm > 0:
            w = wfe_nm / l_nm
            mtf = mtf * np.where(nu < 1, 1 - (w / 0.18) ** 2 * (1 - 4 * (nu - 0.5) ** 2), 0)
        total += l_nm * mtf
    return total / lam.sum()


def projected_solid_angle(diameter, focal_length, eps):
    """Of the annular aperture, seen on axis from the focal plane, sr."""
    def disc(d):
        return math.pi * (d / 2) ** 2 / (focal_length**2 + (d / 2) ** 2)
    return disc(diameter) - disc(eps * diameter)


def blur(L, k, mtf):
    """`L` (render pixels) filtered by `mtf(fy, fx)` (cycles per render pixel), then averaged over k×k pixels."""
    h, w = L.shape
    pad = 16 * k        # at least; to a size the FFT is fast at
    ny, nx = (scipy.fft.next_fast_len(n + 2 * pad, real=True) for n in (h, w))
    padded = np.pad(L, ((pad, ny - h - pad), (pad, nx - w - pad)), mode="reflect")
    spectrum = scipy.fft.rfft2(padded, workers=-1)
    fy = np.fft.fftfreq(ny)[:, None]
    fx = np.fft.rfftfreq(nx)[None, :]
    spectrum *= mtf(fy, fx)
    out = scipy.fft.irfft2(spectrum, s=(ny, nx), workers=-1)[pad:pad + h, pad:pad + w]
    return out.reshape(h // k, k, w // k, k).mean(axis=(1, 3))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("scene_json", type=pathlib.Path)
    p.add_argument("radiance", type=pathlib.Path)
    p.add_argument("instrument", type=pathlib.Path)
    p.add_argument("out", type=pathlib.Path)
    p.add_argument("--electrons", type=pathlib.Path, help="also write the signal's mean electrons per pixel")
    p.add_argument("--seed", type=int, help="of the temporal noise (default: random)")
    args = p.parse_args()

    scene = json.loads(args.scene_json.read_text())
    if "focal_length_m" not in scene:
        raise SystemExit(f"{args.scene_json} has no focal length: rebuild the scene with build_scene.py")
    inst = json.loads(args.instrument.read_text())
    tel, det = inst["telescope"], inst["detector"]

    focal_length, render_pitch = scene["focal_length_m"], scene["pixel_pitch_m"]
    pitch = det["pixel_pitch_m"]
    k = round(pitch / render_pitch)
    if k < 1 or abs(k * render_pitch - pitch) > 1e-6 * pitch:
        raise SystemExit(f"the detector's pixel pitch, {pitch * 1e6:g} µm, is not a multiple of the render's, "
                         f"{render_pitch * 1e6:g} µm")
    D, eps = tel["aperture_diameter_m"], tel.get("obscuration", 0)
    N = focal_length / D
    omega = projected_solid_angle(D, focal_length, eps)
    jitter_px = inst.get("jitter_urad_rms", 0) * 1e-6 * focal_length / render_pitch     # σ, render pixels
    print(f"telescope: f {focal_length:g} m, D {D:g} m (f/{N:.2f}), obscuration {eps:g}, Ω {omega:.5g} sr; "
          f"detector pixels {pitch * 1e6:g} µm, {k}×{k} render pixels each; jitter σ {jitter_px / k:.3f} pixels")

    with OpenEXR.File(str(args.radiance), separate_channels=True) as f:
        header = dict(f.header())
        channels = {name: np.asarray(ch.pixels, dtype=np.float64) for name, ch in f.channels().items()}
    missing = set(scene["bands_nm"]) - set(channels)
    if missing:
        raise SystemExit(f"{args.radiance} has channels {sorted(channels)}, not bands {sorted(missing)}")
    shape = next(iter(channels.values())).shape
    if shape[0] % k or shape[1] % k:
        raise SystemExit(f"{args.radiance} is {shape[1]}×{shape[0]} pixels: not a multiple of {k}")
    det_shape = (shape[0] // k, shape[1] // k)

    # the instrument's fixed patterns, the same for every image (and band: one per pixel and band)
    fixed = np.random.default_rng(det.get("fixed_pattern_seed", 0))
    rng = np.random.default_rng(args.seed)
    t = det["integration_time_s"]
    max_dn = 2 ** int(det["adc_bits"]) - 1
    dn_out, electrons_out = {}, {}
    for band, (lo, hi) in scene["bands_nm"].items():
        def get(section, name, default=None):
            if name not in section and default is not None:
                return default
            return per_band(section[name], band, name)

        tau, qe, wfe = get(tel, "transmittance"), get(det, "quantum_efficiency"), get(tel, "wavefront_error_nm_rms", 0)
        # electrons per unit of band-averaged spectral radiance (W/m²/sr/µm)
        gain_L = (hi - lo) * 1e-3 * ((lo + hi) / 2 * 1e-9 / (H * C)) * pitch**2 * omega * tau * qe * t

        def mtf(fy, fx):
            f2 = fy**2 + fx**2
            m = optics_mtf(np.sqrt(f2) / render_pitch, (lo, hi), N, eps, wfe)
            m = m * np.exp(-2 * math.pi**2 * jitter_px**2 * f2)
            return m / np.exp(-2 * math.pi**2 * RENDER_FILTER_SIGMA**2 * f2)

        L = blur(channels[band], k, mtf)
        signal = np.maximum(L, 0) * gain_L

        prnu = 1 + get(det, "prnu", 0) * fixed.standard_normal(det_shape)
        dark = get(det, "dark_current_e_per_s", 0) * t * np.maximum(
            1 + get(det, "dsnu", 0) * fixed.standard_normal(det_shape), 0)
        e = rng.poisson(np.maximum(signal * prnu, 0) + dark).astype(np.float64)
        saturated = e >= get(det, "full_well_e")
        e = np.minimum(e, get(det, "full_well_e"))
        e += get(det, "read_noise_e", 0) * rng.standard_normal(det_shape)
        dn = np.clip(np.round(e / get(det, "gain_e_per_dn") + get(det, "offset_dn", 0)), 0, max_dn).astype(np.uint32)
        dn_out[band], electrons_out[band] = dn, signal

        # the mean signal's SNR, and the detector's MTF at Nyquist
        s = signal.mean()
        noise = math.sqrt(s + dark.mean() + get(det, "read_noise_e", 0) ** 2 + get(det, "gain_e_per_dn") ** 2 / 12)
        nyquist = 0.5 / pitch
        mtf_nyq = (optics_mtf(np.array([nyquist]), (lo, hi), N, eps, wfe)[0]
                   * math.exp(-2 * math.pi**2 * (jitter_px / k) ** 2 * 0.25) * np.sinc(0.5))
        print(f"{band}: {gain_L:.4g} e⁻ per W/m²/sr/µm; mean {L.mean():.4g} W/m²/sr/µm, {s:.4g} e⁻, "
              f"{dn.mean():.1f} DN; SNR {s / noise:.1f}; saturated {saturated.mean():.2%}; "
              f"λN/p {(lo + hi) / 2e9 * N / pitch:.2f}, MTF at Nyquist {mtf_nyq:.3f}")
        cutoff = 1 / ((lo * 1e-9) * N)       # cycles/m
        if cutoff * render_pitch > 0.5 * 1.25:
            print(f"  warning: the optics pass frequencies to {cutoff * render_pitch:.2f} cycles per render pixel, "
                  "beyond the render's Nyquist frequency (0.5): render finer")

    hdr = {key: v for key, v in header.items() if key in ("renderer", "scene", "sensor", "samples")}
    with OpenEXR.File(hdr, {b: OpenEXR.Channel(v) for b, v in dn_out.items()}) as f:
        f.write(str(args.out))
    if args.electrons:
        with OpenEXR.File(hdr, {b: OpenEXR.Channel(v.astype(np.float32)) for b, v in electrons_out.items()}) as f:
            f.write(str(args.electrons))
    print(f"wrote {args.out}" + (f" and {args.electrons}" if args.electrons else ""))


if __name__ == "__main__":
    main()
