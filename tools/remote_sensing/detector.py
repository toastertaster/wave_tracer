# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "scipy", "OpenEXR"]
# ///
"""Images a render of a scene that build_scene.py built through a telescope onto a moving detector: digital numbers.

The render records band-averaged spectral radiance (W/m²/sr/µm; toa.py's output, or a render without an
atmosphere) on a grid finer than the detector's: its pixel pitch is the detector's divided by an integer k (render
the scene with build_scene.py's --pixel-pitch and --gsd divided by k, and --pixels times k). Per band:
  1. the telescope blurs the radiance: its Fourier transform is multiplied by the modulation transfer functions of
       - diffraction by the aperture, a circle with a central obscuration, averaged over the band's wavelengths
         (weighted by their photons);
       - the wavefront error (Shannon's approximation of a random error: 1 - (W/0.18)² (1 - 4(ν - 1/2)²), W the RMS
         error in waves, ν the frequency over the cut-off; for W up to ~0.15 waves);
       - jitter_urad_rms: jitter faster than the time steps (below), Gaussian, of that RMS angle per axis;
     and divided by the render's own beyond its pixels' area: the reconstruction filter, Gaussian of σ a quarter of
     a render pixel. The image is extended beyond its edges by reflection;
  2. each detector pixel integrates the radiance over its area, where its footprint is at each time step of its
     exposure (the render's pixels' values, as a cubic spline over their positions);
  3. the radiance becomes electrons: L Δλ (λ̄/hc) A Ω τ η t, with Δλ and λ̄ the band's width and centre (a box band),
     A the pixel's area, Ω the projected solid angle of the aperture (an annulus) seen from the focal plane, τ the
     optics' transmittance, η the quantum efficiency and t the integration time;
  4. the detector adds its noise and quantizes: photo-response non-uniformity (PRNU) multiplies the signal, and dark
     current (with its non-uniformity, DSNU) adds to it, both fixed patterns of the instrument's seed; shot noise
     (Poisson) of the signal and dark current; the full well clips the electrons; read noise (Gaussian) adds to them;
     the converter divides by the gain, adds the offset, rounds and clips to its bits.
OUT.exr has one channel of digital numbers (unsigned integers) per band, named after it.

Time. The scene is rendered once, at one instant: the platform's motion moves the pixels' footprints over that
render (along its columns: the render's rows are along track), which ignores the parallax and the change of view
during an acquisition. The footprints move at the ground speed of a circular orbit at the scene's altitude (or
platform.ground_speed_m_s), less platform.motion_compensation (a fraction: 1 for a staring camera), towards
increasing rows (platform.scan "south", default) or decreasing ones ("north"), at platform.drift_deg from the
columns (a crab angle), and with the line of sight's jitter: platform.jitter's sinusoids (frequency_hz, along_urad,
across_urad, phase_deg; random phases without) and random jitter (random_urad_rms per axis, of a Lorentzian spectrum
of corner random_bandwidth_hz), of seed jitter.seed. The camera (camera.type) is
  frame      every pixel integrates during integration_time_s, centred on the render's instant;
  pushbroom  a line of pixels across track; each row of the image is a line period later, and integrates during
             integration_time_s (default: the whole line period);
  tdi        time delay and integration: as pushbroom, but the charge of each row moves with the image across
             camera.stages lines of pixels, one per line period, integrating in each; camera.readout "charge"
             (default: a CCD's, read once) or "digital" (each line read: read noise √stages times; each line's
             full well).
The line period is camera.line_period_s, or by default the time the footprints take to move by one detector pixel,
times 1 + camera.line_period_error (a synchronization error). Each exposure is sampled at camera.time_steps times
(default: enough for its motion to be at most a tenth of a pixel per step, and for ten steps per period of the
jitter's fastest sinusoid or random bandwidth). Lines' PRNU and dark current are per line pixel, averaged over the
stages a row's charge visits.

The instrument (INSTRUMENT.json; see instrument.json for an example) gives, with each value a number or, per band,
an object of numbers:
  telescope:  aperture_diameter_m, obscuration (diameter ratio), transmittance, wavefront_error_nm_rms
  jitter_urad_rms (per axis; jitter faster than the time steps)
  detector:   pixel_pitch_m, integration_time_s, quantum_efficiency, full_well_e, read_noise_e,
              dark_current_e_per_s, prnu (RMS, relative), dsnu (RMS, relative to the dark current),
              gain_e_per_dn, offset_dn, adc_bits, fixed_pattern_seed
  camera, platform (optional): as above; without them, a frame camera, and no motion
The focal length and altitude are the render's (SITE_DIR/scene.json).

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
from scipy.ndimage import map_coordinates, spline_filter
from scipy.signal import lfilter

H, C = 6.62607015e-34, 299792458.0
GM_EARTH, R_EARTH = 3.986004418e14, 6371008.8     # m³/s², m (mean radius, as the satellite_frame sensor's)
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


def pixel_field(L, k, mtf):
    """What a detector pixel of k×k render pixels integrates, wherever it is: `L` (render pixels) filtered by
    `mtf(fy, fx)` (cycles per render pixel), then averaged over the k×k render pixels from each one (at row y, column
    x: over rows y..y+k-1 and columns x..x+k-1). Returned as cubic spline coefficients (for `sample`): a static
    detector's pixel (i, j) is at (k i, k j)."""
    h, w = L.shape
    pad = 16 * k        # at least; to a size the FFT is fast at
    ny, nx = (scipy.fft.next_fast_len(n + 2 * pad, real=True) for n in (h, w))
    padded = np.pad(L, ((pad, ny - h - pad), (pad, nx - w - pad)), mode="reflect")
    spectrum = scipy.fft.rfft2(padded, workers=-1)
    fy = np.fft.fftfreq(ny)[:, None]
    fx = np.fft.rfftfreq(nx)[None, :]
    spectrum *= mtf(fy, fx)
    blurred = scipy.fft.irfft2(spectrum, s=(ny, nx), workers=-1)[pad:pad + h, pad:pad + w]
    c = np.pad(blurred.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    box = (c[k:, k:] - c[:-k, k:] - c[k:, :-k] + c[:-k, :-k]) / (k * k)
    return spline_filter(box, order=3, mode="mirror")


def sample(field, rows, cols):
    """The pixel field (`pixel_field`) at render-pixel positions; and how many of them were outside it."""
    outside = int(np.count_nonzero((rows < 0) | (rows > field.shape[0] - 1) | (cols < 0) | (cols > field.shape[1] - 1)))
    return map_coordinates(field, [rows, cols], order=3, mode="mirror", prefilter=False), outside


class Optics:
    """The instrument's telescope and pixels, and how a render (of `scene`) maps onto them."""

    def __init__(self, inst, scene):
        if "focal_length_m" not in scene:
            raise SystemExit("the scene's JSON has no focal length: rebuild the scene with build_scene.py")
        self.tel, self.det = inst["telescope"], inst["detector"]
        self.focal_length, self.render_pitch = scene["focal_length_m"], scene["pixel_pitch_m"]
        self.pitch = self.det["pixel_pitch_m"]
        self.k = round(self.pitch / self.render_pitch)
        if self.k < 1 or abs(self.k * self.render_pitch - self.pitch) > 1e-6 * self.pitch:
            raise SystemExit(f"the detector's pixel pitch, {self.pitch * 1e6:g} µm, is not a multiple of the "
                             f"render's, {self.render_pitch * 1e6:g} µm")
        self.diameter, self.eps = self.tel["aperture_diameter_m"], self.tel.get("obscuration", 0)
        self.f_number = self.focal_length / self.diameter
        self.omega = projected_solid_angle(self.diameter, self.focal_length, self.eps)
        self.fast_jitter = inst.get("jitter_urad_rms", 0) * 1e-6 * self.focal_length / self.render_pitch  # σ, render px

    def __str__(self):
        return (f"telescope: f {self.focal_length:g} m, D {self.diameter:g} m (f/{self.f_number:.2f}), obscuration "
                f"{self.eps:g}, Ω {self.omega:.5g} sr; detector pixels {self.pitch * 1e6:g} µm, {self.k}×{self.k} "
                f"render pixels each; fast jitter σ {self.fast_jitter / self.k:.3f} pixels")

    def value(self, section, name, band, default=None):
        """The telescope's or detector's (`section`) value `name` for `band`."""
        if name not in section:
            if default is None:
                raise SystemExit(f"instrument: {name} is missing")
            return default
        return per_band(section[name], band, name)

    def field(self, L, band, band_nm):
        """`pixel_field` of the band's render `L`, blurred by the telescope and the fast jitter."""
        wfe = self.value(self.tel, "wavefront_error_nm_rms", band, 0)

        def mtf(fy, fx):
            f2 = fy**2 + fx**2
            m = optics_mtf(np.sqrt(f2) / self.render_pitch, band_nm, self.f_number, self.eps, wfe)
            m = m * np.exp(-2 * math.pi**2 * self.fast_jitter**2 * f2)
            return m / np.exp(-2 * math.pi**2 * RENDER_FILTER_SIGMA**2 * f2)

        return pixel_field(L, self.k, mtf)

    def electrons_per_radiance(self, band, band_nm, exposure):
        """Electrons a pixel collects in `exposure` (s) per unit of band-averaged spectral radiance (W/m²/sr/µm)."""
        lo, hi = band_nm
        return ((hi - lo) * 1e-3 * ((lo + hi) / 2 * 1e-9 / (H * C)) * self.pitch**2 * self.omega
                * self.value(self.tel, "transmittance", band) * self.value(self.det, "quantum_efficiency", band)
                * exposure)

    def static_mtf_at_nyquist(self, band, band_nm):
        """The telescope's, the fast jitter's and the pixel's MTF at the detector's Nyquist frequency."""
        wfe = self.value(self.tel, "wavefront_error_nm_rms", band, 0)
        return (optics_mtf(np.array([0.5 / self.pitch]), band_nm, self.f_number, self.eps, wfe)[0]
                * math.exp(-2 * math.pi**2 * (self.fast_jitter / self.k) ** 2 * 0.25) * np.sinc(0.5))

    def warn_undersampled(self, band_nm):
        cutoff = 1 / (band_nm[0] * 1e-9 * self.f_number) * self.render_pitch      # cycles per render pixel
        if cutoff > 0.5 * 1.25:
            print(f"  warning: the optics pass frequencies to {cutoff:.2f} cycles per render pixel, beyond the "
                  "render's Nyquist frequency (0.5): render finer")


def read_radiance(path, scene, k):
    """The render's bands (W/m²/sr/µm), its EXR header, and the detector's rows and columns (of k×k render pixels)."""
    with OpenEXR.File(str(path), separate_channels=True) as f:
        header = dict(f.header())
        channels = {name: np.asarray(ch.pixels, dtype=np.float64) for name, ch in f.channels().items()}
    missing = set(scene["bands_nm"]) - set(channels)
    if missing:
        raise SystemExit(f"{path} has channels {sorted(channels)}, not bands {sorted(missing)}")
    shape = next(iter(channels.values())).shape
    if shape[0] % k or shape[1] % k:
        raise SystemExit(f"{path} is {shape[1]}×{shape[0]} pixels: not a multiple of {k}")
    return channels, header, shape[0] // k, shape[1] // k


def write_exr(path, header, images):
    hdr = {key: v for key, v in header.items() if key in ("renderer", "scene", "sensor", "samples")}
    with OpenEXR.File(hdr, {b: OpenEXR.Channel(v) for b, v in images.items()}) as f:
        f.write(str(path))


class Platform:
    """The motion of the detector's footprints over the render, in render pixels (rows, columns), at times (s) from
    the render's instant: the platform's motion and the line of sight's jitter."""

    def __init__(self, spec, scene, focal_length, render_pitch, duration):
        """`spec`: the instrument's platform; None for none (no motion)."""
        h = scene["altitude_m"]
        if spec is None:
            spec = {"ground_speed_m_s": 0}
        v = spec.get("ground_speed_m_s", math.sqrt(GM_EARTH / (R_EARTH + h)) * R_EARTH / (R_EARTH + h))
        v *= 1 - spec.get("motion_compensation", 0)
        scan = spec.get("scan", "south")
        if scan not in ("south", "north"):
            raise SystemExit(f"platform.scan: 'south' or 'north', not '{scan}'")
        drift = math.radians(spec.get("drift_deg", 0))
        render_gsd = h * render_pitch / focal_length
        self.ground_speed = v
        self.velocity = ((1 if scan == "south" else -1) * v * math.cos(drift) / render_gsd,
                         v * math.sin(drift) / render_gsd)      # render pixels/s
        to_px = 1e-6 * focal_length / render_pitch                # per µrad
        jitter = spec.get("jitter", {})
        rng = np.random.default_rng(jitter.get("seed", 0))
        self.sinusoids = [(s["frequency_hz"], s.get("along_urad", 0) * to_px, s.get("across_urad", 0) * to_px,
                           math.radians(s["phase_deg"]) if "phase_deg" in s else rng.uniform(0, 2 * math.pi))
                          for s in jitter.get("sinusoids", [])]
        self.max_frequency = max([f for f, *_ in self.sinusoids], default=0)
        self.random = None
        sigma = jitter.get("random_urad_rms", 0) * to_px
        if sigma > 0:
            # a Gauss-Markov process (Lorentzian spectrum): AR(1) on a grid of 1/(200 B), interpolated linearly (which
            # lowers its RMS by ~0.5%)
            bandwidth = jitter["random_bandwidth_hz"]
            self.max_frequency = max(self.max_frequency, bandwidth)
            dt = 1 / (200 * bandwidth)
            n = int(math.ceil(duration / dt)) + 2
            rho = math.exp(-2 * math.pi * bandwidth * dt)
            noise = rng.standard_normal((n, 2)) * sigma * math.sqrt(1 - rho * rho)
            start = rng.standard_normal((1, 2)) * sigma
            self.random = (dt, lfilter([1], [1, -rho], noise, axis=0, zi=rho * start)[0])

    def jitter(self, t):
        """The line of sight's jitter at times `t`: (rows, columns) offsets."""
        t = np.asarray(t, dtype=float)
        dy, dx = np.zeros_like(t), np.zeros_like(t)
        for f, a_along, a_across, phase in self.sinusoids:
            s = np.sin(2 * math.pi * f * t + phase)
            dy, dx = dy + a_along * s, dx + a_across * s
        if self.random is not None:
            dt, x = self.random
            grid = np.arange(len(x)) * dt
            dy = dy + np.interp(t, grid, x[:, 0])
            dx = dx + np.interp(t, grid, x[:, 1])
        return dy, dx

    def time_steps(self, exposure, k):
        """Steps that sample an exposure: of at most a tenth of a detector pixel of motion, and ten per period of
        the jitter's fastest component."""
        motion = math.hypot(*self.velocity) * exposure / (0.1 * k)
        return max(1, math.ceil(motion), math.ceil(10 * self.max_frequency * exposure))



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
    inst = json.loads(args.instrument.read_text())
    optics = Optics(inst, scene)
    det, k = optics.det, optics.k
    cam = inst.get("camera", {"type": "frame"})
    kind = cam.get("type", "frame")
    if kind not in ("frame", "pushbroom", "tdi"):
        raise SystemExit(f"camera.type: 'frame', 'pushbroom' or 'tdi', not '{kind}'")
    stages = int(cam.get("stages", 1)) if kind == "tdi" else 1
    readout = cam.get("readout", "charge")
    if readout not in ("charge", "digital"):
        raise SystemExit(f"camera.readout: 'charge' or 'digital', not '{readout}'")
    print(optics)
    channels, header, rows, cols = read_radiance(args.radiance, scene, k)

    # the acquisition's timing, and the footprints' positions at each time step: (rows, columns) of render pixels
    focal_length, render_pitch = optics.focal_length, optics.render_pitch
    if kind == "frame":
        period = exposure = det["integration_time_s"]
        duration = exposure
    else:
        speed = abs(Platform(inst.get("platform"), scene, focal_length, render_pitch, 0).velocity[0])
        if "line_period_s" in cam:
            period = cam["line_period_s"]
        elif speed > 0:
            period = k / speed * (1 + cam.get("line_period_error", 0))
        else:
            raise SystemExit("camera.line_period_s is needed: the footprints do not move along track")
        exposure = det.get("integration_time_s", period)
        if exposure > period * (1 + 1e-9):
            raise SystemExit(f"detector.integration_time_s ({exposure:g} s) exceeds the line period ({period:g} s)")
        duration = (rows + stages) * period
    platform = Platform(inst.get("platform"), scene, focal_length, render_pitch, duration)
    steps = int(cam.get("time_steps", platform.time_steps(exposure, k)))
    vy, vx = platform.velocity
    tau = (np.arange(steps) + 0.5) / steps * exposure - exposure / 2          # within an exposure, centred
    ii = np.arange(rows)[:, None] * float(k)
    jj = np.arange(cols)[None, :] * float(k)
    positions = []      # (stage, rows, columns) of the footprints at each step
    if kind == "frame":
        for s in range(steps):
            dy, dx = platform.jitter(tau[s] + exposure / 2)
            positions.append((0, ii + vy * tau[s] + dy, jj + vx * tau[s] + dx))
    else:
        # row r's charge, in stage n during [(r+n)T, (r+n)T + exposure]: its footprint is on the ground that stage
        # 0 saw n line periods earlier; registered on the render's grid at the acquisition's middle
        slip = vy * period - k          # per stage: the footprints' motion less the charge's
        t_mid = duration / 2
        for n in range(stages):
            for s in range(steps):
                t = (np.arange(rows)[:, None] + n) * period + tau[s] + exposure / 2
                dy, dx = platform.jitter(t)
                positions.append((n, ii + (n - (stages - 1) / 2) * slip + vy * tau[s] + dy,
                                  jj + vx * (t - t_mid) + dx))
        print(f"{kind}: line period {period * 1e6:.4g} µs, {stages} stage{'s' * (stages > 1)} of "
              f"{exposure * 1e6:.4g} µs ({readout} readout); slip {slip / k:+.4f} pixels per stage")
    print(f"platform: ground speed {platform.ground_speed:.5g} m/s; footprints move {vy / k:.6g}, {vx / k:.6g} "
          f"pixels/s (rows, columns); {steps} time step{'s' * (steps > 1)} per exposure of {exposure * 1e6:.4g} µs")

    # the instrument's fixed patterns, the same for every image (one per pixel, or line pixel and stage, and band)
    fixed = np.random.default_rng(det.get("fixed_pattern_seed", 0))
    rng = np.random.default_rng(args.seed)
    max_dn = 2 ** int(det["adc_bits"]) - 1
    pattern_shape = (rows, cols) if kind == "frame" else (stages, cols)
    dn_out, electrons_out = {}, {}
    for band, band_nm in scene["bands_nm"].items():
        def get(name, default=None):
            return optics.value(det, name, band, default)

        gain_L = optics.electrons_per_radiance(band, band_nm, exposure)
        field = optics.field(channels[band], band, band_nm)
        per_stage = np.zeros((stages, rows, cols))
        outside = 0
        for n, y, x in positions:
            values, out = sample(field, *np.broadcast_arrays(y, x))
            per_stage[n] += np.maximum(values, 0) / steps
            outside += out
        signal_stages = per_stage * gain_L          # electrons, per stage
        signal = signal_stages.sum(0)

        prnu = 1 + get("prnu", 0) * fixed.standard_normal(pattern_shape)
        dark = get("dark_current_e_per_s", 0) * exposure * np.maximum(
            1 + get("dsnu", 0) * fixed.standard_normal(pattern_shape), 0)
        if kind == "frame":
            mean = signal * prnu + dark
        else:   # each stage's line pixel, for every row
            mean = (signal_stages * prnu[:, None, :]).sum(0) + dark.sum(0)[None, :]
        e = rng.poisson(np.maximum(mean, 0)).astype(np.float64)
        full_well = get("full_well_e") * (stages if readout == "digital" else 1)
        saturated = e >= full_well
        e = np.minimum(e, full_well)
        read_noise = get("read_noise_e", 0) * (math.sqrt(stages) if readout == "digital" else 1)
        e += read_noise * rng.standard_normal(e.shape)
        dn = np.clip(np.round(e / get("gain_e_per_dn") + get("offset_dn", 0)), 0, max_dn).astype(np.uint32)
        dn_out[band], electrons_out[band] = dn, signal.astype(np.float32)

        s = signal.mean()
        noise = math.sqrt(s + dark.mean() * stages + read_noise**2 + get("gain_e_per_dn") ** 2 / 12)
        print(f"{band}: {gain_L * stages:.4g} e⁻ per W/m²/sr/µm; mean {s / gain_L / stages:.4g} W/m²/sr/µm, "
              f"{s:.4g} e⁻, {dn.mean():.1f} DN; SNR {s / noise:.1f}; saturated {saturated.mean():.2%}; "
              f"λN/p {sum(band_nm) / 2e9 * optics.f_number / optics.pitch:.2f}, static MTF at Nyquist "
              f"{optics.static_mtf_at_nyquist(band, band_nm):.3f}"
              + (f"; {outside / len(positions) / (rows * cols):.2%} of footprints beyond the render" if outside else ""))
        optics.warn_undersampled(band_nm)

    write_exr(args.out, header, dn_out)
    if args.electrons:
        write_exr(args.electrons, header, electrons_out)
    print(f"wrote {args.out}" + (f" and {args.electrons}" if args.electrons else ""))


if __name__ == "__main__":
    main()
