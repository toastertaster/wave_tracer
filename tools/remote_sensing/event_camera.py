# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "scipy", "OpenEXR", "pillow"]
# ///
"""Images a render of a scene that build_scene.py built onto a moving event camera: its events.

An event camera's pixels report changes of brightness asynchronously: a pixel emits an ON (OFF) event when the
logarithm of its photocurrent rises (falls) by a contrast threshold since its last event. The radiance a pixel
receives is detector.py's (the render, blurred by the telescope, integrated over the pixel's footprint where the
platform's motion and jitter put it; see detector.py), sampled at time steps. Per pixel:
  - photocurrent: the electrons per second of the bands' radiance (each with its quantum efficiency: the camera sees
    all the render's bands, or --band's), plus the dark current;
  - its logarithm, optionally through the photoreceptor's first-order low-pass filter (events.cutoff_hz);
  - events: each crossing of the thresholds (events.threshold_on, events.threshold_off: contrasts in the natural
    logarithm; each pixel's drawn with an RMS of events.threshold_sigma, a fixed pattern of
    detector.fixed_pattern_seed) since the pixel's reference level emits an event, timed by linear interpolation
    within the step; the reference moves by the threshold. During events.refractory_s after an event, the pixel
    emits none (and its reference stays);
  - noise: events of random polarity at events.noise_rate_hz per pixel, at random times.
The time steps (events.time_step_s) default to a tenth of a pixel of the footprints' motion and a tenth of the
jitter's fastest period (the low-pass filter is solved exactly for a current varying linearly within a step). A
pixel emits at most one event per refractory period of a step (all those its change crosses, without one).

The instrument (INSTRUMENT.json; see event_camera.json for an example) is detector.py's telescope, jitter_urad_rms,
platform and detector (pixel_pitch_m, quantum_efficiency, dark_current_e_per_s, fixed_pattern_seed), with events.

    uv run tools/remote_sensing/event_camera.py SITE_DIR/scene.json RADIANCE.exr INSTRUMENT.json OUT.npz
        --duration S [--band NAME] [--seed N] [--preview PNG]

OUT.npz has the events, in time order: x, y (pixel column and row), t (µs from the start), p (+1 ON, -1 OFF); and
the camera's width and height. --preview writes an image of the events' sums per pixel (ON red, OFF blue).
"""

import argparse
import json
import math
import pathlib
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from detector import Optics, Platform, read_radiance, sample      # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("scene_json", type=pathlib.Path)
    p.add_argument("radiance", type=pathlib.Path)
    p.add_argument("instrument", type=pathlib.Path)
    p.add_argument("out", type=pathlib.Path)
    p.add_argument("--duration", type=float, required=True, help="s")
    p.add_argument("--band", help="the only band the camera sees (default: all)")
    p.add_argument("--seed", type=int, help="of the noise events (default: random)")
    p.add_argument("--preview", type=pathlib.Path, help="an image of the events' sums per pixel")
    args = p.parse_args()

    scene = json.loads(args.scene_json.read_text())
    inst = json.loads(args.instrument.read_text())
    optics = Optics(inst, scene)
    det, ev, k = optics.det, inst["events"], optics.k
    print(optics)
    channels, _, rows, cols = read_radiance(args.radiance, scene, k)
    bands = {b: nm for b, nm in scene["bands_nm"].items() if args.band in (None, b)}
    if not bands:
        raise SystemExit(f"--band {args.band}: not one of the scene's bands, {', '.join(scene['bands_nm'])}")

    platform = Platform(inst.get("platform"), scene, optics.focal_length, optics.render_pitch, args.duration)
    refractory = ev.get("refractory_s", 0)
    cutoff = ev.get("cutoff_hz", 0)
    if "time_step_s" in ev:
        dt = ev["time_step_s"]
    else:
        limits = [args.duration]
        speed = math.hypot(*platform.velocity) / k            # pixels/s
        if speed > 0:
            limits.append(0.1 / speed)
        if platform.max_frequency > 0:
            limits.append(0.1 / platform.max_frequency)
        dt = min(limits)
    steps = int(math.ceil(args.duration / dt))
    dt = args.duration / steps
    vy, vx = platform.velocity
    print(f"platform: ground speed {platform.ground_speed:.5g} m/s; footprints move {vy / k:.6g}, {vx / k:.6g} "
          f"pixels/s (rows, columns); {steps} time steps of {dt * 1e6:.4g} µs over {args.duration:g} s")

    # the photocurrent's sources: each band's pixel field, and electrons per second per unit radiance
    fields = [(optics.field(channels[b], b, nm), optics.electrons_per_radiance(b, nm, 1.0)) for b, nm in bands.items()]
    fixed = np.random.default_rng(det.get("fixed_pattern_seed", 0))
    dark = optics.value(det, "dark_current_e_per_s", None, 0)
    sigma = ev.get("threshold_sigma", 0)
    on = np.maximum(ev["threshold_on"] + sigma * fixed.standard_normal((rows, cols)), 0.01)
    off = np.maximum(ev["threshold_off"] + sigma * fixed.standard_normal((rows, cols)), 0.01)
    ii = np.arange(rows)[:, None] * float(k)
    jj = np.arange(cols)[None, :] * float(k)

    def log_current(t):
        dy, dx = platform.jitter(t)
        y, x = np.broadcast_arrays(ii + vy * t + dy, jj + vx * t + dx)
        current = np.full((rows, cols), dark, dtype=np.float64)
        for field, gain in fields:
            current += np.maximum(sample(field, y, x)[0], 0) * gain
        return np.log(np.maximum(current, 1e-3))

    current = log_current(0.0)
    level = current.copy()          # the photoreceptor's output
    reference = level.copy()
    last = np.full((rows, cols), -np.inf)
    time_constant = 1 / (2 * math.pi * cutoff) if cutoff > 0 else 0.0
    decay = math.exp(-dt / time_constant) if cutoff > 0 else 0.0
    events = []         # (x, y, t, p) arrays, per step
    for step in range(1, steps + 1):
        t = step * dt
        previous, before = level, current
        current = log_current(t)
        if cutoff > 0:      # the filter's exact response to a current varying linearly within the step
            g = time_constant * (current - before) / dt
            level = current - g + (previous - before + g) * decay
        else:
            level = current
        for polarity, threshold in ((1, on), (-1, off)):
            n = np.floor(polarity * (level - reference) / threshold)
            ready = (t - last) >= refractory
            if refractory > 0:      # at most as many events as refractory periods fit in the step
                n = np.minimum(n, max(1, math.floor(dt / refractory)))
            fire = (n > 0) & ready
            if not fire.any():
                continue
            y, x = np.nonzero(fire)
            count = n[fire].astype(np.int64)
            # the crossings' times, interpolating the level linearly within the step
            start, end = previous[fire], level[fire]
            ref, thr = reference[fire], threshold[fire]
            rep = np.repeat(np.arange(len(count)), count)
            j = np.arange(count.sum()) - np.repeat(np.cumsum(count) - count, count) + 1
            crossing = ref[rep] + polarity * j * thr[rep]
            span = end[rep] - start[rep]
            frac = np.clip(np.where(span != 0, (crossing - start[rep]) / np.where(span != 0, span, 1), 1), 0, 1)
            times = t - dt + frac * dt
            events.append((x[rep], y[rep], times, np.full(len(rep), polarity, np.int8)))
            reference[fire] += polarity * count * thr
            last[fire] = times[np.cumsum(count) - 1]

    noise_rate = ev.get("noise_rate_hz", 0)
    if noise_rate > 0:
        rng = np.random.default_rng(args.seed)
        n = rng.poisson(noise_rate * args.duration * rows * cols)
        events.append((rng.integers(0, cols, n), rng.integers(0, rows, n), rng.uniform(0, args.duration, n),
                       rng.choice(np.array([-1, 1], np.int8), n)))

    x, y, t, pol = (np.concatenate([e[i] for e in events]) if events else np.zeros(0) for i in range(4))
    order = np.argsort(t, kind="stable")
    np.savez_compressed(args.out, x=x[order].astype(np.uint16), y=y[order].astype(np.uint16),
                        t=np.round(t[order] * 1e6).astype(np.int64), p=pol[order].astype(np.int8),
                        width=cols, height=rows)
    n_on, n_off = int((pol > 0).sum()), int((pol < 0).sum())
    print(f"{len(t)} events ({n_on} ON, {n_off} OFF): {len(t) / args.duration / (rows * cols):.4g} per pixel per s; "
          f"wrote {args.out}")

    if args.preview:
        image = np.zeros((rows, cols, 3))
        np.add.at(image[..., 0], (y[pol > 0].astype(int), x[pol > 0].astype(int)), 1)
        np.add.at(image[..., 2], (y[pol < 0].astype(int), x[pol < 0].astype(int)), 1)
        scale = max(np.percentile(image[image > 0], 99), 1) if (image > 0).any() else 1
        Image.fromarray((np.clip(image / scale, 0, 1) * 255).astype(np.uint8)).save(args.preview)
        print(f"wrote {args.preview}")


if __name__ == "__main__":
    main()
