# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "OpenEXR", "pillow"]
# ///
"""Writes a viewable image (JPEG or PNG) of channels of an EXR: a render, toa.py's or detector.py's output,
materials.exr.

Three channels make a colour image (--bands, default red,green,blue when the EXR has them), one a grey one. Values
are mapped linearly from --range LO,HI (one range for all the channels, or one per channel: LO1,HI1,LO2,HI2,LO3,HI3;
default: the --percentiles of all the chosen channels together, which keeps their balance; with --per-band, each
channel's own) to 0..1, then encoded as sRGB (or not, with --linear).

    uv run tools/remote_sensing/preview.py IN.exr OUT.jpg [--bands R,G,B] [--crop X,Y,W,H] [--bin N] [--zoom N]
        [--range LO,HI[,...] | --percentiles P0,P1 [--per-band]] [--linear] [--quality Q]

--crop takes a window of IN (pixels), --bin averages N×N pixels of it, --zoom repeats each pixel N×N times.
"""

import argparse
import pathlib

import numpy as np
import OpenEXR
from PIL import Image


def floats(text, counts=(2,)):
    values = [float(v) for v in text.split(",")]
    if len(values) not in counts:
        raise argparse.ArgumentTypeError(f"'{text}': {' or '.join(map(str, counts))} values, comma-separated")
    return values


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("exr", type=pathlib.Path)
    p.add_argument("out", type=pathlib.Path)
    p.add_argument("--bands", help="one or three channels, comma-separated")
    p.add_argument("--crop", help="X,Y,W,H: a window of the EXR, pixels")
    p.add_argument("--bin", type=int, default=1, help="average N×N pixels")
    p.add_argument("--zoom", type=int, default=1, help="repeat each pixel N×N times")
    p.add_argument("--range", type=lambda t: floats(t, (2, 6)),
                   help="LO,HI (or LO,HI per channel): the values mapped to black and white")
    p.add_argument("--percentiles", type=floats, default=[0.5, 99.5], help="of the values mapped to black and white")
    p.add_argument("--per-band", action="store_true", help="the percentiles of each channel")
    p.add_argument("--linear", action="store_true", help="no sRGB encoding")
    p.add_argument("--quality", type=int, default=90, help="JPEG quality")
    args = p.parse_args()

    with OpenEXR.File(str(args.exr), separate_channels=True) as f:
        channels = {name: ch.pixels for name, ch in f.channels().items()}
    if args.bands:
        names = args.bands.split(",")
    elif all(b in channels for b in ("red", "green", "blue")):
        names = ["red", "green", "blue"]
    else:
        names = [next(iter(channels))]
    if len(names) not in (1, 3) or any(n not in channels for n in names):
        raise SystemExit(f"--bands: one or three of {', '.join(channels)}")
    image = np.stack([np.asarray(channels[n], dtype=np.float64) for n in names], axis=-1)

    if args.crop:
        x, y, w, h = (int(v) for v in args.crop.split(","))
        image = image[y:y + h, x:x + w]
    if args.bin > 1:
        h, w = (s - s % args.bin for s in image.shape[:2])
        image = image[:h, :w].reshape(h // args.bin, args.bin, w // args.bin, args.bin, -1).mean(axis=(1, 3))

    if args.range:
        if len(args.range) != 2 * image.shape[-1] and len(args.range) != 2:
            raise SystemExit(f"--range: LO,HI, or LO,HI for each of {len(names)} channels")
        lo, hi = (np.resize(args.range[i::2], image.shape[-1]) for i in (0, 1))
    elif args.per_band:
        lo, hi = (np.percentile(image.reshape(-1, image.shape[-1]), q, axis=0) for q in args.percentiles)
    else:
        lo, hi = (np.full(image.shape[-1], np.percentile(image, q)) for q in args.percentiles)
    v = np.clip((image - lo) / np.maximum(hi - lo, 1e-30), 0, 1)
    if not args.linear:
        v = np.where(v <= 0.0031308, 12.92 * v, 1.055 * v ** (1 / 2.4) - 0.055)
    if args.zoom > 1:
        v = v.repeat(args.zoom, axis=0).repeat(args.zoom, axis=1)
    out = (v * 255 + 0.5).astype(np.uint8)
    Image.fromarray(out[..., 0] if out.shape[-1] == 1 else out).save(args.out, quality=args.quality)
    print(f"wrote {args.out}: {out.shape[1]}×{out.shape[0]}, {', '.join(names)} from "
          + ", ".join(f"{a:.4g}..{b:.4g}" for a, b in zip(lo, hi)))


if __name__ == "__main__":
    main()
