#!/usr/bin/env python3
"""Write structured-light pattern sequences to disk as PNG sets.

The Mitsuba renderer (``gen_mitsuba_dataset.py``) projects a *folder of PNGs*, so
this script materialises pattern strategies into folders you can feed straight to
``--patterns``. Each strategy lands in its own folder of zero-padded frames in
projection order::

    patterns/graycode/pat_00.png, pat_01.png, ...     (monochrome)
    patterns/rainbow/pat_00.png                        (colour, single shot)

Two families are produced:

  * **monochrome** decoders from :mod:`lux.methods` (Gray code, phase shift),
  * **colour** single-shot patterns (a hue sweep and an RGB phase triple) that
    the renderer now projects and captures in full colour.

    python scripts/gen_patterns.py --width 1920 --height 1080
    python scripts/gen_mitsuba_dataset.py --scene wavy --patterns patterns/rainbow
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from lux import io  # noqa: E402
from lux.methods import REGISTRY, build_method  # noqa: E402


# --------------------------------------------------------------------------
# Colour single-shot patterns (no monochrome-method equivalent)
# --------------------------------------------------------------------------
def _hsv_to_rgb(h: np.ndarray, s: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Vectorised HSV->RGB; all inputs and outputs in [0, 1]."""
    i = np.floor(h * 6.0).astype(int) % 6
    f = h * 6.0 - np.floor(h * 6.0)
    p, q, t = v * (1 - s), v * (1 - f * s), v * (1 - (1 - f) * s)
    sel = [i == k for k in range(6)]
    r = np.select(sel, [v, q, p, p, t, v])
    g = np.select(sel, [t, v, v, q, p, p])
    b = np.select(sel, [p, p, t, v, v, q])
    return np.stack([r, g, b], axis=-1)


def _rainbow(width: int, height: int) -> np.ndarray:
    """Single colour frame: hue swept across projector columns (column = hue)."""
    hue = np.tile(np.linspace(0.0, 1.0, width, endpoint=False), (height, 1))
    rgb = _hsv_to_rgb(hue, np.ones_like(hue), np.ones_like(hue))
    return rgb[None]  # (1, H, W, 3)


def _rgb_phase(width: int, height: int, periods: int = 24) -> np.ndarray:
    """Single colour frame: R/G/B carry the same sinusoid at 0/120/240 deg phase."""
    x = np.linspace(0.0, 1.0, width, endpoint=False)
    chans = [0.5 + 0.5 * np.sin(2 * np.pi * periods * x + ph)
             for ph in (0.0, 2 * np.pi / 3, 4 * np.pi / 3)]
    img = np.stack([np.tile(c, (height, 1)) for c in chans], axis=-1)
    return img[None]  # (1, H, W, 3)


def _colors(width: int, height: int) -> np.ndarray:
    """Three full-frame primaries in sequence: red, then green, then blue."""
    frames = np.zeros((3, height, width, 3), dtype=np.float64)
    for i in range(3):
        frames[i, ..., i] = 1.0
    return frames  # (3, H, W, 3)


COLOR_BUILDERS = {"rainbow": _rainbow, "rgb_phase": _rgb_phase, "colors": _colors}


# --------------------------------------------------------------------------
# Monochrome single-shot spatial codes
# --------------------------------------------------------------------------
def _marray(width: int, height: int, cell: int = 4, win: int = 5,
            seed: int = 0) -> np.ndarray:
    """Single binary frame where every ``win``x``win``-cell window is unique.

    An M-array in the practical sense: a random binary grid of ``cell``-px
    cells, then iterative repair — hash every sliding window, flip a random
    cell inside each colliding window, repeat until **zero** duplicates. Same
    window-uniqueness guarantee as the algebraic perfect-map constructions but
    for arbitrary sizes. The pattern is reproducible from ``seed``.

    Defaults: 4 px cells -> 480x270 grid; 5x5-cell windows -> 2^25 code space
    for ~125k windows (a 20x20 px spatial decoding footprint).
    """
    from numpy.lib.stride_tricks import sliding_window_view

    gw, gh = width // cell, height // cell
    rng = np.random.default_rng(seed)
    g = rng.integers(0, 2, (gh, gw), dtype=np.uint8)
    weights = (1 << np.arange(win * win, dtype=np.int64)).reshape(win, win)
    for it in range(200):
        codes = np.einsum("ijkl,kl->ij",
                          sliding_window_view(g, (win, win)).astype(np.int64),
                          weights)
        flat = codes.ravel()
        order = np.argsort(flat, kind="stable")
        dup = np.zeros(flat.size, bool)
        dup[order] = np.concatenate([[False], flat[order][1:] == flat[order][:-1]])
        if not dup.any():
            break
        for y, x in zip(*np.divmod(np.flatnonzero(dup), codes.shape[1])):
            dy, dx = rng.integers(0, win, 2)
            g[y + dy, x + dx] ^= 1
    else:
        raise RuntimeError("marray window-uniqueness repair did not converge")
    print(f"  marray: {gw}x{gh} cells of {cell}px, {win}x{win} windows unique "
          f"({codes.size} windows / 2^{win * win} codes, {it} repair rounds)")
    img = np.zeros((height, width))
    up = np.kron(g, np.ones((cell, cell), np.uint8)).astype(float)
    img[:up.shape[0], :up.shape[1]] = up[:height, :width]
    return img[None]  # (1, H, W)


def _speckle(width: int, height: int, sigma_px: float = 1.5,
             seed: int = 0) -> np.ndarray:
    """Single band-limited noise frame (DIC-style speckle) for NCC matching.

    White noise low-pass filtered (Gaussian, in Fourier space) to a feature
    size of ~``sigma_px`` pixels, then percentile-stretched to [0, 1] — the
    correlation-matching complement to the discrete ``marray`` code.
    """
    rng = np.random.default_rng(seed)
    n = rng.normal(size=(height, width))
    fy = np.fft.fftfreq(height)[:, None]
    fx = np.fft.rfftfreq(width)[None, :]
    lp = np.exp(-2 * (np.pi * sigma_px) ** 2 * (fx ** 2 + fy ** 2))
    img = np.fft.irfft2(np.fft.rfft2(n) * lp, s=(height, width))
    lo, hi = np.percentile(img, [1, 99])
    return np.clip((img - lo) / (hi - lo), 0, 1)[None]  # (1, H, W)


MONO_BUILDERS = {"marray": _marray, "speckle": _speckle}


# --------------------------------------------------------------------------
# Horizontal Gray code: the same method, transposed -> encodes projector ROW.
# Paired with the vertical "graycode" set it yields an exact per-pixel
# (column, row) correspondence (decode_columns + decode_rows).
# --------------------------------------------------------------------------
def _graycode_h(width: int, height: int) -> np.ndarray:
    return build_method("graycode").patterns(width, height, axis="y")


AXIS_BUILDERS = {"graycode_h": _graycode_h}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--methods", nargs="+",
                    default=["graycode", "graycode_h", "phaseshift", "rainbow",
                             "rgb_phase", "colors", "marray", "speckle"],
                    help=f"strategies to materialise; available: "
                         f"{sorted(REGISTRY)} + {sorted(COLOR_BUILDERS)} + "
                         f"{sorted(MONO_BUILDERS)} + {sorted(AXIS_BUILDERS)}")
    ap.add_argument("--width", type=int, default=1920, help="pattern width (projector columns)")
    ap.add_argument("--height", type=int, default=1080, help="pattern height (projector rows)")
    ap.add_argument("--out", default="patterns", help="root folder for the pattern sets")
    args = ap.parse_args()

    for name in args.methods:
        if name in COLOR_BUILDERS:
            pats, kind = COLOR_BUILDERS[name](args.width, args.height), "rgb "
        elif name in MONO_BUILDERS:
            pats, kind = MONO_BUILDERS[name](args.width, args.height), "gray"
        elif name in AXIS_BUILDERS:
            pats, kind = AXIS_BUILDERS[name](args.width, args.height), "gray"
        else:
            pats, kind = build_method(name).patterns(args.width, args.height), "gray"
        sdir = io.ensure_dir(os.path.join(args.out, name))
        io.save_image_stack(sdir, pats, prefix="pat")
        print(f"{name:11s} {len(pats):2d} frames {args.width}x{args.height} {kind} -> ./{sdir}/pat_*.png")


if __name__ == "__main__":
    main()
