#!/usr/bin/env python3
"""Carrier-set gate for the coprime hierarchical pattern (review items 2 + 3).

THE gate that must pass before committing a carrier set or running any training:

  1. **Intermod + window-separability** (cheap, analytic): for the chosen coprime integer
     periods, enumerate f_i ± f_j and 2f_i and confirm none collides with another carrier
     within a demodulation bandwidth; and check that spectrally-close carriers (the balanced
     core) are actually *resolvable* in the decoder's receptive-field-sized window.
  2. **SNR-capacity sim** (numerical): superpose the K carriers in 8-bit, push through the
     optics-anchored degradation (camera pixel-footprint area-integration that scales with
     grazing compression + defocus PSF + shot/read noise + quantization), and measure
     per-carrier recovered amplitude and phase error vs K and vs obliquity. This decides how
     many mono carriers fit, which carriers die at grazing ("fine dies, coarse survives"),
     and whether RGB multiplexing is needed.

Per the review: a too-gentle proxy blur makes carriers look artificially separable and
overestimates K, so the degradation here is anchored in optics (footprint + the rig's
defocus_px), NOT the toy isotropic augment.

    python scripts/codesign_carrier_gate.py
"""

from __future__ import annotations

import numpy as np


# Candidate coprime period sets (px). Balanced core: two primes near sqrt(extent) whose
# product >= extent give a unique global range via CRT while staying *fine* (neither near DC,
# so no lighting collision); plus coprime flank(s) for sub-pixel precision.
PERIODS_U = [13, 41, 47]   # core {41,47}: 41*47=1927 >= 1920 (unique over the frame); 13 = fine flank
PERIODS_V = [11, 31, 37]   # core {31,37}: 31*37=1147 >= 1080; 11 = fine flank
W, H = 1920, 1080


def intermod_report(periods, extent, window_px, bw_factor=1.0):
    """Frequencies in cycles-across-extent; bandwidth = bw_factor * (extent/window) cycles
    (a window of `window_px` resolves frequency to ~extent/window cycles)."""
    fs = [extent / p for p in periods]
    bw = bw_factor * extent / window_px
    print(f"  periods {periods}  -> freqs (cyc/frame) {[round(f, 1) for f in fs]}")
    print(f"  demod bandwidth (window {window_px}px): {bw:.1f} cycles")
    collisions, unresolved = [], []
    for i in range(len(fs)):
        for k in range(len(fs)):
            if k != i and abs(2 * fs[i] - fs[k]) < bw:
                collisions.append(f"2f[{periods[i]}] hits f[{periods[k]}]")
        for j in range(i + 1, len(fs)):
            sep = abs(fs[i] - fs[j])
            if sep < bw:
                unresolved.append(f"f[{periods[i]}]~f[{periods[j]}] sep {sep:.1f}<{bw:.1f} cyc")
            for k in range(len(fs)):
                if k not in (i, j):
                    if abs(abs(fs[i] - fs[j]) - fs[k]) < bw:
                        collisions.append(f"f[{periods[i]}]-f[{periods[j]}] hits f[{periods[k]}]")
                    if abs(fs[i] + fs[j] - fs[k]) < bw:
                        collisions.append(f"f[{periods[i]}]+f[{periods[j]}] hits f[{periods[k]}]")
    crt = int(np.prod(periods))
    print(f"  CRT unique range (LCM of coprime periods): {crt}px {'>=' if crt >= extent else '<'} {extent} "
          f"-> {'covers frame' if crt >= extent else 'DOES NOT cover frame'}")
    print(f"  intermod collisions: {collisions or 'none'}")
    print(f"  close-carrier resolvability in {window_px}px window: "
          f"{unresolved or 'all separable'}")
    return fs, bw


def _area_sample(profile_fn, p_centers, c, sub=8):
    """Area-integrate the pattern over each camera pixel's projector footprint of width c
    (the anti-aliasing a point sample misses; footprint grows with grazing compression)."""
    offs = (np.arange(sub) + 0.5) / sub - 0.5
    samples = profile_fn(p_centers[:, None] + c * offs[None, :])   # (N, sub)
    return samples.mean(1)


def _gauss1d(x, sigma):
    r = max(1, int(round(3 * sigma)))
    k = np.exp(-np.arange(-r, r + 1) ** 2 / (2 * sigma ** 2)); k /= k.sum()
    return np.convolve(np.pad(x, r, mode="reflect"), k, "valid")


def snr_sim(periods, extent, obliq_deg, S=256, defocus_px=1.0, full_well=200.0,
            read=0.01, rng=None):
    """Demodulate K superposed carriers from one optics-degraded camera line at a given
    obliquity. Returns per-carrier (recovered amplitude, phase error in projector px)."""
    rng = rng or np.random.default_rng(0)
    c = 1.0 / np.cos(np.deg2rad(obliq_deg))             # projector px per camera px (compression)
    amps = np.full(len(periods), 1.0 / len(periods) * 3.0)
    phases = rng.uniform(0, 2 * np.pi, len(periods))
    bias = 0.0

    def profile(p):                                     # the generated pattern, sigmoid of carriers
        s = bias + sum(a * np.sin(2 * np.pi * p / per + ph)
                       for a, per, ph in zip(amps, periods, phases))
        return 1.0 / (1.0 + np.exp(-s))

    x0 = rng.uniform(0, extent - S * c) if extent > S * c else 0.0
    p_centers = x0 + np.arange(S) * c                   # projector position of each camera px
    cap = _area_sample(profile, p_centers, c)           # footprint area-integration
    cap = _gauss1d(cap, defocus_px)                     # camera/projector defocus PSF (camera px)
    cap = cap + rng.standard_normal(S) * np.sqrt(np.clip(cap, 0, None) / full_well)  # shot
    cap = cap + rng.standard_normal(S) * read           # read
    cap = np.round(np.clip(cap, 0, 1) * 255) / 255      # 8-bit

    # least-squares demod: fit [cos, sin] of every carrier at its known (compressed) phase
    cols = [np.ones(S)]
    for per in periods:
        ph = 2 * np.pi * p_centers / per
        cols += [np.cos(ph), np.sin(ph)]
    A = np.stack(cols, 1)
    coef, *_ = np.linalg.lstsq(A, cap, rcond=None)
    out = []
    for i, (per, ph_true) in enumerate(zip(periods, phases)):
        cc, ss = coef[1 + 2 * i], coef[2 + 2 * i]
        rec_amp = np.hypot(cc, ss)
        rec_ph = np.arctan2(ss, cc)                     # cap = ...+amp*sin(theta+ph)= a*cos*sin? see fit
        # fit uses cos,sin basis: model = cc*cos + ss*sin = rec_amp*sin(theta+phi'), phi'=atan2(cc,ss)
        rec_ph = np.arctan2(cc, ss)
        derr = np.angle(np.exp(1j * (rec_ph - ph_true)))
        px = abs(derr) / (2 * np.pi) * per
        out.append((rec_amp, px))
    return out


def main():
    print("=" * 78)
    print("1. INTERMOD + WINDOW-SEPARABILITY (RF window ~256-300px)")
    print("-" * 78)
    print(" u-axis:")
    intermod_report(PERIODS_U, W, window_px=256)
    print(" v-axis:")
    intermod_report(PERIODS_V, H, window_px=256)

    print("\n" + "=" * 78)
    print("2. SNR-CAPACITY SIM (optics-anchored: footprint + defocus + shot/read + 8-bit)")
    print("-" * 78)
    rng = np.random.default_rng(0)
    bands = [0, 30, 50, 60, 70, 75]
    print(f"  u-axis carriers {PERIODS_U} (mono, all superposed)")
    print(f"  {'obliq':>6} | " + " | ".join(f"p={p:<3d} amp / err_px" for p in PERIODS_U))
    for th in bands:
        rows = [snr_sim(PERIODS_U, W, th, rng=rng) for _ in range(40)]
        agg = np.array(rows)                            # (40, K, 2)
        cells = []
        for i in range(len(PERIODS_U)):
            amp = agg[:, i, 0].mean(); err = np.median(agg[:, i, 1])
            cells.append(f"{amp:4.2f} / {err:6.2f}")
        print(f"  {th:4d}deg | " + " | ".join(cells))
    print("\n  Read: amp -> carrier survival (collapses = band dead); err_px -> usable precision.")
    print("  'fine dies, coarse survives' = small-period amp/precision degrade fastest with obliquity.")


if __name__ == "__main__":
    main()
