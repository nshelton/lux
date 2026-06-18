#!/usr/bin/env python3
"""Validate the analytic anisotropic-Gaussian MTF (the training blur) against the
footprint-supersample ORACLE (the expensive, knob-free physics). Per-carrier amplitude
attenuation vs obliquity; geometric-only (sigma_opt=0) so the two should match by the
box->Gaussian moment match, with possible divergence at 75deg+ (long thin parallelogram tails).

    python scripts/validate_blur_oracle.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from lux import codesign as cd

W, H = 1920, 1080
PERIODS = [13, 19, 33, 139]          # u-axis carriers (theta=0)
S = 256


def fixed_gen():
    g = cd.PatternGenerator((W, H), n_carriers=len(PERIODS))
    with torch.no_grad():
        g.log_freq.copy_(torch.log(torch.tensor([W / p for p in PERIODS], dtype=torch.float32)))
        g.theta.copy_(torch.zeros(len(PERIODS)))              # all along projector-x (u)
        g.phase.copy_(torch.zeros(len(PERIODS)))
        g.log_amp.copy_(torch.log(torch.full((len(PERIODS),), 0.5)))
        g.bias.zero_()
    return g


def oracle_atten(gen, Hinv, ss=8):
    """Footprint-supersample capture -> demod per-carrier amplitude / nominal amp = atten."""
    coords1, _ = cd.grid_from_homography(Hinv, (S, S), "cpu", ss=1)
    coordsS, _ = cd.grid_from_homography(Hinv, (S, S), "cpu", ss=ss)
    cap = torch.nn.functional.avg_pool2d(gen.sample_at(coordsS[None]), ss)[0, 0].detach().numpy()
    u = (coords1[..., 0].numpy() * W).reshape(-1)
    cols = [np.ones_like(u)]
    for p in PERIODS:
        cols += [np.cos(2 * np.pi * u / p), np.sin(2 * np.pi * u / p)]
    A = np.stack(cols, 1)
    coef, *_ = np.linalg.lstsq(A, cap.reshape(-1), rcond=None)
    amp_nom = float(torch.exp(gen.log_amp[0]))                # 0.5
    return [np.hypot(coef[1 + 2 * i], coef[2 + 2 * i]) / amp_nom for i in range(len(PERIODS))]


def analytic_atten(gen, Hinv):
    coords1, _ = cd.grid_from_homography(Hinv, (S, S), "cpu", ss=1)
    fx, fy = gen.freqs()
    at = gen._mtf_atten(coords1[None], fx, fy, (0.0, 0.0))[0]  # (S,S,K), geometric only
    return [float(at[..., i].median()) for i in range(len(PERIODS))]


def main():
    rng = np.random.default_rng(0)
    gen = fixed_gen()
    print(f"carriers (periods px) {PERIODS}; GEOMETRIC-only MTF (sigma_opt=0): analytic vs "
          f"footprint-ss oracle (ss=8).")
    print(f"  attenuation normalized to frontal (removes the obliquity-independent sigmoid factor):")
    print(f"  {'obliq':>6} | " + " | ".join(f"p={p:<4d}[an|orc]" for p in PERIODS))
    rows = {}
    for th in [0, 30, 50, 65, 75]:
        an, orc = [], []
        for _ in range(12):
            Hinv, _ = cd.sample_homography_inv(rng, (S, S), (th, th + 0.01))
            an.append(analytic_atten(gen, Hinv))
            orc.append(oracle_atten(gen, Hinv))
        rows[th] = (np.mean(an, 0), np.mean(orc, 0))
    an0, orc0 = rows[0]
    for th, (an, orc) in rows.items():
        cells = [f"{an[i]/an0[i]:.2f}|{orc[i]/orc0[i]:.2f}" for i in range(len(PERIODS))]
        print(f"  {th:4d}deg | " + " | ".join(f"{c:>11}" for c in cells))
    print("\n  Match => analytic Gaussian reproduces the footprint oracle. NOTE the geometric footprint")
    print("  is a WEAK lever here (p=13 only ~0.92 at 75deg): footprint << carrier periods. The grazing")
    print("  kill comes from OPTICAL variance (sigma_def/sigma_mtf, added analytically) + the radiometric")
    print("  SNR (grazing falloff feeding shot noise) -- not the geometric area-average.")


if __name__ == "__main__":
    main()
