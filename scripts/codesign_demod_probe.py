#!/usr/bin/env python3
"""Build 3: fixed-pattern demod probe (capacity gate). Render the FROZEN coprime carrier pattern
through the faithful proxy (validated analytic anisotropic-Gaussian blur + grazing falloff + shot/
read noise + 8-bit) and measure per-carrier per-pixel phase NOISE sigma_phi vs obliquity vs carrier
count K. No generator loop, no vote tuning -- isolates "is the representation demodulable to the
accuracy the vote needs" from "can co-design improve it."

LOCAL demod (not a global-crop lstsq): each carrier is read by a windowed lock-in over its NATURAL
scale (~3x its period) -- the dense per-pixel estimate a real decoder must produce, with realistic
(not 256x-global) noise averaging and genuine cross-carrier leakage. This is the version that can
FAIL: fine carriers (small window, dim grazing signal) blow the bar first (fine dies), coarse hold
(coarse survives); close carriers leak into each other's lock-in (separability).

Bar: sigma_phi < ~0.2 rad (margin below the vote's 0.3 rad breakdown; real errors are correlated).
A carrier over the bar at an obliquity is dead there -> its floating magnitude zeros its vote.

    python scripts/codesign_demod_probe.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.nn.functional as F

from lux import codesign as cd

W, H = 1920, 1080
S = 512                                   # room for coarse-carrier windows + interior
SS = 4                                    # footprint-integration supersample (matches co-design ss)
SIGMA_DEF, SIGMA_MTF = 1.0, 0.7           # rig-anchored: projector defocus_px, camera MTF
FULL_WELL, READ = 200.0, 0.01
GRAZE_FLOOR = 0.12
APPEARANCE = "raster"                     # "raster" (appearance-fixed) | "analytic" (old proxy)


def gen_for(periods, total_drive=1.5):
    g = cd.PatternGenerator((W, H), n_carriers=len(periods))
    with torch.no_grad():
        g.log_freq.copy_(torch.log(torch.tensor([W / p for p in periods], dtype=torch.float32)))
        g.theta.zero_(); g.phase.zero_()
        g.log_amp.copy_(torch.log(torch.full((len(periods),), total_drive / len(periods))))
        g.bias.zero_()
    return g


def render(gen, Hinv, th_deg, rng):
    """Render one S×S grazing patch through the proxy and return (capture, gt_u_px).

    APPEARANCE='raster' is the appearance-fixed path: the carriers are materialized at projector
    resolution, **8-bit quantized** (the authored PNG), projector-defocus-blurred, then read through
    the warp by bilinear ``grid_sample`` (footprint-supersampled + avg-pooled) and camera-MTF-blurred
    -- the quantized/resampled signal the renderer actually produces. 'analytic' is the pre-fix proxy
    (continuous carriers evaluated directly, MTF as a per-carrier attenuation) for an A/B."""
    cg1, _ = cd.grid_from_homography(Hinv, (S, S), "cpu", ss=1)            # GT coords (pixel centres)
    if APPEARANCE == "raster":
        cgS, _ = cd.grid_from_homography(Hinv, (S, S), "cpu", ss=SS)       # supersampled footprint
        raster_at = cd.raster_appearance(gen, sigma_def=SIGMA_DEF, quantize=True)
        pat = raster_at(cgS[None])                                         # (1,1,SS*S,SS*S) quantized
        pat = F.avg_pool2d(pat, SS)                                        # footprint area integral
        pat = cd._fixed_gauss_blur(pat, SIGMA_MTF)                         # camera MTF
        pat = pat[0, 0].detach().numpy()
    else:
        pat = gen.sample_at(cg1[None], mtf=(SIGMA_MTF, SIGMA_DEF))[0, 0].detach().numpy()
    falloff = max(np.cos(np.deg2rad(th_deg)), GRAZE_FLOOR)
    sig = pat * falloff * rng.uniform(0.55, 1.0)
    sig = sig + rng.standard_normal(sig.shape) * np.sqrt(np.clip(sig, 0, None) / FULL_WELL)
    sig = sig + rng.standard_normal(sig.shape) * READ
    sig = np.round(np.clip(sig, 0, 1) * 255) / 255
    return sig, cg1[..., 0].numpy() * W


def _boxblur(x, win):                      # separable uniform filter (fast, O(N*win))
    r = win // 2
    kx = torch.ones(1, 1, win, 1) / win
    ky = torch.ones(1, 1, 1, win) / win
    x = F.conv2d(F.pad(x, (0, 0, r, r), mode="reflect"), kx)
    x = F.conv2d(F.pad(x, (r, r, 0, 0), mode="reflect"), ky)
    return x


def local_phase(cap, u, period):
    """Per-pixel lock-in phase for one carrier over a ~3x-period window. Returns the interior
    phase field (radians), padding-contaminated border removed."""
    win = int(np.clip(3 * period, 49, 193)) | 1               # odd
    xi = 2 * np.pi * u / period
    c = torch.tensor(np.cos(xi), dtype=torch.float32)[None, None]
    s = torch.tensor(np.sin(xi), dtype=torch.float32)[None, None]
    cap_t = torch.tensor(cap, dtype=torch.float32)[None, None]
    I = _boxblur(cap_t * c, win)[0, 0]
    Q = _boxblur(cap_t * s, win)[0, 0]
    ph = torch.atan2(I, Q).numpy()
    m = win                                                   # drop the window-contaminated border
    return ph[m:-m, m:-m]


def sigma_phi(periods, th, rng, n_homog=2, n_real=6):
    """Per-carrier per-pixel phase noise: circular std across noise realizations (fixed geometry),
    meaned over the interior, averaged over homographies."""
    out = []
    for _ in range(n_homog):
        Hinv, _ = cd.sample_homography_inv(rng, (S, S), (th, th + 0.01))
        per_real = [[local_phase(*render(gen_for(periods), Hinv, th, rng), p) for p in periods]
                    for _ in range(n_real)]                   # [real][carrier] -> phase field
        carr = []
        for k in range(len(periods)):
            stack = np.stack([per_real[r][k] for r in range(n_real)])     # (R, h, w)
            mean = np.angle(np.exp(1j * stack).mean(0))
            cstd = np.angle(np.exp(1j * (stack - mean))).std(0)           # per-pixel circular std
            carr.append(float(np.mean(cstd)))
        out.append(carr)
    return np.mean(out, 0)


def intermod_check(periods, total_drive=1.5, bias=0.0):
    """Re-run the intermod budget on the **quantized rendered spectrum** (the §13-14 check): 8-bit
    quantization is a static nonlinearity that injects harmonic (2f/3f) + intermod (f_i±f_j) energy.
    FFT a row of the quantized authored pattern and report, per carrier, its line energy and the
    worst harmonic/intermod line as a fraction of the weakest carrier -- and flag any harmonic that
    lands within ±1 bin of a carrier (a collision would corrupt that carrier's phase read)."""
    g = gen_for(periods, total_drive)
    with torch.no_grad():
        g.bias.fill_(bias)
    row = g.render_proj((W, H), quantize=True)[0, 0, H // 2].detach().numpy()
    spec = np.abs(np.fft.rfft(row - row.mean()))
    spec = spec / (spec.max() + 1e-12)
    nyq = len(spec) - 1
    cbin = {p: int(round(W / p)) for p in periods}
    carrier_e = {p: float(spec[cbin[p]]) for p in periods}
    cmin = min(carrier_e.values())
    lines = []
    for p in periods:
        lines += [(f"2f·p{p}", 2 * W / p), (f"3f·p{p}", 3 * W / p)]
    for i, pi in enumerate(periods):
        for pj in periods[i + 1:]:
            lines += [(f"f{pi}+f{pj}", W / pi + W / pj), (f"|f{pi}-f{pj}|", abs(W / pi - W / pj))]
    worst, worst_name, collisions = 0.0, "", []
    for name, fb in lines:
        b = int(round(fb))
        if b <= 0 or b >= nyq:
            continue
        e = float(spec[b])
        if e > worst:
            worst, worst_name = e, name
        for p in periods:
            if abs(b - cbin[p]) <= 1:
                collisions.append((name, p, e))
    print(f"\n intermod (quantized spectrum, bias {bias:+.2f}, amp {total_drive/len(periods):.2f}/carrier): "
          f"carrier lines " + " ".join(f"p{p}:{carrier_e[p]:.3f}" for p in periods))
    print(f"   worst harmonic/intermod line: {worst_name} = {worst:.4f}  "
          f"({worst/cmin*100:.1f}% of weakest carrier {cmin:.3f})")
    if collisions:
        print("   !! COLLISION: harmonic within ±1 bin of a carrier -> " +
              "; ".join(f"{n} hits p{p} at {e:.3f}" for n, p, e in collisions))
    else:
        print("   no harmonic/intermod line lands on a carrier (all >±1 bin away). OK.")
    return worst / cmin, collisions


def main():
    rng = np.random.default_rng(0)
    subsets = [[33, 139], [19, 33, 139], [13, 19, 33, 139]]
    bands = [0, 50, 65, 75]
    print(f"APPEARANCE={APPEARANCE}. LOCAL per-pixel demod (window ~3x period). optics sigma_def "
          f"{SIGMA_DEF} sigma_mtf {SIGMA_MTF}, ss {SS}, full_well {FULL_WELL}, graze_floor "
          f"{GRAZE_FLOOR}. Bar sigma_phi < 0.2 rad.")
    for periods in subsets:
        wins = [int(np.clip(3 * p, 49, 193)) | 1 for p in periods]
        print(f"\n carriers {periods} (K={len(periods)}, amp {1.5/len(periods):.2f}, windows {wins}px):")
        print(f"  {'obliq':>6} | " + " | ".join(f"p={p:<3d}" for p in periods))
        for th in bands:
            sig = sigma_phi(periods, th, rng)
            cells = [f"{sig[i]:.3f}{'!' if sig[i] > 0.2 else ' '}" for i in range(len(periods))]
            print(f"  {th:4d}deg | " + " | ".join(f"{c:>6}" for c in cells))
    print("\n  '!' = over the 0.2 bar (dead -> magnitude zeros its vote). Fine carriers should blow")
    print("  the bar first at grazing (fine dies); the coarse core should hold (coarse survives).")

    print("\n=== intermod budget under 8-bit quantization (the appearance-fix re-check) ===")
    for bias in (0.0, 0.4):           # centered + off-center (2f grows with off-center bias, §14)
        intermod_check([13, 19, 33, 139], bias=bias)


if __name__ == "__main__":
    main()
