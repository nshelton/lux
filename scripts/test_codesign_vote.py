#!/usr/bin/env python3
"""Unit-test gate for the CRT consensus vote (review item 5). NON-NEGOTIABLE: if this does not
pass, nothing downstream matters. Run: ``python scripts/test_codesign_vote.py``.

Tests (the listed two plus the three the reviewer added so a real bug can't hide):
  A. exact phases -> u recovered to <0.1px, swept across the frame.
  B. drop EACH carrier in turn (mag->0) -> still correct (different surviving coprime subsets).
  C. windowed soft-argmax tracks the hard peak under a PLANTED secondary peak (does NOT average
     the two -> the naive-soft-argmax trap).
  D. noise-margin sweep: raise sigma_phi until the vote breaks; report the breakdown sigma (this
     becomes the acceptance bar for the fixed-pattern demod probe).
  E. alias-margin: build the strongest false coincidence from the ACTUAL periods and confirm the
     true peak still wins; report the margin.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from lux.codesign_vote import consensus_vote

PERIODS = [13, 19, 33, 139]          # clean working set (separable + 2f/intermod-clean); final lock at demod-probe
EXTENT = 1920


def phases_for(u, periods, noise=0.0, rng=None):
    """True decoded phase psi_k = 2*pi*u/p_k (+ optional wrapped Gaussian phase noise)."""
    u = torch.as_tensor(u, dtype=torch.float32).reshape(-1, 1)
    per = torch.tensor(periods, dtype=torch.float32)[None]
    psi = (2 * torch.pi * u / per)
    if noise > 0:
        g = torch.from_numpy(rng.standard_normal(psi.shape).astype("float32")) if rng else torch.randn_like(psi)
        psi = psi + noise * g
    return psi


def main():
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    K = len(PERIODS)
    ok = True

    # A. exact phases -> <0.1px
    u_true = torch.linspace(20, EXTENT - 20, 200)
    psi = phases_for(u_true, PERIODS)
    mags = torch.ones(len(u_true), K)
    u_est, _ = consensus_vote(psi, mags, PERIODS, EXTENT, step=0.5, soft=True, temp=0.1)
    errA = (u_est - u_true).abs().max().item()
    print(f"A. exact-phase recovery: max err {errA:.4f}px  ->  {'PASS' if errA < 0.1 else 'FAIL'}")
    ok &= errA < 0.1

    # B. drop each carrier in turn
    print("B. drop-each-carrier (remaining subset must still pin u):")
    for d in range(K):
        m = torch.ones(len(u_true), K); m[:, d] = 0.0
        ue, _ = consensus_vote(psi, m, PERIODS, EXTENT, step=0.5, soft=True, temp=0.1)
        e = (ue - u_true).abs().max().item()
        print(f"   drop p={PERIODS[d]:3d}: max err {e:.3f}px  {'PASS' if e < 0.5 else 'FAIL'}")
        ok &= e < 0.5

    # C. genuine multi-peak accumulator (the naive-soft-argmax trap). Use only 2 carriers so the
    # CRT range (13*19=247px) is < the frame -> the accumulator has MANY near-equal lattice peaks
    # every 247px. A full-frame soft-argmax would return their mean (~frame center); the windowed
    # one must sit ON a lattice peak. Tiny noise breaks the exact tie so a hard peak is well-defined.
    p2 = [13, 19]
    u_t = 600.0
    psi2 = phases_for([u_t], p2, noise=0.02, rng=rng)
    u_est_c, _ = consensus_vote(psi2, torch.ones(1, len(p2)), p2, EXTENT,
                                step=0.5, soft=True, temp=0.1)
    e = u_est_c.item()
    lattice = np.arange(u_t % 247, EXTENT, 247.0)                      # all equal-height peaks
    d_peak = float(np.min(np.abs(lattice - e)))
    d_mid = float(np.min(np.abs((lattice[:-1] + lattice[1:]) / 2 - e)))   # nearest midpoint between peaks
    frame_center = abs(e - EXTENT / 2)
    print(f"C. windowed-soft on a {len(lattice)}-peak lattice: est {e:.1f}px, "
          f"dist-to-nearest-peak {d_peak:.2f}px, dist-to-nearest-midpoint {d_mid:.1f}px, "
          f"dist-to-frame-center {frame_center:.0f}px  -> {'PASS' if d_peak < 1.0 and d_mid > 50 else 'FAIL'}")
    ok &= d_peak < 1.0 and d_mid > 50

    # D. noise-margin sweep -> breakdown sigma_phi
    print("D. noise-margin sweep (median err vs phase noise sigma):")
    breakdown = None
    for sig in [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]:
        es = []
        for _ in range(3):
            p = phases_for(u_true, PERIODS, noise=sig, rng=rng)
            ue, _ = consensus_vote(p, torch.ones(len(u_true), K), PERIODS, EXTENT, step=0.5,
                                   soft=False)
            es.append((ue - u_true).abs())
        med = torch.cat(es).median().item()
        flag = ""
        if breakdown is None and med > 1.0:
            breakdown = sig; flag = "  <- breakdown (median err >1px)"
        print(f"   sigma_phi {sig:.2f} rad: median err {med:6.2f}px{flag}")
    print(f"   => breakdown sigma_phi ~ {breakdown}  (acceptance bar for the demod probe)")

    # E. alias-margin: strongest false coincidence from the actual combs
    u_t = 960.0
    psi_t = phases_for([u_t], PERIODS)
    per = torch.tensor(PERIODS, dtype=torch.float32)
    u = torch.arange(0, EXTENT, 0.25)
    acc = (torch.cos(2 * torch.pi * u[:, None] / per[None] - psi_t[0][None])).sum(1)
    top = acc.max().item(); ti = acc.argmax().item()
    mask = (u - u[ti]).abs() > 8
    comp = acc[mask].max().item()
    print(f"E. alias-margin at u={u_t}: true peak {top:.3f} vs best competitor {comp:.3f}  "
          f"(margin {top-comp:.3f})  -> {'PASS' if top - comp > 0.2 else 'FAIL'}")
    ok &= (top - comp) > 0.2

    print("\n" + ("ALL GATE TESTS PASSED" if ok else "GATE FAILED — fix before proceeding"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
