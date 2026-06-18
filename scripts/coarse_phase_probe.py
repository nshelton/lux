#!/usr/bin/env python3
"""Disambiguate the coarse-carrier (p=139) zero-shot unwrap failure: is it a FIXABLE DC/bias/white-ref
mismatch in the appearance-fix proxy, or the planar render-cap (shading the planar proxy can't model)?

On a single near-frontal rendered plane, demodulate the coarse u-carrier phase three ways and compare:
  A) IDEAL    — the authored pattern sampled at gt_proj (no shading/nonlinearity): the reference phase.
  B) RENDER   — the actual rendered capture (renderer's albedo·(ambient + drive·pattern) + 8-bit + clip).
  C) PROXY    — the pattern at gt_proj put through MY proxy appearance (falloff·shading + augment).

Δ_render = wrap(ψ_B − ψ_A), Δ_proxy = wrap(ψ_C − ψ_A) isolate each pipeline's appearance-induced
coarse-phase shift (the intrinsic carrier phase φ_c cancels — same pattern).
  • Δ_render ≈ Δ_proxy                        -> proxy faithful for the coarse carrier; not an appearance
                                                 bug (unwrap failure is decoder/training -> render-train).
  • Δ_proxy − Δ_render ≈ constant (low std)   -> systematic DC/bias mismatch = a BUG to fix in the proxy.
  • Δ_proxy − Δ_render correlates w/ position -> shading-gradient driven = the planar render-cap.
"""
from __future__ import annotations
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
import torch.nn.functional as F
from lux import io, codesign as cd
from scripts.codesign_demod_probe import local_phase

W, H, P = 1920, 1080, 139
DATA = Path("evals/hemisphere/data_v2_val")
PAT = "patterns/codesign_v2/pat_00.png"


def pick_frontal():
    best, bo = None, 1e9
    for d in sorted(DATA.glob("sample_*")):
        j = d / "sample.json"
        if not j.exists():
            continue
        m = json.loads(j.read_text())
        ob = max(m.get("theta_cam_deg", 90), m.get("theta_proj_deg", 90))
        if ob < bo and (d / "codesign_v2" / "cap_pat_00.png").exists():
            bo, best = ob, d
    return best, bo


def biggest_valid_crop(valid, S=384):
    """Find an SxS fully-valid (all-True) window; return (y,x) or None."""
    H, Wd = valid.shape
    iv = np.cumsum(np.cumsum(valid.astype(np.int32), 0), 1)
    def winsum(y, x):
        a = iv[y + S - 1, x + S - 1]
        b = iv[y - 1, x + S - 1] if y else 0
        c = iv[y + S - 1, x - 1] if x else 0
        e = iv[y - 1, x - 1] if (y and x) else 0
        return a - b - c + e
    for y in range(H // 2 - S, H // 2 + 1, 32):
        for x in range(Wd // 2 - S, Wd // 2 + 1, 32):
            if 0 <= y < H - S and 0 <= x < Wd - S and winsum(y, x) == S * S:
                return y, x
    return None


def main():
    d, ob = pick_frontal()
    print(f"frontal plane: {d.name}  max-obliq {ob:.1f} deg")
    cap = io.load_image(str(d / "codesign_v2" / "cap_pat_00.png"), gray=True).astype(np.float32)
    gt = np.load(d / "gt_proj.npy").astype(np.float32)
    gu = gt[..., 0]
    valid = np.isfinite(gu)

    # coarse-carrier lock-in drops a ~193px border each side, so need S well over 2*193
    S, yx = None, None
    for cand in (832, 768, 704, 640):
        yx = biggest_valid_crop(valid, S=cand)
        if yx is not None:
            S = cand
            break
    if yx is None:
        raise SystemExit("no fully-valid crop >=640 on this plane")
    y, x = yx
    print(f"valid crop {S}x{S} at ({y},{x}) -> interior {S - 2*193}px after lock-in border")
    sl = (slice(y, y + S), slice(x, x + S))
    capc = cap[sl]
    guc = gu[sl]                                  # projector col per pixel (smooth on a plane)

    # A) ideal: pattern sampled at gt_proj (normalized) via bilinear grid_sample
    pat = io.load_image(PAT, gray=True).astype(np.float32)
    coords = np.stack([gu[sl] / W, gt[..., 1][sl] / H], -1)
    grid = torch.tensor(coords * 2 - 1, dtype=torch.float32)[None]
    pat_t = torch.tensor(pat, dtype=torch.float32)[None, None]
    ideal = F.grid_sample(pat_t, grid, mode="bilinear", padding_mode="reflection",
                          align_corners=False)[0, 0].numpy()

    # C) proxy appearance on the SAME ideal pattern field: falloff*shading then the photometric augment.
    rng = np.random.default_rng(0)
    th = np.deg2rad(ob)
    falloff = max(np.cos(th), 0.12)
    ys = np.linspace(-1, 1, S); xs = np.linspace(-1, 1, S)
    gyy, gxx = np.meshgrid(ys, xs, indexing="ij")
    shade = float(rng.uniform(0.45, 1.0)) * (1 + float(rng.uniform(-.35, .35)) * gxx
                                             + float(rng.uniform(-.35, .35)) * gyy)
    shade = np.clip(shade, 0.15, 1.0)
    proxy_pre = ideal * falloff * shade
    proxy_aug = cd.differentiable_augment(torch.tensor(proxy_pre, dtype=torch.float32)[None, None],
                                          np.random.default_rng(0), base_psf=0.7)[0, 0].numpy()

    # demod coarse phase from each (lock-in against the known correspondence 2pi*gu/P)
    pa = local_phase(ideal, guc, P)
    pb = local_phase(capc, guc, P)
    pc = local_phase(proxy_aug, guc, P)

    def wrap(a):
        return np.angle(np.exp(1j * a))
    dR = wrap(pb - pa)
    dP = wrap(pc - pa)
    diff = wrap(dP - dR)

    print(f"\ncoarse carrier p={P} phase shift vs IDEAL (authored pattern), on {S}x{S} valid crop:")
    print(f"  Δ_render  mean {np.mean(dR):+.4f} rad  std {np.std(dR):.4f}")
    print(f"  Δ_proxy   mean {np.mean(dP):+.4f} rad  std {np.std(dP):.4f}")
    print(f"  (proxy − render) mean {np.mean(diff):+.4f} rad  std {np.std(diff):.4f}")
    # is the (proxy-render) discrepancy a constant offset or position-correlated?
    yy, xx = np.mgrid[0:diff.shape[0], 0:diff.shape[1]]
    for nm, c in [("x", xx.ravel()), ("y", yy.ravel()), ("gt_u", guc[ :diff.shape[0], :diff.shape[1]].ravel())]:
        r = np.corrcoef(c, diff.ravel())[0, 1]
        print(f"    corr(proxy−render, {nm}) = {r:+.3f}")
    const = np.abs(np.mean(diff)); spread = np.std(diff)
    print(f"\n  READ: |mean| {const:.3f} vs std {spread:.3f} rad. "
          + ("CONSTANT-dominated -> DC/bias bug (fixable)" if const > spread
             else "SPREAD-dominated -> check position corr above (shading=render-cap, else noise)"))
    print(f"  (a coarse-period of {P}px maps 2π; {const:.3f} rad ≈ {const/(2*np.pi)*P:.1f}px of global u offset)")


if __name__ == "__main__":
    main()
