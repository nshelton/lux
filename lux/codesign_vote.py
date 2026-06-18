"""CRT consensus-vote unwrap for the coprime hierarchical phase code.

Build 1 of the hierarchical-pattern leg (`docs/hierarchical_pattern_plan.md`): the piece that
touches neither the proxy nor the network, so it is built and unit-tested *first* — it pins the
interface contract the decoder head must hit (per-carrier decoded phase + a floating
magnitude-confidence) and the accumulator's input format, which is what kills downstream rework.

Each coprime carrier ``k`` of period ``p_k`` constrains the coordinate to
``u ≡ (ψ_k/2π)·p_k  (mod p_k)``, where ``ψ_k`` is the decoded phase (the head's
``atan2(sin_k, cos_k)``, with the carrier's intrinsic offset folded out during training). With
coprime periods the residues pin a unique ``u`` over the frame (CRT). Under noise the residues
disagree, so instead of solving CRT exactly we **accumulate a consensus score**

    acc(u) = Σ_k  m_k · cos(2π u/p_k − ψ_k)

weighted by each carrier's magnitude ``m_k`` (the UNNORMALIZED quadrature vector length — a
carrier that dies at grazing has ``m_k→0`` and simply stops voting). The estimate is the peak.

Two paths (review item 5):
- **hard** (inference): argmax over the ``u`` grid.
- **soft** (training gradient): a **windowed** soft-argmax around the hard peak. The window is the
  whole point — a naive soft-argmax over the full accumulator returns the *mean*, so two near-equal
  peaks yield the catastrophic midpoint, invisible on clean single-peak tests. Softening only
  around the chosen peak tracks it instead of averaging rivals.

Confidence is the **peak margin**: ``(top − best_competitor_outside_window)/top`` — the alias
margin, which is what abstention / fusion downstream should consume (not softmax-max).
"""

from __future__ import annotations

import numpy as np
import torch


def vote_fast(phases: torch.Tensor, mags: torch.Tensor, periods, extent: int):
    """Efficient CRT decode for dense maps: candidates come from the COARSEST carrier (only
    ~extent/max_period of them, ~15 for u) instead of a dense grid, scored against all carriers.
    ``phases`` (..., K), ``mags`` (..., K) -> ``(u (...), margin (...))``. Hard argmax (inference)."""
    dev = phases.device
    per = torch.as_tensor(periods, dtype=torch.float32, device=dev)
    c_idx = int(torch.argmax(per))                              # coarsest carrier
    P0 = float(per[c_idx])
    n = int(np.ceil(extent / P0)) + 1
    base = (phases[..., c_idx] / (2 * np.pi)) % 1.0             # fractional period position
    offs = torch.arange(n, device=dev, dtype=torch.float32)
    cand = (base[..., None] + offs) * P0                        # (..., n) candidate u
    ang = 2 * np.pi * cand[..., None] / per - phases[..., None, :]   # (..., n, K)
    acc = (mags[..., None, :] * torch.cos(ang)).sum(-1)         # (..., n)
    acc = torch.where(cand <= extent, acc, torch.full_like(acc, -1e9))
    best = acc.argmax(-1)
    u = torch.gather(cand, -1, best[..., None]).squeeze(-1)
    top = acc.amax(-1)
    second = torch.where(acc < top[..., None] - 1e-6, acc, torch.full_like(acc, -1e9)).amax(-1)
    margin = (top - second) / top.abs().clamp(min=1e-6)
    return u, margin


def consensus_vote(phases: torch.Tensor, mags: torch.Tensor, periods, extent: int,
                   step: float = 0.5, soft: bool = False, window: float = 4.0,
                   temp: float = 0.15):
    """Vote a 1-D coordinate from per-carrier decoded phases.

    ``phases`` (..., K) decoded ψ_k in radians; ``mags`` (..., K) ≥0 weights; ``periods`` (K,) px;
    ``extent`` the axis length (px). Returns ``(u_est (...), margin (...))``. ``soft=True`` returns
    a differentiable windowed soft-argmax refinement; the window center (hard peak) is detached, so
    gradient flows through the sub-grid refinement only — exactly where it is well-behaved.
    """
    dev = phases.device
    per = torch.as_tensor(periods, dtype=torch.float32, device=dev)        # (K,)
    lead = phases.shape[:-1]
    N = int(torch.tensor(lead).prod()) if lead else 1
    ph = phases.reshape(N, -1)                                             # (N, K)
    mg = mags.reshape(N, -1).clamp(min=0.0)
    u = torch.arange(0, extent, step, device=dev, dtype=torch.float32)     # (G,)
    G = u.shape[0]

    # acc[n, g] = Σ_k m_k cos(2π u_g / p_k − ψ_k)
    ang = 2 * torch.pi * u[None, :, None] / per[None, None, :] - ph[:, None, :]   # (N,G,K)
    acc = (mg[:, None, :] * torch.cos(ang)).sum(-1)                        # (N,G)

    hard_idx = acc.argmax(1)                                              # (N,)
    u_hard = u[hard_idx]

    # peak margin: best competitor outside a ±`window` (in px -> grid steps) guard band
    half = max(1, int(round(window / step)))
    gidx = torch.arange(G, device=dev)
    outside = (gidx[None, :] - hard_idx[:, None]).abs() > half            # (N,G)
    top = acc.gather(1, hard_idx[:, None]).squeeze(1)                     # (N,)
    competitor = torch.where(outside, acc, torch.full_like(acc, -1e9)).max(1).values
    margin = (top - competitor) / top.abs().clamp(min=1e-6)

    if not soft:
        return u_hard.reshape(lead), margin.reshape(lead)

    # windowed soft-argmax: soften ONLY within ±half of the (detached) hard peak
    lo = (hard_idx - half).clamp(0, G - 1)
    offs = torch.arange(2 * half + 1, device=dev)
    win = (lo[:, None] + offs[None, :]).clamp(0, G - 1)                    # (N, 2half+1)
    acc_win = acc.gather(1, win)
    u_win = u[win]
    w = torch.softmax(acc_win / temp, dim=1)
    u_soft = (w * u_win).sum(1)
    return u_soft.reshape(lead), margin.reshape(lead)
