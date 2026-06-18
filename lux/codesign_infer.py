"""Inference + downstream rewiring for the continuous-phase (quadrature) decoder.

The bin model decoded via argmax-bin + softmax-max confidence; the quad model decodes via the CRT
consensus vote (:func:`lux.codesign_vote.vote_fast`) with **peak-margin** confidence. This module
is the drop-in replacement the rest of the pipeline (hemisphere eval, capture viewer, abstention)
consumes, so a continuous-phase checkpoint runs through the same flow as a bin checkpoint.

A conv quad model is shift-invariant, so full-frame ``predict_quad`` is in-distribution (none of the
attention token-scaling that forced tiling for net2); tiling is offered only for very large frames.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from lux.codesign_vote import vote_fast


def load_quad(path: str, device: str = "cpu"):
    """Load a co-design checkpoint -> (model, gen, proj_wh, periods_u, periods_v)."""
    from lux import codesign as cd
    from lux.proj_net import ProjUNet
    ck = torch.load(path, map_location=device, weights_only=False)
    pu, pv = ck["meta"]["u_periods"], ck["meta"]["v_periods"]
    gen = cd.PatternGenerator.ladder(tuple(ck["proj_wh"]), pu, pv)
    gen.load_state_dict(ck["gen_state"])
    model = ProjUNet(base=32, head="quad", n_cu=ck["n_cu"], n_cv=ck["n_cv"]).to(device)
    model.load_state_dict(ck["state"])
    model.eval()
    return model, gen.to(device), tuple(ck["proj_wh"]), pu, pv


@torch.no_grad()
def predict_quad(model, img: np.ndarray, proj_wh, periods_u, periods_v, device: str = "cpu",
                 return_conf: bool = False):
    """Full-frame inference: capture (H,W) in [0,1] -> (H,W,2) projector px, NaN where the validity
    head abstains. ``return_conf`` also returns the **peak-margin** confidence (H,W) =
    ``min(margin_u, margin_v)`` from the vote -- the calibrated abstention signal that replaces
    softmax-max (use a threshold to trade coverage for outlier purity)."""
    nu, nv = len(periods_u), len(periods_v)
    H, W = img.shape
    ph, pw = (-H) % 16, (-W) % 16
    x = F.pad(torch.from_numpy(img.astype(np.float32))[None, None], (0, pw, 0, ph), mode="reflect").to(device)
    pred = model(x)[0, :, :H, :W]
    cu, su = pred[0:2 * nu:2], pred[1:2 * nu:2]
    cv, sv = pred[2 * nu:2 * nu + 2 * nv:2], pred[2 * nu + 1:2 * nu + 2 * nv:2]
    psi_u = torch.atan2(su, cu).permute(1, 2, 0)
    mag_u = torch.sqrt(cu ** 2 + su ** 2).permute(1, 2, 0)
    psi_v = torch.atan2(sv, cv).permute(1, 2, 0)
    mag_v = torch.sqrt(cv ** 2 + sv ** 2).permute(1, 2, 0)
    u, mu = vote_fast(psi_u, mag_u, periods_u, proj_wh[0])
    v, mv = vote_fast(psi_v, mag_v, periods_v, proj_wh[1])
    valid = pred[-1] > 0.0
    uv = torch.stack([u, v], -1)
    uv = torch.where(valid[..., None], uv, torch.full_like(uv, float("nan"))).cpu().numpy()
    if return_conf:
        conf = torch.where(valid, torch.minimum(mu, mv), torch.zeros_like(mu)).cpu().numpy()
        return uv, conf
    return uv
