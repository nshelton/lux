"""One-shot structured-light correspondence network.

A U-Net that maps a single M-array capture (``marray/cap_pat_00.png``) to the
dense projector correspondence ``gt_proj.npy``: per camera pixel, the projector
(col, row) subpixel that lit it, plus a validity logit covering everything
``gt_proj`` marks NaN (projector shadow, out-of-frame, no surface).

Why this can work one-shot: the M-array pattern guarantees every local window
is globally unique, so a ~20 px patch *identifies* its projector cell; the
network learns that codebook implicitly and uses its large receptive field to
ride out defocus blur, oblique foreshortening and albedo texture, which a
literal window-decoder cannot.

Coordinates are normalized to [0, 1] by the projector dimensions; losses are
computed in projector-pixel units (Huber) so the numbers read as px error.
Trained on random crops by ``scripts/train_proj_net.py``; full-frame inference
via ``scripts/predict_proj_net.py``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------
class _Block(nn.Module):
    """Two 3x3 convs with GroupNorm + SiLU."""

    def __init__(self, cin: int, cout: int):
        super().__init__()
        g = max(1, cout // 8)
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1), nn.GroupNorm(g, cout), nn.SiLU(),
            nn.Conv2d(cout, cout, 3, padding=1), nn.GroupNorm(g, cout), nn.SiLU(),
        )

    def forward(self, x):
        return self.net(x)


# Classification + offset head: the projector frame is split into coarse bins
# (32 px / 30 px at 1920x1080); the net classifies the bin (softmax, CE loss —
# well-conditioned, the argmax can jump anywhere immediately) and regresses the
# fractional position within it (dynamic range 1 bin instead of the full frame,
# which is where subpixel precision comes from). coord = (bin + frac) * bin_px.
N_BINS_U = 60
N_BINS_V = 36
_HEAD_CH = N_BINS_U + N_BINS_V + 3   # + offset_u, offset_v, validity logit


def _posenc_2d(h: int, w: int, d: int, device) -> torch.Tensor:
    """Fixed 2D sin/cos positional encoding, (h*w, d); generalizes to any
    resolution so full-frame inference works with crop-trained weights."""
    def enc(n, dd):
        pos = torch.arange(n, device=device, dtype=torch.float32)[:, None]
        i = torch.arange(0, dd, 2, device=device, dtype=torch.float32)[None]
        ang = pos / (10000 ** (i / dd))
        pe = torch.zeros(n, dd, device=device)
        pe[:, 0::2] = torch.sin(ang)
        pe[:, 1::2] = torch.cos(ang)
        return pe
    pey, pex = enc(h, d // 2), enc(w, d // 2)
    pe = torch.cat([pey[:, None, :].expand(h, w, d // 2),
                    pex[None, :, :].expand(h, w, d // 2)], dim=-1)
    return pe.reshape(h * w, d)


class _AttnBlock(nn.Module):
    """Pre-norm transformer block: MHSA + MLP, both residual."""

    def __init__(self, d: int, heads: int = 8, mlp: int = 4):
        super().__init__()
        self.n1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.n2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, mlp * d), nn.GELU(),
                                 nn.Linear(mlp * d, d))

    def forward(self, x):
        h = self.n1(x)
        x = x + self.attn(h, h, h, need_weights=False)[0]
        return x + self.mlp(self.n2(x))


class TransformerMid(nn.Module):
    """Transformer bottleneck: global self-attention over the 1/16-res grid.

    256 tokens on a 256-px training crop; ~8k tokens full-frame (fine one-shot).
    Content-based global routing — a blurred/ambiguous region can gather
    evidence from sharp regions anywhere in the frame, which a fixed conv
    receptive field cannot do.
    """

    def __init__(self, cin: int, d: int, layers: int = 4, heads: int = 8):
        super().__init__()
        self.proj_in = nn.Conv2d(cin, d, 1)
        self.blocks = nn.ModuleList([_AttnBlock(d, heads) for _ in range(layers)])
        self.norm = nn.LayerNorm(d)

    def forward(self, x):
        B, _, H, W = x.shape
        t = self.proj_in(x).flatten(2).transpose(1, 2)       # (B, HW, d)
        t = t + _posenc_2d(H, W, t.shape[-1], x.device)
        for blk in self.blocks:
            t = blk(t)
        return self.norm(t).transpose(1, 2).reshape(B, -1, H, W)


class ProjUNet(nn.Module):
    """U-Net: capture (B, 1, H, W) -> (B, 99, H, W) classification+offset head.

    Channel layout: [0:60] u-bin logits, [60:96] v-bin logits, then offset_u,
    offset_v (fraction within the argmax bin, linear), validity logit.
    Four 2x down/up levels: ~300 px receptive field, full-res skip connections
    keep the 4 px M-array cells sharp. H and W must be multiples of 16 (the
    predict script pads).

    ``mid='conv'`` (default) is a conv bottleneck; ``mid='attn'`` swaps in a
    transformer bottleneck (global attention at 1/16) — same channel count, so
    all other weights stay compatible for warm starts.
    """

    def __init__(self, in_ch: int = 1, base: int = 32, mid: str = "conv",
                 attn_layers: int = 4):
        super().__init__()
        self.arch = mid
        c = [base, base * 2, base * 4, base * 8]
        self.enc = nn.ModuleList()
        prev = in_ch
        for ci in c:
            self.enc.append(_Block(prev, ci))
            prev = ci
        if mid == "attn":
            self.mid = TransformerMid(c[-1], base * 16, layers=attn_layers)
        else:
            self.mid = _Block(c[-1], base * 16)
        self.dec = nn.ModuleList()
        prev = base * 16
        for ci in reversed(c):
            self.dec.append(_Block(prev + ci, ci))
            prev = ci
        self.head = nn.Conv2d(base, _HEAD_CH, 1)
        nn.init.zeros_(self.head.bias)

    def forward(self, x):
        skips = []
        for blk in self.enc:
            x = blk(x)
            skips.append(x)
            x = F.max_pool2d(x, 2)
        x = self.mid(x)
        for blk, s in zip(self.dec, reversed(skips)):
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
            x = blk(torch.cat([x, s], dim=1))
        return self.head(x)


# --------------------------------------------------------------------------
# Loss
# --------------------------------------------------------------------------
def _focal_ce(logits: torch.Tensor, target_idx: torch.Tensor, m: torch.Tensor,
              gamma: float) -> torch.Tensor:
    """Masked, summed bin cross-entropy. ``gamma>0`` makes it *focal*: each
    pixel's CE is scaled by ``(1-p_t)**gamma`` so easy, already-correct bins
    contribute little and gradient concentrates on the hard ones (here, the
    edge / bottom-row v-bins the net keeps missing). The focal weights are
    detached and renormalised to mean 1 over the valid pixels, so the CE term
    keeps its plain-CE magnitude — it only *redistributes* emphasis, leaving the
    balance against the offset/validity terms unchanged. ``gamma=0`` is plain CE."""
    ce_px = F.cross_entropy(logits, target_idx, reduction="none")
    if gamma > 0:
        pt = torch.exp(-ce_px.detach()).clamp(max=1.0)   # prob of the true bin
        w = (1.0 - pt).pow(gamma)
        w = w * (m.sum().clamp(min=1.0) / (w * m).sum().clamp(min=1e-6))
        ce_px = w * ce_px
    return (ce_px * m).sum()


def proj_loss(pred: torch.Tensor, target: torch.Tensor, valid: torch.Tensor,
              proj_wh: tuple[int, int], offset_weight: float = 2.0,
              focal_gamma: float = 0.0, v_weight: float = 1.0):
    """Classification + offset loss, masked to valid pixels.

    Per axis: cross-entropy over the coarse bins + L1 on the within-bin
    fraction (supervised at the GT bin), plus BCE on the validity logit.
    ``target`` is (B, 2, H, W) normalized coords (invalid filled with 0),
    ``valid`` (B, 1, H, W) float {0, 1}.
    ``focal_gamma`` (>0) focuses the bin CE on hard pixels (see :func:`_focal_ce`);
    ``v_weight`` scales the row (v) CE relative to the column (u) CE — v is the
    lagging axis (its deficit is concentrated in the edge / bottom-row bins).
    Returns (total, du_px, dv_px, bce, u_bin_acc, v_bin_acc) for logging — both axes
    are trained (CE + offset per axis), so both are reported separately; v is the
    lagging axis worth watching against the documented row deficit.
    """
    nu, nv = N_BINS_U, N_BINS_V
    lu, lv = pred[:, :nu], pred[:, nu:nu + nv]
    off = pred[:, nu + nv:nu + nv + 2]
    m = valid[:, 0]
    nvalid = m.sum().clamp(min=1.0)

    tu = (target[:, 0].clamp(0, 1 - 1e-6) * nu)          # bin space, (B, H, W)
    tv = (target[:, 1].clamp(0, 1 - 1e-6) * nv)
    iu, iv = tu.long(), tv.long()
    fu, fv = tu - iu, tv - iv                            # within-bin fraction

    ce = (_focal_ce(lu, iu, m, focal_gamma)
          + v_weight * _focal_ce(lv, iv, m, focal_gamma)) / nvalid
    off_l1 = ((off[:, 0] - fu).abs() * m).sum() / nvalid \
        + ((off[:, 1] - fv).abs() * m).sum() / nvalid
    bce = F.binary_cross_entropy_with_logits(pred[:, -1], m)

    with torch.no_grad():                                # decoded px error, per axis, for logs
        bu, bv = lu.argmax(1), lv.argmax(1)
        du = ((bu + off[:, 0].clamp(0, 1) - tu).abs() * (proj_wh[0] / nu) * m).sum() / nvalid
        dv = ((bv + off[:, 1].clamp(0, 1) - tv).abs() * (proj_wh[1] / nv) * m).sum() / nvalid
        ubin = ((bu == iu).float() * m).sum() / nvalid
        vbin = ((bv == iv).float() * m).sum() / nvalid
    return ce + offset_weight * off_l1 + bce, du, dv, bce.detach(), ubin, vbin


# --------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------
def _gaussian_blur(img: np.ndarray, sigma: float) -> np.ndarray:
    """Separable Gaussian blur of a 2D float image — numpy-only (runs in
    DataLoader workers, no scipy), vectorised over kernel taps so it's cheap."""
    r = max(1, int(round(3 * sigma)))
    x = np.arange(-r, r + 1)
    k = np.exp(-x * x / (2 * sigma * sigma))
    k /= k.sum()
    out = img
    for ax in (0, 1):
        pad = [(r, r) if a == ax else (0, 0) for a in range(2)]
        p = np.pad(out, pad, mode="reflect")
        acc = np.zeros_like(out)
        for t, w in enumerate(k):
            sl = [slice(None), slice(None)]
            sl[ax] = slice(t, t + out.shape[ax])
            acc = acc + w * p[tuple(sl)]
        out = acc
    return out


def _augment_crop(ic: np.ndarray, rng) -> np.ndarray:
    """Input-only train-time augmentation (target/validity untouched): re-form the
    (clean) render through a **physically-ordered camera image-formation model**, so
    the degradations have the right *structure*, not just the right magnitude.

    Order and signal-dependence matter — the old version did gain/gamma + flat
    additive noise + blur as independent ops in arbitrary order, which is physically
    wrong: real sensor noise is **shot-limited** (Poisson, std ∝ √signal), so bright
    pattern dots are noisier than dark gaps, and optics blur *before* the sensor adds
    noise, which clips *before* the response curve. Pipeline (≈linear radiance → 8-bit):

        exposure gain → optical PSF (defocus) → shot noise (√signal, linear space)
        → read noise → saturate/clip → response curve (gamma) → 8-bit quantize

    The input crop is treated as ≈linear scene radiance (approximation — enough to
    inject the right structure). All stages probabilistic so sharp/clean crops still
    appear (else the subpixel ceiling caps). Parameterised wider than the old version.
    """
    x = ic.astype(np.float32)
    x = x * rng.uniform(0.55, 1.6)                         # exposure / gain (linear)
    if rng.random() < 0.55:                               # optical PSF: 4px cells survive σ<~1.6
        x = _gaussian_blur(x, rng.uniform(0.4, 1.6))
    if rng.random() < 0.85:
        # shot (photon) noise — Poisson, in linear space: a pixel at fraction s of
        # full well collects s·W photons, std √(s·W) photons = √(s/W) back in [0,1].
        # So std ∝ √signal → bright pattern dots noisier than dark gaps (the structure
        # a flat additive σ misses). Lower full_well = noisier capture.
        full_well = rng.uniform(30.0, 500.0)
        x = x + rng.standard_normal(x.shape).astype(np.float32) * np.sqrt(np.maximum(x, 0.0) / full_well)
    if rng.random() < 0.8:                                # read noise — additive, signal-independent
        x = x + rng.normal(0.0, rng.uniform(0.002, 0.025), x.shape).astype(np.float32)
    # saturate / highlight clip: an exposure push blows some highlights to white
    # (real captures clip ~4% of lit px vs ~0.5% in clean renders) — a hard clip at
    # full well, *before* the response curve.
    x = np.clip(x * rng.uniform(0.9, 1.35), 0.0, 1.0)
    x = x ** rng.uniform(0.75, 1.35)                      # camera response curve (gamma / tone)
    x = np.round(x * 255.0) / 255.0                       # 8-bit quantization
    return np.clip(x, 0.0, 1.0).astype(np.float32)


class ProjSamples(Dataset):
    """Random crops from ``renders/train``-style sample folders.

    Sample-major: one dataset item = one sample folder, loaded fresh (nothing
    is cached — 700-sample sets would not fit in RAM) and returning all
    ``crops_per_sample`` random crops from it stacked as (P, C, S, S); flatten
    the first two batch dims in the train loop. DataLoader workers hide the
    PNG/npy load latency. Photometric jitter (gain/gamma) only — no flips, the
    M-array code is chiral and a mirrored capture can never occur for real.
    """

    def __init__(self, root: str, pattern_set: str = "marray",
                 frame: str = "cap_pat_00.png", crop: int = 256,
                 crops_per_sample: int = 8, jitter: bool = True):
        self.dirs = sorted(d for d in Path(root).glob("sample_*")
                           if not d.name.startswith(".")          # skip ._* sidecars
                           and (d / pattern_set / frame).exists()
                           and (d / "gt_proj.npy").exists())
        if not self.dirs:
            raise FileNotFoundError(f"no samples with {pattern_set}/{frame} under {root!r}")
        self.pattern_set, self.frame = pattern_set, frame
        self.crop, self.per, self.jitter = crop, crops_per_sample, jitter
        rig = json.loads((self.dirs[0] / "rig.json").read_text())
        self.proj_wh = (rig["projector"]["width"], rig["projector"]["height"])

    def __len__(self):
        return len(self.dirs)

    def full(self, i: int):
        """Full frame for eval: capture (H, W) float [0,1], gt (H, W, 2) px with NaN."""
        img, gt = self._load(self.dirs[i])
        return img, gt * np.asarray(self.proj_wh, np.float32)

    def _load(self, d: Path):
        from lux import io
        img = io.load_image(str(d / self.pattern_set / self.frame), gray=True)
        gt = np.load(d / "gt_proj.npy").astype(np.float32)
        gt[..., 0] /= self.proj_wh[0]
        gt[..., 1] /= self.proj_wh[1]
        return img.astype(np.float32), gt

    def __getitem__(self, i):
        rng = np.random.default_rng()
        img, gt = self._load(self.dirs[i])
        H, W, S = img.shape[0], img.shape[1], self.crop
        ics, tcs, vs = [], [], []
        for _ in range(self.per):
            for _ in range(8):  # prefer crops that contain some valid supervision
                y, x = rng.integers(0, H - S + 1), rng.integers(0, W - S + 1)
                v = np.isfinite(gt[y:y + S, x:x + S, 0])
                if v.mean() > 0.05:
                    break
            ic = img[y:y + S, x:x + S].copy()
            if self.jitter:
                ic = _augment_crop(ic, rng)
            ics.append(torch.from_numpy(ic[None]))
            tcs.append(torch.from_numpy(
                np.nan_to_num(gt[y:y + S, x:x + S]).transpose(2, 0, 1).copy()))
            vs.append(torch.from_numpy(v[None].astype(np.float32)))
        return torch.stack(ics), torch.stack(tcs), torch.stack(vs)


# --------------------------------------------------------------------------
# Loaf: the whole dataset as two memory-mapped arrays
# --------------------------------------------------------------------------
# Random 256-px crops from per-sample PNGs decode ~25 MB to read ~130 KB; the
# loaf removes that: captures as one (N, H, W) uint8 array, gt as one
# (N, H, W, 2) uint16 fixed-point array (0xFFFF = invalid/NaN, resolution
# proj_w/65534 ~= 0.03 px), both opened with mmap so a crop touches only the
# pages it needs and the OS page cache keeps the hot set in RAM.

_GT_SENTINEL = np.uint16(0xFFFF)
_GT_MAX = 65534.0


def build_loaf(root: str, out_dir: str, pattern_set: str = "marray",
               frame: str = "cap_pat_00.png") -> None:
    """Pack every sample under ``root`` into ``out_dir``/caps.npy + gt.npy + meta."""
    from lux import io
    dirs = sorted(d for d in Path(root).glob("sample_*")
                  if not d.name.startswith(".")                   # skip ._* sidecars
                  and (d / pattern_set / frame).exists() and (d / "gt_proj.npy").exists())
    if not dirs:
        raise FileNotFoundError(f"no samples under {root!r}")
    rig = json.loads((dirs[0] / "rig.json").read_text())
    pw, ph = rig["projector"]["width"], rig["projector"]["height"]
    probe = io.load_image(str(dirs[0] / pattern_set / frame), gray=True)
    H, W = probe.shape
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    caps = np.lib.format.open_memmap(out / "caps.npy", mode="w+",
                                     dtype=np.uint8, shape=(len(dirs), H, W))
    gts = np.lib.format.open_memmap(out / "gt.npy", mode="w+",
                                    dtype=np.uint16, shape=(len(dirs), H, W, 2))
    for i, d in enumerate(dirs):
        img = io.load_image(str(d / pattern_set / frame), gray=True)
        caps[i] = np.clip(np.round(img * 255), 0, 255).astype(np.uint8)
        gt = np.load(d / "gt_proj.npy")
        q = np.round(np.clip(gt / [pw, ph], 0, 1) * _GT_MAX)
        q = np.where(np.isfinite(gt), q, float(_GT_SENTINEL)).astype(np.uint16)
        gts[i] = q
        if (i + 1) % 100 == 0 or i + 1 == len(dirs):
            print(f"  loaf {i + 1}/{len(dirs)}", flush=True)
    (out / "meta.json").write_text(json.dumps({
        "names": [d.name for d in dirs], "proj_wh": [pw, ph], "hw": [H, W],
        "pattern_set": pattern_set, "frame": frame}, indent=2) + "\n")
    caps.flush()
    gts.flush()


class LoafSamples(Dataset):
    """Same item contract as :class:`ProjSamples` but crops straight out of the
    memory-mapped loaf. Memmaps are opened lazily per process (they don't
    survive DataLoader worker pickling)."""

    def __init__(self, loaf_dir: str, crop: int = 256,
                 crops_per_sample: int = 8, jitter: bool = True):
        self.dir = Path(loaf_dir)
        meta = json.loads((self.dir / "meta.json").read_text())
        self.names: list[str] = meta["names"]
        self.proj_wh = tuple(meta["proj_wh"])
        self.crop, self.per, self.jitter = crop, crops_per_sample, jitter
        self._caps = self._gts = None

    def _open(self):
        if self._caps is None:
            self._caps = np.load(self.dir / "caps.npy", mmap_mode="r")
            self._gts = np.load(self.dir / "gt.npy", mmap_mode="r")
        return self._caps, self._gts

    def __len__(self):
        return len(self.names)

    def full(self, i: int):
        """Full frame for eval: capture (H, W) float [0,1], gt (H, W, 2) px with NaN."""
        caps, gts = self._open()
        img = caps[i].astype(np.float32) / 255.0
        q = gts[i]
        valid = q[..., 0] != _GT_SENTINEL
        gt = q.astype(np.float32) / _GT_MAX * self.proj_wh
        return img, np.where(valid[..., None], gt, np.nan)

    def __getitem__(self, i):
        rng = np.random.default_rng()
        caps, gts = self._open()
        H, W, S = caps.shape[1], caps.shape[2], self.crop
        ics, tcs, vs = [], [], []
        for _ in range(self.per):
            for _ in range(8):
                y, x = rng.integers(0, H - S + 1), rng.integers(0, W - S + 1)
                q = gts[i, y:y + S, x:x + S]
                v = q[..., 0] != _GT_SENTINEL
                if v.mean() > 0.05:
                    break
            ic = caps[i, y:y + S, x:x + S].astype(np.float32) / 255.0
            if self.jitter:
                ic = _augment_crop(ic, rng)
            tc = q.astype(np.float32) / _GT_MAX
            tc = np.where(v[..., None], tc, 0.0)
            ics.append(torch.from_numpy(ic[None]))
            tcs.append(torch.from_numpy(tc.transpose(2, 0, 1).copy()))
            vs.append(torch.from_numpy(v[None].astype(np.float32)))
        return torch.stack(ics), torch.stack(tcs), torch.stack(vs)


class ConcatLoaf(torch.utils.data.ConcatDataset):
    """Several loaves trained as one dataset (e.g. the original render set + a new
    domain like the planar-junction set). Keeps the :class:`LoafSamples` item
    contract — ``ConcatDataset`` already routes ``__getitem__``/``__len__`` across
    the parts by ``cumulative_sizes`` — and additionally exposes ``proj_wh``,
    ``names`` and ``full()`` so the trainer/evaluator treat it like one loaf.

    Crops are still drawn per-sample inside each part, so the parts may differ in
    camera resolution; only ``proj_wh`` (the projector dims that scale ``gt_proj``)
    must agree, since the loss is in projector pixels. The natural mix ratio is the
    parts' size ratio (two ~10k loaves -> ~50/50); pass a sampler to the DataLoader
    to weight them otherwise."""

    def __init__(self, loaves: list[LoafSamples]):
        super().__init__(loaves)
        whs = {tuple(l.proj_wh) for l in loaves}
        if len(whs) != 1:
            raise ValueError(f"loaves disagree on proj_wh (loss is in projector px): {whs}")
        self.proj_wh = loaves[0].proj_wh
        self.names = [n for l in loaves for n in l.names]

    def _route(self, i: int) -> tuple[LoafSamples, int]:
        from bisect import bisect_right
        p = bisect_right(self.cumulative_sizes, i)
        prev = self.cumulative_sizes[p - 1] if p else 0
        return self.datasets[p], i - prev

    def full(self, i: int):
        ds, j = self._route(i)
        return ds.full(j)

    def part_starts(self) -> list[int]:
        """First global index of each part (for held-out splits spanning parts)."""
        return [0] + list(self.cumulative_sizes[:-1])


# --------------------------------------------------------------------------
# Inference
# --------------------------------------------------------------------------
@torch.no_grad()
def predict_full(model: nn.Module, img: np.ndarray, proj_wh: tuple[int, int],
                 device: str = "cpu", return_conf: bool = False,
                 conf_per_axis: bool = False):
    """Full-frame inference: capture (H, W) in [0,1] -> (H, W, 2) projector px,
    NaN where the validity head says the projector can't see the pixel.

    ``return_conf`` also returns a per-pixel confidence (H, W): the **joint**
    correspondence confidence ``min(conf_u, conf_v)`` of the two bin-softmax
    maxima — both axes must be certain for the (col, row) pair to be trusted.
    (Gating on the *column* softmax alone is blind to row-only failures: on an
    oblique plane the foreshortened band stays column-confident while the row
    aliases, so a column-only confidence map reads uniformly high while the row
    is badly wrong.) conf>0.9 keeps ~30% of pixels at ~98% bin accuracy on both
    axes; threshold it to trade coverage for outlier purity.

    ``conf_per_axis`` instead returns ``(uv, conf_u, conf_v)`` so analysis tools
    can gate each axis on its own softmax (and form the joint min themselves).
    """
    model.eval()
    nu, nv = N_BINS_U, N_BINS_V
    H, W = img.shape
    ph, pw = (-H) % 16, (-W) % 16
    x = torch.from_numpy(img.astype(np.float32))[None, None]
    x = F.pad(x, (0, pw, 0, ph), mode="reflect").to(device)
    out = model(x)[0, :, :H, :W]
    # Decode on-device: ship 2-4 result channels to the CPU, not all 99 (~840 MB).
    u = (out[:nu].argmax(0).float() + out[nu + nv].clamp(0, 1)) * (proj_wh[0] / nu)
    v = (out[nu:nu + nv].argmax(0).float() + out[nu + nv + 1].clamp(0, 1)) * (proj_wh[1] / nv)
    uv = torch.stack([u, v], dim=-1)
    valid = out[-1] > 0.0                                # logit > 0 == p > 0.5
    uv = torch.where(valid[..., None], uv, torch.full_like(uv, float("nan")))
    uv = uv.float().cpu().numpy()
    if return_conf or conf_per_axis:
        conf_u = torch.softmax(out[:nu].float(), dim=0).max(0).values
        conf_v = torch.softmax(out[nu:nu + nv].float(), dim=0).max(0).values
        if conf_per_axis:
            return uv, conf_u.cpu().numpy(), conf_v.cpu().numpy()
        conf = torch.minimum(conf_u, conf_v)             # joint correspondence conf
        return uv, conf.cpu().numpy()
    return uv


def _tile_positions(n: int, t: int, stride: int) -> list[int]:
    """Tile start offsets covering [0, n) with a final clamped tile to the edge."""
    if n <= t:
        return [0]
    ps = list(range(0, n - t + 1, stride))
    if ps[-1] != n - t:
        ps.append(n - t)
    return ps


@torch.no_grad()
def predict_tiled(model: nn.Module, img: np.ndarray, proj_wh: tuple[int, int],
                  device: str = "cpu", tile: int = 256, margin: int = 32,
                  overlap: int = 0, reflect: bool = False, return_conf: bool = False,
                  conf_per_axis: bool = False):
    """Full-frame inference by stitching ``tile``x``tile`` predictions instead of
    running the whole frame at once.

    The attention bottleneck does *global* self-attention over the 1/16-res grid, so
    its token count scales with the input: a 256-px training crop is 16x16=256 tokens,
    but a 1080x1920 frame is ~68x120=8160 tokens — a regime the softmax/positional
    encoding never trained on, where accuracy collapses (verified: bin-acc 94%@256-tok
    -> 24%@8160-tok). Tiling keeps every forward pass at the trained ``tile`` size, so
    a resolution-sensitive model runs in-distribution. (Conv models are shift-invariant
    and don't strictly need this — ``predict_full`` works — though tiling also closes
    the conv row deficit, the same train-crop/eval-frame mismatch at the v extremes.)

    Two stitch modes:

    - ``overlap=0`` (default, cheap): **margin-crop** — each output pixel is taken from
      the tile where it sits >= ``margin`` from the edge; border tiles fill to the frame
      edge. A *hard* assignment, so adjacent tiles that disagree in low-context (e.g.
      background) regions leave visible square **seams**.
    - ``overlap>0``: **overlapping tiles + per-pixel max-confidence** — tiles run at
      ``stride = tile - overlap`` so every pixel is covered by several offset tiles, and
      each pixel keeps the prediction from whichever tile is most confident (and valid)
      there. A tile's own edges have the least context -> lowest softmax confidence ->
      they lose to a tile where the pixel is well-centred, so the seams dissolve. Larger
      ``overlap`` = more candidates = smoother, at proportional cost (~(tile/stride)^2
      more forward passes).

    ``tile`` must be a multiple of 16. Mirrors :func:`predict_full`'s returns: ``uv``
    (H,W,2), or with ``return_conf`` ``(uv, conf)``, or with ``conf_per_axis``
    ``(uv, conf_u, conf_v)``.
    """
    H, W = img.shape
    t = min(tile, H, W)

    if overlap > 0:
        stride = max(1, t - overlap)
        uv = np.full((H, W, 2), np.nan, np.float32)
        cu = np.zeros((H, W), np.float32)
        cv = np.zeros((H, W), np.float32)
        best = np.full((H, W), -1.0, np.float32)        # best joint conf seen per pixel
        for y in _tile_positions(H, t, stride):
            for x in _tile_positions(W, t, stride):
                ruv, rcu, rcv = predict_full(model, img[y:y + t, x:x + t], proj_wh,
                                             device=device, conf_per_axis=True)
                # invalid (NaN uv) scores below any valid pixel, so a valid prediction
                # from any tile always beats an abstaining one.
                score = np.where(np.isfinite(ruv[..., 0]), np.minimum(rcu, rcv), -1.0)
                bb = best[y:y + t, x:x + t]
                win = score > bb
                bb[win] = score[win]
                uv[y:y + t, x:x + t][win] = ruv[win]
                cu[y:y + t, x:x + t][win] = rcu[win]
                cv[y:y + t, x:x + t][win] = rcv[win]
        if conf_per_axis:
            return uv, cu, cv
        if return_conf:
            return uv, np.minimum(cu, cv)
        return uv

    # margin-crop stitch (deterministic — use this for the canonical metric; max-conf
    # above is for output). reflect=True pads the frame by `margin` first, so the TRUE
    # outer ring is decoded centrally too — otherwise the outermost margin-px come from a
    # tile where they sit at the real frame edge, with no neighbour to centre them.
    pad = margin if reflect else 0
    src = np.pad(img, pad, mode="reflect") if pad else img
    Hs, Ws = src.shape
    stride = max(1, t - 2 * margin)
    uv = np.full((Hs, Ws, 2), np.nan, np.float32)
    n_extra = 2 if conf_per_axis else (1 if return_conf else 0)
    extra = [np.zeros((Hs, Ws), np.float32) for _ in range(n_extra)]
    for y in _tile_positions(Hs, t, stride):
        for x in _tile_positions(Ws, t, stride):
            r = predict_full(model, src[y:y + t, x:x + t], proj_wh, device=device,
                             return_conf=return_conf, conf_per_axis=conf_per_axis)
            ruv = r[0] if n_extra else r
            rex = r[1:] if n_extra else ()
            top = 0 if y == 0 else margin
            left = 0 if x == 0 else margin
            bot = t if y + t == Hs else t - margin
            right = t if x + t == Ws else t - margin
            uv[y + top:y + bot, x + left:x + right] = ruv[top:bot, left:right]
            for buf, rr in zip(extra, rex):
                buf[y + top:y + bot, x + left:x + right] = rr[top:bot, left:right]
    if pad:
        uv = uv[pad:pad + H, pad:pad + W]
        extra = [e[pad:pad + H, pad:pad + W] for e in extra]
    if conf_per_axis:
        return uv, extra[0], extra[1]
    if return_conf:
        return uv, extra[0]
    return uv


def save_checkpoint(path: str, model: ProjUNet, proj_wh, meta: dict | None = None):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state": model.state_dict(), "proj_wh": proj_wh,
                "bins": (N_BINS_U, N_BINS_V), "arch": getattr(model, "arch", "conv"),
                "meta": meta or {}}, path)


def load_weights_compatible(model: nn.Module, state: dict) -> int:
    """Warm-start: copy every tensor whose name and shape match (e.g. the
    encoder from a checkpoint with a different head). Returns tensors loaded."""
    own = model.state_dict()
    keep = {k: v for k, v in state.items() if k in own and own[k].shape == v.shape}
    model.load_state_dict(keep, strict=False)
    return len(keep)


def load_checkpoint(path: str, device: str = "cpu") -> tuple[ProjUNet, tuple[int, int]]:
    ck = torch.load(path, map_location=device, weights_only=False)
    model = ProjUNet(mid=ck.get("arch", "conv")).to(device)
    model.load_state_dict(ck["state"])
    return model, tuple(ck["proj_wh"])
