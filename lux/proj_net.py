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
def proj_loss(pred: torch.Tensor, target: torch.Tensor, valid: torch.Tensor,
              proj_wh: tuple[int, int], offset_weight: float = 2.0):
    """Classification + offset loss, masked to valid pixels.

    Per axis: cross-entropy over the coarse bins + L1 on the within-bin
    fraction (supervised at the GT bin), plus BCE on the validity logit.
    ``target`` is (B, 2, H, W) normalized coords (invalid filled with 0),
    ``valid`` (B, 1, H, W) float {0, 1}.
    Returns (total, decoded_px_l1, bce, u_bin_acc) for logging.
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

    ce = (F.cross_entropy(lu, iu, reduction="none") * m).sum() / nvalid \
        + (F.cross_entropy(lv, iv, reduction="none") * m).sum() / nvalid
    off_l1 = ((off[:, 0] - fu).abs() * m).sum() / nvalid \
        + ((off[:, 1] - fv).abs() * m).sum() / nvalid
    bce = F.binary_cross_entropy_with_logits(pred[:, -1], m)

    with torch.no_grad():                                # decoded px error, for logs
        bu = lu.argmax(1)
        du = (bu + off[:, 0].clamp(0, 1) - tu).abs() * (proj_wh[0] / nu)
        dv = (lv.argmax(1) + off[:, 1].clamp(0, 1) - tv).abs() * (proj_wh[1] / nv)
        l1_px = ((du + dv) / 2 * m).sum() / nvalid
        bin_acc = ((bu == iu).float() * m).sum() / nvalid
    return ce + offset_weight * off_l1 + bce, l1_px, bce.detach(), bin_acc


# --------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------
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
                ic = np.clip(ic * rng.uniform(0.7, 1.3), 0, 1) ** rng.uniform(0.8, 1.25)
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
                ic = np.clip(ic * rng.uniform(0.7, 1.3), 0, 1) ** rng.uniform(0.8, 1.25)
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
                 device: str = "cpu", return_conf: bool = False):
    """Full-frame inference: capture (H, W) in [0,1] -> (H, W, 2) projector px,
    NaN where the validity head says the projector can't see the pixel.

    ``return_conf`` also returns the u-bin softmax max-probability (H, W) — a
    well-calibrated per-pixel confidence (conf>0.9 keeps ~50% of pixels at ~98%
    bin accuracy); threshold it to trade coverage for outlier purity.
    """
    model.eval()
    nu, nv = N_BINS_U, N_BINS_V
    H, W = img.shape
    ph, pw = (-H) % 16, (-W) % 16
    x = torch.from_numpy(img.astype(np.float32))[None, None]
    x = F.pad(x, (0, pw, 0, ph), mode="reflect").to(device)
    out = model(x)[0, :, :H, :W]
    # Decode on-device: ship 2-3 result channels to the CPU, not all 99 (~840 MB).
    u = (out[:nu].argmax(0).float() + out[nu + nv].clamp(0, 1)) * (proj_wh[0] / nu)
    v = (out[nu:nu + nv].argmax(0).float() + out[nu + nv + 1].clamp(0, 1)) * (proj_wh[1] / nv)
    uv = torch.stack([u, v], dim=-1)
    valid = out[-1] > 0.0                                # logit > 0 == p > 0.5
    uv = torch.where(valid[..., None], uv, torch.full_like(uv, float("nan")))
    if return_conf:
        conf = torch.softmax(out[:nu].float(), dim=0).max(0).values
        return uv.float().cpu().numpy(), conf.cpu().numpy()
    return uv.float().cpu().numpy()


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
