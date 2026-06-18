"""Differentiable homography-warp proxy for structured-light pattern co-design.

The one-shot decoder (:mod:`lux.proj_net`) hits a hard information floor on grazing
surfaces: at 60-75 deg obliquity the fixed random-binary M-array's ~20 px unique
window compresses (x cos 75 ~= 0.26) to ~5 px on the camera, below the resolving
limit, so bin classification collapses to ~8%. The only lever is a *better pattern*
(``docs/cliff_plan.md`` step 7).

This module is the reusable core for learning one. The key reframe: anamorphic
compression at a planar patch *is* a local homography -- cheap and differentiable --
so we don't render with Mitsuba. We:

  1. generate a pattern from a small **structured generator** (multi-scale sinusoidal
     carriers, :class:`PatternGenerator`) whose params carry gradient -- optimizing a
     code *family*, not raw pixels, so windowed-uniqueness stays structural;
  2. sample an oblique-plane **homography** (:func:`sample_homography_inv`) and map a
     camera crop's pixels through it to the projector coordinate each pixel sees --
     that coordinate IS the ground-truth correspondence, exact by construction;
  3. **evaluate the pattern at those coordinates** (analytic for carriers; bilinear
     ``grid_sample`` for a raster pattern), multiply in a smooth shading field, and
     re-form through a differentiable **photometric stack** (:func:`differentiable_augment`,
     the torch port of :func:`lux.proj_net._augment_crop`) so the pattern is optimized
     for a real-capture world, not a clean one;
  4. decode + backprop into BOTH the decoder weights and the generator params.

:class:`ProxyBatcher` ties (1)-(3) into ``(capture, target, valid)`` batches matching
the :func:`lux.proj_net.proj_loss` contract exactly. Everything is GPU-resident and on
the autograd graph -- there is no DataLoader (it would pickle tensors across worker
processes, detaching them from the live generator).

Scope: a homography is a *single plane*, so this trains the oblique-planar **cliff**
only; depth discontinuities / edges still need rendered data (``cliff_plan.md`` line 104).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------
# Pattern generator: a small learnable bank of 2D sinusoidal carriers
# --------------------------------------------------------------------------
class PatternGenerator(nn.Module):
    """K learnable 2D sinusoidal carriers composed into a [0,1] grayscale pattern.

    ``pattern(x, y) = sigmoid( bias + sum_k a_k * sin(2pi (fx_k x + fy_k y) + ph_k) )``
    with ``(x, y)`` the projector-normalized coordinate in [0,1] and frequencies in
    *cycles across the full projector*. Per carrier the learnable params are a
    log-frequency-magnitude, an orientation ``theta``, a phase and a log-amplitude
    (``fx = exp(log_freq) cos theta``, ``fy = exp(log_freq) sin theta``); plus a global
    ``bias``. ~5 params/carrier -> far too few to memorize pixels, so the optimizer
    must find a good *code family*. The sigmoid bounds [0,1] and, as amplitudes grow,
    yields the quasi-binary high-contrast look of an M-array without a non-differentiable
    threshold.

    The decoder reads a fixed coarse-bin grid (``N_BINS_U=60`` / ``N_BINS_V=36`` ->
    one cycle per bin is ``fx=60`` / ``fy=36``); ``init='carrier_bank'`` seeds the two
    lowest carriers there so a coarse phase reading names the bin, with higher octaves
    carrying subpixel precision. ``init='marray_fit'`` instead least-squares-fits the
    carrier params to the real :func:`scripts.gen_patterns._marray`, for the proxy
    fidelity sanity check (Milestone 1).
    """

    def __init__(self, proj_wh: tuple[int, int], n_carriers: int = 12,
                 init: str = "carrier_bank", n_bins: tuple[int, int] = (60, 36),
                 seed: int = 0):
        super().__init__()
        self.proj_wh = tuple(proj_wh)
        self.n_carriers = n_carriers
        rng = np.random.default_rng(seed)
        nu, nv = n_bins

        # Octave-spaced frequency magnitudes from ~the coarse-bin scale up toward the
        # cell-resolving limit; orientations spread over [0, pi); random phases.
        f_lo = float(min(nu, nv))                              # ~one cycle per coarse bin
        f_hi = float(min(proj_wh) / 8.0)                       # high-detail carrier
        mags = np.geomspace(f_lo, f_hi, n_carriers)
        thetas = (np.linspace(0.0, np.pi, n_carriers, endpoint=False)
                  + rng.uniform(-0.15, 0.15, n_carriers))
        # Seed the two lowest carriers axis-aligned at the bin scale (u then v) so the
        # coarse code is bin-nameable from step 0.
        thetas[0], mags[0] = 0.0, float(nu)
        if n_carriers > 1:
            thetas[1], mags[1] = np.pi / 2.0, float(nv)
        phases = rng.uniform(0.0, 2 * np.pi, n_carriers)
        amps = np.full(n_carriers, 1.0 / max(1, n_carriers) * 3.0)

        self.log_freq = nn.Parameter(torch.tensor(np.log(mags), dtype=torch.float32))
        self.theta = nn.Parameter(torch.tensor(thetas, dtype=torch.float32))
        self.phase = nn.Parameter(torch.tensor(phases, dtype=torch.float32))
        self.log_amp = nn.Parameter(torch.tensor(np.log(amps), dtype=torch.float32))
        self.bias = nn.Parameter(torch.zeros((), dtype=torch.float32))

        if init == "marray_fit":
            self._fit_marray(seed)
        elif init != "carrier_bank":
            raise ValueError(f"unknown init {init!r}")

    @classmethod
    def ladder(cls, proj_wh, periods_u, periods_v, seed=0):
        """Frozen-coprime-period hierarchical ladder (Build 4): u-carriers along x (theta=0) at
        ``periods_u``, v-carriers along y (theta=pi/2) at ``periods_v``. Periods are a discrete
        (coprime) constraint, so ``log_freq``/``theta`` are FROZEN (requires_grad=False); only
        amplitude/phase/bias learn (the energy split). Carrier order is u-block then v-block, which
        the quad head / :func:`quad_loss` rely on."""
        g = cls.__new__(cls)
        nn.Module.__init__(g)
        W, H = proj_wh
        g.proj_wh = (W, H)
        g.n_u, g.n_v = len(periods_u), len(periods_v)
        g.n_carriers = g.n_u + g.n_v
        g.periods_u, g.periods_v = list(periods_u), list(periods_v)
        rng = np.random.default_rng(seed)
        mags = np.array([W / p for p in periods_u] + [H / p for p in periods_v])
        thetas = np.array([0.0] * g.n_u + [np.pi / 2] * g.n_v)
        g.log_freq = nn.Parameter(torch.tensor(np.log(mags), dtype=torch.float32), requires_grad=False)
        g.theta = nn.Parameter(torch.tensor(thetas, dtype=torch.float32), requires_grad=False)
        g.phase = nn.Parameter(torch.tensor(rng.uniform(0, 2 * np.pi, g.n_carriers), dtype=torch.float32))
        g.log_amp = nn.Parameter(torch.log(torch.full((g.n_carriers,), 1.5 / g.n_carriers)))
        g.bias = nn.Parameter(torch.zeros((), dtype=torch.float32))
        return g

    # -- frequency / amplitude accessors (kept positive / well-conditioned) ----
    def freqs(self) -> tuple[torch.Tensor, torch.Tensor]:
        mag = torch.exp(self.log_freq)
        return mag * torch.cos(self.theta), mag * torch.sin(self.theta)

    def sample_at(self, coords: torch.Tensor, mtf=None) -> torch.Tensor:
        """Evaluate the pattern at projector-normalized coords ``(..., 2)`` in [0,1].

        Returns the same leading shape with a channel axis, i.e. ``coords`` of shape
        ``(B, H, W, 2)`` -> ``(B, 1, H, W)``. Analytic and fully differentiable in the
        carrier params (no rasterize-then-resample, so no interpolation resolving-limit
        artifact); coords are treated as constants (they are the GT).

        ``mtf=(sigma_mtf_px, sigma_def_px)`` applies the optics-anchored anisotropic-Gaussian
        blur **analytically**, as a per-carrier amplitude attenuation (the Gaussian's Fourier
        transform): ``a_k -> a_k·exp(-2π² f_k^T Σ f_k)`` with ``Σ = (1/12+σ_mtf²)·JJ^T +
        σ_def²·I``. ``J`` is the exact planar camera->projector Jacobian (per pixel, finite-diff
        of the coords grid), so the anisotropy is exact; geometry (footprint, moment-matched box
        variance 1/12) and optics (camera MTF σ_mtf, projector defocus σ_def) add in variance.
        Cheap and differentiable -- the blur the co-design inner loop trains through (vs the
        footprint-supersample, which is the offline fidelity oracle). Requires a grid
        (``coords.dim()==4``)."""
        fx, fy = self.freqs()                                  # (K,)
        x = coords[..., 0].unsqueeze(-1)                       # (..., 1)
        y = coords[..., 1].unsqueeze(-1)
        ang = 2 * np.pi * (x * fx + y * fy) + self.phase       # (..., K)
        amp = torch.exp(self.log_amp)                          # (K,)
        if mtf is not None and coords.dim() == 4:
            amp = amp * self._mtf_atten(coords, fx, fy, mtf)   # (B,H,W,K)
        val = torch.sigmoid(self.bias + (amp * torch.sin(ang)).sum(-1))   # (...)
        return val.unsqueeze(-3) if val.dim() >= 2 else val[None]

    def _mtf_atten(self, coords: torch.Tensor, fx: torch.Tensor, fy: torch.Tensor, mtf):
        """Per-carrier per-pixel amplitude attenuation ``(B,H,W,K)`` from the anisotropic-Gaussian
        MTF. ``coords`` (B,H,W,2) projector-normalized; ``mtf=(sigma_mtf_px, sigma_def_px)``."""
        W, H = self.proj_wh
        sig_mtf, sig_def = mtf
        u, v = coords[..., 0] * W, coords[..., 1] * H          # projector px
        # J (proj-px per cam-px): gradients along rows(=y,dim1) and cols(=x,dim2)
        du_dy, du_dx = torch.gradient(u, dim=(1, 2))
        dv_dy, dv_dx = torch.gradient(v, dim=(1, 2))
        JJt00 = du_dx ** 2 + du_dy ** 2                        # (B,H,W)
        JJt01 = du_dx * dv_dx + du_dy * dv_dy
        JJt11 = dv_dx ** 2 + dv_dy ** 2
        fpx, fpy = fx / W, fy / H                              # carrier freq, cycles/proj-px (K,)
        fc2 = (fpx[None, None, None, :] ** 2 * JJt00[..., None]
               + 2 * fpx[None, None, None, :] * fpy[None, None, None, :] * JJt01[..., None]
               + fpy[None, None, None, :] ** 2 * JJt11[..., None])     # |f_cam|^2 (B,H,W,K)
        fp2 = (fpx ** 2 + fpy ** 2)[None, None, None, :]               # |f_proj|^2 (1,1,1,K)
        var = (1.0 / 12.0 + sig_mtf ** 2) * fc2 + sig_def ** 2 * fp2
        return torch.exp(-2 * np.pi ** 2 * var)

    @torch.no_grad()
    def materialize(self, proj_wh: tuple[int, int] | None = None) -> np.ndarray:
        """Render the pattern at projector resolution as ``(1, Hp, Wp)`` float [0,1]
        (matches :func:`scripts.gen_patterns._marray` so it dumps to a PNG set)."""
        w, h = proj_wh or self.proj_wh
        dev = self.bias.device
        ys = (torch.arange(h, device=dev) + 0.5) / h
        xs = (torch.arange(w, device=dev) + 0.5) / w
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        coords = torch.stack([gx, gy], dim=-1)[None]           # (1, h, w, 2)
        img = self.sample_at(coords)[0, 0].cpu().numpy()       # (h, w)
        return img[None].astype(np.float32)

    def regularizer(self, tau: float = 0.15) -> torch.Tensor:
        """Anti-collapse reg: carrier repulsion in (log-freq, orientation) space so the
        bank doesn't collapse to one frequency, plus a contrast floor so it doesn't
        collapse to flat gray. Returns a scalar to add to the loss (already small)."""
        lf = self.log_freq[:, None] - self.log_freq[None, :]
        dth = self.theta[:, None] - self.theta[None, :]
        dth = torch.atan2(torch.sin(dth), torch.cos(dth))      # wrap to (-pi, pi]
        d2 = lf ** 2 + dth ** 2
        K = self.n_carriers
        eye = torch.eye(K, device=lf.device, dtype=torch.bool)
        repel = torch.exp(-d2 / (tau ** 2))[~eye].mean()       # high when carriers cluster
        # contrast floor on a cheap random probe (variance should stay healthy)
        dev = self.bias.device
        probe = torch.rand(4096, 2, device=dev)
        std = self.sample_at(probe[None])[0, 0].std()
        contrast = F.relu(0.22 - std)                          # push std up toward ~quarter-range
        return repel + 2.0 * contrast

    def _fit_marray(self, seed: int) -> None:
        """Least-squares-fit the carriers to the real M-array pattern (for the proxy
        fidelity check). Fits amplitudes/phases at the seeded frequencies via linear
        regression of [sin, cos] features onto logit(target); leaves frequencies fixed."""
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from scripts.gen_patterns import _marray
        w, h = self.proj_wh
        # fit on a downsampled grid for speed (frequencies are in normalized cycles)
        tgt = _marray(w, h)[0]
        s = max(1, min(w, h) // 256)
        tgt = tgt[::s, ::s]
        gh, gw = tgt.shape
        ys = (np.arange(gh) + 0.5) / gh
        xs = (np.arange(gw) + 0.5) / gw
        gy, gx = np.meshgrid(ys, xs, indexing="ij")
        fx = (np.exp(self.log_freq.detach().numpy()) * np.cos(self.theta.detach().numpy()))
        fy = (np.exp(self.log_freq.detach().numpy()) * np.sin(self.theta.detach().numpy()))
        ang = 2 * np.pi * (gx[..., None] * fx + gy[..., None] * fy)      # (gh, gw, K)
        feats = np.concatenate([np.sin(ang), np.cos(ang),
                                np.ones((gh, gw, 1))], axis=-1).reshape(-1, 2 * len(fx) + 1)
        y = np.clip(tgt.reshape(-1), 1e-3, 1 - 1e-3)
        z = np.log(y / (1 - y))                                          # logit target
        coef, *_ = np.linalg.lstsq(feats, z, rcond=None)
        sin_c, cos_c, bias = coef[:len(fx)], coef[len(fx):2 * len(fx)], coef[-1]
        amp = np.sqrt(sin_c ** 2 + cos_c ** 2) + 1e-6
        ph = np.arctan2(cos_c, sin_c)              # sin(a+ph)=cos_c... -> a*sin+b*cos form
        with torch.no_grad():
            self.log_amp.copy_(torch.tensor(np.log(amp), dtype=torch.float32))
            self.phase.copy_(torch.tensor(ph, dtype=torch.float32))
            self.bias.copy_(torch.tensor(float(bias), dtype=torch.float32))


class RasterPattern(nn.Module):
    """A fixed (non-learnable) raster pattern with the same ``sample_at`` interface as
    :class:`PatternGenerator`, sampled by bilinear ``grid_sample``. Used for the proxy
    fidelity check (project the real M-array through the proxy) and any hand-designed
    pattern A/B. ``buf`` is a (Hp, Wp) float [0,1] array."""

    def __init__(self, buf: np.ndarray):
        super().__init__()
        if buf.ndim == 3:
            buf = buf[0]
        self.register_buffer("pat", torch.tensor(buf, dtype=torch.float32)[None, None])
        self.proj_wh = (buf.shape[1], buf.shape[0])

    def sample_at(self, coords: torch.Tensor) -> torch.Tensor:
        B = coords.shape[0]
        grid = coords * 2.0 - 1.0                              # [0,1] -> [-1,1]
        pat = self.pat.expand(B, -1, -1, -1).to(coords.device)
        return F.grid_sample(pat, grid, mode="bilinear",
                             padding_mode="reflection", align_corners=False)

    @classmethod
    def from_marray(cls, proj_wh: tuple[int, int]) -> "RasterPattern":
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from scripts.gen_patterns import _marray
        w, h = proj_wh
        return cls(_marray(w, h))

    @torch.no_grad()
    def materialize(self, proj_wh=None) -> np.ndarray:
        return self.pat[0].cpu().numpy().astype(np.float32)


# --------------------------------------------------------------------------
# Oblique-plane homography
# --------------------------------------------------------------------------
def _rot3(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def sample_homography_inv(rng, hw: tuple[int, int], obliq_deg: tuple[float, float],
                          dens_range: tuple[float, float] = (0.00048, 0.00056),
                          ) -> tuple[np.ndarray, float]:
    """Sample a camera->projector homography for a random oblique plane.

    Returns ``(Hinv, obliq_rad)`` where ``Hinv`` (3x3) maps a camera *pixel* coordinate
    ``(col, row, 1)`` to a projector-*normalized* coordinate ``(x, y, 1)`` in [0,1]^2,
    and ``obliq_rad`` is the achieved surface-normal angle (the eval band).

    Derivation (tilt about the image y-axis by ``theta``, plane at unit distance, pinhole):
    a surface point ``s`` along the tilt direction images at ``x = s cos th / (1 + s sin th)``;
    inverting gives the projective map with matrix ``[[1,0,0],[0,cos th,0],[-sin th,0,cos th]]``
    acting on the camera-normalized coordinate. Frontal (``th=0``) is the identity (no
    compression); the local surface-units-per-pixel along the tilt axis is ``1/cos th``, i.e.
    the pattern is compressed in the image by ``cos th`` -- exactly the anamorphic foreshortening
    that drives the cliff. We wrap it with an azimuth rotation ``phi`` (varied tilt direction),
    a focal length (perspective strength), an isotropic density (projector px per camera px)
    and a centering translation; the whole composition is one 3x3."""
    H, W = hw
    th = np.deg2rad(rng.uniform(*obliq_deg))
    phi = rng.uniform(0.0, 2 * np.pi)
    fpx = rng.uniform(20.0 * max(H, W), 60.0 * max(H, W))      # near-orthographic (mild perspective)
    # absolute frontal density: projector-normalized units per camera pixel (rig geometry /
    # surface distance). Kept ~fixed across obliquity so the foreshortening genuinely
    # shrinks cells on the camera as 1/cos(theta) -> the absolute resolving-limit crossing
    # that IS the cliff (an isotropic bbox-fit would normalize this away). The default
    # range pins the projector-px-per-camera-px near the rig's ~1:1 (proj_w*d0 ~= 1), so a
    # 20px M-array window (~20 cam px frontal) compresses to ~5 cam px at 75 deg -- the
    # documented resolving limit -- landing the cliff at the right obliquity.
    d0 = rng.uniform(*dens_range)

    cx, cy = (W - 1) / 2.0, (H - 1) / 2.0
    N = np.array([[1.0 / fpx, 0.0, -cx / fpx],
                  [0.0, 1.0 / fpx, -cy / fpx],
                  [0.0, 0.0, 1.0]])
    M = np.array([[1.0, 0.0, 0.0],
                  [0.0, np.cos(th), 0.0],
                  [-np.sin(th), 0.0, np.cos(th)]])             # foreshorten along x
    S = np.diag([d0 * fpx, d0 * fpx, 1.0])                     # frontal density d0 (fpx cancels N)
    core = S @ _rot3(phi) @ M @ _rot3(-phi) @ N               # foreshortening, azimuth phi

    # recenter (translate only -- preserves the absolute compression) so the crop's
    # projector bbox sits at a random interior point. At extreme obliquity the footprint
    # can exceed the frame; the out-of-[0,1] tail is then genuinely invalid (projector
    # shadow / out-of-frame), as in a real capture.
    corners = np.array([[0, 0, 1], [W - 1, 0, 1], [0, H - 1, 1], [W - 1, H - 1, 1]], float)
    pc = (core @ corners.T).T
    pc = pc[:, :2] / pc[:, 2:3]
    lo, hi = pc.min(0), pc.max(0)
    margin = np.maximum(1.0 - (hi - lo), 0.0)
    t = -lo + margin * rng.uniform(0.0, 1.0, 2)              # translate bbox into [0,1]
    # LEFT-multiply by the translation (post-perspective-division shift); modifying
    # core[:2,2] directly would be divided by the perspective denominator and mis-place it.
    T = np.array([[1.0, 0.0, t[0]], [0.0, 1.0, t[1]], [0.0, 0.0, 1.0]])
    return T @ core, float(th)


def grid_from_homography(Hinv: np.ndarray, hw: tuple[int, int], device, ss: int = 1,
                         ) -> tuple[torch.Tensor, torch.Tensor]:
    """Map every camera pixel of an ``hw`` crop through ``Hinv`` to its projector-normalized
    coordinate. Returns ``(coords (ss*H, ss*W, 2) in [0,1], valid bool)`` -- coords is the
    exact GT correspondence; valid is where it lands inside the projector frame.

    ``ss>1`` subdivides each output pixel into ``ss x ss`` sub-samples centered on the pixel
    (positions ``(m+0.5)/ss - 0.5``, which reduces to the integer pixel index at ``ss=1`` --
    the renderer's pixel-index convention, preserving GT-exact-by-construction). Averaging the
    pattern over these sub-samples (in :meth:`ProxyBatcher.sample`) approximates the camera's
    area integral over its pixel footprint, the anti-aliasing a point sample misses -- and the
    footprint spans more projector content under grazing compression, so the blur scales with
    obliquity for free (the M1 fidelity fix)."""
    H, W = hw
    Ht = torch.tensor(Hinv, dtype=torch.float32, device=device)
    ys = (torch.arange(H * ss, device=device, dtype=torch.float32) + 0.5) / ss - 0.5
    xs = (torch.arange(W * ss, device=device, dtype=torch.float32) + 0.5) / ss - 0.5
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    ones = torch.ones_like(gx)
    pix = torch.stack([gx, gy, ones], dim=-1)                 # (H,W,3)
    proj = pix @ Ht.T                                         # (H,W,3)
    coords = proj[..., :2] / proj[..., 2:3].clamp(min=1e-6)
    valid = ((coords[..., 0] >= 0) & (coords[..., 0] <= 1)
             & (coords[..., 1] >= 0) & (coords[..., 1] <= 1)
             & (proj[..., 2] > 0))
    return coords, valid


# --------------------------------------------------------------------------
# Synthetic shading + differentiable photometric stack
# --------------------------------------------------------------------------
def _shading_field(rng, B: int, hw: tuple[int, int], device) -> torch.Tensor:
    """A smooth per-crop (B,1,H,W) multiplicative shading field in ~[0.15,1]: random
    albedo level x a low-order brightness ramp (the plane's cos(N.L) x falloff, which is
    smooth across a single patch). Without it the proxy patch is flatter/brighter than a
    real capture and the pattern co-adapts to absent shading."""
    H, W = hw
    ys = torch.linspace(-1, 1, H, device=device)
    xs = torch.linspace(-1, 1, W, device=device)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    base = torch.tensor(rng.uniform(0.45, 1.0, (B, 1, 1, 1)), dtype=torch.float32, device=device)
    bx = torch.tensor(rng.uniform(-0.35, 0.35, (B, 1, 1, 1)), dtype=torch.float32, device=device)
    by = torch.tensor(rng.uniform(-0.35, 0.35, (B, 1, 1, 1)), dtype=torch.float32, device=device)
    field = base * (1.0 + bx * gx + by * gy)
    return field.clamp(0.15, 1.0)


def _gauss_kernel(sigma: torch.Tensor, radius: int) -> torch.Tensor:
    """Per-sample 2D Gaussian kernels ``(B,1,k,k)`` from per-sample ``sigma (B,)``."""
    dev = sigma.device
    ax = torch.arange(-radius, radius + 1, device=dev, dtype=torch.float32)
    yy, xx = torch.meshgrid(ax, ax, indexing="ij")
    r2 = (xx ** 2 + yy ** 2)[None]                            # (1,k,k)
    k = torch.exp(-r2 / (2.0 * sigma[:, None, None] ** 2))
    k = k / k.sum(dim=(-1, -2), keepdim=True)
    return k[:, None]                                        # (B,1,k,k)


def _ste_clamp(x: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
    """Straight-through clamp: forward clips, gradient is identity (so the pattern still
    gets gradient at blown highlights instead of a dead zone)."""
    return x + (x.clamp(lo, hi) - x).detach()


def differentiable_augment(x: torch.Tensor, rng, base_psf: float = 0.0) -> torch.Tensor:
    """Torch port of :func:`lux.proj_net._augment_crop` (lines 283-320): re-form the clean
    proxy capture through the same **physically-ordered** image-formation model, kept
    differentiable so gradient flows back to the pattern (and so the pattern is not
    optimized for a cleaner-than-real world -- the guardrail in ``cliff_plan.md`` line 101).

    Order/literals mirror the numpy version: exposure gain -> optical PSF (defocus) -> shot
    noise (sqrt(signal), reparameterized so grad flows through the std) -> read noise ->
    saturate/clip -> response curve (gamma) -> 8-bit quantize. ``round``/``clamp`` are zero-
    gradient, so quantization and saturation use straight-through estimators. Each stage is
    probabilistically gated per-sample (clean crops still appear, else the subpixel ceiling caps).
    Input/return ``(B,1,H,W)`` in [0,1]."""
    B = x.shape[0]
    dev = x.device

    def rscalar(lo, hi, shape=(B, 1, 1, 1)):
        return torch.tensor(rng.uniform(lo, hi, shape), dtype=torch.float32, device=dev)

    def gate(p):
        return torch.tensor(rng.random((B, 1, 1, 1)) < p, device=dev)

    x = x * rscalar(0.55, 1.6)                                # exposure / gain (linear)

    # intrinsic camera PSF (MTF + projector defocus): a small ALWAYS-ON blur (vs the
    # probabilistic defocus below). Fixed in camera px, so it is harmless to the ~20px
    # frontal cells but erodes the ~5px grazing-compressed cells -- the obliquity-coupled
    # degradation the proxy was missing at M1 (complements the area-integration supersample).
    if base_psf > 0:
        ker = _gauss_kernel(torch.full((B,), base_psf, device=dev), radius=5)
        xp = F.pad(x, (5,) * 4, mode="reflect")
        x = F.conv2d(xp.reshape(1, B, *xp.shape[-2:]), ker, groups=B).reshape(B, 1, *x.shape[-2:])

    # optical PSF (defocus): per-sample Gaussian blur, depthwise grouped conv.
    sigma = rscalar(0.4, 1.6, (B,)).clamp(min=0.2)
    radius = 5
    ker = _gauss_kernel(sigma, radius)                        # (B,1,k,k)
    xp = F.pad(x, (radius,) * 4, mode="reflect")
    blurred = F.conv2d(xp.reshape(1, B, *xp.shape[-2:]), ker, groups=B).reshape(B, 1, *x.shape[-2:])
    x = torch.where(gate(0.55), blurred, x)

    # shot (photon) noise: std proportional to sqrt(signal); reparameterized.
    full_well = rscalar(30.0, 500.0)
    shot = torch.randn_like(x) * torch.sqrt(x.clamp(min=0.0) / full_well)
    x = torch.where(gate(0.85), x + shot, x)

    # read noise: additive, signal-independent
    read = torch.randn_like(x) * rscalar(0.002, 0.025)
    x = torch.where(gate(0.8), x + read, x)

    # saturate / highlight clip (STE so highlights still pass gradient)
    x = _ste_clamp(x * rscalar(0.9, 1.35), 0.0, 1.0)
    x = x.clamp(min=1e-4) ** rscalar(0.75, 1.35)              # camera response curve (gamma)
    xq = torch.round(x * 255.0) / 255.0                       # 8-bit quantize (STE)
    x = x + (xq - x).detach()
    return _ste_clamp(x, 0.0, 1.0)


# --------------------------------------------------------------------------
# The batcher: pattern + homography + photometrics -> (cap, target, valid)
# --------------------------------------------------------------------------
class ProxyBatcher:
    """Synthesizes on-the-fly GPU batches matching the :func:`lux.proj_net.proj_loss`
    contract: ``cap (B,1,S,S)`` in [0,1] (with grad to the pattern), ``target (B,2,S,S)``
    projector-normalized [0,1] (detached GT), ``valid (B,1,S,S)`` float {0,1}.

    The pattern is supplied per call (``pattern_at`` = a generator's ``sample_at``), so the
    generator forward stays in the batch's graph. Not a ``torch.utils.data`` type: the data
    must carry gradient to the live generator, and there is no I/O latency to hide."""

    def __init__(self, proj_wh: tuple[int, int], crop: int = 256, device: str = "cuda",
                 obliq_deg: tuple[float, float] = (45.0, 75.0), augment: bool = True,
                 shade: bool = True, dens_range: tuple[float, float] = (0.00048, 0.00056),
                 ss: int = 4, base_psf: float = 1.0, grazing_floor: float = 0.12,
                 seed: int | None = None):
        self.proj_wh = tuple(proj_wh)
        self.crop = crop
        self.device = device
        self.obliq_deg = obliq_deg
        self.augment = augment
        self.shade = shade
        self.dens_range = dens_range
        self.ss = ss                  # area-integration supersample (anti-aliasing)
        self.base_psf = base_psf      # always-on intrinsic camera PSF (px)
        self.grazing_floor = grazing_floor   # signal ~ max(cos(theta), floor): grazing irradiance falloff
        self.rng = np.random.default_rng(seed)

    def sample(self, batch: int, pattern_at, hw: tuple[int, int] | None = None,
               obliq_deg: tuple[float, float] | None = None, mtf=None):
        """Build one batch. ``pattern_at`` is a callable ``coords (B,H,W,2) -> (B,1,H,W)``
        (e.g. ``generator.sample_at``). ``hw`` overrides the crop size (used for full-frame
        eval). ``mtf=(sigma_mtf,sigma_def)`` selects the analytic anisotropic-Gaussian blur
        (ss forced to 1, applied inside ``pattern_at``) instead of the footprint supersample --
        the cheap differentiable path for carrier co-design (Build 4)."""
        H, W = hw or (self.crop, self.crop)
        band = obliq_deg or self.obliq_deg
        ss = 1 if mtf is not None else self.ss
        gt_l, valid_l, cap_l, th_l = [], [], [], []
        for _ in range(batch):
            Hinv, th = sample_homography_inv(self.rng, (H, W), band, self.dens_range)
            cgt, v = grid_from_homography(Hinv, (H, W), self.device, ss=1)   # GT at pixel centres
            gt_l.append(cgt)
            valid_l.append(v)
            th_l.append(th)
            cap_l.append(grid_from_homography(Hinv, (H, W), self.device, ss=ss)[0])  # supersampled
        coords = torch.stack(gt_l)                            # (B,H,W,2)
        valid = torch.stack(valid_l)[:, None].float()         # (B,1,H,W)
        ccap = torch.stack(cap_l)                             # (B,ss*H,ss*W,2)

        pat = pattern_at(ccap, mtf) if mtf is not None else pattern_at(ccap)   # (B,1,ss*H,ss*W) w/ grad
        if ss > 1:                                            # area integral over the pixel footprint
            pat = F.avg_pool2d(pat, ss)
        # grazing irradiance falloff: Lambertian signal ~ cos(theta), so grazing patches are dimmer
        # -> the signal-dependent shot noise in differentiable_augment makes them genuinely noisier
        # (a physical, compression-coupled hardener, not unphysical blur).
        falloff = torch.tensor([max(np.cos(t), self.grazing_floor) for t in th_l],
                               dtype=torch.float32, device=self.device).view(batch, 1, 1, 1)
        pat = pat * falloff
        if self.shade:
            pat = pat * _shading_field(self.rng, batch, (H, W), self.device)
        cap = (differentiable_augment(pat, self.rng, base_psf=self.base_psf)
               if self.augment else pat.clamp(0, 1))

        target = coords.permute(0, 3, 1, 2).contiguous().detach()   # (B,2,H,W)
        target = torch.where(valid.bool(), target, torch.zeros_like(target))
        return cap, target, valid


# --------------------------------------------------------------------------
# In-silico eval bank (adapter for scripts/train_proj_net.py:evaluate)
# --------------------------------------------------------------------------
class EvalBank:
    """A fixed, seeded set of full-frame oblique-plane samples, exposing ``.full(i)`` and
    ``.proj_wh`` so the existing :func:`scripts.train_proj_net.evaluate` /
    :func:`lux.proj_net.predict_tiled` run unchanged. Each sample also carries its obliquity
    so results can be banded (mirroring ``scripts/eval_hemisphere.py``)."""

    def __init__(self, pattern_at_np, proj_wh: tuple[int, int], hw=(1080, 1920),
                 bands=((0, 45), (45, 60), (60, 75)), per_band: int = 8,
                 device: str = "cpu", augment: bool = True,
                 dens_range: tuple[float, float] = (0.00048, 0.00056),
                 ss: int = 4, base_psf: float = 1.0, grazing_floor: float = 0.12, seed: int = 12345):
        self.proj_wh = tuple(proj_wh)
        self.hw = hw
        self.device = device
        self.augment = augment
        self.pattern_at = pattern_at_np
        self.bands = bands
        self.ss = ss
        self.base_psf = base_psf
        self.grazing_floor = grazing_floor
        rng = np.random.default_rng(seed)
        self.specs, self.obliq = [], []
        for lo, hi in bands:
            for _ in range(per_band):
                Hinv, th = sample_homography_inv(rng, hw, (lo, hi), dens_range)
                self.specs.append(Hinv)
                self.obliq.append(np.rad2deg(th))
        self._rng = np.random.default_rng(seed + 1)

    def __len__(self):
        return len(self.specs)

    def band_label(self, obliq_deg: float) -> str:
        for lo, hi in self.bands:
            if lo <= obliq_deg < hi or hi == self.bands[-1][1] and obliq_deg <= hi:
                return f"{lo}-{hi}"
        return f">{self.bands[-1][1]}"

    def full(self, i: int):
        """Full-frame capture ``(H,W)`` float [0,1] and gt ``(H,W,2)`` projector px with NaN."""
        H, W = self.hw
        coords, valid = grid_from_homography(self.specs[i], (H, W), self.device, ss=1)
        ccap, _ = grid_from_homography(self.specs[i], (H, W), self.device, ss=self.ss)
        pat = self.pattern_at(ccap[None])                     # (1,1,ss*H,ss*W)
        if self.ss > 1:
            pat = F.avg_pool2d(pat, self.ss)
        falloff = max(np.cos(np.deg2rad(self.obliq[i])), self.grazing_floor)
        field = _shading_field(self._rng, 1, (H, W), self.device) * falloff
        cap = pat * field
        if self.augment:
            cap = differentiable_augment(cap, self._rng, base_psf=self.base_psf)
        img = cap[0, 0].detach().cpu().numpy().clip(0, 1)
        gt = coords.detach().cpu().numpy() * np.asarray(self.proj_wh, np.float32)
        v = valid.detach().cpu().numpy()
        gt = np.where(v[..., None], gt, np.nan).astype(np.float32)
        return img, gt


# --------------------------------------------------------------------------
# Quadrature loss (Build 4): per-carrier circular loss for the continuous-phase head
# --------------------------------------------------------------------------
def quad_loss(pred: torch.Tensor, target: torch.Tensor, valid: torch.Tensor, gen,
              proj_wh: tuple[int, int], carrier_weights=None):
    """Per-carrier circular loss for the continuous-phase (quadrature) head.

    ``pred`` (B, 2*(nu+nv)+1, H, W): u-carriers' (cos,sin) then v-carriers', then a validity logit.
    Each carrier is supervised by **MSE of (cos,sin) toward the unit vector at the true phase**
    ``phi*_k = 2pi*(coord_px)/period_k`` -- which drives the vector to the unit circle where the
    phase is learnable and lets its magnitude **collapse toward 0 where the phase is ambiguous**
    (averaging unit vectors over a noisy posterior shrinks the mean): so the unnormalized magnitude
    becomes the floating confidence the consensus vote weights by, with no normalization and no
    global-collapse risk. ``coord_weight`` is the late, low-weight soft-vote coordinate-L1 refiner
    -- **0 here** (phase-only baseline): unwrap correctness must come from the circular loss, and a
    soft-vote in the wrong period basin would push phases the wrong way (gate it behind coarse
    convergence before turning it on).

    ``carrier_weights`` (len nu+nv, u-block then v-block) scales each carrier's circular loss for the
    coarse-first curriculum (upweight coarse early so the unwrap backbone is trustworthy before fine
    carriers matter). Returns ``(total, u_aligns, v_aligns, bce)`` where ``*_aligns`` are PER-CARRIER
    mean cos(phase error) tensors (1=perfect) -- the coarse carrier's value drives the coord-L1 gate
    and the per-carrier u-vs-v trace diagnoses the row-deficit; periods are ordered fine→coarse so the
    coarse carrier is the last entry."""
    W, H = proj_wh
    nu, nv = gen.n_u, gen.n_v
    m = valid[:, 0]
    nval = m.sum().clamp(min=1.0)
    two_pi = 2 * np.pi
    if carrier_weights is None:
        carrier_weights = [1.0] * (nu + nv)
    phase_loss = pred.new_zeros(())
    u_aligns, v_aligns = [], []
    for k, p in enumerate(gen.periods_u):
        ph = two_pi * (target[:, 0] * W) / p
        tc, ts = torch.cos(ph), torch.sin(ph)
        c, s = pred[:, 2 * k], pred[:, 2 * k + 1]
        phase_loss = phase_loss + carrier_weights[k] * (((c - tc) ** 2 + (s - ts) ** 2) * m).sum() / nval
        u_aligns.append(((c * tc + s * ts) / torch.sqrt(c * c + s * s + 1e-6) * m).sum().detach() / nval)
    for k, p in enumerate(gen.periods_v):
        ph = two_pi * (target[:, 1] * H) / p
        tc, ts = torch.cos(ph), torch.sin(ph)
        b = 2 * nu + 2 * k
        c, s = pred[:, b], pred[:, b + 1]
        phase_loss = phase_loss + carrier_weights[nu + k] * (((c - tc) ** 2 + (s - ts) ** 2) * m).sum() / nval
        v_aligns.append(((c * tc + s * ts) / torch.sqrt(c * c + s * s + 1e-6) * m).sum().detach() / nval)
    bce = F.binary_cross_entropy_with_logits(pred[:, -1], m)
    total = phase_loss + bce            # coord-L1 added in the loop, gated behind coarse convergence
    return total, torch.stack(u_aligns), torch.stack(v_aligns), bce.detach()
