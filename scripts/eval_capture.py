#!/usr/bin/env python3
"""Real-capture eval: build a reference correspondence, score the net's M-array.

The hemisphere bench (`scripts/eval_hemisphere.py`) ported to real hardware. For a
scene captured by `scripts/capture_app.py` (a `captures/<scene>/` folder with a
`graycode/`, a `phaseshift/`, and a `marray/` set of the *same* static scene through
the *same* fixed camera):

  1. build a reference projector-**column** map from the multi-shot sets:
       - `graycode`   : integer column (floor ~1/sqrt(12) = 0.29 px RMS),
       - `phaseshift` : dual-frequency sub-pixel column (self-unwrapping),
       - `hybrid`     : Gray-coded phase shifting (DEFAULT) — Gray-code integer
         column picks the fringe order (robust, no unwrap jumps), the high-frequency
         phase gives the sub-pixel offset. Best of both: phase-shift precision with
         Gray-code robustness, and no integer quantization;
  2. run the net on the single M-array capture -> predicted (column, row) + conf;
  3. compare on the column (`du`), on pixels the reference resolved.

Both sets share the fixed camera + static scene, so they are pixel-aligned — the
comparison needs no calibration. Metrics mirror the hemisphere bench: validity IoU,
u-bin accuracy, median |du|, median |du| within correct bins, p95 |du|, a confidence
sweep, and (for sub-pixel references) the Gray-code-vs-reference agreement, which
empirically exposes the integer quantization floor.

    python scripts/eval_capture.py --captures captures/test2 --ckpt checkpoints/proj_net.pt
    python scripts/eval_capture.py --captures captures/test2 --reference hybrid --min-conf 0.5

NOTE: graycode/phaseshift are VERTICAL patterns -> projector **column** (`du`). Add a
horizontal `graycode_h/` set (gen_patterns.py) and its row reference is decoded too,
enabling `dv` and full **(u,v)** bin accuracy. Outputs land in `<captures>/eval_<ckpt-stem>/`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from lux import io  # noqa: E402
from lux.methods.graycode import GrayCodeMethod  # noqa: E402
from lux.methods.phaseshift import PhaseShiftMethod  # noqa: E402
from lux.proj_net import load_checkpoint, predict_full, predict_tiled, N_BINS_U, N_BINS_V  # noqa: E402


def _load_stack(d: Path) -> np.ndarray:
    """Load a capture set's frames (cap_*.png) as a grayscale (N, H, W) stack."""
    files = sorted(p for ext in ("*.png", "*.jpg", "*.jpeg") for p in d.glob(ext)
                   if not p.name.startswith("."))   # skip macOS ._* AppleDouble sidecars (exFAT)
    if not files:
        raise SystemExit(f"no capture frames in {d}")
    return np.stack([io.load_image(str(f), gray=True) for f in files], axis=0)


def hybrid_column(col_gc, phase_hi, amp_hi, proj_w, high_periods, amp_thresh=0.05):
    """Gray-coded phase shifting: the integer Gray-code column picks the fringe
    order (robust, since Gray error < 0.5 px << half a fringe), the high-frequency
    wrapped phase supplies the sub-pixel offset within the fringe."""
    P = proj_w / high_periods                       # fringe period (px)
    frac = phase_hi / (2 * np.pi)                    # within-fringe fraction [0,1)
    order = np.round(col_gc / P - frac)              # integer fringe index from Gray code
    col = (order + frac) * P
    valid = np.isfinite(col_gc) & (amp_hi > amp_thresh) & (col >= 0) & (col < proj_w)
    return np.where(valid, col, np.nan)


def metrics_on(ref, pred, mask, proj_dim, n_bins=N_BINS_U) -> dict:
    """One-axis metrics over ``mask`` pixels: bin accuracy, |error| medians, p95.
    Defaults to the column axis (``proj_w``, ``N_BINS_U``); pass ``proj_h`` +
    ``N_BINS_V`` for the row axis. (Keys keep the ``du`` names for both axes.)"""
    if not mask.any():
        return dict(n=0, bin_acc=0.0, med_du=np.nan, med_du_correct=np.nan, p95_du=np.nan)
    du = np.abs(pred[mask] - ref[mask])
    binw = proj_dim / n_bins
    rb = (ref[mask] // binw).clip(0, n_bins - 1)
    pb = (pred[mask] // binw).clip(0, n_bins - 1)
    correct = pb == rb
    return dict(
        n=int(mask.sum()),
        bin_acc=float(correct.mean()),
        med_du=float(np.median(du)),
        med_du_correct=float(np.median(du[correct])) if correct.any() else float("nan"),
        p95_du=float(np.percentile(du, 95)),
    )


def uv_bin_acc(ref_u, ref_v, pred_u, pred_v, mask, proj_w, proj_h) -> float:
    """Fraction of ``mask`` pixels whose u-bin AND v-bin both match the reference
    (full 2-D correspondence accuracy)."""
    if not mask.any():
        return 0.0
    bu, bv = proj_w / N_BINS_U, proj_h / N_BINS_V
    ru = (ref_u[mask] // bu).clip(0, N_BINS_U - 1)
    pu = (pred_u[mask] // bu).clip(0, N_BINS_U - 1)
    rv = (ref_v[mask] // bv).clip(0, N_BINS_V - 1)
    pv = (pred_v[mask] // bv).clip(0, N_BINS_V - 1)
    return float(((pu == ru) & (pv == rv)).mean())


def plot_overview(ref_u, pred_u, both, proj_w, out: Path, title: str, ref_label: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("(matplotlib not installed - skipped overview plot)")
        return
    du = np.full_like(ref_u, np.nan)
    du[both] = np.abs(pred_u[both] - ref_u[both])
    fig, ax = plt.subplots(2, 2, figsize=(14, 10))
    cmap = plt.cm.turbo.copy(); cmap.set_bad("0.1")
    ax[0, 0].imshow(np.ma.masked_invalid(ref_u), cmap=cmap, vmin=0, vmax=proj_w)
    ax[0, 0].set_title(f"reference column ({ref_label})")
    ax[0, 1].imshow(np.ma.masked_invalid(pred_u), cmap=cmap, vmin=0, vmax=proj_w)
    ax[0, 1].set_title("predicted column (net)")
    emap = plt.cm.turbo.copy(); emap.set_bad("0.1")
    im = ax[1, 0].imshow(np.ma.masked_invalid(du), cmap=emap, vmin=0, vmax=50)
    ax[1, 0].set_title("|du| (px, clipped at 50)")
    fig.colorbar(im, ax=ax[1, 0], fraction=0.046, pad=0.02)
    for a in (ax[0, 0], ax[0, 1], ax[1, 0]):
        a.set_xticks([]); a.set_yticks([])
    a = ax[1, 1]
    if both.any():
        a.hexbin(ref_u[both], pred_u[both], gridsize=80, bins="log", cmap="viridis",
                 extent=(0, proj_w, 0, proj_w))
        a.plot([0, proj_w], [0, proj_w], "w--", lw=1, alpha=0.7)
    a.set_xlim(0, proj_w); a.set_ylim(0, proj_w)
    a.set_xlabel(f"reference column ({ref_label}, px)"); a.set_ylabel("predicted column (px)")
    a.set_title("pred vs ref column (on-diagonal = correct)")
    fig.suptitle(title)
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"overview -> {out}")


def save_conf_map(conf, mask, out: Path) -> None:
    """Standalone net-confidence map with the turbo colormap + colorbar (0..1),
    over the scene ``mask``; abstained/background pixels are dark."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("(matplotlib not installed - skipped confidence map)")
        return
    c = np.where(mask, np.clip(conf, 0, 1), np.nan)
    cmap = plt.cm.turbo.copy(); cmap.set_bad("0.1")
    fig, ax = plt.subplots(figsize=(11, 6))
    im = ax.imshow(np.ma.masked_invalid(c), cmap=cmap, vmin=0, vmax=1)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("net confidence — joint min(conf_u, conf_v) (turbo)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02, label="confidence")
    fig.savefig(out / "confidence_turbo.png", dpi=120, bbox_inches="tight")
    print(f"confidence -> {out}/confidence_turbo.png")


def save_histograms(ref_u, pred, conf, both, out: Path, ref_label: str,
                    dv_ref=None) -> None:
    """Histograms of du (signed column error), dv (row error — N/A without a row
    reference), and net confidence, over the both-valid comparison pixels."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("(matplotlib not installed - skipped histograms)")
        return
    du = (pred[..., 0] - ref_u)[both]
    cf = conf[both]
    fig, ax = plt.subplots(1, 3, figsize=(18, 5))
    # du — signed column error; log-y so the off-by-bin (±32k px) shoulders show
    ax[0].hist(np.clip(du, -96, 96), bins=193, color="C0")
    ax[0].set_yscale("log"); ax[0].axvline(0, color="k", lw=0.6)
    for k in (-64, -32, 32, 64):
        ax[0].axvline(k, color="0.7", lw=0.5, ls=":")
    med = float(np.median(np.abs(du)))
    ax[0].set_title(f"du = pred - {ref_label} (px, clip ±96)   median|du|={med:.2f}")
    ax[0].set_xlabel("du (px)   dotted = ±1,±2 bin (32px)")
    # dv — needs a row reference
    if dv_ref is not None:
        sel_v = both & np.isfinite(dv_ref)
        dv = (pred[..., 1] - dv_ref)[sel_v]
        ax[1].hist(np.clip(dv, -96, 96), bins=193, color="C1")
        ax[1].set_yscale("log"); ax[1].axvline(0, color="k", lw=0.6)
        for k in (-60, -30, 30, 60):
            ax[1].axvline(k, color="0.7", lw=0.5, ls=":")
        med_dv = float(np.median(np.abs(dv))) if dv.size else float("nan")
        ax[1].set_title(f"dv = pred - graycode_h row (px, clip ±96)   median|dv|={med_dv:.2f}")
        ax[1].set_xlabel("dv (px)   dotted = ±1,±2 bin (30px)")
    else:
        ax[1].text(0.5, 0.5, "dv (row error)\nN/A: vertical patterns only\n"
                   "(no horizontal/row reference)", ha="center", va="center",
                   fontsize=12, color="0.5", transform=ax[1].transAxes)
        ax[1].set_facecolor("0.96"); ax[1].set_xticks([]); ax[1].set_yticks([])
    # confidence
    ax[2].hist(cf, bins=100, range=(0, 1), color="C2")
    ax[2].set_title(f"net confidence   median={float(np.median(cf)):.2f}")
    ax[2].set_xlabel("confidence")
    fig.suptitle(f"error / confidence histograms  —  {out.parent.name} vs {ref_label}")
    fig.savefig(out / "histograms.png", dpi=120, bbox_inches="tight")
    print(f"histograms -> {out}/histograms.png")


def save_uv_grid(col_gc, pred, conf, white_valid, min_conf, proj_w, proj_h,
                 out: Path, row_ref=None) -> None:
    """2x3 grid: rows = x (column) / y (row); columns = graycode | neural | diff.
    Coordinate maps use turbo; the diff uses a diverging map. The y row has no
    graycode reference (vertical patterns encode columns only), so its graycode +
    diff cells are flagged N/A."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("(matplotlib not installed - skipped uv grid)")
        return
    u_pred, v_pred = pred[..., 0], pred[..., 1]
    dmask = white_valid & (conf >= min_conf) & np.isfinite(col_gc) & np.isfinite(u_pred)
    du = np.full(col_gc.shape, np.nan, dtype=np.float64)
    du[dmask] = u_pred[dmask] - col_gc[dmask]
    vmax = float(np.clip(np.nanpercentile(np.abs(du[dmask]), 95) if dmask.any() else 1.0, 5, 100))

    turbo = plt.cm.turbo.copy(); turbo.set_bad("0.12")
    cool = plt.cm.coolwarm.copy(); cool.set_bad("0.12")
    mi = np.ma.masked_invalid

    def show(ax, img, cmap, vmn, vmx, title, cbar_label):
        im = ax.imshow(mi(img), cmap=cmap, vmin=vmn, vmax=vmx)
        ax.set_title(title); ax.set_xticks([]); ax.set_yticks([])
        ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.02, label=cbar_label)

    def na(ax, msg):
        ax.text(0.5, 0.5, msg, ha="center", va="center", fontsize=12, color="0.6",
                transform=ax.transAxes)
        ax.set_facecolor("0.12"); ax.set_xticks([]); ax.set_yticks([])

    fig, ax = plt.subplots(2, 3, figsize=(18, 9))
    # row 0 — x / column
    show(ax[0, 0], col_gc, turbo, 0, proj_w, "graycode — column (x)", "px")
    show(ax[0, 1], np.where(np.isfinite(u_pred), u_pred, np.nan), turbo, 0, proj_w,
         "neural — column (x)", "px")
    show(ax[0, 2], du, cool, -vmax, vmax, f"diff: neural - graycode (x)  [±{vmax:.0f}]", "px")
    # row 1 — y / row (filled when a horizontal Gray-code reference is supplied)
    if row_ref is not None:
        show(ax[1, 0], row_ref, turbo, 0, proj_h, "graycode_h — row (y)", "px")
    else:
        na(ax[1, 0], "graycode — row (y)\nN/A: vertical patterns\n(column-only capture)")
    show(ax[1, 1], np.where(np.isfinite(v_pred), v_pred, np.nan), turbo, 0, proj_h,
         "neural — row (y)", "px")
    if row_ref is not None:
        dmask_v = white_valid & (conf >= min_conf) & np.isfinite(row_ref) & np.isfinite(v_pred)
        dv = np.full(row_ref.shape, np.nan, dtype=np.float64)
        dv[dmask_v] = v_pred[dmask_v] - row_ref[dmask_v]
        vmax_v = float(np.clip(np.nanpercentile(np.abs(dv[dmask_v]), 95)
                               if dmask_v.any() else 1.0, 5, 100))
        show(ax[1, 2], dv, cool, -vmax_v, vmax_v,
             f"diff: neural - graycode_h (y)  [±{vmax_v:.0f}]", "px")
    else:
        na(ax[1, 2], "diff — row (y)\nN/A: no row reference\n(needs horizontal patterns)")
    fig.suptitle(f"u/v coordinates: graycode vs neural  —  {out.parent.name}")
    fig.savefig(out / "uv_grid.png", dpi=120, bbox_inches="tight")
    print(f"uv grid -> {out}/uv_grid.png")


def save_du_maps(ref_u, pred_u, valid, proj_w, out: Path, ref_label: str,
                 col_gc=None) -> np.ndarray:
    """The requested du divergence maps: signed (diverging) + |du| (turbo), masked
    to ``valid``. When the reference is sub-pixel, a third turbo panel shows the
    reference's own floor |Gray code - reference| on the same scale, so net error
    and quantization floor read side by side. Writes du_signed.npy + diff_du.png."""
    m = valid & np.isfinite(ref_u) & np.isfinite(pred_u)
    diff = np.full(ref_u.shape, np.nan, dtype=np.float64)
    diff[m] = pred_u[m] - ref_u[m]
    np.save(out / "du_signed.npy", diff)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("(matplotlib not installed - skipped du plot)")
        return m
    vmax = float(np.clip(np.nanpercentile(np.abs(diff[m]), 95) if m.any() else 1.0, 5, 100))
    show_floor = col_gc is not None and ref_label != "graycode"
    ncol = 3 if show_floor else 2
    fig, ax = plt.subplots(1, ncol, figsize=(7.5 * ncol, 6))
    dmap = plt.cm.coolwarm.copy(); dmap.set_bad("0.12")
    im0 = ax[0].imshow(np.ma.masked_invalid(diff), cmap=dmap, vmin=-vmax, vmax=vmax)
    ax[0].set_title(f"signed du: net - {ref_label} (px)  [±{vmax:.0f}]")
    fig.colorbar(im0, ax=ax[0], fraction=0.046, pad=0.02)
    tmap = plt.cm.turbo.copy(); tmap.set_bad("0.12")
    im1 = ax[1].imshow(np.ma.masked_invalid(np.abs(diff)), cmap=tmap, vmin=0, vmax=vmax)
    ax[1].set_title(f"|du|: net vs {ref_label} (px)  [0,{vmax:.0f}]")
    fig.colorbar(im1, ax=ax[1], fraction=0.046, pad=0.02)
    if show_floor:
        floor = np.full(ref_u.shape, np.nan, dtype=np.float64)
        mf = m & np.isfinite(col_gc)
        floor[mf] = np.abs(col_gc[mf] - ref_u[mf])
        im2 = ax[2].imshow(np.ma.masked_invalid(floor), cmap=tmap, vmin=0, vmax=vmax)
        ax[2].set_title(f"reference floor: |Gray code - {ref_label}| (px)")
        fig.colorbar(im2, ax=ax[2], fraction=0.046, pad=0.02)
    for a in ax:
        a.set_xticks([]); a.set_yticks([])
    fig.savefig(out / "diff_du.png", dpi=120, bbox_inches="tight")
    print(f"du maps -> {out}/diff_du.png")
    return m


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--captures", required=True,
                    help="scene folder with graycode/ phaseshift/ marray/ subdirs")
    ap.add_argument("--ckpt", default="checkpoints/proj_net.pt")
    ap.add_argument("--reference", choices=("graycode", "phaseshift", "hybrid"),
                    default="hybrid", help="reference correspondence (default hybrid: "
                    "Gray-coded phase shifting)")
    ap.add_argument("--graycode-set", default="graycode")
    ap.add_argument("--graycode-h-set", default="graycode_h",
                    help="horizontal Gray-code set -> row reference (dv); optional")
    ap.add_argument("--phaseshift-set", default="phaseshift")
    ap.add_argument("--ps-shifts", type=int, default=4, help="phase shifts per frequency")
    ap.add_argument("--ps-periods", type=int, default=16, help="high-frequency fringe count")
    ap.add_argument("--marray-set", default="marray")
    ap.add_argument("--marray-frame", default="cap_pat_00.png")
    ap.add_argument("--white-thresh", type=float, default=0.10,
                    help="valid-pixel mask = (white capture > this); the du maps "
                         "are clipped to it. white-black (amplitude) is more robust "
                         "to ambient and is what Gray code uses internally.")
    ap.add_argument("--min-conf", type=float, default=0.0,
                    help="also gate the du maps on net confidence >= this "
                         "(0 = white-valid only; ~0.5 removes the low-conf outlier tail)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--tiled", action="store_true",
                    help="stitch full-frame inference from 256-px tiles (training crop "
                         "size). Auto-enabled for attn checkpoints (global attention "
                         "collapses at full-frame token counts); also closes the conv "
                         "row deficit.")
    ap.add_argument("--tile-overlap", type=int, default=64,
                    help="overlapping center-crop stitch: each pixel from the tile it's "
                         "most central in (geometry, not softmax — honest for the metric), "
                         "frame reflect-padded. 64 -> stride 192, ~1.4x passes, seamless; "
                         "0 = hard margin-crop.")
    ap.add_argument("--out", default=None, help="results dir (default <captures>/eval_<ckpt-stem>)")
    args = ap.parse_args()

    cap = Path(args.captures)
    if args.device is None:
        import torch
        args.device = ("mps" if torch.backends.mps.is_available()
                       else "cuda" if torch.cuda.is_available() else "cpu")
    model, proj_wh = load_checkpoint(args.ckpt, device=args.device)
    use_tiled = args.tiled or getattr(model, "arch", "conv") == "attn"
    print(f"inference: {'tiled-256' if use_tiled else 'full-frame'} "
          f"(arch={getattr(model, 'arch', 'conv')})")
    proj_w, proj_h = proj_wh

    # 1. build the reference column ------------------------------------------------
    gc = _load_stack(cap / args.graycode_set)
    col_gc, conf_gc = GrayCodeMethod().decode_columns(gc, proj_w)
    white, black = gc[-2], gc[-1]

    col_ps = col_hybrid = None
    floor_med = floor_p95 = float("nan")
    ps_dir = cap / args.phaseshift_set
    if ps_dir.is_dir():
        ps = _load_stack(ps_dir)
        psm = PhaseShiftMethod(shifts=args.ps_shifts, high_periods=args.ps_periods)
        col_ps, _ = psm.decode_columns(ps, proj_w)
        phase_hi, amp_hi = psm.fringe_phase(ps)
        col_hybrid = hybrid_column(col_gc, phase_hi, amp_hi, proj_w, args.ps_periods)
        # empirical integer-Gray quantization floor: Gray code vs the sub-pixel ref
        mref = np.isfinite(col_gc) & np.isfinite(col_hybrid)
        if mref.any():
            d = np.abs(col_gc[mref] - col_hybrid[mref])
            floor_med, floor_p95 = float(np.median(d)), float(np.percentile(d, 95))

    refs = {"graycode": col_gc, "phaseshift": col_ps, "hybrid": col_hybrid}
    ref_u = refs[args.reference]
    if ref_u is None:
        print(f"[warn] no '{args.phaseshift_set}/' set — falling back to graycode reference")
        args.reference, ref_u = "graycode", col_gc
    ref_valid = np.isfinite(ref_u)

    # valid-pixel mask from the white (all-on) capture — calibration-free scene mask.
    white_valid = white > args.white_thresh
    wv_iou = float((white_valid & ref_valid).sum() / max((white_valid | ref_valid).sum(), 1))

    # 2. net prediction on the M-array capture -------------------------------------
    marray = io.load_image(str(cap / args.marray_set / args.marray_frame), gray=True)
    if marray.shape != ref_u.shape:
        raise SystemExit(f"marray {marray.shape} != reference {ref_u.shape}: same camera/res?")
    if use_tiled:
        pred, conf_u, conf_v = predict_tiled(model, marray, proj_wh, device=args.device,
                                             overlap=args.tile_overlap, conf_per_axis=True,
                                             select="center")
    else:
        pred, conf_u, conf_v = predict_full(model, marray, proj_wh, device=args.device,
                                            conf_per_axis=True)
    # Joint correspondence confidence for the visuals/saves; the per-axis sweeps
    # below gate column on conf_u and row on conf_v (each axis on its own softmax).
    conf = np.minimum(conf_u, conf_v)
    pred_u, pred_v = pred[..., 0], pred[..., 1]
    pred_valid = np.isfinite(pred_u)

    # optional ROW reference from a horizontal Gray-code set -> dv + full (u,v)
    ref_v = None
    gch_dir = cap / args.graycode_h_set
    if gch_dir.is_dir():
        ref_v, _ = GrayCodeMethod().decode_rows(_load_stack(gch_dir), proj_h)
        if ref_v.shape != pred_v.shape:
            print(f"[warn] {args.graycode_h_set} {ref_v.shape} != pred {pred_v.shape}; skipping row")
            ref_v = None

    # 3. compare on the column, over pixels the reference resolved -----------------
    both = ref_valid & pred_valid
    union = ref_valid | pred_valid
    iou = float(both.sum() / max(union.sum(), 1))
    coverage = float(both.sum() / max(ref_valid.sum(), 1))
    m = metrics_on(ref_u, pred_u, both, proj_w)

    print(f"\n{args.ckpt}  on  {cap}   (reference: {args.reference})")
    print(f"  reference valid:       {int(ref_valid.sum()):>9d} valid px  "
          f"({100*ref_valid.mean():.1f}% of frame)")
    if np.isfinite(floor_med):
        print(f"  Gray code vs {args.reference:<9s}: median {floor_med:.2f} px, "
              f"p95 {floor_p95:.2f} px  (≈ integer-Gray quantization floor)")
    print(f"  white>{args.white_thresh:.2f} mask:       {int(white_valid.sum()):>9d} valid px  "
          f"({100*white_valid.mean():.1f}% of frame), IoU vs ref-valid {wv_iou:.3f}")
    print(f"  net prediction:        {int(pred_valid.sum()):>9d} valid px")
    print(f"  validity IoU:          {iou:.3f}")
    print(f"  coverage (ref∩pred):   {coverage:.3f}")
    print(f"  u-bin accuracy:        {100*m['bin_acc']:.1f}%   (binw {proj_w/N_BINS_U:.0f}px)")
    print(f"  median |du|:           {m['med_du']:.2f} px")
    print(f"  median |du| ✓bin:      {m['med_du_correct']:.2f} px")
    print(f"  p95 |du|:              {m['p95_du']:.1f} px")
    uv_acc = float("nan")
    if ref_v is not None:
        rv_valid = np.isfinite(ref_v)
        both_v = rv_valid & pred_valid
        both_uv = ref_valid & rv_valid & pred_valid
        mv = metrics_on(ref_v, pred_v, both_v, proj_h, N_BINS_V)
        uv_acc = uv_bin_acc(ref_u, ref_v, pred_u, pred_v, both_uv, proj_w, proj_h)
        print(f"  --- row (dv) via {args.graycode_h_set} ---")
        print(f"  row reference valid:   {int(rv_valid.sum()):>9d} valid px  "
              f"({100*rv_valid.mean():.1f}% of frame)")
        print(f"  v-bin accuracy:        {100*mv['bin_acc']:.1f}%   (binw {proj_h/N_BINS_V:.0f}px)")
        print(f"  median |dv|:           {mv['med_du']:.2f} px")
        print(f"  median |dv| ✓bin:      {mv['med_du_correct']:.2f} px")
        print(f"  p95 |dv|:              {mv['p95_du']:.1f} px")
        print(f"  full (u,v) bin acc:    {100*uv_acc:.1f}%   (column AND row bin correct)")
    else:
        print(f"  (dv/row: no '{args.graycode_h_set}/' set — column-only, not scored)")

    # confidence-threshold sweeps: trade coverage for precision. Each axis is
    # gated on ITS OWN softmax max (conf_u / conf_v) — the column softmax is blind
    # to row-only failures, so gating the row on conf_u leaves its p95 untouched.
    has_v = ref_v is not None
    thresholds = (0.0, 0.3, 0.5, 0.7, 0.9, 0.95)
    print(f"\n  column (du) sweep (mask conf_u >= t):")
    print(f"  {'t':>5s} {'coverage':>9s} {'bin acc':>8s} {'med|du|':>8s} {'p95|du|':>8s}")
    sweep = []
    for t in thresholds:
        mk = both & (conf_u >= t)
        cov = float(mk.sum() / max(ref_valid.sum(), 1))
        mm = metrics_on(ref_u, pred_u, mk, proj_w)
        sweep.append({"t": t, "coverage": cov, **mm})
        print(f"  {t:>5.2f} {cov:>9.3f} {100*mm['bin_acc']:>7.1f}% "
              f"{mm['med_du']:>7.2f}px {mm['p95_du']:>7.1f}px")

    row_sweep, uv_sweep = [], []
    if has_v:
        print(f"\n  row (dv) sweep (mask conf_v >= t):")
        print(f"  {'t':>5s} {'coverage':>9s} {'bin acc':>8s} {'med|dv|':>8s} {'p95|dv|':>8s}")
        for t in thresholds:
            mk = both_v & (conf_v >= t)
            cov = float(mk.sum() / max(rv_valid.sum(), 1))
            mm = metrics_on(ref_v, pred_v, mk, proj_h, N_BINS_V)
            row_sweep.append({"t": t, "coverage": cov, **mm})
            print(f"  {t:>5.2f} {cov:>9.3f} {100*mm['bin_acc']:>7.1f}% "
                  f"{mm['med_du']:>7.2f}px {mm['p95_du']:>7.1f}px")

        print(f"\n  joint (u,v) sweep (mask min(conf_u,conf_v) >= t):")
        print(f"  {'t':>5s} {'coverage':>9s} {'uv acc':>8s}")
        for t in thresholds:
            mk = both_uv & (conf >= t)
            cov = float(mk.sum() / max(both_uv.sum(), 1))
            uva = uv_bin_acc(ref_u, ref_v, pred_u, pred_v, mk, proj_w, proj_h)
            uv_sweep.append({"t": t, "coverage": cov, "uv_bin_acc": uva})
            print(f"  {t:>5.2f} {cov:>9.3f} {100*uva:>7.1f}%")

    out = Path(args.out or cap / f"eval_{Path(args.ckpt).stem}")
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "ref_col.npy", ref_u)
    np.save(out / "pred.npy", pred)
    np.save(out / "conf.npy", conf)
    if ref_v is not None:
        np.save(out / "ref_row.npy", ref_v)
    # net prediction quick-look: R=col(x), G=row(y), B=confidence (invalid->black)
    io.save_image(str(out / "pred_uv_conf.png"),
                  io.uv_conf_to_rgb(pred, conf, proj_wh[0], proj_wh[1]))
    save_conf_map(conf, white_valid, out)   # readable turbo confidence map
    save_uv_grid(col_gc, pred, conf, white_valid, args.min_conf, proj_wh[0], proj_wh[1], out,
                 row_ref=ref_v)
    save_histograms(ref_u, pred, conf, both, out, args.reference, dv_ref=ref_v)
    io.save_image(str(out / "white.png"), white)
    io.save_image(str(out / "valid_white.png"), white_valid.astype(np.float64))
    diff_valid = white_valid & (conf >= args.min_conf)
    diff_mask = save_du_maps(ref_u, pred_u, diff_valid, proj_w, out, args.reference, col_gc=col_gc)
    summary = {"ckpt": args.ckpt, "captures": str(cap), "reference": args.reference,
               "proj_wh": list(proj_wh), "ref_valid_px": int(ref_valid.sum()),
               "pred_valid_px": int(pred_valid.sum()),
               "gray_vs_ref_med": floor_med, "gray_vs_ref_p95": floor_p95,
               "white_thresh": args.white_thresh, "white_valid_px": int(white_valid.sum()),
               "white_valid_iou_vs_ref": wv_iou, "diff_min_conf": args.min_conf,
               "diff_px": int(diff_mask.sum()),
               "iou": iou, "coverage": coverage, **m, "sweep": sweep}
    if ref_v is not None:
        summary["row"] = {"reference": args.graycode_h_set,
                          "ref_valid_px": int(np.isfinite(ref_v).sum()),
                          "v_bin_acc": mv["bin_acc"], "med_dv": mv["med_du"],
                          "med_dv_correct": mv["med_du_correct"], "p95_dv": mv["p95_du"],
                          "uv_bin_acc": uv_acc,
                          "row_sweep": row_sweep, "uv_sweep": uv_sweep}
    (out / "metrics.json").write_text(json.dumps(summary, indent=2))
    plot_overview(ref_u, pred_u, both, proj_w, out / "overview.png",
                  f"{Path(args.ckpt).name}  vs {args.reference}  —  {cap.name}", args.reference)
    print(f"\nmetrics -> {out}/metrics.json")


if __name__ == "__main__":
    main()
