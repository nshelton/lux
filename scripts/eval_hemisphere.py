#!/usr/bin/env python3
"""Evaluate a correspondence checkpoint on the hemisphere flat-plane set.

For every sample under --data: full-frame inference, then per-sample metrics
(u-bin accuracy, median |du|, median |du| within correct bins, validity IoU)
joined with the sample's pose angles. Writes ``per_sample.csv`` and prints the
metric-vs-obliquity table (binned by max of camera/projector elevation — the
axis along which the code becomes anamorphic). Saves a scatter plot when
matplotlib is available.

    python scripts/eval_hemisphere.py --ckpt checkpoints/proj_net.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from lux import io  # noqa: E402
from lux.proj_net import load_checkpoint, predict_full, predict_tiled, N_BINS_U  # noqa: E402


def eval_sample(model, proj_wh, d: Path, device: str, pattern_set: str, frame: str,
                tiled: bool = False, overlap: int = 0) -> dict:
    img = io.load_image(str(d / pattern_set / frame), gray=True)
    gt = np.load(d / "gt_proj.npy")
    if tiled:
        pred = predict_tiled(model, img, proj_wh, device=device, overlap=overlap)
    else:
        pred = predict_full(model, img, proj_wh, device=device)
    both = np.isfinite(gt[..., 0]) & np.isfinite(pred[..., 0])
    union = np.isfinite(gt[..., 0]) | np.isfinite(pred[..., 0])
    out = {"iou": both.sum() / max(union.sum(), 1),
           "valid_px": int(both.sum())}
    if both.any():
        du = np.abs((pred[..., 0] - gt[..., 0])[both])
        binw = proj_wh[0] / N_BINS_U
        correct = (pred[..., 0][both] // binw).clip(0, N_BINS_U - 1) == \
                  (gt[..., 0][both] // binw).clip(0, N_BINS_U - 1)
        out.update(bin_acc=float(correct.mean()),
                   med_du=float(np.median(du)),
                   med_du_correct=float(np.median(du[correct])) if correct.any() else np.nan,
                   p95_du=float(np.percentile(du, 95)))
    else:
        out.update(bin_acc=0.0, med_du=np.nan, med_du_correct=np.nan, p95_du=np.nan)
    return out


def _disk(ax, theta, phi_deg, err, norm, title):
    """Top-down hemisphere disk: each device pose at radius=tilt-off-normal,
    angle=azimuth (centre = head-on, rings every 15deg out to grazing). Marker
    size and colour both encode error, so failure regions read as big red blobs."""
    import matplotlib.pyplot as plt
    th = np.asarray(theta, float)
    ph = np.radians(np.asarray(phi_deg, float))
    x, y = th * np.cos(ph), th * np.sin(ph)
    e = np.clip(np.asarray(err, float), norm.vmin, norm.vmax)
    s = 18 + 260 * (np.log10(e) - np.log10(norm.vmin)) / (np.log10(norm.vmax) - np.log10(norm.vmin))
    sc = ax.scatter(x, y, c=e, s=s, cmap="RdYlGn_r", norm=norm,
                    edgecolors="k", linewidths=0.3, alpha=0.85, zorder=3)
    for r in (15, 30, 45, 60, 75):
        ax.add_patch(plt.Circle((0, 0), r, fill=False, ls=":", ec="0.6", lw=0.8, zorder=1))
        ax.text(0, r, f"{r}°", fontsize=7, ha="center", va="bottom", color="0.5")
    ax.set_aspect("equal")
    ax.set_xlim(-82, 82)
    ax.set_ylim(-82, 82)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title)
    return sc


def _heatmap_camproj(ax, tc, tp, err, norm):
    """Binned median error over (camera tilt × projector tilt). Disambiguates the
    per-device disks: failure shows as an L (bad whenever *either* device grazes),
    with the only green region in the bottom-left (both near head-on)."""
    edges = np.arange(0, 81, 10)
    nb = len(edges) - 1
    grid = np.full((nb, nb), np.nan)            # [proj_bin, cam_bin]
    for i in range(nb):
        for j in range(nb):
            m = ((tc >= edges[j]) & (tc < edges[j + 1])
                 & (tp >= edges[i]) & (tp < edges[i + 1]))
            if m.any():
                grid[i, j] = np.nanmedian(err[m])
    import matplotlib as mpl
    cmap = mpl.colormaps["RdYlGn_r"].copy()
    cmap.set_bad("0.85")                          # empty bins -> light gray
    im = ax.imshow(grid, origin="lower", extent=[0, 80, 0, 80], aspect="equal",
                   cmap=cmap, norm=norm)
    ax.set_xlabel("camera tilt off normal (°)")
    ax.set_ylabel("projector tilt off normal (°)")
    ax.set_title("error vs (camera × projector) tilt")
    return im


def plot_hemisphere(rows: list[dict], out_dir: Path, ckpt_name: str) -> None:
    """Four readable views: top-down camera & projector pose disks (size+colour =
    error), error vs deflection magnitude (azimuth collapsed), and a binned
    camera×projector heatmap that disambiguates which device drives failure."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
    except ImportError:
        print("\n(matplotlib not installed - skipped plot)")
        return

    err = np.array([r["med_du"] for r in rows], float)
    tc = np.array([r["theta_cam_deg"] for r in rows], float)
    tp = np.array([r["theta_proj_deg"] for r in rows], float)
    pc = np.array([r.get("phi_cam_deg", 0.0) for r in rows], float)
    pp = np.array([r.get("phi_proj_deg", 0.0) for r in rows], float)
    vmax = float(max(100.0, np.nanmax(err)))
    norm = mcolors.LogNorm(vmin=0.3, vmax=vmax)

    fig, axes = plt.subplots(2, 2, figsize=(13, 12))
    sc = _disk(axes[0, 0], tc, pc, err, norm, "camera pose (top-down hemisphere)")
    _disk(axes[0, 1], tp, pp, err, norm, "projector pose (top-down hemisphere)")
    cb = fig.colorbar(sc, ax=axes[0, :], label="median |du| (px)", fraction=0.046, pad=0.02)
    cb.ax.text(0.5, 1.02, "size also = error", transform=cb.ax.transAxes,
               fontsize=7, ha="center", color="0.4")

    # Error vs deflection magnitude (max device tilt off normal), azimuth collapsed.
    defl = np.maximum(tc, tp)
    ax = axes[1, 0]
    ax.scatter(defl, err, s=14, alpha=0.35, color="0.45", zorder=2)
    edges = np.arange(0, 91, 10)
    cen, med, q1, q3 = [], [], [], []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (defl >= a) & (defl < b)
        if m.any():
            cen.append((a + b) / 2)
            med.append(np.nanmedian(err[m]))
            q1.append(np.nanpercentile(err[m], 25))
            q3.append(np.nanpercentile(err[m], 75))
    ax.fill_between(cen, q1, q3, color="C3", alpha=0.18, zorder=1, label="IQR")
    ax.plot(cen, med, "-o", color="C3", lw=2, zorder=3, label="median per 10° bin")
    ax.axhline(1.0, ls="--", lw=0.8, color="0.5", zorder=1)
    ax.text(2, 1.05, "1 px", fontsize=7, color="0.5")
    ax.set_yscale("log")
    ax.set_xlim(0, 80)
    ax.set_xlabel("deflection = max(camera, projector) tilt off normal (°)")
    ax.set_ylabel("median |du| (px, log)")
    ax.set_title("error vs deflection magnitude")
    ax.legend(fontsize=8)

    im = _heatmap_camproj(axes[1, 1], tc, tp, err, norm)
    fig.colorbar(im, ax=axes[1, 1], label="median |du| (px)", fraction=0.046, pad=0.02)

    fig.suptitle(f"{ckpt_name} - hemisphere flat-plane sweep ({len(rows)} samples)")
    fig.savefig(out_dir / "hemisphere_overview.png", dpi=130, bbox_inches="tight")
    print(f"\nplot -> {out_dir}/hemisphere_overview.png")


def _load_rows_csv(out_dir: Path, data_dir: Path) -> list[dict]:
    """Read a per_sample.csv back into rows for --replot, pulling phi angles from
    each sample's sample.json when the csv predates the phi columns."""
    import csv
    rows = []
    with open(out_dir / "per_sample.csv") as f:
        for r in csv.DictReader(f):
            row = {k: (float(v) if v not in ("", "name") and k != "name" else v)
                   for k, v in r.items()}
            if "phi_cam_deg" not in row or row.get("phi_cam_deg") in (None, ""):
                pose = json.loads((data_dir / row["name"] / "sample.json").read_text())
                row["phi_cam_deg"] = pose.get("phi_cam_deg", 0.0)
                row["phi_proj_deg"] = pose.get("phi_proj_deg", 0.0)
            rows.append(row)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="evals/hemisphere/data")
    ap.add_argument("--ckpt", default="checkpoints/proj_net.pt")
    ap.add_argument("--pattern-set", default="marray")
    ap.add_argument("--frame", default="cap_pat_00.png")
    ap.add_argument("--device", default=None)
    ap.add_argument("--tiled", action="store_true",
                    help="stitch full-frame inference from 256-px tiles (the training "
                         "crop size). Auto-enabled for attn checkpoints, whose global "
                         "attention collapses at full-frame token counts; also closes "
                         "the conv row deficit. Force on for conv to A/B it.")
    ap.add_argument("--tile-overlap", type=int, default=0,
                    help="when tiling, overlap tiles + keep max-confidence per pixel "
                         "(removes seams); default 0 = fast hard-stitch for the bench")
    ap.add_argument("--out", default=None, help="results dir (default <data>/../results_<ckpt-stem>)")
    ap.add_argument("--replot", action="store_true",
                    help="skip inference; re-draw plots from an existing per_sample.csv "
                         "(fast iteration on the figure design)")
    args = ap.parse_args()

    if args.replot:
        out_dir = Path(args.out or Path(args.data).parent / f"results_{Path(args.ckpt).stem}")
        rows = _load_rows_csv(out_dir, Path(args.data))
        plot_hemisphere(rows, out_dir, Path(args.ckpt).name)
        return

    if args.device is None:
        import torch
        args.device = ("mps" if torch.backends.mps.is_available()
                       else "cuda" if torch.cuda.is_available() else "cpu")
    model, proj_wh = load_checkpoint(args.ckpt, device=args.device)
    use_tiled = args.tiled or getattr(model, "arch", "conv") == "attn"
    print(f"inference: {'tiled-256' if use_tiled else 'full-frame'} "
          f"(arch={getattr(model, 'arch', 'conv')})")

    dirs = sorted(d for d in Path(args.data).glob("sample_*") if (d / "sample.json").exists())
    if not dirs:
        raise SystemExit(f"no finished samples under {args.data!r}")
    out_dir = Path(args.out or Path(args.data).parent / f"results_{Path(args.ckpt).stem}")
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for i, d in enumerate(dirs):
        pose = json.loads((d / "sample.json").read_text())
        m = eval_sample(model, proj_wh, d, args.device, args.pattern_set, args.frame,
                        tiled=use_tiled, overlap=args.tile_overlap)
        rows.append({**pose, **m, "name": d.name})
        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{len(dirs)} evaluated", flush=True)

    keys = ["name", "theta_cam_deg", "theta_proj_deg", "phi_cam_deg", "phi_proj_deg",
            "r_cam_m", "r_proj_m",
            "bin_acc", "med_du", "med_du_correct", "p95_du", "iou", "valid_px"]
    with open(out_dir / "per_sample.csv", "w") as f:
        f.write(",".join(keys) + "\n")
        for r in rows:
            f.write(",".join(str(r.get(k, "")) for k in keys) + "\n")

    # Binned summary along max(theta_cam, theta_proj).
    theta = np.array([max(r["theta_cam_deg"], r["theta_proj_deg"]) for r in rows])
    edges = [0, 15, 30, 45, 60, 75, 90]
    print(f"\n{args.ckpt} on {len(rows)} hemisphere samples "
          f"(binned by max(cam, proj) obliquity):")
    print(f"{'oblique':>10s} {'n':>4s} {'bin acc':>8s} {'med|du|':>8s} "
          f"{'med|du|✓bin':>11s} {'p95|du|':>8s} {'IoU':>6s}")
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = [r for r, t in zip(rows, theta) if lo <= t < hi]
        if not sel:
            continue
        def agg(k):
            v = np.array([s[k] for s in sel], float)
            return np.nanmedian(v)
        print(f"{lo:3d}-{hi:3d}deg {len(sel):4d} {agg('bin_acc')*100:7.1f}% "
              f"{agg('med_du'):7.2f}px {agg('med_du_correct'):10.2f}px "
              f"{agg('p95_du'):7.1f}px {agg('iou'):6.3f}")

    plot_hemisphere(rows, out_dir, Path(args.ckpt).name)
    print(f"per-sample csv -> {out_dir}/per_sample.csv")


if __name__ == "__main__":
    main()
