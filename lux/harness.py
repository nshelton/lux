"""The benchmark harness — wires scenes, methods, the simulator, and scoring.

One run = (for each scene) render each method's requested patterns through the
simulator, decode to depth, score against ground truth, and emit artifacts. The
harness is deliberately small; its job is orchestration and bookkeeping so the
interesting logic lives in the methods and the simulator.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field

import numpy as np

from . import io
from .geometry import Intrinsics, Rig
from .metrics import DepthMetrics, compare_depth
from .methods import Method, build_method
from .render import RenderConfig, render
from .scene import Scene, build_scene


@dataclass
class HarnessConfig:
    width: int = 320
    height: int = 240
    cam_hfov_deg: float = 50.0
    proj_hfov_deg: float = 45.0
    baseline: float = 0.12
    render: RenderConfig = field(default_factory=RenderConfig)
    out_dir: str = "renders"
    save_artifacts: bool = True
    save_frames: bool = True  # write the projected patterns + captured camera images

    def make_rig(self) -> Rig:
        cam = Intrinsics.from_fov(self.width, self.height, self.cam_hfov_deg)
        proj = Intrinsics.from_fov(self.width, self.height, self.proj_hfov_deg)
        return Rig.make(cam, proj, baseline=self.baseline)


@dataclass
class RunRecord:
    scene: str
    method: str
    metrics: DepthMetrics
    n_patterns: int
    decode_seconds: float


def _evaluate(scene: Scene, method: Method, rig: Rig, cfg: HarnessConfig):
    patterns = method.patterns(rig.projector.width, rig.projector.height)
    cap = render(scene, patterns, rig, cfg.render)

    t0 = time.perf_counter()
    result = method.decode(cap.images, rig)
    dt = time.perf_counter() - t0

    # Score only where the simulator actually delivered pattern light: a method
    # cannot be blamed for projector-shadowed pixels it never had signal for.
    gt_mask = scene.mask & cap.lit_mask
    metrics = compare_depth(result.depth, scene.depth, gt_mask=gt_mask)
    rec = RunRecord(
        scene=scene.name,
        method=method.name,
        metrics=metrics,
        n_patterns=int(patterns.shape[0]),
        decode_seconds=dt,
    )
    return rec, result.depth, cap, patterns


def run(
    scenes: list[str],
    methods: list[str],
    cfg: HarnessConfig | None = None,
) -> list[RunRecord]:
    cfg = cfg or HarnessConfig()
    rig = cfg.make_rig()
    cam = rig.camera
    records: list[RunRecord] = []
    summary = []

    if cfg.save_artifacts:
        io.ensure_dir(cfg.out_dir)

    for scene_name in scenes:
        scene = build_scene(scene_name, cam)
        if cfg.save_artifacts:
            sdir = io.ensure_dir(os.path.join(cfg.out_dir, scene_name))
            io.save_npy(os.path.join(sdir, "gt_depth.npy"), scene.depth)
            pts, col = io.depth_to_points(scene.depth, rig, scene.albedo)
            io.save_ply(os.path.join(sdir, "gt_cloud.ply"), pts, col)

        for method_name in methods:
            method = build_method(method_name)
            rec, depth, cap, patterns = _evaluate(scene, method, rig, cfg)
            records.append(rec)
            summary.append({"scene": scene_name, "method": method_name, **rec.metrics.as_dict(),
                            "n_patterns": rec.n_patterns, "decode_seconds": rec.decode_seconds})

            if cfg.save_artifacts:
                mdir = io.ensure_dir(os.path.join(cfg.out_dir, scene_name, method_name))
                io.save_npy(os.path.join(mdir, "pred_depth.npy"), depth)

                if cfg.save_frames:
                    # Projector patterns (what's displayed) and camera captures
                    # (what the decoder actually consumes), plus glanceable montages.
                    io.save_image_stack(os.path.join(mdir, "patterns"), patterns, prefix="pat")
                    io.save_image_stack(os.path.join(mdir, "captures"), cap.images, prefix="cap")
                    io.save_image(os.path.join(mdir, "patterns_montage.png"), io.montage(patterns))
                    io.save_image(os.path.join(mdir, "captures_montage.png"), io.montage(cap.images))

                pts, _ = io.depth_to_points(depth, rig)
                err_rgb = io.error_colormap(depth, scene.depth)
                _, ecol = io.depth_to_points(depth, rig, err_rgb)
                io.save_ply(os.path.join(mdir, "pred_cloud.ply"), pts, ecol)

    if cfg.save_artifacts:
        io.save_json(os.path.join(cfg.out_dir, "scores.json"), summary)

    return records


def format_table(records: list[RunRecord]) -> str:
    cols = ["scene", "method", "rmse_mm", "median_ae_mm", "bad_px_%", "compl_%", "N", "pat", "t_s"]
    rows = [cols]
    for r in records:
        m = r.metrics
        rows.append([
            r.scene, r.method,
            f"{m.rmse_mm:.2f}", f"{m.median_ae_mm:.2f}", f"{m.bad_pixel_pct:.1f}",
            f"{m.completeness_pct:.1f}", str(m.n_valid), str(r.n_patterns), f"{r.decode_seconds:.3f}",
        ])
    widths = [max(len(row[i]) for row in rows) for i in range(len(cols))]
    out = []
    for j, row in enumerate(rows):
        out.append("  ".join(c.ljust(widths[i]) for i, c in enumerate(row)))
        if j == 0:
            out.append("  ".join("-" * widths[i] for i in range(len(cols))))
    return "\n".join(out)
