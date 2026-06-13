# CLAUDE.md — rendering setup

Guidance for working on the Mitsuba structured-light **dataset renderer**. For the
analytic benchmark and web viewer, see `README.md`; this file covers how synthetic
ground-truth datasets are rendered.

## What it does

`scripts/gen_mitsuba_dataset.py` renders a structured-light capture dataset with
Mitsuba 3: it projects a sequence of patterns through a calibrated projector onto
a scene and renders, per frame, the camera capture — plus exact ground-truth
depth from a position AOV. Triangulation is metric-correct by construction (we
place the rig, so calibration is known, not estimated).

## Two backends

There are two interchangeable renderers behind the same file-driven inputs
(`--scene` / `--rig` / `--patterns`) and the same `renders/<name>/` output layout:

- **Mitsuba** (`lux/datasets/mitsuba_gen.py`, `scripts/gen_mitsuba_dataset.py`) —
  path-traced, accurate, supports the full lens model (DoF + distortion). Slow
  (~4 s/frame at 1080p/128 spp). Output folder `mitsuba_<scene>`.
- **Ray-caster** (`lux/datasets/raster_gen.py`, `scripts/gen_rasterizer_dataset.py`) —
  pure-NumPy analytic ray-caster (plane/sphere/box closed-form + Newton-solved wavy
  heightfield), normal-aware Lambertian shading, projector shadow map. ~100× faster
  (a full 1080p Gray-code set in seconds). Output folder `raster_<scene>`.
  Supports **projector distortion**; camera DoF/distortion are ignored (out of scope).

Both compute `gt_proj` via the shared `lux/datasets/correspondence.py:projector_subpixel`,
and produce the identical artifact set. For the ray-caster, `gt_proj` is exact **by
construction**: it casts exactly `camera_rays` (pixel-index convention), so the hit point
is `depth·ray`, and the capture samples the pattern at the very coordinate `gt_proj`
records (verified: Gray-code decode matches to the ~0.25 px quantization floor).
`scripts/compare_backends.py` cross-checks the two (blocks agree to ~0 mm; wavy differs by
the Mitsuba mesh-tessellation error, where the analytic ray-caster is the truer GT).
`scripts/verify_gt_proj.py --backend {mitsuba,raster}` checks either.

It is a **pure renderer**: there is no built-in decoding strategy and no scoring.
You bring the patterns; it produces captures + GT.

## Three feed-in axes (everything is a file)

| Axis | Flag | Lives in | Loader |
|------|------|----------|--------|
| **Scene** (geometry in front of the rig) | `--scene` | `lux/datasets/scenes/*.json` | `scene_loader.py` |
| **Rig** (camera + projector params) | `--rig` | `lux/datasets/rigs/*.json` | `rig_loader.py` |
| **Patterns** (projection sequence) | `--patterns` | `patterns/<set>/` (PNGs) | loaded inline |

Each flag also accepts a path to your own `.json` (scene/rig) or folder (patterns),
so adding one is just dropping in a file — no code changes. `list_scenes()` /
`list_rigs()` enumerate the built-ins.

### Scenes (`scene_loader.py`)
A scene file is a list of `objects` plus an optional top-level `ambient` (0–1,
default 0.05 via `load_scene_ambient`) for the constant environment light — it
sets the unlit/shadowed floor (`signal = albedo·(ambient + …)` for raster, a
`constant` emitter for Mitsuba). Primitives: `plane`, `sphere`, `box`, `wavy`
(a procedural sinusoidally-displaced surface). A `box` takes an optional
`rotation` `[rx,ry,rz]` (Euler degrees, applied X→Y→Z; `rot_xyz` matches the
Mitsuba rotate composition — verified parity). `reflectance` is a gray scalar or
`[r,g,b]`. Any object may carry a procedural albedo `texture` block
(`{"type": "checker"|"stripes"|"noise", "scale", "contrast", …}`), evaluated in
world space at the hit point — **ray-caster only**, Mitsuba ignores it.
Distances in metres in the scene's own frame (the rig's camera pose
places the viewer). Built-ins: `blocks`, `wavy`.

### Rigs (`rig_loader.py`)
A rig file holds the full camera+projector calibration: each device block is
intrinsics (`hfov_deg` shorthand, or explicit `fx/fy/cx/cy`), plus a placement.
Two placement regimes (`build_rig` picks per file):
- **Pose** (when *both* blocks have a `position`): free world-space placement via
  `position` + `look_at` (point) or `look_dir` (vector) + `up`. `up` is the world
  direction at the image top — `(0,-1,0)` keeps the standard +Y-down image and
  makes a camera at the origin looking `+Z` the identity. See `rigs/posed.json`.
- **Legacy baseline**: `baseline` (m, +X offset) + `toe_in_deg`. Still supported.

Both reduce to the same `Rig`: the relative `R,t` (used by all SL routines) plus
optional world poses `(R_cam,C_cam)`/`(R_proj,C_proj)` the renderers use to place
the devices. Legacy rigs get an identity camera pose, so they render exactly as
before. `look_at_basis` (in `geometry.py`) builds the world→device rotation; depth
stays camera-frame Z, so `gt_proj` is exact for any pose (verified on both backends).
Built-ins: `default`, `wide_baseline`, `hd`, `hd_lens` (all baseline), `posed` (pose).

### Lens model (`optics.py`)
A rig device block may carry optional **lens** fields (parsed by `parse_optics`,
separate from the pinhole `Rig`); absent ⇒ ideal pinhole. Applied **only to the
pattern captures** — GT depth and the albedo/white pass stay ideal pinhole so the
truth stays clean (undistort captures downstream to align).
- camera `aperture_radius` + `focus_distance` → native `thinlens` depth of field.
- camera `distortion {k1,k2,p1,p2}` → Brown-Conrady warp of the capture (post).
- projector `distortion {...}` → warp the pattern before projecting (pre).
- projector `defocus_px` → **approximate** constant-blur defocus (the `projector`
  emitter has no aperture, so true depth-dependent projector DoF isn't possible).
See `rigs/hd_lens.json`. Touchpoints: `_camera_dict` (thinlens), `render_capture`
(`optics=` → projector pre-warp + camera post-distortion), `optics.py` (warps).

### Patterns (`scripts/gen_patterns.py`)
Materialises pattern sets as PNG folders the renderer projects in **filename
order**. Default resolution **1920×1080**. Two families:
- **monochrome** from `lux.methods`: `graycode` (24 frames), `phaseshift` (8).
- **colour** single/multi-shot: `rainbow` (hue sweep), `rgb_phase` (R/G/B phase
  triple), `colors` (full-frame red, green, blue).

Regenerate all: `python scripts/gen_patterns.py`.

## Colour rendering

The pipeline is colour-capable end to end. A pattern may be grayscale `(H,W)` or
colour `(H,W,3)`. The renderer **auto-detects**: it loads each PNG as RGB and
collapses to grayscale only when all channels match, so monochrome sets stay mono
and true colour sets render colour captures (the stats line tags these `rgb`).
Touchpoints if you change this: `mitsuba_gen._projector_dict` (accepts 3-channel),
`mitsuba_gen.render_capture` (returns colour when pattern is colour),
`io.montage` (handles colour stacks), and the loader in `render_pattern_dir`.

## Output layout

Default output root is `renders/` (the web viewer reads this; `results_real/` is
the separate PBRT dataset — leave it alone).

```
renders/<scene>/                 # <scene> defaults to mitsuba_<stem> / raster_<stem>
├── gt_depth.npy                 # GT camera-frame Z depth, NaN off-surface
├── gt_proj.npy                  # (H,W,2) exact projector subpixel (col,row) per camera pixel
├── gt_proj.png                  # quick-look: R=row/proj_h, G=col/proj_w, B=valid
├── gt_cloud.ply                 # GT point cloud, albedo-coloured
├── white.png                    # raw white-lit capture
├── albedo.png                   # white normalised to peak 1.0 (often == white.png)
└── <patterns-set>/
    ├── cap_<pattern-filename>.png   # one capture per projected pattern
    └── captures_montage.png
```

`gt_proj.npy` is the fundamental SL ground truth — the projector subpixel a decoder
estimates (`projector_subpixel(rig, depth, proj_optics)`). It's derived from
`gt_depth` + calibration with lux's pinhole model, so `triangulate_columns(gt_proj
[...,0])` recovers `gt_depth` to machine precision. Channel 0 = column, 1 = row.
Pixels outside the projector frame `[0,w)×[0,h)` are **NaN**, as are pixels the
projector can't see — the gen scripts mask `gt_proj` with `raster_gen.projector_visible`,
which tests each point against the **analytic projector-depth raycast** (the same
oracle the captures' shadows use). So points occluded from the projector (e.g. the
plane behind a box) are invalid (black in `gt_proj.png`), matching where the captures
are unlit, *without* shadow acne on curved surfaces. Both backends use it (occlusion
is a property of the shared geometry). `correspondence.projector_visible` is a
depth-map-only z-buffer fallback — simpler but self-shadows on grazing/curved surfaces.
When the projector has distortion, the coordinate is mapped to the *authored*
(pre-warp) projector coordinate so it matches what a decoder recovers from the
distortion-warped patterns. `scripts/verify_gt_proj.py` checks this end-to-end by
decoding a rendered Gray-code sequence (PASS = matches to sub-pixel; control 0.25px,
distortion edge tail p95 5.3px→0.6px after correction).

Note: `white.png` and `albedo.png` are usually numerically identical — `albedo =
white / white.max()`, a no-op whenever the white frame already saturates at 1.0
(the common case). They diverge only for dim renders.

## Rendering internals (`lux/datasets/mitsuba_gen.py`)

- Mitsuba import is **lazy** (`_ensure_mitsuba`) so `import lux` stays light; macOS
  needs `libLLVM.dylib`, auto-located via `DRJIT_LIBLLVM_PATH`. Variant defaults to
  `llvm_ad_rgb` (all CPU cores + SIMD; ~5× faster than `scalar_rgb` on the GT
  render); override with `LUX_MI_VARIANT`. No GPU backend on macOS — `cuda_*`
  needs an NVIDIA GPU + OptiX.
- `build_scene` defaults baked in (not CLI-exposed): `ambient=0.04`,
  `proj_scale=3.0`, path tracer `max_depth=6`. Edit here to change lighting.
- `render_capture` / `render_ground_truth` take an optional `label`; when set they
  print a per-image stats line (build/render time, throughput, content metric).
  Library callers without a label stay silent.
- GT depth comes from an `aov` integrator emitting `position`; world == camera
  frame, so depth is just the hit point's Z.

## Ray-caster internals (`lux/datasets/raster_gen.py`)

- Scenes are parsed to analytic primitives by `scene_loader.load_scene_primitives`
  (`Plane/Sphere/Box/Wavy` dataclasses, same JSON as Mitsuba, matching defaults).
- Per-pixel vectorised intersectors (origin-generalised so projector shadow rays
  reuse them): ray-plane/sphere/box closed-form; wavy = bisection-bracket + Newton
  solve of the heightfield `z=f(x,y)`. The G-buffer (depth/normal/albedo) and a
  projector shadow-depth map are built **once** and cached, so the per-pattern loop
  is just a pattern lookup + shade.
- Exactness anchor: it casts `camera_rays` verbatim (pixel **index**, not +0.5). Any
  half-pixel offset breaks the `gt_proj`-by-construction guarantee — never change it.
- Shading: `albedo·(ambient + gain·lit·pattern·max(N·L,0)·falloff)`; `RenderConfig`
  knobs (ambient/gain/noise/cast_shadows) reused from `lux/render.py`.
- Capture post-effects (all rig-driven, applied in this order, on the *captures* only —
  white-ref/GT stay clean ideal-pinhole so `gt_proj`/`gt_depth` are unaffected):
  **depth of field** → **camera distortion** → **bloom** → **sensor noise**.
  - DoF: `optics.apply_depth_of_field` — a depth-driven variable blur (CoC =
    `aperture·fx·|D−F|/(F·D)`), a fast post-process approximation of `thinlens`
    (vs Mitsuba's true aperture sampling). Reuses the camera `aperture_radius` /
    `focus_distance` fields.
  - Distortion: `optics.apply_distortion` from `camera.distortion` (captures end up in
    distorted image space; undistort downstream to align with GT).
  - Bloom: `optics.apply_bloom` from a rig `"bloom": {threshold, intensity, radius}` block.
  - Noise: `render.add_sensor_noise` (shot + read + **blue**), from a rig
    `"noise": {blue, read, shot}` block (`blue` = `render.blue_noise_field`, FFT
    frequency-weighted grain).
  See `rigs/hd_lens.json` for a rig exercising all of them.

## Commands

```bash
# generate pattern sets (1920x1080)
python scripts/gen_patterns.py

# render a dataset (Mitsuba): scene + rig + patterns
python scripts/gen_mitsuba_dataset.py --scene wavy --rig default --patterns patterns/graycode --spp 24
python scripts/gen_mitsuba_dataset.py --scene blocks --patterns patterns/colors      # colour capture

# render the same dataset ~100x faster (ray-caster)
python scripts/gen_rasterizer_dataset.py --scene wavy --rig hd --patterns patterns/graycode

# randomized training data (ray-caster): per sample, draws a random scene
# (plane/tilted-wall/wavy/no background + 4-18 oriented boxes & spheres,
# procedural textures, frustum-rejection-sampled into view) and a random rig
# (pose, FOV, baseline + probabilistic noise/bloom/DoF/projector-distortion),
# writes them as scene.json/rig.json in the sample folder (re-renderable with
# gen_rasterizer_dataset.py), then renders the standard artifact set.
# Sample i uses seed+i; sample.json is written last and doubles as the
# completion marker: reruns resume (skip finished samples; --overwrite forces),
# higher --seed extends, disjoint seed ranges give train/val splits.
# --patterns takes multiple sets (G-buffer shared, extra sets ~free);
# --lean drops human-facing extras (ply/montage/quicklooks); --jobs N parallel.
# --cam-distort opts into camera distortion (warps captures off the GT's ideal
# image space); everything else stays pixel-aligned with gt_depth/gt_proj.
python scripts/gen_training_data.py --n 100 --patterns patterns/marray --jobs 4 --lean

# verify / compare
python scripts/verify_gt_proj.py --backend raster
python scripts/compare_backends.py
```

`--patterns` is required. Other flags: `--name` (output folder override), `--out`
(root, default `renders`); Mitsuba-only: `--spp`, `--gt-spp`.

## Training on the Linux box (RTX 2080 Ti) — env & perf gotchas

Hard-won notes for running `scripts/train_proj_net.py` here (not relevant to the
renderer itself, but bit us repeatedly):

- **venv can't live on the repo drive.** The repo sits on an exFAT external drive
  (`/run/media/nshelton/LUX`), which has no symlink support — even
  `python -m venv --copies` fails on the `lib64` link. The working venv is at
  **`~/.venvs/lux`** (on NVMe). Run everything with `~/.venvs/lux/bin/python`.
- **Batch ceiling is 32.** `--mid attn --crop 256 --amp` OOMs at batch 64 and 48
  on the 11 GB card; batch 32 (~10.3 GB) is the max. Use
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
- **Stage the loaf on NVMe — the single biggest speedup.** Training random-crops
  from a memmapped loaf (`caps.npy`+`gt.npy`, ~98 GB), which exceeds the 61 GB RAM
  so it can't be page-cached. Reading those random crops over **exFAT-over-USB
  halves throughput** (~50 vs ~104 img/s) and leaves the GPU sawtoothing 9-99%
  while dataloader workers sit in `D`/`exfat_get_block`. The drive's 500 MB/s
  *sequential* spec is irrelevant — the workload is random-access-latency-bound,
  where NVMe wins ~100x. Fix: `cp` the loaf dir to `~/datasets/<loaf>` and point
  `--loaf` there. Diagnose with `ps -o pid,stat,wchan -p <workers>` (D-state on
  `exfat_get_block` = I/O-bound) and a `/proc/diskstats` read-rate delta.
- **Heat / hangs.** Sustained 99% util uncapped (~260 W) ran the card hot and the
  box hung twice early on. `sudo nvidia-smi -pl 140` caps power (~15% slower, much
  cooler). Also avoid suspend mid-run (NVIDIA-on-Wayland resume failure = black
  screen, live cursor, no input) — wrap training in
  `systemd-inhibit --what=idle:sleep` and/or mask the sleep targets.

## Conventions

- Keep the three feed-in axes file-driven; prefer adding a `.json`/PNG set over
  hardcoding. Match the lazy-import pattern for anything touching Mitsuba.
- `lux/methods/` (graycode/phaseshift/neural) is shared with the analytic
  benchmark (`run_benchmark.py`) — the Mitsuba renderer no longer decodes, so
  don't reintroduce decoding here.
