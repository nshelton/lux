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
A scene file is a list of `objects`. Primitives: `plane`, `sphere`, `box`, `wavy`
(a procedural sinusoidally-displaced mesh emitted as a temp OBJ). `reflectance` is
a gray scalar or `[r,g,b]`. World == camera frame: camera at origin looking down
+Z, +X right, +Y down, metres. Built-ins: `blocks`, `wavy`.

### Rigs (`rig_loader.py`)
A rig file holds the full camera+projector calibration: each device block is
intrinsics (`hfov_deg` shorthand, or explicit `fx/fy/cx/cy`), plus `baseline` (m)
and `toe_in_deg` for the projector pose (`Rig.make`). Built-ins: `default`
(480×360 cam, 512×512 proj, 0.18 m), `wide_baseline`.

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
renders/<scene>/                 # <scene> defaults to mitsuba_<scene-stem>
├── gt_depth.npy                 # GT camera-frame Z depth, NaN off-surface
├── gt_cloud.ply                 # GT point cloud, albedo-coloured
├── white.png                    # raw white-lit capture
├── albedo.png                   # white normalised to peak 1.0 (often == white.png)
└── <patterns-set>/
    ├── cap_<pattern-filename>.png   # one capture per projected pattern
    └── captures_montage.png
```

Note: `white.png` and `albedo.png` are usually numerically identical — `albedo =
white / white.max()`, a no-op whenever the white frame already saturates at 1.0
(the common case). They diverge only for dim renders.

## Rendering internals (`lux/datasets/mitsuba_gen.py`)

- Mitsuba import is **lazy** (`_ensure_mitsuba`) so `import lux` stays light; macOS
  needs `libLLVM.dylib`, auto-located via `DRJIT_LIBLLVM_PATH`. Variant: `scalar_rgb`.
- `build_scene` defaults baked in (not CLI-exposed): `ambient=0.04`,
  `proj_scale=3.0`, path tracer `max_depth=6`. Edit here to change lighting.
- `render_capture` / `render_ground_truth` take an optional `label`; when set they
  print a per-image stats line (build/render time, throughput, content metric).
  Library callers without a label stay silent.
- GT depth comes from an `aov` integrator emitting `position`; world == camera
  frame, so depth is just the hit point's Z.

## Commands

```bash
# generate pattern sets (1920x1080)
python scripts/gen_patterns.py

# render a dataset: scene + rig + patterns
python scripts/gen_mitsuba_dataset.py --scene wavy --rig default --patterns patterns/graycode --spp 24
python scripts/gen_mitsuba_dataset.py --scene blocks --patterns patterns/colors      # colour capture
```

`--patterns` is required. Other flags: `--name` (output folder override), `--spp`
(samples/pixel), `--out` (output root, default `renders`).

## Conventions

- Keep the three feed-in axes file-driven; prefer adding a `.json`/PNG set over
  hardcoding. Match the lazy-import pattern for anything touching Mitsuba.
- `lux/methods/` (graycode/phaseshift/neural) is shared with the analytic
  benchmark (`run_benchmark.py`) — the Mitsuba renderer no longer decodes, so
  don't reintroduce decoding here.
