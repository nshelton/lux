# lux — a structured-light testbed

A harness for evaluating structured-light depth algorithms against ground
truth. Each algorithm is a pluggable **method** that declares the projector
patterns it needs and decodes the resulting camera images into a **metric depth
map**. A synthetic simulator renders the captures and provides exact ground
truth, and a scorer reports per-method depth error.

```
scene (GT depth + albedo)
        │  method.patterns()          ← each method picks its own patterns
        ▼
   render(patterns)  ──►  captured image stack   (analytic simulator: albedo,
        │                                          ambient, noise, projector shadows)
        ▼
   method.decode()   ──►  metric depth map
        │
        ▼
   compare_depth()   ──►  rmse / median / bad-pixel / completeness
```

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e .            # core: numpy, scipy, opencv
pip install -e '.[neural]'  # optional: torch, for the neural method

python scripts/run_benchmark.py            # all scenes × all methods
python scripts/run_benchmark.py --methods graycode phaseshift --shot 0.02
```

Example output:

```
scene             method      rmse_mm  median_ae_mm  bad_px_%  compl_%  pat
slanted_plane     graycode    6.74     5.66          82.3      100.0    20
slanted_plane     phaseshift  1.75     1.15          24.4       99.8     8
spheres_on_plane  graycode    10.57    8.30          88.0      100.0    20
spheres_on_plane  phaseshift  3.20     1.91          48.2       99.9     8
```

Gray code is integer-precise (coarse, quantized depth); phase-shift decodes
sub-pixel and wins on smooth surfaces — exactly the kind of trade-off the
testbed is built to surface.

> **Gray-code depth looks terraced / "sawtoothed" — that's correct.** Gray code
> resolves the projector column to an *integer*, so every camera pixel mapping
> to the same column gets the same depth plane. When the camera over-samples the
> projector (one projector column spans N camera pixels), a block of N pixels
> shares one depth while the true surface ramps across it, giving a depth error
> with **period N pixels**, ~half over- and half under-estimating. The decode is
> bit-exact — feeding the continuous column into the same triangulation gives
> ~0 error. Use phase-shift (or a Gray-coded phase-shift hybrid) for sub-pixel.
> See `lux/methods/graycode.py` for the full explanation.

## Web viewer

The benchmark writes coloured point clouds (`.ply`), captured frames (`.png`),
and `scores.json` to `renders/`. A zero-build Three.js viewer shows ground
truth and the selected method **side by side in one synced camera**, with the
prediction error-coloured and the captured frame strip below:

```bash
python -m http.server      # from the repo root
# open http://localhost:8000/web/
```

- Left panel = ground-truth cloud, right panel = prediction (blue→red = signed
  depth error, ±5 mm). Drag orbits both panels together for direct comparison.
- Scene / method dropdowns switch the data live; the score table highlights the
  selected method; the bottom strip is the exact frames the decoder consumed.

The Python core is decoupled from any viewer — artifacts are plain `.npy` /
`.ply` / `.png` / `.json`, so Rerun or an agent-built 3D viewer can consume them
too.

## Real datasets — PBRT Gray-coded phase shift

`lux/datasets/pbrt_sl.py` loads the `camera-WxH-projector-WxH/` PBRT
scanner-simulation dataset and decodes its real captures. Each scene ships 16
frames: **6 sine phase-shift** patterns, **9 Gray-code** bits, and **1 all-white**
reference. The decoder is a **Gray-coded phase-shift hybrid**:

- The **Gray** code gives a robust integer *fringe index* (the big bits — which
  of 320 fringes).
- The **sine** phase fills in the *sub-fringe* position (the small bits —
  sub-pixel within a 6px fringe).
- `column = (fringe_index + sub_fringe_fraction) × fringe_width`.

The exact phase→column relationship is recovered from the projector patterns
themselves (true column known), so there's no convention guessing — it
reconstructs the projector column to **0.00 px** on the source patterns.

```bash
python scripts/run_pbrt_dataset.py \
    --root camera-4056-3040-projector-1920-1080 \
    --variant perspective --downsample 4
# -> results_real/<scene>/{white,column,phase,depth}.png, cloud.ply, depth.npy
#    + manifest.json, browsable under the "real (PBRT)" tab (3D cloud + 2D maps)
```

Gray bits are thresholded at the **per-pixel mean of the sine frames** — that DC
level is `albedo*(ambient + 0.5*projector)`, exactly the midpoint between a Gray
bit on and off. (Thresholding against `white + min-over-gray` instead leaves
coherent bit flips that surface as ±1 fringe "curtain" layers in the cloud;
the sine-DC threshold drops those from ~6% of pixels to <1%.)

The decoded **correspondence** (column / phase) is exact and calibration-free.
It is then **triangulated to a metric point cloud** using a `PBRTRig` parsed
from the scene's PBRT includes (camera/projector pose + intrinsics). Use the
`perspective` variant — its `Camera "perspective"` is a true pinhole and
triangulates cleanly; the `real-with[-out]-dispersion` variants use a
`realistic` (dgauss 50mm) lens whose distortion breaks the pinhole assumption.

The rig is correct in shape but needs one empirical scalar: depth comes out a
constant ~1.29× too shallow (a baseline/intrinsic convention factor the PBRT
files don't pin down). `METRIC_SCALE` is calibrated once against the reference
**Sphere** (known radius 0.125 m at the origin) — afterwards the Sphere
reconstructs to r≈0.127 m with ~2 mm residual, and since all scenes share the
rig the same scalar makes every scene metric. Swap in a proper geometric
calibration from the checkerboard images to remove the empirical factor.

## Ground-truth datasets — Mitsuba 3 generator

`lux/datasets/mitsuba_gen.py` renders our *own* structured-light data with a
path tracer, so we get what the PBRT captures lacked: **exact ground-truth depth
and exact calibration**. A calibrated `projector` emitter displays each pattern;
the camera captures it; a position AOV gives true camera-frame Z-depth.

```bash
pip install mitsuba            # macOS also needs libLLVM (brew install llvm)
python scripts/gen_mitsuba_dataset.py --methods phaseshift graycode --spp 24
# -> renders/mitsuba_blocks/{gt_cloud.ply, <method>/pred_cloud.ply, captures}
#    + merged scores.json; shows in the synthetic 3D tab (GT vs prediction)
```

The scene is built to match a lux `Rig` exactly, so the existing methods and
metrics work unchanged — and because *we* place the rig, triangulation is
**metric-correct by construction**: decoded depth vs Mitsuba GT gives
`median(pred/gt) = 1.001`, i.e. **no empirical scale factor** (contrast the PBRT
path's `1.29`). The Mitsuba position AOV independently validates the rig
conventions. This makes it the reference loop for scoring any decoder against
true depth, and the basis for generating neural-decoder training data.

> macOS note: Mitsuba's Dr.Jit backend needs `libLLVM.dylib` even for the scalar
> variant. The module auto-detects a Homebrew LLVM install
> (`/opt/homebrew/opt/llvm/lib/libLLVM.dylib`); `brew install llvm` if missing.

## Project layout

```
lux/
  geometry.py     camera/projector rig + ray-plane triangulation
  scene.py        synthetic GT scenes (plane, spheres, depth steps)
  render.py       analytic forward simulator (the data generator)
  metrics.py      depth comparison
  harness.py      orchestration + reporting
  io.py           export depth / point clouds / error maps
  methods/        ← the algorithms (pluggable)
    base.py       Method interface: patterns() + decode() -> DepthResult
    graycode.py   binary Gray-code (integer columns)
    phaseshift.py multi-frequency sinusoidal phase shift (sub-pixel)
    neural.py     learned per-pixel decoder (torch-optional; falls back)
scripts/
  run_benchmark.py
  train_neural.py
web/index.html    Three.js point-cloud viewer
tests/test_pipeline.py
```

## Adding a method

Subclass `Method`, implement the two halves of the contract, and register it:

```python
# lux/methods/mymethod.py
from .base import Method, DepthResult

class MyMethod(Method):
    name = "mymethod"
    def patterns(self, width, height):
        return ...                      # (N, height, width) projector patterns
    def decode(self, images, rig):
        proj_col = ...                  # (H, W) decoded projector column
        depth = self.triangulate(rig, proj_col)   # shared geometry
        return DepthResult(depth=depth, proj_col=proj_col)
```

Add `"mymethod": MyMethod` to `REGISTRY` in `lux/methods/__init__.py`. Methods
that regress depth directly (e.g. a network) can skip `proj_col` and return
`depth` straight from `decode`.

## Adding a scene

Return a `(depth, albedo)` pair from a builder in `lux/scene.py` and add it to
`SCENES`. Depth is metric camera-frame Z; `NaN` marks background.

## The neural method

`scripts/train_neural.py` builds supervision (image stack → GT projector
column) from the simulator and fits a small per-pixel MLP. Without a checkpoint
the method falls back to phase-shift so the benchmark always runs.

A per-pixel model can't disambiguate periodic fringes without spatial context —
that limitation is intentional and visible in the scores. The natural next step
is a CNN/U-Net over the image stack (swap `PixelMLP` in `neural.py`); the
patterns, triangulation, and scoring all stay the same.

## Roadmap / next steps

- **Realism**: defocus/blur, subsurface scattering, specular highlights,
  global illumination, projector gamma — all live in `render.py`.
- **Harder scenes**: imported meshes (depth from a rendered OBJ), thin
  structures, low-albedo and translucent materials.
- **More methods**: single-shot (De Bruijn / speckle), hybrid Gray-code +
  phase, multi-view fusion.
- **Real data**: add a loader alongside the synthetic scenes (e.g. a
  Middlebury-style captured set) behind the same `Scene` interface.
- **Spatial neural decoder**: replace the per-pixel MLP with a U-Net.
