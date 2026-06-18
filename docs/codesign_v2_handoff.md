# codesign_v2 — production-training handoff (Linux box)

Built on the Mac (render + loaves); training + eval handed off. This is everything the Linux box needs.

## What codesign_v2 is
The appearance-fix re-co-designed coprime-carrier pattern, replacing `codesign_learned`. The
proxy now samples the **quantized 8-bit pattern via `grid_sample`** (projector grid + bilinear
resample) instead of analytic carriers, so the pattern is optimized for what the renderer actually
produces. **Bake-off gate PASSED:** render-trained grazing (60–75°) bin-acc **40.6%** vs the
committed pattern's 33.6%, with frontal (92%) and bounded subpixel (`med|du|✓bin` ~1.4 px) held.

- Pattern: `patterns/codesign_v2/pat_00.png` (the artifact the loaves were rendered from).
- Periods FROZEN: u `{13,19,33,139}`, v `{11,17,29,113}`. Quad head (per-carrier cos/sin) + CRT
  consensus vote, `peak_margin` confidence. No coord-L1 (ablation: no du gain).

## Deliverables (transfer these to the Linux box)
| path | what | size |
|---|---|---|
| `loaves/clutter_v2/` | 10k cluttered-hemisphere scenes (`caps.npy` u8 + `gt.npy` u16 + `meta.json`) | ~103 GB |
| `loaves/planar_v2/` | 10k planar-junction-hemisphere scenes | ~103 GB |
| `patterns/codesign_v2/pat_00.png` | the pattern (for re-rendering; training reads the loaf) | ~2 MB |
| `evals/clutter_v2/` | 160 held-out cluttered scenes (keeps `gt_depth`) — **Table 2** | ~4 GB |
| `evals/hemisphere/data/` | existing 160 planar poses, now with `codesign_v2` captures — **Table 1** | — |

**Code to `git pull` first** (all verified bit-identical where claimed): `lux/datasets/raster_gen.py`
(screen-bbox culling 2.84× + wavy chunk/`_wavy_f` opts), `scripts/gen_training_data.py` +
`gen_planar_dataset.py` (hemisphere rigs, `maxtasksperchild`, shared pose helpers),
`scripts/train_quad_rendered.py` (`--loaf`), `lux/codesign.py` + `scripts/codesign_demod_probe.py` +
`scripts/train_codesign_quad.py` (appearance fix).

## Training distribution (read before interpreting results)
- **Both loaves**: camera + projector posed **independently** on the hemisphere, tilt
  grazing-oversampled (`--independent-proj --grazing-frac 0.3 --max-tilt 80`) → `max(cam,proj)`
  obliquity **≥45° 85%, ≥60° 57%** (vs the hemisphere eval's 65%/36% — deliberately *more*
  aggressive on grazing, the regime we're trying to win).
- **Clutter**: origin-centered **big ground plane** (100% surface coverage) + an object cluster near
  the origin; cam↔proj overlap ~20–40% (the eval's independent-pose regime — still ~0.4–0.8 M valid
  px/sample). Provides the depth/occlusion edges (no wavy in this set).
- **Planar**: two big half-plane slabs meeting at a crease (the depth edge), grazing-posed.
- **Seeds (all disjoint — no leakage):** clutter `1.0M`, planar `2.0M`, clutter-eval `8.0M`,
  hemisphere-eval `7.0M` (160 poses), planar-val `7.5M` (400, used for the bake-off).

## 1. Production train (loaf-based quad decoder)
```bash
~/.venvs/lux/bin/python scripts/train_quad_rendered.py \
    --loaf loaves/clutter_v2 loaves/planar_v2 \
    --epochs 40 --batch 32 --crops-per-sample 8 --workers 4 \
    --out checkpoints/codesign_quad_prod.pt --snapshots
```
`ConcatLoaf` mixes the two ~50/50 by size. Coarse-first carrier weighting is built in.

Perf gotchas (from CLAUDE.md): **stage the loaves on NVMe** (`~/datasets/`), not the exFAT drive
(random-access halves throughput); batch ceiling ~32 on the 11 GB card; export
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`; `sudo nvidia-smi -pl 140` to cap heat; wrap in
`systemd-inhibit --what=idle:sleep`.

### Step-6 enhancements still to add (training side, not yet implemented)
- **Edge-weighted sampling.** Edges are ~0.3–0.5% of pixels even in clutter; without upweighting,
  the decoder learns flat surface and treats edges as noise. Bias the crop sampler
  (`LoafSamples.__getitem__`, or a `WeightedRandomSampler`) toward crops with high depth-gradient
  density (precompute a per-sample edge mask from a `gt_depth` gradient, or store it in the loaf).
- **`peak_margin` calibration PER obliquity band.** Calibration drifts with obliquity, so a single
  global τ is wrong. Hold out a calibration split, build a reliability diagram per band, store the
  `τ(band)` mapping for abstention.

## 2. Eval — two separate tables (step 7)
**Table 1 — plane obliquity sweep (160 poses):**
```bash
python scripts/eval_hemisphere_quad.py --ckpt checkpoints/codesign_quad_prod.pt \
    --data evals/hemisphere/data --pattern-set codesign_v2
```
bin-acc + `med|du|✓bin` per band. For the M-array baseline, run `eval_hemisphere.py` (bin model) on
the same poses. **State it as a best-model comparison, not a strictly-matched A/B** (the
matched-budget M-array run is anchored separately).

**Table 2 — clutter edge risk-coverage:**
```bash
python scripts/eval_clutter_quad.py --ckpt checkpoints/codesign_quad_prod.pt \
    --data evals/clutter_v2 --pattern-set codesign_v2
```
Stratify by distance-to-discontinuity; split DEPTH/occlusion vs SHADOW. Watch for **depth-edge
false-consensus** (margin stays high but wrong → `acc@τ` collapses even at high τ) and the
**coverage cliff** (`cov@τ`→0).

## Open risks (don't paper over)
- **Planar render-cap.** codesign_v2 is the best-available *planar-designed* pattern, not
  render-optimal — the proxy is planar (no depth-varying defocus / full-render sampling).
- **Depth-edge false-consensus** may be partly intrinsic to the vote (blended two-surface phases →
  spurious peak). Clutter training may help but not fully close it — report the
  clutter-trained-vs-plane-trained delta honestly.
