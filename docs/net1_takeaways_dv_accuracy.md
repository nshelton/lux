# net1 takeaways — the row (dv) accuracy deficit

Investigation log + conclusions for why **ProjUNet decodes the projector column (u)
better than the row (v)**, and what to do about it. Dated 2026-06-15. All numbers
below are from `checkpoints/proj_net_scratch.pt` (the from-scratch planar+aug net,
val |du| 0.25px).

---

## TL;DR

- The v (row) head lags the u (column) head by **~12 bin-accuracy points**, and this
  is **intrinsic to the synthetic training distribution** — it is *not* a sim-to-real
  or capture artifact (the same gap appears on held-out synthetic val).
- The deficit is **entirely at the edge bins**. Center bins are symmetric (u≈v≈94–95%);
  the gap is at the extremes of the coordinate range (v-edges 33–65% vs u-edges 66–89%).
- Mechanism: the classifier **regresses range-extremes toward the center** when the
  input is ambiguous. Two separable contributors, roughly equal:
  1. **Camera-border context** — edge pixels at the image border lack surrounding
     pattern; the net under-uses the locally-unique window there.
  2. **Intrinsic extreme-bin bias** — even with full context, extreme bins lose to the
     center; the extremes are also the rarest bins.
- **Ruled out** (with evidence): foreshortening/tilt coverage, cell anisotropy,
  reference/registration error, codebook aliasing, bin-coverage starvation, and the
  per-pixel aspect-ratio argument.
- **Cheap principled fix:** retrain with the **full vertical range in view**
  (full-frame, or full-height 1080×W strips). **Deeper fix:** co-design the pattern so
  the bin is *legible* (multi-scale: coarse bin-code + fine M-array).
- **Landed this session:** joint confidence (`min(conf_u, conf_v)`), which makes the
  row failures gate-able (test4 row p95 571px → 1.7px at conf_v≥0.9).

---

## 1. Model setup (context)

`ProjUNet` (`lux/proj_net.py`) maps one M-array capture → dense projector
correspondence. The head (99 channels) is **classification + offset, per axis**:

| | bins | bin width | what it is |
|---|---|---|---|
| u (column) | `N_BINS_U = 60` | 1920/60 = **32 px** | coarse softmax over columns |
| v (row) | `N_BINS_V = 36` | 1080/36 = **30 px** | coarse softmax over rows |
| offsets | 2 | — | within-bin fraction [0,1], linear (subpixel) |
| validity | 1 | — | BCE logit |

Decode: `coord = (argmax_bin + offset) * bin_px`. The loss (`proj_loss`) is
`CE(u) + CE(v) + offset_weight·L1(offsets) + BCE(validity)`.

**Critical architectural fact:** u and v are **two independent marginal softmaxes**,
argmaxed independently — *not* a single joint (u,v) class id over 60×36 cells. So each
v-bin is supervised by every valid pixel in its row band (across all columns and all
images) → v-bins are **not** coverage-starved.

Pattern (`scripts/gen_patterns.py:_marray`, `cell=4, win=5`): random binary grid,
repaired so every **5×5-cell (20 px) window is globally unique**. So 4 px cell,
20 px decodable window, 480×270 cell grid.

---

## 2. The symptom

| metric | real `test4` | synthetic val |
|---|---|---|
| u-bin accuracy | 90.8% | 88.5% |
| v-bin accuracy | **79.0%** | **75.8%** |
| full (u,v) bin | 74.3% | — |
| median \|du\| (✓bin) | 1.11 (1.00) px | — |
| median \|dv\| (✓bin) | 0.63 (0.46) px | — |

The within-bin **offset is sub-pixel and fine on both axes** — the deficit is purely
the **coarse bin classification** for v. The ~12-point u-vs-v gap is the same on real
and synthetic → **intrinsic, not sim-to-real.** (Synthetic median |dv| looks large in
aggregate only because the val set has brutal cluttered scenes; the *gap* is the robust
signal, and on a clean plane like test4 within-bin v error is 0.46 px.)

---

## 3. Hypotheses ruled out (with evidence)

### 3a. Foreshortening / tilt coverage — NO
Measured local cell-compression (camera-px per 4px cell, from the gradient of the
GT projector coordinate). test4's plane is steeply foreshortened (median **3.2 cam-px/cell**,
frontal would be ~12–13), yet the net decodes it at **sub-pixel median**. The training
loaves already contain this compression (val/planar median ~3.9 cam-px/cell). And
`corr(v-bin acc, cell-compression) = +0.05` — essentially zero.

### 3b. Cell anisotropy — NO
test4 cells are **isotropic**: u 3.4 cam-px/cell, v 3.2, ratio **1.05**. Training cells
are also isotropic (per-sample |anisotropy| median ~1.0). So the cells aren't "squashed
vertically."

### 3c. Reference / registration — NO
Confirmed by capture setup: nothing moves between the graycode / graycode_h / marray
captures, so the (u,v) reference is exactly registered.

### 3d. Codebook aliasing — NO (with caveat)
M-array self-similarity (fraction of matching cells vs shift) on the 270×480 cell grid:

```
chance ~ 0.500
VERTICAL  : median 0.500, max 0.513 @ shift 259 cells; @ half-height (135) ~0.50
HORIZONTAL: median 0.500, max 0.520 @ shift 478 cells
```

Flat at chance everywhere, including the ~half-height shift where the net aliases. So
the *pattern* has no near-repeat. (Caveat: a global autocorrelation cannot see a
*localized* near-collision between two small regions — not fully excluded.)

### 3e. Bin coverage / starvation — NO
Because the heads are marginal, each v-bin is heavily sampled. Worst v-bins (30–34) sit
at **average frequency** (2.2–2.9% vs the 2.8% uniform mean), so they're hard, not rare.

### 3f. Aspect ratio, per-pixel — NO (this was an early over-claim)
With independent heads and a roughly **symmetric ~300px receptive field**, a top-border
pixel and a left-border pixel lose the same fraction of context. 16:9 only changes how
*many* bins are border-affected (the aggregate average), not the per-pixel edge depth —
yet v-edges (33–52%) are far deeper than u-edges (66–80%). So aspect ratio alone does
not explain it.

---

## 4. The actual finding

### 4a. It's an edge-bin phenomenon; the center is symmetric
Per-bin accuracy profiles (synthetic val):

```
V-BIN acc, top→bottom (36 bins):
 72 58 52 57 64 68 75 | 82 87 82 84 88 91 94 94 94 95 95 94 94 94 94 95 95 92 92 | 83 68 63 59 33 41 50 64 65 80
 \____ top edge ____/    \_________________ middle: 91-95 _________________/        \________ bottom edge ________/

U-BIN acc, left→right (60 bins):
 80 66 68 81 85 90 ... [middle 93-96] ... 76 80 83 86 84

thirds:  v = top 73% / mid 95% / bottom 66%      u = left 84% / mid 94% / right 89%
```

**Mid-range u ≈ mid-range v ≈ 94–95%.** The entire gap is at the extremes.

### 4b. Errors are distant, pulling edges toward the center
v-bin confusion (top wrong prediction):

```
gt  acc  |err|>3   →
 1  50%    49%     bin 17  (+16)
 2  58%    41%     bin 18  (+16)
28  56%    44%     bin  9  (−19)
30  24%    75%     bin 11  (−19)
31  31%    69%     bin 12  (−19)
```

Misses are **not local neighbors** — they jump ~16–19 bins (≈ half the range) back into
the well-populated middle. The classifier collapses ambiguous edge inputs onto the
center of the range. (Note: the very-bottom bin 35 *recovers* to 80% — so it's not a
clean monotonic border falloff; it's a center-attractor.)

### 4c. Context vs intrinsic — both contribute (the decider)
Edge-v accuracy as a function of camera **vertical border distance** (vbd ∈ [0,1]):

```
EDGE v-bins (0-5, 28-35):  vbd 0.0-0.1: 39.8%   0.1-0.3: 38.5%   0.3-0.6: 61.7%   0.6-1.0: 68.7%
MID  v-bins (10-25):       vbd 0.1-0.3: 84.3%   0.3-0.6: 95.4%   0.6-1.0: 95.0%
```

Two separable effects:
- **Context (~+30 pts):** edge-v jumps 39% → 69% from image-border to interior. The net
  under-uses the locally-unique 20px window at borders and leans on context that's
  missing there.
- **Intrinsic extreme-bin bias (~−26 pts):** even in the interior with full context,
  edge-v (69%) ≪ mid-v (95%). The classifier regresses range-extremes toward the center
  (the extremes are also the rarest bins: 1.4–2.2% vs 3.2%).

`corr(v-bin acc, border-distance) = +0.86` (partly circular, since edge-v-bin ≈ vertical
position, but consistent with the context effect).

### 4d. Why v is worse than u
The center is symmetric, so it's specifically the *extremes*. Leading (not fully
confirmed) reason: the 16:9 projector's **narrow vertical FOV** puts its top/bottom rows
on more grazing / off-surface geometry, so extreme v-bins are rarer and lower-quality in
training than extreme u-bins. (One u-vs-v extreme-bin frequency/validity comparison would
confirm.)

---

## 5. Bin ↔ window/cell theory

- 5×5 cells = **20px window** = smallest globally-unique patch. bin = 30–32px.
- Governing constraint: **bin ≥ window**, so each bin is identifiable from one window.
  Satisfied — the window *over-resolves* the bin (a 20px patch pins position to a few
  px, well inside a 30px bin). **So there is no information shortage**; the failure is
  the net's *calibration/use* of the signal at the extremes, not missing signal.
- Bin size sets the **split between the discrete classifier and the continuous
  regressor**: larger bins = easier classification but more px for the offset to resolve
  (harder subpixel); smaller-than-window bins would be unidentifiable. Current 30–32px is
  a reasonable sweet spot.
- **Wart:** v-bins are 30px = **7.5 cells → misaligned** to the 4px cell grid; u-bins are
  32px = 8 cells (aligned). v-bin boundaries fall mid-cell. Minor (center-v is fine), but
  bins should be `integer × cell` on both axes.

---

## 6. Design directions considered

### Hilbert / joint UV id — rejected
- A joint (u,v) id collapses the **marginal data efficiency** — each marginal bin is
  trained by O(all rows) pixels; each joint cell by ~2160× fewer. Larger bins reduce the
  class count but shift the burden to the offset regressor.
- Plain cross-entropy is **permutation-invariant** — Hilbert ordering buys nothing
  unless you also switch to an ordinal/regression loss. The current per-axis
  "coarse bin + continuous offset" already encodes 2D locality natively.

### Multi-scale / "legible bin" pattern — the deep principled lever
The code is currently **single-scale**: all position info lives in the 20px window, and
the bin is *inferred* by fully decoding absolute position then quantizing — a global
codebook memorization, which is exactly what's fragile at edges/under blur. A
**hierarchical** code — a coarse layer that directly *names the bin* + the fine M-array
for within-bin — turns bin classification into a **local read** (why Gray-code/phase
hierarchies are robust).
- Caveat on "outline the bin with a square": a *periodic* grid gives within-bin **phase**
  but not bin **identity** (ambiguous across bins → needs unwrapping). For identity the
  coarse layer must be **coded** (each bin a distinct coarse symbol).
- Cost: one-shot ⇒ coarse+fine must be multiplexed in one image (e.g. coarse code in one
  color channel, M-array in another; or low-freq + high-freq superimposed). New pattern +
  new capture + full retrain; depends on color/contrast fidelity.

---

## 7. What landed this session (code)

- **Joint confidence** (`lux/proj_net.py:predict_full`): `return_conf=True` now returns
  `min(conf_u, conf_v)` (was column-only, blind to row failures); `conf_per_axis=True`
  returns `(uv, conf_u, conf_v)`. `scripts/eval_capture.py` gates column on conf_u, row on
  conf_v, plus a joint (u,v) sweep. **Result:** gating row on conf_v collapses test4 row
  p95 **571px → 1.7px** at conf_v≥0.9 (58.6% coverage). On bad rows conf_v drops to 0.57
  (vs 0.92 good) while conf_u stays high at 0.80 — i.e. the net *knew*, we just weren't
  reading the row head. **Keep this.**
- **`--focal-gamma` / `--v-weight`** flags in `scripts/train_proj_net.py` + focal CE in
  `proj_loss` (mean-1 normalized, scale-preserving). These are the **abandoned
  symptom-suppressor** path; default off (`0`/`1.0` = original behavior). Harmless to keep
  or remove.

---

## 8. Recommended next steps

1. **Cheap + diagnostic:** retrain (warm-start) with the **full vertical range in view** —
   full-frame, or full-height 1080×W strips. This exposes the genuine frame-border
   decoding condition *and* both range endpoints every step, targeting both the context
   and extreme-bin parts. If it closes the gap, the pattern was always sufficient and the
   net was under-trained; if it plateaus, that's the evidence the single-scale code can't
   express the bin robustly → justifies the redesign. **Success metric:** edge-bin v-acc
   (39%→? at borders, 69%→? interior), not just overall accuracy.
2. **Deeper:** pattern co-design — minimum, cell-aligned bins; maximum, the multi-scale
   coarse-bin-code + fine M-array. Do *after* (1) since it's a capture+retrain cycle.
3. Optional confirmation: u-vs-v extreme-bin frequency/validity comparison to nail the
   asymmetry; and bias the renderer's pose sampling to image the projector frame edges
   more often/frontally (fix the data thinness at the source).

### Corroboration
A separate investigation (see the `hemisphere-obliquity-cliff` note) independently found
proj_net fails past ~45° tilt via **coarse-bin error (not coverage)**, concluding
*"fix the oblique training distribution."* Two investigations point the same way: the
problem is bin classification + training distribution, and the fix is **structural
(data/training or pattern), not a loss knob.**

---

## 9. Reproduction

The diagnostics above were one-off scripts (per-bin accuracy/frequency, v-bin confusion,
pattern autocorrelation, edge-bin-vs-border-distance, compression distributions). Each:
load `checkpoints/proj_net_scratch.pt`, run `predict_full(..., conf_per_axis=True)` over
synthetic samples from `renders/val_loaf` / `renders/planar_loaf` (GT in the loaf's
`gt.npy`), bin to `N_BINS_U/V`, and aggregate. Local cell-compression = `4 / |∂gt/∂axis|`
on a mask-aware-smoothed GT coordinate. Real-capture scoring is
`scripts/eval_capture.py --captures captures/test4 --reference hybrid` (test4 has the
`graycode_h/` horizontal set, so it scores dv).
