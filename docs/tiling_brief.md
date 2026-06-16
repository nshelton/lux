# Tiled Inference for the One-Shot Correspondence Net — Discovery, Evidence, Implications

**Date:** 2026-06-16 · **Status:** for independent review · **Scope:** `lux/proj_net.py` (`ProjUNet`,
`predict_full`, `predict_tiled`), `scripts/eval_hemisphere.py`, `scripts/eval_capture.py`.

> All numbers below were produced on the Mac (`.venv`, torch 2.8, CPU for these evals so the live
> attention training on MPS was not disturbed). Synthetic data: held-out val samples from
> `renders/planar_loaf` / `renders/val_loaf`. Real data: `captures/test0..4`. Models: `proj_net_scratch.pt`
> (conv, fully trained, "CONV") and `proj_net_attn_ep22.pt` (attention, mid-training, "ATTN").

---

## TL;DR

1. **The models were being evaluated and deployed out-of-distribution.** They train on **256-px random
   crops** but `predict_full` runs the whole **1080×1920** frame. That mismatch silently degrades
   accuracy. Running inference as stitched **256-px tiles** (`predict_tiled`) removes the mismatch.
2. **It is a large effect.** Same conv weights, tiled vs full-frame: row-axis bin-acc **78.7% → 99.4%**
   (synthetic), real-capture row **78.6% → 94.5%**, and the **obliquity cliff at 45–60° lifts 50.8% →
   74.6%**. No retraining.
3. **Two different mechanisms, proven by controlled tests.** Conv's loss is **edge-localized** (frame
   top/bottom rows; interior is perfectly resolution-invariant). Attention's loss is **global** —
   even interior pixels collapse as the token count grows, and the **row axis far more than the column**.
4. **Consequence:** the project's two headline limitations — the **45° obliquity cliff** and the
   **row (v) deficit** — were **substantially evaluation/inference artifacts**, not fixed model limits.
   The deployed conv model is materially better than the docs say. **Bound (§9):** the cliff *moved out
   one bin*, it did **not** vanish — residual ~75% @ 45–60° synthetic and worse on real data.
5. **Action:** make tiling the default everywhere (done in code); re-baseline all benchmarks tiled;
   treat the conv-tiled model as the production baseline. Attention, tiled to 256, has no advantage
   over conv (its global-attention benefit is exactly what tiling removes) — a clean negative result.

---

## 1. The phenomenon: accuracy vs input size

Bin accuracy (column u / row v, %) for one synthetic sample, as inference input size grows. Tokens =
(size/16)². Training crop = 256 px = 256 tokens.

| input | tokens | CONV u / v | ATTN u / v |
|---|---|---|---|
| 256 px (train res) | 256 | 99.2 / 99.1 | 94.7 / 94.2 |
| 512 px | 1024 | 99.5 / 99.2 | 68.5 / 44.8 |
| 1080×1920 (full) | 8160 | 95.3 / **78.7** | **23.7 / 26.8** |

- **CONV** is nearly flat for the column (99→95) but drops on the row (99→78.7) at full frame.
- **ATTN** collapses on both axes at full frame (≈ chance), monotonically with token count.

This is *not* a difficulty/foreshortening effect — it is purely a function of the **input size**, with
the content fixed.

## 2. The fix: `predict_tiled`

Run every forward pass at the trained 256-px size and stitch. Two modes:
- `overlap=0` (margin-crop): each pixel from the tile where it sits ≥`margin` from the edge. Fast,
  but leaves square **seams** in low-confidence (background) regions where adjacent tiles disagree.
- `overlap>0` (recommended for deployment): overlapping tiles + **per-pixel max-confidence** selection.
  A tile's own edges have least context → lowest softmax confidence → they lose to a tile where the
  pixel is well-centred, so seams dissolve. `--tile-overlap 128` (50% overlap, ~2× forward passes).

Recovery on the ep9 attention checkpoint (full → tiled), bin u / v:

| sample | CONV full → tiled | ATTN full → tiled |
|---|---|---|
| planar | 95.3 / 78.7 → **99.5 / 99.4** | 23.7 / 26.8 → **95.1 / 94.9** |
| clutter | 94.1 / 79.0 → **97.5 / 97.8** | 26.4 / 18.4 → **87.9 / 89.6** |

Cost: ~4–5 s/frame on CPU at `overlap=0` (~60 tiles), ~2× that at `overlap=128`; trivial on GPU.

## 3. Mechanism — two distinct failures (controlled tests)

### 3a. Attention = global token-count sensitivity (interior-deep, row-asymmetric)

Test: fix a **128×128 central region** (same physical pixels) and grow the surrounding window. By
512 px the region's receptive field is saturated, so for a conv any further change is *not* local
context. Bin-acc of that fixed central region:

| window | tokens | CONV u / v | ATTN u / v |
|---|---|---|---|
| 320 | 400 | 99.8 / 99.7 | 98.8 / 99.4 |
| 512 | 1024 | 99.7 / 99.6 | 98.3 / **82.0** |
| 768 | 2304 | 99.7 / 99.4 | 97.9 / **11.8** |
| 1024 | 4096 | 99.7 / 99.3 | 98.0 / 35.7 |
| 1080 | 4489 | 99.7 / 99.4 | 98.2 / 41.4 |

- **CONV central region is constant (~99.7%)** at every size → conv is genuinely shift-invariant in
  the interior; GroupNorm's global statistics do **not** degrade confident interior predictions.
- **ATTN central region's v-bin collapses 99.4 → 11.8%** for the *same pixels with full local context*
  — this can only be the **global self-attention** (softmax over a token count it never trained on +
  absolute positional encoding extrapolation). The **column stays ~98%; the row collapses.** This
  axis-asymmetry is the direct cause of the "`median_dv` high while `median_du` recovered" observation
  (see §4c).

### 3b. Conv = edge-localized border effect (extreme v-bins only)

Test: conv v-bin accuracy by **camera-row band**, full-frame vs tiled, one synthetic sample:

| camera rows | full-frame v-bin | tiled v-bin |
|---|---|---|
| 0–180 (top) | **35.9%** | 99.4% |
| 180–360 | 82.9% | 99.5% |
| 360–540 (center) | 99.3% | 99.6% |
| 540–720 (center) | 99.3% | 99.4% |
| 720–900 | 94.8% | 99.4% |
| 900–1080 (bottom) | **60.1%** | 99.3% |

Conv's full-frame row loss is **entirely at the top/bottom frame edges**; the center is already 99.3%.
The degradation extends ~one receptive-field (~300 px) in from each edge. This is the documented
"camera-border context" component of the row deficit: edge pixels lack real surrounding pattern
(reflect-padded), and those already-marginal extreme-v-bin predictions regress to the center. Tiling
decodes those regions at training scale and lifts them to ~99%.

**Net:** ATTN fails *globally and interior-deep* (sequence length); CONV fails *only at the frame
edges* (border/RF). Same symptom under `predict_full`, same fix under `predict_tiled`, different cause.

## 4. Quantified impact

### 4a. Hemisphere obliquity benchmark (160 samples, binned by max(cam,proj) tilt)

Column bin-acc (the bench is column-only):

| obliquity | CONV full-frame (prior docs) | **CONV tiled** | ATTN ep22 tiled |
|---|---|---|---|
| 0–15° | 96.6% | **99.2%** | 98.1% |
| 15–30° | 95.1% | **99.3%** | 98.3% |
| 30–45° | 87.0% | **97.8%** | 89.7% |
| 45–60° | 50.8% | **74.6%** | 50.3% |
| 60–75° | 4.7% | **9.7%** | 5.1% |
| med\|du\| (0–15°) | 0.28 px | **0.22 px** | 0.53 px |

Tiling lifts conv by **+11 pts at 30–45° and +24 pts at 45–60°** — the "cliff" is pushed out ~a full
bin. 60–75° stays hard (real anamorphic-resolvability limit).

### 4b. Real captures (5 scenes), both tiled

| scene | CONV u-bin / med\|du\| | ATTN ep22 u-bin / med\|du\| |
|---|---|---|
| test0 | 94.8% / 1.05 px | 92.7% / 1.23 px |
| test1 (grazing) | 47.2% / 18 px | 26.2% / 129 px |
| test2 | 91.0% / 1.23 px | 79.5% / 1.93 px |
| test3 (oblique) | 71.0% / 2.4 px | 51.0% / 11 px |
| test4 | 94.8% / 1.05 px | 92.4% / 1.36 px |
| **test4 row (v)** | 94.5% / 0.47 px | 92.0% / 0.90 px |
| **test4 full (u,v)** | **91.4%** | 87.7% |

For reference, conv **full-frame** on test4 (prior docs) was u 89.6% / **v 78.6%** / uv 73.0% — tiling
adds **+16 pts row, +18 pts joint** on real hardware, same weights.

### 4c. The `median_dv` artifact on the live val curves

The in-loop `evaluate()` uses `predict_full`, so the training val curves are full-frame. Median |d| is
a **threshold metric**: it stays at the wrong-bin distance (~hundreds of px) until bin-acc crosses
50%, then snaps to ~1 px. At ep22 (full-frame): column bin-acc ~50–63% (just over → `median_du` ≈ 1 px)
but row bin-acc ~30–35% (under → `median_dv` ≈ 200 px). **Tiled, both axes are 88–99% and both medians
are 0.4–0.7 px.** So the alarming `median_dv` is (full-frame eval) × (median threshold) × (attention's
row-asymmetric collapse) — not a real row problem in the deployable (tiled) model.

| ep22 val sample | full u / v · md\|du\|/\|dv\| | tiled u / v · md\|du\|/\|dv\| |
|---|---|---|
| planar0 | 51.8 / 32.0 · 1.3 / 209 px | 98.5 / 98.7 · 0.4 / 0.4 px |
| clutter0 | 60.2 / 29.8 · 1.0 / 299 px | 93.0 / 96.2 · 0.5 / 0.5 px |

## 5. What it means

- **The benchmark history understated the model and mischaracterized its limits.** Every conv number
  in the prior docs (cliff at 45°, row deficit) was full-frame, i.e. out-of-distribution. The cliff and
  the row deficit are **substantially evaluation/inference artifacts**. The real conv model: ≥97% to
  45° obliquity, ~75% at 45–60°, sub-0.25 px precision — usable for projection mapping to ~60° tilt.
- **The "fix the oblique training distribution" priority is partly pre-empted.** Much of the 45–60°
  gap closed for free at inference. Only 60–75° remains a genuine information limit (≈10%).
- **The attention experiment is a clean negative result.** Tiling-to-256 is exactly what removes
  attention's global-context advantage, so tiled-attn (ep22) only matches *old full-frame conv*, not
  tiled conv, and it won't close a 24-pt gap at 45–60° in the remaining epochs. Consistent with the
  prior prediction.

## 6. Recommendations

### Eval / deployment (do now — mostly landed)
1. **Tile all inference.** `predict_tiled` is wired into `eval_hemisphere.py`, `eval_capture.py`,
   `predict_proj_net.py` via `--tiled`, auto-enabled for `arch==attn`. **Recommend making it the
   default for conv too** (it strictly helps), and `--tile-overlap 128` for any visual/deployment
   output (removes seams; default in `predict_proj_net.py`).
2. **Re-baseline every benchmark tiled** and update the docs/memory; the full-frame numbers are stale.
3. **Make the in-loop `evaluate()` tiled-aware** so future training val curves are honest (this run's
   curves stay full-frame; judge it by train curves).

### Training (optional — to make models *natively* resolution-robust)
4. **Conv (cheap, removes the residual edge effect):** the only conv weakness is frame-edge marginal
   pixels. Train with the **full vertical range in view** (full-height 1080×W strips, or larger crops),
   so edge rows are in-distribution — or simply accept tiling, which already fixes it.
5. **Attention (only if global context is genuinely wanted):** it must be made resolution-invariant —
   **windowed/local attention** (constant tokens per attention op), or relative positional encoding +
   sequence-length-normalised attention, or training at native resolution (memory-prohibitive here).
   Given tiled-attn ≈ conv, this is **not** recommended over the conv path.
6. **GroupNorm note:** it does *not* harm conv interior predictions (§3a), so it need not be replaced;
   the conv edge effect is border/RF, not a global-norm problem.

### Strategy
7. **Conv-tiled is the production baseline.** The highest-leverage remaining work is the documented
   **real-data Gray-code fine-tune** (closes the sim→real gap that still separates conv-tiled's ~95%
   synthetic from ~91% real on good scenes), not architecture changes.

## 7. Caveats, confounds, open questions (for review)

- **Sample sizes:** §1–3 and §4c use 1–4 held-out synthetic samples (per-pixel statistics, ~10⁶ px
  each, so per-number variance is low, but scene diversity is limited). §4a is 160 samples; §4b is 5
  real scenes. The row-band/central-region tests are single-sample illustrations of a mechanism, not
  population estimates.
- **ATTN is mid-training (ep22/30).** Its absolute numbers will improve; the *comparison* conclusion
  (tiled-attn ≈ old-full-frame-conv < tiled-conv) is unlikely to change but should be re-checked at ep30.
- **Conv mechanism not fully decomposed.** §3 proves conv's loss is edge-localized and that GroupNorm
  doesn't harm the interior, but the edge effect itself may combine reflect-pad context-truncation and
  GroupNorm-sensitivity *of marginal edge activations*; these were not separated. The deployable
  conclusion (tiling fixes it) does not depend on the decomposition.
- **Attention axis-asymmetry (row ≫ column) is observed, not explained.** Why the row collapses first
  under token growth (positional-encoding split? aspect ratio? the pre-existing row deficit amplified?)
  is open. It is robust across samples but the cause is a hypothesis.
- **Tiling is not free of artifacts.** Hard-stitch seams (fixed by overlap+max-conf); a residual
  assumption that 256-px context suffices for correspondence (it does here — the M-array decodable
  window is ~20 px — but a pattern needing longer-range disambiguation would not tile cleanly).
- **`predict_tiled` correctness:** verified that chained `arr[slice][mask]=` writes through the base
  array (basic slicing → view), and that invalid (NaN) tiles score below any valid tile in max-conf.

## 8. Reproduction

```bash
# hemisphere bench, tiled (auto for attn; --tiled to force for conv)
python scripts/eval_hemisphere.py --ckpt checkpoints/proj_net_scratch.pt --tiled
python scripts/eval_hemisphere.py --ckpt checkpoints/proj_net_attn_last.pt          # auto-tiled

# real captures, tiled, seam-free
python scripts/eval_capture.py --captures captures/test4 --ckpt <ckpt> --tile-overlap 128

# the mechanism tests (central-region vs window; conv v-bin by row band) were one-off scripts:
#   load checkpoint, predict_full on centered crops of growing size / on the full frame,
#   bin to N_BINS_U/V, aggregate over a fixed central region / per camera-row band.
```

Core implementation: `lux/proj_net.py:predict_tiled` (margin-crop + overlap/max-confidence modes) and
`predict_full`. See also `docs/net2_plan.md` and `memory/hemisphere-obliquity-cliff.md` (updated).

## 9. Review outcome — ratified, with bounds

Independently reviewed and **ratified** (the fixed-central-region design and the overlap=0/128 split
were called out as cleanly ruling out the "tiling is just test-time ensembling" confound; the
real-hardware delta, not the synthetic, is the convincing number). Binding corrections to the framing:

- **The cliff *moved out one bin*; it did not vanish.** "Substantially an eval artifact" must not slide
  into "solved." Tiled, the residual is still **74.6% @ 45–60° and 9.7% @ 60–75° on synthetic, and far
  worse on real** (test3 oblique 71%, test1 grazing 47%). The residual cliff is real and **larger on
  real than synthetic.** Our cliff diagnostics weren't pre-empted — they were *contaminated* (run
  off-distribution) and can now finally be run clean. **Next: re-run overfit-one-oblique-batch at
  256/training scale on a 60–75° patch** — that, not the full-frame bench, tells us whether the residual
  floor is an information limit or still fixable. The discovery cleans the instrument; it doesn't retire
  the measurement.
- **Attention: closed.** Properly tested now; verdict stands (tiled-attn ≈ old-full-frame-conv <
  tiled-conv). The row-vs-column asymmetry is a real curiosity but lives on a dead path — don't spend
  cycles explaining it unless windowed/RoPE attention is resurrected (the negative result says not to).
- **RF question is half-answered.** The central-region test proves RF-sufficiency *for that sample's
  (frontal/easy) central content* — it does **not** prove RF-sufficiency at 45–75° obliquity, which is
  the cliff. Keep **RF-vs-anamorphic-span on the list, run it tiled** (at training scale).
- **Per-axis Jacobian/SVD weighting: dead — conceded.** The v-deficit localizes to top/bottom **camera
  rows** (center rows were 99.3% full-frame), not to oblique *surfaces*; foreshortening would have hit
  oblique center-row pixels too and didn't. So it's edge context-truncation, not directional
  compression, and tiled u/v come back symmetric. Drop the lever unless a tiled per-axis bench on
  oblique **interior** pixels still shows asymmetry (it is not expected to).
- **Strategic redirect (the real consequence): stop optimizing synthetic obliquity.** Conv-tiled is
  ~99% synthetic but ~91% real on good scenes and 47–71% on hard real scenes, while holding 74.6% on
  *synthetic* 45–60° — i.e. it handles synthetic obliquity far better than real obliquity, so **sim2real
  and obliquity are now entangled in the only numbers that matter (the captures).** The **real-data
  Gray-code fine-tune (§6 rec #7) is the disentangler and the top lever.**
- **Tiling caps usable context at 256 px — so rec #4 is doing double duty.** If the residual cliff is
  RF-limited (diagnostic pending), you **cannot** grow RF past 256 by tiling; only retraining at larger
  crops can. So full-height/larger-crop training is simultaneously the edge fix **and** the only way to
  raise the in-distribution context ceiling for the hardest oblique pixels **and** the only way to bring
  the *true outermost frame border* in-distribution (even tiling can't — the outermost tile still sees a
  reflect-padded edge no neighbor can supply). One intervention, three wins.

**Two residual catches before re-baselining:**
1. **Max-softmax tile selection inherits the very miscalibration the planned learned-correctness
   (ConfidNet) head is meant to fix** — an overconfident-wrong edge prediction can win a seam. When that
   head lands, it should drive **tile stitching**, not just abstention (one calibrated signal, two
   consumers). Until then, spot-check seams in low-texture/background regions.
2. **Re-baseline the full 160-sample hemisphere bench and all captures tiled** before treating any
   specific delta (e.g. +24 @ 45–60°) as final — §1–3 are 1–4 samples (per-pixel n is huge, scene
   diversity is not).

**Methodology log (load-bearing).** This was a confounded *evaluation* — train≠test resolution, never
flagged — that sat under three rounds of cliff-attack planning while a measurement bug ate ~24 points.
Guardrail to add: **"evaluate at the training distribution."** And log the §4c trap: **median-|d| is a
threshold metric** (wrong-bin distance until bin-acc crosses 50%, then it snaps to ~1 px) that masked
the per-axis story and made `median_dv` look like a model defect. Both are the same failure as the
"don't change two things at once" lesson, one level up — *the experiment was clean but the ruler was bent.*
