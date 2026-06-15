# net1 — Takeaways & Handoff for the Next Run

Consolidated learnings from **net1** (`proj_net` / `proj_net_scratch`), the one-shot
M-array correspondence net, written to seed **net2**. Pairs with `session_summary.md`
(which has the earlier history) — this file is the current state + what to do next.

## TL;DR
- **Best model: `checkpoints/proj_net_scratch.pt`** (from-scratch, conv U-Net, epoch 23/30).
  Beats the clutter-only baseline on every obliquity bin; sub-pixel precision ~2× better.
- **From-scratch beat warm-start decisively.** Warm-start entrenched (couldn't relearn the
  oblique codebook at a decaying LR). Train net2 from scratch too.
- **The obliquity cliff is partly an information limit.** 30–45° is now solid (87%), 45–60°
  modest (~49%), 60–75° stays at the floor (~5%) even with targeted hemisphere data →
  confidence-mask it, don't chase it.
- **Sim→real works on non-grazing scenes** (real: 88% col / 81% row / 74% full-(u,v) at
  99.5% coverage). The residual ~1px sub-pixel gap is a clipping/contrast + PSF/noise-shape
  mismatch — the renders are "too clean."
- **Highest-leverage next step: a real-data fine-tune** using the Gray-code (u,v) reference
  as pseudo-GT.

## Where net1 landed (the numbers)

### Hemisphere bench (synthetic, median-agg per bin) — baseline vs from-scratch
| obliquity | baseline `proj_net.pt` (bin% / med\|du\|) | **`proj_net_scratch.pt`** (bin% / med\|du\|) |
|---|---|---|
| 0–15° | 96.1% / 0.56px | 96.7% / 0.28px |
| 15–30° | 94.9% / 0.49px | 95.1% / 0.23px |
| 30–45° | 81.9% / 1.01px | **87.5% / 0.53px** |
| 45–60° | 46.5% / 26.8px | 48.8% / 18.2px |
| 60–75° | 4.2% / 379px | 5.1% / 438px |

Read: precision ~2× better everywhere; frontier pushed out ~one bin (30–45 **+5.6**); 45–60
modest (+2.3); 60–75 flat (information limit).

### Training trajectory (from-scratch, conv, lr 1e-3→1e-4 cosine, offset gated)
- Chance plateau epochs 1–2 (loss ~7.7–8.5, bin ~4%); **phase transition at epoch 3**
  (loss 7.2→3.8, bin 11→28%) — right on schedule.
- By ep23: val median |du| **0.28px**, |dv| 0.31px, bin 87%, validity-IoU 0.969.
- ~53 min/epoch on the Mac (MPS). **Stopped at ep23/30** — killed mid-ep24 (leaked-semaphore
  shutdown, not a crash; likely the Mac slept). Plateaued (val 0.29→0.28 over the last 3
  epochs), so the missing 7 epochs are marginal. Resumable from `proj_net_scratch_last.pt`.

### Real-data eval (`eval_capture.py`, hybrid Gray+phase reference)
- **scene01 / test4** (fairly frontal, captured with `graycode_h`): col **88.2%**, row
  **81.1%**, **full (u,v) 74.1%**, coverage 99.5%. At **conf≥0.9: 80.8% coverage @ 95.6% col
  / 81.9% uv, p95 3.1px**, median |du| 1.17 / |dv| 0.83px. → the net working as intended.
- **test1** (grazing floor): col 33.2%, conf≥0.9 coverage 0.8%. Coverage-limited because the
  whole plane sits at ≥45° obliquity — the cliff, on a real surface (not a model/capture bug).
- baseline→scratch on test1: 23.7%→33.2% (scratch better even on the hard scene).

## Load-bearing learnings (carry to net2)

1. **Conditioning ≫ capacity.** classification(bins)+offset head beat coordinate regression
   massively. Keep the head framing.
2. **Know the chance floor** (ln60+ln36 ≈ 7.7). From-scratch sits at chance ~2–4 epochs then
   phase-transitions (~ep3 here). Don't panic before then. Overfit-one-batch = 2-min bug test.
3. **Offset weight is the sub-pixel lever.** Gate it in once bins form; raising it took net1
   from ~5.7→1.0px median.
4. **LR floor matters** — cosine→0 is wrong while a term hasn't learned; use `--lr-min`.
5. **Aggregate val median is a poor obliquity proxy** (easy-pixel dominated). Use the
   hemisphere bench + per-epoch `--snapshots`; pick **best-on-hemisphere**, not best-on-val.
6. **Warm-start entrenches.** A converged net at a decaying LR can't relearn a new (oblique)
   codebook — it defends its basin. From-scratch (high LR) learns frontal+oblique jointly.
   Confirmed: warm-start flat ~46% with no precision gain; from-scratch moved 30–45 + halved
   precision.
7. **The cliff = part support-hole, part information limit.** Hemisphere data moved 30–45°
   (+5.6) and helped 45–60° a little, but 60–75° didn't budge (5%) despite targeted data →
   anamorphic cell compression below resolvability. Mask 45–75°, don't chase 60–75°.
8. **Confidence (u-bin softmax max) is well-calibrated.** On real data, low-conf pixels are
   genuinely wrong (not hidden-good — verified by the threshold sweep), so masking does its
   job. The conf≥0.9 set is the trustworthy one.
9. **Real-data coverage is geometry-limited, not checkpoint-limited.** test1's poverty was the
   grazing floor, not the model. Shoot non-grazing scenes to see the net shine.
10. **The sim→real residual (~1px sub-pixel) = the renders are "too clean."** Three *shape*
    mismatches (envelope size is fine — aug is broader than real):
    - **contrast/clipping**: renders clip ~0.5% of lit px vs real ~4% — **now partly fixed**
      (see below);
    - **PSF shape**: Gaussian blur aug vs real disc/asymmetric lens+projector PSF;
    - **noise shape**: white i.i.d. Gaussian vs signal-dependent + demosaic-correlated.
    Bins survive these (88%); the **offset head** doesn't (→1px). It only hits sub-pixel.
11. **Don't change two things at once** — isolate variables (net1 once confounded planar-data
    + aug).
12. **MPS is GPU-bound (~50 img/s).** loaf memmap + AMP fixed the data path. Watch CPU
    contention (camera/WindowServer starve the dataloader workers). **barnacle (RTX 2080 Ti)
    is unstable — crashes; train on the Mac for now** (~53 min/epoch).

## Built this session (available for net2)
- **Horizontal Gray codes** — `patterns/graycode_h`, `GrayCodeMethod.patterns(axis="y")` +
  `decode_rows` (the temporal decode is orientation-agnostic, so rows reuse the column
  decoder). Capture both axes → exact per-pixel **(column, row)** correspondence.
- **`eval_capture.py` extended** — decodes `graycode_h` → scores `dv`, **full (u,v) bin
  accuracy**, a `uv acc` confidence-sweep column, and fills the row plots in `uv_grid.png`.
  Column-only still works without `graycode_h`.
- **Capture-app Process panel** (`capture_app.py`) — run the net on the live frame or any
  captured set, per-set decode buttons, result panel beside the feed, Column/Confidence/
  Packed views + min-conf slider, `--device` to keep inference off the training GPU.
- **Augmentation contrast/clipping term** (`lux/proj_net.py:_augment_crop`) — high-contrast /
  highlight-clipped regime on 50% of crops (clip 0.5%→5.5%, std 0.24→0.28, matching real's
  4.1%/0.30). Input-only, probabilistic. **Only affects future training — net1 didn't have it.**
- `preview_augment.py` to eyeball the aug distribution vs a real capture.

## Recommendations for net2

**Priority 1 — real-data fine-tune (closes the ~1px sim→real gap):**
- Fine-tune `proj_net_scratch.pt` on a handful of real captures, using the **hybrid Gray-code
  (u,v) as pseudo-GT**, supervised on the white-mask ∩ reference-confident pixels. Teaches
  real clipping + PSF + noise-shape all at once — cheaper and more direct than modeling them.
- Low LR (~1e-4), few epochs; consider unfreezing decoder+head only (keep encoder features).
  Watch for overfitting to the few scenes — the new clip aug + held-out scenes help.
- Capture several scenes at varied **non-grazing** obliquity, each with marray + graycode +
  graycode_h + phaseshift.

**Priority 2 — if retraining from scratch:**
- Keep from-scratch (not warm-start), conv, lr 1e-3→1e-4, gated offset, 30 epochs. The new
  contrast/clip aug is already in `_augment_crop`.
- Consider explicit **PSF/noise-shape aug**: measure the real PSF (edge-spread across the
  white/black boundary) and real noise (from the black frame), and match those instead of the
  generic Gaussian/white models.
- Pick the **best-on-hemisphere** snapshot.

**Don't bother:**
- Chasing 60–75° (information limit) — confidence-mask 45–75° instead.
- Warm-starting from net1 (entrenchment).
- Finishing net1's ep24–30 (marginal; won't move the cliff).

**For real-world testing:**
- Shoot **non-grazing** scenes (88%/81% there vs ~33% on a grazing floor).
- Always capture `graycode_h` too → full (u,v) + `dv` scoring.
- The confident subset (conf≥0.9 ≈ 80% coverage @ ~95% on a good scene) is what feeds the
  projection-mapping homography — sparse-but-correct beats dense-but-wrong for a warp fit.

## Key files & commands
- **Models:** best `checkpoints/proj_net_scratch.pt` (ep23) + snapshots `_ep01..ep23` +
  rolling `_last`. Baseline `proj_net.pt`. Warm-start (entrenched, don't reuse)
  `proj_net_mixed_*`.
- **Hemisphere bench:** `python scripts/eval_hemisphere.py --ckpt <ckpt>` →
  `evals/hemisphere/results_<stem>/` (per-bin table = the cliff test).
- **Real-capture eval:** `python scripts/eval_capture.py --captures captures/<scene> --ckpt <ckpt>`
  (auto-scores `dv` + full (u,v) when `graycode_h/` is present).
- **Train:** `python scripts/train_proj_net.py --loaf renders/val_loaf renders/planar_loaf
  --mid conv --epochs 30 --batch 64 --amp --lr 1e-3 --lr-min 1e-4 --offset-weight 2
  --gate-offset 6 --snapshots --out checkpoints/<name>.pt --logdir runs/<name>`
- **Patterns (incl. graycode_h):** `python scripts/gen_patterns.py`
- **Aug preview:** `python scripts/preview_augment.py --loaf renders/planar_loaf`
- **Loaves:** `renders/val_loaf` (clutter ~10k), `renders/planar_loaf` (planar hemisphere ~10k).
