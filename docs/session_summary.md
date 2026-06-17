# Session Summary — One-Shot SL Correspondence Net

> **Session 2 recap: `docs/session2_recap.md`** — tiled inference reframed the cliff/row
> deficit as eval artifacts; the physically-ordered aug + full training gave the best
> model yet (val 0.23px / bin 93.6%, hemisphere 45–60° 78%); the 60–75° floor is a
> confirmed information limit. Viz: consolidated summary.png, per-pixel viewer, dashboard.

## What was built
- **`lux/proj_net.py`** — `ProjUNet`: 4-level conv U-Net (7.85M, base 32) + a
  **classification+offset head** (60 u-bins×32px + 36 v-bins×30px + 2 offsets +
  validity). Optional `--mid attn` transformer bottleneck (17M). Loss = CE_u + CE_v +
  w·offset_L1 + BCE(validity). `predict_full(..., return_conf=True)` gives the c2d map
  + per-pixel confidence. Crop augmentation `_augment_crop` (gain/gamma + prob blur
  σ≤1.5 + prob noise σ≤0.05).
- **Data**: `gen_training_data.py` (clutter), `gen_planar_dataset.py` (planar
  junctions over hemisphere), `build_loaf.py` (memmap loaf: caps u8 + gt u16). Loaves:
  `renders/val_loaf` (clutter ~10k), `renders/planar_loaf` (planar ~10k).
- **Train/eval**: `train_proj_net.py` (loaf/ConcatLoaf, AMP, snapshots, gate, TB),
  `predict_proj_net.py` (coverage/bin-precision/|du| + conf masking),
  `eval_hemisphere.py` + `gen_hemisphere_eval.py` (obliquity bench, 4-panel plots).
- **Capture/calib (design/scaffold)**: `capture_app.py` (GLFW), `docs/calibration_design.md`,
  `docs/proj_net_design.html`. Preview: `preview_augment.py`.

## Key results (conv U-Net, clutter-only)
- Regression head → **plateau ~360 px** (abandoned).
- cls+offset → breakthrough; fine-tuned to **0.59 px raw median**, ~**0.45 px / 99%
  bin-precision at 79% coverage** confidence-masked (`conf>0.9`), p95 collapses
  1100→<3 px. Effectively matches Gray-code's 0.25 px regime, single-shot.
- **Hemisphere bench: cliff at ~45°** (bin acc 96/95/82/46/4% over 0–15/…/60–75°).

## Key learnings (the load-bearing ones)
1. **Conditioning ≫ capacity**: classification+offset beats coordinate regression
   massively — the head framing was the single biggest win.
2. **Know the chance floor** (ln60+ln36≈7.7): codebook learning sits at chance ~4
   epochs then phase-transitions. Don't judge a plateau before ~2–3k steps.
   Overfit-one-batch is the 2-min bug-vs-hard-problem test.
3. **Offset was the subpixel lever**: corr(pred,gt frac) 0.17→0.92 after raising
   offset weight 2→6 once bins saturated; median 5.7→1.0 px.
4. **LR**: cosine→0 is wrong while a loss term hasn't learned; use `--lr-min` floor.
5. **Aggregate val median is a poor proxy for obliquity** (easy-pixel dominated) —
   use the hemisphere bench + per-epoch `--snapshots`, pick best-on-hemisphere.
6. **Obliquity cliff = support hole, not density** — more epochs/samples of the same
   (near-frontal) distribution won't move it; needs hemisphere-pose training data.
   60–75° likely an information limit (anamorphic cell compression).
7. **Don't change two things at once**: added planar data + blur/noise aug together →
   confounded the cliff experiment (aug also tested only on clean eval, which can show
   its cost but not its benefit). Lesson logged; from-scratch plan isolates it.
8. **MPS is GPU-bound** (~50 img/s); loaf memmap + AMP fixed the data path. Inference
   ~480 ms/frame full-res after on-device decode.

## Current state / open threads
- **Warm-start mixed run (clutter+planar, aug) running** — conv, 20 epochs, ~epoch 5,
  cliff not moving yet (30–45° improved 82→88%, 45–75° flat). Snapshots saving.
  Decision at ep08 (see `docs/plan_train_from_scratch.md`).
- **Train-from-scratch** = next experiment if ep08 cliff still stuck.
- **Transformer A/B** (`--mid attn`) — queued for the *other machine* (barnacle); tests
  whether global attention helps the oblique/ambiguous regions a conv RF can't.
- **Real-world eval** — pending capture app + `eval_capture.py` (see
  `docs/plan_capture_app.md`).
- **Self-calibration from correspondence** — designed, not built (`calibrate_from_correspondence.py`).
- **Heteroscedastic offset uncertainty** — *code in, run deferred* for a clean conv+aug
  A/B. `--hetero`/`--nll-weight` add a log-variance offset head (β-NLL) + a fused
  calibrated σ map (`predict_full(return_sigma=True)`). Defaults off ⇒ existing runs
  unaffected. Design + launch recipe in `docs/heteroscedastic_uncertainty.md`.
- **Cliff strategy (consensus)** — `docs/cliff_plan.md`: the cliff is coarse-classification
  under anamorphic compression, not subpixel; honest target ~70° + calibrated abstention.
  Diagnose (overfit-oblique-batch, effective-RF-of-bottleneck, conf-vs-obliquity AURC)
  before fixing; obliquity-weighted CE (per-axis `gt_proj` Jacobian) + from-scratch +
  learned bin-correctness head; conv-RF-growth (dilation) over attn if RF-bound (net2
  relabeled **untested**, not failed); pattern co-design vs the homography proxy as the
  swing. NB: CUDA now banked (~107 img/s), so this is affordable.

## Checkpoints
- `proj_net.pt` — best clutter model (epoch 17, 0.59 px). The baseline / warm-start source.
- `proj_net_regress.pt` — abandoned regression baseline.
- `proj_net_mixed*.pt` — current warm-start mixed run (best + `_epNN` snapshots + `_last`).
- (planned) `proj_net_scratch*.pt`, `proj_net_attn.pt`.

## Publishing note
Method is a competent remix of known work (HyperDepth classification decoding,
"Connecting the Dots" learned dot decoding, RAFT/Polka-Lines lineage) — not novel as a
method. Real contributions if pursued: the exact-GT renderer + obliquity benchmark +
bin-vs-offset failure analysis, and (most) a real sim-to-real demonstration. For the
projection-mapping business goal, publishability is orthogonal — a working system is
the deliverable.
