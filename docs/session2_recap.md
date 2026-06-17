# Session 2 Recap — Tiled Inference, the New Aug, and the Honest Cliff

Follows `session_summary.md` (session 1: the cls+offset breakthrough, the 45° cliff, the
row deficit). Session 2 reframed two of those "limits" as **evaluation artifacts**,
trained the best model yet, and built the viz/infra to see it. Dates: 2026-06-16/17.

## The arc (what happened, in order)

1. **Burn-in / clean baseline on the 2080 Ti.** From-scratch `--mid conv` + the old aug,
   both NVMe loaves, batch 32. Doubled as a stability soak — **the box is solid** (the
   earlier long-run hangs were a bad RAM stick, since removed; ~107 img/s, ~24 min/epoch).
2. **Two-machine merge saga.** Untangled a stuck interactive rebase, then reconciled a
   genuine divergence: the Mac built **tiled inference** on `e0d8d95`; this box built the
   **heteroscedastic head** + conv-aug. Merged to one canonical `main` and pushed. A
   second merge brought the Mac's **physically-ordered aug + GGX renderer**; deduped the
   independently-built tiled-eval to the superset (center-crop + `conf_fn`).
3. **Tiling discovery (the big reframe).** The 45° cliff and the v-row deficit were
   **substantially eval artifacts**: train on 256-px crops, infer full-frame (1080×1920)
   = out-of-distribution. Tiled inference at the training scale recovers it. Wired into
   the in-loop `evaluate()` (honest val curves — `median_dv` no longer lies) and the
   benches. See `net2_plan.md` resolution finding.
4. **Heteroscedastic uncertainty** (`--hetero`/`--nll-weight`): a β-NLL offset
   log-variance head + fused-σ in `predict_full` + swappable `conf_fn`. Code in, run
   deferred. See `heteroscedastic_uncertainty.md`.
5. **Cliff consensus** (`cliff_plan.md`, multi-round review): the cliff is
   coarse-classification, not subpixel; honest target ~70° + calibrated abstention;
   diagnose (overfit-batch, effective-RF, AURC) before fixing.
6. **New physically-ordered aug, dropped in at ep11** via warm-restart (`--resume ep10`,
   20-epoch schedule). The aug change is shot-noise-correct (σ∝√signal) + right ordering.
   **Result: the best model yet.**
7. **Viz/infra**: consolidated `summary.png`, clamped error map, an interactive per-pixel
   viewer (scanline scrubber), and a results dashboard.

## Key results — `proj_net_conv_newaug` (final, ep20) is best on every axis

- **Validation:** |du| **0.23 px**, bin **93.6%** (beats documented `proj_net_scratch`
  0.28 px / 87%).
- **Real captures** (tiled center-crop) vs old-aug ep07 — improved on all 5, most on the
  hard ones; confidence-masking lifts all to ~97–98%:

  | scene | old-aug ep07 | newaug final | conf>0.9 |
  |---|---|---|---|
  | test0 | 93.5% | 94.8% | 96.7% |
  | test1 (grazing) | 39.1% | **47.5%** | 98.2% (13% cov) |
  | test2 | 87.9% | 90.9% | 97.6% |
  | test3 (oblique) | 67.9% | 72.1% | 98.1% |
  | test4 | ~90% uv | 94.6% | 97.7% |

- **Hemisphere bench** (tiled center-crop, 160 samples):

  | obliquity | newaug | scratch (doc) |
  |---|---|---|
  | 0–15° | 99.4% | 96.6% |
  | 30–45° | 98.0% | 87.0% |
  | **45–60°** | **78.1%** | ~74.6% |
  | 60–75° | 7.9% | ~8% |

  ≤45° near-perfect (sub-0.4 px); the cliff at 45–60° lifted past scratch; **60–75° stays
  ~8% — the genuine anamorphic information-limit** the overfit diagnostic predicted
  (pattern territory, not trainable). `med|du|✓bin` stays tight (1.1 px @ 45–60°), so the
  residual cliff is pure bin-misclassification.

## Key learnings (session 2)

1. **Two of the headline "limits" were eval artifacts.** Crop-train / full-frame-infer is
   OOD for any resolution-sensitive read (attention catastrophically; conv at the v
   extremes). Always eval at the training scale (tiled). `median_dv` lying for a whole run
   was this.
2. **Center-crop > max-confidence for the tiled metric.** Softmax is overconfident at tile
   edges — exactly where max-conf selection fires. Take each pixel from the tile it's most
   *central* in (geometry), reflect-pad the frame so true edges are central too. Confidence
   is for abstention, not stitching.
3. **The new aug helped sim2real**, biggest on the grazing/oblique scenes — physically-
   ordered, signal-dependent shot noise is the lever (not flat additive). Dropping it in
   at ep11 via warm-restart was fine (codebook is aug-invariant; same degradation family).
4. **net2 / attention reclassified: tested-negative, not untested.** Tiled-attn ≈
   full-frame-conv < tiled-conv. The RF fix, if ever needed, is conv-RF growth (dilation,
   resolution-invariant), not attention.
5. **The 60–75° floor is a real information limit.** ~20 px M-array window × cos(75°)≈0.26
   → ~5 px camera, below the resolving limit. The fix is pattern (coarse scale that
   survives compression), not more training/capacity.
6. **Aggregate val ≠ obliquity.** Pick checkpoints on the hemisphere bench per bin; here
   the best-val final (ep20) is also the hemisphere winner, but that isn't guaranteed.

## Infrastructure built

- **`eval_capture`**: consolidated `summary.png` (du/dv hist linear+log overlaid,
  confidence dist, valid-px-retained-vs-confidence); spatial maps behind `--maps`
  (incl. clamped ±2 px signed-error map); tiled center-crop default; `._*` exFAT
  sidecar fix; `.npy` dumps for re-analysis.
- **`scripts/make_capture_viewer.py`** — interactive per-pixel viewer (`captures/view.html`):
  checkpoint + scene + layer dropdowns, live scanline scrubber (du across row / dv down
  column, residual or absolute), per-pixel readout, real matplotlib colormap LUTs.
- **`scripts/make_results_dashboard.py`** — `results.html` at repo root: ckpt×scene
  capture table + ckpt×obliquity hemisphere table + figure gallery + viewer link. Serve
  from repo root: `python -m http.server` → `:8000/results.html`.
- matplotlib installed in `~/.venvs/lux`.

## Checkpoints

- **`proj_net_conv_newaug.pt`** — the session-2 best (ep20, 0.23 px / 93.6%). Production
  baseline. Snapshots `_ep11.._ep20`.
- `proj_net_conv_aug_ep01..ep10.pt` — the old-aug from-scratch baseline (warm-restart
  source at ep10). Preserved for the aug comparison.
- (Mac line) `proj_net_scratch`, `proj_net_attn`/`net2_ep9` — prior best / attn A/B.

## Open threads / next (priority order)

1. **Pattern co-design** for the 60–75° floor — the homography-proxy differentiable
   pattern/decoder loop (geometric only, planar; carries `_augment_crop` photometrics).
   The genuine contribution; the only lever for the info-limit. See `cliff_plan.md` step 7.
2. **GGX/Fresnel re-render** for test1 grazing (still 47%) — grazing whiteout is a
   *render-distribution* gap, not an aug one; needs a re-rendered glossy loaf
   (`gen_training_data` now randomises ~50% glossy). Then re-train.
3. **Heteroscedastic + learned bin-correctness head** run — calibrated abstention for the
   60–75° floor (accurate where confident, silent where not). Code half-in.
4. **Real-data Gray-code fine-tune** (net1 Priority 1) — closes the residual sim→real px;
   decompose test1/test3 failure (sim2real vs real-obliquity-limit vs context) *before*
   committing (cliff_plan step 8).
5. Re-baseline older checkpoints tiled for an apples-to-apples dashboard.
