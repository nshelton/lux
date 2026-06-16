# Cliff plan — moving 45→70° single-shot (consensus, 2026-06-16)

Consensus from an engineer/consultant debate over `session_summary.md` +
`net2_plan.md`. Supersedes the scattered cliff notes. **Nothing here runs until the
from-scratch `proj_net_conv_aug.pt` run finishes** (clean baseline first).

## Framing (the load-bearing reframes)

- **The cliff is coarse-classification, not subpixel.** Bench bin acc collapses
  96/95/82/46/4 over 0–75°; the 0.59 px is the *frontal offset* regime. Once the bin
  is wrong, offset is irrelevant. So spend on **context + pattern**, not offset weight.
- **Retire "perfect."** At 75° a 32 px u-cell lands on ~8 camera px (cos 75°≈0.26),
  often below the unique-decoding window — Gray-code dies there too. Honest target:
  **accurate to ~65–70° + calibrated abstention beyond**, measured as selective risk.
- **Speed was the bottleneck on the science, and it's banked.** CUDA migration done
  (~107 img/s on the 2080 Ti vs ~50 MPS). Remaining speed = `torch.compile`,
  channels_last, bf16; "go big + widen base 32→48/64" is a **3090** move (the 2080 Ti
  is VRAM/activation-bound at 256² crops — batch 48/64 OOM, not launch-bound).
- **net2 (`--mid attn`) is UNTESTED, not failed.** Its val collapsed on a 256→8160
  token-count train/test mismatch — that's a broken eval, not evidence about
  obliquity. Do not cite it as "attn doesn't help."

## Diagnose before fixing (cheap, hours on CUDA — do first)

The whole strategy forks on **information-limit vs support-hole**. Three tests settle it:

1. **Overfit one 60–75° batch.** Decisive. Memorizes oblique correspondence ⇒ *not*
   info/RF-limited ⇒ support hole (data/weighting/from-scratch fix it). Can't ⇒
   physical floor ⇒ stop pushing accuracy, ship abstention.
2. **Effective RF of the bottleneck path** — measure **empirically** (gradient of one
   bottleneck-output unit w.r.t. input; spatial extent), NOT analytic. Effective RF ≈
   √(theoretical), Gaussian-tailed (Luo et al. 2016); the analytic ~150 px number would
   wrongly declare "RF is huge." Measure the **deep path** specifically — skips feed the
   offset, the coarse bin rides the bottleneck. Then count: at 60° under compression,
   how many distinct u-cells fall inside that RF? Fewer than ~60 ⇒ classification must
   fail, no data fixes it.
   - **RF demand is pattern-dependent** (couples to step 7): higher-local-entropy
     patterns need fewer cells of context to disambiguate 60 bins, so #2's answer moves
     with the pattern. RF-growth and pattern-richness are **partial substitutes**.
3. **Reliability diagram of confidence vs obliquity** (+ obliquity-stratified AURC). Does
   conf already drop in the bad band? If so the system is closer to "working" than the
   cliff suggests — it knows what it doesn't know, which for projection mapping is most
   of the battle.

### Decision rules from the diagnostics

- **#1 says memorizable (support hole)** → obliquity-weighted loss (step 4) +
  from-scratch (step 5) + oblique training distribution (`net2_plan` follow-up #3).
- **#2 says RF-bound** → **grow the conv RF, not attention.** A conv RF is the same
  camera-px extent at any inference resolution, so it has *none* of net2's
  crop→full-frame token-scaling fragility. Prefer **dilation / larger encoder kernels
  over a 5th pooling level** (a 5th downsample on a 256 crop drives the bottleneck to
  8×8 — too coarse to localize the bin). Watch for dilation gridding.
- **#2 says RF-bound AND the pattern is mutable** → **try the pattern first** (step 7):
  more info/cell disambiguates more bins at the *current* RF, at **zero inference cost**,
  whereas RF-growth adds FLOPs/latency on every frame — which matters for the real-time
  projection-mapping deliverable. Grow conv RF only if the pattern is fixed.

## Fixes, priority order (cheapest/highest-confidence first)

4. **Obliquity-weighted CE** — cheapest cliff lever; run before anything exotic. The
   loaf has no normals, but local obliquity is in the **Jacobian of `gt_proj`**
   (anamorphic compression = projector-coords-per-camera-pixel). Refinements:
   - **Per-axis, not scalar.** Weight CE_u and CE_v by the respective singular values of
     the 2×2 Jacobian — u and v foreshorten differently under tilt, and a scalar washes
     that out. Prediction: per-axis weighting may *close the documented v-row deficit*
     (the global `--v-weight` is its crude scalar ancestor); if it does, the deficit was
     directional foreshortening, not an architectural row bias.
   - **Clip + edge-exclude.** Finite-diff Jacobian explodes at `gt_proj` discontinuities
     (the u16 `0xFFFF` validity jumps); clip it and **erode the valid mask** to drop
     pixels adjacent to occlusion edges before weighting, or you up-weight garbage.
5. **From-scratch as the main line** (already adopted for the live run). Warm-starting
   from the frontal `proj_net.pt` seeds a frontal appearance→bin map for which oblique
   appearance is OOD; 5 epochs won't relearn it (the flat mixed-run cliff was *that*,
   not a verdict on the data). From scratch, the codebook phase-transition happens with
   oblique examples in-distribution. Combine with step 4.
6. **Neighbor-bin label smoothing / hierarchical coarse→fine bins.** Under anamorphic
   blur adjacent-cell confusion is "less wrong"; smoothing keeps the classifier from
   collapsing to chance and gives a graceful gradient in the compressed regime.
7. **Pattern co-design** — highest ceiling, the genuine contribution. Optimize the
   **parameters of a structured generator** (multi-scale carrier freqs/phases / small
   basis), NOT raw pixels, so windowed-uniqueness is preserved by construction. Train it
   against the **on-the-fly homography-warp proxy**, not Mitsuba — anamorphic compression
   at a planar patch *is* a local homography, which is cheap and differentiable; reserve
   path-tracing for final sim-to-real validation. One differentiable loop: sample
   homographies in the 45–75° band → warp the generated pattern → decode → backprop into
   generator params + decoder.
   - **Guardrails (so it stays honest):** the inner loop must carry the full
     `_augment_crop` photometric stack (depth-defocus, grazing falloff, saturation,
     noise), or you optimize a pattern for a cleaner-than-real world. And a homography is
     a *single plane* — this loop trains the **cliff (oblique planar)** only; **depth
     discontinuities / edges** still need composited-plane or rendered data. Scope it as
     the cliff's pattern fix, not the edge fix.
   - Hand-designed multi-scale/anisotropic pattern first (cheap, high-confidence);
     learned-pattern as the research bet on top.

If 4–7 plateau: **RAFT-style iterative warped refinement** (first-pass depth →
locally un-anamorphose the camera toward frontal → re-decode). More work; last resort.

## Abstention line (merges the heteroscedastic head + selective prediction)

The deliverable is a **selective-risk** statement, so build and measure it as one. See
`docs/heteroscedastic_uncertainty.md` for the σ_offset half; the partner is a **learned
bin-correctness head** (predicts "will this argmax bin be right?", per-axis), which:
- replaces raw softmax-max in the `predict_full` fused-σ (softmax is overconfident
  exactly in the oblique band);
- **shares the obliquity weighting** (bin-flips concentrate where obliquity is high, so
  the weighting importance-samples the rare "incorrect" class) and **rides the bin-acc
  gate** (meaningless before the codebook phase-transition);
- is scored by **obliquity-stratified risk-coverage / AURC**, not accuracy — add that
  panel to the hemisphere bench; it's the number that proves "knows when to shut up."

## Methodology guardrails (we already got bitten — `session_summary` #7)

One variable per run; judge only **after** the phase transition; **hemisphere bench is
the primary metric** (never val median); best-on-hemisphere snapshot selection. Parallelize
the matrix — from-scratch sweeps on CUDA, independent A/Bs on barnacle; don't serialize
what fans out.

## The sequence

1. CUDA banked; add compile/channels_last/bf16; defer go-big+widen to the 3090.
2. Let the from-scratch conv+aug run finish (clean baseline, ~12 h) — don't perturb it.
3. Diagnostics 1–3 above; let them arbitrate info-limit vs support-hole.
4. In parallel: obliquity-weighted CE (per-axis Jacobian, clipped) + neighbor-bin smoothing.
5. Abstention line: learned bin-correctness head + rewired fused-σ, scored by AURC.
6. Attn only if #2 says RF-bound *and* a conv-RF-growth control loses — default RF fix is
   dilation (pattern first if mutable). Otherwise the support-hole fix is oblique data.
7. Pattern co-design (hand-designed multi-scale → learned generator vs the homography proxy)
   as the swing for the ceiling and the real contribution.
