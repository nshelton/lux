# Is the Residual Obliquity Cliff an Information Limit or Fixable? — Overfit-One-Batch Verdict

**Date:** 2026-06-16 · **Status:** for review · **Depends on:** `docs/tiling_brief.md` (the residual
cliff this tests is what survives tiled inference) · **Code:** `scripts/overfit_oblique.py`.

> Run on the Mac (`.venv`, torch 2.8, MPS). Data: held-out hemisphere samples
> `evals/hemisphere/data/sample_*` (flat planes at known obliquity, **exact synthetic GT** — no
> label noise to confound the ceiling). Model: `proj_net_scratch.pt` (conv) for conv-init.

---

## TL;DR

The residual obliquity cliff that survives tiled inference (~75% @45–60°, ~10% @60–75°) is, **up to
the deepest angle the data contains (74.9°), an INFORMATION-SUFFICIENT generalisation gap — not an
information limit.** A fixed batch of 256-px crops from a **73° plane (where normal eval scores 4%)**
overfits to **100% bin-accuracy**, robust across an LR sweep, a second batch, and both fresh- and
conv-init. So **the residual cliff ≤75° is fixable with oblique training data; pattern redesign is not
needed for this regime.** Mechanism agrees, and locates the *genuine* wall past ~77°, outside the
current data — which has to be rendered to be found.

## The question and why "overfit one batch" answers it

After tiling removed the evaluation artifact (`tiling_brief.md`), a real residual cliff remains. Two
hypotheses with opposite fixes:
- **Information limit:** at high obliquity the foreshortened M-array window aliases — different
  projector positions produce indistinguishable camera windows. No amount of data fixes it; the lever
  is a **new (multi-scale) pattern.**
- **Generalisation gap:** the window is still distinguishable, but the net never trained on that
  oblique distribution. The lever is **oblique training data** (much cheaper).

The discriminator: **can the model *memorise* a fixed batch of oblique crops?** The best achievable
train bin-accuracy on a fixed batch measures input distinguishability — two pixels with the same
appearance but different GT cannot both be fit by *any* model, so a low ceiling = aliasing = info
limit; a high ceiling = the information is present to exploit.

## Method (and why each choice — these are the review-mandated rigour points)

- **Run at 256-px training scale**, not full-frame — so the result isn't contaminated by the
  resolution artifact (`tiling_brief.md`).
- **conv-init primary.** Capacity is obviously present (a trained model), and it converges in hundreds
  of steps, so a low plateau then *isolates* information rather than confounding it with capacity.
- **LR sweep (5e-4 / 1e-3 / 3e-3)** and take the best ceiling → a plateau can't be blamed on a bad LR
  (rules out the optimisation confound).
- **≥2 batches** (crop seeds) at the deepest angle → rules out a one-batch fluke.
- **fresh-init cross-check** at 74° → rules out "conv-init cheats via pretrained features."
- **Frontal control** must hit ~100% → proves the head/capacity can represent the task.
- **Exact synthetic GT** → no label noise floor.
- No augmentation; memorise the raw crops (8 crops/batch, 800 steps).

## Results

Best overfit ceiling (u-bin / v-bin), per obliquity:

| obliquity (sample tilt) | normal tiled eval | best ceiling | window @ cam | verdict |
|---|---|---|---|---|
| frontal ~7° (10°) | 99% | **100 / 100%** | 19.7px | info-sufficient (control ✓) |
| oblique ~60° (62°) | 56% | **100 / 100%** | 9.3px | info-sufficient |
| oblique ~74° (73°) | **4%** | **100 / 100%** | 5.9px | info-sufficient |

Per-run at 73° (the decisive angle):

| init | lr | seed | u / v ceiling | plateaued |
|---|---|---|---|---|
| conv | 5e-4 | 0 | 100 / 100% | yes |
| conv | 1e-3 | 0 | 100 / 100% | yes |
| conv | 3e-3 | 0 | 100 / 100% | yes |
| conv | 1e-3 | 1 *(2nd batch)* | 100 / 100% | yes |
| **fresh** | 1e-3 | 0 | **95 / 93%** | **no — still climbing** (62→87→95) |

The full LR sweep + 2nd batch hit a perfect 100% under conv-init; fresh-init (no pretrained features)
reached 95/93% in 800 steps and was still rising — i.e. on its way to ~100%, just slower from scratch.
Both inits, every LR, both batches → the same conclusion.

## Mechanism corroboration

The decodable M-array window is ~20 projector px (5×5 cells × 4px). Foreshortened at tilt θ it spans
~`20·cos(θ)` camera px; below the camera resolving floor (~4.5px) different projector positions alias.

- 73°: `20·cos(73°) ≈ 5.9px` > 4.5px floor → **resolvable** → 100% ceiling. ✓ (matches the empirics)
- The window crosses the floor at `cos(θ) ≈ 0.225`, i.e. **θ ≈ 77°.** So the genuine information wall
  is **past ~77°** — beyond the deepest sample in the data (74.9°). Empirics and the resolving-limit
  model land on the same place.

## Verdict, bounds, and guardrails

- **Verdict:** the residual cliff ≤74.9° is information-sufficient → **a generalisation gap, fixable
  with oblique training data. Not a pattern-redesign problem in this regime.**
- **Guardrail — ceiling ≠ achievable.** 100% is the *overfit ceiling* (the information exists to
  exploit); a net trained on oblique data will generalise to held-out 74° **below** 100% by an unknown
  margin (memorising one batch is strictly easier than generalising). **Do not let the oblique-data
  plan inherit 100% — or any overfit number — as its target.** Set the goal from a post-training
  held-out 74° eval.
- **Bound — the wall exists, just past the data.** Past ~77° the window drops below the resolving floor
  and pattern redesign (multi-scale: coarse carries the bin under compression, fine carries precision
  frontally) becomes the only lever. Whether that matters depends on whether the use case needs >77°.

## Open / next

- **Render 76/78/80° patches** (ray-caster, exact GT) and rerun this sweep — the data caps at 74.9°
  (zero samples past 75°), so the falloff can only be *located* by rendering past it. This brackets the
  real wall instead of inferring it from one borderline angle.
- This says **oblique-data + sim2real**, not pattern redesign. Sequence it behind the real-side
  decompose (test1/test3: sim2real vs past-real-limit vs context) and renderer-calibration — see
  `tiling_brief.md` §6/§9.

## Reproduction

```bash
python scripts/overfit_oblique.py          # env: OVERFIT_STEPS (default 800), OVERFIT_K (default 8)
```
Picks a frontal / ~60° / ~74° sample from `evals/hemisphere/results_conv_tiled/per_sample.csv` (by
`max(theta_cam,theta_proj)`), loads K fixed 256-px crops (no aug), and overfits across the run matrix
above; prints per-run ceilings + a summary with the resolving-limit calc.
