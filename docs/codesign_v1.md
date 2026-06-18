# Spec: Appearance-faithful co-design pass + 40k production train

## Objective
Produce the best-available coprime-carrier pattern by re-running co-design against an **appearance-faithful** proxy, validate it on real renders, then train the production continuous-phase decoder on a **40k** render set shaped for the hard regimes (grazing + edges). This replaces the current pattern, whose amplitudes/phases were co-designed against an *analytic* appearance that diverges from what the renderer actually produces.

Working dir: `/run/media/nshelton/LUX/lux`.

## Orientation — do this first
Read before touching anything:
- `docs/hierarchical_pattern_plan.md` — design doc (pattern + decoder + the grazing-cliff problem).
- `lux/codesign.py`, `scripts/train_codesign.py` — the co-design / differentiable proxy (where pattern params are learned).
- `lux/codesign_vote.py`, `lux/codesign_infer.py` (`load_quad`, `predict_quad`) — continuous-phase quadrature head + number-theoretic consensus vote decoder.
- `scripts/codesign_demod_probe.py`, `scripts/gen_hemisphere_eval.py`, `scripts/train_quad_rendered.py`, `scripts/eval_hemisphere_quad`, `scripts/eval_clutter_quad.py` — probe, render-gen, decoder train, two eval harnesses.

## Established — treat as settled, do not re-litigate
- **Pattern:** `sigmoid(bias + Σ aₖ·sin(2π fₖ x + φₖ))`, u-carriers on x, v-carriers on y. Decoded per-carrier as unnormalized `(cos,sin)` → phase `ψ=atan2`, magnitude `m=hypot` (= vote weight) → coprime consensus vote → global `(u,v)` + `peak_margin` confidence.
- **Periods are FROZEN:** u `{13,19,33,139}`, v `{11,17,29,113}`. Chosen for coprimality, spectral separability (no 2f/3f collisions), mid-band (out of DC/albedo), grazing-survivability, fault tolerance. **Learn only amps/phases/bias.**
- **Decoder:** coarse-first carrier weighting; **no coord-L1** (ablation showed no du gain — do not re-add); `peak_margin` is the abstention signal.
- **The bug being fixed:** the current pattern's amps/phases were co-designed against the *analytic* carrier appearance, NOT the quantized 8-bit PNG that is actually projected + bilinear-resampled + white-ref'd. That gap made proxy-trained zero-shot collapse on render (58% frontal bin-acc). The pattern is optimized for a world that doesn't exist.
- **Current result to beat** (render-trained current pattern, 60–75° grazing): bin-acc **33.6%** vs M-array 7.9%; bounded subpixel `med|du|✓bin` ~0.9–1.5px flat across obliquity vs M-array 0.22→6.9px. Easy regime regressed (91.6 vs 99.4 frontal). Grazing win is real but render-capped (the proxy claimed 94%).

## Pipeline — execute in order, each step gated

**1. Appearance fix to the proxy.**
Change proxy capture generation to sample the quantized 8-bit pattern PNG via `grid_sample` (model the projection warp + bilinear resampling + white-ref normalization) instead of evaluating carriers analytically. Keep the existing anisotropic grazing blur active in the same proxy.
- **Gate:** re-run `codesign_demod_probe.py` on the quantized-appearance proxy; confirm per-carrier σ_φ stays under bar at every obliquity band. 8-bit quantization adds harmonics — re-check the intermod budget the periods were tuned against. If any period fails under quantization, **STOP and report.**

**2. Re-co-design the pattern.**
Re-run co-design on the appearance-fixed proxy. Periods frozen; learn amps/phases/bias only; coarse-first; no coord-L1. Output = new candidate pattern PNG(s). Expect the energy split to shift (the faithful proxy is harder — likely more energy to coarse carriers for grazing-survival). That shift is the point, not a bug.

**3. Render a small validation set (shared by steps 3–4).**
Render ~400 planes spanning obliquity with the NEW pattern (`gen_hemisphere_eval.py`), held out from the 160-pose eval set (distinct seeds).
- **Appearance-fix canary (cheap go/no-go):** eval the Step-2 *proxy-trained* decoder on these renders **zero-shot** (no render-training). Frontal (0–15°) bin-acc must jump well off the old 58%. If it doesn't recover, the appearance model is still wrong — **STOP and diagnose before spending more render.**

**4. Render bake-off (pick the pattern).**
On the same renders, render-TRAIN a fresh quad decoder (mirror `train_quad_rendered.py`'s schedule) and eval on the held-out 160 poses. If you produced an energy-split variant, do the same and compare. Pick the winner by grazing (60–75°) bin-acc with bounded subpixel.
- **Gate:** the winner's render-trained grazing bin-acc should meet or beat the current **33.6%**. If it doesn't, the appearance fix didn't help the pattern — **report before committing the 40k.**

**5. 40k production render (winner only).**
Render 40k scenes with the winning pattern. Distribution = the original 20k family (clutter + planar), **shaped for the model:** oversample grazing obliquity (60–75° is a narrow band — uniform under-represents exactly the regime we're winning) and oversample edge-proximal pixels (include clutter; the depth-edge case needs edge density).
- **Discipline:** train set **disjoint** from BOTH eval sets (160 hemisphere poses + held-out clutter eval). Verify no pose/scene leakage before kicking off (~22 h render).

**6. Train the production decoder on 40k.**
Mirror the established quad training (coarse-first, no coord-L1, same backbone as the bake-off decoder). **Add edge-weighted sampling** — edges are a sparse fraction even of cluttered scenes; without upweighting the decoder just learns flat surface and treats edges as noise. Calibrate `peak_margin` via a reliability-diagram mapping **per obliquity band** (not a global threshold — calibration drifts with obliquity); hold out a calibration split.

**7. Eval — two separate tables.**
- **Table 1 — plane obliquity sweep** (`eval_hemisphere_quad`, 160 poses): bin-acc + `med|du|✓bin` per band, vs the M-array baseline.
- **Table 2 — clutter edge risk-coverage** (`eval_clutter_quad`): stratify by distance-to-discontinuity, split DEPTH/occlusion vs SHADOW boundaries; report bin-acc, `med|du|`, mean `peak_margin`, `coverage@τ`/`acc@τ`. Watch for depth-edge false-consensus (margin stays high but wrong → `acc@τ` collapses even at high τ) and coverage cliff (`cov@τ`→0).

## Report back
The demod re-validation (step 1 gate), the zero-shot recovery number (step 3 canary), the bake-off winner + grazing bin-acc (step 4 gate), and the two final tables (step 7). **Flag any gate failure immediately — do not proceed past a failed gate.**

## Open risks — do not paper over
- The proxy is still **planar** (no depth-varying defocus, no full renderer sampling). There is a render-cap the appearance fix cannot close; the result is the best-available *planar* pattern, not render-optimal. Don't claim render-optimality from a proxy number.
- The depth-edge false-consensus may be partly **intrinsic to the vote** (blended two-surface phases → spurious peak). Clutter training may help but not fully close it; report the clutter-trained-vs-plane-trained delta honestly.
- The strictly-matched A/B vs M-array is anchored separately (the matched-budget M-array run). The 40k is for the best *model*, so Table 1 is a best-model comparison, not a strictly matched A/B — state it as such.