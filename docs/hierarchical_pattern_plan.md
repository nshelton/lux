# Hierarchical multi-scale phase pattern — co-design plan (for review, 2026-06-17)

Proposal for the next pattern-design leg (`docs/cliff_plan.md` step 7, the #1 open thread in
`docs/session2_recap.md`). Supersedes the "hand-designed multi-scale first" sketch with a
concrete, single proposal: a **hierarchical multi-scale staggered phase code** jointly designed
with a **continuous (bin-free) phase decoder**, trained against a differentiable homography-warp
proxy. This document is written for external review — assumptions and open questions are called
out explicitly.

## 1. Two problems, one root cause

The current one-shot decoder (`lux/proj_net.py:ProjUNet`) represents the projector coordinate as
**hard bin (60 u / 36 v) + within-bin offset**, classified by softmax. Two observed pathologies:

1. **Grazing cliff.** Bench bin-accuracy collapses 99 / 98 / 78 / **8 %** over 0-15 / 30-45 /
   45-60 / **60-75°** (`session2_recap.md:49-61`). Confirmed *information limit* of the fixed
   random-binary M-array: its ~20 px unique window × cos(75°)≈0.26 → ~5 px on camera, below the
   resolving limit. Not trainable — only a better pattern moves it.
2. **Quantization seams.** The confidence map shows a **grid of confident bin interiors and
   low-confidence borders**. At a bin boundary the assignment is genuinely ambiguous between bin
   `k` and `k+1`, so softmax-max dips; and a bin *flip* there is a full 32 px jump (catastrophic).

Both are artifacts of **hard discretization of a continuous quantity** + argmax confidence. The
fix is the same mechanism: replace hard bins with a continuous, multi-scale code where coarse
scales are robust and unambiguous and fine scales add precision.

## 2. The pattern: a hierarchical phase ladder

Encode the projector column `u ∈ [0, W)` (and row `v`, independently, on the orthogonal axis)
as a stack of sinusoidal phases at **geometrically-spaced frequencies**:

```
scale 0 (coarse): φ₀(u) = 2π u/λ₀   λ₀ ≈ W (1920 px) → 1 cycle, never wraps, globally unique
scale 1:          φ₁(u) = 2π u/λ₁   λ₁ = λ₀/r
   ...                               r = 2-3 (chosen: small ratio, see §6)
scale K (fine):   φ_K(u) = 2π u/λ_K  λ_K ≈ 8 px → sub-pixel precision, wraps ~240×
```

Every `φ_k` is a **smooth continuous function of u** — there are no partitions, hence **no
borders and no seams by construction**. Example ladder (r=3, W=1920): periods 1920/640/213/71/24/8
→ 6 u-scales; (r=3, H=1080): 1080/360/120/40/13/8 → ~6 v-scales.

**Single-shot multiplexing.** Classically these phases are separate projected frames; we are
single-shot, so all scales coexist in one grayscale image as a **superposition of carriers** —
exactly what `lux.codesign.PatternGenerator` already computes:

```
pattern(x,y) = sigmoid( bias + Σ_k a_k · sin(2π f_k·(x,y) + φ_k) )
```

The carrier **frequencies are organized into the per-axis ladder**; **amplitudes `a_k` and exact
frequencies stay learnable**, so co-design chooses the energy split between coarse (robust) and
fine (precise) and the precise ladder spacing. u-carriers lie along x, v-carriers along y.

## 3. The decoder: continuous phase, coarse-to-fine unwrap

Replace the `[60 u-logits | 36 v-logits | offset_u | offset_v | validity]` head with a
**per-scale quadrature head**: for each scale `k` and axis, output `(cos φ_k, sin φ_k)`.

- **Quadrature `(cos, sin)` regression** is the key seam-killer: the target is a smooth function of
  `u` (no sawtooth/2π discontinuity), so the loss is continuous and the prediction never has a
  blind spot at a wrap. The **vector magnitude `√(cos²+sin²)`** is a natural, calibrated per-scale
  confidence with *none* of the argmax-at-border pathology.
- Head size: `2 · K_u + 2 · K_v + 1` (validity) channels — far smaller than the 99-channel head.

**Decode = hierarchical unwrap** (coarse picks each finer carrier's period):

```
u₀ = φ₀ · λ₀/2π                                  # coarse, unique over frame, no wrap
for k = 1..K:
    n_k = round( (u_{k-1} − φ_k·λ_k/2π) / λ_k )   # which period of carrier k
    u_k = n_k·λ_k + φ_k·λ_k/2π                    # snap + refine
u = u_K                                           # continuous, precise, seam-free
```

Implemented as a fixed **differentiable** module (soft `round` for gradient). Robustness from the
coarsest scale, precision from the finest.

**Loss:** per-scale circular loss `Σ_k (1 − cos(φ̂_k − φ_k))` + a final-coordinate L1 after unwrap,
**curriculum'd coarse→fine** (the coarse phase must be trustworthy before the fine scales earn
gradient — same discipline as the existing offset/NLL curriculum gates in `proj_net`).

## 4. Why this fixes both problems (same lever)

- **Seams:** no bins ⇒ no grid. The coordinate is continuous everywhere; quadrature regression
  removes the wrap blind spot.
- **Grazing:** at 75° the fine carriers compress below the resolving limit and become noise — but
  the **coarse carrier** (period ≈ ¼-½ frame) is still many camera pixels after ×0.26 compression,
  so it still reads. Result: **graceful loss of precision** (lose fine scales, keep coarse
  correspondence) instead of the current 8 % bin collapse. This is exactly the
  accurate-where-confident / calibrated-abstention behaviour `cliff_plan.md` targets.

## 5. The co-design substrate (already built — `lux/codesign.py`)

A homography is the exact local model of a planar patch under anamorphic compression, so we train
against an on-the-fly differentiable proxy rather than Mitsuba:

- `PatternGenerator` — learnable carrier bank (→ the phase ladder); analytic, differentiable.
- `sample_homography_inv` — oblique-plane camera→projector homography; **verified**: anisotropy
  tracks `1/cos θ` (75°→3.98 vs 3.86), footprint grows with obliquity (256 px crop sees 251→847 px
  of projector along the tilt axis from 0→75°), validity ~1.0.
- `differentiable_augment` — torch port of `proj_net._augment_crop` (gain → PSF → shot → read →
  saturate → gamma → quantize, with straight-through estimators for round/clamp). **Verified** to
  match the numpy stack's statistics (mean 0.587 vs 0.585, std identical, sat 0.146 vs 0.143) and
  to pass gradient to every pattern param.
- `ProxyBatcher` / `EvalBank` — on-the-fly GPU batches in the `proj_loss` contract; per-band
  full-frame eval through the existing `predict_tiled`.
- `scripts/train_codesign.py` — joint trainer (two optimizer groups: decoder + generator;
  decoder warm-up before unfreezing the pattern; anti-collapse regularizer; materializes the
  learned pattern as a PNG set feedable to `gen_rasterizer_dataset.py` for real validation).

**Scope (honest):** a homography is a *single plane*, so this trains the **oblique-planar cliff
only**; depth discontinuities / occlusion edges still need rendered data.

## 6. Current empirical state (Milestone 1: proxy fidelity)

We first trained the decoder against the proxy with the **real M-array** (fixed, non-learnable) to
check whether the proxy reproduces the documented cliff before trusting any learned-pattern result.
Final (28-epoch, full 0-78° training distribution, density pinned to the rig's ~1:1 projector:camera
ratio):

| band | proxy u-bin | proxy v-bin | real bench (M-array) |
|---|---|---|---|
| 0-45° | 93 % | 96 % | ~99 % |
| 45-60° | 92 % | 95 % | 78 % |
| 60-75° | **83 %** | 87 % | **8 %** |

**Verdict: partially faithful.** The easy regime matches reality and a monotone cliff is present,
but the proxy currently **understates grazing difficulty** (83 % vs 8 %). The missing physics is
**obliquity-coupled degradation**: after `1/cos` compression the cells fall under the compounded
camera MTF + projector defocus + grazing whiteout + reduced SNR, which the current isotropic,
obliquity-independent augment does not capture.

**Planned fidelity fix (first build step):** an **anisotropic blur whose σ along the compressed
(tilt) axis scales with the local compression factor**, plus a grazing intensity falloff/whiteout.
This deepens the cliff toward the real shape — necessary so the hierarchical pattern's premise
(fine dies, coarse survives) is genuinely exercised, and so "it helps grazing" cannot be a false
positive off a too-easy proxy.

## 7. Design parameters (current choices)

- Unwrap: **hierarchical, small ratio `r = 2-3`** (chosen over coprime/CRT and learned-soft for
  simplicity and a safe, interpretable unwrap; `r` small enough that the coarser estimate lands
  within ±½ period of the next carrier).
- Scales: K ≈ 6 per axis (ladder spanning full-frame → ~8 px).
- Decoder: same `ProjUNet` backbone, new quadrature head; conv bottleneck (attention reclassified
  tested-negative, `session2_recap.md:76`).
- Bins removed entirely; `proj_loss` replaced by the circular + coordinate loss.

## 8. Open questions for reviewers

1. **Unwrap robustness vs ratio.** With `r=3` and 6 scales, is a single-pass greedy unwrap robust
   enough at grazing where coarse phase is noisy, or do we need redundant/coprime scales + voting
   (CRT) for the worst band? (We chose simple first; flag if this is naive.)
2. **Soft-round gradient.** Differentiable unwrap needs a soft `round`; does that bias the learned
   ladder, and should the unwrap instead be a small learned head (less interpretable)?
3. **Single-shot capacity.** How many independent frequencies can one 8-bit grayscale carry before
   the superposition's local SNR per band is too low for the CNN to demodulate — especially under
   the photometric stack? Is colour multiplexing (RGB channels carry different scales) worth it?
4. **Proxy fidelity ceiling.** Even with anisotropic compression-coupled blur, is the planar
   homography proxy trustworthy enough to *design* a pattern, or must the final ladder be selected
   on rendered data? (We plan real-render validation regardless — see §9.)
5. **Coupling to the bin grid downstream.** Existing tooling (tiled stitch, confidence fusion,
   abstention) assumes the bin head. What breaks when the representation becomes continuous phase?

## 9. Sequence / validation

1. **(done)** Build + verify the proxy core; Milestone 1 fidelity (above) — partial pass.
2. Deepen grazing fidelity (anisotropic compression-coupled blur + falloff); re-confirm the cliff
   approaches the real shape with the M-array baseline.
3. Build the hierarchical pattern ladder + quadrature decoder + circular loss + unwrap.
4. A/B on the proxy vs the M-array baseline: expect **seams gone** (continuous confidence) **and
   grazing lifted** (graceful coarse-survives degradation).
5. **Real-render validation (mandatory):** materialize the learned pattern → `gen_rasterizer_dataset.py`
   / `gen_training_data.py` → `build_loaf.py` → retrain/eval on rendered captures →
   `eval_hemisphere.py`. Closes sim→real and exposes any residual proxy-fidelity gap.

## 11. Review integration + carrier-gate results (2026-06-17)

External review adopted. **Representation switched** from the geometric ladder to a **coprime
balanced-core + consensus vote** (CRT). Three structural changes: (1) coprime-perturb off clean
ratios; (2) cap the coarsest carrier *below* DC (no near-DC carrier — avoids lighting collision)
and let coprime *beats* synthesize global range; (3) replace greedy chain unwrap with a
consensus vote (robust to one bad carrier at grazing). Locked implementer constraints: **freeze
periods, learn only amplitude/phase/bias** (coprime integers are a discrete constraint a grad
optimizer would break; STE-snap only if periods are ever learned); **don't L2-normalize
(cos,sin)** — the floating magnitude *is* the per-carrier "dead at grazing" vote weight; **vote
ships as hard-argmax (inference) + soft-argmax (training) with a unit-test gate** (synthetic
exact phases → u to <0.1px; corrupt one carrier → still correct); **blur must be optics-anchored,
not the toy isotropic augment** (a too-gentle blur overestimates separable carrier count K);
**budget downstream rewiring** — tiled stitch, `min(conf_u,conf_v)` fusion, and abstention assume
the bin head + softmax-max and must be re-pointed at peak-margin confidence.

**Gate run (`scripts/codesign_carrier_gate.py`), candidate set u={13,41,47} / v={11,31,37}:**
- **Intermod: clean** (no f_i±f_j or 2f_i collisions). **CRT range covers the frame** (41·47=1927≥1920).
- **"Fine dies, coarse survives" CONFIRMED** under optics-anchored degradation: the fine flank
  p=13 amplitude collapses 0.18→0.03 over 0→75° (dead at grazing) while the core p=41/47 hold
  ~0.20→0.16. This is the architecture's whole premise, validated. Floating magnitude tracks it
  exactly → it is the right vote weight.
- **One real constraint:** the balanced core {41,47} are **not linearly separable in a 256px
  window** (Δf 6.0 < 7.5-cycle demod bandwidth; need ~320px). v-core {31,37} is fine. → either
  train at ~320px crops, grow the RF, or — cleanest — **RGB-multiplex the close core** (41 on one
  channel, 47 on another → trivially separable, beat still computable). The review anticipated
  exactly this ("evaluate RGB before redesigning" when capacity/separability is tight).
- Caveat: the sim uses a *linear* demod with known phase = a lower bound; the CNN does nonlinear
  demod and may separate the core, but separability is a structural limit worth de-risking via RGB.
- SNR headroom is ample (3 mono carriers sit at amp ~0.2 vs a small noise floor) — capacity is
  limited by close-carrier *separability*, not raw SNR.

**Reconciliation of the M1 "grazing stays 80%" finding:** likely the reviewer's too-gentle-blur
artifact — the training proxy used ss=3 (under-integrates the 75° footprint) vs the gate's
sub-8 footprint integration. Bump the training proxy's area-integration to match before drawing
any decoder-limited-vs-info-limited conclusion.

## 12. Locked build sequence + Build-1 result (2026-06-17)

Sequence (reviewer-locked): **(1) vote + gate → (2) proxy fidelity [gates 3] → (3) fixed-pattern
demod probe [capacity] → (4) full quadrature generator + decoder.** Step 2 *gates* step 3 (you
cannot validate "grazing degrades gracefully" on a proxy that understates grazing); step 3's
fixed-pattern demod probe (frozen pattern, no generator, no vote) isolates "is the representation
demodulable to the accuracy the vote needs" before sinking the full build — and is the real moment
to spend the RGB card if capacity fails (not the separability question).

**Build 1 DONE** (`lux/codesign_vote.py` + `scripts/test_codesign_vote.py`, all gate tests pass):
- CRT consensus accumulator `acc(u)=Σ m_k cos(2πu/p_k − ψ_k)`; **hard** argmax (inference) +
  **windowed** soft-argmax (training) — the window prevents the naive-soft two-peak averaging trap
  (verified on an 8-peak lattice: lands on a peak, 123px from any midpoint). Confidence = peak margin.
- **Interface contract pinned** (what the Build-4 head must emit): per-axis, per-carrier
  *unnormalized* `(cos,sin)` → `ψ_k=atan2(sin,cos)`, `m_k=hypot` (floating magnitude = vote weight);
  vote returns `(u, peak_margin)`.
- **Acceptance bar for the demod probe:** vote holds to <0.5px median up to phase-noise
  **σ_φ ≈ 0.3 rad**, shatters by 0.5 rad. Build 3 must hit per-carrier phase RMSE well under ~0.3 rad
  across all obliquities.
- Working carrier set **u={13,19,33,139}** (separable + 2f/intermod-clean; the proposed {31,67,113}
  fixed separability but reintroduced 2f-harmonic collisions via near-octave ratios). Final set +
  RGB decision deferred to the Build-3 demod probe.

## 13. Build-2 result + blur architecture (2026-06-17)

**Blur architecture (reviewer-steered):** the training/inner-loop blur is an **analytic
Jacobian-scaled anisotropic Gaussian**, applied to sinusoidal carriers as an exact per-carrier
amplitude attenuation `a_k → a_k·exp(−2π² f_k^T Σ f_k)` (the Gaussian's FT — no convolution, no
supersample, differentiable). `Σ = (1/12+σ_mtf²)·JJ^T + σ_def²·I`: geometry (moment-matched box
footprint, `J` = exact planar camera→projector Jacobian) + optics (camera MTF, projector defocus)
**add in variance**. Footprint-supersample is demoted to the **offline fidelity oracle**.
(`lux.codesign.PatternGenerator.sample_at(coords, mtf=(σ_mtf,σ_def))`.)

**Validated** (`scripts/validate_blur_oracle.py`): geometric-only analytic vs footprint-ss oracle
match across 0–75° (finest carrier 0.93|0.93 at 75°; coarse 1.00|1.00) — no tail divergence at
these scales. **Finding:** the geometric footprint is a *weak* lever (footprint ≪ carrier periods);
grazing degradation is dominated by **optical σ + radiometric SNR** (grazing falloff → shot noise),
not the area-average. So σ_opt + the cos θ falloff are the physical levers; both anchored to
optics/radiometry, **not** fit to the M-array cliff (which under faithful planar optics holds
~80%, a consistency check confirming the real 8% is substantially non-planar/decoder-limited).

**Carry-forwards into Build 3 (reviewer):**
- **Demod-probe bar = σ_φ ≈ 0.2 rad** (margin below the 0.3 rad failure point; real demod errors are
  structured/correlated, not the iid the gate injected, and test D's cliff is sharp).
- **Maximize alias margin** in the final set, not just clear zero (test E's 0.41 margin is thin; the
  dense small carriers 13/19/33 nearly coincide, with the coarsest carrier load-bearing — and noise
  erodes the margin). À-la-carte objective: max adjacent-code distinctiveness under the noise model.
- **Judge intermod on the rendered spectrum at operating amplitude/bias, not frequency ratios:**
  sigmoid is near-odd → dominant distortion is **3f, not 2f**; 2f only grows with off-center bias /
  asymmetric saturation. So {31,67,113} may be clean at low amplitude — re-run the intermod check on
  the actual rendered signal once amplitudes are fixed in Build 3 (that's the RGB-decision moment).

## 14. Augment-fidelity verification + demod-probe status (2026-06-17)

**Augment port verified at grazing (not just frontal)** — `differentiable_augment` vs `_augment_crop`
match in mean *and* noise-std across signal levels (0.7→0.3→0.12): e.g. grazing 0.074/0.0439 vs
0.076/0.0450. Op-order confirmed: grazing falloff multiplies the signal **before** the augment, and
shot noise `√(x/full_well)` scales with that reduced `x`, so **SNR drops 3.0→1.7 frontal→grazing in
both pipelines**. So the augment is NOT a confound in the proxy's ~80% grazing.

**Consequence for the 8% check:** 8% is the *real rendered-bench* number (real scenes + `_augment_crop`
+ real decoder), not a proxy number — the proxy can't be pipeline-matched to it (the render substrate
differs). Matching the (already-faithful) augment does **not** move proxy-80→8; the residual is the
render/non-planar substrate, un-closable by blur. So 8% stays a **loose ballpark, never a tuning
target** — optics anchored to specs (defocus_px=1.0, MTF 0.7) + footprint oracle instead.

**Do NOT infer "decoder-limited" from 80→8 (corrected):** that comparison holds the decoder
representation FIXED (hard-bin both sides) and varies only the substrate, so it is no evidence about
the decoder — it points to **geometry/substrate-limited** (depth edges, occlusion, mixed-obliquity
windows absent from the planar proxy), and cannot override §1's information limit (20px→~5px is Nyquist;
no decoder recovers sub-resolution info — a better global decoder buys outlier rejection + subpixel
refinement on present-but-noisy info, lifting the 45–60° mid-cliff, not the deep 60–75° end, per TurboSL).
"Decoder ahead of pattern" is not a coherent ordering: continuous-phase decoding needs a phase-carrying
pattern to read — the lever is the **pair**, which is why they are co-designed. **Sequence unchanged.**
What 80→8 *does* establish: the proxy is faithful for the **planar-oblique lever** (the A/B measures the
M-array-vs-hierarchical *delta* on a common substrate, not the absolute), and the deep cliff is
non-planar → it lives in the §9.5 **render validation**, now more load-bearing. To actually settle
decoder-vs-pattern, hold substrate fixed and swap the decoder (hard-bin vs continuous-phase on identical
inputs) — a side experiment, AFTER the gate, not a reprioritization.

**Build 3 CLOSED — capacity gate PASSED (`scripts/codesign_demod_probe.py`).** Rigorous **local
per-pixel** lock-in (per-carrier window ~3× period, per-pixel phase NOISE = circular std across
noise realizations — the dense estimate a decoder must produce, realistic averaging, genuine
crosstalk; this version *can* fail). Working set {13,19,33,139} at K=4, σ_φ (rad):

| obliq | p=13 (fine) | p=19 | p=33 | p=139 (coarse) |
|---|---|---|---|---|
| 0° | 0.022 | 0.018 | 0.009 | 0.002 |
| 75° | 0.087 | 0.045 | 0.018 | 0.009 |

All **under the 0.2 bar** with margin (worst = fine p=13 at 75°, 0.087 ≈ 2.3× margin). Coarse survives
(0.009 at 75° — CRT backbone solid); fine degrades ~4× with obliquity but holds (first to die under
harsher SNR/more carriers). **No capacity ceiling for this set; RGB not needed (mono suffices)** — the
RGB-card trigger didn't fire. (Earlier global-lstsq probe was an optimistic upper bound; this replaces
it.) **Intermod re-checked on the rendered spectrum at operating amplitude:** 2f/3f < 0.04 of carrier
energy, none lands on a carrier; 2f grows with off-center bias (to ~0.067) but stays off-carrier → keep
bias near-centered in Build 4 and re-run at the learned amplitudes.

## 15. Build-4 render validation + matched-down gate (2026-06-17)

**Build 4 closed (plumbing):** quad generator (frozen coprime ladder u={13,19,33,139} v={11,17,29,113})
+ continuous-phase decoder + CRT consensus vote run end-to-end. Pattern co-designed on the proxy,
materialized (`patterns/codesign_learned/pat_00.png`), then the **production decoder render-trained**
on 400 hemisphere planes (`train_quad_rendered.py`; coord-L1 dropped per ablation). Coarse-align
converged 0.96/0.97. The verdict below is the gate, not the plumbing.

**§9.5 render-trained, plane obliquity (quad-400, full-frame vote):** 91.6/91.2/83.7/63.0/**33.6%**
bin-acc, med|du|✓bin **0.94/0.91/1.05/1.28/1.52px** (0-15…60-75°). Bounded subpixel across all bands.

**§9.5 clutter (plane-trained decoder; split depth-edge vs shadow distance transforms):** depth edges
are the dangerous failure — bin-acc collapses to 7.7% at the edge and the peak-margin **under-abstains**
(acc@τ 10.4%, cov stays ~40% → confidently-wrong false consensus). Shadow boundaries more benign
(22% bin-acc, abstention helps: acc@τ 33 vs 22 raw). Depth-edge under-abstention is the open vote bug.

**Matched-down gate (the load-bearing experiment).** Trained the M-array decoder on the *identical*
400-plane regime (bit-identical geometry via `add_marray_captures.py`; `train_marray_rendered.py`).
First run used the quad's lr 3e-4 → **failed to converge** (5% bin-acc) — the M-array codebook needs
lr 1e-3 and ~50k steps (the production "ep10≈91%" was 10 epochs on the *20k* loaf); the quad converged
in 6k steps because coprime carriers need cheap *local demodulation*, not a global random codebook. Re-ran
with the M-array's **own established recipe** (lr 1e-3, AMP, gate-offset 6, 120 epochs ≈ 24k steps).
Three columns, same 160 poses, each decoder at its honest in-distribution decode (bin models **tiled**,
quad full-frame — its native path; tiling would only lift the quad, so its column is conservative):

| obliq | quad-400 (bin% · du✓bin) | M-array-400 matched | M-array-20k (newaug ref) |
|---|---|---|---|
| 0-15  | 91.6 · **0.94** | 69.5 · 7.17 | **99.4** · 0.22 |
| 30-45 | 83.7 · **1.05** | 65.3 · 7.16 | **98.0** · 0.35 |
| 45-60 | 63.0 · **1.28** | 57.5 · 7.16 | 78.1 · 1.11 |
| 60-75 | **33.6 · 1.52** | 34.6 · 7.23 | 7.9 · 6.90 |

**Finding 1 — the "4× grazing bin-acc" headline was a data-distribution artifact, not a pattern win.**
Matched at 400 planes, quad and M-array **tie at grazing bin-acc (33.6 vs 34.6)**. The original
33.6-vs-7.9 gap existed only because M-array-20k trained on *clutter*, never on grazing planes — train
any decoder on hemisphere planes and grazing jumps 7.9→34.6. Coarse-unwrap survival at grazing is **not**
intrinsically better for the coprime code; it is mostly training-distribution. **Retire the "cliff fix"
framing.**

**Finding 2 (qualified) — the matched subpixel win is regime-specific, not general; sample efficiency
is general.** The first draft's "quad dominates subpixel" repeated Finding 1's mistake in reverse:
it compared the quad only to M-array-400, which sits *below its 70% offset gate* (flat 7.2px =
bin-centering, offset never formed) — a matched-down artifact, not a fair baseline. The gate-crossing
comparison is column three, and it goes the other way: **M-array-20k beats the quad on easy/mid precision
(0.22 vs 0.94px frontal, ~4×)**. So the quad is **not** a general precision winner. What survives matched
is narrower and architectural: **subpixel boundedness in the regimes where the M-array's gated offset
collapses** — at 60-75° quad **1.52px vs the well-trained 20k's 6.90px**. The bin+offset head only trains
its offset above 70% bin-acc, so wherever bin-acc is low (grazing for *any* M-array; everywhere for the
starved 400) the offset never forms and you get bin-centering. The quad's continuous phase has no gate, so
its subpixel stays bounded into those regimes. So the grazing story doesn't vanish — it **moves from
bin-acc (which ties) to subpixel boundedness (which the quad wins even vs the 20k)**. The
**sample-efficiency** half is solid and general: 91.6 vs 69.5% frontal at 400 planes, ~6k vs ~24k steps —
coprime local demod forms faster than a global codebook.

**Consistency — two open ceilings, not one settled win.** By the same matched-down logic, the subpixel
*ceiling* is as open as the grazing-bin-acc one: at 400 the quad's 0.94px beats a crippled M-array but
loses to the well-trained 20k's 0.22px. Whether quad-40k's easy-regime subpixel approaches ~0.22 *while
keeping its grazing boundedness* is unknown. The 40k answers **two** ceilings — grazing bin-acc and
easy-regime subpixel — not one.

**Implication for the 40k:** still worth running, but the honest pitch is **sample-efficient +
grazing-subpixel-bounded, exploratory on two ceilings** — *not* a confirmed win, and *not* a general
precision or cliff-fix claim. Justify on sample efficiency (solid) plus the two open-ceiling questions. The
**appearance-axis re-co-design** (proxy used analytic carriers, not the quantized PNG — frontal 91 vs 99
and proxy-94→render-34 both point here) remains the likeliest single lever to actually move the quad's
numbers; depth-edge under-abstention is the other open fix.

## 10. Risks

- **Proxy-fidelity gap (biggest):** planar + synthetic shading ≠ Mitsuba; mitigated by the
  photometric port, compression-coupled blur, and §9.5. Cliff-only, not edges.
- **Unwrap errors** replace bin-jumps: smoother and bounded by one coarse period, controlled by
  small `r` + coarse-survives-compression, but not eliminated.
- **Generator collapse** to a degenerate ladder: anti-collapse regularizer, weight-decay-free
  generator group, decoder warm-up.
- **Single-shot SNR per band** under the photometric stack (open question 3).
