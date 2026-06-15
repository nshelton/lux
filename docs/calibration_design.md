# Step 2 — Self-Calibration from Network Correspondence

Design for recovering a **metric (up-to-scale) projector–camera calibration** directly
from the network's dense camera→projector correspondence — no physical checkerboard.

Step 1 (the `ProjUNet` decoder, see `docs/proj_net_design.html`) turns one M-array
capture into a dense **c2d** map: for each camera pixel, the projector subpixel
`(u_p, v_p)` that lit it (`pred_proj.npy`), plus a per-pixel confidence. Step 2
consumes that map and solves for the rig.

---

## 1. Purpose & scope

- **In:** one or more c2d correspondence maps (+ confidence) from `predict_proj_net.py`.
- **Out:** camera intrinsics, projector intrinsics, relative pose `(R, t)`, dense
  structure — all **up to a global scale** unless a metric anchor is supplied.
- **Why it's possible:** a projector is a camera in reverse, so a c2d map *is* a
  dense stereo correspondence. This is two-view geometry / self-calibration, with
  the unusual luxury of ~2 M outlier-free correspondences per view.
- **Non-goals:** replacing a one-time metric checkerboard calibration where one is
  available and convenient; photometric/color calibration; pattern defocus modeling.

---

## 2. The observability ladder (what the data can and cannot give)

The single governing fact: **each distinct camera↔projector relative pose yields one
fundamental matrix `F` (7 DOF), which supplies ≈2 constraints on the intrinsics**
(Kruppa / image of the absolute conic). Everything below follows from counting these.

- **`F` alone → projective reconstruction only.** Scene + devices recovered up to an
  arbitrary 3D projective transform (15-DOF ambiguity). Not metric.
- **Metric upgrade needs intrinsic priors.** Two devices have 10 intrinsic params;
  `F` gives only 2 constraints on them. Assume zero skew (−2), square pixels /
  known aspect (−2), principal point ≈ centre (−2 to −4) ⇒ down to **2 unknowns
  (the two focal lengths)**, which exactly matches the 2 available constraints
  (Bougnoux formula → `f_cam`, `f_proj`; then `E = K_pᵀ F K_c` → `R`, `t`-direction).
- **`F` depends only on the rig (intrinsics + relative pose), *not* the scene.** This
  is the pivotal consequence for the two regimes below.

---

## 3. Two capture regimes

### A. Fixed rig (projector bolted to camera) — deployment configuration

One relative pose ⇒ **one `F` forever** ⇒ 2 intrinsic constraints, regardless of how
many scenes you shoot (every scene samples the *same* epipolar geometry).

| | from a single depth-rich capture |
|---|---|
| `f_cam`, `f_proj` | ✅ (with centred principal point) |
| `R`, `t`-direction | ✅ |
| dense depth | ✅ (up to scale) |
| `k1` | ✅-ish — distortion shows as global radial deviation from pinhole epipolar consistency across the full frame (Fitzgibbon-style joint `F`+distortion); needs frame-filling coverage |
| `cx`, `cy` | ❌ — 3 unknowns (`f,cx,cy`) vs 2 constraints; underdetermined. Trades against the epipole/pose. **More scenes do not help.** |
| `k2` | ❌ usually noise (periphery-only, one pose) |
| global scale | ❌ always — needs one metric anchor |

**Hard requirement:** the scene must be **non-coplanar**. A flat surface gives a
homography, which does *not* determine `F` → infinitely many calibrations. A flat
section alone is uncalibratable. Also avoid near-parallel optical axes / tiny
baseline (focal formulas go unstable).

### B. Static projector, moving camera — full self-calibration

`N` camera poses ⇒ **`N` distinct `F`'s** ⇒ `2N` constraints ⇒ the *full* intrinsic
set (camera **and** projector), trajectory, and structure — up to scale. This is the
regime that legitimately recovers `cx, cy, k1, k2`.

Why it's better than ordinary multi-view SfM: the projector pattern assigns every
scene point a **unique global ID** (its M-array projector coordinate), so cross-view
data association is **free, dense, drift-free, outlier-free** — two camera pixels in
different views that decode to the same projector coordinate are the same point. The
static projector is a fixed "anchor view"; each camera pose forms a stereo pair with
it → a fan of baselines around the scene → a very well-conditioned dense BA.

Caveats specific to this regime:
- **Scale still unobservable** → one known displacement / dimension fixes it.
- **Motion must be general** — pure rotation (no parallax) and collinear camera
  centres are critical/degenerate (Sturm). Orbit the scene; translate, don't just pan.
- **Planar-scene requirement relaxes** — multiple views of even an unknown plane give
  multiple homographies that constrain intrinsics (unknown-plane cousin of Zhang's
  method); non-planar is still better conditioned.

---

## 4. Pipeline

1. **Load & filter.** Read c2d map(s) + confidence; keep high-confidence
   correspondences (the bin-softmax confidence from Step 1); subsample to a
   well-distributed set (e.g. ~50 k spanning the whole frame) for the linear stages,
   keep all for final BA weighting.
2. **Robust `F` (degeneracy-aware).** RANSAC with a planar guard (H vs F model
   selection, e.g. GRIC) so a near-planar scene is *detected*, not silently fit.
   With dense data, optionally fit radial distortion jointly (division model).
3. **Intrinsic init.** Priors (centred PP, unit aspect, zero skew) → focal lengths via
   Bougnoux/Kruppa → `E = K_pᵀ F K_c` → decompose to `R, t`-direction; resolve the
   four-fold ambiguity by cheirality (points in front of both devices).
4. **Triangulate** dense structure (up to scale).
5. **Bundle adjustment, incremental.** Refine in stages — `f + pose + depths` →
   add `k1` → admit `cx,cy / k2` **only if observable** (§5). Regularize with light
   priors (PP≈centre, distortion≈0); use Step-1 confidence as measurement weights `Σ`.
6. **Observability gate (§5).** Read the FIM/covariance before trusting any parameter.
7. **Scale.** Apply the metric anchor (known camera displacement or scene dimension).
8. **(Regime B)** Multi-view BA over the trajectory; cross-view association by
   projector-code identity; anchor the world frame to the projector.

---

## 5. Observability & validation (don't trust, measure)

The same matrix under three names: optimization **Hessian** ≈ **`JᵀΣ⁻¹J`** ≈ the
**Fisher Information Matrix**. Covariance `= FIM⁻¹` (Cramér–Rao lower bound), so the
FIM states the best achievable precision of every parameter — independent of solver.

- **Covariance + correlation** → per-parameter uncertainty and coupling
  (`f↔k1`, `cx↔tₓ`).
- **Eigen-spectrum + eigenvectors** → small eigenvalues are low-information
  *directions*; the eigenvector names the weak parameter combination.
- **Null space** → exact gauge freedoms. Global scale is always here; a single-pose
  `cx,cy` show up as near-null. This is how you *prove* a degeneracy, not guess it.
- **Schur complement onto the intrinsic block** → marginalize the millions of
  per-point depths, then inspect the reduced intrinsic information matrix (this is the
  BA "reduced camera system"). Essential for honest intrinsic observability.
- **Held-out residuals** → fit on a subset, measure reprojection error on unused
  correspondences. Train error down but holdout flat = overfitting / noise-chasing.
  The empirical discriminator for "refinement vs noise."
- **Optimal experiment design (Regime B).** The FIM predicts precision *before*
  capture, so pick the next camera pose to maximize information about the weak
  parameters — **E-optimality** (lift the smallest eigenvalue) is the natural
  "next-best-view" for nailing `cx,cy`. Active calibration.

> Rule of thumb: 2 M points kill *variance*, not *degeneracy*. A weakly-observable
> parameter just gets estimated precisely-wrong (absorbing other error). Sample size
> never rescues a structural ambiguity — only priors, a metric anchor, or pose
> diversity do.

---

## 6. Degeneracies (explicit checklist)

- Planar / coplanar scene → homography → `F` undetermined (fixed-rig killer).
- Global scale → always unobservable; needs one metric input.
- No intrinsic priors → projective reconstruction only.
- Near-parallel axes / small baseline → focal-from-`F` unstable.
- Pure camera rotation / collinear camera centres (Regime B) → no parallax / critical.

---

## 7. Recommendation for deployment (projection mapping)

Hybrid, exploiting both regimes:

1. **One-time, Regime B:** scan a handheld (or motorized) camera around a static
   projected scene → full intrinsic self-cal of camera **and** projector + distortion,
   up to scale; fix scale with one measured displacement. Validate with the FIM and a
   holdout split.
2. **Per-install, Regime A:** the fixed deployment rig reuses those intrinsics, so each
   site only solves the **relative pose + scene** from a single depth-rich capture —
   a well-posed problem. Never calibrate intrinsics off a single flat section.

This keeps the field workflow to one shot while keeping the intrinsics trustworthy.

---

## 8. Open questions / future work

- Distortion model for the joint `F`+distortion fit: division model (Fitzgibbon) vs
  Brown–Conrady (matches the renderer's `optics.py`).
- Heteroscedastic weighting: feed the Step-1 per-pixel confidence in as `Σ` so the FIM
  and BA down-weight uncertain correspondences principledly.
- Off-the-shelf back end: the c2d map drops straight into a Ceres/GTSAM factor graph
  (projector = one fixed pose node; camera poses + intrinsics + structure as the rest).
  Sketch the factor graph and wire it.
- Implementation: `scripts/calibrate_from_correspondence.py` — Regime A first
  (single-pose `f`+pose+depth up to scale, with the FIM observability report), then
  Regime B multi-view BA.
