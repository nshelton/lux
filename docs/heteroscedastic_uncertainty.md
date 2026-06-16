# Heteroscedastic offset uncertainty — calibrated σ for the correspondence net

Added 2026-06-16 (code in; **first training run deferred** until the conv+aug run
`proj_net_conv_aug.pt` finishes, to keep that a clean A/B). This note records the
addition so it can be picked up cold on the next run.

## What it is

A second regression head that predicts the **log-variance of each offset**, trained
with a **β-NLL** loss, turning the net's heuristic softmax confidence into a
mathematically grounded, per-pixel **aleatoric (data) uncertainty** map — accounting
for sensor noise, saturation and blur. Same idea as HSU-Net / HSURE-CFPP: "a result
is incomplete without its uncertainty."

The architecture already carried two *different* uncertainty signals:
- **bin classification** (60 u-bins×32px, 36 v-bins×30px) → softmax-max gives
  "which projector cell, how sure" (the bin-flip / unwrap-failure risk).
- **within-bin offset** (the subpixel fraction) → was a bare L1 with **no**
  uncertainty at all.

The offset is exactly where a variance head adds *new* information (not a re-derivation
of the softmax), and where the heteroscedasticity is physically real: a saturated or
blurred cell can be bin-confident yet subpixel-ambiguous. So the new head models the
**subpixel** precision; the existing softmax models the **bin-outlier** risk; they are
complementary and get fused (below).

## The change (backward-compatible — defaults leave existing behavior byte-for-byte)

All in `lux/proj_net.py` + `scripts/train_proj_net.py`. A non-heteroscedastic model
is still a 99-channel head with the identical loss/inference path; `--hetero` opts in.
Existing checkpoints load unchanged (the new `hetero` flag defaults `False`).

| piece | where | what |
|---|---|---|
| `ProjUNet(heteroscedastic=False)` | `proj_net.py` `ProjUNet.__init__` | when `True`, head emits **+2 channels** (`logvar_u, logvar_v`), inserted **before** validity so `pred[...,-1]` (validity) is unchanged. +0 meaningful params (still 7.86M). logvar bias zero-init ⇒ σ²=1 ⇒ NLL starts MSE-like. |
| β-NLL term | `proj_net.py` `proj_loss(..., nll_weight, nll_beta)` | `nll = 0.5·exp(−s)·res² + 0.5·s`, reweighted by `detach(exp(s)^β)` (Seitzer et al. 2022). The **L1 offset term is kept as an anchor** so the coordinate is always fit. `nll_weight=0` (default) = no-op. |
| fused σ | `proj_net.py` `predict_full(..., return_sigma=True)` | returns `(uv, σ_u, σ_v)` in **projector px** (NaN-masked like `uv`): `var = (1−p_bin)·(bin_px²/12) + p_bin·σ_offset²` per axis. `p_bin` = softmax-max (bin-flip risk free from the classifier); `σ_offset` from the log-var head. Non-hetero model ⇒ `σ_offset=0`, degrades to a bin-risk-only map. |
| checkpoint flag | `proj_net.py` `save/load_checkpoint` | `hetero` stored + restored, so models round-trip without specifying arch. |
| curriculum wiring | `train_proj_net.py` train loop | `gate = clip((ep_bin−0.70)/0.25, 0, 1)` (same bin-acc gate as the offset); `nll_warm = min(ep/nll_warmup_epochs, 1)`; **`nll_w = nll_weight · nll_warm · gate`**. Double-gated: the variance head can't learn before the offset **mean** is trustworthy (bin acc clears 70%) *and* ramps in over epochs. |
| CLI | `train_proj_net.py` | `--hetero`, `--nll-weight` (0=off, ~0.04 start; implies `--hetero`), `--nll-beta 0.5`, `--nll-warmup-epochs 10`. Logged as `train/nll_weight`. |

### The one real failure mode it guards against

Plain Gaussian NLL scales the mean gradient by precision `exp(−s)`, so the net can cut
loss by **inflating variance on hard pixels instead of fitting them** ("gradient
shrugging") — a direct threat here, because the hard pixels (oblique cliff, edge rows)
are exactly the documented limiters. Two guards: **β-NLL** (β=0.5 restores the
mean-fitting gradient on high-variance pixels) **and** the retained **L1 anchor**.
Verified on a smoke run: with NLL active, the offset channels still receive a strong
mean gradient (L2≈3.9) while the variance channels learn slowly.

## How to launch the next run

```bash
~/.venvs/lux/bin/python scripts/train_proj_net.py \
  --loaf ~/datasets/val_loaf ~/datasets/planar_loaf \
  --mid conv --epochs 30 --batch 32 --amp --workers 4 --val 4 \
  --lr 1e-3 --lr-min 1e-4 --offset-weight 2 --gate-offset 6 \
  --hetero --nll-weight 0.04 --nll-beta 0.5 --nll-warmup-epochs 10 \
  --snapshots --no-tensorboard \
  --out checkpoints/proj_net_hetero.pt --logdir runs/proj_net_hetero
```

(NVMe loaves, batch-32 ceiling, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` —
see `linux-training-environment` memory.)

### Open decision: from-scratch vs head-graft warm-start

The head is a **single** 1×1 conv emitting bins+offsets+logvars+validity, so a naive
`--resume` from a 99-channel conv checkpoint (shape mismatch 99→101) makes
`load_weights_compatible` skip the **whole head**, reinitializing the trained bin
classifier too. Options:
- **From scratch** (command above) — simplest, matches HSU-Net's from-scratch NLL
  warmup, ~12 h.
- **Head-graft warm-start** (recommended, not yet implemented) — copy the converged 99
  head channels into the new head, leave only the 2 logvar channels fresh, train ~3–5
  epochs. The *right* regime for NLL (means already good → variance head just aligns to
  empirical residuals) and far cheaper. Needs a ~5-line graft helper + `--graft-head`
  flag.

## How to judge it (NOT on val |du|)

Honest framing: this **will not move the headline metric** (mean |du|) — the obliquity
cliff and row deficit are bin-*misclassification* problems the offset NLL doesn't touch.
What it buys: **traceability** (aleatoric vs OOD-model error budgeting), a
**downstream-usable** uncertainty (`1/σ²` weighting in triangulation / temporal fusion —
the real metrology payoff), and mild training robustness (NLL acts like a learned
per-pixel robust loss). Keep the headline runs aimed at the cliff; use this branch to
build the reliability-aware map that eventually *gates* the oblique results.

Validation to add when the checkpoint lands (not yet written):
1. **Reliability diagram** — predicted σ vs actual offset-error percentiles, **bucketed
   by obliquity bin**. "Mathematically calibrated" is aspirational, not automatic (NLL
   assumes Gaussian; saturation is heavier-tailed) — verify, then a one-parameter
   temperature on the variance if it's off.
2. **Downstream-weighted depth RMSE** — `1/σ²`-weighted reconstruction vs unweighted, on
   the hemisphere bench, per bin. This is the test that the map carries real value.

## Priority partner: a learned bin-correctness head (the cliff-relevant half)

σ_offset above is the **frontal-regime** uncertainty (subpixel precision given the bin
is right). The **cliff-regime** uncertainty — bin-flip / unwrap-failure risk — is the
higher-leverage piece for the project's actual deliverable, and it is *not* yet built.

Two facts force it:
- The fused-σ formula uses `p_bin = softmax.max()` as the bin-correctness estimate, but
  raw softmax is **overconfident exactly in the oblique band** — so the fused map is
  least trustworthy where it matters most.
- The validity head can't cover it: `proj_loss` trains validity as `BCE(pred[:,-1], m)`
  with `m = isfinite(gt_proj)`, which is ~always-1 on lit, in-frame, oblique pixels — it
  structurally **cannot learn to abstain on hard-but-valid** pixels.

Fix (queued with the cliff work, see `docs/cliff_plan.md`): add a **learned
bin-correctness head** that predicts "will this argmax bin equal the GT bin?" (per axis,
joint = min), and **rewire `predict_full`'s fused-σ to consume it instead of
softmax-max**. Design constraints, all reusing machinery already here:
- **Shares the obliquity weighting** (step 4 in the cliff plan): bin-flips concentrate
  in the oblique band, so obliquity-weighting importance-samples the rare "incorrect"
  class — one reweighting scheme serves both the main loss and the correctness head.
- **Rides the bin-acc gate** (the same gate the NLL term rides): "will the bin be right?"
  is meaningless before the codebook phase-transition, and its target distribution shifts
  as bins form, so train it late.
- **Scored by obliquity-stratified risk-coverage / AURC**, not accuracy — "accurate to
  70° + knows when to shut up" is a selective-risk statement. Add that panel to the
  hemisphere bench. "Learned" ≠ "calibrated": keep the reliability diagram too, since it
  drifts under the sim→real shift.

Net: σ_offset (here) + learned P(bin-correct) = the full calibrated map; the latter is
the one that turns the 70–75° floor from a failure into a feature.

## Reference

- Seitzer et al., *On the Pitfalls of Heteroscedastic Uncertainty Estimation with
  Probabilistic Neural Networks*, ICLR 2022 (β-NLL).
- Kendall & Gal, *What Uncertainties Do We Need…*, NeurIPS 2017 (aleatoric NLL head).
- HSU-Net / HSURE-CFPP (Source 1): mixed MSE+NLL with the NLL coefficient warmed 0→0.04
  over the first 30 epochs; calibration via reliability diagrams.
