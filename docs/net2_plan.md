# net2 — Attention-Bottleneck Correspondence Net (plan + launch record)

Started 2026-06-15. net2 is the queued transformer A/B from `session_summary.md`:
the same proven **from-scratch** recipe that produced the best conv model
(`proj_net_scratch.pt`, val |du| 0.28px / bin 87%), but with `--mid attn` — a 17M-param
transformer bottleneck (global self-attention at 1/16 res) instead of the 7.85M conv one —
**plus the new `_augment_crop`** (blur / contrast-clip / noise) the conv scratch net never had.

Launch: `scripts/train_net2_attn.sh` → `checkpoints/proj_net_attn.pt` (+ `_epNN`/`_last` snapshots),
log `checkpoints/train_attn.log`. ~60 min/epoch on MPS (attn is ~12% slower than conv), **30 epochs ≈ ~30 h**.

## The recipe (and the deltas vs the conv scratch run)

```
.venv/bin/python scripts/train_proj_net.py \
  --loaf renders/val_loaf renders/planar_loaf \
  --mid attn --epochs 30 --batch 64 --amp --workers 4 --val 4 \
  --lr 1e-3 --lr-min 1e-4 --offset-weight 2 --gate-offset 6 \
  --warmup-steps 400 --grad-clip 1.0 \
  --snapshots --no-tensorboard \
  --out checkpoints/proj_net_attn.pt --logdir runs/proj_net_attn
```

| delta | why |
|---|---|
| `--mid attn` | the architecture under test — can global attention gather edge evidence a conv RF can't reach? |
| no `--resume` | random init — clean attn-vs-conv comparison (not warm-start, which the docs found entrenches). |
| `--warmup-steps 400` | **new flag.** Per-step linear LR warmup 0→1e-3. Transformers from random init want it; the conv net didn't. The per-epoch cosine recomputes from its own counter, so the mid-epoch-1 overrides don't perturb it. |
| `--grad-clip 1.0` | **new flag.** Clip grad L2 norm after AMP unscale — early-spike hygiene for the fresh MHSA stack. |
| (NaN tripwire) | **unconditional** in `train_proj_net.py`: abort after 30 consecutive non-finite losses. A persistent NaN makes GradScaler skip forever without repairing — this fails loud instead of silently burning 30 h and overwriting good snapshots. |

Kept `--lr 1e-3` (the from-scratch codebook phase needs the high LR; a global cut to 5e-4 would
under-drive the conv encoder/decoder where the codebook actually lives — the documented failure
mode is *too-low* LR entrenching, not too-high).

## Why these safety tweaks (adversarial recipe review, 4 lenses)

All four lenses returned **go-with-tweaks**. Key findings:
- **Empirically verified on this MPS box:** attn-from-scratch at lr 1e-3 under fp16 AMP *does*
  converge — the first 1–2 steps overflow fp16, GradScaler skips them (scale 65536→16384), then
  gradnorm decays 52.8→1.0 and loss falls monotonically from chance (~11). Safe as-is, but the
  margin leans on the scaler swallowing the spike → warmup + clip remove that dependence.
- **The one real failure mode:** a NaN that reaches the weights → GradScaler skips forever but never
  repairs → a silent dead run that keeps writing NaN snapshots. Nothing detected it → added the tripwire.
- softmax / LayerNorm auto-upcast to fp32 under MPS autocast (verified), so MHSA runs finite; the mid
  does **not** need to be excluded from autocast.

## ⚠️ Resolution finding (the big one — 2026-06-16, ~epoch 9)

**net2's val collapsed to near-chance while train was healthy — and it's an inference-resolution
artifact, not a training failure.** The attention bottleneck does *global* self-attention over the
1/16-res grid, so its token count scales with the input: a 256-px training crop = 16×16 = **256
tokens**, but `predict_full` runs the whole 1080×1920 frame = ~68×120 = **8160 tokens** — a regime
the softmax + absolute sinusoidal PE never trained on. Bin-acc degrades monotonically with token
count (verified on the ep9 ckpt):

| input | tokens | ATTN bin u/v | CONV bin u/v |
|---|---|---|---|
| 256px crop (train res) | 256 | **94.7 / 94.2** | 99.2 / 99.1 |
| 512px crop | 1024 | 68.5 / 44.8 | 99.5 / 99.2 |
| full frame | 8160 | **23.7 / 26.8** | 95.3 / 78.7 |

Conv is shift-invariant (fixed RF) so crop≈full-frame; attention is sequence-length-sensitive so it
collapses. **More epochs won't fix it** — it's architectural. Ruled out: the new aug, warmup, lr (all
fine; train side is on conv's pace).

**Fix: `lux/proj_net.py:predict_tiled`** — stitch full-frame inference from 256-px tiles (training
crop size), each output pixel taken from a tile where it's ≥`margin` from the edge. Recovers the attn
net **24%→95% / 432px→1.3px** (planar), 26%→88% (clutter) with no retrain. Wired into
`eval_hemisphere.py`, `eval_capture.py`, `predict_proj_net.py` via `--tiled`, **auto-enabled when the
checkpoint arch is `attn`**. Costs ~4–5 s/frame on CPU (~60 tiles), comparable to full-frame.

The default hard-stitch (each pixel from one tile) leaves **square seams** in low-context (background)
regions where adjacent tiles disagree. `--tile-overlap N` (predict_tiled `overlap=N`) fixes it:
overlapping tiles at `stride = tile−N` + per-pixel **max-confidence** selection — a tile's own edges
have least context → lowest conf → they lose to a tile where the pixel is well-centred, so seams
dissolve. `--tile-overlap 128` (50% overlap → 4× per-pixel coverage, ~2× total forward passes vs the
hard-stitch) cleaned test3 (grid on the plane gone) and
*improved* metrics (u-bin 71→73%, more confident coverage). Default 128 in `predict_proj_net.py`
(deployment), 0 in `eval_hemisphere.py` (bench: fast, seams don't move the median).

**Bonus: tiling also closes the conv row deficit.** Conv full-frame v-bin 78.7% → tiled **99.4%** —
the long-documented v<u gap was *partly* a train-256-crop / eval-full-1080-frame mismatch at the
extreme top/bottom v-bins (the vertical-context regime never seen in 256 training). See
`net1_takeaways_dv_accuracy.md`; consider re-running the conv hemisphere bench with `--tiled`.

**Caveat for the live run:** the in-loop `evaluate()` uses full-frame `predict_full`, so net2's
**val curves in TensorBoard stay collapsed** for the whole run (the training process imported the
pre-`predict_tiled` code and we won't restart it). Judge net2 by the **train** curves; measure true
performance with the tiled hemisphere bench at the end. (For *future* attn runs, make `evaluate()`
tiled-aware.)

## Honest expectation (read before reading the results)

The project's own evidence predicts **attention will NOT move the limiter.** The >45° obliquity
cliff and the v-row edge-bin deficit are **training-distribution** limits (coarse-bin
*misclassification* at high validity-IoU), not capacity/bottleneck limits — see
`proj_net_scratch_eval.md` ("Do not reach for … the `--mid attn` bottleneck first") and
`net1_takeaways_dv_accuracy.md` ("the fix is structural, not a loss knob"). So the likely outcome is
attn **ties or marginally beats** conv on the already-working ≤45° regime and does little for the cliff.
Run it as a deliberate architecture A/B with eyes open — the one mechanism that *could* surprise us is
global attention pulling edge-bin evidence a fixed conv RF cannot.

## Confounds to keep straight

net2 changes **two** things vs the conv scratch baseline: the bottleneck (conv→attn) **and** the new
aug. A win/loss is therefore not cleanly attributable. Acceptable because the goal is a better
real-world net (the aug targets sim2real), but the clean control is a **conv + new-aug** run — see
follow-ups.

## After it finishes — evaluate per-bin, not aggregate

`--snapshots` is on. **Pick the checkpoint on the hemisphere bench, per obliquity bin** (aggregate
val |du| mis-ranks them — ep7 once beat ep23 on val but not on the bench):

```
for c in checkpoints/proj_net_attn_ep*.pt; do python scripts/eval_hemisphere.py --ckpt "$c"; done
# compare per-bin (esp. 30–45° / 45–60°, where checkpoints differ) vs results_proj_net_scratch/
python scripts/eval_capture.py --captures captures/test4 --ckpt checkpoints/proj_net_attn_<best>.pt
```

Success criterion: **attn ≥ conv on the ≥45° bins** (and the real-capture (u,v)), not "cliff fixed."

## Queued follow-ups (higher-leverage than attn, per the docs)

1. **conv + new-aug control** — same recipe, `--mid conv`, the new aug. Isolates the aug's
   sim2real benefit from the bottleneck change. Cheaper (~conv speed). Can't run concurrently
   on this one-GPU Mac, so it's the next slot.
2. **Real-data Gray-code fine-tune** (Priority 1 in `net1_takeaways.md`) — fine-tune the best base net
   on real captures using the hybrid Gray-code (u,v) as pseudo-GT. Closes the ~1px sim→real gap.
   Needs a base net first → obvious next run after net2.
3. **Oblique / full-vertical-range training distribution** — the documented *real* fix for the cliff +
   row deficit. Needs re-rendering an oblique-rich loaf and/or a full-height-strip crop change (the
   loaf/model currently use one square crop S for both axes). Multi-hour, separate job.
