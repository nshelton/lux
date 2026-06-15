# proj_net_scratch evaluation — ep23 (2026-06-15)

Re-ran the from-scratch correspondence net on both benchmarks after the latest
training, having copied `proj_net_scratch_ep23.pt` over `checkpoints/proj_net_scratch.pt`.
Two independent evals agree: **the model has plateaued — the extra epochs and the
checkpoint swap do not move accuracy on either synthetic or real data.** The real
limit is *confident-correct coverage on hard inputs*, not validity masking or capacity.

## Checkpoint provenance (read this first)

The `proj_net_scratch_*` files span **two training runs**, which is easy to mix up:

| checkpoint | epoch | mixed-val \|du\| | notes |
|---|---|---|---|
| `proj_net_scratch.pt` (was, pre-copy) | 7 | 0.252 | best-val of the *restarted* Jun-15 run (8 epochs in) |
| `proj_net_scratch_last.pt` | 8 | 0.253 | rolling last of the Jun-15 run |
| `proj_net_scratch_ep23.pt` | 23 | 0.283 | final of the *first* Jun-14 run |

`ep01–ep08` snapshots are Jun-15 (restarted run); `ep09–ep23` are Jun-14 (first run).
`proj_net_scratch.pt` is **now ep23** (copied 2026-06-15). Note ep7 has a *better*
mixed-val (0.252) than ep23 (0.283) — best-val ≠ best on either benchmark below.

## Hemisphere (synthetic obliquity sweep)

`scripts/eval_hemisphere.py`, 160 samples, binned by `max(camera, projector)` tilt
off normal. Three checkpoints — old benchmark, ep7 (best-val), ep23:

**Bin accuracy (%)**
| obliquity | OLD | ep7 | ep23 |
|---|---|---|---|
| 0–15° | 96.7 | 95.1 | 96.6 |
| 15–30° | 95.1 | 94.3 | 95.1 |
| 30–45° | 87.5 | 85.5 | 87.0 |
| 45–60° | 48.8 | 48.5 | 50.8 |
| 60–75° | 5.1 | 4.6 | 4.7 |
| **all** | 60.6 | 54.8 | 61.0 |

**Median |du| (px):** 45–60° is ep23's only clear win (11.4 vs ep7's 18.6); 60–75°
stays catastrophic in all three (~400 px). Within-correct-bin error stays ~1–7 px
even at grazing, and **validity IoU is 0.98–0.999 in every bin**.

Takeaways:
- The **obliquity cliff is structural**: sub-pixel and 85–96% accurate ≤45°, then
  collapses (~50% at 45–60°, ~5% at 60–75°). Checkpoint choice buys ~6 pts of overall
  bin acc, *all of it in the ≤45° regime that already works.*
- It is **coarse-bin misclassification, not coverage** — the validity head keeps the
  pixels (IoU high); the net picks the wrong projector column once foreshortening
  squashes the M-array window. See `memory/hemisphere-obliquity-cliff.md`.

## Real captures (5 scenes, hybrid reference)

`scripts/eval_capture.py`, reference = Gray-coded phase shifting (hybrid). Before =
prior `eval_proj_net_scratch/`; after = ep23. **Before → after, essentially unchanged
(marginally worse):**

| scene | u-bin acc | med \|du\| | coverage | IoU |
|---|---|---|---|---|
| test0 | 91.9 → 91.4% | 1.11 → 1.11px | 0.995 → 0.996 | 0.783 → 0.742 |
| test1 | 34.8 → **33.4%** | 159 → **166px** | 0.988 → 0.989 | 0.986 → 0.987 |
| test2 | 83.8 → 82.9% | 1.39 → 1.41px | 0.990 → 0.992 | 0.938 → 0.930 |
| test3 | 52.7 → **51.7%** | 5.67 → 6.71px | 0.989 → 0.990 | 0.956 → 0.949 |
| test4 | 90.8 → 89.6% | 1.11 → 1.25px | 0.995 → 0.997 | 0.989 → 0.987 |

test4 also has a horizontal Gray-code set (`graycode_h`): **v-bin 78.6%, med |dv|
1.29 px, full (u,v) bin acc 73.0%** — the row axis is the weaker half (corroborates
`memory/proj-net-row-deficit.md`).

### The confidence sweep is the real story

Ungated bin-acc on test1/test3 looks catastrophic, but that's *low-confidence*
predictions dragging the median. Gating on the net's per-axis softmax confidence:

| scene | useful operating point | accuracy @ coverage |
|---|---|---|
| test0 | conf ≥ 0.9 | 97.0% @ 84% |
| test2 | conf ≥ 0.7 | 95.6% @ 74% |
| test4 | conf ≥ 0.7 | 94.9% @ 90% |
| test3 | conf ≥ 0.7 | 94.2% @ **37%** |
| test1 | conf ≥ 0.5 | 92.0% @ **7.6%** |

**When the net is confident, it is accurate (≥92% bin acc at conf ≥ 0.5 on every
scene).** The failure on hard scenes is that almost nothing clears the confidence bar:
on **test1 the net abstains on ~92% of the frame** (only ~8% confident coverage),
test3 only ~a third is usable. This is the real-data face of the "low coverage"
problem — it is *confident-correct coverage*, not the validity mask (IoU ~0.99).

**test1 is a near-total failure** and smells like a domain gap (appearance / SNR /
defocus / obliquity the synthetic training set doesn't cover) — the scene to
investigate next.

## Diagnosis

1. The scratch model has **plateaued**: more epochs and the ep7→ep23 swap do not
   improve synthetic or real accuracy.
2. The recurring failure mode across both benchmarks is the **same**: the net stays
   confident-and-correct in its comfort zone and *abstains or mis-bins* outside it
   (grazing angles synthetically; test1/test3 on real hardware). Validity IoU stays
   high throughout — coverage of the *valid mask* is not the issue; coverage of
   *confident, correct* pixels is.
3. Checkpoint selection is a small lever (~6 pts, in the already-working regime).
   Confidence gating is a real lever but only **trades** coverage for accuracy; it
   does not create usable pixels where the net has none (test1).

## Recommendation

- **Do not** reach for more epochs or the `--mid attn` bottleneck first — neither
  addresses what the data shows is limiting.
- **Fix the training distribution.** Oversample oblique poses / add anamorphic
  (foreshortening) augmentation for the 45°+ cliff; and characterize what makes
  **test1** fail so its conditions can be represented in training.
- Use `--snapshots` + a hemisphere sweep to *pick* the checkpoint (mixed-val
  mis-ranks them), but treat that as fine-tuning, not the fix.

## Repro

```bash
# hemisphere (synthetic)
python scripts/eval_hemisphere.py --ckpt checkpoints/proj_net_scratch.pt   # -> evals/hemisphere/results_proj_net_scratch/

# real captures (per scene -> captures/<scene>/eval_proj_net_scratch/)
for s in test0 test1 test2 test3 test4; do
  python scripts/eval_capture.py --captures captures/$s --ckpt checkpoints/proj_net_scratch.pt
done
```
