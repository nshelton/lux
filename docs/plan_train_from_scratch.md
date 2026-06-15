# Plan — Train From Scratch on Both Loaves

## Why
The warm-start mixed run (resume from clutter model) shows the obliquity cliff
(45–75°) not moving after ~4 epochs — likely **warm-start entrenchment** (the net
defends its near-frontal basin, low plasticity for the new oblique regime). From
scratch learns frontal + oblique *jointly* from init (joint training beats sequential
fine-tuning for balanced multi-distribution performance), same wall-clock. This is the
clean test + the principled balanced-model build.

## Decision gate (do this first, cheap)
Let the current warm-start run reach **epoch 8**, then:
```
python scripts/eval_hemisphere.py --ckpt checkpoints/proj_net_mixed_ep08.pt
```
- 45–60° bin acc climbing meaningfully above baseline (~46%) → patience won; let
  warm-start finish, skip from-scratch.
- 45–60° still ≤ ~46% → entrenchment; run from-scratch below.

## The two loaves
- `renders/val_loaf` — clutter scenes (~10,117 samples).
- `renders/planar_loaf` — planar junctions over the hemisphere (~10,000 samples).
- `ConcatLoaf` mixes them ~50/50 by sample count; first `--val` of *each* held out.

## Command (from scratch — note: NO --resume)
```
python scripts/train_proj_net.py \
  --loaf renders/val_loaf renders/planar_loaf \
  --mid conv --epochs 30 --batch 64 --amp --workers 4 --val 4 \
  --lr 1e-3 --lr-min 1e-4 --offset-weight 2 --gate-offset 6 --snapshots \
  --out checkpoints/proj_net_scratch.pt \
  --logdir runs/proj_net_scratch
```
Rationale for the deltas vs the warm-start run:
- **no `--resume`** → random init (the whole point).
- **`--lr 1e-3`** (not 4e-4): random init needs the high-LR codebook-memorization
  phase; expect a ~4–5 epoch chance plateau (loss ~ ln60+ln36 ≈ 7.7) then a phase
  transition — don't panic before ~epoch 6.
- **`--offset-weight 2 --gate-offset 6`**: the self-paced gate ramps the offset weight
  in as bin accuracy forms (offset can't learn before bins exist). (Fixed
  `--offset-weight 6` also works; the gate is cleaner from scratch.)
- **`--epochs 30`** (vs 20): from scratch needs the extra epochs to re-pay the
  codebook phase. ~54 min/epoch → ~27 h; resumable, snapshots every epoch.
- **new `--out`/`--logdir`** → additive, doesn't touch warm-start or baseline.

## After it finishes
```
python scripts/eval_hemisphere.py --ckpt checkpoints/proj_net_scratch_last.pt
```
Compare `results_proj_net_scratch_last/` vs the baseline `results_proj_net/` and the
warm-start `results_proj_net_mixed_*`. Also pick the best-on-hemisphere snapshot
(`_epNN.pt`) rather than trusting the aggregate val median.

## Expectation
Likely fills **45–60°**; **60–75° probably stays poor** (cells anamorphically
compressed below resolvability = information limit, not a training problem). Report
60–75° as a known boundary + confidence-mask it, rather than chase it.
