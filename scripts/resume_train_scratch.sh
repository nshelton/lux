#!/usr/bin/env bash
# Resume the proj_net_scratch training from the latest checkpoint (epoch 23).
#
# HOW RESUME WORKS (important): train_proj_net.py's --resume loads model WEIGHTS
# only (load_weights_compatible). It does NOT restore the AdamW optimizer state,
# the cosine LR schedule, the epoch counter, or the offset-gate curriculum — all
# of those RESTART. So this is a warm-RESTART from the current weights (a fresh
# cosine run starting at --lr), not a bit-exact continuation. The offset gate
# re-ramps within ~1 epoch since bin-acc is already high. Because --resume ==
# --out and every tensor matches, it inherits the best-val bar, so it won't
# overwrite proj_net_scratch.pt with a worse early epoch.
#
# The weights are at val |du| 0.283px (epoch 23) and were still improving.
#
# Default below = full 30-epoch cosine warm-restart at lr 1e-3. The LR re-warm
# (epoch 23 was at ~2e-4) gives a brief jolt before re-converging — usually fine,
# often helps. For a GENTLE finish instead (no jolt), set: --epochs 8 --lr 3e-4
set -euo pipefail
cd /Users/nshelton/lux

nohup caffeinate -is .venv/bin/python scripts/train_proj_net.py \
  --loaf renders/val_loaf renders/planar_loaf \
  --mid conv --epochs 30 --batch 64 --amp --workers 4 --val 4 \
  --lr 1e-3 --lr-min 1e-4 --offset-weight 2 --gate-offset 6 \
  --snapshots --no-tensorboard \
  --resume checkpoints/proj_net_scratch.pt \
  --out checkpoints/proj_net_scratch.pt \
  --logdir runs/proj_net_scratch \
  >> checkpoints/train_scratch.log 2>&1 &

echo "resumed proj_net_scratch (PID $!) from checkpoints/proj_net_scratch.pt"
echo "watch:  tail -f checkpoints/train_scratch.log"
