#!/usr/bin/env bash
# net2 — the attention-bottleneck correspondence net.
#
# Same proven FROM-SCRATCH recipe that produced the best conv model
# (proj_net_scratch.pt: val |du| 0.28px, bin 87%), but with --mid attn (17M-param
# transformer bottleneck instead of the 7.85M conv one) + the new _augment_crop
# (blur/contrast-clip/noise) the conv scratch net never had.
#
# Deltas vs resume_train_scratch.sh, all to de-risk a fresh transformer at lr 1e-3
# over a ~30h unattended MPS run (see docs/net2_plan.md for the review that motivated them):
#   --mid attn        the architecture under test (global attention at 1/16 res).
#   NO --resume       random init — clean attn-vs-conv comparison (not warm-start).
#   --warmup-steps 400 per-step linear LR warmup 0->1e-3 (transformers want it; conv didn't).
#   --grad-clip 1.0   clip grad norm after AMP unscale — early-spike hygiene.
#   (the NaN tripwire — abort after 30 consecutive non-finite losses — is unconditional
#    in train_proj_net.py so a wedged run fails loud instead of silently burning 30h.)
#
# Keep lr 1e-3 (the from-scratch codebook phase needs it; a global cut to 5e-4 would
# under-drive the conv encoder/decoder where the codebook actually lives).
#
# PICK THE CHECKPOINT POST-HOC on the hemisphere bench (per-bin), NOT the auto *best*
# (aggregate val |du| mis-ranks obliquity). Sweep eval_hemisphere.py over _epNN snapshots.
set -euo pipefail
cd /Users/nshelton/lux

export PYTORCH_ENABLE_MPS_FALLBACK=1

nohup caffeinate -is .venv/bin/python scripts/train_proj_net.py \
  --loaf renders/val_loaf renders/planar_loaf \
  --mid attn --epochs 30 --batch 64 --amp --workers 4 --val 4 \
  --lr 1e-3 --lr-min 1e-4 --offset-weight 2 --gate-offset 6 \
  --warmup-steps 400 --grad-clip 1.0 \
  --snapshots --no-tensorboard \
  --out checkpoints/proj_net_attn.pt \
  --logdir runs/proj_net_attn \
  >> checkpoints/train_attn.log 2>&1 &

echo "launched proj_net_attn (PID $!) — from scratch, --mid attn, both loaves + new aug"
echo "watch:  tail -f checkpoints/train_attn.log"
echo "stop:   kill $!"
