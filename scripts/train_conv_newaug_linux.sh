#!/usr/bin/env bash
# conv + NEW physically-ordered aug, warm-restarted from the old-aug baseline at ep10.
#
# Drops the merged physically-ordered _augment_crop (shot noise ∝√signal, correct
# image-formation order, wider params) into the run at "ep11" by resuming the ep10
# weights. The codebook is already formed (ep10 bin ~91%), and the new aug is the
# same degradation FAMILY, so this warm-restart adapts (refines noise robustness)
# rather than relearning. Expect a brief train-metric dip (harder aug) then recovery.
#
#   - --resume proj_net_conv_aug_ep10.pt : weights only (warm-RESTART: fresh AdamW +
#     cosine + epoch counter; LR re-warms 1e-3 -> brief jolt, re-converges).
#   - NEW --out / --logdir / log: keeps the old-aug ep01-10 snapshots intact as the
#     baseline (do NOT reuse the old --out, the resume's ep01 would overwrite them).
#   - new aug is automatic — it's the merged default _augment_crop (no flag).
#   - GGX/Fresnel specular is NOT exercised here (needs a re-rendered glossy loaf).
#   - in-loop eval is tiled center-crop (honest curves: val |dv| tracks |du|).
#
# NOTE on the comparison: this forfeits a clean old-vs-new aug A/B (codebook formed
# under old aug); it's a valid "trained-with-augmentation" net, not an isolation of
# the new aug's marginal benefit. Deliberate call — see chat.
set -euo pipefail
cd /run/media/nshelton/LUX/lux

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

nohup systemd-inhibit --what=idle:sleep:handle-lid-switch --who=lux-train --why="conv newaug" \
  ~/.venvs/lux/bin/python scripts/train_proj_net.py \
  --loaf ~/datasets/val_loaf ~/datasets/planar_loaf \
  --mid conv --epochs 20 --batch 32 --amp --workers 4 --val 4 \
  --lr 1e-3 --lr-min 1e-4 --offset-weight 2 --gate-offset 6 \
  --resume checkpoints/proj_net_conv_aug_ep10.pt \
  --snapshots \
  --out checkpoints/proj_net_conv_newaug.pt \
  --logdir runs/proj_net_conv_newaug \
  >> checkpoints/train_conv_newaug.log 2>&1 &

echo "launched proj_net_conv_newaug (PID $!) — resumed ep10 weights, NEW physically-ordered aug"
echo "watch:  tail -f checkpoints/train_conv_newaug.log"
echo "stop:   kill $!"
