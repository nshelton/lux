#!/usr/bin/env bash
# conv + new-aug — the documented follow-up #1 from docs/net2_plan.md, on the
# Linux/CUDA box (2080 Ti). This is the CLEAN CONTROL the net2 plan calls for:
# the proven FROM-SCRATCH conv recipe that produced the best model
# (proj_net_scratch.pt: val |du| 0.28px), now WITH the new _augment_crop
# (blur/contrast-clip/noise) — which is always-on in LoafSamples, so "conv +
# new-aug" is just running --mid conv with the current code.
#
# Deltas vs the macOS resume_train_scratch.sh:
#   - FROM SCRATCH (no --resume): proj_net_scratch.pt isn't on this box; the
#     from-scratch recipe is the proven one anyway.
#   - loaves read from NVMe (~/datasets/*), NOT the exFAT repo drive — random
#     crops over exFAT-over-USB halve throughput (see CLAUDE.md).
#   - --batch 32: 64 OOMs even for conv on the 11GB card (verified).
#   - systemd-inhibit: block idle/sleep (NVIDIA-on-Wayland resume = black screen).
#   - NaN tripwire (abort after 30 non-finite losses) is unconditional in the trainer.
#
# PICK THE CHECKPOINT POST-HOC on the hemisphere bench (per obliquity bin), NOT
# the auto *best* — aggregate val |du| mis-ranks obliquity. Sweep the _epNN snaps.
set -euo pipefail
cd /run/media/nshelton/LUX/lux

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

nohup systemd-inhibit --what=idle:sleep:handle-lid-switch --who=lux-train --why="conv+aug training" \
  ~/.venvs/lux/bin/python scripts/train_proj_net.py \
  --loaf ~/datasets/val_loaf ~/datasets/planar_loaf \
  --mid conv --epochs 30 --batch 32 --amp --workers 4 --val 4 \
  --lr 1e-3 --lr-min 1e-4 --offset-weight 2 --gate-offset 6 \
  --snapshots \
  --out checkpoints/proj_net_conv_aug.pt \
  --logdir runs/proj_net_conv_aug \
  >> checkpoints/train_conv_aug.log 2>&1 &

echo "launched proj_net_conv_aug (PID $!) — from scratch, --mid conv, both NVMe loaves + new aug"
echo "watch:  tail -f checkpoints/train_conv_aug.log"
echo "stop:   kill $!"
