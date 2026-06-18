#!/usr/bin/env bash
# Production training for the codesign_v2 continuous-phase (quad) decoder on the CUDA box.
#
# Trains on the clutter_v2 + planar_v2 loaves (ConcatLoaf ~50/50 by size) then runs the two eval
# tables. Full context + the open enhancements (edge-weighted sampling, per-band peak_margin
# calibration) are in docs/codesign_v2_handoff.md.
#
# REQUIRES on the box (NOT in git — transfer separately, e.g. rsync):
#   - the two loaves          (stage on NVMe, NOT the exFAT repo drive — random crops over
#                              exFAT-over-USB halve throughput; see CLAUDE.md)
#   - evals/hemisphere/data   (160 planar poses w/ codesign_v2 captures)  -> Table 1
#   - evals/clutter_v2        (160 cluttered scenes, keeps gt_depth)      -> Table 2
#
# Override any path via env, e.g.:
#   CLUTTER=~/datasets/clutter_v2 PLANAR=~/datasets/planar_v2 EPOCHS=12 bash scripts/train_prod_cuda.sh
set -euo pipefail
cd "$(dirname "$0")/.."

PY=${PY:-$HOME/.venvs/lux/bin/python}            # venv lives on NVMe (exFAT can't host a venv)
CLUTTER=${CLUTTER:-$HOME/datasets/clutter_v2}    # loaves staged on NVMe
PLANAR=${PLANAR:-$HOME/datasets/planar_v2}
OUT=${OUT:-checkpoints/codesign_quad_prod.pt}
EPOCHS=${EPOCHS:-12}
BATCH=${BATCH:-32}                               # 11 GB card ceiling ~32 at crop 256
WORKERS=${WORKERS:-4}

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# heat/stability tips (CLAUDE.md): `sudo nvidia-smi -pl 140` caps power; wrap the run in
# `systemd-inhibit --what=idle:sleep` to avoid a suspend-resume GPU failure mid-train.

echo "== train (ConcatLoaf: $CLUTTER + $PLANAR) -> $OUT | $EPOCHS ep, batch $BATCH =="
"$PY" scripts/train_quad_rendered.py --loaf "$CLUTTER" "$PLANAR" \
    --epochs "$EPOCHS" --batch "$BATCH" --crops-per-sample 8 --workers "$WORKERS" \
    --out "$OUT" --snapshots

echo "== Table 1: plane obliquity sweep (160 hemisphere poses) =="
"$PY" scripts/eval_hemisphere_quad.py --ckpt "$OUT" \
    --data evals/hemisphere/data --pattern-set codesign_v2 --device cuda

echo "== Table 2: clutter edge risk-coverage (distance-to-discontinuity) =="
"$PY" scripts/eval_clutter_quad.py --ckpt "$OUT" \
    --data evals/clutter_v2 --pattern-set codesign_v2 --device cuda

echo "== done -> $OUT  (per-epoch snapshots: ${OUT%.pt}_ep*.pt) =="
