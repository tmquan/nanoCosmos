#!/usr/bin/env bash
# SNEMI3D AC3 -- high-quality overnight run (option A):
#   * Regenerate Phase A blend accumulator (network forward) and KEEP it
#     (--keep-blend-cache) so affinity-based agglomeration can run afterwards.
#   * Phase B = EXACT CPU Mutex Watershed (mws_np), LARGE chunks -> no
#     GPU-approximation blockiness and few chunk seams.
# Phase A runs on the GB300 (cuda:0 in the nanocosmos env); Phase B on CPU.
# Expect ~12-20 h total (exact fine-grid MWS is heavy). Logs to logs/.
set -euo pipefail
cd /localhome/local-tranminhq/nanocosmos

CKPT="outputs/2026-07-04_05-23-24_nanocosmos-2B/checkpoints/crash_recovery.ckpt"  # latest (mtime 2026-07-06 12:07)
OUT_DIR="outputs/submission/snemi3d_AC3"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

mkdir -p logs
LOG="logs/snemi3d_cpu_overnight_$(date +%Y%m%d_%H%M%S).log"
echo "Logging to ${LOG}"

python scripts/infer_submission.py \
  --config-name nanocosmos-2B \
  --ckpt "${CKPT}" \
  --vol AC3_inputs --root data/SNEMI3D \
  --native-resolution 30 6 6 \
  --window-size 400 256 256 \
  --stride-frac 0.25 \
  --blend-batch-size 6 --blend-io-workers 6 --workers-per-gpu 1 \
  --fine-core-size 600 384 384 --fine-context 100 64 64 \
  --mws-workers 2 \
  --save-fine-grid \
  --keep-blend-cache \
  --submission-format snemi3d \
  --overrides training.mutex_watershed.backend=cpu \
  --out-dir "${OUT_DIR}" 2>&1 | tee "${LOG}"

echo "DONE. Kept blend accumulator for agglomeration: ${OUT_DIR}/AC3_inputs_blend_acc.h5"
