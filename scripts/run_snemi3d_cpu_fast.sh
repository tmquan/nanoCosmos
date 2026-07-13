#!/usr/bin/env bash
# SNEMI3D test -- fast exact-CPU Phase B, reusing an already-complete Phase A
# blend accumulator (--reuse-blend-cache skips the network-forward pass
# entirely). Uses SMALL Mutex Watershed chunks (no GPU-sort INT_MAX limit on
# the CPU backend, unlike mws_th) so many more of the host's cores fit within
# RAM at once -- throughput scales with MWS_WORKERS instead of being capped
# at ~2 by oversized chunks. Chunk-boundary over-segmentation from the small
# chunks is expected to be cleaned up afterwards by
# scripts/stitch_affinity_rag.py (global affinity-weighted fragment merge),
# NOT by making chunks big here.
set -euo pipefail
cd /localhome/local-tranminhq/nanocosmos

CKPT="outputs/2026-07-04_05-23-24_nanocosmos-2B/checkpoints/crash_recovery.ckpt"
OUT_DIR="outputs/submission/snemi3d_test"
MWS_WORKERS="${MWS_WORKERS:-16}"
FINE_CORE_SIZE="${FINE_CORE_SIZE:-160 256 256}"
FINE_CONTEXT="${FINE_CONTEXT:-40 48 48}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

mkdir -p logs
LOG="logs/snemi3d_cpu_fast_$(date +%Y%m%d_%H%M%S).log"
echo "Logging to ${LOG}"
echo "MWS_WORKERS=${MWS_WORKERS}  FINE_CORE_SIZE=${FINE_CORE_SIZE}  FINE_CONTEXT=${FINE_CONTEXT}"

python scripts/infer_submission.py \
  --config-name nanocosmos-2B \
  --ckpt "${CKPT}" \
  --vol test_inputs --root data/SNEMI3D \
  --native-resolution 30 6 6 \
  --window-size 400 256 256 \
  --stride-frac 0.25 \
  --blend-batch-size 6 --blend-io-workers 6 --workers-per-gpu 1 \
  --fine-core-size ${FINE_CORE_SIZE} --fine-context ${FINE_CONTEXT} \
  --mws-workers "${MWS_WORKERS}" \
  --save-fine-grid \
  --keep-blend-cache \
  --reuse-blend-cache \
  --submission-format snemi3d \
  --overrides training.mutex_watershed.backend=cpu \
  --out-dir "${OUT_DIR}" 2>&1 | tee "${LOG}"

echo "DONE."
