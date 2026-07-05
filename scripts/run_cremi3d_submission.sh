#!/usr/bin/env bash
# Full CREMI challenge submission run: A+, B+, C+ (padded test volumes),
# two-phase blockwise inference via scripts/infer_submission.py:
#   Phase A -- Gaussian-weighted blend of raw sem+aff logits over
#              heavily-overlapping WINDOW_SIZE windows (full network field
#              of view, 400x256x256 @ 4nm by default) advancing by
#              STRIDE_FRAC * window (0.25 -> 75% overlap between windows).
#   Phase B -- Mutex Watershed once per FINE_CORE_SIZE/FINE_CONTEXT chunk of
#              the already-blended field (no network inference here).
# See infer_submission.py's module docstring for the full design and the
# cross-chunk-merge limitation.
#
# DISK COST: CREMI's padded fine grid is ~2000x3072x3072 -- the Phase A
# accumulator at the defaults below is on the order of 2+ TB of scratch disk
# PER SAMPLE (independent of STRIDE_FRAC; only the I/O volume against it
# scales with STRIDE_FRAC/window count). Run with --dry-run first (copy the
# python invocation below and add --dry-run) to see the exact numbers for
# your hardware, and raise STRIDE_FRAC / shrink WINDOW_SIZE if that's
# impractical for your scratch space.
#
# Edit CKPT below to point at whichever checkpoint you want to submit with
# (the freshest available -- see doc/CURRENT_STATE.md / the training run's
# outputs/<run>/checkpoints/ directory).
#
# Usage:
#   ./scripts/run_cremi3d_submission.sh
#   CKPT=path/to/other.ckpt ./scripts/run_cremi3d_submission.sh   # override
set -euo pipefail

CKPT="${CKPT:-outputs/2026-07-01_15-31-16_nanocosmos-2B/checkpoints/crash_recovery.ckpt}"
CONFIG_NAME="${CONFIG_NAME:-nanocosmos-2B}"
DATA_ROOT="${DATA_ROOT:-data/CREMI3D}"
OUT_ROOT="${OUT_ROOT:-outputs/submission}"
NATIVE_RES="40 4 4"   # CREMI: z y x nm

# Phase A: full network field of view (400x256x256 @ 4nm), 1/4-stride
# (75%) overlap between neighbouring windows, forwarded through the network
# BLEND_BATCH_SIZE windows at a time (keeps the GPU busy instead of one
# window at a time) with BLEND_IO_WORKERS threads reading/resampling
# windows in the background (overlaps disk I/O with GPU compute).
#
# Set GPU_IDS to scale across multiple GPUs (e.g. GPU_IDS="0 1 2 3"); leave
# unset for the original single --device behaviour. WORKERS_PER_GPU > 1
# runs that many concurrent batches per GPU, EACH WITH ITS OWN MODEL
# REPLICA (required -- see infer_submission.py docstring for why sharing
# one instance across threads corrupts results), so it costs that many x
# the model's GPU memory per GPU it applies to.
WINDOW_SIZE="${WINDOW_SIZE:-400 256 256}"
STRIDE_FRAC="${STRIDE_FRAC:-0.25}"
BLEND_BATCH_SIZE="${BLEND_BATCH_SIZE:-4}"
BLEND_IO_WORKERS="${BLEND_IO_WORKERS:-4}"
GPU_IDS="${GPU_IDS:-}"
WORKERS_PER_GPU="${WORKERS_PER_GPU:-1}"

# Phase B (Mutex Watershed on the blended field -- no network inference here,
# so GPU pressure is just MWS scratch, not model activations too). Shrink
# FINE_CORE_SIZE/FINE_CONTEXT and/or force MWS onto the CPU (mws_np -- the
# exact reference impl, just slower; CPU RAM is abundant) if MWS itself OOMs.
# MWS_WORKERS chunks run concurrently; the CPU backend (mws_np) is numba
# nogil=True, so this gives real multi-core speedup there (GPU backend
# mostly overlaps I/O since MWS itself still runs on one device).
FINE_CORE_SIZE="${FINE_CORE_SIZE:-600 384 384}"
FINE_CONTEXT="${FINE_CONTEXT:-100 64 64}"
MWS_BACKEND="${MWS_BACKEND:-cpu}"
MWS_WORKERS="${MWS_WORKERS:-4}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

if [ ! -f "${CKPT}" ]; then
  echo "Checkpoint not found: ${CKPT}" >&2
  echo "Pass CKPT=<path> ./scripts/run_cremi3d_submission.sh to point at a different one." >&2
  exit 1
fi

echo "Checkpoint: ${CKPT}"
echo "Config:     ${CONFIG_NAME}"
echo "Data root:  ${DATA_ROOT}"
echo "Output:     ${OUT_ROOT}"
echo

for s in A B C; do
  vol="cremi3d_sample_${s}+_padded_volume"
  out_dir="${OUT_ROOT}/cremi_${s}+"
  echo "=================================================================="
  echo "=== CREMI sample ${s}+  ->  ${out_dir}"
  echo "=================================================================="
  GPU_ARGS=()
  if [ -n "${GPU_IDS}" ]; then
    GPU_ARGS=(--gpu-ids ${GPU_IDS})
  fi

  python scripts/infer_submission.py \
    --config-name "${CONFIG_NAME}" \
    --ckpt "${CKPT}" \
    --vol "${vol}" --root "${DATA_ROOT}" \
    --native-resolution ${NATIVE_RES} \
    --window-size ${WINDOW_SIZE} \
    --stride-frac ${STRIDE_FRAC} \
    --blend-batch-size ${BLEND_BATCH_SIZE} \
    --blend-io-workers ${BLEND_IO_WORKERS} \
    --workers-per-gpu ${WORKERS_PER_GPU} \
    "${GPU_ARGS[@]}" \
    --fine-core-size ${FINE_CORE_SIZE} \
    --fine-context ${FINE_CONTEXT} \
    --mws-workers ${MWS_WORKERS} \
    --overrides "training.mutex_watershed.backend=${MWS_BACKEND}" \
    --out-dir "${out_dir}"
  echo
done

echo "Done. Submission files:"
find "${OUT_ROOT}" -maxdepth 2 -iname "*_submission.hdf" -exec ls -la {} \;
