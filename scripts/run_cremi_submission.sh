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
#   ./scripts/run_cremi_submission.sh
#   CKPT=path/to/other.ckpt ./scripts/run_cremi_submission.sh   # override
set -euo pipefail

CKPT="${CKPT:-outputs/2026-07-01_15-31-16_nanocosmos-2B/checkpoints/crash_recovery.ckpt}"
CONFIG_NAME="${CONFIG_NAME:-nanocosmos-2B}"
DATA_ROOT="${DATA_ROOT:-data/CREMI3D}"
OUT_ROOT="${OUT_ROOT:-outputs/submission}"
NATIVE_RES="40 4 4"   # CREMI: z y x nm

# Phase A: full network field of view (400x256x256 @ 4nm), 1/4-stride
# (75%) overlap between neighbouring windows.
WINDOW_SIZE="${WINDOW_SIZE:-400 256 256}"
STRIDE_FRAC="${STRIDE_FRAC:-0.25}"

# Phase B (Mutex Watershed on the blended field -- no network inference here,
# so GPU pressure is just MWS scratch, not model activations too). Shrink
# FINE_CORE_SIZE/FINE_CONTEXT and/or force MWS onto the CPU (mws_np -- the
# exact reference impl, just slower; CPU RAM is abundant) if MWS itself OOMs.
FINE_CORE_SIZE="${FINE_CORE_SIZE:-600 384 384}"
FINE_CONTEXT="${FINE_CONTEXT:-100 64 64}"
MWS_BACKEND="${MWS_BACKEND:-cpu}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

if [ ! -f "${CKPT}" ]; then
  echo "Checkpoint not found: ${CKPT}" >&2
  echo "Pass CKPT=<path> ./scripts/run_cremi_submission.sh to point at a different one." >&2
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
  python scripts/infer_submission.py \
    --config-name "${CONFIG_NAME}" \
    --ckpt "${CKPT}" \
    --vol "${vol}" --root "${DATA_ROOT}" \
    --native-resolution ${NATIVE_RES} \
    --window-size ${WINDOW_SIZE} \
    --stride-frac ${STRIDE_FRAC} \
    --fine-core-size ${FINE_CORE_SIZE} \
    --fine-context ${FINE_CONTEXT} \
    --overrides "training.mutex_watershed.backend=${MWS_BACKEND}" \
    --out-dir "${out_dir}"
  echo
done

echo "Done. Submission files:"
find "${OUT_ROOT}" -maxdepth 2 -iname "*_submission.hdf" -exec ls -la {} \;
