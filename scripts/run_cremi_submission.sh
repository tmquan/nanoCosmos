#!/usr/bin/env bash
# Full CREMI challenge submission run: A+, B+, C+ (padded test volumes),
# chunked blockwise inference via scripts/infer_cremi_submission.py.
#
# Edit CKPT below to point at whichever checkpoint you want to submit with
# (the freshest available -- see doc/CURRENT_STATE.md / the training run's
# outputs/<run>/checkpoints/ directory).
#
# Usage:
#   ./scripts/run_cremi_submission.sh
#   CKPT=path/to/other.ckpt ./scripts/run_cremi_submission.sh   # override
#
# Each sample plans to ~256 blocks at the memory-safe defaults below
# (FINE_CORE_SIZE/FINE_CONTEXT), peak accumulator ~26.8 GB/block. Run
# --dry-run first on new hardware to sanity-check the block plan (see
# scripts/infer_cremi_submission.py docstring) -- a dry-run only prints the
# plan, it does not reproduce the MWS/activation memory cost, so a real OOM
# is still possible; shrink FINE_CORE_SIZE/FINE_CONTEXT further if it recurs.
set -euo pipefail

CKPT="${CKPT:-outputs/2026-07-01_15-31-16_nanocosmos-2B/checkpoints/crash_recovery.ckpt}"
CONFIG_NAME="${CONFIG_NAME:-nanocosmos-2B}"
DATA_ROOT="${DATA_ROOT:-data/CREMI3D}"
OUT_ROOT="${OUT_ROOT:-outputs/submission}"
NATIVE_RES="40 4 4"   # CREMI: z y x nm

# Memory-safe defaults (a full-size 800x512x512 core / 200x128x128 context
# block OOM'd at ~274.6/276.5 GB -- MWS's own GPU scratch scales with the
# block's total voxel count, on top of the sliding-window accumulator and the
# 2B model's un-checkpointed eval-time activations). Shrinking the block AND
# forcing Mutex Watershed onto the CPU (mws_np -- the exact reference impl,
# just slower; CPU RAM is abundant) both reduce GPU pressure independently --
# override either via env var if your GPU has more headroom.
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
  python scripts/infer_cremi_submission.py \
    --config-name "${CONFIG_NAME}" \
    --ckpt "${CKPT}" \
    --vol "${vol}" --root "${DATA_ROOT}" \
    --native-resolution ${NATIVE_RES} \
    --fine-core-size ${FINE_CORE_SIZE} \
    --fine-context ${FINE_CONTEXT} \
    --overrides "training.mutex_watershed.backend=${MWS_BACKEND}" \
    --out-dir "${out_dir}"
  echo
done

echo "Done. Submission files:"
find "${OUT_ROOT}" -maxdepth 2 -iname "*_submission.hdf" -exec ls -la {} \;
