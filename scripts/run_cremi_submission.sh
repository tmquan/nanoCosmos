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
# Each sample plans to ~108 blocks (default --fine-core-size/--fine-context),
# peak accumulator ~90.6 GB/block -- needs a GPU with that much free. Run
# --dry-run first on new hardware to sanity-check the block plan (see
# scripts/infer_cremi_submission.py docstring).
set -euo pipefail

CKPT="${CKPT:-outputs/2026-07-01_15-31-16_nanocosmos-2B/checkpoints/crash_recovery.ckpt}"
CONFIG_NAME="${CONFIG_NAME:-nanocosmos-2B}"
DATA_ROOT="${DATA_ROOT:-data/CREMI3D}"
OUT_ROOT="${OUT_ROOT:-outputs/submission}"
NATIVE_RES="40 4 4"   # CREMI: z y x nm

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
    --out-dir "${out_dir}"
  echo
done

echo "Done. Submission files:"
find "${OUT_ROOT}" -maxdepth 2 -iname "*_submission.hdf" -exec ls -la {} \;
