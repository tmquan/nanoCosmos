#!/usr/bin/env bash
# Full SNEMI3D challenge submission run: AC3 (non-padded test volume),
# chunked blockwise inference via scripts/infer_cremi_submission.py.
#
# Output format is SNEMI3D's own (auto-detected, since AC3 carries no
# cropped_region_* attrs): a ZIP containing a single test-input.h5 with
# dataset 'main' -- see https://snemi3d.grand-challenge.org/. This is a
# DIFFERENT format from CREMI's volumes/labels/neuron_ids -- see
# scripts/run_cremi_submission.sh for that challenge instead.
#
# Edit CKPT below to point at whichever checkpoint you want to submit with.
#
# Usage:
#   ./scripts/run_snemi3d_submission.sh
#   CKPT=path/to/other.ckpt ./scripts/run_snemi3d_submission.sh   # override
#
# AC3 is much smaller than CREMI's padded test volumes (100x1024x1024
# native vs CREMI's 200x3072x3072), so this needs far fewer blocks -- still
# uses the same memory-safe block/context defaults and CPU Mutex Watershed
# as the CREMI script (see its comments for why).
set -euo pipefail

CKPT="${CKPT:-outputs/2026-07-01_15-31-16_nanocosmos-2B/checkpoints/crash_recovery.ckpt}"
CONFIG_NAME="${CONFIG_NAME:-nanocosmos-2B}"
DATA_ROOT="${DATA_ROOT:-data/SNEMI3D}"
OUT_DIR="${OUT_DIR:-outputs/submission/snemi3d_AC3}"
NATIVE_RES="30 6 6"   # SNEMI3D: z y x nm
VOL="AC3_inputs"

FINE_CORE_SIZE="${FINE_CORE_SIZE:-600 384 384}"
FINE_CONTEXT="${FINE_CONTEXT:-100 64 64}"
MWS_BACKEND="${MWS_BACKEND:-cpu}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

if [ ! -f "${CKPT}" ]; then
  echo "Checkpoint not found: ${CKPT}" >&2
  echo "Pass CKPT=<path> ./scripts/run_snemi3d_submission.sh to point at a different one." >&2
  exit 1
fi

echo "Checkpoint: ${CKPT}"
echo "Config:     ${CONFIG_NAME}"
echo "Volume:     ${VOL} (${DATA_ROOT})"
echo "Output:     ${OUT_DIR}"
echo

python scripts/infer_cremi_submission.py \
  --config-name "${CONFIG_NAME}" \
  --ckpt "${CKPT}" \
  --vol "${VOL}" --root "${DATA_ROOT}" \
  --native-resolution ${NATIVE_RES} \
  --fine-core-size ${FINE_CORE_SIZE} \
  --fine-context ${FINE_CONTEXT} \
  --submission-format snemi3d \
  --overrides "training.mutex_watershed.backend=${MWS_BACKEND}" \
  --out-dir "${OUT_DIR}"

echo
echo "Done. Submission file:"
find "${OUT_DIR}" -maxdepth 1 -iname "*_submission.zip" -exec ls -la {} \;
