#!/usr/bin/env bash
# Per-rank NUMA binding, tuned for a specific dual-socket B300 node (8 GPUs):
#   GPUs 0-3 -> NUMA node0 (CPUs 0-55,112-167)
#   GPUs 4-7 -> NUMA node1 (CPUs 56-111,168-223)
# (mapping from `nvidia-smi topo -m` + `lscpu`).
#
# NODE-SPECIFIC: the GPU->NUMA split below (LOCAL_RANK < 4) is hardware
# topology for THIS node. On a different machine re-derive it from
# `nvidia-smi topo -m` / `lscpu` and adjust the threshold.
#
# The recipe is overridable without editing this file:
#   NANOCOSMOS_CONFIG=nanocosmos-2B torchrun --standalone --nproc_per_node=8 \
#       --no-python ./numa_run.sh [extra hydra overrides...]
#
# Each rank reads its own LOCAL_RANK (set by torchrun) and binds its
# process + its DataLoader workers to the socket local to its GPU, so the
# higher-numbered ranks stop fetching data across the socket boundary.
#
# Launch with torchrun so every rank runs THIS wrapper (Lightning detects
# the torchrun env and uses the existing process group instead of spawning
# its own subprocesses):
#
#   torchrun --standalone --nproc_per_node=8 --no-python ./numa_run.sh
#
# Requires: numactl installed (check: `which numactl`).
set -euo pipefail

LR="${LOCAL_RANK:-0}"
if [ "${LR}" -lt 4 ]; then
  NODE=0
else
  NODE=1
fi

# Recipe: override via ``NANOCOSMOS_CONFIG`` (default cosmos3nano3d); any extra
# args are forwarded to train.py as Hydra overrides.
CONFIG_NAME="${NANOCOSMOS_CONFIG:-cosmos3nano3d}"

exec numactl --cpunodebind="${NODE}" --membind="${NODE}" \
  python scripts/train.py --config-name "${CONFIG_NAME}" "$@"
