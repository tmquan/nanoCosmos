# Handover: CREMI3D + SNEMI3D submission inference (OOM on GB300)

**Status as of 2026-07-06:** a run of the full-volume submission inference (`scripts/infer_submission.py`, via `scripts/run_cremi3d_submission.sh` / `scripts/run_snemi3d_submission.sh`) on the GB300 server hit an out-of-memory condition and did not complete. Whoever picks this up needs to **diagnose which kind of OOM it was first** (they have different fixes), **clean up any stray large files** the crashed run left behind, then resume. This document is a self-contained runbook — read it fully before touching anything.

---

## 1. Goal

Produce final challenge-submission instance segmentations for:
- **CREMI3D**: samples A+, B+, C+ (padded test volumes) → `*_submission.hdf` (CREMI `volumes/labels/neuron_ids` format).
- **SNEMI3D**: test (non-padded test volume) → `*_submission.zip` (containing `test-input.h5`, dataset `main`).

Both go through the same two-phase pipeline in `scripts/infer_submission.py`:
- **Phase A** — Gaussian-weighted blending of the network's raw affinity+semantic+raw-reconstruction logits over heavily-overlapping windows across the whole volume, accumulated into an on-disk HDF5 file (this is the GPU-heavy, network-forward-pass phase).
- **Phase B** — Mutex Watershed instance agglomeration on the blended field, chunked, writing the final native-resolution segmentation (+ fine-grid raw/sem/label diagnostics if enabled).

**Read `scripts/infer_submission.py`'s module docstring first** — it is the authoritative, up-to-date design reference (two-phase design, disk/IO cost, multi-GPU scaling, known cross-chunk-merge limitation). This document only covers OOM-specific recovery; don't duplicate-maintain the design details here.

---

## 2. Step 1 — Diagnose which OOM actually happened

There are **three distinct OOM types** possible here, with different symptoms and different fixes. Do not guess — find the actual crash log/stack trace first.

```bash
# Find where the run's stdout/stderr went (tmux/screen scrollback, nohup.out,
# a redirected log file, or the shell's own history). Also check:
dmesg -T | tail -100 | grep -i -E "out of memory|oom|killed process"
journalctl -k --since "-2 hours" | grep -i oom          # if systemd/journald available
df -h                                                    # disk usage on the output/scratch filesystem
nvidia-smi                                               # any zombie processes still holding GPU memory?
free -h                                                  # host RAM headroom
```

**(a) CUDA OOM** (`torch.OutOfMemoryError` / `RuntimeError: CUDA out of memory` in the log). Caused by Phase A's GPU-resident model replicas/batches. See §4a.

**(b) Disk full (`OSError: [Errno 28] No space left on device` or the h5py write silently failing)**. Almost certainly the Phase A blend accumulator or the `--save-fine-grid` diagnostics — see §3 (these are *huge*, especially for CREMI) — or a leftover accumulator from an earlier crash still sitting on disk consuming space. See §4b.

**(c) Host RAM OOM / process killed by the kernel OOM-killer** (exit code 137, `dmesg` shows `Killed process`, no Python traceback at all). Likely candidates: too many `--blend-io-workers` / `--workers-per-gpu` replicas, or the final submission-packaging step which reads the **entire** native-resolution segmentation into host RAM at once (`data = fsrc["main"][:]` in `infer_submission()` — for CREMI's ~200×3072×3072 int64 volume that's ~38 GB in one array; not usually fatal on a big node, but worth knowing about). See §4c.

---

## 3. IMPORTANT correction — disk cost is NOT what the earlier comments implied

An earlier pass at this script's comments suggested "raise `--stride-frac` / shrink `--window-size` if the accumulator is too big for your disk." **This is wrong and has been corrected in the code/comments as of this handover.** Verified empirically (`--dry-run` at `stride-frac` 0.25/0.5/1.0 all print the *identical* accumulator size for the same volume):

> The Phase A accumulator's size is **fixed** by `(real fine-grid shape) × (N_AFF+2 channels) × 4 bytes` — essentially the volume's native size/resolution and the model's channel count. `--window-size`/`--stride-frac` only change how many *overlapping windows* are processed (i.e. runtime and I/O volume against that fixed-size file), not the file's size on disk.

Concretely, at current defaults:
- **SNEMI3D test**: ~240 GB accumulator + ~7.5 GB weight map + (if `--save-fine-grid`, default on) ~28 GB diagnostics ≈ **~275 GB peak**, one sample.
- **CREMI3D** (each of A+/B+/C+): ~2+ TB accumulator + (if `--save-fine-grid`) ~300 GB diagnostics ≈ **~2.3+ TB peak PER SAMPLE**. The wrapper script processes A/B/C sequentially and deletes the accumulator after each sample succeeds, so peak usage should be ~1 sample's worth *if nothing crashed* — but see below.

**There is currently no CLI knob to shrink the accumulator itself.** The only real levers today:
- `--no-save-fine-grid` (or `SAVE_FINE_GRID=false` in the wrapper scripts) — cuts the ~28–300 GB diagnostic files, not the accumulator.
- Get more scratch disk, or point `--out-dir` at a filesystem with more room.
- (Not implemented, candidate follow-up if disk remains the blocker): blend in float16 instead of float32 (~halves the accumulator), or make Phase A skip the raw channel when `--no-save-fine-grid` is set (~3% smaller, marginal).

**Before resuming, always run the exact same command with `--dry-run` appended and compare the printed disk estimate against `df -h` output for the target filesystem.**

---

## 4. Cleanup + fixes by OOM type

### First, in all cases: check for and remove stray leftover files

If the crash happened **during or after Phase A**, the accumulator file (`<out_dir>/<vol>_blend_acc.h5`, hundreds of GB to multiple TB) is **only deleted on successful completion** of the whole run (or if `--keep-blend-cache` was passed, never). A crash leaves it sitting on disk, silently consuming the space you need for the retry.

```bash
# List any leftover accumulator/output files and their sizes:
find outputs/submission -iname "*_blend_acc.h5" -exec ls -lh {} \;
find outputs/submission -iname "*_pred_*_full.h5" -o -iname "*_pred_*_fine.h5" -exec ls -lh {} \;
```

**Do NOT blindly delete these** — if Phase A actually completed before the crash, that `_blend_acc.h5` is exactly what `--reuse-blend-cache` (see §5, added as part of this handover) can reuse to skip re-running the most expensive part. Only delete it if it's confirmed incomplete/corrupt (e.g. `--reuse-blend-cache` reports "missing/incompatible" when you try it) or if you've decided to change `--window-size`/`--stride-frac`/`--gpu-ids` in a way that changes the block plan (the shapes won't match and it'll be rejected + recomputed automatically anyway — see §5).

### 4a. CUDA OOM

Phase A's GPU memory scales with: `--blend-batch-size` (linear) × `--workers-per-gpu` (each worker is a **full separate model replica**, not a shared one — see the "IMPORTANT" note in the module docstring about why sharing one instance across threads corrupts results) × number of `--gpu-ids`.

Fix, in order of preference:
1. Reduce `--workers-per-gpu` to `1` first (each extra worker costs a full 2B-param model's worth of GPU memory *per GPU*, on top of N× the activation memory).
2. Reduce `--blend-batch-size` (env var `BLEND_BATCH_SIZE`, currently defaults to `6` in both wrapper scripts — try `2` or `1`).
3. Confirm `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is set (both wrapper scripts already export this).
4. If using `--gpu-ids` with multiple GPUs, confirm no other process is already resident on those GPUs (`nvidia-smi`) — a previous crashed run's process may not have released GPU memory (zombie process holding the CUDA context: find it with `nvidia-smi` and `kill -9 <pid>` if it's actually dead/orphaned).
5. Phase B's Mutex Watershed can also OOM the GPU if `training.mutex_watershed.backend` resolves to a GPU backend — both wrapper scripts already force `MWS_BACKEND=cpu` by default (the exact `mws_np` reference impl; slower but CPU RAM is comparatively abundant). Confirm this wasn't overridden.

### 4b. Disk full

1. Do the cleanup in §4 first (stray accumulator files from earlier crashes).
2. Re-run with `--dry-run` and compare against `df -h` on the target filesystem — see §3 for realistic numbers per dataset.
3. If genuinely short on space: set `SAVE_FINE_GRID=false` (env var, both wrapper scripts) to skip the fine-grid diagnostics (only helps by tens to a few hundred GB, not the dominant cost).
4. If the accumulator itself doesn't fit: point `--out-dir` (or `OUT_ROOT`/`OUT_DIR` env vars) at a larger scratch volume, or free up space elsewhere on that filesystem. There is no algorithmic fix available today (see §3's "candidate follow-up" notes if this becomes a hard blocker).
5. For CREMI specifically: confirm the wrapper script is processing A+/B+/C+ **sequentially** (it is, by default, in a `for s in A B C` loop) so you never need >1 sample's worth of accumulator space at once — but if a previous run crashed mid-loop, an earlier sample's directory might still hold its now-orphaned accumulator; check all three `outputs/submission/cremi_{A,B,C}+/` directories, not just the one that was in progress when it crashed.

### 4c. Host RAM OOM / killed process

1. Reduce `--blend-io-workers` (env var `BLEND_IO_WORKERS`, defaults to `6`) and/or `--mws-workers` (`MWS_WORKERS`, defaults to `6`) — each is a thread, not a process, but each holds its own working buffers (a full Phase A window, or a full Phase B core+context chunk, in host RAM while processing).
2. Reduce `--fine-core-size`/`--fine-context` (`FINE_CORE_SIZE`/`FINE_CONTEXT` env vars, default `600 384 384` / `100 64 64`) — each Phase B chunk's buffer is `n_fields × (core+context)³` float32 in host RAM, and up to `--mws-workers` of these exist concurrently.
3. If it died in the final submission-packaging step (right after "Full native-resolution segmentation (...) total ids" prints, before "Submission file written"): that step currently reads the *entire* native-resolution output array into RAM at once (`fsrc["main"][:]`). For CREMI this is ~38 GB in one shot — not usually fatal, but if the node is memory-constrained or shared with other jobs, this is a plausible culprit. No CLI workaround exists today; would need a small code change to stream that read in chunks if it recurs (flag as a follow-up, not urgent).

---

## 5. Resuming — use `--reuse-blend-cache`

Added as part of this handover specifically for this scenario. If a compatible `<vol>_blend_acc.h5` already exists in `--out-dir` (e.g. Phase A completed but the crash happened in Phase B or the final packaging step), passing `--reuse-blend-cache` (or `REUSE_BLEND_CACHE=true` in the wrapper scripts) will:
- Check the existing file's `acc`/`weight` dataset shapes against what *this* invocation's parameters would produce.
- If they match exactly → **skip Phase A entirely** and go straight to Phase B (saves potentially hours of GPU-heavy compute).
- If they don't match (or the file is missing/corrupt) → print a warning and recompute Phase A from scratch, exactly as if the flag hadn't been passed. **Safe to always pass** — it's a no-op when there's nothing usable to reuse.

**Caveat:** the shape check only verifies dimensional compatibility, not that Phase A actually finished writing every window (if it crashed *mid*-Phase-A, the file will have the right shape but incomplete/partial data in some regions — this would silently produce a wrong segmentation, not an error). Only rely on `--reuse-blend-cache` if you're confident Phase A's log showed all windows completed (search the crash log for `[blend] window <last>/<total>` matching the total window count printed near the top of the run) before the crash happened in Phase B or later.

Verified this feature works correctly via a smoke test (synthetic volume, real 2B model architecture): second run's log shows `Phase A: SKIPPED -- reusing existing compatible blend cache`, and Phase B/submission packaging completed identically to a from-scratch run.

---

## 6. Recommended conservative resume commands

Start conservative, verify with `--dry-run`, then scale back up once you've confirmed the actual OOM cause and headroom on this specific node. Do **not** just re-run the exact command that OOM'd — you don't yet know which resource was exhausted.

**SNEMI3D test:**
```bash
cd /localhome/local-tranminhq/nanoCosmos
CKPT=<path to your actual checkpoint>              # see §7 — verify this path exists on THIS server
BLEND_BATCH_SIZE=2 WORKERS_PER_GPU=1 GPU_IDS="" \
BLEND_IO_WORKERS=4 MWS_WORKERS=4 \
REUSE_BLEND_CACHE=true \
CKPT="${CKPT}" ./scripts/run_snemi3d_submission.sh
```

**CREMI3D (A+/B+/C+):**
```bash
cd /localhome/local-tranminhq/nanoCosmos
CKPT=<path to your actual checkpoint>
BLEND_BATCH_SIZE=2 WORKERS_PER_GPU=1 GPU_IDS="" \
BLEND_IO_WORKERS=4 MWS_WORKERS=4 \
REUSE_BLEND_CACHE=true \
CKPT="${CKPT}" ./scripts/run_cremi3d_submission.sh
```

Before either, **always** dry-run first with the exact same env vars by copying the `python scripts/infer_submission.py ...` invocation the wrapper script would run and appending `--dry-run` (or temporarily add `set -x` to the wrapper script to see the exact invocation, then re-run that line by hand with `--dry-run`). Confirm the printed disk estimate fits on the target filesystem (`df -h`) before letting it proceed for real.

Once conservative settings are confirmed working, scale `--blend-batch-size`/`--workers-per-gpu`/`--gpu-ids` back up incrementally to use the GB300's actual GPU capacity efficiently — don't leave it artificially throttled if the real bottleneck turns out to be disk, not GPU memory.

---

## 7. Things to verify on the new server before running anything

- **Checkpoint path.** Both wrapper scripts default `CKPT` to `outputs/2026-07-01_15-31-16_nanocosmos-2B/checkpoints/crash_recovery.ckpt` — this is almost certainly **stale/wrong** on a different server (it was the training run's local path at some earlier point in time). Find the actual latest checkpoint on this machine:
  ```bash
  find outputs -iname "*.ckpt" -newer outputs -printf "%T@ %p\n" 2>/dev/null | sort -rn | head -20
  ```
  and pass it explicitly via `CKPT=<real path>`.
- **Data availability.** Confirm `data/SNEMI3D/test_inputs.h5` and `data/CREMI3D/cremi3d_sample_{A,B,C}+_padded_volume.h5` exist and are the *padded* CREMI downloads (not the cropped ones) — see `scripts/download_cremi3d.py --padded` and `doc/DATASETS.md` if they need re-downloading.
- **`numba` availability.** The Phase B CPU Mutex Watershed backend (`mws_np`) is numba-JIT'd with `nogil=True` for real multi-core parallelism across `--mws-workers` — confirm `python -c "import numba"` succeeds on this server; if numba is missing, Phase B silently falls back to slow, non-parallel plain Python (functionally correct, just much slower — was actually observed missing in the dev sandbox used to build this feature, so don't assume it's present without checking).
- **GPU count/ids.** `nvidia-smi -L` to see what's actually available before setting `GPU_IDS`.

---

## 8. Success criteria

- **SNEMI3D**: `outputs/submission/snemi3d_test/test_inputs_submission.zip` exists and contains exactly one `test-input.h5` (dataset `main`, uint32, same shape as `test_inputs.h5`'s native shape).
- **CREMI3D**: `outputs/submission/cremi_{A,B,C}+/*_submission.hdf` exist, each with dataset `volumes/labels/neuron_ids` (uint64) at the *official* (unpadded) CREMI test region shape, plus a `resolution` attribute.
- Both scripts print a final `Submission file written: ... (N unique ids incl. background)` line with a plausible id count (thousands, not 1 or millions) — a suspiciously low count (~1-2) suggests the semantic/affinity gating collapsed everything to background, worth sanity-checking against the checkpoint quality before trusting the submission.
- Sanity-check with the `--save-fine-grid` diagnostics (`pred_sem_fine.h5`, `pred_raw_fine.h5`) if disk allows — visually or statistically confirm the semantic probability map isn't degenerate (all-0 or all-1) before spending the CREMI compute budget on all three samples.

---

## 9. Reference — all `infer_submission.py` CLI flags relevant to this recovery

| Flag | Wrapper env var | Default | Relevant to |
|---|---|---|---|
| `--blend-batch-size` | `BLEND_BATCH_SIZE` | 6 | GPU memory (§4a) |
| `--blend-io-workers` | `BLEND_IO_WORKERS` | 6 | Host RAM (§4c) |
| `--workers-per-gpu` | `WORKERS_PER_GPU` | 1 | GPU memory, multiplies model replicas (§4a) |
| `--gpu-ids` | `GPU_IDS` | unset (single `--device`) | GPU memory / scaling |
| `--fine-core-size` / `--fine-context` | `FINE_CORE_SIZE` / `FINE_CONTEXT` | `800 512 512` / `200 128 128` (script defaults: `600 384 384` / `100 64 64`) | Phase B host RAM / GPU (if MWS backend is GPU) |
| `--mws-workers` | `MWS_WORKERS` | 6 | Host RAM (§4c), CPU parallelism |
| `--save-fine-grid` / `--no-save-fine-grid` | `SAVE_FINE_GRID` | on | Disk (§3, §4b) |
| `--reuse-blend-cache` | `REUSE_BLEND_CACHE` | off | Recovery (§5) — new in this handover |
| `--keep-blend-cache` | n/a (not wired to an env var) | off | Keeps the accumulator after a *successful* run instead of deleting it (useful if you want to deliberately preserve it for a future `--reuse-blend-cache`, e.g. to experiment with different Phase B chunking without redoing Phase A) |
| `--dry-run` | n/a | off | Always run this first after any parameter change |

Full authoritative docs: `scripts/infer_submission.py` module docstring (design, disk cost, multi-GPU scaling, known limitations) and `--help`.
