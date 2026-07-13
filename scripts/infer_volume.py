#!/usr/bin/env python
"""Run trained-model inference on a raw (label-free) EM volume crop.

Generic, **dataset-agnostic** inference path: given a trained checkpoint and
any native-resolution EM ``.h5`` volume (CREMI test A+/B+/C+, SNEMI3D test, an
unlabeled COSEM/FLYEM/MitoEM crop, ...), this script:

1. Reads a native-resolution region from disk (``--origin`` / ``--size``).
2. Resamples it onto the network's fixed **fine grid** (trilinear; same
   ``native_res / fine_nm`` scaling ``ToFineGridd`` uses at train time -- see
   ``nanocosmos/transforms/fine_grid.py`` and
   ``nanocosmos.datamodules.joint3d._scaled``).
3. Builds the exact model architecture from a Hydra training config
   (:func:`scripts.train.build_module`) and loads the checkpoint weights.
4. Runs Gaussian-blended sliding-window inference
   (:func:`nanocosmos.inference.sliding_window.sliding_window_inference`) to
   get the full unified head (affinity + sem + raw) over the region.
5. Agglomerates instances with the SAME :class:`MutexWatershed` the model was
   validated with (``self.agglomerator``, built by ``build_module`` from
   ``training.mutex_watershed`` -- no separate construction needed).
   Affinities always run at the fine grid: the Gaussian-weighted sliding
   window already blended every overlapping tile's affinity logits into ONE
   continuous fine-grid field, and Mutex Watershed is run exactly **once**
   on that assembled result -- never per-tile (per-tile instance ids from
   neighbouring patches are not mutually consistent and cannot be stitched).
6. Resamples the resulting instance segmentation back down to the volume's
   **native** resolution (nearest-neighbour -- a discrete id map must never be
   trilinear/average-resampled) -- this is the artifact a CREMI / SNEMI3D
   benchmark actually scores, since their ground truth lives on the native
   grid, however finely the network itself inferred.
7. Saves the native-resolution instance segmentation as ``.h5`` (nanocosmos
   on-disk convention: key ``main``, axes ``[Z,Y,X]``) -- always produced --
   plus, when ``--save-fine-grid`` (default **true**), the extra fine-grid
   diagnostics: the super-resolved raw reconstruction, sem probability,
   fine-grid instance segmentation, and a central-slice PNG preview panel.

Because every dataset-specific bit (volume path, native resolution, region)
is a plain CLI argument, this is the SAME code path for CREMI test volumes,
SNEMI3D test, or anything else -- no per-dataset branching.

Examples
--------
    # CREMI padded test volume A+ (40 x 4 x 4 nm), a modest region:
    python scripts/infer_volume.py \\
        --config-name nanocosmos-2B \\
        --ckpt outputs/2026-07-01_15-31-16_nanocosmos-2B/checkpoints/crash_recovery.ckpt \\
        --vol cremi3d_sample_A+_volume --root data/CREMI3D \\
        --native-resolution 40 4 4 \\
        --origin 0 400 400 --size 100 256 256 \\
        --out-dir outputs/infer/cremi_A+

    # SNEMI3D test (30 x 6 x 6 nm) -- identical invocation, different dataset:
    python scripts/infer_volume.py \\
        --config-name nanocosmos-2B \\
        --ckpt <ckpt> \\
        --vol test_inputs --root data/SNEMI3D \\
        --native-resolution 30 6 6 \\
        --origin 0 384 384 --size 80 256 256 \\
        --out-dir outputs/infer/snemi3d_test

Note on region size: the network's fine grid is ``pixel_size`` (e.g. 4 nm)
regardless of the source's native resolution, so a native region is resampled
by ``native_res / fine_nm`` per axis -- e.g. CREMI's 40 nm z becomes 10x more
voxels at the fine grid. Keep ``--size`` modest (this defaults to one
network patch's worth) or inference will try to allocate a very large
fine-grid volume. Pass ``--size`` bigger than one patch to genuinely exercise
the sliding-window blending across multiple tiles.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence, Tuple

import h5py
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train import build_module  # noqa: E402  (local import after sys.path tweak)


# ----------------------------------------------------------------------
# Volume I/O (same on-disk convention as LazyVolDataset)
# ----------------------------------------------------------------------

def _get_h5_dataset(f: h5py.File):
    for k in ("main", "data", "raw", "volume", "image"):
        if k in f:
            return f[k]
    return f[list(f.keys())[0]]


def _read_region(
    path: Path, origin: Sequence[int], size: Sequence[int],
) -> np.ndarray:
    """Read a ``[Z,Y,X]`` region, clamped to the volume bounds."""
    with h5py.File(str(path), "r", locking=False) as f:
        ds = _get_h5_dataset(f)
        shape = ds.shape[-3:]
        slices = tuple(
            slice(max(0, o), min(shape[d], o + size[d]))
            for d, o in enumerate(origin)
        )
        return np.asarray(ds[slices], dtype=np.float32)


def _norm_range(path: Path, region: np.ndarray) -> Tuple[float, float]:
    """Per-volume [min, max] -- reuse the cached ``.norm.json`` sidecar
    (written by ``LazyVolDataset``) when present, else fall back to the
    region's own range."""
    import json

    norm_path = path.with_suffix(path.suffix + ".norm.json")
    if norm_path.exists():
        try:
            with open(norm_path) as fh:
                d = json.load(fh)
            return float(d["min"]), float(d["max"])
        except (OSError, ValueError, KeyError):
            pass
    return float(region.min()), float(region.max())


# ----------------------------------------------------------------------
# Fine-grid resampling (mirrors ToFineGridd / joint3d._scaled)
# ----------------------------------------------------------------------

def _fine_grid_shape(native_shape: Sequence[int], native_res: Sequence[float], fine_nm: float) -> Tuple[int, ...]:
    """Native [Z,Y,X] voxel count -> fine-grid voxel count: round(n * native_res / fine_nm)."""
    return tuple(
        max(1, int(round(native_shape[d] * float(native_res[d]) / fine_nm)))
        for d in range(3)
    )


def _to_fine_grid(image01: np.ndarray, native_res: Sequence[float], fine_nm: float) -> torch.Tensor:
    """``[Z,Y,X]`` normalised image -> ``[1,1,D,H,W]`` fine-grid tensor (trilinear)."""
    t = torch.from_numpy(image01)[None, None]
    fine_shape = _fine_grid_shape(t.shape[-3:], native_res, fine_nm)
    if tuple(t.shape[-3:]) == fine_shape:
        return t
    return F.interpolate(t, size=fine_shape, mode="trilinear", align_corners=False)


# ----------------------------------------------------------------------
# Saving
# ----------------------------------------------------------------------

def _save_h5(arr: np.ndarray, path: Path, source_note: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(str(path), "w") as f:
        ds = f.create_dataset("main", data=arr, compression="gzip", compression_opts=4)
        ds.attrs["source"] = source_note


def _save_preview(
    image01: torch.Tensor, raw: torch.Tensor, sem: torch.Tensor, seg: torch.Tensor, out_path: Path,
) -> None:
    """A quick central-slice PNG panel: input | raw recon | sem | instances."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from nanocosmos.callbacks.tensorboard.viz import _label_to_rgb, _normalise, _to_2d

    z = image01.shape[-3] // 2
    panels = {
        "input (fine grid)": _normalise(_to_2d(image01)[:1])[0, 0].cpu().numpy(),
        "pred/raw (recon)": ((_to_2d(raw[:1]).clamp(-1, 1) + 1) / 2)[0, 0].cpu().numpy(),
        "pred/sem": _to_2d(sem[:1])[0, 0].clamp(0, 1).cpu().numpy(),
    }
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for ax, (title, arr) in zip(axes, panels.items()):
        ax.imshow(arr, cmap="gray")
        ax.set_title(title)
        ax.axis("off")
    # ``seg`` is [B, D, H, W] (no channel dim); _to_2d needs 5D [B, C, D, H, W]
    # to know which axis to slice, so add + drop a singleton channel dim
    # around the call (matches the pattern in joint3d_logger.py).
    seg_2d = _to_2d(seg[:1].long().unsqueeze(1))[:, 0]  # [1, D, H, W] -> [1, H, W]
    seg_rgb = _label_to_rgb(seg_2d)[0].permute(1, 2, 0).cpu().numpy()
    axes[3].imshow(seg_rgb)
    axes[3].set_title("pred/label (Mutex Watershed)")
    axes[3].axis("off")
    fig.suptitle(f"z-slice {z} (of {image01.shape[-3]} fine-grid slices)")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# ----------------------------------------------------------------------
# Core reusable inference routine
# ----------------------------------------------------------------------

def infer_volume(
    cfg,
    ckpt_path: str,
    vol_path: Path,
    native_resolution: Sequence[float],
    origin: Sequence[int],
    size: Sequence[int],
    out_dir: Path,
    device: str = "cuda",
    sem_threshold: float = 0.5,
    save_fine_grid: bool = True,
) -> dict:
    """Run the full inference pipeline on one native-resolution region.

    Dataset-agnostic: ``vol_path`` / ``native_resolution`` / ``origin`` /
    ``size`` are the only per-dataset inputs -- the exact same function runs
    CREMI, SNEMI3D test, or any other single-channel EM ``.h5`` volume.

    Args:
        save_fine_grid: When ``True`` (default), additionally save the raw
            fine-grid (network-native, e.g. 4 nm) outputs -- ``pred_recon``,
            ``pred_sem``, ``pred_label_fine``, and the preview panel -- on top
            of the always-produced native-resolution instance segmentation.
            These are diagnostic / super-resolution artifacts, not needed for
            benchmark submission; set to ``False`` to skip them and only
            write the native-grid segmentation.

    Returns a dict of the saved output paths (``label`` always present;
    ``recon`` / ``sem`` / ``label_fine`` / ``preview`` only when
    ``save_fine_grid``).
    """
    from nanocosmos.losses import slice_head
    from nanocosmos.inference.sliding_window import sliding_window_inference

    fine_nm = float(cfg.data.pixel_size[0])
    patch_size = tuple(int(s) for s in cfg.data.patch_size)

    # ---- 1-2. read + normalise + resample onto the fine grid ----
    region = _read_region(vol_path, origin, size)
    vmin, vmax = _norm_range(vol_path, region)
    image01 = np.clip((region - vmin) / max(vmax - vmin, 1e-6), 0.0, 1.0)
    fine_image = _to_fine_grid(image01, native_resolution, fine_nm)[0]  # [1, D, H, W]
    print(
        f"Region native shape {region.shape} (z,y,x) @ {tuple(native_resolution)} nm "
        f"-> fine grid {tuple(fine_image.shape[-3:])} @ {fine_nm} nm"
    )

    # ---- 3. build model + load checkpoint ----
    device_t = torch.device(device if torch.cuda.is_available() else "cpu")
    module = build_module(cfg)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("state_dict", ckpt)
    missing, unexpected = module.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"  (warm start: {len(missing)} missing keys, kept fresh init)")
    if unexpected:
        print(f"  (warm start: {len(unexpected)} unexpected keys, ignored)")
    module.to(device_t).eval()

    # ---- 4. sliding-window inference over the fine-grid region ----
    stride = tuple(max(1, p // 2) for p in patch_size)
    head = sliding_window_inference(
        module, fine_image, patch_size=patch_size, stride=stride,
        aggregation="gaussian", batch_size=1, device=device_t,
    )  # [C, D, H, W]

    # ---- 5. split head, blend-then-agglomerate ONCE on the fine (4 nm) grid ----
    # Affinities always run at the network's fine grid: the Gaussian-weighted
    # sliding window above already blended every overlapping tile's affinity
    # logits into ONE continuous fine-grid volume (this is why we never run
    # Mutex Watershed per-tile -- per-tile instance ids from neighbouring
    # patches are not mutually consistent and cannot be stitched; only the
    # continuous affinity FIELD can be blended, and MWS runs exactly once on
    # the assembled result).
    fields = slice_head(head[None])  # add batch dim -> [1, C, D, H, W] view
    raw = fields["raw"]
    sem = fields["sem"].sigmoid()
    aff = fields["aff"].sigmoid().float()
    sem_fg = (sem[:, 0] > sem_threshold) if getattr(module.agglomerator, "gate_with_sem", True) else None
    seg_fine = module.agglomerator(aff, sem_fg)  # [1, D, H, W], still @ fine_nm

    # ---- 6. resample the FINAL instance segmentation back to native resolution ----
    # CREMI / SNEMI3D (and any benchmark) score against native-grid ground
    # truth, so the fine-grid segmentation -- however finely it was inferred --
    # must be pooled back down to the volume's native voxel grid before
    # submission. This is a DISCRETE instance-id map: nearest-neighbour only
    # (never trilinear/average -- interpolating between two different integer
    # ids produces meaningless intermediate values). ``region.shape`` (the
    # native region read from disk) is the exact target size.
    seg_native = F.interpolate(
        seg_fine[None].float(), size=region.shape, mode="nearest",
    )[0].long()  # [1, d_native, h_native, w_native]

    # ---- 7. save ----
    # ``label`` (native-resolution instance segmentation) is ALWAYS produced --
    # it is the benchmark-submission artifact. The fine-grid diagnostics
    # (recon / sem / label_fine / preview) are gated on ``save_fine_grid``
    # (default True) since they are the extra super-resolution outputs, not
    # required for scoring.
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = vol_path.stem
    out = {"label": out_dir / f"{stem}_pred_label_native.h5"}  # NATIVE grid -- the submission artifact
    _save_h5(
        seg_native[0].cpu().numpy().astype(np.int64), out["label"],
        f"nanocosmos inference from {ckpt_path}; MWS @ {fine_nm} nm fine grid, "
        f"nearest-neighbour resampled to native {tuple(native_resolution)} nm for scoring",
    )

    if save_fine_grid:
        out["recon"] = out_dir / f"{stem}_pred_recon.h5"            # fine grid (super-resolved)
        out["sem"] = out_dir / f"{stem}_pred_sem.h5"                # fine grid (super-resolved)
        out["label_fine"] = out_dir / f"{stem}_pred_label_fine.h5"  # fine grid, diagnostic
        out["preview"] = out_dir / f"{stem}_preview.png"
        _save_h5(raw[0, 0].cpu().numpy(), out["recon"], f"nanocosmos inference from {ckpt_path} (fine grid, {fine_nm} nm)")
        _save_h5(sem[0, 0].cpu().numpy(), out["sem"], f"nanocosmos inference from {ckpt_path} (fine grid, {fine_nm} nm)")
        _save_h5(
            seg_fine[0].cpu().numpy().astype(np.int64), out["label_fine"],
            f"nanocosmos inference from {ckpt_path} (fine grid, {fine_nm} nm; diagnostic only)",
        )
        try:
            _save_preview(fine_image[None], raw, sem, seg_fine, out["preview"])
        except Exception as exc:  # noqa: BLE001 -- preview is best-effort
            print(f"  (preview panel skipped: {exc})")

    n_ids = int(torch.unique(seg_fine).numel())
    print(
        f"Done: {n_ids} instance ids. Fine-grid seg {tuple(seg_fine.shape[-3:])} "
        f"@ {fine_nm} nm -> native seg {tuple(seg_native.shape[-3:])} @ {tuple(native_resolution)} nm."
    )
    print(f"Outputs written to {out_dir}/")
    for k, v in out.items():
        print(f"  {k:12s} -> {v}")
    return out


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config-name", default="nanocosmos-2B", help="Hydra config under configs/ (model/loss/training).")
    p.add_argument("--overrides", nargs="*", default=[], help="Extra Hydra overrides, e.g. model.variant=2B.")
    p.add_argument("--ckpt", required=True, help="Path to a Lightning .ckpt (full or weights-only).")
    p.add_argument("--vol", required=True, help="Volume basename (without _volume.h5), e.g. cremi3d_sample_A+_volume.")
    p.add_argument("--root", required=True, help="Directory containing <vol>.h5.")
    p.add_argument("--native-resolution", type=float, nargs=3, required=True, metavar=("Z", "Y", "X"), help="Native voxel size (nm).")
    p.add_argument("--origin", type=int, nargs=3, default=[0, 0, 0], metavar=("Z", "Y", "X"), help="Region origin in native voxels.")
    p.add_argument("--size", type=int, nargs=3, default=None, metavar=("Z", "Y", "X"), help="Region size in native voxels. Default: one fine-grid patch's worth.")
    p.add_argument("--out-dir", required=True, help="Directory to write predictions + preview.")
    p.add_argument("--device", default="cuda")
    p.add_argument("--sem-threshold", type=float, default=0.5)
    p.add_argument(
        "--save-fine-grid", dest="save_fine_grid", action=argparse.BooleanOptionalAction,
        default=True,
        help="Also save the fine-grid (network-native, e.g. 4 nm) recon/sem/label + preview "
             "diagnostics on top of the always-produced native-resolution segmentation "
             "(default: true; use --no-save-fine-grid to only write the native-grid label).",
    )
    return p.parse_args()


def _load_cfg(config_name: str, overrides):
    from hydra import compose, initialize_config_dir

    config_dir = str(Path(__file__).resolve().parent.parent / "configs")
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        return compose(config_name=config_name, overrides=list(overrides))


def main() -> None:
    args = _parse_args()
    cfg = _load_cfg(args.config_name, args.overrides)

    vol_path = Path(args.root) / f"{args.vol}.h5"
    if not vol_path.exists():
        raise SystemExit(f"Volume not found: {vol_path}")

    if args.size is None:
        fine_nm = float(cfg.data.pixel_size[0])
        patch = [int(s) for s in cfg.data.patch_size]
        size = [
            max(1, int(round(patch[d] * fine_nm / args.native_resolution[d])))
            for d in range(3)
        ]
        print(f"--size not given; defaulting to one fine-grid patch's worth of native voxels: {size}")
    else:
        size = args.size

    infer_volume(
        cfg,
        ckpt_path=args.ckpt,
        vol_path=vol_path,
        native_resolution=args.native_resolution,
        origin=args.origin,
        size=size,
        out_dir=Path(args.out_dir),
        device=args.device,
        save_fine_grid=args.save_fine_grid,
        sem_threshold=args.sem_threshold,
    )


if __name__ == "__main__":
    main()
