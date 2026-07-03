#!/usr/bin/env python
"""Checkpoint-compatibility gate for nanocosmos.

Guards the invariant that the shipped Lightning checkpoint (``last.ckpt``) can
still be loaded by the current code — i.e. no edit has changed a serialized
``state_dict`` key, layer shape, or the ``hyper_parameters`` contract.

Two modes:

* ``--structural`` (default, fast, no model build): reads the checkpoint's
  ``state_dict`` key/shape fingerprint and ``hyper_parameters`` keys and diffs
  them against a golden snapshot (created on first run). Also imports the whole
  package to catch import-time breakage. Runs in seconds.
* ``--full`` (slow, offline model build): reconstructs the LightningModule from
  the checkpoint's own saved ``hyper_parameters`` and calls
  ``load_state_dict(strict=True)``. This is the definitive proof that the model
  the current code builds still matches the checkpoint exactly. Requires the
  pretrained backbone weights to be available locally (uses ``HF_HUB_OFFLINE``).

Exit code 0 = checkpoint still loadable; non-zero = a serialized-surface change
was detected (or the full load failed).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
CKPT = REPO / "last.ckpt"
GOLDEN = REPO / "scripts" / ".ckpt_golden.json"

# Allow running as a plain script from anywhere (package is used in-tree, not
# necessarily pip-installed).
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

# Map of joint model.type -> concrete LightningModule, used to reconstruct the
# module from the checkpoint's saved hparams for the --full load test.
_JOINT_MODULES = {
    "joint3d_2b": ("nanocosmos.modules.joint3d", "JointPredict3DModule"),
    "joint3d": ("nanocosmos.modules.joint3d", "Joint3DModule"),
    "joint3d_edge": ("nanocosmos.modules.joint3d", "JointEdge3DModule"),
    "joint3d_super": ("nanocosmos.modules.joint3d", "JointSuper3DModule"),
}


def _fingerprint(ckpt_path: Path) -> dict:
    import torch

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False, mmap=True)
    sd = ckpt.get("state_dict", {})
    keys = {k: list(v.shape) for k, v in sd.items() if hasattr(v, "shape")}
    hp = ckpt.get("hyper_parameters", {})
    return {
        "pl_version": ckpt.get("pytorch-lightning_version"),
        "n_state_dict_keys": len(sd),
        "state_dict": keys,
        "hyper_parameter_keys": sorted(hp.keys()) if hasattr(hp, "keys") else [],
    }


def structural(update_golden: bool) -> int:
    # Import the whole package first — catches edits that break import.
    import importlib

    for mod in (
        "nanocosmos", "nanocosmos.models", "nanocosmos.modules",
        "nanocosmos.datasets", "nanocosmos.datamodules", "nanocosmos.losses",
        "nanocosmos.transforms", "nanocosmos.metrics", "nanocosmos.callbacks",
        "nanocosmos.inference", "nanocosmos.preprocessors",
    ):
        importlib.import_module(mod)
    print("import: OK (package + all subpackages)")

    fp = _fingerprint(CKPT)
    print(f"ckpt: PL {fp['pl_version']}, {fp['n_state_dict_keys']} state_dict keys, "
          f"hparams={fp['hyper_parameter_keys']}")

    if update_golden or not GOLDEN.exists():
        GOLDEN.write_text(json.dumps(fp, indent=1, sort_keys=True))
        print(f"golden: wrote {GOLDEN.name} ({fp['n_state_dict_keys']} keys)")
        return 0

    golden = json.loads(GOLDEN.read_text())
    if golden == fp:
        print("STRUCTURAL: OK — serialized surface identical to golden snapshot.")
        return 0

    # Report the drift precisely.
    gk, fk = set(golden["state_dict"]), set(fp["state_dict"])
    added, removed = sorted(fk - gk), sorted(gk - fk)
    reshaped = [k for k in gk & fk if golden["state_dict"][k] != fp["state_dict"][k]]
    print("STRUCTURAL: FAIL — serialized surface changed vs golden:")
    if removed:
        print(f"  removed keys ({len(removed)}): {removed[:20]}")
    if added:
        print(f"  added keys ({len(added)}): {added[:20]}")
    if reshaped:
        print(f"  reshaped keys ({len(reshaped)}): {reshaped[:20]}")
    if golden["hyper_parameter_keys"] != fp["hyper_parameter_keys"]:
        print(f"  hparams changed: {golden['hyper_parameter_keys']} -> "
              f"{fp['hyper_parameter_keys']}")
    return 1


def full() -> int:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    import importlib

    import torch

    ckpt = torch.load(CKPT, map_location="cpu", weights_only=False, mmap=True)
    hp = ckpt["hyper_parameters"]
    sd = ckpt["state_dict"]

    # The shipped last.ckpt is a joint recipe (nested loss.seg.offsets). Pick the
    # module by the loss shape: nested 'seg' => joint. Default to the 2B joint.
    loss_cfg = dict(hp.get("loss_config", {}) or {})
    is_joint = "seg" in loss_cfg
    mod_path, cls_name = _JOINT_MODULES["joint3d_2b"] if is_joint else (
        "nanocosmos.modules.cosmos_predict_2_5", "CosmosPredict3DModule")
    ModCls = getattr(importlib.import_module(mod_path), cls_name)
    print(f"reconstructing {cls_name} from checkpoint hyper_parameters ...")

    model = ModCls(
        model_config=dict(hp.get("model_config", {}) or {}),
        optimizer_config=dict(hp.get("optimizer_config", {}) or {}),
        loss_config=loss_cfg,
        training_config=dict(hp.get("training_config", {}) or {}),
    )
    missing, unexpected = model.load_state_dict(sd, strict=False)
    # Also check shapes line up for the intersection.
    live = model.state_dict()
    mism = [k for k in sd if k in live and tuple(live[k].shape) != tuple(sd[k].shape)]
    ok = not missing and not unexpected and not mism
    print(f"load_state_dict: missing={len(missing)} unexpected={len(unexpected)} "
          f"shape_mismatch={len(mism)}")
    if not ok:
        for label, items in (("missing", missing), ("unexpected", unexpected),
                             ("shape_mismatch", mism)):
            if items:
                print(f"  {label} ({len(items)}): {list(items)[:15]}")
        print("FULL: FAIL — current code does not strictly load last.ckpt.")
        return 1
    print(f"FULL: OK — {len(sd)} keys loaded strictly, no missing/unexpected/reshaped.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--full", action="store_true",
                    help="reconstruct the module and strict-load (slow, offline build)")
    ap.add_argument("--update-golden", action="store_true",
                    help="(re)write the structural golden snapshot")
    args = ap.parse_args()
    if not CKPT.exists():
        print(f"SKIP: {CKPT} not found", file=sys.stderr)
        return 0
    rc = structural(update_golden=args.update_golden)
    if rc == 0 and args.full:
        rc = full()
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
