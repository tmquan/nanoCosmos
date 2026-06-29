"""Reduce a larger Cosmos 3 ``Cosmos3OmniTransformer`` to a smaller tier.

Cosmos3-Edge (4B total / 2B dense generator tower) is **announced but not yet
released** on HuggingFace.  Until the official weights ship, we approximate it
by *reducing* the released Cosmos3-Nano (8B dense tower) generator down to the
Edge geometry reported in the Cosmos 3 technical report (arXiv 2606.02800):

    tier   hidden  layers  heads  kv  head_dim  intermediate
    Nano     4096      36     32   8       128         12288
    Edge     2048      28     16   8       128          6144   (paper)

The reduction is a structured-pruning **warm start**, not a trained Edge model:

* **Depth** (36 -> 28 layers): keep an evenly-spaced subset of the parent's
  decoder blocks (preserves coverage from shallow to deep features).
* **Width** (every weight matrix): copy the top-left sub-block of the parent
  tensor into the child (truncate each dimension).  Because attention heads are
  contiguous ``head_dim``-sized row blocks, truncating the first
  ``num_heads * head_dim`` rows of ``q_proj`` / ``o_proj`` keeps the first
  ``num_heads`` heads intact; ``num_key_value_heads`` is unchanged (8 across all
  tiers), so ``k_proj`` / ``v_proj`` only lose input columns.

The parent's Wan2.2 VAE is identical across tiers and is reused unchanged by
the caller (only the transformer is reduced here).

This is deliberately generic (shape-driven) so it does not hard-code the omni
transformer's exact module names: any child parameter whose name maps onto an
existing parent parameter (after remapping the block index) and whose every
dimension is ``<=`` the parent's is copied by truncation; everything else keeps
the child's fresh initialisation.
"""

import logging
import re
from typing import Any, Dict, Optional

import torch

logger = logging.getLogger(__name__)

# Config keys that carry the geometry, with the aliases diffusers / Qwen3-VL
# style configs are known to use.  We override whichever alias is actually
# present in the parent config so the child ``from_config`` sees a consistent
# shape.
_GEOMETRY_ALIASES: Dict[str, tuple] = {
    "hidden_dim": ("hidden_size", "hidden_dim", "dim"),
    "num_layers": ("num_hidden_layers", "num_layers"),
    "num_heads": ("num_attention_heads", "num_heads"),
    "num_key_value_heads": ("num_key_value_heads", "num_kv_heads"),
    "head_dim": ("head_dim",),
    "intermediate_size": ("intermediate_size", "ffn_dim", "intermediate_dim"),
}

_LAYER_KEY_ALIASES = ("num_hidden_layers", "num_layers")
_BLOCK_CONTAINER_ATTRS = ("transformer_blocks", "blocks", "layers")


def _first_present(cfg: Dict[str, Any], aliases: tuple) -> Optional[str]:
    for k in aliases:
        if k in cfg:
            return k
    return None


def _resolve_num_layers(cfg: Dict[str, Any]) -> Optional[int]:
    key = _first_present(cfg, _LAYER_KEY_ALIASES)
    return int(cfg[key]) if key is not None else None


def _block_container_attr(model: torch.nn.Module) -> Optional[str]:
    for attr in _BLOCK_CONTAINER_ATTRS:
        if hasattr(model, attr):
            return attr
    return None


def _truncate_to(src: torch.Tensor, shape: torch.Size) -> Optional[torch.Tensor]:
    """Return the top-left ``shape`` sub-block of ``src`` (or None if it can't).

    The child must not be *larger* than the parent in any dimension, and the
    tensor rank must match.  A no-op slice (identical shape) returns a clone.
    """
    if src.dim() != len(shape):
        return None
    if any(t > s for t, s in zip(shape, src.shape)):
        return None
    idx = tuple(slice(0, int(t)) for t in shape)
    return src[idx].contiguous().clone()


def build_geometry_overrides(child_cfg_fields: Dict[str, int]) -> Dict[str, int]:
    """Map nanoCosmos variant fields -> the canonical override dict.

    ``child_cfg_fields`` keys are the :class:`_VariantConfig` field names
    (``hidden_dim``, ``num_layers``, ``num_heads``, ``num_key_value_heads``,
    ``head_dim``, ``intermediate_size``); returns the same dict unchanged --
    the alias resolution against the *parent* config happens in
    :func:`reduce_omni_transformer`.
    """
    return dict(child_cfg_fields)


def reduce_omni_transformer(
    parent: torch.nn.Module,
    child_geometry: Dict[str, int],
) -> torch.nn.Module:
    """Build a smaller ``Cosmos3OmniTransformer`` warm-started from ``parent``.

    Args:
        parent: A loaded (pretrained) ``Cosmos3OmniTransformer`` (e.g. Nano).
        child_geometry: nanoCosmos variant fields for the target tier
            (``hidden_dim`` / ``num_layers`` / ``num_heads`` /
            ``num_key_value_heads`` / ``head_dim`` / ``intermediate_size``).

    Returns:
        A new ``Cosmos3OmniTransformer`` at the child geometry, on the parent's
        dtype/device, with all shape-compatible weights truncation-copied from
        the parent and the rest left at fresh init.
    """
    from diffusers import Cosmos3OmniTransformer  # type: ignore[attr-defined]

    parent_cfg: Dict[str, Any] = dict(parent.config)
    child_cfg: Dict[str, Any] = dict(parent_cfg)

    # Override only the geometry keys that actually exist in the parent config,
    # using whichever alias the config uses.
    for field, aliases in _GEOMETRY_ALIASES.items():
        if field not in child_geometry:
            continue
        key = _first_present(parent_cfg, aliases)
        if key is None:
            logger.warning(
                "reduce: parent config has none of %s; cannot set %s=%s",
                aliases, field, child_geometry[field],
            )
            continue
        child_cfg[key] = int(child_geometry[field])

    parent_layers = _resolve_num_layers(parent_cfg)
    child_layers = _resolve_num_layers(child_cfg)
    if parent_layers is None or child_layers is None:
        raise ValueError(
            "reduce: could not resolve layer count from the omni config "
            f"(tried {_LAYER_KEY_ALIASES}).",
        )

    # diffusers stamps these onto a loaded config; drop so from_config rebuilds.
    for meta in ("_name_or_path",):
        child_cfg.pop(meta, None)

    child = Cosmos3OmniTransformer.from_config(child_cfg)
    ref_param = next(parent.parameters(), None)
    if ref_param is not None:
        child = child.to(device=ref_param.device, dtype=ref_param.dtype)

    # Evenly-spaced depth map: child block j <- parent block round(j*(P-1)/(C-1)).
    if child_layers >= parent_layers:
        layer_map = {j: j for j in range(child_layers)}
    elif child_layers == 1:
        layer_map = {0: 0}
    else:
        layer_map = {
            j: int(round(j * (parent_layers - 1) / (child_layers - 1)))
            for j in range(child_layers)
        }

    attr = _block_container_attr(child) or _block_container_attr(parent) or "transformer_blocks"
    block_pat = re.compile(rf"(?:^|\.){re.escape(attr)}\.(\d+)\.")

    def _remap_key(child_key: str) -> Optional[str]:
        m = block_pat.search(child_key)
        if m is None:
            return child_key  # non-block param: same name
        child_idx = int(m.group(1))
        parent_idx = layer_map.get(child_idx)
        if parent_idx is None:
            return None
        start, end = m.span(1)
        return child_key[:start] + str(parent_idx) + child_key[end:]

    parent_sd = parent.state_dict()
    child_sd = child.state_dict()

    new_sd: Dict[str, torch.Tensor] = {}
    n_copied = n_truncated = n_fresh = 0
    for ckey, cval in child_sd.items():
        pkey = _remap_key(ckey)
        if pkey is not None and pkey in parent_sd:
            sliced = _truncate_to(parent_sd[pkey], cval.shape)
            if sliced is not None:
                new_sd[ckey] = sliced.to(dtype=cval.dtype, device=cval.device)
                n_copied += 1
                if tuple(sliced.shape) != tuple(parent_sd[pkey].shape):
                    n_truncated += 1
                continue
        new_sd[ckey] = cval  # keep fresh init
        n_fresh += 1

    missing, unexpected = child.load_state_dict(new_sd, strict=False)
    logger.info(
        "reduce: Cosmos3OmniTransformer %d->%d layers; copied %d params "
        "(%d truncated), %d kept fresh; load_state_dict missing=%d unexpected=%d.",
        parent_layers, child_layers, n_copied, n_truncated, n_fresh,
        len(missing), len(unexpected),
    )
    return child


__all__ = ["reduce_omni_transformer", "build_geometry_overrides"]
