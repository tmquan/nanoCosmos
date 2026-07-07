"""Warm-start a Cosmos 3 ``Cosmos3OmniTransformer`` from a larger tier.

Two related operations, both structured-pruning **warm starts** (not a
trained model), sharing the same depth-remap + width-truncation machinery:

* :func:`reduce_omni_transformer` -- build an ENTIRE smaller tier
  (e.g. Edge) from a larger parent (e.g. Nano): every child parameter is
  either truncation-copied from the (depth-remapped) parent or left at
  fresh init.
* :func:`fill_missing_from_parent` -- given a child model that is MOSTLY
  already correctly loaded (e.g. from its own native pretrained checkpoint),
  fill in ONLY a specific list of missing parameters from a parent tier,
  leaving every other already-loaded parameter untouched. Used by
  Cosmos3-Edge to source a small subset of parameters its own released
  checkpoint doesn't have (a diffusers/checkpoint architecture gap -- see
  the "WHY SOME WEIGHTS ARE MISSING" note in
  ``nanocosmos/models/cosmos_3_edge/variants.py``) from Nano's
  corresponding (larger) tensors, rather than leaving them at fresh init.

Depth remap (both operations): keep an evenly-spaced subset of the parent's
decoder blocks (preserves coverage from shallow to deep features) --
child block ``j`` <- parent block ``round(j * (P-1) / (C-1))``.

Width truncation (both operations): copy the top-left sub-block of the
parent tensor into the child (truncate each dimension). Because attention
heads are contiguous ``head_dim``-sized row blocks, truncating the first
``num_heads * head_dim`` rows of ``q_proj`` / ``o_proj`` keeps the first
``num_heads`` heads intact; when ``num_key_value_heads`` is unchanged across
tiers (8, for Nano/Edge/Super), ``k_proj`` / ``v_proj`` only lose input
columns.

This is deliberately generic (shape-driven) so it does not hard-code the omni
transformer's exact module names: any child parameter whose name maps onto an
existing parent parameter (after remapping the block index) and whose every
dimension is ``<=`` the parent's is copied by truncation; everything else
(for :func:`reduce_omni_transformer`) keeps the child's fresh initialisation.
"""

import logging
import re
import warnings
from typing import Any, Callable, Dict, Optional, Sequence

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


def _first_real_device(model: torch.nn.Module) -> torch.device:
    """First non-meta parameter's device, else CPU.

    Used to materialise newly-filled parameters onto whatever device the
    rest of an already-(mostly)-loaded model actually lives on, since a
    meta parameter's own ``.device`` is ``meta`` and useless for this.
    """
    for p in model.parameters():
        if not p.is_meta:
            return p.device
    return torch.device("cpu")


def _replace_param_or_buffer(
    model: torch.nn.Module, key: str, new_tensor: torch.Tensor,
) -> bool:
    """Replace the Parameter/buffer at ``key`` (dotted ``state_dict`` path)
    with ``new_tensor``, IN PLACE on its owning submodule.

    Unlike ``existing_tensor.data.copy_(new_tensor)``, this works even when
    the existing tensor is on the ``meta`` device (no real storage to copy
    into) -- e.g. ``diffusers``' ``low_cpu_mem_usage`` loading leaves any
    parameter absent from the checkpoint as a literal meta placeholder
    (despite logging "newly initialized"), and a plain ``.data.copy_()``
    onto it is silently a no-op, leaving the model impossible to
    ``.to(device)`` / ``.cpu()`` later ("Cannot copy out of meta tensor").

    Returns False (no-op) if ``key`` doesn't resolve to an existing
    parameter or buffer.
    """
    parts = key.split(".")
    owner: Any = model
    for p in parts[:-1]:
        owner = getattr(owner, p, None)
        if owner is None:
            return False
    leaf = parts[-1]
    if leaf in owner._parameters:
        old = owner._parameters[leaf]
        if old is None:
            return False
        owner._parameters[leaf] = torch.nn.Parameter(
            new_tensor.to(dtype=old.dtype), requires_grad=old.requires_grad,
        )
        return True
    if leaf in owner._buffers:
        old = owner._buffers[leaf]
        if old is None:
            return False
        owner._buffers[leaf] = new_tensor.to(dtype=old.dtype)
        return True
    return False


def _build_layer_map(parent_layers: int, child_layers: int) -> Dict[int, int]:
    """Evenly-spaced depth map: child block j <- parent block round(j*(P-1)/(C-1))."""
    if child_layers >= parent_layers:
        return {j: j for j in range(child_layers)}
    if child_layers == 1:
        return {0: 0}
    return {
        j: int(round(j * (parent_layers - 1) / (child_layers - 1)))
        for j in range(child_layers)
    }


def _make_remap_key_fn(
    child: torch.nn.Module, parent: torch.nn.Module, layer_map: Dict[int, int],
) -> Callable[[str], Optional[str]]:
    """Build a ``child_key -> parent_key`` remapper using ``layer_map`` for
    the block-container attribute (``transformer_blocks`` / ``blocks`` /
    ``layers``) present on either model. Non-block params map unchanged."""
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

    return _remap_key


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

    layer_map = _build_layer_map(parent_layers, child_layers)
    _remap_key = _make_remap_key_fn(child, parent, layer_map)

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
    # Warn loudly when the reduction warm-started only a small fraction of the
    # child: a near-cold init (mostly-fresh tensors) usually means the layer map
    # or geometry did not line up with the parent, not an intended random start.
    n_total = n_copied + n_fresh
    if n_total and n_fresh > 0.5 * n_total:
        warnings.warn(
            f"reduce_omni_transformer kept {n_fresh}/{n_total} child tensors at "
            f"fresh init (>50%); the reduced backbone is largely un-warm-started "
            f"-- check the child geometry / layer map against the parent.",
            stacklevel=2,
        )
    return child


def fill_missing_from_parent(
    child: torch.nn.Module,
    parent: torch.nn.Module,
    missing_keys: Sequence[str],
) -> Dict[str, Any]:
    """Fill ONLY ``missing_keys`` in ``child`` by truncation-copying the
    corresponding (depth-remapped) tensor from ``parent``.

    Unlike :func:`reduce_omni_transformer` (which rebuilds and warm-starts
    the WHOLE child model from the parent), this targets a specific,
    caller-supplied list of parameter names -- typically the
    ``missing_keys`` a diffusers ``from_pretrained(...,
    output_loading_info=True)`` call reports for the child's OWN native
    checkpoint -- and leaves every other (already correctly loaded) child
    parameter completely untouched.

    Args:
        child: The already-built, already-(mostly)-loaded model whose
            ``missing_keys`` parameters are still at fresh init.
        parent: A loaded (pretrained) model of the SAME architecture family
            at a larger (or equal) geometry (e.g. Nano for Edge). Only used
            as a tensor source; not modified.
        missing_keys: Parameter names (``child.state_dict()`` keys) to fill.
            Keys not found in the remapped parent, or whose shape doesn't
            truncate cleanly (child dim > parent dim), are left untouched
            and reported under ``"skipped"``.

    Returns:
        ``{"filled": [...], "skipped": [...], "n_filled": int, "n_skipped": int}``.
    """
    parent_cfg: Dict[str, Any] = dict(parent.config)
    child_cfg: Dict[str, Any] = dict(child.config)
    parent_layers = _resolve_num_layers(parent_cfg)
    child_layers = _resolve_num_layers(child_cfg)
    if parent_layers is None or child_layers is None:
        raise ValueError(
            "fill_missing_from_parent: could not resolve layer count from "
            f"the omni config (tried {_LAYER_KEY_ALIASES}).",
        )

    layer_map = _build_layer_map(parent_layers, child_layers)
    _remap_key = _make_remap_key_fn(child, parent, layer_map)

    parent_sd = parent.state_dict()
    child_params: Dict[str, torch.Tensor] = dict(child.named_parameters())
    child_buffers: Dict[str, torch.Tensor] = dict(child.named_buffers())
    child_device = _first_real_device(child)

    filled, skipped = [], []
    for ckey in missing_keys:
        target = child_params.get(ckey, child_buffers.get(ckey))
        if target is None:
            skipped.append(ckey)  # not an actual child param/buffer name
            continue
        pkey = _remap_key(ckey)
        if pkey is None or pkey not in parent_sd:
            skipped.append(ckey)
            continue
        sliced = _truncate_to(parent_sd[pkey], target.shape)
        if sliced is None:
            skipped.append(ckey)
            continue
        # NOTE: NOT `target.data.copy_(...)` -- `target` may be a literal
        # meta-device placeholder (see `_replace_param_or_buffer`), so we
        # must REPLACE the Parameter/buffer object, not write into it.
        new_tensor = sliced.to(device=child_device)
        if not _replace_param_or_buffer(child, ckey, new_tensor):
            skipped.append(ckey)
            continue
        filled.append(ckey)

    logger.info(
        "fill_missing_from_parent: filled %d/%d missing param(s) from parent "
        "(%d->%d layers, depth-remapped); %d could not be filled (skipped).",
        len(filled), len(missing_keys), parent_layers, child_layers, len(skipped),
    )
    if skipped:
        logger.warning(
            "fill_missing_from_parent: left %d param(s) at fresh init "
            "(no matching/compatible parent tensor): %s",
            len(skipped), skipped[:10],
        )
    return {
        "filled": filled, "skipped": skipped,
        "n_filled": len(filled), "n_skipped": len(skipped),
    }


__all__ = ["reduce_omni_transformer", "fill_missing_from_parent"]
