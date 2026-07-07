"""Cosmos3-Edge (4B) backbone wrapper -- direct load + parent-fill for the
handful of parameters the released checkpoint is missing.

A tier specialisation of
:class:`nanocosmos.models.cosmos_3_common.wrapper.Cosmos3OmniWrapper`.
``nvidia/Cosmos3-Edge`` has released weights on HuggingFace and the ``EDGE``
variant loads them **directly** (own transformer + own Wan2.2 VAE). The
released checkpoint is missing a small, fixed set of parameters relative to
what the installed ``diffusers`` version's ``Cosmos3VLTextMoTDecoderLayer``
always instantiates -- see the "WHY SOME WEIGHTS ARE MISSING" section in
:mod:`nanocosmos.models.cosmos_3_edge.variants` for the full finding (short
version: Edge's real MLP is non-gated/ReLU^2 and has no QK-norm on the text
pathway, so it never shipped ``mlp.gate_proj`` / ``mlp_moe_gen.gate_proj`` /
``self_attn.norm_q`` / ``self_attn.norm_k``, but diffusers unconditionally
builds all four regardless of config). :meth:`_post_load_diffusers` catches
these as ``missing_keys`` (via ``from_pretrained(...,
output_loading_info=True)``) and fills each one from the released
Cosmos3-Nano checkpoint by depth-remapping Edge's 28 layers onto Nano's 36
and truncating Nano's (larger) tensor down to Edge's shape -- the same
mechanism as the older full-model reduction, just targeted at only the
actually-missing parameters instead of every parameter.

A legacy ``reduce_from_parent=True`` path (full-model reduction from a
parent tier, e.g. for a from-scratch comparison or if a future variant has
no released checkpoint at all) is still supported and takes priority when
set.

With ``pretrained=false`` this falls back to the shared standalone path (a
random-init Edge-geometry DiT + learned conv tokenizer, no Wan2.2 VAE).
"""

import logging
from typing import Any, Dict, Optional

from nanocosmos.models.cosmos_3_common.reduce import (
    fill_missing_from_parent,
    reduce_omni_transformer,
)
from nanocosmos.models.cosmos_3_common.wrapper import Cosmos3OmniWrapper
from nanocosmos.models.cosmos_3_edge.variants import _VARIANT_CONFIGS

logger = logging.getLogger(__name__)


class Cosmos3EdgeWrapper(Cosmos3OmniWrapper):
    """Cosmos3-Edge (4B) omni transformer as a volumetric EM feature extractor.

    Loads the released ``nvidia/Cosmos3-Edge`` checkpoint directly and fills
    the small set of parameters it's missing (relative to what the
    installed diffusers class expects) from Cosmos3-Nano by truncation --
    see the module docstring.
    """

    _variant_configs = _VARIANT_CONFIGS

    def __init__(self, *args: Any, variant: str = "EDGE", **kwargs: Any) -> None:
        super().__init__(*args, variant=variant, **kwargs)

    def _post_load_diffusers(
        self,
        local_path: Any,
        cache_dir: Optional[str],
        hf_token: Optional[str],
        dit_loading_info: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Legacy full reduction, or targeted parent-fill of missing keys.

        Legacy path (``reduce_from_parent=True``): at entry ``self.dit`` is
        a *parent* tier's (e.g. Nano's) ``Cosmos3OmniTransformer`` loaded in
        full; replace it with the reduced Edge transformer (depth + width
        truncation of EVERY parameter). See :func:`reduce_omni_transformer`.

        Default path (``reduce_from_parent=False``, the normal case):
        ``self.dit`` is already Edge's OWN loaded transformer. If
        ``dit_loading_info['missing_keys']`` is non-empty and
        ``self.cfg.fill_missing_parent_repo_id`` is set, download that
        parent tier's transformer and fill ONLY those specific parameters
        by depth-remap + truncation, leaving every other (already correctly
        loaded, native Edge) parameter untouched. See
        :func:`fill_missing_from_parent`.

        No-op if neither applies (e.g. ``reduce_from_parent=False`` and no
        missing keys / no fill parent configured) -- the base class's
        normal diffusers path already loaded a complete, usable transformer.

        CHECKPOINT INVARIANT: whichever path ran (full reduction vs.
        native-load-plus-fill vs. plain native load) determines
        ``self.dit``'s exact weights; a saved training checkpoint's
        ``model.dit.*`` tensors reflect the loading path used to produce
        them. Do not change ``reduce_from_parent`` /
        ``fill_missing_parent_repo_id`` for a variant whose checkpoints are
        already in use without also re-deriving/re-training from the new
        path.
        """
        if bool(getattr(self.cfg, "reduce_from_parent", False)):
            self._reduce_full_from_parent()
            return

        missing = list((dit_loading_info or {}).get("missing_keys", []))
        parent_repo_id = getattr(self.cfg, "fill_missing_parent_repo_id", None)
        if not missing or not parent_repo_id:
            return
        self._fill_missing_from_parent_repo(
            missing, parent_repo_id, cache_dir, hf_token,
        )

    def _reduce_full_from_parent(self) -> None:
        """Legacy path: replace ``self.dit`` (a loaded parent tier) with a
        full structured-pruning reduction to the Edge geometry."""
        parent = getattr(self, "dit", None)
        if parent is None:
            return
        child_geometry = {
            "hidden_dim": self.cfg.hidden_dim,
            "num_layers": self.cfg.num_layers,
            "num_heads": self.cfg.num_heads,
            "num_key_value_heads": self.cfg.num_key_value_heads,
            "head_dim": self.cfg.head_dim,
            "intermediate_size": self.cfg.intermediate_size,
        }
        logger.info(
            "Cosmos3-Edge: reducing loaded parent (%s) -> Edge geometry %s "
            "(Wan2.2 VAE reused unchanged).",
            self.cfg.hf_repo_id, child_geometry,
        )
        edge_dit = reduce_omni_transformer(parent, child_geometry)
        self.dit = edge_dit.to(self._dtype)
        del parent  # dereferenced; free before the rest of init.

    def _fill_missing_from_parent_repo(
        self,
        missing_keys: list,
        parent_repo_id: str,
        cache_dir: Optional[str],
        hf_token: Optional[str],
    ) -> None:
        """Download ``parent_repo_id``'s transformer and fill ``missing_keys``
        in ``self.dit`` (Edge's own, already-loaded transformer) by
        depth-remap + truncation, leaving every other parameter untouched."""
        import diffusers  # type: ignore[import-untyped]

        from nanocosmos.models.cosmos_2_5_common.hf_loader import _download_from_hf

        parent_revision = getattr(self.cfg, "fill_missing_parent_revision", "main")
        logger.info(
            "Cosmos3-Edge: %d param(s) missing from the native checkpoint "
            "(%s) -- downloading parent %s (rev=%s) to fill by truncation: %s",
            len(missing_keys), self.cfg.hf_repo_id, parent_repo_id,
            parent_revision, missing_keys[:5],
        )
        parent_path = _download_from_hf(
            parent_repo_id,
            revision=parent_revision,
            cache_dir=cache_dir,
            token=hf_token,
            ignore_patterns=self._hf_ignore_patterns(),
        )
        parent_transformer = diffusers.Cosmos3OmniTransformer.from_pretrained(
            str(parent_path),
            subfolder="transformer",
            torch_dtype=self._dtype,
        )
        try:
            report = fill_missing_from_parent(
                self.dit, parent_transformer, missing_keys,
            )
            logger.info(
                "Cosmos3-Edge: filled %d/%d missing param(s) from %s "
                "(%d could not be filled and stay at fresh init).",
                report["n_filled"], len(missing_keys), parent_repo_id,
                report["n_skipped"],
            )
        finally:
            del parent_transformer


__all__ = ["Cosmos3EdgeWrapper"]
