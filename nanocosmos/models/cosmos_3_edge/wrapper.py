"""Cosmos3-Edge (4B) backbone wrapper -- reduced from Nano until release.

A tier specialisation of
:class:`nanocosmos.models.cosmos_3_common.wrapper.Cosmos3OmniWrapper`.  Because
Cosmos3-Edge is not yet released, the pretrained path loads the **Nano**
checkpoint (transformer + the shared Wan2.2 VAE) and then reduces the loaded
Nano ``Cosmos3OmniTransformer`` down to the Edge geometry (depth + width
truncation, weights partially copied) in :meth:`_post_load_diffusers`, reusing
Nano's VAE unchanged.

With ``pretrained=false`` this falls back to the shared standalone path (a
random-init Edge-geometry DiT + learned conv tokenizer, no Wan2.2 VAE).
"""

import logging
from typing import Any, Optional

from nanocosmos.models.cosmos_3_common.reduce import reduce_omni_transformer
from nanocosmos.models.cosmos_3_common.wrapper import Cosmos3OmniWrapper
from nanocosmos.models.cosmos_3_edge.variants import _VARIANT_CONFIGS

logger = logging.getLogger(__name__)


class Cosmos3EdgeWrapper(Cosmos3OmniWrapper):
    """Cosmos3-Edge (4B) omni transformer as a volumetric EM feature extractor.

    Warm-started from the released Cosmos3-Nano via structured reduction until
    the official Edge weights ship.
    """

    _variant_configs = _VARIANT_CONFIGS

    def __init__(self, *args: Any, variant: str = "EDGE", **kwargs: Any) -> None:
        super().__init__(*args, variant=variant, **kwargs)

    def _post_load_diffusers(
        self,
        local_path: Any,
        cache_dir: Optional[str],
        hf_token: Optional[str],
    ) -> None:
        """Reduce the freshly-loaded **parent** (Nano) transformer to Edge.

        Runs inside the base ``_build_backbone`` right after a successful
        diffusers load, so at entry ``self.dit`` is the parent (Nano)
        ``Cosmos3OmniTransformer`` and ``self.vae_*`` is the shared Wan2.2
        VAE.  We replace ``self.dit`` with the reduced Edge transformer and
        leave the VAE untouched; everything built afterwards
        (feature_projector / hooks / decoder_adapter) already keys on the
        Edge ``self.cfg`` and the new ``self.dit.config``.
        """
        if not bool(getattr(self.cfg, "reduce_from_parent", False)):
            return
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
        # The parent is now dereferenced; free it before the rest of init.
        del parent


__all__ = ["Cosmos3EdgeWrapper"]
