"""Cosmos3-Nano (16B) backbone wrapper.

A ~20-line tier specialisation of
:class:`nanocosmos.models.cosmos_3_common.wrapper.Cosmos3OmniWrapper`: it only
pins the Nano variant registry + default variant.  All the heavy Cosmos 3
machinery (omni forward, ``latent_patch_size`` repatch, feature hooks,
residual Wan2.2 VAE decode) is inherited from the shared base.

``Cosmos3Nano3DWrapper`` is kept as a back-compat alias for the historical
class name (used by ``nanocosmos-16B.yaml`` / ``model.type: joint3d``).
"""

from typing import Any

from nanocosmos.models.cosmos_3_common.wrapper import Cosmos3OmniWrapper
from nanocosmos.models.cosmos_3_nano.variants import _VARIANT_CONFIGS


class Cosmos3NanoWrapper(Cosmos3OmniWrapper):
    """Cosmos3-Nano (16B) omni transformer as a volumetric EM feature extractor."""

    _variant_configs = _VARIANT_CONFIGS

    def __init__(self, *args: Any, variant: str = "NANO", **kwargs: Any) -> None:
        super().__init__(*args, variant=variant, **kwargs)


# Historical name (pre-tier-split).  Keep so existing configs / checkpoints /
# imports that reference ``Cosmos3Nano3DWrapper`` keep working unchanged.
Cosmos3Nano3DWrapper = Cosmos3NanoWrapper

__all__ = ["Cosmos3NanoWrapper", "Cosmos3Nano3DWrapper"]
