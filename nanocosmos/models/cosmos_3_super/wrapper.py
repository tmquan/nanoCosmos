"""Cosmos3-Super (64B) backbone wrapper.

A ~20-line tier specialisation of
:class:`nanocosmos.models.cosmos_3_common.wrapper.Cosmos3OmniWrapper`: it only
pins the Super variant registry + default variant.  All the heavy Cosmos 3
machinery is inherited from the shared base.
"""

from typing import Any

from nanocosmos.models.cosmos_3_common.wrapper import Cosmos3OmniWrapper
from nanocosmos.models.cosmos_3_super.variants import _VARIANT_CONFIGS


class Cosmos3SuperWrapper(Cosmos3OmniWrapper):
    """Cosmos3-Super (64B) omni transformer as a volumetric EM feature extractor."""

    _variant_configs = _VARIANT_CONFIGS

    def __init__(self, *args: Any, variant: str = "SUPER", **kwargs: Any) -> None:
        super().__init__(*args, variant=variant, **kwargs)


__all__ = ["Cosmos3SuperWrapper"]
