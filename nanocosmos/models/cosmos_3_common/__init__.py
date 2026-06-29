"""Shared Cosmos 3 family backbone package (Edge / Nano / Super).

Cosmos 3 tiers share ONE architecture (a dual-tower Mixture-of-Transformers
``Cosmos3OmniTransformer`` + Wan2.2-TI2V VAE) and differ only in scale.  This
package holds everything that is tier-agnostic:

* :class:`Cosmos3OmniWrapper` -- the volumetric-EM feature-extractor wrapper
  (omni forward, latent-patch repatch, feature hooks, residual-VAE decode).
* :class:`_VariantConfig` -- the MoT / GQA variant-config dataclass.
* :func:`reduce_omni_transformer` -- the Nano -> Edge structured-pruning
  warm start used until the official Edge weights ship.

The concrete tiers live in sibling packages and are ~20-line subclasses:
:mod:`nanocosmos.models.cosmos_3_edge`, :mod:`...cosmos_3_nano`,
:mod:`...cosmos_3_super`.
"""

from nanocosmos.models.cosmos_3_common.reduce import reduce_omni_transformer
from nanocosmos.models.cosmos_3_common.variants import _VariantConfig
from nanocosmos.models.cosmos_3_common.wrapper import (
    Cosmos3OmniWrapper,
    _CacheTolerantIdentity,
)

__all__ = [
    "Cosmos3OmniWrapper",
    "_CacheTolerantIdentity",
    "_VariantConfig",
    "reduce_omni_transformer",
]
