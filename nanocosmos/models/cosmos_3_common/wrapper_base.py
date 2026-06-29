"""Cosmos 3 family wrapper base (soft re-export).

All the shared backbone scaffolding (HuggingFace download, VAE
encode/decode, multi-layer feature extraction, the unified-head decoder
adapter, freeze / gradient-checkpointing plumbing) lives in
:class:`nanocosmos.models.cosmos_2_5_common.wrapper_base._BaseCosmos25Wrapper`.
The Cosmos 3 omni wrapper subclasses it, but imports it through this module so
the ``cosmos_3_*`` packages never reference the ``cosmos_2_5_common``
namespace directly -- a thin "soft link" that keeps the Cosmos 3 backbone
self-contained at the import level while the heavy shared logic stays in one
place.
"""

from nanocosmos.models.cosmos_2_5_common.wrapper_base import _BaseCosmos25Wrapper

__all__ = ["_BaseCosmos25Wrapper"]
