"""Shared variant-config dataclass for the Cosmos 3 family.

Cosmos 3 (NVIDIA's omnimodal world-model family, released 2026-05-31) is a
dual-tower **Mixture-of-Transformers** (``Cosmos3OmniTransformer``): an
autoregressive *reasoner* tower and a diffusion *generator* tower with
per-stream projections, a generation softmax over concatenated keys/values,
and a unified 3D mRoPE.  For volumetric EM segmentation we drive only the
generator (video) tower as a feature extractor. Image, audio, and action
conditioning stay null. Text is a metadata prompt when the batch supplies
one, otherwise a single null token. A tier is fully described by its
generator-tower geometry.

All three tiers share the SAME architecture and differ only in scale; the
numbers below are mirrored verbatim from the published HuggingFace
``transformer/config.json`` of each tier (see each tier's ``variants.py``):

================  ======  ======  =====  ===  ========  ============  ========
tier              hidden  layers  heads  kv   head_dim  intermediate  init
================  ======  ======  =====  ===  ========  ============  ========
Edge (4B)           2048      28     16    8       128          9216  scratch
Nano (16B)          4096      36     32    8       128         12288  Qwen3-VL 8B
Super (64B)         5120      64     64    8       128         25600  Qwen3-VL 32B
================  ======  ======  =====  ===  ========  ============  ========

Every tier shares the Wan2.2-TI2V VAE (``AutoencoderKLWan``: ``z_dim=48``,
16x spatial / 4x temporal), published ``latent_patch_size=2`` (nanoCosmos
repatches this to 1 after load), ``head_dim=128``,
``num_key_value_heads=8`` (GQA), and mRoPE section ``[24, 20, 20]``.
The released Edge ``transformer/config.json`` sets ``rope_theta`` to ``1e8``.
``_VariantConfig.use_moe`` is local metadata. The tower keeps separate
``mlp`` and ``mlp_moe_gen`` modules.

:class:`_VariantConfigBase` carries the download and shape metadata
common to the tiers.  This subclass adds the MoT / GQA fields so the
standalone fallback and the Edge reduction path can reconstruct the
shape.  Cosmos 3 trains in BF16.
"""

from dataclasses import dataclass
from typing import Optional

from nanocosmos.models.cosmos_2_5_common.variants import _VariantConfigBase


@dataclass
class _VariantConfig(_VariantConfigBase):
    """Cosmos 3 variant config (adds the MoT / GQA shape fields).

    The extra fields are not consumed by the shared diffusers load path
    (which reads the shape straight off the pretrained checkpoint) but are
    required by the random-init standalone fallback, document the
    omni-transformer geometry at a glance, and drive the Edge reduction.
    """

    # Per-head attention dim (``head_dim`` in the HF config).  Decoupled from
    # ``hidden_dim`` / ``num_heads`` in the Qwen3-VL style: e.g. Super has
    # ``num_heads * head_dim = 64 * 128 = 8192 != hidden_dim (5120)``.
    head_dim: int = 128
    # Grouped-query attention: KV heads < query heads (8 across all tiers).
    num_key_value_heads: int = 8
    # Feed-forward inner dim (``intermediate_size``); ``mlp_ratio`` on the base
    # mirrors ``intermediate_size / hidden_dim`` for the standalone DiT.
    intermediate_size: int = 12288
    # Mixture-of-Transformers routing (``use_moe`` in the config).
    use_moe: bool = True
    # Edge-only (legacy path, kept for standalone-from-scratch comparisons):
    # when True the published ``hf_repo_id`` is a *parent* tier (Nano) whose
    # loaded ``Cosmos3OmniTransformer`` is reduced in FULL (depth + width
    # truncation, every weight either copied or fresh) to THIS variant's
    # geometry, while the parent's Wan2.2 VAE is reused unchanged.  See
    # :func:`nanocosmos.models.cosmos_3_common.reduce.reduce_omni_transformer`.
    reduce_from_parent: bool = False
    # When ``hf_repo_id`` is this variant's OWN native checkpoint (the normal
    # case) and a load still reports ``missing_keys``, those parameters are
    # filled by truncation-copying the depth-remapped tensor from this parent
    # tier. diffusers 0.42.1 matches the Edge checkpoint (``hidden_act: relu2``,
    # ``qk_norm_for_text: false``), so the shipped EDGE variant leaves this
    # unset and missing keys stay at fresh init. See
    # :func:`nanocosmos.models.cosmos_3_common.reduce.fill_missing_from_parent`.
    fill_missing_parent_repo_id: Optional[str] = None
    fill_missing_parent_revision: str = "main"


__all__ = ["_VariantConfig"]
