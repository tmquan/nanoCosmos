"""Variant registry for Cosmos3-Edge (4B total / 2B dense generator tower).

Cosmos3-Edge is **announced but not yet released** on HuggingFace (planned for
Jetson / on-device).  Per the Cosmos 3 technical report (arXiv 2606.02800) it
is the only tier trained *from scratch*, with the dual-tower generator
geometry:

    hidden_size=2048, num_hidden_layers=28, num_attention_heads=16,
    num_key_value_heads=8 (GQA), head_dim=128, intermediate_size=6144 (MLP 3.0)

Provenance (high confidence): each Cosmos 3 tier's generator tower is the
corresponding **Qwen3 dense** backbone shape, verified field-for-field against
the published configs:

    Nano  == Qwen3-8B   (4096 / 36 / 32 / 8 / head_dim 128 / inter 12288)
    Super == Qwen3-32B  (5120 / 64 / 64 / 8 / head_dim 128 / inter 25600)
    Edge  == Qwen3-1.7B (2048 / 28 / 16 / 8 / head_dim 128 / inter  6144)

The report's stated Edge dims (2048 / 28 / 16 / 8) match Qwen3-1.7B exactly,
which fixes ``intermediate_size=6144`` -- it is Qwen3-1.7B's FFN width, not a
free guess.  "Edge trained from scratch" follows naturally: there is no
Qwen3-VL at the 1.7B size to initialise from, unlike Nano/Super (Qwen3-VL
8B / 32B).  Everything else (VAE, ``latent_patch_size``, ``head_dim``, KV heads,
Cosmos ``rope_theta=5e6``, mRoPE section ``[24, 20, 20]``, ``use_moe``) is
shared with Nano/Super -- the same Wan2.2-TI2V VAE (``z_dim=48``, 16x spatial /
4x temporal) -- and is inherited verbatim from the Nano parent config during
the reduction.

Until the official weights ship we warm-start Edge by REDUCING the released
Cosmos3-Nano (see :mod:`nanocosmos.models.cosmos_3_common.reduce`): the
variant's ``hf_repo_id`` therefore points at the **Nano** repo (the parent we
load + reduce) and ``reduce_from_parent=True`` triggers the depth+width
truncation in :class:`Cosmos3EdgeWrapper`.  Set ``model.pretrained=false`` in
the config to skip the warm start entirely and train the Edge geometry from
scratch (standalone DiT + learned conv tokenizer, no Wan2.2 VAE).
"""

from typing import Dict

from nanocosmos.models.cosmos_3_common.variants import _VariantConfig

_VARIANT_CONFIGS: Dict[str, _VariantConfig] = {
    "EDGE": _VariantConfig(
        # Parent we reduce from: the released Nano checkpoint (transformer +
        # the shared Wan2.2 VAE).  When ``pretrained=true`` the base loader
        # downloads this repo, then ``Cosmos3EdgeWrapper`` reduces the loaded
        # Nano transformer to the Edge geometry and reuses Nano's VAE as-is.
        hf_repo_id="nvidia/Cosmos3-Nano",
        hf_revision="main",
        reduce_from_parent=True,
        hidden_dim=2048,
        num_layers=28,
        num_heads=16,
        latent_channels=48,
        spatial_compression=16,
        temporal_compression=4,
        # ~2B dense generator tower + VAE; warm-started reduction fits a
        # frozen-VAE forward comfortably under ~16 GB.
        estimated_vram_gb=14.0,
        max_sequence_length=131072,
        patch_size=2,
        mlp_ratio=6144 / 2048,  # = 3.0
        head_dim=128,
        num_key_value_heads=8,
        intermediate_size=6144,  # INFERRED (report omits the FFN width)
        use_moe=True,
    ),
}

__all__ = ["_VARIANT_CONFIGS", "_VariantConfig"]
