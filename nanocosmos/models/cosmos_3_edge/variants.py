"""Variant registry for Cosmos3-Edge (4B total / 2B dense generator tower).

Cosmos3-Edge is **announced but not yet released** on HuggingFace (planned for
Jetson / on-device).  Per the Cosmos 3 technical report (arXiv 2606.02800) it
is the only tier trained *from scratch* (Nano/Super initialise from Qwen3-VL),
with the dual-tower generator geometry:

    hidden_size=2048, num_hidden_layers=28, num_attention_heads=16,
    num_key_value_heads=8 (GQA), head_dim=128, intermediate_size=6144 (MLP 3.0)

The report lists layers / hidden / heads / kv explicitly; ``intermediate_size``
is inferred at the family's MLP ratio 3.0 (Nano's 12288/4096).  Everything else
(VAE, ``latent_patch_size``, ``head_dim``, KV heads, rope, mRoPE section,
``use_moe``) is shared with Nano/Super -- the same Wan2.2-TI2V VAE
(``z_dim=48``, 16x spatial / 4x temporal).

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
