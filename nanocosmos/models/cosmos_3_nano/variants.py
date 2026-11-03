"""Variant registry for Cosmos3-Nano (16B total / 8B dense generator tower).

Numbers mirrored verbatim from the published HF config
(``nvidia/Cosmos3-Nano`` ``transformer/config.json``,
``Cosmos3OmniTransformer``): ``hidden_size=4096``, ``num_hidden_layers=36``,
``num_attention_heads=32``, ``num_key_value_heads=8`` (GQA), ``head_dim=128``,
``intermediate_size=12288`` (MLP ratio 3.0), ``latent_channel=48``,
``latent_patch_size=2``, ``max_position_embeddings=262144``, ``use_moe=true``.
The VAE (``vae/config.json``, ``AutoencoderKLWan``, Wan2.2-TI2V) is
``z_dim=48``, 16x spatial / 4x temporal.

Nano initialises from pre-trained Qwen3-VL 8B and is loadable from HuggingFace.
Only BF16 is officially supported (per the model card); keep ``model.dtype``
at ``bf16``.
"""

from typing import Dict

from nanocosmos.models.cosmos_3_common.variants import _VariantConfig

_VARIANT_CONFIGS: Dict[str, _VariantConfig] = {
    "NANO": _VariantConfig(
        hf_repo_id="nvidia/Cosmos3-Nano",
        hf_revision="main",
        hidden_dim=4096,
        num_layers=36,
        num_heads=32,
        latent_channels=48,
        spatial_compression=16,
        temporal_compression=4,
        # 16B params @ bf16 ~= 32 GB of weights before activations /
        # optimiser state; a frozen feature-extractor forward fits in
        # ~40 GB, an end-to-end fine-tune needs materially more.
        estimated_vram_gb=40.0,
        max_sequence_length=262144,
        patch_size=2,
        mlp_ratio=12288 / 4096,  # = 3.0
        head_dim=128,
        num_key_value_heads=8,
        intermediate_size=12288,
        use_moe=True,
    ),
}

__all__ = ["_VARIANT_CONFIGS", "_VariantConfig"]
