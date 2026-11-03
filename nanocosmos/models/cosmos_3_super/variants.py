"""Variant registry for Cosmos3-Super (64B total / 32B dense generator tower).

Numbers mirrored verbatim from the published HF config
(``nvidia/Cosmos3-Super`` ``transformer/config.json``,
``Cosmos3OmniTransformer``): ``hidden_size=5120``, ``num_hidden_layers=64``,
``num_attention_heads=64``, ``num_key_value_heads=8`` (GQA), ``head_dim=128``,
``intermediate_size=25600`` (MLP ratio 5.0), ``latent_channel=48``,
``latent_patch_size=2``, ``max_position_embeddings=262144``, ``use_moe=true``.

Note the Qwen3-VL-style decoupling: ``num_heads * head_dim = 64 * 128 = 8192``
which is NOT equal to ``hidden_size=5120`` (``q_proj`` projects to 8192, then
``o_proj`` back to 5120).  The VAE is the same Wan2.2-TI2V VAE as the other
tiers (``z_dim=48``, 16x spatial / 4x temporal).

Super initialises from pre-trained Qwen3-VL 32B and is loadable from
HuggingFace.  Only BF16 is officially supported.  This is a very large
backbone (~32 GB dense tower @ bf16 before activations); an end-to-end
fine-tune needs multi-GPU sharding (FSDP).
"""

from typing import Dict

from nanocosmos.models.cosmos_3_common.variants import _VariantConfig

_VARIANT_CONFIGS: Dict[str, _VariantConfig] = {
    "SUPER": _VariantConfig(
        hf_repo_id="nvidia/Cosmos3-Super",
        hf_revision="main",
        hidden_dim=5120,
        num_layers=64,
        num_heads=64,
        latent_channels=48,
        spatial_compression=16,
        temporal_compression=4,
        # 32B dense tower @ bf16 ~= 64 GB of weights before activations /
        # optimiser state; assume FSDP sharding for any training.
        estimated_vram_gb=80.0,
        max_sequence_length=262144,
        patch_size=2,
        mlp_ratio=25600 / 5120,  # = 5.0
        head_dim=128,
        num_key_value_heads=8,
        intermediate_size=25600,
        use_moe=True,
    ),
}

__all__ = ["_VARIANT_CONFIGS", "_VariantConfig"]
