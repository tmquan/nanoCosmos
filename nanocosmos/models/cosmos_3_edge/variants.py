"""Variant registry for Cosmos3-Edge (4B total / 2B dense generator tower).

Cosmos3-Edge has released weights on HuggingFace (``nvidia/Cosmos3-Edge``,
gated -- needs an HF token). The ``EDGE`` variant loads them directly.

diffusers 0.42.1 reads ``hidden_act: relu2`` and
``qk_norm_for_text: false``, so the direct load reports no missing keys.
``fill_missing_parent_repo_id`` is unset. A legacy ``reduce_from_parent=True``
path remains for comparisons; the shipped configs leave it off.

Older diffusers (0.39) always built a gated-SiLU MLP and text QK norms the
Edge checkpoint does not contain. Do not run this variant on that build.

Geometry, verified against the released ``transformer/config.json`` (the
report's Qwen3-1.7B guess used FFN 6144; the checkpoint is wider):

    hidden_size=2048, num_hidden_layers=28, num_attention_heads=16,
    num_key_value_heads=8 (GQA), head_dim=128, intermediate_size=9216

The VAE is Edge's own Wan2.2-TI2V (``z_dim=48``, 16x spatial / 4x temporal).

Set ``model.pretrained=false`` to skip the download and train the Edge
geometry from scratch (standalone DiT + learned conv tokenizer).
"""

from typing import Dict

from nanocosmos.models.cosmos_3_common.variants import _VariantConfig

_VARIANT_CONFIGS: Dict[str, _VariantConfig] = {
    "EDGE": _VariantConfig(
        # Direct load of Edge's own transformer + Wan2.2 VAE.
        hf_repo_id="nvidia/Cosmos3-Edge",
        hf_revision="main",
        reduce_from_parent=False,
        # fill_missing_parent_repo_id stays unset: the pinned diffusers
        # build matches the checkpoint, so the Nano fill is a no-op.
        hidden_dim=2048,
        num_layers=28,
        num_heads=16,
        latent_channels=48,
        spatial_compression=16,
        temporal_compression=4,
        # ~2B dense generator tower + VAE; comfortably fits a frozen-VAE
        # forward under ~16 GB.
        estimated_vram_gb=14.0,
        max_sequence_length=131072,
        patch_size=2,
        mlp_ratio=9216 / 2048,  # = 4.5; verified against the released
                                # nvidia/Cosmos3-Edge transformer/config.json
                                # (supersedes the earlier Qwen3-1.7B-inferred 6144/3.0)
        head_dim=128,
        num_key_value_heads=8,
        intermediate_size=9216,  # verified against the released transformer/config.json
        use_moe=True,
    ),
}

__all__ = ["_VARIANT_CONFIGS", "_VariantConfig"]
