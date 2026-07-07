"""Variant registry for Cosmos3-Edge (4B total / 2B dense generator tower).

Cosmos3-Edge has **released weights** on HuggingFace (``nvidia/Cosmos3-Edge``,
first published ~2026-07-01). The ``EDGE`` variant loads them **directly**
(``hf_repo_id="nvidia/Cosmos3-Edge"``) and fills the small subset of
parameters the checkpoint is missing (a checkpoint/library architecture gap,
see "WHY SOME WEIGHTS ARE MISSING" below) from the released Cosmos3-Nano by
truncation, rather than either (a) approximating the WHOLE model via
Nano-reduction (the old default, which discarded Edge's own real weights
everywhere they DO exist) or (b) leaving the gap at fresh init (a plain
direct load, which would corrupt the generation-pathway features this
wrapper extracts).

Per the Cosmos 3 technical report (arXiv 2606.02800) Edge is the only tier
trained *from scratch*, with the dual-tower generator geometry (verified
against the released ``nvidia/Cosmos3-Edge`` repo's ``transformer/config.json``,
superseding the report-inferred guess below):

    hidden_size=2048, num_hidden_layers=28, num_attention_heads=16,
    num_key_value_heads=8 (GQA), head_dim=128, intermediate_size=9216

Provenance (high confidence): each Cosmos 3 tier's generator tower is *close*
to the corresponding **Qwen3 dense** backbone shape, per the published
configs:

    Nano  == Qwen3-8B   (4096 / 36 / 32 / 8 / head_dim 128 / inter 12288)
    Super == Qwen3-32B  (5120 / 64 / 64 / 8 / head_dim 128 / inter 25600)
    Edge  ~= Qwen3-1.7B (2048 / 28 / 16 / 8 / head_dim 128 / inter  6144)

The report's stated Edge dims (2048 / 28 / 16 / 8) match Qwen3-1.7B exactly,
but the released checkpoint's actual FFN width is ``intermediate_size=9216``
-- WIDER than the naive Qwen3-1.7B match (6144) would suggest.

WHY SOME WEIGHTS ARE MISSING FROM THE DIRECT LOAD (as of 2026-07-07)
----------------------------------------------------------------------
The released ``nvidia/Cosmos3-Edge`` checkpoint's ``transformer/config.json``
sets ``hidden_act: "relu2"`` and ``qk_norm_for_text: false`` -- i.e. a
NON-gated, ReLU^2-activated MLP and no per-head QK RMSNorm on the
"understanding" attention pathway. But the installed ``diffusers`` package's
``Cosmos3VLTextMoTDecoderLayer`` (``transformer_cosmos3.py``) does not yet
read either flag: it unconditionally builds a gated-SiLU MLP
(``down_proj(silu(gate_proj(x)) * up_proj(x))``, hardcoded ``nn.SiLU()``) for
BOTH ``mlp`` (understanding) AND ``mlp_moe_gen`` (**generation/video** --
what this wrapper's forward hooks capture as backbone features) and always
constructs ``self_attn.norm_q``/``norm_k``. Confirmed via the actual
safetensors index: ``nvidia/Cosmos3-Edge`` genuinely has NO
``mlp.gate_proj`` / ``mlp_moe_gen.gate_proj`` / ``self_attn.norm_q`` /
``self_attn.norm_k`` keys at all (unlike Nano's checkpoint, which has all
four).

``diffusers.from_pretrained(..., output_loading_info=True)`` reports these as
``missing_keys`` rather than raising, so ``Cosmos3EdgeWrapper`` catches them
in :meth:`_post_load_diffusers` and fills each one -- by depth-remapping
Edge's 28 layers onto Nano's 36 (the same evenly-spaced map used for full
reduction) and truncating Nano's (larger) tensor down to Edge's shape -- via
:func:`nanocosmos.models.cosmos_3_common.reduce.fill_missing_from_parent`.
``self_attn.norm_q``/``norm_k`` are ``head_dim``-shaped (128), identical
across every tier, so those specifically are copied verbatim (no truncation
needed); only the ``gate_proj`` matrices (``[intermediate_size, hidden_dim]``)
are actually truncated. This is still a structured warm start for those ~4
tensors per layer, not trained Edge weights -- but it now only touches the
small gap instead of the whole model, and the other ~95% of Edge's
parameters (attention q/k/v/o, up_proj/down_proj, all norms, embeddings,
patchifier, MoE router, ...) are the checkpoint's OWN trained weights.
Revisit (drop the fill) once diffusers ships support for Edge's
``hidden_act``/``qk_norm_for_text`` config flags.

Everything else (VAE, ``latent_patch_size``, ``head_dim``, KV heads, Cosmos
``rope_theta``, mRoPE section ``[24, 20, 20]``) is shared with Nano/Super --
the same Wan2.2-TI2V VAE (``z_dim=48``, 16x spatial / 4x temporal), confirmed
identical in the released repo's ``vae/config.json`` -- and here is loaded
straight from Edge's OWN checkpoint (no fill needed; the VAE is complete).

Set ``model.pretrained=false`` in the config to skip the download entirely
and train the Edge geometry from scratch (standalone DiT + learned conv
tokenizer, no Wan2.2 VAE).
"""

from typing import Dict

from nanocosmos.models.cosmos_3_common.variants import _VariantConfig

_VARIANT_CONFIGS: Dict[str, _VariantConfig] = {
    "EDGE": _VariantConfig(
        # Direct load: Edge's OWN released checkpoint (transformer + its own
        # Wan2.2 VAE, identical to Nano/Super's).  See the module docstring
        # ("WHY SOME WEIGHTS ARE MISSING") for the small parameter subset
        # this checkpoint lacks and how it's filled below.
        hf_repo_id="nvidia/Cosmos3-Edge",
        hf_revision="main",
        reduce_from_parent=False,
        # Parent tier to source the missing gate_proj / norm_q / norm_k
        # weights from (by depth-remap + truncation) -- see
        # Cosmos3EdgeWrapper._post_load_diffusers.
        fill_missing_parent_repo_id="nvidia/Cosmos3-Nano",
        fill_missing_parent_revision="main",
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
