"""The shared :class:`Cosmos3OmniWrapper` (Edge / Nano / Super).

Cosmos 3 is NVIDIA's omnimodal world model.  Its generator is a
Mixture-of-Transformers (``Cosmos3OmniTransformer``) that jointly models
text / image / video / audio / action; for volumetric EM segmentation we
drive only its diffusion (video) tower as a feature extractor and feed null
conditioning for every other modality.

This wrapper is **tier-agnostic**: the Edge (4B) / Nano (16B) / Super (64B)
tiers share one architecture and differ only in scale, so each concrete tier
(:mod:`nanocosmos.models.cosmos_3_edge` / ``...cosmos_3_nano`` /
``...cosmos_3_super``) is a ~20-line subclass that sets ``_variant_configs``
(its single-entry registry) and the default ``variant``.

Almost all scaffolding (HuggingFace download, VAE encode, multi-layer feature
extraction via persistent block hooks, the unified-head decoder adapter,
freeze / gradient-checkpointing plumbing) is inherited verbatim from
:class:`nanocosmos.models.cosmos_2_5_common.wrapper_base._BaseCosmos25Wrapper`.
This file owns only the Cosmos 3-specific pieces:

* the diffusers class names to load (``Cosmos3OmniTransformer`` +
  the shared ``AutoencoderKLWan``);
* the HF-snapshot ignore list (skip the unused omni sub-towers);
* the ``latent_patch_size 2 -> 1`` repatch (blocky-artifact mitigation);
* the omni forward call (:meth:`_run_dit_forward`) + feature-capture hooks;
* the residual Wan2.2 VAE decoder wrap.

Because the base DiT is kept on the ``self.dit`` attribute -- exactly as
Predict / Transfer do -- every downstream convention that keys on
``model.dit.*`` keeps working unchanged: the optimiser param-group split, the
``freeze_dit_backbone`` schedule, and the
``ckpt_path_skip_prefixes=[model.dit.]`` warm-start filter.

References:
    - HuggingFace: nvidia/Cosmos3-Nano, nvidia/Cosmos3-Super
    - https://github.com/nvidia/cosmos
"""

import logging
from typing import Any, Callable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.utils.checkpoint

from nanocosmos.models.cosmos_3_common.wrapper_base import _BaseCosmos25Wrapper

logger = logging.getLogger(__name__)


def _checkpointed_decode_frame(
    forward_fn: Callable[..., torch.Tensor],
    frame: torch.Tensor,
    entry_cache: List[Any],
    first_chunk: bool,
) -> Tuple[torch.Tensor, List[Any]]:
    """Run one residual-Wan decode frame under activation checkpointing.

    The Wan decoder threads a stateful causal-conv cache (a mix of tensors and
    ``None`` / ``"Rep"`` sentinels) plus a positional ``feat_idx`` counter
    through every conv.  Activation checkpointing re-runs the frame in backward,
    so that state must be reproduced *exactly* or the recompute desyncs (wrong
    ``feat_idx`` -> ``IndexError``; wrong sentinel -> temporal-size mismatch).

    We therefore pass only the cache **tensors** through ``checkpoint`` -- so
    gradients flow across frames AND the tensors are saved for a bit-exact
    recompute -- and carry the non-tensor sentinel *structure* via a plain
    closure, re-merging the two inside the checkpointed function.  ``feat_idx``
    is reset to ``[0]`` inside the function so recompute starts from the same
    counter.  Returns ``(frame_output, updated_cache)``.
    """
    struct: List[Any] = ["__T__" if torch.is_tensor(c) else c for c in entry_cache]
    tensor_vals = [c for c in entry_cache if torch.is_tensor(c)]
    holder: dict = {}

    def run(fr: torch.Tensor, *tvals: torch.Tensor):
        cache: List[Any] = list(struct)
        ti = 0
        for k, slot in enumerate(cache):
            if slot == "__T__":
                cache[k] = tvals[ti]
                ti += 1
        conv_idx = [0]
        out = forward_fn(
            fr, feat_cache=cache, feat_idx=conv_idx, first_chunk=first_chunk,
        )
        holder["struct"] = [
            "__T__" if torch.is_tensor(c) else c for c in cache
        ]
        return (out, *[c for c in cache if torch.is_tensor(c)])

    results = torch.utils.checkpoint.checkpoint(
        run, frame, *tensor_vals, use_reentrant=False,
    )
    out = results[0]
    new_tensors = list(results[1:])
    updated_cache: List[Any] = []
    ti = 0
    for slot in holder["struct"]:
        if slot == "__T__":
            updated_cache.append(new_tensors[ti])
            ti += 1
        else:
            updated_cache.append(slot)
    return out, updated_cache


class _CacheTolerantIdentity(nn.Module):
    """Identity that tolerates the Wan cached-conv ``(x, cache)`` call.

    The shared decoder adapter swaps the Wan decoder's ``conv_out`` for
    ``nn.Identity`` (it consumes the pre-``conv_out`` features in its
    task head).  The residual Wan2.2 VAE's *cached* decode path calls
    ``conv_out(x, feat_cache[idx])`` with an extra positional argument,
    which plain ``nn.Identity`` rejects.  This passthrough ignores any
    extra args / kwargs and returns ``x`` unchanged, preserving the
    adapter's "stop before conv_out" contract on the cached path too.
    """

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        return x


class Cosmos3OmniWrapper(_BaseCosmos25Wrapper):
    """Cosmos 3 omni transformer as a volumetric EM feature extractor.

    A single unified task head produces ``[B, head_channels, D, H, W]``
    (default ``HEAD_CHANNELS = N_AFF + 2``: per-offset affinities + a
    foreground/semantic channel + a linear raw-reconstruction channel);
    the channel layout is owned by :mod:`nanocosmos.losses._common`.

    The depth axis of the EM volume maps to the model's temporal (video)
    axis, exactly as for the Cosmos 2.5 wrappers::

        EM volume  [B, C, D, H, W]  <->  video  [B, C, T, H, W]

    The Wan2.2-TI2V VAE compresses 16x spatially and 4x temporally into a
    48-channel latent grid, which the omni transformer's diffusion tower
    then processes.

    Subclasses MUST set :attr:`_variant_configs` (a ``variant -> _VariantConfig``
    registry) and SHOULD set a tier-appropriate default ``variant`` in their
    ``__init__``.

    Args:
        in_channels: Number of input channels (1 for EM volumes).
        head_channels: Unified head width (default ``HEAD_CHANNELS``).
        feature_size: Internal feature map channel count after projection.
        variant: Cosmos 3 variant key (resolved against ``_variant_configs``).
        dtype: Weight dtype.  Cosmos 3 is officially BF16-only; keep
            ``"bf16"`` unless you know what you are doing.
        pretrained: Auto-pull the tier's HF checkpoint on first instantiation.
        feature_layers: Omni-transformer block indices to extract features
            from.  Defaults (from the base class) to four evenly-spaced layers
            across the tier's layer stack.
    """

    # Subclasses populate this with their single-tier registry.
    _variant_configs: dict = {}

    def __init__(self, *args: Any, variant: str = "NANO", **kwargs: Any) -> None:
        super().__init__(*args, variant=variant, **kwargs)
        # Halve the DiT's spatial patchification (``latent_patch_size 2 -> 1``)
        # so the captured feature grid is the *full* VAE latent grid
        # (H/16 x W/16) instead of H/32 x W/32 -- this is the blocky-artifact
        # mitigation (4x finer spatial features).  It swaps two ``nn.Linear``
        # modules and mutates the DiT config, so it MUST run pre-FSDP
        # (construction is always pre-wrap).
        self._repatch_to_unit_latent_patch()
        # Cosmos 3 ships the residual Wan2.2-TI2V VAE, whose decoder the
        # shared decode path can only drive frame-by-frame; patch *this*
        # model's decoder forward accordingly (no-op for non-residual VAEs).
        self._maybe_wrap_residual_vae_decoder()
        # Keep the fp32 ``Timesteps`` embedding compatible with whatever
        # dtype ``time_embedder`` ends up at (bf16 after the base load, or
        # bf16 under FSDP MixedPrecision) without mutating the module.
        self._install_time_embedder_dtype_guard()

    # ------------------------------------------------------------------
    # Backbone selection
    # ------------------------------------------------------------------

    def _diffusers_transformer_cls_name(self) -> str:
        # The Cosmos 3 omni generator (``model_index.json`` ->
        # ``transformer: ["diffusers", "Cosmos3OmniTransformer"]``).
        return "Cosmos3OmniTransformer"

    def _diffusers_vae_cls_name(self) -> str:
        # Cosmos 3 ships the Wan2.2-TI2V VAE under the same diffusers
        # class the Cosmos 2.5 stack already uses (just a different
        # config: z_dim=48, 16x spatial / 4x temporal).
        return "AutoencoderKLWan"

    def _hf_ignore_patterns(self) -> Optional[List[str]]:
        # We only load ``transformer/`` + ``vae/`` and feed null
        # conditioning for text / image / audio, so skip the heavy omni
        # sub-towers and tokenizers (saves tens of GB per snapshot).
        from nanocosmos.models.cosmos_2_5_common.hf_loader import (
            _DEFAULT_IGNORE_PATTERNS,
        )

        return list(_DEFAULT_IGNORE_PATTERNS) + [
            "vision_encoder/*",
            "sound_tokenizer/*",
            "text_tokenizer/*",
        ]

    # ------------------------------------------------------------------
    # Spatial-resolution mitigation (latent_patch_size 2 -> 1)
    # ------------------------------------------------------------------

    def _repatch_to_unit_latent_patch(self) -> None:
        """Rebuild ``proj_in`` / ``proj_out`` for ``latent_patch_size = 1``.

        Cosmos 3's omni transformer patchifies the VAE latent's H/W axes by
        ``latent_patch_size`` (``p``) *before* the token stack: a ``p x p``
        spatial patch of the 48-channel latent is flattened to a
        ``patch_latent_dim = p*p*48`` vector and projected to ``hidden`` by
        ``proj_in`` (``proj_out`` does the inverse for the diffusion
        prediction).  With the published ``p = 2`` the feature grid is only
        ``H/(16*2) x W/(16*2) = H/32 x W/32`` (an 8x8 grid for a 256x256
        crop), which the decoder upsamples 32x -> the visible ~32px blocky
        pattern in the ``pred/*`` panels.

        Setting ``p = 1`` makes the feature grid the *full* VAE latent grid
        (``H/16 x W/16`` -> 16x16 for 256x256), 4x finer spatially.  But
        ``p`` is baked into the pretrained projection dims
        (``proj_in: 192 -> hidden``, ``proj_out: hidden -> 192``), so it is
        NOT a free config flag: both layers must be rebuilt at
        ``patch_latent_dim = 1*1*48 = 48``.

        To preserve the backbone's warm start we do **not** randomly
        re-initialise them -- we down-project the pretrained patch-embed by
        **averaging over the ``p x p`` spatial sub-positions** (the standard,
        scale-preserving way to retarget a patch embedding to a finer patch;
        cf. ViT patch-resize).  The flatten order in
        ``_patchify_and_pack_latents`` is ``(p_h, p_w, c)``
        (einsum ``cthpwq->thwpqc`` then ``reshape(-1, p*p*c)``), so the
        pretrained ``[hidden, p*p*c]`` weight reshapes to
        ``[hidden, p, p, c]`` and the mean over dims ``(p, p)`` yields the
        ``[hidden, c]`` unit-patch weight (bias is ``hidden``-dim, unchanged).
        ``proj_out`` is the mirror (``[p*p*c, hidden] -> [c, hidden]``; its
        ``[p*p*c]`` bias -> ``[c]``); note its output is *not* used by the
        feature path (we hook intermediate blocks), but it must stay
        shape-consistent so the model's (ignored) prediction tail does not
        crash.

        Finally ``register_to_config(latent_patch_size=1, patch_latent_dim=48)``
        makes ``_patchify_and_pack_latents`` / ``_unpatchify_and_unpack_latents``
        (which read ``self.config.latent_patch_size`` live) agree with the new
        projection dims.  Idempotent: a no-op if the DiT is already at ``p=1``
        (e.g. when resuming from a nanocosmos checkpoint saved post-repatch).

        Cost: ``num_vision_tokens`` (hence the DiT sequence length) grows 4x,
        so self-attention compute grows ~16x and activation memory ~4x.
        """
        dit = getattr(self, "dit", None)
        if dit is None:
            return
        # The standalone (pretrained=False) ``_StandaloneDiT3D`` has no diffusers
        # ``config`` / patch projections to repatch -- it already operates at the
        # full latent grid -- so this mitigation is a no-op there.
        cfg = getattr(dit, "config", None)
        if cfg is None:
            return
        p = int(getattr(cfg, "latent_patch_size", 2))
        if p == 1:
            return  # already unit-patch (e.g. resumed checkpoint)

        c = int(getattr(cfg, "latent_channel", 48))
        proj_in = getattr(dit, "proj_in", None)
        proj_out = getattr(dit, "proj_out", None)
        if not isinstance(proj_in, nn.Linear) or not isinstance(proj_out, nn.Linear):
            logger.warning(
                "Cosmos 3: proj_in/proj_out not nn.Linear; skipping "
                "latent_patch_size->1 repatch (blocky-artifact mitigation).",
            )
            return

        expected = p * p * c
        if proj_in.in_features != expected or proj_out.out_features != expected:
            logger.warning(
                "Cosmos 3: unexpected patch dims (proj_in.in=%d, proj_out.out=%d, "
                "expected %d=%d*%d*%d); skipping latent_patch_size->1 repatch to "
                "avoid corrupting the backbone.",
                proj_in.in_features, proj_out.out_features, expected, p, p, c,
            )
            return

        hidden = int(proj_in.out_features)
        device = proj_in.weight.device
        dtype = proj_in.weight.dtype

        with torch.no_grad():
            # proj_in:  [hidden, p, p, c] -> mean over (p, p) -> [hidden, c]
            w_in = (
                proj_in.weight.detach()
                .reshape(hidden, p, p, c)
                .mean(dim=(1, 2))
                .contiguous()
            )
            new_in = nn.Linear(c, hidden, bias=proj_in.bias is not None)
            new_in = new_in.to(device=device, dtype=dtype)
            new_in.weight.copy_(w_in)
            if proj_in.bias is not None:
                new_in.bias.copy_(proj_in.bias.detach())

            # proj_out: [p, p, c, hidden] -> mean over (p, p) -> [c, hidden]
            w_out = (
                proj_out.weight.detach()
                .reshape(p, p, c, hidden)
                .mean(dim=(0, 1))
                .contiguous()
            )
            new_out = nn.Linear(hidden, c, bias=proj_out.bias is not None)
            new_out = new_out.to(device=device, dtype=dtype)
            new_out.weight.copy_(w_out)
            if proj_out.bias is not None:
                new_out.bias.copy_(
                    proj_out.bias.detach().reshape(p, p, c).mean(dim=(0, 1)).contiguous(),
                )

        dit.proj_in = new_in
        dit.proj_out = new_out
        dit.register_to_config(latent_patch_size=1, patch_latent_dim=c)
        logger.info(
            "Cosmos 3: repatched latent_patch_size %d -> 1 (proj_in %d->%d, "
            "proj_out %d->%d, patch_latent_dim %d -> %d) -- feature grid is now "
            "the full H/16 x W/16 VAE latent grid (blocky-artifact mitigation). "
            "Sequence length grows ~%dx; pair with gradient checkpointing.",
            p, expected, hidden, hidden, expected, expected, c, p * p,
        )

    # ------------------------------------------------------------------
    # Omni forward
    # ------------------------------------------------------------------

    def _run_dit_forward(
        self,
        latent: torch.Tensor,
        timestep: torch.Tensor,
    ) -> None:
        """Drive the omni transformer's diffusion (video) tower over ``latent``.

        Features are captured by the persistent block hooks (see
        :meth:`_register_persistent_hooks` below), so this only has to
        make ``self.dit`` *run* a single denoising-step forward over one
        video latent with **null text / image / audio conditioning**.

        Unlike the Cosmos 2.5 ``CosmosTransformer3DModel`` (which takes a
        ``hidden_states`` latent + cross-attention text embeddings),
        ``Cosmos3OmniTransformer.forward`` consumes a *packed, unbatched*
        joint token stream and explicit index/position bookkeeping.  We
        reconstruct the minimal version of that stream the diffusers
        ``Cosmos3OmniPipeline`` builds for an unconditional video forward.
        """
        cfg = self.dit.config
        p = int(getattr(cfg, "latent_patch_size", self.cfg.patch_size))
        device = latent.device

        # The omni transformer is unbatched (it works on a single packed
        # ``[sequence_length, hidden]`` stream); the batch loop lives in
        # ``_extract_features_hook``, so here ``latent`` is one sample.
        if latent.shape[0] != 1:
            latent = latent[:1]
        grid_t = int(latent.shape[2])
        grid_h = (int(latent.shape[3]) + p - 1) // p
        grid_w = (int(latent.shape[4]) + p - 1) // p
        num_vision_tokens = grid_t * grid_h * grid_w

        # Minimal null-text prefix: a single causal "understanding" token.
        und_len = 1
        input_ids = torch.zeros(und_len, dtype=torch.long, device=device)
        text_indexes = torch.arange(und_len, dtype=torch.long, device=device)

        vision_sequence_indexes = torch.arange(
            und_len, und_len + num_vision_tokens, dtype=torch.long, device=device,
        )
        # No image conditioning => every frame (hence every token) is noisy,
        # so the MSE-read indices are just the full vision span.
        vision_mse_loss_indexes = vision_sequence_indexes.clone()
        vision_noisy_frame_indexes = [
            torch.arange(grid_t, dtype=torch.long, device=device),
        ]
        vision_token_shapes: List[Tuple[int, int, int]] = [(grid_t, grid_h, grid_w)]
        sequence_length = und_len + num_vision_tokens

        position_ids = self._build_position_ids(
            und_len, grid_t, grid_h, grid_w, device,
        )

        ts_val = float(timestep.flatten()[0].item()) if timestep.numel() else 0.0
        vision_timesteps = torch.full(
            (num_vision_tokens,), ts_val, dtype=torch.float32, device=device,
        )

        # The block hooks need to know which positions of the generation
        # stream (``hidden_states[und_len:]``) are the vision tokens.  In
        # this packing the generation half *is* the vision tokens, so the
        # selection is the identity ``arange(N)``; we store it explicitly
        # so the hook stays correct if the layout ever grows extra
        # generation tokens.
        self._c3_gen_vision_index = vision_sequence_indexes - und_len

        # Disable autocast *at the DiT chokepoint* (not just in ``forward``).
        # The omni DiT is reached from multiple entry points -- the training
        # ``forward`` AND the ``wan_decoder_output`` diagnostic -- so pinning
        # the disable here covers every caller; the backbone is already
        # natively bf16 so the outer autocast is redundant anyway.
        with torch.autocast(device_type=device.type, enabled=False), \
                self._dit_forward_without_ckpt_when_eval():
            self.dit(
                input_ids=input_ids,
                text_indexes=text_indexes,
                position_ids=position_ids,
                und_len=und_len,
                sequence_length=sequence_length,
                vision_tokens=[latent.to(self._dtype)],
                vision_token_shapes=vision_token_shapes,
                vision_sequence_indexes=vision_sequence_indexes,
                vision_mse_loss_indexes=vision_mse_loss_indexes,
                vision_timesteps=vision_timesteps,
                vision_noisy_frame_indexes=vision_noisy_frame_indexes,
            )

    def _build_position_ids(
        self,
        und_len: int,
        grid_t: int,
        grid_h: int,
        grid_w: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Build ``[3, und_len + grid_t*grid_h*grid_w]`` unified-3D-mRoPE ids.

        Mirrors the diffusers ``Cosmos3OmniPipeline`` position-id
        construction for the null-conditioning case (integer positions,
        no FPS modulation):

        * text tokens share one monotonically-increasing id across all
          three (T, H, W) axes;
        * VAE vision tokens get a ``(t, h, w)`` grid, the temporal axis
          offset past the text prefix by
          ``unified_3d_mrope_temporal_modality_margin`` so text and vision
          never collide in rotary phase.  Spatial ids reset to 0 when
          ``unified_3d_mrope_reset_spatial_ids`` is set (it is for Nano).
        """
        cfg = self.dit.config
        reset_spatial = bool(
            getattr(cfg, "unified_3d_mrope_reset_spatial_ids", True),
        )
        margin = int(
            getattr(cfg, "unified_3d_mrope_temporal_modality_margin", 15000),
        )

        text_ids = (
            torch.arange(und_len, dtype=torch.long, device=device)
            .unsqueeze(0)
            .expand(3, -1)
        )

        offset = und_len + margin
        t_index = (
            torch.arange(grid_t, device=device)
            .view(-1, 1)
            .expand(-1, grid_h * grid_w)
            .flatten()
            + offset
        )
        h_index = (
            torch.arange(grid_h, device=device)
            .view(1, -1, 1)
            .expand(grid_t, -1, grid_w)
            .flatten()
        )
        w_index = (
            torch.arange(grid_w, device=device)
            .view(1, 1, -1)
            .expand(grid_t, grid_h, -1)
            .flatten()
        )
        if not reset_spatial:
            h_index = h_index + offset
            w_index = w_index + offset
        vision_ids = torch.stack([t_index, h_index, w_index], dim=0).to(torch.long)

        return torch.cat([text_ids, vision_ids], dim=1)

    # ------------------------------------------------------------------
    # Feature-capture overrides
    #
    # The omni transformer interleaves a causal text "understanding"
    # stream and a full "generation" stream: each block returns
    # ``(und_seq, gen_seq)`` rather than a single tensor, and it never
    # patchifies the temporal axis.  Both facts break the base class's
    # capture assumptions, so Cosmos 3 overrides the two hook-side methods.
    # ------------------------------------------------------------------

    def _register_persistent_hooks(self) -> None:
        """Register block hooks that capture the *vision* generation tokens.

        Cosmos 3 decoder layers return ``(und_seq, gen_seq)``; the vision
        latent lives in ``gen_seq`` (``hidden_states[und_len:]``).  We
        therefore capture ``output[1]`` and select the vision span, as a
        ``[num_vision_tokens, hidden]`` tensor (the batch axis is added
        back when samples are stacked in :meth:`_extract_features_hook`).
        """
        self._hook_buffer: List[torch.Tensor] = []
        self._hook_handles: List[Any] = []
        self._hook_block_container = None
        self._hooks_active = False
        self._c3_gen_vision_index: Optional[torch.Tensor] = None

        for attr in ("transformer_blocks", "blocks", "layers"):
            if hasattr(self.dit, attr):
                # Unregistered reference (see base ``_register_persistent_hooks``):
                # a normal assignment would alias the DiT block list as a second
                # submodule, which makes FSDP's recursive auto-wrap double-wrap
                # the same decoder layer ("already wrapped by FSDP").
                object.__setattr__(
                    self, "_hook_block_container", getattr(self.dit, attr),
                )
                break

        if self._hook_block_container is None:
            return

        def _make_hook(_idx: int):
            def hook_fn(_module: nn.Module, _input: Any, output: Any) -> None:
                if not self._hooks_active:
                    return
                # Vision tokens are the generation stream (``output[1]``).
                if isinstance(output, (tuple, list)) and len(output) > 1:
                    gen_seq = output[1]
                elif isinstance(output, (tuple, list)):
                    gen_seq = output[0]
                else:
                    gen_seq = output
                idx = self._c3_gen_vision_index
                if idx is not None and idx.numel() != gen_seq.shape[0]:
                    gen_seq = gen_seq.index_select(0, idx)
                if self._hook_should_detach():
                    gen_seq = gen_seq.detach()
                # ``clone()`` is REQUIRED, not just an optimisation: under
                # ``torch.compile(mode="reduce-overhead")`` the DiT runs in
                # CUDA graphs with reused static buffers, so ``gen_seq`` is a
                # view into graph-owned memory.  Cloning lifts the capture out
                # of the cudagraph pool.
                self._hook_buffer.append(gen_seq.clone())
            return hook_fn

        for idx in self._feature_layers:
            if idx < len(self._hook_block_container):
                h = self._hook_block_container[idx].register_forward_hook(
                    _make_hook(idx),
                )
                self._hook_handles.append(h)

    def _extract_features_hook(
        self,
        latent: torch.Tensor,
        timestep: torch.Tensor,
        d_tok: int,
        h_tok: int,
        w_tok: int,
    ) -> torch.Tensor:
        """Capture omni features, fixing the token grid and the batch axis.

        Two Cosmos-3-specific corrections relative to the base method:

        * **Token grid.** The base class derives ``(d_tok, h_tok, w_tok)``
          assuming all three latent axes are patchified by ``patch_size``.
          Cosmos 3 patchifies only H/W (by ``latent_patch_size``) and
          leaves the temporal axis intact, so we recompute the grid as
          ``(D, ceil(H/p), ceil(W/p))`` from the (already padded) latent.
        * **Batch.** The omni transformer is unbatched, so we run one
          forward per sample and stack the per-layer ``[N, hidden]``
          captures into ``[B, N, hidden]`` for the projector.
        """
        if self._hook_block_container is None:
            return super()._extract_features_hook(
                latent, timestep, d_tok, h_tok, w_tok,
            )

        p = int(getattr(self.dit.config, "latent_patch_size", self.cfg.patch_size))
        D_p, H_p, W_p = (
            int(latent.shape[-3]),
            int(latent.shape[-2]),
            int(latent.shape[-1]),
        )
        grid_d = D_p
        grid_h = (H_p + p - 1) // p
        grid_w = (W_p + p - 1) // p
        B = latent.shape[0]
        num_layers = len(self._feature_layers)

        # Grad must be enabled whenever any DiT-side branch is trainable.
        any_trainable = self._any_trainable()
        ctx = torch.enable_grad() if any_trainable else torch.no_grad()

        per_sample_layers: List[List[torch.Tensor]] = []
        self._hooks_active = True
        try:
            with ctx:
                for b in range(B):
                    self._hook_buffer.clear()
                    self._run_dit_forward(latent[b : b + 1], timestep)
                    per_sample_layers.append(list(self._hook_buffer))
        finally:
            self._hooks_active = False
            self._hook_buffer.clear()

        collected: List[torch.Tensor] = []
        for layer in range(num_layers):
            feats = [
                ps[layer] for ps in per_sample_layers if layer < len(ps)
            ]
            if len(feats) == B:
                collected.append(torch.stack(feats, dim=0))  # [B, N, hidden]

        # Defensive: keep the projector input length == #feature layers
        # even if a hook somehow failed to fire.
        while 0 < len(collected) < num_layers:
            collected.append(collected[-1])

        # Match the projector's *runtime* parameter dtype rather than forcing
        # fp32: the projector is fp32 in the eager / DDP-bf16-mixed path but
        # bf16 under FSDP ``MixedPrecision`` (bf16-true), so a hard ``.float()``
        # would mismatch the sharded bf16 projector weights.
        proj_param = next(self.feature_projector.parameters(), None)
        proj_dtype = proj_param.dtype if proj_param is not None else torch.float32
        collected = [f.to(proj_dtype) for f in collected]
        return self.feature_projector(collected, grid_d, grid_h, grid_w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Full forward: encode -> omni DiT features -> unified head.

        Differs from the shared base forward in two Cosmos-3-only ways,
        both confined to this override (shared code is untouched):

        1. **Autocast is disabled for the whole wrapper forward.**  The
           Cosmos 3 graph is a *deliberately* mixed-dtype stack -- bf16
           omni DiT + bf16 Wan VAE, fp32 feature projector / ``to_latent``
           / task head, and an fp32-pinned ``time_embedder`` -- whose
           dtype boundaries we manage explicitly with casts.  Under an
           *outer* bf16 autocast the autocast "promote" ops hit
           ``at::autocast::prioritize`` -> *"Unexpected floating
           ScalarType"*.  The backbone is already natively bf16, so the
           outer autocast is redundant; disabling it makes the autocast
           path run byte-for-byte like the verified eager path.

        2. **fp32 head dtype shim.**  The projector / ``to_latent`` / head
           run in fp32, but the shared ``_encode_and_extract`` casts the
           features back to the *input* dtype; a bf16 input would feed
           bf16 activations into the fp32 head.  We upcast the features to
           the head dtype (a no-op for the usual fp32-input path).
        """
        with torch.autocast(device_type=x.device.type, enabled=False):
            features, target_size = self._encode_and_extract(x)
            head_param = next(self.decoder_adapter.head.parameters(), None)
            if head_param is not None and features.dtype != head_param.dtype:
                features = features.to(head_param.dtype)
            return self.decoder_adapter(
                features, target_size=target_size, image=x,
            )

    def _install_time_embedder_dtype_guard(self) -> None:
        """Cast the fp32 timestep embedding to ``time_embedder``'s dtype.

        ``Cosmos3OmniTransformer.forward`` runs
        ``self.time_embedder(self.time_proj(timesteps))`` and diffusers'
        ``Timesteps``/``get_timestep_embedding`` ALWAYS returns fp32.  If
        ``time_embedder``'s weights are bf16 (they are after the base load,
        and under FSDP ``MixedPrecision``) the matmul raises a Float/BFloat16
        mismatch.  A ``forward_pre_hook`` that casts the (activation) input to
        the module's current weight dtype fixes this for every precision /
        sharding mode without ever mutating the parameter dtype.
        """
        if getattr(self, "_c3_time_embedder_guarded", False):
            return
        time_embedder = getattr(self.dit, "time_embedder", None)
        if time_embedder is None:
            return

        def _cast_inputs(module: nn.Module, args: Any) -> Any:
            if not args:
                return args
            weight = next(module.parameters(), None)
            first = args[0]
            if (
                weight is not None
                and isinstance(first, torch.Tensor)
                and first.dtype != weight.dtype
            ):
                return (first.to(weight.dtype),) + tuple(args[1:])
            return args

        time_embedder.register_forward_pre_hook(_cast_inputs)
        self._c3_time_embedder_guarded = True

    # ------------------------------------------------------------------
    # Decoder override (residual Wan2.2 VAE)
    # ------------------------------------------------------------------

    def _maybe_wrap_residual_vae_decoder(self) -> None:
        """Make the shared decode path work with the residual Wan2.2 VAE.

        ``_DecoderAdapter3D._decode_body`` decodes by calling
        ``decoder_body(latent)`` exactly once with ``feat_cache=None``.
        That is correct for the *non-residual* Wan VAE used by Cosmos 2.5,
        but the Wan2.2-TI2V VAE Cosmos 3 ships (``is_residual=True``) only
        performs temporal upsampling on its *cached* path, while its
        residual ``avg_shortcut`` (``DupUp3D``) always doubles the
        temporal axis -- so a single un-cached call hits a ``t`` vs
        ``2t`` size mismatch (e.g. ``4`` vs ``8``).

        Rather than touch the shared adapter, we wrap *this* model's
        decoder ``forward`` so an un-cached call internally performs the
        VAE's own frame-by-frame cached decode.  The wrap is gated on
        ``is_residual`` and a diffusers backend, so non-residual VAEs
        (Cosmos 2.5) are left byte-for-byte untouched.
        """
        if getattr(self, "_c3_residual_decode_wrapped", False):
            return
        if getattr(self, "_backend", None) != "diffusers":
            return
        vae_ref = getattr(self, "_vae_ref", None)
        if not vae_ref:
            return
        vae_config = getattr(vae_ref[0], "config", None)
        if not bool(getattr(vae_config, "is_residual", False)):
            return
        adapter = getattr(self, "decoder_adapter", None)
        decoder = getattr(adapter, "decoder_body", None)
        if decoder is None or not hasattr(decoder, "forward"):
            return

        # Block-level decoder-body checkpointing (set up by the base
        # ``enable_gradient_checkpointing``) is INCOMPATIBLE with the residual
        # VAE's stateful frame-by-frame cached decode below -- recompute would
        # desync the causal-conv cache / ``feat_idx`` counter.  Opt out of it
        # and undo any wrappers already applied; this path checkpoints per
        # frame instead (see ``_cached_chunked_forward``).
        self._skip_decoder_body_checkpoint = True
        for m in self._decoder_body_modules():
            self._unwrap_forward(m)

        # Cache length: one slot per causal conv plus one for the (already
        # swapped-to-Identity) ``conv_out`` site, which its forward still
        # indexes unconditionally.  (Class checked by name to avoid importing
        # a private diffusers symbol.)
        num_causal_convs = sum(
            1 for m in decoder.modules()
            if type(m).__name__ == "WanCausalConv3d"
        )
        cache_len = num_causal_convs + 1

        # The adapter replaced ``conv_out`` with a plain ``nn.Identity``;
        # the residual VAE's cached path calls it as ``conv_out(x, cache)``,
        # so make it tolerate (and ignore) the extra cache argument.
        if isinstance(getattr(decoder, "conv_out", None), nn.Identity):
            decoder.conv_out = _CacheTolerantIdentity()

        original_forward = decoder.forward

        def _cached_chunked_forward(
            x: torch.Tensor,
            feat_cache: Optional[List[Any]] = None,
            feat_idx: Optional[List[int]] = None,
            first_chunk: bool = False,
        ) -> torch.Tensor:
            # Defer to the native forward when a caller already manages the
            # causal-conv cache.
            if feat_cache is not None:
                return original_forward(
                    x,
                    feat_cache=feat_cache,
                    feat_idx=feat_idx,
                    first_chunk=first_chunk,
                )
            # Activation-checkpoint each frame in training: the un-checkpointed
            # loop retains every frame's full-resolution decoder activations for
            # backward (the dominant memory cost of this recipe).  Gated on
            # autograd being enabled (``torch.utils.checkpoint`` cannot wrap the
            # ``inference_mode`` tensors Lightning uses for eval) and on
            # gradient checkpointing being active.
            use_ckpt = (
                getattr(self, "_gradient_checkpointing", False)
                and torch.is_grad_enabled()
            )
            feat_map: List[Any] = [None] * cache_len
            decoded: Optional[torch.Tensor] = None
            for i in range(x.shape[2]):
                frame = x[:, :, i : i + 1, :, :]
                first = i == 0
                if use_ckpt:
                    out_i, feat_map = _checkpointed_decode_frame(
                        original_forward, frame, feat_map, first,
                    )
                else:
                    conv_idx = [0]
                    out_i = original_forward(
                        frame,
                        feat_cache=feat_map,
                        feat_idx=conv_idx,
                        first_chunk=first,
                    )
                decoded = (
                    out_i if decoded is None
                    else torch.cat([decoded, out_i], dim=2)
                )
            return decoded

        decoder.forward = _cached_chunked_forward
        self._c3_residual_decode_wrapped = True
        logger.info(
            "Cosmos 3: wrapped residual Wan2.2 VAE decoder with frame-by-frame "
            "cached decode (%d causal convs) for the feature-extraction "
            "decode path.",
            num_causal_convs,
        )


__all__ = ["Cosmos3OmniWrapper", "_CacheTolerantIdentity"]
