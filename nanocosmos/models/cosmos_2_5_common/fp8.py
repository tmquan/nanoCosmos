"""Optional FP8 training conversion for the DiT (Blackwell+).

The Cosmos DiT is matmul-dominated, so casting its ``nn.Linear`` layers to FP8
(via torchao's ``float8`` training path) accelerates the attention / MLP GEMMs
on FP8-capable tensor cores (Hopper sm_89/sm_90, Blackwell sm_10x) and lowers
their activation/weight memory -- which in turn lets the DiT run *without*
gradient checkpointing.  The convolutional VAE decoder is **not** touched: FP8
conv is unsupported in these paths and the VAE is frozen anyway.

Only ``nn.Linear`` modules whose in/out features are both multiples of 16 (an
FP8 tensor-core tiling requirement) are converted; sensitive / small layers
(patch embed/unembed, positional & timestep embeddings, normalisation, the
latent projection, the final task projections) are excluded by name so the
numerically delicate parts stay in BF16.  Rowwise (per-row) scaling is the
default as it is the most accuracy-robust dynamic recipe for training.
"""

import logging
from typing import Optional

import torch.nn as nn

logger = logging.getLogger(__name__)

# Substrings (matched against the module's FQN relative to the DiT) that keep a
# Linear in BF16.  Two categories:
#
# 1. Numerically delicate / small: embeddings, norms, latent in/out
#    projections, timestep MLPs, task / vocab heads.
# 2. Joint-attention CONDITIONING stream (MMDiT ``add_*`` / ``to_add_out`` /
#    ``context_*``): this model drives only the generator (video/latent) tower
#    and feeds NULL conditioning, so the conditioning stream carries ~1 token.
#    FP8 ``_scaled_mm`` requires every GEMM dim (including the dynamic token
#    dim) to be divisible by 16, so a 1-token stream fails
#    ("trailing dimension ... divisible by 16 but got ... (Nx1)").  These
#    projections are tiny anyway -- FP8 buys nothing there -- so keep them BF16.
#    The big token-parallel ``self_attn.to_{q,k,v,out}`` + MLP linears (N =
#    batch * latent_tokens, 16-aligned) stay in FP8.
_FP8_EXCLUDE_SUBSTRINGS = (
    "embed",        # x_embedder / t_embedder / pos_embed / patch_embed ...
    "proj_in",
    "proj_out",
    "to_latent",
    "norm",
    "head",
    "final",
    "lm_head",
    "time",         # timestep / time_embedder MLPs
    # MMDiT conditioning-stream projections (small / 1-token -> not 16-aligned).
    "add_q",
    "add_k",
    "add_v",
    "add_out",
    "to_add_out",
    "context",
)


def _default_filter(module: nn.Module, fqn: str) -> bool:
    """Return True iff this Linear should be converted to FP8."""
    if not isinstance(module, nn.Linear):
        return False
    name = fqn.lower()
    if any(s in name for s in _FP8_EXCLUDE_SUBSTRINGS):
        return False
    # FP8 tensor cores require the contraction/output dims to be 16-aligned.
    if (module.in_features % 16 != 0) or (module.out_features % 16 != 0):
        return False
    return True


def apply_float8_to_dit(dit: nn.Module, recipe: str = "rowwise") -> int:
    """In-place convert eligible DiT ``nn.Linear`` layers to FP8 training.

    Args:
        dit: The transformer module (converted in place).
        recipe: torchao float8 recipe -- ``"rowwise"`` (default, most robust),
            ``"tensorwise"``, or ``"rowwise_with_gw_hp"``.

    Returns:
        Number of Linear modules converted to FP8.  ``0`` (and a warning) if
        torchao is unavailable, so training degrades gracefully to BF16.
    """
    try:
        from torchao.float8 import (
            Float8LinearConfig,
            convert_to_float8_training,
        )
    except Exception as exc:  # pragma: no cover - optional dependency
        logger.warning(
            "FP8 requested but torchao.float8 is unavailable (%s); "
            "falling back to BF16 for the DiT.", exc,
        )
        return 0

    try:
        config = Float8LinearConfig.from_recipe_name(recipe)
    except Exception as exc:
        logger.warning(
            "FP8 recipe %r not recognised by torchao (%s); using 'rowwise'.",
            recipe, exc,
        )
        config = Float8LinearConfig.from_recipe_name("rowwise")

    eligible = [
        name for name, m in dit.named_modules()
        if _default_filter(m, name)
    ]
    n_eligible = len(eligible)
    if n_eligible == 0:
        logger.warning("FP8: no eligible DiT Linear layers found; nothing converted.")
        return 0

    convert_to_float8_training(
        dit, module_filter_fn=_default_filter, config=config,
    )
    logger.info(
        "FP8 enabled on DiT: converted %d Linear layers (recipe=%s); "
        "embeddings / norms / proj_in / proj_out / heads kept in BF16.",
        n_eligible, recipe,
    )
    return n_eligible


__all__ = ["apply_float8_to_dit"]
