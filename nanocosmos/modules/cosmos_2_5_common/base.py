"""
Cosmos 2.5 Lightning-module base for 3-D volumetric segmentation.

Specialisation of :class:`~nanocosmos.modules.base.BaseCircuitModule`
for the Cosmos 2.5 backbone (Cosmos-Predict).  The base
``BaseCircuitModule`` owns training, evaluation and logging;
this module adds:

* HuggingFace token handling (kept out of ``save_hyperparameters``)
* static freeze of the VAE encoder, DiT backbone and VAE decoder
  applied once at construction via the model wrapper's ``freeze_*``
  kwargs
* fused-norm NaN/Inf gradient guard (one device->host sync per step
  via :func:`torch._foreach_norm` rather than one per parameter --
  see :meth:`BaseCosmosModule.configure_gradient_clipping`)
* backbone-specific AdamW learning rate and param-group split
  (``model.dit.*`` at ``dit_backbone_lr`` vs everything else at ``lr``)
  in :meth:`configure_optimizers`; empty groups are filtered out
  before AdamW sees them.

Only the **automatic** training mode is supported (predict from the
volume alone).  Point-prompt / proofread training is a Vista-only path.

Subclasses with architecture-specific model kwargs override
:meth:`_extra_model_kwargs` to add them; the base ``_build_model`` only
forwards the kwargs the shared ``_BaseCosmos25Wrapper`` accepts.
"""

import logging
import os
from typing import Any, Dict, Optional

import torch

from nanocosmos.losses import HEAD_CHANNELS
from nanocosmos.modules.base import BaseCircuitModule

logger = logging.getLogger(__name__)

# Log level for the per-step non-finite-gradient guard in
# ``BaseCosmosModule.configure_gradient_clipping``.  Default DEBUG keeps
# it silent under the usual INFO/WARNING root logger -- bf16-mixed +
# ``gradient_clip_val=1.0`` triggers a handful of harmless guard fires
# per run, each previously duplicated across DDP ranks.  Override via
# ``NANOCOSMOS_GRAD_GUARD_LOG=warning`` (or ``info`` / ``error``) when
# investigating loss spikes.
_GRAD_GUARD_LOG_FN = getattr(
    logger,
    os.environ.get("NANOCOSMOS_GRAD_GUARD_LOG", "debug").lower(),
    logger.debug,
)


class BaseCosmosModule(BaseCircuitModule):
    """Abstract base for Cosmos-Predict / Cosmos-Transfer 3-D modules.

    Subclasses **must** define :attr:`_model_cls` and :attr:`_loss_cls`.

    Args:
        model_config: Forwarded to ``_model_cls`` (see
            :class:`nanocosmos.models.cosmos_2_5_common.wrapper_base._BaseCosmos25Wrapper`).
        optimizer_config: Optimizer / scheduler settings.
        loss_config: Forwarded as ``**loss_config`` to ``_loss_cls``.
        training_config: Training behaviour (mutex_watershed, freeze schedule, ...).
    """

    _SPATIAL_DIMS = 3

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def __init__(
        self,
        model_config: Optional[Dict[str, Any]] = None,
        optimizer_config: Optional[Dict[str, Any]] = None,
        loss_config: Optional[Dict[str, Any]] = None,
        training_config: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        model_config = dict(model_config or {})
        # ``hf_token`` is intentionally not persisted via save_hyperparameters.
        hf_token = model_config.pop("hf_token", None)
        self.save_hyperparameters()
        if hf_token is not None:
            model_config["hf_token"] = hf_token

        super().__init__(
            model_config=model_config,
            optimizer_config=optimizer_config,
            loss_config=loss_config,
            training_config=training_config,
            **kwargs,
        )

        if self.model._backbone_loaded and self.model.vae_encoder is not None:
            self.model._fallback_down.requires_grad_(False)

        logger.info(
            "Validation agglomerator: %s",
            repr(self.agglomerator),
        )

    def _build_model(self, model_config: Dict[str, Any]) -> torch.nn.Module:
        model = self._model_cls(
            in_channels=model_config.get("in_channels", 1),
            head_channels=model_config.get("head_channels", HEAD_CHANNELS),
            feature_size=model_config.get("feature_size", 64),
            variant=model_config.get("variant", "2B"),
            dtype=model_config.get("dtype", "bf16"),
            pretrained=model_config.get("pretrained", True),
            freeze_dit_backbone=model_config.get("freeze_dit_backbone", False),
            freeze_vae_decoder=model_config.get("freeze_vae_decoder", False),
            freeze_vae_encoder=model_config.get("freeze_vae_encoder", True),
            gradient_checkpointing=model_config.get("gradient_checkpointing", False),
            feature_layers=model_config.get("feature_layers"),
            cache_dir=model_config.get("cache_dir"),
            hf_token=model_config.get("hf_token"),
            dropout=model_config.get("dropout", 0.0),
            decode_chunk=model_config.get("decode_chunk", 16),
            vae_symmetrize_z=model_config.get("vae_symmetrize_z", False),
            **self._extra_model_kwargs(model_config),
        )
        # Optional FP8 (Blackwell+) on the DiT's matmul-heavy Linear layers.
        # Module-swap (torchao float8) decoupled from Lightning precision, so
        # ``training.precision: bf16-true`` is unchanged; the conv VAE is never
        # touched.  Done here (pre-DDP wrap, post all backbone surgery).
        if model_config.get("fp8", False):
            from nanocosmos.models.cosmos_2_5_common.fp8 import apply_float8_to_dit

            dit = getattr(model, "dit", None)
            if dit is not None:
                apply_float8_to_dit(
                    dit, recipe=str(model_config.get("fp8_recipe", "rowwise")),
                )
        return model

    def _extra_model_kwargs(
        self, model_config: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Subclass hook: forward arch-specific kwargs to ``_model_cls``.

        Cosmos-Predict has no extras and inherits the empty-dict default;
        a future backbone with extra constructor kwargs would override this.
        """
        return {}

    # ------------------------------------------------------------------
    # Scheduled DiT thaw (``model.freeze_dit_backbone: N`` warm-up)
    # ------------------------------------------------------------------

    def on_train_epoch_start(self) -> None:
        """Thaw the DiT backbone when the configured warm-up epoch arrives.

        Wired to ``model.freeze_dit_backbone: N`` (non-negative int):
        the DiT stays frozen for epochs ``0..N-1`` and is unfrozen at
        the start of epoch ``N``.  ``configure_optimizers`` already
        included the DiT params in the optimizer's backbone group up
        front (see ``include_frozen_dit`` there), so the only work
        here is flipping ``requires_grad`` -- AdamW lazily allocates
        moment buffers on the first step that produces gradients, and
        the existing LR scheduler state is preserved verbatim.
        """
        super().on_train_epoch_start()
        thaw_epoch = getattr(self.model, "dit_thaw_epoch", None)
        if thaw_epoch is None:
            return
        if self.current_epoch < thaw_epoch:
            return
        if not self.model._freeze_dit_backbone:
            return
        logger.info(
            "Scheduled DiT thaw at epoch %d (freeze_dit_backbone: %d).",
            self.current_epoch, thaw_epoch,
        )
        self.model.unfreeze_dit_backbone()

    # ------------------------------------------------------------------
    # Optimizer (backbone vs heads split + NaN/Inf gradient zeroing)
    # ------------------------------------------------------------------

    def configure_gradient_clipping(
        self, optimizer, gradient_clip_val=None, gradient_clip_algorithm=None,
    ) -> None:
        """Zero every gradient when any is non-finite, then clip.

        Hot-path version of the previous per-parameter scan (which
        called ``p.grad.isnan().any()`` and ``.isinf().any()`` on every
        parameter and forced one device->host sync per call -- O(P) in
        the 2 B-param DiT, ~thousands of syncs per step).  The new path
        builds the gradient list once, computes a fused stack of
        per-tensor L2 norms with ``torch._foreach_norm`` (single fused
        kernel over all tensors), and bails out only when
        ``torch.isfinite(norm_stack).all()`` is False -- one scalar
        sync per step instead of O(P).

        On a clean step the cost is one ``_foreach_norm`` call plus a
        single ``isfinite`` reduction + bool cast (a few microseconds);
        on a bad step we additionally zero every gradient with
        ``_foreach_zero_``, which is itself a fused kernel.

        Behaviour is otherwise identical to the previous path: on a
        non-finite batch every gradient is zeroed before the clipper
        runs, so the optimiser sees a no-op step rather than NaN
        weights.

        Non-finite-batch events are emitted at DEBUG level (silent by
        default) -- the bf16-mixed + ``gradient_clip_val=1.0`` stack
        triggers a handful per run that are otherwise harmless, and
        each event was being duplicated per DDP rank.  Raise the
        ``nanocosmos.modules.cosmos_2_5_common.base`` logger to DEBUG
        (or set ``NANOCOSMOS_GRAD_GUARD_LOG=warning`` -- see
        :data:`_GRAD_GUARD_LOG_FN`) to surface them again.
        """
        grads = []
        for group in optimizer.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    grads.append(p.grad)
        if grads:
            norms = torch._foreach_norm(grads)
            norm_stack = torch.stack(norms)
            if not bool(torch.isfinite(norm_stack).all()):
                torch._foreach_zero_(grads)
                _GRAD_GUARD_LOG_FN(
                    "Zeroed all %d gradients at step %d (one or more "
                    "non-finite per-tensor norms).",
                    len(grads), self.global_step,
                )
        if not gradient_clip_val:
            return

        # FSDP path: Lightning's ``FSDPPrecision`` rejects
        # ``self.clip_gradients(..., algorithm='norm')`` -- sharded grads
        # need a cross-rank norm reduction, which the FSDP root module's
        # own ``clip_grad_norm_`` performs.  Route there under FSDP and use
        # the standard ``self.clip_gradients`` everywhere else (DDP / single).
        from pytorch_lightning.strategies import FSDPStrategy

        if isinstance(self.trainer.strategy, FSDPStrategy):
            fsdp_module = self.trainer.model  # FSDP-wrapped root
            clip_fn = getattr(fsdp_module, "clip_grad_norm_", None)
            if callable(clip_fn):
                clip_fn(float(gradient_clip_val))
                return

        self.clip_gradients(
            optimizer,
            gradient_clip_val=gradient_clip_val,
            gradient_clip_algorithm=gradient_clip_algorithm,
        )

    def configure_optimizers(self) -> Any:
        lr = self.optimizer_config.get("lr", 1e-4)
        wd = self.optimizer_config.get("weight_decay", 1e-5)
        # AdamW betas: default to PyTorch's (0.9, 0.999); a lower beta2
        # (e.g. 0.99, as in the Cosmos-3 generator recipe) adapts the
        # second moment faster when fine-tuning a pretrained backbone.
        betas = tuple(self.optimizer_config.get("betas", (0.9, 0.999)))
        # AdamW epsilon.  PyTorch default 1e-8; the Cosmos 3 framework SFT
        # recipe notes 1e-6 is preferable under bf16 (avoids underflow in the
        # squared-gradient denominator).  Default keeps the historical 1e-8 so
        # configs that don't set it are byte-for-byte unchanged.
        eps = float(self.optimizer_config.get("eps", 1e-8))
        # Use explicit ``is None`` so a deliberate ``dit_backbone_lr: 0``
        # (e.g. to keep the unfrozen DiT weights pinned via gradient-
        # only updates from learned LR schedulers) is honoured rather
        # than silently falling back to ``lr`` via ``or``-truthiness.
        backbone_lr = self.optimizer_config.get("dit_backbone_lr")
        if backbone_lr is None:
            backbone_lr = lr

        # ``include_frozen_dit`` keeps the DiT params in the optimizer
        # even while they're still frozen, so the warm-up schedule
        # (``freeze_dit_backbone: N`` -- frozen for epochs 0..N-1, thawed
        # at epoch N) can simply flip ``requires_grad`` at epoch N
        # without rebuilding the optimizer (and without resetting the
        # LR scheduler).  AdamW skips params whose ``grad is None``,
        # so frozen DiT params sitting in the optimizer are a no-op
        # until they thaw.
        include_frozen_dit = (
            getattr(self.model, "dit_thaw_epoch", None) is not None
        )

        backbone_decay, backbone_no_decay = [], []
        head_decay, head_no_decay = [], []
        for name, param in self.named_parameters():
            is_backbone = name.startswith("model.dit.")
            if not param.requires_grad and not (
                include_frozen_dit and is_backbone
            ):
                continue
            no_decay = param.dim() <= 1 or name.endswith(".bias")
            if is_backbone:
                (backbone_no_decay if no_decay else backbone_decay).append(param)
            else:
                (head_no_decay if no_decay else head_decay).append(param)

        param_groups = [
            {"params": backbone_decay,      "lr": backbone_lr,   "weight_decay": wd},
            {"params": backbone_no_decay,   "lr": backbone_lr,   "weight_decay": 0.0},
            {"params": head_decay,          "lr": lr,            "weight_decay": wd},
            {"params": head_no_decay,       "lr": lr,            "weight_decay": 0.0},
        ]
        param_groups = [g for g in param_groups if g["params"]]

        clip_val = self.training_config.get("gradient_clip_val")
        use_fused = (
            not clip_val
            and torch.cuda.is_available()
            and all(p.is_cuda for g in param_groups for p in g["params"])
        )
        optimizer = torch.optim.AdamW(
            param_groups, lr=lr, betas=betas, eps=eps, weight_decay=wd,
            fused=use_fused,
        )

        return self._maybe_wrap_scheduler(optimizer)


__all__ = ["BaseCosmosModule"]
