"""Rectified-flow helpers for the joint reconstruction recipe.

The straight path runs in the VAE latent. ``t = 0`` is the clean volume
and ``t = 1`` is the degraded volume:

    x_t = (1 - t) x0 + t x1
    v   = x1 - x0

``proj_out`` is trained to predict ``v``. ``pred/Euler`` integrates that
velocity from ``t = 1`` down to ``t = 0`` with a fixed Euler step, then the
task head reads the resulting latent. ``pred/recon`` is the task head on the
degraded latent alone; this module does not build it.
"""

from typing import Callable

import torch


def interpolate_rectified_flow(
    x0: torch.Tensor,
    x1: torch.Tensor,
    t: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Straight-path sample ``x_t`` and constant velocity ``v = x1 - x0``.

    ``t`` is a per-sample value in ``(0, 1)`` with shape ``[B]`` (or any
    shape broadcastable to ``[B, 1, ...]``). ``x0`` and ``x1`` share a shape.
    The velocity does not depend on ``t``.
    """
    if x0.shape != x1.shape:
        raise ValueError(
            f"rectified-flow endpoints must match, got {tuple(x0.shape)} "
            f"and {tuple(x1.shape)}."
        )
    t_b = t.to(device=x0.device, dtype=x0.dtype).reshape(
        t.shape[0], *([1] * (x0.dim() - 1)),
    )
    x_t = (1.0 - t_b) * x0 + t_b * x1
    return x_t, x1 - x0


def euler_integrate(
    x1: torch.Tensor,
    velocity_fn: Callable[[torch.Tensor, float], torch.Tensor],
    steps: int,
) -> torch.Tensor:
    """Integrate ``dx/dt = v(x, t)`` from ``t = 1`` (``x1``) down to ``t = 0``.

    ``steps`` equal Euler increments. ``velocity_fn(x, t)`` receives the unit
    timestep ``t`` in ``(0, 1]``, not the transformer's scaled timestep.
    """
    n = int(steps)
    if n < 1:
        raise ValueError(f"few-step Euler needs at least 1 step, got {steps}.")
    dt = 1.0 / n
    x = x1
    for k in range(n):
        t = 1.0 - k * dt
        v = velocity_fn(x, t)
        if v.shape != x.shape:
            raise ValueError(
                f"velocity shape {tuple(v.shape)} != state {tuple(x.shape)}."
            )
        x = x - dt * v
    return x
