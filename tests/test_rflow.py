"""Rectified-flow path geometry, without a DiT."""

import torch

from nanocosmos.losses.rflow import euler_integrate, interpolate_rectified_flow


def test_straight_path_endpoints_and_velocity():
    x0 = torch.zeros(2, 4, 2, 2, 2)
    x1 = torch.ones(2, 4, 2, 2, 2)
    t = torch.tensor([0.0, 1.0])
    x_t, v = interpolate_rectified_flow(x0, x1, t)
    assert torch.equal(x_t[0], x0[0])
    assert torch.equal(x_t[1], x1[1])
    assert torch.equal(v, x1 - x0)


def test_euler_with_constant_velocity_reaches_clean():
    x0 = torch.randn(1, 2, 2, 2, 2)
    x1 = torch.randn(1, 2, 2, 2, 2)
    target = x1 - x0

    def velocity_fn(state, t_unit):
        assert state.shape == x1.shape
        assert 0.0 < t_unit <= 1.0
        return target

    reached = euler_integrate(x1, velocity_fn, steps=4)
    assert torch.allclose(reached, x0, atol=1e-5)
