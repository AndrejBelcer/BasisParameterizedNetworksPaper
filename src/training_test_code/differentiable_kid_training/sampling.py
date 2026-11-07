"""Explicit grids and full-gradient Euler sampling."""
import torch
from torch.utils.checkpoint import checkpoint


def shifted_grid(steps, minimum, phase, start=1.0):
    if steps < 3 or not 0 < minimum < start <= 1 or not 0 < phase < 1:
        raise ValueError('Require steps >= 3, 0 < minimum < start <= 1, and 0 < phase < 1.')
    count = steps - 2
    interior = minimum + (torch.arange(count, dtype=torch.float64) + phase) * ((start-minimum)/count)
    grid = torch.cat((torch.tensor([start], dtype=torch.float64), interior.flip(0),
                      torch.tensor([minimum, 0.0], dtype=torch.float64))).float()
    if not bool((grid[:-1] > grid[1:]).all()):
        raise ValueError('Phase produced duplicate FP32 nodes; draw another phase.')
    return grid


def random_grid(steps, minimum, seed, start=1.0):
    rng = torch.Generator().manual_seed(seed)
    for _ in range(100):
        phase = float(torch.rand((), generator=rng, dtype=torch.float64))
        if 0 < phase < 1:
            try:
                return shifted_grid(steps, minimum, phase, start), phase
            except ValueError:
                pass
    raise RuntimeError('Could not construct strictly descending FP32 grid.')


def noise_bank(count, channels, image_size, seed):
    return torch.randn(count, channels, image_size, image_size,
                       generator=torch.Generator().manual_seed(seed), dtype=torch.float32)


class EulerSampler:
    def __init__(self, velocity, grid, device, checkpoint_steps=True):
        self.velocity = velocity
        self.times = tuple(t.reshape(1).to(device) for t in grid[:-1])
        self.deltas = tuple(float(x) for x in grid[1:] - grid[:-1])
        self.checkpoint_steps = checkpoint_steps

    def __call__(self, noise):
        x = noise
        for t, delta in zip(self.times, self.deltas):
            if self.checkpoint_steps and torch.is_grad_enabled():
                # Time is an explicit input, never a late-bound loop closure.
                v = checkpoint(self.velocity, x, t, use_reentrant=False)
            else:
                v = self.velocity(x, t)
            x = x + delta * v
        return x


def rgb01(raw):
    return (raw * 0.5 + 0.5).clamp(0, 1)
