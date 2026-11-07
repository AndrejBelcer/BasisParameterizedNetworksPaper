"""Fixed-grid Euler integration around the original, uncompiled velocity model."""
import torch


class EulerGenerator(torch.nn.Module):
    def __init__(self, model, grid, device, model_dtype):
        super().__init__()
        self.model = model
        self.model_dtype = model_dtype
        self.register_buffer("times", grid[:-1].reshape(-1, 1).to(device))
        self.deltas = tuple((grid[1:]-grid[:-1]).tolist())

    def forward(self, noise):
        # No in-place updates: the same initial noise bank serves every candidate.
        # Autocast is supplied by Runtime outside this complete compiled region.
        x = noise.float().clone()
        for index, delta in enumerate(self.deltas):
            velocity = self.model(x.to(self.model_dtype), self.times[index])
            x = x + delta*velocity.float()
        return x
