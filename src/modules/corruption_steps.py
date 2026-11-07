import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment
import torch._dynamo as dynamo

from src.utilities.tensor_utils import NoiseImage_G

# ===================================================================
# --- Base Class ---
# ===================================================================

class CorruptionStep(nn.Module):
    """
    Abstract base class for all corruption operations.
    
    Each step defines a target state. The main module then performs:
    N_new = lerp(N_old, Step(N_old), scale)
    
    Attributes:
        scale (float): The strength of this specific corruption step.
        uses_randomness (bool): Must be set to True if the step uses RNG, 
                                as this prevents 'fullgraph' compilation.
    """
    def __init__(self, scale: float = 1.0):
        super().__init__()
        self.scale = scale
        self.uses_randomness = False

    def forward(self, G: torch.Tensor) -> torch.Tensor:
        """Returns the modified tensor (the target for the interpolation)."""
        raise NotImplementedError

# ===================================================================
# --- Corruption Steps ---
# ===================================================================

class NoiseStep_G(CorruptionStep):
    """
    Generates pure Gaussian noise matching the input shape.
    Used to simulate the standard 'NoiseOnly' flow matching strategy.
    """
    def __init__(self, scale: float = 1.0):
        super().__init__(scale)
        self.uses_randomness = True

    def forward(self, G: torch.Tensor) -> torch.Tensor:
        # Generate fresh noise for every forward pass in the project's float range
        return NoiseImage_G(G.shape, G.device, G.dtype) * self.scale


class HungarianNoiseStep_G(CorruptionStep):
    """
    Optimal Transport Noise.
    
    Generates a pool of noise larger than the batch size and uses the 
    Hungarian Algorithm to assign the 'closest' noise sample to each image.
    This reduces the 'velocity distance' the model has to learn.
    """
    def __init__(self, scale: float = 1.0, pool_multiplier: int = 1):
        super().__init__(scale)
        self.uses_randomness = True
        self.pool_multiplier = pool_multiplier

    def forward(self, G: torch.Tensor) -> torch.Tensor:
        B, C, H, W = G.shape
        device, dtype = G.device, G.dtype

        # 1. Generate a larger pool of noise
        pool_size = B * self.pool_multiplier
        N_pool = NoiseImage_G((pool_size, C, H, W), device, dtype)

        # 2. Compute pairwise L2 distances (Cost Matrix)
        # This part is fully differentiable and runs on GPU
        G_flat = G.view(B, -1)
        N_flat = N_pool.view(pool_size, -1)
        cost_matrix = torch.cdist(G_flat, N_flat, p=2.0)**2

        # 3. Solve Assignment (Graph Break)
        # We move to CPU for the combinatorial solver
        assigned_indices_cpu = self.solve_hungarian_matching(cost_matrix)

        # 4. Gather assigned noise back on GPU
        assigned_indices = assigned_indices_cpu.to(device)
        return N_pool[assigned_indices] * self.scale
    
    @dynamo.disable
    def solve_hungarian_matching(self, cost_matrix: torch.Tensor) -> torch.Tensor:
        """Runs the SciPy solver. Isolated from torch.compile."""
        cost_cpu = cost_matrix.detach().cpu().numpy()
        # row_ind is [0...B], col_ind is the assigned noise index
        _, col_ind = linear_sum_assignment(cost_cpu)
        return torch.from_numpy(col_ind).long()




