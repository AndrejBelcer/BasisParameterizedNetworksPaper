import torch
from typing import Callable, Dict, Any

from src.core.globals import GPU_CONFIG
from src.layers.base_time_layer import BaseLayer

class CovarianceExpansionLayer(BaseLayer):
    """
    Computes quadratic features x_i * x_j for all pairs (i, j).
    Static layer (no learnable parameters).
    """
    def __init__(self, a: int, x: int, degree: int):
        super().__init__()
        # Shape validation
        if a <= 0 or x <= 0:
            raise ValueError("CovarianceExpansionLayer params 'a', 'x' must be positive.")

        # Shape attributes for validation
        shape_f = (a * (a + 1)) // 2

        self.a_in, self.x_in = a, x
        self.f_out, self.y_out = shape_f, x

        self.rows, self.cols = torch.triu_indices(a, a, offset=1, device=GPU_CONFIG.device)
        
    def forward(self, X: torch.Tensor, L: torch.Tensor) -> torch.Tensor:
        # Compute the x_i x_j
        T = X[:, self.rows, :, :] * X[:, self.cols, :, :]
        # Concat with original features
        return torch.cat([X, T], dim=1)
    
    def PrintLayerInfo(self, detailed_info: bool):
        print(f"--- Layer: CovarianceExpansionLayer ---")
        self.PrintLayerInputOutput()
        print("  (Static Layer: No learnable parameters)")

    def get_config(self) -> Dict[str, Any]:
        return {
            'type': 'CovarianceExpansionLayer',
            'params': {
                'a': self.a_in,
                'x': self.x_in
            }
        }
    
