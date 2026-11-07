import torch
import torch.nn as nn
from torch.nn.attention import sdpa_kernel, SDPBackend

from src.core.globals import GPU_CONFIG
from src.core.enums import CompileMode
from src.modules.velocity_model import VelocityModel

class VelocityTrainingWrapper(nn.Module):
    """
    Wrapper used for training the VelocityModel.
    Evaluates independent image batches at multiple times using vmap.
    """
    def __init__(self, model: VelocityModel):
        super().__init__()
        self.model = model
        
        # ---------------------------------------------------------
        # 1. VMAP Forward (Flow Matching)
        # in_dims=(0, 0) maps over the T dimension of X_tz and t_t
        # Inside the vmap, the model sees X: [Z, C, H, W] and t: [1]
        # ---------------------------------------------------------
        self.vmap_forward = torch.vmap(self.model, in_dims=(0, 0), randomness='error')
        
        # ---------------------------------------------------------
        # 2. Compilation
        # ---------------------------------------------------------
        compile_mode = GPU_CONFIG.compile_mode_velocity
        if GPU_CONFIG.compile_model and compile_mode != CompileMode.NONE:
            print(f"Compiling Training Wrapper (Mode: {compile_mode.value})...")
            self.compiled_forward = torch.compile(self.vmap_forward, mode=compile_mode.value, fullgraph=True)
        else:
            print("Compiling wrapper skipped.")
            self.compiled_forward = self.vmap_forward

    def predict_velocity(self, X_tz: torch.Tensor, t_t: torch.Tensor) -> torch.Tensor:
        """
        Predicts the velocity field for a batched set of times and images.
        
        Args:
            X_tz (torch.Tensor): Corrupted images of shape [T, Z, C, H, W]
            t_t (torch.Tensor): Time steps of shape [T, 1]
            
        Returns:
            torch.Tensor: Predicted velocity of shape [T, Z, C, H, W]
        """
        with sdpa_kernel(SDPBackend.MATH):
            pred_V = self.compiled_forward(X_tz, t_t)
        return pred_V

