import torch
import torch.nn.functional as F
from typing import Dict, Any

from src.layers.base_time_layer import BaseLayer

class StaticAttentionLayer(BaseLayer):
    """
    Parameterless Multi-Head Self-Attention.
    Expects input channels to be 3 * C.
    Splits input into Q, K, V, computes attention, and outputs C channels.
    """
    def __init__(self, a: int, x: int, degree: int, head_dim: int = 4):
        super().__init__()

        # 0. Precompute values
        f = a // 3
        num_heads = f // head_dim

        # 1. Validate Parameters
        if not all(val > 0 for val in [a, x, f, head_dim, num_heads]):
            raise ValueError("StaticAttentionLayer params must be positive.")
        if a % 3 != 0:
            raise ValueError(f"Input channels {a} must be divisible by 3 to split into Q, K, V.")
        if f % head_dim != 0:
            raise ValueError(f"Output channels {f} must be divisible by head_dim {head_dim}.")

        # 2. Set Shape Attributes
        self.a_in = a
        self.x_in = x
        self.f_out = f
        self.y_out = x
        self.head_dim = head_dim
        self.num_heads = num_heads

        # 3. Precompute Constants for fast forward pass
        self.n = x * x  # Sequence length (H * W)

    def forward(self, X: torch.Tensor, L: torch.Tensor) -> torch.Tensor:
        # X shape: [B, 3*C, H, W]
        B = X.shape[0]
        
        # 1. Reshape and Permute to get Q, K, V
        # Input X: [B, 3*C, H, W] -> [B, 3, Heads, Head_Dim, N]
        X_reshaped = X.view(B, 3, self.num_heads, self.head_dim, self.n)
        
        # Permute to [3, B, Heads, N, Head_Dim] and make contiguous.
        # Doing this ONCE for the combined tensor is much faster than doing it 3 times.
        # This guarantees memory contiguity, which is strictly required for FlashAttention.
        X_permuted = X_reshaped.permute(1, 0, 2, 4, 3).contiguous()
        
        # 2. Unpack Q, K, V
        # Because X_permuted is contiguous, slicing along dim 0 yields contiguous tensors.
        Q, K, V = X_permuted[0], X_permuted[1], X_permuted[2]
        
        # 3. Scaled Dot-Product Attention
        # Automatically routes to the highly optimized FlashAttention backend.
        attn_output = F.scaled_dot_product_attention(Q, K, V)
        
        # 4. Reshape back to Image Dimensions
        # attn_output: [B, Heads, N, Head_Dim]
        # Permute to [B, Heads, Head_Dim, N] to group the channel dimensions together
        attn_output = attn_output.permute(0, 1, 3, 2).contiguous()
        
        # View as [B, C, H, W]
        Y = attn_output.view(B, self.f_out, self.y_out, self.y_out)
        
        return Y

    def PrintLayerInfo(self, detailed_info: bool):
        print(f"--- Layer: StaticAttentionLayer (Parameterless) ---")
        self.PrintLayerInputOutput()
        print(f"  Heads: {self.num_heads}, Head Dim: {self.head_dim}")
        print("  (Zero Learnable Parameters)")

    def get_config(self) -> Dict[str, Any]:
        return {
            'type': 'StaticAttentionLayer',
            'params': {
                'a': self.a_in,
                'x': self.x_in,
                'head_dim': self.head_dim
            }
        }
