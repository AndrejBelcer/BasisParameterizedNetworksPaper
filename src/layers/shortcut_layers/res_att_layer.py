import torch
from typing import Callable, Dict, Any
import torch.nn.functional as F

from src.core.globals import GPU_CONFIG
from src.core.enums import ActivationType
from src.core.mappings_activations import get_activation_function

from src.layers.base_time_layer import BaseLayer
from src.layers.static_layers.attention_layer import StaticAttentionLayer
from src.layers.time_layers.matmul_layer import MatMulLayer


class ResAttLayer(BaseLayer):
    def __init__(self, a: int, x: int, degree: int, act_in: ActivationType = ActivationType.IDENTITY, act_out: ActivationType = ActivationType.IDENTITY, head_dim: int = 32):
        super().__init__() 
        # 1. Set Shape Attributes
        self.a_in, self.f_out = a, a
        self.x_in, self.y_out = x, x
        self.degree = degree

        self.act_in = act_in
        self.act_out = act_out
        self.head_dim = head_dim

        self.mat_in = MatMulLayer(f = 3 * a, a = a, x = x, degree=degree, activation_function = act_in)
        self.att = StaticAttentionLayer(a = 3 * a, x = x, degree=degree, head_dim = head_dim)
        self.mat_out = MatMulLayer(f = a, a = a, x = x, degree=degree, activation_function = act_out)

    def forward(self, X: torch.Tensor, L: torch.Tensor) -> torch.Tensor:        
        # Attention Path
        Y = self.mat_in(X, L)
        Y = self.att(Y, L)
        Y = self.mat_out(Y, L)
        
        # Update: X_new = X + delta_X
        return X + Y

    def PrintLayerInfo(self, detailed_info: bool):
        print(f"--- Layer: ResAttLayer ---")
        self.PrintLayerInputOutput()
        print("Submodules:")
        print(f"--------------------------:")
        self.mat_in.PrintLayerInfo(detailed_info)
        self.att.PrintLayerInfo(detailed_info)
        self.mat_out.PrintLayerInfo(detailed_info)
        print(f"--------------------------:")


    def get_config(self) -> Dict[str, Any]:
        return {
            'type': 'ResAttLayer',
            'params': {
                'a': self.a_in,
                'x': self.x_in,
                'act_in': self.act_in,
                'act_out': self.act_out,
                'head_dim': self.head_dim,
                'degree': self.degree,
            }
        }
