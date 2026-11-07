import torch
import math
from typing import Callable, Dict, Any

from src.core.globals import GPU_CONFIG
from src.core.enums import ActivationType
from src.core.mappings_activations import get_activation_function

from src.layers.base_time_layer import BaseLayer
from src.math.time_interpolation import InterpolateWeights
from src.math.parameter_initialization import CreateSpectralParameter, CreateZerosParameter

class MatMulLayer(BaseLayer):
    """
    Channel projection with Legendre-parameterized weights and bias.
    
    Projects [B, A, X, Y] -> [B, F, X, Y].
    Operates on the channel dimension (A -> F).
    """
    def __init__(self, f: int, a: int, x: int, degree: int, activation_function: ActivationType = ActivationType.XN):
        super().__init__()
        
        # 1. Validate Parameters
        if not all(val > 0 for val in [f, a, x, degree]):
            raise ValueError("MatMulLayer params 'f', 'a', 'x', 'degree' must be positive.")

        # 2. Set Shape Attributes
        self.a_in, self.x_in = a, x
        self.f_out, self.y_out = f, x # Spatial dims preserved
        self.degree = degree
        
        # 3. Activation function
        self.activation_function = activation_function
        self.ActivationFunction: Callable = get_activation_function(activation_function)

        # 4. Parameter Initialization
        self.matrix = CreateSpectralParameter((self.f_out, self.a_in), degree=self.degree, device=GPU_CONFIG.device, dtype=GPU_CONFIG.dtype, main_std=math.sqrt(2.0 / (self.a_in + self.f_out)))
        self.bias = CreateZerosParameter(self.f_out, degree=self.degree, device=GPU_CONFIG.device, dtype=GPU_CONFIG.dtype)
        

    def forward(self, X: torch.Tensor, L: torch.Tensor) -> torch.Tensor:
        # 1. Interpolate Parameters
        M_t = InterpolateWeights(self.matrix, L)
        B_t = torch.tanh(InterpolateWeights(self.bias, L)).view(1, self.f_out, 1, 1)

        # 2. Einsum
        Y = torch.einsum('fa, baxy -> bfxy', M_t, X)
        
        # 3. Apply Bias
        Y = Y + B_t
        
        return self.ActivationFunction(Y)

    def PrintLayerInfo(self, detailed_info: bool):
        print(f"--- Layer: MatMulLayer ---")
        self.PrintLayerInputOutput()
        if(detailed_info):
            self.PrintTimeLayerInfo("Flat Weights", self.matrix.view(self.degree, -1))
            self.PrintTimeLayerInfo("Bias", self.bias)
        self.PrintActivationFunction(self.activation_function)


    def get_config(self) -> Dict[str, Any]:
        return {
            'type': 'MatMulLayer',
            'params': {
                'f': self.f_out,
                'a': self.a_in,
                'x': self.x_in,
                'degree': self.degree,
                'activation_function': self.activation_function
            }
        }
