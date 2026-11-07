import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Callable, Dict, Any

from src.core.globals import GPU_CONFIG
from src.core.enums import ActivationType
from src.core.mappings_activations import get_activation_function

from src.layers.base_time_layer import BaseLayer
from src.math.time_interpolation import InterpolateWeights
from src.math.parameter_initialization import CreateSpectralParameter, CreateZerosParameter
from src.math.convolution_shape import GetRequiredP

class ConvolutionUpscalingLayer(BaseLayer):
    """
    Time-Dependent Grouped Convolution Upscaling Layer.
    Upscales input bilinearly, then applies grouped convolution.
    """
    def __init__(self, a: int, x: int, k: int, degree: int, d: int = 1, p: int = -1, c: int = 1, y: int = 200, activation_function: ActivationType = ActivationType.XN):
        super().__init__()
        
        # 1. Validate Parameters
        if not all(val > 0 for val in [a, x, k, d, c, y]):
            raise ValueError("ConvolutionUpscalingLayer params must be positive integers.")

        if (p < 0):
            p = GetRequiredP(y, k, d, 1)

        # Calculate required Upscale size (U) before convolution
        # d * (1 - k) + (U + 2*p) = y  =>  U = y - d*(1-k) - 2*p
        self.U = y - d * (1 - k) - (2 * p)
        
        if self.U <= 0:
             raise ValueError(f"Calculated intermediate upscale size U ({self.U}) is non-positive. Check target y ({y}) and kernel params.")

        # 2. Set Shape Attributes
        self.a_in, self.x_in = a, x
        self.degree = degree
        self.c = c
        self.f_out, self.y_out = a * c, y
        self.k, self.d, self.p = k, d, p
        
        # 3. Activation function
        self.activation_function = activation_function
        self.ActivationFunction: Callable = get_activation_function(activation_function)

        # 4. Parameter Initialization
        self.kernels = CreateSpectralParameter((self.f_out, 1, self.k, self.k), degree=self.degree, device=GPU_CONFIG.device, dtype=GPU_CONFIG.dtype, main_std=math.sqrt(2.0 / ((self.k**2))))
        self.bias = CreateZerosParameter(self.f_out, degree=self.degree, device=GPU_CONFIG.device, dtype=GPU_CONFIG.dtype)

    def forward(self, X: torch.Tensor, L: torch.Tensor) -> torch.Tensor:
        # 1. Interpolate Parameters
        k_t = InterpolateWeights(self.kernels, L)
        b_t = InterpolateWeights(self.bias, L)

        # 2. Upscale the X
        # [B, A, X, X] -> [B, A, U, U]
        X_up = F.interpolate(X, size=(self.U, self.U), mode='bilinear', align_corners=True)

        # 3. Apply Convolution
        # [B, A, U, U] -> [B, A*c, Y, Y]
        Y = F.conv2d(input=X_up, weight=k_t, bias=b_t, stride=1, padding=self.p, dilation=self.d, groups=self.a_in)
        
        return self.ActivationFunction(Y)

    def PrintLayerInfo(self, detailed_info: bool):
        print(f"--- Layer: ConvolutionUpscalingLayer (Grouped Expansion) ---")
        self.PrintLayerInputOutput()
        print(f"  Multiplier c: {self.c}")
        print(f"  Upscale Intermediate: {self.U}x{self.U}")
        print(f"  Kernel Size: {self.k}x{self.k}, Padding: {self.p}")
        if(detailed_info):
            self.PrintTimeLayerInfo("Weights", self.kernels.view(self.degree, -1))
            self.PrintTimeLayerInfo("Bias", self.bias)
        self.PrintActivationFunction(self.activation_function)

    def get_config(self) -> Dict[str, Any]:
        return {
            'type': 'ConvolutionUpscalingLayer',
            'params': {
                'a': self.a_in,
                'x': self.x_in,
                'y': self.y_out,
                'k': self.k,
                'd': self.d,
                'p': self.p,
                'c': self.c,
                'degree': self.degree,
                'activation_function': self.activation_function
            }
        }

