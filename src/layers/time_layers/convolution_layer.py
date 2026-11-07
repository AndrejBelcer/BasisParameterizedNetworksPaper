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
from src.math.convolution_shape import GetConvolutionNumerator, GetRequiredP
from src.math.parameter_initialization import CreateSpectralParameter, CreateZerosParameter, CreateConstParameter

class ConvolutionLayer(BaseLayer):
    """
    Time-Dependent Grouped Convolution Layer (Free Parameters).
    
    Performs a Depthwise Convolution with a Channel Multiplier 'c'.
    Input Channels: A
    Output Channels: A * c
    Groups: A (Each input channel is convolved with 'c' independent kernels)
    """
    def __init__(self, a: int, x: int, k: int, degree: int, s: int = 1, d: int = 1, p: int = -1, c: int = 1, activation_function: ActivationType = ActivationType.XN):
        super().__init__()
        
        # 1. Validate Parameters
        if not all(val > 0 for val in [a, x, k, s, d, c]):
            raise ValueError("ConvolutionLayer params must be positive integers.")

        if (p < 0):
            p = GetRequiredP(x, k, d, s)

        numerator_y = GetConvolutionNumerator(x=x, k=k, d=d, s=s, p=p)
        if numerator_y % s != 0:
             raise ValueError(f"Invalid convolution: x={x}, k={k}, d={d}, s={s}, p={p} results in non-integer output.")
        
        y_out_val = numerator_y // s
        if y_out_val <= 0:
            raise ValueError(f"Calculated output spatial dimension y ({y_out_val}) must be positive.")
    
        # 2. Set Shape Attributes
        self.a_in, self.x_in = a, x
        self.degree = degree
        self.c = c
        self.f_out = a * c
        self.y_out = y_out_val
        
        self.k, self.s, self.d, self.p = k, s, d, p
        
        # 3. Activation function
        self.activation_function = activation_function
        self.ActivationFunction: Callable = get_activation_function(activation_function)

        # 4. Parameter Initialization
        self.kernels = CreateSpectralParameter((self.f_out, 1, self.k, self.k), degree=self.degree, device=GPU_CONFIG.device, dtype=GPU_CONFIG.dtype, main_std=math.sqrt(2.0 / ((self.k**2))))
        self.bias = CreateZerosParameter(self.f_out, device=GPU_CONFIG.device, degree=self.degree, dtype=GPU_CONFIG.dtype)

    def forward(self, X: torch.Tensor, L: torch.Tensor) -> torch.Tensor:
        # 1. Interpolate Parameters
        # k_t: [A*c, 1, K, K]
        k_t = InterpolateWeights(self.kernels, L)
        # b_t: [A*c]
        b_t = InterpolateWeights(self.bias, L)

        # 2. Apply Convolution
        # groups=self.a_in ensures each input channel maps to 'c' output channels
        # Input: [B, A, H, W] -> Output: [B, A*c, H', W']
        Y = F.conv2d(input=X, weight=k_t, bias=b_t, stride=self.s, padding=self.p, dilation=self.d, groups=self.a_in)
        
        return self.ActivationFunction(Y)

    def PrintLayerInfo(self, detailed_info: bool):
        print(f"--- Layer: ConvolutionLayer (Grouped Expansion) ---")
        self.PrintLayerInputOutput()
        print(f"  Multiplier c: {self.c} (Groups={self.a_in})")
        print(f"  Kernel Size: {self.k}x{self.k}, Stride: {self.s}, Padding: {self.p}")
        if(detailed_info):
            self.PrintTimeLayerInfo("Weights", self.kernels.view(self.degree, -1))
            self.PrintTimeLayerInfo("Bias", self.bias)
        self.PrintActivationFunction(self.activation_function)

    def get_config(self) -> Dict[str, Any]:
        return {
            'type': 'ConvolutionLayer',
            'params': {
                'a': self.a_in,
                'x': self.x_in,
                'k': self.k,
                'd': self.d,
                's': self.s,
                'p': self.p,
                'c': self.c,
                'degree': self.degree,
                'activation_function': self.activation_function
            }
        }

