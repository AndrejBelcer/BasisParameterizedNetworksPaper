import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Callable, Dict, Any

from src.core.globals import GPU_CONFIG

from src.layers.base_time_layer import BaseLayer
from src.core.enums import ActivationType
from src.core.mappings_activations import get_activation_function


from src.math.time_interpolation import InterpolateWeights
from src.math.parameter_initialization import CreateZerosParameter

class GroupNormLayer(BaseLayer):
    """
    Time-Dependent Group Normalization Layer.
    
    Args:
        a (int): Number of input channels.
        x (int): Input spatial resolution (height/width).
        c (int): Number of channels per group. Total groups = a // c.
    """
    def __init__(self, a: int, x: int, degree: int, c: int = 3, activation_function: ActivationType = ActivationType.IDENTITY):
        super().__init__()
        
        # 1. Validate Parameters
        if c <= 0:
            raise ValueError(f"GroupNormLayer: channels per group 'c' must be positive, got {c}.")
        if a % c != 0:
            raise ValueError(f"GroupNormLayer: Input channels 'a' ({a}) must be divisible by channels per group 'c' ({c}).")

        # 2. Set Shape Attributes
        self.a_in, self.x_in = a, x
        self.f_out, self.y_out = a, x
        self.degree = degree
        self.channels_per_group = c
        self.num_groups = a // c

        # 3. Parameter Initialization
        # Raw gain and shift coefficients have shape [degree, a].
        # Zero raw gain gives 1 + ELU(0) = 1; zero shift gives identity offset.
        self.gamma = CreateZerosParameter(self.a_in, degree=self.degree, device=GPU_CONFIG.device, dtype=GPU_CONFIG.dtype)
        self.beta = CreateZerosParameter(self.a_in, degree=self.degree, device=GPU_CONFIG.device, dtype=GPU_CONFIG.dtype)

        # 4. Activation function
        self.activation_function = activation_function
        self.ActivationFunction: Callable = get_activation_function(activation_function)


    def forward(self, X: torch.Tensor, L: torch.Tensor) -> torch.Tensor:
        # 1. Interpolate Parameters
        # gamma_t, beta_t: [a]
        gamma_t = InterpolateWeights(self.gamma, L)
        beta_t = InterpolateWeights(self.beta, L)

        # 2. Compute Scaling Factor
        # Constrain scale to (0, \infty)
        scale = 1.0 + F.elu(gamma_t)

        # 3. Apply Group Norm
        # F.group_norm(input, num_groups, weight, bias, eps)
        # We pass our time-dependent scale and beta directly as weight/bias.
        X = F.group_norm(X, self.num_groups, weight=scale, bias=beta_t, eps=1e-5)
        return self.ActivationFunction(X)

    def PrintLayerInfo(self, detailed_info: bool):
        print(f"--- Layer: GroupNormLayer ---")
        self.PrintLayerInputOutput()
        print(f"  Channels per Group: {self.channels_per_group}")
        print(f"  Total Groups: {self.num_groups}")
        if(detailed_info):
            self.PrintTimeLayerInfo("Gamma (Scale)", self.gamma)
            self.PrintTimeLayerInfo("Beta (Shift)", self.beta)
        self.PrintActivationFunction(self.activation_function)


    def get_config(self) -> Dict[str, Any]:
        return {
            'type': 'GroupNormLayer',
            'params': {
                'a': self.a_in,
                'x': self.x_in,
                'c': self.channels_per_group,
                'degree': self.degree,
                'activation_function': self.activation_function
            }
        }
    
