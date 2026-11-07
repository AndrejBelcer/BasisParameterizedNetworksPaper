from typing import Callable
from src.core.enums import ActivationType
import torch
import torch.nn.functional as F


# Activation functions
from src.math.activation_functions import Activation_XEXP, Activation_XEXP2, Activation_X3, Activation_XN, Activation_Gauss

def get_activation_function(act_type: ActivationType | str) -> Callable:
    """Returns the activation function based on the enum or string."""
    # Allow string fallback for config parsing
    if isinstance(act_type, str):
        act_type = ActivationType(act_type.lower())

    match act_type:
        case ActivationType.XEXP: return Activation_XEXP
        case ActivationType.XEXP2: return Activation_XEXP2
        case ActivationType.X3: return Activation_X3
        case ActivationType.XN: return Activation_XN
        case ActivationType.GAUSS: return Activation_Gauss
        case ActivationType.TANH: return torch.tanh
        case ActivationType.RELU: return F.relu
        case ActivationType.SIGMOID: return torch.sigmoid
        case ActivationType.SILU: return F.silu
        case ActivationType.IDENTITY: return lambda x: x
        case _: raise ValueError(f"Unknown activation type: {act_type}")

