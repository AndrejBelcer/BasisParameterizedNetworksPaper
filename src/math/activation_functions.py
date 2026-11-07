import torch
import torch.nn.functional as F
from src.core.enums import ActivationType

"""
Custom activation functions.
Registered for lookup by string name in the Model config.
"""

def Activation_XEXP(x: torch.Tensor) -> torch.Tensor:
    """
    y = x * exp(-(x^4) / 4) * C
    C (approx 1.28402) normalizes the peak to y=1.0 at x=1.0 and falls of to y=0 around x=2.
    """
    x_safe = torch.clamp(x, min=-3.0, max=3.0)
    return x_safe * torch.exp(-(x_safe**4) / 4.0) * 1.28402

def Activation_XEXP2(x: torch.Tensor) -> torch.Tensor:
    """
    y = t * exp(-(t^4) / 4) * C;
    t = 0.9 x
    Maximum at 2
    """
    x_safe = torch.clamp(0.9 * x, min=-3.2, max=3.2)
    return x_safe * torch.exp(-(x_safe**4) / 4.0) * 2.56804

def Activation_X3(x: torch.Tensor) -> torch.Tensor:
    """
    y = t^3 / (1 + |t^3|) where t = 2x.
    Sigmoidal-like, bounded [-1, 1].
    """
    x_safe = torch.clamp(x, min=-3.5, max=3.5)
    t_cubed = 8.0 * (x_safe ** 3)
    return t_cubed / (1.0 + torch.abs(t_cubed))

def Activation_XN(x: torch.Tensor) -> torch.Tensor:
    """
    y = 2 * t^3 / (1 + |t^3|) where t = x.
    Sigmoidal-like, bounded [-2, 2].
    """
    x_safe = torch.clamp(x, min=-7, max=7)
    x_cubed = x_safe**3
    return 2 * x_cubed / (1 + torch.abs(x_cubed))


def Activation_Gauss(x: torch.Tensor) -> torch.Tensor:
    """y = 2 exp(-x^2) - 1"""
    return 2 * torch.exp(-x**2) - 1
