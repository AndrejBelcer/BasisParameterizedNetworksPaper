import torch
import torch.nn as nn
from typing import Tuple, Dict, Any

from src.core.enums import ActivationType

class BaseLayer(nn.Module):
    """
    An abstract base class for all layers in the model.
    Enforces strict input/output shape validation and provides debugging utilities.
    """
    def __init__(self):
        super().__init__()
        # Shape attributes for validation (must be set by subclass __init__)
        self.a_in: int = 0
        self.x_in: int = 0
        self.f_out: int = 0
        self.y_out: int = 0

    def Shape(self, CurrentInputChannels: int, CurrentInputSpatial: int) -> Tuple[int, int]:
        """
        Validates the layer's expected input shape against provided dimensions
        and returns the calculated output shape (channels, spatial_size).
        """
        if CurrentInputChannels != self.a_in:
            raise ValueError(f"{self.__class__.__name__}: Input channel mismatch. Expected {self.a_in}, got {CurrentInputChannels}.")
        if CurrentInputSpatial != self.x_in:
            raise ValueError(f"{self.__class__.__name__}: Input spatial size mismatch. Expected {self.x_in}, got {CurrentInputSpatial}.")
        
        return self.f_out, self.y_out

    def forward(self, X: torch.Tensor, L: torch.Tensor) -> torch.Tensor:
        """
        The forward pass must be implemented by the subclass.
        It must accept the interpolation coefficients tensor 'L' even if the layer is static.
        """
        raise NotImplementedError("Subclasses must implement the `forward` method.")

    def get_config(self) -> Dict[str, Any]:
        """
        Returns the explicit configuration dictionary for this layer.
        Must be implemented by subclasses to ensure reproducibility.
        """
        raise NotImplementedError(f"{self.__class__.__name__} must implement get_config()")

    @torch.no_grad()
    def AfterStep(self):
        """
        Is applied after an optimizer step. 
        Can be used for clamping or re-normalization if strictly necessary.
        """
        pass

    def PrintLayerInputOutput(self):
        """
        Prints information about input output shapes.
        """
        print(f"Shape: [_, {self.a_in}, {self.x_in}, {self.x_in}] -> [_, {self.f_out}, {self.y_out}, {self.y_out}]")

    def PrintLayerInfo(self, detailed_info: bool):
        """
        Prints information about the layer's parameters.
        Must be overridden by subclasses to call PrintTimeLayerInfo.
        """
        pass

    @torch.no_grad()    
    def PrintTimeLayerInfo(self, name: str, flat_tensor: nn.Parameter):
        """
        Displays the statistics for the learnable parameter tensor.
        Assumes a Legendre Polynomial basis where Row 0 is the static term,
        and Rows 1 to degree-1 are the time-dependent dynamic terms.
        
        Args:
            name (str): A descriptive name (e.g., "Kernel Weights").
            flat_tensor (nn.Parameter): The parameter tensor of shape [degree, flat_size].
        """
        if flat_tensor.ndim != 2:
            print(f"  {name}: [Shape mismatch, expected [degree, flat_size], got {flat_tensor.shape}]")
            return

        degree = flat_tensor.shape[0]
        print(f"  Parameter Group: '{name}' (Shape: {flat_tensor.shape})")
        
        # Dynamically generate labels based on the degree
        labels = ["Static (P0)"] + [f"Dyn {i} (P{i})" for i in range(1, degree)]
        
        for i in range(degree):
            data = flat_tensor[i].detach().float() # Detach and cast to float for stats
            mean = data.mean().item()
            std = data.std().item()
            min_val = data.min().item()
            max_val = data.max().item()
            print(f"    {labels[i]}:\tμ={mean:+.4f}, σ={std:.4f}, min={min_val:+.4f}, max={max_val:+.4f}")
        print("    " + "-"*40)

    @torch.no_grad()   
    def PrintActivationFunction(self, activation: ActivationType):
        print(f"\tActivation function: [{activation}]")

