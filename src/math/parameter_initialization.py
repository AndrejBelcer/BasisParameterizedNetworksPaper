import torch
import torch.nn as nn
from typing import Tuple, Union

def CreateSpectralParameter(shape: Union[int, Tuple[int, ...]], degree: int, device: torch.device, dtype: torch.dtype, main_std: float = 1.0, dyn_std: float = 3e-2) -> nn.Parameter:
    if main_std <= 0.0 or dyn_std <= 0.0:
        raise ValueError("Initialization can't be negative.")

    if isinstance(shape, int):
        shape = (shape,)
        
    # 1. Allocate zero tensor [degree, *shape]
    full_shape = (degree,) + shape
    tensor = torch.zeros(full_shape, device=device, dtype=dtype)
    
    # 2. Initialize Row 0 (The Mean/Static term)
    nn.init.normal_(tensor[0], mean=0.0, std=main_std)
        
    # 3. Initialize Rows 1 to degree-1 (The Variations)
    if degree > 1:
        time_std = main_std * dyn_std
        nn.init.normal_(tensor[1:], mean=0.0, std=time_std)
        
    return nn.Parameter(tensor)

def CreateZerosParameter(shape: Union[int, Tuple[int, ...]], degree: int, device: torch.device, dtype: torch.dtype) -> nn.Parameter:
    if isinstance(shape, int):
        shape = (shape,)
        
    full_shape = (degree,) + shape
    tensor = torch.zeros(full_shape, device=device, dtype=dtype)
    return nn.Parameter(tensor) 

def CreateConstParameter(shape: Union[int, Tuple[int, ...]], value: float, degree: int, device: torch.device, dtype: torch.dtype) -> nn.Parameter:
    if isinstance(shape, int):
        shape = (shape,)
        
    full_shape = (degree,) + shape
    tensor = torch.full(full_shape, value, device=device, dtype=dtype)
    return nn.Parameter(tensor)
