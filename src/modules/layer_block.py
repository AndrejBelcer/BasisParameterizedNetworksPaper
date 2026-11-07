import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Any

from src.core.mappings_layers import get_layer_class
from src.core.enums import ActivationType
from src.layers.base_time_layer import BaseLayer
from src.layers.time_layers.matmul_layer import MatMulLayer
from src.utilities.debug_utilities import save_tensor_as_image_grid

# ===================================================================
# --- Helper Adapters for Branchless Forward ---
# ===================================================================

class IdentityResizer(nn.Module):
    """Pass-through for when spatial dimensions match perfectly."""
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return X

class SpatialResizer(nn.Module):
    """Parameter-free spatial interpolation."""
    def __init__(self, target_size: int):
        super().__init__()
        self.target_size = target_size
            
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return F.interpolate(X, size=(self.target_size, self.target_size), mode='bilinear', align_corners=True)


class MatMulProjectionAdapter(nn.Module):
    """Uses a Time-Dependent MatMul to project channels for a residual connection."""
    def __init__(self, matmul_layer: MatMulLayer, resizer: nn.Module):
        super().__init__()
        self.matmul = matmul_layer
        self.resizer = resizer
        
    def forward(self, X: torch.Tensor, L: torch.Tensor) -> torch.Tensor:
        X = self.resizer(X)
        return self.matmul(X, L)
        
class LayerBlock(BaseLayer):
    """
    A container that holds a sequence of Time-Dependent layers.
    It acts as a single layer with a defined input and output shape.
    """
    def __init__(self, config_list: List[Dict[str, Any]], a_in: int, x_in: int, degree: int = 4):
        super().__init__()
        
        self.layers: nn.ModuleList = nn.ModuleList()
        self.degree = degree
        
        current_a = a_in
        current_x = x_in
        
        for i, conf in enumerate(config_list):
            layer_type = conf.get('type')
            params = conf.get('params', {}).copy()
            
            LayerClass = get_layer_class(layer_type)
            
            params['a'] = current_a
            params['x'] = current_x
            params['degree'] = self.degree
            
            try:
                layer: BaseLayer = LayerClass(**params)
            except Exception as e:
                raise RuntimeError(f"LayerBlock: Failed to instantiate {layer_type} (Index {i}). Error: {e}")
            
            next_a, next_x = layer.Shape(current_a, current_x)
            current_a, current_x = next_a, next_x
            
            self.layers.append(layer)

        self.a_in = a_in
        self.x_in = x_in
        self.f_out = current_a
        self.y_out = current_x

    def forward(self, X: torch.Tensor, L: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            X = layer(X, L)
        return X

    def DebugForward(self, X: torch.Tensor, L: torch.Tensor, save_dir: str, block_idx: int) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            X = layer(X, L)
            fname = f"B{block_idx:02d}_L{i:02d}_{layer.__class__.__name__}.png"
            full_path = os.path.join(save_dir, fname)
            try:
                tensor_to_save = X[0] if X.ndim == 4 else X
                save_tensor_as_image_grid(tensor_to_save, full_path)
            except Exception as e:
                print(f"  [Debug] Failed to save {fname}: {e}")
        return X

    def AfterStep(self):
        for layer in self.layers:
            layer.AfterStep()

    def PrintLayerInfo(self, detailed_info: bool):
        print(f"--- {self.__class__.__name__} [{self.a_in}, {self.x_in}] -> [{self.f_out}, {self.y_out}] ---")
        for i, layer in enumerate(self.layers):
            print(f"  [{i}] ", end="")
            layer.PrintLayerInfo(detailed_info)

    def get_config(self) -> List[Dict[str, Any]]:
        config_list = []
        for layer in self.layers:
            config_list.append(layer.get_config())
        return config_list

class ProjectionResidualLayerBlock(LayerBlock):
    """
    A LayerBlock with a Learned Residual Connection.
    Implements: Output = Block(x) + MatMulProjection(x)
    """
    def __init__(self, config_list: List[Dict[str, Any]], a_in: int, x_in: int, degree: int = 4):
        super().__init__(config_list, a_in, x_in, degree)
        
        # 1. Resolve Spatial Resizing
        resizer = IdentityResizer() if self.x_in == self.y_out else SpatialResizer(self.y_out)

        # 2. Resolve Channel Matching via Time-Dependent MatMul
        # Note: MatMul takes the *output* spatial dimension because resizing happens first.
        matmul = MatMulLayer(f=self.f_out, a=self.a_in, x=self.y_out, degree=self.degree, activation_function=ActivationType.IDENTITY)
        self.res_adapter = MatMulProjectionAdapter(matmul, resizer)


    def forward(self, X: torch.Tensor, L: torch.Tensor) -> torch.Tensor:
        Y_main = super().forward(X, L)
        # Projection adapter requires L for the MatMul
        Y_res = self.res_adapter(X, L)
        return Y_main + Y_res

    def PrintLayerInfo(self, detailed_info: bool):
        super().PrintLayerInfo(detailed_info)
        print("  [Skip Connection Projection]")
        self.res_adapter.matmul.PrintLayerInfo(detailed_info)


