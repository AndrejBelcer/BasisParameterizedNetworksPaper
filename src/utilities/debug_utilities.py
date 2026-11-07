import os
import torch
import numpy as np
from PIL import Image
from typing import Union, List, Tuple







# ===================================================================
# --- Debugging ---
# ===================================================================

def save_tensor_as_image_grid(tensor: torch.Tensor, save_path: str, low: float = -2.0, high: float = 2.0):
    """
    Visualize a [C, H, W] tensor by creating a horizontal grid of its channels.
    - Values in [low, high] are mapped to grayscale [black, white] (0..1).
    - Values < low are colored blue.
    - Values > high are colored red.
    - NaN values are colored yellow.

    Args:
        tensor (torch.Tensor): input tensor with shape [C, H, W]
        save_path (str): output PNG path
        low (float): lower bound for grayscale mapping (default -2.0)
        high (float): upper bound for grayscale mapping (default 2.0)
    """
    if tensor.ndim != 3:
        raise ValueError(f"Input tensor must be 3D [C, H, W], but got shape {tensor.shape}")

    # ensure float for arithmetic and comparisons
    tensor = tensor.float()
    C, H, W = tensor.shape

    # Create canvas with a consistent float dtype
    canvas_dtype = torch.float32
    canvas = torch.zeros((C, H, W, 3), device=tensor.device, dtype=canvas_dtype)

    # Normalize values in [low, high] to [0, 1] for grayscale
    grayscale_part = (tensor.clamp(min=low, max=high) - low) / (high - low)
    grayscale_part = grayscale_part.to(dtype=canvas_dtype)

    # Fill R,G,B with the grayscale
    canvas[..., 0] = grayscale_part
    canvas[..., 1] = grayscale_part
    canvas[..., 2] = grayscale_part

    # Masks
    is_too_low = tensor < low
    is_too_high = tensor > high
    is_nan = torch.isnan(tensor)

    # Colors as same dtype & device as canvas
    blue = torch.tensor([0.0, 0.0, 1.0], device=tensor.device, dtype=canvas_dtype)
    red = torch.tensor([1.0, 0.0, 0.0], device=tensor.device, dtype=canvas_dtype)
    yellow = torch.tensor([1.0, 1.0, 0.0], device=tensor.device, dtype=canvas_dtype)

    # Assign colors (order: low, high, then NaN to override)
    canvas[is_too_low] = blue
    canvas[is_too_high] = red
    canvas[is_nan] = yellow

    # Convert to uint8 [0..255]
    canvas = canvas.clamp(0.0, 1.0)
    canvas_uint8 = (canvas * 255.0).to(dtype=torch.uint8).cpu().numpy()

    # Stitch horizontally: each channel becomes an H x W x 3 image
    stitched = np.concatenate([canvas_uint8[i] for i in range(C)], axis=1)

    # Ensure directory exists and save
    parent_dir = os.path.dirname(save_path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)
    Image.fromarray(stitched).save(save_path)

