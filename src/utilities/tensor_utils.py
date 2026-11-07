import os
import torch
import numpy as np
from PIL import Image
from typing import Union, List, Tuple

# ===================================================================
# --- Constants ---
# ===================================================================

# Range: [-CR, CR] roughly corresponds to "safe" signal range for different functions (like tanh/xexp)
CONVERSION_RATIO = 1.0


# Automatic Precomputation
CONVERSION_FACTOR = 2.0 * CONVERSION_RATIO
INVERSE_CONVERSION_FACTOR = 1.0 / CONVERSION_FACTOR

# ===================================================================
# --- Conversion Functions ---
# ===================================================================

def normalize_tensor(tensor_01: torch.Tensor) -> torch.Tensor:
    """ Maps [0, 1] -> [-CR, CR] """
    return (tensor_01 - 0.5) * CONVERSION_FACTOR

def unnormalize_tensor(tensor_scaled: torch.Tensor) -> torch.Tensor:
    """ Maps [-CR, CR] -> [0, 1] """
    return torch.clamp((tensor_scaled / CONVERSION_FACTOR) + 0.5, 0.0, 1.0)

def ByteTensorToFloatTensor(byte_tensor: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    """
    [0, 255] -> [0, 1] -> [-0.5, 0.5] -> [-CR, CR]
    """
    return normalize_tensor(byte_tensor / 255.0)

def FloatTensorToByteTensor(float_tensor: torch.Tensor) -> torch.Tensor:
    """
    [-CR, CR] -> [-0.5, 0.5] -> [0, 1] -> [0, 255]
    """
    return torch.clamp(torch.round(unnormalize_tensor(float_tensor) * 255), 0, 255).to(torch.uint8)

def LoadImageAsTensor(image_path: str) -> torch.Tensor:
    """Loads image -> RGB -> Tensor [C, H, W] in float range."""
    try:
        img = Image.open(image_path).convert('RGB')
        img_np = np.array(img, dtype=np.float32)
        # HWC -> CHW
        img_np = np.transpose(img_np, (2, 0, 1))
        tensor = torch.from_numpy(img_np)
        return ByteTensorToFloatTensor(tensor)
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        raise

def FloatTensorToImage(tensor: torch.Tensor) -> Image.Image:
    """Tensor [C, H, W] -> PIL Image."""
    # Ensure CPU and detach
    t = tensor.detach().cpu()
    # Clamp to valid range before conversion
    t = torch.clamp(t, -CONVERSION_RATIO, CONVERSION_RATIO)
    byte_t = FloatTensorToByteTensor(t)
    # CHW -> HWC
    np_img = byte_t.permute(1, 2, 0).numpy()
    return Image.fromarray(np_img)

def NoiseImage_U(shape: tuple, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Returns a random noise tensor of a given shape in the project's float range."""
    return (torch.rand(shape, device=device, dtype=dtype) - 0.5) * CONVERSION_FACTOR

def NoiseImage_G(shape: tuple, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Returns a random noise tensor of a given shape in the project's float range."""
    return (torch.randn(shape, device=device, dtype=dtype)) * CONVERSION_RATIO

def PILImageToFloatTensor(pil_img: Image.Image) -> torch.Tensor:
    img_np = np.array(pil_img, dtype=np.float32)
    img_np = np.transpose(img_np, (2, 0, 1))
    tensor = torch.from_numpy(img_np)
    return ByteTensorToFloatTensor(tensor)

