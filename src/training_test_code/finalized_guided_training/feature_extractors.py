"""Two explicit feature pipelines; both consume unquantized RGB floats in [0,1]."""
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision.models import inception_v3, Inception_V3_Weights

try:
    from cleanfid.inception_torchscript import InceptionV3W
except ImportError:
    InceptionV3W = None

from .image_resize import PILBicubicResize, PILLOW_VERSION
from .io_utils import file_identity


class TorchvisionFeatures:
    name = "torchvision"
    preprocessing = "RGB01-bilinear299-antialias-ImageNet-normalization"

    def __init__(self, runtime):
        self.runtime = runtime
        model = inception_v3(weights=Inception_V3_Weights.DEFAULT, transform_input=False)
        model.fc = torch.nn.Identity()
        model = model.to(device=runtime.device, dtype=runtime.model_dtype,
                         memory_format=torch.channels_last).eval().requires_grad_(False)
        self.model = runtime.compile(model, "torchvision Inception features")
        self.mean = torch.tensor([.485, .456, .406], device=runtime.device).view(1, 3, 1, 1)
        self.std = torch.tensor([.229, .224, .225], device=runtime.device).view(1, 3, 1, 1)

    def cache_details(self):
        return dict(extractor=self.name, weights=Inception_V3_Weights.DEFAULT.name)

    def __call__(self, images):
        batch = F.interpolate(images.float(), size=(299, 299), mode="bilinear",
                              align_corners=False, antialias=True)
        batch = ((batch-self.mean)/self.std).to(memory_format=torch.channels_last)
        with self.runtime.autocast():
            return self.model(batch).float().clone()


class CleanFeatures:
    name = "clean_float"
    preprocessing = "RGB01-float255-PIL-bicubic299-separable-v1-clamp-TF-Inception"

    def __init__(self, runtime):
        if InceptionV3W is None:
            raise ImportError("use_two_kid_models=True requires clean-fid; install it with pip install clean-fid")
        self.runtime = runtime
        weights_folder = Path("cache/finalized_guided_training/inception")
        weights_folder.mkdir(parents=True, exist_ok=True)
        self.model = InceptionV3W(str(weights_folder), download=True, resize_inside=False)
        self.model = self.model.to(device=runtime.device, dtype=runtime.model_dtype).eval().requires_grad_(False)
        self.weights_identity = file_identity(weights_folder/"inception-2015-12-05.pt")
        # Clean-FID ships a TorchScript network. It already has a compiled forward;
        # do not nest torch.compile around its unsupported ScriptModule boundary.
        print("Clean Inception: official TorchScript forward; precision follows GPU_CONFIG.", flush=True)
        resize = PILBicubicResize().to(runtime.device)
        error = resize.check_against_pil()
        print(f"Clean floating-point resize: max PIL difference={error:.6g}/255 (startup only).", flush=True)
        self.resize = runtime.compile(resize, "Clean floating-point resize")

    def cache_details(self):
        return dict(extractor=self.name, weights_sha256=self.weights_identity, pillow=PILLOW_VERSION)

    def __call__(self, images):
        # Clean-FID's clean path expects floating pixels in 0..255 after resizing.
        # No round(), byte(), uint8 conversion, PNG encode or CPU resize here.
        batch = self.resize(images.float()*255).clone()
        with self.runtime.autocast():
            return self.model(batch).float().clone()
