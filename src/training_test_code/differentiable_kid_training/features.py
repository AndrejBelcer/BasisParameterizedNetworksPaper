"""A float adapter around torch-fidelity's TensorFlow-compatible Inception.

Uses upstream weights, modules and interpolation. The module sequence and
normalization match torch-fidelity 0.4.0's pool3 path. Startup checks parity
against the installed integer-input implementation before training.
"""
import hashlib
import importlib.metadata
import inspect
import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from .reporting import atomic_save, file_hash


class FloatInception(torch.nn.Module):
    module_names = (
        'Conv2d_1a_3x3', 'Conv2d_2a_3x3', 'Conv2d_2b_3x3', 'MaxPool_1',
        'Conv2d_3b_1x1', 'Conv2d_4a_3x3', 'MaxPool_2',
        'Mixed_5b', 'Mixed_5c', 'Mixed_5d', 'Mixed_6a', 'Mixed_6b',
        'Mixed_6c', 'Mixed_6d', 'Mixed_6e', 'Mixed_7a', 'Mixed_7b',
        'Mixed_7c', 'AvgPool')

    def __init__(self, device, weights_path=None):
        super().__init__()
        try:
            from torch_fidelity.feature_extractor_inceptionv3 import FeatureExtractorInceptionV3, URL_INCEPTION_V3
            from torch_fidelity.interpolate_compat_tensorflow import interpolate_bilinear_2d_like_tensorflow1x
        except ImportError as exc:
            raise ImportError('Install torch-fidelity 0.4.0 and a working torchvision in the training environment.') from exc
        print('Loading reference-style Inception weights...', flush=True)
        self.official = FeatureExtractorInceptionV3(
            'inception-v3-compat', ['2048', 'logits_unbiased'],
            feature_extractor_weights_path=weights_path,
            feature_extractor_internal_dtype='float32', verbose=True).to(device).eval().requires_grad_(False)
        for name in self.module_names:
            if not hasattr(self.official, name):
                raise RuntimeError(f'Incompatible Inception implementation: missing {name}.')
        self.resize = interpolate_bilinear_2d_like_tensorflow1x
        weight_file = Path(weights_path) if weights_path else Path(torch.hub.get_dir())/'checkpoints'/Path(URL_INCEPTION_V3).name
        self.identity = {'weights': file_hash(weight_file),
                         'upstream_source': file_hash(inspect.getfile(FeatureExtractorInceptionV3)),
                         'adapter_source': file_hash(__file__),
                         'torch_fidelity': importlib.metadata.version('torch-fidelity'),
                         'torch': torch.__version__, 'preprocessing': 'TF-compatible float255, (x-128)/128'}
        self.eval()

    def pixels255(self, pixels):
        x = self.resize(pixels.float(), size=(299, 299), align_corners=False)
        x = (x-128)/128
        for name in self.module_names:
            x = getattr(self.official, name)(x)
        return x.flatten(1).float()

    def forward(self, images01):
        return self.pixels255(images01*255)

    def verify(self, device):
        pixels = torch.randint(0, 256, (2, 3, 32, 32), generator=torch.Generator().manual_seed(917)).to(device, torch.uint8)
        with torch.no_grad():
            expected = self.official(pixels)[0]
            actual = self.pixels255(pixels.float())
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=2e-5)
        x = (pixels.float()/255).requires_grad_(True)
        gradient = torch.autograd.grad(self(x).square().mean(), x)[0]
        if not torch.isfinite(gradient).all() or float(gradient.abs().max()) == 0:
            raise RuntimeError('Feature adapter has invalid image gradients.')
        print(f'Inception parity/input-gradient checks passed (max error {float((actual-expected).abs().max()):.3g}).', flush=True)

    @torch.no_grad()
    def evaluation(self, images01):
        pixels = (images01.clamp(0, 1)*255).round().to(torch.uint8)
        return self.official(pixels)


def reference_features(folder, extractor, device, cache_folder, batch_size=64):
    paths = sorted({p.resolve() for p in Path(folder).rglob('*.png')})
    if len(paths) < 2:
        raise ValueError(f'No usable reference PNG set in {folder}.')
    print(f'Identifying {len(paths)} reference PNGs...', flush=True)
    digest = hashlib.sha256(json.dumps(extractor.identity, sort_keys=True).encode())
    for p in paths:
        digest.update(str(p).encode())
        digest.update(file_hash(p).encode())
    key = digest.hexdigest()
    path = Path(cache_folder)/f'reference_{key[:24]}.pt'
    if path.exists():
        data = torch.load(path, map_location='cpu', weights_only=True)
        if data['identity'] != key or len(data['features']) != len(paths):
            raise RuntimeError('Invalid reference feature cache.')
        print(f'Reusing reference features: {path}', flush=True)
        return data['features'].to(device), key
    parts = []
    with torch.no_grad():
        for i in range(0, len(paths), batch_size):
            arrays = []
            for p in paths[i:i+batch_size]:
                with Image.open(p) as image:
                    arrays.append(torch.from_numpy(np.array(image.convert('RGB'), dtype=np.uint8, copy=True)).permute(2, 0, 1))
            pixels = torch.stack(arrays).to(device)
            # Exactly integer-valued pixels, using the parity-checked float path.
            parts.append(extractor.pixels255(pixels.float()).cpu())
            if i == 0 or i//batch_size % 100 == 0:
                print(f'Reference features: {min(i+batch_size,len(paths))}/{len(paths)}', flush=True)
    values = torch.cat(parts)
    if not torch.isfinite(values).all():
        raise FloatingPointError('Nonfinite real features.')
    atomic_save({'identity': key, 'features': values}, path)
    return values.to(device), key
