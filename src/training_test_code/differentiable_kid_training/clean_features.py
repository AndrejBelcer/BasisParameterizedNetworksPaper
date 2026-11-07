"""Clean-FID features: differentiable training resize and official PIL evaluation."""
import importlib.metadata
import inspect
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .features import FloatInception
from .reporting import file_hash


def differentiable_clean_resize(pixels):
    """Continuous approximation of Clean-FID's per-channel PIL F-mode bicubic."""
    return F.interpolate(pixels.float(), size=(299, 299), mode='bicubic',
                         align_corners=False, antialias=True).clamp(0, 255)


class CleanFloatInception(torch.nn.Module):
    """Eager training gradients, checked against official clean Inception.

    pixels255() uses PIL for cached reference images. forward() resizes on the
    GPU with gradients through the eager TensorFlow-compatible backbone.
    The official TorchScript backbone is used only without gradients for
    references and monitoring. Evaluation quantizes before PIL resizing.
    """
    def __init__(self, device, weights_path=None):
        super().__init__()
        try:
            from cleanfid.inception_torchscript import InceptionV3W
            from cleanfid.resize import build_resizer
        except ImportError as exc:
            raise ImportError('Install clean-fid to use use_clean_fid=True.') from exc
        self.device = torch.device(device)
        self.resize_official = build_resizer('clean')
        cache = Path(torch.hub.get_dir())/'checkpoints'
        cache.mkdir(parents=True, exist_ok=True)
        print('Loading Clean-FID Inception weights (PIL bicubic protocol)...', flush=True)
        self.official = InceptionV3W(str(cache), download=True, resize_inside=False)
        self.official.to(self.device).eval().requires_grad_(False)
        # The released TorchScript artifact can fail in CUDA BatchNorm backward.
        # Keep it as the reference evaluator and use the eager implementation
        # for differentiation, with mandatory feature parity checked below.
        self.eager = FloatInception(self.device, weights_path)
        self.identity = {
            'protocol': 'clean-fid/clean',
            'weights': file_hash(cache/'inception-2015-12-05.pt'),
            'upstream_source': file_hash(inspect.getfile(InceptionV3W)),
            'resize_source': file_hash(inspect.getfile(build_resizer)),
            'adapter_source': file_hash(__file__),
            'clean_fid': importlib.metadata.version('clean-fid'),
            'Pillow': importlib.metadata.version('Pillow'),
            'torch': torch.__version__,
            'preprocessing': 'PIL F-mode bicubic 299x299, clip [0,255], (x-128)/128',
            'training_resize': 'torch bicubic, align_corners=False, antialias=True, clip [0,255]',
            'training_backbone': self.eager.identity,
        }
        self.eval()

    def forward(self, images01):
        x = differentiable_clean_resize(images01*255)
        x = (x-128)/128
        for name in self.eager.module_names:
            x = getattr(self.eager.official, name)(x)
        return x.flatten(1).float()

    @torch.no_grad()
    def pixels255(self, pixels):
        arrays = pixels.detach().cpu().permute(0, 2, 3, 1).numpy()
        resized = [torch.from_numpy(np.array(self.resize_official(a), copy=True)).permute(2, 0, 1)
                   for a in arrays]
        batch = torch.stack(resized).to(device=self.device, dtype=torch.float32)
        return self.official(batch).float()

    def verify(self, device):
        pixels = torch.randint(0, 256, (3, 3, 32, 32),
                               generator=torch.Generator().manual_seed(917)).to(device, torch.uint8)
        checker = (torch.arange(32)[:, None]+torch.arange(32)[None, :]) % 2*255
        pixels[1] = checker.to(device, torch.uint8)
        pixels[2] = torch.linspace(0, 255, 32, device=device).round().byte()[None, None, :]
        with torch.no_grad():
            expected = self.pixels255(pixels)
            actual = self(pixels.float()/255)
        torch.testing.assert_close(actual, expected, rtol=3e-4, atol=3e-4)
        for count in (1, 2, 3):
            x = (pixels[:count].float()/255).requires_grad_(True)
            # A non-leaf input mirrors the sampler -> resize -> Inception path.
            gradient = torch.autograd.grad(self(x*.9+.05).square().mean(), x)[0]
            if not torch.isfinite(gradient).all() or not bool(gradient.abs().max() > 0):
                raise RuntimeError('Clean-FID eager adapter has invalid image gradients.')
        print('Clean-FID official/eager feature parity and input-gradient checks passed '
              f'(max feature error {float((actual-expected).abs().max()):.3g}).', flush=True)

    @torch.no_grad()
    def evaluation(self, images01, compute_logits=True):
        pixels = (images01.clamp(0, 1)*255).round().to(torch.uint8)
        features = self.pixels255(pixels)
        logits = None
        if compute_logits:
            _, logits = self.eager.evaluation(images01)
        return features, logits
