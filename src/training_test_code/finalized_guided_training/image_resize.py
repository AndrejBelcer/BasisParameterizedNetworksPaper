"""PIL's separable floating-point bicubic resize, evaluated in batches on GPU.

The one-dimensional interpolation weights are obtained from PIL itself once.
Horizontal and vertical passes preserve PIL's intermediate FP32 representation.
No integer conversion or pixel rounding is performed.
"""
import numpy as np
from PIL import Image, __version__ as PILLOW_VERSION
import torch


def pil_resize_weights(input_size, output_size):
    impulses = np.eye(input_size, dtype=np.float32)
    columns = []
    for impulse in impulses:
        resized = Image.fromarray(impulse[None, :]).resize(
            (output_size, 1), resample=Image.Resampling.BICUBIC)
        columns.append(np.array(resized, dtype=np.float32, copy=True)[0])
    return torch.from_numpy(np.stack(columns, axis=1))


class PILBicubicResize(torch.nn.Module):
    def __init__(self, input_height=32, input_width=32, output_size=299):
        super().__init__()
        self.input_shape = (input_height, input_width)
        self.output_size = output_size
        self.register_buffer("horizontal", pil_resize_weights(input_width, output_size))
        self.register_buffer("vertical", pil_resize_weights(input_height, output_size))

    def forward(self, images):
        if images.shape[-2:] != self.input_shape:
            raise ValueError("Clean KID resize expects the configured image dimensions")
        # Pixel preprocessing stays floating point, even under network autocast.
        # Backend FP32/TF32 policy is inherited from GPU_CONFIG, never overridden.
        with torch.autocast(device_type=images.device.type, enabled=False):
            horizontal = images.float() @ self.horizontal.T
            return (self.vertical @ horizontal).clamp(0, 255)

    @torch.no_grad()
    def check_against_pil(self):
        """One small startup check, before weights/candidates are evaluated.

        Matrix accumulation differs from PIL's CPU arithmetic; permit tiny
        numerical differences, but reject an incompatible resize implementation.
        """
        height, width = self.input_shape
        values = torch.rand(2, 3, height, width,
                            generator=torch.Generator().manual_seed(417)).mul_(255)
        values[0, :, :, :width//2] = 0
        values[0, :, :, width//2:] = 255
        expected = []
        for picture in values.numpy():
            channels = []
            for channel in picture:
                resized = Image.fromarray(channel).resize(
                    (self.output_size, self.output_size), Image.Resampling.BICUBIC)
                channels.append(np.array(resized, dtype=np.float32, copy=True).clip(0, 255))
            expected.append(np.stack(channels))
        actual = self(values.to(self.horizontal.device)).cpu()
        error = float((actual-torch.from_numpy(np.stack(expected))).abs().max())
        # In 0..255 pixel units. Much smaller than one 8-bit intensity level.
        if error > 0.002:
            raise RuntimeError(f"Clean KID GPU resize differs from PIL by {error:.6g}; "
                               "check backend precision before training")
        return error
