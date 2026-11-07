"""Independent full-interval RGB gradients from cached image/noise pairs."""
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.utils.data import DataLoader, TensorDataset

from src.training_helpers.time_sampler import TimeSampler, TimeSamplingType
from .io_utils import load_pairs
from .parameter_space import require_finite, vector_norm


@dataclass
class Guides:
    directions: list
    losses: list
    norms: list

    def agreement(self):
        return [dict(first=i+1, second=j+1, cosine=float(torch.dot(a, b)))
                for i, a in enumerate(self.directions) for j, b in enumerate(self.directions)
                if i < j and self.norms[i] and self.norms[j]]

    def step_alignment(self, displacement):
        """Raw mean RGB gradient dot step, plus cosine with RGB descent.

        Uses cached CPU gradients; no additional forward/backward computation.
        Negative dot product predicts a decrease of RGB loss to first order.
        """
        mean = torch.zeros_like(displacement)
        for direction, norm in zip(self.directions, self.norms):
            mean.add_(direction, alpha=norm/len(self.directions))
        denominator = vector_norm(mean)*vector_norm(displacement)
        dot = float(torch.dot(displacement, mean))
        return dot, -dot/denominator if denominator else None


class RGBGuidance:
    def __init__(self, model, space, runtime, *, cache_path, num_workers,
                 num_times, batch_size_per_time, grad_accum_steps,
                 gradient_min_time, gradient_max_time, num_rgb_guides, seed):
        self.model, self.space, self.runtime = model, space, runtime
        self.num_times = num_times
        self.batch_size_per_time = batch_size_per_time
        self.grad_accum_steps = grad_accum_steps
        self.num_rgb_guides = num_rgb_guides
        images, noises = load_pairs(cache_path)
        size = num_times*batch_size_per_time
        if len(images) < size:
            raise ValueError("Pair cache is smaller than one full RGB batch")
        self.loader = DataLoader(TensorDataset(images, noises), batch_size=size, shuffle=True,
                                 drop_last=True, num_workers=num_workers,
                                 persistent_workers=num_workers > 0,
                                 pin_memory=runtime.device.type == "cuda",
                                 generator=torch.Generator().manual_seed(seed))
        self.iterator = None
        self.sampler = TimeSampler(method=TimeSamplingType.STRATIFIED_UNIFORM,
                                   lowest_t=gradient_min_time,
                                   highest_t=gradient_max_time, device="cpu")
        self.forward = runtime.compile(torch.vmap(model, in_dims=(0, 0), randomness="error"), "RGB vmap forward")
        self.loss_scale = 1024. if runtime.use_autocast and runtime.dtype == torch.float16 else 1.

    def _batch(self):
        if self.iterator is None:
            self.iterator = iter(self.loader)
        try:
            return next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.loader)
            return next(self.iterator)

    def collect(self):
        rt = self.runtime
        directions, losses, norms = [], [], []
        old_mode = self.model.training
        self.model.eval()
        try:
            for _ in range(self.num_rgb_guides):
                self.model.zero_grad(set_to_none=True)
                loss_sum = torch.zeros((), device=rt.device)
                for _ in range(self.grad_accum_steps):
                    images, noises = self._batch()
                    shape = (self.num_times, self.batch_size_per_time, 3, 32, 32)
                    g = images.to(rt.device, dtype=rt.model_dtype, non_blocking=True).view(shape)
                    n = noises.to(rt.device, dtype=rt.model_dtype, non_blocking=True).view(shape)
                    t = self.sampler.sample(self.num_times).to(rt.device, dtype=rt.model_dtype)
                    target = n-g
                    x = g+t.view(-1, 1, 1, 1, 1)*target
                    with sdpa_kernel(SDPBackend.MATH), rt.autocast():
                        prediction = self.forward(x, t[:, None])
                        loss = F.mse_loss(prediction.float(), target.float())
                    (loss*(self.loss_scale/self.grad_accum_steps)).backward()
                    loss_sum.add_(loss.detach(), alpha=1/self.grad_accum_steps)
                require_finite(loss_sum, "RGB loss")
                gradient = self.space.gradient().div_(self.loss_scale)
                require_finite(gradient, "RGB gradient")
                norm = vector_norm(gradient)
                directions.append(gradient.div_(norm) if norm else gradient)
                losses.append(float(loss_sum))
                norms.append(norm)
        except FloatingPointError:
            self.loss_scale = max(1., self.loss_scale*.5)
            raise
        finally:
            self.model.zero_grad(set_to_none=True)
            self.model.train(old_mode)
        return Guides(directions, losses, norms)
