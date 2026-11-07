import math
import torch
import torch.nn as nn
from typing import Optional
from enum import Enum
import numpy as np
import scipy.stats as stats

from src.core.globals import GPU_CONFIG

class TimeSamplingType(str, Enum):
    """The type of sampling distribution for time steps."""
    STRATIFIED_UNIFORM = "stratified_uniform"      # Stratified / Latin Hypercube (Default)
    STRATIFIED_EXPONENT = "stratified_exponent"
    STRATIFIED_BETA = "stratified_beta"
    RANDOM_UNIFORM = "random_uniform_sampling"  # Pure random
    RANDOM_LOGIT = "logit_normal_sampling"      # Gaussian on logit scale
    RANDOM_POWER = "power_sampling"             # Power law distribution
    DETERMINISTIC_UNIFORM = "deterministic_uniform"
    DETERMINISTIC_UNIFORM_BOUNDARY = "deterministic_uniform_boundary"


# ==========================================
# Samplers (Generate Uniform [0, 1] Samples)
# ==========================================

class BaseSampler:
    """Base class for uniform time step samplers."""
    def __init__(self, device: torch.device):
        self.device = device

    def sample(self, k: int) -> torch.Tensor:
        raise NotImplementedError


class UniformSampler(BaseSampler):
    """Standard Uniform sampling in [0, 1]."""
    def sample(self, k: int) -> torch.Tensor:
        return torch.rand(k, device=self.device, dtype=GPU_CONFIG.dtype)


class StratifiedSampler(BaseSampler):
    """Stratified Sampling (Latin Hypercube in 1D) over [0, 1]."""
    def sample(self, k: int) -> torch.Tensor:
        steps = torch.arange(k, device=self.device, dtype=GPU_CONFIG.dtype)
        jitter = torch.rand(k, device=self.device, dtype=GPU_CONFIG.dtype)
        return (steps + jitter) / k


class DeterministicSampler(BaseSampler):
    """Equally spaced deterministic sampling over (0, 1) excluding boundaries."""
    def sample(self, k: int) -> torch.Tensor:
        # arange(1, k + 1) / (k + 1) is equivalent to linspace(0, 1, k + 2)[1:-1]
        # but avoids allocating and slicing boundary elements.
        return torch.arange(1, k + 1, device=self.device, dtype=GPU_CONFIG.dtype) / (k + 1)


class DeterministicSamplerBoundary(BaseSampler):
    """Equally spaced deterministic sampling over [0, 1], including boundaries."""
    def sample(self, k: int) -> torch.Tensor:
        return torch.linspace(0.0, 1.0, k, device=self.device, dtype=GPU_CONFIG.dtype)

# ==========================================
# Transforms (Map Uniform to Target Dist)
# ==========================================

class BaseTransform:
    """Base class for transforming uniform samples to other distributions."""
    def __init__(self, device: torch.device):
        self.device = device

    def __call__(self, u: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class IdentityTransform(BaseTransform):
    """Passes uniform samples through unchanged."""
    def __call__(self, u: torch.Tensor) -> torch.Tensor:
        return u


class PowerTransform(BaseTransform):
    """Power law distribution: t = u^exponent."""
    def __init__(self, exponent: float, device: torch.device):
        super().__init__(device)
        self.exponent = exponent

    def __call__(self, u: torch.Tensor) -> torch.Tensor:
        return torch.pow(u, self.exponent)


class LogitTransform(BaseTransform):
    """Maps uniform samples to a LogitNormal distribution."""
    def __init__(self, mean: float, sigma: float, device: torch.device):
        super().__init__(device)
        self.mean = mean
        self.sigma = sigma
        self.sqrt_2 = math.sqrt(2.0)

    def __call__(self, u: torch.Tensor) -> torch.Tensor:
        # Clamp slightly to avoid numerical instability at exact boundaries 0 and 1
        u_clamped = torch.clamp(u, min=1e-7, max=1.0 - 1e-7)
        # Inverse CDF of a standard normal distribution
        normal_samples = torch.erfinv(2.0 * u_clamped - 1.0) * self.sqrt_2
        # Scale, shift, and map to (0, 1) using Sigmoid
        scaled_samples = normal_samples * self.sigma + self.mean
        return torch.sigmoid(scaled_samples)


class BetaTransform(BaseTransform):
    """Transforms uniform samples to a Beta distribution using a precomputed GPU LUT."""
    def __init__(self, alpha: float, beta: float, device: torch.device, lut_size: int = 16384):
        super().__init__(device)
        self.lut_size = lut_size
        self.scale = float(lut_size - 1)
        self.max_index = lut_size - 1

        # Precompute the inverse CDF (PPF) on the CPU
        quantiles = np.linspace(0.0, 1.0, lut_size, dtype=np.float32)
        lut_values = stats.beta.ppf(quantiles, alpha, beta)

        # Cast and send directly to the target device during initialization
        self.lut = torch.tensor(lut_values, dtype=GPU_CONFIG.dtype, device=device)

    def __call__(self, u: torch.Tensor) -> torch.Tensor:
        # Scale to LUT index space
        idx = u * self.scale
        
        # Get surrounding indices
        i0 = idx.long()
        i1 = torch.clamp(i0 + 1, max=self.max_index)
        
        # Compute weights and interpolate
        w = idx - i0
        return torch.lerp(self.lut[i0], self.lut[i1], w)


# ==========================================
# Orchestrator (Unified Interface)
# ==========================================

class TimeSampler:
    """
    Generates time steps 't' for training batches.

    Pipeline:
        1. Sampler draws u in [0, 1) (stratified or i.i.d.).
        2. Transform maps u to t in [0, 1] with the desired distribution shape.
        3. Affine range map squeezes [0, 1] onto [lowest_t, highest_t],
           preserving the shape of the distribution (no clamping artifacts).
        4. Final clamp is only a numerical safety net (no-op for current transforms).
        5. Sorting ensures the output is always descendingly sorted.
    """
    def __init__(self,
                 method: TimeSamplingType = TimeSamplingType.STRATIFIED_UNIFORM,
                 alpha: float = 1.5,       # alpha for beta distribution
                 beta: float = 1.3,        # beta for beta distribution
                 sigma: float = 1.0,       # sigma for random_logit
                 mean: float = 0.0,        # mean for random_logit
                 exponent: float = 0.65,   # exponent for power / stratified_exponent
                 lowest_t: float = 0.08,    # lower bound of the training time interval
                 highest_t: float = 1.0 - 1e-3,  # upper bound of the training time interval
                 device: Optional[torch.device] = None,
                 ):
        """
        Args:
            method (TimeSamplingType): Sampling strategy.
            alpha (float): Beta distribution parameter.
            beta (float): Beta distribution parameter.
            sigma (float): Standard deviation for LogitNormal.
            mean (float): Mean for LogitNormal.
            exponent (float): Exponent power.
            lowest_t (float): Lower bound of the sampled interval. The model is
                never trained below this time.
            highest_t (float): Upper bound of the sampled interval.
            device (torch.device): Target device for tensors.

        Note:
            The transform parameters (mean, sigma, alpha, beta, exponent)
            describe the distribution *on the remapped interval*
            [lowest_t, highest_t], not on [0, 1].
        """

        # Dynamically resolve device at instantiation time
        self.device: torch.device = GPU_CONFIG.device if device is None else torch.device(device)

        if not (0.0 < lowest_t < highest_t <= 1.0):
            raise ValueError(f"Invalid time range: require 0.0 < lowest_t < highest_t <= 1.0, got lowest_t={lowest_t}, highest_t={highest_t}.")

        self.lowest_t = float(lowest_t)
        self.highest_t = float(highest_t)
        self.t_span = self.highest_t - self.lowest_t

        # 1. Assign Sampler & Determine if it naturally produces ascending values
        if method in (TimeSamplingType.STRATIFIED_UNIFORM, TimeSamplingType.STRATIFIED_EXPONENT, TimeSamplingType.STRATIFIED_BETA):
            self.sampler = StratifiedSampler(device=self.device)
            self.is_ascending = True
        elif method in (TimeSamplingType.RANDOM_UNIFORM, TimeSamplingType.RANDOM_LOGIT, TimeSamplingType.RANDOM_POWER):
            self.sampler = UniformSampler(device=self.device)
            self.is_ascending = False
        elif method in (TimeSamplingType.DETERMINISTIC_UNIFORM):
            self.sampler = DeterministicSampler(device=self.device)
            self.is_ascending = True
        elif method in (TimeSamplingType.DETERMINISTIC_UNIFORM_BOUNDARY):
            self.sampler = DeterministicSamplerBoundary(device=self.device)
            self.is_ascending = True
        else:
            raise ValueError(f"Unknown sampling method: {method}")

        # 2. Assign Transform
        if method in (TimeSamplingType.STRATIFIED_UNIFORM, TimeSamplingType.RANDOM_UNIFORM, TimeSamplingType.DETERMINISTIC_UNIFORM, TimeSamplingType.DETERMINISTIC_UNIFORM_BOUNDARY):
            self.transform = IdentityTransform(device=self.device)
        elif method in (TimeSamplingType.STRATIFIED_EXPONENT, TimeSamplingType.RANDOM_POWER):
            self.transform = PowerTransform(exponent=exponent, device=self.device)
        elif method == TimeSamplingType.RANDOM_LOGIT:
            self.transform = LogitTransform(mean=mean, sigma=sigma, device=self.device)
        elif method == TimeSamplingType.STRATIFIED_BETA:
            self.transform = BetaTransform(alpha=alpha, beta=beta, device=self.device)
        else:
            raise ValueError(f"Unknown sampling method: {method}")

    def sample(self, k: int) -> torch.Tensor:
        """
        Returns a tensor of shape [k] containing time values in
        [lowest_t, highest_t], guaranteed to be sorted in descending order.
        """
        u = self.sampler.sample(k)
        t = self.transform(u)

        # Affine remap of the full [0, 1] range onto [lowest_t, highest_t].
        # Preserves the distribution shape (stratification, beta, logit-normal,
        # power) instead of piling up probability mass at the boundaries,
        # which is what a clamp would do now that lowest_t is large.
        t = self.lowest_t + self.t_span * t

        # Numerical safety net; a no-op for the current transforms.
        t = torch.clamp(t, min=self.lowest_t, max=self.highest_t)

        # 5. Ensure descending order efficiently
        if self.is_ascending:
            # Since the base sampler is ascending and all transforms are monotonically 
            # increasing, the array is already ascending. We just flip it (O(1) view).
            t = torch.flip(t, dims=[0]).contiguous()
        else:
            # For random samplers, we must perform an actual sort.
            t, _ = torch.sort(t, descending=True)

        return t
