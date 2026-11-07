import os
import hashlib
from abc import ABC, abstractmethod
from typing import Optional, Tuple, List, Union, Dict

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import inception_v3, Inception_V3_Weights
from torch.optim.swa_utils import AveragedModel

from src.core.globals import GPU_CONFIG
from src.core.enums import CompileMode
from src.utilities.tensor_utils import unnormalize_tensor
from src.training_helpers.time_sampler import TimeSampler
from src.modules.velocity_model import VelocityModel


# =====================================================================
# --- Abstract Base Class ---
# =====================================================================

class BaseValidationTracker(ABC):
    """
    Abstract Base Class for Flow Matching validation trackers.
    Handles deterministic noise generation, ODE Euler trajectory sampling,
    and direction-agnostic score improvement tracking.
    """
    def __init__(
        self,
        time_sampler: TimeSampler,
        device: torch.device,
        batch_size: int = 2048,
        num_batches: int = 2,
        channels: int = 3,
        img_size: int = 32,
        steps: int = 30,
        seed: int = 11,
        metric_chunk_size: Optional[int] = 512,
    ):
        self.device = device
        self.batch_size = batch_size
        self.num_batches = num_batches
        self.channels = channels
        self.img_size = img_size
        self.steps = steps
        self.metric_chunk_size = metric_chunk_size if metric_chunk_size else batch_size

        # ImageNet normalization statistics as fixed GPU broadcasting tensors
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=device, dtype=torch.float32).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=device, dtype=torch.float32).view(1, 3, 1, 1)

        # Precompute deterministic noise for ALL batches
        total_images = num_batches * batch_size
        print(f"[{self.__class__.__name__}] Pre-computing deterministic noise for {total_images} images (Seed: {seed})...")
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)
        self.fixed_noise = torch.randn(
            (total_images, channels, img_size, img_size), 
            device=device, 
            dtype=GPU_CONFIG.dtype, 
            generator=generator
        )

        # Precompute Time Grid
        # The TimeSampler now guarantees that sampled_t is sorted descendingly.
        sampled_t = time_sampler.sample(steps - 1)
        t_boundary_start = torch.tensor([1.0], device=device, dtype=GPU_CONFIG.dtype)
        t_boundary_end = torch.tensor([0.0], device=device, dtype=GPU_CONFIG.dtype)

        # Concatenating [1.0, descending_array, 0.0] naturally yields a perfectly sorted descending array.
        t_grid = torch.cat([t_boundary_start, sampled_t, t_boundary_end])

        # Precompute tensors and Python floats to eliminate GPU-CPU sync stalls (.item())
        t_steps = t_grid[:-1]
        dt_steps = t_grid[1:] - t_grid[:-1]
        self.list_t_tensor: List[torch.Tensor] = [t.view(1) for t in t_steps]
        self.list_delta_t_float: List[float] = [float(dt.item()) for dt in dt_steps]

        # Lazy cache for compiled models: id(raw_model) -> compiled_model
        self._compiled_cache: Dict[int, nn.Module] = {}

        # Best score tracking (direction-agnostic)
        self.best_score_base: float = -float("inf") if self.higher_is_better else float("inf")
        self.best_score_ema: float = -float("inf") if self.higher_is_better else float("inf")

        print(f"Prepared validator with {total_images} noises ({num_batches} batches of {batch_size}).")

    @property
    @abstractmethod
    def metric_name(self) -> str:
        """Name of the metric (e.g., 'IS', 'KID')."""
        pass

    @property
    @abstractmethod
    def higher_is_better(self) -> bool:
        """True if maximizing (IS), False if minimizing (KID, FID)."""
        pass

    @abstractmethod
    def _compute_metric(self, images: torch.Tensor) -> float:
        """Computes metric given generated images in [0, 1] range of shape [B, C, H, W]."""
        pass

    def _is_better(self, current_score: float, best_score: float) -> bool:
        """Checks if current score represents an improvement."""
        if self.higher_is_better:
            return current_score > best_score
        return current_score < best_score

    def _get_compiled_model(self, model: Union[VelocityModel, AveragedModel]) -> nn.Module:
        """
        Retrieves or lazily compiles the underlying VelocityModel.
        Does not mutate the raw model, keeping training vmap safe.
        """
        raw_model: VelocityModel = model.module if isinstance(model, AveragedModel) else model
        model_id = id(raw_model)

        if model_id not in self._compiled_cache:
            compile_mode = GPU_CONFIG.compile_mode_velocity
            if GPU_CONFIG.compile_model and compile_mode != CompileMode.NONE:
                print(f"[{self.__class__.__name__}] Compiling {raw_model.__class__.__name__} for validation (Mode: {compile_mode.value})...")
                try:
                    compiled = torch.compile(raw_model, mode=compile_mode.value, fullgraph=True)
                    self._compiled_cache[model_id] = compiled
                except Exception as e:
                    print(f"[{self.__class__.__name__}] Model compilation fallback: {e}")
                    self._compiled_cache[model_id] = raw_model
            else:
                self._compiled_cache[model_id] = raw_model

        return self._compiled_cache[model_id]

    @torch.no_grad()
    def _generate_images(self, model: nn.Module) -> torch.Tensor:
        """
        Fast Euler integration without GPU-CPU synchronization stalls.
        Denoises images in `num_batches` segments to manage VRAM efficiently.
        Returns unnormalized [0, 1] clamped images.
        """
        generated_batches = []

        for i in range(self.num_batches):
            start_idx = i * self.batch_size
            end_idx = start_idx + self.batch_size
            
            # Clone only the segment we are currently denoising
            x = self.fixed_noise[start_idx:end_idx].clone()

            with torch.amp.autocast(device_type=self.device.type, dtype=GPU_CONFIG.dtype, enabled=GPU_CONFIG.use_autocast):
                # Pure asynchronous dispatch: no .item() host synchronizations
                for t_tensor, dt in zip(self.list_t_tensor, self.list_delta_t_float):
                    v = model(x, t_tensor)
                    x.add_(v, alpha=dt)
            
            generated_batches.append(x)

        # Concatenate all generated segments into one large tensor
        all_generated_images = torch.cat(generated_batches, dim=0)
        return unnormalize_tensor(all_generated_images).clamp(0.0, 1.0)

    def evaluate(
        self,
        base_model: Union[VelocityModel, AveragedModel],
        ema_model: Optional[Union[VelocityModel, AveragedModel]] = None,
    ) -> Tuple[float, float, bool, bool]:
        """
        Evaluates models using their compiled counterparts, updates best scores,
        and restores training mode.
        Returns: (base_score, ema_score, base_improved, ema_improved)
        """
        base_improved = False
        ema_improved = False
        ema_score = 0.0

        # --- 1. Base Model ---
        raw_base: VelocityModel = base_model.module if isinstance(base_model, AveragedModel) else base_model
        raw_base.eval()
        compiled_base = self._get_compiled_model(raw_base)
        base_images = self._generate_images(compiled_base)
        base_score = self._compute_metric(base_images)
        raw_base.train()

        if self._is_better(base_score, self.best_score_base):
            self.best_score_base = base_score
            base_improved = True

        # --- 2. EMA Model ---
        if ema_model is not None:
            raw_ema: VelocityModel = ema_model.module if isinstance(ema_model, AveragedModel) else ema_model
            raw_ema.eval()
            compiled_ema = self._get_compiled_model(raw_ema)
            ema_images = self._generate_images(compiled_ema)
            ema_score = self._compute_metric(ema_images)
            raw_ema.train()

            if self._is_better(ema_score, self.best_score_ema):
                self.best_score_ema = ema_score
                ema_improved = True

        return base_score, ema_score, base_improved, ema_improved


# =====================================================================
# --- Inception Score Implementation ---
# =====================================================================

class InceptionScoreTracker(BaseValidationTracker):
    """Computes Inception Score (IS). Higher is better."""
    def __init__(
        self,
        time_sampler: TimeSampler,
        device: torch.device,
        batch_size: int = 2048,
        num_batches: int = 2,
        channels: int = 3,
        img_size: int = 32,
        steps: int = 30,
        seed: int = 11,
        metric_chunk_size: Optional[int] = 512,
    ):
        print("[InceptionScoreTracker] Initializing InceptionV3 for Inception Score...")
        self.inception = inception_v3(weights=Inception_V3_Weights.DEFAULT, transform_input=False).to(device=device, memory_format=torch.channels_last)
        self.inception.eval()

        if GPU_CONFIG.compile_model:
            try:
                self.inception = torch.compile(self.inception, mode=GPU_CONFIG.compile_mode_velocity.value)
            except Exception as e:
                print(f"[InceptionScoreTracker] Compilation fallback: {e}")

        super().__init__(
            time_sampler=time_sampler, 
            device=device,
            batch_size=batch_size, 
            num_batches=num_batches,
            channels=channels, 
            img_size=img_size, 
            steps=steps, 
            seed=seed, 
            metric_chunk_size=metric_chunk_size
        )
        self._warmup()

    @property
    def metric_name(self) -> str:
        return "IS"

    @property
    def higher_is_better(self) -> bool:
        return True

    @torch.no_grad()
    def _warmup(self):
        dummy = torch.zeros((self.metric_chunk_size, 3, 299, 299), device=self.device, dtype=torch.float32).to(memory_format=torch.channels_last)
        with torch.amp.autocast(device_type=self.device.type, dtype=GPU_CONFIG.dtype, enabled=GPU_CONFIG.use_autocast):
            _ = self.inception(dummy)

    @torch.no_grad()
    def _compute_metric(self, images: torch.Tensor) -> float:
        probs_list = []
        num_images = images.size(0)

        for i in range(0, num_images, self.metric_chunk_size):
            batch = images[i : i + self.metric_chunk_size]
            batch_resized = F.interpolate(batch.float(), size=(299, 299), mode="bilinear", align_corners=False, antialias=True)
            batch_norm = ((batch_resized - self.mean) / self.std).to(memory_format=torch.channels_last)

            with torch.amp.autocast(device_type=self.device.type, dtype=GPU_CONFIG.dtype, enabled=GPU_CONFIG.use_autocast):
                logits = self.inception(batch_norm)

            probs_list.append(F.softmax(logits.float(), dim=1))

        probs = torch.cat(probs_list, dim=0)
        p_y = probs.mean(dim=0, keepdim=True)
        kl = probs * (torch.log(probs + 1e-16) - torch.log(p_y + 1e-16))
        return torch.exp(kl.sum(dim=1).mean()).item()


# =====================================================================
# --- Kernel Inception Distance (KID) Implementation ---
# =====================================================================

class KIDTracker(BaseValidationTracker):
    """
    Computes Kernel Inception Distance (KID). Lower is better.
    Uses Inception-V3 2048-D features and polynomial kernel k(x, y) = ((x^T y)/d + 1)^3.
    Precomputes and caches reference features and self-kernel constant to cache/validator/.
    """
    def __init__(
        self,
        reference_folder: str,
        time_sampler: TimeSampler,
        device: torch.device,
        batch_size: int = 2048,
        num_batches: int = 2,
        channels: int = 3,
        img_size: int = 32,
        steps: int = 30,
        seed: int = 11,
        metric_chunk_size: Optional[int] = 512,
        cache_dir: str = os.path.join("cache", "validator"),
    ):
        self.reference_folder = reference_folder
        self.cache_dir = cache_dir

        print("[KIDTracker] Initializing InceptionV3 feature extractor...")
        self.inception = inception_v3(weights=Inception_V3_Weights.DEFAULT, transform_input=False).to(device=device, memory_format=torch.channels_last)
        self.inception.fc = nn.Identity()  # Outputs 2048-D pool3 features directly
        self.inception.eval()

        if GPU_CONFIG.compile_model:
            try:
                self.inception = torch.compile(self.inception, mode=GPU_CONFIG.compile_mode_velocity.value)
            except Exception as e:
                print(f"[KIDTracker] Compilation fallback: {e}")

        super().__init__(
            time_sampler=time_sampler, 
            device=device,
            batch_size=batch_size, 
            num_batches=num_batches,
            channels=channels, 
            img_size=img_size, 
            steps=steps, 
            seed=seed, 
            metric_chunk_size=metric_chunk_size
        )

        self._warmup()

        # Load or compute cached reference features & self-kernel constant
        self.ref_features, self.c_ref = self._load_or_create_reference_cache()

    @property
    def metric_name(self) -> str:
        return "KID"

    @property
    def higher_is_better(self) -> bool:
        return False

    @torch.no_grad()
    def _warmup(self):
        dummy = torch.zeros((self.metric_chunk_size, 3, 299, 299), device=self.device, dtype=torch.float32).to(
            memory_format=torch.channels_last
        )
        with torch.amp.autocast(device_type=self.device.type, dtype=GPU_CONFIG.dtype, enabled=GPU_CONFIG.use_autocast):
            _ = self.inception(dummy)

    def _collect_reference_png_files(self) -> List[str]:
        """Scans the reference folder recursively for .png files without duplicates."""
        if not os.path.exists(self.reference_folder):
            raise FileNotFoundError(f"Reference folder not found: {self.reference_folder}")

        seen_paths = set()
        file_list = []

        for root, _, files in os.walk(self.reference_folder):
            for fname in files:
                if fname.lower().endswith(".png"):
                    full_path = os.path.join(root, fname)
                    # Resolve symlinks and normalize path separators
                    canonical = os.path.realpath(os.path.abspath(full_path))
                    if canonical not in seen_paths:
                        seen_paths.add(canonical)
                        file_list.append(canonical)

        if not file_list:
            raise ValueError(f"No .png images found in reference folder: {self.reference_folder}")

        # Deterministic order across all operating systems
        file_list.sort()
        return file_list

    def _get_cache_path(self, file_count: int) -> str:
        """Generates an OS-independent, deterministic cache path."""
        norm_ref_path = os.path.normpath(os.path.realpath(os.path.abspath(self.reference_folder))).replace("\\", "/").lower()
        path_hash = hashlib.sha256(f"{norm_ref_path}_{file_count}".encode("utf-8")).hexdigest()[:12]
        folder_base = os.path.basename(os.path.normpath(self.reference_folder))
        filename = f"kid_cache_{folder_base}_{path_hash}.pt"
        return os.path.join(self.cache_dir, filename)

    @torch.no_grad()
    def _extract_features(self, images_01: torch.Tensor) -> torch.Tensor:
        """Extracts 2048-D Inception features from [0, 1] images in small chunks."""
        features_list = []
        num_images = images_01.size(0)

        for i in range(0, num_images, self.metric_chunk_size):
            batch = images_01[i : i + self.metric_chunk_size]
            batch_resized = F.interpolate(batch.float(), size=(299, 299), mode="bilinear", align_corners=False, antialias=True)
            batch_norm = ((batch_resized - self.mean) / self.std).to(memory_format=torch.channels_last)

            with torch.amp.autocast(device_type=self.device.type, dtype=GPU_CONFIG.dtype, enabled=GPU_CONFIG.use_autocast):
                feat = self.inception(batch_norm)

            features_list.append(feat.float())

        return torch.cat(features_list, dim=0)

    @torch.no_grad()
    def _compute_self_kernel_constant(self, features: torch.Tensor) -> float:
        """
        Computes C_ref = 1 / (N * (N - 1)) * sum_{i != j} k(y_i, y_j)
        in blocks to prevent GPU OOM on large reference sets.
        """
        N = features.size(0)
        inv_d = 1.0 / features.size(1)
        block_size = 5000
        total_sum = 0.0

        for i in range(0, N, block_size):
            feat_i = features[i : i + block_size]
            for j in range(0, N, block_size):
                feat_j = features[j : j + block_size]
                k_ij = torch.pow((feat_i @ feat_j.t()) * inv_d + 1.0, 3)
                total_sum += k_ij.sum().item()

        # Subtract self-diagonal elements k(y_i, y_i)
        diag_dots = (features * features).sum(dim=1)
        diag_sum = torch.pow(diag_dots * inv_d + 1.0, 3).sum().item()

        c_ref = (total_sum - diag_sum) / (N * (N - 1))
        return float(c_ref)

    def _load_or_create_reference_cache(self) -> Tuple[torch.Tensor, float]:
        """Loads cached reference features and C_ref or computes and saves them."""
        os.makedirs(self.cache_dir, exist_ok=True)
        file_list = self._collect_reference_png_files()
        cache_file = self._get_cache_path(len(file_list))

        if os.path.exists(cache_file):
            print(f"[KIDTracker] Loading cached reference data from: {cache_file}")
            cached_data = torch.load(cache_file, map_location=self.device)
            ref_features = cached_data["ref_features"].to(self.device)
            c_ref = cached_data["c_ref"]
            print(f"[KIDTracker] Loaded {ref_features.size(0)} reference features. C_ref: {c_ref:.6f}")
            return ref_features, c_ref

        print(f"[KIDTracker] Cache miss. Computing features for {len(file_list)} reference PNGs...")
        load_batch_size = 512
        all_features = []

        for i in range(0, len(file_list), load_batch_size):
            batch_paths = file_list[i : i + load_batch_size]
            tensors = []
            for path in batch_paths:
                with Image.open(path) as img:
                    img = img.convert("RGB")
                    arr = np.array(img, dtype=np.float32) / 255.0  # Range: [0, 1]
                    tensors.append(torch.from_numpy(arr).permute(2, 0, 1))

            batch_tensor = torch.stack(tensors).to(self.device)
            feat = self._extract_features(batch_tensor)
            all_features.append(feat)

        ref_features = torch.cat(all_features, dim=0)

        print("[KIDTracker] Computing reference self-kernel constant C_ref...")
        c_ref = self._compute_self_kernel_constant(ref_features)

        # Cache reference features to disk
        cache_data = {
            "ref_features": ref_features.cpu(),
            "c_ref": c_ref,
            "num_samples": len(file_list),
        }
        torch.save(cache_data, cache_file)
        print(f"[KIDTracker] Successfully saved cache to: {cache_file} (C_ref: {c_ref:.6f})")

        return ref_features.to(self.device), c_ref

    @torch.no_grad()
    def _compute_metric(self, images: torch.Tensor) -> float:
        """
        Computes exact unbiased MMD^2 with polynomial kernel:
        MMD^2 = 1/(M(M-1)) sum_{i!=j} k(x_i, x_j) + C_ref - 2/(M*N) sum_{i,j} k(x_i, y_j)
        """
        X = self._extract_features(images)  # [M, 2048]
        Y = self.ref_features               # [N, 2048]

        M = X.size(0)
        inv_d = 1.0 / X.size(1)

        # 1. K_XX term (generated vs generated)
        s_xx = X @ X.t()
        k_xx = torch.pow(s_xx * inv_d + 1.0, 3)
        diag_sum_xx = torch.pow((X * X).sum(dim=1) * inv_d + 1.0, 3).sum()
        term_xx = (k_xx.sum() - diag_sum_xx) / (M * (M - 1))

        # 2. K_XY term (generated vs reference)
        s_xy = X @ Y.t()
        k_xy = torch.pow(s_xy * inv_d + 1.0, 3)
        term_xy = k_xy.mean()

        # 3. Combine with cached C_ref
        mmd_squared = term_xx + self.c_ref - 2.0 * term_xy
        return mmd_squared.item()
