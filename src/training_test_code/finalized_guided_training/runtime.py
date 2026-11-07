
"""Read GPU_CONFIG once; never reinitialize or override the shared configuration."""
from contextlib import nullcontext

import torch
import torchvision

from src.core.enums import CompileMode
from src.core.globals import GPU_CONFIG


class Runtime:
    def __init__(self):
        self.device = torch.device(GPU_CONFIG.device)
        self.dtype = GPU_CONFIG.dtype
        self.model_dtype = GPU_CONFIG.model_dtype
        self.use_autocast = GPU_CONFIG.use_autocast and self.dtype in (torch.float16, torch.bfloat16)
        self.compile_enabled = GPU_CONFIG.compile_model and GPU_CONFIG.compile_mode_velocity != CompileMode.NONE
        self.compile_mode = GPU_CONFIG.compile_mode_velocity.value
        if self.model_dtype != torch.float32:
            raise ValueError("KID-guided training needs GPU_CONFIG.model_dtype=float32 for its small parameter perturbations. Use an FP32 master-weight profile; configured autocast remains supported.")

    def autocast(self):
        if not self.use_autocast:
            return nullcontext()
        return torch.amp.autocast(device_type=self.device.type, dtype=self.dtype, enabled=True)

    def compile(self, forward, label):
        if not self.compile_enabled:
            return forward
        print(f"Compiling {label}: {self.compile_mode} (first call includes warmup).", flush=True)
        return torch.compile(forward, mode=self.compile_mode, fullgraph=True)

    def signature(self):
        # Used by checkpoints and feature caches; changing precision starts a new run.
        is_cuda = self.device.type == "cuda"
        return dict(torch=str(torch.__version__), torchvision=str(torchvision.__version__),
                    device_type=self.device.type, model_dtype=str(self.model_dtype), dtype=str(self.dtype),
                    autocast=self.use_autocast, compile=self.compile_enabled, compile_mode=self.compile_mode,
                    cuda_matmul_fp32_precision=torch.backends.cuda.matmul.fp32_precision if is_cuda else None,
                    cudnn_fp32_precision=torch.backends.cudnn.fp32_precision if is_cuda else None,
                    cudnn_benchmark=bool(torch.backends.cudnn.benchmark),
                    cudnn_deterministic=bool(torch.backends.cudnn.deterministic))

    def synchronize(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
