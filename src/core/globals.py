
from dataclasses import dataclass, field
from typing import Optional

import torch

from src.core.enums import HardwareProfile, DeviceType, DataType, CompileMode, CompilerStance, FP32Precision
from src.utilities.print_helper import PrintFields, PrintSummary, PrintYellow, PrintRed


@dataclass
class GPUConfiguration:
    """
    Central configuration for device, numerical precision, AMP,
    and compiler behavior.

    This object applies global PyTorch settings only when Initialize()
    is called. Hardware profiles determine their associated defaults.
    """

    profile: HardwareProfile = HardwareProfile.SAFE_FP32

    # Device
    device_name: DeviceType = DeviceType.CUDA

    # Model parameters / master weights
    model_dtype_name: DataType = DataType.FP32

    # AMP target dtype. Ignored when use_autocast is False.
    autocast_dtype_name: DataType = DataType.BF16
    use_autocast: bool = False

    # Modern PyTorch 2.9+ precision API
    fp32_precision: FP32Precision = FP32Precision.IEEE

    # cuDNN
    enable_cudnn_benchmark: bool = True

    # Compiler
    compile_model: bool = True
    compile_mode_corruption: CompileMode = CompileMode.DEFAULT
    compile_mode_velocity: CompileMode = CompileMode.DEFAULT
    compile_mode_loss: CompileMode = CompileMode.DEFAULT
    compiler_stance: CompilerStance = CompilerStance.DEFAULT

    # Debugging
    debug_mode: bool = False

    # Internal state, populated by Initialize()
    device: torch.device = field(init=False)
    dtype: torch.dtype = field(init=False)
    model_dtype: torch.dtype = field(init=False)
    min_divisor: float = field(init=False)
    min_divisor_tensor: torch.Tensor = field(init=False)

    def __post_init__(self):
        # Safe placeholders. No global backend settings are applied here.
        self.device = torch.device("cpu")
        self.dtype = torch.float32
        self.model_dtype = torch.float32
        self.min_divisor = 1e-18
        self.min_divisor_tensor = torch.tensor(self.min_divisor, dtype=self.dtype, device=self.device)

    def _apply_profile(self):
        """Apply a complete, repeatable set of profile defaults."""

        # Reset profile-controlled fields first. This prevents settings
        # from a previously initialized profile leaking into this one.
        self.model_dtype_name = DataType.FP32
        self.autocast_dtype_name = DataType.BF16
        self.use_autocast = False
        self.fp32_precision = FP32Precision.IEEE

        self.compile_model = True
        self.compile_mode_velocity = CompileMode.DEFAULT
        self.compile_mode_corruption = CompileMode.DEFAULT
        self.compile_mode_loss = CompileMode.DEFAULT

        if self.profile == HardwareProfile.NATIVE_3060_WIN:
            self.autocast_dtype_name = DataType.FP16
            self.use_autocast = True
            self.fp32_precision = FP32Precision.TF32

        elif self.profile == HardwareProfile.NATIVE_5070_WIN:
            self.autocast_dtype_name = DataType.BF16
            self.use_autocast = True
            self.fp32_precision = FP32Precision.TF32

        elif self.profile == HardwareProfile.NATIVE_L40S_LINUX:
            self.autocast_dtype_name = DataType.BF16
            self.use_autocast = True
            self.fp32_precision = FP32Precision.TF32

        elif self.profile == HardwareProfile.PRECISE:
            self.model_dtype_name = DataType.FP32
            self.use_autocast = False
            self.fp32_precision = FP32Precision.IEEE

        elif self.profile == HardwareProfile.MAX_SPEED:
            # Keep FP32 model/master weights for compatibility with
            # KID-guided training. Use FP16 autocast for speed.
            self.use_autocast = False
            self.fp32_precision = FP32Precision.TF32
            self.model_dtype_name = DataType.BF16
            self.dtype = DataType.BF16

        elif self.profile == HardwareProfile.SAFE_FP32:
            self.model_dtype_name = DataType.FP32
            self.use_autocast = False
            self.fp32_precision = FP32Precision.IEEE
            self.compile_model = False

        else:
            raise KeyError(f"Unsupported HardwareProfile: {self.profile}")

    @staticmethod
    def _resolve_dtype(dtype_name: DataType) -> torch.dtype:
        """Convert the project's DataType enum into a torch.dtype."""

        mapping = {
            DataType.FP32: torch.float32,
            DataType.FP16: torch.float16,
            DataType.BF16: torch.bfloat16,
        }

        try:
            return mapping[dtype_name]
        except KeyError as exc:
            raise ValueError(f"Unsupported DataType: {dtype_name}") from exc

    def _setup_device_and_dtype(self):
        """Resolve the selected device and numerical dtypes."""

        # Fail fast rather than silently running a CUDA profile on CPU.
        if self.device_name == DeviceType.CUDA:
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False. Check the NVIDIA driver and PyTorch build.")

            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        # Model weights have their own dtype, independently of AMP.
        self.model_dtype = self._resolve_dtype(self.model_dtype_name)

        if self.use_autocast:
            if self.autocast_dtype_name not in (DataType.FP16, DataType.BF16):
                raise ValueError("Autocast requires FP16 or BF16 as its target dtype. For FP32 computation, disable autocast.")

            self.dtype = self._resolve_dtype(self.autocast_dtype_name)
        else:
            self.dtype = self.model_dtype

        # Validate BF16 support on CUDA.
        if ((self.device.type == "cuda") and (torch.bfloat16 in (self.dtype, self.model_dtype)) and (not torch.cuda.is_bf16_supported())):
            raise ValueError("BF16 was requested, but the CUDA device does not report BF16 support.")

        # Numerical safeguard used by the project.
        self.min_divisor = (1e-6 if self.dtype in (torch.float16, torch.bfloat16) else 1e-18)

        self.min_divisor_tensor = torch.tensor(self.min_divisor, dtype=self.dtype, device=self.device)

    def _apply_precision_settings(self):
        """
        Configure precision using only the modern backend-specific API.

        Do not also call set_float32_matmul_precision() or assign
        allow_tf32. Mixing those old controls with fp32_precision
        is unsupported in current PyTorch.
        """

        if self.device.type != "cuda":
            return

        precision = self.fp32_precision.value

        torch.backends.cuda.matmul.fp32_precision = precision
        torch.backends.cudnn.fp32_precision = precision

    def Initialize(self, profile: Optional[HardwareProfile] = None):
        """
        Apply the selected profile to the global PyTorch configuration.

        Profile-controlled fields are reset on each initialization.
        """

        if profile is not None:
            self.profile = profile

        self._apply_profile()
        self._setup_device_and_dtype()
        self._apply_precision_settings()

        PrintSummary(f"Initializing GPU Config: {self.profile.value.upper()}")

        config_rows = [
            ("Compute Device", self.device),
            ("Model DType (Weights)", self.model_dtype),
            ("Autocast", "Enabled" if self.use_autocast else "Disabled"),
            ("Autocast Target DType", self.dtype if self.use_autocast else "N/A"),
        ]

        if self.device.type == "cuda":
            config_rows.extend([
                ("CUDA MatMul FP32 Precision", torch.backends.cuda.matmul.fp32_precision),
                ("cuDNN FP32 Precision", torch.backends.cudnn.fp32_precision),
            ])

            torch.backends.cudnn.benchmark = (self.enable_cudnn_benchmark)

            config_rows.append(("cuDNN Benchmark", "Enabled" if self.enable_cudnn_benchmark else "Disabled"))

        # Public compiler API in PyTorch 2.14.
        try:
            torch.compiler.set_stance(self.compiler_stance.value)
            config_rows.append(("Compiler Stance", self.compiler_stance.value))
        except Exception as exc:
            PrintYellow(f"Failed to set compiler stance: {exc}")

        compile_enabled = (self.compile_model and self.compile_mode_velocity != CompileMode.NONE)

        if compile_enabled:
            config_rows.extend([
                ("Compiler Mode Velocity", self.compile_mode_velocity.value),
                ("Compiler Mode Corruption", self.compile_mode_corruption.value),
                ("Compiler Mode Loss", self.compile_mode_loss.value),
            ])
        else:
            config_rows.append(("Compiler", "Disabled"))

        PrintFields(config_rows)

        # Explicitly reset this global state on repeated initialization.
        torch.autograd.set_detect_anomaly(self.debug_mode)

        if self.debug_mode:
            PrintRed("Debug Mode: Enabled (Anomaly Detection ON)")


def RunGPUOptions():
    GPU_CONFIG.Initialize()


# Construct without applying global PyTorch settings at import time.
GPU_CONFIG = GPUConfiguration()
