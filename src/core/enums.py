from enum import Enum

# =========================================================================
# --- Hardware and Numerical Precision ---
# =========================================================================

class HardwareProfile(str, Enum):
    """Pre-configured hardware setups for different machines."""
    NATIVE_3060_WIN = "native_3060_win"
    NATIVE_5070_WIN = "native_5070_win"
    NATIVE_L40S_LINUX = "native_l40s_linux"
    PRECISE = "precise"
    MAX_SPEED = "max_speed"
    SAFE_FP32 = "safe_fp32"


class DeviceType(str, Enum):
    CPU = "cpu"
    CUDA = "cuda"


class DataType(str, Enum):
    """Tensor dtype for model weights or autocast."""
    FP32 = "float32"
    FP16 = "float16"
    BF16 = "bfloat16"


class FP32Precision(str, Enum):
    """Backend precision for CUDA FP32 matrix multiplication and cuDNN."""
    IEEE = "ieee"
    TF32 = "tf32"


class CompileMode(str, Enum):
    """Mode for torch.compile; NONE is the project's skip-compilation option."""
    DEFAULT = "default"
    REDUCE_OVERHEAD = "reduce-overhead" # CUDA graphs for lower overhead when supported
    MAX_AUTOTUNE = "max-autotune"       # More tuning at compile time
    MAX_AUTOTUNE_NO_CUDAGRAPHS = "max-autotune-no-cudagraphs"
    NONE = "none"


class CompilerStance(str, Enum):
    """Runtime behavior for torch.compile through torch.compiler.set_stance()."""
    DEFAULT = "default"
    FORCE_EAGER = "force_eager"
    EAGER_ON_RECOMPILE = "eager_on_recompile"
    FAIL_ON_RECOMPILE = "fail_on_recompile"
    EAGER_THEN_COMPILE = "eager_then_compile"
    AOT_EAGER_THEN_COMPILE = "aot_eager_then_compile"


# =========================================================================
# --- Math & Model ---
# =========================================================================

class ActivationType(str, Enum):
    XEXP = "xexp"      # x * exp(-x^4/4)
    XEXP2 = "xexp2"
    X3 = "x3"          # x^3 / (1 + |x|^3)
    XN = "xn"          # 2 * x^3 / (1 + |x|^3)
    TANH = "tanh"
    RELU = "relu"
    SIGMOID = "sigmoid"
    IDENTITY = "identity"
    GAUSS = "gauss"
    SILU = "silu"


class BlockType(str, Enum):
    """Defines the structural type of a LayerBlock."""
    VANILLA = "vanilla"
    RESIDUAL_PROJECTION = "residual_projection"

