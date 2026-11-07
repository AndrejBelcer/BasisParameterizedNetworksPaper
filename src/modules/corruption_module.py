import torch
import torch.nn as nn
from typing import List, Dict, Any

from src.core.globals import GPU_CONFIG
from src.core.enums import CompileMode
from src.core.mappings_corruptions import get_corruption_step_class
from src.utilities.tensor_utils import CONVERSION_RATIO
from src.utilities.print_helper import PrintGreen, PrintYellow

# ===================================================================
# --- Main Corruption Module ---
# ===================================================================

class CorruptionModule(nn.Module):
    """
    Unified Corruption Module.
    Constructs the target state 'N' by sequentially applying corruption steps.
    
    The final target for Flow Matching is:
    X(t) = (1 - t) * G + t * N
    """
    def __init__(self, steps_config: List[Dict[str, Any]], clamp: bool = False):
        super().__init__()
        self.steps = nn.ModuleList()
        self.clamp = clamp
        self._uses_randomness = False
        
        # Initialize steps using the Mappings factory
        for step_conf in steps_config:
            name = step_conf.get('name')
            scale = step_conf.get('scale', 1.0)
            
            try:
                StepClass = get_corruption_step_class(name)
                step_instance = StepClass(scale=scale)
                self.steps.append(step_instance)
                
                # Track if any step prevents 'fullgraph' compilation
                if step_instance.uses_randomness:
                    self._uses_randomness = True
            except Exception as e:
                raise ValueError(f"Could not construct corruption step '{name}': {e}") from e

    @property
    def uses_randomness(self) -> bool:
        """Returns True if any step uses RNG, necessitating a graph break."""
        return self._uses_randomness

    def forward(self, G: torch.Tensor) -> torch.Tensor:
        # Start with the ground truth
        N = G.clone()
        
        # Apply each step as a linear interpolation towards the corrupted state
        for step in self.steps:
            # N_new = (1 - scale) * N_old + scale * step(N_old)
            N = torch.lerp(N, step(N), step.scale)
        
        # Ensure the final target stays within the project's valid float range if clamp is on
        if (self.clamp):
            N = torch.clamp(N, -CONVERSION_RATIO, CONVERSION_RATIO)
        
        return N

# ===================================================================
# --- Factory & Compilation ---
# ===================================================================

def CreateCorruptionModule(config: List[Dict[str, Any]]) -> nn.Module:
    """
    Factory to instantiate and compile the CorruptionModule.
    Automatically optimizes the compilation strategy based on the steps used.
    """
    module = CorruptionModule(config or [])
    module = module.to(GPU_CONFIG.device, dtype=GPU_CONFIG.model_dtype)

    # Skip compilation if disabled in global config
    if not GPU_CONFIG.compile_model or GPU_CONFIG.compile_mode_corruption == CompileMode.NONE:
        return module

    # If the module is deterministic, we use 'fullgraph=True' for maximum speed.
    # If it uses RNG, we allow graph breaks.
    fullgraph_opt = not module.uses_randomness
    
    try:
        print(f"Compiling Corruption [Mode: {GPU_CONFIG.compile_mode_corruption.value}, Fullgraph: {fullgraph_opt}]...")
        return torch.compile(module, mode=GPU_CONFIG.compile_mode_corruption.value, fullgraph=fullgraph_opt)
    except Exception as e:
        PrintYellow(f"Corruption compilation failed: {e}. Returning eager module.")
        return module

