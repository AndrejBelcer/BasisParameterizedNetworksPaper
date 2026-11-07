from typing import Any, Type

# Corruptions
from src.modules.corruption_steps import NoiseStep_G, HungarianNoiseStep_G

def get_corruption_step_class(step_name: str) -> Type[Any]:
    """Returns the Corruption Step class."""

    match step_name:
        case "Noise_G": return NoiseStep_G
        case "HungarianNoise_G": return HungarianNoiseStep_G
        case _: raise ValueError(f"Unknown corruption step: {step_name}")
