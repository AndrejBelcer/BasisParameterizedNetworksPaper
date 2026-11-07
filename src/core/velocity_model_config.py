from dataclasses import dataclass, field, asdict
from typing import Any, List, Dict

from src.core.enums import BlockType


@dataclass
class LayerConfig:
    """Explicit configuration for a single neural network layer."""
    type: str
    params: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Converts configuration to a standard Python dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LayerConfig":
        """Constructs instance from a dictionary representation."""
        return cls(type=data["type"], params=data.get("params", {}).copy())


@dataclass
class BlockConfig:
    """Configuration for a single LayerBlock, including its structural type."""
    block_type: BlockType
    layers: List[LayerConfig]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "block_type": self.block_type.value,
            "layers": [layer.to_dict() for layer in self.layers]
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any], default_type: BlockType) -> "BlockConfig":
        """
        Constructs a BlockConfig. Handles backward compatibility if 'block_type' is missing.
        """
        # If the data is just a list (old checkpoint format), wrap it.
        if isinstance(data, list):
            b_type = default_type
            layers_data = data
        else:
            b_type = BlockType(data.get("block_type", default_type.value))
            layers_data = data.get("layers", [])

        return cls(block_type=b_type, layers=[LayerConfig.from_dict(l) for l in layers_data])


@dataclass
class VelocityModelConfig:
    """The root configuration class defining the entire VelocityModel structure."""
    initializer: BlockConfig
    encoder: List[BlockConfig]
    bottleneck: BlockConfig
    decoder: List[BlockConfig]
    finalizer: BlockConfig
    
    # Global architectural hyper-parameters
    degree: int = 4

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the configuration hierarchy into a plain nested dictionary."""
        return {
            "initializer": self.initializer.to_dict(),
            "encoder": [block.to_dict() for block in self.encoder],
            "bottleneck": self.bottleneck.to_dict(),
            "decoder": [block.to_dict() for block in self.decoder],
            "finalizer": self.finalizer.to_dict(),
            "degree": self.degree
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VelocityModelConfig":
        """
        Parse current or earlier checkpoint dictionaries into block configurations.
        Missing degree metadata defaults to four terms with a warning.
        """
        degree = data.get("degree")
        if degree is None:
            print("⚠️ [Warning] 'degree' not found in checkpoint. Defaulting to 4.")
            degree = 4

        # --- Backward Compatibility & Parsing ---
        
        # Initializer (Default: VANILLA)
        init_data = data.get("initializer") or data.get("initializer_layers", [])
        initializer = BlockConfig.from_dict(init_data, default_type=BlockType.VANILLA)

        # Encoder (Default: RESIDUAL_PROJECTION)
        enc_data = data.get("encoder") or data.get("encoder_layers", [])
        encoder = [BlockConfig.from_dict(b, default_type=BlockType.RESIDUAL_PROJECTION) for b in enc_data]

        # Bottleneck (Default: RESIDUAL_PROJECTION)
        bot_data = data.get("bottleneck") or data.get("bottleneck_layers", [])
        bottleneck = BlockConfig.from_dict(bot_data, default_type=BlockType.RESIDUAL_PROJECTION)

        # Decoder (Default: RESIDUAL_PROJECTION)
        dec_data = data.get("decoder") or data.get("decoder_layers", [])
        decoder = [BlockConfig.from_dict(b, default_type=BlockType.RESIDUAL_PROJECTION) for b in dec_data]

        # Finalizer (Default: VANILLA)
        fin_data = data.get("finalizer") or data.get("finalizer_layers", [])
        finalizer = BlockConfig.from_dict(fin_data, default_type=BlockType.VANILLA)

        return cls(initializer=initializer, encoder=encoder, bottleneck=bottleneck, decoder=decoder, finalizer=finalizer, degree=degree)
