from typing import Any, Type


# Static Layers
from src.layers.static_layers.covariance_expansion_layer import CovarianceExpansionLayer
from src.layers.static_layers.attention_layer import StaticAttentionLayer


# Time Layers
from src.layers.time_layers.matmul_layer import MatMulLayer
from src.layers.time_layers.convolution_layer import ConvolutionLayer
from src.layers.time_layers.convolution_upscaling_layer import ConvolutionUpscalingLayer
from src.layers.time_layers.group_norm_layer import GroupNormLayer


# Shortcuts
from src.layers.shortcut_layers.res_att_layer import ResAttLayer


def get_layer_class(layer_name: str) -> Type[Any]:
    """Returns the Layer class based on its string name."""


    match layer_name:
        case "CovarianceExpansionLayer": return CovarianceExpansionLayer
        case "StaticAttentionLayer": return StaticAttentionLayer
        case "ResAttLayer": return ResAttLayer
        case "MatMulLayer": return MatMulLayer
        case "ConvolutionLayer": return ConvolutionLayer
        case "ConvolutionUpscalingLayer": return ConvolutionUpscalingLayer
        case "GroupNormLayer": return GroupNormLayer
        case _: raise ValueError(f"Unknown layer type: {layer_name}")

