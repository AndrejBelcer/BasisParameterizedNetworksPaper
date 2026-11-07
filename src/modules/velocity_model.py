import os
import torch
import torch.nn as nn
from typing import List, Dict, Any, Tuple

from src.core.globals import GPU_CONFIG
from src.core.enums import CompileMode, BlockType
from src.core.velocity_model_config import VelocityModelConfig, BlockConfig
from src.math.time_interpolation import GetLegendreCoefficients

from src.modules.layer_block import LayerBlock, ProjectionResidualLayerBlock
from src.utilities.print_helper import PrintGreen, PrintYellow, Human_Format

# ===================================================================
# --- Compilation Wrapper ---
# ===================================================================

def CompileModel(model: VelocityModel) -> nn.Module:
    if model.is_compiled == True:
        print("Model already compiled.")
        return model

    if GPU_CONFIG.compile_model and GPU_CONFIG.compile_mode_velocity != CompileMode.NONE:
        print(f"Compiling Velocity Model (Mode: {GPU_CONFIG.compile_mode_velocity.value})...")
        model.to(device=GPU_CONFIG.device, dtype=GPU_CONFIG.model_dtype)
        try:
            model.is_compiled = True
            compiled_model = torch.compile(model, mode=GPU_CONFIG.compile_mode_velocity.value, fullgraph=True)
            PrintGreen("Velocity Model compiled successfully.")
            return compiled_model
        except Exception as e:
            print(f"Model compilation failed: {e}. Returning eager model.")
            return model
    else:
        print("Compiling model skipped.")
        return model

# ===================================================================
# --- Main Model Class ---
# ===================================================================

class VelocityModel(nn.Module):
    """
    The main U-Net-like model composed of LayerBlocks.
    
    Architecture:
    - Initializer (1 Block)
    - Encoder (List of Blocks) -> Generates Skips
    - Bottleneck (1 Block)
    - Decoder (List of Blocks) -> Consumes Skips
    - Finalizer (1 Block)
    
    The model is Time-Dependent: weights evolve based on input 't'.
    """
    def __init__(self, config: VelocityModelConfig):
        super().__init__()
        self.config = config
        self.is_compiled: bool = False
        self.degree = config.degree
        
        # Extract initial shapes
        first_layer_params = config.initializer.layers[0].params
        initial_channels = first_layer_params.get('a', 3)
        initial_spatial = first_layer_params.get('x', 32)
        
        # --- 1. Initializer ---
        self.initializer = self._create_block(config.initializer, initial_channels, initial_spatial)
        c_curr, s_curr = self.initializer.f_out, self.initializer.y_out
        
        # --- 2. Encoder ---
        self.encoder = nn.ModuleList()
        self.skip_shapes = []
        for block_conf in config.encoder:
            block = self._create_block(block_conf, c_curr, s_curr)
            self.encoder.append(block)
            self.skip_shapes.append((block.f_out, block.y_out))
            c_curr, s_curr = block.f_out, block.y_out
            
        # --- 3. Bottleneck ---
        self.bottleneck = self._create_block(config.bottleneck, c_curr, s_curr)
        c_curr, s_curr = self.bottleneck.f_out, self.bottleneck.y_out
        
        # --- 4. Decoder ---
        self.decoder = nn.ModuleList()
        skips_to_use = list(reversed(self.skip_shapes))
        
        if len(config.decoder) != len(skips_to_use):
            raise ValueError(f"Error: Decoder has {len(config.decoder)} blocks but {len(skips_to_use)} skips available.")

        for i, block_conf in enumerate(config.decoder):
            skip_c, skip_s = skips_to_use[i]
            if s_curr != skip_s:
                raise ValueError(f"Shape Mismatch in Decoder Block {i}: Current Spatial {s_curr} != Skip Spatial {skip_s}")
            
            # The input to the decoder block is the current channels + the skip channels
            c_input_for_block = c_curr + skip_c
            
            block = self._create_block(block_conf, c_input_for_block, s_curr)
            self.decoder.append(block)
            c_curr, s_curr = block.f_out, block.y_out
            
        # --- 5. Finalizer ---
        self.finalizer = self._create_block(config.finalizer, c_curr, s_curr)
        
        # Collect all blocks for easy iteration
        self.blocks = [self.initializer] + list(self.encoder) + [self.bottleneck] + list(self.decoder) + [self.finalizer]

        self.count_params()
        #self.print_model_visualization()

    # ===================================================================
    # --- Construction Helpers ---
    # ===================================================================

    def _create_block(self, block_config: BlockConfig, c_in: int, s_in: int) -> LayerBlock:
        """Instantiates the correct LayerBlock subclass based on BlockType."""
        block_mapping = {
            BlockType.VANILLA: LayerBlock,
            BlockType.RESIDUAL_PROJECTION: ProjectionResidualLayerBlock,
        }
        
        BlockClass = block_mapping[block_config.block_type]
        raw_layers = [layer.to_dict() for layer in block_config.layers]
        
        return BlockClass(raw_layers, c_in, s_in, self.degree)

    def _get_block_type(self, block: LayerBlock) -> BlockType:
        """Reverse mapping to serialize the model architecture."""
        if isinstance(block, ProjectionResidualLayerBlock): return BlockType.RESIDUAL_PROJECTION
        return BlockType.VANILLA

    # ===================================================================
    # --- Forward Pass ---
    # ===================================================================

    def forward(self, X: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # 1. Compute Interpolation Coefficients ONCE
        L = GetLegendreCoefficients(t, self.degree)
        
        # 2. Initializer
        X = self.initializer(X, L)
        
        # 3. Encoder
        skips = []
        for block in self.encoder:
            X = block(X, L)
            skips.append(X)
            
        # 4. Bottleneck
        X = self.bottleneck(X, L)
        
        # 5. Decoder
        current_skips = list(reversed(skips))
        for i, block in enumerate(self.decoder):
            # Concatenate Skip Connection
            X = torch.cat([X, current_skips[i]], dim=1)
            X = block(X, L)
            
        # 6. Finalizer
        X = self.finalizer(X, L)
            
        return X

    # ===================================================================
    # --- Visualization & Debugging ---
    # ===================================================================

    def print_model_visualization(self):
        """Prints a clean, vertical ASCII representation of the U-Net architecture."""
        print("\n" + "="*50)
        print(f"{'VELOCITY MODEL ARCHITECTURE':^50}")
        print("="*50)
        
        def print_step(name: str, b_type: BlockType, c_in: int, s_in: int, c_out: int, s_out: int, extra: str = ""):
            print(f"[{name:^14}] : {b_type.value.upper()}")
            print(f"       |          : [{c_in:<4}, {s_in:<3}] -> [{c_out:<4}, {s_out:<3}] {extra}")
            print(f"       |")

        # Initializer
        print_step("Initializer", self._get_block_type(self.initializer), 
                   self.initializer.a_in, self.initializer.x_in, 
                   self.initializer.f_out, self.initializer.y_out)

        # Encoder
        for i, block in enumerate(self.encoder):
            extra = f"---> (Save Skip {i})"
            print_step(f"Encoder {i}", self._get_block_type(block), block.a_in, block.x_in, block.f_out, block.y_out, extra)

        # Bottleneck
        print_step("Bottleneck", self._get_block_type(self.bottleneck), self.bottleneck.a_in, self.bottleneck.x_in, self.bottleneck.f_out, self.bottleneck.y_out)

        # Decoder
        skip_idx = len(self.encoder) - 1
        for i, block in enumerate(self.decoder):
            skip_c, skip_s = self.skip_shapes[skip_idx]
            c_before_concat = block.a_in - skip_c
            
            print(f"[Concat Skip {skip_idx}] : <--- (Consume Skip {skip_idx})")
            print(f"       |          : [{c_before_concat:<4}, {skip_s:<3}] + [{skip_c:<4}, {skip_s:<3}] = [{block.a_in:<4}, {skip_s:<3}]")
            print(f"       |")
            
            print_step(f"Decoder {i}", self._get_block_type(block), block.a_in, block.x_in, block.f_out, block.y_out)
            skip_idx -= 1

        # Finalizer
        print(f"[  Finalizer   ] : {self._get_block_type(self.finalizer).value.upper()}")
        print(f"                 : [{self.finalizer.a_in:<4}, {self.finalizer.x_in:<3}] -> [{self.finalizer.f_out:<4}, {self.finalizer.y_out:<3}]")
        print("="*50 + "\n")

    @torch.no_grad()
    def DebugForwardPass(self, X: torch.Tensor, t: torch.Tensor, save_dir: str):
        """
        Runs a forward pass and triggers the DebugForward of each LayerBlock.
        """
        print(f"\n--- Running Debug Forward Pass (t={t.item():.2f}) ---")
        os.makedirs(save_dir, exist_ok=True)
        self.eval()

        L = GetLegendreCoefficients(t, self.degree)
        block_counter = 0
        
        X = self.initializer.DebugForward(X, L, save_dir, block_counter)
        block_counter += 1
        
        skips = []
        for block in self.encoder:
            X = block.DebugForward(X, L, save_dir, block_counter)
            skips.append(X)
            block_counter += 1
            
        X = self.bottleneck.DebugForward(X, L, save_dir, block_counter)
        block_counter += 1
        
        current_skips = list(reversed(skips))
        for i, block in enumerate(self.decoder):
            X = torch.cat([X, current_skips[i]], dim=1)
            X = block.DebugForward(X, L, save_dir, block_counter)
            block_counter += 1
            
        X = self.finalizer.DebugForward(X, L, save_dir, block_counter)
        
        self.train()
        print("--- Debug Pass Complete ---")

    # ===================================================================
    # --- Management ---
    # ===================================================================

    def AfterStep(self):
        """Called after optimizer step."""
        for block in self.blocks:
            block.AfterStep()

    @torch.no_grad()
    def PrintModelInfo(self, detailed_info: bool = False):
        print("\n=== Model Architecture Info ===")
        for block in self.blocks:
            block.PrintLayerInfo(detailed_info)
        print("===============================\n")

    def get_config(self) -> Dict[str, Any]:
        """
        Reconstructs the exact architecture configuration from the instantiated layers.
        """
        def serialize_block(block: LayerBlock) -> Dict[str, Any]:
            return {
                "block_type": self._get_block_type(block).value,
                "layers": block.get_config()
            }

        return {
            "initializer": serialize_block(self.initializer),
            "encoder": [serialize_block(b) for b in self.encoder],
            "bottleneck": serialize_block(self.bottleneck),
            "decoder": [serialize_block(b) for b in self.decoder],
            "finalizer": serialize_block(self.finalizer),
            "degree": self.degree
        }

    def save(self, filepath: str):
        print(f"Saving model to {filepath}...")
        
        # Get and Serialize config dataclass directly into a flat dictionary
        config_serialized = self.get_config()

        first_param = next(self.parameters())
        original_dtype = first_param.dtype
        if original_dtype != torch.float32:
            self.to(torch.float32)
            
        checkpoint = {
            'arch_config': config_serialized,
            'state_dict': self.state_dict()
        }
        torch.save(checkpoint, filepath)
        
        if original_dtype != torch.float32:
            self.to(original_dtype)
            
        print("Model saved with explicitly resolved config.")

    def compile_model(self) -> VelocityModel:
        return CompileModel(self)

    @classmethod
    def load(cls, filepath: str, compile: bool = True) -> VelocityModel:
        if not os.path.exists(filepath):
            return None
            
        print(f"Loading model from {filepath}...")
        checkpoint = torch.load(filepath, map_location=GPU_CONFIG.device, weights_only=False)
        
        # Handle state keys (supports UI and cluster formats)
        state_key = 'state_dict' if 'state_dict' in checkpoint else 'model_state'
        
        # Reconstruct the VelocityModelConfig safely
        raw_config = checkpoint['arch_config']
        config = VelocityModelConfig.from_dict(raw_config)
        
        model = cls(config)
        model.load_state_dict(checkpoint[state_key])
        
        model.to(device=GPU_CONFIG.device, dtype=GPU_CONFIG.model_dtype)
        if(compile):
            model = CompileModel(model)
        return model

    @classmethod
    def create(cls, config: VelocityModelConfig, compile: bool = True) -> VelocityModel:
        """Creates a new model instance."""
        model = cls(config)
        model.to(device=GPU_CONFIG.device, dtype=GPU_CONFIG.model_dtype)
        if(compile):
            model = CompileModel(model)
        return model

    def count_params(self, human_format: bool = True) -> int:
        """Returns the total number of trainable parameters."""
        number_of_parameters = sum(p.numel() for p in self.parameters() if p.requires_grad)
        if(human_format):
            print(f"The model has {Human_Format(number_of_parameters)} parameters")
        else:
            print(f"The model has {number_of_parameters} parameters")
        return number_of_parameters

    def __repr__(self) -> str:
        """
        Bypasses default nn.Module.__repr__ to prevent list.pop tracing 
        errors during torch.compile + torch.vmap.
        """
        return "VelocityModel"
