import torch
import torch.nn as nn
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from typing import Optional, Dict

class EMAHelper:
    """
    Maintains a shadow copy of the model parameters using Exponential Moving Average.
    Uses PyTorch's official optimized AveragedModel under the hood.
    """
    def __init__(self, model: nn.Module, decay: float = 0.999, device: Optional[str] = None):
        self.decay = decay
        
        # Determine storage device
        if device is None:
            self.device = next(model.parameters()).device
        else:
            self.device = torch.device(device)
            
        # Create PyTorch's official EMA model
        # This automatically handles the shadow copy creation and uses optimized _foreach operations
        self.ema_model = AveragedModel(model, device=self.device, multi_avg_fn=get_ema_multi_avg_fn(decay))
        
        # Backup storage for when we swap weights into the model
        self.backup: Dict[str, torch.Tensor] = {}

    def update(self, model: nn.Module):
        """
        Updates the shadow parameters using highly optimized fused operations.
        """
        # PyTorch handles the no_grad() and lerp logic internally and safely
        self.ema_model.update_parameters(model)

    def save(self, filepath: str):
        """
        Save the averaged VelocityModel using its checkpoint format.
        """
        if hasattr(self.ema_model.module, 'save'):
            self.ema_model.module.save(filepath)
        else:
            torch.save(self.ema_model.module.state_dict(), filepath)

    @torch.no_grad()
    def copy_to(self, model: nn.Module):
        """
        Saves the model's current parameters to backup and loads the EMA weights.
        Call this BEFORE inference/saving.
        """
        # 1. Save current weights to backup
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.detach().clone()
                
        # 2. Load EMA weights into the model
        # AveragedModel stores the averaged weights in its '.module' attribute
        for ema_param, model_param in zip(self.ema_model.module.parameters(), model.parameters()):
            if model_param.requires_grad:
                model_param.data.copy_(ema_param.data.to(model_param.device))

    @torch.no_grad()
    def restore(self, model: nn.Module):
        """
        Restores the model's original parameters from backup.
        Call this AFTER inference/saving to resume training.
        """
        for name, param in model.named_parameters():
            if name in self.backup:
                param.data.copy_(self.backup[name])
        
        self.backup = {} # Clear memory
