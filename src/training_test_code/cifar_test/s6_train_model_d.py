import os
import time
import math

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, TensorDataset

# --- Project Imports ---
from src.core.globals import GPU_CONFIG
from src.modules.velocity_model import VelocityModel
from src.modules.ema import EMAHelper
from src.training_helpers.time_sampler import TimeSampler, TimeSamplingType
from src.modules.corruption_module import CorruptionModule, CreateCorruptionModule
from src.modules.velocity_training_wrapper import VelocityTrainingWrapper
from src.utilities.tensor_utils import normalize_tensor
from src.training_helpers.validation_tracker import KIDTracker, BaseValidationTracker


# ===================================================================
# --- Data Loading ---
# ===================================================================

def _load_training_data(folder_path: str, image_size: int, apply_horizontal_flip: bool, batch_size: int, num_workers: int) -> DataLoader:
    """Loads images into RAM, applies normalization and optional flipping, returns a DataLoader."""
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Data folder not found: {folder_path}")

    print(f"Scanning {os.path.basename(folder_path)} for training images...")
    images_list = []
    
    for root, _, files in os.walk(folder_path):
        for fname in files:
            if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                path = os.path.join(root, fname)
                try:
                    with Image.open(path) as img:
                        img = img.convert('RGB')
                        if img.size != (image_size, image_size):
                            img = img.resize((image_size, image_size), Image.Resampling.LANCZOS)
                        
                        arr = np.array(img).astype(np.float32) / 255.0
                        tensor = normalize_tensor(torch.from_numpy(arr).permute(2, 0, 1))
                        images_list.append(tensor)
                except Exception as e:
                    print(f"  [Warning] Failed to load {fname}: {e}")

    if not images_list:
        raise ValueError(f"No images found in {folder_path}")

    train_tensor = torch.stack(images_list)
    
    if apply_horizontal_flip:
        print(f"  [Augmentation] Applying horizontal flip...")
        flipped_tensor = torch.flip(train_tensor, dims=[3])
        train_tensor = torch.cat([train_tensor, flipped_tensor], dim=0)
        
    print(f"  Loaded Training Tensor: {train_tensor.shape}. Range: [{train_tensor.min():.3f}, {train_tensor.max():.3f}]")
    
    dataset = TensorDataset(train_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True, pin_memory=True)


# ===================================================================
# --- Training Logic ---
# ===================================================================

def get_exponential_schedule_with_warmup(optimizer, warmup_steps: int, total_steps: int, lr_start: float, lr_end: float):
    """Linear warmup followed by exponential (geometric) decay to lr_end."""
    decay_ratio = lr_end / lr_start

    def lr_lambda(current_step: int):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        progress = min(max(progress, 0.0), 1.0)
        
        # Exponential decay: (lr_end / lr_start) ^ progress
        return decay_ratio ** progress
        
    return LambdaLR(optimizer, lr_lambda)

def get_log_cosine_schedule_with_warmup(optimizer, warmup_steps: int, total_steps: int, lr_start: float, lr_end: float):
    """Smooth cosine transition between lr_start and lr_end in log-space."""
    decay_ratio = lr_end / lr_start

    def lr_lambda(current_step: int):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))

        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        progress = min(max(progress, 0.0), 1.0)

        # Smooth S-curve in exponent: 0.0 at progress=0 -> 1.0 at progress=1
        cosine_power = 0.5 * (1.0 - math.cos(math.pi * progress))
        
        # decay_ratio^0.0 = 1.0 (lr_start), decay_ratio^1.0 = lr_end / lr_start
        return decay_ratio ** cosine_power

    return LambdaLR(optimizer, lr_lambda)

def TrainVelocityModel(
        model_path: str,
        attempt_load_training_state: bool = True,
        batch_size_per_time: int = 4,  # Z (number of samples per time step)
        num_times: int = 64,          # T (number of unique time steps evaluated)
        grad_accum_steps: int = 1,
        use_EMA: bool = True,
        ema_decay: float = 0.9999,     # Multiplicative factor for EMA
        learning_rate_start: float = 2e-4,
        learning_rate_end: float = 5e-6,
        warmup_steps: int = 2000,
        weight_decay: float = 1e-2,
        total_epochs: int = 500,
        image_training_folder: str = "data/cifar_official/fid_reference_flat",
        apply_horizontal_flip: bool = True,
        image_size: int = 32,
        channels: int = 3,
        num_workers: int = 8,
        precompute_size_multiplier: int = 4,
        noise_method: str = "HungarianNoise_G",
        ):
    """Train velocity MSE using Noise_G or minibatch-assigned HungarianNoise_G."""
    
    device = GPU_CONFIG.device
    trainer_state_path = model_path.replace(".pth", "_trainer.pth")
    
    # 1. Setup dynamic time subsampling configuration
    Z = batch_size_per_time
    T = num_times
    
    effective_batch_size = Z * T
    precompute_size = precompute_size_multiplier * effective_batch_size
    
    # Load a larger batch for noise assignment before splitting it across times.
    train_loader = _load_training_data(
        folder_path=image_training_folder, 
        image_size=image_size, 
        apply_horizontal_flip=apply_horizontal_flip, 
        batch_size=precompute_size, 
        num_workers=num_workers
    )

    # len(train_loader) is the number of super-batches. 
    # Multiply by precompute_size_multiplier to get actual mini-batch steps.
    steps_per_epoch = (len(train_loader) * precompute_size_multiplier) // grad_accum_steps
    total_steps = total_epochs * steps_per_epoch
    
    # 3. Model & Wrapper Initialization
    print(f"\nLoading VelocityModel from {model_path}...")

    model = VelocityModel.load(model_path, compile=False).to(device)
    model.train()
    
    wrapper = VelocityTrainingWrapper(model)

    # 4. Training Components
    time_sampler: TimeSampler = TimeSampler(lowest_t=0.085)
    corruption_module: CorruptionModule = CreateCorruptionModule([{'name': noise_method, 'scale': 1.0}])
    
    # Optimize all trainable Legendre coefficients.
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=learning_rate_start, weight_decay=weight_decay, eps = 1e-5)
    
    ema = EMAHelper(model, decay=ema_decay) if use_EMA else None
    
    scaler_enabled = (GPU_CONFIG.dtype == torch.float16) and GPU_CONFIG.use_autocast
    scaler = GradScaler(device=device.type, enabled=scaler_enabled)

    # 5. Resume State (Optimizer & EMA buffers only)
    if attempt_load_training_state and os.path.exists(trainer_state_path):
        try:
            print("Restoring optimizer state...")
            state = torch.load(trainer_state_path, map_location=device)
            optimizer.load_state_dict(state['optimizer_state_dict'])
            
            # Keep the requested learning rate and weight decay when resuming.
            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate_start
                param_group['initial_lr'] = learning_rate_start
                param_group['weight_decay'] = weight_decay
            
            if use_EMA and ema is not None and 'ema_state_dict' in state:
                ema.ema_model.load_state_dict(state['ema_state_dict'])
        except Exception as e:
            print(f"  [Warning] Failed to restore optimizer state: {e}")

    # 6. Initialize Scheduler AFTER loading optimizer state    
    lr_scheduler = get_log_cosine_schedule_with_warmup(optimizer, warmup_steps=warmup_steps, total_steps=total_steps, lr_start=learning_rate_start, lr_end=learning_rate_end)

    # 7. Validation Tracker Initialization (KID)
    val_time_sampler = TimeSampler(method=TimeSamplingType.STRATIFIED_UNIFORM, lowest_t=0.13)
    val_tracker: BaseValidationTracker = KIDTracker(reference_folder=image_training_folder, batch_size=512, channels=channels, img_size=image_size, steps=30, time_sampler=val_time_sampler, device=device, seed=13, chunk_size=128)
    
    print(f"\n--- Running Initial Validation (Epoch 0) ---")
    base_metric, ema_metric, base_improved, ema_improved = val_tracker.evaluate(base_model=wrapper.model, ema_model=ema.ema_model if (use_EMA and ema is not None) else None)
    print(f"Epoch 0 | Base {val_tracker.metric_name}: {base_metric:.5f} | EMA {val_tracker.metric_name}: {ema_metric:.5f}")

    # 8. Training Loop
    print(f"\n--- Starting Training ---")
    print(f"Z (Batch per time): {Z} | T (Num times): {T} | Grad Accum: {grad_accum_steps}")
    print(f"Effective Mini-Batch Size: {effective_batch_size}")
    print(f"Precompute Super-Batch Size: {precompute_size} (Multiplier: {precompute_size_multiplier})")
    print(f"Total Epochs: {total_epochs} | Total Steps: {total_steps}")
    
    for epoch in range(1, total_epochs + 1):
        epoch_loss = 0.0
        start_t = time.time()
        accum_step = 0
        mini_batches_processed = 0
        
        optimizer.zero_grad(set_to_none=True)
        
        for step, (super_images,) in enumerate(train_loader):
            # super_images shape is [precompute_size, C, H, W]
            super_images = super_images.to(device, non_blocking=True)
            
            # --- 1. Corrupt the entire super-batch at once (Hungarian Matching) ---
            with torch.no_grad():
                with autocast(device_type=device.type, dtype=GPU_CONFIG.dtype, enabled=GPU_CONFIG.use_autocast):
                    super_N_flat = corruption_module(super_images)
            
            # --- 2. Inner Loop: Iterate through the super-batch in chunks of effective_batch_size ---
            for i in range(0, precompute_size, effective_batch_size):
                
                # Slice the mini-batch
                G_mini = super_images[i : i + effective_batch_size]
                N_mini = super_N_flat[i : i + effective_batch_size]
                
                # Reshape Tensors for VMAP: [T, Z, C, H, W]
                G_tz = G_mini.view(T, Z, channels, image_size, image_size)
                N_tz = N_mini.view(T, Z, channels, image_size, image_size)
                
                # --- 3. Sample T times (Shape: [T, 1]) ---
                t_sampled = time_sampler.sample(T) # [T]
                t_t = t_sampled.view(T, 1)         # [T, 1]
                
                # --- 4. Flow Matching Math: X(t) = G + t(N - G) ---
                t_spatial = t_t.view(T, 1, 1, 1, 1)
                X_tz = G_tz + t_spatial * (N_tz - G_tz)
                target_V_tz = N_tz - G_tz
            
                # --- Forward & Loss ---
                with autocast(device_type=device.type, dtype=GPU_CONFIG.dtype, enabled=GPU_CONFIG.use_autocast):
                    
                    # 1. Predict Velocity
                    pred_V_tz = wrapper.predict_velocity(X_tz, t_t)
                    
                    # 2. RGB Velocity Loss (Per sample shape: [T, Z])
                    loss_fm_elementwise = F.mse_loss(pred_V_tz, target_V_tz, reduction='none')
                    loss_rgb_per_sample = loss_fm_elementwise.mean(dim=(2, 3, 4)) # [T, Z]
                    loss_rgb = loss_rgb_per_sample.mean()

                    total_loss = loss_rgb
                    scaled_loss = total_loss / grad_accum_steps
                    

                # --- Backward ---
                scaler.scale(scaled_loss).backward()
                epoch_loss += total_loss.item()
                accum_step += 1
                mini_batches_processed += 1

                # --- Optimizer Step ---
                if accum_step == grad_accum_steps:
                    scaler.unscale_(optimizer)
                    
                    # NaN/Inf Check
                    has_overflow = False
                    for p in model.parameters():
                        if p.grad is not None and not torch.isfinite(p.grad).all():
                            has_overflow = True
                            break
                            
                    if has_overflow:
                        if not scaler.is_enabled():
                            print("  [WARNING] NaN/Inf in gradients! Sanitizing to 0.0...")
                            for p in model.parameters():
                                if p.grad is not None:
                                    torch.nan_to_num(p.grad, nan=0.0, posinf=0.0, neginf=0.0, out=p.grad)
                            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    
                    lr_scheduler.step()
                    if use_EMA and ema is not None:
                        ema.update(model)
                        
                    accum_step = 0

        # --- Epoch Logging & Saving ---
        avg_loss = epoch_loss / max(1, mini_batches_processed)
        
        duration = time.time() - start_t
        current_lr = lr_scheduler.get_last_lr()[0]
        
        print(f"Epoch {epoch}/{total_epochs} | LR {current_lr:.3e} | Velocity MSE {avg_loss:.5f} | Time {duration:.1f}s")

        # --- Validation & Conditional Saving ---
        base_metric, ema_metric, base_improved, ema_improved = val_tracker.evaluate(base_model=wrapper.model, ema_model=ema.ema_model if (use_EMA and ema is not None) else None)
        
        print(f"  -> Validation {val_tracker.metric_name} | Base: {base_metric:.6f} {'(NEW BEST)' if base_improved else ''} | EMA: {ema_metric:.6f} {'(NEW BEST)' if ema_improved else ''}")

        # Save Base Model if improved
        if base_improved:
            wrapper.model.save(model_path)
            # Save optimizer state alongside the base model
            trainer_state = {'optimizer_state_dict': optimizer.state_dict()}
            if use_EMA and ema is not None:
                trainer_state['ema_state_dict'] = ema.ema_model.state_dict()
            torch.save(trainer_state, trainer_state_path)
            print(f"  -> [Saved] Base model & Trainer state to {model_path}")
            
        # Save EMA Model if improved
        if ema_improved and use_EMA and ema is not None:
            ema_path = model_path.replace(".pth", "_ema.pth")
            ema.save(ema_path)
            print(f"  -> [Saved] EMA model to {ema_path}")


def TrainTheModel():
    TrainVelocityModel(
        model_path=r"saved_models/D.pth",
        image_training_folder=r"data/cifar_official/fid_reference_flat",
        attempt_load_training_state=False,
        use_EMA=False,
        batch_size_per_time=2,
        precompute_size_multiplier=32,
        num_times=16,
        grad_accum_steps=4,
        weight_decay=3e-4,
        total_epochs=300,
        learning_rate_start=6e-4,
        learning_rate_end=1e-5,
        warmup_steps=200,
    )


if __name__ == "__main__":
    TrainTheModel()
