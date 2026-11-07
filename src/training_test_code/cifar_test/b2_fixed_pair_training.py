import os
import time
import math

import torch
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

# --- Project Imports ---
from src.core.globals import GPU_CONFIG
from src.modules.velocity_model import VelocityModel
from src.modules.ema import EMAHelper
from src.training_helpers.time_sampler import TimeSampler, TimeSamplingType
from src.modules.velocity_training_wrapper import VelocityTrainingWrapper
from src.training_helpers.validation_tracker import KIDTracker, BaseValidationTracker

# ===================================================================
# --- Data Loading ---
# ===================================================================

def _load_and_mirror_seed_data_to_vram(seeds: list[int], device: torch.device) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
    """Loads datasets for all seeds directly into VRAM and mirrors them."""
    print("\nLoading and mirroring datasets directly to VRAM...")
    vram_datasets = {}
    
    for seed in seeds:
        cache_path = f"cache/training_pairs/cifar_fixed_pairs_{seed}.pt"
        if not os.path.exists(cache_path):
            raise FileNotFoundError(f"Cached pairs not found: {cache_path}. Create them with a1_pair_images.CreateInitialPairs and PrepareDataset first.")
        
        # Load directly to the target device
        data = torch.load(cache_path, map_location=device, weights_only=True)
        G_base = data["G"]
        N_base = data["N"]
        
        # Mirror both images and noises to preserve optimal assignment
        G_flipped = torch.flip(G_base, dims=[3])
        N_flipped = torch.flip(N_base, dims=[3])
        
        # Concatenate to double the dataset size
        G_full = torch.cat([G_base, G_flipped], dim=0)
        N_full = torch.cat([N_base, N_flipped], dim=0)
        
        vram_datasets[seed] = (G_full, N_full)
        print(f"  -> Seed {seed}: Loaded and mirrored to {len(G_full)} pairs.")
        
    return vram_datasets

# ===================================================================
# --- Training Logic ---
# ===================================================================

def get_log_cosine_schedule_with_warmup(optimizer, warmup_steps: int, total_steps: int, lr_start: float, lr_end: float):
    decay_ratio = lr_end / lr_start
    def lr_lambda(current_step: int):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        progress = min(max(progress, 0.0), 1.0)
        cosine_power = 0.5 * (1.0 - math.cos(math.pi * progress))
        return decay_ratio ** cosine_power
    return LambdaLR(optimizer, lr_lambda)

def TrainVelocityModelFixedPairs(
        model_path: str,
        seeds: list[int],
        attempt_load_training_state: bool = True,
        batch_size_per_time: int = 4,  # Z
        num_times: int = 64,           # T
        grad_accum_steps: int = 1,
        use_EMA: bool = True,
        ema_decay: float = 0.9999,
        learning_rate_start: float = 2e-4,
        learning_rate_end: float = 5e-6,
        warmup_steps: int = 2000,
        weight_decay: float = 1e-2,
        total_epochs: int = 500,
        image_training_folder: str = "data/cifar_official/fid_reference_flat", # Still needed for KIDTracker
        image_size: int = 32,
        channels: int = 3,
        ):
    
    device = GPU_CONFIG.device
    trainer_state_path = model_path.replace(".pth", "_trainer.pth")
    
    # 1. Setup dynamic time subsampling configuration
    Z = batch_size_per_time
    T = num_times
    effective_batch_size = Z * T
    
    # 2. Data Loading (Directly to VRAM)
    vram_datasets = _load_and_mirror_seed_data_to_vram(seeds, device)
    
    # Assuming all seeds have the same number of samples
    samples_per_epoch = len(vram_datasets[seeds[0]][0])
    steps_per_epoch = (samples_per_epoch // effective_batch_size) // grad_accum_steps
    total_steps = total_epochs * steps_per_epoch
    
    # 3. Model & Wrapper Initialization
    print(f"\nLoading VelocityModel from {model_path}...")
    model = VelocityModel.load(model_path, compile=False).to(device)
    model.train()
    wrapper = VelocityTrainingWrapper(model)
    
    # 4. Training Components
    time_sampler = TimeSampler(lowest_t=1e-3)
    
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=learning_rate_start, weight_decay=weight_decay, eps=1e-5)
    ema = EMAHelper(model, decay=ema_decay) if use_EMA else None
    
    scaler_enabled = (GPU_CONFIG.dtype == torch.float16) and GPU_CONFIG.use_autocast
    scaler = GradScaler(device=device.type, enabled=scaler_enabled)

    # 5. Resume State
    if attempt_load_training_state and os.path.exists(trainer_state_path):
        try:
            print("Restoring optimizer state...")
            state = torch.load(trainer_state_path, map_location=device, weights_only=True)
            optimizer.load_state_dict(state['optimizer_state_dict'])
            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate_start
                param_group['initial_lr'] = learning_rate_start
                param_group['weight_decay'] = weight_decay
            if use_EMA and ema is not None and 'ema_state_dict' in state:
                ema.ema_model.load_state_dict(state['ema_state_dict'])
        except Exception as e:
            print(f"  [Warning] Failed to restore optimizer state: {e}")

    lr_scheduler = get_log_cosine_schedule_with_warmup(optimizer, warmup_steps=warmup_steps, total_steps=total_steps, lr_start=learning_rate_start, lr_end=learning_rate_end)

    # 6. Validation Tracker Initialization
    val_time_sampler = TimeSampler(method=TimeSamplingType.STRATIFIED_UNIFORM, lowest_t=0.09)
    val_tracker: BaseValidationTracker = KIDTracker(reference_folder=image_training_folder, batch_size=2048, channels=channels, img_size=image_size, time_sampler=val_time_sampler, device=device, seed=13)
    
    print(f"\n--- Running Initial Validation (Epoch 0) ---")
    base_metric, ema_metric, base_improved, ema_improved = val_tracker.evaluate(base_model=wrapper.model, ema_model=ema.ema_model if (use_EMA and ema is not None) else None)
    print(f"Epoch 0 | Base {val_tracker.metric_name}: {base_metric:.5f} | EMA {val_tracker.metric_name}: {ema_metric:.5f}")

    # 7. Training Loop
    print(f"\n--- Starting Training ---")
    print(f"Z (Batch per time): {Z} | T (Num times): {T} | Grad Accum: {grad_accum_steps}")
    print(f"Effective Mini-Batch Size: {effective_batch_size}")
    print(f"Total Epochs: {total_epochs} | Total Steps: {total_steps}")
    
    for epoch in range(1, total_epochs + 1):
        epoch_loss = 0.0
        start_t = time.time()
        accum_step = 0
        mini_batches_processed = 0
        
        # Cycle through seeds round-robin
        current_seed = seeds[(epoch - 1) % len(seeds)]
        G_all, N_all = vram_datasets[current_seed]
        
        # Generate random permutation for the custom batching loop
        perm = torch.randperm(len(G_all), device=device)
        num_batches = len(G_all) // effective_batch_size
        
        optimizer.zero_grad(set_to_none=True)
        
        for batch_idx in range(num_batches):
            # Slice the VRAM tensors directly
            start_idx = batch_idx * effective_batch_size
            end_idx = start_idx + effective_batch_size
            idx = perm[start_idx:end_idx]
            
            G_mini = G_all[idx]
            N_mini = N_all[idx]
            
            # Reshape Tensors for VMAP: [T, Z, C, H, W]
            G_tz = G_mini.view(T, Z, channels, image_size, image_size)
            N_tz = N_mini.view(T, Z, channels, image_size, image_size)
            
            # Sample T times (Shape: [T, 1])
            t_sampled = time_sampler.sample(T)
            t_t = t_sampled.view(T, 1)
            
            # Flow Matching Math: X(t) = G + t(N - G)
            t_spatial = t_t.view(T, 1, 1, 1, 1)
            X_tz = G_tz + t_spatial * (N_tz - G_tz)
            target_V_tz = N_tz - G_tz
        
            # Forward & Loss
            with autocast(device_type=device.type, dtype=GPU_CONFIG.dtype, enabled=GPU_CONFIG.use_autocast):
                pred_V_tz = wrapper.predict_velocity(X_tz, t_t)
                
                loss_fm_elementwise = F.mse_loss(pred_V_tz, target_V_tz, reduction='none')
                loss_rgb_per_sample = loss_fm_elementwise.mean(dim=(2, 3, 4)) # [T, Z]
                
                loss_rgb = loss_rgb_per_sample.mean()

                total_loss = loss_rgb
                scaled_loss = total_loss / grad_accum_steps
                
            # Backward
            scaler.scale(scaled_loss).backward()
            epoch_loss += total_loss.item()
            accum_step += 1
            mini_batches_processed += 1

            # Optimizer Step
            if accum_step == grad_accum_steps:
                scaler.unscale_(optimizer)
                
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

        # Epoch Logging & Saving
        avg_loss = epoch_loss / max(1, mini_batches_processed)
        
        duration = time.time() - start_t
        current_lr = lr_scheduler.get_last_lr()[0]
        
        print(f"Epoch {epoch}/{total_epochs} (Seed: {current_seed}) | LR {current_lr:.3e} | Total Loss {avg_loss:.5f} | Time {duration:.1f}s")

        base_metric, ema_metric, base_improved, ema_improved = val_tracker.evaluate(base_model=wrapper.model, ema_model=ema.ema_model if (use_EMA and ema is not None) else None)
        print(f"  -> Validation {val_tracker.metric_name} | Base: {base_metric:.6f} {'(NEW BEST)' if base_improved else ''} | EMA: {ema_metric:.6f} {'(NEW BEST)' if ema_improved else ''}")

        if base_improved:
            wrapper.model.save(model_path)
            trainer_state = {'optimizer_state_dict': optimizer.state_dict()}
            if use_EMA and ema is not None:
                trainer_state['ema_state_dict'] = ema.ema_model.state_dict()
            torch.save(trainer_state, trainer_state_path)
            print(f"  -> [Saved] Base model & Trainer state to {model_path}")
            
        if ema_improved and use_EMA and ema is not None:
            ema_path = model_path.replace(".pth", "_ema.pth")
            ema.save(ema_path)
            print(f"  -> [Saved] EMA model to {ema_path}")

def TrainTheModel():
    TrainVelocityModelFixedPairs(
        model_path=r"saved_models/Leg_C.pth",
        seeds=[100001, 100002, 100003], # Pass your list of seeds here
        image_training_folder=r"data/cifar_official/fid_reference_flat",
        attempt_load_training_state=False,
        use_EMA=True,
        batch_size_per_time=2,
        num_times=128,
        grad_accum_steps=3,
        weight_decay=1e-2,
        total_epochs=600,
        learning_rate_start=2e-4,
        learning_rate_end=5e-6,
        warmup_steps=2000,
    )

if __name__ == "__main__":
    TrainTheModel()
