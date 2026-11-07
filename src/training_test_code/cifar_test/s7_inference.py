import os
import shutil
from pathlib import Path
import torch
from PIL import Image

from typing import List, Optional

from src.core.globals import GPU_CONFIG
from src.modules.velocity_model import VelocityModel
from src.utilities.tensor_utils import FloatTensorToByteTensor
from src.utilities.print_helper import PrintFields, PrintSummary


from src.training_helpers.time_sampler import TimeSampler


def save_images_to_folder(images: torch.Tensor, folder: str):
    """Save a [N, 3, H, W] batch in the project's float range as RGB PNGs.

    The shared conversion unnormalizes, clamps and rounds to the nearest
    8-bit value. PNG preserves those integer pixels losslessly.
    """
    if images.ndim != 4 or images.shape[1] != 3:
        raise ValueError("Expected RGB images with shape [N, 3, H, W]")

    output_folder = Path(folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    next_index = 1 + max((int(path.stem) for path in output_folder.iterdir() if path.is_file() and path.suffix.lower() == ".png" and path.stem.isdecimal()), default=-1)
    # Convert in FP32 before quantization, including when inference uses FP16/BF16.
    pixels = FloatTensorToByteTensor(images.detach().float()).cpu()
    pixels = pixels.permute(0, 2, 3, 1).contiguous().numpy()
    for index, array in enumerate(pixels, start=next_index):
        Image.fromarray(array).save(output_folder / f"{index:05d}.png")


def generate_initial_noise(
    batch_size: int,
    channels: int,
    img_size: int,
    device: torch.device,
    dtype: torch.dtype,
    generator: torch.Generator,
    noise_factor: float = 1.0,
):
    """
    Generates deterministic Gaussian noise using the supplied generator.
    """
    return (torch.randn((batch_size, channels, img_size, img_size), device=device, dtype=dtype, generator=generator) * noise_factor)


@torch.no_grad()
def generate_images_standard(
    model: VelocityModel,
    num_samples: int,
    batch_size: int,
    steps: int,
    time_sampler: TimeSampler,
    img_size: int,
    channels: int,
    save_dir: str,
    device: torch.device,
    generator: torch.Generator,
    noise_factor: float = 1.0,
):
    """
    Generates images using deterministic initial noise and Euler integration.
    Integrates from t = 1.0 down to t = 0.0, evaluating the model at points
    sampled from time_sampler (stopping model queries at lowest_t).
    """
    model.eval()

    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)

    os.makedirs(save_dir, exist_ok=True)

    count = 0
    print(f"Generating {num_samples} images ({steps} steps)")

    while count < num_samples:

        current_bs = min(batch_size, num_samples - count)

        # ---------------------------------------------------
        # Deterministic initial noise (t = 1.0)
        # ---------------------------------------------------
        x = generate_initial_noise(batch_size=current_bs, channels=channels, img_size=img_size, device=device, dtype=GPU_CONFIG.dtype, generator=generator, noise_factor=noise_factor)


        # ---------------------------------------------------
        # Compute Time Grid and Delta T
        # ---------------------------------------------------
        # 1. Sample (steps - 1) intermediate points from time_sampler (within [lowest_t, highest_t])
        sampled_t = time_sampler.sample(steps - 1).to(device=device, dtype=GPU_CONFIG.dtype)

        # 2. Form the full boundary grid [1.0, sampled_t..., 0.0]
        t_boundary_start = torch.tensor([1.0], device=device, dtype=GPU_CONFIG.dtype)
        t_boundary_end = torch.tensor([0.0], device=device, dtype=GPU_CONFIG.dtype)
        
        t_grid = torch.cat([t_boundary_start, sampled_t, t_boundary_end])
        t_grid, _ = torch.sort(t_grid, descending=True)

        # 3. Model evaluation points: t_grid[:-1] (all points >= lowest_t)
        #    Step deltas: t_grid[1:] - t_grid[:-1] (final delta lands exactly on t=0.0)
        list_t = t_grid[:-1]
        list_delta_t = t_grid[1:] - t_grid[:-1]

        # ---------------------------------------------------
        # Euler integration
        # ---------------------------------------------------
        for t, dt in zip(list_t, list_delta_t):
            t_tensor = t.view(1)

            v = model(x.to(GPU_CONFIG.dtype), t_tensor).float()

            x = x + dt * v

        save_images_to_folder(x, save_dir)

        count += current_bs

        if count % (batch_size * 4) == 0 or count == num_samples:
            print(f"Progress: {count}/{num_samples}")

def GetEvaluationFolder(
    model_name: str,
    steps: int,
    num_samples: int,
    seed: int,
    noise_factor: float = 1.0,
    *,
    output_root: Optional[str] = None,
    lowest_t: Optional[float] = None,
    run_id: str = "",
):
    """Expose the folder name without changing the sampling procedure."""
    if Path(model_name).name != model_name or any(c in model_name for c in "/\\"):
        raise ValueError("model_name must be a filename stem")
    if any(c not in "0123456789abcdef" for c in run_id):
        raise ValueError("run_id must be a hexadecimal identity")
    root = Path(output_root or os.path.join("outputs", "benchmarks", "full_evaluations")).resolve()
    cutoff = "" if lowest_t is None else f"_t{float(lowest_t)}"
    identity = "" if not run_id else f"_{run_id}"
    folder = root / f"{model_name}_S{steps}{cutoff}_{num_samples}_s{seed}_f{round(noise_factor, 2)}{identity}"
    if folder.resolve().parent != root:
        raise ValueError("Evaluation folder must stay directly inside output_root")
    return str(folder)


def run_full_evaluation(
    model_name: str,
    model: VelocityModel,
    num_samples: int = 50000,
    batch_size: int = 500,
    steps: int = 250,
    time_sampler: TimeSampler = TimeSampler(),
    seed: int = 11,
    noise_factor: float = 1.0,
    *,
    output_root: Optional[str] = None,
    include_cutoff_in_name: bool = False,
    run_id: str = "",
):
    """
    seed:
        Makes the initial Gaussian noises deterministic.

    noise_factor:
        Multiplies the initial Gaussian noise.
    """
    device = GPU_CONFIG.device

    generator = torch.Generator(device=device)
    generator.manual_seed(seed)

    model.eval()


    PrintSummary("RUNNING FULL BENCHMARK")
    PrintFields([
        ("Model", model_name),
        ("Samples", num_samples),
        ("Steps", steps),
        ("Seed", seed),
        ("Noise factor", noise_factor),
    ])

    try:
        img_size = model.arch_config["initializer_layers"][0]["params"]["x"]
        channels = model.arch_config["initializer_layers"][0]["params"]["a"]
    except Exception:
        img_size = 32
        channels = 3

    save_dir = GetEvaluationFolder(model_name, steps, num_samples, seed, noise_factor,
        output_root=output_root, lowest_t=time_sampler.lowest_t if include_cutoff_in_name else None,
        run_id=run_id)

    # -------------------------------------------------------
    # Generate images
    # -------------------------------------------------------

    generate_images_standard(model=model, num_samples=num_samples, batch_size=batch_size, steps = steps, time_sampler = time_sampler, img_size=img_size, channels=channels, save_dir=save_dir, device=device, generator=generator, noise_factor=noise_factor)
    return save_dir



def MakeInference(
        model_name: str = "CifEModel",
        seeds: List[int] = [11, 12, 13, 14, 15],
        steps: int = 55,
        batch_size: int = 512,
        num_samples: int = 50000,
        time_sampler: TimeSampler = TimeSampler(lowest_t=0.085),
        *,
        model_path: Optional[str] = None,
        output_root: Optional[str] = None,
        include_cutoff_in_name: bool = False,
        run_id: str = "",
        ):

    model_path = model_path or f"saved_models/{model_name}.pth"
    model = VelocityModel.load(model_path)
    folders = []
    for seed in seeds:
        folders.append(run_full_evaluation(model_name=model_name, model=model,
            batch_size=batch_size, num_samples=num_samples, steps=steps, seed=seed,
            time_sampler=time_sampler, output_root=output_root,
            include_cutoff_in_name=include_cutoff_in_name, run_id=run_id))
    return folders
