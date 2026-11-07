import os
import sys
import glob
import shutil
import tempfile
import hashlib
import json
from importlib.metadata import version
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import scipy.linalg
import torch
import torch.nn.functional as F
import torchvision
from torchvision.models import inception_v3, Inception_V3_Weights
from PIL import Image

from src.core.globals import GPU_CONFIG
from src.utilities.print_helper import FormatFields, PrintStatus

# ==============================================================================
# Compatibility for evaluators that pass disp=False to scipy.linalg.sqrtm
# and expect the earlier two-value return format.
# ==============================================================================
_original_sqrtm = scipy.linalg.sqrtm

def _patched_sqrtm(*args, **kwargs):
    expects_tuple = "disp" in kwargs
    kwargs.pop("disp", None)
    result = _original_sqrtm(*args, **kwargs)
    if expects_tuple:
        return result, None
    return result

scipy.linalg.sqrtm = _patched_sqrtm
# ==============================================================================

# 1. PyTorch-FID
from pytorch_fid.fid_score import compute_statistics_of_path, calculate_fid_given_paths
from pytorch_fid.inception import InceptionV3

# 2. Torch-Fidelity
import torch_fidelity

# 3. Clean-FID
from cleanfid import fid
import cleanfid
import cleanfid.utils

# ==============================================================================
# --- WINDOWS BUG FIX (Monkey-Patch for Clean-FID) ---
# Clean-FID double-counts images on Windows because its internal EXTENSIONS 
# list contains both lowercase and uppercase extensions on a case-insensitive filesystem.
# ==============================================================================
cleanfid.utils.EXTENSIONS = ["png", "jpg", "jpeg"]
if hasattr(fid, "EXTENSIONS"):
    fid.EXTENSIONS = ["png", "jpg", "jpeg"]
# ==============================================================================

SUPPORTED_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
METRIC_RANDOM_SEED = 2020

# Display names are shared by the per-folder reports and the multi-run summary.
# Dictionary/CSV keys stay unchanged so existing analysis scripts still work.
METRIC_LABELS = {
    "PyTorch_FID": "FID [pytorch-fid; TensorFlow Inception-v3]",
    "Fidelity_FID": "FID [torch-fidelity; inception-v3-compat]",
    "Clean_FID": "FID [clean-fid; Inception-v3; clean resize]",
    "Clean_Legacy_FID": "FID [clean-fid; Inception-v3; legacy_tensorflow]",
    "Fidelity_IS_mean": "IS [torch-fidelity; inception-v3-compat]",
    "Fidelity_KID_mean_x1k": "KID [torch-fidelity; inception-v3-compat]",
    "Clean_KID_x1k": "KID [clean-fid; Inception-v3; clean resize]",
    "Clean_Legacy_KID_x1k": "KID [clean-fid; Inception-v3; legacy_tensorflow]",
    "Torchvision_KID_x1k": "KID [torchvision; Inception-v3 IMAGENET1K_V1; all pairs]",
}


def get_image_files(folder: str) -> List[str]:
    """
    Scans a folder and returns a deterministically sorted list of valid image paths.
    """
    if not os.path.isdir(folder):
        return []
    valid_files = [
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if os.path.splitext(f.lower())[1] in SUPPORTED_IMAGE_EXTS
        and os.path.isfile(os.path.join(folder, f))
    ]
    valid_files.sort()
    return valid_files


def get_image_manifest(folder: str, take_first_50k: bool = False) -> Dict[str, Any]:
    """Identify the selected images by filename and contents, excluding reports."""
    paths = get_image_files(folder)
    if take_first_50k:
        paths = paths[:50000]
    if len(paths) < 2:
        raise ValueError(f"Metrics require at least two images: {folder}")
    manifest = hashlib.sha256()
    for path in paths:
        image_hash = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                image_hash.update(chunk)
        record = [os.path.basename(path), image_hash.hexdigest()]
        manifest.update((json.dumps(record, ensure_ascii=True) + "\n").encode("utf-8"))
    return {
        "folder": os.path.abspath(folder),
        "image_count": len(paths),
        "images_sha256": manifest.hexdigest(),
    }


def get_metric_protocol(device: str) -> Dict[str, Any]:
    """Describe the evaluators without loading their models or reference features."""
    packages = ("torch", "torchvision", "pytorch-fid", "torch-fidelity", "clean-fid", "numpy", "scipy", "Pillow")
    with open(__file__, "r", encoding="utf-8") as f:
        implementation_hash = hashlib.sha256(f.read().encode("utf-8")).hexdigest()
    evaluation_device = torch.device(device)
    return {
        "implementation_sha256": implementation_hash,
        "package_versions": {name: version(name) for name in packages},
        "metric_labels": METRIC_LABELS,
        "feature_dimensions": 2048,
        "torchvision_weights": "Inception_V3_Weights.IMAGENET1K_V1",
        "torchvision_preprocessing": "RGB [0,1]; bilinear 299x299; antialias=True; ImageNet mean/std",
        "torchvision_estimator": "unbiased all-pairs cubic MMD; FP32 features/products, FP64 sums",
        "torchvision_batch_size": 128,
        "torchvision_kernel_block_size": 2048,
        "clean_fid_weights": "inception-2015-12-05.pt",
        "clean_fid_modes": ["clean", "legacy_tensorflow"],
        "torch_fidelity_extractor": "inception-v3-compat",
        "kid_kernel": "(x dot y / 2048 + 1)^3",
        "subset_kid": "100 subsets; size=min(1000, generated count, reference count)",
        "isc_splits": 10,
        "metric_random_seed": METRIC_RANDOM_SEED,
        "device": str(evaluation_device),
        "device_name": torch.cuda.get_device_name(evaluation_device) if evaluation_device.type == "cuda" else "CPU",
        "compile": bool(GPU_CONFIG.compile_model),
        "compile_mode": GPU_CONFIG.compile_mode_velocity.value,
        "cuda_matmul_fp32_precision": torch.backends.cuda.matmul.fp32_precision if evaluation_device.type == "cuda" else None,
        "cudnn_fp32_precision": torch.backends.cudnn.fp32_precision if evaluation_device.type == "cuda" else None,
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
    }


def cache_real_statistics(ref_folder: str, cache_dir: str, device: str, cache_identity: Optional[str] = None) -> Dict[str, str]:
    """
    Caches reference statistics for PyTorch-FID, Clean-FID (Clean & Legacy), 
    and prepares the Torch-Fidelity cache directory under cache/metrics/<hash>/.
    Runs feature extraction once per reference contents and evaluator protocol.
    The judger supplies cache_identity to reuse its already computed manifest.
    """
    try:
        rel_path = os.path.relpath(ref_folder)
    except ValueError:
        rel_path = os.path.abspath(ref_folder)
    normalized_path = rel_path.replace('\\', '/')
    if cache_identity is None:
        identity = dict(reference=get_image_manifest(ref_folder), protocol=get_metric_protocol(device))
        cache_identity = hashlib.sha256(json.dumps(identity, sort_keys=True).encode("utf-8")).hexdigest()
    ref_hash = hashlib.sha256((normalized_path + cache_identity).encode('utf-8')).hexdigest()[:20]
    ref_cache_dir = os.path.join(cache_dir, ref_hash)
    os.makedirs(ref_cache_dir, exist_ok=True)
    
    # Count and print reference images
    ref_images = get_image_files(ref_folder)
    print(f"\n[INFO] Found {len(ref_images):,} reference images in: {ref_folder}")
    if len(ref_images) < 2:
        raise ValueError(f"Metrics require at least two reference images: {ref_folder}")

    # Clean-FID resolves custom statistics relative to its package __file__.
    cleanfid.__file__ = os.path.join(ref_cache_dir, "dummy.py")
    if hasattr(cleanfid, "custom_stats_folder"):
        cleanfid.custom_stats_folder = os.path.join(ref_cache_dir, "stats")

    pt_fid_cache = os.path.join(ref_cache_dir, "pytorch_fid_stats.npz")
    tf_cache_root = os.path.join(ref_cache_dir, "torch_fidelity")
    os.makedirs(tf_cache_root, exist_ok=True)

    print(f"\n[CACHE] Checking reference statistics in: {ref_cache_dir}")

    # --- 1. PyTorch-FID Cache ---
    if not os.path.exists(pt_fid_cache):
        print(f"[CACHE] Computing PyTorch-FID statistics for {ref_folder}...")
        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
        model = InceptionV3([block_idx]).to(device)
        m, s = compute_statistics_of_path(ref_folder, model, batch_size=128, dims=2048, device=device)
        np.savez(pt_fid_cache, mu=m, sigma=s)
        print(f"[CACHE] PyTorch-FID stats saved to: {pt_fid_cache}")
    else:
        print("[CACHE] PyTorch-FID cache found. Skipping computation.")

    # --- 2. Clean-FID Cache (Clean Mode) ---
    clean_name = f"ref_{ref_hash}"
    if not fid.test_stats_exists(name=clean_name, mode="clean"):
        print("[CACHE] Computing Clean-FID (Clean mode) custom stats...")
        fid.make_custom_stats(name=clean_name, fdir=ref_folder, mode="clean", num_workers=0, device=device)
    else:
        print("[CACHE] Clean-FID cache (Clean mode) found. Skipping.")

    # --- 3. Clean-FID Cache (Legacy Mode) ---
    legacy_name = f"ref_{ref_hash}_legacy"
    if not fid.test_stats_exists(name=legacy_name, mode="legacy_tensorflow"):
        print("[CACHE] Computing Clean-FID (Legacy TF mode) custom stats...")
        fid.make_custom_stats(name=legacy_name, fdir=ref_folder, mode="legacy_tensorflow", num_workers=0, device=device)
    else:
        print("[CACHE] Clean-FID cache (Legacy mode) found. Skipping.")

    return {
        "pt_fid_cache": pt_fid_cache,
        "tf_cache_root": tf_cache_root,
        "clean_name": clean_name,
        "legacy_name": legacy_name,
        "ref_cache_dir": ref_cache_dir,
        "torchvision_cache": os.path.join(ref_cache_dir, "torchvision_kid_reference.pt"),
    }


def compute_pytorch_fid(gen_folder: str, pt_fid_cache: str, device: str) -> float:
    """Computes standard PyTorch-FID against cached reference statistics."""
    fid_value = calculate_fid_given_paths(
        paths=[pt_fid_cache, gen_folder],
        batch_size=128,
        device=device,
        dims=2048,
    )
    return float(fid_value)


def compute_torch_fidelity(
    gen_folder: str,
    ref_folder: str,
    tf_cache_root: str,
    device: str,
    num_images: int,
) -> Dict[str, float]:
    """
    Computes exact TensorFlow-matching FID, Inception Score (mean + std), 
    and Kernel Inception Distance (mean + std, scaled x10^3).
    
    Caches reference statistics under tf_cache_root using input2_cache_name='ref_stats'.
    """
    is_cuda = "cuda" in str(device)
    
    # KID uses a default subset size of 1000. Clamp if sample count is lower to prevent crashes.
    kid_subset_size = min(1000, num_images, len(get_image_files(ref_folder)))
    if kid_subset_size < 2:
        raise ValueError("KID requires at least two generated and reference images")

    metrics = torch_fidelity.calculate_metrics(
        input1=gen_folder,
        input2=ref_folder,
        input2_cache_name="ref_stats",  # Isolated by the reference-folder cache root
        cache_root=tf_cache_root,
        cuda=is_cuda,
        isc=True,
        fid=True,
        kid=True,
        feature_extractor="inception-v3-compat",
        rng_seed=METRIC_RANDOM_SEED,
        isc_splits=10,
        kid_subsets=100,
        kid_subset_size=kid_subset_size,
        verbose=False,
        num_workers=0,
    )

    return {
        "Fidelity_FID": float(metrics["frechet_inception_distance"]),
        "Fidelity_IS_mean": float(metrics["inception_score_mean"]),
        "Fidelity_IS_std": float(metrics["inception_score_std"]),
        "Fidelity_KID_mean_x1k": float(metrics["kernel_inception_distance_mean"]) * 1000.0,
        "Fidelity_KID_std_x1k": float(metrics["kernel_inception_distance_std"]) * 1000.0,
    }


def compute_clean_fid(gen_folder: str, clean_name: str, legacy_name: str, ref_cache_dir: str, device: str) -> Dict[str, float]:
    """
    Computes Clean-FID (Clean & Legacy) and Clean-KID (Clean & Legacy, scaled x10^3).
    """
    cleanfid.__file__ = os.path.join(ref_cache_dir, "dummy.py")
    if hasattr(cleanfid, "custom_stats_folder"):
        cleanfid.custom_stats_folder = os.path.join(ref_cache_dir, "stats")

    clean_fid_val = fid.compute_fid(
        gen_folder,
        dataset_name=clean_name,
        dataset_split="custom",
        mode="clean",
        device=device,
        num_workers=0,
    )
    legacy_fid_val = fid.compute_fid(
        gen_folder,
        dataset_name=legacy_name,
        dataset_split="custom",
        mode="legacy_tensorflow",
        device=device,
        num_workers=0,
    )

    # Clean-FID custom KID evaluation
    clean_kid_val = fid.compute_kid(
        gen_folder,
        dataset_name=clean_name,
        dataset_split="custom",
        mode="clean",
        device=device,
        num_workers=0,
    )
    legacy_kid_val = fid.compute_kid(
        gen_folder,
        dataset_name=legacy_name,
        dataset_split="custom",
        mode="legacy_tensorflow",
        device=device,
        num_workers=0,
    )

    return {
        "Clean_FID": float(clean_fid_val),
        "Clean_Legacy_FID": float(legacy_fid_val),
        "Clean_KID_x1k": float(clean_kid_val) * 1000.0,
        "Clean_Legacy_KID_x1k": float(legacy_kid_val) * 1000.0,
    }


@torch.no_grad()
def polynomial_kernel_mean(a: torch.Tensor, b: torch.Tensor, block_size: int = 2048, same: bool = False) -> float:
    """All-pairs cubic kernel mean; omit self-pairs for an unbiased KID estimate.

    FP32 kernel products and FP64 accumulation match the training evaluator.
    Inputs live on the evaluation device; only a block-sized kernel is allocated.
    """
    if min(len(a), len(b)) < 2 or a.shape[1] != b.shape[1] or block_size < 1:
        raise ValueError("Expected compatible feature matrices with at least two rows and a positive block size")
    if same and a.shape != b.shape:
        raise ValueError("A self-kernel requires matching feature shapes")
    total = torch.zeros((), device=a.device, dtype=torch.float64)
    with torch.autocast(device_type=a.device.type, enabled=False):
        for i in range(0, len(a), block_size):
            aa = a[i:i+block_size].float()
            for j in range(i if same else 0, len(b), block_size):
                bb = b[j:j+block_size].float()
                kernel = (aa @ bb.T / a.shape[1] + 1).pow(3)
                if same and i == j:
                    kernel = (kernel + kernel.T) * .5
                    kernel.diagonal().zero_()
                total += kernel.sum(dtype=torch.float64) * (2 if same and i != j else 1)
    denominator = len(a)*(len(a)-1) if same else len(a)*len(b)
    return float(total / denominator)


class TorchvisionKIDEvaluator:
    """KID using torchvision Inception-v3 IMAGENET1K_V1 and all feature pairs.

    Matches kid_guided_training's torchvision weights, resize, normalization and
    kernel. Benchmark features use FP32, without autocast. PNG quantization and
    any training mixed precision can still change scores; this is not TF KID.
    """
    def __init__(self, ref_folder: str, cache_path: str, device: str,
                 batch_size: int = 128, kernel_block_size: int = 2048):
        self.device = torch.device(device)
        self.batch_size = batch_size
        self.kernel_block_size = kernel_block_size
        weights = Inception_V3_Weights.IMAGENET1K_V1
        model = inception_v3(weights=weights, transform_input=False)
        model.fc = torch.nn.Identity()
        model = model.to(device=self.device, dtype=torch.float32,
                         memory_format=torch.channels_last).eval().requires_grad_(False)
        compile_mode = GPU_CONFIG.compile_mode_velocity.value
        if GPU_CONFIG.compile_model and compile_mode.lower() != "none":
            print(f"Compiling torchvision KID features: {compile_mode}")
            model = torch.compile(model, mode=compile_mode, fullgraph=True)
        self.model = model
        self.mean = torch.tensor([.485, .456, .406], device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor([.229, .224, .225], device=self.device).view(1, 3, 1, 1)

        paths = get_image_files(ref_folder)
        if len(paths) < 2:
            raise ValueError("Torchvision KID requires at least two reference images")
        manifest = hashlib.sha256()
        for path in paths:
            stat = os.stat(path)
            manifest.update(json.dumps([os.path.basename(path), stat.st_size, stat.st_mtime_ns]).encode())
        metadata = dict(format="torchvision_kid_png_v1", weights=weights.name,
                        torch=str(torch.__version__), torchvision=str(torchvision.__version__),
                        reference=os.path.abspath(ref_folder), manifest=manifest.hexdigest(),
                        preprocessing="RGB01-bilinear299-antialias-ImageNet-normalization",
                        precision="float32-no-autocast", device=str(self.device),
                        batch_size=batch_size, kernel_block_size=kernel_block_size,
                        compile=bool(GPU_CONFIG.compile_model), compile_mode=compile_mode,
                        cuda_matmul_fp32_precision=torch.backends.cuda.matmul.fp32_precision if self.device.type == "cuda" else None,
                        cudnn_fp32_precision=torch.backends.cudnn.fp32_precision if self.device.type == "cuda" else None)
        cached = torch.load(cache_path, map_location="cpu", weights_only=True) if os.path.exists(cache_path) else None
        if cached is not None and cached.get("metadata") == metadata:
            reference, self.c_ref = cached["features"], cached["c_ref"]
            PrintStatus(f"[CACHE] Loaded torchvision KID reference features: {cache_path}")
        else:
            reference = self.folder_features(ref_folder)
            on_device = reference.to(self.device)
            self.c_ref = polynomial_kernel_mean(on_device, on_device, kernel_block_size, same=True)
            del on_device
            os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
            temporary_path = cache_path + ".tmp"
            torch.save(dict(metadata=metadata, features=reference, c_ref=self.c_ref), temporary_path)
            os.replace(temporary_path, cache_path)
            PrintStatus(f"[CACHE] Saved torchvision KID reference features: {cache_path}", "success")
        if reference.shape != (len(paths), 2048) or not torch.isfinite(reference).all() or not np.isfinite(self.c_ref):
            raise ValueError("Invalid torchvision KID reference cache")
        self.reference = reference

    @torch.no_grad()
    def folder_features(self, folder: str) -> torch.Tensor:
        paths = get_image_files(folder)
        if len(paths) < 2:
            raise ValueError(f"Torchvision KID requires at least two images: {folder}")
        parts = []
        with torch.autocast(device_type=self.device.type, enabled=False):
            for start in range(0, len(paths), self.batch_size):
                images = []
                for path in paths[start:start+self.batch_size]:
                    with Image.open(path) as picture:
                        array = np.array(picture.convert("RGB"), dtype=np.float32) / 255.
                    images.append(torch.from_numpy(array).permute(2, 0, 1))
                batch = torch.stack(images).to(self.device)
                batch = F.interpolate(batch, size=(299, 299), mode="bilinear",
                                      align_corners=False, antialias=True)
                batch = ((batch-self.mean)/self.std).to(memory_format=torch.channels_last)
                features = self.model(batch).float().cpu().clone()
                if not torch.isfinite(features).all():
                    raise FloatingPointError("Nonfinite torchvision Inception features")
                parts.append(features)
                if start == 0 or start+self.batch_size >= len(paths) or (start//self.batch_size+1) % 100 == 0:
                    print(f"  Torchvision features: {min(start+self.batch_size, len(paths)):,}/{len(paths):,}")
        return torch.cat(parts)

    def score(self, gen_folder: str) -> Dict[str, float]:
        features = self.folder_features(gen_folder).to(self.device)
        reference = self.reference.to(self.device)
        self_mean = polynomial_kernel_mean(features, features, self.kernel_block_size, same=True)
        cross_mean = polynomial_kernel_mean(features, reference, self.kernel_block_size)
        kid = self_mean + self.c_ref - 2*cross_mean
        if not np.isfinite(kid):
            raise FloatingPointError("Nonfinite torchvision KID")
        # Do not clip negative unbiased estimates or invent a subset standard deviation.
        return {"Torchvision_KID_x1k": kid * 1000.0}


def score_dictionary_to_string(folder_name: str, eval_count: int, results: Dict[str, Any]) -> str:
    result_string = "=" * 96 + "\n"
    result_string += f"Folder: {folder_name} | Images Evaluated: {eval_count:,}\n"
    result_string += "-" * 96 + "\n"
    groups = {
        "FID Scores": ("PyTorch_FID", "Fidelity_FID", "Clean_FID", "Clean_Legacy_FID"),
        "Inception Score (IS)": ("Fidelity_IS_mean",),
        "KID Scores (x 10³)": ("Fidelity_KID_mean_x1k", "Clean_KID_x1k", "Clean_Legacy_KID_x1k", "Torchvision_KID_x1k"),
    }
    std_keys = {"Fidelity_IS_mean": "Fidelity_IS_std", "Fidelity_KID_mean_x1k": "Fidelity_KID_std_x1k"}
    for heading, keys in groups.items():
        result_string += f"{heading}:\n"
        rows = []
        for key in keys:
            value = f"{results[key]:.4f}"
            if key in std_keys:
                value += f" ± {results[std_keys[key]]:.4f}"
            rows.append((f"• {METRIC_LABELS[key]}", value))
        result_string += FormatFields(rows, indent="   ") + "\n"
    result_string += "\nFID/KID features: 2048-dimensional Inception-v3 pool features.\n"
    result_string += "clean-fid: TensorFlow Inception weights (inception-2015-12-05.pt);\n"
    result_string += "  clean = PIL bicubic resize; legacy_tensorflow = TensorFlow-compatible resize.\n"
    result_string += "torchvision: IMAGENET1K_V1 weights; RGB [0,1], bilinear 299x299 with antialiasing,\n"
    result_string += "  ImageNet mean/std normalization; FP32 features, no autocast.\n"
    result_string += "KID: unbiased cubic MMD, k(x,y)=(x·y/2048+1)^3.\n"
    result_string += "  torchvision uses all pairs; other KIDs average 100 subsets of up to 1000 images.\n"
    result_string += f"  Subset sampling seed: {METRIC_RANDOM_SEED}. KID ± is subset SD; IS ± is split SD (10 splits).\n"
    result_string += "=" * 96

    print(result_string)
    return result_string

def master_score_generator(
    gen_folder: str,
    ref_folder: str,
    cache_info: Dict[str, str],
    device: str,
    take_first_50k: bool = False,
    torchvision_evaluator: Optional[TorchvisionKIDEvaluator] = None,
) -> Tuple[Dict[str, Any], str]:
    """
    Evaluates a generated image folder across all metric engines.
    
    If take_first_50k is True and folder contains > 50,000 images, creates a 
    temporary hardlinked staging folder of the first 50,000 deterministically sorted 
    images to ensure strict academic benchmark comparability.
    """
    folder_name = os.path.basename(os.path.normpath(gen_folder))
    image_files = get_image_files(gen_folder)
    total_found = len(image_files)

    if total_found == 0:
        raise FileNotFoundError(f"No valid image files found in folder: {gen_folder}")

    staging_dir: Optional[str] = None
    eval_folder = gen_folder
    eval_count = total_found

    # Subsampling logic
    if take_first_50k and total_found > 50000:
        print(f"\n[INFO] Found {total_found:,} images. Subsetting to first 50,000 for strict benchmarking...")
        parent_dir = os.path.dirname(os.path.abspath(gen_folder))
        staging_dir = tempfile.mkdtemp(prefix="_staging_50k_", dir=parent_dir)
        
        # Hardlink (or copy fallback) the first 50,000 images
        selected_files = image_files[:50000]
        for src_path in selected_files:
            dst_path = os.path.join(staging_dir, os.path.basename(src_path))
            try:
                os.link(src_path, dst_path)
            except OSError:
                shutil.copyfile(src_path, dst_path)

        eval_folder = staging_dir
        eval_count = 50000
    else:
        print(f"\n[INFO] Evaluating all {eval_count:,} images in: {folder_name}")

    results: Dict[str, Any] = {
        "Folder": folder_name,
        "Num_Images": eval_count,
    }

    try:
        print("  -> 1/4 Computing FID [pytorch-fid, TF-compatible]...")
        results["PyTorch_FID"] = compute_pytorch_fid(eval_folder, cache_info["pt_fid_cache"], device)

        print("  -> 2/4 Computing FID, IS and KID [torch-fidelity]...")
        fidelity_res = compute_torch_fidelity(eval_folder, ref_folder, cache_info["tf_cache_root"], device, eval_count)
        results.update(fidelity_res)

        print("  -> 3/4 Computing FID and KID [clean-fid: clean / legacy TF]...")
        # Clean-FID samples KID subsets through NumPy's global RNG. Keep the
        # benchmark reproducible without changing the caller's random stream.
        random_state = np.random.get_state()
        try:
            np.random.seed(METRIC_RANDOM_SEED)
            clean_res = compute_clean_fid(eval_folder, cache_info["clean_name"], cache_info["legacy_name"], cache_info["ref_cache_dir"], device)
        finally:
            np.random.set_state(random_state)
        results.update(clean_res)

        print(f"  -> 4/4 Computing {METRIC_LABELS['Torchvision_KID_x1k']}...")
        if torchvision_evaluator is None:
            torchvision_evaluator = TorchvisionKIDEvaluator(ref_folder, cache_info["torchvision_cache"], device)
        results.update(torchvision_evaluator.score(eval_folder))

    finally:
        # Guarantee cleanup of staging folder
        if staging_dir is not None and os.path.exists(staging_dir):
            shutil.rmtree(staging_dir, ignore_errors=True)

    # Detailed console logging for this folder
    result_string = score_dictionary_to_string(folder_name, eval_count, results)

    return results, result_string
