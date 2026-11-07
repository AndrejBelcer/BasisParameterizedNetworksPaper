"""Cached data, reference identity, atomic model saving and human-readable outputs."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import math
import uuid

from PIL import Image
import torch

from .parameter_space import require_finite


def file_identity(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def reference_manifest(folder):
    root = Path(folder).resolve()
    unique = {}
    for path in sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() == ".png"):
        unique.setdefault(path.resolve(), path)
    paths = list(unique.values())  # Keep logical names for symlinks outside root.
    if len(paths) < 2:
        raise ValueError("Reference folder needs at least two PNGs")
    digest = hashlib.sha256()
    for index, path in enumerate(paths):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(bytes.fromhex(file_identity(path)))
        if (index+1) % 10000 == 0:
            print(f"Reference manifest: {index+1:,}/{len(paths):,}", flush=True)
    return paths, digest.hexdigest()


def load_pairs(path):
    data = torch.load(path, map_location="cpu", weights_only=True)
    g, n = data["G"], data["N"]
    if g.shape != n.shape or g.ndim != 4 or tuple(g.shape[1:]) != (3, 32, 32) or len(g) < 2:
        raise ValueError("Cached G and N must have shape [N, 3, 32, 32]")
    if not g.is_floating_point() or not n.is_floating_point():
        raise ValueError("Cached G and N must be normalized floating-point tensors")
    for start in range(0, len(g), 1024):
        require_finite(g[start:start+1024], "cached images")
        require_finite(n[start:start+1024], "cached noise")
        if float(g[start:start+1024].abs().max()) > 1.001:
            raise ValueError("Cached G must lie in [-1, 1]")
    print(f"Loaded {len(g):,} fixed G/N pairs.", flush=True)
    return g, n


def atomic_save(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def save_model(model, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem+".tmp"+path.suffix)
    model.save(str(temporary))
    temporary.replace(path)


class RunLog:
    """New output directory per invocation; JSONL is diagnostic, never resume state."""
    def __init__(self, model_path, seed, runtime, *, output_root="outputs/finalized_guided_training", **run_details):
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        self.folder = Path(output_root)/Path(model_path).stem/(stamp+"_"+uuid.uuid4().hex[:8])
        self.folder.mkdir(parents=True, exist_ok=False)
        self.image_folder = self.folder/"denoised_images"
        self.image_folder.mkdir()
        protocol = dict(settings=run_details, noise_seed_base=seed, runtime=runtime.signature())
        (self.folder/"settings.json").write_text(
            json.dumps(protocol, indent=2, default=str, allow_nan=False), encoding="utf-8")
        print(f"Logs and image grids: {self.folder}", flush=True)

    def event(self, event, **values):
        with (self.folder/"events.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(dict(event=event, **values), allow_nan=False)+"\n")


def save_grid(images, path):
    if images is None or not len(images):
        return
    images = images.detach().cpu()
    if images.ndim != 4 or images.shape[1] != 3:
        raise ValueError("Image grids require RGB images with shape [N, 3, H, W]")
    require_finite(images, "image grid")
    images = (images.clamp(0, 1)*255).round().byte()
    columns = min(8, len(images))
    h, w = images.shape[-2:]
    canvas = torch.zeros(3, math.ceil(len(images)/columns)*h, columns*w, dtype=torch.uint8)
    for i, picture in enumerate(images):
        canvas[:, (i//columns)*h:(i//columns+1)*h, (i%columns)*w:(i%columns+1)*w] = picture
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(canvas.permute(1, 2, 0).numpy()).save(path)
