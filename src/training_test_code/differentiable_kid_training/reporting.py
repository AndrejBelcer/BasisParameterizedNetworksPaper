"""Checkpoints, bounded logs and resource measurements. No backend settings."""
import csv
import ast
import hashlib
import json
import os
from pathlib import Path
import torch


def file_hash(path):
    result = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            result.update(block)
    return result.hexdigest()


def implementation_hash(path):
    """Caller edits are checked by argument identity, not as algorithm changes."""
    tree = ast.parse(Path(path).read_text(encoding='utf-8'))
    tree.body = [node for node in tree.body
                 if not isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)) or node.name != 'TrainTheModel']
    return hashlib.sha256(ast.dump(tree,include_attributes=False).encode()).hexdigest()


def atomic_save(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+'.tmp')
    torch.save(data, temporary)
    os.replace(temporary, path)


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(data, indent=2, default=str, allow_nan=False), encoding='utf-8')
    os.replace(temporary, path)


def cpu_copy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu_copy(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(cpu_copy(v) for v in value)
    return value


def save_checkpoint(path, model, optimizer, update, pairs, rgb_weight, best_kid,
                    best_fid, identity, microbatch, stale, microbatch_cap=32, rgb_time_chunk=16):
    # arch_config/state_dict keep compatibility with VelocityModel.load().
    atomic_save({'arch_config': model.get_config(), 'state_dict': cpu_copy(model.state_dict()),
                 'optimizer': cpu_copy(optimizer.state_dict()), 'update': update,
                 'rgb_pairs': pairs.state_dict(), 'rgb_weight': rgb_weight,
                 'best_kid': best_kid, 'best_fid': best_fid, 'identity': identity,
                 'microbatch': microbatch, 'stale': stale,
                 'microbatch_cap': microbatch_cap, 'rgb_time_chunk': rgb_time_chunk}, path)


class Logs:
    def __init__(self, folder):
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)

    def append(self, filename, row):
        path = self.folder/filename
        exists = path.exists() and path.stat().st_size > 0
        if exists:
            with path.open(newline='', encoding='utf-8') as stream:
                fields = next(csv.reader(stream))
            if set(fields) != set(row):
                raise RuntimeError(f'CSV schema mismatch: {path}')
        else:
            fields = list(row)
        with path.open('a', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            if not exists:
                writer.writeheader()
            writer.writerow(row)

    def event(self, event, **values):
        with (self.folder/'events.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps({'event': event, **values}, default=str, allow_nan=False)+'\n')


def memory_gb(device):
    if device.type != 'cuda':
        return 0.0
    return torch.cuda.max_memory_allocated(device)/1e9


def montage(images01, path, count=100):
    from PIL import Image
    pixels = (images01[:count].clamp(0, 1)*255).round().byte().cpu()
    width, height = pixels.shape[-1], pixels.shape[-2]
    columns = min(10, len(pixels))
    canvas = Image.new('RGB', (columns*width, ((len(pixels)+columns-1)//columns)*height))
    for i, pixel in enumerate(pixels):
        canvas.paste(Image.fromarray(pixel.permute(1, 2, 0).numpy()), ((i%columns)*width, (i//columns)*height))
    canvas.save(path)
