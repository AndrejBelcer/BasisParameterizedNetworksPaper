"""Shared-noise KIDs with component and combined paired uncertainty."""
from dataclasses import dataclass, field
from pathlib import Path
import hashlib
import json
import math
import time

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset

from src.training_helpers.time_sampler import TimeSampler, TimeSamplingType
from .feature_extractors import TorchvisionFeatures, CleanFeatures
from .generation import EulerGenerator
from .io_utils import atomic_save, reference_manifest
from .parameter_space import require_finite


@dataclass
class NoiseBank:
    seed: int
    noises: torch.Tensor


@dataclass
class KIDResult:
    score: float
    self_rows: torch.Tensor
    cross_rows: torch.Tensor
    noise_seed: int
    start: int
    count: int
    component_scores: dict = field(default_factory=dict)
    component_self_rows: dict = field(default_factory=dict)
    component_cross_rows: dict = field(default_factory=dict)
    sample_ranges: tuple = ()

    @property
    def intervals(self):
        ranges = self.sample_ranges or ((self.noise_seed, self.start, self.start+self.count),)
        merged = []
        for seed, start, end in ranges:
            if merged and merged[-1][0] == seed and merged[-1][2] == start:
                merged[-1] = (seed, merged[-1][1], end)
            else:
                merged.append((seed, start, end))
        return tuple(merged)

    def require_same_samples(self, other):
        if self.intervals != other.intervals or self.count != other.count:
            raise ValueError("Paired comparisons and fits require exactly the same noise samples")

    def leave_one_out(self):
        # The fixed real-reference constant is omitted; it cancels in all
        # paired differences and finite-difference derivatives.
        n = self.count
        if n < 3:
            raise ValueError("Delete-one KID requires at least three samples")
        rows, cross = self.self_rows.double(), self.cross_rows.double()
        return (rows.sum()-2*rows)/((n-1)*(n-2))-2*(cross.sum()-cross)/(n-1)

    def component(self, name):
        """View one feature pipeline with exactly the same paired sample identity."""
        if name not in self.component_scores:
            raise KeyError(f"Unknown KID component: {name}")
        return KIDResult(self.component_scores[name], self.component_self_rows[name],
                         self.component_cross_rows[name], self.noise_seed, self.start, self.count,
                         sample_ranges=self.sample_ranges)

    def component_leave_one_out(self, name):
        """Per-component delete-one values; shared sample ordering keeps covariance."""
        return self.component(name).leave_one_out()


@dataclass
class FeatureHistory:
    noise_seed: int
    features: dict = field(default_factory=dict)
    images: torch.Tensor | None = None
    scores: dict = field(default_factory=dict)
    partitions: dict = field(default_factory=dict)

    @property
    def count(self):
        return len(next(iter(self.features.values()))) if self.features else 0

    @property
    def feature_mib(self):
        return sum(value.numel()*value.element_size() for value in self.features.values())/1024**2


def kid_standard_error(result):
    """Delete-one SE of standalone KID, conditional on fixed real references."""
    values = result.leave_one_out()
    require_finite(values, "standalone jackknife values")
    return float(((len(values)-1)/len(values)*(values-values.mean()).square().sum()).sqrt())


def paired_difference(candidate, incumbent):
    """Approximate paired jackknife SE conditional on the fixed reference set.

    This does not correct selection bias when the samples also selected a model.
    """
    candidate.require_same_samples(incumbent)
    values = candidate.leave_one_out()-incumbent.leave_one_out()
    require_finite(values, "paired jackknife values")
    count = candidate.count
    se = float(((count-1)/count*(values-values.mean()).square().sum()).sqrt())
    return candidate.score-incumbent.score, se


def component_paired_difference(candidate, incumbent, name):
    """Paired difference and SE for an individual feature pipeline."""
    return paired_difference(candidate.component(name), incumbent.component(name))


class ReferenceImages(Dataset):
    def __init__(self, paths):
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        with Image.open(self.paths[index]) as picture:
            array = np.array(picture.convert("RGB"), dtype=np.float32)/255.
        return torch.from_numpy(array).permute(2, 0, 1)


class KIDEvaluator:
    def __init__(self, model, runtime, *, image_validation_folder, num_workers,
                 generation_steps, generation_min_time, generation_max_time,
                 generation_batch_size, inception_batch_size, kid_kernel_block_size,
                 reference_on_gpu, sample_grid_image_count, sample_count,
                 use_two_kid_models=True):
        self.model, self.runtime = model, runtime
        self.num_workers = num_workers
        # Fixed tile shapes make a model's samples independent of which cached
        # prefix is requested. The tile never exceeds the requested batch size.
        self.generation_batch_size = min(generation_batch_size, sample_count)
        if sample_count % self.generation_batch_size:
            raise ValueError("generation_batch_size must divide the initial tier, or be at least its size")
        self.inception_batch_size = inception_batch_size
        self.kid_kernel_block_size = kid_kernel_block_size
        self.sample_grid_image_count = sample_grid_image_count
        reference_paths, reference_identity = reference_manifest(image_validation_folder)
        self.score_evaluations = 0
        self.generated_images = 0
        self.generation_seconds = self.feature_seconds = self.kernel_seconds = 0.
        self.feature_seconds_by_model = {}
        self._pending_timers = []
        sampler = TimeSampler(method=TimeSamplingType.DETERMINISTIC_UNIFORM,
                              lowest_t=generation_min_time, highest_t=generation_max_time,
                              device="cpu")
        middle = sampler.sample(generation_steps-1).float().reshape(-1)
        require_finite(middle, "Euler interior times")
        if len(middle) != generation_steps-1 or not bool(((middle > 0) & (middle < 1)).all()):
            raise ValueError("Euler sampler must provide the requested number of interior times in (0, 1)")
        # The external sampler's return order must not determine integration
        # direction. Preserve its time locations and explicitly integrate 1 -> 0.
        middle = middle.sort(descending=True).values
        self.grid = torch.cat([torch.ones(1), middle, torch.zeros(1)])
        if not bool((self.grid[1:] < self.grid[:-1]).all()):
            raise ValueError("Euler time grid must be strictly decreasing without duplicate times")
        # Compile the complete wrapper around the ORIGINAL model. RGB guidance
        # has its own compiled vmap; no compiled model is nested inside this loop.
        generator = EulerGenerator(model, self.grid, runtime.device, runtime.model_dtype).eval()
        self.generate = runtime.compile(generator, "complete Euler generation loop")
        print(f"Generation tile={self.generation_batch_size:,} (requested batch={generation_batch_size:,}); "
              "fixed shapes across evaluations.", flush=True)

        print("Initializing KID Inception features...", flush=True)
        self.extractors = {"torchvision": TorchvisionFeatures(runtime)}
        if use_two_kid_models:
            self.extractors["clean_float"] = CleanFeatures(runtime)
        self.feature_seconds_by_model = dict.fromkeys(self.extractors, 0.)
        self.references, self.reference_constants = {}, {}
        for name, extractor in self.extractors.items():
            self._load_reference(extractor, reference_paths, reference_identity, reference_on_gpu)
        print(f"KID objective: equal-weight mean of {', '.join(self.extractors)}; "
              "unquantized floating-point images.", flush=True)

    def _load_reference(self, extractor, reference_paths, reference_identity, reference_on_gpu):
        runtime = self.runtime
        metadata = dict(format="guided_es_features_v1", reference=reference_identity,
                        runtime=runtime.signature(), inception_batch_size=self.inception_batch_size,
                        kernel_block_size=self.kid_kernel_block_size,
                        preprocessing=extractor.preprocessing, **extractor.cache_details())
        key = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()
        # Generation noises, time grid and search settings do not affect real features.
        cache_path = Path("cache/finalized_guided_training/reference_features")/(key+".pt")
        if cache_path.exists():
            cached = torch.load(cache_path, map_location="cpu", weights_only=True)
            if cached["metadata"] != metadata:
                raise ValueError("Reference feature cache metadata mismatch")
            reference, c_ref = cached["features"], cached["c_ref"]
            print(f"Loaded {extractor.name} reference feature cache: {cache_path}", flush=True)
        else:
            reference = self._reference_features(reference_paths, extractor)
            reference_for_kernel = reference.to(runtime.device) if reference_on_gpu else reference
            rows = self._kernel_rows(reference_for_kernel, reference_for_kernel, same=True)
            c_ref = float(rows.sum()/(len(reference)*(len(reference)-1)))
            del rows, reference_for_kernel
            atomic_save(dict(metadata=metadata, features=reference, c_ref=c_ref), cache_path)
            print(f"Saved {extractor.name} reference feature cache: {cache_path}", flush=True)
        if reference.shape != (len(reference_paths), 2048) or not math.isfinite(c_ref):
            raise ValueError("Invalid reference features or kernel constant")
        require_finite(reference, "reference features")
        self.references[extractor.name] = reference.to(runtime.device) if reference_on_gpu else reference
        self.reference_constants[extractor.name] = c_ref
        print(f"Reference features ({extractor.name}): {len(reference):,}, "
              f"stored on {self.references[extractor.name].device} "
              f"({reference.numel()*reference.element_size()/1024**2:.1f} MiB).", flush=True)

    @torch.no_grad()
    def _features(self, images, extractor):
        parts = []
        for start in range(0, len(images), self.inception_batch_size):
            batch = images[start:start+self.inception_batch_size]
            # Each extractor returns an owned result, safe across compiled calls.
            parts.append(extractor(batch))
        return torch.cat(parts)

    def _reference_features(self, paths, extractor):
        parts = []
        loader = DataLoader(ReferenceImages(paths), batch_size=self.inception_batch_size,
                            num_workers=self.num_workers, shuffle=False,
                            pin_memory=self.runtime.device.type == "cuda")
        processed = 0
        for index, images in enumerate(loader):
            parts.append(self._features(images.to(self.runtime.device, non_blocking=True), extractor).cpu())
            processed += len(images)
            if index == 0 or (index+1) % 100 == 0:
                print(f"Reference features: {processed:,}/{len(paths):,}", flush=True)
        return torch.cat(parts)

    @torch.no_grad()
    def _kernel_rows(self, features, reference, same=False):
        block = self.kid_kernel_block_size
        rows = torch.zeros(len(features), device=self.runtime.device, dtype=torch.float64)
        for i in range(0, len(features), block):
            a = features[i:i+block].to(self.runtime.device)
            for j in range(i if same else 0, len(reference), block):
                b = reference[j:j+block].to(self.runtime.device)
                kernel = (a @ b.T/features.shape[1]+1).pow(3)
                if same and i == j:
                    kernel = (kernel+kernel.T)*.5  # Enforce symmetry despite GEMM rounding.
                    kernel.diagonal().zero_()
                rows[i:i+len(a)].add_(kernel.sum(dim=1, dtype=torch.float64))
                if same and i != j:
                    # Reuse symmetry rather than multiply the opposite block again.
                    rows[j:j+len(b)].add_(kernel.sum(dim=0, dtype=torch.float64))
        return rows

    @torch.no_grad()
    def kernel_pair_rows(self, first, second):
        """Both row sums of an off-diagonal kernel rectangle; no dense cache."""
        block = self.kid_kernel_block_size
        left = torch.zeros(len(first), device=self.runtime.device, dtype=torch.float64)
        right = torch.zeros(len(second), device=self.runtime.device, dtype=torch.float64)
        for i in range(0, len(first), block):
            a = first[i:i+block].to(self.runtime.device)
            for j in range(0, len(second), block):
                b = second[j:j+block].to(self.runtime.device)
                kernel = (a @ b.T/first.shape[1]+1).pow(3)
                left[i:i+len(a)].add_(kernel.sum(dim=1, dtype=torch.float64))
                right[j:j+len(b)].add_(kernel.sum(dim=0, dtype=torch.float64))
        return left.cpu(), right.cpu()

    def noise_bank(self, seed, count):
        # Generate the whole bank once: every slice is an exact prefix, rather
        # than depending on random-number behavior for different draw shapes.
        noises = torch.randn(count, 3, 32, 32, device="cpu",
                             generator=torch.Generator().manual_seed(seed))
        if self.runtime.device.type == "cuda":
            noises = noises.pin_memory()
        return NoiseBank(seed, noises.to(self.runtime.device, non_blocking=True))

    @torch.no_grad()
    def preview(self, bank):
        """Fixed-noise visual comparison, without computing features or KID.

        Complete generation tiles preserve the compiled sampler's input shape.
        The caller supplies a bank long enough for the rounded-up grid count.
        """
        count = self.sample_grid_image_count
        if count == 0:
            return None
        tile = self.generation_batch_size
        needed = ((count+tile-1)//tile)*tile
        if len(bank.noises) < needed:
            raise ValueError(f"Preview noise bank requires at least {needed} samples")
        old_mode = self.model.training
        self.model.eval()
        self.runtime.synchronize()
        started = time.monotonic()
        pictures = []
        try:
            for start in range(0, needed, tile):
                with self.runtime.autocast():
                    output = self.generate(bank.noises[start:start+tile]).clone()
                self.generated_images += len(output)
                require_finite(output, "preview images")
                images = (output*.5+.5).clamp(0, 1)
                pictures.append(images[:min(tile, count-start)].cpu())
        finally:
            self.model.train(old_mode)
            self.runtime.synchronize()
            self.generation_seconds += time.monotonic()-started
        return torch.cat(pictures)

    def _flush_timings(self):
        if self._pending_timers:
            self.runtime.synchronize()
            for begin, middle, stages in self._pending_timers:
                self.generation_seconds += begin.elapsed_time(middle)/1000
                for name, stage_begin, stage_end in stages:
                    elapsed = stage_begin.elapsed_time(stage_end)/1000
                    self.feature_seconds += elapsed
                    self.feature_seconds_by_model[name] += elapsed
            self._pending_timers.clear()

    @torch.no_grad()
    def extend(self, history, bank, count):
        """Caller installs the candidate's exact weights before every extension."""
        if history.noise_seed != bank.seed or count < 0 or count > len(bank.noises):
            raise ValueError("Feature history does not match the requested noise bank")
        if count <= history.count:
            return
        if history.features and (set(history.features) != set(self.extractors)
                                 or any(len(value) != history.count for value in history.features.values())):
            raise ValueError("Feature history does not match the active KID pipelines")
        tile = self.generation_batch_size
        if count % tile or history.count % tile:
            raise ValueError("Feature-history boundaries must align with the fixed generation tile")
        rt = self.runtime
        old_mode = self.model.training
        self.model.eval()
        additions = {name: [] for name in self.extractors}
        pictures = []
        try:
            for start in range(history.count, count, tile):
                if rt.device.type == "cuda":
                    begin, middle = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
                    stream = torch.cuda.current_stream(rt.device)
                    begin.record(stream)
                else:
                    started = time.monotonic()
                with rt.autocast():
                    # Own the output before another compiled module is invoked.
                    x = self.generate(bank.noises[start:start+tile]).clone()
                if rt.device.type == "cuda":
                    middle.record(stream)
                else:
                    self.generation_seconds += time.monotonic()-started
                    started = time.monotonic()
                self.generated_images += len(x)
                require_finite(x, "generated images")
                images = (x*.5+.5).clamp(0, 1)
                stages = []
                for name, extractor in self.extractors.items():
                    if rt.device.type == "cuda":
                        stage_begin, stage_end = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
                        stage_begin.record(stream)
                    else:
                        feature_started = time.monotonic()
                    additions[name].append(self._features(images, extractor))
                    if rt.device.type == "cuda":
                        stage_end.record(stream)
                        stages.append((name, stage_begin, stage_end))
                    else:
                        elapsed = time.monotonic()-feature_started
                        self.feature_seconds += elapsed
                        self.feature_seconds_by_model[name] += elapsed
                if rt.device.type == "cuda":
                    self._pending_timers.append((begin, middle, stages))
                if start < self.sample_grid_image_count:
                    pictures.append(images[:self.sample_grid_image_count-start].cpu())
        finally:
            self.model.train(old_mode)
            self._flush_timings()
        new_features = {name: torch.cat(parts) for name, parts in additions.items()}
        for name, features in new_features.items():
            require_finite(features, f"generated {name} Inception features")
        for name, features in new_features.items():
            history.features[name] = (torch.cat([history.features[name], features.to(history.features[name].device)])
                                      if name in history.features else features)
        if pictures:
            new_pictures = torch.cat(pictures)
            history.images = new_pictures if history.images is None else torch.cat([history.images, new_pictures])

    @torch.no_grad()
    def score(self, history, count, *, start=0):
        """Exact KID on a cached feature slice, including all cross-block pairs.

        Cheap kernels are recomputed for new slices. Generated images/features
        are never recomputed. This method never averages subset KIDs. The search
        controller can also pool cached features and partition their kernel rows
        for fold diagnostics, without generating any more images.
        """
        if count < 3 or start < 0 or start+count > history.count:
            raise ValueError("KID needs an available feature slice with at least three samples")
        key = (start, count)
        if key in history.scores:
            return history.scores[key]
        self.runtime.synchronize()
        started = time.monotonic()
        weight = 1./len(self.extractors)
        self_rows = torch.zeros(count, dtype=torch.float64, device=self.runtime.device)
        cross_rows = torch.zeros_like(self_rows)
        component_scores = {}
        component_self_rows = {}
        component_cross_rows = {}
        for name in self.extractors:
            features = history.features[name][start:start+count]
            reference = self.references[name]
            component_self = self._kernel_rows(features, features, same=True)
            component_cross = self._kernel_rows(features, reference)/len(reference)
            value = float(component_self.sum()/(count*(count-1))
                          - 2*component_cross.mean() + self.reference_constants[name])
            if not math.isfinite(value):
                raise FloatingPointError(f"Nonfinite {name} KID")
            component_scores[name] = value
            component_self_rows[name] = component_self.cpu()
            component_cross_rows[name] = component_cross.cpu()
            # Averages of per-image contributions preserve inter-metric covariance
            # in both paired jackknife differences and quadratic-fit uncertainty.
            self_rows.add_(component_self, alpha=weight)
            cross_rows.add_(component_cross, alpha=weight)
        score = sum(component_scores.values())*weight
        if not math.isfinite(score):
            raise FloatingPointError("Nonfinite KID")
        result = KIDResult(score, self_rows.cpu(), cross_rows.cpu(), history.noise_seed, start, count,
                           component_scores, component_self_rows, component_cross_rows)
        history.scores[key] = result
        self.kernel_seconds += time.monotonic()-started
        self.score_evaluations += 1
        return result
