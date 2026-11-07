# Basis-Parameterized Networks

Code and checkpoints for the flow-matching experiment with Legendre-parameterized networks. Each raw trainable tensor is represented by four shifted Legendre terms. At each time, their learned coefficients produce the effective tensors of a U-Net, without a separate time embedding. The reported architecture contains 24,391,948 learned coefficients and 6,097,987 effective parameter values.

## Environment

The code targets Python 3.14+. Use a compatible PyTorch and torchvision installation with CUDA support for GPU runs. Direct dependencies are listed in [requirements.txt](requirements.txt) without version pins. Packages were updated during development, so this file declares the dependencies rather than reconstructing one fixed historical environment.

The differentiable TensorFlow-compatible Inception adapter was written against `torch-fidelity` 0.4.0. Metric reports record the evaluator package versions. Dataset preparation and the metric extractors download upstream data or weights when these are not cached.

Use the repository root, containing `src`, as the working directory. Initialize `GPU_CONFIG` before constructing models or time samplers. It controls the device, precision and compilation settings centrally.

## Released checkpoints

All three checkpoints have the same architecture. The architecture configuration is stored with the weights.

| File | Training stage | Euler evaluations | Lowest positive evaluation time |
|---|---|---:|---:|
| `Checkpoint_A.pth` | Velocity training and forward-only feature-based refinement | 40 | 0.12 |
| `Checkpoint_B.pth` | Further differentiable KID refinement, mainly with TensorFlow-compatible features | 32 | 0.003375 |
| `Checkpoint_C.pth` | Further differentiable KID refinement, with greater emphasis on clean KID | 40 | 0.001 |

The refinement histories of B and C partly overlap. These are checkpoints of one architecture, rather than independently trained models.

Sampling starts from independent standard Gaussian noise at time 1. The boundary-uniform sampler includes equally spaced interior times from 0.999 down to the listed lower time. The final Euler step reaches time 0, without evaluating the model there.

## Generate images

Call the functions from a Python entry point. This example generates 128 images with Checkpoint B:

```python
from src.core.enums import HardwareProfile
from src.core.globals import GPU_CONFIG
from src.training_helpers.time_sampler import TimeSampler, TimeSamplingType
from src.training_test_code.cifar_test.s7_inference import MakeInference


def main():
    GPU_CONFIG.Initialize(profile=HardwareProfile.PRECISE)

    return MakeInference(
        model_name="Checkpoint_B",
        model_path="Checkpoint_B.pth",
        seeds=[2000],
        steps=32,
        batch_size=128,
        num_samples=128,
        time_sampler=TimeSampler(
            method=TimeSamplingType.DETERMINISTIC_UNIFORM_BOUNDARY,
            lowest_t=0.003375,
            highest_t=0.999,
        ),
        output_root="outputs/examples",
        include_cutoff_in_name=True,
    )


if __name__ == "__main__":
    folders = main()
```

`MakeInference` returns the generated folder paths. Generation replaces an existing folder with the same output name, so use a different output root when preserving an earlier run. For another checkpoint, change the file, model name and sampling settings together. Batch size can be adjusted to available GPU memory.

The reported evaluations use six folders of 50,000 images, with seeds 2000–2005 and the settings in the table. The small example above is not a benchmark run. Exact pixels can also depend on batch size, hardware and numerical settings, even with the same seed.

## Dataset and metrics

`src/training_test_code/cifar_test/s1_download.py` provides `download_and_save_cifar(data_root="./data")`. It extracts the 50,000 CIFAR-10 training images to `data/cifar_official/fid_reference_flat`, without class labels. Training maps RGB bytes to floating-point values in [-1, 1]. Evaluation uses the original training images as its real reference.

After generating the six full evaluation folders, pass the returned paths to:

```python
from src.training_test_code.cifar_test.s8_metrics_judge import MakeJudgement

MakeJudgement(
    output_csv="Checkpoint_B_metrics",
    gen_folders=folders,
    ref_folder="data/cifar_official/fid_reference_flat",
    take_first_50k=True,
)
```

Call this in the initialized entry point. It writes per-folder JSON/text reports and `outputs/benchmarks/Checkpoint_B_metrics.csv`, including means and sample standard deviations across folders. KID columns ending in `_x1k` are multiplied by 1,000. KID subset variation and IS split variation are separate from variation across generation seeds.

`s4_prepare_metrics.py` implements the named evaluator protocols. The additional torchvision KID uses an unbiased, all-pairs cubic-kernel estimator with ImageNet Inception-v3 features and the preprocessing recorded in each report. It is a custom evaluation variant, separate from the native `pytorch-fid` FID computation. Metric and reference caches are stored under `cache/metrics`.

## Training code

Training involved repeated experiments and resumed runs with changing settings, including time ranges, pair construction and refinement options. A complete run-by-run history was not retained. The files below provide the training procedures used during development, rather than a single script that reproduces the full history of a released checkpoint. All reported training used only the 50,000 CIFAR-10 training images.

| Location under `src/training_test_code/` | Callable | Purpose |
|---|---|---|
| `cifar_test/s6_train_model_d.py` | `TrainVelocityModel` | Velocity MSE with independent Gaussian noise (`noise_method="Noise_G"`) or minibatch Hungarian assignment (`"HungarianNoise_G"`) |
| `cifar_test/a1_pair_images.py` | `CreateInitialPairs`, `PrepareDataset` | Create and improve cached image–noise assignments |
| `cifar_test/a2_fixed_pair_training.py` | `TrainVelocityModelFixedPairs` | Velocity MSE using one pair cache, supplied as `cache_path` |
| `cifar_test/b2_fixed_pair_training.py` | `TrainVelocityModelFixedPairs` | Velocity MSE using multiple caches identified by `seeds` |
| `finalized_guided_training/train_kid_guided.py` | `TrainFinalizedGuided` | Forward-only KID refinement in a search space derived from velocity-loss gradients |
| `differentiable_kid_training/train_kid_guided.py` | `TrainDifferentiableKID` | KID refinement differentiated through the sampler and frozen feature extractor |

Import the chosen function and pass its parameters directly. Keep calls that use multiprocessing inside an `if __name__ == "__main__":` guard. The `TrainTheModel()` presets retain settings from particular development runs. They are not the final evaluation settings or a recovered training recipe.

The velocity trainers and forward-only refinement load an existing model and can overwrite that model file. Train a working copy when preserving a released checkpoint. Differentiable refinement writes to its separate `output_folder`; `use_clean_fid` selects between clean and TensorFlow-compatible feature paths.

For multiple pair caches, first call `CreateInitialPairs` with the desired seed and cache path, then `PrepareDataset` with that existing path. `PrepareDataset` otherwise creates a missing cache with its default seed of 9999. The multi-cache trainers expect files named `cache/training_pairs/cifar_fixed_pairs_<seed>.pt`. Dataset images, pair caches, generated images and optimizer states are created locally.

## Model implementation

- `src/math/time_interpolation.py`: shifted Legendre basis values and coefficient contraction.
- `src/layers/`: the layers used by the reported architecture.
- `src/modules/layer_block.py`: layer blocks and residual projections.
- `src/modules/velocity_model.py`: network construction, forward evaluation, checkpoint loading and saving.
- `src/core/velocity_model_config.py`: architecture configuration types, used by `VelocityModel.create(config)` for new models.
- `src/core/globals.py`: central GPU configuration.

This release focuses on the methods described in the paper. Unused experimental losses, layers and UI code are omitted.
