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
        time_sampler=TimeSampler(method=TimeSamplingType.DETERMINISTIC_UNIFORM_BOUNDARY, lowest_t=0.003375, highest_t=0.999),
        output_root="outputs/examples",
        include_cutoff_in_name=True,
    )


if __name__ == "__main__":
    folders = main()
