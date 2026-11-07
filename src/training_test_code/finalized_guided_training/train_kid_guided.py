"""Three-axis velocity-gradient-guided KID refinement."""
import math
from pathlib import Path
import secrets
import time

import torch

from src.modules.velocity_model import VelocityModel
from src.utilities.print_helper import PrintFields, PrintStatus, PrintSummary, PrintTable
from .guidance import RGBGuidance
from .gradient_history import GradientSecondMoment
from .io_utils import RunLog, save_grid, save_model
from .kid_evaluator import FeatureHistory, KIDEvaluator
from .parameter_space import ParameterSpace, orthonormal_basis, vector_dot, vector_norm
from .runtime import Runtime
from .scaling import AxisScaleController
from .search import NoiseSeeds, RadiusController, search_round


def _basis_overlap(previous, current):
    if not previous or not current:
        return {}
    gram = torch.tensor([[vector_dot(a, b) for b in current] for a in previous], dtype=torch.float64)
    cosines = torch.linalg.svdvals(gram).clamp(0, 1)
    return dict(cross_gram=gram.tolist(), principal_cosines=cosines.tolist(),
                principal_angles_degrees=(torch.acos(cosines)*180/math.pi).tolist(),
                mean_squared_overlap=float(cosines.square().mean()),
                previous_rank=len(previous), current_rank=len(current))


def TrainFinalizedGuided(
        model_path: str,
        cache_path: str = "cache/training_pairs/cifar_fixed_pairs.pt",
        image_validation_folder: str = "data/cifar_official/fid_reference_flat",
        total_rounds: int = 500,
        num_workers: int = 8,
        batch_size_per_time: int = 2,
        num_times: int = 128,
        grad_accum_steps: int = 5,
        gradient_min_time: float = 0.075,
        gradient_max_time: float = 0.999,
        num_rgb_guides: int = 4,
        removed_gradient_fraction: float = 0.2,
        basis_residual_tolerance: float = 1e-3,
        preconditioned_axis: bool = True,
        second_moment_decay: float = 0.9,
        second_moment_damping: float = 0.05,
        initial_probe_relative_radius: float = 2e-5,
        initial_step_relative_radius: float = 2e-5,
        min_probe_relative_radius: float = 1e-6,
        max_probe_relative_radius: float = 1e-3,
        min_step_relative_radius: float = 1e-6,
        max_step_relative_radius: float = 1e-3,
        initial_mean_axis_scale: float = 0.25,
        initial_residual_axis_scale: float = 1.5,
        initial_remaining_axis_scale: float = 4.0,
        target_probe_kid_change: float = 5e-6,
        axis_scale_confidence_z: float = 2.0,
        min_axis_scale: float = 0.05,
        max_axis_scale: float = 4.0,
        max_axis_scale_change: float = 2.0,
        sample_count: int = 1024*8,
        validation_count: int = 1024*8,
        data_folds: int = 4,
        max_candidate_attempts: int = 5,
        max_refinement_actions: int = 2,
        max_exploratory_attempts: int = 1,
        fold_regression_penalty: float = 1.0,
        fold_regression_z: float = 1.0,
        proposal_penalty: float = 1.0,
        acceptance_confidence_z: float = 0.3,
        component_regression_tolerance: float = 2e-6,
        component_regression_z: float = 1.0,
        min_kid_improvement: float = 0.0,
        minimum_worthwhile_gain: float = 1e-6,
        candidate_minimum_separation: float = 0.2,
        fit_ridge: float = 1e-4,
        cross_term_ridge_multiplier: float = 10.0,
        covariance_floor: float = 1e-5,
        repeatability_tolerance: float = 1e-6,
        generation_steps: int = 50,
        generation_min_time: float = 0.10,
        generation_max_time: float = 0.999,
        generation_batch_size: int = 1024,
        inception_batch_size: int = 512,
        kid_kernel_block_size: int = 2048,
        reference_on_gpu: bool = True,
        sample_grid_image_count: int = 64,
        use_two_kid_models: bool = True,
        seed: int | None = None,
        output_root: str = "outputs/finalized_guided_training",
    ):
    """Forward-only KID search in up to three orthogonal RGB-derived axes.

    Each location uses sample_count shared noises. Disjoint data_folds guide
    candidate ranking; overlapping delete-fold fits estimate sensitivity.
    A full quadratic includes cross terms from the initial stencil. Uncertainty
    uses the larger paired or delete-fold variance, without adding them.
    fold_regression_penalty is a separate soft risk preference against confident
    fold degradation, never an extra standard error or an acceptance veto.

    removed_gradient_fraction removes the largest absolute mean-gradient
    coordinates by COUNT from the tail input. The current preconditioned mean
    uses a bias-corrected history of squared gradients, not momentum.
    Basis, anchor, scales and sampler stay fixed throughout each generation.

    At most max_candidate_attempts full/half candidate cycles and
    max_refinement_actions targeted measurements are allowed. Rejected fresh
    data may guide salvage but cannot validate another model. Every finalist
    uses unused paired validation_count noises with the same fixed z.
    z=.3 is a permissive optimization margin, not a confidence guarantee.
    Accepted checkpoints atomically overwrite model_path. No resume from logs.
    """
    if type(total_rounds) is not int or total_rounds < 1:
        raise ValueError("total_rounds must be a positive integer")
    if type(num_workers) is not int or num_workers < 0:
        raise ValueError("num_workers must be a nonnegative integer")
    if type(batch_size_per_time) is not int or batch_size_per_time < 1:
        raise ValueError("batch_size_per_time must be a positive integer")
    if type(num_times) is not int or num_times < 1:
        raise ValueError("num_times must be a positive integer")
    if type(grad_accum_steps) is not int or grad_accum_steps < 1:
        raise ValueError("grad_accum_steps must be a positive integer")
    if type(num_rgb_guides) is not int or num_rgb_guides < 2:
        raise ValueError("num_rgb_guides must be an integer of at least two")
    if not math.isfinite(removed_gradient_fraction) or not 0 < removed_gradient_fraction <= 1:
        raise ValueError("removed_gradient_fraction must be in (0, 1]")
    if not math.isfinite(basis_residual_tolerance) or not 0 < basis_residual_tolerance < 1:
        raise ValueError("basis_residual_tolerance must be in (0, 1)")
    if type(preconditioned_axis) is not bool:
        raise ValueError("preconditioned_axis must be a bool")
    history = GradientSecondMoment(second_moment_decay, second_moment_damping)
    if not 0 < gradient_min_time < gradient_max_time < 1:
        raise ValueError("RGB gradient times must satisfy 0 < min < max < 1")
    if not 0 < generation_min_time < generation_max_time < 1:
        raise ValueError("Generation times must satisfy 0 < min < max < 1")
    if (not 0 < min_probe_relative_radius <= initial_probe_relative_radius <= max_probe_relative_radius
            or not math.isfinite(max_probe_relative_radius)):
        raise ValueError("Invalid probe radius bounds")
    if (not 0 < min_step_relative_radius <= initial_step_relative_radius <= max_step_relative_radius
            or not math.isfinite(max_step_relative_radius)):
        raise ValueError("Invalid step radius bounds")
    if min_probe_relative_radius > max_step_relative_radius:
        raise ValueError("Minimum probe radius must fit within the maximum candidate radius")
    if not 0 < min_axis_scale <= 1 <= max_axis_scale or not math.isfinite(max_axis_scale):
        raise ValueError("Axis factor bounds must be finite, positive, and include one")
    if not math.isfinite(initial_mean_axis_scale) or not min_axis_scale <= initial_mean_axis_scale <= max_axis_scale:
        raise ValueError("initial_mean_axis_scale must lie within the axis factor bounds")
    if not math.isfinite(initial_residual_axis_scale) or not min_axis_scale <= initial_residual_axis_scale <= max_axis_scale:
        raise ValueError("initial_residual_axis_scale must lie within the axis factor bounds")
    if not math.isfinite(initial_remaining_axis_scale) or not min_axis_scale <= initial_remaining_axis_scale <= max_axis_scale:
        raise ValueError("initial_remaining_axis_scale must lie within the axis factor bounds")
    if type(sample_count) is not int or sample_count < 3:
        raise ValueError("sample_count must be an integer of at least three")
    if type(validation_count) is not int or validation_count < 3:
        raise ValueError("validation_count must be an integer of at least three")
    if type(data_folds) is not int or not 3 <= data_folds <= 8:
        raise ValueError("data_folds must be an integer from three to eight")
    if (sample_count % data_folds or validation_count % data_folds
            or min(sample_count, validation_count)//data_folds < 3):
        raise ValueError("Search and confirmation banks must split into equal folds of at least three images")
    if type(max_candidate_attempts) is not int or max_candidate_attempts < 1:
        raise ValueError("max_candidate_attempts must be a positive integer")
    if type(max_refinement_actions) is not int or max_refinement_actions < 0:
        raise ValueError("max_refinement_actions must be a nonnegative integer")
    if type(max_exploratory_attempts) is not int or not 0 <= max_exploratory_attempts <= max_candidate_attempts:
        raise ValueError("max_exploratory_attempts must lie between zero and max_candidate_attempts")
    if not math.isfinite(fold_regression_penalty) or fold_regression_penalty < 0:
        raise ValueError("fold_regression_penalty must be finite and nonnegative")
    if not math.isfinite(fold_regression_z) or fold_regression_z < 0:
        raise ValueError("fold_regression_z must be finite and nonnegative")
    if type(generation_steps) is not int or generation_steps < 2:
        raise ValueError("generation_steps must be an integer of at least two")
    if type(generation_batch_size) is not int or generation_batch_size < 1:
        raise ValueError("generation_batch_size must be a positive integer")
    if type(inception_batch_size) is not int or inception_batch_size < 1:
        raise ValueError("inception_batch_size must be a positive integer")
    if type(kid_kernel_block_size) is not int or kid_kernel_block_size < 1:
        raise ValueError("kid_kernel_block_size must be a positive integer")
    if type(sample_grid_image_count) is not int or sample_grid_image_count < 0:
        raise ValueError("sample_grid_image_count must be a nonnegative integer")
    tile = min(generation_batch_size, sample_count)
    if sample_count % tile or validation_count % tile:
        raise ValueError("Search and confirmation image counts must be multiples of the fixed generation tile")
    if not math.isfinite(proposal_penalty) or proposal_penalty < 0:
        raise ValueError("proposal_penalty must be finite and nonnegative")
    if not math.isfinite(acceptance_confidence_z) or acceptance_confidence_z < 0:
        raise ValueError("acceptance_confidence_z must be finite and nonnegative")
    if not math.isfinite(component_regression_tolerance) or component_regression_tolerance < 0:
        raise ValueError("component_regression_tolerance must be finite and nonnegative")
    if not math.isfinite(component_regression_z) or component_regression_z < 0:
        raise ValueError("component_regression_z must be finite and nonnegative")
    if not math.isfinite(min_kid_improvement) or min_kid_improvement < 0:
        raise ValueError("min_kid_improvement must be finite and nonnegative")
    if not math.isfinite(minimum_worthwhile_gain) or minimum_worthwhile_gain <= 0:
        raise ValueError("minimum_worthwhile_gain must be finite and positive")
    if not math.isfinite(candidate_minimum_separation) or not 0 < candidate_minimum_separation < 1:
        raise ValueError("candidate_minimum_separation must be in (0, 1)")
    if not math.isfinite(fit_ridge) or fit_ridge < 0:
        raise ValueError("fit_ridge must be finite and nonnegative")
    if not math.isfinite(cross_term_ridge_multiplier) or cross_term_ridge_multiplier < 1:
        raise ValueError("cross_term_ridge_multiplier must be finite and at least one")
    if not math.isfinite(covariance_floor) or not 0 < covariance_floor <= 1:
        raise ValueError("covariance_floor must be in (0, 1]")
    if not math.isfinite(repeatability_tolerance) or repeatability_tolerance < 0:
        raise ValueError("repeatability_tolerance must be finite and nonnegative")
    if seed is None:
        seed = secrets.randbits(60)
    last_start_seed = 2**63-2-1-int(sample_grid_image_count > 0)-total_rounds*(1+max_candidate_attempts+max_refinement_actions)
    if type(seed) is not int or not 0 <= seed <= last_start_seed:
        raise ValueError("seed must be nonnegative and leave room for this run's unique noise banks")
    if Path(model_path).resolve() == Path(cache_path).resolve():
        raise ValueError("Model and training-pair cache paths must differ")
    axis_controller = AxisScaleController(target_cost=target_probe_kid_change,
        curvature_z=axis_scale_confidence_z, min_factor=min_axis_scale, max_factor=max_axis_scale,
        max_change=max_axis_scale_change, initial_mean_factor=initial_mean_axis_scale,
        initial_residual_factor=initial_residual_axis_scale, initial_remaining_factor=initial_remaining_axis_scale)

    # Explicit settings are for reporting only. Function arguments directly
    # control every call below; there is no settings/override dispatch layer.
    settings = dict(model_path=str(model_path), cache_path=str(cache_path),
        image_validation_folder=str(image_validation_folder), total_rounds=total_rounds,
        num_workers=num_workers, batch_size_per_time=batch_size_per_time, num_times=num_times,
        grad_accum_steps=grad_accum_steps, gradient_min_time=gradient_min_time,
        gradient_max_time=gradient_max_time, num_rgb_guides=num_rgb_guides,
        removed_gradient_fraction=removed_gradient_fraction, basis_residual_tolerance=basis_residual_tolerance,
        preconditioned_axis=preconditioned_axis, second_moment_decay=second_moment_decay,
        second_moment_damping=second_moment_damping,
        initial_probe_relative_radius=initial_probe_relative_radius,
        initial_step_relative_radius=initial_step_relative_radius,
        min_probe_relative_radius=min_probe_relative_radius, max_probe_relative_radius=max_probe_relative_radius,
        min_step_relative_radius=min_step_relative_radius, max_step_relative_radius=max_step_relative_radius,
        initial_mean_axis_scale=initial_mean_axis_scale, initial_residual_axis_scale=initial_residual_axis_scale,
        initial_remaining_axis_scale=initial_remaining_axis_scale, target_probe_kid_change=target_probe_kid_change,
        axis_scale_confidence_z=axis_scale_confidence_z, min_axis_scale=min_axis_scale,
        max_axis_scale=max_axis_scale, max_axis_scale_change=max_axis_scale_change,
        sample_count=sample_count, validation_count=validation_count,
        data_folds=data_folds, max_candidate_attempts=max_candidate_attempts,
        max_refinement_actions=max_refinement_actions, max_exploratory_attempts=max_exploratory_attempts,
        fold_regression_penalty=fold_regression_penalty, fold_regression_z=fold_regression_z,
        proposal_penalty=proposal_penalty,
        acceptance_confidence_z=acceptance_confidence_z, min_kid_improvement=min_kid_improvement,
        component_regression_tolerance=component_regression_tolerance,
        component_regression_z=component_regression_z,
        minimum_worthwhile_gain=minimum_worthwhile_gain, candidate_minimum_separation=candidate_minimum_separation,
        fit_ridge=fit_ridge, cross_term_ridge_multiplier=cross_term_ridge_multiplier, covariance_floor=covariance_floor, repeatability_tolerance=repeatability_tolerance,
        generation_steps=generation_steps, generation_min_time=generation_min_time,
        generation_max_time=generation_max_time, generation_batch_size=generation_batch_size,
        inception_batch_size=inception_batch_size, kid_kernel_block_size=kid_kernel_block_size,
        reference_on_gpu=reference_on_gpu, sample_grid_image_count=sample_grid_image_count,
        use_two_kid_models=use_two_kid_models, seed=seed, output_root=str(output_root))
    runtime = Runtime()
    torch.manual_seed(seed)
    if runtime.device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    model = VelocityModel.load(model_path, compile=False).to(runtime.device, dtype=runtime.model_dtype)
    model.eval()
    space = ParameterSpace(model)
    reference_norm = vector_norm(space.vector())
    if not math.isfinite(reference_norm) or reference_norm <= 0:
        raise ValueError("Initial trainable parameter norm must be finite and positive")
    radii = RadiusController(reference_norm, initial_probe_relative_radius, initial_step_relative_radius,
                             min_probe_relative_radius, max_probe_relative_radius,
                             min_step_relative_radius, max_step_relative_radius)
    log = RunLog(model_path, seed, runtime, output_root=output_root, configuration=settings,
                 schema_version=5, search="full_quadratic_fold_consensus_targeted_salvage",
                 acceptance="fixed_z_unused_paired_confirmation", reference_conditioning="fixed_real_dataset")
    PrintSummary("Finalized RGB-guided KID training")
    PrintFields([("Parameters", f"{space.size:,}"), ("Largest gradient fraction removed", removed_gradient_fraction),
                 ("Initial mean / residual / remaining probe factors",
                  f"{initial_mean_axis_scale:g} / {initial_residual_axis_scale:g} / {initial_remaining_axis_scale:g}"),
                 ("Target paired symmetric probe cost", target_probe_kid_change),
                 ("Search / fresh images per model", f"{sample_count:,} / {validation_count:,}"),
                 ("Maximum candidate attempts", max_candidate_attempts),
                 ("Data folds / images per fold", f"{data_folds} / {sample_count//data_folds:,}"),
                 ("Maximum targeted refinement actions", max_refinement_actions),
                 ("Maximum exploratory attempts", max_exploratory_attempts),
                 ("Proposal uncertainty penalty", proposal_penalty),
                 ("Preconditioned second axis", preconditioned_axis),
                 ("Acceptance z (all attempts)", acceptance_confidence_z), ("Reports", log.folder)])
    metric = KIDEvaluator(model=model, runtime=runtime,
        image_validation_folder=image_validation_folder, num_workers=num_workers,
        generation_steps=generation_steps, generation_min_time=generation_min_time,
        generation_max_time=generation_max_time, generation_batch_size=generation_batch_size,
        inception_batch_size=inception_batch_size, kid_kernel_block_size=kid_kernel_block_size,
        reference_on_gpu=reference_on_gpu, sample_grid_image_count=sample_grid_image_count,
        sample_count=sample_count, use_two_kid_models=use_two_kid_models)
    guidance = RGBGuidance(model, space, runtime, cache_path=cache_path, num_workers=num_workers,
        num_times=num_times, batch_size_per_time=batch_size_per_time, grad_accum_steps=grad_accum_steps,
        gradient_min_time=gradient_min_time, gradient_max_time=gradient_max_time,
        num_rgb_guides=num_rgb_guides, seed=seed)
    noise_seeds = NoiseSeeds(seed)

    # Warm up and check repeatability on identical full tiles, not on an extra
    # 13-location landscape. These observations never enter a training fit.
    repeat_count = max(tile, math.ceil(3/tile)*tile)
    repeat_bank = metric.noise_bank(noise_seeds.next(), repeat_count)
    first, second = FeatureHistory(repeat_bank.seed), FeatureHistory(repeat_bank.seed)
    PrintStatus("Startup repeatability check; includes compilation warmup")
    metric.extend(first, repeat_bank, repeat_count)
    first_score = metric.score(first, repeat_count)
    metric.extend(second, repeat_bank, repeat_count)
    second_score = metric.score(second, repeat_count)
    differences = {name: abs(first_score.component_scores[name]-second_score.component_scores[name])
                   for name in first_score.component_scores}
    log.event("repeatability", image_count=repeat_count, differences=differences,
              tolerance=repeatability_tolerance)
    if max(differences.values()) > repeatability_tolerance:
        raise RuntimeError("Identical-noise KID repeatability exceeded tolerance; see repeatability event")
    del first, second, first_score, second_score, repeat_bank

    fixed_grid_bank = None
    if sample_grid_image_count:
        preview_count = math.ceil(sample_grid_image_count/tile)*tile
        fixed_grid_bank = metric.noise_bank(noise_seeds.next(), preview_count)
        save_grid(metric.preview(fixed_grid_bank), log.image_folder/"initial_fixed_noise.png")
        log.event("fixed_grid", noise_seed=fixed_grid_bank.seed, image_count=sample_grid_image_count)

    accepted_count, previous_basis = 0, None
    for round_index in range(1, total_rounds+1):
        started = time.monotonic()
        before_images = metric.generated_images
        generation_before, feature_before, kernel_before = (metric.generation_seconds,
                                                           metric.feature_seconds, metric.kernel_seconds)
        if runtime.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(runtime.device)
        PrintSummary(f"Round {round_index}/{total_rounds}; accepted {accepted_count}")
        log.event("round_start", round=round_index, accepted_total=accepted_count,
                  probe_relative_radius=radii.probe, step_relative_radius=radii.step)
        space.begin_round()
        result, committed = None, False
        info = dict(round=round_index, accepted_basis_coordinates=[])
        try:
            guides = guidance.collect()
            PrintTable(["Guide", "RGB MSE", "Gradient norm"],
                       [(index+1, f"{loss:.7f}", f"{norm:.6g}")
                        for index, (loss, norm) in enumerate(zip(guides.losses, guides.norms))])
            second_direction = None
            history_info = None
            if preconditioned_axis:
                second_direction, history_info = history.direction(guides.directions, guides.norms)
            basis, basis_info = orthonormal_basis(guides.directions, guides.norms,
                removed_gradient_fraction=removed_gradient_fraction, tolerance=basis_residual_tolerance,
                second_direction=second_direction)
            if history_info is not None:
                basis_info["second_moment"] = history_info
            del second_direction
            overlap = _basis_overlap(previous_basis, basis)
            previous_basis = basis
            axis_labels = basis_info["axis_labels"]
            factors = axis_controller.factors_for(axis_labels)
            base_probe_radius = min(radii.probe, radii.max_probe, radii.max_step)*reference_norm
            coordinate_scales = (factors*base_probe_radius).clamp(
                min=radii.min_probe*reference_norm,
                max=min(radii.max_probe, radii.max_step)*reference_norm)
            log.event("basis", round=round_index, basis_rank=len(basis), basis=basis_info,
                      basis_overlap=overlap, rgb_losses=guides.losses, rgb_norms=guides.norms,
                      rgb_agreement=guides.agreement(), coordinate_scales=coordinate_scales.tolist(),
                      base_probe_radius=base_probe_radius)
            PrintFields([("Search dimensions", len(basis)),
                         ("Removed / remaining gradient energy",
                          f"{basis_info['removed_energy_fraction']:.6g} / {basis_info['remaining_energy_fraction']:.6g}"),
                         ("Remaining norm surviving projection", math.sqrt(basis_info["third_projection_retained_fraction"])),
                         ("Physical probe scales", coordinate_scales.tolist())])
            if not basis:
                info["reason"] = "No nondegenerate RGB search directions"
            else:
                bank = metric.noise_bank(noise_seeds.next(), sample_count)
                result = search_round(space, metric, basis, bank, radii, log, round_index=round_index,
                    coordinate_scales=coordinate_scales, axis_labels=axis_labels,
                    sample_count=sample_count, validation_count=validation_count,
                    data_folds=data_folds, max_refinement_actions=max_refinement_actions,
                    max_candidate_attempts=max_candidate_attempts, proposal_penalty=proposal_penalty,
                    max_exploratory_attempts=max_exploratory_attempts,
                    fold_regression_penalty=fold_regression_penalty, fold_regression_z=fold_regression_z,
                    acceptance_confidence_z=acceptance_confidence_z, min_kid_improvement=min_kid_improvement,
                    component_regression_tolerance=component_regression_tolerance,
                    component_regression_z=component_regression_z,
                    minimum_worthwhile_gain=minimum_worthwhile_gain,
                    candidate_minimum_separation=candidate_minimum_separation,
                    fit_ridge=fit_ridge, cross_term_ridge_multiplier=cross_term_ridge_multiplier,
                    covariance_floor=covariance_floor, noise_seeds=noise_seeds)
                del bank
                info.update(result.diagnostics)
                if result.accepted:
                    space.assign(result.displacement)
                    # Render before commit: failures here leave the checkpoint
                    # and the in-memory incumbent at their previous values.
                    if fixed_grid_bank is not None:
                        save_grid(metric.preview(fixed_grid_bank),
                                  log.image_folder/f"accepted_{accepted_count+1:05d}_fixed_noise.png")
                    save_model(model, model_path)
                    committed = True
                    accepted_count += 1
                    dot, cosine = guides.step_alignment(result.displacement)
                    info.update(rgb_gradient_dot_step=dot, rgb_descent_cosine=cosine)
        except (FloatingPointError, torch.linalg.LinAlgError) as error:
            # A numerical anomaly cannot commit partially evaluated weights.
            result = None
            info.update(reason=str(error), outcome="numerical_rejection")
            log.event("numerical_rejection", round=round_index, reason=str(error))
        finally:
            if not committed:
                space.assign()
            model.zero_grad(set_to_none=True)
            model.eval()
        # Axis calibration uses the ORIGINAL radius that generated this data,
        # before any global radius changes. It affects the next generation only.
        if result is not None and result.diagnostics.get("axis_probe_statistics"):
            calibration = axis_controller.update(result.diagnostics["axis_probe_statistics"],
                                                  result.diagnostics["base_probe_radius"])
            log.event("axis_calibration", round=round_index, axes=calibration)
            PrintTable(["Axis", "Even cost", "Paired SE", "Factor now", "Next factor", "Decision"],
                       [(entry["axis_label"], f"{entry['cost']:+.3g}", f"{entry['cost_se']:.3g}",
                         f"{entry['current_factor']:.3g}", f"{entry['next_factor']:.3g}", entry["reason"])
                        for entry in calibration])
        adaptation = radii.adapt(result)
        info.update(accepted=committed, accepted_total=accepted_count, seconds=time.monotonic()-started,
                    generated_images=metric.generated_images-before_images,
                    generation_seconds=metric.generation_seconds-generation_before,
                    inception_seconds=metric.feature_seconds-feature_before,
                    kernel_seconds=metric.kernel_seconds-kernel_before,
                    next_probe_relative_radius=radii.probe, next_step_relative_radius=radii.step,
                    radius_reason=adaptation)
        if runtime.device.type == "cuda":
            info.update(peak_allocated_gib=torch.cuda.max_memory_allocated(runtime.device)/1024**3,
                        peak_reserved_gib=torch.cuda.max_memory_reserved(runtime.device)/1024**3)
        log.event("round_complete", **info)
        PrintStatus(info["reason"], "success" if committed else "warning")
        PrintFields([("Generated images", info["generated_images"]),
                     ("Round seconds", f"{info['seconds']:.1f}"),
                     ("Next radius decision", adaptation), ("Reports", log.folder)])
    PrintStatus(f"Finished: {accepted_count}/{total_rounds} accepted updates. Logs: {log.folder}", "success")
    return str(model_path)


def TrainTheModel(model_path: str = "saved_models/C_Kai_fp.pth", computer_string: str = "l40s"):
    """Training settings for an already initialized GPU_CONFIG."""
    generation_batch_size: int = 2048
    inception_batch_size: int = 512
    num_times: int = 64
    batch_size_per_time: int = 4
    grad_accum_steps: int = 5
    if computer_string == "laptop_8gb":
        PrintStatus("8GB mode chosen.")
        generation_batch_size = 256
        inception_batch_size = 64
        num_times = 16
        grad_accum_steps = 4
    elif computer_string == "l40s":
        PrintStatus("L40S mode chosen.")
    elif computer_string == "cluster_40gb":
        PrintStatus("40GB mode chosen.")
    else:
        raise ValueError("computer_string must be l40s, cluster_40gb or laptop_8gb")
    return TrainFinalizedGuided(
        model_path=model_path,
        generation_batch_size=generation_batch_size,
        inception_batch_size=inception_batch_size,
        num_times=num_times,
        batch_size_per_time=batch_size_per_time,
        grad_accum_steps=grad_accum_steps,
    )
