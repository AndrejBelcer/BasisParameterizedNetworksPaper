"""Full quadratics fitted jointly to paired, correlated KID observations.

Coefficient uncertainty is conditional on the measured design and estimated
jackknife covariance; it does not remove adaptive selection or model bias.
Only tiny CPU coordinate problems use autograd. The generator never does.
"""
from dataclasses import dataclass, field
import math

import torch

from .parameter_space import require_finite


def coefficient_count(dimension, diagonal=False):
    return 2*dimension if diagonal else dimension+dimension*(dimension+1)//2


def axis_scale_vector(scale, dimension):
    scales = torch.as_tensor(scale, dtype=torch.float64, device="cpu")
    if scales.ndim == 0:
        scales = scales.repeat(dimension)
    if scales.shape != (dimension,) or not bool(torch.isfinite(scales).all()) or bool((scales <= 0).any()):
        raise ValueError("Coordinate scales must be positive finite scalars or one value per axis")
    return scales


def coordinate_design(dimension, scale, diagonal=False):
    """Unit-radius normalized probes, scaled physically per axis; no incumbent."""
    if dimension < 1 or dimension > 3:
        raise ValueError("The landscape needs 1-3 dimensions and a positive probe scale")
    scales = axis_scale_vector(scale, dimension)
    axes = torch.eye(dimension, dtype=torch.float64)*scales
    points = []
    for axis in axes:
        points.extend((axis.clone(), -axis.clone()))
    for first in range(0 if diagonal else dimension):
        for second in range(first+1, dimension):
            mixed = (axes[first]+axes[second])/math.sqrt(2.)
            points.extend((mixed, -mixed))
    return points


def design(coordinates, scale, diagonal=False):
    """Columns: u_i, .5*u_i^2, u_i*u_j (i<j); Hessian has no hidden factor 2."""
    coordinates = torch.as_tensor(coordinates, dtype=torch.float64)
    u = coordinates/axis_scale_vector(scale, coordinates.shape[-1])
    terms = [u, .5*u.square()]
    mixed = [u[..., first]*u[..., second] for first in range(0 if diagonal else u.shape[-1])
             for second in range(first+1, u.shape[-1])]
    if mixed:
        terms.append(torch.stack(mixed, dim=-1))
    return torch.cat(terms, dim=-1)


def jackknife_covariance(values):
    values = values.double()
    if values.ndim != 2 or len(values) < 3:
        raise ValueError("Jackknife covariance needs at least three paired samples")
    centered = values-values.mean(dim=0)
    return (len(values)-1)/len(values)*(centered.T @ centered)


@dataclass
class FitBlock:
    """One independent shared-noise bank; merge candidates before fitting it."""
    coordinates: list
    incumbent: object
    results: list


@dataclass
class QuadraticFit:
    coefficients: torch.Tensor
    covariance: torch.Tensor
    dimension: int
    scale: float | torch.Tensor
    image_count: int
    observations: int
    block_count: int
    residual_rms: float
    condition: float
    design_rank: int
    variance_inflation: float = 1.
    resolution_diagonal: torch.Tensor | None = None
    diagonal: bool = False
    resampling_covariance: torch.Tensor | None = None
    fold_models: list = field(default_factory=list)
    fold_regression_penalty: float = 0.
    fold_regression_z: float = 1.

    @property
    def scales(self):
        return axis_scale_vector(self.scale, self.dimension)

    @property
    def gradient(self):
        return self.coefficients[:self.dimension]/self.scales

    @property
    def hessian(self):
        d = self.dimension
        matrix = torch.diag(self.coefficients[d:2*d]).clone()
        offset = 2*d
        for first in range(0 if self.diagonal else d):
            for second in range(first+1, d):
                matrix[first, second] = matrix[second, first] = self.coefficients[offset]
                offset += 1
        return matrix/(self.scales[:, None]*self.scales[None, :])

    @property
    def curvature(self):
        return self.hessian.diag()

    @property
    def coefficient_errors(self):
        variance = self.covariance.diag()
        if self.resampling_covariance is not None:
            variance = torch.maximum(variance, self.resampling_covariance.diag())
        return variance.clamp_min(0).sqrt()

    @property
    def hessian_se(self):
        errors = self.coefficient_errors
        d = self.dimension
        matrix = torch.diag(errors[d:2*d]).clone()
        offset = 2*d
        for first in range(0 if self.diagonal else d):
            for second in range(first+1, d):
                matrix[first, second] = matrix[second, first] = errors[offset]
                offset += 1
        return matrix/(self.scales[:, None]*self.scales[None, :])

    def reduction(self, coordinates):
        return -float(design(coordinates, self.scale, self.diagonal) @ self.coefficients)

    def reduction_se(self, coordinates):
        row = design(coordinates, self.scale, self.diagonal)
        variance = row @ self.covariance @ row
        if self.resampling_covariance is not None:
            variance = torch.maximum(variance, row @ self.resampling_covariance @ row)
        return float(variance.clamp_min(0).sqrt())

    def fold_downside(self, coordinates):
        """Soft risk preference, in KID units; not another variance estimate."""
        if not self.fold_models:
            return 0.
        return sum(max(0., -fit.reduction(coordinates)-self.fold_regression_z*fit.reduction_se(coordinates))
                   for fit in self.fold_models)/len(self.fold_models)

    def utility(self, coordinates, penalty):
        return (self.reduction(coordinates)-penalty*self.reduction_se(coordinates)
                -self.fold_regression_penalty*self.fold_downside(coordinates))

    def propose(self, radius, penalty, *, iterations=100, forbidden=None,
                separation=.2, hard_radius=None, component_fits=(), component_tolerance=0.):
        """Search a dimensionless coordinate ball, returning physical coordinates.

        Physical x = scales*z. Exclusions are measured in normalized z units;
        hard_radius separately bounds the physical parameter displacement norm.
        Feasible starts and intermediate iterates are retained. Soft penalties
        help move away from excluded locations, but only feasible points can win.
        """
        if (not math.isfinite(radius) or radius <= 0 or not math.isfinite(penalty)
                or penalty < 0 or iterations < 1 or not math.isfinite(separation)
                or not 0 < separation < 1
                or hard_radius is not None and (not math.isfinite(hard_radius) or hard_radius <= 0)):
            raise ValueError("Invalid acquisition search settings")
        scales = self.scales
        excluded = (torch.stack([torch.as_tensor(point, dtype=torch.float64)/scales
                                 for point in forbidden]) if forbidden
                    else torch.empty(0, self.dimension, dtype=torch.float64))
        if excluded.ndim != 2 or excluded.shape[1] != self.dimension:
            raise ValueError("Excluded points must have the landscape dimension")
        require_finite(excluded, "excluded proposal coordinates")
        generator = torch.Generator(device="cpu").manual_seed(71823)
        rays = torch.randn(96, self.dimension, generator=generator, dtype=torch.float64)
        rays /= rays.norm(dim=1, keepdim=True).clamp_min(1e-30)
        axes = torch.eye(self.dimension, dtype=torch.float64)
        directions = torch.cat([axes, -axes, rays])
        starts = torch.cat([torch.zeros(1, self.dimension, dtype=torch.float64)]+
                           [directions*fraction for fraction in (.01, .03, .1, .25, .5, 1.)])
        objective_scale = max(float(self.coefficients.abs().max()),
                              float(self.coefficient_errors.max()), 1e-15)
        beta = self.coefficients/objective_scale
        if self.diagonal:
            # Use conservative positive curvature for proposing only. Reported
            # predictions and SEs continue to describe the actual fitted model.
            beta = beta.clone()
            errors = self.coefficient_errors/objective_scale
            beta[self.dimension:] = torch.maximum(beta[self.dimension:],
                                                   errors[self.dimension:].clamp_min(1e-12))
        covariance = self.covariance/objective_scale**2
        resampling = (self.resampling_covariance/objective_scale**2
                      if self.resampling_covariance is not None else None)
        component_beta = (torch.stack([fit.coefficients for fit in component_fits])/objective_scale
                          if len(component_fits) > 1 else None)

        fold_beta = (torch.stack([fit.coefficients for fit in self.fold_models])/objective_scale
                     if self.fold_models else None)
        fold_covariance = (torch.stack([fit.covariance for fit in self.fold_models])/objective_scale**2
                           if self.fold_models else None)

        def values(points):
            rows = design(points*radius*scales, scales, self.diagonal)
            variance = ((rows @ covariance)*rows).sum(dim=-1).clamp_min(0)
            if resampling is not None:
                variance = torch.maximum(variance, ((rows @ resampling)*rows).sum(dim=-1).clamp_min(0))
            scores = -(rows @ beta)-penalty*(variance+1e-24).sqrt()
            if component_beta is not None:
                regressions = (rows @ component_beta.T-component_tolerance/objective_scale).clamp_min(0)
                scores = scores-regressions.sum(dim=-1)
            if fold_beta is not None:
                fold_gains = -(rows @ fold_beta.T)
                fold_variance = torch.einsum("bi,kij,bj->bk", rows, fold_covariance, rows).clamp_min(0)
                downside = (-fold_gains-self.fold_regression_z*(fold_variance+1e-24).sqrt()).clamp_min(0).mean(dim=-1)
                scores = scores-self.fold_regression_penalty*downside
            return scores

        def project(points):
            points.div_(points.norm(dim=1, keepdim=True).clamp_min(1.))
            if hard_radius is not None:
                lengths = (points*radius*scales).norm(dim=1, keepdim=True)
                points.div_((lengths/hard_radius).clamp_min(1.))

        def margins(points):
            if not len(excluded):
                return torch.empty(len(points), 0, dtype=torch.float64)
            normalized = points*radius
            lengths = normalized.norm(dim=1, keepdim=True)
            excluded_lengths = excluded.norm(dim=1).unsqueeze(0)
            local = torch.minimum(lengths, excluded_lengths).clamp_min(.05*radius).clamp_max(radius)
            thresholds = (separation*local).clamp_min(radius*1e-9)
            distances = (normalized[:, None, :]-excluded[None, :, :]).norm(dim=-1)
            return distances/thresholds-1.

        def feasible(points):
            good = torch.isfinite(points).all(dim=1) & (points.norm(dim=1) <= 1.+1e-10)
            if hard_radius is not None:
                good &= (points*radius*scales).norm(dim=1) <= hard_radius*(1.+1e-10)
            return good & (margins(points) > 1e-8).all(dim=1)

        with torch.enable_grad():
            with torch.no_grad():
                project(starts)
            initial = values(starts).detach()
            allowed = feasible(starts)
            if not bool(allowed.any()):
                return torch.zeros(self.dimension, dtype=torch.float64)
            ranked = initial.masked_fill(~allowed, -torch.inf)
            best_index = int(ranked.argmax())
            best_value, best = float(ranked[best_index]), starts[best_index].clone()
            keep = min(32, int(allowed.sum()))
            points = starts[ranked.topk(keep).indices].clone().requires_grad_(True)
            optimizer = torch.optim.Adam([points], lr=.04)
            barrier = max(float(initial.abs().max()), 1.)*10.
            for _ in range(iterations):
                optimizer.zero_grad(set_to_none=True)
                violations = (-margins(points)).clamp_min(0)
                objective = values(points)-barrier*violations.square().sum(dim=1)
                (-objective.sum()).backward()
                optimizer.step()
                with torch.no_grad():
                    project(points)
                    scores = values(points).masked_fill(~feasible(points), -torch.inf)
                    index = int(scores.argmax())
                    if float(scores[index]) > best_value:
                        best_value, best = float(scores[index]), points[index].clone()
        coordinates = best.detach()*radius*scales
        require_finite(coordinates, "quadratic proposal")
        component_cost = sum(max(0., -fit.reduction(coordinates)-component_tolerance)
                             for fit in component_fits) if len(component_fits) > 1 else 0.
        if self.utility(coordinates, penalty)-component_cost <= 0:
            coordinates.zero_()
        return coordinates

    def diagnostics(self, radius, penalty, hard_radius=None):
        point = self.propose(radius, penalty, hard_radius=hard_radius)
        predicted, error = self.reduction(point), self.reduction_se(point)
        errors = self.coefficient_errors
        scaled_hessian = self.scales[:, None]*self.hessian*self.scales[None, :]
        return dict(dimension=self.dimension, model="diagonal" if self.diagonal else "full",
                    coefficient_count=len(self.coefficients),
                    coefficients=self.coefficients.tolist(), gradient=self.gradient.tolist(),
                    scaled_gradient=(self.gradient*self.scales).tolist(),
                    hessian=self.hessian.tolist(), hessian_se=self.hessian_se.tolist(),
                    scaled_hessian=scaled_hessian.tolist(),
                    scaled_hessian_eigenvalues=torch.linalg.eigvalsh(scaled_hessian).tolist(),
                    curvature=self.curvature.tolist(),
                    gradient_se=(errors[:self.dimension]/self.scales).tolist(),
                    hessian_eigenvalues=torch.linalg.eigvalsh(self.hessian).tolist(),
                    coefficient_standard_errors=errors.tolist(),
                    coefficient_covariance=self.covariance.tolist(), coordinate_scale=self.scales.tolist(),
                    fold_coefficient_covariance=(self.resampling_covariance.tolist()
                                                 if self.resampling_covariance is not None else None),
                    coefficient_error_rule="max_paired_and_delete_fold",
                    coordinate_scales=self.scales.tolist(),
                    fit_image_count=self.image_count, observations=self.observations,
                    block_count=self.block_count, design_rank=self.design_rank,
                    residual_rms=self.residual_rms, normal_condition=self.condition, condition=self.condition,
                    variance_inflation=self.variance_inflation,
                    ridge_resolution_diagonal=(self.resolution_diagonal.tolist()
                                               if self.resolution_diagonal is not None else None),
                    proposed_coordinates=point.tolist(),
                    scaled_proposed_coordinates=(point/self.scales).tolist(),
                    predicted_reduction=predicted, predicted_reduction_se=error,
                    acquisition=self.utility(point, penalty), uncertainty_penalty=penalty,
                    fold_downside=self.fold_downside(point),
                    fold_regression_penalty=self.fold_regression_penalty,
                    fold_regression_z=self.fold_regression_z)


@dataclass
class FitBundle:
    combined: QuadraticFit
    components: dict
    component_names: list
    joint_covariance: torch.Tensor
    observation_details: dict = field(default_factory=dict)

    def diagnostics(self, radius, penalty, hard_radius=None):
        return dict(combined=self.combined.diagnostics(radius, penalty, hard_radius),
                    components={name: fit.diagnostics(radius, penalty, hard_radius)
                                for name, fit in self.components.items()},
                    component_names=self.component_names,
                    joint_coefficient_covariance=self.joint_covariance.tolist(),
                    **self.observation_details)

    # Explicit delegates keep acquisition code focused on the combined objective.
    @property
    def dimension(self):
        return self.combined.dimension

    @property
    def gradient(self):
        return self.combined.gradient

    @property
    def hessian(self):
        return self.combined.hessian

    @property
    def covariance(self):
        return self.combined.covariance

    @property
    def variance_inflation(self):
        return self.combined.variance_inflation

    def reduction(self, point):
        return self.combined.reduction(point)

    def reduction_se(self, point):
        return self.combined.reduction_se(point)

    def utility(self, point, penalty):
        return self.combined.utility(point, penalty)

    def propose(self, radius, penalty, *, iterations=100, forbidden=None,
                separation=.2, hard_radius=None, component_tolerance=0.):
        return self.combined.propose(radius, penalty, iterations=iterations, forbidden=forbidden,
                                     separation=separation, hard_radius=hard_radius,
                                     component_fits=list(self.components.values()),
                                     component_tolerance=component_tolerance)


def fit_quadratic(blocks, scale, *, ridge=1e-4, covariance_floor=1e-5,
                  diagonal=False, cross_term_ridge_multiplier=10.):
    """Joint GLS of all component KIDs with paired location/metric covariance.

    Coefficients are metric-major; observations are location-major. Each noise
    block has an independent covariance. Reusing a bank in two blocks is rejected.
    The equal-weight combined coefficient covariance includes cross-metric terms.
    """
    if (not blocks or not blocks[0].coordinates
            or not math.isfinite(ridge) or ridge < 0 or not math.isfinite(covariance_floor)
            or not 0 < covariance_floor <= 1 or not math.isfinite(cross_term_ridge_multiplier)
            or cross_term_ridge_multiplier < 1):
        raise ValueError("Invalid quadratic fitting configuration")
    dimension = len(blocks[0].coordinates[0])
    if not 1 <= dimension <= 3:
        raise ValueError("Only one-, two-, and three-dimensional landscapes are supported")
    scales = axis_scale_vector(scale, dimension)
    width = coefficient_count(dimension, diagonal)
    names = list(blocks[0].incumbent.component_scores)
    # Synthetic combined-only results remain useful for small mathematical tests.
    component_count = len(names) or 1
    joint_width = component_count*width
    normal = torch.zeros(joint_width, joint_width, dtype=torch.float64)
    rhs = torch.zeros(joint_width, dtype=torch.float64)
    seen, observations, designs = [], [], []
    image_count = 0
    for block in blocks:
        parent = block.incumbent
        if len(block.coordinates) != len(block.results) or not block.results:
            raise ValueError("Each fit block needs matching coordinates and model results")
        if set(parent.component_scores) != set(names):
            raise ValueError("Every fit block must use the same component KIDs")
        intervals = getattr(parent, "intervals", ((parent.noise_seed, parent.start, parent.start+parent.count),))
        for interval in intervals:
            if any(seed == interval[0] and max(start, interval[1]) < min(end, interval[2])
                   for seed, start, end in seen):
                raise ValueError("Overlapping noise blocks must be merged before fitting")
            seen.append(interval)
        image_count += parent.count
        coordinates = torch.stack([torch.as_tensor(point, dtype=torch.float64)
                                   for point in block.coordinates])
        if coordinates.shape != (len(block.results), dimension):
            raise ValueError("Inconsistent coordinate dimension")
        x = design(coordinates, scales, diagonal)
        require_finite(x, "quadratic design")
        designs.append(x)
        values, jackknife = [], []
        for result in block.results:
            parent.require_same_samples(result)
            if set(result.component_scores) != set(names):
                raise ValueError("Model results use inconsistent component KIDs")
            if names:
                values.append([result.component_scores[name]-parent.component_scores[name]
                               for name in names])
                jackknife.append(torch.stack([
                    result.component_leave_one_out(name)-parent.component_leave_one_out(name)
                    for name in names], dim=1))
            else:
                values.append([result.score-parent.score])
                jackknife.append((result.leave_one_out()-parent.leave_one_out()).unsqueeze(1))
        y = torch.tensor(values, dtype=torch.float64).reshape(-1)
        loo = torch.cat(jackknife, dim=1)
        covariance = jackknife_covariance(loo)
        require_finite(y, "observed KID differences")
        require_finite(covariance, "paired observation covariance")
        joint_x = torch.zeros(len(y), joint_width, dtype=torch.float64)
        for component in range(component_count):
            joint_x[component::component_count, component*width:(component+1)*width] = x
        try:
            eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
        except torch.linalg.LinAlgError as error:
            raise FloatingPointError("Paired covariance decomposition failed") from error
        floor = max(float(eigenvalues.max())*covariance_floor, 1e-24)
        weight = (eigenvectors/eigenvalues.clamp_min(floor)) @ eigenvectors.T
        normal += joint_x.T @ weight @ joint_x
        rhs += joint_x.T @ weight @ y
        observations.append((joint_x, y, weight))
    rank = int(torch.linalg.matrix_rank(torch.cat(designs)))
    if rank < width:
        raise ValueError(f"Quadratic design has rank {rank}, but needs {width} independent columns")
    penalty = max(float(normal.diag().mean())*ridge, 1e-24)
    penalty_factors = torch.ones(width, dtype=torch.float64)
    penalty_factors[2*dimension:] = cross_term_ridge_multiplier
    penalty_diagonal = penalty*penalty_factors.repeat(component_count)
    regularized = normal+torch.diag(penalty_diagonal)
    try:
        coefficients = torch.linalg.solve(regularized, rhs)
        inverse = torch.linalg.solve(regularized, torch.eye(joint_width, dtype=torch.float64))
    except torch.linalg.LinAlgError as error:
        raise FloatingPointError("Regularized full-quadratic solve failed") from error
    covariance = inverse @ normal @ inverse.T
    resolution = inverse @ normal
    covariance = .5*(covariance+covariance.T)
    residuals = torch.cat([x @ coefficients-y for x, y, _ in observations])
    discrepancy = sum(float((x @ coefficients-y) @ weight @ (x @ coefficients-y))
                      for x, y, weight in observations)
    residual_degrees = max(len(residuals)-joint_width, 0)
    # Observation covariance already estimates sampling noise. Residuals also
    # contain spatial misspecification and ridge bias: report them separately
    # instead of pretending extra images will eliminate that discrepancy.
    inflation = 1.
    condition = float(torch.linalg.cond(regularized))
    require_finite(coefficients, "quadratic coefficients")
    require_finite(covariance, "quadratic coefficient covariance")
    require_finite(torch.tensor(condition, dtype=torch.float64), "quadratic condition")
    # Equal weights across metrics; summing diagonal blocks alone is incorrect.
    average = torch.cat([torch.eye(width, dtype=torch.float64)/component_count
                         for _ in range(component_count)], dim=1)
    combined_coefficients = average @ coefficients
    combined_covariance = average @ covariance @ average.T
    location_count = sum(len(block.results) for block in blocks)
    residual_matrix = residuals.reshape(-1, component_count)

    def make_fit(beta, cov, residual):
        return QuadraticFit(beta, cov, dimension, scales, image_count, location_count, len(blocks),
                            float(residual.square().mean().sqrt()), condition, rank, inflation,
                            diagonal=diagonal)

    combined = make_fit(combined_coefficients, combined_covariance, residual_matrix.mean(dim=1))
    components = {}
    for index, name in enumerate(names):
        selection = slice(index*width, (index+1)*width)
        components[name] = make_fit(coefficients[selection], covariance[selection, selection],
                                    residual_matrix[:, index])
        # Diagonal entries describe ridge attenuation in this component's
        # coefficient coordinates. The full resolution matrix also mixes metrics.
        components[name].resolution_diagonal = resolution.diag()[selection]
    details = dict(joint_observation_count=len(residuals),
                   joint_residual_chi_squared=discrepancy,
                   joint_residual_degrees_of_freedom=residual_degrees,
                   residual_variance_estimable=bool(residual_degrees),
                   residual_discrepancy_ratio=discrepancy/residual_degrees if residual_degrees else None,
                   residuals_used_as_sampling_variance=False,
                   ridge_penalty=penalty, ridge_penalty_diagonal=penalty_diagonal.tolist(),
                   cross_term_ridge_multiplier=cross_term_ridge_multiplier, ridge_resolution_diagonal=resolution.diag().tolist(),
                   ridge_effective_parameter_count=float(torch.trace(resolution)))
    return FitBundle(combined, components, names, covariance, details)
