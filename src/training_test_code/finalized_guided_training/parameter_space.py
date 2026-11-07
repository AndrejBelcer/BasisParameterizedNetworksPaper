"""CPU parameter anchors and the three-axis RGB-guided search space."""
import math

import torch


def require_finite(value, label):
    if not torch.isfinite(value).all():
        raise FloatingPointError(f"Nonfinite {label}; abandoning the round without changing the incumbent")


def vector_norm(value):
    return float(torch.linalg.vector_norm(value, dtype=torch.float64))


def vector_dot(first, second):
    return float(torch.sum(first*second, dtype=torch.float64))


def _orthogonalize(vector, basis):
    """Modified Gram-Schmidt twice; return accumulated projection coefficients."""
    coefficients = torch.zeros(len(basis), dtype=torch.float64)
    for _ in range(2):
        for index, direction in enumerate(basis):
            coefficient = vector_dot(direction, vector)
            coefficients[index] += coefficient
            vector.add_(direction, alpha=-coefficient)
    return coefficients


def basis_vector(basis, coordinates):
    if not basis or len(basis) != len(coordinates):
        raise ValueError("A displacement needs one coordinate for every basis direction")
    result = torch.zeros_like(basis[0])
    for direction, coefficient in zip(basis, coordinates):
        value = float(coefficient)
        if value:
            result.add_(direction, alpha=value)
    require_finite(result, "parameter displacement")
    return result


def orthonormal_basis(vectors, norms, removed_gradient_fraction=.1, tolerance=1e-3,
                      second_direction=None):
    """Mean descent, leading residual SVD, then remaining-gradient descent.

    removed_gradient_fraction selects
    the largest absolute mean-gradient coordinates globally, then REMOVES them
    from the third input vector. Projection generally makes that vector dense.
    Since mean descent is already in the basis, projecting a mask or its
    complement gives opposite directions in exactly the same one-dimensional
    span. A nearly dependent vector is omitted rather than normalized.
    Inputs are normalized gradient guides accompanied by their original norms.
    Optional second_direction replaces the residual SVD; the mean and tail are
    retained first, then its independent residual is inserted as axis 2.
    """
    if (not vectors or len(vectors) != len(norms)
            or not math.isfinite(removed_gradient_fraction)
            or not 0 < removed_gradient_fraction <= 1
            or not math.isfinite(tolerance) or not 0 < tolerance < 1):
        raise ValueError("Invalid guides, selected gradient fraction, or basis tolerance")
    if any(not math.isfinite(norm) or norm < 0 for norm in norms):
        raise FloatingPointError("Invalid guide norms")
    if any(vector.ndim != 1 or vector.numel() != vectors[0].numel() for vector in vectors):
        raise ValueError("Guides must be equal-length parameter vectors")
    info = dict(method="mean_residual_svd_remaining_gradient", axis_labels=[],
                mean_gradient_norm=0., mean_axis=False, mean_relative_norm=0.,
                singular_values=[], singular_value_cutoff=0., orthogonality_error=0.,
                removed_gradient_fraction=removed_gradient_fraction,
                third_source="remaining_gradient", removed_coordinate_count=0,
                remaining_coordinate_count=vectors[0].numel(), removed_energy_fraction=0.,
                remaining_energy_fraction=0., third_input_norm=0., third_projected_norm=0.,
                third_projection_retained_fraction=0.)
    scale = max(norms)
    if scale == 0:
        return [], info
    mean = torch.zeros_like(vectors[0], device="cpu", dtype=torch.float32)
    for vector, norm in zip(vectors, norms):
        require_finite(vector, "RGB guide")
        mean.add_(vector.detach().cpu().float(), alpha=(norm/scale)/len(vectors))
    mean_norm = vector_norm(mean)
    numerical_tolerance = 8*torch.finfo(torch.float32).eps
    cutoff = max(tolerance, numerical_tolerance)
    info.update(mean_gradient_norm=mean_norm*scale, mean_relative_norm=mean_norm,
                singular_value_cutoff=cutoff*scale)
    if mean_norm <= numerical_tolerance:
        info["rank_reason"] = "Mean RGB gradient is numerically negligible"
        return [], info
    basis = [mean.clone().div_(-mean_norm)]
    info.update(mean_axis=True, axis_labels=["mean RGB descent"])

    # QR compresses the huge residual matrix; only the small R needs FP64 SVD.
    residual_basis = []
    small = torch.zeros(len(vectors), len(vectors), dtype=torch.float64)
    svd_vectors = vectors if second_direction is None else []
    for column, (vector, norm) in enumerate(zip(svd_vectors, norms)):
        residual = vector.detach().cpu().float().clone().mul_(norm/scale)
        _orthogonalize(residual, basis)
        coefficients = _orthogonalize(residual, residual_basis)
        small[:len(coefficients), column] = coefficients
        length = vector_norm(residual)
        if length > numerical_tolerance:
            small[len(residual_basis), column] = length
            residual_basis.append(residual.div_(length))
    if residual_basis:
        try:
            left, singular, _ = torch.linalg.svd(small[:len(residual_basis)], full_matrices=False)
        except torch.linalg.LinAlgError as error:
            raise FloatingPointError("RGB residual SVD failed") from error
        require_finite(singular, "RGB residual singular values")
        info["singular_values"] = (singular*scale).tolist()
        if float(singular[0]) > cutoff:
            residual = basis_vector(residual_basis, left[:, 0])
            _orthogonalize(residual, basis)
            length = vector_norm(residual)
            if length > numerical_tolerance:
                residual.div_(length)
                pivot = int(left[:, 0].abs().argmax())
                if float(left[pivot, 0]) < 0:
                    residual.neg_()
                basis.append(residual)
                info["axis_labels"].append("leading RGB residual SVD")
    del residual_basis, small

    selected = max(1, math.ceil(removed_gradient_fraction*mean.numel()))
    indices = torch.topk(mean.abs(), selected, sorted=False).indices
    removed_norm = vector_norm(mean[indices])
    remaining = -mean.clone()
    remaining[indices] = 0
    del indices
    original_norm = vector_norm(remaining)
    _orthogonalize(remaining, basis)
    projected_norm = vector_norm(remaining)
    info.update(removed_coordinate_count=selected,
                remaining_coordinate_count=mean.numel()-selected,
                removed_energy_fraction=(removed_norm/mean_norm)**2,
                remaining_energy_fraction=(original_norm/mean_norm)**2,
                third_input_norm=original_norm*scale,
                third_projected_norm=projected_norm*scale,
                third_projection_retained_fraction=(projected_norm/original_norm)**2
                if original_norm else 0.)
    if projected_norm > cutoff:
        basis.append(remaining.div_(projected_norm))
        info["axis_labels"].append("remaining-gradient residual")
    if second_direction is not None:
        if second_direction.shape != mean.shape:
            raise ValueError("Preconditioned direction has the wrong parameter shape")
        residual = second_direction.detach().cpu().float().clone()
        require_finite(residual, "preconditioned basis input")
        original = vector_norm(residual)
        # Preserve the tail first, then insert the optional new direction as
        # axis 2. All three returned axes remain mutually perpendicular.
        _orthogonalize(residual, basis)
        length = vector_norm(residual)
        info.update(method="mean_preconditioned_remaining_gradient",
                    second_projection_retained_fraction=(length/original)**2 if original else 0.)
        if length > max(tolerance*original, numerical_tolerance):
            basis.insert(1, residual.div_(length))
            info["axis_labels"].insert(1, "preconditioned RGB residual")
    info["orthogonality_error"] = max(
        abs(vector_dot(first, second)-(1. if i == j else 0.))
        for i, first in enumerate(basis) for j, second in enumerate(basis) if j >= i)
    if info["orthogonality_error"] > 1e-4:
        raise FloatingPointError("Constructed search basis is not orthonormal")
    info["dimension"] = len(basis)
    return basis, info


class ParameterSpace:
    """Assign candidates from one immutable CPU anchor without accumulated drift."""
    def __init__(self, model):
        self.model = model
        self.parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
        if not self.parameters:
            raise ValueError("Model has no trainable parameters")
        self.size = sum(parameter.numel() for parameter in self.parameters)
        self.anchor = None
        self.buffers = {}

    def begin_round(self):
        self.anchor = self.vector()
        require_finite(self.anchor, "incumbent parameters")
        self.buffers = {name: value.detach().clone() for name, value in self.model.named_buffers()}

    def vector(self):
        return torch.cat([parameter.detach().reshape(-1) for parameter in self.parameters]).cpu().float()

    def gradient(self):
        return torch.cat([(parameter.grad.detach().reshape(-1) if parameter.grad is not None
                           else torch.zeros_like(parameter).reshape(-1))
                          for parameter in self.parameters]).cpu().float()

    @torch.no_grad()
    def assign(self, displacement=None):
        if self.anchor is None:
            raise RuntimeError("begin_round() must precede candidate evaluation")
        if displacement is not None and displacement.shape != self.anchor.shape:
            raise ValueError("Parameter displacement has the wrong shape")
        values = self.anchor if displacement is None else self.anchor+displacement
        require_finite(values, "candidate parameters")
        values = values.to(device=self.parameters[0].device, dtype=torch.float32)
        offset = 0
        for parameter in self.parameters:
            size = parameter.numel()
            parameter.copy_(values[offset:offset+size].view_as(parameter))
            offset += size
        for name, value in self.model.named_buffers():
            value.copy_(self.buffers[name])
