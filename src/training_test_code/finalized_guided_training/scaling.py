"""Adapt physical probe lengths between generations using paired curvature.

The semantic axes can rotate between generations. Retained scale factors are
only starting choices; no old landscape observations are transported or reused.
"""
import math

import torch

from .parameter_space import require_finite


def _symmetric_contrast(plus_score, minus_score, parent_score,
                        plus_values, minus_values, parent_values):
    values = .5*(plus_values.double()+minus_values.double())-parent_values.double()
    require_finite(values, "paired symmetric probe jackknife values")
    count = len(values)
    if count < 3:
        raise ValueError("Symmetric probe uncertainty needs at least three samples")
    centered = values-values.mean()
    error = float(((count-1)/count*centered.square().sum()).sqrt())
    cost = .5*(plus_score+minus_score)-parent_score
    if not math.isfinite(cost) or not math.isfinite(error):
        raise FloatingPointError("Nonfinite paired symmetric probe contrast")
    return dict(cost=cost, cost_se=error)


def symmetric_probe_statistics(plus, minus, incumbent):
    """Even KID change and paired delete-one SE on one shared noise bank.

    q = (KID(+r) + KID(-r))/2 - KID(0) measures the even cost at
    this actual physical radius. It is not divided by the radius squared.
    Combined rows retain correlation between the component feature pipelines.
    """
    plus.require_same_samples(minus)
    plus.require_same_samples(incumbent)
    result = _symmetric_contrast(plus.score, minus.score, incumbent.score,
                                 plus.leave_one_out(), minus.leave_one_out(),
                                 incumbent.leave_one_out())
    names = set(incumbent.component_scores)
    if set(plus.component_scores) != names or set(minus.component_scores) != names:
        raise ValueError("Symmetric probes must use the same component KIDs")
    components = {}
    for name in incumbent.component_scores:
        components[name] = _symmetric_contrast(
            plus.component_scores[name], minus.component_scores[name],
            incumbent.component_scores[name], plus.component_leave_one_out(name),
            minus.component_leave_one_out(name), incumbent.component_leave_one_out(name))
    result["components"] = components
    return result


class AxisScaleController:
    """Persistent semantic-axis scales, updated only between local searches.

    Starting factors are .25 for mean RGB descent, 1.5 for the residual axis,
    and 4 for the remaining-gradient axis. Positive measured even cost must exceed
    curvature_z paired standard errors before a scale can change in either
    direction. A flat, negative, or unresolved axis is never expanded blindly.
    """
    def __init__(self, target_cost=5e-6, curvature_z=2., min_factor=.05,
                 max_factor=4., max_change=2., initial_mean_factor=.25,
                 initial_residual_factor=1.5, initial_remaining_factor=4.):
        if not math.isfinite(target_cost) or target_cost <= 0:
            raise ValueError("target_cost must be finite and positive")
        if not math.isfinite(curvature_z) or curvature_z < 0:
            raise ValueError("curvature_z must be finite and nonnegative")
        if (not math.isfinite(min_factor) or not math.isfinite(max_factor)
                or not 0 < min_factor <= max_factor):
            raise ValueError("Axis factor bounds must be finite, positive, and ordered")
        if not math.isfinite(max_change) or max_change < 1:
            raise ValueError("max_change must be finite and at least one")
        if (not math.isfinite(initial_mean_factor) or initial_mean_factor <= 0
                or not math.isfinite(initial_residual_factor) or initial_residual_factor <= 0
                or not math.isfinite(initial_remaining_factor) or initial_remaining_factor <= 0):
            raise ValueError("Initial semantic axis factors must be finite and positive")
        self.target_cost = target_cost
        self.curvature_z = curvature_z
        self.min_factor, self.max_factor, self.max_change = min_factor, max_factor, max_change
        self.initial_mean_factor = initial_mean_factor
        self.initial_residual_factor = initial_residual_factor
        self.initial_remaining_factor = initial_remaining_factor
        self.factors = {}

    def _initial_factor(self, label):
        if label == "mean RGB descent":
            factor = self.initial_mean_factor
        elif label in ("leading RGB residual SVD", "preconditioned RGB residual"):
            factor = self.initial_residual_factor
        elif label == "remaining-gradient residual":
            factor = self.initial_remaining_factor
        else:
            raise ValueError(f"Unknown semantic axis label: {label}")
        return min(self.max_factor, max(self.min_factor, factor))

    def factors_for(self, axis_labels):
        """Return an owned CPU vector; one generation keeps this vector fixed."""
        if len(set(axis_labels)) != len(axis_labels):
            raise ValueError("Semantic axis labels must be unique")
        values = []
        for label in axis_labels:
            if label not in self.factors:
                self.factors[label] = self._initial_factor(label)
            values.append(self.factors[label])
        return torch.tensor(values, device="cpu", dtype=torch.float64)

    def update(self, axis_statistics, base_radius):
        """Set next-generation factors from actual, possibly capped radii.

        Each entry contains axis_label, cost, cost_se, and physical_radius;
        symmetric_probe_statistics supplies the cost fields and component data.
        base_radius is the physical scalar h used in that generation. The
        caller applies any hard physical-radius cap again in the next search.
        """
        if not math.isfinite(base_radius) or base_radius <= 0:
            raise ValueError("base_radius must be finite and positive")
        labels = [entry["axis_label"] for entry in axis_statistics]
        self.factors_for(labels)
        diagnostics = []
        updates = {}
        for entry in axis_statistics:
            label = entry["axis_label"]
            cost, error, radius = entry["cost"], entry["cost_se"], entry["physical_radius"]
            if (not math.isfinite(cost) or not math.isfinite(error) or error < 0
                    or not math.isfinite(radius) or radius <= 0):
                raise ValueError("Axis probe cost, uncertainty, and physical radius must be valid")
            current = radius/base_radius
            if not math.isfinite(current) or current <= 0:
                raise ValueError("Actual axis factor must be finite and positive")
            next_factor = current
            supported = cost > 0 and cost > self.curvature_z*error
            if supported:
                multiplier = math.sqrt(self.target_cost/cost)
                multiplier = min(self.max_change, max(1/self.max_change, multiplier))
                next_factor = min(self.max_factor,
                                  max(self.min_factor, current*multiplier))
                reason = ("Reduced after resolved positive even cost" if next_factor < current else
                          "Expanded after resolved positive even cost" if next_factor > current else
                          "Held at the target cost or factor bound")
            else:
                reason = ("Held: even cost is nonpositive" if cost <= 0 else
                          "Held: positive even cost is unresolved")
            updates[label] = next_factor
            diagnostics.append(dict(axis_label=label, current_factor=current,
                next_factor=next_factor, cost=cost, cost_se=error,
                physical_radius=radius, base_radius=base_radius,
                target_cost=self.target_cost, curvature_z=self.curvature_z,
                resolved_positive_cost=supported, reason=reason,
                components=entry.get("components", {})))
        # Reject malformed statistics without partially updating existing axes.
        self.factors.update(updates)
        return diagnostics
