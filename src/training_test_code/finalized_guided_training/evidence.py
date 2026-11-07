"""Paired measurements within banks, independent aggregation between banks."""
import math

import torch

from .kid_evaluator import paired_difference, component_paired_difference


def combine_estimates(rows):
    """Count-weighted (value, SE, count) estimates from independent banks.

    This averages unbiased block KIDs, not the exact pooled-image U-statistic.
    Between-bank scatter can enlarge the estimated uncertainty. Component
    errors must never be substituted for the combined paired KID error.
    """
    if not rows:
        raise ValueError("Independent aggregation needs measurements")
    if any(not math.isfinite(value) or not math.isfinite(se) or se < 0 or count < 3
           for value, se, count in rows):
        raise FloatingPointError("Invalid independent-bank estimate")
    total = sum(row[2] for row in rows)
    weights = [row[2]/total for row in rows]
    value = sum(weight*row[0] for weight, row in zip(weights, rows))
    variance = sum((weight*row[1])**2 for weight, row in zip(weights, rows))
    if len(rows) > 1:
        square_sum = sum(weight**2 for weight in weights)
        scatter = sum(weight*(row[0]-value)**2 for weight, row in zip(weights, rows))
        variance = max(variance, scatter*square_sum/max(1-square_sum, 1e-15))
    return value, math.sqrt(max(variance, 0.))


def candidate_evidence(coordinates, blocks):
    """Use only exact-location observations, preserving sample pairing."""
    banks, component_rows, seen = [], {}, []
    for block in blocks:
        for point, result in zip(block.coordinates, block.results):
            if not torch.equal(point, coordinates):
                continue
            for identity in result.intervals:
                if any(seed == identity[0] and max(start, identity[1]) < min(end, identity[2])
                       for seed, start, end in seen):
                    raise ValueError("A noise bank was counted twice in candidate evidence")
                seen.append(identity)
            if set(result.component_scores) != set(block.incumbent.component_scores):
                raise ValueError("Candidate and incumbent use different KID components")
            if banks and set(result.component_scores) != set(component_rows):
                raise ValueError("Independent banks use different KID components")
            delta, se = paired_difference(result, block.incumbent)
            banks.append(dict(delta=delta, se=se, count=result.count, noise_seed=result.noise_seed))
            for name in result.component_scores:
                change, error = component_paired_difference(result, block.incumbent, name)
                component_rows.setdefault(name, []).append((change, error, result.count))
            break
    if not banks:
        return None
    delta, se = combine_estimates([(row["delta"], row["se"], row["count"]) for row in banks])
    components = {}
    for name, rows in component_rows.items():
        value, error = combine_estimates(rows)
        components[name] = dict(delta=value, se=error)
    return dict(delta=delta, se=se, banks=banks, components=components,
                support_count=sum(row["delta"] < 0 for row in banks),
                required_support=len(banks)//2+1)


def regression_cost(components, tolerance):
    """Soft cost in raw KID units; improving one metric cannot hide a large loss."""
    return sum(max(0., row["delta"]-tolerance) for row in components.values())


def component_regressions(components, tolerance, z):
    """Flag material worsening beyond a fixed SE allowance; not a confidence test."""
    return [name for name, row in components.items() if row["delta"]-z*row["se"] > tolerance]
