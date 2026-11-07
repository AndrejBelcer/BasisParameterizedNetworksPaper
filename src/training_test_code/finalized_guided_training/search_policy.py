"""Shared search gates and coordinate distances; no model or runtime imports."""
import math

from .parameter_space import vector_norm


def same_location(first, second, scales, radius, separation):
    """Physical points compared in the dimensionless proposal metric."""
    a, b = first/scales, second/scales
    scale = min(radius, max(min(vector_norm(a), vector_norm(b)), .05*radius))
    return vector_norm(a-b) <= max(separation*scale, radius*1e-9)


def search_eligibility(gain, se, penalty, worthwhile_gain):
    """Identical criteria for predicted and measured search evidence.

    A marginal is positive and passes just one normal condition. It remains a
    diagnostic label for older logs/tests; exploratory measurements never
    bypass measured utility or unused-confirmation acceptance requirements.
    """
    if not math.isfinite(gain) or not math.isfinite(se) or se < 0:
        raise FloatingPointError("Nonfinite search gain or invalid paired uncertainty")
    utility = gain-penalty*se
    raw_pass = gain >= worthwhile_gain
    margin_pass = utility > 0
    normal = raw_pass and margin_pass
    marginal = gain > 0 and (raw_pass or margin_pass) and not normal
    reason = ("eligible" if normal else "positive_marginal" if marginal else
              "nonpositive_gain" if gain <= 0 else "gain_and_margin_insufficient")
    return dict(gain=gain, se=se, utility=utility, raw_gain_pass=raw_pass,
                uncertainty_margin_pass=margin_pass, normal=normal, marginal=marginal, reason=reason)
