"""Full-quadratic fold-guided searches with finite, targeted salvage."""
import time
from dataclasses import dataclass, field

import torch

from src.utilities.print_helper import PrintFields, PrintStatus, PrintTable
from .kid_evaluator import FeatureHistory, kid_standard_error, paired_difference, component_paired_difference
from .parameter_space import basis_vector, require_finite, vector_norm
from .quadratic import FitBlock, coordinate_design
from .evidence import regression_cost, component_regressions
from .folds import FoldLandscape, partition_scores
from .scaling import symmetric_probe_statistics
from .search_policy import search_eligibility, same_location as _same_location


class NoiseSeeds:
    """Allocate unique bank identities for search, previews, and confirmation."""
    def __init__(self, seed):
        self.current = seed

    def next(self):
        if self.current >= 2**63-2:
            raise OverflowError("Noise-bank seed stream exhausted")
        self.current += 1
        return self.current


@dataclass
class Candidate:
    coordinates: torch.Tensor
    label: str
    history: FeatureHistory


@dataclass
class SearchResult:
    accepted: bool = False
    displacement: torch.Tensor | None = None
    norm: float = 0.
    diagnostics: dict = field(default_factory=dict)


class CandidatePool:
    """One immutable anchor/basis and one shared search bank."""
    def __init__(self, space, metric, basis, bank, log, round_index, count, coordinate_scales, phase="search"):
        self.space, self.metric, self.basis, self.bank = space, metric, basis, bank
        self.log, self.round_index, self.count, self.phase = log, round_index, count, phase
        self.role = "validation" if phase == "confirmation" else "search"
        self.source_id = f"noise_{bank.seed}"
        self.scales = coordinate_scales.detach().cpu().double().clone()
        self.incumbent = Candidate(torch.zeros(len(basis), dtype=torch.float64),
                                   "incumbent", FeatureHistory(bank.seed))
        self.candidates = [self.incumbent]

    def add(self, coordinates, label):
        coordinates = coordinates.detach().cpu().double().clone()
        require_finite(coordinates, "candidate coordinates")
        if coordinates.shape != self.incumbent.coordinates.shape:
            raise ValueError("Candidate coordinate dimension differs from the fixed basis")
        for candidate in self.candidates:
            # Only numerical duplicates are merged; statistical separation is
            # enforced separately by proposal selection.
            scale = max(vector_norm(coordinates), vector_norm(candidate.coordinates), 1e-30)
            if vector_norm(coordinates-candidate.coordinates) <= scale*1e-8:
                return candidate
        candidate = Candidate(coordinates, label, FeatureHistory(self.bank.seed))
        self.candidates.append(candidate)
        return candidate

    def evaluate(self, candidate):
        self.metric.runtime.synchronize()
        started = time.monotonic()
        before = (self.metric.generation_seconds, self.metric.feature_seconds, self.metric.kernel_seconds)
        new_images = max(0, self.count-candidate.history.count)
        if new_images:
            self.space.assign(basis_vector(self.basis, candidate.coordinates))
            self.metric.extend(candidate.history, self.bank, self.count)
        result = self.metric.score(candidate.history, self.count)
        # The small score rows stay on CPU too. No GPU model copy is retained.
        candidate.history.features = {name: values.cpu()
                                      for name, values in candidate.history.features.items()}
        PrintFields([("Evaluation", f"{self.phase}: {candidate.label}"),
                     ("Images / new", f"{self.count:,} / {new_images:,}"),
                     ("KID", f"{result.score:.8f}")])
        self.log.event("evaluation", round=self.round_index, phase=self.phase,
                       label=candidate.label, coordinates=candidate.coordinates.tolist(),
                       scaled_coordinates=(candidate.coordinates/self.scales).tolist(),
                       coordinate_scales=self.scales.tolist(),
                       location_id=",".join(float(value).hex() for value in candidate.coordinates),
                       source_id=self.source_id, role=self.role,
                       noise_seed=self.bank.seed, image_count=self.count, new_images=new_images,
                       cached_images=self.count-new_images, seconds=time.monotonic()-started,
                       generation_seconds=self.metric.generation_seconds-before[0],
                       inception_seconds=self.metric.feature_seconds-before[1],
                       kernel_seconds=self.metric.kernel_seconds-before[2],
                       kid=result.score, kid_se=kid_standard_error(result),
                       kid_components=result.component_scores)
        return result

    def block(self, start=0, count=None):
        count = self.count if count is None else count
        candidates = [candidate for candidate in self.candidates[1:] if candidate.history.count >= start+count]
        parent = self.metric.score(self.incumbent.history, count, start=start)
        results = [self.metric.score(candidate.history, count, start=start) for candidate in candidates]
        return FitBlock([candidate.coordinates for candidate in candidates], parent, results)


class RadiusController:
    """No automatic expansion. Model discrepancy can shrink future searches."""
    def __init__(self, reference_norm, probe, step, min_probe, max_probe, min_step, max_step):
        self.reference_norm, self.probe, self.step = reference_norm, probe, step
        self.min_probe, self.max_probe = min_probe, max_probe
        self.min_step, self.max_step = min_step, max_step

    def adapt(self, result):
        if result is None:
            return "Held after numerical failure"
        if result.diagnostics.get("fit_mismatch"):
            self.probe = max(self.min_probe, self.probe*.8)
            self.step = max(self.min_step, self.step*.7)
            return "Reduced after measured local-model discrepancy"
        return "Held; automatic radius expansion is disabled"


def _assessment(evidence, penalty, worthwhile_gain, tolerance, component_z,
                fold_penalty=1., fold_z=1.):
    status = search_eligibility(-evidence["delta"], evidence["se"], penalty, worthwhile_gain)
    cost = regression_cost(evidence["components"], tolerance)
    regressions = component_regressions(evidence["components"], tolerance, component_z)
    folds = evidence.get("folds", [])
    downside = (sum(max(0., row["delta"]-fold_z*row["se"]) for row in folds)/len(folds)
                if folds else 0.)
    status.update(component_cost=cost, fold_downside=downside,
                  utility=status["utility"]-cost-fold_penalty*downside,
                  component_regressions=regressions)
    status["eligible"] = status["normal"] and status["utility"] > 0 and not regressions
    status["promising"] = status["gain"] > 0 and status["gain"]+status["se"] >= worthwhile_gain and not regressions
    status["reason"] = ("eligible" if status["eligible"] else "component_regression" if regressions else
                        "risk_cost" if status["normal"] else status["reason"])
    return status


def _predicted_assessment(landscape, point, penalty, worthwhile, tolerance, component_z, fold_penalty, fold_z):
    fit = landscape.bundle.combined
    evidence = dict(delta=-fit.reduction(point), se=fit.reduction_se(point),
                    components={name: dict(delta=-model.reduction(point), se=model.reduction_se(point))
                                for name, model in landscape.bundle.components.items()},
                    folds=[dict(delta=-model.combined.reduction(point), se=model.combined.reduction_se(point))
                           for model in landscape.fold_fits])
    return _assessment(evidence, penalty, worthwhile, tolerance, component_z, fold_penalty, fold_z)


def _choose_candidate(landscape, pools, radius, hard_radius, penalty, worthwhile_gain,
                      separation, tried, rejected, confirmed, tolerance, component_z,
                      allow_exploration, fold_penalty=1., fold_z=1.):
    bundle, fit = landscape.bundle, landscape.bundle.combined
    exclusions = tried+confirmed
    points = [(bundle.propose(radius, penalty, forbidden=exclusions, separation=separation,
                               hard_radius=hard_radius, component_tolerance=tolerance), "proposal")]
    for index, model in enumerate(landscape.training_fits):
        points.append((model.propose(radius, penalty, iterations=50, forbidden=exclusions,
                                     separation=separation, hard_radius=hard_radius,
                                     component_tolerance=tolerance), f"fold_proposal_{index+1}"))
    points.extend((point*factor, "interpolation") for point in reversed(rejected) for factor in (.5, .25))
    if allow_exploration:
        points.append((bundle.propose(radius, 0., forbidden=exclusions, separation=separation,
                                      hard_radius=hard_radius, component_tolerance=tolerance), "exploratory"))
    points.extend((candidate.coordinates, "cached_probe") for pool in pools for candidate in pool.candidates[1:]
                  if candidate.history.count >= pool.count)
    available, exploratory, records, seen = [], [], [], []
    for coordinates, kind in points:
        if any(torch.equal(coordinates, point) for point in seen):
            continue
        seen.append(coordinates)
        record = dict(kind=kind, coordinates=coordinates.tolist(),
                      scaled_coordinates=(coordinates/fit.scales).tolist())
        norm = vector_norm(coordinates/fit.scales)
        if norm <= radius*1e-8:
            records.append(dict(record, reason="zero_or_no_feasible_proposal"))
            continue
        if norm > radius*(1+1e-8) or vector_norm(coordinates) > hard_radius*(1+1e-8):
            records.append(dict(record, reason="outside_trust_region"))
            continue
        if any(_same_location(coordinates, point, fit.scales, radius, separation) for point in confirmed):
            records.append(dict(record, reason="already_confirmed_neighbourhood"))
            continue
        attempted = any(_same_location(coordinates, point, fit.scales, radius, separation) for point in tried)
        evidence = landscape.evidence(coordinates)
        if attempted and evidence is None:
            records.append(dict(record, reason="tried_neighbourhood"))
            continue
        predicted = _predicted_assessment(landscape, coordinates, penalty, worthwhile_gain, tolerance,
                                          component_z, fold_penalty, fold_z)
        status = (_assessment(evidence, penalty, worthwhile_gain, tolerance, component_z, fold_penalty, fold_z)
                  if evidence is not None else predicted)
        # A cached outlier must also have support from the full-data fit.
        ranking = min(status["utility"], predicted["utility"]) if evidence is not None else status["utility"]
        record.update(status, ranking_utility=ranking, predicted_gain=predicted["gain"],
                      predicted_utility=predicted["utility"],
                      evidence_source="measured" if evidence is not None else "predicted")
        if status["eligible"] and ranking > 0 and not attempted:
            available.append((ranking, int(evidence is not None), coordinates, kind))
        elif (allow_exploration and not attempted and evidence is None and not status["component_regressions"]
              and predicted["gain"] >= .25*worthwhile_gain
              and predicted["gain"]+predicted["se"] >= worthwhile_gain and ranking >= -2*worthwhile_gain):
            exploratory.append((ranking, 0, coordinates, "exploratory"))
            record["reason"] = "bounded_exploration_available"
        record["repair_plausible"] = (not status["component_regressions"]
                                      and max(status["gain"], predicted["gain"]) > 0
                                      and max(status["gain"]+status["se"],
                                              predicted["gain"]+predicted["se"]) >= worthwhile_gain
                                      and ranking >= -2*worthwhile_gain)
        records.append(record)
    choices = available or exploratory
    excluded = {}
    for record in records:
        if record["reason"] != "eligible":
            excluded[record["reason"]] = excluded.get(record["reason"], 0)+1
    details = dict(options=records, excluded_counts=excluded, eligible_count=len(available), exploration_count=len(exploratory),
                   reason="Candidate available" if choices else "No supported worthwhile distinct candidate")
    if not choices:
        return None, details
    _, _, point, kind = max(choices, key=lambda row: (row[0], row[1]))
    details.update(selected_kind=kind, selected_coordinates=point.tolist())
    return (point.clone(), kind), details


def search_round(space, metric, basis, bank, radii, log, *, round_index, coordinate_scales, axis_labels,
                 sample_count, validation_count, max_candidate_attempts, max_refinement_actions,
                 data_folds, max_exploratory_attempts, fold_regression_penalty, fold_regression_z,
                 proposal_penalty, acceptance_confidence_z, min_kid_improvement,
                 component_regression_tolerance, component_regression_z, minimum_worthwhile_gain,
                 candidate_minimum_separation, fit_ridge, cross_term_ridge_multiplier,
                 covariance_floor, noise_seeds):
    scales = coordinate_scales.detach().cpu().double().clone()
    if not basis or len(scales) != len(basis) or len(axis_labels) != len(basis):
        raise ValueError("Every basis axis needs a frozen physical scale and label")
    require_finite(scales, "coordinate scales")
    if bool((scales <= 0).any()):
        raise ValueError("Physical scales must be positive")
    h = min(radii.probe, radii.max_probe, radii.max_step)*radii.reference_norm
    radius = min(1., radii.step*radii.reference_norm/h)
    hard_radius = min(radii.max_step*radii.reference_norm, float(scales.max())*radius)
    result = SearchResult(diagnostics=dict(reason="No worthwhile confirmed improvement", fit_mismatch=False,
        base_probe_radius=h, coordinate_scales=scales.tolist(), scaled_step_radius=radius, hard_step_radius=hard_radius))
    primary = CandidatePool(space, metric, basis, bank, log, round_index, sample_count, scales)
    pools, tried, rejected, confirmed, refined = [primary], [], [], [], []
    attempts = actions = explorations = fit_version = 0
    pending = None
    stencil = coordinate_design(len(basis), scales)
    worst_images = ((len(stencil)+1)*sample_count
                    +max_candidate_attempts*(2*sample_count+2*validation_count)
                    +max_refinement_actions*max(sample_count, 2*validation_count))
    result.diagnostics["maximum_generated_images"] = worst_images
    log.event("search_limits", round=round_index, sample_count=sample_count, validation_count=validation_count,
              max_candidate_attempts=max_candidate_attempts, max_refinement_actions=max_refinement_actions,
              max_exploratory_attempts=max_exploratory_attempts, maximum_generated_images=worst_images,
              note="Finite action bound, not an image-count stopping rule")
    try:
        parent = primary.evaluate(primary.incumbent)
        for index, point in enumerate(stencil, start=1):
            primary.evaluate(primary.add(point, f"landscape {index}"))
        origin = partition_scores(metric, [(primary.incumbent.history, sample_count)], data_folds)
        log.event("search_origin", round=round_index, kid=origin.full.score,
                  kid_se=kid_standard_error(origin.full), kid_components=origin.full.component_scores,
                  noise_seeds=[bank.seed], source_id=primary.source_id, sample_count=sample_count,
                  data_folds=data_folds, images_per_fold=sample_count//data_folds,
                  fold_membership=[row.sample_ranges for row in origin.folds], pooled_features=True)
        axes = []
        for axis, label in enumerate(axis_labels):
            plus = metric.score(primary.candidates[2*axis+1].history, sample_count)
            minus = metric.score(primary.candidates[2*axis+2].history, sample_count)
            axes.append(dict(axis_label=label, physical_radius=float(scales[axis]),
                             **symmetric_probe_statistics(plus, minus, parent)))
        result.diagnostics["axis_probe_statistics"] = axes
        log.event("axis_probe_statistics", round=round_index, axes=axes,
                  coordinate_scales=scales.tolist(), base_probe_radius=h)
        # Each pass consumes a candidate cycle or a refinement action, or stops.
        for _ in range(max_candidate_attempts+max_refinement_actions+1):
            if attempts >= max_candidate_attempts:
                result.diagnostics["reason"] = "Finite candidate allowance exhausted"
                break
            fit_version += 1
            landscape = FoldLandscape(metric, pools, scales, data_folds, fit_ridge, covariance_floor,
                                       diagonal=False, cross_term_ridge_multiplier=cross_term_ridge_multiplier,
                                       fold_regression_penalty=fold_regression_penalty, fold_regression_z=fold_regression_z)
            bundle, fit = landscape.bundle, landscape.bundle.combined
            log.event("fold_stability", round=round_index, attempt=attempts+1, fit_version=fit_version,
                      **landscape.diagnostics())
            log.event("fit", round=round_index, attempt=attempts+1, fit_version=fit_version,
                      combined=fit.diagnostics(radius, proposal_penalty, hard_radius=hard_radius),
                      components={name: model.diagnostics(radius, proposal_penalty, hard_radius=hard_radius)
                                  for name, model in bundle.components.items()},
                      coordinate_scales=scales.tolist(), scaled_step_radius=radius,
                      hard_step_radius=hard_radius, **bundle.observation_details)
            if fit_version == 1:
                log.event("spatial_fit_check", round=round_index, fit_version=fit_version,
                          **landscape.spatial_diagnostics())
            if pending is not None and actions < max_refinement_actions:
                pool, point = pending
                point = primary.add(point, "retained salvage half step").coordinates
                pending = None
                predicted_status = _predicted_assessment(
                    landscape, point, proposal_penalty, minimum_worthwhile_gain, component_regression_tolerance,
                    component_regression_z, fold_regression_penalty, fold_regression_z)
                existing = landscape.evidence(point)
                plausible = (predicted_status["gain"]+predicted_status["se"] >= minimum_worthwhile_gain
                             or existing is not None and -existing["delta"]+existing["se"] >= minimum_worthwhile_gain)
                if (plausible and not any(_same_location(point, old, scales, radius, candidate_minimum_separation)
                                         for old in confirmed) and vector_norm(point/scales) > radius*1e-8):
                    actions += 1
                    log.event("refinement_prediction", round=round_index, fit_version=fit_version,
                              action=actions, kind="promoted_half_step", source_id=pool.source_id,
                              coordinates=point.tolist(), **landscape.prediction(point))
                    before = metric.generated_images
                    child = pool.evaluate(pool.add(point, f"salvage half step {actions}"))
                    old = metric.score(pool.incumbent.history, pool.count)
                    delta, error = paired_difference(child, old)
                    log.event("refinement", round=round_index, action=actions, fit_version=fit_version,
                              kind="promoted_half_step", source_id=pool.source_id, role=pool.role,
                              coordinates=point.tolist(), gain=-delta, se=error,
                              new_images=metric.generated_images-before,
                              reason="Test a revised length using the rejected validation evidence")
                    continue
            suggestion, selection = _choose_candidate(
                landscape, pools, radius, hard_radius, proposal_penalty, minimum_worthwhile_gain,
                candidate_minimum_separation, tried, rejected, confirmed, component_regression_tolerance,
                component_regression_z, explorations < max_exploratory_attempts,
                fold_regression_penalty, fold_regression_z)
            log.event("candidate_selection", round=round_index, attempt=attempts+1, fit_version=fit_version, **selection)
            if suggestion is None:
                repairs = [row for row in selection["options"] if row.get("repair_plausible")
                           and not any(_same_location(torch.tensor(row["coordinates"], dtype=torch.float64),
                                                      old, scales, radius, candidate_minimum_separation) for old in refined)]
                if actions < max_refinement_actions and repairs:
                    repair = max(repairs, key=lambda row: row["ranking_utility"])
                    point = primary.add(torch.tensor(repair["coordinates"], dtype=torch.float64),
                                        f"targeted refinement {actions+1}").coordinates
                    actions += 1
                    refined.append(point.clone())
                    log.event("refinement_prediction", round=round_index, fit_version=fit_version,
                              action=actions, kind="decision_repair", coordinates=point.tolist(),
                              **landscape.prediction(point))
                    before = metric.generated_images
                    if landscape.evidence(point) is None:
                        pool, kind = primary, "interior_geometry"
                        parent = metric.score(pool.incumbent.history, pool.count)
                    else:
                        pool = CandidatePool(space, metric, basis, metric.noise_bank(noise_seeds.next(), validation_count),
                                             log, round_index, validation_count, scales, phase="search_refinement")
                        pools.append(pool)
                        kind = "independent_paired_check"
                        parent = pool.evaluate(pool.incumbent)
                    observation = pool.evaluate(pool.add(point, f"targeted refinement {actions}"))
                    delta, error = paired_difference(observation, parent)
                    tried = [old for old in tried if not _same_location(point, old, scales, radius,
                                                                       candidate_minimum_separation)]
                    log.event("refinement", round=round_index, action=actions, fit_version=fit_version,
                              kind=kind, source_id=pool.source_id, role="search", coordinates=point.tolist(),
                              gain=-delta, se=error, new_images=metric.generated_images-before,
                              reason="Plausible decision can change with a targeted measurement")
                    continue
                result.diagnostics["reason"] = selection["reason"]
                break
            attempts += 1
            proposed, kind = suggestion
            explorations += int(kind == "exploratory")
            proposed = primary.add(proposed, f"candidate {attempts} ({kind})").coordinates
            cached = landscape.evidence(proposed)
            cached_ready = cached is not None and _assessment(
                cached, proposal_penalty, minimum_worthwhile_gain, component_regression_tolerance,
                component_regression_z, fold_regression_penalty, fold_regression_z)["eligible"]
            rows = []
            for factor in ((1.,) if cached_ready else (1., .5)):
                point = primary.add(proposed*factor, f"candidate {attempts} ({kind}, scale {factor:g})").coordinates
                if any(_same_location(point, old, scales, radius, candidate_minimum_separation) for old in tried+confirmed):
                    continue
                point_kind = kind if factor == 1 else "interpolation"
                label = f"candidate {attempts} ({point_kind})"
                prediction = landscape.prediction(point)
                log.event("candidate_prediction", round=round_index, attempt=attempts, fit_version=fit_version,
                          label=label, kind=point_kind, coordinates=point.tolist(), **prediction)
                primary.evaluate(primary.add(point, label))
                evidence = landscape.evidence(point)
                status = _assessment(evidence, proposal_penalty, minimum_worthwhile_gain,
                                     component_regression_tolerance, component_regression_z,
                                     fold_regression_penalty, fold_regression_z)
                was_in_fit = any(torch.equal(point, old) for block in landscape.blocks for old in block.coordinates)
                mismatch = (not was_in_fit and abs(prediction["predicted_reduction"]+evidence["delta"])
                            > 2*(prediction["predicted_reduction_se"]+evidence["se"]))
                result.diagnostics["fit_mismatch"] |= mismatch
                rows.append((point, label, point_kind, evidence, status, prediction, mismatch))
            if not rows:
                tried.append(proposed.clone())
                continue
            eligible = [row for row in rows if row[4]["eligible"]]
            point, label, kind, evidence, status, prediction, mismatch = max(
                eligible or rows, key=lambda row: row[4]["utility"])
            log.event("line_search", round=round_index, attempt=attempts, fit_version=fit_version,
                      alternatives=[dict(assessment, coordinates=coords.tolist(), kind=point_kind,
                                         delta=ev["delta"], se=ev["se"], folds=ev["folds"],
                                         paired_se=ev["paired_se"], fold_se=ev["fold_se"],
                                         spatial_holdout=not any(torch.equal(coords, old)
                                                                 for block in landscape.blocks for old in block.coordinates),
                                         **pred)
                                    for coords, _, point_kind, ev, assessment, pred, _ in rows])
            details = dict(round=round_index, attempt=attempts, fit_version=fit_version, label=label, kind=kind,
                           coordinates=point.tolist(), scaled_coordinates=(point/scales).tolist(),
                           coordinate_scales=scales.tolist(), search_delta=evidence["delta"], search_se=evidence["se"],
                           evidence_delta=evidence["delta"], evidence_se=evidence["se"],
                           scaled_step_norm=vector_norm(point/scales), physical_step_norm=vector_norm(point),
                           scaled_step_radius=radius, hard_step_radius=hard_radius, search_gain=status["gain"],
                           search_utility=status["utility"], minimum_worthwhile_gain=minimum_worthwhile_gain,
                           proposal_penalty=proposal_penalty, fold_regression_penalty=fold_regression_penalty,
                           fold_regression_z=fold_regression_z, measured_fold_downside=status["fold_downside"],
                           component_cost=status["component_cost"], raw_gain_pass=status["raw_gain_pass"],
                           uncertainty_margin_pass=status["uncertainty_margin_pass"], fit_mismatch=mismatch,
                           support_count=evidence["support_count"], search_banks=evidence["banks"],
                           search_components=evidence["components"], supporting_folds=evidence["supporting_folds"],
                           search_folds=evidence["folds"], paired_search_se=evidence["paired_se"],
                           fold_search_se=evidence["fold_se"], pooled_search=True, cached_finalist=cached_ready,
                           sample_count=sample_count, validation_count=validation_count, **prediction)
            PrintFields([("Candidate cycle", f"{attempts}/{max_candidate_attempts}"),
                         ("Refinement actions", f"{actions}/{max_refinement_actions}"),
                         ("Search gain / SE", f"{status['gain']:+.3g} / {evidence['se']:.3g}"),
                         ("Fold downside / component cost", f"{status['fold_downside']:.3g} / {status['component_cost']:.3g}"),
                         ("Risk-adjusted search utility", f"{status['utility']:+.3g}")])
            if not eligible:
                log.event("proposal", **details, reason=status["reason"])
                tried.extend(coords.clone() for coords, *_ in rows)
                rejected.append(point.clone())
                result.diagnostics["reason"] = "Measured search does not support a worthwhile improvement"
                continue
            tried.append(point.clone())
            confirmed.append(point.clone())
            confirmation_seed = noise_seeds.next()
            log.event("finalist_frozen", **details, confirmation_noise_seed=confirmation_seed,
                      acceptance_confidence_z=acceptance_confidence_z)
            fresh = CandidatePool(space, metric, basis, metric.noise_bank(confirmation_seed, validation_count),
                                  log, round_index, validation_count, scales, phase="confirmation")
            parent = fresh.evaluate(fresh.incumbent)
            observation = fresh.evaluate(fresh.add(point, label))
            delta, se = paired_difference(observation, parent)
            components, component_deltas = {}, {}
            for name in observation.component_scores:
                change, error = component_paired_difference(observation, parent, name)
                components[name] = dict(delta=change, se=error)
                component_deltas[name] = dict(search_delta=evidence["components"][name]["delta"],
                                              search_se=evidence["components"][name]["se"],
                                              fresh_delta=change, fresh_se=error)
            regressions = component_regressions(components, component_regression_tolerance, component_regression_z)
            accepted = delta+acceptance_confidence_z*se < -min_kid_improvement and not regressions
            reason = ("Fresh paired improvement passed" if accepted else
                      "Fresh component regression: "+", ".join(regressions) if regressions else
                      "Fresh evaluation indicates worsening" if delta-2*se > 0 else
                      "Fresh improvement did not pass the fixed acceptance margin")
            log.event("finalist", **details, fresh_delta=delta, fresh_se=se, component_deltas=component_deltas,
                      accepted=accepted, reason=reason, confirmation_noise_seed=confirmation_seed,
                      validation_source_id=fresh.source_id, acceptance_value=delta+acceptance_confidence_z*se,
                      acceptance_confidence_z=acceptance_confidence_z,
                      component_regressions=regressions, component_regression_tolerance=component_regression_tolerance,
                      component_regression_z=component_regression_z,
                      fresh_incumbent=parent.score, fresh_candidate=observation.score)
            comparison = [("Search (selected)", f"{evidence['delta']:+.8f}", f"{evidence['se']:.8f}"),
                          ("Unused fresh noise", f"{delta:+.8f}", f"{se:.8f}")]
            comparison.extend((name, f"{row['delta']:+.8f}", f"{row['se']:.8f}") for name, row in components.items())
            PrintTable(["Comparison", "KID change", "Paired SE"], comparison)
            PrintStatus(f"Candidate {attempts}/{max_candidate_attempts}: {reason}; z={acceptance_confidence_z:g}",
                        "success" if accepted else "warning")
            result.diagnostics.update(reason=reason, selected_candidate=label, final_delta=delta, final_se=se,
                                      scaled_step_norm=vector_norm(point/scales),
                                      prediction_ratio=-evidence["delta"]/prediction["predicted_reduction"]
                                      if prediction["predicted_reduction"] > 0 else None)
            if accepted:
                result.accepted, result.displacement = True, basis_vector(basis, point)
                result.norm = vector_norm(result.displacement)
                result.diagnostics["accepted_basis_coordinates"] = point.tolist()
                break
            rejected.append(point.clone())
            fresh.role, fresh.phase = "promoted_search", "salvage_promoted"
            pools.append(fresh)
            pending = (fresh, point*.5)
            log.event("confirmation_reclassified", round=round_index, noise_seed=confirmation_seed,
                      source_id=fresh.source_id, previous_role="validation", new_role="promoted_search",
                      reason="Failed validation is search evidence; next acceptance uses unused noise")
        result.diagnostics.update(candidate_attempts=attempts, refinement_actions=actions,
                                  exploratory_attempts=explorations, search_bank_count=len(pools),
                                  full_curvature_refined=True)
        log.event("salvage_stop", round=round_index, reason=result.diagnostics["reason"],
                  candidate_attempts=attempts, refinement_actions=actions,
                  exploratory_attempts=explorations, accepted=result.accepted)
        return result
    finally:
        space.assign()
