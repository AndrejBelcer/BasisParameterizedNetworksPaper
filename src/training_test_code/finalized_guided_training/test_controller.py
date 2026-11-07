"""Optional synthetic controller checks; no model, dataset or GPU runtime."""
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from . import search, folds as fold_module
from .evidence import candidate_evidence
from .kid_evaluator import KIDResult
from .folds import PartitionScores
from .quadratic import FitBlock, QuadraticFit, FitBundle


class FakeSpace:
    def __init__(self):
        self.point = torch.zeros(1)

    def assign(self, displacement=None):
        self.point = torch.zeros(1) if displacement is None else displacement.clone()


class FakeMetric:
    def __init__(self, space, fresh_worsens=False, disagree=False):
        self.space, self.fresh_worsens, self.disagree = space, fresh_worsens, disagree
        self.generated = self.generated_images = 0
        self.generation_seconds = self.feature_seconds = self.kernel_seconds = 0.
        self.runtime = SimpleNamespace(synchronize=lambda: None)
        self.seeds = []

    def noise_bank(self, seed, count):
        self.seeds.append(seed)
        return SimpleNamespace(seed=seed)

    def extend(self, history, bank, count):
        self.generated += count
        self.generated_images += count
        history.features = {"fake": torch.ones(count, 1)}
        history.point = float(self.space.point[0])

    def score(self, history, count, start=0):
        point = history.point
        slope = -2e-4
        if self.disagree and history.noise_seed == 2:
            slope = 5e-5
        if self.fresh_worsens and history.noise_seed >= 2:
            slope = 2e-4
        curvature = 1e-5 if self.disagree else 1e-4
        value = slope*point+curvature*point*point
        scores = {"first": .01+value, "second": .02+value}
        index = torch.arange(count, dtype=torch.float64)-(count-1)/2
        # Small shared sample variation, larger in displaced models.
        rows = {name: -.5*score+index*(1+abs(point))*1e-9 for name, score in scores.items()}
        zeros = torch.zeros(count, dtype=torch.float64)
        return KIDResult(sum(scores.values())/2, zeros, sum(rows.values())/2,
                         history.noise_seed, start, count, scores,
                         {name: zeros for name in scores}, rows)


class FakeLog:
    def __init__(self):
        self.events = []

    def event(self, name, **values):
        self.events.append(dict(event=name, **values))


def fake_partition(metric, entries, fold_count):
    """Analytic paired responses; exact-kernel pooling has separate tests."""
    entries = sorted(entries, key=lambda entry: entry[0].noise_seed)
    originals = [metric.score(history, count) for history, count in entries]
    names = list(originals[0].component_scores)

    def subset(selected):
        cross, scores = {}, {}
        ranges = tuple((history.noise_seed, fold*(count//fold_count), (fold+1)*(count//fold_count))
                       for fold in selected for history, count in entries)
        for name in names:
            cross[name] = torch.cat([row.component_cross_rows[name][fold*(count//fold_count):
                                                                    (fold+1)*(count//fold_count)]
                                     for fold in selected for row, (_, count) in zip(originals, entries)])
            scores[name] = -2*float(cross[name].mean())
        count = len(cross[names[0]])
        zeros = torch.zeros(count, dtype=torch.float64)
        return KIDResult(sum(scores.values())/len(scores), zeros, sum(cross.values())/len(cross),
                         entries[0][0].noise_seed, 0, count, scores, {name: zeros for name in names}, cross, ranges)

    folds = list(range(fold_count))
    return PartitionScores(subset(folds), [subset([fold]) for fold in folds],
                           [subset([other for other in folds if other != fold]) for fold in folds])


class ControllerTests(unittest.TestCase):
    def run_search(self, *, fresh_worsens=False, disagree=False, failing=False, proposal=.8):
        space, log = FakeSpace(), FakeLog()
        metric = FakeMetric(space, fresh_worsens, disagree)
        bank = metric.noise_bank(1, 16)
        radii = search.RadiusController(1., 1., 1., .1, 10., .1, 10.)

        def propose(fit, *args, **kwargs):
            return torch.tensor([proposal], dtype=torch.float64)

        with (patch.object(QuadraticFit, "propose", propose),
              patch.object(fold_module, "partition_scores", fake_partition),
              patch.object(search, "partition_scores", fake_partition)):
            if failing:
                with patch.object(metric, "extend", side_effect=FloatingPointError("synthetic failure")):
                    with self.assertRaises(FloatingPointError):
                        self.call_search(space, metric, bank, radii, log)
                return space, metric, log, None
            result = self.call_search(space, metric, bank, radii, log)
        return space, metric, log, result

    def call_search(self, space, metric, bank, radii, log):
        return search.search_round(
            space, metric, [torch.ones(1)], bank, radii, log,
            round_index=1, coordinate_scales=torch.ones(1, dtype=torch.float64),
            axis_labels=["mean RGB descent"], sample_count=16, validation_count=16,
            max_candidate_attempts=3, max_refinement_actions=2, proposal_penalty=1.,
            data_folds=4, max_exploratory_attempts=1, fold_regression_penalty=1., fold_regression_z=1.,
            cross_term_ridge_multiplier=10.,
            acceptance_confidence_z=.3, min_kid_improvement=0., minimum_worthwhile_gain=1e-6,
            candidate_minimum_separation=.2, component_regression_tolerance=2e-6,
            component_regression_z=1., fit_ridge=1e-4, covariance_floor=1e-5,
            noise_seeds=search.NoiseSeeds(1))

    def test_one_search_set_and_unused_confirmation_accept_real_improvement(self):
        space, metric, log, result = self.run_search()
        self.assertTrue(result.accepted)
        self.assertEqual(float(space.point[0]), 0.)  # Caller performs the commit.
        origin = next(event for event in log.events if event["event"] == "search_origin")
        self.assertEqual(origin["noise_seeds"], [1])
        finalists = [event for event in log.events if event["event"] == "finalist"]
        self.assertEqual(len(finalists), 1)
        self.assertNotIn(finalists[0]["confirmation_noise_seed"], origin["noise_seeds"])
        line = next(event for event in log.events if event["event"] == "line_search")
        self.assertIn(len(line["alternatives"]), (1, 2))
        self.assertEqual(result.diagnostics["scaled_step_radius"], 1.)
        self.assertLessEqual(result.diagnostics["scaled_step_norm"], 1.+1e-10)

    def test_failed_confirmations_have_finite_attempts_and_never_reuse_noise(self):
        space, metric, log, result = self.run_search(fresh_worsens=True)
        self.assertFalse(result.accepted)
        self.assertEqual(float(space.point[0]), 0.)
        finalists = [event for event in log.events if event["event"] == "finalist"]
        seeds = [event["confirmation_noise_seed"] for event in finalists]
        self.assertLessEqual(len(finalists), 3)
        self.assertEqual(len(seeds), len(set(seeds)))
        self.assertEqual(len(metric.seeds), len(set(metric.seeds)))
        self.assertLessEqual(result.diagnostics["search_bank_count"], 6)
        self.assertLessEqual(metric.generated, result.diagnostics["maximum_generated_images"])

    def test_exception_restores_anchor(self):
        space, _, _, _ = self.run_search(failing=True)
        self.assertEqual(float(space.point[0]), 0.)

    def test_nearly_duplicate_proposal_uses_identical_cached_coordinates(self):
        _, _, log, result = self.run_search(proposal=1.+1e-10)
        self.assertTrue(result.accepted)
        finalist = next(event for event in log.events if event["event"] == "finalist")
        self.assertEqual(finalist["coordinates"], [1.])

    def test_failed_validation_is_promoted_but_next_finalist_uses_new_noise(self):
        _, metric, log, result = self.run_search(fresh_worsens=True)
        promotions = [event for event in log.events if event["event"] == "confirmation_reclassified"]
        self.assertTrue(promotions)
        self.assertTrue(all(event["new_role"] == "promoted_search" for event in promotions))
        self.assertLessEqual(result.diagnostics["refinement_actions"], 2)
        for index, event in enumerate(log.events):
            if event["event"] != "finalist_frozen":
                continue
            known = {row["noise_seed"] for row in log.events[:index] if row["event"] == "evaluation"}
            self.assertNotIn(event["confirmation_noise_seed"], known)
        self.assertLessEqual(metric.generated, result.diagnostics["maximum_generated_images"])

    def test_fold_diagnostics_and_finite_limits_are_recorded(self):
        _, metric, log, _ = self.run_search(fresh_worsens=True)
        limits = next(event for event in log.events if event["event"] == "search_limits")
        folds = [event for event in log.events if event["event"] == "fold_stability"]
        self.assertTrue(folds)
        self.assertEqual(folds[0]["images_per_fold"], 4)
        self.assertEqual(limits["max_exploratory_attempts"], 1)
        self.assertLessEqual(metric.generated, limits["maximum_generated_images"])
        confirmations = [event for event in log.events if event["event"] == "finalist"]
        self.assertTrue(any(event["cached_finalist"] for event in confirmations))

    def test_accepted_boundary_does_not_expand(self):
        controller = search.RadiusController(1., 1., 1., .1, 10., .1, 10.)
        result = search.SearchResult(accepted=True, norm=1., diagnostics=dict(
            scaled_step_norm=1., scaled_step_radius=1., prediction_ratio=10.))
        controller.adapt(result)
        self.assertEqual(controller.probe, 1.)
        self.assertEqual(controller.step, 1.)

    def test_fresh_rejection_alone_does_not_change_radii(self):
        controller = search.RadiusController(1., 1., 1., .1, 10., .1, 10.)
        controller.adapt(search.SearchResult(diagnostics=dict(fit_mismatch=False)))
        self.assertEqual(controller.probe, 1.)
        self.assertEqual(controller.step, 1.)

    def test_no_duplicate_bank_evidence(self):
        space = FakeSpace()
        metric = FakeMetric(space)
        parent_history = SimpleNamespace(point=0., noise_seed=1)
        child_history = SimpleNamespace(point=.5, noise_seed=1)
        parent, child = metric.score(parent_history, 16), metric.score(child_history, 16)
        point = torch.tensor([.5], dtype=torch.float64)
        block = FitBlock([point], parent, [child])
        with self.assertRaisesRegex(ValueError, "counted twice"):
            candidate_evidence(point, [block, block])

    def test_exploration_is_measurement_only_and_can_be_disabled(self):
        fit = QuadraticFit(torch.tensor([-.8e-6, 1e-6], dtype=torch.float64),
                           torch.diag(torch.tensor([1e-12, 1e-20], dtype=torch.float64)),
                           1, 1., 32, 2, 1, 0., 1., 2, diagonal=True)
        bundle = FitBundle(fit, {"first": fit, "second": fit}, ["first", "second"], torch.eye(4))
        landscape = SimpleNamespace(bundle=bundle, training_fits=[], fold_fits=[], evidence=lambda point: None)
        with patch.object(QuadraticFit, "propose", return_value=torch.tensor([.8], dtype=torch.float64)):
            suggestion, _ = search._choose_candidate(landscape, [], 1., 1., 1., 1e-6,
                                                       .2, [], [], [], 2e-6, 1., True)
            self.assertEqual(suggestion[1], "exploratory")
            disabled, _ = search._choose_candidate(landscape, [], 1., 1., 1., 1e-6,
                                                     .2, [], [], [], 2e-6, 1., False)
            self.assertIsNone(disabled)

    def test_fold_downside_distinguishes_agreement_from_confident_regression(self):
        base = dict(delta=-3e-6, se=1e-7, components={})
        agreeing = dict(base, folds=[dict(delta=value, se=1e-7) for value in (-1e-6, -2e-6, -4e-6, -5e-6)])
        conflicting = dict(base, folds=[dict(delta=value, se=1e-7) for value in (-8e-6, -6e-6, -2e-6, 4e-6)])
        first = search._assessment(agreeing, 1., 1e-6, 2e-6, 1.)
        second = search._assessment(conflicting, 1., 1e-6, 2e-6, 1.)
        self.assertEqual(first["fold_downside"], 0.)
        self.assertGreater(first["utility"], second["utility"])
        self.assertEqual(first["se"], second["se"])  # Risk cost never changes SE.

    def test_no_progress_path_consumes_finite_candidate_allowance(self):
        original = search._assessment

        def insufficient(evidence, *args):
            status = original(evidence, *args)
            status["eligible"] = False
            return status

        with patch.object(search, "_assessment", insufficient):
            _, metric, log, result = self.run_search()
        self.assertFalse(result.accepted)
        self.assertLessEqual(result.diagnostics["candidate_attempts"], 3)
        self.assertLessEqual(result.diagnostics["refinement_actions"], 2)
        self.assertLessEqual(metric.generated, result.diagnostics["maximum_generated_images"])


if __name__ == "__main__":
    unittest.main()
