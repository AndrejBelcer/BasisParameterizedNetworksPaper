"""Optional CPU checks of pooled kernels, pairing, covariance and fold fits."""
from types import SimpleNamespace
import unittest

import torch

from .folds import partition_scores, delete_fold_covariance, FoldLandscape
from .kid_evaluator import FeatureHistory, KIDEvaluator, paired_difference
from .quadratic import FitBlock, QuadraticFit, fit_quadratic


class FoldTests(unittest.TestCase):
    def setUp(self):
        self.metric = KIDEvaluator.__new__(KIDEvaluator)
        self.metric.runtime = SimpleNamespace(device=torch.device("cpu"), synchronize=lambda: None)
        self.metric.extractors = {"first": None, "second": None}
        self.metric.kid_kernel_block_size = 5
        self.metric.kernel_seconds, self.metric.score_evaluations = 0., 0
        generator = torch.Generator().manual_seed(420)
        self.metric.references = {name: torch.randn(19, 4, generator=generator, dtype=torch.float64)*.2
                                  for name in self.metric.extractors}
        self.metric.reference_constants = {}
        for name, features in self.metric.references.items():
            kernel = self.kernel(features, features)
            self.metric.reference_constants[name] = float((kernel.sum()-kernel.diag().sum())/(19*18))
        self.histories = [FeatureHistory(seed, features={
            name: torch.randn(24, 4, generator=generator, dtype=torch.float64)*.2+seed*.03
            for name in self.metric.extractors}) for seed in (101, 102)]

    @staticmethod
    def kernel(first, second):
        return (first @ second.T/first.shape[1]+1).pow(3)

    def assert_exact(self, result, histories):
        by_seed = {history.noise_seed: history for history in histories}
        component_scores = []
        for name in self.metric.extractors:
            features = torch.cat([by_seed[seed].features[name][start:end]
                                  for seed, start, end in result.intervals])
            kernel = self.kernel(features, features)
            kernel.diagonal().zero_()
            cross = self.kernel(features, self.metric.references[name]).mean(dim=1)
            expected = float(kernel.sum()/(len(features)*(len(features)-1))-2*cross.mean()
                             +self.metric.reference_constants[name])
            self.assertAlmostEqual(result.component_scores[name], expected, places=10)
            torch.testing.assert_close(result.component_self_rows[name], kernel.sum(dim=1))
            torch.testing.assert_close(result.component_cross_rows[name], cross)
            component_scores.append(expected)
        self.assertAlmostEqual(result.score, sum(component_scores)/len(component_scores), places=10)

    def test_pooled_and_every_subset_include_exact_kernel_pairs(self):
        result = partition_scores(self.metric, [(history, 24) for history in self.histories], 4)
        for row in [result.full]+result.folds+result.left_out:
            self.assert_exact(row, self.histories)
        self.assertEqual(result.full.count, 48)
        self.assertEqual([row.count for row in result.folds], [12]*4)
        self.assertEqual([row.count for row in result.left_out], [36]*4)
        block_mean = sum(self.metric.score(history, 24).score for history in self.histories)/2
        self.assertGreater(abs(result.full.score-block_mean), 1e-10)
        cached = partition_scores(self.metric, [(history, 24) for history in reversed(self.histories)], 4)
        self.assertIs(cached, result)

    def test_pairing_and_overlap_use_real_sample_ranges(self):
        result = partition_scores(self.metric, [(history, 24) for history in self.histories], 4)
        self.assertEqual(paired_difference(result.full, result.full), (0., 0.))
        with self.assertRaisesRegex(ValueError, "same noise"):
            paired_difference(result.folds[0], result.folds[1])
        with self.assertRaisesRegex(ValueError, "counted twice"):
            partition_scores(self.metric, [(self.histories[0], 24)]*2, 4)
        point = torch.tensor([1.], dtype=torch.float64)
        first = FitBlock([point], result.full, [result.full])
        overlapping = FitBlock([point], result.folds[0], [result.folds[0]])
        with self.assertRaisesRegex(ValueError, "Overlapping"):
            fit_quadratic([first, overlapping], 1., diagonal=True)

    def test_delete_fold_scaling_and_pointwise_max_do_not_double_count(self):
        values = torch.tensor([[3.], [8/3], [7/3], [2.]], dtype=torch.float64)
        self.assertAlmostEqual(float(delete_fold_covariance(values)[0, 0]), 5/12)
        fit = QuadraticFit(torch.zeros(2, dtype=torch.float64), torch.diag(torch.tensor([4., 1.], dtype=torch.float64)),
                           1, 1., 48, 2, 1, 0., 1., 2, diagonal=True)
        fit.resampling_covariance = torch.diag(torch.tensor([1., 16.], dtype=torch.float64))
        point = torch.tensor([1.], dtype=torch.float64)
        # Design [1, .5]: paired variance=4.25, fold variance=5; use 5, not 9.25.
        self.assertAlmostEqual(fit.reduction_se(point)**2, 5.)

    def test_landscape_does_not_count_overlapping_fits_as_new_data(self):
        pools = []
        for parent in self.histories:
            candidates = []
            for value in (-1., 1.):
                history = FeatureHistory(parent.noise_seed, features={name: features+value*.01
                                                                     for name, features in parent.features.items()})
                candidates.append(SimpleNamespace(coordinates=torch.tensor([value], dtype=torch.float64), history=history))
            pools.append(SimpleNamespace(count=24, bank=SimpleNamespace(seed=parent.noise_seed),
                                         incumbent=SimpleNamespace(history=parent), candidates=[None]+candidates))
        landscape = FoldLandscape(self.metric, pools, torch.ones(1, dtype=torch.float64), 4, 1e-4, 1e-5)
        self.assertEqual(len(landscape.blocks), 1)
        self.assertEqual(landscape.bundle.combined.image_count, 48)
        self.assertFalse(landscape.bundle.observation_details["residual_variance_estimable"])
        self.assertEqual(landscape.bundle.combined.variance_inflation, 1.)
        self.assertEqual([fit.combined.image_count for fit in landscape.training_fits], [36]*4)
        evidence = landscape.evidence(torch.tensor([1.], dtype=torch.float64))
        self.assertEqual(len(evidence["folds"]), 4)
        self.assertEqual(len(evidence["banks"]), 2)
        self.assertEqual(evidence["se"], max(evidence["paired_se"], evidence["fold_se"]))


if __name__ == "__main__":
    unittest.main()
