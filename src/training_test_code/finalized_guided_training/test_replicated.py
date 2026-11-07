"""Optional CPU regressions for replicated fitting, guards and gradient history."""
import math
import unittest

import torch

from .evidence import combine_estimates, component_regressions, regression_cost
from .gradient_history import GradientSecondMoment
from .parameter_space import orthonormal_basis, vector_dot, vector_norm
from .quadratic import FitBlock, coordinate_design, design, fit_quadratic
from .test_math import SyntheticResult


class ReplicatedMathTests(unittest.TestCase):
    def test_independent_aggregation_preserves_counts_and_scatter(self):
        value, error = combine_estimates([(2., 1., 8), (2., 1., 8)])
        self.assertEqual(value, 2.)
        self.assertAlmostEqual(error, 1/math.sqrt(2))
        value, error = combine_estimates([(-2., .01, 8), (2., .01, 8)])
        self.assertEqual(value, 0.)
        self.assertAlmostEqual(error, 2.)
        value, _ = combine_estimates([(1., 1., 8), (3., 1., 24)])
        self.assertEqual(value, 2.5)

    def test_two_diagonal_banks_recover_slopes_and_physical_curvature(self):
        generator = torch.Generator().manual_seed(34)
        scales = torch.tensor([.1, .2, .4], dtype=torch.float64)
        points = coordinate_design(3, scales, diagonal=True)
        self.assertEqual(len(points), 6)
        rows = design(torch.stack(points), scales, diagonal=True)
        self.assertEqual(rows.shape, (6, 6))
        beta = torch.tensor([-.2, .1, -.3, 1., 2., 3.], dtype=torch.float64)*1e-4
        blocks = []
        for seed in (1, 2):
            parent = SyntheticResult([.01, .02], torch.zeros(96, 2), seed=seed)
            results = []
            for row in rows:
                change = float(row @ beta)
                noise = torch.randn(96, 2, generator=generator, dtype=torch.float64)*1e-7
                scores = [.01+change, .02+change]
                results.append(SyntheticResult(scores, noise+torch.tensor(scores), seed=seed))
            blocks.append(FitBlock(points, parent, results))
        bundle = fit_quadratic(blocks, scales, ridge=0., diagonal=True)
        fit = bundle.combined
        torch.testing.assert_close(fit.coefficients, beta, rtol=1e-7, atol=1e-10)
        torch.testing.assert_close(fit.gradient, beta[:3]/scales, rtol=1e-7, atol=1e-10)
        torch.testing.assert_close(fit.hessian, torch.diag(beta[3:]/scales.square()), rtol=1e-7, atol=1e-10)
        self.assertTrue(fit.diagonal)
        self.assertEqual(fit.block_count, 2)
        with self.assertRaisesRegex(ValueError, "Overlapping"):
            fit_quadratic([blocks[0], blocks[0]], scales, diagonal=True)

    def test_component_guard_allows_uncertainty_but_rejects_material_regression(self):
        components = dict(first=dict(delta=-10e-6, se=1e-6), second=dict(delta=8e-6, se=1e-6))
        self.assertEqual(component_regressions(components, 2e-6, 1.), ["second"])
        self.assertAlmostEqual(regression_cost(components, 2e-6), 6e-6)
        components["second"]["se"] = 10e-6
        self.assertEqual(component_regressions(components, 2e-6, 1.), [])

    def test_second_moment_uses_raw_squares_and_current_mean(self):
        history = GradientSecondMoment(decay=.5, damping=.05)
        raw = [torch.tensor([3., 4.]), torch.tensor([-3., 4.])]
        directions = [value/vector_norm(value) for value in raw]
        first, info = history.direction(directions, [5., 5.])
        torch.testing.assert_close(history.second, torch.tensor([4.5, 8.]))
        torch.testing.assert_close(first, torch.tensor([0., -1.]))
        raw = [torch.tensor([0., -2.]), torch.tensor([0., -2.])]
        second, info = history.direction([value/2 for value in raw], [2., 2.])
        # Current gradients reversed; the numerator must not retain momentum.
        torch.testing.assert_close(second, torch.tensor([0., 1.]))
        self.assertEqual(info["updates"], 2)
        self.assertEqual(history.second.device.type, "cpu")

    def test_optional_axis_is_orthogonal_to_mean_and_retained_tail(self):
        mean = torch.tensor([8., 4., 2., 1., .5, .2])
        vectors = [mean/vector_norm(mean)]*2
        second = torch.tensor([0., 1., -1., 2., -3., 1.])
        basis, info = orthonormal_basis(vectors, [vector_norm(mean)]*2,
                                        removed_gradient_fraction=.2, second_direction=second)
        self.assertEqual(len(basis), 3)
        self.assertEqual(info["axis_labels"][1], "preconditioned RGB residual")
        gram = torch.tensor([[vector_dot(a, b) for b in basis] for a in basis])
        torch.testing.assert_close(gram, torch.eye(3), atol=1e-6, rtol=1e-6)


if __name__ == "__main__":
    unittest.main()
