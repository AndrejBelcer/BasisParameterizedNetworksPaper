"""Optional mathematical regressions; no model, dataset, or GPU initialization.

The module path is src.training_test_code.finalized_guided_training.test_math.
"""
import math
import unittest

import torch

from .parameter_space import orthonormal_basis, vector_dot, vector_norm
from .quadratic import (FitBlock, QuadraticFit, axis_scale_vector, coefficient_count,
                        coordinate_design, design, fit_quadratic)


class SyntheticResult:
    """Only the paired result interface needed by the quadratic fitter."""
    def __init__(self, scores, jackknife, seed=123):
        self.component_scores = dict(zip(("first", "second"), scores))
        self.score = sum(scores)/len(scores)
        self.values = jackknife.double()
        self.noise_seed, self.start, self.count = seed, 0, len(jackknife)

    def component_leave_one_out(self, name):
        index = list(self.component_scores).index(name)
        return self.values[:, index]

    def leave_one_out(self):
        return self.values.mean(dim=1)

    def require_same_samples(self, other):
        if (self.noise_seed, self.start, self.count) != (other.noise_seed, other.start, other.count):
            raise ValueError("Synthetic paired samples differ")


def coefficients_for(gradient, hessian, scale):
    d = len(gradient)
    scales = axis_scale_vector(scale, d)
    mixed = [hessian[first, second]*scales[first]*scales[second] for first in range(d)
             for second in range(first+1, d)]
    pieces = [gradient*scales, hessian.diag()*scales.square()]
    if mixed:
        pieces.append(torch.stack(mixed))
    return torch.cat(pieces)


class QuadraticMathTests(unittest.TestCase):
    def test_physical_gradient_hessian_and_mixed_terms(self):
        for dimension in (1, 2, 3):
            with self.subTest(dimension=dimension):
                scale = .07
                gradient = torch.tensor([.2, -.5, .3], dtype=torch.float64)[:dimension]
                hessian = torch.tensor([[2., .4, -.3], [.4, 3., .6], [-.3, .6, 4.]],
                                       dtype=torch.float64)[:dimension, :dimension]
                coefficients = coefficients_for(gradient, hessian, scale)
                width = coefficient_count(dimension)
                fit = QuadraticFit(coefficients, torch.eye(width, dtype=torch.float64),
                                   dimension, scale, 96, 12, 1, 0., 1., width)
                torch.testing.assert_close(fit.gradient, gradient)
                torch.testing.assert_close(fit.hessian, hessian)
                torch.testing.assert_close(fit.hessian_se,
                                           torch.full_like(hessian, 1/scale**2))
                probes = coordinate_design(dimension, scale)
                self.assertEqual(len(probes), dimension*(dimension+1))
                matrix = design(torch.stack(probes), scale)
                self.assertEqual(int(torch.linalg.matrix_rank(matrix)), width)
                for point in probes:
                    expected = float(gradient @ point+.5*point @ hessian @ point)
                    self.assertAlmostEqual(fit.reduction(point), -expected, places=13)
                    self.assertAlmostEqual(vector_norm(point), scale, places=13)

    def test_joint_fit_recovers_components_and_correlated_combination(self):
        generator = torch.Generator().manual_seed(1123)
        d, count = 3, 96
        scale = torch.tensor([.02, .04, .08], dtype=torch.float64)
        coordinates = coordinate_design(d, scale)
        rows = design(torch.stack(coordinates), scale)
        width = coefficient_count(d)
        first = torch.linspace(-2e-4, 3e-4, width, dtype=torch.float64)
        second = torch.linspace(1e-4, -4e-4, width, dtype=torch.float64)
        parent = SyntheticResult([0., 0.], torch.zeros(count, 2, dtype=torch.float64))
        results = []
        for row in rows:
            shared = torch.randn(count, generator=generator, dtype=torch.float64)*1e-5
            independent = torch.randn(count, generator=generator, dtype=torch.float64)*1e-5
            # Strong positive metric correlation should survive averaging.
            loo = torch.stack([shared, .8*shared+.6*independent], dim=1)
            scores = [float(row @ first), float(row @ second)]
            loo += torch.tensor(scores, dtype=torch.float64)
            results.append(SyntheticResult(scores, loo))
        block = FitBlock(coordinates, parent, results)
        bundle = fit_quadratic([block], scale, ridge=0.)
        torch.testing.assert_close(bundle.components["first"].coefficients, first, atol=1e-10, rtol=1e-8)
        torch.testing.assert_close(bundle.components["second"].coefficients, second, atol=1e-10, rtol=1e-8)
        torch.testing.assert_close(bundle.combined.coefficients, (first+second)/2, atol=1e-10, rtol=1e-8)
        average = torch.cat([torch.eye(width, dtype=torch.float64)/2]*2, dim=1)
        expected = average @ bundle.joint_covariance @ average.T
        torch.testing.assert_close(bundle.combined.covariance, expected)
        diagonal_only = (bundle.components["first"].covariance+
                         bundle.components["second"].covariance)/4
        self.assertGreater(float((expected-diagonal_only).abs().max()), 1e-16)
        with self.assertRaisesRegex(ValueError, "Overlapping"):
            fit_quadratic([block, block], scale, ridge=0.)

    def test_cross_terms_get_stronger_ridge_without_relabeling_variance(self):
        points = coordinate_design(3, 1.)
        matrix = design(torch.stack(points), 1.)
        count = 64
        generator = torch.Generator().manual_seed(610)
        beta = torch.linspace(-2e-5, 3e-5, 9, dtype=torch.float64)
        parent = SyntheticResult([0., 0.], torch.zeros(count, 2, dtype=torch.float64))
        children = []
        for row in matrix:
            value = float(row @ beta)
            noise = torch.randn(count, 2, generator=generator, dtype=torch.float64)*1e-7
            children.append(SyntheticResult([value, value], noise+value))
        bundle = fit_quadratic([FitBlock(points, parent, children)], 1.,
                               ridge=1e-3, cross_term_ridge_multiplier=10.)
        penalties = torch.tensor(bundle.observation_details["ridge_penalty_diagonal"])
        for offset in (0, 9):
            torch.testing.assert_close(penalties[offset:offset+6], penalties[offset].expand(6))
            torch.testing.assert_close(penalties[offset+6:offset+9], (10*penalties[offset]).expand(3))
        self.assertLess(bundle.observation_details["ridge_effective_parameter_count"], 18.)

    def test_fold_downside_is_a_risk_cost_not_extra_sampling_variance(self):
        covariance = torch.eye(2, dtype=torch.float64)*1e-16
        fit = QuadraticFit(torch.tensor([-4e-6, 0.], dtype=torch.float64), covariance,
                           1, 1., 16, 2, 1, 0., 1., 2, diagonal=True)
        bad = QuadraticFit(torch.tensor([2e-6, 0.], dtype=torch.float64), covariance,
                           1, 1., 4, 2, 1, 0., 1., 2, diagonal=True)
        point = torch.ones(1, dtype=torch.float64)
        se = fit.reduction_se(point)
        utility = fit.utility(point, 1.)
        fit.fold_models = [fit, bad]
        fit.fold_regression_penalty = 1.
        self.assertEqual(fit.reduction_se(point), se)
        self.assertLess(fit.utility(point, 1.), utility)

    def test_acquisition_is_finite_and_bounded(self):
        scale, radius = .05, 1.6
        gradient = torch.tensor([.2, -.5, .3], dtype=torch.float64)
        hessian = torch.diag(torch.tensor([2., 3., 4.], dtype=torch.float64))
        coefficients = coefficients_for(gradient, hessian, scale)
        width = coefficient_count(3)
        fit = QuadraticFit(coefficients, torch.eye(width, dtype=torch.float64)*1e-10,
                           3, scale, 96, 12, 1, 0., 1., width)
        point = fit.propose(radius, penalty=.3, iterations=8)
        self.assertTrue(bool(torch.isfinite(point).all()))
        self.assertLessEqual(vector_norm(point), scale*radius*(1+1e-12))
        self.assertGreaterEqual(fit.utility(point, .3), -1e-12)

    def test_vector_scales_and_acquisition_exclusions(self):
        scales = torch.tensor([.02, .05, .08], dtype=torch.float64)
        gradient = torch.tensor([.2, -.5, .3], dtype=torch.float64)
        hessian = torch.tensor([[2., .4, -.3], [.4, 3., .6], [-.3, .6, 4.]], dtype=torch.float64)
        coefficients = coefficients_for(gradient, hessian, scales)
        width = coefficient_count(3)
        fit = QuadraticFit(coefficients, torch.eye(width, dtype=torch.float64)*1e-10,
                           3, scales, 96, 12, 1, 0., 1., width)
        torch.testing.assert_close(fit.gradient, gradient)
        torch.testing.assert_close(fit.hessian, hessian)
        expected_se = 1e-5/(scales[:, None]*scales[None, :])
        torch.testing.assert_close(fit.hessian_se, expected_se)
        for point in coordinate_design(3, scales):
            self.assertAlmostEqual(vector_norm(point/scales), 1., places=13)
            expected = float(gradient @ point+.5*point @ hessian @ point)
            self.assertAlmostEqual(fit.reduction(point), -expected, places=13)
        radius, hard_radius = 1.6, .06
        first = fit.propose(radius, .3, iterations=8, hard_radius=hard_radius)
        second = fit.propose(radius, .3, iterations=8, forbidden=[first],
                             separation=.2, hard_radius=hard_radius)
        self.assertGreater(vector_norm(second), 0.)
        self.assertLessEqual(vector_norm(second/scales), radius*(1+1e-10))
        self.assertLessEqual(vector_norm(second), hard_radius*(1+1e-10))
        a, b = first/scales, second/scales
        exclusion = .2*min(radius, max(min(vector_norm(a), vector_norm(b)), .05*radius))
        self.assertGreater(vector_norm(a-b), exclusion)

    def test_all_gradient_coordinates_do_not_manufacture_third_axis(self):
        mean = torch.tensor([4., 3., 2., 1., .4, .3, .2, .1])
        residual = torch.tensor([.1, -.2, .3, -.1, .2, .1, -.3, .2])
        raw = [mean+residual, mean-residual]
        norms = [vector_norm(value) for value in raw]
        guides = [value/norm for value, norm in zip(raw, norms)]
        basis, details = orthonormal_basis(guides, norms, removed_gradient_fraction=1.)
        self.assertEqual(len(basis), 2)
        self.assertEqual(details["removed_coordinate_count"], len(mean))
        self.assertEqual(details["remaining_coordinate_count"], 0)
        self.assertEqual(details["third_input_norm"], 0.)

    def test_mask_is_global_and_final_basis_is_orthonormal(self):
        mean = torch.tensor([4., 3., 2., 1., .4, .3, .2, .1])
        residual = torch.tensor([.1, -.2, .3, -.1, .2, .1, -.3, .2])
        other = torch.tensor([-.2, .1, .1, .2, -.1, .3, .2, -.3])
        raw = [mean+residual, mean-residual, mean+other, mean-other]
        norms = [vector_norm(value) for value in raw]
        guides = [value/norm for value, norm in zip(raw, norms)]
        basis, details = orthonormal_basis(guides, norms, removed_gradient_fraction=.25)
        self.assertEqual(len(basis), 3)
        self.assertEqual(details["removed_coordinate_count"], math.ceil(.25*len(mean)))
        self.assertAlmostEqual(details["removed_energy_fraction"], 25/vector_norm(mean)**2, places=6)
        self.assertAlmostEqual(details["remaining_energy_fraction"], 1-25/vector_norm(mean)**2, places=6)
        gram = torch.tensor([[vector_dot(a, b) for b in basis] for a in basis], dtype=torch.float64)
        torch.testing.assert_close(gram, torch.eye(3, dtype=torch.float64), atol=1e-6, rtol=1e-6)
        removed = torch.zeros_like(mean)
        removed[:2] = -mean[:2]
        remaining = -mean-removed
        for _ in range(2):
            for axis in basis[:2]:
                removed.add_(axis, alpha=-vector_dot(removed, axis))
                remaining.add_(axis, alpha=-vector_dot(remaining, axis))
        # Complement masking changes the projected sign, not the search span.
        torch.testing.assert_close(remaining, -removed, atol=1e-6, rtol=1e-6)
        remaining /= vector_norm(remaining)
        self.assertGreater(vector_dot(remaining, basis[2]), 1-1e-6)


if __name__ == "__main__":
    unittest.main()
