"""Optional CPU-only regressions for paired curvature and physical axis scales.

Run in the configured environment with unittest; these checks need no model,
dataset, feature extractor, or GPU configuration.
"""
import math
import unittest

import torch

from .scaling import AxisScaleController, symmetric_probe_statistics


class SyntheticResult:
    def __init__(self, scores, values, seed=13):
        self.component_scores = dict(zip(("first", "second"), scores))
        self.score = sum(scores)/len(scores)
        self.values = values.double()
        self.noise_seed, self.start, self.count = seed, 0, len(values)

    def leave_one_out(self):
        return self.values.mean(dim=1)

    def component_leave_one_out(self, name):
        return self.values[:, list(self.component_scores).index(name)]

    def require_same_samples(self, other):
        if (self.noise_seed, self.start, self.count) != (other.noise_seed, other.start, other.count):
            raise ValueError("Synthetic paired samples differ")


class SymmetricProbeTests(unittest.TestCase):
    def test_paired_common_and_odd_noise_cancel_before_uncertainty(self):
        count = 7
        index = torch.arange(-3, 4, dtype=torch.float64)
        common = index*1e-3
        odd = index*2e-4
        even = index*1e-8
        parent_values = torch.stack([.01+common, .02+2*common], dim=1)
        odd_values = torch.stack([odd, -3*odd], dim=1)
        even_values = torch.stack([even, 2*even], dim=1)
        cost = torch.tensor([4e-6, 8e-6], dtype=torch.float64)
        parent = SyntheticResult([.01, .02], parent_values)
        plus = SyntheticResult([.01+.002+4e-6, .02-.003+8e-6],
                               parent_values+odd_values+cost+even_values)
        minus = SyntheticResult([.01-.002+4e-6, .02+.003+8e-6],
                                parent_values-odd_values+cost+even_values)
        measured = symmetric_probe_statistics(plus, minus, parent)
        expected_se = math.sqrt((count-1)/count*float(even.square().sum()))
        self.assertAlmostEqual(measured["cost"], 6e-6, delta=1e-14)
        self.assertAlmostEqual(measured["cost_se"], 1.5*expected_se, delta=1e-13)
        self.assertAlmostEqual(measured["components"]["first"]["cost"], 4e-6, delta=1e-14)
        self.assertAlmostEqual(measured["components"]["second"]["cost"], 8e-6, delta=1e-14)
        self.assertAlmostEqual(measured["components"]["first"]["cost_se"], expected_se, delta=1e-13)
        self.assertAlmostEqual(measured["components"]["second"]["cost_se"], 2*expected_se, delta=1e-13)
        self.assertGreater(measured["cost"], 2*measured["cost_se"])
        # An unpaired error calculation would retain the much larger common noise.
        self.assertLess(measured["cost_se"], float(common.std())/1000)

    def test_different_noise_banks_cannot_form_a_symmetric_contrast(self):
        values = torch.zeros(5, 2, dtype=torch.float64)
        parent = SyntheticResult([0., 0.], values)
        plus = SyntheticResult([1e-5, 1e-5], values, seed=14)
        minus = SyntheticResult([1e-5, 1e-5], values)
        with self.assertRaisesRegex(ValueError, "samples differ"):
            symmetric_probe_statistics(plus, minus, parent)


class AxisScaleTests(unittest.TestCase):
    labels = ["mean RGB descent", "leading RGB residual SVD", "remaining-gradient residual"]

    def test_initial_factors_are_semantic_and_returned_vector_is_owned(self):
        controller = AxisScaleController()
        factors = controller.factors_for(self.labels)
        torch.testing.assert_close(factors, torch.tensor([.25, 1.5, 4.], dtype=torch.float64))
        self.assertEqual(factors.device.type, "cpu")
        factors[0] = 99.
        self.assertEqual(float(controller.factors_for(self.labels)[0]), .25)
        custom = AxisScaleController(initial_mean_factor=.5, initial_residual_factor=.3,
                                     initial_remaining_factor=2.)
        torch.testing.assert_close(custom.factors_for(self.labels),
                                   torch.tensor([.5, .3, 2.], dtype=torch.float64))

    def test_positive_resolved_cost_moves_toward_target(self):
        controller = AxisScaleController()
        rows = controller.update([
            dict(axis_label=self.labels[0], cost=20e-6, cost_se=1e-7, physical_radius=1.),
            # The actual factor is .1 despite the stored initial value being .2.
            dict(axis_label=self.labels[1], cost=1.25e-6, cost_se=1e-7, physical_radius=.4),
        ], base_radius=4.)
        self.assertAlmostEqual(rows[0]["current_factor"], .25)
        self.assertAlmostEqual(rows[0]["next_factor"], .125)
        self.assertAlmostEqual(rows[1]["current_factor"], .1)
        self.assertAlmostEqual(rows[1]["next_factor"], .2)
        self.assertTrue(all(row["resolved_positive_cost"] for row in rows))

    def test_unresolved_zero_and_negative_cost_hold_actual_factor(self):
        for cost, se in ((1e-6, 1e-6), (2e-6, 1e-6), (0., 0.), (-1e-5, 1e-7)):
            with self.subTest(cost=cost, se=se):
                controller = AxisScaleController()
                row = controller.update([dict(axis_label=self.labels[0], cost=cost,
                                             cost_se=se, physical_radius=.3)], 1.)[0]
                self.assertFalse(row["resolved_positive_cost"])
                self.assertAlmostEqual(row["next_factor"], .3)
                self.assertAlmostEqual(float(controller.factors_for([self.labels[0]])[0]), .3)

    def test_factor_bounds_and_per_update_limits(self):
        cases = ((1., 80e-6, 1.5, 2/3),
                 (1., 5e-8, 1.5, 1.5),
                 (.05, 1e-3, 2., .05),
                 (4., 1e-9, 2., 4.))
        for current, cost, limit, expected in cases:
            with self.subTest(current=current, cost=cost):
                controller = AxisScaleController(max_change=limit)
                row = controller.update([dict(axis_label=self.labels[2], cost=cost,
                                             cost_se=0., physical_radius=current)], 1.)[0]
                self.assertAlmostEqual(row["next_factor"], expected)
                self.assertGreaterEqual(row["next_factor"], controller.min_factor)
                self.assertLessEqual(row["next_factor"], controller.max_factor)
                self.assertGreaterEqual(row["next_factor"], current/limit-1e-14)
                self.assertLessEqual(row["next_factor"], current*limit+1e-14)

    def test_invalid_statistics_do_not_partially_update_existing_axes(self):
        controller = AxisScaleController()
        before = controller.factors_for(self.labels).clone()
        with self.assertRaises(ValueError):
            controller.update([
                dict(axis_label=self.labels[0], cost=20e-6, cost_se=0., physical_radius=.25),
                dict(axis_label=self.labels[1], cost=20e-6, cost_se=-1., physical_radius=.2),
            ], 1.)
        torch.testing.assert_close(controller.factors_for(self.labels), before)
