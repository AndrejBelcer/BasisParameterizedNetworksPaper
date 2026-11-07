"""Optional gate/distance regressions; no model, dataset or GPU setup required."""
import unittest

import torch

from .search_policy import search_eligibility, same_location


class SearchPolicyTests(unittest.TestCase):
    def test_reported_small_gain_has_no_second_worthwhile_margin(self):
        # This would fail the old gain-.3*SE>1e-6 gate despite passing selection.
        evidence = search_eligibility(1.66e-6, 3e-6, .3, 1e-6)
        self.assertTrue(evidence["normal"])
        self.assertFalse(evidence["marginal"])
        self.assertLess(evidence["utility"], 1e-6)

    def test_positive_marginal_must_pass_one_condition(self):
        cases = [(0.5e-6, 0.1e-6, True), (1.2e-6, 5e-6, True),
                 (0.5e-6, 3e-6, False), (-1e-6, 0.1e-6, False), (0., 0., False)]
        for gain, error, expected in cases:
            with self.subTest(gain=gain, error=error):
                result = search_eligibility(gain, error, .3, 1e-6)
                self.assertFalse(result["normal"])
                self.assertEqual(result["marginal"], expected)

    def test_scaled_distance_is_invariant_to_physical_coordinate_units(self):
        scales = torch.tensor([.1, 10.], dtype=torch.float64)
        first = torch.tensor([.05, 0.], dtype=torch.float64)
        near = torch.tensor([.055, 0.], dtype=torch.float64)
        far = torch.tensor([0., 5.], dtype=torch.float64)
        conversion = torch.tensor([1e4, .01], dtype=torch.float64)
        self.assertTrue(same_location(first, near, scales, 1., .2))
        self.assertFalse(same_location(first, far, scales, 1., .2))
        for second in (near, far):
            self.assertEqual(same_location(first, second, scales, 1., .2),
                             same_location(first*conversion, second*conversion, scales*conversion, 1., .2))


if __name__ == "__main__":
    unittest.main()
