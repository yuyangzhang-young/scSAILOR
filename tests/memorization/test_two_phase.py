"""The two-phase retraining's readings on small hand-made arrays; no study data.

    python -m pytest tests/memorization/test_two_phase.py
"""
import unittest

import numpy as np

from scsailor.resolution import comparison
from reproduce.memorization import two_phase
from reproduce.memorization.data import Design

PLATES = ('plate6', 'plate14')


def units(lines=4, drugs=3, genes=5, seed=0):
    """Every drug on every context (a line on one of the two plates), random truth and weights."""
    rng = np.random.default_rng(seed)
    contexts = [f'L{i}|{p}' for i in range(lines) for p in PLATES]
    context = np.repeat(contexts, drugs)
    n = len(context)
    return {'truth': rng.uniform(0, 2, (n, genes)), 'context': context,
            'p_code': np.tile(np.arange(drugs), len(contexts)), 'weight': rng.uniform(.5, 1.5, n)}


class OutputEnergy(unittest.TestCase):
    def test_is_the_energy_of_the_drug_differences_around_the_anchor(self):
        u = units()
        D = Design(u)
        rng = np.random.default_rng(1)
        n, genes = u['truth'].shape
        A = rng.normal(size=(n, genes))
        x = rng.normal(size=(n, genes))
        x -= D.project_ctx(x)   # drug differences: zero weighted mean over each context's drugs
        shift = D.project_ctx(rng.normal(size=(n, genes)))   # a context-level part, ignored
        expected = 1e3 * np.sum(D.w * np.mean(x * x, axis=1))
        self.assertAlmostEqual(two_phase.output_energy(A + shift + x, A, D), expected)


class SeedRow(unittest.TestCase):
    def setUp(self):
        self.u = units()
        self.D = Design(self.u)
        rng = np.random.default_rng(2)
        shape = self.u['truth'].shape
        self.Q, self.F = rng.normal(1, 1, shape), rng.normal(1, 1, shape)
        self.base = 3.

    def test_errors_in_percent_of_the_base_and_delta_their_difference(self):
        row, _ = two_phase.seed_row(self.Q, self.F, self.D, self.u['truth'], self.base)
        T, w = self.u['truth'], self.D.w

        def pct(P):
            return 100 * np.sum(w * np.mean((np.maximum(P, 0) - T) ** 2, axis=1)) / self.base

        self.assertAlmostEqual(row['two_phase_readers_pct_of_design_aware'], pct(self.Q))
        self.assertAlmostEqual(row['original_readers_pct_of_design_aware'], pct(self.F))
        self.assertAlmostEqual(row['delta_points'], pct(self.Q) - pct(self.F))
        self.assertLessEqual(row['delta_low'], row['delta_high'])

    def test_seed_mean_averages_the_seeds_and_resamples_lines_with_seeds_pooled(self):
        rng = np.random.default_rng(3)
        rows, per_line = [], []
        for _ in range(3):
            Q = self.Q + rng.normal(0, .5, self.Q.shape)
            row, lines = two_phase.seed_row(Q, self.F, self.D, self.u['truth'], self.base)
            rows.append(row)
            per_line.append(lines)
        mean = two_phase.seed_mean_row(rows, per_line, self.base)
        for k in ('two_phase_readers_pct_of_design_aware', 'original_readers_pct_of_design_aware'):
            self.assertAlmostEqual(mean[k], np.mean([r[k] for r in rows]))
        pooled = {line: np.mean([p[line] for p in per_line]) for line in per_line[0]}
        iv = comparison.bootstrap(pooled, self.base)
        self.assertAlmostEqual(mean['delta_points'], np.mean([r['delta_points'] for r in rows]))
        self.assertEqual([mean['delta_low'], mean['delta_high']], iv['interval_pct'])


if __name__ == '__main__':
    unittest.main()
