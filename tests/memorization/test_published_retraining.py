"""The target replacements of the published models' retraining on small hand-made arrays; no study data.

    python -m pytest tests/memorization/test_published_retraining.py
"""
import importlib.util
import unittest

import numpy as np

TORCH = importlib.util.find_spec('torch') is not None
if TORCH:
    from reproduce.memorization import published_retraining as pr
    from reproduce.memorization.retraining import context_levels

GENES = 3


class FakeData:
    """Four training units in two contexts, with control pools, unit weights, drug codes and the drug average."""

    def __init__(self):
        rng = np.random.default_rng(0)
        self.parent = np.array([0, 0, 1, 1])
        self.pcode = np.array([0, 1, 0, 1])
        self.prob = np.array([.1, .3, .2, .4])
        self.r = rng.normal(size=(2, GENES)).astype(np.float32)
        self.target = rng.normal(size=(4, GENES)).astype(np.float32)
        self.pools = {c: (rng.normal(size=(6, GENES)).astype(np.float32), np.arange(6)) for c in (0, 1)}

    def selected_rows(self, step, seed):
        return np.array([step % 4, (step + 1) % 4])


class FakeStream:
    """Two units per update, two treated cells per unit; the input set is two cells."""

    def __init__(self, data):
        self.data, self.seed, self.state = data, 7, {'drawn': 0}
        rng = np.random.default_rng(1)
        self.atoms = {s: rng.normal(size=(2, 2, GENES)).astype(np.float32) for s in range(1, 5)}
        self.cells = {s: rng.normal(size=(4, GENES)).astype(np.float32) for s in range(1, 5)}

    def batches(self, first, stop, workers):
        for step in range(first, stop + 1):
            yield step, {'atoms': self.atoms[step], 'target': self.cells[step].copy(), 'codes': np.zeros(2)}

    def state_dict(self):
        return self.state

    def load_state_dict(self, state):
        self.state = state


def targets(stream, change):
    return {s: b['target'] for s, b in pr.Retargeted(stream, change, cells=2).batches(1, 4, 1)}


@unittest.skipUnless(TORCH, 'the training layer needs torch')
class Replacement(unittest.TestCase):
    def setUp(self):
        self.data = FakeData()
        self.stream = FakeStream(self.data)

    def test_planted_shift_adds_each_units_shift_to_all_its_treated_cells_in_single_precision(self):
        shift = np.arange(4 * GENES, dtype=np.float32).reshape(4, GENES) / 7
        got = targets(self.stream, pr.shifted(shift))
        for step, t in got.items():
            rows = self.data.selected_rows(step, 7)
            expected = self.stream.cells[step].reshape(2, 2, GENES) + shift[rows][:, None]
            self.assertEqual(t.dtype, np.float32)
            self.assertTrue(np.array_equal(t, expected.reshape(4, GENES)))

    def test_task_shift_is_the_task_target_minus_the_natural_target(self):
        task = self.data.target + np.float32(.25)
        shift = pr.task_shift(self.data.target, task)
        self.assertEqual(shift.dtype, np.float32)
        self.assertTrue(np.allclose(shift, .25))

    def test_readable_only_shift_removes_the_unreadable_part_of_each_units_context(self):
        E = np.array([[1., 2., 3.], [4., 5., 6.]])
        shift = pr.unreadable_shift(np.array(['A|p1', 'B|p1']), E, np.array(['B|p1', 'A|p1', 'B|p1']))
        self.assertTrue(np.array_equal(shift, -E[[1, 0, 1]].astype(np.float32)))

    def test_input_relative_subtracts_pool_mean_minus_input_set_mean_from_every_treated_cell(self):
        got = targets(self.stream, pr.input_relative(self.data))
        for step, t in got.items():
            rows = self.data.selected_rows(step, 7)
            for slot, row in enumerate(rows):
                pool = self.data.pools[self.data.parent[row]][0].astype(np.float64).mean(0)
                gap = pool - self.stream.atoms[step][slot].astype(np.float64).mean(0)
                cells = self.stream.cells[step][2 * slot:2 * slot + 2]
                self.assertTrue(np.allclose(t[2 * slot:2 * slot + 2], cells - gap, atol=1e-6))

    def test_context_level_is_pool_mean_plus_the_weighted_offset_beyond_pool_and_drug_average(self):
        level = context_levels(self.data, self.data.r)
        d = self.data
        for c in (0, 1):
            pool = d.pools[c][0].astype(np.float64).mean(0)
            units = d.parent == c
            rest = d.target[units] - pool - d.r[d.pcode[units]]
            expected = pool + np.average(rest, axis=0, weights=d.prob[units])
            self.assertTrue(np.allclose(level[c], expected))

    def test_two_phase_keeps_the_targets_up_to_the_switch_and_then_subtracts_level_minus_input_set_mean(self):
        level = context_levels(self.data, self.data.r)
        got = targets(self.stream, pr.two_phase(self.data, switch=2))
        for step, t in got.items():
            if step <= 2:
                self.assertTrue(np.array_equal(t, self.stream.cells[step]))
                continue
            rows = self.data.selected_rows(step, 7)
            for slot, row in enumerate(rows):
                amount = level[self.data.parent[row]] - self.stream.atoms[step][slot].astype(np.float64).mean(0)
                cells = self.stream.cells[step][2 * slot:2 * slot + 2]
                self.assertTrue(np.allclose(t[2 * slot:2 * slot + 2], cells - amount, atol=1e-6))

    def test_two_phase_on_a_planted_task_shifts_the_targets_in_both_phases_and_the_level_by_the_mean_shift(self):
        shift = np.full((4, GENES), .5, dtype=np.float32)
        d = self.data
        plain, planted = context_levels(d, d.r), context_levels(d, d.r, d.target.astype(np.float64) + shift)
        for c in (0, 1):
            self.assertTrue(np.allclose(planted[c], plain[c] + .5))
        got = targets(self.stream, pr.two_phase(self.data, switch=2, shift=shift))
        self.assertTrue(np.allclose(got[1], self.stream.cells[1] + .5))
        rows = self.data.selected_rows(3, 7)
        amount = planted[self.data.parent[rows[0]]] - self.stream.atoms[3][0].astype(np.float64).mean(0)
        self.assertTrue(np.allclose(got[3][:2], self.stream.cells[3][:2] + .5 - amount, atol=1e-6))

    def test_the_stream_state_is_the_wrapped_streams(self):
        stream = pr.Retargeted(self.stream, pr.shifted(np.zeros((4, GENES), np.float32)), cells=2)
        stream.load_state_dict({'drawn': 3})
        self.assertEqual(self.stream.state_dict(), {'drawn': 3})
        self.assertEqual(stream.state_dict(), {'drawn': 3})


if __name__ == '__main__':
    unittest.main()
