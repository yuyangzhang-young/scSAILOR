"""The built models' retraining on small hand-made arrays; no study data: the model constants, the target changes of
the designs, the units they are applied to, and a short training run.

    python -m pytest tests/memorization/test_retraining.py
"""
import importlib.util
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from reproduce.memorization import retraining

TORCH = importlib.util.find_spec('torch') is not None
GENES = 2000   # the natural task's panel size, fixed by its stream


def write_fold(directory, contexts=2, cells=300, units=6, drugs=3, seed=0):
    """A training directory: control pools and training targets in which column 0 of each unit's target is its row."""
    rng = np.random.default_rng(seed)
    with h5py.File(directory / 'controls.h5', 'w') as h:
        for c in range(contexts):
            h[f'{c}/input/x'] = rng.poisson(.3, (cells, GENES)).astype(np.float32)
            h[f'{c}/input/depth'] = rng.integers(1000, 5000, cells)
    target = rng.uniform(0, 1, (units, GENES)).astype(np.float32)
    target[:, 0] = np.arange(units)
    np.savez(directory / 'train_targets.npz', target=target, parent=np.arange(units) % contexts,
             p_code=np.arange(units) % drugs, weight=np.full(units, 1 / units), intercept=target.mean(axis=0),
             p_ids=np.array([f"[('d{p}', 1.0, 'uM')]" for p in range(drugs)]))


class Constants(unittest.TestCase):
    def test_are_the_weighted_drug_change_pool_mean_and_target_mean(self):
        target = np.array([[1., 2.], [3., 6.], [5., 4.]], np.float32)
        parent, p_code, weight = np.array([0, 1, 1]), np.array([0, 0, 1]), np.array([1., 3., 2.])
        pool = {0: np.array([.5, .5]), 1: np.array([1., 2.])}
        r, c, intercept = retraining.model_constants(target, parent, p_code, weight, pool, 2)
        self.assertTrue(np.allclose(r, [[(1 * .5 + 3 * 2) / 4, (1 * 1.5 + 3 * 4) / 4], [4., 2.]]))
        self.assertTrue(np.allclose(c, [(.5 + 5 * 1.) / 6, (.5 + 5 * 2.) / 6]))
        self.assertTrue(np.allclose(intercept, [(1 + 9 + 10) / 6, (2 + 18 + 8) / 6]))
        self.assertEqual(r.dtype, np.float32)


class Designs(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(1)
        self.pools = {0: (rng.uniform(size=(5, 3)).astype(np.float32), None),
                      1: (rng.uniform(size=(4, 3)).astype(np.float32), None)}
        self.x = rng.uniform(size=(2, 6, 3)).astype(np.float32)
        self.y = rng.uniform(size=(2, 3)).astype(np.float32)
        self.parent = np.array([1, 0])

    def test_input_relative_removes_the_gap_between_the_pool_and_the_carrier_rows(self):
        stream = type('S', (), {'pools': self.pools})()
        got = retraining.input_relative(stream)(1, self.parent, self.x, self.y)
        gap = np.stack([self.pools[1][0].mean(axis=0), self.pools[0][0].mean(axis=0)]) - self.x.mean(axis=1)
        self.assertTrue(np.allclose(got, self.y - gap, atol=1e-6))
        self.assertEqual(got.dtype, np.float32)

    def test_readable_only_removes_the_contexts_unreadable_part(self):
        E = {0: np.array([1., 2., 3.]), 1: np.array([-1., 0., 1.])}
        got = retraining.readable_only(E)(1, self.parent, self.x, self.y)
        self.assertTrue(np.allclose(got, self.y - np.stack([E[1], E[0]])))

    def test_two_phase_keeps_the_targets_then_removes_the_context_level_beyond_the_carrier_rows(self):
        stream = type('S', (), {})()
        stream.pools = self.pools
        stream.target = np.array([[1., 1., 1.], [2., 0., 1.], [0., 3., 1.]], np.float32)
        stream.parent, stream.pcode, stream.prob = np.array([0, 0, 1]), np.array([0, 1, 0]), np.array([.2, .6, .2])
        r = np.array([[.1, .2, .3], [0., 0., 1.]], np.float32)
        change = retraining.two_phase(stream, r, switch=5)
        self.assertIs(change(5, self.parent, self.x, self.y), self.y)
        pool = {c: p.mean(axis=0) for c, (p, _) in self.pools.items()}
        level0 = pool[0] + (.2 * (stream.target[0] - pool[0] - r[0]) + .6 * (stream.target[1] - pool[0] - r[1])) / .8
        level1 = pool[1] + (stream.target[2] - pool[1] - r[0])
        got = change(6, self.parent, self.x, self.y)
        self.assertTrue(np.allclose(got, self.y - (np.stack([level1, level0]) - self.x.mean(axis=1)), atol=1e-6))


class StreamRows(unittest.TestCase):
    def test_the_change_is_applied_to_the_units_the_natural_stream_drew(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_fold(Path(tmp))
            seen = []
            stream = retraining.Stream(Path(tmp), change=lambda step, parent, x, y: seen.append((parent, y)) or y)
            stream.batch(3, 256, 'L1', 7)
            parent, y = seen[0]
            self.assertTrue(np.array_equal(parent, stream.parent[y[:, 0].astype(int)]))

    def test_replaced_targets_are_what_the_stream_draws(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_fold(Path(tmp))
            plain = retraining.Stream(Path(tmp))
            replaced = retraining.Stream(Path(tmp), target=plain.target + 1)
            a, b = plain.batch(2, 256, 'L1', 7), replaced.batch(2, 256, 'L1', 7)
            self.assertTrue(np.array_equal(a[0], b[0]) and np.allclose(b[3], a[3] + 1))


@unittest.skipUnless(TORCH, 'needs torch')
class Training(unittest.TestCase):
    def test_a_short_run_is_repeatable_and_changes_the_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_fold(Path(tmp))
            stream = retraining.Stream(Path(tmp))
            values = retraining.constants('tahoe', stream, None)
            runs = []
            for _ in range(2):
                model = retraining.build('summary', values, stream.p_count, 5)
                start = {k: v.clone() for k, v in model.state_dict().items()}
                losses = retraining.train(model, stream, 'summary', 5, updates=2, workers=1)
                runs.append((losses, model.state_dict()))
            self.assertEqual(runs[0][0], runs[1][0])
            self.assertTrue(all((runs[0][1][k] == runs[1][1][k]).all() for k in start))
            self.assertTrue(any(not (runs[0][1][k] == start[k]).all() for k in start))
