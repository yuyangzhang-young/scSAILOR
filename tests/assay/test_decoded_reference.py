"""The decoded-cue reference (``reproduce/assay/references.py``) on the tiny random instance; no study data.

    python -m pytest tests/assay/test_decoded_reference.py
"""
import tempfile
import unittest
from pathlib import Path

import numpy as np

from test_tiny_instance import write_instance

from reproduce.assay import references
from scsailor.assay import Instance, generator
from scsailor.assay.constants import N_BITS


class Embed(unittest.TestCase):
    def test_embedded_profiles_carry_the_given_half_differences_and_are_not_clipped(self):
        rng = np.random.default_rng(1)
        differences = rng.normal(0, .3, (N_BITS, 3, 7))
        states = np.stack([generator.base_states(5, 0, base) for base in range(4)])
        profiles = references.embed(differences, states)
        self.assertTrue((profiles >= 0).all())
        for a in range(N_BITS):
            sign = states[:, 0, a, None, None]
            half = sign * (profiles[:, 0] - profiles[:, a + 1]) / 2
            self.assertTrue(np.allclose(half, differences[a][None], atol=1e-12))


class TinyInstance(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        write_instance(Path(cls.temporary.name), np.random.default_rng(0))
        cls.instance = Instance(cls.temporary.name)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_training_average_is_the_mean_signed_half_difference_over_training_backgrounds_and_base_states(self):
        average = references.training_average(self.instance)
        g, a, p = self.instance.generator, 6, 2
        direct = []
        for q, context in enumerate(self.instance.train_contexts):
            background = generator.Background(self.instance.pool("train", context)[0])
            for base in range(4):
                states = generator.base_states(references.TRAINING_ROOT, q, base)
                y = [generator.known_answer(g, background, [p], states[[c]])[0] for c in (0, a + 1)]
                direct.append(states[0, a] * (y[0] - y[1]) / 2)
        self.assertTrue(np.allclose(average[a, p], np.mean(direct, axis=0), atol=1e-14))

    def test_every_state_is_decoded_from_the_input_at_its_cues_first_level(self):
        table = references.decoded_cue_reference(self.instance)
        self.assertEqual(len(table), N_BITS)
        self.assertEqual(set(table.decoded), {4 * 26})        # four base states and their flips, one bag
        self.assertEqual(set(table.decoded_wrong), {0})
        self.assertTrue(table.recovery[table.response == "null"].isna().all())
        self.assertTrue(np.isfinite(table.recovery[table.response != "null"]).all())


if __name__ == "__main__":
    unittest.main()
