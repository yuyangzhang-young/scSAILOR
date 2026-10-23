"""The assay's premise on random cell bags (no study data): a bit is hidden below its first visible
level and readable from it on.


    python -m pytest tests/assay/test_design_rule.py
"""
import unittest

import numpy as np

from scsailor.assay import design
from scsailor.assay.carrier import carrier, decode_bit
from scsailor.assay.constants import N_BITS, N_CELLS, N_GENES
from scsailor.assay.generator import Parameters


class DesignRule(unittest.TestCase):
    def test_bits_are_hidden_below_their_first_level_and_decoded_from_it_on(self):
        for seed in range(3):
            rng = np.random.default_rng(seed)
            x = rng.gamma(2., .5, size=(N_CELLS, N_GENES))
            depth = rng.choice(np.arange(1000, 60000), size=N_CELLS, replace=False)
            state = rng.choice([-1, 1], size=N_BITS)
            for row in design.bits():
                a = row[design.BIT]
                plus, minus = state.copy(), state.copy()
                plus[a], minus[a] = 1, -1
                for level in design.LEVELS:
                    p_rows, p_depth = carrier(x, depth, plus, level, np.random.default_rng(31001), Parameters())
                    m_rows, m_depth = carrier(x, depth, minus, level, np.random.default_rng(31001), Parameters())
                    if design.visible(a, level):
                        self.assertEqual((decode_bit(p_rows, p_depth, a), decode_bit(m_rows, m_depth, a)), (1, -1),
                                         f"bag {seed} bit {a} at {level}")
                    else:
                        self.assertTrue(np.array_equal(p_rows, m_rows) and np.array_equal(p_depth, m_depth),
                                        f"bag {seed} bit {a} leaks at {level}")


if __name__ == "__main__":
    unittest.main()
