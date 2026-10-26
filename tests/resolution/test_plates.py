"""The plate read-outs take the screen's two plate names: renamed plates give the same twins, contrasts and test.

    python -m pytest tests/resolution/test_plates.py
"""
import unittest

import numpy as np

from scsailor.resolution import attribution, menu


def units(plates, lines=4, drugs=3, genes=5):
    rng = np.random.default_rng(0)
    contexts = np.repeat([f"L{i}|{p}" for i in range(lines) for p in plates], drugs)
    return {"truth": rng.uniform(0, 2, (len(contexts), genes)), "context": contexts,
            "p_code": np.tile(np.arange(drugs), lines * len(plates)), "weight": rng.uniform(.5, 1.5, len(contexts))}


class RenamedPlates(unittest.TestCase):
    tahoe, other = ("plate6", "plate14"), ("rep1", "rep2")

    def test_twins_contrast_and_test_do_not_depend_on_the_names(self):
        a, b = units(self.tahoe), units(self.other)
        self.assertTrue(np.array_equal(attribution.replicate_twins(a, plates=self.tahoe),
                                       attribution.replicate_twins(b, plates=self.other)))
        parts = {"common": np.random.default_rng(1).uniform(size=len(a["truth"]))}
        self.assertEqual(attribution.plate_contrast(parts, a, plates=self.tahoe),
                         attribution.plate_contrast(parts, b, plates=self.other))
        test_a = attribution.plate_test(parts, a, plates=self.tahoe)
        test_b = attribution.plate_test(parts, b, plates=self.other)
        self.assertEqual(test_a.pop("lines_plate6_higher"), test_b.pop("lines_rep1_higher"))
        self.assertEqual(test_a, test_b)

    def test_each_context_reads_its_plate_from_its_bag(self):
        rng = np.random.default_rng(2)
        direction = rng.normal(size=6)
        bags = {f"L{i}|{p}": (rng.normal(size=6) + (direction if p == "rep1" else -direction), None)
                for i in range(5) for p in self.other}
        read = menu.plates_from_bags(bags, [f"L{i}" for i in range(5)], bags, sorted(bags), plates=self.other)
        self.assertEqual(read, {c: c.split("|")[1] for c in bags})


if __name__ == "__main__":
    unittest.main()
