"""One name per model. The built attention model is ``attention_model``; ``state`` is the published State, which is not
a built model. Every table of the built models names the same three.

    python -m pytest tests/test_model_names.py
"""
import importlib.util
import unittest

from scsailor.assay.contract import READOUT_MODULES
from scsailor.localization.contract import CUTS, FRONT_READS_EMBEDDING, PRESENCE_SITES
from scsailor.localization.presence import KEY_CUTS

BUILT = {"attention_model", "deepsets", "summary"}


class Tables(unittest.TestCase):
    def test_every_table_names_the_built_models(self):
        for table in (READOUT_MODULES, CUTS, PRESENCE_SITES, KEY_CUTS):
            self.assertEqual(set(table), BUILT)
        self.assertEqual(set(FRONT_READS_EMBEDDING), {"attention_model"})


@unittest.skipUnless(importlib.util.find_spec("torch"), "needs torch")
class Build(unittest.TestCase):
    def test_the_attention_model_is_built_by_its_name_and_state_is_not_a_built_model(self):
        import numpy as np

        from scsailor.models.backbones import AttentionModel, build

        intercept = np.zeros(2000, np.float32)
        self.assertIsInstance(build("attention_model", intercept), AttentionModel)
        with self.assertRaises(KeyError):
            build("state", intercept)


if __name__ == "__main__":
    unittest.main()
