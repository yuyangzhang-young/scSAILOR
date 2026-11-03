"""Importing origin resolution has no side effects: a fresh process imports every module; torch stays unloaded and
``sys.path`` is unchanged.

    python -m pytest tests/resolution/test_resolution_imports.py
"""
import json
import subprocess
import sys
import unittest

CHECK = """
import json, sys
before = list(sys.path)
from scsailor.resolution import (attribution, card, comparison, grids, il_cells, menu, net_cost, origins, policies, probes,
                                 references, scoring, slopes, stages)
print(json.dumps({"torch": "torch" in sys.modules, "path_unchanged": sys.path == before}))
"""


class Imports(unittest.TestCase):
    def test_imports_load_no_torch_and_leave_the_path(self):
        done = subprocess.run([sys.executable, "-c", CHECK], capture_output=True, text=True, check=True)
        self.assertEqual(json.loads(done.stdout), {"torch": False, "path_unchanged": True})


if __name__ == "__main__":
    unittest.main()
