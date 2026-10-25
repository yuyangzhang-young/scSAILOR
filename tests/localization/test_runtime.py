"""A run records the GPU memory of its own device: the query names the device the run uses, so a run on
``cuda:1`` does not record the current device ``cuda:0``.

    python -m pytest tests/localization/test_runtime.py
"""
import importlib.util
import unittest
from unittest import mock

from scsailor.localization import runtime


@unittest.skipUnless(importlib.util.find_spec("torch"), "needs torch")
class PeakMemory(unittest.TestCase):
    def test_the_query_names_the_run_device(self):
        import torch

        with mock.patch.object(torch.cuda, "max_memory_allocated", return_value=3 * 2**20) as query:
            self.assertEqual(runtime.peak_gpu_mib("cuda:1"), 3)
        query.assert_called_once_with("cuda:1")

    def test_a_cpu_run_records_none(self):
        self.assertIsNone(runtime.peak_gpu_mib("cpu"))


if __name__ == "__main__":
    unittest.main()
