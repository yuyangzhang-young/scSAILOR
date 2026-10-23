"""Process settings of every run: one numerical thread and full-precision matrix products.

Import this module before NumPy or PyTorch: the thread variables take effect only if they are set before
those libraries load. The paper's runs used these settings; with more threads, sums can be taken in a
different order and the results differ in the last digits.
"""
import os

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_name] = "1"

import json  # noqa: E402
from pathlib import Path  # noqa: E402

import torch  # noqa: E402


def configure(device):
    """One thread, no TF32, 'highest' float32 matrix precision; checks that a requested GPU exists."""
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"{device} was requested but CUDA is not available")


def save_json(path, value):
    """Write JSON through a temporary file, so that a reader never sees a partial file."""
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)
