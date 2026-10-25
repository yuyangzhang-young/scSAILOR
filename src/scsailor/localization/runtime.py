"""The numerical runtime of every localization run: the only place that sets threads, TF32 and matmul precision."""

__all__ = ["set_precision", "peak_gpu_mib"]


def set_precision(threads=None, interop_threads=None):
    """FP32 at full precision: TF32 off on CUDA matmul and cuDNN; optionally fix torch's CPU threads."""
    import torch

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    if threads is not None:
        torch.set_num_threads(threads)
    if interop_threads is not None:
        torch.set_num_interop_threads(interop_threads)


def peak_gpu_mib(device):
    """The peak memory torch allocated on the run's own device, in MiB; ``None`` on the CPU."""
    import torch

    return torch.cuda.max_memory_allocated(device) / 2**20 if str(device).startswith("cuda") else None
