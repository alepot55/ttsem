"""A CUDA driver stub, enough for Triton to compile a kernel on a machine with no GPU.

Triton asks the active driver for the current target while compiling, and some backend code
asks ``torch.cuda`` the same questions. Nothing is ever launched through this: it exists so
that :mod:`ttsem.harness` can compile a kernel for a chosen compute capability and read the
IR out of the result.
"""

from __future__ import annotations


class _FakeDriver:
    """Enough of a driver to import and specialise a kernel without a GPU.

    Triton asks the active driver for the current target at import time; here nothing is ever
    launched, so a target is all it needs.
    """

    def __init__(self, cc: int) -> None:
        from triton.backends.compiler import GPUTarget

        self._target = GPUTarget("cuda", cc, 32)

    @staticmethod
    def is_active():
        return True

    def get_current_target(self):
        return self._target

    @staticmethod
    def get_current_device():
        return 0

    @staticmethod
    def get_current_stream(device=None):
        return 0

    def get_device_properties(self, device=None):
        return {
            "max_shared_mem": 101376,
            "multiprocessor_count": 128,
            "sm_clock_rate": 1000,
            "mem_clock_rate": 1000,
            "mem_bus_width": 256,
        }

    @staticmethod
    def get_active_torch_device():
        import torch

        return torch.device("cpu")

    @staticmethod
    def get_device_interface():
        import torch

        return torch.cpu

    @staticmethod
    def map_python_to_cpp_type(ty):
        return ty

    @staticmethod
    def get_benchmarker():
        return None

    @staticmethod
    def get_empty_cache_for_benchmark():
        return None

    @staticmethod
    def clear_cache(cache):
        return None


def _stub_torch_cuda(cc: int) -> None:
    """Answer the ``torch.cuda`` queries as the chosen target would.

    Kernel programs query ``torch.cuda`` at import time and in skip conditions; without a GPU
    those queries have to be answered for the compile to be reached at all.
    """
    import contextlib

    import torch

    cap = (cc // 10, cc % 10)
    torch.cuda.get_device_capability = lambda device=None: cap
    torch.cuda.get_device_name = lambda device=None: f"fake-sm{cc}"
    torch.cuda.is_available = lambda: True
    torch.cuda.device_count = lambda: 1
    torch.cuda.current_device = lambda: 0
    torch.cuda.synchronize = lambda device=None: None
    torch.cuda.empty_cache = lambda: None
    torch.cuda.manual_seed = lambda seed: None
    torch.cuda.manual_seed_all = lambda seed: None
    torch.cuda.device = lambda idx=None: contextlib.nullcontext()

    class _Props:
        name = f"fake-sm{cc}"
        major, minor = cap
        multi_processor_count = 128
        total_memory = 1 << 34

    torch.cuda.get_device_properties = lambda device=None: _Props()


__all__ = ["_FakeDriver", "_stub_torch_cuda"]
