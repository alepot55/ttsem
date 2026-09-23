"""A CUDA driver stub, enough for Triton to compile a kernel on a machine with no GPU.

Triton asks the active driver for the current target while compiling, and some backend code
asks ``torch.cuda`` the same questions. Nothing is ever launched through this: it exists so
that :mod:`ttsem.harness` can compile a kernel for a chosen compute capability and read the
IR out of the result.
"""

from __future__ import annotations

from typing import Any


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


# ``torch.cuda`` as torch defined it, kept by the first stub: code that asks torch itself (Dynamo
# saving the RNG state, Inductor reading device properties) needs the real answers while Triton
# sees the target, and :func:`unstub_torch_cuda` gives them back.
_REAL_TORCH_CUDA: dict[str, Any] = {}


def _stub_torch_cuda(cc: int) -> None:
    """Answer the ``torch.cuda`` queries as the chosen target would.

    Kernel programs query ``torch.cuda`` at import time and in skip conditions; without a GPU
    those queries have to be answered for the compile to be reached at all.
    """
    import contextlib

    import torch

    cap = (cc // 10, cc % 10)

    class _Props:
        name = f"fake-sm{cc}"
        major, minor = cap
        multi_processor_count = 128
        total_memory = 1 << 34

    stubs: dict[str, Any] = {
        "get_device_capability": lambda device=None: cap,
        "get_device_name": lambda device=None: f"fake-sm{cc}",
        "is_available": lambda: True,
        "device_count": lambda: 1,
        "current_device": lambda: 0,
        "synchronize": lambda device=None: None,
        "empty_cache": lambda: None,
        "manual_seed": lambda seed: None,
        "manual_seed_all": lambda seed: None,
        "device": lambda idx=None: contextlib.nullcontext(),
        "get_device_properties": lambda device=None: _Props(),
    }
    for name, stub in stubs.items():
        _REAL_TORCH_CUDA.setdefault(name, getattr(torch.cuda, name))
        setattr(torch.cuda, name, stub)


def unstub_torch_cuda() -> None:
    """Put back the ``torch.cuda`` functions :func:`_stub_torch_cuda` replaced."""
    import torch

    for name, real in _REAL_TORCH_CUDA.items():
        setattr(torch.cuda, name, real)


__all__ = ["_FakeDriver", "_stub_torch_cuda", "unstub_torch_cuda"]
