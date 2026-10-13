"""The install line for the optional Triton extra, in one place for every message that prints it.

PyTorch's CPU index comes first so that pip takes the CPU build of torch, which needs no CUDA
libraries and is a fraction of the download; ttsem, Triton and the rest come from PyPI.
"""

from __future__ import annotations

CPU_INDEX = "https://download.pytorch.org/whl/cpu"
PYPI_INDEX = "https://pypi.org/simple"

# Until the first release is on PyPI, only the git form of the requirement resolves.
GIT_REQUIREMENT = "'ttsem[triton] @ git+https://github.com/alepot55/ttsem'"

PIP_TRITON = f"pip install --index-url {CPU_INDEX} --extra-index-url {PYPI_INDEX} 'ttsem[triton]'"

GIT_NOTE = f"until ttsem's first release is on PyPI, write {GIT_REQUIREMENT} for 'ttsem[triton]'"


def triton_hint(indent: str = "    ") -> str:
    """The install command as lines to print, the command indented and split for a shell."""
    return (
        f"{indent}pip install --index-url {CPU_INDEX} \\\n"
        f"{indent}    --extra-index-url {PYPI_INDEX} 'ttsem[triton]'\n"
        f"({GIT_NOTE})\n"
    )
