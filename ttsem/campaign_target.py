"""ttsem as a target of an external program-generation campaign: the device, then every pass.

A campaign driver generates SIMT programs and grades them against *targets* (the `Target`
protocol: `__call__(program, inputs) -> Verdict`, plus optional `prewarm`, `forget`, `source`,
`facts`, `close`). This target compiles the tile family through Triton, launches it recorded,
compares the device with the campaign's own interpreter, and, when they agree, replays the
compilation pass by pass through :func:`ttsem.validate.validate_stages`: a pass whose output
the semantics reads differently from the device becomes a difference of its own, so the
campaign's triage sees it as a finding with the pass named. The campaign registers it with

    TARGETS += ("triton",)
    if kind == "triton": return campaign_target.TritonTarget(device=device, ...)

The driver is not vendored here and is not a dependency of this package: name its top-level
package in ``TTSEM_CAMPAIGN_PACKAGE`` and the directory holding it in ``TTSEM_CAMPAIGN_SRC``.
It is expected to provide ``<package>.fuzz.campaign``, ``.fuzz.emit_triton`` and
``.fuzz.emit_cuda``.
"""

from __future__ import annotations

import importlib
import os
import sys
import time
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent


def _campaign_package() -> str:
    return os.environ.get("TTSEM_CAMPAIGN_PACKAGE", "simtfuzz")


def _campaign_src(package: str) -> Path:
    override = os.environ.get("TTSEM_CAMPAIGN_SRC")
    candidates = [Path(override)] if override else []
    candidates += [HERE.parent.parent / "src", HERE.parent / "src"]
    for root in candidates:
        if (root / package / "fuzz" / "campaign.py").is_file():
            return root
    raise RuntimeError(
        f"{package}.fuzz.campaign was not found; set TTSEM_CAMPAIGN_SRC to the directory "
        f"holding the {package} package (and TTSEM_CAMPAIGN_PACKAGE to its name)"
    )


def _fuzz() -> tuple[Any, Any, Any]:
    package = _campaign_package()
    root = str(_campaign_src(package))
    if root not in sys.path:
        sys.path.insert(0, root)
    campaign = importlib.import_module(f"{package}.fuzz.campaign")
    emit_triton = importlib.import_module(f"{package}.fuzz.emit_triton")
    emit_cuda = importlib.import_module(f"{package}.fuzz.emit_cuda")
    return campaign, emit_triton, emit_cuda


class TritonTarget:
    """The campaign's tile family through Triton, graded on the device and then pass by pass."""

    name = "triton"

    def __init__(
        self,
        *,
        device: int = 0,
        cache_dir: Path | None = None,
        per_pass: bool = True,
        triton_opt: str | None = None,
        run_timeout: float = 120.0,
        below_llvm: Any = None,  # a validate.BelowLLVM, e.g. simtllvm's BelowLlvmAdapter()
        **_ignored: Any,  # the campaign's toolkit/toolchain/workers knobs mean nothing here
    ) -> None:
        self.device = device
        self.cache_dir = cache_dir
        self.per_pass = per_pass
        self.triton_opt = triton_opt
        self.below_llvm = below_llvm
        self.run_timeout = run_timeout
        self.compile_seconds: dict[str, float] = {}
        self.reports: dict[str, Any] = {}  # program hash -> validate.Report of the last run
        self.campaign, self.emit_triton, self.emit_cuda = _fuzz()

    # -- the protocol -----------------------------------------------------------
    def __call__(self, program: Any, inputs: Mapping[str, Any]) -> Any:
        c, et = self.campaign, self.emit_triton
        device = f"cuda:{self.device}"
        t0 = time.time()
        try:
            launched = et.launch_recorded(program, inputs, device=device, directory=self.cache_dir)
        except (et.EmitError, et.HarnessError) as exc:
            raise c.CompileError(str(exc)) from exc
        except Exception as exc:  # the launch itself, or Triton's compiler
            raise c.LaunchError(f"{type(exc).__name__}: {exc}"[:400]) from exc
        digest = launched.kernel.program_hash
        self.compile_seconds[digest] = time.time() - t0
        try:
            expected = c.execute(program, inputs, facts=False).outputs
        except c.ConvergenceError as exc:
            raise c.OracleFailure(str(exc)) from exc
        differences = list(self.emit_cuda.compare(expected, launched.outputs))
        if not differences and self.per_pass:
            differences += self._per_pass_differences(launched, digest)
        return et.Verdict(
            program_hash=digest,
            differences=tuple(differences),
            oracle=expected,
            target=launched.outputs,
            source=launched.kernel.source,
        )

    def _per_pass_differences(self, launched: Any, digest: str) -> list[Any]:
        """A mismatching stage as a `Difference`: the array it names is the pass."""
        et = self.emit_triton
        harness, validate = et.ttsem()
        try:  # the MLIR_ENABLE_DUMP trace of the same compile: "Before P" stages
            import torch

            major, minor = torch.cuda.get_device_capability(self.device)
            target = harness.GPUTarget("cuda", major * 10 + minor, 32)
            stages = validate.split_dump(harness.dump_for_launch(launched.record, target))
        except Exception as exc:  # no dump, no verdict on the passes; the device verdict stands
            self.reports[digest] = f"stages unavailable: {exc!r}"[:300]
            return []
        report = validate.validate_stages(
            launched.record,
            stages,
            self.triton_opt,
            f"{digest[:12]} ({len(stages)} stages)",
            below_llvm=self.below_llvm,
        )
        self.reports[digest] = report
        bad = [r for r in report.passes if r.verdict == "mismatch"]
        if not bad:
            return []
        first = bad[0]
        return [
            self.emit_cuda.Difference(
                array=f"pass:{first.pass_name}",
                index=-1,
                oracle=0,
                target=0,
                count=max(int(first.n_diff), 1),
            )
        ]

    def prewarm(self, work: Iterable[tuple[Any, Mapping[str, Any]]]) -> Any:
        class _Done:
            def wait(self) -> None:
                return None

        return _Done()

    def forget(self, program_hash: str) -> None:
        self.compile_seconds.pop(program_hash, None)
        self.reports.pop(program_hash, None)

    def source(self, program: Any, inputs: Mapping[str, Any]) -> tuple[str, str]:
        return "kernel.py", self.emit_triton.emit(program).source

    def facts(self) -> Mapping[str, str]:
        import triton

        try:
            import torch

            major, minor = torch.cuda.get_device_capability(self.device)
            cc = f"{major}{minor}"
        except Exception:
            cc = "?"
        return {
            "target": self.name,
            "triton": getattr(triton, "__version__", "?"),
            "cc": cc,
            "per_pass": str(self.per_pass),
            "triton_opt": self.triton_opt or "wheel printer",
        }

    def close(self) -> None:
        return None


__all__ = ["TritonTarget"]
