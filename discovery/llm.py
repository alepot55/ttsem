"""The language models the search and the prover talk to.

* :class:`ClaudeCLI` runs ``claude -p`` with every tool off, no MCP server, no session saved and
  a working directory of its own (so no project instructions or hooks reach the model): the
  account's subscription, no API key.
* :class:`OpenAICompatible` posts to any ``/v1/chat/completions`` endpoint, such as a local model
  served by llama.cpp, Ollama or vLLM on the GPU machine.
* :class:`Scripted` returns canned answers, for the tests.

Every call is recorded (seconds, tokens, cost when the backend reports one) in ``calls``.
"""

from __future__ import annotations

import dataclasses
import json
import os
import shutil
import subprocess
import tempfile
import time
import urllib.request
from collections.abc import Iterable


@dataclasses.dataclass
class Call:
    seconds: float
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    error: str = ""


class LLMError(Exception):
    pass


class ClaudeCLI:
    """``claude -p`` as a plain text model."""

    def __init__(
        self, model: str = "sonnet", timeout: float = 600.0, effort: str | None = None
    ) -> None:
        self.model = model
        self.timeout = timeout
        self.effort = effort
        self.calls: list[Call] = []
        self.exe = shutil.which("claude") or "claude"
        self.cwd = os.path.join(tempfile.gettempdir(), "discovery-llm")
        os.makedirs(self.cwd, exist_ok=True)

    @property
    def name(self) -> str:
        return f"claude-cli:{self.model}"

    def __call__(
        self,
        prompt: str,
        system: str = "You are a careful expert. Answer exactly in the format asked.",
    ) -> str:
        cmd = [
            self.exe,
            "-p",
            "--output-format",
            "json",
            "--model",
            self.model,
            "--tools",
            "",
            "--strict-mcp-config",
            "--no-session-persistence",
            "--setting-sources",
            "",
            "--system-prompt",
            system,
        ]
        if self.effort:
            cmd += ["--effort", self.effort]
        t0 = time.time()
        try:
            proc = subprocess.run(
                cmd,
                input=prompt,
                capture_output=True,
                text=True,
                encoding="utf-8",
                timeout=self.timeout,
                cwd=self.cwd,
            )
        except subprocess.TimeoutExpired as exc:
            self.calls.append(Call(time.time() - t0, error="timeout"))
            raise LLMError(f"claude -p timed out after {self.timeout:.0f}s") from exc
        dt = time.time() - t0
        try:
            out = json.loads(proc.stdout)
        except json.JSONDecodeError as exc:
            self.calls.append(Call(dt, error=proc.stderr[-300:]))
            raise LLMError(
                f"claude -p returned no JSON (exit {proc.returncode}): {proc.stderr[-300:]}"
            ) from exc
        usage = out.get("usage") or {}
        call = Call(
            dt,
            int(usage.get("input_tokens", 0))
            + int(usage.get("cache_creation_input_tokens", 0))
            + int(usage.get("cache_read_input_tokens", 0)),
            int(usage.get("output_tokens", 0)),
            float(out.get("total_cost_usd") or 0.0),
        )
        if out.get("is_error") or out.get("subtype", "success") != "success":
            call.error = str(out.get("result", ""))[:300]
            self.calls.append(call)
            raise LLMError(f"claude -p failed: {call.error}")
        self.calls.append(call)
        return str(out.get("result", ""))


class OpenAICompatible:
    """A chat-completions endpoint, e.g. ``http://localhost:8080/v1`` for a local model."""

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key_env: str = "OPENAI_API_KEY",
        timeout: float = 600.0,
        temperature: float = 0.8,
        max_tokens: int = 4096,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.key = os.environ.get(api_key_env, "")
        self.timeout = timeout
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.calls: list[Call] = []

    @property
    def name(self) -> str:
        return f"openai:{self.model}"

    def __call__(
        self,
        prompt: str,
        system: str = "You are a careful expert. Answer exactly in the format asked.",
    ) -> str:
        body = json.dumps(
            {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
            }
        ).encode()
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=body,
            method="POST",
            headers={
                "Content-Type": "application/json",
                **({"Authorization": f"Bearer {self.key}"} if self.key else {}),
            },
        )
        t0 = time.time()
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                out = json.loads(resp.read())
        except Exception as exc:  # noqa: BLE001  network, HTTP, JSON: all the same to the caller
            self.calls.append(Call(time.time() - t0, error=str(exc)[:300]))
            raise LLMError(f"{self.base_url}: {exc}") from exc
        usage = out.get("usage") or {}
        self.calls.append(
            Call(
                time.time() - t0,
                int(usage.get("prompt_tokens", 0)),
                int(usage.get("completion_tokens", 0)),
            )
        )
        return out["choices"][0]["message"]["content"]


class Scripted:
    """Answers from a list, in order; for tests."""

    def __init__(self, answers: Iterable[str]) -> None:
        self.answers = list(answers)
        self.prompts: list[str] = []
        self.calls: list[Call] = []

    @property
    def name(self) -> str:
        return "scripted"

    def __call__(self, prompt: str, system: str = "") -> str:
        self.prompts.append(prompt)
        self.calls.append(Call(0.0))
        if not self.answers:
            raise LLMError("no scripted answer left")
        return self.answers.pop(0)


def from_spec(spec: str):
    """A model from ``claude:MODEL`` (``claude:sonnet``, ``claude:opus``, ...) or
    ``openai:MODEL@URL`` (any OpenAI-compatible endpoint, default ``http://localhost:8080/v1``).
    :class:`Scripted` replies are built in code, not from a spec."""
    kind, _, rest = spec.partition(":")
    if kind == "claude":
        return ClaudeCLI(rest or "sonnet")
    if kind == "openai":
        model, _, url = rest.partition("@")
        return OpenAICompatible(url or "http://localhost:8080/v1", model)
    raise ValueError(f"unknown model spec {spec!r}")


def usage(model) -> dict:
    calls = getattr(model, "calls", [])
    return {
        "calls": len(calls),
        "seconds": round(sum(c.seconds for c in calls), 1),
        "input_tokens": sum(c.input_tokens for c in calls),
        "output_tokens": sum(c.output_tokens for c in calls),
        "cost_usd": round(sum(c.cost_usd for c in calls), 4),
        "errors": sum(1 for c in calls if c.error),
    }
