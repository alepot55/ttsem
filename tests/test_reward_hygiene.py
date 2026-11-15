"""What the reward package and its documentation keep out: an absolute path of the machine they
were written on, an evaluation figure, and the long dash. The package is source that travels in a
wheel, and the text is read by people who do not know where it came from."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
PACKAGE = sorted((ROOT / "ttsem" / "reward").glob("*.py"))
DOC = ROOT / "docs" / "REWARD.md"
README = ROOT / "README.md"

# the roots of a person's or a runner's own directories, which no public text should name
LOCAL_PATHS = (r"/home/\w", r"/Users/\w", r"/root/")
# the prose is written without the long dash
EM_DASH = chr(0x2014)


def reward_section(readme: str) -> str:
    """The README's own section on the reward: from its heading to the next of the same level."""
    start = re.search(r"^## .*Reward.*$", readme, re.M)
    assert start is not None, "README.md has no Reward section"
    end = re.search(r"^## ", readme[start.end() :], re.M)
    return readme[start.start() : start.end() + (end.start() if end else len(readme))]


def texts() -> dict[str, str]:
    found = {str(p.relative_to(ROOT)): p.read_text(encoding="utf-8") for p in PACKAGE}
    found["docs/REWARD.md"] = DOC.read_text(encoding="utf-8")
    found["README.md (Reward)"] = reward_section(README.read_text(encoding="utf-8"))
    return found


def test_the_package_is_where_the_checks_look() -> None:
    names = {p.name for p in PACKAGE}
    assert {"__init__.py", "core.py", "cap.py", "worker.py", "verl_adapter.py"} <= names
    assert DOC.is_file()


@pytest.mark.parametrize("pattern", LOCAL_PATHS)
def test_no_local_path_in_the_reward_source_or_text(pattern: str) -> None:
    hits = [
        f"{name}:{number}: {line.strip()[:80]}"
        for name, text in texts().items()
        for number, line in enumerate(text.splitlines(), 1)
        if re.search(pattern, line)
    ]
    assert not hits, f"{pattern!r} in public text:\n" + "\n".join(hits[:10])


def test_the_documentation_holds_no_evaluation_figures() -> None:
    """The text says what the reward does and what it misses, not how often: a percentage in it
    would be a measurement, and a measurement lives with the work that reports it."""
    for name in ("docs/REWARD.md", "README.md (Reward)"):
        hits = [
            line.strip()[:80] for line in texts()[name].splitlines() if re.search(r"\d\s?%", line)
        ]
        assert not hits, f"{name} has a percentage:\n" + "\n".join(hits[:10])


def test_no_long_dash_in_the_reward_source_or_text() -> None:
    hits = [name for name, text in texts().items() if EM_DASH in text]
    assert not hits, f"{hits} hold the long dash"
