"""The proposer: its prompt, and the program read back from its reply.

The prompt holds the IR's grammar (the docstring of :mod:`discovery.ir` and its operator list),
the problem, what the checks are and what is scored, the parent program with its verdict, up to
two other elites with their scores, the last rejected candidates with their reasons and the proved
lemmas (from :mod:`discovery.knowledge`), and the request for one whole program in a single fenced
block. The problem's known reformulations are not put in the prompt as programs, but a proved
lemma states the merges of the program it proves: when the library index holds a lemma about a
known form, the prompt shows that form's merges (a :class:`Knowledge` with ``library=None``
shows no lemma). In blind mode the problem is described without its name
(``Problem.blind_description``) and every text shown is rewritten: programs are renamed
``candidate`` and the operator's name is replaced; the field names (elements, state, outputs) are
not renamed. A structure the model knows can still give the answer away; the run log says which
runs were blind.
"""

from __future__ import annotations

import re
from collections.abc import Callable

from discovery import ir
from discovery.archive import TOLERANCE, Record
from discovery.evaluate import EXACT_TOL, FP_TOL
from discovery.knowledge import Knowledge
from discovery.problems import Problem

SYSTEM = (
    "You design numerical algorithms as programs in a small language of blocked folds. You end "
    "every answer with exactly one complete program in one fenced code block."
)
NEUTRAL = "candidate"  # the name of every program shown in blind mode

_HEAD = re.compile(r"\(\s*program\s+[^\s()]+")
_FENCE = re.compile(r"```[^\n`]*\r?\n(.*?)```", re.DOTALL)


class ProposalError(Exception):
    """A reply with no program in it."""


def shown(problem: Problem, blind: bool) -> Callable[[str], str]:
    """How a text reaches the proposer: unchanged, or in blind mode with every program renamed
    and the operator's name replaced."""
    if not blind:
        return str
    name = re.compile(re.escape(problem.name), re.IGNORECASE)

    def show(text: str) -> str:
        return name.sub("op", _HEAD.sub(f"(program {NEUTRAL}", text))

    return show


def score(rec: Record) -> str:
    v = rec.verdict
    took = (
        f"measured {rec.measured_us:.0f} us"
        if rec.measured_us is not None
        else f"estimated {v.cost['time_us']:.0f} us"
    )
    return (
        f"{v.passes} pass(es) over memory, {v.state_fields} state field(s), "
        f"float32 {v.fp_class}, {took}"
    )


def _grammar() -> str:
    doc = (ir.__doc__ or "").split("\n", 2)[2].replace("``", "`").replace("::", ":").strip()
    ops = (
        f"Operators: {' '.join(sorted(ir.VARIADIC))} (two or more arguments); "
        f"{' '.join(sorted(ir.BINARY))} (two); {' '.join(sorted(ir.UNARY))} (one); "
        f"{' '.join(sorted(ir.REDUCTIONS))} (reductions over the elements of a block, one)."
    )
    runs = (
        "How a program runs: each pass reads every element once. The elements are cut into "
        "consecutive nonempty blocks, anywhere; `block` gives the summary of each block, and "
        "`merge` turns two adjacent summaries (L the left one, R the right one) into the summary "
        "of their union, in any order and along any tree, until one summary is left. Its fields "
        "are then readable as `PASS.FIELD` by the later passes and the output. An elementwise "
        "output reads every element once more."
    )
    return f"{doc}\n\n{ops}\n\n{runs}"


def _signature(problem: Problem) -> str:
    fields = ", ".join(f"{f} a {t}" for f, t in problem.elems.items())
    line = f"Its elements: (input {' '.join(problem.elems)}), {fields}."
    if problem.outputs == "elementwise":
        return f"{line}\nIts output: (output elementwise EXPR), one value per element."
    assert isinstance(problem.outputs, dict)
    names = " ".join(f"({n} EXPR)" for n in problem.outputs)
    types = ", ".join(f"{n} a {t}" for n, t in problem.outputs.items())
    return f"{line}\nIts output: (output final {names}), {types}."


def _checks(problem: Problem) -> str:
    return f"""# How a program is checked and scored

1. It must parse and type in the language above, with this operator's elements and output.
2. Exact over the reals: it runs in float64 on random rows of many lengths, from a single \
element up. Each pass is cut into blocks in many ways (for example one block, one element per \
block, random cuts) and the block summaries are merged left to right, right to left and along \
random trees; the outputs must match the operator's every time (relative error at most \
{EXACT_TOL:g} on ordinary rows). Only the outputs decide this stage, on the blockings drawn: the \
merge's own law (step 5) is not required here. The first failure is reported with its input.
3. Floating point: a program that passed runs in float32 (blocks of 128, merged in order) on \
{len(problem.fp_families)} input families, ordinary and adversarial \
({", ".join(problem.fp_families)}). Its float32 class is `stable` (relative error at most \
{FP_TOL:g} on every family), `inaccurate`, or `nonfinite` (an inf or a nan came out).
4. Score: the passes over memory (each pass reads the elements from memory again, and an \
elementwise output is one more pass), the state fields, the float32 class, and the time: \
measured on the GPU when a program shows a measured time, estimated otherwise. On these \
operators the memory time is many times the compute time, so the estimate is set by the bytes \
moved per element (the element fields each pass reads, and an elementwise output's writes); the \
exps computed do not change it. Fewer passes and less time are better; `stable` beats \
`inaccurate`, which beats `nonfinite`. The archive keeps the best program for each combination of \
passes, state fields and float32 class; two times within {TOLERANCE:.0%} of each other count as \
equal, and then the program with fewer expression nodes wins.
5. Proof, apart from the score: the proof tier asks Lean to prove, for every pass, \
merge(block(A), block(B)) = \
block(A ++ B) over the reals for any nonempty blocks A and B (the merge is a homomorphism), and \
that the output on one block is the operator's. A program whose outputs are right but whose merge \
is not a homomorphism (its summaries depend on where the blocks fall) can pass the tests above, \
but it cannot be proved."""


def build_prompt(
    problem: Problem,
    parent: Record,
    inspirations: list[Record],
    knowledge: Knowledge | None,
    blind: bool,
) -> str:
    show = shown(problem, blind)
    description = problem.blind_description if blind else problem.description
    measured = (
        f" Measured on the GPU: {parent.measured_us:.0f} us."
        if parent.measured_us is not None
        else ""
    )
    parts = [
        "You are improving a program for one operator, written in a small language for "
        "reductions on a GPU. Every program you propose is checked automatically and scored.",
        f"# The language\n\n{_grammar()}",
        f"# The operator\n\n{description}\n\n{_signature(problem)}",
        _checks(problem),
        "# The program to improve\n\n"
        f"```lisp\n{show(parent.verdict.text)}\n```\n\n"
        f"Its verdict: {show(parent.verdict.message)}{measured}",
    ]
    if inspirations:
        lines = ["# Other programs in the archive"]
        for rec in inspirations:
            lines += ["", f"```lisp\n{show(rec.verdict.text)}\n```", f"Score: {score(rec)}."]
        parts.append("\n".join(lines))
    if knowledge is not None:
        section = knowledge.prompt_section(show)
        if section:
            parts.append(section)
    parts.append(
        "# Your answer\n\n"
        "Propose one new program for this operator that passes the checks and scores better than "
        "the program to improve: fewer passes over memory, or as many passes with less time or a "
        "better float32 class. You may change anything: the passes, the state, the "
        "blocks, the merges, the output. Do not repeat a program shown above. Reason in a few "
        "sentences at most, then end your reply with the whole program in a single fenced block:"
        "\n\n```lisp\n(program NAME\n  ...)\n```"
    )
    return "\n\n".join(parts) + "\n"


def parse_reply(reply: str) -> str:
    """The program of a reply: the last fenced block holding one, else the last ``(program``
    form written bare. Raises :class:`ProposalError` when there is none."""
    blocks = [b for b in _FENCE.findall(reply) if "(program" in b]
    if blocks:
        return blocks[-1].strip()
    start = reply.rfind("(program")
    if start < 0:
        raise ProposalError("the reply holds no program: expected (program ...) in a fenced block")
    depth = 0
    for i in range(start, len(reply)):
        depth += {"(": 1, ")": -1}.get(reply[i], 0)
        if depth == 0:
            return reply[start : i + 1]
    return reply[start:].strip()  # unbalanced: the parser will say where


def complete(llm, prompt: str) -> str:
    """Ask ``llm`` once and return the program of its reply."""
    return parse_reply(llm(prompt, system=SYSTEM))


def propose(
    llm,
    problem: Problem,
    parent: Record,
    inspirations: list[Record],
    knowledge: Knowledge | None,
    blind: bool,
) -> str:
    """One candidate program, as text: the reply of ``llm`` to the prompt for this parent."""
    return complete(llm, build_prompt(problem, parent, inspirations, knowledge, blind))
