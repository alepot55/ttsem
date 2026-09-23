"""The torch.compile demo runs end to end without a GPU and gives a verdict for every case."""

import io

import pytest

pytest.importorskip("torch")
pytest.importorskip("triton")

from ttsem import torch_demo


def test_every_case_gets_a_verdict():
    out = io.StringIO()
    assert torch_demo.main([], out) == 0
    text = out.getvalue()
    for case in torch_demo.CASES:
        assert case.issue in text
    assert "flagged on this torch" in text


def test_a_race_names_the_store_and_the_load():
    out = io.StringIO()
    verdict = torch_demo.run_case(torch_demo.CASES[0], out)
    if verdict == "race":  # the torch releases before pytorch#198010
        assert "tl.store" in out.getvalue() and "tl.load" in out.getvalue()
    else:
        assert verdict == "ok"
