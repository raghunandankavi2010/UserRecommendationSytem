import json
from pathlib import Path

import numpy as np
import pytest

from matcher.linking import InterestLinker, normalize

VEC = {  # tiny fake embedding space: x = coffee-ness, y = trekking-ness
    "Coffee": [1, 0], "Trekking": [0, 1],
    "espresso": [1, 0.05], "walks": [0.6, 0.8], "beekeeping": [-1, 0],
}


def fake(texts):
    v = np.array([VEC[t] for t in texts], float)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


@pytest.fixture
def linker():
    return InterestLinker(["Coffee", "Trekking"], fake, auto_min=0.9, ask_min=0.5)


def test_three_decisions(linker):
    assert linker.link("espresso").decision == "linked" and linker.link("espresso").interest == "Coffee"
    r = linker.link("walks")
    assert r.decision == "did_you_mean" and r.interest is None and r.suggestions[0] == "Trekking"
    assert linker.link("beekeeping").decision == "new"


def test_exact_match_ignores_case_and_spaces(linker):
    r = linker.link("  coffee ")
    assert (r.decision, r.interest, r.score) == ("linked", "Coffee", 1.0)


def test_rejects_bad_input(linker):
    with pytest.raises(ValueError):
        linker.link("   ")
    with pytest.raises(ValueError):
        InterestLinker(["Coffee"], fake, auto_min=0.5, ask_min=0.9)


def test_normalize():
    assert normalize(" weekend\t treks \n") == "weekend treks"


def test_real_model_on_clear_cases():
    pytest.importorskip("sentence_transformers")
    from matcher.linking import st_encoder
    tax = json.loads((Path(__file__).parents[1] / "ml/data/taxonomy.json").read_text())
    linker = InterestLinker([l for g in tax.values() for l in g], st_encoder())
    hiking = linker.link("hiking")  # gte-small ranks Walking first; aliases/fine-tuning should fix this
    assert "Trekking" in hiking.suggestions and hiking.interest != "Walking"
    assert "Filter coffee" in linker.link("filter kaapi").suggestions
    assert linker.link("beekeeping").decision != "linked"
