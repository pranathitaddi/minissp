"""
tests/test_data.py

Corpus scoping and the raw readers: the parts of data.py that the Kaggle /
Colab pipeline depends on before any model is loaded.
"""

from __future__ import annotations

import gzip
import json

import pytest

from minissp.data import _build_text_matcher, iter_wiki, read_test, scope_corpus

try:
    import ahocorasick  # noqa: F401
    HAVE_AHO = True
except ImportError:
    HAVE_AHO = False


# ---------------------------------------------------------------------------
# read_test / iter_wiki
# ---------------------------------------------------------------------------

def test_read_test_strips_the_searchr1_source_prefix(tmp_path):
    path = tmp_path / "test.jsonl"
    rows = [
        {"data_source": "searchR1_nq",
         "reward_model": {"ground_truth": {"target": ["Paris"]}},
         "extra_info": {"question": "capital of france?"}},
        {"data_source": "hotpotqa",
         "reward_model": {"ground_truth": {"target": ["Rome"]}},
         "extra_info": {"question": "capital of italy?"}},
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows))
    out = read_test(path)
    assert [r["source"] for r in out] == ["nq", "hotpotqa"]


def test_iter_wiki_unquotes_search_r1_titles(tmp_path):
    path = tmp_path / "wiki.jsonl.gz"
    with gzip.open(path, "wt", encoding="utf-8") as f:
        f.write(json.dumps({"id": "0", "contents": '"Aaron"\nAaron is a prophet.'}) + "\n")
        f.write(json.dumps({"id": "1", "title": "Bob", "text": "Bob text."}) + "\n")
    docs = list(iter_wiki(path))
    assert docs[0] == {"id": "0", "title": "Aaron", "text": "Aaron is a prophet."}
    assert docs[1]["title"] == "Bob"


# ---------------------------------------------------------------------------
# text matcher: automaton must agree with the regex reference exactly
# ---------------------------------------------------------------------------

PATTERNS = ["paris", "new york", "york", "art", "c++", "san jose", "jose"]
TEXTS = [
    "she moved from paris to new york in 1990.",
    "parisian art museums",          # 'paris' not a whole word; 'art' is
    "the yorkshire terrier",         # 'york' not a whole word
    "c++ and san jose, california",
    "party at the start",            # 'art' only inside words
    "",
    "jose",
]


@pytest.mark.skipif(not HAVE_AHO, reason="pyahocorasick not installed")
def test_automaton_matcher_agrees_with_regex_matcher():
    naive = _build_text_matcher(PATTERNS, use_automaton=False)
    fast = _build_text_matcher(PATTERNS, use_automaton=True)
    for text in TEXTS:
        assert fast(text) == naive(text), text


def test_regex_matcher_respects_word_boundaries():
    match = _build_text_matcher(PATTERNS, use_automaton=False)
    found = {PATTERNS[i] for i in match("she moved from paris to new york in 1990.")}
    assert found == {"paris", "new york", "york"}
    assert match("parisian party") == []


# ---------------------------------------------------------------------------
# scope_corpus
# ---------------------------------------------------------------------------

def _wiki():
    return [
        {"id": "1", "title": "Paris", "text": "Paris is the capital of France."},
        {"id": "2", "title": "Louvre", "text": "The Louvre in Paris holds art from Rome."},
        {"id": "3", "title": "Unrelated", "text": "Nothing to see here."},
        {"id": "4", "title": "Rome", "text": "Rome is in Italy."},
    ]


@pytest.mark.parametrize("use_automaton", [False, pytest.param(
    True, marks=pytest.mark.skipif(not HAVE_AHO, reason="pyahocorasick not installed"))])
def test_scope_counts_a_shared_passage_toward_every_entity(use_automaton):
    kept, coverage = scope_corpus(iter(_wiki()), answers={"Paris"}, eval_targets={"Rome"},
                                  n_distractors=10, use_automaton=use_automaton)
    # Passage 2 mentions both entities: stored once, counted for each.
    assert coverage == {"Paris": 2, "Rome": 2}
    ids = [p["id"] for p in kept]
    assert sorted(ids) == ["1", "2", "3", "4"]  # "3" is the only distractor
    assert len(ids) == len(set(ids))


def test_scope_respects_per_entity_cap():
    _, coverage = scope_corpus(iter(_wiki()), answers={"Paris"}, eval_targets=set(),
                               per_entity_cap=1, n_distractors=0, use_automaton=False)
    assert coverage == {"Paris": 1}
