"""
tests/test_pipeline.py

Phase 2 proof-adjacent tests (plan.md §2, "Tests (test_pipeline.py)"):
  - Retriever on a small real FAISS index returns the expected title in top-3
  - Judge.is_correct / rag_solve parsing on MOCKED generate() output
  - retrieval.py and judge.py import with zero side effects

Nothing here needs a GPU. The retrieval test needs sentence-transformers,
faiss and a locally cached e5 checkpoint; it skips cleanly otherwise.
"""

from __future__ import annotations

import json
import sys

import pytest

from minissp.judge import Judge, JudgeParseError, parse_answer, parse_verdict
from minissp.retrieval import Retriever, format_information

# ---------------------------------------------------------------------------
# A ~50-passage fixture corpus
# ---------------------------------------------------------------------------

_FIXTURE_TOPICS = [
    ("Hamlet", "Hamlet is a tragedy written by William Shakespeare around 1600."),
    ("Macbeth", "Macbeth is a tragedy by William Shakespeare about a Scottish general."),
    ("Mount Everest", "Mount Everest is Earth's highest mountain above sea level."),
    ("Marie Curie", "Marie Curie was a physicist and chemist who researched radioactivity."),
    ("Photosynthesis", "Photosynthesis converts light energy into chemical energy in plants."),
    ("The Great Gatsby", "The Great Gatsby is a 1925 novel by F. Scott Fitzgerald."),
    ("Amazon River", "The Amazon River in South America is the largest river by discharge."),
    ("Python (programming language)",
     "Python is a high-level programming language created by Guido van Rossum."),
    ("Apollo 11", "Apollo 11 was the spaceflight that first landed humans on the Moon in 1969."),
    ("Pacific Ocean", "The Pacific Ocean is the largest and deepest of Earth's oceans."),
]

_FILLER = [
    "regional transport", "municipal history", "seasonal weather", "local cuisine",
    "school district", "railway station", "county council", "botanical garden",
    "parish church", "village festival",
]


def _fixture_passages() -> list[dict]:
    passages = [
        {"id": f"doc{i}", "title": title, "text": text}
        for i, (title, text) in enumerate(_FIXTURE_TOPICS)
    ]
    for i in range(40):
        topic = _FILLER[i % len(_FILLER)]
        passages.append({
            "id": f"filler{i}",
            "title": f"Filler article {i} on {topic}",
            "text": f"This unrelated article number {i} describes {topic} in a small town.",
        })
    return passages


# ---------------------------------------------------------------------------
# Import hygiene (mirrors test_core.test_import_minissp_config)
# ---------------------------------------------------------------------------

def test_importing_retrieval_and_judge_has_no_side_effects(tmp_path, monkeypatch):
    for module in ("minissp.retrieval", "minissp.judge"):
        sys.modules.pop(module, None)
    monkeypatch.chdir(tmp_path)

    import minissp.judge  # noqa: F401
    import minissp.retrieval  # noqa: F401

    # No heavy dependency may be pulled in by the import itself.
    for heavy in ("faiss", "sentence_transformers", "bitsandbytes"):
        assert heavy not in sys.modules, f"{heavy} imported at module import time"
    # And nothing may be written to disk.
    assert list(tmp_path.iterdir()) == []


def test_constructing_objects_loads_nothing(tmp_path):
    retriever = Retriever(tmp_path / "nonexistent", "intfloat/e5-base-v2")
    assert retriever.is_loaded() is False
    with pytest.raises(RuntimeError):
        retriever.search(["anything"])

    judge = Judge("Qwen/Qwen2.5-3B-Instruct")
    assert judge.is_loaded() is False
    with pytest.raises(RuntimeError):
        judge._generate(["anything"], max_new_tokens=8)


def test_retriever_load_raises_when_index_is_missing(tmp_path):
    pytest.importorskip("faiss")
    pytest.importorskip("sentence_transformers")
    with pytest.raises(FileNotFoundError):
        Retriever(tmp_path, "intfloat/e5-base-v2").load()


# ---------------------------------------------------------------------------
# format_information
# ---------------------------------------------------------------------------

def test_format_information_shape():
    out = format_information([
        {"title": "Hamlet", "text": "A tragedy."},
        {"title": "Macbeth", "text": "Another tragedy."},
    ])
    assert out == '(Title: "Hamlet") A tragedy.\n(Title: "Macbeth") Another tragedy.'


def test_format_information_empty():
    assert format_information([]) == ""


# ---------------------------------------------------------------------------
# Real (tiny) index + Retriever.search
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def tiny_index(tmp_path_factory):
    pytest.importorskip("faiss", reason="faiss-cpu not installed")
    pytest.importorskip("sentence_transformers",
                        reason="sentence-transformers not installed")

    from minissp.data import build_index

    out_dir = tmp_path_factory.mktemp("index")
    corpus_path = out_dir / "corpus.jsonl"
    with open(corpus_path, "w", encoding="utf-8") as f:
        for passage in _fixture_passages():
            f.write(json.dumps(passage) + "\n")

    try:
        build_index(corpus_path, out_dir, embed_name="intfloat/e5-base-v2")
    except Exception as exc:  # noqa: BLE001 — offline / uncached checkpoint
        pytest.skip(f"could not build the e5 index (no cached checkpoint or no network): {exc}")
    return out_dir


def test_retriever_search_returns_expected_title_in_top3(tiny_index):
    retriever = Retriever(tiny_index, "intfloat/e5-base-v2").load()
    results = retriever.search(["who wrote the tragedy Hamlet"], topk=3)
    assert len(results) == 1
    titles = [d["title"] for d in results[0]]
    assert "Hamlet" in titles
    assert all("score" in d and "text" in d and "id" in d for d in results[0])


def test_retriever_search_is_batched(tiny_index):
    retriever = Retriever(tiny_index, "intfloat/e5-base-v2").load()
    results = retriever.search(
        ["what is the highest mountain on Earth", "which language did Guido van Rossum create"],
        topk=3,
    )
    assert len(results) == 2
    assert "Mount Everest" in [d["title"] for d in results[0]]
    assert "Python (programming language)" in [d["title"] for d in results[1]]


def test_retriever_search_empty_query_list(tiny_index):
    retriever = Retriever(tiny_index, "intfloat/e5-base-v2").load()
    assert retriever.search([]) == []


# ---------------------------------------------------------------------------
# Judge parsing on mocked generate() output — no model, no GPU, no network
# ---------------------------------------------------------------------------

class StubJudge(Judge):
    """Judge with the single generation seam replaced by canned completions."""

    def __init__(self, completions):
        super().__init__("stub-model", device="cpu")
        self._completions = list(completions)
        self.prompts_seen = None

    def _generate(self, prompts, max_new_tokens):
        self.prompts_seen = list(prompts)
        self.max_new_tokens_seen = max_new_tokens
        return self._completions[:len(prompts)]


@pytest.mark.parametrize("completion,expected", [
    ("Correct", True),
    ("Wrong", False),
    ("  Correct\n", True),
    ("correct", True),
    ("Correct.", True),
    ("Wrong — the prediction names a different person", False),
])
def test_parse_verdict_accepts(completion, expected):
    assert parse_verdict(completion) is expected


@pytest.mark.parametrize("completion", ["", "   ", "Maybe", "I think it is correct", "1"])
def test_parse_verdict_raises_on_anything_else(completion):
    with pytest.raises(JudgeParseError):
        parse_verdict(completion)


def test_is_correct_single_form():
    judge = StubJudge(["Correct"])
    assert judge.is_correct("Who wrote Hamlet?", ["Shakespeare"], "William Shakespeare") is True
    assert judge.max_new_tokens_seen == 8
    assert "Who wrote Hamlet?" in judge.prompts_seen[0]


def test_is_correct_batched_form():
    judge = StubJudge(["Correct", "Wrong"])
    out = judge.is_correct(
        ["q1", "q2"], [["a1"], ["a2"]], ["p1", "p2"])
    assert out == [True, False]


def test_is_correct_raises_judge_parse_error_never_coerces_to_false():
    judge = StubJudge(["Hmm, hard to say"])
    with pytest.raises(JudgeParseError):
        judge.is_correct("q", ["a"], "p")


def test_is_correct_rejects_mismatched_batch_sizes():
    judge = StubJudge(["Correct"])
    with pytest.raises(ValueError):
        judge.is_correct(["q1", "q2"], [["a1"]], ["p1", "p2"])


@pytest.mark.parametrize("completion,expected", [
    ("<answer>Paris</answer>", "Paris"),
    ("some preamble <answer> Paris </answer> trailing", "Paris"),
    ("Paris</answer>", "Paris"),          # prompt pre-seeded the opening tag
    ("no tags at all", ""),
    ("", ""),
])
def test_parse_answer(completion, expected):
    assert parse_answer(completion) == expected


def test_rag_solve_single_and_batched():
    docs = [{"id": "d0", "title": "Hamlet", "text": "A tragedy by Shakespeare."}]
    judge = StubJudge(["<answer>William Shakespeare</answer>"])
    assert judge.rag_solve("Who wrote Hamlet?", docs) == "William Shakespeare"
    assert judge.max_new_tokens_seen == 256
    assert '(Title: "Hamlet")' in judge.prompts_seen[0]

    judge = StubJudge(["<answer>a</answer>", "no answer tag"])
    assert judge.rag_solve(["q1", "q2"], [docs, docs]) == ["a", ""]


def test_rag_solve_rejects_mismatched_batch_sizes():
    judge = StubJudge([])
    with pytest.raises(ValueError):
        judge.rag_solve(["q1", "q2"], [[]])
