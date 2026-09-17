
import dataclasses
import random

import pytest

from minissp.config import Config
from minissp.prompts import (
    TAGS,
    judge_prompt,
    proposer_prompt,
    rag_solver_prompt,
    solver_prompt,
)
from minissp.rollout import (
    Traj,
    Turn,
    build_rag_materials,
    parse,
    rag_verify,
    rule_filter,
)


def test_import_minissp_config():
    # Phase 0 proof step: importing config must have zero side effects
    # (no downloads, no GPU/model calls). If this import does anything
    # heavier than defining the dataclass, that's a Phase 0 regression.
    import minissp.config  # noqa: F401


def test_run_id_is_required():
    with pytest.raises(TypeError):
        Config()  # run_id has no default on purpose


def test_minimal_construction():
    cfg = Config(run_id="t4-smoke")
    assert cfg.run_id == "t4-smoke"


def test_t4_defaults():
    cfg = Config(run_id="t4-smoke")
    assert cfg.seed == 0
    assert cfg.policy_name == "Qwen/Qwen2.5-1.5B-Instruct"
    assert cfg.judge_name == "Qwen/Qwen2.5-3B-Instruct"
    assert cfg.embed_name == "intfloat/e5-base-v2"
    assert cfg.rag_solver == "judge"
    assert cfg.proposer_reward == "1-acc"
    assert cfg.batch_answers == 8
    assert cfg.n_solver == 4
    assert cfg.max_turns_proposer == 3
    assert cfg.max_turns_solver == 3
    assert cfg.topk == 3
    assert cfg.noise_docs == 4
    assert cfg.max_prompt_tokens == 1536
    assert cfg.max_new_tokens_per_turn == 256
    assert cfg.max_total_tokens == 2048
    assert cfg.lora_r == 16
    assert cfg.lora_alpha == 32
    assert cfg.lr == 2e-5
    assert cfg.kl_beta == 0.01
    assert cfg.replay_reset_every == 10
    assert cfg.eval_every == 10
    assert cfg.drive_root == "/content/drive/MyDrive/minissp"


def test_rag_solver_accepts_policy_alternative():
    cfg = Config(run_id="t4-smoke", rag_solver="policy")
    assert cfg.rag_solver == "policy"


def test_proposer_reward_accepts_intermediate_difficulty():
    cfg = Config(run_id="t4-smoke", proposer_reward="intermediate_difficulty")
    assert cfg.proposer_reward == "intermediate_difficulty"


def test_every_field_is_overridable_via_kwargs():
    # Every field must be settable by name — this is what the CLI
    # override loop (0.6) relies on, so the test doubles as a guard
    # against a field being renamed without updating that loop.
    overrides = {
        "run_id": "override-run",
        "seed": 7,
        "batch_answers": 16,
        "lr": 1e-4,
        "kl_beta": 0.05,
    }
    cfg = Config(**overrides)
    for key, value in overrides.items():
        assert getattr(cfg, key) == value


def test_config_field_names_match_plan():
    # Fixes the field set so a silent rename/drop fails loudly here
    # rather than surfacing later as a confusing CLI or training bug.
    expected = {
        "run_id", "seed", "policy_name", "judge_name", "embed_name",
        "rag_solver", "proposer_reward", "batch_answers", "n_solver",
        "max_turns_proposer", "max_turns_solver", "topk", "noise_docs",
        "max_prompt_tokens", "max_new_tokens_per_turn", "max_total_tokens",
        "lora_r", "lora_alpha", "lr", "kl_beta", "replay_reset_every",
        "eval_every", "drive_root",
    }
    actual = {f.name for f in dataclasses.fields(Config)}
    assert actual == expected

# ---------------------------------------------------------------------------
# Phase 2 — rollout.py: parser, rule filter, RAG materials
# ---------------------------------------------------------------------------

GOOD_SOLVER = (
    "<think>I need to look this up.</think>\n"
    "<search>who wrote hamlet</search>\n"
    "<information>\n(Title: \"Hamlet\") Hamlet is a tragedy by Shakespeare.\n</information>\n"
    "<think>Shakespeare then.</think>\n"
    "<answer>William Shakespeare</answer>"
)

GOOD_PROPOSER = (
    "<think>Let me ground this.</think>\n"
    "<search>tragedy written around 1600</search>\n"
    "<information>\n(Title: \"Hamlet\") Hamlet is a tragedy by Shakespeare.\n</information>\n"
    "<question>Which English playwright wrote the tragedy Hamlet around the year 1600?</question>"
)

# (name, text, terminal_tag, expected_format_valid)
PARSER_TABLE = [
    ("well_formed_solver", GOOD_SOLVER, "answer", True),
    ("well_formed_proposer", GOOD_PROPOSER, "question", True),
    ("unclosed_tag", "<think>never closed<answer>x</answer>", "answer", False),
    ("nested_tags", "<think>a<search>b</search></think><answer>x</answer>", "answer", False),
    ("two_terminal_tags", "<answer>a</answer><answer>b</answer>", "answer", False),
    ("text_outside_tags", "<think>a</think> loose text <answer>x</answer>", "answer", False),
    ("empty_trajectory", "", "answer", False),
    ("think_only_no_terminal", "<think>hmm</think>", "answer", False),
    ("trailing_whitespace_ok", "<think>a</think>\n<answer>x</answer>   \n\n", "answer", True),
    ("stray_closing_tag", "</think><answer>x</answer>", "answer", False),
    ("content_after_terminal", "<answer>x</answer><think>more</think>", "answer", False),
    ("wrong_terminal_for_role", GOOD_SOLVER, "question", False),
    # Decision (documented in rollout.parse): the parser is context-free and
    # does NOT police turn ORDER, so <information> before any <search> is
    # shape-valid. Ordering is enforced by generation (stop strings), not here.
    ("information_before_search", "<information>x</information><answer>y</answer>", "answer", True),
]


@pytest.mark.parametrize("name,text,terminal,expected", PARSER_TABLE,
                         ids=[row[0] for row in PARSER_TABLE])
def test_parser_table(name, text, terminal, expected):
    _turns, valid = parse(text, terminal_tag=terminal)
    assert valid is expected


def test_parse_returns_turns_with_offsets():
    turns, valid = parse(GOOD_SOLVER, terminal_tag="answer")
    assert valid is True
    assert [t.kind for t in turns] == ["think", "search", "information", "think", "answer"]
    assert turns[-1].text == "William Shakespeare"
    for turn in turns:
        assert GOOD_SOLVER[turn.start:turn.end].startswith(f"<{turn.kind}>")
        assert GOOD_SOLVER[turn.start:turn.end].endswith(f"</{turn.kind}>")


def test_parse_is_role_agnostic_without_terminal_tag():
    assert parse(GOOD_SOLVER)[1] is True
    assert parse(GOOD_PROPOSER)[1] is True


def _traj(text):
    turns, valid = parse(text, terminal_tag="question")
    return Traj(text=text, turns=turns, format_valid=valid)


def test_rule_filter_pass():
    ok, reason = rule_filter(_traj(GOOD_PROPOSER), "William Shakespeare")
    assert (ok, reason) == (True, "ok")


def test_rule_filter_rejects_invalid_format():
    ok, reason = rule_filter(_traj("<question>unclosed"), "William Shakespeare")
    assert (ok, reason) == (False, "format_invalid")


def test_rule_filter_rejects_no_search():
    text = ("<think>no searching at all</think>"
            "<question>Which English playwright wrote the tragedy Hamlet around 1600?</question>")
    ok, reason = rule_filter(_traj(text), "William Shakespeare")
    assert (ok, reason) == (False, "no_search")


def test_rule_filter_rejects_short_question():
    text = ("<search>hamlet</search><information>x</information>"
            "<question>Who wrote Hamlet?</question>")
    ok, reason = rule_filter(_traj(text), "William Shakespeare")
    assert (ok, reason) == (False, "question_too_short")


def test_rule_filter_rejects_answer_in_question():
    text = ("<search>hamlet</search><information>x</information>"
            "<question>Did the playwright william shakespeare, write Hamlet in England "
            "around the year 1600?</question>")
    ok, reason = rule_filter(_traj(text), "William Shakespeare")
    assert (ok, reason) == (False, "answer_in_question")


def test_rule_filter_rejects_missing_question_turn():
    # format_valid forced True to exercise the defensive no_question branch
    traj = Traj(text="", turns=[Turn("answer", "x", 0, 1)], format_valid=True)
    ok, reason = rule_filter(traj, "x")
    assert (ok, reason) == (False, "no_question")


# -- rag_verify -------------------------------------------------------------

class FakeSolver:
    """Stands in for Judge / policy: no model, no GPU, no network."""

    def __init__(self, answer="William Shakespeare", correct=True):
        self.answer = answer
        self.correct = correct
        self.last_docs = None

    def rag_solve(self, question, docs):
        self.last_docs = docs
        return self.answer

    def is_correct(self, question, targets, prediction):
        return self.correct


def _docs(prefix, k):
    return [{"id": f"{prefix}{i}", "title": f"T{prefix}{i}", "text": f"body {prefix}{i}"}
            for i in range(k)]


def test_rag_verify_materials_are_exactly_proposer_plus_k_noise_unique():
    proposer_docs = _docs("p", 3)
    pool = proposer_docs + _docs("o", 20)  # overlapping pool must be deduped
    fake = FakeSolver()
    assert rag_verify("q?", "William Shakespeare", proposer_docs, pool, fake,
                      k_noise=4, rng=random.Random(0)) is True
    materials = fake.last_docs
    assert len(materials) == len(proposer_docs) + 4
    ids = [d["id"] for d in materials]
    assert len(set(ids)) == len(ids)
    assert set(d["id"] for d in proposer_docs).issubset(set(ids))


def test_build_rag_materials_is_deterministic_under_a_seeded_rng():
    proposer_docs = _docs("p", 2)
    pool = _docs("o", 10)
    a = build_rag_materials(proposer_docs, pool, 4, random.Random(1))
    b = build_rag_materials(proposer_docs, pool, 4, random.Random(1))
    assert [d["id"] for d in a] == [d["id"] for d in b]


def test_build_rag_materials_caps_at_pool_size():
    materials = build_rag_materials(_docs("p", 2), _docs("o", 1), 4, random.Random(0))
    assert len(materials) == 3


def test_rag_verify_requires_an_explicit_rng():
    with pytest.raises(ValueError):
        rag_verify("q?", "a", _docs("p", 1), _docs("o", 5), FakeSolver())


def test_rag_verify_false_when_solver_produces_nothing():
    fake = FakeSolver(answer="")
    assert rag_verify("q?", "a", _docs("p", 1), _docs("o", 5), fake,
                      rng=random.Random(0)) is False


def test_rag_verify_false_when_judge_says_wrong():
    fake = FakeSolver(correct=False)
    assert rag_verify("q?", "a", _docs("p", 1), _docs("o", 5), fake,
                      rng=random.Random(0)) is False


# ---------------------------------------------------------------------------
# Phase 2 — prompts.py
# ---------------------------------------------------------------------------

def test_tags_constant_matches_plan():
    assert TAGS == ("think", "search", "information", "answer", "question")


def test_proposer_prompt_embeds_answer_n_and_examples():
    out = proposer_prompt("Leo Tolstoy", 2, "Question 1: ... Question 2: ... Question 3: ...")
    assert "Leo Tolstoy" in out
    assert "exactly 2" in out
    assert "Question 1:" in out
    assert "<question>" in out


def test_solver_prompt_asks_for_answer_tag():
    out = solver_prompt("Who wrote Hamlet?")
    assert "Who wrote Hamlet?" in out
    assert "<answer>" in out and "<search>" in out


def test_rag_solver_prompt_forbids_search():
    out = rag_solver_prompt("Who wrote Hamlet?", '(Title: "Hamlet") A tragedy.')
    assert "(Title: \"Hamlet\")" in out
    assert "ONLY the documents" in out
    assert "Do not write <search>" in out


def test_judge_prompt_lists_all_targets_and_pins_the_verdict_format():
    out = judge_prompt("Who wrote Hamlet?", ["Shakespeare", "William Shakespeare"], "Bill S.")
    assert "- Shakespeare" in out and "- William Shakespeare" in out
    assert "Bill S." in out
    assert "exactly one word: Correct or Wrong" in out


# ---------------------------------------------------------------------------
# Phase 0.6 — the CLI override loop
# ---------------------------------------------------------------------------

def test_cli_override_loop_casts_by_field_type():
    from minissp.config import parse_args

    cfg = parse_args(["--run-id", "x", "--lr", "0.001", "--batch-answers", "16",
                      "--policy-name", "Qwen/Qwen2.5-0.5B-Instruct"])
    assert cfg.run_id == "x"
    assert cfg.lr == 0.001 and isinstance(cfg.lr, float)
    assert cfg.batch_answers == 16 and isinstance(cfg.batch_answers, int)
    assert cfg.policy_name == "Qwen/Qwen2.5-0.5B-Instruct"
    # untouched fields keep the dataclass defaults
    assert cfg.kl_beta == 0.01
    assert cfg.n_solver == 4


def test_cli_exposes_one_hyphenated_flag_per_field():
    import dataclasses as _dc

    from minissp.config import build_parser

    flags = {a.option_strings[0] for a in build_parser()._actions
             if a.option_strings and a.option_strings[0] != "-h"}
    expected = {"--" + f.name.replace("_", "-") for f in _dc.fields(Config)}
    assert flags == expected


def test_cli_requires_run_id():
    from minissp.config import parse_args

    with pytest.raises(SystemExit):
        parse_args(["--seed", "1"])


def test_cli_rejects_a_non_numeric_value_for_a_numeric_field():
    from minissp.config import parse_args

    with pytest.raises(SystemExit):
        parse_args(["--run-id", "x", "--seed", "banana"])


def test_cli_can_be_extended_with_entrypoint_flags():
    # train.py adds --max-steps to the same parser; config_from_namespace must
    # ignore attributes that are not Config fields.
    import argparse

    from minissp.config import add_config_args, config_from_namespace

    parser = argparse.ArgumentParser()
    add_config_args(parser)
    parser.add_argument("--max-steps", dest="max_steps", type=int, default=50)
    args = parser.parse_args(["--run-id", "y", "--max-steps", "3"])
    cfg = config_from_namespace(args)
    assert cfg.run_id == "y"
    assert args.max_steps == 3
    assert not hasattr(cfg, "max_steps")
