"""
tests/test_train.py

Phase 3 of plan.md (§3.2-3.5): rewards, GRPO advantages, the loss mask, the
zero-gradient property the loss mask exists to provide, and the replay buffer.

None of this needs a GPU or an LLM:
  - the loss-mask tests use a whitespace-free character tokenizer stub
  - the gradient test uses a two-layer toy model with a `.logits` output, which
    is the only thing `train.token_logprobs` actually requires
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from minissp.judge import JudgeParseError
from minissp.rollout import Traj, parse
from minissp.train import (
    DUMMY_QUESTION,
    ReplayBuffer,
    ReplayItem,
    baseline_advantages,
    build_loss_mask,
    compute_update,
    grpo_advantages,
    k3_kl,
    proposer_reward,
    solver_rewards,
    token_logprobs,
    traj_answer,
)

# ---------------------------------------------------------------------------
# §3.3 advantages
# ---------------------------------------------------------------------------


def test_grpo_advantages_sum_to_zero():
    # The plan's named case: one group of four.
    adv = grpo_advantages([1.0, 0.0, 0.0, 1.0], 4)
    assert len(adv) == 4
    assert sum(adv) == pytest.approx(0.0, abs=1e-6)


def test_grpo_advantages_are_per_contiguous_group():
    # Two groups of two: the second group is all-equal and gets ~0 advantage,
    # which must not be contaminated by the first group's spread.
    adv = grpo_advantages([1.0, 0.0, 1.0, 1.0], 2)
    assert sum(adv[:2]) == pytest.approx(0.0, abs=1e-6)
    assert adv[0] > 0 > adv[1]
    assert adv[2] == pytest.approx(0.0, abs=1e-6)
    assert adv[3] == pytest.approx(0.0, abs=1e-6)


def test_grpo_advantages_zero_variance_group_is_finite():
    # All-correct or all-wrong groups give GRPO no signal; the 1e-6 floor keeps
    # that a zero rather than a NaN.
    adv = grpo_advantages([1.0, 1.0, 1.0, 1.0], 4)
    assert all(math.isfinite(a) and a == pytest.approx(0.0, abs=1e-6) for a in adv)


def test_grpo_advantages_rejects_a_ragged_batch():
    with pytest.raises(ValueError):
        grpo_advantages([1.0, 0.0, 1.0], 2)


def test_baseline_advantages_centre_on_the_batch_mean():
    adv = baseline_advantages([1.0, 0.0, 0.5])
    assert sum(adv) == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# §3.3 proposer reward
# ---------------------------------------------------------------------------


def test_proposer_reward_one_minus_acc():
    assert proposer_reward("1-acc", 0.0) == 1.0
    assert proposer_reward("1-acc", 1.0) == 0.0
    assert proposer_reward("1-acc", 0.25) == pytest.approx(0.75)


def test_intermediate_difficulty_peaks_at_half():
    # [PAPER-UNCONFIRMED] placeholder formula; the test pins its *shape*
    # (peak at 0.5, zero at both extremes), not the exact reference formula.
    assert proposer_reward("intermediate_difficulty", 0.5) == pytest.approx(1.0)
    assert proposer_reward("intermediate_difficulty", 0.0) == pytest.approx(0.0)
    assert proposer_reward("intermediate_difficulty", 1.0) == pytest.approx(0.0)
    assert (proposer_reward("intermediate_difficulty", 0.4)
            > proposer_reward("intermediate_difficulty", 0.1))


def test_unknown_proposer_reward_raises():
    with pytest.raises(ValueError):
        proposer_reward("mystery", 0.5)


# ---------------------------------------------------------------------------
# §3.3 solver rewards
# ---------------------------------------------------------------------------


class StubJudge:
    """Judge stand-in: no model, no GPU. Verdict fixed per prediction string."""

    def __init__(self, correct=True, raise_on=None):
        self.correct = correct
        self.raise_on = raise_on or set()
        self.seen: list[str] = []

    def is_correct(self, question, targets, prediction):
        single = isinstance(question, str)
        predictions = [prediction] if single else list(prediction)
        self.seen.extend(predictions)
        out = []
        for p in predictions:
            if p in self.raise_on:
                raise JudgeParseError(f"unparseable verdict for {p!r}")
            out.append(self.correct)
        return out[0] if single else out


GOOD = ("<think>a</think><search>q</search>"
        "<information>x</information><answer>Shakespeare</answer>")
BROKEN = "<think>a</think><answer>Shakespeare"


def _traj(text: str) -> Traj:
    turns, valid = parse(text, terminal_tag="answer")
    return Traj(text=text, turns=turns, format_valid=valid)


def test_invalid_format_trajectory_reward_is_exactly_zero():
    judge = StubJudge(correct=True)  # would say "correct" if it were ever asked
    rewards = solver_rewards(judge, ["q?"], [["Shakespeare"]], [_traj(BROKEN)])
    assert rewards == [0.0]
    assert judge.seen == [], "a malformed trajectory must never reach the judge"


def test_valid_trajectory_scores_one_when_the_judge_agrees():
    assert solver_rewards(StubJudge(True), ["q?"], [["Shakespeare"]],
                          [_traj(GOOD)]) == [1.0]


def test_valid_trajectory_scores_zero_when_the_judge_disagrees():
    assert solver_rewards(StubJudge(False), ["q?"], [["Shakespeare"]],
                          [_traj(GOOD)]) == [0.0]


def test_judge_parse_errors_are_counted_not_silently_zero():
    judge = StubJudge(True, raise_on={"Shakespeare"})
    stats: dict = {}
    rewards = solver_rewards(judge, ["q?"], [["Shakespeare"]], [_traj(GOOD)], stats)
    assert rewards == [0.0]
    assert stats["judge_errors"] == 1


def test_traj_answer_takes_the_terminal_answer():
    assert traj_answer(_traj(GOOD)) == "Shakespeare"


# ---------------------------------------------------------------------------
# §3.2 loss mask
# ---------------------------------------------------------------------------


class CharTokenizer:
    """One token per character; ids are code points.

    This is deliberately the simplest tokenizer for which
    "re-encode the prefix and take the token count" is exactly checkable.
    """

    pad_token_id = 0

    def __call__(self, text, add_special_tokens=False, **kwargs):
        return {"input_ids": [ord(c) for c in text]}


PROMPT = "PROMPT:"
INFO = "<information>\ndoc\n</information>"


def _traj_with_info() -> Traj:
    body = "<think>a</think>\n" + INFO + "\n<answer>x</answer>"
    start = body.index("<information>")
    end = start + len(INFO)
    return Traj(text=body, prompt=PROMPT, info_spans=[(start, end)])


def test_loss_mask_zeroes_the_prompt():
    tok = CharTokenizer()
    ids, mask = build_loss_mask(tok, _traj_with_info())
    assert len(ids) == len(mask)
    assert mask[:len(PROMPT)] == [0] * len(PROMPT)
    assert mask[len(PROMPT)] == 1  # first generated character


def test_loss_mask_zeroes_information_spans_including_the_tags():
    tok = CharTokenizer()
    traj = _traj_with_info()
    ids, mask = build_loss_mask(tok, traj)
    full = traj.prompt + traj.text
    masked = "".join(c for c, m in zip(full, mask) if m == 0)
    assert PROMPT in masked
    assert INFO in masked
    # the tags themselves are masked (the plan's stated default)
    assert "<information>" in masked and "</information>" in masked
    # everything the policy actually wrote is still trained on
    kept = "".join(c for c, m in zip(full, mask) if m == 1)
    assert "<think>a</think>" in kept
    assert "<answer>x</answer>" in kept
    assert "doc" not in kept


def test_loss_mask_can_keep_the_tags_unmasked():
    # [INFERRED] the plan flags tag masking as an unconfirmed repo detail, so
    # the decision is a flag rather than a hard-coded behaviour.
    tok = CharTokenizer()
    traj = _traj_with_info()
    _ids, mask = build_loss_mask(tok, traj, include_tags=False)
    full = traj.prompt + traj.text
    kept = "".join(c for c, m in zip(full, mask) if m == 1)
    assert "<information>" in kept and "</information>" in kept
    assert "doc" not in kept


def test_loss_mask_with_no_information_spans_masks_only_the_prompt():
    tok = CharTokenizer()
    traj = Traj(text="<answer>x</answer>", prompt=PROMPT)
    _ids, mask = build_loss_mask(tok, traj)
    assert sum(mask) == len(traj.text)


# ---------------------------------------------------------------------------
# §3.2 the property the mask exists for: masked tokens get zero gradient
# ---------------------------------------------------------------------------


class ToyLM(nn.Module):
    """Two-layer per-position language model with a transformers-shaped output.

    Position t's logits depend only on input_ids[t], which is what makes the
    gradient attribution below unambiguous.
    """

    def __init__(self, vocab: int = 8, hidden: int = 6) -> None:
        super().__init__()
        self.emb = nn.Embedding(vocab, hidden)
        self.proj = nn.Linear(hidden, vocab)

    def forward(self, input_ids, attention_mask=None):
        return SimpleNamespace(logits=self.proj(torch.tanh(self.emb(input_ids))))


def test_masked_tokens_receive_zero_gradient():
    torch.manual_seed(0)
    model = ToyLM()
    ids = torch.tensor([[0, 1, 2, 3, 4, 5]])
    attn = torch.ones_like(ids)

    # Positions 0 and 1 (which consume embeddings 0 and 1) are masked; the
    # rest are policy tokens.
    mask = torch.tensor([[0.0, 0.0, 1.0, 1.0, 1.0]])
    advantage = torch.tensor([1.5])

    logp = token_logprobs(model, ids, attn)
    assert logp.shape == mask.shape
    loss = -(mask * advantage.unsqueeze(1) * logp).sum() / mask.sum()
    loss.backward()

    grad = model.emb.weight.grad
    assert grad is not None
    # embeddings used only at masked positions
    assert torch.count_nonzero(grad[0]) == 0
    assert torch.count_nonzero(grad[1]) == 0
    # embeddings used at unmasked positions
    assert torch.count_nonzero(grad[2]) > 0
    assert torch.count_nonzero(grad[3]) > 0
    # position 5 is never consumed (it is only ever a target), and token 6/7
    # never appear at all
    assert torch.count_nonzero(grad[6]) == 0


def test_an_all_masked_sequence_produces_no_gradient_at_all():
    torch.manual_seed(0)
    model = ToyLM()
    ids = torch.tensor([[0, 1, 2, 3]])
    mask = torch.zeros(1, 3)
    logp = token_logprobs(model, ids, torch.ones_like(ids))
    (-(mask * logp).sum()).backward()
    assert torch.count_nonzero(model.emb.weight.grad) == 0
    assert torch.count_nonzero(model.proj.weight.grad) == 0


def test_k3_kl_is_non_negative_and_zero_at_equality():
    logp = torch.tensor([-1.0, -2.0, -0.5])
    assert torch.allclose(k3_kl(logp, logp), torch.zeros(3), atol=1e-7)
    assert bool((k3_kl(logp, logp + 0.3) >= 0).all())
    assert bool((k3_kl(logp, logp - 0.3) >= 0).all())


# ---------------------------------------------------------------------------
# §3.5 replay buffer
# ---------------------------------------------------------------------------


def _item(i: int) -> ReplayItem:
    return ReplayItem(question=f"q{i}", answer=f"a{i}",
                      docs=[{"id": str(i), "title": f"T{i}", "text": "body"}])


def test_replay_buffer_clears_on_multiples_of_reset_every():
    buf = ReplayBuffer()
    buf.extend(_item(i) for i in range(3))

    assert buf.maybe_clear(9, 10) is False
    assert len(buf) == 3
    assert buf.maybe_clear(10, 10) is True
    assert len(buf) == 0

    buf.extend(_item(i) for i in range(2))
    assert buf.maybe_clear(11, 10) is False
    assert buf.maybe_clear(20, 10) is True


def test_replay_buffer_does_not_clear_at_step_zero():
    # Step 0 is before any step has run; clearing there would be a no-op that
    # masks an off-by-one in the reset schedule.
    buf = ReplayBuffer()
    buf.push(_item(0))
    assert buf.maybe_clear(0, 10) is False
    assert len(buf) == 1


def test_fill_batch_prefers_fresh_then_replay_then_dummy():
    buf = ReplayBuffer()
    buf.extend(_item(i) for i in range(2))          # 2 replayable
    rows, counts = buf.fill_batch([_item(90)], size=5)
    assert counts == {"fresh": 1, "replay": 2, "dummy": 2}
    assert [r.question for r in rows[:3]] == ["q90", "q1", "q0"]  # newest replay first
    assert all(r.is_dummy for r in rows[3:])
    assert rows[-1].question == DUMMY_QUESTION


def test_fill_batch_is_all_fresh_when_fresh_is_enough():
    buf = ReplayBuffer()
    buf.push(_item(0))
    rows, counts = buf.fill_batch([_item(1), _item(2)], size=2)
    assert counts == {"fresh": 2, "replay": 0, "dummy": 0}
    assert [r.question for r in rows] == ["q1", "q2"]


def test_fill_batch_is_all_dummy_when_nothing_is_available():
    rows, counts = ReplayBuffer().fill_batch([], size=3)
    assert counts == {"fresh": 0, "replay": 0, "dummy": 3}
    assert all(r.is_dummy for r in rows)


def test_replay_buffer_serialises_round_trip(tmp_path):
    buf = ReplayBuffer()
    buf.extend(_item(i) for i in range(4))
    path = tmp_path / "replay.jsonl"
    buf.to_jsonl(path)

    restored = ReplayBuffer.from_jsonl(path)
    assert len(restored) == 4
    assert restored.serialize() == buf.serialize()
    assert list(restored.items)[0].docs == [{"id": "0", "title": "T0", "text": "body"}]


def test_replay_buffer_from_a_missing_file_is_empty(tmp_path):
    assert len(ReplayBuffer.from_jsonl(tmp_path / "nope.jsonl")) == 0


def test_replay_buffer_respects_maxlen():
    buf = ReplayBuffer(maxlen=2)
    buf.extend(_item(i) for i in range(5))
    assert [i.question for i in buf.items] == ["q3", "q4"]


# ---------------------------------------------------------------------------
# §3.4 the update itself, on a real peft adapter (CPU, no LLM)
# ---------------------------------------------------------------------------


def _peft_toy():
    from peft import LoraConfig, get_peft_model

    torch.manual_seed(0)
    return get_peft_model(ToyLM(vocab=16, hidden=8),
                          LoraConfig(r=4, lora_alpha=8, target_modules=["proj"]))


def test_compute_update_runs_end_to_end_and_moves_the_adapter():
    from minissp.config import Config

    model = _peft_toy()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-2)
    scaler = torch.amp.GradScaler("cpu", enabled=False)
    tok = SimpleNamespace(pad_token_id=0)
    samples = [
        ([1, 2, 3, 4, 5], [0, 0, 1, 1, 1], 1.2),
        ([1, 2, 3], [0, 1, 1], -0.8),          # ragged lengths must pad cleanly
        ([2, 3, 4, 5, 6, 7], [0, 0, 0, 1, 1, 1], 0.5),
    ]
    before = {k: v.clone() for k, v in model.state_dict().items() if "lora_B" in k}

    out = compute_update(model, tok, opt, scaler, samples, Config(run_id="x"))

    assert out["skipped"] is False
    assert math.isfinite(out["loss"]) and math.isfinite(out["grad_norm"])
    assert out["grad_norm"] > 0
    after = {k: v for k, v in model.state_dict().items() if "lora_B" in k}
    assert any(not torch.equal(before[k], after[k]) for k in before)


def test_compute_update_skips_when_every_token_is_masked():
    from minissp.config import Config

    model = _peft_toy()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-2)
    out = compute_update(model, SimpleNamespace(pad_token_id=0), opt, None,
                         [([1, 2, 3], [0, 0, 0], 1.0)], Config(run_id="x"))
    assert out["skipped"] is True


def test_kl_is_zero_against_the_disabled_adapter_at_initialisation():
    # peft initialises lora_B to zeros, so policy == reference exactly; a
    # non-zero KL here would mean disable_adapter() is not actually the
    # reference policy.
    from minissp.config import Config

    model = _peft_toy()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-2)
    out = compute_update(model, SimpleNamespace(pad_token_id=0), opt, None,
                         [([1, 2, 3, 4], [0, 1, 1, 1], 1.0)], Config(run_id="x"))
    assert out["kl"] == pytest.approx(0.0, abs=1e-6)
