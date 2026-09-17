"""
minissp/train.py

Phase 3 of plan.md (§3.1-3.6, §3.8): rewards, advantages, the loss mask, the
REINFORCE/GRPO loss with a k3 KL penalty, the replay buffer, one self-play
step, and the resume loop.

    python -m minissp.train --run-id t4-001 --max-steps 50

There is no `--resume` flag: the presence of `runs/<run_id>/latest` decides
(plan §3.8).

Unlike retrieval.py / judge.py, this module is allowed to import torch and
friends at module level — it is inherently the training entry point. What it
must NOT do at import time is download a model or touch the GPU or disk; all
of that lives inside functions and `main()`.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
import uuid
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch

from minissp.config import Config, add_config_args, config_from_namespace
from minissp.data import ANSWERS_TOY_PATH, EVAL_TOY_PATH
from minissp.judge import JudgeParseError
from minissp.prompts import proposer_prompt, solver_prompt
from minissp.rollout import Traj, rag_verify, rule_filter, run_trajectories
from minissp.runtime import (
    METRICS_COLUMNS,
    MetricsWriter,
    capture_rng_state,
    config_hash,
    git_sha,
    gpu_name,
    index_sha,
    load_checkpoint,
    read_latest,
    restore_rng_state,
    save_checkpoint,
)

TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",
                  "gate_proj", "up_proj", "down_proj"]

# The padding row used when neither fresh nor replayed questions fill the
# batch (plan §3.5). Its advantage is forced to zero, so it contributes shape
# but no gradient — it exists only to keep tensor sizes and timing stable.
DUMMY_QUESTION = "What is the capital city of France?"
DUMMY_ANSWER = "Paris"

MICRO_BATCH = 2          # trajectories per forward/backward on a T4
GRAD_CLIP = 1.0


# ---------------------------------------------------------------------------
# §3.3 rewards and advantages
# ---------------------------------------------------------------------------

def traj_answer(traj: Traj) -> str:
    """The content of the trajectory's terminal <answer>, or ""."""
    answers = [t for t in traj.turns if t.kind == "answer"]
    return answers[-1].text if answers else ""


def traj_question(traj: Traj) -> str:
    questions = [t for t in traj.turns if t.kind == "question"]
    return questions[-1].text if questions else ""


def solver_rewards(judge, questions: list[str], targets: list[list[str]],
                   trajs: list[Traj], stats: dict | None = None) -> list[float]:
    """1.0 if the judge calls the solver's answer equivalent, else 0.0.

    An invalid-format trajectory is exactly 0.0 and is never sent to the judge
    (plan §2.4: invalid ⇒ reward 0, never negative). A JudgeParseError is not
    coerced to a verdict either: the item scores 0.0 and is counted in
    `stats["judge_errors"]` so a degrading judge is visible in metrics rather
    than looking like a solver that got worse.
    """
    rewards = [0.0] * len(trajs)
    live: list[int] = []
    for i, traj in enumerate(trajs):
        if traj.format_valid and traj_answer(traj).strip():
            live.append(i)

    errors = 0
    if live:
        q = [questions[i] for i in live]
        t = [list(targets[i]) for i in live]
        p = [traj_answer(trajs[i]) for i in live]
        try:
            verdicts = judge.is_correct(q, t, p)
        except JudgeParseError:
            # Fall back to one call per item so a single unparseable verdict
            # cannot zero out the whole batch.
            verdicts = []
            for qi, ti, pi in zip(q, t, p):
                try:
                    verdicts.append(bool(judge.is_correct(qi, ti, pi)))
                except JudgeParseError:
                    verdicts.append(False)
                    errors += 1
        for i, verdict in zip(live, verdicts):
            rewards[i] = 1.0 if verdict else 0.0

    if stats is not None:
        stats["judge_errors"] = stats.get("judge_errors", 0) + errors
    return rewards


def grpo_advantages(rewards: list[float], group: int) -> list[float]:
    """(r - mean_g) / (std_g + 1e-6) over each contiguous group of `group`.

    Groups are contiguous because the solver rollouts for one question are
    laid out contiguously (question q occupies [q*n, (q+1)*n)).
    """
    if group <= 0:
        raise ValueError("group size must be positive")
    if len(rewards) % group != 0:
        raise ValueError(
            f"{len(rewards)} rewards is not a whole number of groups of {group}"
        )
    out: list[float] = []
    for start in range(0, len(rewards), group):
        chunk = rewards[start:start + group]
        mean = sum(chunk) / group
        var = sum((r - mean) ** 2 for r in chunk) / group
        std = math.sqrt(var)
        out.extend((r - mean) / (std + 1e-6) for r in chunk)
    return out


def proposer_reward(kind: str, solver_acc: float) -> float:
    """Reward for a proposer whose question the solver answered with `solver_acc`.

    "1-acc"                  the paper's headline objective: harder is better.
    "intermediate_difficulty" a peak at acc = 0.5 — questions that are neither
                              trivially easy nor impossible carry the most
                              learning signal.

    # TODO [PAPER-UNCONFIRMED] The exact `intermediate_difficulty` formula is
    # plan.md's "Still to confirm" item #1 (`grep -rn intermediate_difficulty
    # quarl/` in the reference repo). The triangular peak below is a stand-in
    # with the right shape and range [0, 1]; it must be replaced once the repo
    # is available. Nothing else in the pipeline depends on its exact form.
    """
    if kind == "1-acc":
        return 1.0 - solver_acc
    if kind == "intermediate_difficulty":
        return 1.0 - abs(2.0 * solver_acc - 1.0)
    raise ValueError(
        f"unknown proposer_reward {kind!r}; expected '1-acc' or 'intermediate_difficulty'"
    )


def baseline_advantages(rewards: list[float]) -> list[float]:
    """REINFORCE with a batch-mean baseline (plan §3.3): n=1 per answer, so
    there is no GRPO group to normalise within."""
    if not rewards:
        return []
    mean = sum(rewards) / len(rewards)
    return [r - mean for r in rewards]


# ---------------------------------------------------------------------------
# §3.2 loss mask
# ---------------------------------------------------------------------------

def build_loss_mask(tok, traj: Traj, include_tags: bool = True) -> tuple[list[int], list[int]]:
    """Tokenizes prompt+completion once and returns (input_ids, loss_mask).

    loss_mask is 1 for tokens the policy generated and 0 for
      - every prompt token, and
      - every `<information>…</information>` span.

    `include_tags=True` (the default) masks the tag tokens themselves along
    with the retrieved text. [INFERRED — plan §3.2 flags this as a reference
    repo detail that is not confirmed; the flag exists so flipping the
    decision is a one-word change rather than a rewrite.]

    Spans are located by re-encoding the prefix up to each boundary and taking
    token counts, never by aligning character offsets onto tokens after the
    fact (plan §3.2 is explicit about this).
    """
    full = traj.prompt + traj.text
    input_ids = tok(full, add_special_tokens=False)["input_ids"]
    n = len(input_ids)

    def prefix_len(char_offset: int) -> int:
        if char_offset <= 0:
            return 0
        return len(tok(full[:char_offset], add_special_tokens=False)["input_ids"])

    mask = [1] * n

    # prompt
    prompt_tokens = min(prefix_len(len(traj.prompt)), n)
    for i in range(prompt_tokens):
        mask[i] = 0

    offset = len(traj.prompt)
    for start, end in traj.info_spans:
        if not include_tags:
            # Narrow the span to the body between the tags.
            body = traj.text[start:end]
            open_tag, close_tag = "<information>", "</information>"
            if body.startswith(open_tag):
                start += len(open_tag)
            if body.endswith(close_tag):
                end -= len(close_tag)
        lo = min(prefix_len(offset + start), n)
        hi = min(prefix_len(offset + end), n)
        for i in range(lo, hi):
            mask[i] = 0

    return input_ids, mask


# ---------------------------------------------------------------------------
# §3.5 replay buffer
# ---------------------------------------------------------------------------

@dataclass
class ReplayItem:
    question: str
    answer: str
    docs: list[dict] = field(default_factory=list)
    # Dummy rows keep the batch rectangular but must never produce gradient.
    is_dummy: bool = False
    # Index into the step's proposer trajectories, when this row is fresh.
    proposer_idx: int | None = None


def dummy_item() -> ReplayItem:
    return ReplayItem(question=DUMMY_QUESTION, answer=DUMMY_ANSWER,
                      docs=[], is_dummy=True)


class ReplayBuffer:
    """Verified (question, answer, docs) triples, cleared every N steps.

    Paper B.1: of the four batch-fill strategies tested, a replay buffer reset
    every 10 steps won; oversampling was rejected as too expensive.
    """

    def __init__(self, maxlen: int = 512) -> None:
        self.items: deque[ReplayItem] = deque(maxlen=maxlen)

    def __len__(self) -> int:
        return len(self.items)

    def push(self, item: ReplayItem) -> None:
        self.items.append(item)

    def extend(self, items) -> None:
        for item in items:
            self.push(item)

    def clear(self) -> None:
        self.items.clear()

    def maybe_clear(self, step: int, every: int) -> bool:
        """Clears on steps that are a positive multiple of `every`."""
        if every <= 0 or step <= 0 or step % every != 0:
            return False
        self.clear()
        return True

    def fill_batch(self, fresh: list[ReplayItem], size: int) -> tuple[list[ReplayItem], dict]:
        """Fresh verified first, then replay (newest first), then dummies."""
        rows = list(fresh[:size])
        n_fresh = len(rows)

        n_replay = 0
        if len(rows) < size:
            for item in reversed(self.items):
                if len(rows) >= size:
                    break
                rows.append(item)
                n_replay += 1

        n_dummy = 0
        while len(rows) < size:
            rows.append(dummy_item())
            n_dummy += 1

        return rows, {"fresh": n_fresh, "replay": n_replay, "dummy": n_dummy}

    # -- serialization ----------------------------------------------------

    def serialize(self) -> list[dict]:
        return [asdict(item) for item in self.items]

    def load(self, rows) -> "ReplayBuffer":
        self.clear()
        for row in rows:
            self.push(ReplayItem(**row))
        return self

    def to_jsonl(self, path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for row in self.serialize():
                f.write(json.dumps(row) + "\n")

    @classmethod
    def from_jsonl(cls, path, maxlen: int = 512) -> "ReplayBuffer":
        buf = cls(maxlen=maxlen)
        path = Path(path)
        if not path.exists():
            return buf
        with open(path, "r", encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]
        return buf.load(rows)


# ---------------------------------------------------------------------------
# §3.1 / §3.4 model setup and loss
# ---------------------------------------------------------------------------

def build_policy(cfg: Config, device: str = "cuda"):
    """fp16 base + LoRA adapter (plan §3.1). Returns (model, tokenizer)."""
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(cfg.policy_name, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        cfg.policy_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map=device,
    )
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=0.05,
        target_modules=TARGET_MODULES,
        task_type="CAUSAL_LM",
    ))
    return model, tok


def token_logprobs(model, input_ids, attention_mask):
    """log p(y_t | y_<t) for every position t>=1. Shape [B, T-1]."""
    out = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = out.logits[:, :-1, :].float()
    targets = input_ids[:, 1:]
    logp = torch.log_softmax(logits, dim=-1)
    return logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)


def k3_kl(logp_policy, logp_ref):
    """Schulman's k3 estimator: exp(d) - d - 1 with d = logp_ref - logp_policy.

    Non-negative, unbiased, and far lower variance than the naive difference.
    """
    diff = (logp_ref - logp_policy).clamp(-20.0, 20.0)
    return torch.exp(diff) - diff - 1.0


def pad_batch(rows: list[tuple[list[int], list[int], float]], pad_id: int, device):
    """Right-pads (ids, mask, advantage) triples into tensors."""
    width = max(len(ids) for ids, _, _ in rows)
    ids_t, attn_t, mask_t, adv_t = [], [], [], []
    for ids, mask, adv in rows:
        pad = width - len(ids)
        ids_t.append(ids + [pad_id] * pad)
        attn_t.append([1] * len(ids) + [0] * pad)
        mask_t.append(mask + [0] * pad)
        adv_t.append(adv)
    return (
        torch.tensor(ids_t, dtype=torch.long, device=device),
        torch.tensor(attn_t, dtype=torch.long, device=device),
        torch.tensor(mask_t, dtype=torch.float32, device=device),
        torch.tensor(adv_t, dtype=torch.float32, device=device),
    )


def compute_update(model, tok, optimizer, scaler, samples, cfg,
                   micro_batch: int = MICRO_BATCH) -> dict:
    """One on-policy gradient update over `samples` (plan §3.4).

    samples: list of (input_ids, loss_mask, advantage).

        L = - Σ mask·A·logπ_θ / Σ mask  +  β · Σ mask·KL_k3 / Σ mask

    No PPO ratio or clipping: exactly one update per rollout batch, so the
    behaviour policy is the current policy.

    A non-finite loss or grad-norm skips the update entirely (plan §3.4) and
    reports `skipped=True` so the caller does not advance the step.
    """
    device = next(model.parameters()).device
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0

    total_mask = float(sum(sum(mask[1:]) for _, mask, _ in samples))
    if total_mask <= 0:
        return {"loss": 0.0, "kl": 0.0, "grad_norm": 0.0, "skipped": True}

    optimizer.zero_grad(set_to_none=True)
    loss_sum = 0.0
    kl_sum = 0.0
    use_amp = device.type == "cuda"

    for start in range(0, len(samples), micro_batch):
        chunk = samples[start:start + micro_batch]
        ids, attn, mask, adv = pad_batch(chunk, pad_id, device)
        mask = mask[:, 1:]  # align with the shifted logprobs

        with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
            logp = token_logprobs(model, ids, attn)
            with torch.no_grad():
                with model.disable_adapter():
                    logp_ref = token_logprobs(model, ids, attn)

        pg = -(mask * adv.unsqueeze(1) * logp).sum()
        kl = (mask * k3_kl(logp, logp_ref)).sum()
        loss = (pg + cfg.kl_beta * kl) / total_mask

        if not torch.isfinite(loss):
            optimizer.zero_grad(set_to_none=True)
            return {"loss": float("nan"), "kl": float("nan"),
                    "grad_norm": float("nan"), "skipped": True}

        if scaler is not None and scaler.is_enabled():
            scaler.scale(loss).backward()
        else:
            loss.backward()

        loss_sum += float(loss.detach())
        kl_sum += float(kl.detach()) / total_mask

    if scaler is not None and scaler.is_enabled():
        scaler.unscale_(optimizer)
    grad_norm = torch.nn.utils.clip_grad_norm_(
        [p for p in model.parameters() if p.requires_grad], GRAD_CLIP)

    if not torch.isfinite(grad_norm):
        optimizer.zero_grad(set_to_none=True)
        if scaler is not None and scaler.is_enabled():
            scaler.update()
        return {"loss": loss_sum, "kl": kl_sum,
                "grad_norm": float("nan"), "skipped": True}

    if scaler is not None and scaler.is_enabled():
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    return {"loss": loss_sum, "kl": kl_sum,
            "grad_norm": float(grad_norm), "skipped": False}


# ---------------------------------------------------------------------------
# §3.6 one step
# ---------------------------------------------------------------------------

def _read_jsonl(path) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


class Trainer:
    """Holds everything one step needs; `step()` is plan §3.6 verbatim."""

    def __init__(self, cfg: Config, run_dir, model, tok, retriever, judge,
                 answers, optimizer=None, scaler=None, session_id: str | None = None,
                 index_dir=None):
        self.cfg = cfg
        self.run_dir = Path(run_dir)
        self.model = model
        self.tok = tok
        self.retriever = retriever
        self.judge = judge
        self.answers = answers
        self.optimizer = optimizer
        self.scaler = scaler
        self.session_id = session_id or str(uuid.uuid4())
        self.index_dir = index_dir
        self.replay = ReplayBuffer()
        self.step_idx = 0
        self.metrics = MetricsWriter(self.run_dir / "metrics.jsonl")

    # -- verification ------------------------------------------------------

    def _rag_verifier(self):
        return self.model if self.cfg.rag_solver == "policy" else self.judge

    def step(self) -> dict:
        cfg = self.cfg
        step_idx = self.step_idx + 1
        # Deterministic per (seed, step): ints hash stably, unlike strings.
        rng = random.Random(hash((cfg.seed, step_idx)))
        stats: dict = {"judge_errors": 0}

        # 1 sample B answers
        rows = [self.answers[rng.randrange(len(self.answers))]
                for _ in range(cfg.batch_answers)]

        # 2 proposer rollouts
        t0 = time.time()
        prompts = [proposer_prompt(r["answer"], r["n"], r["examples"]) for r in rows]
        proposer_trajs = run_trajectories(self.model, self.tok, self.retriever,
                                          prompts, "proposer", cfg)
        t_propose = time.time() - t0

        # 3 rule filter + RAG verification
        t0 = time.time()
        pool = [d for t in proposer_trajs for d in t.docs_seen]
        rule_pass = 0
        fresh: list[ReplayItem] = []
        for i, (traj, row) in enumerate(zip(proposer_trajs, rows)):
            ok, _reason = rule_filter(traj, row["answer"])
            if not ok:
                continue
            rule_pass += 1
            question = traj_question(traj)
            verified = rag_verify(question, row["answer"], traj.docs_seen, pool,
                                  self._rag_verifier(), k_noise=cfg.noise_docs, rng=rng)
            if verified:
                fresh.append(ReplayItem(question=question, answer=row["answer"],
                                        docs=list(traj.docs_seen), proposer_idx=i))
        t_verify = time.time() - t0

        # 4 fill the batch
        batch, fill = self.replay.fill_batch(fresh, cfg.batch_answers)

        # 5 solver rollouts (B x n_solver, contiguous per question)
        t0 = time.time()
        solver_prompts = [solver_prompt(item.question)
                          for item in batch for _ in range(cfg.n_solver)]
        solver_trajs = run_trajectories(self.model, self.tok, self.retriever,
                                        solver_prompts, "solver", cfg)
        t_solve = time.time() - t0

        # 6 judge, rewards, GRPO advantages
        t0 = time.time()
        questions = [item.question for item in batch for _ in range(cfg.n_solver)]
        targets = [[item.answer] for item in batch for _ in range(cfg.n_solver)]
        rewards = solver_rewards(self.judge, questions, targets, solver_trajs, stats)
        advantages = grpo_advantages(rewards, cfg.n_solver)
        t_judge = time.time() - t0

        per_question_acc = [
            sum(rewards[q * cfg.n_solver:(q + 1) * cfg.n_solver]) / cfg.n_solver
            for q in range(len(batch))
        ]
        nonzero_var = sum(
            1 for q in range(len(batch))
            if len(set(rewards[q * cfg.n_solver:(q + 1) * cfg.n_solver])) > 1
        )

        # Dummy rows keep the batch rectangular but must not move the policy.
        for q, item in enumerate(batch):
            if item.is_dummy:
                for j in range(cfg.n_solver):
                    advantages[q * cfg.n_solver + j] = 0.0

        # 7 proposer rewards from per-question solver accuracy
        prop_rewards: list[float] = []
        prop_indices: list[int] = []
        for q, item in enumerate(batch):
            if item.proposer_idx is None:  # replayed or dummy: not this step's proposer
                continue
            prop_indices.append(item.proposer_idx)
            prop_rewards.append(proposer_reward(cfg.proposer_reward, per_question_acc[q]))
        prop_advantages = baseline_advantages(prop_rewards)

        # 8 loss over all solver + proposer trajectories
        t0 = time.time()
        samples: list[tuple[list[int], list[int], float]] = []
        for traj, adv in zip(solver_trajs, advantages):
            if adv == 0.0:
                continue
            ids, mask = build_loss_mask(self.tok, traj)
            samples.append((ids, mask, adv))
        for idx, adv in zip(prop_indices, prop_advantages):
            if adv == 0.0:
                continue
            ids, mask = build_loss_mask(self.tok, proposer_trajs[idx])
            samples.append((ids, mask, adv))

        if samples and self.optimizer is not None:
            update = compute_update(self.model, self.tok, self.optimizer,
                                    self.scaler, samples, cfg)
        else:
            update = {"loss": 0.0, "kl": 0.0, "grad_norm": 0.0, "skipped": True}
        t_update = time.time() - t0

        # 9 replay push/clear, checkpoint, metrics
        self.replay.extend(fresh)
        self.replay.maybe_clear(step_idx, cfg.replay_reset_every)

        if not update["skipped"]:
            self.step_idx = step_idx

        t0 = time.time()
        self._write_samples(step_idx, batch, solver_trajs)
        self._checkpoint(step_idx)
        t_ckpt = time.time() - t0

        all_trajs = list(proposer_trajs) + list(solver_trajs)
        row = {
            "step": step_idx,
            "session_id": self.session_id,
            "t_propose": round(t_propose, 3),
            "t_verify": round(t_verify, 3),
            "t_solve": round(t_solve, 3),
            "t_judge": round(t_judge, 3),
            "t_update": round(t_update, 3),
            "t_ckpt": round(t_ckpt, 3),
            "rule_pass_rate": rule_pass / max(1, len(proposer_trajs)),
            "rag_verified_rate": len(fresh) / max(1, len(proposer_trajs)),
            "fresh_verified": fill["fresh"],
            "replay_filled": fill["replay"],
            "dummy_filled": fill["dummy"],
            "solver_acc": sum(rewards) / max(1, len(rewards)),
            "proposer_mean_reward": (sum(prop_rewards) / len(prop_rewards)
                                     if prop_rewards else 0.0),
            "frac_groups_nonzero_var": nonzero_var / max(1, len(batch)),
            "avg_turns_proposer": (sum(t.turns_used for t in proposer_trajs)
                                   / max(1, len(proposer_trajs))),
            "avg_turns_solver": (sum(t.turns_used for t in solver_trajs)
                                 / max(1, len(solver_trajs))),
            "format_valid_rate": (sum(1 for t in all_trajs if t.format_valid)
                                  / max(1, len(all_trajs))),
            "loss": update["loss"],
            "kl": update["kl"],
            "grad_norm": update["grad_norm"],
            "skipped_update": bool(update["skipped"]),
            "judge_errors": stats["judge_errors"],
            "eval_acc": None,
        }
        return row

    # -- side files --------------------------------------------------------

    def _write_samples(self, step_idx: int, batch, solver_trajs) -> None:
        out_dir = self.run_dir / "samples"
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"step_{step_idx:05d}.jsonl"
        n = self.cfg.n_solver
        with open(path, "w", encoding="utf-8") as f:
            for q, item in enumerate(batch):
                traj = solver_trajs[q * n] if q * n < len(solver_trajs) else None
                f.write(json.dumps({
                    "question": item.question,
                    "answer": item.answer,
                    "is_dummy": item.is_dummy,
                    "solver_text": traj.text if traj else "",
                    "solver_format_valid": traj.format_valid if traj else False,
                }) + "\n")

    def _checkpoint(self, step_idx: int) -> None:
        state = {
            "step": step_idx,
            "session_id": self.session_id,
            "gpu_name": gpu_name(),
            "git_sha": git_sha(),
            "config_hash": config_hash(self.cfg),
            "index_sha": index_sha(self.index_dir) if self.index_dir else "unknown",
            "judge_name": self.cfg.judge_name,
        }
        save_checkpoint(self.run_dir, self.model, self.optimizer, self.scaler,
                        capture_rng_state(), self.replay, state)


# ---------------------------------------------------------------------------
# §3.8 resume loop / CLI
# ---------------------------------------------------------------------------

def run_dir_for(runs_root, run_id: str) -> Path:
    return Path(runs_root) / run_id


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m minissp.train")
    add_config_args(parser)
    parser.add_argument("--max-steps", dest="max_steps", type=int, default=50)
    parser.add_argument("--runs-root", dest="runs_root", default="runs")
    parser.add_argument("--index-dir", dest="index_dir", default=None,
                        help="dir holding corpus.faiss + docs.jsonl "
                             "(default: <drive_root>/index)")
    parser.add_argument("--answers", default=str(ANSWERS_TOY_PATH))
    parser.add_argument("--eval-file", dest="eval_file", default=str(EVAL_TOY_PATH))
    args = parser.parse_args(argv)
    cfg = config_from_namespace(args)

    from minissp import eval as eval_mod
    from minissp.judge import Judge
    from minissp.retrieval import Retriever

    device = "cuda" if torch.cuda.is_available() else "cpu"
    index_dir = args.index_dir or str(Path(cfg.drive_root) / "index")
    run_dir = run_dir_for(args.runs_root, cfg.run_id)
    run_dir.mkdir(parents=True, exist_ok=True)

    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    model, tok = build_policy(cfg, device=device)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=cfg.lr)
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))

    retriever = Retriever(index_dir, cfg.embed_name).load()
    judge = Judge(cfg.judge_name, device=device).load()

    answers = _read_jsonl(args.answers)
    eval_rows = _read_jsonl(args.eval_file)

    trainer = Trainer(cfg, run_dir, model, tok, retriever, judge, answers,
                      optimizer=optimizer, scaler=scaler, index_dir=index_dir)

    # The presence of `latest` decides init vs resume — there is no flag.
    latest = read_latest(run_dir)
    if latest is not None:
        ckpt = load_checkpoint(run_dir)
        if ckpt["adapter_dir"] is not None:
            model.load_adapter(str(ckpt["adapter_dir"]), adapter_name="default",
                               is_trainable=True)
        if ckpt["optimizer"] is not None:
            optimizer.load_state_dict(ckpt["optimizer"])
        if ckpt["scaler"] is not None:
            scaler.load_state_dict(ckpt["scaler"])
        if ckpt["rng_state"] is not None:
            restore_rng_state(ckpt["rng_state"])
        trainer.replay.load(ckpt["replay"])
        trainer.step_idx = ckpt["step"]
        dropped = trainer.metrics.truncate_after(ckpt["step"])
        print(f"[train] resumed at step {ckpt['step']} "
              f"(dropped {dropped} metrics rows past the checkpoint)")
    else:
        print(f"[train] fresh run {cfg.run_id} in {run_dir}")

    consecutive_skips = 0
    while trainer.step_idx < args.max_steps:
        before = trainer.step_idx
        row = trainer.step()
        if cfg.eval_every > 0 and row["step"] % cfg.eval_every == 0:
            result = eval_mod.run(model, tok, retriever, judge, cfg, eval_rows,
                                  step=row["step"], run_dir=run_dir)
            row["eval_acc"] = result["acc"]
        assert set(row) == set(METRICS_COLUMNS)
        trainer.metrics.write(row)
        print(f"[train] step {row['step']:>4}  loss={row['loss']:.4f} "
              f"solver_acc={row['solver_acc']:.3f} "
              f"rag_verified_rate={row['rag_verified_rate']:.3f}")
        if trainer.step_idx == before:
            # Non-finite loss/grad: the step is retried rather than advanced
            # (plan §3.4). Three in a row means something is structurally
            # broken (fp16 overflow, dead judge) — stop instead of spinning.
            consecutive_skips += 1
            print(f"[train] update skipped ({consecutive_skips}); step not advanced")
            if consecutive_skips >= 3:
                print("[train] three consecutive skipped updates; stopping")
                break
        else:
            consecutive_skips = 0


if __name__ == "__main__":
    main()
