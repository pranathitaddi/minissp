"""
minissp/rollout.py

Phase 2 of plan.md (§2.4-2.6): the tag-protocol parser, the batched multi-turn
generation loop that replaces sglang's async engine, the proposer rule filter
and RAG verification.

    turns, ok = parse(trajectory_text)                      # §2.4
    trajs = run_trajectories(model, tok, retriever, prompts, "solver", cfg)   # §2.5
    ok, reason = rule_filter(traj, answer)                  # §2.6
    verified = rag_verify(question, answer, docs, pool, judge, rng=rng)       # §2.6

Smoke test (Colab cell 2):

    python -m minissp.rollout --smoke --index-dir /content/drive/MyDrive/minissp/index

Importing this module must have zero side effects: torch/transformers are
imported inside the functions that need them, exactly as in data.py,
retrieval.py and judge.py.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from dataclasses import dataclass, field
from pathlib import Path

from minissp.data import normalize
from minissp.prompts import TAGS, TERMINAL_TAG_BY_ROLE, proposer_prompt, solver_prompt
from minissp.retrieval import format_information

TERMINAL_TAGS = ("answer", "question")

# Matches an opening or closing tag for any protocol tag, and nothing else.
_TAG_RE = re.compile(r"</?(" + "|".join(TAGS) + r")>")

STOP_STRINGS = ["</search>", "</answer>", "</question>"]

# rule_filter thresholds [INFERRED — plan §2.6 flags these as not confirmed
# against the reference repo].
MIN_QUESTION_WORDS = 10


# ---------------------------------------------------------------------------
# §2.4 parser
# ---------------------------------------------------------------------------

@dataclass
class Turn:
    kind: str
    text: str
    start: int  # char offset of "<kind>" in the trajectory string
    end: int    # char offset just past "</kind>"


@dataclass
class Traj:
    text: str                                   # the completion (no prompt)
    info_spans: list[tuple[int, int]] = field(default_factory=list)
    docs_seen: list[dict] = field(default_factory=list)
    turns_used: int = 0                         # number of <search> turns taken
    turns: list[Turn] = field(default_factory=list)
    format_valid: bool = False
    # The rendered prompt this completion continues. Kept separate from `text`
    # so parse() only ever sees model-protocol output; P3's loss mask needs
    # both halves.
    prompt: str = ""


def parse(text: str, terminal_tag: str | None = None) -> tuple[list[Turn], bool]:
    """
    Strict protocol parser. Returns (turns, format_valid).

    format_valid requires ALL of:
      - every opened tag is closed, and closed by its own closing tag
      - no nesting (a tag may not open while another is open)
      - no stray closing tag
      - exactly one terminal tag (<answer> or <question>); if `terminal_tag`
        is given, it must be that one and the other must not appear
      - the terminal tag is the last turn (nothing but whitespace follows it)
      - no non-whitespace text outside of tags

    Turns are returned for whatever was parsed before the first violation, so
    callers can still inspect a malformed trajectory.

    Decision on turn ORDER: parse() is deliberately context-free. It does not
    reject `<information>` appearing before any `<search>`, nor an empty
    `<think>`, nor any other ordering. Order is a semantic property, and the
    only <information> blocks in a real trajectory are the ones this module
    inserts itself; a model that emits its own <information> is handled by the
    stop strings (generation halts at </search>) rather than by the parser.
    Keeping parse() shape-only makes it cheap to reason about and to test.
    """
    turns: list[Turn] = []
    valid = True
    pos = 0
    open_kind: str | None = None
    open_start = 0
    open_body_start = 0

    for match in _TAG_RE.finditer(text):
        kind = match.group(1)
        is_close = match.group(0).startswith("</")
        between = text[pos:match.start()]

        if open_kind is None:
            if between.strip():
                valid = False  # text outside of tags
            if is_close:
                valid = False  # stray closing tag
                break
            open_kind = kind
            open_start = match.start()
            open_body_start = match.end()
        else:
            if not is_close:
                valid = False  # nesting
                break
            if kind != open_kind:
                valid = False  # mismatched closing tag
                break
            turns.append(Turn(kind=open_kind,
                              text=text[open_body_start:match.start()].strip(),
                              start=open_start,
                              end=match.end()))
            open_kind = None
        pos = match.end()

    if open_kind is not None:
        valid = False  # unclosed tag at end of trajectory
    elif text[pos:].strip():
        valid = False  # trailing text outside of tags

    terminals = [t for t in turns if t.kind in TERMINAL_TAGS]
    if len(terminals) != 1:
        valid = False
    elif terminals[0] is not turns[-1]:
        valid = False  # something follows the terminal tag
    elif terminal_tag is not None and terminals[0].kind != terminal_tag:
        valid = False

    return turns, valid


def terminal_tag_for_role(role: str) -> str:
    try:
        return TERMINAL_TAG_BY_ROLE[role]
    except KeyError:
        raise ValueError(
            f"unknown role {role!r}; expected one of {sorted(TERMINAL_TAG_BY_ROLE)}"
        ) from None


# ---------------------------------------------------------------------------
# §2.5 the generation loop
# ---------------------------------------------------------------------------

def _max_turns(role: str, cfg) -> int:
    if role == "proposer":
        return cfg.max_turns_proposer
    return cfg.max_turns_solver


def _last_tag_body(text: str, tag: str) -> str:
    matches = re.findall(rf"<{tag}>(.*?)</{tag}>", text, flags=re.DOTALL)
    return matches[-1].strip() if matches else ""


def run_trajectories(model, tok, retriever, prompts: list[str], role: str, cfg) -> list[Traj]:
    """
    Batched multi-turn rollout (plan §2.5). Deliberately dumb: every turn
    re-prefills the whole sequence rather than keeping a KV cache alive, which
    is what lets a single HF `generate` call stand in for sglang's async
    multi-turn engine.

    `prompts` are raw prompt strings (from prompts.py); they are wrapped in the
    model's chat template here. `role` is "proposer" or "solver".
    """
    import torch

    terminal = terminal_tag_for_role(role)
    max_turns = _max_turns(role, cfg)

    rendered = [
        tok.apply_chat_template([{"role": "user", "content": p}],
                                tokenize=False, add_generation_prompt=True)
        for p in prompts
    ]
    n = len(prompts)
    completions = [""] * n
    info_spans: list[list[tuple[int, int]]] = [[] for _ in range(n)]
    docs_seen: list[list[dict]] = [[] for _ in range(n)]
    searches_used = [0] * n
    done = [False] * n

    prev_padding_side = getattr(tok, "padding_side", "right")
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    try:
        # max_turns search turns plus one final turn to emit the terminal tag.
        for _ in range(max_turns + 1):
            active = [i for i in range(n) if not done[i]]
            if not active:
                break

            batch = tok([rendered[i] + completions[i] for i in active],
                        return_tensors="pt", padding=True,
                        add_special_tokens=False).to(model.device)
            with torch.no_grad():
                out = model.generate(
                    **batch,
                    max_new_tokens=cfg.max_new_tokens_per_turn,
                    stop_strings=STOP_STRINGS,
                    tokenizer=tok,
                    do_sample=True,
                    temperature=1.0,
                    top_p=1.0,
                    pad_token_id=tok.pad_token_id,
                )
            prompt_len = batch["input_ids"].shape[1]
            new_texts = tok.batch_decode(out[:, prompt_len:], skip_special_tokens=True)

            pending: list[tuple[int, str]] = []
            for i, new_text in zip(active, new_texts):
                completions[i] += new_text
                tail = completions[i].rstrip()
                if tail.endswith("</search>") and searches_used[i] < max_turns:
                    query = _last_tag_body(completions[i], "search")
                    searches_used[i] += 1
                    pending.append((i, query))
                else:
                    # terminal tag, an over-budget search, or the token cap
                    done[i] = True

            if pending:
                results = retriever.search([q for _, q in pending], topk=cfg.topk)
                for (i, _), docs in zip(pending, results):
                    block = ("\n<information>\n" + format_information(docs)
                             + "\n</information>\n")
                    start = len(completions[i]) + 1  # skip the leading newline
                    completions[i] += block
                    info_spans[i].append((start, len(completions[i]) - 1))
                    docs_seen[i].extend(docs)
                    n_tokens = len(tok(rendered[i] + completions[i],
                                       add_special_tokens=False)["input_ids"])
                    if n_tokens >= cfg.max_total_tokens:
                        done[i] = True
    finally:
        tok.padding_side = prev_padding_side

    trajs: list[Traj] = []
    for i in range(n):
        turns, ok = parse(completions[i], terminal_tag=terminal)
        trajs.append(Traj(
            text=completions[i],
            info_spans=info_spans[i],
            docs_seen=docs_seen[i],
            turns_used=searches_used[i],
            turns=turns,
            format_valid=ok,
            prompt=rendered[i],
        ))
    return trajs


# ---------------------------------------------------------------------------
# §2.6 rule filter
# ---------------------------------------------------------------------------

def rule_filter(traj: Traj, answer: str) -> tuple[bool, str]:
    """
    Proposer-side rule filter. Returns (ok, reason); reason is "ok" when the
    trajectory passes, otherwise one of:

        "format_invalid"      — parse() rejected the trajectory
        "no_question"         — no <question> turn (defensive; implies invalid)
        "no_search"           — the proposer never searched
        "question_too_short"  — fewer than MIN_QUESTION_WORDS words
        "answer_in_question"  — the answer is given away in the question

    Rejection means the trajectory is dropped, never punished: rewards stay at
    0 and never go negative (plan §2.4, enforced in train.py).
    """
    if not traj.format_valid:
        return False, "format_invalid"

    questions = [t for t in traj.turns if t.kind == "question"]
    if not questions:
        return False, "no_question"
    question = questions[-1].text

    if not any(t.kind == "search" for t in traj.turns):
        return False, "no_search"

    if len(question.split()) < MIN_QUESTION_WORDS:
        return False, "question_too_short"

    norm_answer = normalize(answer)
    if norm_answer and norm_answer in normalize(question):
        return False, "answer_in_question"

    return True, "ok"


# ---------------------------------------------------------------------------
# §2.6 RAG verification
# ---------------------------------------------------------------------------

def _doc_key(doc: dict) -> str:
    doc_id = doc.get("id")
    if doc_id is not None:
        return f"id:{doc_id}"
    return f"tt:{doc.get('title', '')}|{doc.get('text', '')}"


def build_rag_materials(proposer_docs: list[dict], other_docs_pool: list[dict],
                        k_noise: int, rng: random.Random) -> list[dict]:
    """
    proposer_docs (deduped) + up to k_noise distinct noise docs sampled without
    replacement from other_docs_pool and deduped against proposer_docs, then
    shuffled. Split out of rag_verify so the "exactly len(proposer_docs)+k_noise
    unique docs" property is directly testable without a model.
    """
    materials: list[dict] = []
    seen: set[str] = set()
    for doc in proposer_docs:
        key = _doc_key(doc)
        if key not in seen:
            seen.add(key)
            materials.append(doc)

    candidates: list[dict] = []
    candidate_keys: set[str] = set()
    for doc in other_docs_pool:
        key = _doc_key(doc)
        if key in seen or key in candidate_keys:
            continue
        candidate_keys.add(key)
        candidates.append(doc)

    k = min(k_noise, len(candidates))
    materials.extend(rng.sample(candidates, k))
    rng.shuffle(materials)
    return materials


def rag_verify(question: str, answer: str, proposer_docs: list[dict],
               other_docs_pool: list[dict], judge_or_policy, k_noise: int = 4,
               rng: random.Random | None = None) -> bool:
    """
    A question is verified when a solver, given only the proposer's own
    retrieved documents plus k_noise distractors, reproduces the intended
    answer (plan §2.6).

    `judge_or_policy` is duck-typed: anything exposing
        .rag_solve(question: str, docs: list[dict]) -> str
        .is_correct(question: str, targets: list[str], prediction: str) -> bool
    works — Judge by default, the policy when cfg.rag_solver == "policy".

    `rng` is required (keyword, but not optional in practice): noise sampling
    must be reproducible from the run seed.
    """
    if rng is None:
        raise ValueError("rag_verify requires an explicit random.Random for determinism")

    materials = build_rag_materials(proposer_docs, other_docs_pool, k_noise, rng)
    prediction = judge_or_policy.rag_solve(question, materials)
    if not prediction or not prediction.strip():
        return False
    return bool(judge_or_policy.is_correct(question, [answer], prediction))


# ---------------------------------------------------------------------------
# CLI: --smoke  (plan §2 "Proof (cell 2)")
# ---------------------------------------------------------------------------

def _print_vram(label: str) -> None:
    try:
        import torch
        if torch.cuda.is_available():
            gb = torch.cuda.memory_allocated() / 1e9
            print(f"[smoke] VRAM after {label}: {gb:.2f} GB")
            return
    except Exception:  # noqa: BLE001 — VRAM reporting must never break the smoke run
        pass
    print(f"[smoke] VRAM after {label}: n/a (no CUDA)")


def _read_jsonl(path: Path, limit: int | None = None) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
            if limit is not None and len(rows) >= limit:
                break
    return rows


def _smoke(args: argparse.Namespace) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from minissp.config import Config
    from minissp.judge import Judge
    from minissp.retrieval import Retriever

    cfg = Config(run_id=args.run_id)
    rng = random.Random(cfg.seed)
    data_dir = Path(args.data_dir)
    index_dir = args.index_dir or str(Path(cfg.drive_root) / "index")

    tok = AutoTokenizer.from_pretrained(cfg.policy_name, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(
        cfg.policy_name, torch_dtype=torch.float16,
        device_map="cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    _print_vram("policy")

    judge = Judge(cfg.judge_name, device="cuda" if torch.cuda.is_available() else "cpu").load()
    _print_vram("judge")

    retriever = Retriever(index_dir, cfg.embed_name).load()
    _print_vram("retriever")

    n = args.n
    eval_rows = _read_jsonl(data_dir / "eval_toy.jsonl", limit=n)
    answer_rows = _read_jsonl(data_dir / "answers_toy.jsonl", limit=n)

    solver_trajs = run_trajectories(
        model, tok, retriever,
        [solver_prompt(r["question"]) for r in eval_rows], "solver", cfg)
    proposer_trajs = run_trajectories(
        model, tok, retriever,
        [proposer_prompt(r["answer"], r["n"], r["examples"]) for r in answer_rows],
        "proposer", cfg)

    all_trajs = solver_trajs + proposer_trajs
    format_valid_rate = sum(t.format_valid for t in all_trajs) / max(1, len(all_trajs))
    avg_search_turns = sum(t.turns_used for t in all_trajs) / max(1, len(all_trajs))

    pool = [d for t in proposer_trajs for d in t.docs_seen]
    rule_pass = 0
    rag_verified = 0
    reasons: dict[str, int] = {}
    for traj, row in zip(proposer_trajs, answer_rows):
        ok, reason = rule_filter(traj, row["answer"])
        reasons[reason] = reasons.get(reason, 0) + 1
        if not ok:
            continue
        rule_pass += 1
        question = [t for t in traj.turns if t.kind == "question"][-1].text
        if rag_verify(question, row["answer"], traj.docs_seen, pool, judge,
                      k_noise=cfg.noise_docs, rng=rng):
            rag_verified += 1

    denom = max(1, len(proposer_trajs))
    print(f"[smoke] format_valid_rate  {format_valid_rate:.3f}")
    print(f"[smoke] avg_search_turns   {avg_search_turns:.2f}")
    print(f"[smoke] rule_pass_rate     {rule_pass / denom:.3f}  reasons={reasons}")
    print(f"[smoke] rag_verified_rate  {rag_verified / denom:.3f}")

    out_dir = Path(args.out) / "samples"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "smoke.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for role, trajs in (("solver", solver_trajs), ("proposer", proposer_trajs)):
            for traj in trajs:
                f.write(json.dumps({
                    "role": role,
                    "prompt": traj.prompt,
                    "text": traj.text,
                    "turns_used": traj.turns_used,
                    "format_valid": traj.format_valid,
                    "info_spans": traj.info_spans,
                }) + "\n")
    print(f"[smoke] wrote {len(all_trajs)} trajectories -> {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m minissp.rollout")
    parser.add_argument("--smoke", action="store_true",
                        help="load policy+judge+retriever and run 8+8 trajectories")
    parser.add_argument("--n", type=int, default=8, help="trajectories per role")
    parser.add_argument("--run-id", dest="run_id", default="smoke")
    parser.add_argument("--index-dir", dest="index_dir", default=None,
                        help="dir holding corpus.faiss + docs.jsonl "
                             "(default: <drive_root>/index)")
    parser.add_argument("--data-dir", dest="data_dir",
                        default=str(Path(__file__).resolve().parent.parent / "data"))
    parser.add_argument("--out", default="runs/smoke")
    args = parser.parse_args()

    if not args.smoke:
        parser.error("nothing to do; pass --smoke")
    _smoke(args)


if __name__ == "__main__":
    main()
