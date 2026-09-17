"""
minissp/eval.py

Phase 4 of plan.md (§4.1): greedy solver rollouts over the frozen `eval_toy`
set, judge-scored, with a 95% bootstrap CI over questions (1,000 resamples).

    python -m minissp.eval --run-id t4-001 --step 20

and, from inside the training loop:

    result = eval.run(model, tok, retriever, judge, cfg, eval_rows, step, run_dir)

Writes `runs/<run_id>/eval_step_<N>.json`.

With 300 questions the CI is roughly +-5 points; the plan is explicit that
only movement beyond that is interpretable, which is exactly why the interval
is computed and stored alongside the point estimate rather than left to be
eyeballed off a curve.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from minissp.config import add_config_args, config_from_namespace
from minissp.data import EVAL_TOY_PATH
from minissp.judge import JudgeParseError
from minissp.prompts import solver_prompt
from minissp.rollout import run_trajectories

BOOTSTRAP_RESAMPLES = 1000
EVAL_BATCH = 16


def bootstrap_ci(values: list[float], n_resamples: int = BOOTSTRAP_RESAMPLES,
                 alpha: float = 0.05, seed: int = 0) -> tuple[float, float]:
    """Percentile bootstrap CI of the mean, resampling questions with replacement."""
    if not values:
        return (0.0, 0.0)
    rng = random.Random(seed)
    n = len(values)
    means = []
    for _ in range(n_resamples):
        means.append(sum(values[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    lo = means[max(0, int((alpha / 2) * n_resamples) - 1)]
    hi = means[min(n_resamples - 1, int((1 - alpha / 2) * n_resamples))]
    return (lo, hi)


class _GreedyModel:
    """Transparent proxy that forces `do_sample=False` on every generate().

    `rollout.run_trajectories` is the single multi-turn loop and it samples
    (temperature 1.0) because that is what training needs. Evaluation must be
    greedy and deterministic (plan §4.1). Proxying here keeps one rollout
    implementation instead of two, and keeps rollout.py untouched.
    """

    def __init__(self, model):
        self._model = model

    def __getattr__(self, name):
        return getattr(self._model, name)

    def generate(self, *args, **kwargs):
        kwargs = dict(kwargs)
        kwargs["do_sample"] = False
        # Sampling knobs are meaningless under greedy decoding and make
        # transformers warn on every call.
        kwargs.pop("temperature", None)
        kwargs.pop("top_p", None)
        return self._model.generate(*args, **kwargs)


def run(model, tok, retriever, judge, cfg, eval_rows: list[dict],
        step: int = 0, run_dir=None, out_path=None) -> dict:
    """Greedy solver pass over `eval_rows`; returns the result dict it writes.

    Result: {"step", "n", "acc", "ci95": [lo, hi], "judge_errors",
             "format_valid_rate", "by_source": {...}}
    """
    import torch

    was_training = getattr(model, "training", False)
    if hasattr(model, "eval"):
        model.eval()
    greedy = _GreedyModel(model)

    correct: list[float] = []
    sources: list[str] = []
    judge_errors = 0
    format_valid = 0
    total = 0

    with torch.no_grad():
        for start in range(0, len(eval_rows), EVAL_BATCH):
            chunk = eval_rows[start:start + EVAL_BATCH]
            prompts = [solver_prompt(r["question"]) for r in chunk]
            trajs = run_trajectories(greedy, tok, retriever, prompts, "solver", cfg)
            total += len(trajs)
            format_valid += sum(1 for t in trajs if t.format_valid)

            live = [i for i, t in enumerate(trajs)
                    if t.format_valid and _answer_of(t).strip()]
            verdicts = {}
            if live:
                qs = [chunk[i]["question"] for i in live]
                ts = [list(chunk[i]["targets"]) for i in live]
                ps = [_answer_of(trajs[i]) for i in live]
                try:
                    results = judge.is_correct(qs, ts, ps)
                except JudgeParseError:
                    results = []
                    for q, t, p in zip(qs, ts, ps):
                        try:
                            results.append(bool(judge.is_correct(q, t, p)))
                        except JudgeParseError:
                            results.append(False)
                            judge_errors += 1
                verdicts = dict(zip(live, results))

            for i, row in enumerate(chunk):
                correct.append(1.0 if verdicts.get(i) else 0.0)
                sources.append(row.get("source", "unknown"))

    if was_training and hasattr(model, "train"):
        model.train()

    acc = sum(correct) / max(1, len(correct))
    lo, hi = bootstrap_ci(correct, seed=cfg.seed)

    by_source: dict[str, dict] = {}
    for source in sorted(set(sources)):
        vals = [c for c, s in zip(correct, sources) if s == source]
        by_source[source] = {"n": len(vals), "acc": sum(vals) / max(1, len(vals))}

    result = {
        "step": step,
        "n": len(correct),
        "acc": acc,
        "ci95": [lo, hi],
        "judge_errors": judge_errors,
        "format_valid_rate": format_valid / max(1, total),
        "by_source": by_source,
    }

    if out_path is None and run_dir is not None:
        out_path = Path(run_dir) / f"eval_step_{step}.json"
    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
    return result


def _answer_of(traj) -> str:
    answers = [t for t in traj.turns if t.kind == "answer"]
    return answers[-1].text if answers else ""


def _read_jsonl(path) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m minissp.eval")
    add_config_args(parser)
    parser.add_argument("--step", type=int, default=0,
                        help="checkpoint step to evaluate (0 = the base policy)")
    parser.add_argument("--runs-root", dest="runs_root", default="runs")
    parser.add_argument("--index-dir", dest="index_dir", default=None)
    parser.add_argument("--eval-file", dest="eval_file", default=str(EVAL_TOY_PATH))
    args = parser.parse_args(argv)
    cfg = config_from_namespace(args)

    import torch

    from minissp.judge import Judge
    from minissp.retrieval import Retriever
    from minissp.runtime import checkpoint_dir, verify_manifest
    from minissp.train import build_policy

    device = "cuda" if torch.cuda.is_available() else "cpu"
    index_dir = args.index_dir or str(Path(cfg.drive_root) / "index")
    run_dir = Path(args.runs_root) / cfg.run_id

    model, tok = build_policy(cfg, device=device)
    if args.step > 0:
        ckpt = checkpoint_dir(run_dir, args.step)
        verify_manifest(ckpt)
        model.load_adapter(str(ckpt / "adapter"), adapter_name="default")

    retriever = Retriever(index_dir, cfg.embed_name).load()
    judge = Judge(cfg.judge_name, device=device).load()

    result = run(model, tok, retriever, judge, cfg, _read_jsonl(args.eval_file),
                 step=args.step, run_dir=run_dir)
    print(f"[eval] step {result['step']}  acc={result['acc']:.3f} "
          f"95% CI [{result['ci95'][0]:.3f}, {result['ci95'][1]:.3f}]  "
          f"n={result['n']}  format_valid_rate={result['format_valid_rate']:.3f}")


if __name__ == "__main__":
    main()
