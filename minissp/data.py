
import hashlib
import json
import random
import re
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import hf_hub_download

TOY_ANSWERS_PATH = Path("data/answers_toy.jsonl")
TOY_EVAL_PATH = Path("data/eval_toy.jsonl")
MANIFEST_PATH = Path("data/manifest.json")


def load_raw():
    """Load SSP train (via datasets, uniform schema) and test (manual, to
    avoid Arrow schema-unification errors across files)."""
    train_ds = load_dataset(
        "Quark-LLM/SSP",
        data_files="train_answers_random_hop_50000_proposer.jsonl",
    )["train"]

    test_path = hf_hub_download(
        repo_id="Quark-LLM/SSP",
        filename="test.jsonl",
        repo_type="dataset",
    )
    with open(test_path) as f:
        test_rows = [json.loads(line) for line in f]

    return {"train": train_ds, "test": test_rows}


def _parse_examples(sys_question_example: str) -> list:
    """Split the newline-separated 'Question N: ...' block into a clean list."""
    lines = [ln.strip() for ln in sys_question_example.split("\n") if ln.strip()]
    cleaned = [re.sub(r"^Question\s*\d+:\s*", "", ln) for ln in lines]
    return cleaned


def build_answers_toy(raw, n_target: int = 800, seed: int = 0):
    """Stratified sample of train answers for proposer training.

    Stratifies on search_turns (n), since train has no topic/category field.
    Within each search_turns bucket, sampling is proportional to that
    bucket's share of the full train set, so the toy set's difficulty
    mix mirrors the full data.
    """
    train_ds = raw["train"]
    rng = random.Random(seed)

    buckets = {}
    for i, row in enumerate(train_ds):
        key = row["search_turns"]
        buckets.setdefault(key, []).append(i)

    total = len(train_ds)
    selected_indices = []
    for key, indices in buckets.items():
        share = len(indices) / total
        take = max(1, round(n_target * share))
        rng.shuffle(indices)
        selected_indices.extend(indices[:take])

    rng.shuffle(selected_indices)
    selected_indices = selected_indices[:n_target]

    out = []
    for i in selected_indices:
        row = train_ds[i]
        out.append({
            "question": None,  # train rows have no single "the" question,
                                # only the 3 example questions below
            "examples": _parse_examples(row["sys_question_example"]),
            "answer": row["ground_truth"],
            "n": int(row["search_turns"]),
            "source": "train",
        })
    return out


def build_eval_toy(raw, train_answers: set, n_nq: int = 150, n_hotpot: int = 150, seed: int = 0):
    """150 NQ + 150 HotpotQA questions from test, held out for eval.

    Excludes any row whose answer collides with a train_answers entry,
    guaranteeing no leakage regardless of sampling luck.
    """
    test_rows = raw["test"]
    rng = random.Random(seed)

    def is_clean(row):
        targets = row["reward_model"]["ground_truth"]["target"]
        return targets[0] not in train_answers

    nq_rows = [r for r in test_rows if "nq" in r["data_source"].lower() and is_clean(r)]
    hotpot_rows = [r for r in test_rows if "hotpot" in r["data_source"].lower() and is_clean(r)]

    if len(nq_rows) < n_nq:
        raise ValueError(f"Only {len(nq_rows)} clean NQ rows available, need {n_nq}")
    if len(hotpot_rows) < n_hotpot:
        raise ValueError(f"Only {len(hotpot_rows)} clean HotpotQA rows available, need {n_hotpot}")

    rng.shuffle(nq_rows)
    rng.shuffle(hotpot_rows)
    picked = nq_rows[:n_nq] + hotpot_rows[:n_hotpot]
    rng.shuffle(picked)

    out = []
    for row in picked:
        targets = row["reward_model"]["ground_truth"]["target"]
        out.append({
            "question": row["extra_info"]["question"],
            "answer": targets[0],
            "answers": targets,
            "n": None,
            "source": "nq" if "nq" in row["data_source"].lower() else "hotpotqa",
        })
    return out

def write_jsonl(rows, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_manifest(paths: list) -> dict:
    manifest = {str(p): sha256_of(p) for p in paths}
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))
    return manifest


def answer_set(jsonl_path: Path) -> set:
    with jsonl_path.open() as f:
        return {json.loads(line)["answer"] for line in f}