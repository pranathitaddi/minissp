"""
minissp/data.py

Phase 1 of plan.md: raw jsonl readers, toy subset freeze, manifest,
corpus scoping, index build.

CLI:
    python -m minissp.data download
    python -m minissp.data scope --wiki ~/wiki-18.jsonl.gz
    python -m minissp.data freeze
    python -m minissp.data build-index --out ~/minissp_index/

This module must be cheap to import (no downloads, no model loads at
import time) — heavy deps (datasets/huggingface_hub, faiss,
sentence-transformers) are imported lazily inside the functions that
need them, mirroring retrieval.py's lazy-load discipline.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import random
import re
import string
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Iterator

# ---------------------------------------------------------------------------
# Paths (repo-relative; override via CLI flags where it matters)
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

TRAIN_PATH = DATA_DIR / "train.jsonl"
TEST_PATH = DATA_DIR / "test.jsonl"
CORPUS_SCOPED_PATH = DATA_DIR / "corpus_scoped.jsonl"
CORPUS_IDS_PATH = DATA_DIR / "corpus_ids.json"
ANSWERS_TOY_PATH = DATA_DIR / "answers_toy.jsonl"
EVAL_TOY_PATH = DATA_DIR / "eval_toy.jsonl"
MANIFEST_PATH = DATA_DIR / "manifest.json"

HF_TRAIN_REPO = "Quark-LLM/SSP"
HF_TRAIN_FILENAME = "train_answers_random_hop_50000_proposer.jsonl"
# Confirm the exact test-file name on the dataset page before relying on
# this default — the plan flags this dataset as [HF]-confirmed only for
# the schema, not necessarily this literal filename.
HF_TEST_FILENAME = "test.jsonl"

HF_WIKI_REPO = "PeterJinGo/wiki-18-corpus"
HF_WIKI_FILENAME = "wiki-18.jsonl.gz"  # [INFERRED] confirm on the dataset page

# Default local destination for the wiki-18 download (~13 GB). Kept outside
# the repo (per plan 1.2: "Keep it outside the repo"), in the home directory
# by default so `scope` can find it without an explicit --wiki flag.
WIKI_PATH = Path.home() / "wiki-18.jsonl.gz"

_ARTICLES = {"a", "an", "the"}
_PUNCT_STRIP = str.maketrans("", "", string.punctuation)


# ---------------------------------------------------------------------------
# 1.1 normalize / readers
# ---------------------------------------------------------------------------

def normalize(s: str) -> str:
    """Lower, strip, drop leading articles + trailing punctuation, collapse whitespace."""
    if s is None:
        return ""
    s = s.strip().lower()
    s = s.rstrip(string.punctuation + " ")
    s = re.sub(r"\s+", " ", s).strip()
    words = s.split(" ")
    if words and words[0] in _ARTICLES:
        words = words[1:]
    return " ".join(words).strip()


def read_train(path: str | Path) -> list[dict]:
    """
    Reads the Quark-LLM/SSP proposer training file.

    Row schema on disk:
        {"sys_question_example": "...", "ground_truth": "...", "search_turns": "2"}

    Returns:
        [{"answer": str, "n": int, "examples": str}, ...]

    Drops rows where normalize(ground_truth) is empty or longer than 60 chars
    (junk rows per plan 1.1), and rows whose search_turns doesn't cast to
    {1, 2, 3}.
    """
    out = []
    dropped = 0
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                dropped += 1
                continue

            answer_raw = row.get("ground_truth", "")
            norm = normalize(answer_raw)
            if not norm or len(norm) > 60:
                dropped += 1
                continue

            try:
                n = int(row["search_turns"])
            except (KeyError, ValueError, TypeError):
                dropped += 1
                continue
            if n not in (1, 2, 3):
                dropped += 1
                continue

            examples = row.get("sys_question_example", "")
            out.append({"answer": answer_raw, "n": n, "examples": examples})

    if dropped:
        print(f"[read_train] dropped {dropped} junk rows out of "
              f"{dropped + len(out)} total", file=sys.stderr)
    return out


def read_test(path: str | Path) -> list[dict]:
    """
    Reads the veRL-format SSP test file.

    Row schema on disk (fields that matter):
        {"data_source": "nq", "prompt": [...],
         "reward_model": {"ground_truth": {"target": ["..."]}, "style": "rule"},
         "extra_info": {"question": "...", "index": 0, "split": "test"}}

    Returns:
        [{"question": str, "targets": list[str], "source": str}, ...]

    Uses extra_info.question, NOT prompt[0].content (that is the repo's
    rendered solver prompt with its own system text baked in; we render
    our own in prompts.py).
    """
    out = []
    skipped = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            try:
                question = row["extra_info"]["question"]
                targets = row["reward_model"]["ground_truth"]["target"]
                source = row["data_source"]
            except KeyError:
                skipped += 1
                continue
            if not question or not targets:
                skipped += 1
                continue
            out.append({"question": question, "targets": list(targets), "source": source})

    if skipped:
        print(f"[read_test] skipped {skipped} malformed rows", file=sys.stderr)
    return out


# ---------------------------------------------------------------------------
# 1.2 wiki-18 streaming
# ---------------------------------------------------------------------------

def _open_maybe_gz(path: str | Path):
    path = str(path)
    # errors="replace" swaps any byte that can't be decoded as UTF-8 (e.g. a
    # source file that's actually Latin-1/cp1252, or a handful of corrupted
    # bytes) for U+FFFD rather than crashing the whole stream partway
    # through a multi-hour pass over 21M wiki-18 passages. A few mangled
    # characters in rare passages is an acceptable trade for not losing the
    # entire run.
    if path.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return open(path, "r", encoding="utf-8", errors="replace")


def iter_wiki(path: str | Path) -> Iterator[dict]:
    """
    Streams wiki-18 as {"id": str, "title": str, "text": str} dicts.
    Never loads the file into memory.

    wiki-18 dumps vary in field naming across mirrors; this accepts the
    two most common shapes:
      - {"id", "title", "text"}
      - {"id", "contents"} where contents is "Title\ntext..." (Search-R1 style)
    Confirm which shape your downloaded file actually uses before a full
    scope_corpus run — the first few lines are cheap to eyeball.
    """
    with _open_maybe_gz(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if "title" in row and "text" in row:
                yield {"id": row.get("id"), "title": row["title"], "text": row["text"]}
            elif "contents" in row:
                contents = row["contents"]
                title, _, text = contents.partition("\n")
                yield {"id": row.get("id"), "title": title, "text": text}
            else:
                # Unrecognized shape — skip rather than guess wrong.
                continue


# ---------------------------------------------------------------------------
# 1.3 corpus scoping
# ---------------------------------------------------------------------------

def _whole_word_match(entity_lower: str, text_lower: str) -> bool:
    return re.search(r"(?<!\w)" + re.escape(entity_lower) + r"(?!\w)", text_lower) is not None


def scope_corpus(
    wiki_stream: Iterable[dict],
    answers: set[str],
    eval_targets: set[str],
    per_entity_cap: int = 20,
    n_distractors: int = 20_000,
    seed: int = 0,
) -> tuple[list[dict], dict[str, int]]:
    """
    Single pass over wiki_stream. For each entity in answers | eval_targets:
      - title match: normalize(title) == normalize(entity) -> keep (up to per_entity_cap)
      - text match: entity is a whole-word, case-insensitive match in text,
        AND len(entity) >= 4 chars, AND entity is not purely numeric -> keep
        (up to per_entity_cap)
    Also reservoir-samples n_distractors passages that matched no entity.

    Returns:
        (kept_passages, coverage) where kept_passages is a list of
        {"id","title","text"} dicts (deduplicated) and coverage maps
        each raw entity string -> number of passages kept for it.
    """
    rng = random.Random(seed)

    entities = sorted(answers | eval_targets)
    norm_to_entity: dict[str, str] = {}
    text_candidates: list[tuple[str, str]] = []  # (normalized_lower, raw_entity)
    for e in entities:
        norm_to_entity[normalize(e)] = e
        e_lower = e.strip().lower()
        if len(e_lower) >= 4 and not e_lower.isdigit():
            text_candidates.append((e_lower, e))

    coverage: dict[str, int] = defaultdict(int)
    kept: dict[str, dict] = {}  # id -> passage, dedup

    distractor_reservoir: list[dict] = []
    seen_distractors = 0

    for passage in wiki_stream:
        pid = passage.get("id")
        title = passage.get("title", "")
        text = passage.get("text", "")
        title_norm = normalize(title)
        text_lower = text.lower()

        matched_any = False

        # Title match
        entity = norm_to_entity.get(title_norm)
        if entity is not None and coverage[entity] < per_entity_cap:
            key = pid if pid is not None else f"title:{title_norm}:{len(kept)}"
            if key not in kept:
                kept[key] = passage
                coverage[entity] += 1
            matched_any = True

        # Text match (whole word, case-insensitive)
        for e_lower, raw_entity in text_candidates:
            if coverage[raw_entity] >= per_entity_cap:
                continue
            if _whole_word_match(e_lower, text_lower):
                key = pid if pid is not None else f"text:{raw_entity}:{len(kept)}"
                if key not in kept:
                    kept[key] = passage
                    coverage[raw_entity] += 1
                matched_any = True

        if not matched_any:
            seen_distractors += 1
            if len(distractor_reservoir) < n_distractors:
                distractor_reservoir.append(passage)
            else:
                j = rng.randint(0, seen_distractors - 1)
                if j < n_distractors:
                    distractor_reservoir[j] = passage

    all_kept = list(kept.values()) + distractor_reservoir
    # coverage should report 0 for entities that never matched, not be absent
    full_coverage = {e: coverage.get(e, 0) for e in entities}
    return all_kept, full_coverage


# ---------------------------------------------------------------------------
# 1.4 index build
# ---------------------------------------------------------------------------

def build_index(corpus_path: str | Path, out_dir: str | Path,
                 embed_name: str = "intfloat/e5-base-v2", force: bool = False) -> None:
    """
    Embeds every passage in corpus_path with e5 ("passage: {title}\\n{text}",
    normalize_embeddings=True) and writes an IndexFlatIP FAISS index plus a
    parallel docs.jsonl (same order as the index) to out_dir.

    Idempotent: skips if both output files already exist, unless force=True.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    index_path = out_dir / "corpus.faiss"
    docs_path = out_dir / "docs.jsonl"

    if index_path.exists() and docs_path.exists() and not force:
        print(f"[build_index] {index_path} and {docs_path} already exist; "
              f"skipping (--force to rebuild)", file=sys.stderr)
        return

    import faiss
    from sentence_transformers import SentenceTransformer

    passages = []
    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                passages.append(json.loads(line))

    if not passages:
        raise ValueError(f"No passages found in {corpus_path}")

    model = SentenceTransformer(embed_name)
    texts = [f"passage: {p.get('title', '')}\n{p.get('text', '')}" for p in passages]
    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
    )

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    faiss.write_index(index, str(index_path))

    with open(docs_path, "w", encoding="utf-8") as f:
        for p in passages:
            f.write(json.dumps({"id": p.get("id"), "title": p.get("title", ""),
                                 "text": p.get("text", "")}) + "\n")

    print(f"[build_index] wrote {len(passages)} passages to {index_path} / {docs_path}")


# ---------------------------------------------------------------------------
# CLI: download
# ---------------------------------------------------------------------------

def _download(args: argparse.Namespace) -> None:
    from huggingface_hub import hf_hub_download

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    train_src = hf_hub_download(repo_id=HF_TRAIN_REPO, filename=HF_TRAIN_FILENAME,
                                 repo_type="dataset")
    test_src = hf_hub_download(repo_id=HF_TRAIN_REPO, filename=HF_TEST_FILENAME,
                                repo_type="dataset")

    import shutil
    shutil.copy(train_src, TRAIN_PATH)
    shutil.copy(test_src, TEST_PATH)
    print(f"[download] wrote {TRAIN_PATH} and {TEST_PATH}")

    if args.skip_wiki:
        print("[download] --skip-wiki set; not downloading wiki-18 corpus", file=sys.stderr)
        return

    wiki_dest = Path(args.wiki_out)
    if wiki_dest.exists() and not args.force:
        print(f"[download] {wiki_dest} already exists; skipping wiki-18 download "
              f"(--force to re-download)", file=sys.stderr)
        return

    print("[download] fetching wiki-18 corpus (~13 GB, this will take a while)...",
          file=sys.stderr)
    wiki_src = hf_hub_download(repo_id=HF_WIKI_REPO, filename=HF_WIKI_FILENAME,
                                repo_type="dataset")
    wiki_dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(wiki_src, wiki_dest)
    print(f"[download] wrote {wiki_dest}")


# ---------------------------------------------------------------------------
# CLI: scope
# ---------------------------------------------------------------------------

def _scope(args: argparse.Namespace) -> None:
    train_rows = read_train(TRAIN_PATH)
    test_rows = read_test(TEST_PATH)

    answers = {row["answer"] for row in train_rows}
    eval_targets = {t for row in test_rows for t in row["targets"]}

    print(f"[scope] {len(answers)} unique train answers, "
          f"{len(eval_targets)} unique eval targets", file=sys.stderr)

    wiki_stream = iter_wiki(args.wiki)
    kept, coverage = scope_corpus(
        wiki_stream,
        answers=answers,
        eval_targets=eval_targets,
        per_entity_cap=args.per_entity_cap,
        n_distractors=args.n_distractors,
        seed=args.seed,
    )

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(CORPUS_SCOPED_PATH, "w", encoding="utf-8") as f:
        for p in kept:
            f.write(json.dumps(p) + "\n")

    ids = [p.get("id") for p in kept]
    with open(CORPUS_IDS_PATH, "w", encoding="utf-8") as f:
        json.dump({"ids": ids, "coverage": coverage}, f)

    zero_cov = sum(1 for c in coverage.values() if c == 0)
    print(f"[scope] kept {len(kept)} passages -> {CORPUS_SCOPED_PATH}")
    print(f"[scope] {zero_cov} / {len(coverage)} entities have zero coverage "
          f"(these can't be used for answers_toy / eval_toy)", file=sys.stderr)


# ---------------------------------------------------------------------------
# CLI: freeze
# ---------------------------------------------------------------------------

def _freeze(args: argparse.Namespace) -> None:
    with open(CORPUS_IDS_PATH, "r", encoding="utf-8") as f:
        corpus_meta = json.load(f)
    coverage = corpus_meta["coverage"]

    train_rows = read_train(TRAIN_PATH)
    test_rows = read_test(TEST_PATH)

    rng = random.Random(args.seed)

    # answers_toy: 800 answers with coverage >= 3, stratified by n in {1,2,3}
    eligible = [r for r in train_rows if coverage.get(r["answer"], 0) >= 3]
    by_n: dict[int, list[dict]] = defaultdict(list)
    for r in eligible:
        by_n[r["n"]].append(r)
    for bucket in by_n.values():
        rng.shuffle(bucket)

    target_total = 800
    per_bucket = target_total // 3
    answers_toy: list[dict] = []
    for n in (1, 2, 3):
        answers_toy.extend(by_n[n][:per_bucket])
    # top up if any bucket was short
    shortfall = target_total - len(answers_toy)
    if shortfall > 0:
        leftover = [r for n in (1, 2, 3) for r in by_n[n][per_bucket:]]
        rng.shuffle(leftover)
        answers_toy.extend(leftover[:shortfall])

    answers_toy_norms = {normalize(r["answer"]) for r in answers_toy}

    # eval_toy: 150 nq + 150 hotpotqa, any target with coverage >= 1,
    # and no overlap with answers_toy_norms
    by_source: dict[str, list[dict]] = defaultdict(list)
    for r in test_rows:
        if r["source"] in ("nq", "hotpotqa"):
            has_coverage = any(coverage.get(t, 0) >= 1 for t in r["targets"])
            no_overlap = all(normalize(t) not in answers_toy_norms for t in r["targets"])
            if has_coverage and no_overlap:
                by_source[r["source"]].append(r)

    for bucket in by_source.values():
        rng.shuffle(bucket)

    eval_toy = by_source["nq"][:150] + by_source["hotpotqa"][:150]

    if len(answers_toy) < target_total:
        print(f"[freeze] WARNING: only {len(answers_toy)}/800 eligible answers "
              f"(coverage >= 3) — widen corpus scoping (per_entity_cap / "
              f"n_distractors) or accept a smaller toy set.", file=sys.stderr)
    for src in ("nq", "hotpotqa"):
        if len(by_source[src]) < 150:
            print(f"[freeze] WARNING: only {len(by_source[src])}/150 eligible "
                  f"eval rows for source={src}", file=sys.stderr)

    with open(ANSWERS_TOY_PATH, "w", encoding="utf-8") as f:
        for r in answers_toy:
            f.write(json.dumps(r) + "\n")

    with open(EVAL_TOY_PATH, "w", encoding="utf-8") as f:
        for r in eval_toy:
            f.write(json.dumps(r) + "\n")

    manifest = {
        "seed": args.seed,
        "answers_toy_count": len(answers_toy),
        "eval_toy_count": len(eval_toy),
        "answers_toy_sha256": _sha256(ANSWERS_TOY_PATH),
        "eval_toy_sha256": _sha256(EVAL_TOY_PATH),
        "corpus_ids_sha256": _sha256(CORPUS_IDS_PATH),
    }
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"[freeze] wrote {len(answers_toy)} rows -> {ANSWERS_TOY_PATH}")
    print(f"[freeze] wrote {len(eval_toy)} rows -> {EVAL_TOY_PATH}")
    print(f"[freeze] wrote manifest -> {MANIFEST_PATH}")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# CLI: build-index
# ---------------------------------------------------------------------------

def _build_index_cmd(args: argparse.Namespace) -> None:
    build_index(CORPUS_SCOPED_PATH, args.out, embed_name=args.embed_name, force=args.force)


# ---------------------------------------------------------------------------
# argparse entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m minissp.data")
    sub = parser.add_subparsers(dest="command", required=True)

    p_download = sub.add_parser(
        "download",
        help="HF -> data/train.jsonl, data/test.jsonl, and (by default) the wiki-18 corpus",
    )
    p_download.add_argument("--skip-wiki", action="store_true",
                             help="skip the ~13 GB wiki-18 corpus download")
    p_download.add_argument("--wiki-out", dest="wiki_out", default=str(WIKI_PATH),
                             help=f"where to save wiki-18 (default: {WIKI_PATH})")
    p_download.add_argument("--force", action="store_true",
                             help="re-download wiki-18 even if wiki-out already exists")
    p_download.set_defaults(func=_download)

    p_scope = sub.add_parser("scope", help="wiki-18 -> data/corpus_scoped.jsonl, data/corpus_ids.json")
    p_scope.add_argument("--wiki", default=str(WIKI_PATH),
                          help=f"path to wiki-18.jsonl(.gz) (default: {WIKI_PATH})")
    p_scope.add_argument("--per-entity-cap", dest="per_entity_cap", type=int, default=20)
    p_scope.add_argument("--n-distractors", dest="n_distractors", type=int, default=20_000)
    p_scope.add_argument("--seed", type=int, default=0)
    p_scope.set_defaults(func=_scope)

    p_freeze = sub.add_parser("freeze", help="-> answers_toy, eval_toy, manifest.json")
    p_freeze.add_argument("--seed", type=int, default=0)
    p_freeze.set_defaults(func=_freeze)

    p_index = sub.add_parser("build-index", help="corpus_scoped.jsonl -> corpus.faiss, docs.jsonl")
    p_index.add_argument("--out", required=True, help="output directory")
    p_index.add_argument("--embed-name", dest="embed_name", default="intfloat/e5-base-v2")
    p_index.add_argument("--force", action="store_true")
    p_index.set_defaults(func=_build_index_cmd)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()