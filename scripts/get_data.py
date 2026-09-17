"""Fetch Quark-LLM/SSP, build + freeze toy answer/eval subsets, write manifest.
"""
from minissp.data import (
    load_raw,
    build_answers_toy,
    build_eval_toy,
    write_jsonl,
    write_manifest,
    TOY_ANSWERS_PATH,
    TOY_EVAL_PATH,
)


def main():
    print("Loading raw dataset from Hugging Face...")
    raw = load_raw()
    print(f"  train: {len(raw['train'])} rows")
    print(f"  test:  {len(raw['test'])} rows")

    print("Building answers_toy...")
    answers = build_answers_toy(raw)
    print(f"  {len(answers)} rows")

    print("Building eval_toy (excluding any train answer overlap)...")
    train_answer_set = {a["answer"] for a in answers}
    eval_qs = build_eval_toy(raw, train_answers=train_answer_set)
    print(f"  {len(eval_qs)} rows")

    write_jsonl(answers, TOY_ANSWERS_PATH)
    write_jsonl(eval_qs, TOY_EVAL_PATH)
    manifest = write_manifest([TOY_ANSWERS_PATH, TOY_EVAL_PATH])

    print(f"\nWrote {TOY_ANSWERS_PATH}, {TOY_EVAL_PATH}, and manifest with {len(manifest)} entries.")


if __name__ == "__main__":
    main()