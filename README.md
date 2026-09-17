# minissp

A standalone reimplementation of **Search Self-Play** (SSP, arXiv:2510.18821) sized to
train end-to-end on a single **Colab free-tier T4 GPU**.

Everything runs in one Python process on the GPU: retrieval is a function call, and the
judge is a second model in the same process. There is no server, no tunnel, and no
background process to supervise. A Mac (or any local machine) is used only to download
data, build the corpus/index, and write code — training itself happens on Colab.

## How it works

A **proposer** model is given an answer and must write a question that a **solver**
model can answer by searching a scoped Wikipedia corpus. Both models are the same
policy (`Qwen/Qwen2.5-1.5B-Instruct`, LoRA fine-tuned), self-play-trained with GRPO. A
frozen **judge** model (`Qwen/Qwen2.5-3B-Instruct`, 4-bit) verifies solver answers and,
via RAG, verifies that proposed questions are actually answerable from retrieved
documents before they're used as training signal.

| Component | Choice |
|---|---|
| Policy | `Qwen/Qwen2.5-1.5B-Instruct`, fp16, LoRA r=16 α=32 |
| Judge | `Qwen/Qwen2.5-3B-Instruct`, 4-bit (nf4), frozen, in-process |
| Retriever | `intfloat/e5-base-v2` + FAISS `IndexFlatIP`, CPU, top-3 |
| Corpus | wiki-18, scoped to passages covering the frozen toy answer/eval set |
| Tier | T4 only — one config, overridable via CLI flags |

See the divergence table below for exactly what fidelity is traded away to fit a free
GPU tier.

## Package layout

```
minissp/
  config.py     Config dataclass + T4 defaults + --field value CLI overrides
  prompts.py    PROPOSER, SOLVER, RAG_SOLVER, JUDGE prompt templates + shared tag set
  data.py       raw jsonl readers, toy subset freeze, manifest, corpus scoping, index build
  retrieval.py  Retriever: lazy e5 + FAISS, search(), format_information()
  judge.py      Judge: lazy 4-bit Qwen2.5-3B, is_correct(), rag_solve()
  rollout.py    tag parser, batched multi-turn generation, rule filters, RAG verification
  train.py      rewards, GRPO advantages, loss, replay buffer, one training step, resume loop
  runtime.py    metrics.jsonl writer, checkpoint save/load with manifest verification
  eval.py       greedy eval on eval_toy, bootstrap CI
data/
  answers_toy.jsonl, eval_toy.jsonl, manifest.json, corpus_ids.json   # frozen, committed
colab/
  run.ipynb     3 cells: bootstrap, rollout smoke test, train
tests/
  test_core.py, test_pipeline.py, test_runtime.py, test_train.py
```

`retrieval.py` and `judge.py` are lazy by design — importing them never touches disk,
network, or GPU; only calling `.load()` does. This keeps the test suite fast and keeps
`data.py`'s CLI usable without pulling in ML dependencies at import time.

## Local setup

```bash
python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt -r requirements-dev.txt
pip install -e .
```

Verify:

```bash
pytest -q
```

All tests should pass without a GPU or network access — heavy-dependency tests
(retrieval/judge) either mock the model call or are skipped with an explicit reason if
`faiss`/`sentence-transformers` aren't installed.

## Building the data and corpus (local, one-time)

```bash
python -m minissp.data download                          # -> data/train.jsonl, data/test.jsonl
python -m minissp.data scope --wiki ~/wiki-18.jsonl.gz    # -> data/corpus_scoped.jsonl, data/corpus_ids.json
python -m minissp.data freeze                             # -> answers_toy.jsonl, eval_toy.jsonl, manifest.json
python -m minissp.data build-index --out ~/minissp_index/ # -> corpus.faiss, docs.jsonl
```

Upload the resulting `corpus.faiss` and `docs.jsonl` to `MyDrive/minissp/index/` by
hand before the first Colab session — see `colab/run.ipynb`.

## Training on Colab

1. Push your branch — Colab pulls the repo via git.
2. In `MyDrive/minissp/`, create `wheels/`, `index/`, `hf_cache/`, `runs/`, and upload
   `index/corpus.faiss` + `index/docs.jsonl` from the step above.
3. Open `colab/run.ipynb`, set the runtime to a **T4 GPU**.
4. **Cell 1** — bootstrap: mounts Drive, points `HF_HOME` at Drive so weights survive
   runtime resets, clones/pulls the repo, installs dependencies. Uncomment the
   `pip download ... -d wheels/` line on the very first run only, then commit the
   generated `requirements.lock` back into the repo.
5. **Cell 2** — `python -m minissp.rollout --smoke`: loads policy + judge + retriever
   together, prints VRAM usage and `format_valid_rate` / `avg_search_turns` /
   `rule_pass_rate` / `rag_verified_rate` on 16 trajectories, writes samples to
   `runs/smoke/samples/`. Read them before training — a broken tag format won't be
   fixed by training.
6. **Cell 3** — `python -m minissp.train --run-id t4-001 --max-steps 50`. Presence of
   `runs/<run-id>/latest` decides resume vs. init, so after a free-tier disconnect just
   re-run the same cell in a fresh session to continue.

### What to watch

- `format_valid_rate` should stay above 70%.
- `rag_verified_rate` should trend upward — the headline curve.
- `frac_groups_nonzero_var` below 10% for many steps means GRPO has no gradient signal.
- **Collapse signature**: `proposer_mean_reward → 1.0` with `solver_acc → 0` for 3+
  steps — abort and inspect samples; this also catches a dead judge or broken parser.

### Success bar at this scale

50 contiguous steps across ≥2 sessions with no duplicated or missing step in
`metrics.jsonl`, no collapse, `rag_verified_rate` non-decreasing in trend, and
`eval_acc` at step 50 not below step 0 outside its bootstrap CI.

## What's traded away for a free-tier T4

| Official (paper / reference repo) | Here |
|---|---|
| veRL + Ray + FSDP, multi-GPU, full fine-tune | One process, LoRA, one T4 |
| sglang async multi-turn rollout | HF `generate` with `stop_strings`, re-prefill per turn |
| HTTP retrieval server, FAISS-GPU, full wiki-18 (21M passages) | In-process FAISS-CPU over a ~30–60k scoped subset |
| Qwen2.5-32B judge | Qwen2.5-3B-Instruct, 4-bit, in-process |
| Qwen2.5-7B policy, B=256, n=5, 10 turns, 8k tokens | 1.5B policy, B=8, n=4, 3 turns, 2k tokens |

Because retrieval runs over a scoped (artificially easy) corpus, absolute accuracy
numbers are **not** comparable to the paper — only trends (format validity, RAG
verification rate, turn count) are.

Available at https://github.com/pranathitaddi/minissp.git.
