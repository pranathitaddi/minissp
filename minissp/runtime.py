"""
minissp/runtime.py

Phase 3 of plan.md (§3.7): the two pieces of run bookkeeping that must be
boringly correct because a Colab runtime can die between any two lines —

    MetricsWriter(path).write(row)              one validated JSON line, fsynced
    save_checkpoint(run_dir, ...)               adapter + optimizer + rng + manifest
    load_checkpoint(run_dir_or_ckpt_dir)        manifest verified BEFORE loading

Checkpoint layout (plan §2 "Drive layout"):

    runs/<run_id>/
      metrics.jsonl
      latest                   text file: "00042", written via tmp + os.replace
      ckpt/step_00042/
        adapter/               PEFT save_pretrained()
        optimizer.pt scaler.pt rng.pt
        replay.jsonl
        state.json             step, session_id, gpu_name, git_sha, config_hash,
                               index_sha, judge_name
        manifest.json          SHA256 of every other file in the directory

`model`, `optimizer` and `scaler` are duck-typed: anything with
`save_pretrained()` / `state_dict()` respectively works, which is what lets the
round-trip be tested on CPU without an LLM.

torch is imported inside functions so `import minissp.runtime` stays cheap.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import shutil
from pathlib import Path

# ---------------------------------------------------------------------------
# §3.7 metrics
# ---------------------------------------------------------------------------

METRICS_COLUMNS: tuple[str, ...] = (
    "step", "session_id",
    "t_propose", "t_verify", "t_solve", "t_judge", "t_update", "t_ckpt",
    "rule_pass_rate", "rag_verified_rate",
    "fresh_verified", "replay_filled", "dummy_filled",
    "solver_acc", "proposer_mean_reward", "frac_groups_nonzero_var",
    "avg_turns_proposer", "avg_turns_solver", "format_valid_rate",
    "loss", "kl", "grad_norm", "skipped_update", "judge_errors", "eval_acc",
)

_METRICS_KEYS = frozenset(METRICS_COLUMNS)


class MetricsSchemaError(ValueError):
    """Raised when a metrics row does not match METRICS_COLUMNS exactly."""


class MetricsWriter:
    """Append-only JSONL writer with a fixed, fully-enforced column set.

    Both directions are errors: a missing column means a metric silently
    stopped being computed, an unknown column means a typo that would never
    show up in the exported CSV. Neither should be discoverable weeks later.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def validate(row: dict) -> None:
        keys = set(row)
        missing = _METRICS_KEYS - keys
        extra = keys - _METRICS_KEYS
        if missing or extra:
            parts = []
            if missing:
                parts.append(f"missing columns {sorted(missing)}")
            if extra:
                parts.append(f"unknown columns {sorted(extra)}")
            raise MetricsSchemaError("; ".join(parts))

    def write(self, row: dict) -> None:
        self.validate(row)
        ordered = {k: row[k] for k in METRICS_COLUMNS}
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(ordered) + "\n")
            f.flush()
            os.fsync(f.fileno())

    def read_rows(self) -> list[dict]:
        if not self.path.exists():
            return []
        rows = []
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows

    def truncate_after(self, step: int) -> int:
        """Drops rows with step > `step` (resume reconciliation, plan §3.8).

        Returns the number of rows dropped.
        """
        rows = self.read_rows()
        kept = [r for r in rows if r.get("step", 0) <= step]
        if len(kept) == len(rows):
            return 0
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            for r in kept:
                f.write(json.dumps(r) + "\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, self.path)
        return len(rows) - len(kept)


# ---------------------------------------------------------------------------
# hashing / atomic writes
# ---------------------------------------------------------------------------

def sha256_file(path: str | Path, chunk: int = 65536) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def atomic_write_text(path: str | Path, text: str) -> None:
    """tmp file in the same directory + os.replace, so a reader never sees half."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _iter_files(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.is_file())


def build_manifest(ckpt_dir: str | Path) -> dict:
    """SHA256 of every file in the checkpoint dir except manifest.json itself."""
    ckpt_dir = Path(ckpt_dir)
    files = {}
    for p in _iter_files(ckpt_dir):
        rel = p.relative_to(ckpt_dir).as_posix()
        if rel == MANIFEST_NAME:
            continue
        files[rel] = sha256_file(p)
    return {"files": files}


class ManifestError(RuntimeError):
    """Raised when a checkpoint's manifest does not match what is on disk."""


def verify_manifest(ckpt_dir: str | Path) -> None:
    """Recomputes every SHA256 and compares. Raises before anything is loaded."""
    ckpt_dir = Path(ckpt_dir)
    manifest_path = ckpt_dir / MANIFEST_NAME
    if not manifest_path.exists():
        raise ManifestError(f"no {MANIFEST_NAME} in {ckpt_dir}")
    with open(manifest_path, "r", encoding="utf-8") as f:
        recorded = json.load(f).get("files", {})
    actual = build_manifest(ckpt_dir)["files"]

    missing = sorted(set(recorded) - set(actual))
    unexpected = sorted(set(actual) - set(recorded))
    changed = sorted(k for k in set(recorded) & set(actual) if recorded[k] != actual[k])
    if missing or unexpected or changed:
        raise ManifestError(
            f"checkpoint {ckpt_dir} failed manifest verification: "
            f"missing={missing} unexpected={unexpected} changed={changed}"
        )


# ---------------------------------------------------------------------------
# §3.7 checkpoints
# ---------------------------------------------------------------------------

LATEST_NAME = "latest"
CKPT_DIRNAME = "ckpt"
MANIFEST_NAME = "manifest.json"
STATE_NAME = "state.json"
ADAPTER_DIRNAME = "adapter"
REPLAY_NAME = "replay.jsonl"
KEEP_LAST = 2
KEEP_EVERY = 10


def step_dirname(step: int) -> str:
    return f"step_{step:05d}"


def ckpt_root(run_dir: str | Path) -> Path:
    return Path(run_dir) / CKPT_DIRNAME


def checkpoint_dir(run_dir: str | Path, step: int) -> Path:
    return ckpt_root(run_dir) / step_dirname(step)


def read_latest(run_dir: str | Path) -> int | None:
    """The step recorded in `runs/<id>/latest`, or None if there is none."""
    path = Path(run_dir) / LATEST_NAME
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return None
    return int(text)


def write_latest(run_dir: str | Path, step: int) -> None:
    atomic_write_text(Path(run_dir) / LATEST_NAME, f"{step:05d}\n")


def existing_steps(run_dir: str | Path) -> list[int]:
    root = ckpt_root(run_dir)
    if not root.exists():
        return []
    steps = []
    for p in root.iterdir():
        if p.is_dir() and p.name.startswith("step_"):
            try:
                steps.append(int(p.name[len("step_"):]))
            except ValueError:
                continue
    return sorted(steps)


def prune_checkpoints(run_dir: str | Path, keep_last: int = KEEP_LAST,
                      keep_every: int = KEEP_EVERY) -> list[int]:
    """Keeps the last `keep_last` checkpoints and every `keep_every`-th step.

    Returns the steps that were deleted.
    """
    steps = existing_steps(run_dir)
    keep = set(steps[-keep_last:]) if keep_last > 0 else set()
    if keep_every > 0:
        keep |= {s for s in steps if s % keep_every == 0}
    deleted = []
    for s in steps:
        if s in keep:
            continue
        shutil.rmtree(checkpoint_dir(run_dir, s), ignore_errors=True)
        deleted.append(s)
    return deleted


def capture_rng_state() -> dict:
    """Python + numpy + torch RNG state, for an exactly-resumable run."""
    import random

    import torch

    state = {"python": random.getstate(), "torch": torch.get_rng_state()}
    try:
        import numpy as np
        state["numpy"] = np.random.get_state()
    except Exception:  # noqa: BLE001 — numpy state is a nicety, never fatal
        pass
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict) -> None:
    import random

    import torch

    if "python" in state:
        random.setstate(state["python"])
    if "torch" in state:
        torch.set_rng_state(state["torch"])
    if "numpy" in state:
        try:
            import numpy as np
            np.random.set_state(state["numpy"])
        except Exception:  # noqa: BLE001
            pass
    if "torch_cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def save_checkpoint(run_dir: str | Path, model, optimizer, scaler,
                    rng_state: dict | None, replay, state: dict) -> Path:
    """Writes one checkpoint under `run_dir/ckpt/step_XXXXX/` and updates `latest`.

    `state` must carry at least `step`; the remaining plan §3.7 keys
    (session_id, gpu_name, git_sha, config_hash, index_sha, judge_name) are
    written through as given so the caller decides how to compute them.

    Order matters: everything is written, then the manifest, then `latest`
    (atomically). A run killed mid-save leaves a checkpoint that `latest`
    never points at, which is the safe failure.
    """
    import torch

    if "step" not in state:
        raise ValueError("checkpoint state must include 'step'")
    step = int(state["step"])

    run_dir = Path(run_dir)
    ckpt_dir = checkpoint_dir(run_dir, step)
    if ckpt_dir.exists():
        shutil.rmtree(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # adapter (PEFT) — duck-typed on save_pretrained
    if model is not None:
        model.save_pretrained(str(ckpt_dir / ADAPTER_DIRNAME))

    if optimizer is not None:
        torch.save(optimizer.state_dict(), ckpt_dir / "optimizer.pt")
    if scaler is not None:
        torch.save(scaler.state_dict(), ckpt_dir / "scaler.pt")
    if rng_state is not None:
        torch.save(rng_state, ckpt_dir / "rng.pt")

    rows = _replay_rows(replay)
    with open(ckpt_dir / REPLAY_NAME, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    with open(ckpt_dir / STATE_NAME, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, sort_keys=True, default=str)

    with open(ckpt_dir / MANIFEST_NAME, "w", encoding="utf-8") as f:
        json.dump(build_manifest(ckpt_dir), f, indent=2, sort_keys=True)

    write_latest(run_dir, step)
    prune_checkpoints(run_dir)
    return ckpt_dir


def _replay_rows(replay) -> list[dict]:
    """Accepts a ReplayBuffer, a list of dicts, a list of dataclasses, or None."""
    if replay is None:
        return []
    if hasattr(replay, "serialize"):
        return list(replay.serialize())
    rows = []
    for item in replay:
        if dataclasses.is_dataclass(item) and not isinstance(item, type):
            rows.append(dataclasses.asdict(item))
        else:
            rows.append(dict(item))
    return rows


def resolve_checkpoint_dir(path: str | Path) -> Path:
    """Accepts either a checkpoint dir or a run dir (resolved via `latest`)."""
    path = Path(path)
    if (path / MANIFEST_NAME).exists():
        return path
    step = read_latest(path)
    if step is None:
        raise FileNotFoundError(
            f"{path} is neither a checkpoint dir (no {MANIFEST_NAME}) nor a run "
            f"dir with a `{LATEST_NAME}` file"
        )
    return checkpoint_dir(path, step)


def load_checkpoint(path: str | Path, map_location: str = "cpu") -> dict:
    """Verifies the manifest, THEN loads.

    Returns {"dir", "step", "state", "replay", "adapter_dir", "optimizer",
    "scaler", "rng_state"}; the caller applies them (load_state_dict /
    PeftModel.from_pretrained) because runtime.py must not know about peft.
    """
    import torch

    ckpt_dir = resolve_checkpoint_dir(path)
    verify_manifest(ckpt_dir)  # before touching anything else

    with open(ckpt_dir / STATE_NAME, "r", encoding="utf-8") as f:
        state = json.load(f)

    replay: list[dict] = []
    replay_path = ckpt_dir / REPLAY_NAME
    if replay_path.exists():
        with open(replay_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    replay.append(json.loads(line))

    def _maybe_load(name):
        p = ckpt_dir / name
        if not p.exists():
            return None
        return torch.load(p, map_location=map_location, weights_only=False)

    adapter_dir = ckpt_dir / ADAPTER_DIRNAME
    return {
        "dir": ckpt_dir,
        "step": int(state.get("step", 0)),
        "state": state,
        "replay": replay,
        "adapter_dir": adapter_dir if adapter_dir.exists() else None,
        "optimizer": _maybe_load("optimizer.pt"),
        "scaler": _maybe_load("scaler.pt"),
        "rng_state": _maybe_load("rng.pt"),
    }


# ---------------------------------------------------------------------------
# provenance helpers used to fill state.json
# ---------------------------------------------------------------------------

def git_sha(repo_dir: str | Path | None = None) -> str:
    import subprocess
    repo_dir = Path(repo_dir) if repo_dir else Path(__file__).resolve().parent.parent
    try:
        out = subprocess.run(["git", "-C", str(repo_dir), "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or "unknown"
    except Exception:  # noqa: BLE001 — provenance must never break a run
        return "unknown"


def config_hash(cfg) -> str:
    payload = json.dumps(dataclasses.asdict(cfg), sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def index_sha(index_dir: str | Path) -> str:
    """SHA256 of corpus.faiss, truncated — pins which corpus a run used."""
    path = Path(index_dir) / "corpus.faiss"
    if not path.exists():
        return "unknown"
    return sha256_file(path)[:16]


def gpu_name() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except Exception:  # noqa: BLE001
        pass
    return "cpu"
