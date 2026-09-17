"""
tests/test_runtime.py

Phase 3 of plan.md (§3.7): the metrics writer's fixed schema and the
checkpoint save -> verify -> load round trip.

Nothing here needs a GPU or a downloaded model: the checkpoint test uses a
real `peft` LoRA adapter wrapped around a two-layer nn.Module, so the actual
PEFT save_pretrained / load_adapter path and real file I/O are exercised.
"""

from __future__ import annotations

import json
import random

import pytest
import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model

from minissp.runtime import (
    METRICS_COLUMNS,
    ManifestError,
    MetricsSchemaError,
    MetricsWriter,
    build_manifest,
    capture_rng_state,
    checkpoint_dir,
    existing_steps,
    load_checkpoint,
    prune_checkpoints,
    read_latest,
    restore_rng_state,
    save_checkpoint,
    verify_manifest,
    write_latest,
)
from minissp.train import ReplayBuffer, ReplayItem

# ---------------------------------------------------------------------------
# MetricsWriter
# ---------------------------------------------------------------------------


def _full_row(**overrides) -> dict:
    row = {name: 0 for name in METRICS_COLUMNS}
    row["session_id"] = "s"
    row["skipped_update"] = False
    row["eval_acc"] = None
    row.update(overrides)
    return row


def test_metrics_writer_accepts_a_complete_row(tmp_path):
    writer = MetricsWriter(tmp_path / "metrics.jsonl")
    writer.write(_full_row(step=1))
    rows = writer.read_rows()
    assert len(rows) == 1
    assert set(rows[0]) == set(METRICS_COLUMNS)
    # Column order on disk is the declared order, so an exported CSV is stable.
    assert list(rows[0]) == list(METRICS_COLUMNS)


def test_metrics_writer_rejects_a_row_missing_a_column(tmp_path):
    writer = MetricsWriter(tmp_path / "metrics.jsonl")
    row = _full_row(step=1)
    del row["kl"]
    with pytest.raises(MetricsSchemaError) as exc:
        writer.write(row)
    assert "kl" in str(exc.value)
    assert not (tmp_path / "metrics.jsonl").exists()


def test_metrics_writer_rejects_an_unknown_column(tmp_path):
    # A typo'd column would otherwise be written forever and never plotted.
    writer = MetricsWriter(tmp_path / "metrics.jsonl")
    with pytest.raises(MetricsSchemaError) as exc:
        writer.write(_full_row(step=1, solvr_acc=0.5))
    assert "solvr_acc" in str(exc.value)


def test_metrics_writer_appends(tmp_path):
    writer = MetricsWriter(tmp_path / "metrics.jsonl")
    for i in (1, 2, 3):
        writer.write(_full_row(step=i))
    assert [r["step"] for r in writer.read_rows()] == [1, 2, 3]


def test_metrics_truncate_after_drops_rows_past_the_checkpoint(tmp_path):
    # plan §3.8: on resume, metrics rows past `latest` are reconciled away.
    writer = MetricsWriter(tmp_path / "metrics.jsonl")
    for i in range(1, 6):
        writer.write(_full_row(step=i))
    assert writer.truncate_after(3) == 2
    assert [r["step"] for r in writer.read_rows()] == [1, 2, 3]


# ---------------------------------------------------------------------------
# latest / pruning
# ---------------------------------------------------------------------------


def test_write_latest_is_zero_padded_and_readable(tmp_path):
    write_latest(tmp_path, 42)
    assert (tmp_path / "latest").read_text().strip() == "00042"
    assert read_latest(tmp_path) == 42


def test_read_latest_is_none_for_a_fresh_run(tmp_path):
    assert read_latest(tmp_path) is None


def test_prune_keeps_last_two_and_every_tenth(tmp_path):
    for step in range(1, 15):
        checkpoint_dir(tmp_path, step).mkdir(parents=True)
    deleted = prune_checkpoints(tmp_path, keep_last=2, keep_every=10)
    assert existing_steps(tmp_path) == [10, 13, 14]
    assert set(deleted) == set(range(1, 15)) - {10, 13, 14}


# ---------------------------------------------------------------------------
# checkpoints
# ---------------------------------------------------------------------------


class Tiny(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lin1 = nn.Linear(8, 8)
        self.lin2 = nn.Linear(8, 4)

    def forward(self, x):
        return self.lin2(torch.tanh(self.lin1(x)))


def _lora_model(seed: int = 0):
    torch.manual_seed(seed)
    model = get_peft_model(Tiny(), LoraConfig(r=4, lora_alpha=8, target_modules=["lin1"]))
    # lora_B initialises to zeros; randomise so a save/load round trip that
    # silently dropped the tensors could not pass by accident.
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "lora_" in name:
                param.copy_(torch.randn_like(param))
    return model


def _lora_tensors(model) -> dict:
    return {k: v.clone() for k, v in model.state_dict().items() if "lora_" in k}


def _state(step: int) -> dict:
    return {
        "step": step,
        "session_id": "sess-1",
        "gpu_name": "cpu",
        "git_sha": "deadbeef",
        "config_hash": "0123456789abcdef",
        "index_sha": "fedcba9876543210",
        "judge_name": "Qwen/Qwen2.5-3B-Instruct",
    }


def test_checkpoint_round_trip_restores_adapter_and_rng(tmp_path):
    model = _lora_model(seed=0)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=1e-3)
    # One real optimizer step so the saved state is non-trivial.
    model(torch.randn(2, 8)).sum().backward()
    optimizer.step()
    scaler = torch.amp.GradScaler("cpu", enabled=False)

    replay = ReplayBuffer()
    replay.push(ReplayItem(question="who wrote hamlet?", answer="Shakespeare",
                           docs=[{"id": "1", "title": "Hamlet", "text": "a tragedy"}]))

    random.seed(1234)
    torch.manual_seed(1234)
    rng_state = capture_rng_state()
    expected_python = random.random()
    expected_torch = torch.randn(3)

    ckpt_path = save_checkpoint(tmp_path, model, optimizer, scaler, rng_state,
                                replay, _state(1))
    assert ckpt_path == checkpoint_dir(tmp_path, 1)
    assert read_latest(tmp_path) == 1

    loaded = load_checkpoint(tmp_path)
    assert loaded["step"] == 1
    assert loaded["state"]["judge_name"] == "Qwen/Qwen2.5-3B-Instruct"
    assert loaded["replay"] == [{
        "question": "who wrote hamlet?", "answer": "Shakespeare",
        "docs": [{"id": "1", "title": "Hamlet", "text": "a tragedy"}],
        "is_dummy": False, "proposer_idx": None,
    }]

    # identical adapter tensors
    fresh = _lora_model(seed=99)
    before = _lora_tensors(fresh)
    fresh.load_adapter(str(loaded["adapter_dir"]), adapter_name="default")
    after = _lora_tensors(fresh)
    saved = _lora_tensors(model)
    assert set(after) == set(saved)
    assert any(not torch.equal(before[k], saved[k]) for k in saved), \
        "the two models started identical; the test would prove nothing"
    for k in saved:
        assert torch.equal(after[k], saved[k]), k

    # identical next RNG draw
    random.seed(0)
    torch.manual_seed(0)
    restore_rng_state(loaded["rng_state"])
    assert random.random() == expected_python
    assert torch.equal(torch.randn(3), expected_torch)

    # optimizer state survived
    assert loaded["optimizer"]["state"]


def test_manifest_verification_catches_a_tampered_file(tmp_path):
    model = _lora_model()
    save_checkpoint(tmp_path, model, None, None, None, ReplayBuffer(), _state(1))
    ckpt = checkpoint_dir(tmp_path, 1)
    verify_manifest(ckpt)  # clean

    (ckpt / "replay.jsonl").write_text('{"question": "tampered"}\n', encoding="utf-8")
    with pytest.raises(ManifestError):
        verify_manifest(ckpt)
    with pytest.raises(ManifestError):
        load_checkpoint(tmp_path)


def test_manifest_verification_catches_a_deleted_file(tmp_path):
    model = _lora_model()
    save_checkpoint(tmp_path, model, None, None, None, ReplayBuffer(), _state(2))
    (checkpoint_dir(tmp_path, 2) / "replay.jsonl").unlink()
    with pytest.raises(ManifestError):
        load_checkpoint(tmp_path)


def test_manifest_does_not_hash_itself(tmp_path):
    model = _lora_model()
    save_checkpoint(tmp_path, model, None, None, None, ReplayBuffer(), _state(3))
    ckpt = checkpoint_dir(tmp_path, 3)
    manifest = json.loads((ckpt / "manifest.json").read_text())
    assert "manifest.json" not in manifest["files"]
    assert "state.json" in manifest["files"]
    assert build_manifest(ckpt) == manifest


def test_save_checkpoint_requires_a_step(tmp_path):
    with pytest.raises(ValueError):
        save_checkpoint(tmp_path, None, None, None, None, ReplayBuffer(), {})


def test_load_checkpoint_without_latest_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_checkpoint(tmp_path)
