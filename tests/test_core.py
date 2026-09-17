import json
from pathlib import Path

from minissp.config import TIER_PRESETS
from minissp.data import (
    TOY_ANSWERS_PATH, TOY_EVAL_PATH, MANIFEST_PATH,
    sha256_of, answer_set,
)


def test_tier_presets_fit_vram_cap():
    for name, preset in TIER_PRESETS.items():
        used = preset.vram.policy_gb + preset.vram.judge_gb
        assert used < preset.vram.cap_gb, f"{name} preset exceeds VRAM cap"
        assert preset.vram.headroom_gb() > 0


def test_package_imports():
    import minissp.config
    import minissp.prompts
    import minissp.data  # noqa: F401
    import minissp.retrieval
    import minissp.judge
    import minissp.rollout  # noqa: F401
    import minissp.train
    import minissp.runtime
    import minissp.eval  # noqa: F401


def test_manifest_matches_committed_files():
    manifest = json.loads(MANIFEST_PATH.read_text())
    for path_str, expected_hash in manifest.items():
        assert sha256_of(Path(path_str)) == expected_hash, f"{path_str} hash mismatch"


def test_no_answer_overlap_between_train_and_eval():
    train_answers = answer_set(TOY_ANSWERS_PATH)
    eval_answers = answer_set(TOY_EVAL_PATH)
    overlap = train_answers & eval_answers
    assert not overlap, f"{len(overlap)} answers leak between train and eval sets"


def test_eval_toy_composition():
    with TOY_EVAL_PATH.open() as f:
        rows = [json.loads(line) for line in f]
    nq = [r for r in rows if r.get("source") == "nq"]
    hotpot = [r for r in rows if r.get("source") == "hotpotqa"]
    assert len(nq) == 150 and len(hotpot) == 150