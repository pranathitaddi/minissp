from minissp.config import TIER_PRESETS


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