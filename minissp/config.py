
from dataclasses import dataclass


@dataclass(frozen=True)
class VRAMBudget:
    policy_gb: float
    judge_gb: float
    cap_gb: float

    def headroom_gb(self) -> float:
        return self.cap_gb - (self.policy_gb + self.judge_gb)


@dataclass(frozen=True)
class TierPreset:
    name: str
    vram: VRAMBudget
    batch_size: int
    n_solver: int
    max_search_turns_proposer: int
    max_search_turns_solver: int


TIER_PRESETS = {
    "t4": TierPreset("t4", VRAMBudget(policy_gb=5.1, judge_gb=5.0, cap_gb=16.0),
                      batch_size=8, n_solver=4,
                      max_search_turns_proposer=3, max_search_turns_solver=3),
    "l4": TierPreset("l4", VRAMBudget(policy_gb=6.1, judge_gb=9.0, cap_gb=24.0),
                      batch_size=16, n_solver=4,
                      max_search_turns_proposer=4, max_search_turns_solver=4),
    "a100": TierPreset("a100", VRAMBudget(policy_gb=12.0, judge_gb=20.0, cap_gb=40.0),
                        batch_size=32, n_solver=5,
                        max_search_turns_proposer=5, max_search_turns_solver=6),
}