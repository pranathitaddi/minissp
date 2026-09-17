"""
minissp/config.py

Phase 0 of plan.md (§0.6): the single T4 preset, plus the CLI override loop
that makes every field settable as `--field-name value`.

    from minissp.config import Config, parse_args
    cfg = parse_args(["--run-id", "t4-001", "--lr", "1e-4"])

Importing this module must have zero side effects.
"""

import argparse
import dataclasses
import typing
from dataclasses import dataclass


@dataclass
class Config:
    run_id: str
    seed: int = 0
    policy_name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    judge_name: str = "Qwen/Qwen2.5-3B-Instruct"
    embed_name: str = "intfloat/e5-base-v2"
    rag_solver: str = "judge"            # or "policy"
    proposer_reward: str = "1-acc"       # or "intermediate_difficulty"
    batch_answers: int = 8               # B
    n_solver: int = 4                    # GRPO group size
    max_turns_proposer: int = 3
    max_turns_solver: int = 3
    topk: int = 3
    noise_docs: int = 4
    max_prompt_tokens: int = 1536
    max_new_tokens_per_turn: int = 256
    max_total_tokens: int = 2048
    lora_r: int = 16
    lora_alpha: int = 32
    lr: float = 2e-5
    kl_beta: float = 0.01
    replay_reset_every: int = 10
    eval_every: int = 10
    drive_root: str = "/content/drive/MyDrive/minissp"


# ---------------------------------------------------------------------------
# 0.6 CLI override loop
#
# One `--field-name` flag per dataclass field, generated from
# dataclasses.fields(Config) so a renamed field can never silently lose its
# flag. Defaults are deliberately None: only flags the user actually passed
# are forwarded to Config(), so the dataclass stays the single source of
# truth for default values.
# ---------------------------------------------------------------------------

def _field_types() -> dict:
    """Resolved annotations (robust to `from __future__ import annotations`)."""
    return typing.get_type_hints(Config)


def _flag_for(name: str) -> str:
    return "--" + name.replace("_", "-")


def add_config_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Adds one override flag per Config field to an existing parser.

    Entry points (train.py, eval.py) call this and then add their own
    non-Config flags (`--max-steps`, `--step`, ...) to the same parser.
    """
    hints = _field_types()
    for f in dataclasses.fields(Config):
        annotated = hints.get(f.name, str)
        kwargs: dict = {"dest": f.name, "default": None}
        if annotated is bool:
            # Gives both `--flag` and `--no-flag`, so a bool field defaulting
            # to True is still overridable. (Config has no bool fields today;
            # this branch keeps the loop honest if one is ever added.)
            kwargs["action"] = argparse.BooleanOptionalAction
        else:
            kwargs["type"] = annotated if annotated in (int, float, str) else str
            kwargs["metavar"] = f.name.upper()
        default = (f.default if f.default is not dataclasses.MISSING else None)
        kwargs["help"] = f"override Config.{f.name}" + (
            f" (default: {default!r})" if default is not None else " (required)"
        )
        parser.add_argument(_flag_for(f.name), **kwargs)
    return parser


def build_parser(prog: str | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=prog)
    return add_config_args(parser)


def config_from_namespace(args: argparse.Namespace) -> Config:
    """Builds a Config from a namespace produced by `add_config_args`.

    Unset flags (None) are dropped so dataclass defaults apply; extra
    attributes belonging to the caller's own flags are ignored.
    """
    names = {f.name for f in dataclasses.fields(Config)}
    overrides = {
        name: value for name, value in vars(args).items()
        if name in names and value is not None
    }
    if "run_id" not in overrides:
        raise SystemExit("--run-id is required")
    return Config(**overrides)


def parse_args(argv: list[str] | None = None, prog: str | None = None) -> Config:
    """Parses `--field value` overrides into a Config."""
    return config_from_namespace(build_parser(prog).parse_args(argv))