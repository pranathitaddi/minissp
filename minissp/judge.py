"""
minissp/judge.py

Phase 2 of plan.md (§2.3): the frozen second model that lives beside the
policy on the same GPU. It does two jobs:

    Judge.is_correct(question, targets, prediction) -> bool
        answer-equivalence grading for solver rewards and for RAG verification
    Judge.rag_solve(question, docs) -> str
        one-shot RAG solving used by rollout.rag_verify

Both accept a single item or a list of items (list in, list out) — the rollout
calls them 40+ times per step, so batching is the normal path.

Loaded in 4-bit nf4 with fp16 compute (bitsandbytes), greedy decoding.
Lazy exactly like retrieval.py: `import minissp.judge` must not import torch,
transformers or bitsandbytes and must not touch disk or the network.
"""

from __future__ import annotations

import re

from minissp.prompts import judge_prompt, rag_solver_prompt
from minissp.retrieval import format_information

VERDICTS = {"Correct", "Wrong"}

_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)
_ANSWER_CLOSE_RE = re.compile(r"^(.*?)</answer>", re.DOTALL | re.IGNORECASE)


class JudgeParseError(Exception):
    """Raised when the judge's verdict token is neither "Correct" nor "Wrong".

    Never coerced to False (plan §2.3): a silent False would look exactly like
    a legitimately wrong answer and would quietly depress solver rewards.
    """


def parse_verdict(text: str) -> bool:
    """
    Parses a judge completion: the first whitespace-delimited token must be
    "Correct" or "Wrong" (case-insensitively, ignoring trailing punctuation).
    Anything else raises JudgeParseError.
    """
    tokens = (text or "").strip().split()
    if not tokens:
        raise JudgeParseError("judge produced an empty completion")
    first = tokens[0].strip(".,:;!?'\"*")
    for verdict in VERDICTS:
        if first.lower() == verdict.lower():
            return verdict == "Correct"
    raise JudgeParseError(
        f"judge first token {first!r} is not one of {sorted(VERDICTS)} "
        f"(full completion: {text.strip()[:120]!r})"
    )


def parse_answer(text: str) -> str:
    """Extracts the content of <answer>...</answer>, or "" if there is none.

    Also accepts a completion that opens with the answer body because the
    prompt already seeded "<answer>" (see prompts.rag_solver_prompt).
    """
    text = text or ""
    match = _ANSWER_RE.search(text)
    if match:
        return match.group(1).strip()
    match = _ANSWER_CLOSE_RE.search(text)
    if match:
        return match.group(1).strip()
    return ""


class Judge:
    """Frozen 4-bit Qwen used as equivalence judge and RAG verifier."""

    def __init__(self, name: str = "Qwen/Qwen2.5-3B-Instruct", device: str = "cuda") -> None:
        # Nothing loaded here on purpose (plan §2.3).
        self.name = name
        self.device = device
        self.model = None
        self.tok = None

    # -- loading ----------------------------------------------------------

    def is_loaded(self) -> bool:
        return self.model is not None and self.tok is not None

    def load(self) -> "Judge":
        """Loads tokenizer + nf4-quantized model. Idempotent."""
        if self.is_loaded():
            return self

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        self.tok = AutoTokenizer.from_pretrained(self.name, padding_side="left")
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            self.name,
            quantization_config=quant_config,
            device_map=self.device,
        )
        self.model.eval()
        return self

    # -- generation seam --------------------------------------------------

    def _render(self, prompt: str) -> str:
        """Wraps a raw prompt in the model's chat template."""
        return self.tok.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )

    def _generate(self, prompts: list[str], max_new_tokens: int) -> list[str]:
        """
        Greedy batched generation. This is the single seam the tests stub, so
        no test ever needs a GPU or a downloaded model.
        """
        if not self.is_loaded():
            raise RuntimeError(
                "Judge.load() must be called before generating; the model is "
                "loaded lazily on purpose."
            )
        if not prompts:
            return []

        import torch

        rendered = [self._render(p) for p in prompts]
        batch = self.tok(rendered, return_tensors="pt", padding=True,
                         add_special_tokens=False).to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(
                **batch,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tok.pad_token_id,
            )
        prompt_len = batch["input_ids"].shape[1]
        return self.tok.batch_decode(out[:, prompt_len:], skip_special_tokens=True)

    # -- public API -------------------------------------------------------

    def is_correct(self, question, targets, prediction):
        """
        Judges answer equivalence.

        Single form:  is_correct(question: str, targets: list[str], prediction: str) -> bool
        Batched form: is_correct(questions: list[str], targets: list[list[str]],
                                 predictions: list[str]) -> list[bool]

        Raises JudgeParseError if any verdict token is unparseable.
        """
        single = isinstance(question, str)
        questions = [question] if single else list(question)
        target_lists = [targets] if single else list(targets)
        predictions = [prediction] if single else list(prediction)

        if not (len(questions) == len(target_lists) == len(predictions)):
            raise ValueError(
                f"is_correct got mismatched batch sizes: {len(questions)} questions, "
                f"{len(target_lists)} target lists, {len(predictions)} predictions"
            )

        prompts = [judge_prompt(q, list(t), p)
                   for q, t, p in zip(questions, target_lists, predictions)]
        completions = self._generate(prompts, max_new_tokens=8)
        verdicts = [parse_verdict(c) for c in completions]
        return verdicts[0] if single else verdicts

    def rag_solve(self, question, docs):
        """
        One-shot RAG solving over pre-retrieved documents.

        Single form:  rag_solve(question: str, docs: list[dict]) -> str
        Batched form: rag_solve(questions: list[str], docs: list[list[dict]]) -> list[str]

        Returns the text inside <answer>...</answer>, or "" if the model did
        not produce one.
        """
        single = isinstance(question, str)
        questions = [question] if single else list(question)
        doc_lists = [docs] if single else list(docs)

        if len(questions) != len(doc_lists):
            raise ValueError(
                f"rag_solve got mismatched batch sizes: {len(questions)} questions, "
                f"{len(doc_lists)} doc lists"
            )

        prompts = [rag_solver_prompt(q, format_information(d))
                   for q, d in zip(questions, doc_lists)]
        completions = self._generate(prompts, max_new_tokens=256)
        answers = [parse_answer(c) for c in completions]
        return answers[0] if single else answers
