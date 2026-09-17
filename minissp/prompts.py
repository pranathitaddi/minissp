"""
minissp/prompts.py

Phase 2 of plan.md: the four prompt templates used by the self-play loop —
PROPOSER, SOLVER, RAG_SOLVER and JUDGE.

The paper (App. D) is not reproduced here verbatim; these templates are
written to the tag protocol described in plan.md §2.4 and §1:

    <think>...</think>            free-form reasoning (any number of turns)
    <search>...</search>          a retrieval query; the environment answers
    <information>...</information>  retrieved passages, inserted by us
    <answer>...</answer>          terminal tag for solver / rag_solver
    <question>...</question>      terminal tag for proposer

Exactly one terminal tag ends a trajectory. `TAGS` lives here rather than in
rollout.py (where plan §2.4 sketches it) so the parser and the prompts can
never drift apart — rollout.py imports it from this module.

Pure strings only: importing this module must have zero side effects.
"""

from __future__ import annotations

# Every tag the protocol recognises. rollout.parse() treats any other
# angle-bracketed token as literal text (and therefore as a format violation
# if it falls outside a tag body).
TAGS = ("think", "search", "information", "answer", "question")

# The two tags that may terminate a trajectory, by role.
TERMINAL_TAG_BY_ROLE = {
    "solver": "answer",
    "rag_solver": "answer",
    "proposer": "question",
}

_PROTOCOL_BLOCK = """\
You must follow this output protocol exactly:
- Put all of your reasoning inside <think> and </think>.
- To search the web, write a query inside <search> and </search>. The search \
results will be returned to you inside <information> and </information>. Do \
not ever write <information> yourself.
- Every tag you open must be closed. Tags are never nested inside one another.
- Write nothing outside of tags.\
"""


def proposer_prompt(answer: str, n: int, examples: str) -> str:
    """
    Prompt for the proposer role.

    Args:
        answer:   the ground-truth answer the generated question must have.
        n:        the required number of <search> turns (dataset `search_turns`).
        examples: the verbatim `sys_question_example` block from the training
                  row — three unrelated example questions, used purely as
                  style exemplars.

    The proposer explores the corpus with exactly `n` searches, then emits a
    single <question>...</question> whose answer is `answer`, without naming
    the answer in the question.
    """
    turn_word = "search" if n == 1 else "searches"
    return f"""\
You are a question designer. You are given an answer, and your job is to write \
one search-intensive question whose correct answer is exactly that answer.

The answer is: {answer}

{_PROTOCOL_BLOCK}
- When you are done searching, write your final question inside <question> and \
</question>. That is the last thing you write.

Requirements for your question:
1. Use exactly {n} {turn_word} (exactly {n} <search>...</search> blocks) before \
writing your question. Not fewer, not more.
2. Use the retrieved information to ground the question in real, verifiable \
facts, so that another model with the same search tool can find the answer.
3. The question must be answerable with the given answer and with nothing else \
— it must have a single unambiguous answer.
4. Do NOT reveal the answer in the question. The answer string must not appear \
in the question, in any casing or wording.
5. The question must be self-contained: it may not refer to "the passage", \
"the text above" or the search results.
6. Write at least 10 words.

Here are examples of the style of question expected. They are unrelated to \
your answer; copy only their style, never their content.

{examples}

Begin now with <think>.\
"""


def solver_prompt(question: str) -> str:
    """
    Prompt for the solver role: multi-turn think/search/information cycling
    that ends in a single <answer>...</answer>.
    """
    return f"""\
You are a research assistant answering a question with the help of a search \
tool.

Question: {question}

{_PROTOCOL_BLOCK}
- You may alternate <think> and <search> as many times as you need. After each \
search you will receive an <information> block; read it, think again, and \
search again if the answer is still not certain.
- When you know the answer, write it inside <answer> and </answer>. That is the \
last thing you write.

Keep the content of <answer> short: just the answer itself (an entity, a date, \
a number or a short phrase), with no explanation and no full sentence.

Begin now with <think>.\
"""


def rag_solver_prompt(question: str, information: str) -> str:
    """
    Prompt for the RAG verifier: one shot, no searching, answer strictly from
    the documents supplied (already rendered by `retrieval.format_information`).
    """
    return f"""\
You are a research assistant. Answer the question using ONLY the documents \
below. You have no search tool and you may not use outside knowledge.

Documents:
{information}

Question: {question}

Rules:
- Answer in a single turn. Do not write <search> or <think>.
- Write only <answer>your answer</answer> and nothing else.
- Keep the answer short: just the answer itself (an entity, a date, a number or \
a short phrase).
- If the documents do not contain the answer, write <answer>unknown</answer>.

<answer>\
"""


def judge_prompt(question: str, targets: list[str], prediction: str) -> str:
    """
    Prompt for the equivalence judge.

    `judge.Judge.is_correct` parses the FIRST whitespace-delimited token of the
    completion and requires it to be exactly "Correct" or "Wrong", so the
    instruction is stated twice and the completion is pre-seeded with a label
    line to make any other first token very unlikely.
    """
    gold = "\n".join(f"- {t}" for t in targets)
    return f"""\
You are grading a short answer. Decide whether the predicted answer means the \
same thing as any one of the gold answers.

Question: {question}

Gold answers (any one of these counts as correct):
{gold}

Predicted answer: {prediction}

Grading rules:
- Judge meaning, not wording: different casing, punctuation, articles, word \
order, abbreviations, or extra qualifying words are fine if the substance \
matches.
- A prediction that is more specific than the gold answer but still names the \
same thing is Correct.
- An empty, evasive, or "unknown" prediction is Wrong.
- A prediction naming a different entity, date or number is Wrong.

Answer with exactly one word: Correct or Wrong. Write nothing else — no \
punctuation, no explanation.

Verdict:\
"""
