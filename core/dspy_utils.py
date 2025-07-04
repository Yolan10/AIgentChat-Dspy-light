from __future__ import annotations
from collections import deque
from typing import List, Dict

from dspy import Example, Program, MIPROv2


def build_dataset(history: deque[Dict]) -> List[Example]:
    """Return a list of DSPy Examples derived from the conversation history.

    Only conversations with an attached ``score`` field are included. The turns
    from each conversation are collapsed into a single block of text.
    """
    dataset: List[Example] = []
    for entry in history:
        score = entry.get("score")
        if score is None:
            continue
        text = "\n".join(turn.get("text", "") for turn in entry.get("turns", []))
        dataset.append(Example(text=text, score=score))

    return dataset


class PromptProgram(Program):
    """Simple DSPy program that always returns its prompt text."""

    def __init__(self, prompt: str):
        super().__init__()
        self.prompt = prompt

    def forward(self, text: str) -> str:  # pragma: no cover - trivial
        return self.prompt


def optimize_prompt(
    initial_prompt: str, dataset: List[Example], minibatch_size: int
) -> tuple[str, Dict]:
    """Run MIPROv2 on ``dataset`` to optimise ``initial_prompt``.

    Returns the best prompt and a dictionary of optimizer metrics.
    """
    optimizer = MIPROv2(metric=lambda ex, pred: ex.score)
    program = PromptProgram(initial_prompt)
    optimized = optimizer.compile(
        program, trainset=dataset, minibatch_size=minibatch_size
    )
    return optimized.prompt, optimizer.get_params()
