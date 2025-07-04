"""Example script demonstrating MIPROv2 prompt optimization for a dummy agent.

This script creates a simple DummyAgent with a basic system prompt. A few
conversation logs are added along with scores to simulate evaluation
feedback. The agent then invokes DSPy's MIPROv2 optimizer to generate a
better version of its prompt based on the provided history.

Set the OPENAI_API_KEY environment variable before running this script.
"""

import os
from collections import deque

import dspy
from dspy.teleprompt.mipro_optimizer_v2 import MIPROv2

from core.dspy_utils import build_dataset, prompt_program


class DummyAgent:
    """A minimal agent storing a prompt and conversation history."""

    def __init__(self, prompt: str):
        self.prompt = prompt
        self.history = deque()

    def log(self, message: str, score: float):
        """Record a conversation turn with an associated score."""
        self.history.append({
            "turns": [{"speaker": "user", "text": message}],
            "score": score,
        })

    def improve_prompt(self) -> str:
        """Run MIPROv2 to optimize the current prompt using the history."""
        dataset = build_dataset(self.history)
        trainset = dataset.train
        if len(trainset) < 2:
            trainset = trainset * 2
        metric = lambda ex: ex.score

        optimizer = MIPROv2(metric=metric, auto="light")
        program = prompt_program(self.prompt)
        new_prog = optimizer.compile(program, trainset=trainset)
        self.prompt = new_prog.signature.__doc__ or self.prompt
        return self.prompt


if __name__ == "__main__":
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY environment variable not set")

    lm = dspy.LM("openai/gpt-4o-mini", api_key=api_key)
    dspy.configure(lm=lm)

    agent = DummyAgent("You are a helpful assistant.")
    agent.log("hello", 0.4)
    agent.log("what is the capital of France?", 0.8)

    print("Original prompt:\n", agent.prompt)
    print("Running MIPROv2 to improve prompt...")
    improved = agent.improve_prompt()
    print("Improved prompt:\n", improved)
