import os
from typing import List

# Basic configuration
POPULATION_SIZE = 36
SELF_IMPROVE_AFTER: List[int] = [1, 10, 35]
LLM_MODEL = os.environ.get("LLM_MODEL", "gpt-4.1-nano")
PARALLEL_CONVERSATIONS = True
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# DSPy optimizer settings
DSPY_MIPRO_MINIBATCH_SIZE = 4
DSPY_BOOTSTRAP_MINIBATCH_SIZE = 2
MAX_TURNS = 6
HISTORY_BUFFER_LIMIT = 50


class ConfigError(Exception):
    """Raised when configuration validation fails."""


def validate_configuration():
    if OPENAI_API_KEY is None:
        raise ConfigError("Missing OPENAI_API_KEY")
    if any(point > POPULATION_SIZE for point in SELF_IMPROVE_AFTER):
        raise ConfigError(
            "SELF_IMPROVE_AFTER schedule exceeds population size"
        )

