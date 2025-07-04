import os
from typing import List

# Basic configuration loaded from environment variables
POPULATION_SIZE = int(os.environ.get("POPULATION_SIZE", 36))
SELF_IMPROVE_AFTER: List[int] = [
    int(x) for x in os.environ.get("SELF_IMPROVE_AFTER", "1,10,35").split(",") if x
]
LLM_MODEL = os.environ.get("LLM_MODEL", "gpt-4.1-nano")
PARALLEL_CONVERSATIONS = os.environ.get("PARALLEL_CONVERSATIONS", "true").lower() == "true"
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")

# DSPy optimizer settings
DSPY_MIPRO_MINIBATCH_SIZE = int(os.environ.get("DSPY_MIPRO_MINIBATCH_SIZE", 4))
DSPY_BOOTSTRAP_MINIBATCH_SIZE = int(os.environ.get("DSPY_BOOTSTRAP_MINIBATCH_SIZE", 2))
MAX_TURNS = int(os.environ.get("MAX_TURNS", 6))
HISTORY_BUFFER_LIMIT = int(os.environ.get("HISTORY_BUFFER_LIMIT", 50))


class ConfigError(Exception):
    """Raised when configuration validation fails."""


def validate_configuration():
    if OPENAI_API_KEY is None:
        raise ConfigError("Missing OPENAI_API_KEY")
    if any(point > POPULATION_SIZE for point in SELF_IMPROVE_AFTER):
        raise ConfigError(
            "SELF_IMPROVE_AFTER schedule exceeds population size"
        )

