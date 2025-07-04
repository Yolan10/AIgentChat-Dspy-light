# DSPy integration helpers
from collections import deque
from typing import Iterable, Any

from dspy.datasets.dataset import Dataset
from dspy.teleprompt.mipro_optimizer_v2 import MIPROv2

import config


def build_dataset(history: deque) -> Dataset:
    """Return a DSPy Dataset of (conversation, score) records.

    Each entry in ``history`` should either be a tuple ``(log, score)`` or
    a mapping with ``"turns"`` and ``"score"`` keys. ``log`` is expected to have
    a ``"turns"`` list of ``{"speaker": str, "text": str}`` dictionaries and
    ``score`` may be a number or a mapping with an ``"overall"`` field.
    """
    records = []
    for item in history:
        if isinstance(item, tuple) and len(item) == 2:
            conv_log, score = item
        elif isinstance(item, dict):
            conv_log = item
            score = item.get("score")
        else:
            continue

        turns = conv_log.get("turns", [])
        convo = "\n".join(f"{t.get('speaker')}: {t.get('text')}" for t in turns)
        if isinstance(score, dict):
            score_val = score.get("overall")
        else:
            score_val = score
        records.append({"conversation": convo, "score": score_val})

    ds = Dataset(train_size=len(records), dev_size=0, test_size=0, input_keys=["conversation"])
    ds._train = records
    ds._dev = []
    ds._test = []
    return ds


def get_miprov2(metric, prompt_model=None, task_model=None) -> MIPROv2:
    """Return a configured MIPROv2 optimizer instance."""
    return MIPROv2(
        metric=metric,
        prompt_model=prompt_model,
        task_model=task_model,
        max_bootstrapped_demos=config.DSPY_BOOTSTRAP_MINIBATCH_SIZE,
        max_labeled_demos=config.DSPY_MIPRO_MINIBATCH_SIZE,
    )

