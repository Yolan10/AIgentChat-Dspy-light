# DSPy integration helpers
from collections import deque
from typing import Iterable, Any

from dspy.datasets.dataset import Dataset
from dspy.teleprompt.mipro_optimizer_v2 import MIPROv2
import dspy

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
        convo = "\n".join(
            f"{t.get('speaker')}: {t.get('text')}" for t in turns
        )
        if isinstance(score, dict):
            score_val = score.get("overall")
        else:
            score_val = score
        records.append(dspy.Example(conversation=convo, score=score_val))

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


def apply_dspy_optimizer(
    prompt: str,
    history: Iterable[Any],
    metric=None,
    prompt_model=None,
    task_model=None,
    get_opt=None,
) -> str:
    """Return ``prompt`` optimized on the provided ``history`` using DSPy.

    ``history`` should be an iterable of conversation logs accepted by
    :func:`build_dataset`. ``metric`` defaults to using the ``score`` attribute
    of each :class:`dspy.Example`.
    """
    ds = build_dataset(deque(history))
    trainset = ds.train
    if len(trainset) < 2:
        trainset = trainset * 2
    if metric is None:
        metric = lambda ex: ex.score
    if get_opt is None:
        get_opt = get_miprov2
    opt_kwargs = {}
    if prompt_model is not None:
        opt_kwargs["prompt_model"] = prompt_model
    if task_model is not None:
        opt_kwargs["task_model"] = task_model
    optimizer = get_opt(metric, **opt_kwargs)
    try:
        new_prompt = optimizer.compile(prompt, trainset=trainset)
    except AttributeError as e:
        if "predictors" in str(e):
            return prompt
        raise
    return new_prompt if isinstance(new_prompt, str) else prompt

