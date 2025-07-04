"""DSPy helper utilities."""

from __future__ import annotations

from typing import Callable, List, Any

from dspy.teleprompt import BootstrapFewShot, MIPROv2

import config


def apply_dspy_optimizer(student: Any, metric: Callable, trainset: List) -> Any:
    """Return a DSPy program optimized with BootstrapFewShot and MIPROv2.

    Parameters
    ----------
    student : Any
        DSPy module or program to optimize.
    metric : Callable
        Scoring function used by the optimizers.
    trainset : List
        Example training data.

    Returns
    -------
    Any
        The optimized DSPy program.
    """

    bootstrap = BootstrapFewShot(
        metric=metric,
        max_labeled_demos=config.DSPY_BOOTSTRAP_MINIBATCH_SIZE,
    )
    student = bootstrap.compile(student, trainset=trainset)

    mipro = MIPROv2(
        metric=metric,
        max_bootstrapped_demos=config.DSPY_MIPRO_MINIBATCH_SIZE,
    )
    student = mipro.compile(student, trainset=trainset)
    return student
