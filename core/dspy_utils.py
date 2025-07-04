"""DSPy helper utilities."""

from __future__ import annotations

from typing import Callable, List, Any

from dspy.teleprompt import BootstrapFewShot, MIPROv2

import config


def apply_dspy_optimizer(program: Any, metric: Callable, trainset: List) -> Any:
    """Return a DSPy program optimized with BootstrapFewShot and MIPROv2.

    Parameters
    ----------
    program : Any
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
    program = bootstrap.compile(program, trainset=trainset)

    mipro = MIPROv2(
        metric=metric,
        max_bootstrapped_demos=config.DSPY_MIPRO_MINIBATCH_SIZE,
    )
    program = mipro.compile(program, trainset=trainset)
    return program
