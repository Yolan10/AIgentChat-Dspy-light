from collections import deque
import config
from core.dspy_utils import build_dataset, get_miprov2
from dspy.teleprompt.mipro_optimizer_v2 import MIPROv2


def test_build_dataset_basic():
    history = deque([
        ({"turns": [{"speaker": "wiz", "text": "hi"}, {"speaker": "pop", "text": "ok"}]}, {"overall": 0.7}),
        ({"turns": [{"speaker": "wiz", "text": "bye"}]}, {"overall": 0.5}),
    ])
    ds = build_dataset(history)
    train = ds.train
    assert len(train) == 2
    assert train[0].conversation == "wiz: hi\npop: ok"
    assert train[0].score == 0.7


def dummy_metric(*args, **kwargs):
    return 0.0


def test_get_miprov2_configured():
    opt = get_miprov2(dummy_metric)
    assert isinstance(opt, MIPROv2)
    assert opt.max_labeled_demos == config.DSPY_MIPRO_MINIBATCH_SIZE
    assert opt.max_bootstrapped_demos == config.DSPY_BOOTSTRAP_MINIBATCH_SIZE

