from types import SimpleNamespace
from core.utils import get_usage_tokens


def test_get_usage_tokens_dict():
    usage = {"input_tokens": 1, "output_tokens": 2}
    assert get_usage_tokens(usage) == (1, 2)


def test_get_usage_tokens_dict_alt_keys():
    usage = {"prompt_tokens": 3, "completion_tokens": 4}
    assert get_usage_tokens(usage) == (3, 4)


def test_get_usage_tokens_object():
    usage = SimpleNamespace(input_tokens=5, output_tokens=6)
    assert get_usage_tokens(usage) == (5, 6)
