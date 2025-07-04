import dspy
import importlib.util
import sys
import types
from pathlib import Path

sys.modules.setdefault(
    "langchain_openai", types.SimpleNamespace(ChatOpenAI=lambda *a, **k: None)
)
sys.modules.setdefault(
    "langchain.schema",
    types.SimpleNamespace(SystemMessage=object, HumanMessage=object),
)

spec = importlib.util.spec_from_file_location("wizard_agent", Path("agents/wizard_agent.py"))
wizard_agent = importlib.util.module_from_spec(spec)
spec.loader.exec_module(wizard_agent)
build_dataset = wizard_agent.build_dataset
WizardAgent = wizard_agent.WizardAgent


def test_build_dataset_format():
    logs = [
        {"turns": [{"text": "a"}, {"text": "b"}], "overall": 0.3},
        {"turns": [{"text": "c"}, {"text": "d"}], "score": 0.9},
    ]
    dataset = build_dataset(logs)
    assert all(isinstance(ex, dspy.Example) for ex in dataset)
    assert dataset[0].conversation == "a b"
    assert dataset[0].score == 0.3
    assert dataset[1].conversation == "c d"
    assert dataset[1].score == 0.9


def test_self_improve_updates_prompt(monkeypatch):
    improved = "new prompt"

    class DummyMIPRO:
        def __init__(self, metric):
            pass
        def compile(self, current_prompt, *, trainset):
            return improved

    monkeypatch.setattr(wizard_agent, "MIPROv2", DummyMIPRO)
    agent = WizardAgent("w1")
    agent.current_prompt = "old"
    agent.history_buffer.append({"turns": [{"text": "hi"}], "score": 1.0})
    agent.self_improve()
    assert agent.current_prompt == improved




