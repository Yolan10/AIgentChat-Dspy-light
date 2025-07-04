import dspy
import importlib.util
import sys
import types
from pathlib import Path

sys.modules.setdefault(
    "langchain_openai", types.SimpleNamespace(ChatOpenAI=lambda *a, **k: None)
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


def test_self_improve_updates_prompt(monkeypatch, tmp_path):
    improved = "new prompt"

    class DummyMIPRO:
        def __init__(self, metric):
            pass
        def compile(self, current_prompt, *, trainset):
            return improved

    monkeypatch.setattr(wizard_agent, "get_miprov2", lambda *a, **k: DummyMIPRO(*a, **k))
    monkeypatch.setattr(wizard_agent, "IMPROVED_PROMPTS_LOG", tmp_path / "imp.log")
    agent = WizardAgent("w1")
    agent.set_run(1)
    agent.current_prompt = "old"
    agent.conversation_count = 1
    agent.history_buffer.append({"turns": [{"text": "hi"}], "score": 1.0})
    agent.self_improve()
    assert agent.current_prompt == improved
    log_entry = (tmp_path / "imp.log").read_text().strip().splitlines()[0]
    assert improved in log_entry


def test_add_judge_feedback_updates_history():
    agent = WizardAgent("w1")
    agent.history_buffer.append({"turns": []})
    agent.add_judge_feedback({"overall": 0.6})
    assert agent.history_buffer[-1]["score"] == 0.6





