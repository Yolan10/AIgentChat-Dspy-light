import importlib.util
import sys
import types
import json
from pathlib import Path

import dspy
from dspy.teleprompt.mipro_optimizer_v2 import MIPROv2

# Avoid importing real OpenAI components
sys.modules.setdefault(
    "langchain_openai", types.SimpleNamespace(ChatOpenAI=lambda *a, **k: None)
)

spec = importlib.util.spec_from_file_location("wizard_agent", Path("agents/wizard_agent.py"))
wizard_agent = importlib.util.module_from_spec(spec)
spec.loader.exec_module(wizard_agent)
WizardAgent = wizard_agent.WizardAgent


def test_miprov2_self_improve(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setattr(wizard_agent, "IMPROVED_PROMPTS_LOG", tmp_path / "imp.log")

    improved = "optimized"
    called = {"flag": False}

    def fake_compile(self, program, *, trainset):
        called["flag"] = True
        program.signature.__doc__ = improved
        return program

    monkeypatch.setattr(MIPROv2, "compile", fake_compile)

    agent = WizardAgent("A")
    agent.set_run(1)
    agent.current_prompt = "old"
    agent.conversation_count = 1
    agent.history_buffer.append({"turns": [{"speaker": "u", "text": "hi"}], "score": 0.2})

    agent.self_improve()

    assert called["flag"]
    assert agent.current_prompt == improved
    entry = json.loads((tmp_path / "imp.log").read_text().strip())
    assert entry["prompt"] == improved
