import types
from agents.judge_agent import EnhancedJudgeAgent
from core import token_tracker

class DummyResp:
    def __init__(self, content="0.6"):
        self.content = content
        self.usage_metadata = {"prompt_tokens": 1, "completion_tokens": 1}

def test_judge_parses_score(monkeypatch, tmp_path):
    agent = EnhancedJudgeAgent("J")
    monkeypatch.setattr(agent, "llm", types.SimpleNamespace(invoke=lambda msgs: DummyResp("0.75")))
    monkeypatch.setattr(token_tracker, "LOG_FILE", tmp_path / "usage.json")
    token_tracker.tracker.set_run(1)
    log = {"pop_agent_id": "A", "turns": [{"speaker": "pop", "text": "hi"}]}
    result = agent.judge(log)
    assert result["overall"] == 0.75
    assert result["success"] is True

def test_judge_handles_bad_output(monkeypatch, tmp_path):
    agent = EnhancedJudgeAgent("J")
    monkeypatch.setattr(agent, "llm", types.SimpleNamespace(invoke=lambda msgs: DummyResp("n/a")))
    monkeypatch.setattr(token_tracker, "LOG_FILE", tmp_path / "usage.json")
    token_tracker.tracker.set_run(1)
    log = {"pop_agent_id": "A", "turns": []}
    result = agent.judge(log)
    assert result["overall"] == 0.0
    assert result["success"] is False
