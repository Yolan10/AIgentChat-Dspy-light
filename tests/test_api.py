import pytest


class _MockCompletions:
    def create(self, model: str, messages: list):
        return {"choices": [{"message": {"content": "pong"}}]}


class _MockChat:
    def __init__(self):
        self.completions = _MockCompletions()


class _MockClient:
    def __init__(self, api_key: str | None = None):
        self.chat = _MockChat()


@pytest.fixture
def mock_openai_client(monkeypatch):
    """Return a mocked OpenAI client."""
    import openai

    monkeypatch.setattr(openai, "OpenAI", _MockClient)
    return _MockClient()


def test_openai_api(mock_openai_client):
    response = mock_openai_client.chat.completions.create(
        model="gpt-3.5-turbo", messages=[{"role": "user", "content": "ping"}]
    )
    assert response["choices"][0]["message"]["content"] == "pong"
