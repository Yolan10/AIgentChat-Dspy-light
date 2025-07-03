def test_openai_api(monkeypatch):
    import os
    import openai
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        import pytest
        pytest.skip("No API key configured")
    client = openai.OpenAI(api_key=key)
    try:
        client.chat.completions.create(model="gpt-3.5-turbo", messages=[{"role": "user", "content": "ping"}])
    except Exception as e:
        import pytest
        pytest.fail(f"OpenAI API call failed: {e}")
