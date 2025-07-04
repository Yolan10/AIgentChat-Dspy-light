import importlib
from pathlib import Path


def test_env_file_loading(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("OPENAI_API_KEY=test-key\n")

    import dotenv

    real_load = dotenv.load_dotenv

    def fake_load(dotenv_path=None, override=False):
        return real_load(dotenv_path=env_file, override=override)

    monkeypatch.setattr("dotenv.load_dotenv", fake_load)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    cfg = importlib.import_module("config")
    importlib.reload(cfg)

    assert cfg.OPENAI_API_KEY == "test-key"
