import json
from core import utils
from core import token_tracker


def test_extract_json_array_valid():
    text = "prefix [ {\"a\": 1}, {\"b\": 2} ] suffix"
    assert utils.extract_json_array(text) == [{"a": 1}, {"b": 2}]


def test_extract_json_array_invalid():
    assert utils.extract_json_array("no array") == []


def test_increment_run_number(tmp_path, monkeypatch):
    log_dir = tmp_path / "logs"
    run_file = log_dir / "run_counter.txt"
    monkeypatch.setattr(utils, "LOG_DIR", log_dir)
    monkeypatch.setattr(utils, "RUN_COUNTER_FILE", run_file)

    run1 = utils.increment_run_number()
    run2 = utils.increment_run_number()

    assert run1 == 1
    assert run2 == 2
    assert run_file.read_text() == "2"


def test_token_tracker_add_usage(tmp_path, monkeypatch):
    log_file = tmp_path / "usage.json"
    monkeypatch.setattr(token_tracker, "LOG_FILE", log_file)

    tracker = token_tracker.TokenTracker()
    tracker.set_run(1)
    tracker.add_usage(3, 4)

    data = json.loads(log_file.read_text())
    assert data == {"1": {"prompt": 3, "completion": 4}}
