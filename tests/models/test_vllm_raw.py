from pathlib import Path
from unittest.mock import patch

from harmonyagent.models.vllm_raw import VllmRawModel, VllmRawModelConfig
from harmonyagent.utils.tape import read_tape


def _make_response(text="response", finish_reason="stop"):
    return {"choices": [{"text": text, "finish_reason": finish_reason}]}


@patch.object(VllmRawModel, "_query")
def test_query_returns_response(mock_query):
    response = _make_response("hello")
    mock_query.return_value = response
    model = VllmRawModel(VllmRawModelConfig())

    result = model.query("prompt", stop_token_ids=[1, 2], max_tokens=1)

    assert result == response
    mock_query.assert_called_once_with("prompt", [1, 2], 1)


@patch.object(VllmRawModel, "_query")
def test_query_increments_n_calls(mock_query):
    mock_query.return_value = _make_response()
    model = VllmRawModel(VllmRawModelConfig())

    assert model.n_calls == 0
    model.query("p1", stop_token_ids=[], max_tokens=1)
    assert model.n_calls == 1
    model.query("p2", stop_token_ids=[], max_tokens=1)
    assert model.n_calls == 2


@patch.object(VllmRawModel, "_query")
def test_query_records_tape(mock_query, tmp_path):
    mock_query.return_value = _make_response("resp0", "stop")
    tape_path = str(tmp_path / "tape")
    model = VllmRawModel(VllmRawModelConfig(tape_path=tape_path))

    model.query("prompt0", stop_token_ids=[], max_tokens=1)

    tape = read_tape(Path(tape_path))
    assert len(tape) == 1
    assert tape[0] == {"prompt_text": "prompt0", "response": {"choices": [{"text": "resp0", "finish_reason": "stop"}]}}


@patch.object(VllmRawModel, "_query")
def test_query_no_tape_when_path_empty(mock_query, tmp_path):
    mock_query.return_value = _make_response()
    model = VllmRawModel(VllmRawModelConfig())

    model.query("prompt", stop_token_ids=[], max_tokens=1)

    # No tape directory should be created
    assert not (tmp_path / "tape").exists()


@patch.object(VllmRawModel, "_query")
def test_query_records_multiple_steps(mock_query, tmp_path):
    tape_path = str(tmp_path / "tape")
    model = VllmRawModel(VllmRawModelConfig(tape_path=tape_path))

    mock_query.return_value = _make_response("r0", "stop")
    model.query("p0", stop_token_ids=[], max_tokens=1)
    mock_query.return_value = _make_response("r1", "length")
    model.query("p1", stop_token_ids=[], max_tokens=1)

    tape = read_tape(Path(tape_path))
    assert len(tape) == 2
    assert tape[0]["prompt_text"] == "p0"
    assert tape[1]["prompt_text"] == "p1"
    assert tape[1]["response"]["choices"][0]["finish_reason"] == "length"


def test_get_template_vars():
    model = VllmRawModel(VllmRawModelConfig())
    vars = model.get_template_vars()
    assert vars["model_name"] == "gpt-oss"
    assert vars["n_model_calls"] == 0
    assert vars["model_cost"] == 0.0
