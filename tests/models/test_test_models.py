import pytest

from harmonyagent.models.test_models import PromptTextMismatch, TapeModel, TapeModelConfig


def test_tape_model_query():
    tape = [
        {"prompt_text": "hello", "response": {"choices": [{"text": "world", "finish_reason": "stop"}]}},
    ]
    model = TapeModel(config=TapeModelConfig(tape=tape))
    result = model.query(prompt_text="hello", stop_token_ids=[], max_tokens=1000)
    assert result == {"choices": [{"text": "world", "finish_reason": "stop"}]}
    assert model.n_calls == 1


def test_tape_model_prompt_mismatch():
    tape = [
        {"prompt_text": "hello", "response": {"choices": [{"text": "world", "finish_reason": "stop"}]}},
    ]
    model = TapeModel(config=TapeModelConfig(tape=tape))
    with pytest.raises(PromptTextMismatch):
        model.query(prompt_text="wrong", stop_token_ids=[], max_tokens=1000)


def test_tape_model_get_template_vars():
    model = TapeModel(config=TapeModelConfig(tape=[]))
    vars = model.get_template_vars()
    assert vars["model_name"] == "tape"
    assert vars["n_model_calls"] == 0
    assert vars["model_cost"] == 0.0
