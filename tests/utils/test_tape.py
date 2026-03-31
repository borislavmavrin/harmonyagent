from harmonyagent.utils.tape import read_tape, record_step


def _make_response(text, finish_reason="stop"):
    return {"choices": [{"text": text, "finish_reason": finish_reason}]}


def test_read_tape_empty_dir(tmp_path):
    tape_path = tmp_path / "tape"
    tape_path.mkdir()
    assert read_tape(tape_path) == []


def test_read_tape_nonexistent_dir(tmp_path):
    assert read_tape(tmp_path / "nonexistent") == []


def test_read_tape_single_step(tmp_path):
    tape_path = tmp_path / "tape"
    record_step(tape_path, 0, "prompt0", _make_response("resp0", "stop"))

    result = read_tape(tape_path)
    assert result == [{"prompt_text": "prompt0", "response": {"choices": [{"text": "resp0", "finish_reason": "stop"}]}}]


def test_read_tape_multiple_steps(tmp_path):
    tape_path = tmp_path / "tape"
    record_step(tape_path, 0, "p0", _make_response("r0", "stop"))
    record_step(tape_path, 1, "p1", _make_response("r1", "length"))
    record_step(tape_path, 2, "p2", _make_response("r2", "stop"))

    result = read_tape(tape_path)
    assert len(result) == 3
    assert result[0] == {"prompt_text": "p0", "response": {"choices": [{"text": "r0", "finish_reason": "stop"}]}}
    assert result[1] == {"prompt_text": "p1", "response": {"choices": [{"text": "r1", "finish_reason": "length"}]}}
    assert result[2] == {"prompt_text": "p2", "response": {"choices": [{"text": "r2", "finish_reason": "stop"}]}}


def test_read_tape_ordering(tmp_path):
    tape_path = tmp_path / "tape"
    # Write steps out of order
    record_step(tape_path, 2, "p2", _make_response("r2"))
    record_step(tape_path, 0, "p0", _make_response("r0"))
    record_step(tape_path, 1, "p1", _make_response("r1"))

    result = read_tape(tape_path)
    assert [r["prompt_text"] for r in result] == ["p0", "p1", "p2"]
