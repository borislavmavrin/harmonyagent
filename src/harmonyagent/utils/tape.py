from pathlib import Path

FIELDS = ["prompt_text", "response_text", "finish_reason"]


def record_step(tape_path: Path, step: int, prompt_text: str, response: dict):
    response_text = response["choices"][0]["text"]
    finish_reason = response["choices"][0]["finish_reason"]

    if not tape_path.is_dir():
        tape_path.mkdir(exist_ok=True, parents=True)

    for field, content in zip(FIELDS, [prompt_text, response_text, finish_reason]):
        debug_file = tape_path / f"{step}_{field}.txt"
        with debug_file.open(mode="w") as f:
            f.write(content)


def read_tape(tape_path: Path) -> list[dict]:
    if not tape_path.is_dir():
        return []

    steps = sorted({int(f.name.split("_", 1)[0]) for f in tape_path.glob("*_*.txt")})

    result = []
    for step in steps:
        entry = {}
        for field in FIELDS:
            file = tape_path / f"{step}_{field}.txt"
            entry[field] = file.read_text() if file.exists() else None
        llm_format = {
            "prompt_text": entry["prompt_text"],
            "response": {"choices": [{"text": entry["response_text"], "finish_reason": entry["finish_reason"]}]},
        }
        result.append(llm_format)
    return result
