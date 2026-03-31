import re
from pathlib import Path

import pandas as pd

LOG_LINE_RE = re.compile(
    r"^(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) - [^-]+ - (?P<level>[A-Z]+) - (?P<task_id>\S+) step=(?P<step>\d+) calls=(?P<calls>\d+)\s*(?P<text>.*)$"
)
TOOL_RE = re.compile(r"^tool:\s*(?P<tool_name>\S+)\s+received arguments:\s*(?P<tool_args>.*)$")


def parse_tool_line(line: str) -> dict:
    if not isinstance(line, str):
        return {"is_match": False, "original_line": line}
    m = TOOL_RE.match(line.strip())
    if not m:
        return {"is_match": False, "original_line": line}
    return {
        "is_match": True,
        "tool_name": m.group("tool_name"),
        "tool_args": m.group("tool_args"),
        "original_line": line,
    }


def parse_log_line(line: str) -> dict:
    if not isinstance(line, str):
        return {"is_match": False, "original_line": line}

    m = LOG_LINE_RE.match(line.strip())
    if not m:
        return {"is_match": False, "original_line": line}

    gd = m.groupdict()
    tool = parse_tool_line(gd["text"])
    return {
        "is_match": True,
        "timestamp": gd["timestamp"],
        "level": gd["level"],
        "task_id": gd["task_id"],
        "step": int(gd["step"]),
        "call": int(gd["calls"]),
        "text": gd["text"],
        "original_line": line,
        "is_tool": tool["is_match"],
        "tool_name": tool["tool_name"] if tool["is_match"] else "",
        "tool_args": tool["tool_args"] if tool["is_match"] else "",
    }


log_path = Path("evaluation/tmp_0/harmonyagent.log")
log_rows = log_path.read_text().splitlines()
log_rows = map(parse_log_line, log_rows)
log_rows = filter(lambda row: row["is_match"], log_rows)
log_rows = list(log_rows)
log_rows = pd.DataFrame(log_rows)
task_ids = log_rows["task_id"].unique().tolist()


# ingle task_id
task_id = task_ids[1]
task_logs = log_rows[log_rows["task_id"] == task_id].sort_values(by=["step", "call"])

# get tools and args
print(task_logs[task_logs["is_tool"]][["tool_name", "tool_args"]])
# print(tools)
# print(parse_tool_line('tool: repo_browser.print_tree received arguments: \'{"path": "", "depth": 3}\\n\'') == {
#     "is_match": True,
#     "tool_name": "repo_browser.print_tree",
#     "tool_args": '\'{"path": "", "depth": 3}\\n\'',
#     "original_line": 'tool: repo_browser.print_tree received arguments: \'{"path": "", "depth": 3}\\n\'',
# })
# for each step take last call
# last_call_idx = task_logs.groupby("step")["call"].idxmax()
# task_logs = task_logs.loc[last_call_idx]
# assert task_logs["step"].tolist() == list(range(task_logs["step"].shape[0])), "sequence of steps is corrupted"
