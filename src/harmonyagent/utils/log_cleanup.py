import json
from pathlib import Path

import yaml

eval_path = Path("evaluation/full_high")

exit_statuses_path = list(eval_path.glob("exit_statuses_*.yaml"))[0]
resolved = []
submitted = []
with open(exit_statuses_path) as file:
    results = yaml.safe_load(file)

resolved = results["instances_by_exit_status"]["Resolved"] if "Resolved" in results["instances_by_exit_status"] else []
submitted = results["instances_by_exit_status"]["Submitted"] if "Submitted" in results["instances_by_exit_status"] else []

print(f"Resolved: {len(resolved)}")
print(f"Submitted: {len(submitted)}")


other_exit_status_instances = []
other_exit_statuses = []
for exit_status in results["instances_by_exit_status"]:
    if exit_status not in {"Resolved", "Submitted"}:
        other_exit_statuses.append(exit_status)
        exit_status_lst = results["instances_by_exit_status"][exit_status]
        other_exit_status_instances.extend(exit_status_lst)
        print(f"{exit_status}: {len(exit_status_lst)}")

# cleanup preds
pred_path = eval_path / "preds.json"
predictions = json.loads(pred_path.read_text())
for i in other_exit_status_instances:
    del predictions[i]

input("we are about to rewrite preds.json, are you sure?")
with open(pred_path, "w") as f:
    json.dump(predictions, f, indent=2)
