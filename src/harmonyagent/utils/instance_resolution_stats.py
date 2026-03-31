from collections import Counter
from pathlib import Path

import yaml

resolution_files = list(Path("evaluation/").glob("debug_repo_browser_run_?/*.yaml"))
resolved = []
submitted = []
for resolution_file in resolution_files:
    with open(resolution_file) as file:
        results = yaml.safe_load(file)
        resolved.extend(results["instances_by_exit_status"]["Resolved"])
        submitted.extend(results["instances_by_exit_status"]["Submitted"])

print(Counter(resolved))
print(len(resolved) + len(submitted))
print(len(resolved) / (len(resolved) + len(submitted)))
never_resolved = {i for i in submitted if i not in resolved}
print("never resolved:")

print("\n".join(sorted([str(i) for i in never_resolved])))
