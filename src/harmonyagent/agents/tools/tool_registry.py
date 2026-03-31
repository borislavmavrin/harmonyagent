from harmonyagent.agents.tools.container_exec import *
from harmonyagent.agents.tools.repo_browser import *
from harmonyagent.domain_model import Tool

TOOL_REGISTRY: dict[str, Tool] = {
    "container.exec": ContainerExec,
    "repo_browser.print_tree": PrintTree,
    "repo_browser.search": Search,
    "repo_browser.open_file": OpenFile,
    "repo_browser.list_dir": RepoBrowserListDir,
    "repo_browser.apply_patch": ApplyPatch,
}
