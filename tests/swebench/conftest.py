import json
from pathlib import Path

import pytest

DATA_DIR = Path(__file__).parent / "data"

BASH_ONLY_20 = [
    "django__django-11119",
    "django__django-12125",
    "django__django-13195",
    "django__django-13512",
    "django__django-14122",
    "django__django-14349",
    "django__django-14725",
    "django__django-14771",
    "matplotlib__matplotlib-21568",
    "pydata__xarray-3151",
    "pylint-dev__pylint-6528",
    "pytest-dev__pytest-5631",
    "pytest-dev__pytest-5840",
    "pytest-dev__pytest-7432",
    "scikit-learn__scikit-learn-14087",
    "sympy__sympy-12489",
    "sympy__sympy-14976",
    "sympy__sympy-18199",
    "sympy__sympy-19040",
    "sympy__sympy-19346",
]


@pytest.fixture(scope="session")
def swebench_instances():
    """Load bash-only-20 SWE-bench instances from local JSON data."""
    instances = json.loads((DATA_DIR / "bash_only_20_instances.json").read_text())
    return {inst["instance_id"]: inst for inst in instances}
