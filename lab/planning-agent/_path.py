"""Make `lab/planning-agent` importable as `lab.planning_agent`.

Python cannot import a directory with a hyphen, and the directory name is
fixed by the lab playbook. Registering an alias in sys.modules is the smallest
fix that keeps both the directory name and ordinary import statements.

This lives in its own module rather than in tests/conftest.py because pytest is
not the only entry point: run.py is a documented command in the README, and a
conftest is never loaded for it.
"""

import importlib.util
import sys
from pathlib import Path

TOPIC = Path(__file__).resolve().parent


def install_alias() -> None:
    if "lab.planning_agent" in sys.modules:
        return
    lab = importlib.util.module_from_spec(
        importlib.util.spec_from_loader("lab", loader=None, is_package=True))
    lab.__path__ = [str(TOPIC.parent)]
    sys.modules.setdefault("lab", lab)

    spec = importlib.util.spec_from_file_location(
        "lab.planning_agent", TOPIC / "__init__.py",
        submodule_search_locations=[str(TOPIC)])
    module = importlib.util.module_from_spec(spec)
    sys.modules["lab.planning_agent"] = module
    spec.loader.exec_module(module)


install_alias()
