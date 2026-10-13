"""Install the lab.planning_agent alias before any test imports it."""

import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "_planning_agent_path", Path(__file__).resolve().parent.parent / "_path.py")
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
