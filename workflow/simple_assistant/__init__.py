"""Simple assistant workflow package.

This is a facade module that re-exports from the new structure.
Legacy import: from workflow.simple_assistant import app
Preferred import: from workflow.graph import app
"""

from workflow.graph.simple_assistant_graph import app, build_simple_assistant_graph, run_simple_assistant

__all__ = ["app", "build_simple_assistant_graph", "run_simple_assistant"]
