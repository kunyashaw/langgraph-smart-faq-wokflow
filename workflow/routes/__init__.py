"""Workflow routes package."""

from workflow.routes.simple_assistant_routes import detect_intent, route_after_router

__all__ = ["detect_intent", "route_after_router"]
