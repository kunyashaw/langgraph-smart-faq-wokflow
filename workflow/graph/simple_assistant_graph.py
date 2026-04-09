"""
Simple Assistant Workflow Graph

作用：
- 组装一个尽量简单、但有条件分支的 LangGraph 工作流。
- graph 会先判断走 prompt_agent 还是 code_agent。
- 其中 prompt_agent 内部还会继续决定是否调用 math_tools / search_tools。
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage
from langgraph.graph import END, START, StateGraph

from core.tracing import build_run_config, configure_langsmith
from workflow.nodes.simple_assistant_nodes import (
    code_agent_node,
    prompt_agent_node,
    router_node,
)
from workflow.routes.simple_assistant_routes import route_after_router
from workflow.states.simple_assistant_state import SimpleAssistantState


DEFAULT_WORKFLOW_PROJECT = "demos-simple-assistant"


def build_simple_assistant_graph():
    graph = StateGraph(SimpleAssistantState)

    graph.add_node(
        "router_node",
        router_node,
        metadata={
            "step": "routing",
        },
    )
    graph.add_node(
        "prompt_agent_node",
        prompt_agent_node,
        metadata={
            "step": "agent",
            "agent": "prompt_agent",
        },
    )
    graph.add_node(
        "code_agent_node",
        code_agent_node,
        metadata={
            "step": "agent",
            "agent": "code_agent",
        },
    )

    graph.add_edge(START, "router_node")
    graph.add_conditional_edges("router_node", route_after_router)
    graph.add_edge("prompt_agent_node", END)
    graph.add_edge("code_agent_node", END)

    return graph.compile(name="simple_assistant_graph")


app = build_simple_assistant_graph()


def run_simple_assistant(
    messages: list,
    *,
    code: str = "",
    error_message: str = "",
    expected_behavior: str = "",
    language: str = "Python",
    project_name: str = DEFAULT_WORKFLOW_PROJECT,
) -> dict[str, Any]:
    langsmith_project = configure_langsmith(project_name)

    state: SimpleAssistantState = {
        "messages": messages,
        "code": code,
        "error_message": error_message,
        "expected_behavior": expected_behavior,
        "language": language,
    }

    run_config = build_run_config(
        run_name="simple_assistant_run",
        tags=["workflow", "simple_assistant"],
        metadata={
            "workflow": "simple_assistant",
            "langsmith_project": langsmith_project,
            "has_code_context": bool(code or error_message),
        },
    )
    return app.invoke(state, config=run_config)
