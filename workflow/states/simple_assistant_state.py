"""
Simple Assistant Workflow State

作用：
- 定义这个 workflow 在各个节点之间传递的数据结构。
- 使用 messages 列表支持多轮对话历史。
"""

from __future__ import annotations

from typing import Annotated, Literal, TypedDict

from langgraph.graph import add_messages


class SimpleAssistantState(TypedDict):
    messages: Annotated[list, add_messages]

    code: str
    error_message: str
    expected_behavior: str
    language: str

    intent: Literal["prompt", "code"]
    route_reason: str

    agent_name: str
    scenario: str
    tool_route: str
