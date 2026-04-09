"""
Simple Assistant Workflow Routes

作用：
- 负责根据输入内容做轻量级条件判断。
- 这里只做"应该走哪个 agent"的判断，不做真正的模型调用。
"""

from __future__ import annotations

from typing import Literal

from langchain_core.messages import BaseMessage

from workflow.states.simple_assistant_state import SimpleAssistantState


CODE_HINT_KEYWORDS = (
    "python",
    "java",
    "javascript",
    "typescript",
    "golang",
    "代码",
    "函数",
    "类",
    "报错",
    "异常",
    "错误",
    "修复",
    "debug",
    "bug",
    "traceback",
    "stack trace",
    "review",
)


def _get_latest_user_message(messages: list[BaseMessage]) -> str:
    for msg in reversed(messages):
        if hasattr(msg, "type") and msg.type == "human":
            return msg.content
    return ""


def detect_intent(state: SimpleAssistantState) -> tuple[Literal["prompt", "code"], str]:
    if state.get("code") or state.get("error_message"):
        return "code", "state中包含code或error_message"

    messages = state.get("messages", [])
    user_input = _get_latest_user_message(messages).strip().lower()

    if any(keyword in user_input for keyword in CODE_HINT_KEYWORDS):
        return "code", "命中了编程/报错关键词"

    return "prompt", "默认走通用问答agent"


def route_after_router(
    state: SimpleAssistantState,
) -> Literal["prompt_agent_node", "code_agent_node"]:
    if state.get("intent") == "code":
        return "code_agent_node"
    return "prompt_agent_node"
