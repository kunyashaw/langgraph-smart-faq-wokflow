"""
Simple Assistant Workflow Nodes

作用：
- 定义 graph 中每个节点的执行逻辑。
"""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig

from agents.code_agent import code_agent
from agents.prompt_agent import prompt_agent
from core.tracing import extend_run_config
from workflow.routes.simple_assistant_routes import detect_intent
from workflow.states.simple_assistant_state import SimpleAssistantState


def router_node(state: SimpleAssistantState) -> dict:
    intent, route_reason = detect_intent(state)
    return {
        "intent": intent,
        "route_reason": route_reason,
    }


def prompt_agent_node(
    state: SimpleAssistantState,
    *,
    config: RunnableConfig,
) -> dict:
    messages = state.get("messages", [])
    last_human_msg = ""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            last_human_msg = msg.content
            break

    agent_config = extend_run_config(
        config,
        run_name="prompt_agent_node",
        tags=["node", "prompt_agent_node"],
        metadata={
            "workflow_node": "prompt_agent_node",
            "intent": state.get("intent", "prompt"),
        },
    )
    try:
        result = prompt_agent.reply(last_human_msg, config=agent_config)
        return {
            "messages": [AIMessage(content=result["answer"])],
            "agent_name": result["agent_name"],
            "answer": result["answer"],
            "thinking": result["thinking"],
            "tool_route": result["tool_route"],
            "scenario": "prompt_answer",
        }
    except Exception as exc:
        return {
            "messages": [AIMessage(content=f"prompt_agent 调用失败：{exc}")],
            "agent_name": "prompt_agent",
            "answer": f"prompt_agent 调用失败：{exc}",
            "thinking": "",
            "tool_route": "general",
            "scenario": "prompt_error",
        }


def code_agent_node(
    state: SimpleAssistantState,
    *,
    config: RunnableConfig,
) -> dict:
    messages = state.get("messages", [])
    last_human_msg = ""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            last_human_msg = msg.content
            break

    agent_config = extend_run_config(
        config,
        run_name="code_agent_node",
        tags=["node", "code_agent_node"],
        metadata={
            "workflow_node": "code_agent_node",
            "intent": state.get("intent", "code"),
        },
    )

    try:
        if state.get("code") or state.get("error_message"):
            result = code_agent.debug_reply(
                task=last_human_msg,
                code=state.get("code", ""),
                error_message=state.get("error_message", ""),
                expected_behavior=state.get("expected_behavior", ""),
                language=state.get("language", "Python"),
                config=agent_config,
            )
        else:
            result = code_agent.reply(last_human_msg, config=agent_config)
    except Exception as exc:
        return {
            "messages": [AIMessage(content=f"code_agent 调用失败：{exc}")],
            "agent_name": "code_agent",
            "answer": f"code_agent 调用失败：{exc}",
            "thinking": "",
            "scenario": "code_error",
            "tool_route": "general",
        }

    return {
        "messages": [AIMessage(content=result["answer"])],
        "agent_name": result["agent_name"],
        "answer": result["answer"],
        "thinking": result["thinking"],
        "scenario": result["scenario"],
        "tool_route": state.get("tool_route", "general"),
    }
