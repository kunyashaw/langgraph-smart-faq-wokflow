# LangGraph 工作流项目从零到运行

手把手记录一个 LangGraph + MiniMax 的智能问答工作流项目，支持多轮对话、路由分发、Agent 调用和 LangSmith 追踪。

---

## 项目背景

想做一个智能问答入口：根据用户输入的内容，自动判断该走通用问答还是代码处理。重要的是要支持多轮对话——用户可以追问、补充代码、继续深入。

最终方案：LangGraph 搭路由 + Agent 组合的工作流，接 MiniMax API，配合 LangSmith 做调用追踪，State 中用 `add_messages` reducer 自动累积对话历史。

---

## 环境准备

### 1. 初始化项目

```bash
mkdir demos && cd demos
uv init
```

### 2. 安装依赖

```bash
uv add langchain-openai langgraph python-dotenv
uv add langchain-core langchain-community
```

`pyproject.toml` 最终长这样：

```toml
[project]
name = "demos"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = [
    "langchain-openai>=1.1.12",
    "langgraph>=1.0.0",
    "python-dotenv>=1.2.2",
]
```

### 3. 配置 API Key

根目录新建 `.env`：

```
MINIMAX_API_KEY = "你的MiniMax密钥"
SMITH_API_KEY = "你的LangSmith密钥"
```

---

## 目录结构

```
demos/
├── agents/              # Agent 定义
│   ├── __init__.py
│   ├── code_agent.py    # 代码处理 Agent
│   └── prompt_agent.py   # 通用问答 Agent
├── core/                # 核心工具
│   ├── __init__.py
│   ├── llm.py           # LLM 初始化和调用封装
│   └── tracing.py       # LangSmith 配置
├── tools/               # 工具集
│   ├── __init__.py
│   ├── math_tools.py    # 数学计算工具
│   └── search_tools.py  # 搜索工具
├── workflow/            # 工作流
│   ├── nodes/           # 节点实现
│   │   └── simple_assistant_nodes.py
│   ├── graph/            # 图定义
│   │   └── simple_assistant_graph.py
│   ├── routes/          # 路由逻辑
│   │   └── simple_assistant_routes.py
│   ├── states/          # 状态定义
│   │   └── simple_assistant_state.py
│   └── simple_assistant/
│       └── __init__.py   # facade
└── run_workflow.py      # 入口脚本
```

---

## 核心模块

### LLM 封装

`core/llm.py` 统一管理模型调用：

```python
import os
from langchain_openai import ChatOpenAI

def build_llm() -> ChatOpenAI:
    api_key = os.getenv("MINIMAX_API_KEY")
    return ChatOpenAI(
        model="MiniMax-M2.7",
        base_url="https://api.minimaxi.com/v1",
        api_key=api_key,
        temperature=0.7,
        max_tokens=1000,
        timeout=60,
    )
```

### LangSmith 配置

`core/tracing.py` 处理链路追踪：

```python
import os
from dotenv import load_dotenv

def configure_langsmith(project_name: str) -> str:
    load_dotenv()
    api_key = os.getenv("LANGSMITH_API_KEY") or os.getenv("SMITH_API_KEY")
    if api_key:
        os.environ["LANGSMITH_API_KEY"] = api_key
        os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGSMITH_PROJECT"] = project_name
    return project_name

def build_run_config(run_name: str, tags=None, metadata=None):
    return {
        "run_name": run_name,
        "tags": list(tags or []),
        "metadata": dict(metadata or {}),
    }

def extend_run_config(config, run_name=None, tags=None, metadata=None):
    from langchain_core.runnables.config import merge_configs
    extra = {
        "tags": list(tags or []),
        "metadata": dict(metadata or {}),
    }
    if run_name:
        extra["run_name"] = run_name
    return merge_configs(config, extra)
```

---

## Agent 实现

### 通用问答 Agent

`agents/prompt_agent.py`，支持自动识别计算和搜索意图并调用对应工具。这里不展开完整代码，核心是 `PromptAgent` 类提供 `reply()` 方法，输入问题字符串返回 dict。

### 代码处理 Agent

`agents/code_agent.py`，专注于代码排错和修复建议。`CodeAgent` 类提供 `reply()` 和 `debug_reply()` 方法。

---

## 工作流设计

### 状态定义（支持多轮对话）

`workflow/states/simple_assistant_state.py`：

```python
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
```

关键点：`messages: Annotated[list, add_messages]` 让新消息自动追加到列表，而不是覆盖。

### 路由逻辑

`workflow/routes/simple_assistant_routes.py`：

```python
from typing import Literal
from langchain_core.messages import BaseMessage
from workflow.states.simple_assistant_state import SimpleAssistantState

CODE_HINT_KEYWORDS = (
    "python", "java", "javascript", "代码", "函数", "类",
    "报错", "异常", "错误", "修复", "debug", "bug",
    "traceback", "stack trace", "review",
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

def route_after_router(state: SimpleAssistantState) -> Literal["prompt_agent_node", "code_agent_node"]:
    if state.get("intent") == "code":
        return "code_agent_node"
    return "prompt_agent_node"
```

### 节点实现

`workflow/nodes/simple_assistant_nodes.py`：

```python
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from agents.code_agent import code_agent
from agents.prompt_agent import prompt_agent
from core.tracing import extend_run_config
from workflow.routes.simple_assistant_routes import detect_intent
from workflow.states.simple_assistant_state import SimpleAssistantState

def router_node(state: SimpleAssistantState) -> dict:
    intent, route_reason = detect_intent(state)
    return {"intent": intent, "route_reason": route_reason}

def prompt_agent_node(state: SimpleAssistantState, *, config: RunnableConfig) -> dict:
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

def code_agent_node(state: SimpleAssistantState, *, config: RunnableConfig) -> dict:
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
```

### 图构建

`workflow/graph/simple_assistant_graph.py`：

```python
from typing import Any
from langgraph.graph import END, START, StateGraph
from workflow.nodes.simple_assistant_nodes import code_agent_node, prompt_agent_node, router_node
from workflow.routes.simple_assistant_routes import route_after_router
from workflow.states.simple_assistant_state import SimpleAssistantState
from core.tracing import build_run_config, configure_langsmith

DEFAULT_WORKFLOW_PROJECT = "demos-simple-assistant"

def build_simple_assistant_graph():
    graph = StateGraph(SimpleAssistantState)

    graph.add_node("router_node", router_node, metadata={"step": "routing"})
    graph.add_node("prompt_agent_node", prompt_agent_node, metadata={"step": "agent", "agent": "prompt_agent"})
    graph.add_node("code_agent_node", code_agent_node, metadata={"step": "agent", "agent": "code_agent"})

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
```

---

## 入口脚本

`run_workflow.py`，支持单轮和多轮对话：

```python
import argparse
from langchain_core.messages import HumanMessage
from workflow.graph import app, run_simple_assistant

def print_graph():
    print("\n" + "=" * 60)
    print("Workflow Graph:")
    print("=" * 60)
    graph = app.get_graph()
    print(graph.draw_mermaid())
    print("=" * 60 + "\n")

def main():
    parser = argparse.ArgumentParser(description="Simple Assistant Workflow")
    parser.add_argument("--show-graph", action="store_true", help="Show workflow graph")
    parser.add_argument("--user-input", type=str, help="User input for the workflow")
    parser.add_argument("--code", type=str, default="", help="Code snippet (optional)")
    parser.add_argument("--error", type=str, default="", help="Error message (optional)")
    parser.add_argument("--expected", type=str, default="", help="Expected behavior (optional)")
    parser.add_argument("--language", type=str, default="Python", help="Programming language")

    args = parser.parse_args()

    if args.show_graph:
        print_graph()
        return

    messages = []

    if args.user_input:
        messages.append(HumanMessage(content=args.user_input))
        result = run_simple_assistant(
            messages=messages,
            code=args.code,
            error_message=args.error,
            expected_behavior=args.expected,
            language=args.language,
        )
        messages = result.get("messages", [])
        print("\n" + "=" * 60)
        print("Answer:")
        print("=" * 60)
        if messages and hasattr(messages[-1], "content"):
            print(messages[-1].content)
        print("=" * 60)
    else:
        print("Simple Assistant Workflow CLI (多轮对话模式)")
        print("=" * 60)
        print("输入 exit 或 quit 退出对话")
        print()

        while True:
            user_input = input("你: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ("exit", "quit", "q"):
                print("再见!")
                break

            messages.append(HumanMessage(content=user_input))
            print("思考中...\n")

            result = run_simple_assistant(
                messages=messages,
                code="",
                error_message="",
                expected_behavior="",
                language="Python",
            )
            messages = result.get("messages", [])

            if messages and hasattr(messages[-1], "content"):
                print(f"\n助手: {messages[-1].content}\n")

if __name__ == "__main__":
    main()
```

---

## 运行方式

### 查看工作流图

```bash
uv run python run_workflow.py --show-graph
```

会输出 Mermaid 格式的图结构，复制到 [mermaid.live](https://mermaid.live) 可以直接渲染。

### 单轮对话

```bash
uv run python run_workflow.py --user-input "你好，介绍一下自己"
```

### 多轮对话

```bash
uv run python run_workflow.py
```

进入多轮模式后，可以连续对话：

```
Simple Assistant Workflow CLI (多轮对话模式)
============================================================
输入 exit 或 quit 退出对话

你: 帮我解释一下什么是闭包
助手: 闭包是...

你: 能举个Python的例子吗
助手: 当然，这里是一个例子...

你: exit
再见!
```

---

## 多轮对话实现原理

关键在于 State 中的 `messages: Annotated[list, add_messages]`：

- `add_messages` 是一个 reducer 函数，告诉 LangGraph 如何合并同一字段的多次更新
- 每次新增消息时，LangGraph 自动把新消息 append 到列表，而不是覆盖整个列表
- 这样 `app.invoke(state)` 会保留之前的所有消息

节点的返回值也需要返回 `messages` 字段：

```python
return {
    "messages": [AIMessage(content=result["answer"])],
    "agent_name": ...,
    ...
}
```

LangGraph 会自动把这条 AI 消息合并到历史中。

---

## 项目结构最终版

```
demos/
├── agents/
│   ├── __init__.py
│   ├── code_agent.py
│   └── prompt_agent.py
├── core/
│   ├── __init__.py
│   ├── llm.py
│   └── tracing.py
├── tools/
│   ├── __init__.py
│   ├── math_tools.py
│   └── search_tools.py
├── workflow/
│   ├── graph/
│   │   ├── __init__.py
│   │   └── simple_assistant_graph.py
│   ├── nodes/
│   │   ├── __init__.py
│   │   └── simple_assistant_nodes.py
│   ├── routes/
│   │   ├── __init__.py
│   │   └── simple_assistant_routes.py
│   ├── states/
│   │   ├── __init__.py
│   │   └── simple_assistant_state.py
│   └── simple_assistant/
│       └── __init__.py
├── .env
├── pyproject.toml
└── run_workflow.py
```

---

*项目代码本地调试通过，依赖版本见 pyproject.toml*
