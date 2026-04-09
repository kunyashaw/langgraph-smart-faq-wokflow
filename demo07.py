import os
import re
from typing import Any, Literal, NotRequired, TypedDict

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.messages import BaseMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.runnables.config import merge_configs
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

load_dotenv()


def configure_langsmith() -> str:
    langsmith_api_key = os.getenv("LANGSMITH_API_KEY") or os.getenv("SMILE_API_KEY")
    project_name = os.getenv("LANGSMITH_PROJECT", "demos-demo07")

    if langsmith_api_key:
        os.environ.setdefault("LANGSMITH_API_KEY", langsmith_api_key)
        os.environ.setdefault("LANGSMITH_TRACING", "true")
        os.environ.setdefault("LANGSMITH_PROJECT", project_name)

    return project_name


LANGSMITH_PROJECT = configure_langsmith()

llm = ChatOpenAI(
    model="MiniMax-M2.7",
    base_url="https://api.minimaxi.com/v1",
    api_key=os.getenv("MINIMAX_API_KEY"),
    temperature=0.7,
    max_tokens=1000,
    timeout=60,
    max_retries=2,
)


@tool
def multiply(a: int, b: int) -> int:
    """计算两个整数相乘"""
    return a * b


@tool
def get_weather(city: str) -> str:
    """查询城市天气"""
    weather_db = {"深圳": "晴天28C", "北京": "多云22C", "上海": "小雨18C"}
    return weather_db.get(city, f"暂无{city}数据")


@tool
def search_web(query: str) -> str:
    """在互联网上搜索信息"""
    return f"关于「{query}」的搜索结果（模拟）：来自 Wikipedia、知乎..."


toolAgent = create_agent(
    model=llm,
    tools=[multiply, get_weather, search_web],
    system_prompt=(
        "你是一个工具调用助手。"
        "遇到计算就用 multiply，遇到天气就用 get_weather，"
        "遇到搜索就用 search_web。"
        "不要输出思考过程，也不要输出 <think> 标签，只返回最终答案。"
    ),
    name="demo07_tool_agent",
)


class AgentState(TypedDict):
    number1: int
    operation: str
    number2: int
    number3: int
    operation2: str
    number4: int
    finalNumber: NotRequired[int]
    finalNumber2: NotRequired[int]
    toolAgentTriggered: NotRequired[bool]
    toolAgentResult: NotRequired[str]


def extract_agent_text(agent_result: dict[str, Any]) -> str:
    messages = agent_result.get("messages", [])
    if not messages:
        return str(agent_result)

    last_message = messages[-1]
    if isinstance(last_message, BaseMessage):
        content = last_message.content
    else:
        content = last_message

    if isinstance(content, str):
        return re.sub(r"<think>.*?</think>\s*", "", content, flags=re.DOTALL).strip()
    if isinstance(content, list):
        text_parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                text_parts.append(item)
            elif isinstance(item, dict) and item.get("text"):
                text_parts.append(str(item["text"]))
        text = "\n".join(text_parts)
        return re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL).strip()
    return re.sub(r"<think>.*?</think>\s*", "", str(content), flags=re.DOTALL).strip()


def add_node1(state: AgentState):
    return {"finalNumber": state["number1"] + state["number2"]}


def substract_node(state: AgentState):
    return {"finalNumber": state["number1"] - state["number2"]}


def add_node2(state: AgentState):
    return {"finalNumber2": state["number3"] + state["number4"]}


def substract_node2(state: AgentState, *, config: RunnableConfig):
    final_number2 = state["number3"] - state["number4"]

    agent_input = {
        "messages": [
            {
                "role": "user",
                "content": (
                    f"当前走到了 substract_node2 分支。"
                    f"请务必调用 multiply 工具计算 {state['number3']} * {state['number4']}，"
                    f"然后告诉我这个分支已执行，当前减法结果是 {final_number2}。"
                ),
            }
        ]
    }
    agent_config = merge_configs(
        config,
        {
            "run_name": "substract_node2_toolAgent",
            "tags": ["demo07", "substract_node2", "toolAgent"],
            "metadata": {
                "demo": "demo07",
                "branch": "substract_node2",
                "number3": state["number3"],
                "number4": state["number4"],
                "finalNumber2": final_number2,
                "langsmith_project": LANGSMITH_PROJECT,
            },
        },
    )
    agent_result = toolAgent.invoke(agent_input, config=agent_config)

    return {
        "finalNumber2": final_number2,
        "toolAgentTriggered": True,
        "toolAgentResult": extract_agent_text(agent_result),
    }


def route1(state: AgentState) -> Literal["add_node1", "substract_node"]:
    # 第一段路由：根据 operation 决定走 + 还是 -
    if state["operation"] == "+":
        return "add_node1"
    if state["operation"] == "-":
        return "substract_node"
    raise ValueError("operation 仅支持 + 或 -")


def route2(state: AgentState) -> Literal["add_node2", "substract_node2"]:
    # 第二段路由：根据 operation2 决定走 + 还是 -
    if state["operation2"] == "+":
        return "add_node2"
    if state["operation2"] == "-":
        return "substract_node2"
    raise ValueError("operation2 仅支持 + 或 -")


graph = StateGraph(AgentState)
graph.add_node("add_node1", add_node1, metadata={"step": "first_branch", "operation": "add"})
graph.add_node(
    "substract_node",
    substract_node,
    metadata={"step": "first_branch", "operation": "subtract"},
)
graph.add_node("add_node2", add_node2, metadata={"step": "second_branch", "operation": "add"})
graph.add_node(
    "substract_node2",
    substract_node2,
    metadata={
        "step": "second_branch",
        "operation": "subtract",
        "uses_tool_agent": True,
    },
)

# 从 START 直接做第一次条件路由
graph.add_conditional_edges(START, route1)

# 第一次分支算完后，进入第二次路由
graph.add_conditional_edges("add_node1", route2)
graph.add_conditional_edges("substract_node", route2)

# 第二次分支算完后结束
graph.add_edge("add_node2", END)
graph.add_edge("substract_node2", END)

app = graph.compile(name="demo07_calculation_graph")


def run_demo() -> dict[str, Any]:
    result = app.invoke(
        {
            "number1": 10,
            "operation": "+",
            "number2": 5,
            "number3": 20,
            "operation2": "-",
            "number4": 8,
        },
        config={
            "run_name": "demo07_entrypoint",
            "tags": ["demo07", "langgraph", "smith-demo"],
            "metadata": {
                "demo": "demo07",
                "entrypoint": "demo07.py",
                "langsmith_project": LANGSMITH_PROJECT,
            },
        },
    )
    print(result["finalNumber"])  # 15
    print(result["finalNumber2"])  # 12
    print(result.get("toolAgentTriggered", False))
    print(result.get("toolAgentResult", ""))
    return result


if __name__ == "__main__":
    run_demo()
