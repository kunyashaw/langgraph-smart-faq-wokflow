"""
Prompt Agent 使用说明

作用：
- 这是一个通用的提示词型 agent，适合处理解释、问答、总结、概念说明这类自然语言任务。
- 当问题被判断为“计算”时，会优先调用 `tools/math_tools.py` 中的数学工具。
- 当问题被判断为“实时查询/搜索”时，会优先调用 `tools/search_tools.py` 中的搜索工具。

常用用法：
- 直接拿最终答案：
    from agents.prompt_agent import prompt_agent
    answer = prompt_agent.invoke("帮我算一下 (3 + 5) * 2")

- 拿答案和 thinking：
    result = prompt_agent.invoke_with_meta("今天可以搜索到哪些 LangGraph 公开信息？")
    print(result.answer)
    print(result.thinking)

- 在 workflow node 中使用：
    from agents.prompt_agent import prompt_agent

    def prompt_node(state):
        result = prompt_agent.reply(state["question"])
        return {
            "answer": result["answer"],
            "thinking": result["thinking"],
            "tool_route": result["tool_route"],
        }

适用场景：
- 普通问答
- 概念解释
- 文本总结
- 计算问题自动走数学工具
- 搜索/实时问题自动走搜索工具
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Iterator

from dotenv import load_dotenv
from langchain_core.runnables import RunnableConfig

from core.llm import DEFAULT_SYSTEM_PROMPT, UtilsLLMService
from core.tracing import extend_run_config
from tools.math_tools import (
    add_numbers,
    calculate_expression,
    divide_numbers,
    multiply_numbers,
    subtract_numbers,
)
from tools.search_tools import search_project_files, search_web


@dataclass(slots=True)
class PromptAgentResponse:
    """统一封装 prompt agent 的输出，方便在 graph/state 中传递。"""

    answer: str
    thinking: str

    def to_dict(self) -> dict[str, str]:
        """转成普通 dict，便于写入 LangGraph state。"""
        return asdict(self)


class PromptAgent:
    """可复用的提示词型 agent，适合在 workflow node 中直接调用。"""

    def __init__(
        self,
        *,
        name: str = "prompt_agent",
        character: str = "LangChain 助手",
        extra_instructions: str = "",
    ) -> None:
        self.name = name
        self.character = character
        self.extra_instructions = extra_instructions.strip()
        self._service: UtilsLLMService | None = None

    def _build_system_prompt(self) -> str:
        """把基础系统提示词、角色设定和额外约束拼成最终 system prompt。"""
        prompt_parts = [
            DEFAULT_SYSTEM_PROMPT,
            f"你的角色设定是：{self.character}。",
            "请先认真理解用户问题，再给出准确、自然、可直接使用的最终回复。",
            "如果前面已经给出了工具结果，请严格基于工具结果回答，不要忽略工具结果，也不要自行编造新的搜索内容或计算结果。",
        ]
        if self.extra_instructions:
            prompt_parts.append(self.extra_instructions)
        return "\n".join(prompt_parts)

    def _ensure_service(self) -> UtilsLLMService:
        """懒加载底层 LLM service，避免 import 模块时就触发模型初始化。"""
        if self._service is not None and self._service.ready:
            return self._service

        load_dotenv()

        service = UtilsLLMService()
        # 每个 agent 在初始化时把自己的 system prompt 注入到底层服务。
        service.set_system_prompt(self._build_system_prompt())
        service.initialize()
        if not service.ready:
            raise RuntimeError(f"{self.name} initialize failed: {service.init_error}")

        self._service = service
        return service

    def _format_number(self, value: float) -> str:
        """把数值格式化为更自然的字符串，避免无意义的 .0。"""
        if float(value).is_integer():
            return str(int(value))
        return str(value)

    def _extract_expression(self, question: str) -> str:
        """从问题中尽量提取一个可直接用于表达式计算的片段。"""
        expression_match = re.search(r"[\d\.\s\+\-\*\/\(\)%]{3,}", question)
        if not expression_match:
            return ""
        expression = expression_match.group(0).strip()
        if any(operator in expression for operator in "+-*/%"):
            return expression
        return ""

    def _is_math_query(self, question: str) -> bool:
        """用轻量规则判断当前问题是否更像计算请求。"""
        clean_question = question.strip().lower()
        if self._extract_expression(clean_question):
            return True

        numbers = re.findall(r"-?\d+(?:\.\d+)?", clean_question)
        if len(numbers) < 2:
            return False

        math_keywords = (
            "计算",
            "算一下",
            "等于",
            "加",
            "减",
            "乘",
            "除",
            "相加",
            "相减",
            "相乘",
            "相除",
            "plus",
            "minus",
            "multiply",
            "divide",
        )
        return any(keyword in clean_question for keyword in math_keywords)

    def _is_search_query(self, question: str) -> bool:
        """用轻量规则判断当前问题是否更像搜索/实时查询请求。"""
        clean_question = question.strip().lower()
        search_keywords = (
            "搜索",
            "查询",
            "查一下",
            "检索",
            "最新",
            "实时",
            "现在",
            "今天",
            "今日",
            "当前",
            "recent",
            "latest",
            "today",
            "news",
            "price",
            "weather",
            "股价",
            "汇率",
            "天气",
        )
        project_search_keywords = (
            "项目",
            "代码",
            "代码库",
            "仓库",
            "repo",
            "文件",
            "函数",
            "类",
            "workflow",
            "promptagent",
            "codeagent",
        )
        return any(keyword in clean_question for keyword in search_keywords + project_search_keywords)

    def _is_project_search_query(self, question: str) -> bool:
        """判断搜索是否更偏向项目内文件，而不是公开网络信息。"""
        clean_question = question.strip().lower()
        project_search_keywords = (
            "项目",
            "代码",
            "代码库",
            "仓库",
            "repo",
            "文件",
            "函数",
            "类",
            "workflow",
            "promptagent",
            "codeagent",
        )
        return any(keyword in clean_question for keyword in project_search_keywords)

    def _extract_search_query(self, question: str) -> str:
        """尽量从自然语言问题里提取更适合搜索的核心关键词。"""
        english_tokens = re.findall(r"[A-Za-z][A-Za-z0-9_.-]*", question)
        if english_tokens:
            # 优先保留像 LangGraph / PromptAgent 这样的核心英文标识。
            return " ".join(dict.fromkeys(english_tokens))

        cleaned = question
        stop_phrases = (
            "帮我",
            "请帮我",
            "请",
            "可以",
            "能不能",
            "搜索",
            "查询",
            "查一下",
            "检索",
            "实时",
            "最新",
            "今天",
            "今日",
            "现在",
            "当前",
            "公开信息",
            "相关信息",
            "有哪些",
            "是什么",
        )
        for phrase in stop_phrases:
            cleaned = cleaned.replace(phrase, " ")

        cleaned = re.sub(r"[？?，,。.!！:：]+", " ", cleaned)
        cleaned = " ".join(cleaned.split())
        return cleaned or question.strip()

    def _run_math_tool(self, question: str) -> str:
        """优先用 math_tools 处理明确的计算请求。"""
        expression = self._extract_expression(question)
        if expression:
            result = calculate_expression.invoke({"expression": expression})
            return f"已调用数学工具 calculate_expression，计算结果：{expression} = {self._format_number(result)}"

        numbers = [float(item) for item in re.findall(r"-?\d+(?:\.\d+)?", question)]
        if len(numbers) < 2:
            raise ValueError("未能从问题中提取出足够的数字来执行计算")

        first, second = numbers[0], numbers[1]
        clean_question = question.lower()

        if "相乘" in clean_question or "乘" in clean_question or "multiply" in clean_question:
            result = multiply_numbers.invoke({"a": first, "b": second})
            return (
                "已调用数学工具 multiply_numbers，"
                f"计算结果：{self._format_number(first)} * {self._format_number(second)} = {self._format_number(result)}"
            )
        if "相除" in clean_question or "除" in clean_question or "divide" in clean_question:
            result = divide_numbers.invoke({"a": first, "b": second})
            return (
                "已调用数学工具 divide_numbers，"
                f"计算结果：{self._format_number(first)} / {self._format_number(second)} = {self._format_number(result)}"
            )
        if "相减" in clean_question or "减" in clean_question or "minus" in clean_question:
            result = subtract_numbers.invoke({"a": first, "b": second})
            return (
                "已调用数学工具 subtract_numbers，"
                f"计算结果：{self._format_number(first)} - {self._format_number(second)} = {self._format_number(result)}"
            )

        result = add_numbers.invoke({"a": first, "b": second})
        return (
            "已调用数学工具 add_numbers，"
            f"计算结果：{self._format_number(first)} + {self._format_number(second)} = {self._format_number(result)}"
        )

    def _run_search_tool(self, question: str) -> str:
        """优先用 search_tools 处理实时查询或搜索请求。"""
        query = self._extract_search_query(question)
        if self._is_project_search_query(question):
            result = search_project_files.invoke(
                {
                    "query": query,
                    "search_path": ".",
                    "max_results": 5,
                }
            )
            return f"已调用搜索工具 search_project_files（query={query}），搜索结果如下：\n{result}"

        result = search_web.invoke({"query": query, "max_results": 3})
        return f"已调用搜索工具 search_web（query={query}），搜索结果如下：\n{result}"

    def _prepare_question(self, question: str) -> tuple[str, str, str]:
        """根据问题类型决定是否先调工具，再把工具结果交给大模型组织回复。"""
        use_math = self._is_math_query(question)
        use_search = self._is_search_query(question)

        if not use_math and not use_search:
            return "general", question, ""

        tool_outputs: list[str] = []
        route_parts: list[str] = []

        if use_search:
            route_parts.append("search")
            try:
                tool_outputs.append(self._run_search_tool(question))
            except Exception as exc:
                tool_outputs.append(f"搜索工具调用失败：{exc}")

        if use_math:
            route_parts.append("math")
            try:
                tool_outputs.append(self._run_math_tool(question))
            except Exception as exc:
                tool_outputs.append(f"数学工具调用失败：{exc}")

        route = "+".join(route_parts)
        prepared_question = "\n\n".join(
            [
                f"用户原始问题：{question}",
                f"工具路由：{route}",
                "工具执行结果：",
                "\n\n".join(tool_outputs),
                (
                    "请基于上面的工具结果，直接给出对用户有帮助的最终回答。"
                    "如果搜索结果不足以覆盖“实时/最新”诉求，请明确说明当前仅检索到了这些结果，"
                    "不要编造未搜索到的信息。"
                ),
            ]
        )
        fallback_answer = "\n\n".join(tool_outputs)
        return route, prepared_question, fallback_answer

    def _invoke_internal(
        self,
        question: str,
        config: RunnableConfig | None = None,
    ) -> tuple[str, PromptAgentResponse]:
        """统一执行入口：先判断是否要调工具，再让大模型生成最终回复。"""
        service = self._ensure_service()
        route, prepared_question, fallback_answer = self._prepare_question(question)
        agent_config = extend_run_config(
            config,
            run_name=f"{self.name}.invoke",
            tags=["agent", self.name, route],
            metadata={
                "agent_name": self.name,
                "agent_type": "prompt",
                "tool_route": route,
            },
        )
        try:
            answer, thinking = service.invoke_with_meta(prepared_question, config=agent_config)
            return route, PromptAgentResponse(answer=answer, thinking=thinking)
        except Exception:
            if route == "general":
                raise
            return route, PromptAgentResponse(
                answer=(
                    "工具已经执行完成，但当前大模型整理回复失败。"
                    "下面先返回工具结果，供你继续使用：\n\n"
                    f"{fallback_answer}"
                ),
                thinking="",
            )

    def invoke(
        self,
        question: str,
        config: RunnableConfig | None = None,
    ) -> str:
        """返回清洗后的最终回复文本。"""
        _, response = self._invoke_internal(question, config=config)
        return response.answer

    def invoke_with_meta(
        self,
        question: str,
        config: RunnableConfig | None = None,
    ) -> PromptAgentResponse:
        """返回最终答案，以及模型隐藏的 thinking 文本。"""
        _, response = self._invoke_internal(question, config=config)
        return response

    def reply(
        self,
        question: str,
        config: RunnableConfig | None = None,
    ) -> dict[str, str]:
        """适合在 graph node 中使用的标准返回结构。"""
        route, result = self._invoke_internal(question, config=config)
        return {
            "agent_name": self.name,
            "answer": result.answer,
            "thinking": result.thinking,
            "tool_route": route,
        }

    def stream(
        self,
        question: str,
        config: RunnableConfig | None = None,
    ) -> Iterator[str]:
        """流式返回最终可展示文本。"""
        service = self._ensure_service()
        route, prepared_question, fallback_answer = self._prepare_question(question)
        agent_config = extend_run_config(
            config,
            run_name=f"{self.name}.stream",
            tags=["agent", self.name, route, "stream"],
            metadata={
                "agent_name": self.name,
                "agent_type": "prompt",
                "tool_route": route,
            },
        )
        if route != "general":
            try:
                yield from service.stream(prepared_question, config=agent_config)
                return
            except Exception:
                yield (
                    "工具已经执行完成，但当前大模型整理回复失败。"
                    "下面先返回工具结果：\n\n"
                    f"{fallback_answer}"
                )
                return
        yield from service.stream(prepared_question, config=agent_config)

    def stream_with_thinking(
        self,
        question: str,
        config: RunnableConfig | None = None,
    ) -> Iterator[tuple[str, str]]:
        """流式返回 answer/think 两类片段。"""
        service = self._ensure_service()
        route, prepared_question, fallback_answer = self._prepare_question(question)
        agent_config = extend_run_config(
            config,
            run_name=f"{self.name}.stream_with_thinking",
            tags=["agent", self.name, route, "stream"],
            metadata={
                "agent_name": self.name,
                "agent_type": "prompt",
                "tool_route": route,
            },
        )
        if route != "general":
            try:
                yield from service.stream_with_thinking(prepared_question, config=agent_config)
                return
            except Exception:
                yield (
                    "answer",
                    "工具已经执行完成，但当前大模型整理回复失败。"
                    "下面先返回工具结果：\n\n"
                    f"{fallback_answer}",
                )
                return
        yield from service.stream_with_thinking(prepared_question, config=agent_config)


def build_prompt_agent(
    *,
    name: str = "prompt_agent",
    character: str = "LangChain 助手",
    extra_instructions: str = "",
) -> PromptAgent:
    """工厂函数：按不同角色配置创建新的 prompt agent。"""
    return PromptAgent(
        name=name,
        character=character,
        extra_instructions=extra_instructions,
    )


# 默认实例，适合在 workflow node 中直接 import 后调用。
prompt_agent = build_prompt_agent()


def main() -> None:
    """本地调试入口。"""
    for question in [
        "帮我算一下 (3 + 5) * 2",
        "今天可以搜索到哪些 LangGraph 的公开信息？",
        "什么是 RAG？",
    ]:
        result = prompt_agent.reply(question)
        print("question:", question)
        print("route:", result["tool_route"])
        print("answer:", result["answer"])
        if result["thinking"]:
            print("thinking:", result["thinking"])
        print("-" * 40)


if __name__ == "__main__":
    main()
