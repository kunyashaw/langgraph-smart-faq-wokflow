"""
Code Agent 使用说明

作用：
- 这是一个面向编程场景的 agent，当前主打“代码排错与修复”。
- 适合输入任务描述、代码片段、报错信息，让大模型输出问题定位、修复思路、修改代码和验证步骤。

常用用法：
- 通用编程问答：
    from agents.code_agent import code_agent
    answer = code_agent.invoke("Python 里 dataclass 是做什么的？")

- 代码排错：
    result = code_agent.debug_code(
        task="帮我定位为什么报错",
        code="def average(nums): return sum(nums) / len(nums)",
        error_message="ZeroDivisionError: division by zero",
        expected_behavior="空列表时不要直接报错",
    )
    print(result.answer)

- 在 workflow node 中使用：
    from agents.code_agent import code_agent

    def code_debug_node(state):
        result = code_agent.debug_reply(
            task=state["task"],
            code=state.get("code", ""),
            error_message=state.get("error_message", ""),
            expected_behavior=state.get("expected_behavior", ""),
            language=state.get("language", "Python"),
        )
        return {
            "code_answer": result["answer"],
            "code_thinking": result["thinking"],
        }

适用场景：
- 代码排错
- 修复建议
- 代码 review
- 作为 graph 中的编程分析节点
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterator

from dotenv import load_dotenv
from langchain_core.runnables import RunnableConfig

from core.llm import DEFAULT_SYSTEM_PROMPT, UtilsLLMService
from core.tracing import extend_run_config


@dataclass(slots=True)
class CodeAgentResponse:
    """统一封装 code agent 的输出，方便在 graph/state 中透传。"""

    answer: str
    thinking: str

    def to_dict(self) -> dict[str, str]:
        """转成普通 dict，便于直接写入 LangGraph state。"""
        return asdict(self)


class CodeAgent:
    """面向编程场景的通用 agent，当前主打“代码排错与修复”场景。"""

    def __init__(
        self,
        *,
        name: str = "code_agent",
        specialty: str = "代码排错与修复助手",
        extra_instructions: str = "",
    ) -> None:
        self.name = name
        self.specialty = specialty
        self.extra_instructions = extra_instructions.strip()
        self._service: UtilsLLMService | None = None

    def _build_system_prompt(self) -> str:
        """构造适合编程任务的 system prompt。"""
        prompt_parts = [
            DEFAULT_SYSTEM_PROMPT,
            f"你的角色设定是：{self.specialty}。",
            "你擅长阅读代码、理解报错、定位根因，并给出最小可行修复方案。",
            "当用户给出代码和错误信息时，请优先输出：问题定位、修复思路、修复代码、验证步骤。",
            "如果信息不足，请基于现有信息给出最可能原因，并明确写出你的假设。",
            "回答尽量使用清晰的小标题，涉及代码时优先给出可直接复制的代码块。",
        ]
        if self.extra_instructions:
            prompt_parts.append(self.extra_instructions)
        return "\n".join(prompt_parts)

    def _ensure_service(self) -> UtilsLLMService:
        """懒加载底层 LLM service，避免 import 模块时就初始化模型。"""
        if self._service is not None and self._service.ready:
            return self._service

        load_dotenv()

        service = UtilsLLMService()
        service.set_system_prompt(self._build_system_prompt())
        service.initialize()
        if not service.ready:
            raise RuntimeError(f"{self.name} initialize failed: {service.init_error}")

        self._service = service
        return service

    def _run_with_meta(
        self,
        question: str,
        *,
        config: RunnableConfig | None = None,
        scenario: str = "general_code_help",
    ) -> CodeAgentResponse:
        """统一执行入口，返回答案和隐藏 thinking。"""
        service = self._ensure_service()
        agent_config = extend_run_config(
            config,
            run_name=f"{self.name}.{scenario}",
            tags=["agent", self.name, scenario],
            metadata={
                "agent_name": self.name,
                "agent_type": "code",
                "scenario": scenario,
            },
        )
        answer, thinking = service.invoke_with_meta(question, config=agent_config)
        return CodeAgentResponse(answer=answer, thinking=thinking)

    def invoke(
        self,
        question: str,
        config: RunnableConfig | None = None,
    ) -> str:
        """通用调用：直接返回最终回复。"""
        return self._run_with_meta(question, config=config).answer

    def invoke_with_meta(
        self,
        question: str,
        config: RunnableConfig | None = None,
    ) -> CodeAgentResponse:
        """通用调用：返回最终回复和隐藏 thinking。"""
        return self._run_with_meta(question, config=config)

    def reply(
        self,
        question: str,
        config: RunnableConfig | None = None,
    ) -> dict[str, str]:
        """适合在 graph node 中使用的标准返回结构。"""
        result = self._run_with_meta(
            question,
            config=config,
            scenario="general_code_help",
        )
        return {
            "agent_name": self.name,
            "scenario": "general_code_help",
            "answer": result.answer,
            "thinking": result.thinking,
        }

    def debug_code(
        self,
        *,
        task: str,
        code: str = "",
        error_message: str = "",
        language: str = "Python",
        expected_behavior: str = "",
        config: RunnableConfig | None = None,
    ) -> CodeAgentResponse:
        """代码排错场景：让模型定位问题并给出修复建议。"""
        sections = [
            "你现在要处理一个编程排错任务。",
            f"编程语言：{language}",
            f"任务描述：{task.strip()}",
        ]
        if expected_behavior.strip():
            sections.append(f"期望行为：{expected_behavior.strip()}")
        if error_message.strip():
            sections.append(f"报错信息：\n```text\n{error_message.strip()}\n```")
        if code.strip():
            sections.append(f"相关代码：\n```{language.lower()}\n{code.strip()}\n```")
        sections.append(
            "\n".join(
                [
                    "请按下面结构回答：",
                    "1. 问题定位：说明最可能的根因。",
                    "2. 修复方案：给出最小改动建议。",
                    "3. 修复后的代码：给出完整代码或关键修改片段。",
                    "4. 验证步骤：告诉我怎么确认修复成功。",
                ]
            )
        )
        question = "\n\n".join(sections)
        return self._run_with_meta(
            question,
            config=config,
            scenario="code_debug",
        )

    def debug_reply(
        self,
        *,
        task: str,
        code: str = "",
        error_message: str = "",
        language: str = "Python",
        expected_behavior: str = "",
        config: RunnableConfig | None = None,
    ) -> dict[str, str]:
        """适合在 workflow node 中直接调用的代码排错入口。"""
        result = self.debug_code(
            task=task,
            code=code,
            error_message=error_message,
            language=language,
            expected_behavior=expected_behavior,
            config=config,
        )
        return {
            "agent_name": self.name,
            "scenario": "code_debug",
            "answer": result.answer,
            "thinking": result.thinking,
        }

    def review_code(
        self,
        *,
        code: str,
        focus: str = "正确性、可读性和可维护性",
        language: str = "Python",
        config: RunnableConfig | None = None,
    ) -> CodeAgentResponse:
        """代码 review 场景：给出问题、风险和优化建议。"""
        question = "\n\n".join(
            [
                "请帮我做一次代码 review。",
                f"编程语言：{language}",
                f"关注重点：{focus}",
                f"代码如下：\n```{language.lower()}\n{code.strip()}\n```",
                "\n".join(
                    [
                        "请按下面结构回答：",
                        "1. 主要问题：优先说 bug、风险和行为错误。",
                        "2. 优化建议：给出更合理的实现方式。",
                        "3. 示例代码：如果合适，请给出修改后的关键片段。",
                    ]
                ),
            ]
        )
        return self._run_with_meta(
            question,
            config=config,
            scenario="code_review",
        )

    def stream(
        self,
        question: str,
        config: RunnableConfig | None = None,
    ) -> Iterator[str]:
        """流式返回最终可展示文本。"""
        service = self._ensure_service()
        agent_config = extend_run_config(
            config,
            run_name=f"{self.name}.stream",
            tags=["agent", self.name, "stream"],
            metadata={
                "agent_name": self.name,
                "agent_type": "code",
            },
        )
        yield from service.stream(question, config=agent_config)

    def stream_with_thinking(
        self,
        question: str,
        config: RunnableConfig | None = None,
    ) -> Iterator[tuple[str, str]]:
        """流式返回 answer/think 两类片段。"""
        service = self._ensure_service()
        agent_config = extend_run_config(
            config,
            run_name=f"{self.name}.stream_with_thinking",
            tags=["agent", self.name, "stream"],
            metadata={
                "agent_name": self.name,
                "agent_type": "code",
            },
        )
        yield from service.stream_with_thinking(question, config=agent_config)


def build_code_agent(
    *,
    name: str = "code_agent",
    specialty: str = "代码排错与修复助手",
    extra_instructions: str = "",
) -> CodeAgent:
    """工厂函数：按不同编程场景创建新的 code agent。"""
    return CodeAgent(
        name=name,
        specialty=specialty,
        extra_instructions=extra_instructions,
    )


# 默认实例，适合在 workflow node 中直接 import 后调用。
code_agent = build_code_agent()


def main() -> None:
    """本地调试入口。"""
    result = code_agent.debug_code(
        task="帮我定位这段代码为什么会抛出除零错误",
        code=(
            "def average(nums):\n"
            "    total = sum(nums)\n"
            "    return total / len(nums)\n\n"
            "print(average([]))"
        ),
        error_message="ZeroDivisionError: division by zero",
        expected_behavior="当列表为空时，不要直接报错，而是给出合理处理。",
    )
    print("answer:", result.answer)
    if result.thinking:
        print("thinking:", result.thinking)


if __name__ == "__main__":
    main()
