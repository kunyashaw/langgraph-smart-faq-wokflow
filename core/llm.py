# ========== Utils: LLM Service 封装 ==========
# 文件：utils_llm_service.py
#
# 这个文件只负责“模型和链路调用”相关逻辑：
# 1. 创建 LLM 实例；
# 2. 构建 LCEL 链；
# 3. 提供 initialize / invoke / stream / shutdown 方法；
# 4. 不依赖 FastAPI，便于单测和复用。

import os
import re
from typing import Iterator

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI

# 这些标签通常是“思考过程”或“中间推理”，对前端展示不是最终答案。
HIDDEN_TAG_NAMES = ("think", "reasoning", "analysis")
HIDDEN_SECTION_PATTERN = re.compile(
    r"<\s*(think|reasoning|analysis)\b[^>]*>.*?<\s*/\s*\1\s*>",
    flags=re.IGNORECASE | re.DOTALL,
)
HIDDEN_CONTENT_PATTERN = re.compile(
    r"<\s*(think|reasoning|analysis)\b[^>]*>(.*?)<\s*/\s*\1\s*>",
    flags=re.IGNORECASE | re.DOTALL,
)
DEFAULT_SYSTEM_PROMPT = (
    "你是一个清晰、耐心的技术助手。回答要准确、结构清楚、尽量简洁。"
    "只输出最终答案，不要输出<think>、<reasoning>、<analysis>等中间思考标签。"
)


def build_llm() -> ChatOpenAI:
    """创建聊天模型实例。"""
    # 从环境变量读取 MiniMax 的 API Key。
    api_key = os.getenv("MINIMAX_API_KEY")
    # 如果没配置 key，直接抛异常，避免后续请求才报错。
    if not api_key:
        # 抛出运行时错误，提示调用方先设置 MINIMAX_API_KEY。
        raise RuntimeError("MINIMAX_API_KEY is not set")

    # 返回一个 ChatOpenAI 客户端实例（这里通过兼容接口连接 MiniMax）。
    return ChatOpenAI(
        # 指定要使用的模型名称。
        model="MiniMax-M2.7",
        # 指定服务端 API 基地址（MiniMax 兼容 OpenAI 协议入口）。
        base_url="https://api.minimaxi.com/v1",
        # 传入鉴权密钥，用于请求签名认证。
        api_key=api_key,
        # 控制随机性，值越高越发散，值越低越稳定。
        temperature=0.7,
        # 限制单次回复的最大 token 数，防止返回过长。
        max_tokens=1000,
        # 设置单次请求超时时间（秒），避免长时间阻塞。
        timeout=60,
        # 失败时自动重试次数，提高临时网络抖动下的稳定性。
        max_retries=2,
    )


def build_chain(system_prompt: str = DEFAULT_SYSTEM_PROMPT):
    """构建基础 LCEL 链路：Prompt -> LLM -> StrOutputParser。"""
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                system_prompt,
            ),
            ("human", "{question}"),
        ]
    )
    return prompt | build_llm() | StrOutputParser()


class UtilsLLMService:
    """封装链路生命周期和调用细节。"""

    def __init__(self) -> None:
        self.chain = None
        self.init_error: str | None = None
        self.system_prompt: str = DEFAULT_SYSTEM_PROMPT

    @property
    def ready(self) -> bool:
        """链路是否已经可用。"""
        return self.chain is not None

    def initialize(self) -> None:
        """启动阶段初始化链路。"""
        try:
            self.chain = build_chain(self.system_prompt)
            self.init_error = None
        except Exception as exc:
            self.chain = None
            self.init_error = str(exc)

    def shutdown(self) -> None:
        """关闭阶段清理资源。"""
        self.chain = None

    def set_system_prompt(self, system_prompt: str) -> None:
        """更新系统提示词，并重建链路。"""
        self.system_prompt = str(system_prompt or "").strip() or DEFAULT_SYSTEM_PROMPT
        if self.ready:
            self.chain = build_chain(self.system_prompt)

    def set_prompt(self, system_prompt: str) -> None:
        """set_system_prompt 的别名，方便外部语义化调用。"""
        self.set_system_prompt(system_prompt)

    def _require_chain(self):
        """拿到可用链路，不可用时抛出运行时异常。"""
        if self.chain is None:
            detail = self.init_error or "chain is not initialized"
            raise RuntimeError(f"service unavailable: {detail}")
        return self.chain

    def _clean_text_output(self, raw_output) -> str:
        """
        清洗模型输出，只保留可展示的有效答案文本。
        - 先转字符串；
        - 去掉 <think>/<reasoning>/<analysis> 包裹内容；
        - 再做首尾空白清理。
        """
        text = str(raw_output or "")
        # 反复替换，兼容多个隐藏段落。
        previous = None
        while previous != text:
            previous = text
            text = HIDDEN_SECTION_PATTERN.sub("", text)

        # 兼容“不完整标签”场景：有开标签但没闭合时，丢弃开标签后的内容。
        text = re.sub(
            r"<\s*(think|reasoning|analysis)\b[^>]*>.*$",
            "",
            text,
            flags=re.IGNORECASE | re.DOTALL,
        )
        # 清理孤立闭标签。
        text = re.sub(r"</\s*(think|reasoning|analysis)\s*>", "", text, flags=re.IGNORECASE)
        return text.strip()

    def _clean_answer_fragment(self, fragment: str) -> str:
        """
        清理流式答案片段（轻量版）：
        - 去掉残留隐藏标签；
        - 保留原始空格和换行，不做 strip，避免拼接后词语粘连。
        """
        text = str(fragment or "")
        text = re.sub(r"</?\s*(think|reasoning|analysis)\b[^>]*>", "", text, flags=re.IGNORECASE)
        return text

    def _clean_stream_chunks(self, chunks: Iterator[str]) -> Iterator[str]:
        """
        增量清洗流式输出。
        目标：即使标签被拆到不同 chunk，也尽量不把隐藏内容透传给前端。
        """
        open_pattern = re.compile(
            r"<\s*(think|reasoning|analysis)\b[^>]*>",
            flags=re.IGNORECASE,
        )
        close_pattern = re.compile(
            r"</\s*(think|reasoning|analysis)\s*>",
            flags=re.IGNORECASE,
        )
        # 为了处理“标签被切分到多个 chunk”的场景，保留一段尾部缓冲。
        hold_back = 64
        in_hidden = False
        buffer = ""

        for raw_chunk in chunks:
            piece = str(raw_chunk or "")
            if not piece:
                continue
            buffer += piece

            while True:
                if not in_hidden:
                    match_open = open_pattern.search(buffer)
                    if not match_open:
                        # 暂留尾巴，避免把不完整标签片段提前输出。
                        if len(buffer) <= hold_back:
                            break
                        emit_text = buffer[:-hold_back]
                        buffer = buffer[-hold_back:]
                        cleaned = self._clean_text_output(emit_text)
                        if cleaned:
                            yield cleaned
                        break

                    # 遇到开标签：先输出标签前内容，再进入隐藏区。
                    visible = buffer[: match_open.start()]
                    cleaned_visible = self._clean_text_output(visible)
                    if cleaned_visible:
                        yield cleaned_visible
                    buffer = buffer[match_open.end() :]
                    in_hidden = True
                else:
                    match_close = close_pattern.search(buffer)
                    if not match_close:
                        # 隐藏区里没有闭标签时，丢弃大部分内容，仅保留尾巴用于拼接闭标签。
                        if len(buffer) > hold_back:
                            buffer = buffer[-hold_back:]
                        break
                    # 找到闭标签后，移除隐藏内容，继续正常输出。
                    buffer = buffer[match_close.end() :]
                    in_hidden = False

        if not in_hidden and buffer:
            cleaned_tail = self._clean_text_output(buffer)
            if cleaned_tail:
                yield cleaned_tail

    def _extract_hidden_text(self, raw_output) -> str:
        """
        提取模型输出里的隐藏思考内容（如 think/reasoning/analysis 标签内文本）。
        如果没有相关标签，返回空字符串。
        """
        text = str(raw_output or "")
        segments: list[str] = []
        for match in HIDDEN_CONTENT_PATTERN.finditer(text):
            content = match.group(2).strip()
            if content:
                segments.append(content)
        return "\n".join(segments)

    def _split_stream_visible_hidden(self, chunks: Iterator[str]) -> Iterator[tuple[str, str]]:
        """
        把流式 chunk 拆成两类事件：
        - ("answer", 可展示答案文本)
        - ("think", 标签内思考文本)
        这样上层可以一边展示答案，一边给用户“正在思考”的状态提示。
        """
        open_pattern = re.compile(  # 匹配 think/reasoning/analysis 开始标签。
            r"<\s*(think|reasoning|analysis)\b[^>]*>",  # 支持标签内带属性。
            flags=re.IGNORECASE,  # 标签名大小写不敏感。
        )  # 开始标签正则定义结束。
        hold_back = 64  # 尾部缓冲长度，避免把被切断的标签片段提前输出。
        in_hidden = False  # 标记当前是否处于隐藏思考片段内部。
        current_tag = ""  # 记录当前打开的标签名，用于匹配对应闭标签。
        buffer = ""  # 流式拼接缓冲区，承接跨 chunk 的文本。

        for raw_chunk in chunks:  # 逐个处理底层模型返回的 chunk。
            piece = str(raw_chunk or "")  # 把 chunk 统一转成字符串（兼容 None）。
            if not piece:  # 空 chunk 直接跳过，避免无意义处理。
                continue  # 进入下一轮 chunk。
            buffer += piece  # 把当前 chunk 追加到缓冲区。

            while True:  # 对当前缓冲区循环消费，直到不能再安全解析为止。
                if not in_hidden:  # 当前不在隐藏区，优先找开标签。
                    open_match = open_pattern.search(buffer)  # 查找最近的隐藏开标签。
                    if not open_match:  # 没找到开标签，说明当前内容都属于可见答案。
                        if len(buffer) <= hold_back:  # 缓冲长度太短时先保留，防止尾部是半截标签。
                            break  # 暂停消费，等待下一 chunk 拼接后再判断。
                        emit_text = buffer[:-hold_back]  # 安全输出前半部分，把尾巴留在缓冲区。
                        buffer = buffer[-hold_back:]  # 把尾部保留用于下轮拼接解析。
                        if emit_text:  # 只有有内容时才产出事件。
                            yield ("answer", emit_text)  # 产出可见答案片段。
                        break  # 当前缓冲区已处理到安全边界，等待下一 chunk。

                    visible = buffer[: open_match.start()]  # 截取开标签前的可见内容。
                    if visible:  # 可见内容非空时才输出。
                        yield ("answer", visible)  # 产出答案事件。
                    current_tag = open_match.group(1)  # 记录开标签名（think/reasoning/analysis）。
                    buffer = buffer[open_match.end() :]  # 从缓冲区移除已处理部分（含开标签）。
                    in_hidden = True  # 状态切换为“已进入隐藏区”。
                else:  # 当前位于隐藏区，需要找对应闭标签。
                    close_pattern = re.compile(  # 为当前标签名动态构造闭标签正则。
                        rf"</\s*{re.escape(current_tag)}\s*>",  # 只匹配与当前开标签同名的闭标签。
                        flags=re.IGNORECASE,  # 闭标签大小写不敏感。
                    )  # 闭标签正则定义结束。
                    close_match = close_pattern.search(buffer)  # 在隐藏区中查找闭标签。
                    if not close_match:  # 若还没等到闭标签，说明隐藏内容可能被拆包了。
                        if len(buffer) <= hold_back:  # 缓冲区太短时先全部保留，继续等待拼接。
                            break  # 暂停消费当前缓冲区。
                        think_text = buffer[:-hold_back]  # 先输出安全部分隐藏文本，尾巴继续保留。
                        buffer = buffer[-hold_back:]  # 保留尾部用于拼接潜在闭标签。
                        if think_text:  # 仅当文本非空时产出事件。
                            yield ("think", think_text)  # 产出思考文本事件。
                        break  # 到达安全边界，等待下一 chunk。

                    think_text = buffer[: close_match.start()]  # 截取闭标签前的隐藏文本。
                    if think_text:  # 隐藏文本非空才输出。
                        yield ("think", think_text)  # 产出思考事件。
                    buffer = buffer[close_match.end() :]  # 从缓冲区移除隐藏内容和闭标签。
                    in_hidden = False  # 退出隐藏区，回到可见区解析流程。
                    current_tag = ""  # 清空当前标签名，避免影响后续匹配。

        if buffer:  # 流结束后如果缓冲区还有残留，按当前状态收尾输出。
            if in_hidden:  # 若结束时仍在隐藏区，残留文本按思考内容处理。
                yield ("think", buffer)  # 产出尾部思考文本。
            else:  # 若不在隐藏区，残留文本属于可见答案。
                yield ("answer", buffer)  # 产出尾部答案文本。

    def invoke_with_meta(
        self,
        question: str,
        config: RunnableConfig | None = None,
    ) -> tuple[str, str]:
        """
        普通调用（带元信息）：
        - answer: 清洗后的最终答案
        - thinking: 提取到的思考片段（可能为空字符串）
        """
        chain = self._require_chain()
        raw_text = chain.invoke({"question": question}, config=config)
        answer = self._clean_text_output(raw_text)
        thinking = self._extract_hidden_text(raw_text)
        return answer, thinking

    def invoke_raw(
        self,
        question: str,
        config: RunnableConfig | None = None,
    ) -> str:
        """普通调用：返回模型原始输出，不做清洗。"""
        chain = self._require_chain()
        return str(chain.invoke({"question": question}, config=config) or "")

    def invoke(
        self,
        question: str,
        config: RunnableConfig | None = None,
    ) -> str:
        """普通调用：返回完整文本。"""
        answer, _ = self.invoke_with_meta(question, config=config)
        return answer

    def batch(
        self,
        questions: list[str],
        config: RunnableConfig | list[RunnableConfig] | None = None,
    ) -> list[str]:
        """批量调用：输入多个问题，返回清洗后的答案列表。"""
        chain = self._require_chain()
        inputs = [{"question": question} for question in questions]
        raw_outputs = chain.batch(inputs, config=config)
        return [self._clean_text_output(output) for output in raw_outputs]

    def stream_with_thinking(
        self,
        question: str,
        config: RunnableConfig | None = None,
    ) -> Iterator[tuple[str, str]]:
        """
        流式调用（带事件类型）：
        - ("answer", 文本)：可直接拼接到最终回答；
        - ("think", 文本)：可用作“模型正在思考”的状态提示。
        """
        chain = self._require_chain()
        raw_chunks = chain.stream({"question": question}, config=config)
        for kind, text in self._split_stream_visible_hidden(raw_chunks):
            if kind == "answer":
                cleaned = self._clean_answer_fragment(text)
                if cleaned:
                    yield ("answer", cleaned)
            elif text.strip():
                yield ("think", text)

    def stream(
        self,
        question: str,
        config: RunnableConfig | None = None,
    ) -> Iterator[str]:
        """流式调用：逐块返回文本。"""
        for kind, text in self.stream_with_thinking(question, config=config):
            if kind == "answer":
                yield text
