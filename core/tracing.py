"""
Tracing 使用说明

作用：
- 统一管理 LangSmith tracing 的环境配置。
- 为 graph、agent、tool 调用提供统一的 run config 构造函数。

常用用法：
- 初始化 Smith：
    from core.tracing import configure_langsmith
    configure_langsmith("demos-simple-assistant")

- 构造一次调用的 config：
    from core.tracing import build_run_config
    config = build_run_config(
        run_name="simple_assistant_run",
        tags=["workflow", "demo"],
        metadata={"entrypoint": "graph.py"},
    )
"""

from __future__ import annotations

import os
from typing import Any, Sequence

from dotenv import load_dotenv
from langchain_core.runnables import RunnableConfig
from langchain_core.runnables.config import merge_configs


DEFAULT_LANGSMITH_PROJECT = "python-langchain-demos"


def _first_non_empty_env(*names: str) -> str | None:
    """按顺序读取多个环境变量，返回第一个非空值。"""
    for name in names:
        value = os.getenv(name)
        if value and value.strip():
            return value.strip()
    return None


def configure_langsmith(project_name: str = DEFAULT_LANGSMITH_PROJECT) -> str:
    """统一配置 LangSmith 环境变量，兼容当前工程里的自定义 key 名。"""
    load_dotenv()

    api_key = _first_non_empty_env(
        "LANGSMITH_API_KEY",
        "SMITH_API_KEY",
        "SMILE_API_KEY",
        "LANGCHAIN_API_KEY",
    )
    endpoint = _first_non_empty_env(
        "LANGSMITH_ENDPOINT",
        "LANGCHAIN_ENDPOINT",
    )

    if api_key:
        os.environ.setdefault("LANGSMITH_API_KEY", api_key)
        os.environ.setdefault("LANGSMITH_TRACING", "true")
    if endpoint:
        os.environ.setdefault("LANGSMITH_ENDPOINT", endpoint)

    os.environ["LANGSMITH_PROJECT"] = project_name
    return os.environ["LANGSMITH_PROJECT"]


def build_run_config(
    run_name: str,
    *,
    tags: Sequence[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> RunnableConfig:
    """构造一份适合 LangSmith 追踪的基础 RunnableConfig。"""
    return {
        "run_name": run_name,
        "tags": list(tags or []),
        "metadata": dict(metadata or {}),
    }


def extend_run_config(
    config: RunnableConfig | None,
    *,
    run_name: str | None = None,
    tags: Sequence[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> RunnableConfig:
    """在父 config 基础上追加当前节点/agent 的 tracing 信息。"""
    extra_config: RunnableConfig = {
        "tags": list(tags or []),
        "metadata": dict(metadata or {}),
    }
    if run_name:
        extra_config["run_name"] = run_name
    return merge_configs(config, extra_config)
