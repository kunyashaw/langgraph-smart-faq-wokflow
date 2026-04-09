"""
Search Tools 使用说明

作用：
- 提供一组可被 LangChain/LangGraph agent 调用的搜索工具。
- 包含“公开知识搜索”和“项目内文本搜索”两类能力。

常用用法：
- 导入工具列表：
    from tools.search_tools import SEARCH_TOOLS

- 搜索公开知识：
    from tools.search_tools import search_web
    result = search_web.invoke({"query": "LangGraph"})

- 搜索项目代码：
    from tools.search_tools import search_project_files
    result = search_project_files.invoke({"query": "PromptAgent"})

适用场景：
- tool-calling agent
- 编程分析 agent
- graph 中的检索节点
"""

from __future__ import annotations

import html
import json
import re
import subprocess
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

from langchain_core.tools import tool


WIKIPEDIA_SEARCH_API = "https://en.wikipedia.org/w/api.php"


def _normalize_text(text: str) -> str:
    """清理 HTML 片段和多余空白。"""
    no_tags = re.sub(r"<[^>]+>", "", text)
    return " ".join(html.unescape(no_tags).split())


def _safe_read_text(path: Path) -> str:
    """尽量以文本方式读取文件，读取失败时返回空字符串。"""
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return ""


@tool
def search_web(query: str, max_results: int = 3) -> str:
    """搜索公开知识信息，优先返回 Wikipedia 的搜索摘要。"""
    clean_query = query.strip()
    if not clean_query:
        return "query 不能为空"

    limit = max(1, min(max_results, 5))
    params = urlencode(
        {
            "action": "query",
            "list": "search",
            "srsearch": clean_query,
            "srlimit": limit,
            "utf8": 1,
            "format": "json",
        }
    )
    request = Request(
        f"{WIKIPEDIA_SEARCH_API}?{params}",
        headers={"User-Agent": "python-langchain-demos/1.0"},
    )

    try:
        with urlopen(request, timeout=10) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except Exception as exc:
        return f"搜索失败：{exc}"

    items = payload.get("query", {}).get("search", [])
    if not items:
        return f"未找到与「{clean_query}」相关的公开结果。"

    lines: list[str] = []
    for index, item in enumerate(items[:limit], start=1):
        title = item.get("title", "未知标题")
        snippet = _normalize_text(item.get("snippet", ""))
        url = f"https://en.wikipedia.org/wiki/{quote(title.replace(' ', '_'))}"
        lines.append(f"{index}. {title}: {snippet}\n   {url}")

    return "\n".join(lines)


@tool
def search_project_files(
    query: str,
    search_path: str = ".",
    max_results: int = 5,
) -> str:
    """在当前项目文件中搜索关键词，优先使用 rg。"""
    clean_query = query.strip()
    if not clean_query:
        return "query 不能为空"

    limit = max(1, min(max_results, 20))
    base_path = Path(search_path).resolve()
    if not base_path.exists():
        return f"路径不存在：{base_path}"

    try:
        result = subprocess.run(
            [
                "rg",
                "-n",
                "--no-heading",
                "--color",
                "never",
                clean_query,
                str(base_path),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode not in (0, 1):
            return f"搜索失败：{result.stderr.strip() or 'rg 执行异常'}"

        lines = [line for line in result.stdout.splitlines() if line.strip()][:limit]
        if lines:
            return "\n".join(f"{index}. {line}" for index, line in enumerate(lines, start=1))
    except FileNotFoundError:
        pass

    matches: list[str] = []
    for path in base_path.rglob("*"):
        if not path.is_file():
            continue
        text = _safe_read_text(path)
        if not text:
            continue

        for line_number, line in enumerate(text.splitlines(), start=1):
            if clean_query in line:
                relative_path = path.relative_to(base_path)
                matches.append(f"{relative_path}:{line_number}:{line.strip()}")
                if len(matches) >= limit:
                    return "\n".join(
                        f"{index}. {line}" for index, line in enumerate(matches, start=1)
                    )

    if not matches:
        return f"在 {base_path} 中未找到与「{clean_query}」相关的内容。"

    return "\n".join(f"{index}. {line}" for index, line in enumerate(matches, start=1))


def get_search_tools() -> list[Any]:
    """返回当前模块下所有搜索工具，方便 agent 统一注册。"""
    return [
        search_web,
        search_project_files,
    ]


SEARCH_TOOLS = get_search_tools()
