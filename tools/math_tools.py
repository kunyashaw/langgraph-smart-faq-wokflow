"""
Math Tools 使用说明

作用：
- 提供一组可被 LangChain/LangGraph agent 直接调用的数学工具。
- 适合处理基础四则运算，以及较简单的表达式计算场景。

常用用法：
- 直接导入工具列表：
    from tools.math_tools import MATH_TOOLS

- 单独调用某个工具：
    from tools.math_tools import calculate_expression
    result = calculate_expression.invoke({"expression": "(3 + 5) * 2"})

适用场景：
- 工具调用型 agent
- graph node 中的辅助计算
- demo / 教学示例
"""

from __future__ import annotations

import ast
import operator
from typing import Any

from langchain_core.tools import tool


_ALLOWED_BINARY_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

_ALLOWED_UNARY_OPERATORS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def _evaluate_expression_node(node: ast.AST) -> float:
    """安全递归计算表达式 AST，仅允许基础算术运算。"""
    if isinstance(node, ast.Expression):
        return _evaluate_expression_node(node.body)

    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)

    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_UNARY_OPERATORS:
        operand = _evaluate_expression_node(node.operand)
        return float(_ALLOWED_UNARY_OPERATORS[type(node.op)](operand))

    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_BINARY_OPERATORS:
        left = _evaluate_expression_node(node.left)
        right = _evaluate_expression_node(node.right)
        if isinstance(node.op, (ast.Div, ast.FloorDiv, ast.Mod)) and right == 0:
            raise ValueError("除数不能为 0")
        return float(_ALLOWED_BINARY_OPERATORS[type(node.op)](left, right))

    raise ValueError("表达式中包含不支持的语法，仅支持 + - * / // % ** 和括号")


@tool
def add_numbers(a: float, b: float) -> float:
    """计算两个数字相加。"""
    return a + b


@tool
def subtract_numbers(a: float, b: float) -> float:
    """计算两个数字相减。"""
    return a - b


@tool
def multiply_numbers(a: float, b: float) -> float:
    """计算两个数字相乘。"""
    return a * b


@tool
def divide_numbers(a: float, b: float) -> float:
    """计算两个数字相除。"""
    if b == 0:
        raise ValueError("除数不能为 0")
    return a / b


@tool
def calculate_expression(expression: str) -> float:
    """安全计算数学表达式，支持 + - * / // % ** 和括号。"""
    clean_expression = expression.strip()
    if not clean_expression:
        raise ValueError("expression 不能为空")

    try:
        parsed = ast.parse(clean_expression, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"表达式语法错误: {exc}") from exc

    return _evaluate_expression_node(parsed)


def get_math_tools() -> list[Any]:
    """返回当前模块下所有数学工具，方便 agent 统一注册。"""
    return [
        add_numbers,
        subtract_numbers,
        multiply_numbers,
        divide_numbers,
        calculate_expression,
    ]


MATH_TOOLS = get_math_tools()

