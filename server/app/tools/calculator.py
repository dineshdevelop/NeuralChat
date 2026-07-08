# =============================================================================
# app/tools/calculator.py — Safe Math Calculator Tool
# =============================================================================
#
# 🧠 LEARNING NOTE — LangChain Tools:
#
# A LangChain Tool is a function that an LLM agent can call.
# Tools have:
#   • name        → how the LLM refers to it (e.g., "calculator")
#   • description → tells the LLM WHEN and HOW to use it (critical for good routing)
#   • func        → the actual Python function to call
#   • args_schema → Pydantic model validating the tool's input
#
# The @tool decorator from LangChain is the simplest way to create a tool.
# It auto-extracts the name and description from the function name + docstring.
#
# Security — Why NOT eval()?
#   eval("__import__('os').system('rm -rf /')") would be catastrophic.
#   We use Python's `ast` module to parse the expression into an Abstract
#   Syntax Tree (AST) and only allow safe numeric operations.
#   This prevents code injection attacks while still supporting:
#     • Basic arithmetic: + - * / // % **
#     • Math functions: sqrt, abs, floor, ceil, round, log, sin, cos, tan, pi, e
# =============================================================================

import ast
import math
import operator
from typing import Union

from langchain_core.tools import tool


# Allowed operators mapped to their Python functions
_SAFE_OPERATORS = {
    ast.Add:      operator.add,
    ast.Sub:      operator.sub,
    ast.Mult:     operator.mul,
    ast.Div:      operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod:      operator.mod,
    ast.Pow:      operator.pow,
    ast.USub:     operator.neg,   # unary minus: -5
    ast.UAdd:     operator.pos,   # unary plus: +5
}

# Allowed math functions and constants
_SAFE_FUNCTIONS = {
    "sqrt":  math.sqrt,
    "abs":   abs,
    "floor": math.floor,
    "ceil":  math.ceil,
    "round": round,
    "log":   math.log,
    "log2":  math.log2,
    "log10": math.log10,
    "sin":   math.sin,
    "cos":   math.cos,
    "tan":   math.tan,
    "pi":    math.pi,
    "e":     math.e,
    "inf":   math.inf,
}


def _safe_eval(node: ast.AST) -> Union[int, float]:
    """
    Recursively evaluates an AST node containing only safe numeric operations.

    🧠 LEARNING NOTE — AST-based safe eval:
    ast.parse("2 + 3 * 4") returns an AST like:
      Module(body=[Expr(value=BinOp(
          left=Constant(value=2),
          op=Add(),
          right=BinOp(
              left=Constant(value=3),
              op=Mult(),
              right=Constant(value=4)
          )
      ))])

    We recursively walk this tree and evaluate each node.
    If we encounter anything unexpected (variable names, imports, etc.),
    we raise a ValueError — preventing code injection.

    Parameters:
      node → an AST node to evaluate

    Returns:
      int or float — the numeric result

    Raises:
      ValueError  → if the expression contains unsafe operations
      ZeroDivisionError → if dividing by zero
    """
    # A plain number: 42, 3.14, etc.
    if isinstance(node, ast.Constant):
        if not isinstance(node.value, (int, float)):
            raise ValueError(f"Only numeric constants allowed, got: {type(node.value).__name__}")
        return node.value

    # Named constant: pi, e, inf
    elif isinstance(node, ast.Name):
        if node.id in _SAFE_FUNCTIONS:
            val = _SAFE_FUNCTIONS[node.id]
            if callable(val):
                raise ValueError(f"'{node.id}' is a function, not a constant")
            return val
        raise ValueError(f"Unknown name: '{node.id}'")

    # Binary operator: a + b, a * b, etc.
    elif isinstance(node, ast.BinOp):
        op_func = _SAFE_OPERATORS.get(type(node.op))
        if op_func is None:
            raise ValueError(f"Unsupported operator: {type(node.op).__name__}")
        left = _safe_eval(node.left)
        right = _safe_eval(node.right)
        return op_func(left, right)

    # Unary operator: -5, +3
    elif isinstance(node, ast.UnaryOp):
        op_func = _SAFE_OPERATORS.get(type(node.op))
        if op_func is None:
            raise ValueError(f"Unsupported unary operator: {type(node.op).__name__}")
        operand = _safe_eval(node.operand)
        return op_func(operand)

    # Function call: sqrt(16), abs(-5), round(3.7)
    elif isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise ValueError("Only simple function calls allowed (e.g., sqrt(4))")
        func_name = node.func.id
        func = _SAFE_FUNCTIONS.get(func_name)
        if func is None or not callable(func):
            raise ValueError(f"Unknown function: '{func_name}'")
        args = [_safe_eval(arg) for arg in node.args]
        return func(*args)

    else:
        raise ValueError(f"Unsupported expression type: {type(node).__name__}")


def calculate(expression: str) -> str:
    """
    Safely evaluates a mathematical expression string and returns the result.

    Parameters:
      expression → math expression string (e.g., "sqrt(144) + 2**8")

    Returns:
      Result as a string (e.g., "268.0") or an error message.
    """
    expression = expression.strip()
    try:
        tree = ast.parse(expression, mode="eval")
        result = _safe_eval(tree.body)

        # Format nicely: integer if whole number, float otherwise
        if isinstance(result, float) and result.is_integer():
            return str(int(result))
        return str(round(result, 10))  # limit decimal places

    except ZeroDivisionError:
        return "Error: Division by zero"
    except ValueError as e:
        return f"Error: {e}"
    except SyntaxError:
        return f"Error: Invalid mathematical expression: '{expression}'"
    except Exception as e:
        return f"Error: {e}"


@tool
def calculator_tool(expression: str) -> str:
    """
    Evaluates a mathematical expression and returns the numeric result.

    Use this tool for any arithmetic or math computation. Supports:
    - Basic arithmetic: +, -, *, /, //, %, ** (power)
    - Math functions: sqrt(), abs(), floor(), ceil(), round(), log(), log2(), log10()
    - Trigonometry: sin(), cos(), tan()
    - Constants: pi, e

    Examples:
    - "2 + 3 * 4" → 14
    - "sqrt(144)" → 12
    - "2 ** 10" → 1024
    - "round(pi, 4)" → 3.1416
    - "(100 + 200) / 3" → 100.0

    Args:
        expression: A mathematical expression string to evaluate.

    Returns:
        The numeric result as a string, or an error message if evaluation fails.
    """
    return calculate(expression)
