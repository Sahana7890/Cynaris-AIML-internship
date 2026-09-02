"""Python syntax validation module."""

import ast


def check_syntax(code: str) -> bool:
    """
    Check whether the supplied Python code has valid syntax.

    Args:
        code: Python source code as a string.

    Returns:
        True if the syntax is valid, otherwise False.
    """
    try:
        ast.parse(code)
        return True
    except (SyntaxError, ValueError, TypeError):
        return False