"""Tests for preprocessing and syntax checking."""

from preprocessing.syntax_checker import check_syntax


def test_valid_python_code():
    """Valid Python code should return True."""
    code = "x = 10"
    assert check_syntax(code) is True


def test_invalid_python_code():
    """Invalid Python code should return False."""
    code = "x = "
    assert check_syntax(code) is False


def test_function_code():
    """A valid function should pass syntax checking."""
    code = """
def add(a, b):
    return a + b
"""
    assert check_syntax(code) is True