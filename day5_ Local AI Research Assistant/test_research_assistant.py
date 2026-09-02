"""
Unit tests for the Local AI Research Assistant.
"""

import pytest

from capstone.research_assistant import validate_question


def test_valid_question():
    """A valid question should be accepted."""

    state = {
        "question": "What is Generative AI?",
        "answer": "",
    }

    result = validate_question(state)

    assert result["question"] == "What is Generative AI?"


def test_question_with_spaces():
    """Leading and trailing spaces should be removed."""

    state = {
        "question": "  What is RAG?  ",
        "answer": "",
    }

    result = validate_question(state)

    assert result["question"] == "What is RAG?"


def test_empty_question():
    """An empty question should raise ValueError."""

    state = {
        "question": "",
        "answer": "",
    }

    with pytest.raises(ValueError):
        validate_question(state)


def test_whitespace_question():
    """A whitespace-only question should raise ValueError."""

    state = {
        "question": "     ",
        "answer": "",
    }

    with pytest.raises(ValueError):
        validate_question(state)