"""Tests for code quality score calculation."""

from analysis.quality_score import calculate_quality_score


def test_no_issues():
    """Code with no issues should receive 100."""
    score = calculate_quality_score("", "", "")
    assert score == 100


def test_quality_issues():
    """Quality score should decrease when issues are detected."""
    pylint_output = "C0114 Missing module docstring"
    flake8_output = "E225 missing whitespace around operator"
    bandit_output = "B101 Use of assert"

    score = calculate_quality_score(
        pylint_output,
        flake8_output,
        bandit_output,
    )

    assert score < 100


def test_score_never_negative():
    """Quality score should not become negative."""
    pylint_output = "C0114 " * 30

    score = calculate_quality_score(
        pylint_output,
        "",
        "",
    )

    assert score >= 0