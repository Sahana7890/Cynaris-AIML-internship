"""Calculate the overall code quality score."""


def count_issues(output: str) -> int:
    """
    Count detected static-analysis issues.

    Args:
        output: Analyzer output.

    Returns:
        Number of detected issues.
    """
    if not output:
        return 0

    return len(
        [
            line
            for line in output.splitlines()
            if line.strip()
        ]
    )


def calculate_quality_score(
    pylint_output: str,
    flake8_output: str,
    bandit_output: str,
) -> int:
    """
    Calculate a code quality score.

    Five points are deducted for every detected issue.

    Args:
        pylint_output: Pylint results.
        flake8_output: Flake8 results.
        bandit_output: Bandit results.

    Returns:
        Quality score between 0 and 100.
    """
    total_issues = (
        count_issues(pylint_output)
        + count_issues(flake8_output)
        + count_issues(bandit_output)
    )

    score = 100 - (total_issues * 5)

    return max(score, 0)