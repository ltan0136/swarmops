"""Small, dependency-free statistics for comparing success rates.

- Wilson score interval for one success rate.
- Exact McNemar test for two variants run on the *same* episodes (paired).
- Newcombe's hybrid score interval for two rates on *different* episodes (independent).
"""

from __future__ import annotations

import math

Z_95 = 1.959963984540054


def wilson_interval(successes: int, n: int, z: float = Z_95) -> tuple[float, float]:
    """Confidence interval for a proportion that behaves well near 0 % and 100 %."""
    if n <= 0:
        return (0.0, 1.0)
    p = successes / n
    denominator = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denominator
    half_width = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return (max(0.0, centre - half_width), min(1.0, centre + half_width))


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact p-value for paired binary outcomes.

    b = episodes the first variant got right and the second got wrong; c = the reverse.
    Episodes both variants got right (or both wrong) carry no information about the difference.
    """
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2**n
    return min(1.0, 2 * tail)


def newcombe_interval(
    successes_1: int, n_1: int, successes_2: int, n_2: int, z: float = Z_95
) -> tuple[float, float]:
    """Confidence interval for p1 - p2 when the two rates come from different episodes."""
    p1, p2 = successes_1 / n_1, successes_2 / n_2
    low_1, high_1 = wilson_interval(successes_1, n_1, z)
    low_2, high_2 = wilson_interval(successes_2, n_2, z)
    difference = p1 - p2
    low = difference - math.sqrt((p1 - low_1) ** 2 + (high_2 - p2) ** 2)
    high = difference + math.sqrt((high_1 - p1) ** 2 + (p2 - low_2) ** 2)
    return (low, high)
