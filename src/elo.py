"""Elo ratings with a decaying K-factor (FiveThirtyEight tennis variant).

K = 250 / (n + 5) ** 0.4, where n is the player's number of prior rated matches. The
constants are taken from FiveThirtyEight's published method and are not tuned on our data.
"""

INITIAL_RATING = 1500.0


def k_factor(n_matches: int) -> float:
    return 250.0 / (n_matches + 5) ** 0.4


def expected_score(rating_a: float, rating_b: float) -> float:
    return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))


def updated_ratings(winner: float, loser: float, n_winner: int, n_loser: int) -> tuple[float, float]:
    surprise = 1.0 - expected_score(winner, loser)
    return winner + k_factor(n_winner) * surprise, loser - k_factor(n_loser) * surprise


def blend(overall: float, surface: float, weight: float) -> float:
    """Weighted mix of overall and surface Elo; weight=1 means overall only."""
    return weight * overall + (1.0 - weight) * surface
