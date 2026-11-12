"""Pre-match player features computed in a single chronological pass.

Leakage prevention: for every match we first record each player's state (Elo, form,
experience) as it stood *before* the match, and only then update that state with the
result. Matches are processed in (date, tournament, round) order, so a player's
features only ever reflect strictly earlier matches. In-match statistics (aces, serve
points, score) are never used.
"""

from collections import defaultdict, deque
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.elo import INITIAL_RATING, updated_ratings

RECENT_WINDOW = 20
UNRANKED_RANK = 2000

# Per-player features; the model uses the A-minus-B difference of each.
PLAYER_FEATURES = [
    "elo", "surface_elo", "log_rank", "log_points", "recent_win_rate",
    "surface_win_rate", "log_matches", "log_surface_matches", "age",
]
MODEL_FEATURES = [f"diff_{f}" for f in PLAYER_FEATURES] + ["best_of_5"]


@dataclass
class PlayerState:
    name: str = ""
    elo: float = INITIAL_RATING
    n: int = 0
    surface_elo: dict = field(default_factory=lambda: defaultdict(lambda: INITIAL_RATING))
    surface_n: dict = field(default_factory=lambda: defaultdict(int))
    surface_wins: dict = field(default_factory=lambda: defaultdict(int))
    recent: deque = field(default_factory=lambda: deque(maxlen=RECENT_WINDOW))
    last_date: pd.Timestamp | None = None
    last_rank: float = np.nan
    last_points: float = np.nan
    birth_year: float = np.nan


def smoothed_rate(wins: float, n: int, prior_n: int = 4) -> float:
    # Beta(2, 2)-style shrinkage toward 0.5 so players with few matches are not at 0 or 1.
    return (wins + prior_n / 2) / (n + prior_n)


def rank_features(rank: float, points: float) -> tuple[float, float]:
    rank = UNRANKED_RANK if pd.isna(rank) else rank
    points = 0.0 if pd.isna(points) else points
    return float(np.log(rank)), float(np.log1p(points))


def player_snapshot(state: PlayerState, surface: str, rank: float, points: float, age: float) -> dict:
    log_rank, log_points = rank_features(rank, points)
    return {
        "elo": state.elo,
        "surface_elo": state.surface_elo[surface],
        "log_rank": log_rank,
        "log_points": log_points,
        "recent_win_rate": smoothed_rate(sum(state.recent), len(state.recent)),
        "surface_win_rate": smoothed_rate(state.surface_wins[surface], state.surface_n[surface]),
        "log_matches": float(np.log1p(state.n)),
        "log_surface_matches": float(np.log1p(state.surface_n[surface])),
        "age": age,
    }


def update_states(winner: PlayerState, loser: PlayerState, surface: str) -> None:
    winner.elo, loser.elo = updated_ratings(winner.elo, loser.elo, winner.n, loser.n)
    winner.surface_elo[surface], loser.surface_elo[surface] = updated_ratings(
        winner.surface_elo[surface], loser.surface_elo[surface],
        winner.surface_n[surface], loser.surface_n[surface],
    )
    for player, won in ((winner, 1), (loser, 0)):
        player.n += 1
        player.surface_n[surface] += 1
        player.surface_wins[surface] += won
        player.recent.append(won)


def build_player_features(matches: pd.DataFrame) -> tuple[pd.DataFrame, dict[int, PlayerState]]:
    """Return one row per match with pre-match winner (w_*) and loser (l_*) features,
    plus the final state of every player (the prediction snapshot)."""
    states: dict[int, PlayerState] = defaultdict(PlayerState)
    rows = []
    for m in matches.itertuples(index=False):
        w, l = states[m.winner_id], states[m.loser_id]
        w_feat = player_snapshot(w, m.surface, m.winner_rank, m.winner_rank_points, m.winner_age)
        l_feat = player_snapshot(l, m.surface, m.loser_rank, m.loser_rank_points, m.loser_age)
        rows.append({f"w_{k}": v for k, v in w_feat.items()} | {f"l_{k}": v for k, v in l_feat.items()})

        update_states(w, l, m.surface)
        for state, pid_name, rank, points, age in (
            (w, m.winner_name, m.winner_rank, m.winner_rank_points, m.winner_age),
            (l, m.loser_name, m.loser_rank, m.loser_rank_points, m.loser_age),
        ):
            state.name = pid_name
            state.last_date = m.tourney_date
            state.last_rank, state.last_points = rank, points
            if not pd.isna(age):
                state.birth_year = m.tourney_date.year + m.tourney_date.day_of_year / 365.25 - age

    features = pd.DataFrame(rows, index=matches.index)
    return pd.concat([matches, features], axis=1), dict(states)


def to_model_frame(df: pd.DataFrame, a_is_winner: np.ndarray) -> pd.DataFrame:
    """Orient each match as player A vs player B. `a_is_winner` must not be derived from
    anything that leaks the result into the row (e.g. a seeded coin flip or both orientations)."""
    a_is_winner = np.asarray(a_is_winner, dtype=bool)
    out = pd.DataFrame(index=df.index)
    for f in PLAYER_FEATURES:
        diff = df[f"w_{f}"] - df[f"l_{f}"]
        out[f"diff_{f}"] = np.where(a_is_winner, diff, -diff)
    out["best_of_5"] = (df["best_of"] == 5).astype(int)
    out["y"] = a_is_winner.astype(int)
    return out


def symmetric_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Both orientations of every match: removes any ordering signal and makes the
    training data exactly antisymmetric in the difference features."""
    n = len(df)
    return pd.concat([to_model_frame(df, np.ones(n)), to_model_frame(df, np.zeros(n))], ignore_index=True)
