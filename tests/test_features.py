import numpy as np
import pandas as pd
import pytest

from src.elo import expected_score, updated_ratings
from src.features import MODEL_FEATURES, build_player_features, symmetric_frame, to_model_frame


def synthetic_matches(n: int = 60, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        a, b = rng.choice(6, size=2, replace=False)
        rows.append({
            "tourney_date": pd.Timestamp("2020-01-06") + pd.Timedelta(weeks=i), "surface": rng.choice(["Hard", "Clay"]),
            "best_of": 3, "winner_id": int(a), "winner_name": f"P{a}", "winner_rank": float(a + 1),
            "winner_rank_points": 1000.0, "winner_age": 25.0,
            "loser_id": int(b), "loser_name": f"P{b}", "loser_rank": float(b + 1),
            "loser_rank_points": 900.0, "loser_age": 26.0,
        })
    return pd.DataFrame(rows)


def test_expected_score_is_symmetric():
    assert expected_score(1600, 1500) + expected_score(1500, 1600) == pytest.approx(1.0)
    assert expected_score(1500, 1500) == pytest.approx(0.5)


def test_elo_update_is_zero_sum_with_equal_experience():
    w, l = updated_ratings(1500, 1500, 10, 10)
    assert w > 1500 > l
    assert w + l == pytest.approx(3000)


def test_features_do_not_use_future_matches():
    """Features of the first k matches must be identical whether or not later matches exist."""
    full = synthetic_matches(60)
    truncated = full.iloc[:30]
    feats_full, _ = build_player_features(full)
    feats_trunc, _ = build_player_features(truncated)
    cols = [c for c in feats_full.columns if c.startswith(("w_", "l_"))]
    pd.testing.assert_frame_equal(feats_full.iloc[:30][cols], feats_trunc[cols])


def test_first_match_uses_initial_state():
    feats, _ = build_player_features(synthetic_matches(5))
    first = feats.iloc[0]
    assert first["w_elo"] == first["l_elo"] == 1500
    assert first["w_recent_win_rate"] == 0.5


def test_orientation_flips_sign_and_label():
    feats, _ = build_player_features(synthetic_matches(20))
    a = to_model_frame(feats, np.ones(len(feats)))
    b = to_model_frame(feats, np.zeros(len(feats)))
    diff_cols = [c for c in MODEL_FEATURES if c.startswith("diff_")]
    np.testing.assert_allclose(a[diff_cols].to_numpy(), -b[diff_cols].to_numpy())
    assert a["y"].eq(1).all() and b["y"].eq(0).all()


def test_symmetric_frame_is_balanced():
    feats, _ = build_player_features(synthetic_matches(20))
    frame = symmetric_frame(feats)
    assert len(frame) == 40
    assert frame["y"].mean() == 0.5
