"""Fit the Elo baseline and XGBoost model with chronological validation, then save artifacts.

Protocol
- Study period rows: 2015-2025 (2010-2014 only warm up Elo/form features).
- Hyperparameters (Elo blend weight, XGBoost grid) are chosen by mean log loss over
  expanding-window folds that validate on 2022, 2023 and 2024 separately.
- The final model is refit on 2015-2024. 2025 is never touched here; evaluate_models.py
  scores it once.
"""

import itertools
import json
from datetime import date

import numpy as np
import pandas as pd
import xgboost as xgb

from src.data import ROOT, SURFACES, load_matches
from src.elo import blend, expected_score
from src.features import MODEL_FEATURES, PLAYER_FEATURES, build_player_features, player_snapshot, symmetric_frame

ARTIFACTS = ROOT / "artifacts"
MODEL_PATH = ARTIFACTS / "xgb_model.json"
META_PATH = ARTIFACTS / "model_meta.json"
SNAPSHOT_PATH = ARTIFACTS / "player_snapshot.csv"
STATS_PATH = ARTIFACTS / "player_stats.csv"

MODEL_VERSION = "xgb-2015-2024-v1"
TRAIN_START, TEST_YEAR = 2015, 2025
VALIDATION_YEARS = (2022, 2023, 2024)
ELO_WEIGHTS = (0.0, 0.25, 0.5, 0.75, 1.0)
XGB_GRID = {"max_depth": [2, 3, 4], "n_estimators": [150, 400], "learning_rate": [0.05]}
XGB_FIXED = {"subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 5,
             "objective": "binary:logistic", "eval_metric": "logloss", "random_state": 0, "n_jobs": 4}


def elo_win_prob(df: pd.DataFrame, weight: float) -> np.ndarray:
    """P(winner wins) under blended Elo; computed from the winner's perspective."""
    w = blend(df["w_elo"], df["w_surface_elo"], weight)
    l = blend(df["l_elo"], df["l_surface_elo"], weight)
    return expected_score(w.to_numpy(), l.to_numpy())


def fit_xgb(train: pd.DataFrame, params: dict) -> xgb.XGBClassifier:
    frame = symmetric_frame(train)
    model = xgb.XGBClassifier(**XGB_FIXED, **params)
    model.fit(frame[MODEL_FEATURES], frame["y"])
    return model


def xgb_win_prob(model: xgb.XGBClassifier, df: pd.DataFrame) -> np.ndarray:
    """Orientation-invariant P(winner wins): average of P(A wins | A=winner) and
    1 - P(A wins | A=loser). Trees are not exactly antisymmetric, so this guarantees
    P(A beats B) + P(B beats A) = 1."""
    frame = symmetric_frame(df)
    p = model.predict_proba(frame[MODEL_FEATURES])[:, 1]
    n = len(df)
    return 0.5 * (p[:n] + 1.0 - p[n:])


def loss_from_winner_prob(p: np.ndarray) -> float:
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return float(-np.mean(np.log(p)))


def select_elo_weight(df: pd.DataFrame) -> tuple[float, dict]:
    val = df[df["year"].isin(VALIDATION_YEARS)]
    scores = {w: loss_from_winner_prob(elo_win_prob(val, w)) for w in ELO_WEIGHTS}
    return min(scores, key=scores.get), scores


def select_xgb_params(df: pd.DataFrame) -> tuple[dict, list]:
    results = []
    for values in itertools.product(*XGB_GRID.values()):
        params = dict(zip(XGB_GRID, values, strict=True))
        fold_losses = []
        for year in VALIDATION_YEARS:
            train = df[(df["year"] >= TRAIN_START) & (df["year"] < year)]
            val = df[df["year"] == year]
            fold_losses.append(loss_from_winner_prob(xgb_win_prob(fit_xgb(train, params), val)))
        results.append({**params, "fold_log_loss": fold_losses, "mean_log_loss": float(np.mean(fold_losses))})
        print(params, round(results[-1]["mean_log_loss"], 4))
    best = min(results, key=lambda r: r["mean_log_loss"])
    return {k: best[k] for k in XGB_GRID}, results


def build_snapshot(states: dict, cutoff: pd.Timestamp) -> pd.DataFrame:
    """Each player's features as of the data cutoff, one row per surface."""
    rows = []
    for pid, s in states.items():
        age = cutoff.year + cutoff.day_of_year / 365.25 - s.birth_year
        for surface in SURFACES:
            feats = player_snapshot(s, surface, s.last_rank, s.last_points, age)
            rows.append({
                "player_id": pid, "name": s.name, "surface": surface, "last_match": s.last_date.date().isoformat(),
                "rank": s.last_rank, "rank_points": s.last_points, "matches": s.n, **feats,
            })
    return pd.DataFrame(rows)


def build_player_stats(matches: pd.DataFrame) -> pd.DataFrame:
    """Aggregated win/loss counts per player, year and surface (derived, not raw match data)."""
    year = matches["tourney_date"].dt.year
    wins = matches.assign(year=year).groupby(["winner_id", "winner_name", "year", "surface"]).size()
    losses = matches.assign(year=year).groupby(["loser_id", "loser_name", "year", "surface"]).size()
    wins.index.names = losses.index.names = ["player_id", "name", "year", "surface"]
    stats = pd.concat([wins.rename("wins"), losses.rename("losses")], axis=1).fillna(0).astype(int)
    return stats.reset_index()


def main() -> None:
    matches = load_matches()
    df, states = build_player_features(matches)
    df["year"] = df["tourney_date"].dt.year
    train_period = df[(df["year"] >= TRAIN_START) & (df["year"] < TEST_YEAR)]

    elo_weight, elo_scores = select_elo_weight(train_period)
    print(f"Elo blend weight (overall share): {elo_weight}  validation log loss: {elo_scores}")
    best_params, cv_results = select_xgb_params(train_period)
    print(f"best XGBoost params: {best_params}")

    model = fit_xgb(train_period, best_params)
    ARTIFACTS.mkdir(exist_ok=True)
    model.save_model(MODEL_PATH)

    cutoff = matches["tourney_date"].max()
    build_snapshot(states, cutoff).to_csv(SNAPSHOT_PATH, index=False, float_format="%.5f")
    build_player_stats(matches).to_csv(STATS_PATH, index=False)

    meta = {
        "model_version": MODEL_VERSION,
        "trained_on": f"{TRAIN_START}-01-01 to {train_period['tourney_date'].max().date()}",
        "n_train_matches": len(train_period),
        "test_year": TEST_YEAR,
        "data_cutoff": cutoff.date().isoformat(),
        "features": MODEL_FEATURES,
        "player_features": PLAYER_FEATURES,
        "elo_weight": elo_weight,
        "elo_validation_log_loss": {str(k): v for k, v in elo_scores.items()},
        "xgb_params": best_params,
        "xgb_cv_results": cv_results,
        "validation_years": list(VALIDATION_YEARS),
        "built": date.today().isoformat(),
    }
    META_PATH.write_text(json.dumps(meta, indent=2))
    print(f"saved model and artifacts to {ARTIFACTS.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
