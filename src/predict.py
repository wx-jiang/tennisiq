"""Predictions from the saved model and the feature snapshot at the data cutoff."""

import json
import re
import unicodedata
from functools import lru_cache

import numpy as np
import pandas as pd
import xgboost as xgb

from src.data import SURFACES
from src.elo import blend, expected_score
from src.train import META_PATH, MODEL_PATH, SNAPSHOT_PATH, STATS_PATH

MIN_MATCHES = 10  # players with fewer tour-level matches have unreliable features
STALE_DAYS = 365


class PlayerNotFound(ValueError):
    pass


@lru_cache(maxsize=1)
def load_model() -> tuple[xgb.XGBClassifier, dict]:
    model = xgb.XGBClassifier()
    model.load_model(MODEL_PATH)
    return model, json.loads(META_PATH.read_text())


@lru_cache(maxsize=1)
def load_snapshot() -> pd.DataFrame:
    return pd.read_csv(SNAPSHOT_PATH)


@lru_cache(maxsize=1)
def load_stats() -> pd.DataFrame:
    return pd.read_csv(STATS_PATH)


def normalize(name: str) -> str:
    ascii_name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    return " ".join(ascii_name.lower().replace("-", " ").replace(".", "").split())


@lru_cache(maxsize=1)
def player_index() -> pd.DataFrame:
    snap = load_snapshot()
    players = snap.drop_duplicates("player_id")[["player_id", "name", "matches", "last_match"]].copy()
    players["key"] = players["name"].map(normalize)
    players["last_key"] = players["key"].str.split().str[-1]
    return players


def resolve_player(query: str) -> pd.Series:
    """Match a user-supplied name to one player: exact full name, then unique surname
    or name substring. Ambiguous surnames resolve to the player with far more tour matches."""
    if not query or not query.strip():
        raise PlayerNotFound("Player name is empty.")
    players = player_index()
    key = normalize(query)
    candidates = players[players["key"] == key]
    if candidates.empty:
        candidates = players[(players["last_key"] == key) | players["key"].str.contains(rf"\b{re.escape(key)}\b", regex=True)]
    if candidates.empty:
        raise PlayerNotFound(f"No ATP tour-level player matching '{query}' in the dataset (2010-2025).")
    if len(candidates) > 1:
        candidates = candidates.sort_values("matches", ascending=False)
        top, second = candidates.iloc[0], candidates.iloc[1]
        if top["matches"] < 3 * second["matches"]:
            names = ", ".join(candidates["name"].head(5))
            raise PlayerNotFound(f"'{query}' is ambiguous: {names}. Please use the full name.")
    return candidates.iloc[0]


def player_features(player_id: int, surface: str) -> pd.Series:
    snap = load_snapshot()
    return snap[(snap["player_id"] == player_id) & (snap["surface"] == surface)].iloc[0]


def validate_surface(surface: str) -> str:
    surface = (surface or "").strip().capitalize()
    if surface not in SURFACES:
        raise ValueError(f"Unsupported surface '{surface}'. Use one of: {', '.join(SURFACES)}.")
    return surface


def predict(player1: str, player2: str, surface: str, best_of: int = 3) -> dict:
    surface = validate_surface(surface)
    if best_of not in (3, 5):
        raise ValueError("best_of must be 3 or 5.")
    p1, p2 = resolve_player(player1), resolve_player(player2)
    if p1["player_id"] == p2["player_id"]:
        raise ValueError("Please choose two different players.")
    for p in (p1, p2):
        if p["matches"] < MIN_MATCHES:
            raise PlayerNotFound(f"{p['name']} has only {p['matches']} tour-level matches; too few to predict reliably.")

    model, meta = load_model()
    a, b = player_features(p1["player_id"], surface), player_features(p2["player_id"], surface)
    diffs = {f"diff_{f}": a[f] - b[f] for f in meta["player_features"]}
    x = pd.DataFrame([diffs | {"best_of_5": int(best_of == 5)}])[meta["features"]]
    x_flipped = x.copy()
    x_flipped[[c for c in x.columns if c.startswith("diff_")]] *= -1
    # Average both orientations so P(A beats B) = 1 - P(B beats A) exactly.
    p_xgb = 0.5 * (model.predict_proba(x)[0, 1] + 1 - model.predict_proba(x_flipped)[0, 1])

    w = meta["elo_weight"]
    elo_a, elo_b = blend(a["elo"], a["surface_elo"], w), blend(b["elo"], b["surface_elo"], w)
    p_elo = expected_score(elo_a, elo_b)

    cutoff = pd.Timestamp(meta["data_cutoff"])
    warnings = [
        f"{p['name']}'s last match in the data is {p['last_match']}; features may be outdated."
        for p in (p1, p2) if (cutoff - pd.Timestamp(p["last_match"])).days > STALE_DAYS
    ]
    return {
        "player1": p1["name"], "player2": p2["name"], "surface": surface, "best_of": best_of,
        "xgboost_win_prob": {p1["name"]: round(float(p_xgb), 3), p2["name"]: round(float(1 - p_xgb), 3)},
        "elo_win_prob": {p1["name"]: round(float(p_elo), 3), p2["name"]: round(float(1 - p_elo), 3)},
        "elo": {
            p1["name"]: {"overall": round(a["elo"]), f"{surface.lower()}": round(a["surface_elo"])},
            p2["name"]: {"overall": round(b["elo"]), f"{surface.lower()}": round(b["surface_elo"])},
            "overall_diff": round(a["elo"] - b["elo"]),
            "surface_diff": round(a["surface_elo"] - b["surface_elo"]),
        },
        "inputs": {name: _readable_inputs(row) for name, row in ((p1["name"], a), (p2["name"], b))},
        "model_version": meta["model_version"],
        "data_cutoff": meta["data_cutoff"],
        "warnings": warnings,
    }


def _readable_inputs(row: pd.Series) -> dict:
    return {
        "atp_rank": None if pd.isna(row["rank"]) else int(row["rank"]),
        "rank_points": None if pd.isna(row["rank_points"]) else int(row["rank_points"]),
        "recent_win_rate_last20_smoothed": round(float(row["recent_win_rate"]), 3),
        "surface_win_rate_smoothed": round(float(row["surface_win_rate"]), 3),
        "tour_matches_since_2010": int(row["matches"]),
        "surface_matches_since_2010": int(round(np.expm1(row["log_surface_matches"]))),
        "age": None if pd.isna(row["age"]) else round(float(row["age"]), 1),
        "last_match": row["last_match"],
    }
