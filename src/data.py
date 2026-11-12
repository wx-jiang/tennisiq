"""Download and clean ATP tour-level singles matches (Jeff Sackmann format)."""

from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_PATH = ROOT / "data" / "matches.parquet"

# The original GitHub repo (JeffSackmann/tennis_atp) returned 404 as of Nov 2026.
# We use a Hugging Face mirror of the same files (CC BY-NC-SA 4.0), pinned to a commit.
MIRROR_REVISION = "8ac86f7c62889b5d5c4cfec4c6403edcb8c5d993"
MIRROR_URL = (
    "https://huggingface.co/datasets/Aneeshers/tennis-sackmann-archive/resolve/"
    f"{MIRROR_REVISION}/atp/atp_matches_{{year}}.csv"
)

# 2010-2014 are used only to warm up Elo / rolling features; no model rows come from them.
WARMUP_YEARS = range(2010, 2015)
STUDY_YEARS = range(2015, 2026)
SURFACES = ("Hard", "Clay", "Grass")

COLUMNS = [
    "tourney_id", "tourney_name", "surface", "tourney_level", "tourney_date", "match_num",
    "round", "best_of", "score",
    "winner_id", "winner_name", "winner_rank", "winner_rank_points", "winner_age", "winner_ht", "winner_hand",
    "loser_id", "loser_name", "loser_rank", "loser_rank_points", "loser_age", "loser_ht", "loser_hand",
]

ROUND_ORDER = {"Q1": 0, "Q2": 1, "Q3": 2, "ER": 3, "R128": 4, "RR": 5, "R64": 6, "R32": 7,
               "R16": 8, "QF": 9, "SF": 10, "BR": 11, "F": 12}


def download(years=None, force: bool = False) -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for year in years or [*WARMUP_YEARS, *STUDY_YEARS]:
        path = RAW_DIR / f"atp_matches_{year}.csv"
        if path.exists() and not force:
            continue
        resp = requests.get(MIRROR_URL.format(year=year), timeout=60)
        resp.raise_for_status()
        path.write_bytes(resp.content)
        print(f"downloaded {path.name}")


def clean(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw[COLUMNS].copy()
    df["tourney_date"] = pd.to_datetime(df["tourney_date"].astype(str), format="%Y%m%d")
    df = df[df["surface"].isin(SURFACES)]
    # Walkovers/defaults carry no information about playing strength.
    score = df["score"].fillna("").str.upper()
    df = df[~score.str.contains("W/O|WALKOVER|DEF|UNFINISHED") & (score != "")]
    df = df.dropna(subset=["winner_id", "loser_id"])
    df["round_order"] = df["round"].map(ROUND_ORDER).fillna(6)
    # tourney_date is the tournament start date, so order within a tournament by round
    # to ensure each match only sees results from earlier rounds.
    df = df.sort_values(["tourney_date", "tourney_id", "round_order", "match_num"], kind="stable")
    return df.reset_index(drop=True)


def build() -> pd.DataFrame:
    download()
    raw = pd.concat(
        [pd.read_csv(RAW_DIR / f"atp_matches_{y}.csv", low_memory=False) for y in [*WARMUP_YEARS, *STUDY_YEARS]],
        ignore_index=True,
    )
    df = clean(raw)
    df.to_parquet(PROCESSED_PATH, index=False)
    print(f"saved {len(df):,} matches to {PROCESSED_PATH.relative_to(ROOT)}")
    return df


def load_matches() -> pd.DataFrame:
    return pd.read_parquet(PROCESSED_PATH)


if __name__ == "__main__":
    build()
